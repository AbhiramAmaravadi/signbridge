from __future__ import annotations

import base64
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemini_client import GeminiClient
from gemini_prompts import build_scene_prompt, build_translation_prompt
from mediapipe_pipeline import FeatureExtractor, HolisticConfig, HolisticDetector
from sentence_state import SentenceState, SentenceStateConfig


AI_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_SERVICE_DIR.parent
TRANSFORMER_DIR = PROJECT_ROOT / "transformer"

if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))

from predict_topk import normalize_landmark_sequence, predict_topk_from_array  # noqa: E402


DEFAULT_SEQUENCE_LENGTH = 45
LANDMARKS_PER_FRAME = 543
VALUES_PER_LANDMARK = 3


class PredictRequest(BaseModel):
    """Payload for SignBridge AI service inference.

    Provide either `landmarks` for direct model inference or `image_base64` for
    MediaPipe processing on the server. Direct landmarks should already be in
    SignBridge training order: face, left_hand, pose, right_hand.
    """

    landmarks: list[Any] | None = Field(default=None)
    image_base64: str | None = Field(
        default=None,
        description="Base64 image bytes or a data URL such as data:image/jpeg;base64,...",
    )
    sequence_length: int = Field(default=DEFAULT_SEQUENCE_LENGTH, ge=1, le=120)
    top_k: int = Field(default=5, ge=1, le=20)
    reset_buffer: bool = False
    force_finalize: bool = False
    scene_context: str | None = None


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    timestamp: str
    ready: bool
    sequence_length: int
    buffer_length: int
    top_k: list[PredictionItem]
    candidate: str | None = None
    locked_word: str | None = None
    lock_progress: float = 0.0
    words: list[str] = Field(default_factory=list)
    raw_sentence: str = ""
    finalized_sentence: str | None = None
    eos_trigger: str | None = None
    next_word: str | None = None
    idle_seconds: float = 0.0
    motion_score: float = 0.0
    translation_prompt: str | None = None
    detail: str | None = None


class TranslateRequest(BaseModel):
    words: list[str]
    scene_context: str | None = None
    image_base64: str | None = None
    mime_type: str = "image/jpeg"


class TranslateResponse(BaseModel):
    raw_sentence: str
    polished_sentence: str
    detected_emotion: str = "neutral"
    detected_scene: str = "unknown"
    prompt: str
    used_gemini: bool


class SceneRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"


class SceneResponse(BaseModel):
    scene_context: str
    prompt: str
    used_gemini: bool


app = FastAPI(title="SignBridge AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_lock = Lock()
_detector: HolisticDetector | None = None
_extractor: FeatureExtractor | None = None
_frame_buffer: deque[np.ndarray] = deque(maxlen=DEFAULT_SEQUENCE_LENGTH)
_sentence_state = SentenceState(
    SentenceStateConfig(
        confidence_threshold=0.30,
        stable_prediction_count=3,
        stable_prediction_seconds=2.75,
        cooldown_seconds=1.0,
        idle_seconds_to_finalize=5.0,
        stillness_epsilon=0.003,
    )
)
_gemini_client = GeminiClient.from_env()
print(
    f"[GEMINI] Runtime mode: {'remote Gemini enabled' if _gemini_client else 'local fallback only'}",
    flush=True,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_pipeline() -> tuple[HolisticDetector, FeatureExtractor]:
    global _detector, _extractor

    if _detector is None:
        _detector = HolisticDetector(
            HolisticConfig(
                model_complexity=1,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    if _extractor is None:
        _extractor = FeatureExtractor()

    return _detector, _extractor


def _decode_base64_image(image_base64: str) -> np.ndarray:
    payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError("image_base64 is not valid base64 data.") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("image_base64 could not be decoded as an image.")

    return frame_bgr


def _format_predictions(result: dict) -> list[PredictionItem]:
    return [
        PredictionItem(label=prediction["label"], confidence=prediction["confidence"])
        for prediction in result["output"]["predictions"]
    ]


def _predict_sequence(sequence: np.ndarray, top_k: int) -> list[PredictionItem]:
    normalized = normalize_landmark_sequence(sequence)
    result = predict_topk_from_array(normalized, k=top_k)
    return _format_predictions(result)


def _state_payload(
    predictions: list[PredictionItem],
    sequence: np.ndarray | None,
    force_finalize: bool,
    scene_context: str | None,
) -> dict[str, Any]:
    state = _sentence_state.process(
        [prediction.model_dump() for prediction in predictions],
        landmarks=sequence,
        force_finalize=force_finalize,
    )
    finalized = state["finalized_sentence"]

    return {
        **state,
        "translation_prompt": (
            build_translation_prompt(finalized.split(), scene_context)
            if finalized
            else None
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "signbridge-ai"}


@app.get("/api/v1/gemini/scene-prompt")
def gemini_scene_prompt() -> dict[str, str]:
    return {"prompt": build_scene_prompt()}


@app.post("/api/v1/gemini/translate", response_model=TranslateResponse)
def gemini_translate(request: TranslateRequest) -> TranslateResponse:
    prompt = build_translation_prompt(request.words, request.scene_context)
    fallback = _fallback_polish(request.words)

    if _gemini_client is None:
        return TranslateResponse(
            raw_sentence=" ".join(request.words),
            polished_sentence=fallback,
            detected_emotion="neutral",
            detected_scene=request.scene_context or "unknown",
            prompt=prompt,
            used_gemini=False,
        )

    try:
        payload = _gemini_client.generate_json_with_optional_image(
            prompt,
            image_base64=request.image_base64,
            mime_type=request.mime_type,
        )
    except Exception as exc:
        print(
            f"[GEMINI] Translation multimodal call failed. Falling back locally. "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return TranslateResponse(
            raw_sentence=" ".join(request.words),
            polished_sentence=fallback,
            detected_emotion="neutral",
            detected_scene=request.scene_context or "unknown",
            prompt=prompt,
            used_gemini=False,
        )

    return TranslateResponse(
        raw_sentence=" ".join(request.words),
        polished_sentence=str(payload.get("polished_sentence") or fallback),
        detected_emotion=str(payload.get("detected_emotion") or "neutral").lower(),
        detected_scene=str(payload.get("detected_scene") or request.scene_context or "unknown").lower(),
        prompt=prompt,
        used_gemini=True,
    )


@app.post("/api/v1/gemini/scene", response_model=SceneResponse)
def gemini_scene(request: SceneRequest) -> SceneResponse:
    prompt = build_scene_prompt()

    if _gemini_client is None:
        return SceneResponse(
            scene_context="Unknown scene",
            prompt=prompt,
            used_gemini=False,
        )

    try:
        scene_context = _gemini_client.analyze_image(prompt, request.image_base64, request.mime_type)
    except Exception as exc:
        print(f"[GEMINI] Scene analysis failed: {type(exc).__name__}: {exc}", flush=True)
        return SceneResponse(
            scene_context="Unknown scene",
            prompt=prompt,
            used_gemini=False,
        )

    return SceneResponse(
        scene_context=scene_context or "Unknown scene",
        prompt=prompt,
        used_gemini=True,
    )


@app.post("/api/v1/sentence/finalize", response_model=PredictResponse)
def finalize_sentence(scene_context: str | None = None) -> PredictResponse:
    finalized = _sentence_state.finalize("keypress")
    words = finalized.split() if finalized else []

    return PredictResponse(
        timestamp=_utc_now(),
        ready=True,
        sequence_length=0,
        buffer_length=len(_frame_buffer),
        top_k=[],
        words=[],
        raw_sentence="",
        finalized_sentence=finalized,
        eos_trigger="keypress" if finalized else None,
        translation_prompt=build_translation_prompt(words, scene_context) if finalized else None,
        detail="Sentence finalized by keypress." if finalized else "No words were available to finalize.",
    )


@app.post("/api/v1/sentence/reset")
def reset_sentence() -> dict[str, str]:
    _sentence_state.reset()
    return {"status": "reset"}


def _fallback_polish(words: list[str]) -> str:
    raw = " ".join(word.strip() for word in words if word.strip())
    if not raw:
        return ""
    return raw[0].upper() + raw[1:] + ("." if raw[-1] not in ".!?" else "")


@app.post("/predict", response_model=PredictResponse)
@app.post("/api/v1/inference", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if request.landmarks is None and request.image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide either 'landmarks' or 'image_base64'.")

    if request.landmarks is not None:
        try:
            sequence = normalize_landmark_sequence(np.asarray(request.landmarks, dtype=np.float32))
            top_k = _predict_sequence(sequence, request.top_k)
            state = _state_payload(top_k, sequence, request.force_finalize, request.scene_context)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Landmark inference failed: {exc}") from exc

        return PredictResponse(
            timestamp=_utc_now(),
            ready=True,
            sequence_length=int(sequence.shape[0]),
            buffer_length=int(sequence.shape[0]),
            top_k=top_k,
            candidate=state["candidate"],
            locked_word=state["locked_word"],
            lock_progress=state["lock_progress"],
            words=state["words"],
            raw_sentence=state["raw_sentence"],
            finalized_sentence=state["finalized_sentence"],
            eos_trigger=state["eos_trigger"],
            next_word=state["next_word"],
            idle_seconds=state["idle_seconds"],
            motion_score=state["motion_score"],
            translation_prompt=state["translation_prompt"],
        )

    try:
        frame_bgr = _decode_base64_image(request.image_base64 or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _pipeline_lock:
        global _frame_buffer

        if request.reset_buffer or _frame_buffer.maxlen != request.sequence_length:
            _frame_buffer = deque(maxlen=request.sequence_length)

        try:
            detector, extractor = _get_pipeline()
            results = detector.process(frame_bgr)
            feature_vector = extractor.extract(results)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"MediaPipe processing failed: {exc}") from exc

        frame = feature_vector.reshape(LANDMARKS_PER_FRAME, VALUES_PER_LANDMARK)
        _frame_buffer.append(frame)

        if len(_frame_buffer) < request.sequence_length:
            return PredictResponse(
                timestamp=_utc_now(),
                ready=False,
                sequence_length=request.sequence_length,
                buffer_length=len(_frame_buffer),
                top_k=[],
                detail="Frame accepted; waiting for enough frames to fill the sequence buffer.",
            )

        try:
            sequence = np.stack(tuple(_frame_buffer), axis=0).astype(np.float32, copy=False)
            top_k = _predict_sequence(sequence, request.top_k)
            state = _state_payload(top_k, sequence, request.force_finalize, request.scene_context)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return PredictResponse(
        timestamp=_utc_now(),
        ready=True,
        sequence_length=request.sequence_length,
        buffer_length=len(_frame_buffer),
        top_k=top_k,
        candidate=state["candidate"],
        locked_word=state["locked_word"],
        lock_progress=state["lock_progress"],
        words=state["words"],
        raw_sentence=state["raw_sentence"],
        finalized_sentence=state["finalized_sentence"],
        eos_trigger=state["eos_trigger"],
        next_word=state["next_word"],
        idle_seconds=state["idle_seconds"],
        motion_score=state["motion_score"],
        translation_prompt=state["translation_prompt"],
    )


@app.on_event("shutdown")
def shutdown() -> None:
    if _detector is not None:
        _detector.close()
