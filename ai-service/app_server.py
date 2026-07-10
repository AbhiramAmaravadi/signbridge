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

from mediapipe_pipeline import FeatureExtractor, HolisticConfig, HolisticDetector


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


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    timestamp: str
    ready: bool
    sequence_length: int
    buffer_length: int
    top_k: list[PredictionItem]
    detail: str | None = None


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "signbridge-ai"}


@app.post("/api/v1/inference", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if request.landmarks is None and request.image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide either 'landmarks' or 'image_base64'.")

    if request.landmarks is not None:
        try:
            sequence = normalize_landmark_sequence(np.asarray(request.landmarks, dtype=np.float32))
            top_k = _predict_sequence(sequence, request.top_k)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Landmark inference failed: {exc}") from exc

        return PredictResponse(
            timestamp=_utc_now(),
            ready=True,
            sequence_length=int(sequence.shape[0]),
            buffer_length=int(sequence.shape[0]),
            top_k=top_k,
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return PredictResponse(
        timestamp=_utc_now(),
        ready=True,
        sequence_length=request.sequence_length,
        buffer_length=len(_frame_buffer),
        top_k=top_k,
    )


@app.on_event("shutdown")
def shutdown() -> None:
    if _detector is not None:
        _detector.close()
