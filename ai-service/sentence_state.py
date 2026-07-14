from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import numpy as np


EOS_LABELS = {"done", "finish", "finished", "period", "stop"}

NEXT_WORD_HINTS: dict[str, list[str]] = {
    "i": ["want", "need", "go", "am"],
    "you": ["want", "need", "help", "go"],
    "want": ["water", "food", "help", "more"],
    "need": ["help", "water", "doctor", "bathroom"],
    "go": ["home", "store", "school", "work"],
    "water": ["please", "now"],
    "help": ["please", "me"],
    "thank": ["you"],
    "good": ["morning", "night"],
}


@dataclass
class SentenceStateConfig:
    confidence_threshold: float = 0.30
    stable_prediction_count: int = 3
    cooldown_seconds: float = 1.0
    idle_seconds_to_finalize: float = 5.0
    stillness_epsilon: float = 0.003


@dataclass
class SentenceState:
    """Debounces top-k sign predictions and builds sentence-level state."""

    config: SentenceStateConfig = field(default_factory=SentenceStateConfig)
    words: list[str] = field(default_factory=list)
    finalized_sentences: list[str] = field(default_factory=list)
    candidate_label: str | None = None
    candidate_confidence: float = 0.0
    candidate_hits: int = 0
    last_locked_at: float = 0.0
    last_frame: np.ndarray | None = None
    idle_started_at: float | None = None
    last_motion_score: float = 0.0

    def process(
        self,
        top_predictions: list[dict[str, Any]],
        landmarks: np.ndarray | None = None,
        force_finalize: bool = False,
    ) -> dict[str, Any]:
        now = monotonic()
        locked_word = self._process_prediction(top_predictions, now)
        idle_duration = self._update_motion(landmarks, now)

        eos_trigger: str | None = None
        if force_finalize:
            eos_trigger = "keypress"
        elif locked_word and locked_word.lower() in EOS_LABELS:
            eos_trigger = "dedicated_sign"
            if self.words and self.words[-1].lower() in EOS_LABELS:
                self.words.pop()
        elif self.words and idle_duration >= self.config.idle_seconds_to_finalize:
            eos_trigger = "idle"

        finalized_sentence = self.finalize(eos_trigger) if eos_trigger else None

        return {
            "candidate": self.candidate_label,
            "candidate_confidence": self.candidate_confidence,
            "candidate_hits": self.candidate_hits,
            "locked_word": locked_word,
            "words": list(self.words),
            "raw_sentence": self.raw_sentence,
            "finalized_sentence": finalized_sentence,
            "finalized_sentences": list(self.finalized_sentences),
            "eos_trigger": eos_trigger,
            "idle_seconds": idle_duration,
            "motion_score": self.last_motion_score,
            "next_word": self.predict_next_word(top_predictions),
            "status": self._status(locked_word, finalized_sentence),
        }

    def finalize(self, trigger: str | None = None) -> str | None:
        if not self.words:
            return None

        sentence = self.raw_sentence
        self.finalized_sentences.append(sentence)
        self.words.clear()
        self.candidate_label = None
        self.candidate_hits = 0
        self.candidate_confidence = 0.0
        self.idle_started_at = None
        return sentence

    def reset(self) -> None:
        self.words.clear()
        self.finalized_sentences.clear()
        self.candidate_label = None
        self.candidate_confidence = 0.0
        self.candidate_hits = 0
        self.last_locked_at = 0.0
        self.last_frame = None
        self.idle_started_at = None
        self.last_motion_score = 0.0

    @property
    def raw_sentence(self) -> str:
        return " ".join(self.words)

    def predict_next_word(self, top_predictions: list[dict[str, Any]]) -> str | None:
        if self.words:
            hints = NEXT_WORD_HINTS.get(self.words[-1].lower())
            if hints:
                return hints[0]

        for prediction in top_predictions:
            label = str(prediction.get("label") or "")
            if label and (not self.words or label != self.words[-1]):
                return label

        return None

    def _process_prediction(self, top_predictions: list[dict[str, Any]], now: float) -> str | None:
        if not top_predictions:
            self._clear_candidate()
            return None

        top_prediction = top_predictions[0]
        label = str(top_prediction.get("label") or "").strip()
        confidence = float(top_prediction.get("confidence") or 0.0)

        if not label or confidence < self.config.confidence_threshold:
            self._clear_candidate()
            return None

        if label != self.candidate_label:
            self.candidate_label = label
            self.candidate_confidence = confidence
            self.candidate_hits = 1
            return None

        self.candidate_hits += 1
        self.candidate_confidence = max(self.candidate_confidence, confidence)

        if self.candidate_hits < self.config.stable_prediction_count:
            return None

        if now - self.last_locked_at < self.config.cooldown_seconds:
            return None

        if self.words and self.words[-1] == label:
            return None

        self.words.append(label)
        self.last_locked_at = now
        self.candidate_hits = 0
        return label

    def _update_motion(self, landmarks: np.ndarray | None, now: float) -> float:
        if landmarks is None:
            self.idle_started_at = None
            self.last_motion_score = 0.0
            return 0.0

        frame = np.asarray(landmarks, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[-1]

        if self.last_frame is None or self.last_frame.shape != frame.shape:
            self.last_frame = frame
            self.idle_started_at = now
            self.last_motion_score = 0.0
            return 0.0

        self.last_motion_score = float(np.mean(np.abs(frame - self.last_frame)))
        self.last_frame = frame

        if self.last_motion_score <= self.config.stillness_epsilon:
            if self.idle_started_at is None:
                self.idle_started_at = now
            return now - self.idle_started_at

        self.idle_started_at = None
        return 0.0

    def _clear_candidate(self) -> None:
        self.candidate_label = None
        self.candidate_confidence = 0.0
        self.candidate_hits = 0

    @staticmethod
    def _status(locked_word: str | None, finalized_sentence: str | None) -> str:
        if finalized_sentence:
            return "sentence_finalized"
        if locked_word:
            return "word_locked"
        return "listening"
