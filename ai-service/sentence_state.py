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
    "look": ["here", "at me", "there"],
    "shhh": ["here", "at me", "there"],
    "hello": ["how", "are", "you"],
    "happy": ["to", "see", "you"],
    "flower": ["is", "beautiful", "outside"],
}


@dataclass
class SentenceStateConfig:
    # A prediction must be both confident and sustained before it can affect the
    # sentence.  This filters the noisy outputs commonly produced while a hand
    # is entering or leaving the camera frame.
    confidence_threshold: float = 0.35
    stable_prediction_count: int = 8
    confidence_std_threshold: float = 0.045
    stable_prediction_seconds: float = 0.0
    cooldown_seconds: float = 1.0
    idle_seconds_to_finalize: float = 5.0
    release_idle_seconds: float = 0.7
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
    candidate_started_at: float | None = None
    confidence_history: list[float] = field(default_factory=list)
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
        idle_duration = self._update_motion(landmarks, now)
        hands_present = self._hands_present(landmarks)
        hands_resting = self._hands_resting(landmarks)
        locked_word = self._process_prediction(
            top_predictions,
            now,
            suppress=hands_present is False,
        )

        eos_trigger: str | None = None
        if force_finalize:
            eos_trigger = "keypress"
        elif locked_word and locked_word.lower() in EOS_LABELS:
            eos_trigger = "dedicated_sign"
            if self.words and self.words[-1].lower() in EOS_LABELS:
                self.words.pop()
        elif self.words and (
            hands_present is False
            or (hands_resting is True and idle_duration >= self.config.release_idle_seconds)
        ):
            eos_trigger = "resting_posture"
        elif self.words and idle_duration >= self.config.idle_seconds_to_finalize:
            eos_trigger = "idle"

        finalized_sentence = self.finalize(eos_trigger) if eos_trigger else None

        return {
            "candidate": self.candidate_label,
            "candidate_confidence": self.candidate_confidence,
            "candidate_hits": self.candidate_hits,
            "lock_progress": self._lock_progress(now),
            "locked_word": locked_word,
            "words": list(self.words),
            "raw_sentence": self.raw_sentence,
            "finalized_sentence": finalized_sentence,
            "finalized_sentences": list(self.finalized_sentences),
            "eos_trigger": eos_trigger,
            "idle_seconds": idle_duration,
            "motion_score": self.last_motion_score,
            "next_words": self.predict_next_words(top_predictions, hands_present),
            "next_word": self.predict_next_word(top_predictions, hands_present),
            "status": self._status(locked_word, finalized_sentence, hands_present),
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
        self.candidate_started_at = None
        self.confidence_history.clear()
        self.idle_started_at = None
        return sentence

    def reset(self) -> None:
        self.words.clear()
        self.finalized_sentences.clear()
        self.candidate_label = None
        self.candidate_confidence = 0.0
        self.candidate_hits = 0
        self.candidate_started_at = None
        self.confidence_history.clear()
        self.last_locked_at = 0.0
        self.last_frame = None
        self.idle_started_at = None
        self.last_motion_score = 0.0

    @property
    def raw_sentence(self) -> str:
        return " ".join(self.words)

    def predict_next_words(
        self,
        top_predictions: list[dict[str, Any]],
        hands_present: bool | None = None,
    ) -> list[str]:
        """Return up to three context-aware suggestions for the live buffer."""
        if hands_present is False:
            return []

        suggestions: list[str] = []
        if self.words:
            suggestions.extend(NEXT_WORD_HINTS.get(self.words[-1].lower(), []))

        for prediction in top_predictions:
            label = str(prediction.get("label") or "")
            confidence = float(prediction.get("confidence") or 0.0)
            if (
                label
                and confidence >= self.config.confidence_threshold
                and (not self.words or label != self.words[-1])
                and label not in suggestions
            ):
                suggestions.append(label)
            if len(suggestions) >= 3:
                break

        return suggestions[:3]

    def predict_next_word(
        self,
        top_predictions: list[dict[str, Any]],
        hands_present: bool | None = None,
    ) -> str | None:
        suggestions = self.predict_next_words(top_predictions, hands_present)
        return suggestions[0] if suggestions else None

    def append_word(self, word: str) -> str | None:
        """Append a user-confirmed suggestion without waiting for a gesture."""
        normalized = " ".join(word.strip().split())
        if not normalized or (self.words and self.words[-1].lower() == normalized.lower()):
            return None

        self.words.append(normalized)
        self._clear_candidate()
        self.idle_started_at = None
        return normalized

    def _process_prediction(
        self,
        top_predictions: list[dict[str, Any]],
        now: float,
        suppress: bool = False,
    ) -> str | None:
        if suppress or not top_predictions:
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
            self.candidate_started_at = now
            self.confidence_history = [confidence]
            return None

        self.candidate_hits += 1
        self.confidence_history.append(confidence)
        self.confidence_history = self.confidence_history[-self.config.stable_prediction_count :]
        self.candidate_confidence = float(np.mean(self.confidence_history))

        stable_duration = now - (self.candidate_started_at or now)
        confidence_std = float(np.std(self.confidence_history))
        if (
            self.candidate_hits < self.config.stable_prediction_count
            or stable_duration < self.config.stable_prediction_seconds
            or len(self.confidence_history) < self.config.stable_prediction_count
            or confidence_std > self.config.confidence_std_threshold
        ):
            return None

        if now - self.last_locked_at < self.config.cooldown_seconds:
            return None

        if self.words and self.words[-1] == label:
            return None

        self.words.append(label)
        self.last_locked_at = now
        self.candidate_hits = 0
        self.candidate_started_at = None
        self.confidence_history.clear()
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

    @staticmethod
    def _hands_present(landmarks: np.ndarray | None) -> bool | None:
        """Return whether either MediaPipe hand landmark group contains data.

        ``None`` means landmark presence is unavailable (for example, callers
        supplying predictions without camera landmarks), so confidence gating
        remains the safe fallback.
        """
        if landmarks is None:
            return None

        frame = np.asarray(landmarks, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[-1]
        if frame.shape != (543, 3):
            return None

        left_hand = frame[468:489]
        right_hand = frame[522:543]
        return bool(np.any(np.abs(left_hand) > 1e-6) or np.any(np.abs(right_hand) > 1e-6))

    @staticmethod
    def _hands_resting(landmarks: np.ndarray | None) -> bool | None:
        """Detect hands lowered to chest level or absent from the frame."""
        if landmarks is None:
            return None

        frame = np.asarray(landmarks, dtype=np.float32)
        if frame.ndim == 3:
            frame = frame[-1]
        if frame.shape != (543, 3):
            return None

        left_hand = frame[468:489]
        right_hand = frame[522:543]
        hand_groups = [group for group in (left_hand, right_hand) if np.any(np.abs(group) > 1e-6)]
        if not hand_groups:
            return True

        pose = frame[489:522]
        shoulder_points = pose[[11, 12]]
        if not np.any(np.abs(shoulder_points) > 1e-6):
            return None

        shoulder_y = float(np.mean(shoulder_points[:, 1]))
        hand_y = float(np.mean([np.mean(group[:, 1]) for group in hand_groups]))
        return hand_y >= shoulder_y - 0.02

    def _clear_candidate(self) -> None:
        self.candidate_label = None
        self.candidate_confidence = 0.0
        self.candidate_hits = 0
        self.candidate_started_at = None
        self.confidence_history.clear()

    def _lock_progress(self, now: float) -> float:
        if not self.candidate_label or self.candidate_started_at is None:
            return 0.0

        elapsed = now - self.candidate_started_at
        duration_progress = (
            1.0
            if self.config.stable_prediction_seconds <= 0
            else elapsed / self.config.stable_prediction_seconds
        )
        count_progress = self.candidate_hits / max(1, self.config.stable_prediction_count)
        return max(0.0, min(1.0, min(duration_progress, count_progress)))

    @staticmethod
    def _status(
        locked_word: str | None,
        finalized_sentence: str | None,
        hands_present: bool | None,
    ) -> str:
        if finalized_sentence:
            return "sentence_finalized"
        if locked_word:
            return "word_locked"
        if hands_present is False:
            return "idle"
        if hands_present is None:
            return "searching"
        return "listening"
