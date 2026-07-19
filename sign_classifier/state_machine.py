from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignBufferManager:
    """Manage a stream of sign predictions and produce a stable word queue."""

    confidence_threshold: float = 0.60
    stable_frame_count: int = 3
    cooldown_frames: int = 2
    word_queue: list[str] = field(default_factory=list)
    _current_candidate: str | None = None
    _candidate_hits: int = 0
    _cooldown_remaining: int = 0

    def process_new_prediction(self, top_predictions: Sequence[dict[str, Any]]) -> bool:
        """Process a new frame of model predictions.

        The method only considers the Rank 1 prediction. If the top score is below the
        confidence threshold, the frame is treated as noise. Stable words are appended
        only after they persist for enough consecutive frames and are not duplicates
        of the most recent finalized word.

        Returns True when a new word is appended to the queue.
        """
        if not top_predictions:
            self._decrement_cooldown()
            self._reset_candidate()
            return False

        top_prediction = top_predictions[0]
        label = top_prediction.get("label")
        confidence = float(top_prediction.get("confidence", 0.0))

        if confidence < self.confidence_threshold or not label:
            self._decrement_cooldown()
            self._reset_candidate()
            return False

        if self._current_candidate != label:
            self._current_candidate = label
            self._candidate_hits = 1
            self._decrement_cooldown()
            return False

        self._candidate_hits += 1

        if self._candidate_hits < self.stable_frame_count:
            self._decrement_cooldown()
            return False

        if self.word_queue and self.word_queue[-1] == label:
            self._decrement_cooldown()
            return False

        if self._cooldown_remaining > 0:
            self._decrement_cooldown()
            return False

        self.word_queue.append(label)
        self._cooldown_remaining = self.cooldown_frames
        self._candidate_hits = 0
        self._decrement_cooldown()
        return True

    def get_current_sentence(self) -> str:
        """Return the finalized word stream as a single human-readable sentence."""
        return " ".join(self.word_queue)

    def _reset_candidate(self) -> None:
        self._current_candidate = None
        self._candidate_hits = 0

    def _decrement_cooldown(self) -> None:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1


if __name__ == "__main__":
    manager = SignBufferManager(confidence_threshold=0.60, stable_frame_count=3, cooldown_frames=2)

    sample_stream = [
        [{"label": "hello", "confidence": 0.65}],
        [{"label": "hello", "confidence": 0.70}],
        [{"label": "hello", "confidence": 0.75}],
        [{"label": "hello", "confidence": 0.72}],
        [{"label": "hello", "confidence": 0.74}],
        [{"label": "water", "confidence": 0.80}],
        [{"label": "water", "confidence": 0.82}],
        [{"label": "water", "confidence": 0.85}],
        [{"label": "water", "confidence": 0.88}],
        [{"label": "please", "confidence": 0.55}],
        [{"label": "please", "confidence": 0.62}],
        [{"label": "please", "confidence": 0.68}],
        [{"label": "please", "confidence": 0.72}],
    ]

    print("Starting mock prediction stream...")
    for index, frame in enumerate(sample_stream, start=1):
        appended = manager.process_new_prediction(frame)
        print(
            f"Frame {index}: top={frame[0]['label']} conf={frame[0]['confidence']:.2f} "
            f"-> appended={appended} sentence='{manager.get_current_sentence()}'"
        )

    print("Final queue:", manager.word_queue)
    print("Final sentence:", manager.get_current_sentence())
