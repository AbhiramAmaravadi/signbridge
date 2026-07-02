from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class SequenceBuffer:
    """Maintains a rolling frame-feature buffer and persists full sequences."""

    def __init__(
        self,
        sequence_length: int,
        feature_dimension: int,
        output_dir: str | Path,
        save_stride: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be greater than zero.")

        self.sequence_length = sequence_length
        self.feature_dimension = feature_dimension
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_stride = save_stride or sequence_length
        self._buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        self._frames_since_save = 0

    def add(self, feature_vector: np.ndarray) -> Path | None:
        if feature_vector.shape != (self.feature_dimension,):
            raise ValueError(
                "feature_vector shape must be "
                f"({self.feature_dimension},), got {feature_vector.shape}."
            )

        self._buffer.append(feature_vector.astype(np.float32, copy=False))
        self._frames_since_save += 1

        if self.is_full and self._frames_since_save >= self.save_stride:
            self._frames_since_save = 0
            return self.save()

        return None

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.sequence_length

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._buffer), self.feature_dimension)

    def to_numpy(self) -> np.ndarray:
        if not self.is_full:
            raise RuntimeError(
                f"Sequence buffer is not full: {len(self._buffer)}/"
                f"{self.sequence_length} frames."
            )
        return np.stack(tuple(self._buffer), axis=0).astype(np.float32, copy=False)

    def save(self) -> Path:
        sequence = self.to_numpy()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = self.output_dir / f"sequence_{timestamp}.npy"
        np.save(output_path, sequence)
        return output_path
