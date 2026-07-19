from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class StageStats:
    """Aggregated timing values for one real-time pipeline stage."""

    count: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    min_ms: float = field(default=float("inf"))
    max_ms: float = 0.0

    def add(self, elapsed_ms: float) -> None:
        self.count += 1
        self.total_ms += elapsed_ms
        self.last_ms = elapsed_ms
        self.min_ms = min(self.min_ms, elapsed_ms)
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "last_ms": self.last_ms,
            "average_ms": self.average_ms,
            "min_ms": 0.0 if self.count == 0 else self.min_ms,
            "max_ms": self.max_ms,
            "total_ms": self.total_ms,
        }


class RealtimeMetrics:
    """Reusable timers for webcam MediaPipe and model inference pipelines.

    Example:
        metrics = RealtimeMetrics()
        metrics.start("mediapipe")
        ...
        metrics.stop("mediapipe")

        with metrics.track("inference"):
            ...

        metrics.tick_frame()
        print(metrics.report())
    """

    DEFAULT_STAGES = (
        "mediapipe",
        "feature_extraction",
        "inference",
        "post_processing",
        "end_to_end",
    )

    def __init__(self, fps_window: int = 120) -> None:
        if fps_window <= 1:
            raise ValueError("fps_window must be greater than 1.")

        self._active: dict[str, float] = {}
        self._stats: defaultdict[str, StageStats] = defaultdict(StageStats)
        self._frame_times: deque[float] = deque(maxlen=fps_window)

        for stage in self.DEFAULT_STAGES:
            self._stats[stage]

    def start(self, stage: str) -> None:
        """Start timing a named stage."""
        if stage in self._active:
            raise RuntimeError(f"Timer for stage '{stage}' is already running.")
        self._active[stage] = perf_counter()

    def stop(self, stage: str) -> float:
        """Stop timing a named stage and return elapsed milliseconds."""
        started_at = self._active.pop(stage, None)
        if started_at is None:
            raise RuntimeError(f"Timer for stage '{stage}' was not started.")

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        self._stats[stage].add(elapsed_ms)
        return elapsed_ms

    @contextmanager
    def track(self, stage: str) -> Iterator[None]:
        """Context manager form of start/stop."""
        self.start(stage)
        try:
            yield
        finally:
            self.stop(stage)

    def tick_frame(self) -> None:
        """Record one completed frame for rolling FPS calculation."""
        self._frame_times.append(perf_counter())

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0

        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._frame_times) - 1) / elapsed

    def report(self) -> dict[str, object]:
        """Return all timing statistics as a serializable dictionary."""
        return {
            "fps": self.fps,
            "stages": {stage: stats.as_dict() for stage, stats in self._stats.items()},
            "active_stages": sorted(self._active),
        }

    def reset(self) -> None:
        """Clear all collected timings and active timers."""
        self._active.clear()
        self._stats.clear()
        self._frame_times.clear()

        for stage in self.DEFAULT_STAGES:
            self._stats[stage]
