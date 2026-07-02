from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class WebcamConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    buffer_size: int = 1


class Webcam:
    """Thin OpenCV webcam wrapper with explicit resource cleanup."""

    def __init__(self, config: WebcamConfig | None = None) -> None:
        self.config = config or WebcamConfig()
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open webcam at index {self.config.camera_index}."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        self._capture = capture

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        if self._capture is None:
            raise RuntimeError("Webcam must be opened before reading frames.")
        return self._capture.read()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Webcam":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
