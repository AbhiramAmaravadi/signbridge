from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp


@dataclass(frozen=True)
class HolisticConfig:
    model_complexity: int = 1
    smooth_landmarks: bool = True
    refine_face_landmarks: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class HolisticDetector:
    """MediaPipe Holistic wrapper for real-time frame processing."""

    def __init__(self, config: HolisticConfig | None = None) -> None:
        self.config = config or HolisticConfig()
        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=self.config.smooth_landmarks,
            enable_segmentation=False,
            refine_face_landmarks=self.config.refine_face_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def process(self, frame_bgr: cv2.typing.MatLike):
        frame_bgr.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(frame_rgb)
        frame_bgr.flags.writeable = True
        return results

    def draw_landmarks(self, frame_bgr: cv2.typing.MatLike, results) -> cv2.typing.MatLike:
        self._mp_drawing.draw_landmarks(
            frame_bgr,
            results.face_landmarks,
            self._mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
        )
        self._mp_drawing.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            self._mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
        )
        self._mp_drawing.draw_landmarks(
            frame_bgr,
            results.left_hand_landmarks,
            self._mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self._mp_styles.get_default_hand_connections_style(),
        )
        self._mp_drawing.draw_landmarks(
            frame_bgr,
            results.right_hand_landmarks,
            self._mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self._mp_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self._mp_styles.get_default_hand_connections_style(),
        )
        return frame_bgr

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
