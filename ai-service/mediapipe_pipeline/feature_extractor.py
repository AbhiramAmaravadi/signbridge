from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LandmarkGroupSpec:
    name: str
    count: int


class FeatureExtractor:
    """Converts MediaPipe Holistic results into a fixed-size numpy vector."""

    LANDMARK_GROUPS = (
        LandmarkGroupSpec("face", 468),
        LandmarkGroupSpec("pose", 33),
        LandmarkGroupSpec("left_hand", 21),
        LandmarkGroupSpec("right_hand", 21),
    )
    VALUES_PER_LANDMARK = 3
    FEATURE_DIMENSION = sum(group.count for group in LANDMARK_GROUPS) * VALUES_PER_LANDMARK

    def extract(self, results) -> np.ndarray:
        vectors = [
            self._landmarks_to_vector(results.face_landmarks, 468),
            self._landmarks_to_vector(results.pose_landmarks, 33),
            self._landmarks_to_vector(results.left_hand_landmarks, 21),
            self._landmarks_to_vector(results.right_hand_landmarks, 21),
        ]
        return np.concatenate(vectors).astype(np.float32, copy=False)

    @classmethod
    def describe_feature_layout(cls) -> list[dict[str, int | str]]:
        offset = 0
        layout = []
        for group in cls.LANDMARK_GROUPS:
            dimensions = group.count * cls.VALUES_PER_LANDMARK
            layout.append(
                {
                    "name": group.name,
                    "landmarks": group.count,
                    "values_per_landmark": cls.VALUES_PER_LANDMARK,
                    "start": offset,
                    "end": offset + dimensions,
                }
            )
            offset += dimensions
        return layout

    @staticmethod
    def _landmarks_to_vector(landmark_list, expected_count: int) -> np.ndarray:
        vector = np.zeros(expected_count * 3, dtype=np.float32)
        if landmark_list is None:
            return vector

        for index, landmark in enumerate(landmark_list.landmark[:expected_count]):
            base = index * 3
            vector[base : base + 3] = (landmark.x, landmark.y, landmark.z)

        return vector
