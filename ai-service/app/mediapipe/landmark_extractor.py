"""
SignBridge - Landmark Extraction

Purpose:
    Extract MediaPipe hand landmarks and convert them
    into numerical feature vectors.

Output:
    List of x, y, z coordinates.

Example:
    [
        [0.45, 0.22, -0.01],
        [0.47, 0.25, -0.03],
        ...
    ]

Future Usage:
    Transformer Sign Recognition Model
"""

import cv2
import mediapipe as mp


mp_holistic = mp.solutions.holistic


def extract_hand_landmarks(hand_landmarks):
    """
    Convert MediaPipe landmarks into coordinate list.

    Parameters
    ----------
    hand_landmarks : MediaPipe landmark object

    Returns
    -------
    list
        List of [x, y, z] coordinates.
    """

    coordinates = []

    for landmark in hand_landmarks.landmark:

        coordinates.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return coordinates


def main():

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = holistic.process(rgb)

            if results.right_hand_landmarks:

                coordinates = extract_hand_landmarks(
                    results.right_hand_landmarks
                )

                print(
                    f"Detected {len(coordinates)} landmarks"
                )

            cv2.imshow(
                "Landmark Extraction",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()