"""
SignBridge - MediaPipe Landmark Test

Purpose:
    Detect and visualize hand landmarks using MediaPipe.

Usage:
    python landmark_test.py

Controls:
    Press 'q' to quit.

Expected Result:
    Hand skeleton overlay appears on webcam feed.
"""
import cv2
import mediapipe as mp
# ---------------------------------------------------------
# MediaPipe Initialization
# ---------------------------------------------------------

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def main():
    """
    Capture webcam feed and draw detected hand landmarks.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not access webcam.")
        return

    print("MediaPipe started.")
    print("Press 'q' to exit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Run MediaPipe inference
            results = holistic.process(rgb_frame)

            # Draw left hand landmarks
            if results.left_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            # Draw right hand landmarks
            if results.right_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            cv2.imshow(
                "SignBridge Landmark Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()