"""
SignBridge - Webcam Test

Purpose:
    Verify that OpenCV can access the user's webcam.

Usage:
    python webcam_test.py

Controls:
    Press 'q' to quit.

Expected Result:
    A window displaying the live webcam feed.
"""

import cv2

def main():
    """
    Open webcam and display live video feed.
    """

    # Open default camera (usually laptop webcam)
    cap = cv2.VideoCapture(0)

    # Ensure camera opened successfully
    if not cap.isOpened():
        print("ERROR: Could not access webcam.")
        return

    print("Webcam started successfully.")
    print("Press 'q' to exit.")

    while True:

        # Read current frame
        success, frame = cap.read()

        if not success:
            print("ERROR: Failed to capture frame.")
            break

        # Display frame
        cv2.imshow("SignBridge Webcam Test", frame)

        # Exit on q key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()