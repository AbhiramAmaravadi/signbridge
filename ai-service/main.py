from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from mediapipe_pipeline import FeatureExtractor, HolisticDetector, SequenceBuffer, Webcam
from mediapipe_pipeline.holistic_detector import HolisticConfig
from mediapipe_pipeline.webcam import WebcamConfig


WINDOW_NAME = "SignBridge MediaPipe Holistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SignBridge real-time MediaPipe Holistic webcam demo."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--save-stride",
        type=int,
        default=30,
        help="Save one full sequence every N processed frames after the buffer fills.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "saved_sequences",
    )
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--refine-face-landmarks", action="store_true")
    return parser.parse_args()


def draw_status(frame, fps: float, buffer_shape: tuple[int, int], last_saved: Path | None) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Buffer: {buffer_shape[0]} x {buffer_shape[1]}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if last_saved is not None:
        cv2.putText(
            frame,
            f"Saved: {last_saved.name}",
            (12, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    webcam_config = WebcamConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
    )
    holistic_config = HolisticConfig(
        model_complexity=args.model_complexity,
        refine_face_landmarks=args.refine_face_landmarks,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    extractor = FeatureExtractor()
    buffer = SequenceBuffer(
        sequence_length=args.sequence_length,
        feature_dimension=FeatureExtractor.FEATURE_DIMENSION,
        output_dir=args.output_dir,
        save_stride=args.save_stride,
    )

    last_frame_time = time.perf_counter()
    fps = 0.0
    last_saved: Path | None = None

    print("Starting SignBridge webcam demo. Press 'q' to quit.")
    print(f"Feature dimension: {FeatureExtractor.FEATURE_DIMENSION}")
    print(f"Sequence shape: ({args.sequence_length}, {FeatureExtractor.FEATURE_DIMENSION})")
    print(f"Saving sequences to: {args.output_dir}")

    try:
        with Webcam(webcam_config) as webcam, HolisticDetector(holistic_config) as detector:
            while True:
                ok, frame = webcam.read()
                if not ok or frame is None:
                    print("Ignoring empty camera frame.")
                    continue

                now = time.perf_counter()
                elapsed = now - last_frame_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                last_frame_time = now

                results = detector.process(frame)
                feature_vector = extractor.extract(results)
                saved_path = buffer.add(feature_vector)
                if saved_path is not None:
                    last_saved = saved_path
                    print(f"Saved sequence: {saved_path}")

                detector.draw_landmarks(frame, results)
                draw_status(frame, fps, buffer.shape, last_saved)
                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
