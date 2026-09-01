import json
import os
import sys
from pathlib import Path
from threading import Lock

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf


ROWS_PER_FRAME = 543
DIMS = 3
FEATURE_DIM = ROWS_PER_FRAME * DIMS

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "emotion_classifier.tflite"
LABEL_MAP_PATH = MODEL_DIR / "emotion_to_prediction_index_map.json"

_interpreter: tf.lite.Interpreter | None = None
_input_detail: dict | None = None
_output_detail: dict | None = None
_index_to_emotion: dict[int, str] | None = None
_interpreter_lock = Lock()


def normalize_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize supported inputs to float32 [T, 543, 3]."""
    array = np.asarray(sequence, dtype=np.float32)

    if array.ndim == 1:
        if array.size % FEATURE_DIM != 0:
            raise ValueError(
                f"Flat landmark payload length must be divisible by {FEATURE_DIM}."
            )
        array = array.reshape(-1, ROWS_PER_FRAME, DIMS)
    elif array.ndim == 2:
        if array.shape[1] == FEATURE_DIM:
            # SignBridge saved sequence: [T, 1629].
            array = array.reshape(array.shape[0], ROWS_PER_FRAME, DIMS)
        elif array.shape[1] == DIMS and array.shape[0] % ROWS_PER_FRAME == 0:
            # GISLR-style rows: [T * 543, 3].
            array = array.reshape(-1, ROWS_PER_FRAME, DIMS)
        else:
            raise ValueError(
                "2D landmark payload must be shaped as "
                f"[T, {FEATURE_DIM}] or [T * {ROWS_PER_FRAME}, {DIMS}]; "
                f"received {array.shape}."
            )
    elif array.ndim != 3:
        raise ValueError("Landmark payload must be a 1D, 2D, or 3D array.")

    if array.shape[1:] != (ROWS_PER_FRAME, DIMS):
        raise ValueError(
            f"Landmark sequence must be [T, {ROWS_PER_FRAME}, {DIMS}]; "
            f"received {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError("Landmark sequence must contain at least one frame.")

    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def load_sequence(path: Path) -> np.ndarray:
    """Load a SignBridge .npy or supported landmark Parquet sequence."""
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return normalize_landmark_sequence(np.load(path, allow_pickle=False))

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        if {"x", "y", "z"}.issubset(frame.columns):
            array = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
        else:
            numeric = frame.select_dtypes(include=[np.number])
            array = numeric.to_numpy(dtype=np.float32)
        return normalize_landmark_sequence(array)

    raise ValueError(
        f"Unsupported sequence file {path}. Use a .npy or .parquet file."
    )


def _get_labels() -> dict[int, str]:
    global _index_to_emotion

    if _index_to_emotion is None:
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as file:
            emotion_to_index = json.load(file)
        _index_to_emotion = {
            int(index): emotion
            for emotion, index in emotion_to_index.items()
        }

    return _index_to_emotion


def _get_interpreter() -> tuple[tf.lite.Interpreter, dict, dict]:
    global _interpreter, _input_detail, _output_detail

    if _interpreter is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"TFLite model not found: {MODEL_PATH}")

        _interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_detail = _interpreter.get_input_details()[0]
        _output_detail = _interpreter.get_output_details()[0]

    if _input_detail is None or _output_detail is None:
        raise RuntimeError("TFLite interpreter details were not initialized.")

    return _interpreter, _input_detail, _output_detail


def predict_topk_from_array(sequence: np.ndarray, k: int = 3) -> dict:
    """Run the emotion model and return the same top-k schema as the sign model."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    index_to_emotion = _get_labels()
    model_input = normalize_landmark_sequence(sequence)

    global _input_detail, _output_detail

    with _interpreter_lock:
        interpreter, input_detail, output_detail = _get_interpreter()

        if tuple(input_detail["shape"]) != tuple(model_input.shape):
            interpreter.resize_tensor_input(
                input_detail["index"],
                model_input.shape,
                strict=False,
            )
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            _input_detail = input_detail
            _output_detail = output_detail

        if input_detail["dtype"] != np.float32:
            raise TypeError(
                f"Expected a float32 TFLite input, received {input_detail['dtype']}."
            )

        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_detail["index"]).reshape(-1)

    if len(index_to_emotion) != probabilities.shape[0]:
        raise ValueError(
            f"Label map has {len(index_to_emotion)} classes but the model "
            f"returned {probabilities.shape[0]}."
        )

    k = min(k, probabilities.shape[0])
    top_indices = np.argsort(probabilities)[-k:][::-1]
    predictions = [
        {
            "rank": rank,
            "label": index_to_emotion[int(index)],
            "confidence": float(probabilities[index]),
        }
        for rank, index in enumerate(top_indices, start=1)
    ]

    return {
        "output": {
            "num_classes": int(probabilities.shape[0]),
            "top_k": k,
            "predictions": predictions,
        }
    }


def predict_emotion_from_array(sequence: np.ndarray) -> dict:
    """Return the raw top-1 emotion for the pipeline stabilizer."""
    prediction = predict_topk_from_array(sequence, k=1)["output"]["predictions"][0]
    return {
        "label": prediction["label"],
        "confidence": prediction["confidence"],
    }


def predict_topk(path: Path, k: int = 3) -> dict:
    return predict_topk_from_array(load_sequence(path), k=k)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m emotion_classifier.predict_emotion "
            "path/to/sequence.npy [k]"
        )

    sequence_path = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    sequence = load_sequence(sequence_path)
    result = predict_topk_from_array(sequence, k=k)

    interpreter, input_detail, output_detail = _get_interpreter()
    del interpreter

    payload = {
        "file": str(sequence_path),
        "input_shape": list(sequence.shape),
        "input_shape_signature": input_detail["shape_signature"].tolist(),
        "output_shape": output_detail["shape"].tolist(),
        **result,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
