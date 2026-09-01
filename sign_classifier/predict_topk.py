import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import tensorflow as tf


ROWS_PER_FRAME = 543
DIMS = 3
TARGET_SEQUENCE_LENGTH = 64
MODEL_INPUT_SHAPE = (1, TARGET_SEQUENCE_LENGTH, ROWS_PER_FRAME * DIMS)
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "sign_classifier.tflite"
LABEL_MAP_PATH = MODEL_DIR / "sign_to_prediction_index_map.json"

_interpreter: tf.lite.Interpreter | None = None
_input_detail: dict | None = None
_output_detail: dict | None = None
_index_to_label: dict[int, str] | None = None
_interpreter_lock = Lock()


def normalize_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % (ROWS_PER_FRAME * DIMS) != 0:
            raise ValueError(
                "Flat landmark payload length must be divisible by "
                f"{ROWS_PER_FRAME * DIMS}."
            )
        arr = arr.reshape(-1, ROWS_PER_FRAME, DIMS)
    elif arr.ndim == 2:
        if arr.shape[1] == ROWS_PER_FRAME * DIMS:
            arr = arr.reshape(arr.shape[0], ROWS_PER_FRAME, DIMS)
        elif arr.shape[1] == DIMS and arr.shape[0] % ROWS_PER_FRAME == 0:
            arr = arr.reshape(-1, ROWS_PER_FRAME, DIMS)
        else:
            raise ValueError(
                "2D landmark payload must be shaped as "
                f"(frames, {ROWS_PER_FRAME * DIMS}) or "
                f"(frames * {ROWS_PER_FRAME}, {DIMS})."
            )
    elif arr.ndim != 3:
        raise ValueError("Landmark payload must be a 1D, 2D, or 3D array.")

    if arr.shape[1:] != (ROWS_PER_FRAME, DIMS):
        raise ValueError(
            "Landmark sequence must be shaped as "
            f"(frames, {ROWS_PER_FRAME}, {DIMS}); got {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("Landmark sequence must contain at least one frame.")

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_parquet_sequence(path: Path) -> np.ndarray:
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    return normalize_landmark_sequence(df.to_numpy(dtype=np.float32))


def interpolate_sequence(sequence: np.ndarray) -> np.ndarray:
    """Linearly resample a landmark sequence to the v2 model's 64 frames."""
    sequence = normalize_landmark_sequence(sequence)
    if sequence.shape[0] == TARGET_SEQUENCE_LENGTH:
        return sequence

    source_positions = np.linspace(0, sequence.shape[0] - 1, TARGET_SEQUENCE_LENGTH)
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.ceil(source_positions).astype(np.int64)
    weights = (source_positions - lower).astype(np.float32)[:, None, None]
    return (
        sequence[lower] * (1.0 - weights) + sequence[upper] * weights
    ).astype(np.float32, copy=False)


def _get_labels() -> dict[int, str]:
    global _index_to_label

    if _index_to_label is None:
        with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
            sign2ord = json.load(f)
        _index_to_label = {int(index): label for label, index in sign2ord.items()}

    return _index_to_label


def _get_interpreter() -> tuple[tf.lite.Interpreter, dict, dict]:
    global _interpreter, _input_detail, _output_detail

    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        _interpreter.allocate_tensors()
        _input_detail = _interpreter.get_input_details()[0]
        _output_detail = _interpreter.get_output_details()[0]
        input_shape = tuple(int(value) for value in _input_detail["shape"])
        if input_shape != MODEL_INPUT_SHAPE:
            raise RuntimeError(
                "v2 sign model must expose a fixed input shape "
                f"{MODEL_INPUT_SHAPE}; received {input_shape}."
            )

    if _input_detail is None or _output_detail is None:
        raise RuntimeError("TFLite interpreter details were not initialized.")

    return _interpreter, _input_detail, _output_detail


def predict_topk_from_array(sequence: np.ndarray, k: int = 5) -> dict:
    index_to_label = _get_labels()
    x = interpolate_sequence(sequence).reshape(MODEL_INPUT_SHAPE)

    with _interpreter_lock:
        interpreter, input_detail, output_detail = _get_interpreter()
        interpreter.set_tensor(input_detail["index"], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_detail["index"]).reshape(-1)

    if len(index_to_label) != probs.shape[0]:
        raise ValueError(
            f"v2 label map has {len(index_to_label)} classes but the model "
            f"returned {probs.shape[0]} output values."
        )

    k = min(k, probs.shape[0])
    top_idx = np.argsort(probs)[-k:][::-1]

    predictions = [
        {
            "rank": rank,
            "label": index_to_label[int(idx)],
            "confidence": float(probs[idx]),
        }
        for rank, idx in enumerate(top_idx, start=1)
    ]

    return {
        "output": {
            "num_classes": int(probs.shape[0]),
            "top_k": k,
            "predictions": predictions,
        }
    }


def predict_topk(parquet_path: Path, k: int = 5) -> dict:
    x = load_parquet_sequence(parquet_path)
    return predict_topk_from_array(x, k)


def main():
    global _interpreter
    _interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    print(_interpreter.get_input_details())
    print(_interpreter.get_output_details())
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict_topk_json.py path/to/sample.parquet [k]")

    parquet_path = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    result = predict_topk(parquet_path, k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
