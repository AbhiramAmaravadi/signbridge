import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


ROWS_PER_FRAME = 543
DIMS = 3


def load_parquet_sequence(path: Path) -> np.ndarray:
    df = pd.read_parquet(path, columns=["x", "y", "z"])
    arr = df.to_numpy(dtype=np.float32)

    n_frames = len(arr) // ROWS_PER_FRAME
    arr = arr[: n_frames * ROWS_PER_FRAME]
    arr = arr.reshape(n_frames, ROWS_PER_FRAME, DIMS)

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def predict_topk(parquet_path: Path, k: int = 5) -> dict:
    with open("sign_to_prediction_index_map.json", "r", encoding="utf-8") as f:
        sign2ord = json.load(f)

    ord2sign = {int(v): label for label, v in sign2ord.items()}

    x = load_parquet_sequence(parquet_path)

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    interpreter.resize_tensor_input(input_detail["index"], x.shape, strict=False)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_detail["index"], x)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_detail["index"]).reshape(-1)
    top_idx = np.argsort(probs)[-k:][::-1]

    predictions = [
        {
            "rank": rank,
            "label": ord2sign[int(idx)],
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


def main():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    print(interpreter.get_input_details())
    print(interpreter.get_output_details())
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict_topk_json.py path/to/sample.parquet [k]")

    parquet_path = Path(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    result = predict_topk(parquet_path, k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()