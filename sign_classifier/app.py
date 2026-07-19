from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict_topk import ROWS_PER_FRAME, normalize_landmark_sequence, predict_topk, predict_topk_from_array

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)


class InferenceRequest(BaseModel):
    file_path: str | None = None
    landmarks: list[Any] | None = Field(
        default=None,
        description="Live MediaPipe payload shaped as [frames][543][3], [frames][1629], or flattened rows.",
    )


class InferenceResponse(BaseModel):
    timestamp: str
    sequence_length: int
    top_k: list[dict]


@app.post("/api/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    script_dir = Path(__file__).resolve().parent

    if request.landmarks is None and not (request.file_path and request.file_path.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'landmarks' for live inference or 'file_path' for parquet inference.",
        )

    try:
        if request.landmarks is not None:
            sequence = normalize_landmark_sequence(np.asarray(request.landmarks, dtype=np.float32))
            result = predict_topk_from_array(sequence, k=5)
            sequence_length = int(sequence.shape[0])
        else:
            dataset_path = Path(request.file_path or "")
            if not dataset_path.is_absolute():
                dataset_path = script_dir.parent / dataset_path

            if not dataset_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Parquet file not found: {dataset_path}",
                )

            result = predict_topk(dataset_path, k=5)
            sequence_length = ROWS_PER_FRAME
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    predictions = result["output"]["predictions"]
    top_k = [
        {"label": prediction["label"], "confidence": prediction["confidence"]}
        for prediction in predictions
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence_length": sequence_length,
        "top_k": top_k,
    }
