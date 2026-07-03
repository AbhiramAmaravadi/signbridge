import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict_topk import ROWS_PER_FRAME, predict_topk

app = FastAPI()


class InferenceRequest(BaseModel):
    file_path: str


class InferenceResponse(BaseModel):
    timestamp: str
    sequence_length: int
    top_k: list[dict]


@app.post("/api/v1/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    if not request.file_path or not request.file_path.strip():
        raise HTTPException(status_code=400, detail="The 'file_path' field is required and must be a non-empty string.")

    script_dir = Path(__file__).resolve().parent
    dataset_path = Path(request.file_path)
    if not dataset_path.is_absolute():
        dataset_path = script_dir.parent / dataset_path

    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Parquet file not found: {dataset_path}",
        )

    os.chdir(script_dir)

    try:
        result = predict_topk(dataset_path, k=5)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    predictions = result["output"]["predictions"]
    top_k = [
        {"label": prediction["label"], "confidence": prediction["confidence"]}
        for prediction in predictions
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence_length": ROWS_PER_FRAME,
        "top_k": top_k,
    }
