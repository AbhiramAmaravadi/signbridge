# SignBridge

Real-time sign language recognition with a React frontend, a Spring Boot backend, and a Python MediaPipe + TensorFlow Lite AI service.

## Project Structure

```text
signbridge/
  frontend/      React + Vite web app
  backend/       Java 21 Spring Boot API
  ai-service/    Python MediaPipe webcam/API service
  transformer/   TensorFlow Lite sign recognition model and inference helpers
  dataset/       Local landmark parquet dataset
```

## Prerequisites

- Node.js 20+ for the Vite frontend
- Java 21 for the Spring Boot backend
- Maven wrapper from `backend/mvnw.cmd`
- Python 3.11 recommended for the AI service
- PostgreSQL running locally if you use the Java backend database features

The backend currently expects:

```text
Database : jdbc:postgresql://localhost:5432/signbridge
Username : postgres
Password : 1234
```

Create the database before starting Spring Boot:

```sql
CREATE DATABASE signbridge;
```

## Install Dependencies

### Frontend

```powershell
cd signbridge\frontend
npm install
```

### Java Backend

Make sure Java 21 is active:

```powershell
java -version
```

The Maven project uses Spring Boot `4.1.0` and `<java.version>21</java.version>`, so Java 8, 11, or 17 can fail with `Process terminated with exit code: 1`.

### AI Service

```powershell
cd signbridge\ai-service
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is not available, install Python 3.11 or create the virtual environment with your installed Python executable.

## Run The System

Open three terminals.

### Terminal 1: AI Service

This starts the FastAPI wrapper around MediaPipe and the TFLite transformer model.

```powershell
cd signbridge\ai-service
.\.venv\Scripts\Activate.ps1
uvicorn app_server:app --host 127.0.0.1 --port 8001 --reload
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8001/health
```

Prediction endpoint:

```text
POST http://127.0.0.1:8001/predict
POST http://127.0.0.1:8001/api/v1/inference
```

The endpoint accepts either:

```json
{
  "landmarks": [[[0.0, 0.0, 0.0]]],
  "top_k": 5
}
```

or:

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "sequence_length": 45,
  "top_k": 5
}
```

For image frames, the server fills an in-memory rolling buffer and returns `ready: false` until enough frames have arrived.

Sentence and Gemini endpoints:

```text
POST http://127.0.0.1:8001/api/v1/sentence/finalize
POST http://127.0.0.1:8001/api/v1/sentence/reset
POST http://127.0.0.1:8001/api/v1/gemini/translate
POST http://127.0.0.1:8001/api/v1/gemini/scene
GET  http://127.0.0.1:8001/api/v1/gemini/scene-prompt
```

Set `GEMINI_API_KEY` to enable Gemini translation and scene perception. Without a key, the API returns a safe local fallback so the webcam flow still works.

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
uvicorn app_server:app --host 127.0.0.1 --port 8001 --reload
```

The frontend uses the browser Web Speech API for text-to-speech. When a sentence is finalized by Enter, idle detection, or a dedicated done/period sign, the final text is polished and spoken automatically.

### Terminal 2: Java Backend

```powershell
cd signbridge\backend
.\mvnw.cmd spring-boot:run
```

If this fails:

- Confirm `java -version` reports Java 21.
- Confirm PostgreSQL is running and the `signbridge` database exists.
- Confirm port `8080` is free.

### Terminal 3: Frontend

```powershell
cd signbridge\frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://127.0.0.1:5173
```

## AI Service Modes

### FastAPI Server

Use this for browser and service integration:

```powershell
cd signbridge\ai-service
uvicorn app_server:app --host 127.0.0.1 --port 8001 --reload
```

### Local OpenCV Webcam Demo

Use this for local desktop debugging:

```powershell
cd signbridge\ai-service
python main.py
```

The desktop script opens an OpenCV webcam window and saves rolling landmark sequences. Press `q` to quit.

## Transformer Benchmarking

Run latency benchmarks without changing model inference logic:

```powershell
cd signbridge
python transformer\benchmark.py --samples 500 --shuffle
```

Full dataset:

```powershell
python transformer\benchmark.py
```

Benchmark CSV output is written under:

```text
transformer/benchmark_results/
```

## Import Path Notes

The AI service package exports the nested MediaPipe modules through `mediapipe_pipeline/__init__.py`.

Preferred imports:

```python
from mediapipe_pipeline import (
    FeatureExtractor,
    HolisticConfig,
    HolisticDetector,
    SequenceBuffer,
    Webcam,
    WebcamConfig,
)
```

Run AI service commands from:

```text
C:signbridge\ai-service
```

This keeps `uvicorn app_server:app` and `python main.py` on the same clean import path.

## Service URLs

```text
Frontend      http://127.0.0.1:5173
Java backend  http://127.0.0.1:8080
AI service    http://127.0.0.1:8001
Transformer   http://127.0.0.1:8000/api/v1/inference, if transformer/app.py is run separately
```
