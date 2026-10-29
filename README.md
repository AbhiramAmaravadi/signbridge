<h1 align="center">SignBridge</h1>

<p align="center">
  <strong>Bridging Communication Gaps with Real-Time Sign Language Translation &amp; Machine Learning.</strong>
</p>

<p align="center">
  <a href="https://signbridge-web-two.vercel.app/#overview">🚀 Launch the SignBridge Live Demo</a>
</p>

<p align="center">
  <a href="https://github.com/AbhiramAmaravadi/signbridge/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11 or newer" /></a>
  <a href="https://github.com/AbhiramAmaravadi/signbridge"><img src="https://img.shields.io/github/stars/AbhiramAmaravadi/signbridge?style=flat&logo=github" alt="GitHub stars" /></a>
  <a href="https://github.com/AbhiramAmaravadi/signbridge/actions"><img src="https://img.shields.io/badge/build-not_configured-lightgrey.svg" alt="Build status" /></a>
  <a href="https://youtu.be/WMk-TVJiK8Q"><img src="https://img.shields.io/badge/YouTube-watch_explanation-FF0000.svg?logo=youtube&logoColor=white" alt="Watch the explanation video" /></a>
</p>

## Overview

SignBridge is a modular, real-time sign language recognition and translation platform. It captures video, extracts body/hand/face landmarks, classifies temporal gesture sequences, assembles recognized signs into sentences, and optionally uses Gemini for natural-language refinement and scene-aware translation.

Try the deployed application: [SignBridge Live Demo](https://signbridge-web-two.vercel.app/#overview).

The repository is organized as three cooperating applications:

```text
SignBridge/
├── frontend/          React 19 + Vite web application
├── backend/            Java 21 + Spring Boot API gateway: Gemini, TTS, PostgreSQL history
├── ai-service/         Python FastAPI + MediaPipe + TensorFlow Lite inference service (internal)
├── sign_classifier/    Transformer-based TFLite sign model and evaluation tools
├── emotion_classifier/ TFLite emotion model used by the AI service
└── docs/               Architecture and API notes
```

## Explanation Video

<p align="center">
  <a href="https://youtu.be/WMk-TVJiK8Q"><img src="https://img.youtube.com/vi/WMk-TVJiK8Q/maxresdefault.jpg" alt="Watch the SignBridge explanation video" width="720" /></a>
</p>

The video covers the system architecture, machine-learning pipeline, and a live demonstration of SignBridge.

## Videos

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://signbridge-web-two.vercel.app/media/intro.mp4"><img src="https://signbridge-web-two.vercel.app/media/intro.jpg" alt="Play the 20-second SignBridge introduction" width="100%" /></a>
      <br /><strong>Meet SignBridge</strong> · 0:20, with sound
    </td>
    <td align="center" width="50%">
      <a href="https://signbridge-web-two.vercel.app/media/demo-session.mp4"><img src="https://signbridge-web-two.vercel.app/media/demo-session.jpg" alt="Play the recorded live demo session" width="100%" /></a>
      <br /><strong>Recorded live session</strong> · 0:37: “look”, “wolf” and “shhh” become <em>“Look over there, a wolf! Shhh!”</em>
    </td>
  </tr>
</table>

Click a thumbnail to play the video. Both also play on the website: the introduction on the [Overview](https://signbridge-web-two.vercel.app/#overview) page and the recorded session at the top of the [Live Demo](https://signbridge-web-two.vercel.app/#demo) page.

## Key Features

- **Real-time gesture recognition:** Processes webcam frames with MediaPipe Holistic landmark detection and a temporal TensorFlow Lite classifier.
- **Multi-modal translation:** Converts `video → landmarks → gestures → text`, with optional text-to-speech and Gemini-assisted sentence refinement.
- **Low-latency local inference:** The core landmark and sign-classification path runs locally and supports CPU execution.
- **Modular services:** Use the browser UI, the FastAPI inference service, or the model utilities independently.
- **Stateful sentence assembly:** Stabilizes predictions, tracks words, detects completion, and finalizes sentences.
- **Extensible platform:** Designed for new signs, additional languages, mobile clients, and alternative model runtimes.

## Architecture

The browser talks only to the **Spring Boot backend**. Spring Boot forwards recognition traffic to the internal **FastAPI ML service** and runs the translation pipeline itself.

```text
React (browser: webcam + MediaPipe Holistic landmarks)
  ↓  HTTP REST
Spring Boot  (public API gateway, port 8080)
  ↓  HTTP REST  (FASTAPI_BASE_URL)
FastAPI ML Service  (internal, port 8001)
  ↓
MediaPipe + Transformer (TFLite)  →  top-k signs + sentence state
  ↓
ASL Gloss  (back in Spring Boot)
  ↓
Gemini  →  Natural English
  ↓
Google Cloud TTS  →  MP3 audio
  ↓
PostgreSQL  (translation history)
  ↓
Response to React  (text + audio; falls back to the browser voice)
```

```mermaid
flowchart LR
    UI[React + Vite\nMediaPipe in browser] -->|/api/v1/inference\n/api/v1/sentence/*| SB[Spring Boot\nAPI gateway]
    UI -->|/api/v1/translate| SB
    SB -->|forward| ML[FastAPI ML service\nTFLite Transformer\nsentence state]
    SB --> GM[Gemini]
    SB --> TTS[Google Cloud TTS]
    SB --> DB[(PostgreSQL)]
```

Two request flows go through Spring Boot:

| Flow | Frontend call | What Spring Boot does |
| --- | --- | --- |
| Live recognition (several times per second) | `POST /api/v1/inference`, `POST /api/v1/sentence/{finalize,reset,append,delete}` | Forwards the JSON unchanged to FastAPI and returns FastAPI's response unchanged |
| Translation (once per sentence) | `POST /api/v1/translate` | Builds the ASL gloss → Gemini → Google Cloud TTS → saves to PostgreSQL → returns `aslGloss`, `englishSentence`, `speechAudio` |

`/api/v1/translate` also accepts a raw `landmarks` sequence instead of `words`; Spring Boot then asks FastAPI to classify it (statelessly) before translating.

**Why two backend technologies?** Python FastAPI is responsible for computer vision and machine-learning inference because the ASL recognition pipeline relies on Python ML tooling (MediaPipe, TensorFlow Lite). Java Spring Boot acts as the main application backend, handling API orchestration, business logic, persistence, Gemini integration, TTS integration, and communication with the ML service.

Graceful degradation: without `GEMINI_API_KEY` the sentence is polished locally (`usedGemini: false`); without `TTS_API_KEY` no audio is returned and the browser speaks instead; if PostgreSQL is down the translation is still returned (`id: null`); if FastAPI is down the recognition endpoints return `503`.

The classifier consumes landmark sequences in the project’s training order and predicts 250 ASL sign classes. The documented held-out validation results are 67.25% top-1 accuracy and 87.57% top-5 accuracy; see [`sign_classifier/README.md`](sign_classifier/README.md) for model details.

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Web client | React 19, TypeScript, Vite, Axios, Framer Motion |
| AI service | Python 3.11+, FastAPI, Uvicorn, OpenCV, MediaPipe, NumPy |
| ML inference | TensorFlow Lite, Transformer encoder model, MediaPipe Holistic landmarks |
| Application API | Java 21, Spring Boot 4.1, Spring Web MVC, Spring Data JPA |
| Persistence | PostgreSQL |
| Generative and speech capabilities | Gemini API, Google Cloud Text-to-Speech, browser Web Speech API |

## Getting Started

### Prerequisites

- Python 3.11 recommended for the AI service
- Node.js 20+ and npm for the frontend
- Java 21 for the Spring Boot backend
- PostgreSQL for translation history and backend database features
- A webcam for live recognition
- CUDA is **optional**. CPU inference is supported; if you enable GPU acceleration, install a TensorFlow runtime and CUDA/cuDNN combination supported by your target platform.

### 1. Clone the repository

```bash
git clone https://github.com/AbhiramAmaravadi/signbridge.git
cd signbridge
```

### 2. Install the AI service

```bash
cd ai-service
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows PowerShell
# .\\venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..
```

### 3. Configure PostgreSQL

Create the database used by the Spring Boot backend:

```sql
CREATE DATABASE signbridge;
```

The development defaults are defined in [`backend/src/main/resources/application.properties`](backend/src/main/resources/application.properties):

```text
Host: localhost
Port: 5432
Database: signbridge
Username: postgres
Password: (set via DB_PASSWORD env var)
```

For shared or production environments, override these values through your normal Spring configuration mechanism and do not commit credentials.

### 4. Start the services

Run each service in its own terminal, in this order.

| Service | URL | Reached by |
| --- | --- | --- |
| FastAPI ML service | `http://127.0.0.1:8001` | Spring Boot only |
| Spring Boot backend | `http://localhost:8080` | Browser |
| React frontend | `http://localhost:5173` | Browser |

**1) AI service — FastAPI inference API**

```bash
cd ai-service

# Activate the virtual environment first if it is not active.
uvicorn app_server:app --host 127.0.0.1 --port 8001 --reload
```

**2) Spring Boot backend**

```bash
cd backend
export DB_PASSWORD="your_postgres_password"
export FASTAPI_BASE_URL="http://127.0.0.1:8001"   # default
# Optional: export GEMINI_API_KEY=...  TTS_API_KEY=...
./mvnw spring-boot:run
```

PowerShell:

```powershell
cd backend
$env:DB_PASSWORD = "your_postgres_password"
.\mvnw.cmd spring-boot:run
```

**3) React frontend**

```bash
cd frontend
npm install
npm run dev
```

In development the frontend calls `http://localhost:8080` (Spring Boot) automatically. Open `http://localhost:5173`.

### Alternative: Docker Compose

Runs PostgreSQL, the FastAPI ML service (internal only), Spring Boot and the frontend:

```bash
# optional: export GEMINI_API_KEY=... TTS_API_KEY=...
docker compose up --build
```

Then open `http://localhost:5173`. The Spring Boot API is on `http://localhost:8080`; FastAPI is not published to the host.

### Optional: local OpenCV webcam demo

The desktop demo runs landmark extraction and saves rolling sequences locally:

```bash
cd ai-service
python main.py --camera-index 0 --width 640 --height 480 --sequence-length 30
```

Press `q` to exit. Useful options include `--output-dir`, `--save-stride`, `--model-complexity`, `--min-detection-confidence`, and `--min-tracking-confidence`.

## Configuration

### Frontend API target

`VITE_API_BASE_URL` must point at the **Spring Boot backend**. When unset, `npm run dev` uses `http://localhost:8080` and production builds use the fallback URL in [`frontend/src/config.ts`](frontend/src/config.ts).

```bash
cd frontend
printf "VITE_API_BASE_URL=http://localhost:8080\n" > .env.local
npm run dev
```

### Backend environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `FASTAPI_BASE_URL` | `http://localhost:8001` | Where Spring Boot reaches the ML service |
| `FASTAPI_READ_TIMEOUT` | `30s` | Allows for an ML-service cold start |
| `DB_URL`, `DB_USERNAME`, `DB_PASSWORD` | `jdbc:postgresql://localhost:5432/signbridge`, `postgres`, empty | PostgreSQL |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | unset, `gemini-2.5-flash` | Sentence refinement |
| `TTS_API_KEY` | unset | Google Cloud Text-to-Speech (API key with the TTS API enabled) |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173` + deployed frontend | Browser origins allowed to call Spring Boot |
| `PORT` | `8080` | HTTP port (set by Cloud Run) |

API keys live only in the backend environment; nothing secret is shipped to the browser.

### Gemini integration

Gemini is optional and is called by Spring Boot. Without a key, a local fallback capitalizes the gloss into a sentence. (The FastAPI service keeps its own `/api/v1/gemini/*` endpoints for standalone use; the frontend no longer calls them.)

```bash
export GEMINI_API_KEY="your_api_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

PowerShell:

```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
$env:GEMINI_MODEL = "gemini-2.5-flash"
```

## API Reference

### Spring Boot (public API, `http://localhost:8080`, OpenAPI UI at `/docs`)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Backend status plus `mlService: ok/unavailable` |
| `POST` | `/api/v1/inference` (alias `/inference`) | Forwarded to FastAPI `/api/v1/inference` |
| `POST` | `/api/v1/sentence/{finalize,reset,append,delete}` | Forwarded to FastAPI sentence state |
| `POST` | `/api/v1/translate` | Gloss → Gemini → TTS → PostgreSQL |
| `GET` | `/api/v1/translations` | Translation history |
| `GET` | `/api/v1/translations/{id}` | One translation (`404` if missing) |
| `DELETE` | `/api/v1/translations/{id}` | Delete a translation |

```bash
curl -X POST http://localhost:8080/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"words": ["STORE", "GO", "TOMORROW"]}'
# {"id":1,"aslGloss":"STORE GO TOMORROW","englishSentence":"...","speechAudio":"<base64 mp3 or null>",
#  "detectedEmotion":"neutral","detectedScene":"unknown","usedGemini":true}
```

### FastAPI ML service (internal, `http://127.0.0.1:8001`, OpenAPI UI at `/docs`)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | AI service health check |
| `POST` | `/predict` or `/api/v1/inference` | Predict signs from landmarks or a base64 image frame |
| `POST` | `/api/v1/sentence/finalize` | Finalize the current sentence |
| `POST` | `/api/v1/sentence/reset` | Reset sentence state |
| `POST` | `/api/v1/sentence/append` | Append a word manually |
| `POST` | `/api/v1/sentence/delete` | Delete a word manually |
| `POST` | `/api/v1/gemini/translate` | Refine recognized words into natural language |
| `POST` | `/api/v1/gemini/scene` | Analyze an image scene when Gemini is enabled |
| `GET` | `/api/v1/gemini/scene-prompt` | Inspect the scene-analysis prompt |

Example inference request using a landmark sequence:

```bash
curl -X POST http://127.0.0.1:8001/api/v1/inference \\
  -H "Content-Type: application/json" \\
  -d '{"landmarks": [[[0.0, 0.0, 0.0]]], "top_k": 5}'
```

Add `"update_sentence": false` for a one-off classification that leaves the live sentence state untouched (Spring Boot uses this for `/api/v1/translate` with landmarks).

## Model Evaluation

Run the classifier benchmark from the repository root:

```bash
python sign_classifier/benchmark.py --samples 500 --shuffle
```

See [`sign_classifier/README.md`](sign_classifier/README.md) for preprocessing, the 250-class output space, and evaluation methodology.

## Roadmap

- [x] Real-time MediaPipe landmark extraction
- [x] TensorFlow Lite temporal sign classification
- [x] FastAPI inference and sentence-state endpoints
- [x] React/Vite browser interface
- [x] Optional Gemini sentence refinement and scene context
- [ ] Expand sign coverage and multilingual sign-language support
- [ ] Fine-tune and continuously evaluate models on broader, representative datasets
- [ ] Add mobile deployment targets and optimized on-device inference
- [ ] Add automated CI, API contract tests, and release artifacts
- [ ] Improve observability, calibration, and accessibility testing

## Contributing

Contributions are welcome. To propose a change:

1. Open an issue describing the problem, use case, or proposed design.
2. Fork the repository and create a focused branch.
3. Keep changes scoped, documented, and covered by tests where practical.
4. Run the relevant frontend lint/build, backend tests, and AI-service checks.
5. Open a pull request with a clear summary, validation steps, and screenshots for UI changes.

Please do not include API keys, database passwords, user video, or other sensitive data in commits or issue attachments. When reporting model behavior, include reproducible context and label the result as an observation rather than a guarantee of recognition accuracy.

## License

SignBridge is intended to be released under the [MIT License](LICENSE). If you are packaging or redistributing this repository, verify that the repository’s `LICENSE` file is present and reflects the applicable licensing terms.

## Acknowledgments

SignBridge builds on the work of the MediaPipe, TensorFlow, OpenCV, FastAPI, React, Spring Boot, and PostgreSQL communities. We also acknowledge the research and dataset contributors whose work enables accessible sign-language machine learning, and the SAIL (Software AI / Accessibility Lab) community for its accessibility-focused collaboration.

<p align="center">
  <sub>Built to make communication more accessible, one gesture at a time.</sub>
</p>
