# Deployment & reliability notes

## Cloud Run (AI / API service) — avoid cold-start "page won't load"

The service loads TensorFlow, MediaPipe and two TFLite models. On a scaled-to-zero instance the
first request can take 10-30 s, longer than the old frontend timeouts. Keep one instance warm:

```bash
gcloud run services update signbridge-backend \
  --region us-central1 \
  --min-instances 1 \
  --memory 2Gi --cpu 2 \
  --cpu-boost \
  --no-cpu-throttling \
  --concurrency 1 \
  --timeout 60 \
  --set-env-vars CORS_ALLOWED_ORIGINS=https://signbridge-frontend-389644353290.us-central1.run.app,http://localhost:5173
```

* `--min-instances 1` + `--cpu-boost` removes most cold starts.
* `--max-instances 1` is recommended until sentence state is moved out of process memory
  (`_sentence_state` / `_frame_buffer` in `ai-service/app_server.py` are per-process globals).
* The frontend pings `/health` on page load (`warmUpBackend()` in `frontend/src/config.ts`) and the
  service preloads MediaPipe on startup (`_warm_up` in `app_server.py`).

## Frontend

MediaPipe is self-hosted: `npm run dev|build` runs `scripts/copy-mediapipe.mjs`, which copies
`node_modules/@mediapipe/*` into `public/mediapipe/` (git-ignored). No jsDelivr dependency at runtime.

## Secrets

Nothing sensitive is committed. Provide these through the environment / Secret Manager:

| Variable | Used by |
| --- | --- |
| `DB_URL`, `DB_USERNAME`, `DB_PASSWORD` | Spring Boot backend |
| `GEMINI_API_KEY`, `TTS_API_KEY` | backend / ai-service |
| `CORS_ALLOWED_ORIGINS` | backend and ai-service (comma separated) |

## Purging old secrets/binaries from git history (needs team coordination)

Rotating the credentials is what actually fixes the leak. Rewriting history is optional (repo size)
and requires everyone to re-clone:

```bash
pip install git-filter-repo
git clone --mirror https://github.com/AbhiramAmaravadi/signbridge.git signbridge-backup.git   # backup
git filter-repo --invert-paths \
  --path .env.local --path .vercel --path SignBridge_Deck.pptx --path SignBridge_Demo_video.mp4 \
  --path ai-service/saved_sequences
git remote add origin https://github.com/AbhiramAmaravadi/signbridge.git
git push --force --all && git push --force --tags
```
