# Build from the repository root so sibling sign_classifier/ is available:
#   docker build -f ai-service/Dockerfile .
#
# Cloud Run / gcloud example:
#   gcloud builds submit --config=<(echo '...')  OR
#   docker build -f ai-service/Dockerfile -t SIGNBRIDGE_AI .

FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app:/app/sign_classifier:/app/signclassifier"
ENV PYTHONUNBUFFERED=1

COPY ai-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY ai-service/ .

# Classifier module + TFLite assets (predict_topk.py lives here, not under ai-service/)
COPY sign_classifier/ ./sign_classifier/

EXPOSE 8080

CMD uvicorn app_server:app --host 0.0.0.0 --port ${PORT:-8080}
