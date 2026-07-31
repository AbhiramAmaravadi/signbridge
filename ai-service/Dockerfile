FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app:/app/sign_classifier:/app/signclassifier"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn app_server:app --host 0.0.0.0 --port ${PORT:-8080}
