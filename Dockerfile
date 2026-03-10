# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Copy models — ต้องมี models/whisper/ และ models/best_model.pt
COPY models/ /app/models/

# ตรวจสอบว่า whisper model มีอยู่
RUN echo "Models directory:" && ls -la /app/models/ && \
    echo "Whisper directory:" && ls -la /app/models/whisper/ 2>/dev/null || echo "WARNING: whisper not found"

ENV PYTHONPATH=/app/backend
ENV PORT=8080
ENV MODEL_DIR=/app/models

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "/app/backend/main.py"]