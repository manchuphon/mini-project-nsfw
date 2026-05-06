# ── Stage 1: Install all dependencies ───────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Download & prepare model ───────────────────────
FROM python:3.11-slim AS model-builder
WORKDIR /build

# Reuse installed packages from Stage 1
COPY --from=builder /install /usr/local

COPY scripts/prepare_model.py ./scripts/prepare_model.py

RUN python scripts/prepare_model.py \
      --output-dir ./model \
      --warmup 1 \
      --runs 1

# ── Stage 3: Runtime image ───────────────────────────────────
FROM python:3.11-slim AS runtime

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY --from=model-builder /build/model/ ./model/

ENV MODEL_BACKEND=quantized \
    MODEL_DIR=/app/model \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    ORT_NUM_THREADS=2

USER appuser
EXPOSE 7860
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "4", \
     "--timeout-keep-alive", "30", \
     "--log-level", "warning"]