# Build dependencies 
FROM python:3.11-slim AS builder
WORKDIR /build
# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# Runtime image
FROM python:3.11-slim AS runtime
# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app
# Copy installed packages from builder
COPY --from=builder /install /usr/local
# Copy app code
COPY app/ ./app/
# Copy pre-built model files
COPY model/ ./model/
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