import asyncio
import io
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.model import load_model, predict

# process pool
executor: ProcessPoolExecutor | None = None
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    # warm-up
    load_model()
    executor = ProcessPoolExecutor(max_workers=4, initializer=load_model)
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="NSFW Image Classification API",
    description="High-throughput content moderation service using ViT + ONNX quantized model",
    version="1.0.0",
    lifespan=lifespan,
)


# Helper 
def _validate_and_decode(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((224, 224), Image.BILINEAR)
        return np.array(img, dtype=np.float32)
    except UnidentifiedImageError:
        raise ValueError("INVALID_IMAGE")
    except Exception as e:
        raise ValueError(f"DECODE_ERROR:{e}")


def _run_prediction(image_bytes: bytes) -> dict:
    arr = _validate_and_decode(image_bytes)
    label, confidence, scores = predict(arr)
    return {"label": label, "confidence": round(float(confidence), 4), "scores": scores}


# Endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Content-type check
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    # Read & size 
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) / 1e6:.1f} MB). Max {MAX_FILE_SIZE_MB} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    # CPU-bound inference in process pool
    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _run_prediction, image_bytes)
    except ValueError as e:
        msg = str(e)
        if "INVALID_IMAGE" in msg:
            raise HTTPException(status_code=400, detail="File is not a valid image or is corrupted.")
        raise HTTPException(status_code=400, detail=f"Image processing error: {msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return JSONResponse(
        content={
            "filename": file.filename,
            "label": result["label"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "latency_ms": latency_ms,
        }
    )


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 16:
        raise HTTPException(status_code=400, detail="Max 16 images per batch.")

    async def _classify_one(f: UploadFile):
        ct = f.content_type
        if ct not in ALLOWED_CONTENT_TYPES:
            return {"filename": f.filename, "error": f"Unsupported type: {ct}"}
        data = await f.read()
        if len(data) > MAX_FILE_SIZE_BYTES:
            return {"filename": f.filename, "error": "File too large"}
        loop = asyncio.get_event_loop()
        try:
            res = await loop.run_in_executor(executor, _run_prediction, data)
            return {"filename": f.filename, **res}
        except Exception as e:
            return {"filename": f.filename, "error": str(e)}

    results = await asyncio.gather(*[_classify_one(f) for f in files])
    return JSONResponse(content={"results": results, "count": len(results)})