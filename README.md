---
title: NSFW Image Classification API
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
---

# NSFW Image Classification API

High-throughput content moderation service using **ViT + ONNX Quantized model** (Falconsai/nsfw_image_detection)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Classify single image |
| POST | `/predict/batch` | Classify up to 16 images |

## cURL Examples

```bash
# Health check
curl https://manchuphonhu25-mini-project-nsfw.hf.space/health

# Single image prediction
curl -X POST https://manchuphonhu25-mini-project-nsfw.hf.space/predict \
  -F "file=@your_image.jpg"

# Batch prediction
curl -X POST https://manchuphonhu25-mini-project-nsfw.hf.space/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## Response Example

```json
{
  "filename": "photo.jpg",
  "label": "normal",
  "confidence": 0.9823,
  "scores": {
    "normal": 0.9823,
    "nsfw": 0.0177
  },
  "latency_ms": 142.5
}
```

## Supported Formats

`image/jpeg` `image/png` `image/webp` `image/gif` `image/bmp` — Max **10 MB**

## Model

- **Base model:** [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) (Vision Transformer)
- **Optimized:** ONNX + Dynamic INT8 Quantization
- **Backend:** ONNX Runtime CPU