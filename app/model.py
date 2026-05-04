import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Backend selection 
BACKEND = os.getenv("MODEL_BACKEND", "quantized")  # pytorch | onnx | quantized
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

_session = None          # ONNX 
_hf_pipeline = None     # HuggingFace 
_id2label = {0: "normal", 1: "nsfw"}

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)          
    arr = np.expand_dims(arr, 0)         
    return arr.astype(np.float32)


def load_model():
    global _session, _hf_pipeline

    if BACKEND == "pytorch":
        if _hf_pipeline is None:
            from transformers import pipeline as hf_pipeline_fn
            _hf_pipeline = hf_pipeline_fn(
                "image-classification",
                model=str(MODEL_DIR / "pytorch"),
                device=-1,
            )
    else:
        if _session is None:
            import onnxruntime as ort
            model_file = (
                "model_quantized.onnx" if BACKEND == "quantized" else "model.onnx"
            )
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                str(MODEL_DIR / model_file),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )


def predict(image_array: np.ndarray) -> Tuple[str, float, dict]:
    if BACKEND == "pytorch":
        from PIL import Image
        img = Image.fromarray(image_array.astype(np.uint8))
        results = _hf_pipeline(img)
        top = results[0]
        scores = {r["label"]: round(r["score"], 4) for r in results}
        return top["label"], top["score"], scores

    # ONNX path 
    tensor = _normalize(image_array)
    input_name = _session.get_inputs()[0].name
    logits = _session.run(None, {input_name: tensor})[0][0]  

    # Softmax
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    pred_idx = int(probs.argmax())
    label = _id2label[pred_idx]
    confidence = float(probs[pred_idx])
    scores = {_id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    return label, confidence, scores