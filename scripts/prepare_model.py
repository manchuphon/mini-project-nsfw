import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="./model", help="Where to save model files")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--runs", type=int, default=50)
args = parser.parse_args()

OUT = Path(args.output_dir)
(OUT / "pytorch").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Falconsai/nsfw_image_detection"

def make_dummy_image() -> Image.Image:
    arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)

dummy_img = make_dummy_image()


# ============================================================
# PHASE 2: PyTorch Baseline
# ============================================================
print("\n" + "="*60)
print("PHASE 2: Loading PyTorch model (baseline)")
print("="*60)

import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained(MODEL_ID)
pt_model  = AutoModelForImageClassification.from_pretrained(MODEL_ID)
pt_model.eval()

pt_model.save_pretrained(str(OUT / "pytorch"))
processor.save_pretrained(str(OUT / "pytorch"))
pt_size_mb = sum(f.stat().st_size for f in (OUT / "pytorch").rglob("*") if f.is_file()) / 1e6
print(f"PyTorch model saved  ({pt_size_mb:.1f} MB)")

def pytorch_infer(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = pt_model(**inputs).logits
    return logits.argmax(-1).item()

for _ in range(args.warmup):
    pytorch_infer(dummy_img)

pt_times = []
for _ in range(args.runs):
    t0 = time.perf_counter()
    pytorch_infer(dummy_img)
    pt_times.append((time.perf_counter() - t0) * 1000)

pt_mean = np.mean(pt_times)
pt_p95  = np.percentile(pt_times, 95)
print(f"  Mean latency : {pt_mean:.2f} ms")
print(f"  P95  latency : {pt_p95:.2f} ms")


# ============================================================
# PHASE 3: ONNX Export  — ใช้ legacy exporter (dynamo=False)
# ============================================================
print("\n" + "="*60)
print("PHASE 3: Exporting to ONNX")
print("="*60)

onnx_path = OUT / "model.onnx"
dummy_inputs = processor(images=dummy_img, return_tensors="pt")
dummy_tensor = dummy_inputs["pixel_values"]

# ── Legacy exporter: เสถียรกว่าและ quantize ได้ ──────────────
with torch.no_grad():
    torch.onnx.export(
        pt_model,
        (dummy_tensor,),
        str(onnx_path),
        export_params=True,
        opset_version=14,          # 14 เสถียรที่สุดกับ ViT + quantize
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits":        {0: "batch_size"},
        },
    )

onnx_size_mb = onnx_path.stat().st_size / 1e6
print(f"ONNX model saved  ({onnx_size_mb:.1f} MB)")

import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
onnx_session = ort.InferenceSession(
    str(onnx_path),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)

def onnx_infer(arr: np.ndarray):
    out = onnx_session.run(None, {"pixel_values": arr})
    return out[0].argmax(-1)[0]

np_input = dummy_tensor.numpy()

for _ in range(args.warmup):
    onnx_infer(np_input)

onnx_times = []
for _ in range(args.runs):
    t0 = time.perf_counter()
    onnx_infer(np_input)
    onnx_times.append((time.perf_counter() - t0) * 1000)

onnx_mean = np.mean(onnx_times)
onnx_p95  = np.percentile(onnx_times, 95)
print(f"  Mean latency : {onnx_mean:.2f} ms")
print(f"  P95  latency : {onnx_p95:.2f} ms")


# ============================================================
# PHASE 4: Dynamic INT8 Quantization
# ============================================================
print("\n" + "="*60)
print("PHASE 4: Dynamic INT8 Quantization…")
print("="*60)

from onnxruntime.quantization import QuantType, quantize_dynamic

quant_path = OUT / "model_quantized.onnx"
quantize_dynamic(
    str(onnx_path),
    str(quant_path),
    weight_type=QuantType.QUInt8,
)

quant_size_mb = quant_path.stat().st_size / 1e6
print(f"Quantized model saved  ({quant_size_mb:.1f} MB)")

quant_session = ort.InferenceSession(
    str(quant_path),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)

def quant_infer(arr: np.ndarray):
    out = quant_session.run(None, {"pixel_values": arr})
    return out[0].argmax(-1)[0]

for _ in range(args.warmup):
    quant_infer(np_input)

quant_times = []
for _ in range(args.runs):
    t0 = time.perf_counter()
    quant_infer(np_input)
    quant_times.append((time.perf_counter() - t0) * 1000)

quant_mean = np.mean(quant_times)
quant_p95  = np.percentile(quant_times, 95)
print(f"  Mean latency : {quant_mean:.2f} ms")
print(f"  P95  latency : {quant_p95:.2f} ms")


# ============================================================
# PHASE 5: Benchmark Summary
# ============================================================
print("\n" + "="*60)
print("PHASE 5: Benchmark Summary")
print("="*60)

header = f"{'Model':<20} {'Size (MB)':>10} {'Mean (ms)':>12} {'P95 (ms)':>12} {'Speedup':>10}"
print(header)
print("-" * len(header))
print(f"{'PyTorch (baseline)':<20} {pt_size_mb:>10.1f} {pt_mean:>12.2f} {pt_p95:>12.2f} {'1.00×':>10}")
print(f"{'ONNX':<20} {onnx_size_mb:>10.1f} {onnx_mean:>12.2f} {onnx_p95:>12.2f} {pt_mean/onnx_mean:>9.2f}×")
print(f"{'ONNX + INT8 Quant':<20} {quant_size_mb:>10.1f} {quant_mean:>12.2f} {quant_p95:>12.2f} {pt_mean/quant_mean:>9.2f}×")

results = {
    "pytorch":   {"size_mb": pt_size_mb,   "mean_ms": pt_mean,   "p95_ms": pt_p95},
    "onnx":      {"size_mb": onnx_size_mb,  "mean_ms": onnx_mean, "p95_ms": onnx_p95},
    "quantized": {"size_mb": quant_size_mb, "mean_ms": quant_mean, "p95_ms": quant_p95},
}
with open(OUT / "benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nBenchmark results saved to {OUT}/benchmark.json")
print("Model preparation complete!")