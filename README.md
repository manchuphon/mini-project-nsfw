# JMeter Load Testing Guide – NSFW API

## Prerequisites
- Apache JMeter 5.6+ ([download](https://jmeter.apache.org/download_jmeter.cgi))
- A sample image at `/tmp/sample.png` (any 224×224 PNG/JPEG works)
- API running locally (`docker run -p 7860:7860 nsfw-api`) or on HF Spaces

---

## Quick Start

### 1. Prepare sample image
```bash
python - <<'EOF'
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0,256,(224,224,3),dtype='uint8'))
img.save('/tmp/sample.png')
EOF
```

### 2. Run load test (CLI mode – no GUI needed)
```bash
# Local Docker
jmeter -n \
  -t jmeter/nsfw_api_loadtest.jmx \
  -JBASE_URL=http://localhost:7860 \
  -JTHREADS=20 \
  -JRAMP_UP=10 \
  -JDURATION=60 \
  -Jimage_path=/tmp/sample.png \
  -l results/raw.jtl \
  -e -o results/html_report

# Hugging Face Spaces (replace URL)
jmeter -n \
  -t jmeter/nsfw_api_loadtest.jmx \
  -JBASE_URL=https://YOUR_USERNAME-mini-project-nsfw.hf.space \
  -JTHREADS=20 \
  -JRAMP_UP=10 \
  -JDURATION=60 \
  -Jimage_path=/tmp/sample.png \
  -l results/cloud_raw.jtl \
  -e -o results/cloud_html_report
```

### 3. View results
Open `results/html_report/index.html` in a browser.

---

## Thread Group Configuration

| Group | Purpose | Threads | Ramp-up | Duration |
|-------|---------|---------|---------|----------|
| TG1 – Smoke | Verify /health only | 1 | 1 s | 5 s |
| TG2 – Load | Normal load /predict | 20 | 10 s | 60 s |
| TG3 – Stress | Find saturation point | 100 | 60 s | 120 s |

> TG3 is **disabled by default**. Enable it in JMeter GUI or add `-JTG3_enabled=true`.

---

## Metrics to Collect & Analyse

### Key Metrics
| Metric | Target (local) | Target (cloud) |
|--------|---------------|----------------|
| Throughput (TPS) | > 5 req/s | > 2 req/s |
| P50 Latency | < 500 ms | < 1000 ms |
| P95 Latency | < 1500 ms | < 3000 ms |
| Error Rate | < 1% | < 1% |

### How to read Aggregate Report
```
Label          #Samples  Average  Min   Max    P90    P95    P99   Error%  Throughput
POST /predict  1200      423 ms   180   2100   850    1200   1800  0.08%   18.5/s
```

- **Average**: Mean latency – avoid relying on this alone (affected by outliers)
- **P95**: 95th percentile – 95% of requests finish within this time → use for SLA
- **Throughput**: Requests per second (TPS) – main capacity indicator
- **Error%**: Should be < 1%; spikes indicate server saturation

---

## Bottleneck Analysis

### Signs of CPU saturation
- Latency climbs steeply as thread count increases
- P95 > 3× P50 (long tail)
- Error rate rises above 1%

### Typical bottleneck for this API
Since inference is CPU-bound (ONNX on CPU), the bottleneck is the **ProcessPoolExecutor**
with `max_workers=4`. When concurrent requests exceed 4, requests queue up.

**Fix options:**
1. Increase `max_workers` (limited by CPU cores)
2. Use a quantized model (already done – INT8 is ~2–3× faster)
3. Use GPU inference (`CUDAExecutionProvider`)
4. Scale horizontally (multiple Docker containers behind a load balancer)

---

## Exporting Results for Report

```bash
# Generate CSV summary from .jtl file
jmeter -g results/raw.jtl -o results/html_report

# Quick stats via Python
python - <<'EOF'
import pandas as pd
df = pd.read_csv('results/raw.jtl')
print(df.groupby('label')['elapsed'].describe(percentiles=[.5,.9,.95,.99]))
EOF
```