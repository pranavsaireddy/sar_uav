# SAR UAV Detection System
### Multimodal RGB-Thermal Fusion for Human Survivor Detection
**Local Deployment — RTX 4060 Ti · FastAPI Backend · React Frontend**

---

## What This Is

A production-quality AI system that fuses RGB + thermal (IR) camera feeds from a UAV to detect human survivors in disaster scenarios. It solves a core problem:

- **RGB alone** fails in smoke, dust, debris, low light
- **Thermal alone** generates false positives from hot debris, engines, fires
- **This system**: RGB body shape + thermal heat signature together = confident detection

The model learns three rules via cross-modal attention:
| Condition | Result |
|---|---|
| Heat aligned with body shape (RGB) | ✅ High confidence detection |
| Heat with no body shape (hot debris) | ❌ Suppressed by anomaly filter |
| Body shape with no heat (cold survivor) | ⚠️ Flagged for follow-up |

---

## Hardware Target

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 5 7600X (6c/12t) |
| GPU | NVIDIA RTX 4060 Ti **16 GB VRAM** |
| RAM | 16 GB DDR5 (32 GB recommended) |
| Storage | 20 GB free SSD |
| OS | Windows 11 + WSL2 **or** Ubuntu 22.04 LTS |

---

## Project Structure

```
sar_system/
├── backend/
│   ├── models/fusion_model.py        # PyTorch SARFusionModel (~8M params)
│   ├── training/
│   │   ├── train.py                  # Full training loop + hard negative mining
│   │   ├── dataset.py                # LLVIP / KAIST / SAR JSON loaders
│   │   └── losses.py                 # DetectionLoss + Consistency + Alignment
│   ├── inference/engine.py           # InferenceEngine (NMS, preprocessing)
│   ├── api/
│   │   ├── main.py                   # FastAPI app entry
│   │   ├── routes/detect.py          # POST /detect (base64)
│   │   ├── routes/upload.py          # POST /upload (multipart files)
│   │   ├── routes/stats.py           # GET /stats, /history, /detections/map
│   │   └── routes/ws.py              # WS /ws/live, /ws/monitor
│   ├── deployment/edge_optimizer.py  # ONNX export + INT8 quantization
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx                   # Router + nav bar
│       ├── pages/
│       │   ├── Dashboard.jsx         # Live feed, camera canvases, GPS map
│       │   ├── Upload.jsx            # Drag-drop inference, bbox overlay
│       │   ├── History.jsx           # Filterable detection log
│       │   └── Stats.jsx             # Recharts dashboards
│       ├── components/               # Gauge, bars, map, overlay, dropzone
│       ├── hooks/                    # useWebSocket, useDetection
│       ├── services/api.js           # Axios client
│       └── store/detectionStore.js   # Zustand global state
├── simulation/uav_simulation.py      # Synthetic dataset + demo frame generator
└── docker-compose.yml
```

---

## Phase 1 — Environment Setup

### Install prerequisites (in order)

```bash
# 1. CUDA 12.1
# → https://developer.nvidia.com/cuda-downloads

# 2. Python 3.11
python --version   # confirm 3.11.x

# 3. Node.js 20 LTS
# → https://nodejs.org

# 4. Python virtual environment
cd sar_system/backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 5. Install Python deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 6. Verify GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True  /  NVIDIA GeForce RTX 4060 Ti
```

### Install frontend deps

```bash
cd sar_system/frontend
npm install
```

---

## Phase 2 — Dataset

### Option A: LLVIP (recommended, ~2.6 GB, no registration)

```
HuggingFace: https://huggingface.co/datasets/srivastavagaurang/LLVIP
Official:    https://bupt-ai-cz.github.io/LLVIP/
```

Place extracted dataset at `backend/data/`:
```
data/
├── visible/train/    ← 19903 RGB JPEGs
├── visible/test/     ← 3463 RGB JPEGs
├── infrared/train/   ← matching thermal PNGs
├── infrared/test/
└── Annotations/YOLO_Format/train/ + test/
```

### Option B: Synthetic dataset (zero download, instant)

```bash
cd sar_system
python simulation/uav_simulation.py
# Generates 1000 labeled pairs in ~3 minutes at data/synthetic/
```

---

## Phase 3 — Training

```bash
cd backend

# Full model (50 epochs, RTX 4060 Ti, ~35-45 min total)
python training/train.py \
  --data data/ \
  --epochs 50 \
  --batch-size 16 \
  --model full \
  --amp

# Quick smoke test (5 epochs, verify pipeline)
python training/train.py \
  --data data/ \
  --epochs 5 \
  --batch-size 8 \
  --model edge

# Windows: wrap in if __name__ == '__main__' guard to avoid worker crash
```

**Expected progression:**

| Epoch | Loss | Val Accuracy |
|---|---|---|
| 1–5 (warmup) | 3.0–4.5 | 50–60% |
| 5–15 | 1.5–2.5 | 65–75% |
| 15–30 | 0.8–1.5 | 75–83% |
| 30–50 | 0.3–0.8 | 83–90% |

Best checkpoint saved to `backend/checkpoints/best_sar_model.pt`

---

## Phase 4 — Backend API

```bash
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI at **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Status + device info |
| POST | /detect | Base64 inference |
| POST | /upload | **Multipart RGB + thermal files** |
| GET | /stats | Live system metrics |
| GET | /history | Detection log |
| GET | /detections/map | GPS survivor points |
| WS | /ws/live | Real-time frame streaming |
| WS | /ws/monitor | Broadcast subscriber |

### Test upload endpoint

```bash
curl -X POST http://localhost:8000/upload \
  -F "rgb_file=@path/to/rgb.jpg" \
  -F "thermal_file=@path/to/thermal.png" \
  -F "lat=17.385" -F "lon=78.487" -F "altitude=50"
```

---

## Phase 5 — Frontend

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

**Four pages:**

| Page | Route | What it does |
|---|---|---|
| Dashboard | `/` | Live WebSocket feed, dual camera canvases, GPS map |
| Upload | `/upload` | Drag-drop RGB + thermal → real-time inference with bbox overlay |
| History | `/history` | Filterable detection log with CSV export |
| Stats | `/stats` | Detection rate, confidence histograms, latency charts |

---

## Phase 6 — Edge Export (ONNX)

```bash
cd backend
python deployment/edge_optimizer.py \
  --checkpoint checkpoints/best_sar_model.pt \
  --output checkpoints/sar_edge.onnx \
  --mode edge \
  --benchmark
# Expected: ~100 FPS (full), ~200+ FPS (edge INT8) on RTX 4060 Ti
```

---

## Docker (optional)

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## Performance Targets

| Metric | Target |
|---|---|
| Val accuracy | ≥ 85% |
| False positive rate | ≤ 15% |
| Avg inference latency | ≤ 12 ms (RTX 4060 Ti, FP16, 320px) |
| Consistency (true positive) | ≥ 0.65 |
| Consistency (hot debris FP) | ≤ 0.25 |

---

## Common Issues

| Error | Cause | Fix |
|---|---|---|
| CUDA out of memory | Batch size too large | Reduce `--batch-size` or ensure `--amp` is set |
| RuntimeError: workers | Windows multiprocessing | Add `if __name__ == '__main__':` guard |
| 422 Unprocessable Entity | Missing form field | Check multipart field names match schema |
| WebSocket disconnects | No keepalive | Auto-reconnect with exponential backoff is built in |
| Thermal all black | 16-bit PNG not normalized | Dataset.py handles this automatically |
| CORS error in browser | Origins mismatch | Ensure `CORS_ORIGINS=http://localhost:5173` in `.env` |
| Model not detecting | Threshold too high | Lower `CONF_THRESHOLD` to 0.35 in `.env` |

---

## Key Design Decisions

- **No rotation augmentation** — UAV nadir perspective makes rotation unrealistic
- **Single Uvicorn worker** — GPU is not fork-safe; multi-worker would require model sharding
- **asyncio.Lock on inference** — prevents concurrent GPU calls via WebSocket
- **Hard negative mining** — pool of 512 high-confidence false positives injected every 10 batches
- **AMP throughout** — saves ~6 GB VRAM, doubles throughput on RTX 4060 Ti

---

*SAR UAV Detection System — RGB-Thermal Fusion — RTX 4060 Ti Local Build*
