# SAR UAV Detection System — Full Project Documentation
*Last updated: 2026-04-11*

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model Architecture](#3-model-architecture)
4. [Dataset](#4-dataset)
5. [Training](#5-training)
6. [Inference & Survival Scoring](#6-inference--survival-scoring)
7. [Backend API](#7-backend-api)
8. [Frontend](#8-frontend)
9. [File Structure](#9-file-structure)
10. [Environment & Dependencies](#10-environment--dependencies)
11. [Run Commands](#11-run-commands)
12. [Training Results](#12-training-results)
13. [Known Issues & Notes](#13-known-issues--notes)
14. [Next Steps](#14-next-steps)

---

## 1. Project Overview

A full-stack AI system for Search and Rescue (SAR) operations using UAV-mounted cameras. The system fuses RGB (visible light) and thermal infrared images to detect human survivors in disaster scenarios.

**Core idea:** A living person emits body heat (~37°C) that shows up bright in thermal imagery but may be hidden or hard to spot in RGB. By fusing both modalities with cross-modal attention, the model achieves higher accuracy than either camera alone.

**Stack:**
| Layer | Technology |
|-------|-----------|
| AI Model | PyTorch 2.6.0 + CUDA 12.4 |
| Backend API | Python 3.12 + FastAPI + Uvicorn |
| Frontend | React 18 + Vite + Tailwind CSS |
| GPU | NVIDIA RTX 4060 Ti (8 GB VRAM) |

---

## 2. System Architecture

```
UAV Camera Feed
      │
      ├── RGB image (JPG)
      └── Thermal image (JPG/PNG)
              │
              ▼
    ┌─────────────────────┐
    │   FastAPI Backend   │  ← POST /upload
    │   port 8000         │
    │                     │
    │  InferenceEngine    │
    │    └─ SARFusionModel│
    │    └─ NMS           │
    │    └─ SurvivalScore │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  React Frontend     │  ← http://localhost:5173
    │  port 5173          │
    │                     │
    │  /upload  page      │
    │  /stats   page      │
    │  WebSocket /ws/live │
    └─────────────────────┘
```

**WebSocket endpoints:**
- `/ws/live` — real-time frame streaming for UAV live feed
- `/ws/monitor` — broadcast channel for multi-client monitoring

---

## 3. Model Architecture

**Model:** `SARFusionModel` — 15,740,878 parameters (full mode)

```
RGB   (B, 3, 320, 320) ──┐
                          ├─► ModalityEncoder (EfficientNet-B0 backbone)
Thermal (B, 1, 320, 320) ─┘        │
                                    ▼
                           CrossModalAttention
                           (bidirectional: RGB↔Thermal)
                                    │
                                    ▼
                             FusionModule
                           (3 layers, 8 heads)
                                    │
                                    ▼
                            AnomalyFilter
                           (produces consistency_score)
                                    │
                                    ▼
                            DetectionHead
                           (YOLO-style, 3 anchors)
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
            bounding_boxes                      confidence
            (B, H, W, 3, 6)                       (B,)
```

### Key components:

**ModalityEncoder** — EfficientNet-B0 pretrained on ImageNet. For thermal input, the first conv layer is adapted from 3-channel to 1-channel by averaging RGB weights (preserves pretrained features).

**CrossModalAttention** — Bidirectional: RGB features attend to thermal keys/values, and thermal features attend to RGB keys/values simultaneously. This aligns body shape (RGB) with heat signature (thermal).

**FusionModule** — Stacks N CrossModalAttention layers with residual connections. Learns a `modality_weight` parameter (soft weighting of RGB vs thermal contribution).

**AnomalyFilter** — Suppresses false positives (e.g., hot engine parts, reflections) by scoring cross-modal consistency. Outputs a `global_score` (consistency_score in API) and a spatial heatmap.

**DetectionHead** — YOLO-style: predicts (cx, cy, w, h, objectness, class) per anchor per grid cell on a 10×10 grid. Three anchors: (0.28, 0.22), (0.38, 0.48), (0.9, 0.78). Activations: sigmoid for cx/cy/obj/class, exp for w/h.

**Edge mode** — Smaller variant: feature_dim=128, 2 fusion layers, 4 attention heads. For deployment on embedded hardware.

---

## 4. Dataset

### Real Dataset: LLVIP (Low-Light Visible-Infrared Person Pair)
- **Source:** BUPT AI-CZ Group — https://bupt-ai-cz.github.io/LLVIP/
- **Paper:** "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision" (ICCV 2021)
- **License:** Free for academic/non-commercial use

| Split | Pairs | Location |
|-------|-------|----------|
| Train | 12,025 | `data/LLVIP/visible/train/` + `data/LLVIP/infrared/train/` |
| Test | 3,463 | `data/LLVIP/visible/test/` + `data/LLVIP/infrared/test/` |

**Important:** Both RGB and thermal images in LLVIP are `.jpg` format. The dataset loader handles this — it first tries `infrared/{split}/{stem}.jpg` before falling back to `.png`.

**Annotations:** Originally Pascal VOC XML format (`LLVIP/Annotations/*.xml`). Converted to YOLO format (`.txt`) during setup using `prepare_llvip.py`. Labels stored at `data/LLVIP/Annotations/YOLO_Format/{train,test}/`.

### YOLO Label Format
```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```
All values normalized to [0,1]. Class 0 = person. Image resolution: 1280×1024.

### Preparation Script
`prepare_llvip.py` at project root:
1. Wipes any existing synthetic data from `data/LLVIP/`
2. Moves images from `LLVIP/` to `data/LLVIP/`
3. Converts XML annotations to YOLO `.txt` format

---

## 5. Training

### Configuration
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP \
  --epochs 50 \
  --batch-size 16 \
  --model full \
  --amp \
  --workers 2 \
  --ckpt-dir checkpoints
```

**Key flags:**
| Flag | Value | Reason |
|------|-------|--------|
| `--amp` | enabled | Mixed precision (FP16 forward, FP32 gradients) — fits 8GB VRAM |
| `--workers 2` | 2 | DataLoader multiprocessing — raised GPU util from 2% → 67-80% |
| `--workers 0` | (old) | Was needed to avoid Windows DataLoader crash; fixed by `if __name__ == "__main__"` guard |
| `--batch-size 16` | 16 | 8GB VRAM limit with AMP and 320px input |

### Resume from Checkpoint
```bash
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP --epochs 50 --batch-size 16 --model full \
  --amp --workers 2 --ckpt-dir checkpoints \
  --resume checkpoints/sar_epoch_040.pt
```
- Supports both periodic checkpoints (`sar_epoch_040.pt`) and best checkpoints (full dict with optimizer state)
- Advances `global_step` correctly so LR schedule continues from the right point

### Training Features
- **Cosine LR schedule** with linear warmup (5 epochs)
- **Hard negative mining** — pool of 512 false positives, injected at 20% of each batch every 10 steps
- **AMP** via `torch.amp.GradScaler` + `torch.amp.autocast`
- **Checkpoints saved** every 10 epochs (`sar_epoch_010.pt`, etc.) and whenever val loss improves (`best_sar_model.pt`)

### Epoch Timing
| Config | Time/epoch | Total (50 epochs) |
|--------|-----------|-------------------|
| workers=0 | ~212s | ~3h |
| workers=2 + persistent_workers | ~100s | ~83 min |

### Results (Real LLVIP, 50 epochs)
| Metric | Value |
|--------|-------|
| Best val loss | 0.0169 |
| Best val accuracy | ~92.87% (epoch 39) |
| Final epoch accuracy | 92.61% |
| Checkpoint | `checkpoints/best_sar_model.pt` |

---

## 6. Inference & Survival Scoring

### Preprocessing
- **RGB:** PIL → RGB → resize(320,320) → float32/255 → tensor (1,3,320,320)
- **Thermal:** PIL → L (grayscale) → resize(320,320) → float32/255 → tensor (1,1,320,320)
- Both cast to `.float()` before model — critical fix: AMP saves weights as float16; `.model.float()` after checkpoint load ensures fp32 inference

### Inference Pipeline
1. Preprocess both images
2. Forward pass with `torch.amp.autocast("cuda")`
3. Decode YOLO outputs → NMS → bounding boxes
4. Compute `consistency_score` from AnomalyFilter
5. Compute `survival_likelihood` (physics-informed, see below)
6. Build response dict

### Physics-Informed Survival Likelihood

Replaces a blind MLP head (no supervision signal) with a formula grounded in observable physics:

```
survival = 0.30 × thermal_signal
         + 0.25 × thermal_contrast
         + 0.25 × confidence
         + 0.10 × consistency_score
         + 0.10 × posture_factor
```

| Component | How computed | Why meaningful |
|-----------|-------------|----------------|
| `thermal_signal` | Mean thermal intensity inside detected bbox | Living body (~37°C) reads brighter than surroundings |
| `thermal_contrast` | sigmoid((bbox_mean − global_mean) / global_std) | How many std-devs above ambient temp; high = genuinely warm |
| `confidence` | Model's max objectness × class score | Detection quality anchor |
| `consistency_score` | AnomalyFilter cross-modal agreement | Real heat+shape vs artifact |
| `posture_factor` | Bbox aspect ratio: h/w ≥ 1.2 → 0.85 (upright), ≤ 0.7 → 0.30 (prone), else 0.60 | Mobility proxy |

**Implementation note:** Background stats use global image mean/std (pre-computed once), not a per-box mask. Person bbox is <1% of 320×320, so global ≈ background. Eliminates per-box `(320×320)` bool mask allocation.

**Typical scores:**
| Scenario | Score |
|----------|-------|
| No detection | 0.16 |
| Warm prone person | 0.58 |
| Warm upright person | 0.68–0.76 |
| Cold / no contrast | 0.53 |

### Limitations of Survival Scoring
The score is observable-signal-based, not medically validated. True survival probability depends on factors not in a single image: time since accident, injuries, weather, motion over time. For production use, this score should be treated as a **priority signal**, not a medical estimate.

---

## 7. Backend API

**Base URL:** `http://localhost:8000`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model status, device, uptime |
| POST | `/upload` | Upload RGB + thermal → DetectionResult |
| POST | `/detect` | Base64 inference (JSON body) |
| GET | `/stats` | Aggregate system statistics |
| WS | `/ws/live` | Real-time frame streaming |
| WS | `/ws/monitor` | Broadcast monitoring channel |
| GET | `/docs` | Swagger UI |

### DetectionResult Schema
```json
{
  "detected": true,
  "confidence": 0.6295,
  "consistency_score": 0.4460,
  "survival_likelihood": 0.6295,
  "bounding_boxes": [
    { "cx": 0.5, "cy": 0.3, "w": 0.05, "h": 0.12, "confidence": 0.62 }
  ],
  "gps_location": { "lat": 17.385, "lon": 78.487, "altitude": 50.0 },
  "explanation": "Detected human — RGB body shape aligned with thermal heat signature.",
  "frame_id": "F000001",
  "inference_ms": 35.2,
  "modality_weights": { "rgb": 0.45, "thermal": 0.55 },
  "timestamp": "2026-04-11T17:00:00Z"
}
```

### Configuration
Backend reads from `.env` at project root AND `backend/.env`:
```
MODEL_PATH=checkpoints/best_sar_model.pt
MODEL_MODE=full
CONF_THRESHOLD=0.45
IMG_SIZE=320
```

**Critical:** Backend MUST be launched from `d:/sar_system` (project root), not from `backend/`. Imports use `backend.` prefix.

---

## 8. Frontend

**URL:** `http://localhost:5173`

| Page | Route | Description |
|------|-------|-------------|
| Upload | `/upload` | Drag-and-drop RGB + thermal, shows DetectionResult card with bounding boxes |
| Stats | `/stats` | Live system statistics dashboard |

**Stack:** React 18, Vite 5, Tailwind CSS, WebSocket hooks

---

## 9. File Structure

```
d:/sar_system/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app, lifespan, engine init
│   │   ├── schemas.py           # Pydantic models (DetectionResult, etc.)
│   │   └── routes/
│   │       ├── upload.py        # POST /upload
│   │       ├── detect.py        # POST /detect (base64)
│   │       ├── stats.py         # GET /stats
│   │       └── ws.py            # WebSocket /ws/live, /ws/monitor
│   ├── models/
│   │   └── fusion_model.py      # SARFusionModel, DetectionHead, etc.
│   ├── inference/
│   │   └── engine.py            # InferenceEngine, compute_survival_likelihood
│   ├── training/
│   │   ├── train.py             # Training loop, --resume support
│   │   ├── dataset.py           # LLVIPDataset, DataLoader factory
│   │   └── losses.py            # SARLoss (detection + consistency + alignment)
│   ├── deployment/
│   │   └── edge_optimizer.py    # ONNX export for edge deployment
│   └── venv/                    # Python 3.12 virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── pages/               # Upload, Stats pages
│   │   ├── components/          # Reusable UI components
│   │   ├── hooks/               # WebSocket, API hooks
│   │   ├── services/            # API client
│   │   └── store/               # State management
│   └── node_modules/
│
├── simulation/
│   └── uav_simulation.py        # (Legacy) Synthetic frame generator
│
├── data/
│   └── LLVIP/
│       ├── visible/{train,test}/ # RGB JPEGs (12025 train, 3463 test)
│       ├── infrared/{train,test}/ # Thermal JPEGs (same filenames)
│       └── Annotations/YOLO_Format/{train,test}/ # YOLO .txt labels
│
├── checkpoints/
│   ├── best_sar_model.pt        # Best val loss checkpoint (USE THIS)
│   ├── sar_epoch_010.pt         # Periodic saves
│   ├── sar_epoch_020.pt
│   ├── sar_epoch_030.pt
│   ├── sar_epoch_040.pt         # Used for power-cut resume
│   └── sar_epoch_050.pt
│
├── logs/                        # Training logs (TensorBoard)
├── .env                         # Root env (copy of backend/.env)
├── prepare_llvip.py             # Dataset preparation script
├── generate_synthetic_llvip.py  # (Legacy) Synthetic data generator
└── PROJECT_DOCUMENTATION.md    # This file
```

---

## 10. Environment & Dependencies

### Python Environment
```
Path: d:/sar_system/backend/venv/
Python: 3.12
```

Key packages:
```
torch==2.6.0+cu124
torchvision==0.21.0+cu124
onnxruntime-gpu
fastapi
uvicorn[standard]
opencv-python-headless
Pillow
numpy
python-dotenv
pydantic
```

### Node.js
```
Path: d:/sar_system/frontend/node_modules/
```

### GPU
```
NVIDIA GeForce RTX 4060 Ti
VRAM: 8 GB
CUDA Driver: 13.2 (supports cu124 wheels)
torch.cuda.is_available(): True
```

---

## 11. Run Commands

### Backend
```bash
cd d:/sar_system
backend/venv/Scripts/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```
Expected output:
```
Loaded checkpoint: checkpoints/best_sar_model.pt
SAR backend ready on device: cuda
INFO: Application startup complete.
```

### Frontend
```bash
cd d:/sar_system/frontend
npm run dev
```

### Training (from scratch)
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP --epochs 50 --batch-size 16 \
  --model full --amp --workers 2 --ckpt-dir checkpoints
```

### Training (resume from checkpoint)
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP --epochs 50 --batch-size 16 \
  --model full --amp --workers 2 --ckpt-dir checkpoints \
  --resume checkpoints/sar_epoch_040.pt
```

### Verify GPU
```bash
cd d:/sar_system
backend/venv/Scripts/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Edge ONNX Export (optional)
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/deployment/edge_optimizer.py \
  --checkpoint checkpoints/best_sar_model.pt \
  --output checkpoints/sar_edge.onnx \
  --mode edge --benchmark
```

---

## 12. Training Results

### Phase 1 — Synthetic Data (superseded)
- 2,000 synthetic pairs (colored blobs + Gaussian thermal)
- 50 epochs, peak accuracy ~99% (on synthetic test set — meaningless)
- Model learned to detect synthetic artifacts, NOT real humans

### Phase 2 — Real LLVIP Data (current)
- 12,025 real night-vision pairs
- 50 epochs with resume after power cut (resumed from epoch 40)
- Training time: ~100s/epoch with workers=2

| Epoch | Val Loss | Val Accuracy |
|-------|----------|--------------|
| 1 | 0.300 | 0.00% |
| 4 | 0.018 | 69.59% |
| 11 | 0.014 | 85.27% |
| 21 | 0.010 | 81.95% ← best val loss |
| 29 | 0.011 | 91.11% |
| 39 | 0.017 | **92.87%** ← best accuracy |
| 50 | 0.022 | 92.61% |
| **Best** | **0.0169** | **~92.87%** |

**GPU utilization improvement:**
| Config | GPU Util | Epoch time |
|--------|----------|-----------|
| workers=0 | 2% | 212s |
| workers=2 + persistent_workers | 67-80% | ~100s |

---

## 13. Known Issues & Notes

### Port Conflict on Restart
After power cut or abnormal shutdown, Windows may retain ghost TCP sockets on port 8000 (PIDs exist in netstat but process is dead). Wait ~4 minutes for TCP TIME_WAIT to expire, or use a different port (`--port 8080`).

### Checkpoint dtype after AMP Training
AMP training saves model weights as float16. The inference engine casts back to float32 with `model.float()` after loading. Without this, inference fails with:
```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
```

### Windows DataLoader Workers
`--workers 0` was historically required on Windows to avoid multiprocessing crashes. This is fixed — the `if __name__ == "__main__":` guard in `train.py` makes `--workers 2` safe on Windows.

### Real vs Synthetic Performance
The 92% accuracy is on real LLVIP night-time pedestrian data. LLVIP images are fixed-camera street scenes with clear upright pedestrians. UAV aerial imagery is harder (top-down view, varying altitude, more occlusion) — real-world UAV performance will be lower.

### Survival Score Limitations
The survival likelihood score is physics-informed but not medically validated. It uses thermal intensity, thermal contrast, detection confidence, consistency, and posture (aspect ratio). It does NOT account for: time since injury, actual body temperature (only relative brightness), motion over time, or environmental conditions.

---

## 14. Next Steps

### Immediate
- [ ] Test end-to-end through the frontend UI at http://localhost:5173/upload
- [ ] Verify WebSocket live feed at `/ws/live` with streaming images

### Model Improvements
- [ ] **UAV-specific fine-tuning** — LLVIP is fixed-camera street scenes; collect or find top-down UAV thermal data for aerial perspective
- [ ] **Multi-scale detection** — add FPN (Feature Pyramid Network) for detecting people at varying UAV altitudes
- [ ] **Temporal fusion** — use multiple consecutive frames for motion detection (increases survival score reliability)
- [ ] **Real survival scoring** — integrate ISRID statistical priors (terrain type, time missing) as additional survival score inputs

### Deployment
- [ ] **ONNX export** — run `edge_optimizer.py` to produce `sar_edge.onnx` for edge/embedded deployment
- [ ] **Docker containerization** — containerize backend for reproducible deployment
- [ ] **RTSP stream integration** — connect `/ws/live` to actual UAV camera feed

### Infrastructure
- [ ] Swap ghost ports issue — add proper process cleanup script for Windows restarts
- [ ] Add TensorBoard logging (logs dir already created)
