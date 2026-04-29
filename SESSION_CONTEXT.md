# SAR UAV Detection System — Session Context
*Last updated: 2026-04-11*

---

## What Was Accomplished This Session

### 1. Environment Setup — COMPLETE
| Item | Status | Detail |
|------|--------|--------|
| Python 3.12 venv | Done | `d:/sar_system/backend/venv/` |
| PyTorch 2.6.0+cu124 | Done | CUDA 13.2 on RTX 4060 Ti (8 GB VRAM) confirmed |
| Backend pip deps | Done | fastapi, uvicorn, opencv, onnxruntime-gpu, pydantic, etc. |
| Node.js npm deps | Done | `d:/sar_system/frontend/node_modules/` installed |
| Required directories | Done | `checkpoints/`, `logs/`, `data/` created at project root |
| `.env` config | Done | `backend/.env` + copy at project root `.env` |

**Verified GPU:** `torch.cuda.is_available() = True`, `NVIDIA GeForce RTX 4060 Ti`

---

### 2. Bug Fixes Applied
| File | Bug | Fix |
|------|-----|-----|
| `simulation/uav_simulation.py:51` | `np.random.randint(..., dtype=float32)` not supported | Changed to `.astype(np.float32)` |
| `simulation/uav_simulation.py:58` | `np.random.randn(h, w)` shape mismatch vs actual slice | Changed to `np.random.randn(*slc.shape)` |
| `backend/training/losses.py:69` | `F.binary_cross_entropy` unsafe with AMP | Wrapped in `torch.amp.autocast("cuda", enabled=False)` |
| `backend/training/train.py` | Deprecated `torch.cuda.amp.GradScaler/autocast` | Updated to `torch.amp.GradScaler/autocast` |
| `backend/training/train.py:267` | Unicode `✓` char fails on Windows cp1252 | Replaced with `[BEST]` ASCII text |
| `backend/training/train.py:160` | Checkpoint saved to wrong dir (`data/checkpoints/`) | Added `--ckpt-dir` argument, defaults to correct path |

---

### 3. Synthetic Dataset — COMPLETE
- **2000 samples** generated in LLVIP format at `data/LLVIP/`
- **1700 train / 300 test** (85/15 split)
- Structure matches what `LLVIPDataset` expects:
  ```
  data/LLVIP/
    visible/train/     ← 1700 RGB .jpg files
    visible/test/      ← 300 RGB .jpg files
    infrared/train/    ← 1700 thermal .png files
    infrared/test/     ← 300 thermal .png files
    Annotations/YOLO_Format/train/  ← 1700 .txt label files
    Annotations/YOLO_Format/test/   ← 300 .txt label files
  ```
- Generator script: `generate_synthetic_llvip.py` at project root

---

### 4. Training — IN PROGRESS (running as background process)
- **Command:** `backend/venv/Scripts/python backend/training/train.py --data data/LLVIP --epochs 50 --batch-size 16 --model full --amp --workers 0 --ckpt-dir checkpoints`
- **Run from:** `d:/sar_system` (project root)
- **Epoch 1 verified:** Loss=1.0005 → Val Loss=0.6089, Accuracy=48.67% (takes ~35s/epoch)
- **Expected total time:** ~30 minutes for 50 epochs
- **Checkpoint output:** `d:/sar_system/checkpoints/best_sar_model.pt`
- **Status:** Still running in background — may or may not be complete when next session starts

---

### 5. Both Servers — RUNNING
| Server | URL | Process |
|--------|-----|---------|
| Backend (uvicorn) | http://localhost:8000 | Running (background task `bt0z9vxsk`) |
| Frontend (Vite) | http://localhost:5173 | Running (background task `bb4yn6mcz`) |
| Swagger docs | http://localhost:8000/docs | Active |

**Note:** Both servers will need to be restarted in a new terminal session (they don't persist across Windows restarts or shell sessions).

---

## IMPORTANT: Real LLVIP Dataset

The HuggingFace dataset `srivastavagaurang/LLVIP` is **private/gated (401)** — cannot be auto-downloaded.

To get the real LLVIP dataset (~2.6 GB):
1. Go to: https://bupt-ai-cz.github.io/LLVIP/
2. Download the full dataset (Google Drive or Baidu Pan link on the page)
3. Extract and organize as:
   ```
   data/LLVIP/
     visible/train/    ← 19903 RGB JPEGs
     visible/test/     ← 3463 RGB JPEGs
     infrared/train/   ← 19903 thermal PNGs (same filenames)
     infrared/test/    ← 3463 thermal PNGs
     Annotations/YOLO_Format/train/  ← .txt YOLO labels
     Annotations/YOLO_Format/test/
   ```
4. **Delete the synthetic data** first: `rm -rf data/LLVIP/*`
5. Retrain with: `python backend/training/train.py --data data/LLVIP --epochs 50 --batch-size 16 --model full --amp --workers 4 --ckpt-dir checkpoints`

---

## What Is Left To Do (Next Session)

### Step 1 — Training checkpoints already exist
Training ran through at least **epoch 10** before session ended. Two checkpoints are saved:
- `checkpoints/best_sar_model.pt` — best val loss so far (use this)
- `checkpoints/sar_epoch_010.pt` — epoch 10 periodic save

**Option A — Use existing checkpoint (ready now):** Skip retraining and go straight to Step 2.

**Option B — Continue training for full 50 epochs (recommended):**
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP \
  --epochs 50 \
  --batch-size 16 \
  --model full \
  --amp \
  --workers 0 \
  --ckpt-dir checkpoints
```
*(Training restarts from epoch 1 — checkpoint saving will overwrite `best_sar_model.pt` only if val loss improves)*

### Step 2 — Restart backend with trained model
```bash
# Kill any existing uvicorn process first (Task Manager or Ctrl+C)
cd d:/sar_system
backend/venv/Scripts/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
# Should print: "Loaded checkpoint: checkpoints/best_sar_model.pt"
```

### Step 3 — Restart frontend
```bash
cd d:/sar_system/frontend
npm run dev
```

### Step 4 — Test the model end-to-end

**Option A — Swagger UI (easiest):**
1. Open http://localhost:8000/docs
2. Go to `POST /upload`
3. Upload any two images (one as RGB, one as thermal)
4. Check DetectionResult response

**Option B — Curl test:**
```bash
# Use any two images from the synthetic dataset
cd d:/sar_system
curl -X POST http://localhost:8000/upload \
  -F "rgb_file=@data/LLVIP/visible/test/SYN000001.jpg" \
  -F "thermal_file=@data/LLVIP/infrared/test/SYN000001.png"
```

**Option C — Frontend Upload page:**
1. Open http://localhost:5173/upload
2. Drag RGB image into first dropzone
3. Drag thermal image into second dropzone
4. Click Submit — see detection result card with bounding boxes

### Step 5 — Run edge optimization (optional)
```bash
cd d:/sar_system
backend/venv/Scripts/python backend/deployment/edge_optimizer.py \
  --checkpoint checkpoints/best_sar_model.pt \
  --output checkpoints/sar_edge.onnx \
  --mode edge \
  --benchmark
```

---

## Key File Locations

| File | Purpose |
|------|---------|
| `d:/sar_system/backend/venv/` | Python 3.12 virtual environment |
| `d:/sar_system/.env` | Root .env (copy of backend/.env) |
| `d:/sar_system/backend/.env` | Backend environment config |
| `d:/sar_system/data/LLVIP/` | Synthetic training dataset (2000 pairs) |
| `d:/sar_system/checkpoints/best_sar_model.pt` | Trained model weights (after training completes) |
| `d:/sar_system/generate_synthetic_llvip.py` | Script to regenerate synthetic data |
| `d:/sar_system/frontend/node_modules/` | npm packages (already installed) |

---

## Run Commands Summary

```bash
# --- BACKEND ---
cd d:/sar_system
backend/venv/Scripts/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# --- FRONTEND ---
cd d:/sar_system/frontend
npm run dev

# --- TRAINING ---
cd d:/sar_system
backend/venv/Scripts/python backend/training/train.py \
  --data data/LLVIP \
  --epochs 50 \
  --batch-size 16 \
  --model full \
  --amp \
  --workers 0 \
  --ckpt-dir checkpoints

# --- VERIFY GPU ---
cd d:/sar_system
backend/venv/Scripts/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# --- CHECK TRAINING PROGRESS ---
ls checkpoints/
```

---

## Known Issues / Notes

- **GPU VRAM:** System has 8 GB (not 16 GB as the build doc says). `batch-size 16 + AMP` fits fine (~6 GB used).
- **workers=0** required on Windows to avoid DataLoader multiprocessing crash. Set to 4 on Linux.
- **Demo mode:** Until the checkpoint is present, the backend runs with random weights but is fully functional for testing the API pipeline.
- **Synthetic data limitation:** Model trained on synthetic data will have lower accuracy than LLVIP-trained model. Use real LLVIP for production.
- **AMP deprecation warnings:** Fixed — now uses `torch.amp` API instead of `torch.cuda.amp`.
- **Unicode print:** Fixed — Windows cp1252 doesn't support the checkmark character.

---

## Architecture Reminder

```
RGB (B,3,H,W) ─┐
                ├─► ModalityEncoder (EfficientNet-B0) ─► CrossModalAttention ─► FusionModule ─► AnomalyFilter ─► DetectionHead
Thermal (B,1,H,W)─┘
```

- **Model:** SARFusionModel (~15.7M params, full mode)
- **Inference:** ~12 ms on RTX 4060 Ti at 320px FP16
- **API:** FastAPI + uvicorn (port 8000), React + Vite (port 5173)
- **WebSocket:** `/ws/live` for real-time streaming, `/ws/monitor` for broadcast
