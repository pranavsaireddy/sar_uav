# SAR UAV Detection System — Setup Guide

## Prerequisites

Install these before anything else:

| Tool | Version | Link |
|---|---|---|
| Python | 3.11 or 3.12 | https://python.org |
| Node.js | 20 LTS | https://nodejs.org |
| CUDA Toolkit | 12.4 | https://developer.nvidia.com/cuda-downloads |

> No GPU? Skip CUDA — the system falls back to CPU automatically (slower inference).

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/pranavsaireddy/sar_uav.git
cd sar_uav
```

---

## Step 2 — Download the pre-trained model weights

```bash
python download_weights.py
```

This downloads `best_sar_model.pt` (~181 MB) into the `checkpoints/` folder.
Trained on the LLVIP dataset — no training needed.

---

## Step 3 — Set up the Python backend

```bash
cd backend
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
# With NVIDIA GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only (no GPU)
pip install torch torchvision

# Remaining packages
pip install -r requirements.txt
```

Go back to project root:

```bash
cd ..
```

---

## Step 4 — Set up the frontend

```bash
cd frontend
npm install
cd ..
```

---

## Step 5 — Configure environment

Create a `.env` file in the project root:

```bash
# Windows
copy backend\.env.example .env 2>nul || echo MODEL_PATH=checkpoints/best_sar_model.pt > .env

# Linux / macOS
cp backend/.env.example .env 2>/dev/null || echo "MODEL_PATH=checkpoints/best_sar_model.pt" > .env
```

Default values work out of the box — no changes needed unless you want to tweak ports or thresholds.

---

## Step 6 — Run

Open **two terminals**, both from the project root (`sar_uav/`).

**Terminal 1 — Backend:**

```bash
backend/venv/Scripts/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
# Linux/macOS: backend/venv/bin/uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
Loaded checkpoint: checkpoints/best_sar_model.pt
SAR backend ready on device: cuda
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
npm run dev
```

You should see:
```
VITE ready in Xms → http://localhost:5173/
```

---

## Step 7 — Open the app

Go to **http://localhost:5173** in your browser.

| Page | What to do |
|---|---|
| **Upload** | Drop an RGB image + a thermal image → see detection result with bounding boxes |
| **Dashboard** | Live WebSocket stream — connect a UAV feed or use the simulator |
| **History** | Browse past detections, filter by confidence, export CSV |
| **Stats** | Detection rates, latency charts, modality weight breakdown |

### Quick test (no UAV needed)

Go to **Upload**, use any two images:
- RGB: a regular photo (`.jpg` / `.png`)
- Thermal: a grayscale image of the same scene (or any grayscale image to test the pipeline)

The API is also accessible at **http://localhost:8000/docs** (Swagger UI).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'backend'` | Make sure you run uvicorn from the project root, not from `backend/` |
| `Port 8000 already in use` | Add `--port 8001` to the uvicorn command, update `frontend/vite.config.js` proxy accordingly |
| `CUDA out of memory` | The model runs on CPU if CUDA fails — check console for `device: cpu` |
| `Weights already exist` | `download_weights.py` skips re-download automatically |
| Frontend shows CORS error | Confirm backend is running on port 8000 and frontend on 5173 |
| `npm install` fails | Delete `frontend/node_modules/` and retry |

---

## Want to retrain?

You don't need to — the downloaded weights are production-ready. But if you want to:

```bash
# Download the LLVIP dataset first: https://bupt-ai-cz.github.io/LLVIP/
# Place it at data/LLVIP/

python backend/venv/Scripts/python -m backend.training.train \
  --data data/LLVIP \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --amp \
  --workers 2 \
  --dataset-type llvip \
  --ckpt-dir checkpoints
```

Requires ~45 min on an RTX 4060 Ti with CUDA.
