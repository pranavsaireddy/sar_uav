"""
Download pre-trained SAR model weights from GitHub Releases.
Run from the project root: python download_weights.py
"""

import urllib.request
import os
from pathlib import Path

WEIGHTS_URL = "https://github.com/pranavsaireddy/sar_uav/releases/download/v1.0/best_sar_model.pt"
DEST = Path("checkpoints/best_sar_model.pt")


def download():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"Weights already exist at {DEST} — skipping download.")
        return

    print(f"Downloading weights from GitHub Releases...")
    print(f"  -> {DEST}  (~181 MB)")

    def progress(count, block_size, total):
        pct = min(100, count * block_size * 100 // total)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(WEIGHTS_URL, DEST, reporthook=progress)
    print(f"\nDone. Saved to {DEST}")


if __name__ == "__main__":
    download()
