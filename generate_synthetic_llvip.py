"""
Generate synthetic data in LLVIP folder structure so the existing
LLVIPDataset loader works without any code changes.

Output layout:
  data/LLVIP/
    visible/train/*.jpg
    visible/test/*.jpg
    infrared/train/*.png
    infrared/test/*.png
    Annotations/YOLO_Format/train/*.txt
    Annotations/YOLO_Format/test/*.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulation.uav_simulation import SyntheticFrameGenerator
import base64
import argparse


def generate(output_dir: str, n_samples: int = 2000, split_ratio: float = 0.85):
    root = Path(output_dir)
    n_train = int(n_samples * split_ratio)

    for split in ("train", "test"):
        (root / "visible" / split).mkdir(parents=True, exist_ok=True)
        (root / "infrared" / split).mkdir(parents=True, exist_ok=True)
        (root / "Annotations" / "YOLO_Format" / split).mkdir(parents=True, exist_ok=True)

    gen = SyntheticFrameGenerator(size=320, human_probability=0.65)

    for i in range(n_samples):
        split = "train" if i < n_train else "test"
        frame = gen.generate_frame()
        fid = frame["frame_id"]

        # Save RGB as JPEG
        rgb_bytes = base64.b64decode(frame["rgb_b64"])
        (root / "visible" / split / f"{fid}.jpg").write_bytes(rgb_bytes)

        # Save thermal as PNG (grayscale)
        thm_bytes = base64.b64decode(frame["thermal_b64"])
        (root / "infrared" / split / f"{fid}.png").write_bytes(thm_bytes)

        # Save YOLO label
        lbl_path = root / "Annotations" / "YOLO_Format" / split / f"{fid}.txt"
        with open(lbl_path, "w") as f:
            for box in frame["boxes"]:
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples} ({split})")

    print(f"\nDone. {n_train} train + {n_samples - n_train} test samples in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/LLVIP")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    generate(args.output, args.samples)
