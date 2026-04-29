"""
Prepare real LLVIP dataset:
1. Delete synthetic data from data/LLVIP/
2. Move real images from LLVIP/ to data/LLVIP/
3. Convert Pascal VOC XML annotations → YOLO .txt format
   into data/LLVIP/Annotations/YOLO_Format/{train,test}/
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC  = Path("d:/sar_system/LLVIP")
DEST = Path("d:/sar_system/data/LLVIP")

# --- Step 1: wipe synthetic data ---
print("Step 1: Removing synthetic data...")
for split in ("train", "test"):
    for sub in ("visible", "infrared"):
        folder = DEST / sub / split
        if folder.exists():
            shutil.rmtree(folder)
            folder.mkdir(parents=True)
    yolo_dir = DEST / "Annotations" / "YOLO_Format" / split
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    yolo_dir.mkdir(parents=True)
print("  Done.\n")

# --- Step 2: move images ---
print("Step 2: Moving images...")
for split in ("train", "test"):
    for sub in ("visible", "infrared"):
        src_dir  = SRC  / sub / split
        dest_dir = DEST / sub / split
        files = list(src_dir.glob("*.jpg"))
        print(f"  {sub}/{split}: {len(files)} files")
        for f in files:
            shutil.move(str(f), dest_dir / f.name)
print("  Done.\n")

# --- Step 3: convert XML → YOLO ---
print("Step 3: Converting XML annotations to YOLO format...")

IMG_W, IMG_H = 1280, 1024  # fixed for LLVIP

def xml_to_yolo(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        w = int(size.findtext("width") or IMG_W)
        h = int(size.findtext("height") or IMG_H)
    else:
        w, h = IMG_W, IMG_H

    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines

for split in ("train", "test"):
    img_dir  = DEST / "visible" / split
    yolo_dir = DEST / "Annotations" / "YOLO_Format" / split
    xml_dir  = SRC  / "Annotations"

    imgs = list(img_dir.glob("*.jpg"))
    converted = 0
    no_label = 0
    for img in imgs:
        xml_path = xml_dir / (img.stem + ".xml")
        txt_path = yolo_dir / (img.stem + ".txt")
        if xml_path.exists():
            lines = xml_to_yolo(xml_path)
            txt_path.write_text("\n".join(lines))
            converted += 1
        else:
            # No annotation = no person in frame, write empty label
            txt_path.write_text("")
            no_label += 1

    print(f"  {split}: {converted} with labels, {no_label} empty (no person)")

print("\nAll done! data/LLVIP/ is ready with real LLVIP data.")
print(f"  visible/train:  {len(list((DEST/'visible'/'train').glob('*.jpg')))}")
print(f"  visible/test:   {len(list((DEST/'visible'/'test').glob('*.jpg')))}")
print(f"  infrared/train: {len(list((DEST/'infrared'/'train').glob('*.jpg')))}")
print(f"  infrared/test:  {len(list((DEST/'infrared'/'test').glob('*.jpg')))}")
