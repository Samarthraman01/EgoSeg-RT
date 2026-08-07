import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import time

FRAMES_DIR = os.path.expanduser("~/egoseg_rt/data/frames/")
MASKS_DIR  = os.path.expanduser("~/egoseg_rt/data/masks/")
OUTPUT     = os.path.expanduser("~/egoseg_rt/outputs/demo_video.mp4")

os.makedirs(os.path.expanduser("~/egoseg_rt/outputs/"), exist_ok=True)

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
sample = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))
H, W   = sample.shape[:2]

pip_W = W // 3
pip_H = H // 3

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT, fourcc, 3, (W, H))

print(f"Creating demo video — {len(frames)} frames...")

# load latencies from script 02 if available
# otherwise use fixed average
avg_latency = 137.3
avg_fps     = 7.29

for i, fname in enumerate(frames):
    original = cv2.imread(os.path.join(FRAMES_DIR, fname))

    mask_path = os.path.join(MASKS_DIR, fname)
    if not os.path.exists(mask_path):
        continue
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (W, H))

    # full screen mask as background
    canvas = mask.copy()

    # pip — original in bottom left
    pip = cv2.resize(original, (pip_W, pip_H))
    pip = cv2.copyMakeBorder(pip, 3, 3, 3, 3,
        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    pip_Hb = pip.shape[0]
    pip_Wb = pip.shape[1]

    y1 = H - pip_Hb - 20
    y2 = H - 20
    x1 = 20
    x2 = 20 + pip_Wb
    canvas[y1:y2, x1:x2] = pip

    # ── Numerical overlays ────────────────────────────────────────────

    # top left — model info
    # top left — model info (was 0.7, make 1.0)
    cv2.putText(canvas,
        f"EgoSeg-RT | SegFormer-B2 | ADE20K",
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 4)

    # top right — FPS (was 0.7, make 1.0)
    cv2.putText(canvas,
        f"FPS: {avg_fps:.1f}",
        (W - 200, 45), cv2.FONT_HERSHEY_SIMPLEX,
        2, (0, 255, 0), 4)

    # second line — metrics (was 0.6, make 0.9)
    cv2.putText(canvas,
        f"mIoU: 0.350 | Latency: {avg_latency:.0f}ms",
        (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 255, 0), 4)

    # dominant class (new — large and centre)
    

    # frame counter (was 0.6, make 0.8)
    cv2.putText(canvas,
        f"Frame: {i+1:03d}/{len(frames)}",
        (W - 250, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 255, 255), 2)

    # hardware info (was 0.5, make 0.7)
    cv2.putText(canvas,
        "Apple M2 Pro (MPS)",
        (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX,
        2, (200, 200, 200), 4)

# pip label (was 0.5, make 0.7)
    cv2.putText(canvas, "Original POV",
        (x1 + 5, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 255, 255), 4)

    writer.write(canvas)

    if (i + 1) % 10 == 0:
        print(f"Written {i+1}/{len(frames)} frames")

writer.release()
print(f"\nDemo video saved to: {OUTPUT}")