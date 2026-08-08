import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from PIL import Image
import torchvision.transforms as transforms

# ── Setup ─────────────────────────────────────────────────────────────
FRAMES_DIR  = os.path.expanduser("~/egoseg_rt/data/frames/")
MASKS_DIR   = os.path.expanduser("~/egoseg_rt/data/masks_b0_onnx/")
MODEL_PATH  = os.path.expanduser("~/egoseg_rt/models/segformer_b0.onnx")

os.makedirs(MASKS_DIR, exist_ok=True)

# ── Load ONNX model ───────────────────────────────────────────────────
print("Loading SegFormer-B0 ONNX...")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CPUExecutionProvider']
)
print("Model loaded!")

# ── Preprocessing ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Colour map ────────────────────────────────────────────────────────
np.random.seed(42)
colour_map = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
colour_map[0] = [0, 0, 0]

# ── Process frames ────────────────────────────────────────────────────
frames    = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
latencies = []

print(f"Processing {len(frames)} frames with B0 ONNX on CPU...")

for i, fname in enumerate(frames):
    frame_bgr = cv2.imread(os.path.join(FRAMES_DIR, fname))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # preprocess → numpy for ONNX
    import torch
    input_tensor = transform(pil_image).unsqueeze(0).numpy()

    # run ONNX inference on CPU
    start = time.perf_counter()
    outputs    = session.run(None, {"pixel_values": input_tensor})
    logits     = outputs[0]  # (1, 150, 128, 128)
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    # argmax
    predicted = np.argmax(logits[0], axis=0).astype(np.uint8)

    # upsample to original size
    predicted = cv2.resize(
        predicted,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # colour map
    mask_rgb = colour_map[predicted]
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(MASKS_DIR, fname), mask_bgr)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(frames)} — "
              f"latency: {latency_ms:.1f}ms", flush=True)

# ── Results ───────────────────────────────────────────────────────────
avg_latency = np.mean(latencies)
avg_fps     = 1000 / avg_latency

print(f"\n── B0 ONNX Benchmark Results ────────────")
print(f"Model:        SegFormer-B0 ONNX FP32")
print(f"Hardware:     M2 CPU")
print(f"Frames:       {len(frames)}")
print(f"Avg latency:  {avg_latency:.1f} ms")
print(f"Avg FPS:      {avg_fps:.2f}")
print(f"\n── Comparison ───────────────────────────")
print(f"B0 PyTorch MPS:  47.5ms  → 21.05 FPS")
print(f"B0 ONNX CPU:     {avg_latency:.1f}ms → {avg_fps:.2f} FPS")
print(f"B2 PyTorch MPS:  137.3ms → 7.29 FPS")
print(f"B2 ONNX CPU:     450.5ms → 2.22 FPS")