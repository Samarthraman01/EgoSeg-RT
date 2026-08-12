!pip install transformers -q

import torch
import numpy as np
import time
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import torchvision.transforms as transforms

# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:  {device}")
print(f"GPU:     {torch.cuda.get_device_name(0)}")
print(f"Memory:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Preprocessing ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Create dummy frames ───────────────────────────────────────────────
# FPS does not depend on image content
# same computation for any 512x512 input
NUM_FRAMES = 112
dummy_frames = [
    Image.fromarray(
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    )
    for _ in range(NUM_FRAMES)
]
print(f"\nCreated {NUM_FRAMES} dummy frames (1280x720)")

# ── Benchmark function ────────────────────────────────────────────────
def benchmark(model, frames, device, label):
    model.eval()
    latencies = []

    # warmup — GPU needs a few runs to reach stable speed
    print(f"\nWarming up {label}...")
    for _ in range(10):
        inp = transform(frames[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(pixel_values=inp)
        torch.cuda.synchronize()

    # benchmark
    print(f"Benchmarking {label}...")
    for i, frame in enumerate(frames):
        inp = transform(frame).unsqueeze(0).to(device)

        torch.cuda.synchronize()  # wait for GPU to finish previous ops
        start = time.perf_counter()

        with torch.no_grad():
            outputs   = model(pixel_values=inp)
            predicted = torch.argmax(outputs.logits, dim=1)

        torch.cuda.synchronize()  # wait for this op to finish
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        if (i + 1) % 20 == 0:
            print(f"  frame {i+1}/{len(frames)} — {latency_ms:.1f}ms")

    avg_latency = np.mean(latencies[10:])  # skip first 10 warmup
    avg_fps     = 1000 / avg_latency

    print(f"\n── {label} ──────────────────")
    print(f"Avg latency:  {avg_latency:.1f} ms")
    print(f"Avg FPS:      {avg_fps:.2f}")

    return avg_latency, avg_fps

# ── SegFormer-B2 GPU ──────────────────────────────────────────────────
print("\n" + "="*50)
print("Loading SegFormer-B2...")
model_b2 = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
).to(device)

b2_latency, b2_fps = benchmark(
    model_b2, dummy_frames, device,
    "SegFormer-B2 PyTorch T4 GPU")

del model_b2
torch.cuda.empty_cache()

# ── SegFormer-B0 GPU ──────────────────────────────────────────────────
print("\n" + "="*50)
print("Loading SegFormer-B0...")
model_b0 = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
).to(device)

b0_latency, b0_fps = benchmark(
    model_b0, dummy_frames, device,
    "SegFormer-B0 PyTorch T4 GPU")

del model_b0
torch.cuda.empty_cache()

# ── Complete paper table ──────────────────────────────────────────────
print("\n" + "="*60)
print("COMPLETE BENCHMARK TABLE — EgoSeg-RT")
print("="*60)
print(f"{'Model':<22} {'Hardware':<14} {'FPS':>7}  {'Latency':>10}")
print("-"*60)
print(f"{'U-Net':<22} {'M2 CPU':<14} {'1.3':>7}  {'769ms':>10}")
print(f"{'SegFormer-B2':<22} {'M2 CPU':<14} {'1.3':>7}  {'769ms':>10}")
print(f"{'SegFormer-B2':<22} {'M2 MPS':<14} {'7.3':>7}  {'137ms':>10}")
print(f"{'SegFormer-B2 ONNX':<22} {'M2 CPU':<14} {'2.2':>7}  {'450ms':>10}")
print(f"{'SegFormer-B2 INT8':<22} {'M2 CPU':<14} {'1.0':>7}  {'1000ms':>10}")
print(f"{'SegFormer-B0':<22} {'M2 MPS':<14} {'21.1':>7}  {'47ms':>10}")
print(f"{'SegFormer-B0 ONNX':<22} {'M2 CPU':<14} {'12.6':>7}  {'79ms':>10}")
print(f"{'SegFormer-B2':<22} {'T4 GPU':<14} {b2_fps:>7.1f}  {b2_latency:>9.1f}ms")
print(f"{'SegFormer-B0':<22} {'T4 GPU':<14} {b0_fps:>7.1f}  {b0_latency:>9.1f}ms")
print("="*60)