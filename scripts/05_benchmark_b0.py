import os 
import cv2
import torch
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import time 
import numpy as np

device = torch.device(("mps") if torch.backends.mps.is_available() else "cpu")
print(f"running on : {device}")

FRAMES_DIR = os.path.expanduser("~/egoseg_rt/data/frames/")
MASKS_DIR  = os.path.expanduser("~/egoseg_rt/data/masks_b0/")
MODEL_PATH = os.path.expanduser("~/EGOSEG_RT/models/segformer_b0_epoch10.pt")

os.makedirs(MASKS_DIR, exist_ok=True)

#_-load model-
print("loading segformer...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",  # ← b0 not b2
    num_labels=150,
    ignore_mismatched_sizes=True
)
model = model.to(device)
model.eval()

#-preproccessing-
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#--colour maps--
np.random.seed(42)
colour_map = np.random.randint(0, 255, (150, 3), dtype = np.uint8)
colour_map[0] = [0,0,0]

#_-process each frames--
frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
print(f"Processing {len(frames)} frames.....")

latencies = []

for i, fnames in enumerate(frames):

    frames_bgr = cv2. imread(os.path.join(FRAMES_DIR, fnames))
    frames_rgb = cv2.cvtColor(frames_bgr, cv2.COLOR_BGR2RGB) #to convert into rgb format
    pil_images = Image.fromarray(frames_rgb)

    input_tensor = transform(pil_images).unsqueeze(0).to(device)

    #inference time
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(pixel_values = input_tensor)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1)
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1).float(),
            size=(frames_bgr.shape[0], frames_bgr.shape[1]),
            mode='nearest'
        ).squeeze().long().cpu().numpy()
    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

    mask_rgb = colour_map[predicted]
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    #save mask
    cv2.imwrite(os.path.join(MASKS_DIR, fnames), mask_bgr)

    if (i + 1) % 10 == 0:
        print(f"processed {i+1}/{len(frames)} "
        f"latency: {latency_ms:.1f}ms", flush = True)

#Result        
avg_latency = np.mean(latencies)
avg_fps     = 1000 / avg_latency

print(f"\n── Results ──────────────────────────")
print(f"Frames processed: {len(frames)}")
print(f"Avg latency:      {avg_latency:.1f} ms")
print(f"Avg FPS:          {avg_fps:.2f}")
print(f"Masks saved to:   {MASKS_DIR}")