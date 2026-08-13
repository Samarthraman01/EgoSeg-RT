import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
from PIL import Image, ImageDraw, ImageFont
import os

device = torch.device("mps")

MODEL_PATH = os.path.expanduser("~/mediseg/models/segformer_b2_epoch9.pt")
FRAME_DIR  = "data/visor_hos/val_frames/"
MASK_DIR   = "data/visor_hos/val_masks/"
OUTPUT_DIR = "data/visor_hos/qualitative/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load model
print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# colour maps
np.random.seed(42)
colour_map = np.random.randint(0, 255, (150, 3), dtype=np.uint8)
colour_map[0] = [0, 0, 0]

# VISOR colour map
np.random.seed(99)
visor_colour_map = np.random.randint(0, 255, (400, 3), dtype=np.uint8)
visor_colour_map[0] = [0, 0, 0]

# interesting frames to show
interesting_frames = [
    "P01_107_frame_0000003671.jpg",  # sink frame
    "P01_107_frame_0000001369.jpg",  # door + hand frame
    "P02_02_frame_0000001369.jpg",   # different kitchen
    "P03_14_frame_0000001369.jpg",   # another kitchen
    "P07_101_frame_0000001369.jpg",  # another kitchen
]

# use first 5 frames that exist
frames_to_show = []
all_frames = sorted(os.listdir(FRAME_DIR))
for f in all_frames:
    if len(frames_to_show) >= 5:
        break
    mask_path = os.path.join(MASK_DIR, f.replace('.jpg', '.png'))
    if os.path.exists(mask_path):
        frames_to_show.append(f)

print(f"Creating qualitative analysis for {len(frames_to_show)} frames...")

for fname in frames_to_show:
    # load frame
    frame_bgr = cv2.imread(os.path.join(FRAME_DIR, fname))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_bgr.shape[:2]

    # run model
    inp = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = model(pixel_values=inp)
        pred = torch.argmax(out.logits, dim=1)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(),
            size=(H, W), mode='nearest'
        ).squeeze().long().cpu().numpy()

    # load GT mask
    gt_raw = np.array(Image.open(
        os.path.join(MASK_DIR, fname.replace('.jpg', '.png'))))

    # create colour images
    pred_colour = colour_map[pred]
    gt_colour   = visor_colour_map[np.clip(gt_raw, 0, 399)]

    # resize all to same size for display
    display_W, display_H = 640, 360
    orig_small = cv2.resize(frame_bgr, (display_W, display_H))
    pred_small = cv2.resize(
        cv2.cvtColor(pred_colour, cv2.COLOR_RGB2BGR),
        (display_W, display_H))
    gt_small   = cv2.resize(
        cv2.cvtColor(gt_colour, cv2.COLOR_RGB2BGR),
        (display_W, display_H))

    # add text labels
    def add_label(img, text):
        cv2.putText(img, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255,255,255), 2)
        return img

    add_label(orig_small, "Original (egocentric)")
    add_label(pred_small, "SegFormer prediction")
    add_label(gt_small,   "VISOR ground truth")

    # combine 3 panels
    combined = np.hstack([orig_small, pred_small, gt_small])

    # add frame name at top
    header = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, fname, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    final = np.vstack([header, combined])

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, final)
    print(f"Saved: {out_path}")

print(f"\nDone! Qualitative figures saved to {OUTPUT_DIR}")