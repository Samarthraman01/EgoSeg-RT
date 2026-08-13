import os
import cv2
import torch
import numpy as np 
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import time

#---class mapping ------- visor to ade20k
VISOR_TO_ADE20K = {
    # structural (high pixel count → most reliable)
    63:  26,   # sink → sink
    159:  3,   # floor → floor
    3:   10,   # cupboard → cabinet
    12:  28,   # fridge → refrigerator

    # hands (high pixel count)
    300: 12,   # hand:left → person
    301: 12,   # hand:right → person
    303: 12,   # glove:left → person
    304: 12,   # glove:right → person

    # smaller but valid matches
    15:  44,   # bottle → bottle
    
    23:  38,   # box → box
}

#-----Setup-------------------------------
# -------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

FRAMES_DIR  = "data/visor_hos/val_frames/"
MASKS_DIR   = "data/visor_hos/val_masks/"
OUTPUT_DIR  = "data/visor_hos/comparisons/"
MODEL_PATH  = os.path.expanduser("~/mediseg/models/segformer_b2_epoch9.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

#----model------------------
print("loading model")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded!")

#------preprocessing---------
transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

np.random.seed(42)
colour_map = np.random.randint(0, 255, (150,3), dtype=np.uint8)
colour_map[0] = [0,0,0]

iou_scores = {ade_cls: [] for ade_cls in set(VISOR_TO_ADE20K.values())}

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
print(f"Processing {len(frames)} frames...")

for i, fname in enumerate(frames):
    # load RGB frame
    frame_bgr = cv2.imread(os.path.join(FRAMES_DIR, fname))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_bgr.shape[:2]

    # load ground truth mask
    mask_name = fname.replace('.jpg', '.png')
    mask_path = os.path.join(MASKS_DIR, mask_name)
    if not os.path.exists(mask_path):
        continue

    gt_mask_raw = np.array(Image.open(mask_path))  # values = class_id + 1

    # run SegFormer inference
    pil_image    = Image.fromarray(frame_rgb)
    input_tensor = transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs   = model(pixel_values=input_tensor)
        predicted = torch.argmax(outputs.logits, dim=1)
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        ).squeeze().long().cpu().numpy()

    # calculate IoU for each matched class
    for visor_id, ade_id in VISOR_TO_ADE20K.items():
        # ground truth: pixels where VISOR class = visor_id
        # mask value = visor_id + 1
        gt_binary   = (gt_mask_raw == visor_id + 1)

        if gt_binary.sum() == 0:
            continue  # this class not in frame

        # prediction: pixels where SegFormer predicted ade_id
        pred_binary = (predicted == ade_id)

        intersection = (gt_binary & pred_binary).sum()
        union        = (gt_binary | pred_binary).sum()

        if union > 0:
            iou = intersection / union
            iou_scores[ade_id].append(float(iou))

    # save 3-panel comparison: original | prediction | gt overlay
    pred_colour = colour_map[predicted]
    pred_bgr    = cv2.cvtColor(pred_colour, cv2.COLOR_RGB2BGR)

    # resize all to same height for comparison
    orig_small = cv2.resize(frame_bgr, (640, 360))
    pred_small = cv2.resize(pred_bgr,  (640, 360))

    comparison = np.hstack([orig_small, pred_small])
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), comparison)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(frames)}", flush=True)

# ── Results ───────────────────────────────────────────────────────────
print(f"\n── VISOR Distribution Shift Results ─────────────")
print(f"ADE20K mIoU (same domain):    0.350")
print(f"\nMatched class IoU on VISOR:")

class_names = {
    26:  'sink',
    3:   'floor',
    10:  'cabinet/cupboard',
    28:  'refrigerator',
    12:  'person/hand',
    44:  'bottle',
    38:  'box',
}
all_ious = []

for ade_id, scores in iou_scores.items():
    if scores:
        mean_iou = np.mean(scores)
        all_ious.append(mean_iou)
        print(f"  {class_names.get(ade_id, ade_id)}: {mean_iou:.3f} "
              f"(from {len(scores)} frames)")

if all_ious:
    print(f"\nMean IoU (matched classes): {np.mean(all_ious):.3f}")
    print(f"Distribution shift:         {0.350 - np.mean(all_ious):.3f} drop")

print(f"\nComparisons saved to: {OUTPUT_DIR}")