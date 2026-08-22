import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
from PIL import Image

# ── MPS fix — patch decode head ───────────────────────────────────────
def _patched_decode_head_forward(self, encoder_hidden_states):
    batch_size = encoder_hidden_states[-1].shape[0]
    all_hidden_states = ()

    for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
        if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
            height = width = int(encoder_hidden_state.shape[-1] ** 0.5)
            encoder_hidden_state = (
                encoder_hidden_state
                .reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
        encoder_hidden_state = mlp(encoder_hidden_state)
        encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
        encoder_hidden_state = encoder_hidden_state.reshape(
            batch_size, -1, height, width)
        encoder_hidden_state = nn.functional.interpolate(
            encoder_hidden_state,
            size=encoder_hidden_states[0].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        all_hidden_states += (encoder_hidden_state,)

    hidden_states = self.linear_fuse(
        torch.cat(all_hidden_states[::-1], dim=1))
    hidden_states = hidden_states.contiguous()  # ← MPS fix
    hidden_states = self.batch_norm(hidden_states)
    hidden_states = self.activation(hidden_states)
    hidden_states = self.dropout(hidden_states)
    return self.classifier(hidden_states)

SegformerDecodeHead.forward = _patched_decode_head_forward

# ── Class mapping ─────────────────────────────────────────────────────
VISOR_TO_ADE20K = {
    63:  26,   # sink
    159:  3,   # floor
    3:   10,   # cupboard
    12:  28,   # fridge
    300: 12,   # hand left
    301: 12,   # hand right
    303: 12,   # glove left
    304: 12,   # glove right
    15:  44,   # bottle
    23:  38,   # box
}

lookup = np.full(400, 255, dtype=np.uint8)
for visor_id, ade_id in VISOR_TO_ADE20K.items():
    lookup[visor_id] = ade_id

# ── Dataset ───────────────────────────────────────────────────────────
class VISORDataset(Dataset):
    def __init__(self, frames_dir, masks_dir):
        self.frames_dir = frames_dir
        self.masks_dir  = masks_dir
        self.samples    = []

        for fname in sorted(os.listdir(frames_dir)):
            if not fname.endswith('.jpg'):
                continue
            mask_path = os.path.join(
                masks_dir, fname.replace('.jpg', '.png'))
            if os.path.exists(mask_path):
                self.samples.append(fname)

        print(f"Dataset: {len(self.samples)} samples found")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]

        image = Image.open(
            os.path.join(self.frames_dir, fname)
        ).convert('RGB')

        mask_raw = np.array(Image.open(
            os.path.join(self.masks_dir,
                fname.replace('.jpg', '.png'))
        ))

        mask_clipped   = np.clip(mask_raw, 0, 399)
        mask_converted = lookup[mask_clipped].astype(np.int64)

        mask_pil = Image.fromarray(mask_converted.astype(np.uint8))
        mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

        return self.transform(image), mask_tensor

# ── Evaluate ──────────────────────────────────────────────────────────
def evaluate(model, device):
    val_dataset = VISORDataset(
        frames_dir = "data/visor_hos/val_frames/",
        masks_dir  = "data/visor_hos/val_masks/"
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=0)

    model.eval()
    intersection = torch.zeros(150)
    union        = torch.zeros(150)

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(pixel_values=images).logits
            logits = torch.nn.functional.interpolate(
                logits, size=masks.shape[-2:],
                mode='bilinear', align_corners=False
            )
            predicted = torch.argmax(logits, dim=1)

            for cls in set(VISOR_TO_ADE20K.values()):
                pred_cls = (predicted == cls).cpu()
                gt_cls   = (masks == cls).cpu()
                intersection[cls] += (pred_cls & gt_cls).sum().item()
                union[cls]        += (pred_cls | gt_cls).sum().item()

    ious = []
    for cls in set(VISOR_TO_ADE20K.values()):
        if union[cls] > 0:
            iou = intersection[cls] / union[cls]
            ious.append(float(iou))
            print(f"  class {cls}: IoU = {iou:.3f}")

    miou = float(np.mean(ious)) if ious else 0.0
    print(f"VISOR mIoU: {miou:.3f}")
    return miou

# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}")

# ── Model ─────────────────────────────────────────────────────────────
print("Loading SegFormer-B2...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model = model.to(device)

# ── Optimizer ─────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW([
    {"params": model.segformer.parameters(),   "lr": 0.000006},
    {"params": model.decode_head.parameters(), "lr": 0.00006}
])

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

# ── DataLoader ────────────────────────────────────────────────────────
train_dataset = VISORDataset(
    frames_dir = "data/visor_hos/train_frames/",
    masks_dir  = "data/visor_hos/train_masks/"
)
train_loader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=0)

# ── Training ──────────────────────────────────────────────────────────
NUM_EPOCHS = 10
os.makedirs("models", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss  = 0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(pixel_values=images).logits
        logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:],
            mode='bilinear', align_corners=False
        )

        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                  f"Batch {batch_idx}/{len(train_loader)} "
                  f"Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / num_batches
    print(f"\nEpoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(),
        f"models/segformer_b2_visor_epoch{epoch+1}.pt")
    print(f"Saved checkpoint epoch {epoch+1}")

    miou = evaluate(model, device)
    print(f"Epoch {epoch+1} VISOR mIoU: {miou:.3f}\n")

print("Fine-tuning complete!")