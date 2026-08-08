import torch
from transformers import SegformerForSemanticSegmentation
import os

device = torch.device("cpu")
MODEL_PATH = os.path.expanduser("~/egoseg_rt/models/segformer_b0_epoch10.pt")
ONNX_PATH  = os.path.expanduser("~/egoseg_rt/models/segformer_b0.onnx")

print("Loading B0 model...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)

print("Exporting to ONNX...")
with torch.no_grad():
    torch.onnx.export(
        model,
        {"pixel_values": dummy_input},
        ONNX_PATH,
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamo=False   # ← forces old exporter
    )

print(f"Saved to {ONNX_PATH}")
size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"Model size: {size_mb:.1f} MB")