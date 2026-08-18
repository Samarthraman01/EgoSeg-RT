"""
Rasterize VISOR sparse JSON polygon annotations into per-pixel class-index
PNG masks for the validation frames in data/visor_hos/val_frames.

Source: data/visor_hos/.../annotations/val/<video>.json, one per video
(downloaded from data.bris.ac.uk -- VISOR is not on HuggingFace, no auth needed)
"""
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw


ANNOTATIONS_DIR = "data/visor_hos/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations/annotations/train"
FRAMES_DIR = "data/visor_hos/train_frames"
OUTPUT_DIR      = "data/visor_hos/train_masks"

# Hand/glove classes (300-303) sit on top of everything else in VISOR;
# draw them last so object polygons underneath don't occlude them.
FOREGROUND_CLASS_IDS = {300, 301, 302, 303}

# Pixel value 0 is reserved for "unannotated background". VISOR's own
# class_id 0 is a real class (e.g. "tap"), so every stored value is
# class_id + 1 -- subtract 1 from any non-zero mask pixel to recover
# the original VISOR class_id.
BACKGROUND_VALUE = 0


def rasterize(image_size, annotations):
    mask = np.full((image_size[1], image_size[0]), BACKGROUND_VALUE, dtype=np.uint16)
    background = [a for a in annotations if a["class_id"] not in FOREGROUND_CLASS_IDS]
    foreground = [a for a in annotations if a["class_id"] in FOREGROUND_CLASS_IDS]

    for ann in background + foreground:
        layer = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(layer)
        for polygon in ann["segments"]:
            points = [(p[0], p[1]) for p in polygon]
            if len(points) >= 3:
                draw.polygon(points, fill=1)
        mask[np.array(layer) == 1] = ann["class_id"] + 1
    return mask


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_paths = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json")))

    written, skipped = 0, 0
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)

        for entry in data["video_annotations"]:
            frame_name = entry["image"]["name"]
            frame_path = os.path.join(FRAMES_DIR, frame_name)
            if not os.path.exists(frame_path):
                skipped += 1
                continue

            with Image.open(frame_path) as im:
                size = im.size

            mask = rasterize(size, entry["annotations"])
            out_name = os.path.splitext(frame_name)[0] + ".png"
            Image.fromarray(mask).save(os.path.join(OUTPUT_DIR, out_name))
            written += 1

    print(f"Wrote {written} masks to {OUTPUT_DIR} from {len(json_paths)} videos ({skipped} frames skipped, no matching jpg)")


if __name__ == "__main__":
    main()
