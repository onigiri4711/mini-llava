"""
Download and format a Stage 1 (alignment) dataset.

Suggested source: COCO Captions, via the HuggingFace `datasets` library
(e.g. "HuggingFaceM4/COCO" or similar), filtered down to a manageable
subset (20-50k image-caption pairs is plenty to see real alignment happen).

TODO:
1. Pick the exact dataset/subset size given your compute budget.
2. Download images locally (or stream) and write out a JSON list of
   {"image_path": ..., "caption": ...} matching Stage1CaptionDataset's
   expected format.
3. Hold out a small val split (e.g. 500 pairs) for tracking alignment loss.

Example skeleton:

    from datasets import load_dataset
    import json, os

    ds = load_dataset("HuggingFaceM4/COCO", split="train[:30000]")
    os.makedirs("data/stage1_images", exist_ok=True)
    records = []
    for i, ex in enumerate(ds):
        img_path = f"data/stage1_images/{i}.jpg"
        ex["image"].save(img_path)
        records.append({"image_path": img_path, "caption": ex["sentences"]["raw"][0]})

    with open("data/stage1_train.json", "w") as f:
        json.dump(records, f)
"""

raise NotImplementedError("Fill in after choosing your exact dataset source.")
