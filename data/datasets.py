"""
Dataset classes for the two training stages.

Stage1CaptionDataset: (image, caption) pairs — e.g. a COCO Captions subset.
Stage2InstructDataset: (image, instruction, response) triples — e.g. a
    filtered subset of LLaVA-Instruct-150k or a small VQA slice.

Both produce a fixed prompt template with a single contiguous run of
<image> placeholder tokens, sized to match num_patches from the vision
encoder (e.g. 49 patches for CLIP ViT-B/32 at 224x224 -> 7x7 grid).

TODO: confirm num_patches for your chosen encoder/resolution combo and set
IMAGE_TOKEN_COUNT accordingly. Get this wrong and embed_inputs() in
models/vlm.py will raise a shape-mismatch error (which is a good thing —
better to fail loudly there than silently misalign tokens).
"""

from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_COUNT = 49  # TODO: set to your encoder's patch grid size

CAPTION_PROMPT_TEMPLATE = (
    "{image_tokens}\nDescribe this image."
)
INSTRUCT_PROMPT_TEMPLATE = (
    "{image_tokens}\n{instruction}"
)


class Stage1CaptionDataset(Dataset):
    """Expects a list of dicts: {"image_path": str, "caption": str}"""

    def __init__(self, samples: List[Dict], tokenizer, image_processor, max_len: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.image_tokens = IMAGE_TOKEN * IMAGE_TOKEN_COUNT

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        prompt = CAPTION_PROMPT_TEMPLATE.format(image_tokens=self.image_tokens)
        full_text = prompt + " " + sample["caption"] + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Mask loss on everything except the caption tokens (prompt + image
        # placeholders should not contribute to the loss).
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


class Stage2InstructDataset(Dataset):
    """Expects a list of dicts: {"image_path": str, "instruction": str, "response": str}"""

    def __init__(self, samples: List[Dict], tokenizer, image_processor, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.image_tokens = IMAGE_TOKEN * IMAGE_TOKEN_COUNT

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        prompt = INSTRUCT_PROMPT_TEMPLATE.format(
            image_tokens=self.image_tokens, instruction=sample["instruction"]
        )
        full_text = prompt + " " + sample["response"] + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
