"""
Interpretability: for a given (image, question, generated answer), visualize
how much attention the LLM places on each image patch token while generating
the answer. Overlay as a heatmap on the original image.

This is what turns a "look, it works" demo into "look, I know why it works
and where it breaks" — the more valuable claim for a portfolio.

Approach:
1. Run a forward pass with output_attentions=True.
2. For a chosen generated token (e.g. the key noun in the answer), average
   attention weights across heads/layers (or pick a specific layer — worth
   comparing early vs. late layers) from that token back to the image patch
   token positions.
3. Reshape the per-patch attention scores back into the patch grid (e.g. 7x7
   for CLIP ViT-B/32 @ 224px) and upsample to overlay on the original image.

TODO once Stage 2 model is trained:
- decide which layer(s) to visualize (early layers often show low-level
  edge/color attention, late layers more semantic/object-level attention —
  showing both is a good qualitative result)
- build a small gallery of examples: correct answers (attention on the
  right object) vs. wrong answers (attention on the wrong region or
  diffuse/uninformative) — this pairing is the actual "failure mode"
  analysis for your writeup
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def get_patch_grid_size(num_patches: int) -> int:
    side = int(num_patches ** 0.5)
    assert side * side == num_patches, "non-square patch grid, adjust logic"
    return side


def overlay_attention(image: Image.Image, attn_scores: np.ndarray, out_path: str):
    """
    attn_scores: 1D array of length num_patches, attention weight per patch.
    """
    grid_size = get_patch_grid_size(len(attn_scores))
    attn_map = attn_scores.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(
        attn_map,
        cmap="jet",
        alpha=0.45,
        extent=(0, image.width, image.height, 0),
        interpolation="bilinear",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")


# TODO: function to actually extract attn_scores from the model's
# output_attentions given a specific answer token index and image token
# span — this depends on the exact LLM architecture's attention output
# shape, confirm against whichever model (Qwen2/TinyLlama/Phi-2) you land on.
