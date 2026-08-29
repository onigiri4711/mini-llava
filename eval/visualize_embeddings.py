"""
The single most convincing plot for your writeup: a t-SNE/UMAP projection of
projected image-patch embeddings vs. text token embeddings, before vs. after
Stage 1 training.

Before training: the projector is randomly initialized, so projected visual
features land nowhere near the LLM's text embedding manifold — two disjoint
clouds.

After Stage 1: the visual features should have moved into (or near) the
region of the embedding space the LLM's text tokens occupy — literally
visualizing "alignment."

Usage sketch (fill in once you have both checkpoints):

    python eval/visualize_embeddings.py \
        --projector-before checkpoints/stage1_epoch0_projector.pt \
        --projector-after checkpoints/stage1_epoch{N}_projector.pt \
        --n-samples 500
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from models.vlm import MiniLLaVA


def collect_embeddings(model: MiniLLaVA, images, sample_texts, device):
    """Returns (visual_embeds, text_embeds) as numpy arrays, pooled per-sample
    (mean over patches / mean over tokens) for a clean 2D scatter."""
    model.eval()
    with torch.no_grad():
        pixel_values = model.vision_encoder.preprocess(images).to(device)
        vis_feats = model.vision_encoder(pixel_values)
        vis_tokens = model.projector(vis_feats)  # (B, N_patches, D)
        vis_pooled = vis_tokens.mean(dim=1).cpu().numpy()

        text_ids = model.tokenizer(sample_texts, padding=True, return_tensors="pt")["input_ids"].to(device)
        text_embeds = model.llm.get_input_embeddings()(text_ids)
        text_pooled = text_embeds.mean(dim=1).cpu().numpy()

    return vis_pooled, text_pooled


def plot_comparison(vis_before, text_before, vis_after, text_after, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vis, text, title in [
        (axes[0], vis_before, text_before, "Before Stage 1 training"),
        (axes[1], vis_after, text_after, "After Stage 1 training"),
    ]:
        combined = np.concatenate([vis, text], axis=0)
        reduced = UMAP(n_components=2, random_state=0).fit_transform(combined)
        n_vis = len(vis)
        ax.scatter(reduced[:n_vis, 0], reduced[:n_vis, 1], label="image embeds", alpha=0.6, s=10)
        ax.scatter(reduced[n_vis:, 0], reduced[n_vis:, 1], label="text embeds", alpha=0.6, s=10)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    # TODO: wire up argparse + checkpoint loading once you have real
    # before/after projector weights to compare. The plotting logic above
    # is ready to go as soon as you feed it two sets of embeddings.
    pass
