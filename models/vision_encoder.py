"""
Frozen vision encoder wrapper.

Loads a pretrained vision transformer (CLIP or SigLIP) via transformers and
exposes patch-level embeddings (not just the pooled/CLS embedding) — LLaVA
uses the grid of patch tokens, not a single pooled vector, so the LLM gets
spatial information rather than one summary vector.

TODO (after reading the paper): confirm whether to use the second-to-last
hidden layer instead of the final layer. LLaVA-1.5 uses the penultimate layer
output on the theory that the very last layer is over-specialized for the
encoder's own pretraining objective (contrastive alignment) and loses some
fine-grained visual detail useful for a downstream LLM.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoImageProcessor


class FrozenVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        select_layer: int = -2,
        select_feature: str = "patch",  # "patch" or "cls_patch"
    ):
        super().__init__()
        self.model_name = model_name
        self.select_layer = select_layer
        self.select_feature = select_feature

        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Freeze everything — this encoder is never updated during training.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, H, W), already preprocessed.
        returns: (B, num_patches, hidden_size)
        """
        outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            # drop the CLS token, keep only patch tokens
            return hidden_states[:, 1:, :]
        elif self.select_feature == "cls_patch":
            return hidden_states
        else:
            raise ValueError(f"Unknown select_feature: {self.select_feature}")

    def preprocess(self, images):
        """images: list of PIL.Image -> pixel_values tensor"""
        return self.image_processor(images=images, return_tensors="pt")["pixel_values"]
