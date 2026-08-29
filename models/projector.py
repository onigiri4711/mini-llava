"""
The projector is the only module trained in Stage 1, and it's the whole
point of the architecture: it maps vision-encoder patch embeddings into the
LLM's embedding space so the LLM can treat visual tokens like word tokens.

TODO (after reading the paper): LLaVA v1 used a single linear layer.
LLaVA-1.5 switched to a 2-layer MLP with GELU and found it meaningfully
improved downstream benchmarks. Worth implementing both and comparing —
that comparison alone is a good ablation for your writeup.
"""

import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, x):
        return self.proj(x)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, hidden_mult: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim * hidden_mult),
            nn.GELU(),
            nn.Linear(llm_dim * hidden_mult, llm_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_projector(kind: str, vision_dim: int, llm_dim: int) -> nn.Module:
    if kind == "linear":
        return LinearProjector(vision_dim, llm_dim)
    elif kind == "mlp":
        return MLPProjector(vision_dim, llm_dim)
    raise ValueError(f"Unknown projector kind: {kind}")
