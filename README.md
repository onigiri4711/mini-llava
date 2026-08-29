# mini-llava

A small-scale reproduction of the LLaVA architecture: frozen vision encoder →
trainable projection layer → language model (frozen or LoRA fine-tuned),
trained in two stages (feature alignment, then instruction tuning).

Goal is not SOTA performance — it's to demonstrate real understanding of how
modern VLMs are built: cross-modal alignment, projection layers, staged
training, and where/why these models fail.

## Architecture

```
image ──> [Vision Encoder (frozen)] ──> patch embeddings
                                             │
                                    [Projector (trainable)]
                                             │
                                    projected visual tokens
                                             │
text prompt ──> [tokenizer] ──> text embeddings
                                             │
                    [concat: visual tokens + text tokens]
                                             │
                              [LLM (frozen / LoRA)]
                                             │
                                       generated text
```

## Repo layout

```
configs/            training configs for each stage (yaml)
data/                dataset loading + preprocessing
models/              vision encoder, projector, full VLM wrapper
scripts/             one-off data prep scripts
eval/                quantitative eval + interpretability (attention viz)
notebooks/           demo notebook for the writeup
train.py             training entrypoint, stage 1 or stage 2
```

## Stages

**Stage 1 — Feature alignment.** Vision encoder and LLM both frozen. Only the
projector is trained, on image-caption pairs. Objective: teach the projector
to place visual features somewhere the LLM's embedding space can interpret.

**Stage 2 — Instruction tuning.** Projector continues training, LLM gets LoRA
adapters. Trained on image + instruction + response triples. Objective: teach
the model to follow instructions about image content, not just caption it.

## Status

- [ ] Stage 1 data pipeline
- [ ] Stage 1 training loop + loss curves
- [ ] Stage 2 data pipeline
- [ ] Stage 2 training loop (LoRA)
- [ ] Eval set + metrics
- [ ] Attention/patch visualization
- [ ] Failure mode writeup

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes to self (fill in as you read the paper)

- Which vision encoder, and why (CLIP ViT vs SigLIP — check patch count,
  resolution, whether it was trained with a [CLS] token or not)
- Projector: linear vs 2-layer MLP — LLaVA-1.5 found MLP beats linear, worth
  confirming at this scale
- How image tokens get spliced into the text sequence (a fixed placeholder
  token gets replaced with the projected visual embeddings — check exactly
  where this happens in the reference implementation)
- Loss masking: only compute loss on response tokens, not the prompt or image
  tokens
