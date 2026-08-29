"""
Training entrypoint. Run with a stage-specific config:

    python train.py --config configs/stage1_alignment.yaml
    python train.py --config configs/stage2_instruct.yaml

Stage 1: only projector params get an optimizer (vision encoder + LLM frozen).
Stage 2: projector + LoRA params get an optimizer (vision encoder still frozen,
         base LLM weights still frozen, only LoRA adapters train).

TODO once you've read the paper's training section:
- confirm learning rates per stage (LLaVA uses a notably higher LR for the
  projector in Stage 1 than the LoRA LR in Stage 2 — worth replicating that
  ratio even at small scale)
- decide on warmup schedule / cosine decay
- decide gradient accumulation steps based on your actual GPU memory
"""

import argparse
import json
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vlm import MiniLLaVA
from data.datasets import Stage1CaptionDataset, Stage2InstructDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_trainable_params(model: MiniLLaVA, stage: str):
    if stage == "stage1":
        return list(model.projector.parameters())
    elif stage == "stage2":
        params = list(model.projector.parameters())
        params += [p for p in model.llm.parameters() if p.requires_grad]  # LoRA params
        return params
    raise ValueError(stage)


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniLLaVA(
        vision_model_name=cfg["vision_model_name"],
        llm_model_name=cfg["llm_model_name"],
        projector_kind=cfg["projector_kind"],
        use_lora=(cfg["stage"] == "stage2"),
    ).to(device)

    with open(cfg["train_data"]) as f:
        records = json.load(f)

    if cfg["stage"] == "stage1":
        dataset = Stage1CaptionDataset(records, model.tokenizer, model.vision_encoder.image_processor)
    else:
        dataset = Stage2InstructDataset(records, model.tokenizer, model.vision_encoder.image_processor)

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)

    trainable = get_trainable_params(model, cfg["stage"])
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"])

    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    model.train()
    step = 0
    for epoch in range(cfg["epochs"]):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, pixel_values, attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=loss.item())

            # TODO: log to wandb, save checkpoints on interval,
            # run a val-loss pass periodically

        # TODO: save checkpoint at end of each epoch
        # torch.save(model.projector.state_dict(), f"checkpoints/{cfg['stage']}_epoch{epoch}_projector.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
