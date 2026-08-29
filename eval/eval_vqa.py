"""
Quantitative eval on a held-out VQA-style set. This is what turns "it seems
to work" into a number you can put in the writeup and track across
checkpoints/ablations (linear vs MLP projector, frozen vs LoRA LLM, etc).

Metric: exact-match / soft accuracy against reference answers is standard
for VQA-style short-answer eval. For free-form instruction responses,
consider also reporting a small hand-graded rubric (e.g. 50 examples,
graded correct/partial/wrong by hand) since automatic metrics undersell
free-form generation quality.

TODO:
1. Build/hold out eval set (~200-500 examples) from your Stage 2 data prep,
   never seen during training.
2. Implement generate + compare loop below.
3. Break results down by question type (counting, color, spatial relation,
   OCR, etc.) if your eval set has that metadata — a single overall accuracy
   number is much less informative than a breakdown, and the breakdown is
   what feeds your failure-mode analysis.
"""

import json
import argparse
import torch

from models.vlm import MiniLLaVA


def normalize_answer(text: str) -> str:
    return text.strip().lower().rstrip(".")


def evaluate(model: MiniLLaVA, eval_records, device):
    correct = 0
    results = []

    model.eval()
    for rec in eval_records:
        # TODO: build input_ids/pixel_values the same way
        # Stage2InstructDataset does, then call model.generate(...)
        # and compare normalize_answer(generated) == normalize_answer(rec["answer"])
        pass

    # TODO: aggregate + optionally break down by rec.get("question_type")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_records = json.load(f)

    # TODO: load model + checkpoint, then evaluate(...)
