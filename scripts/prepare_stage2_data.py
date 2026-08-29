"""
Download and format a Stage 2 (instruction tuning) dataset.

Suggested source: a filtered subset of LLaVA-Instruct-150k (liuhaotian/LLaVA-
Instruct-150K on HF Hub) or a VQAv2 subset reformatted as instruction/response
pairs. A few thousand examples is enough to see instruction-following
behavior emerge on top of the Stage 1 alignment.

TODO:
1. Pick dataset + subset size.
2. Reformat multi-turn conversations (if using LLaVA-Instruct) into single
   instruction/response pairs for simplicity, or extend
   Stage2InstructDataset to handle multi-turn if you want that complexity.
3. Write out JSON list of {"image_path", "instruction", "response"} matching
   Stage2InstructDataset's expected format.
4. Hold out a val split for eval/eval_vqa.py.
"""

raise NotImplementedError("Fill in after choosing your exact dataset source.")
