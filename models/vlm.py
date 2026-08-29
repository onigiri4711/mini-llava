"""
Full VLM: vision encoder (frozen) + projector (trainable) + LLM (frozen or
LoRA). This module handles the part that's easy to get subtly wrong: splicing
projected visual tokens into the text token sequence at the right position,
and masking the loss so it's only computed on the response tokens.

TODO (after reading the paper/code): confirm the exact convention for where
the image placeholder sits in the prompt template (LLaVA uses a fixed
"<image>" token position, usually right after a system prompt, before the
instruction). Your prompt template in data/datasets.py must match whatever
you implement here.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from models.vision_encoder import FrozenVisionEncoder
from models.projector import build_projector


class MiniLLaVA(nn.Module):
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        llm_model_name: str = "Qwen/Qwen2-0.5B",
        projector_kind: str = "mlp",
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        image_token: str = "<image>",
    ):
        super().__init__()

        self.vision_encoder = FrozenVisionEncoder(vision_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        self.image_token = image_token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Freeze LLM by default; Stage 1 trains projector only.
        for p in self.llm.parameters():
            p.requires_grad = False

        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],  # adjust per LLM arch
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

        llm_dim = self.llm.config.hidden_size
        self.projector = build_projector(
            projector_kind, self.vision_encoder.hidden_size, llm_dim
        )

    def embed_inputs(self, input_ids: torch.Tensor, pixel_values: torch.Tensor):
        """
        Replace every occurrence of the image placeholder token with the
        projected visual patch embeddings, in order. Returns inputs_embeds
        ready to feed to the LLM directly (bypassing its own embedding
        lookup for the image positions).
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, T, D)

        with torch.no_grad():
            vis_feats = self.vision_encoder(pixel_values)  # (B, N_patches, D_v)
        vis_tokens = self.projector(vis_feats)  # (B, N_patches, D_llm)

        # TODO: this assumes exactly one contiguous <image> placeholder run
        # per sample, sized to N_patches, inserted upstream in the dataset
        # collator. Verify that assumption holds once you write the
        # tokenization/collation logic — it's the most common source of
        # silent bugs in LLaVA reimplementations.
        image_mask = (input_ids == self.image_token_id)
        inputs_embeds = text_embeds.clone()
        for b in range(input_ids.shape[0]):
            idx = image_mask[b].nonzero(as_tuple=True)[0]
            if len(idx) != vis_tokens.shape[1]:
                raise ValueError(
                    f"Sample {b}: {len(idx)} image placeholder tokens but "
                    f"{vis_tokens.shape[1]} visual patches. Check your "
                    f"prompt template / patch count."
                )
            inputs_embeds[b, idx] = vis_tokens[b]

        return inputs_embeds

    def forward(self, input_ids, pixel_values, attention_mask, labels=None):
        inputs_embeds = self.embed_inputs(input_ids, pixel_values)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, pixel_values, attention_mask, **gen_kwargs):
        inputs_embeds = self.embed_inputs(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
