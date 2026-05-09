"""Shared model loading + inference helpers for Northstar-CUA-Fast +
optional DPO-QA LoRA adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Override via env / CLI when running on a different machine.
DEFAULT_BASE_MODEL = "Tzafon/Northstar-CUA-Fast"
DEFAULT_ADAPTER = str(Path(__file__).resolve().parent.parent / "adapter")


@dataclass
class LoadedModel:
    model: Any
    processor: Any


def load_model(base_model: str = DEFAULT_BASE_MODEL, adapter_path: str | None = None) -> LoadedModel:
    """Load Northstar (from HF or local path) + optional PEFT LoRA adapter."""
    print(f"[load] base={base_model} adapter={adapter_path}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model)
    if torch.cuda.is_available():
        print(f"[load] VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    return LoadedModel(model=model, processor=processor)


@torch.no_grad()
def infer_qa(lm: LoadedModel, image: Image.Image, system_prompt: str, user_text: str,
             max_new_tokens: int = 256) -> str:
    """QA-mode inference (no tools): system prompt + image + user text -> prose response.

    This is the mode CyberSecEval3 tests."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]
    prompt = lm.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = lm.processor(
        text=[prompt], images=[image], return_tensors="pt", padding=True
    ).to(lm.model.device)
    gen = lm.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None,
    )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    return lm.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
