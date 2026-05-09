"""DPO fine-tune Tzafon/Northstar-CUA-Fast on prose preference pairs to
defend against visual prompt injection.

Each pair: same image + same task, two candidate prose responses
(chosen=safe description, rejected=secret leak). KL-regularized DPO
pushes the model to prefer chosen over rejected.

Usage:
    python scripts/train_dpo.py \\
        --train data/dpo_pairs_qa.jsonl \\
        --root data/qa_train \\
        --out adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import DPOConfig, DPOTrainer


def build_qa_messages(system_text: str, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]


def assistant_message(text: str) -> list[dict]:
    return [{"role": "assistant", "content": [{"type": "text", "text": text}]}]


def load_pairs(jsonl_path: Path, root: Path) -> Dataset:
    rows = []
    for ln in open(jsonl_path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        img_path = (root / r["image_path"]).resolve()
        rows.append({
            "images": [Image.open(img_path).convert("RGB")],
            "prompt": build_qa_messages(r["_meta"]["system_prompt"], r["prompt"]),
            "chosen": assistant_message(r["chosen"]),
            "rejected": assistant_message(r["rejected"]),
        })
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to dpo_pairs_qa.jsonl")
    ap.add_argument("--root", required=True, help="Root for resolving image_path")
    ap.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    ap.add_argument("--out", default="adapter", help="Where to save the trained LoRA adapter")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1, help="KL strength")
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--logging-steps", type=int, default=10)
    args = ap.parse_args()

    print(f"[train] loading model: {args.base_model}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.base_model)

    print(f"[train] loading pairs from {args.train}", flush=True)
    ds = load_pairs(Path(args.train), Path(args.root))
    print(f"[train] dataset size: {len(ds)}", flush=True)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        max_length=None,  # critical for VLM — don't truncate image tokens
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=lora_cfg,
    )

    print(f"[train] starting DPO. output -> {args.out}", flush=True)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[train] done. adapter saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
