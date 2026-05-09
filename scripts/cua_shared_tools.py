"""CUA-mode inference helper (with Qwen tool calls) — separate from the
QA-mode helper in cua_shared.py. Used by the live Kernel demo where the
model needs to actually click."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from PIL import Image

from cua_shared import LoadedModel

QWEN_TOOLS = [
    {"type": "function", "function": {
        "name": "click", "description": "Click at normalized 0-999 coordinates.",
        "parameters": {"type": "object",
                       "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                       "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "type", "description": "Type text into the focused field.",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string"}},
                       "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "key", "description": "Press a keyboard shortcut.",
        "parameters": {"type": "object",
                       "properties": {"keys": {"type": "string"}},
                       "required": ["keys"]}}},
    {"type": "function", "function": {
        "name": "terminate", "description": "End the task.",
        "parameters": {"type": "object",
                       "properties": {"message": {"type": "string"}}}}},
]

SYS = ("You are a computer-use agent. Look at the screenshot and emit one tool call. "
       "Coordinates are normalized 0-999.")

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_NAME_RE = re.compile(r'"name"\s*:\s*"([a-z_]+)"')


def parse_tool_call(text: str) -> dict[str, Any] | None:
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        # salvage: regex out name + numbers
        nm = _NAME_RE.search(m.group(1))
        if not nm:
            return None
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        return {"name": nm.group(1), "arguments": {"x": nums[0], "y": nums[1]} if len(nums) >= 2 else {}}
    args = obj.get("arguments") or {}
    if isinstance(args, dict):
        # Coerce list-typed x to (x,y)
        if isinstance(args.get("x"), list) and len(args["x"]) == 2:
            args["y"] = float(args["x"][1])
            args["x"] = float(args["x"][0])
        for k in ("x", "y"):
            if k in args:
                try:
                    args[k] = float(args[k])
                except (TypeError, ValueError):
                    pass
    return {"name": obj.get("name"), "arguments": args}


@torch.no_grad()
def infer_action_with_tools(lm: LoadedModel, image: Image.Image, task: str,
                            max_new_tokens: int = 128) -> dict[str, Any]:
    user_text = f"User task: {task}\nLook at the screenshot and emit ONE tool call."
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt = lm.processor.apply_chat_template(
        messages, tools=QWEN_TOOLS, add_generation_prompt=True, tokenize=False
    )
    inputs = lm.processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(lm.model.device)
    gen = lm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            temperature=None, top_p=None)
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    text = lm.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"raw": text, "parsed": parse_tool_call(text)}
