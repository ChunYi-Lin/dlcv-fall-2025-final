from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class HFLLMConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    device_map: Optional[Any] = None
    dtype: str = "auto"
    quantization: str = "none"


def _parse_torch_dtype(dtype: str):
    dtype_norm = (dtype or "auto").strip().lower()
    if dtype_norm == "auto":
        return "auto"
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_norm not in mapping:
        raise ValueError(
            f"Unsupported dtype: {dtype!r}. Use one of: auto, fp16, bf16, fp32."
        )
    return mapping[dtype_norm]


def _default_device_map():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _coerce_device_map(device_map: Optional[str]):
    if device_map is None:
        return _default_device_map()
    device_map_norm = str(device_map).strip().lower()
    if device_map_norm in {"", "none", "null"}:
        return None
    return device_map


def _fallback_chat_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    lines = []
    for message in messages:
        role = (message.get("role") or "").strip().lower()
        content = message.get("content") or ""
        if role == "system":
            prefix = "System"
        elif role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = role.capitalize() or "Message"
        lines.append(f"{prefix}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


class HFChatLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _device(self):
        device = getattr(self.model, "device", None)
        if device is not None:
            return device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def build_prompt(self, messages: Sequence[Mapping[str, str]]) -> str:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return _fallback_chat_prompt(messages)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        prompt = self.build_prompt(messages)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self._device())

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_hf_chat_llm(config: HFLLMConfig) -> HFChatLLM:
    tokenizer_path = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = (config.quantization or "none").strip().lower()
    device_map = _coerce_device_map(config.device_map)

    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "device_map": device_map,
    }

    if quantization == "4bit":
        compute_dtype = _parse_torch_dtype(config.dtype)
        if compute_dtype == "auto":
            compute_dtype = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "none":
        model_kwargs["torch_dtype"] = _parse_torch_dtype(config.dtype)
    else:
        raise ValueError(
            f"Unsupported quantization: {config.quantization!r}. Use one of: none, 4bit, 8bit."
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return HFChatLLM(model=model, tokenizer=tokenizer)
