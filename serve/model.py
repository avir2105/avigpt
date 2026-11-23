"""4-bit model load + generate/stream helpers for Colab T4."""

from __future__ import annotations

import os
import time
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

from serve.prompts import build_prompt, clean_answer, extract_assistant_reply

MODEL_NAME = os.getenv("MODEL_NAME", "avi2135/avigpt2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
USE_4BIT = os.getenv("USE_4BIT", "1") == "1"
ALLOW_CPU = os.getenv("ALLOW_CPU", "0") == "1"

model = None
tokenizer = None
_load_meta: dict = {}


def vram_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def load_model_and_tokenizer() -> None:
    global model, tokenizer, _load_meta
    if model is not None and tokenizer is not None:
        return

    if not torch.cuda.is_available() and not ALLOW_CPU:
        raise RuntimeError(
            "CUDA GPU required for quantized 7B on Colab T4. "
            "Set ALLOW_CPU=1 only for tiny debug models."
        )

    t0 = time.time()
    quantization_config = None
    if USE_4BIT and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _load_meta = {
        "model_name": MODEL_NAME,
        "use_4bit": bool(quantization_config),
        "load_seconds": round(time.time() - t0, 2),
        "vram": vram_stats(),
    }


def get_load_meta() -> dict:
    return dict(_load_meta)


def _gen_kwargs(max_new_tokens: int | None, temperature: float) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens or MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.9
    else:
        kwargs["do_sample"] = False
    return kwargs


@torch.inference_mode()
def generate_answer(
    question: str,
    *,
    history: list[dict] | None = None,
    context: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict:
    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded")

    prompt = build_prompt(question, history=history, context=context)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    outputs = model.generate(**inputs, **_gen_kwargs(max_new_tokens, temperature))
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    answer = extract_assistant_reply(decoded, prompt)
    answer = clean_answer(answer, question)
    latency = round(time.time() - t0, 3)
    new_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])
    return {
        "answer": answer,
        "latency_s": latency,
        "new_tokens": new_tokens,
        "vram": vram_stats(),
    }


def stream_answer(
    question: str,
    *,
    history: list[dict] | None = None,
    context: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
) -> Iterator[str]:
    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded")

    prompt = build_prompt(question, history=history, context=context)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inputs)
    kwargs.update(_gen_kwargs(max_new_tokens, temperature))
    kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    for token in streamer:
        yield token
    thread.join()
