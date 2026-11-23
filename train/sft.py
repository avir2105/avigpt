"""
QLoRA SFT script for Colab T4.

Example (Colab):
  !python -m train.sft --config train/configs/sft.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_row(question: str, answer: str, system: str) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{question} [/INST] {answer} </s>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "train" / "configs" / "sft.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    # Heavy imports after argparse so `--help` works without GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    data_path = ROOT / cfg["data_path"]
    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    system = cfg.get(
        "system",
        "You are AVIGPT, a factual assistant about Avinashsingh Rajput. Do not invent facts.",
    )
    texts = [format_row(r["question"], r["answer"], system) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora)

    out_dir = ROOT / cfg.get("output_dir", "outputs/sft-qlora")
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.get("epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        learning_rate=cfg.get("lr", 2e-4),
        fp16=True,
        logging_steps=10,
        save_steps=200,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 512),
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    print(f"Saved adapter to {out_dir / 'adapter'}")
    if cfg.get("push_to_hub"):
        trainer.model.push_to_hub(cfg["push_to_hub"])
        print(f"Pushed to {cfg['push_to_hub']}")


if __name__ == "__main__":
    main()
