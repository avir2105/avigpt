# Training (QLoRA on Colab T4)

Reproducible SFT path for Week 3 / future refreshes.

## Run (GPU runtime, separate from serve)

```bash
%cd /content/avigpt   # or your clone path
!pip install -q -r requirements.txt
!huggingface-cli login   # if pushing to Hub
!python -m train.sft --config train/configs/sft.yaml
```

Config: `train/configs/sft.yaml`

- 4-bit base + LoRA
- `data/train.jsonl`
- Writes adapter to `outputs/sft-qlora/adapter`
- Optional: set `push_to_hub: avi2135/avigpt2-qlora` after login

## DPO

Preference seed: `data/prefs.jsonl` (50 pairs).

**Decision (Week 3):** skip 7B DPO on free T4 for now — document in `reports/week3_safety.md`. Prefer 1B–3B proxy or a longer session later.
