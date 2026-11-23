# AVIGPT

Personal Q&A LLM about **Avinashsingh Rajput** — QLoRA fine-tune, **4-bit** serve, **RAG**, **safety**, and **versioned evals** — built and measured on a **free Google Colab T4 (~15 GB)** and demoed over **ngrok**.

**Demo video:** _add link after recording — see [docs/DEMO.md](docs/DEMO.md)_  
**Model:** [avi2135/avigpt2](https://huggingface.co/avi2135/avigpt2) · **Card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)  
**Plan:** [docs/ANTHROPIC_ROADMAP.md](docs/ANTHROPIC_ROADMAP.md)

---

## Problem → method

| Goal | Approach |
|------|----------|
| Answer personal bio questions | QLoRA SFT on Llama-2-7B-chat |
| Stay on free T4 | 4-bit NF4 inference + modest `max_new_tokens` |
| Reduce hallucination | TF-IDF RAG over `data/corpus/` + `/verify` tool |
| Be honest about risk | Constitution + hard refuse filter + red-team report |
| Prove progress | Versioned runs in `reports/runs/` (before → after) |

Not a wrapper chat app — the portfolio artifact is **measure → ground → refuse → demo under constraints**.

---

## Results (same 80 eval questions, live T4 + ngrok)

| Mode | Exact | Token F1 | Contains gold | Run |
|------|-------|----------|---------------|-----|
| SFT only | 0.0375 | 0.4066 | 0.1875 | v002 |
| **SFT + RAG** | **0.1000** | **0.5260** | **0.6250** | v004 |

Red-team (25 prompts):

| Era | Pass rate | Run |
|-----|-----------|-----|
| Old Colab build (rescored) | 0.16 | v001 |
| Redeployed serve + filter | **0.96** | v003 |

Full tables: [reports/LEADERBOARD.md](reports/LEADERBOARD.md) · [week2](reports/week2_rag.md) · [week3](reports/week3_safety.md)

---

## Failures (kept on purpose)

- **Pre-redeploy:** model invented bank / Aadhaar-style details when asked (see v001).
- **v003:** still complied once with “invent three fake companies” (`rt02`) — filter extended afterward; re-measure after next Colab pull.
- **Exact match** stays low without RAG; paraphrases ≠ gold strings.

---

## Quick demo (Colab T4 + local UI)

1. Open [`notebooks/serve_ngrok.ipynb`](notebooks/serve_ngrok.ipynb) → Runtime **GPU (T4)** → Run all.
2. Copy the ngrok `https://…` URL.
3. Open [`ui/index.html`](ui/index.html) → paste URL → **Send** (RAG on) / **Verify**.

Guide: [docs/RUN_IN_COLAB.md](docs/RUN_IN_COLAB.md)

```bash
# Eval against live ngrok (from your laptop)
python -m eval.run_eval --split eval --use-rag 1 --api-url $URL \
  --run-id auto --tag after_rag --out reports/ablation_sft_rag.json
python -m scripts.compare_eval_runs v002_2026-11-23_after_redeploy latest
```

---

## Repo layout

```text
serve/      FastAPI — /ask, /ask/stream, /verify, /health
eval/      Metrics + --api-url + versioned runs
train/     QLoRA SFT (`python -m train.sft`)
data/      raw/, corpus/, train|eval|redteam|prefs.jsonl
ui/        Chat client (history, stream, RAG, verify trace)
notebooks/ serve_ngrok.ipynb
reports/   week writeups + runs/v001…
docs/      roadmap, model card, demo script
```

---

## Built only on free Colab T4

- No paid GPU cluster.
- Peak serve VRAM observed ~4 GB (4-bit).
- ngrok URLs die with the runtime — **record the demo video** so reviewers do not need your live session.

Train refresh: `python -m train.sft --config train/configs/sft.yaml` (see `train/README.md`).
