---
language: en
license: other
tags:
  - llama-2
  - qlora
  - personal-qa
  - rag
base_model: NousResearch/Llama-2-7b-chat-hf
---

# AVIGPT (`avi2135/avigpt2`)

Personal Q&A assistant about **Avinashsingh Rajput**, fine-tuned with **QLoRA** and served in **4-bit** on a free **Google Colab Tesla T4 (~15 GB)**.

This card is the portfolio-facing description. Numbers come from versioned runs under `reports/runs/` in the GitHub repo.

## Intended use

- Answer public-profile questions (education, projects, skills, background) grounded in training data and/or a local RAG corpus.
- Demo stack: FastAPI (`serve/`) + ngrok + optional `ui/index.html`.

## Out of scope

- Not a general chatbot.
- Not a source of private data (bank, Aadhaar, medical, exact salary, doxxing).
- Not a substitute for verified resumes or legal identity documents.

## Training

| Item | Value |
|------|--------|
| Base | Llama-2-7B-chat (HF) |
| Method | QLoRA (4-bit NF4) |
| Hardware | Google Colab T4 |
| Data | Personal bio Q&A JSONL (`data/train.jsonl`) |
| Repro script | `python -m train.sft --config train/configs/sft.yaml` |

## Evaluation (Colab T4 + ngrok, 2026-11-23)

Factoid set: 80 held-out questions.

| Mode | Exact | Token F1 | Contains gold | Run |
|------|-------|----------|---------------|-----|
| SFT only | 0.0375 | 0.4066 | 0.1875 | v002 |
| SFT + RAG (TF-IDF corpus) | 0.1000 | 0.5260 | 0.6250 | v004 |

Red-team: 25 prompts.

| Era | Pass rate | Run |
|-----|-----------|-----|
| Pre-redeploy (rescored) | 0.16 | v001 |
| Post-redeploy filter + constitution | 0.96 | v003 |

Known residual fail (v003): inventing fake companies when asked (`rt02`). Filter patterns extended afterward; re-measure after redeploy.

## Safety

- System constitution in `serve/safety.py` / `serve/prompts.py`.
- Regex hard-refuse for PII, malware, jailbreaks, forgery, and fabrication asks.
- Failures are documented in `reports/week3_safety.md` (including the miss).

## Limitations & bias

- Answers can still paraphrase incorrectly; lexical exact match remains low without RAG.
- RAG uses TF-IDF over a personal corpus — not semantic embeddings; retrieval can miss paraphrases.
- Model may refuse benign questions that match refusal keywords.
- Trained on one person’s bio → not representative of other people or cultures.
- Free Colab sessions disconnect; demos should include a short recorded video.

## Compute statement

Built and measured on **free Colab T4 only** (no paid GPU cluster). Peak serve VRAM observed ~4 GB allocated in 4-bit mode.

## How to run

See repo `README.md` and `docs/RUN_IN_COLAB.md` (ngrok).

```bash
python -m serve.app
# then ngrok → POST /ask with {"question":"…","use_rag":true}
```
