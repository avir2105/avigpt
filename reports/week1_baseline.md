# Week 1 — Baseline eval

> First measured run: 2026-11-23 via live Colab+ngrok (`--api-url`). See also `reports/CURRENT_STATUS.md`.

## How to run

```bash
# Preferred: model already serving on Colab+ngrok — run from Windows (CPU scores, no local GPU):
python -m eval.run_eval --split eval --use-rag 0 --api-url https://YOUR-NGROK.ngrok-free.app --out reports/baseline_sft.json
python -m eval.run_eval --split redteam --api-url https://YOUR-NGROK.ngrok-free.app --out reports/redteam.json

# Or load model in-process on Colab T4 (stop serve first — don't double-load):
python -m eval.run_eval --split eval --use-rag 0 --out reports/baseline.json
python -m eval.run_eval --split redteam --out reports/redteam.json
```

Optional local/Colab in-process base vs SFT:

```bash
MODEL_NAME=NousResearch/Llama-2-7b-chat-hf python -m eval.run_eval --split eval --out reports/baseline_base.json
MODEL_NAME=avi2135/avigpt2 python -m eval.run_eval --split eval --out reports/baseline_sft.json
```

## Results (2026-11-23)

| Model | Quant | n | Exact Match | Token F1 | Contains gold | Peak VRAM (GB) | Notes |
|-------|-------|---|-------------|----------|---------------|----------------|-------|
| Llama-2-7B-chat | 4-bit | 80 | TBD | TBD | TBD | TBD | base — **not run yet** |
| avi2135/avigpt2 (live API) | 4-bit (assumed) | 80 | **0.00** | **0.4285** | **0.00** | n/a (`/health` 404) | SFT via ngrok; RAG off |

Red-team (same backend, n=25): reported pass **0.44**; stricter refusal-only read **~0.20** (scoring inflated when question matched local refuse regex).

## Decoding fixes shipped (in repo; verify on Colab redeploy)

- Proper Llama-2 `[INST]` / `<<SYS>>` prompt builder (`serve/prompts.py`)
- Stopped brittle `[/]` + first-`.` truncation
- SSE streaming: `POST /ask/stream`
- `/health` reports VRAM peak / load meta

## Failures observed

1. **Question echo** — model often reprints the question before / inside the answer (~60/80).
2. **Prompt / markup leakage** — `<<SYS>>`, `[INST]`, `[/html]`, “Keywords:” junk (~41/80).
3. **Name / token corruption** — `Aviash`, glued words, truncated mid-sentence.
4. **Zero exact / contains-gold** — lexical metrics stay at 0 even when F1 is mid; answers are fuzzy paraphrases.
5. **Safety leaks on live API** — invented bank/Aadhaar/phone/address on red-team PII prompts; constitution filter not effective on deployed build.
6. **Deploy skew** — live server missing `/health` → older Colab code than local `serve/app.py`.
