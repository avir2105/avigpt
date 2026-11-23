# Week 2 — RAG ablation

RAG is implemented (`serve/rag.py` + `use_rag` on `/ask`). Measured on live Colab T4 + ngrok, 2026-11-23.

Corpus: `data/corpus/facts.txt` (TF-IDF on CPU).

## Status checklist

| Task | Status |
|------|--------|
| W2.1 Corpus in `data/corpus/` | Done |
| W2.2 Index builds on serve startup | Done |
| W2.3 `/ask` with `use_rag: true` returns `sources` | Done (smoke + full eval) |
| W2.4 Ablation table SFT vs SFT+RAG | Done |
| W2.5 Hallucination probes | Optional follow-up |

## Ablation table (n=80, same model `avi2135/avigpt2` 4-bit)

| Mode | Run | Exact | Token F1 | Contains gold | Notes |
|------|-----|-------|----------|---------------|-------|
| SFT only | v002 | 0.0375 | 0.4066 | 0.1875 | `use_rag=0` |
| SFT + RAG | v004 | **0.1000** | **0.5260** | **0.6250** | `use_rag=1` |

### Deltas (v002 → v004)

| Metric | Change |
|--------|--------|
| Exact match | +0.0625 |
| Token F1 | **+0.1194** |
| Contains gold | **+0.4375** |

**Verdict:** RAG clearly helped. Biggest win is **contains_gold** (19% → 63%) — answers more often include the real fact tokens from the corpus. F1 and exact match also rose. This is the portfolio money shot for free-tier Colab.

## Hallucination probe

| Probe | RAG off | RAG on |
|-------|---------|--------|
| NASA JPL projects? | TBD (optional) | TBD |
| Nobel Prize 2020? | TBD | TBD |

## How it was run

```powershell
python -m eval.run_eval --split eval --use-rag 1 --api-url https://4c3a-34-139-195-43.ngrok-free.app --run-id auto --tag after_rag --out reports/ablation_sft_rag.json
python -m scripts.compare_eval_runs v002_2026-11-23_after_redeploy latest
```
