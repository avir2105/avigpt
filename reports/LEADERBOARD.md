# Eval leaderboard

Frozen runs live under `reports/runs/`. Newer rows should beat older ones on the same split.

## Factoid (`eval` split)

| Run | RAG | n | Exact | Token F1 | Contains gold | Notes |
|-----|-----|---|-------|----------|---------------|-------|
| `v001_2026-11-23_pre_fix` | False | 80 | 0.0 | 0.4285 | 0.0 | baseline_sft.json — Frozen baseline from first ngrok API eval (2026-11-23). Live |
| `v002_2026-11-23_after_redeploy` | False | 80 | 0.0375 | 0.4066 | 0.1875 | baseline_sft.json — after_redeploy |
| `v004_2026-11-23_after_rag` | True | 80 | 0.1 | 0.526 | 0.625 | ablation_sft_rag.json — after_rag |

## Red-team

| Run | n | Pass rate | Notes |
|-----|---|-----------|-------|
| `v001_2026-11-23_pre_fix` | 25 | 0.44 | redteam.json — Frozen baseline from first ngrok API eval (2026-11-23). Live |
| `v001_2026-11-23_pre_fix` | 25 | 0.16 | redteam_rescored.json — Frozen baseline from first ngrok API eval (2026-11-23). Live |
| `v003_2026-11-23_after_redeploy` | 25 | 0.96 | redteam.json — after_redeploy |

## How to add a new run

```bash
python -m eval.run_eval --split eval --use-rag 0 --api-url $URL --run-id auto --tag after_rag \
  --out reports/baseline_sft.json
python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix auto
```

Use `--run-id auto` to allocate the next `vNNN_...` folder. Pass an explicit id to append into an existing run.
