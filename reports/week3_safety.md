# Week 3 — Safety / red-team

## How to run

```bash
python -m eval.run_eval --split redteam --out reports/redteam.json
```

## Pass rate (fill in)

| Filter | n | Pass rate | Notes |
|--------|---|-----------|-------|
| Heuristic + model | 25 | TBD | see `serve/safety.py` |

## Failures (must list)

_Paste failing `id`s from `reports/redteam.json` and what went wrong._

## DPO note (T4)

Full 7B DPO on free Colab T4 is optional / often too tight. Preference seed lives at `data/prefs.jsonl`. If OOM: document skip or run DPO on a 1B–3B proxy.
