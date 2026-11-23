# Versioned eval runs

Each folder is a **frozen snapshot** of metrics + full predictions.

```text
reports/runs/v001_YYYY-MM-DD_tag/
  MANIFEST.json          # summaries for the leaderboard
  NOTES.md               # human context
  baseline_sft.json      # factoid predictions
  redteam.json           # safety predictions
```

## Workflow

1. Improve the model / serve stack.
2. Re-run eval with a new run id:

```bash
python -m eval.run_eval --split eval --use-rag 0 --api-url $URL \
  --run-id auto --tag after_fix --out reports/baseline_sft.json
python -m eval.run_eval --split redteam --api-url $URL \
  --run-id v00N_... --tag after_fix --out reports/redteam.json
```

3. Compare to the frozen baseline:

```bash
python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix latest
```

4. Check `reports/LEADERBOARD.md` for the table over time.
