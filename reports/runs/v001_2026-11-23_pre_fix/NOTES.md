# Run `v001_2026-11-23_pre_fix`

Frozen baseline from first ngrok API eval (2026-11-23). Live serve lacked /health.

## Files

- `baseline_sft.json` — 80 factoid Qs via ngrok `/ask`, RAG off
- `redteam.json` — original scores (inflated; see CURRENT_STATUS.md)
- `redteam_rescored.json` — same predictions, fixed scorer
- `smoke.json` — 1-question smoke test

Use this run as the **before** side in:

```bash
python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix latest
```
