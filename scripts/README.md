# Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Rebuild JSONL splits + corpus from `data/raw/` |
| `freeze_eval_run.py` | Snapshot existing report JSON into `reports/runs/vNNN_…` |
| `compare_eval_runs.py` | Diff two frozen runs (before → after) |
| `run_docker.ps1` | Build/run the API container from repo root |

```bash
python -m scripts.prepare_data
python -m scripts.freeze_eval_run --run-id auto --tag pre_fix
python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix latest
```
