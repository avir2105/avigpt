# AVIGPT

Personal Q&A LLM about Avinashsingh Rajput — **QLoRA SFT + 4-bit serve + RAG + evals**, built to run on **free Google Colab T4 (15 GB)** and exposed via **ngrok**.

## Why this project

Not just a chatbot wrapper. The portfolio loop is:

1. Fine-tune (QLoRA) → 2. Measure (eval harness) → 3. Ground (RAG) → 4. Refuse unsafe asks → 5. Demo (ngrok + chat UI)

Full plan: [docs/ANTHROPIC_ROADMAP.md](docs/ANTHROPIC_ROADMAP.md)

## Repository layout

```text
avigpt/
  app.py                 # Thin entrypoint → serve.app (Docker / Colab)
  requirements.txt
  Dockerfile             # Optional GPU image
  serve/                 # FastAPI, model, RAG, safety, prompts
  eval/                  # Metrics + versioned run harness
  train/                 # QLoRA SFT script + config
  data/
    raw/                 # Source dumps (mydata / mybig)
    corpus/              # RAG facts
    *.jsonl              # train / eval / redteam / prefs
  ui/                    # Production chat client
  notebooks/             # Colab train / serve notebooks
  scripts/               # Data prep, eval freeze/compare, docker helper
  reports/
    runs/                # Frozen eval versions (before → after)
  docs/                  # Roadmap + Colab+ngrok guide
  tests/                 # CPU unit tests
  archive/               # Superseded files (do not use)
```

## Stack

| Piece | Path |
|-------|------|
| API | `serve/app.py` — `/ask`, `/ask/stream`, `/verify`, `/health` |
| Prompts / safety | `serve/prompts.py`, `serve/safety.py` |
| RAG (CPU TF-IDF) | `serve/rag.py` + `data/corpus/facts.txt` |
| Eval | `eval/run_eval.py` |
| Train | `train/sft.py` + `train/configs/sft.yaml` |
| UI | `ui/index.html` |
| Data | `data/*.jsonl`, `data/raw/`, `data/corpus/` |

Default model: `avi2135/avigpt2` (override with `MODEL_NAME`).

## Colab quickstart (T4)

1. GPU runtime → clone repo → `pip install -r requirements.txt`
2. Serve: `python -m serve.app` (or `python app.py`)
3. ngrok: [docs/RUN_IN_COLAB.md](docs/RUN_IN_COLAB.md)
4. Open `ui/index.html` locally → paste the ngrok URL

### Eval (versioned)

```bash
python -m eval.run_eval --split eval --use-rag 0 --api-url $URL \
  --run-id auto --tag after_fix --out reports/baseline_sft.json
python -m eval.run_eval --split redteam --api-url $URL \
  --run-id auto --tag after_fix --out reports/redteam.json
python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix latest
```

See `reports/LEADERBOARD.md` and `reports/runs/README.md`.

### Rebuild data splits

```bash
python -m scripts.prepare_data
```

### Train (QLoRA)

```bash
python -m train.sft --config train/configs/sft.yaml
```

### Docker (optional)

```powershell
.\scripts\run_docker.ps1
```

## API sketch

```bash
curl -s $BASE/health
curl -s -X POST $BASE/ask -H "Content-Type: application/json" \
  -d '{"question":"What'\''s my name?","use_rag":true}'
curl -s -X POST $BASE/verify -H "Content-Type: application/json" \
  -d '{"claim":"He interned at Clairvoyant"}'
```

## Constraints (by design)

- Free **Colab T4 only** — 4-bit / QLoRA, modest `MAX_NEW_TOKENS`
- No paid GPU cluster
- ngrok URLs are ephemeral; keep a short demo video for portfolio

## Tests (CPU)

```bash
python -c "from tests.test_core import *; test_refusal_heuristic(); test_prompt_and_extract(); test_clean_answer_strips_echo_and_junk(); test_redteam_scoring_not_inflated_by_question_pattern(); test_metrics(); test_rag_search(); print('ok')"
```
