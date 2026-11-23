# Week 2 — RAG ablation

## How to run

```bash
python -m eval.run_eval --split eval --use-rag 0 --out reports/ablation_sft.json
python -m eval.run_eval --split eval --use-rag 1 --out reports/ablation_sft_rag.json
```

## Ablation table (fill in)

| Mode | Token F1 | Contains gold | Notes |
|------|----------|---------------|-------|
| SFT only | TBD | TBD | `use_rag=0` |
| SFT + RAG | TBD | TBD | TF-IDF corpus `data/corpus/facts.txt` |

## Hallucination probe

Ask facts **not** in docs (e.g. NASA / Nobel). With RAG on, model should say unknown / contradict.

| Probe | RAG off | RAG on |
|-------|---------|--------|
| NASA JPL projects? | TBD | TBD |
| Nobel Prize 2020? | TBD | TBD |
