# Raw source text

Unprocessed bio / Q&A dumps used to rebuild splits:

```bash
python -m scripts.prepare_data
```

That writes `data/train.jsonl`, `data/eval.jsonl`, `data/redteam.jsonl`, `data/prefs.jsonl`, and `data/corpus/facts.txt`.

Do not serve these files directly — use the JSONL / corpus outputs.
