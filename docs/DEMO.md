# Demo video script (2–3 minutes)

Record once Colab+ngrok is up. Paste the finished link into the README (`Demo video` section).

## Checklist before record

- [ ] `notebooks/serve_ngrok.ipynb` cells 1–4 succeeded
- [ ] `/health` shows model + T4 VRAM
- [ ] `ui/index.html` has the ngrok URL; status pill says **API: ok**

## Suggested takes

1. **Setup (15s)** — Show Colab GPU runtime + ngrok URL briefly. Say: “7B QLoRA, 4-bit on free T4.”
2. **Ask + RAG (45s)** — Toggle RAG on. Ask “Where was Avinash born?” Show streaming answer + source chips.
3. **Verify tool (45s)** — Paste claim “He interned at Clairvoyant” → **Verify**. Show verdict + yellow tool trace.
4. **Safety (30s)** — Ask for bank account / Aadhaar. Show refusal.
5. **Evals (30s)** — Flash `reports/LEADERBOARD.md` or week2 table: SFT vs SFT+RAG contains_gold 0.19 → 0.63; red-team 0.96.

## Upload

YouTube unlisted / Loom / Drive → put URL in README:

```markdown
**Demo video:** <paste-link-here>
```
