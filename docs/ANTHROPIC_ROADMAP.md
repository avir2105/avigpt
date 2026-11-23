# AVIGPT → Core AI Portfolio (Anthropic-Oriented Roadmap)

**Constraint (non-negotiable):** no personal GPU / cloud budget. Only **Google Colab T4 (~15 GB VRAM)** + **ngrok** for demos. Everything below is designed for that.

This is not a disadvantage if you frame it right: Anthropic (and strong teams) care whether you can **measure, reason, and ship under limits** — not whether you rented an A100.

---

## Honest framing

A prettier chat UI alone will not get you hired. What already proves you can do ML work:

- QLoRA SFT on Llama-2 on Colab
- 4-bit NF4 inference on T4
- FastAPI + ngrok public demo

What hiring managers want next:

- **Evals** (numbers, not vibes)
- **Grounding** (RAG / citations — less hallucination)
- **Safety thinking** (refusals, red-team, honesty about failures)
- **Clear writeups** (what worked, what failed, why)

You do **not** need a GPU cluster to look serious. You need a **tight experiment log** and a **working Colab+ngrok demo**.

This project can **support** an Anthropic-style application. It will not replace fundamentals + interview performance. It *does* prove resourcefulness.

---

## Hard limits (design for these)

| Resource | Reality | Rule |
|----------|---------|------|
| GPU | Colab T4 ~15 GB | Stay on **4-bit / QLoRA**; never load 7B in fp16 |
| Session | Disconnects, quotas | Checkpoint often; Hub push after every successful train |
| Demo | ngrok + Colab | One-click Colab notebook + short demo video (backup when Colab dies) |
| Time | Free tier | Prefer **small eval sets**, **short gens**, **cached baselines** |
| Disk | Colab ephemeral | Save adapters + reports to **Hugging Face Hub** / Drive |

### What fits on T4 (15 GB)

| Workload | Fits? | Notes |
|----------|-------|-------|
| 7B QLoRA SFT | Yes | What you already did; keep batch small, grad accum |
| 7B 4-bit inference | Yes | Keep `max_new_tokens` modest (128–256), not huge |
| 7B DPO full | Risky / slow | Prefer **tiny pref set** or skip; do DPO on **1B–3B** instead |
| RAG embeddings | Yes (often CPU) | `sentence-transformers` on CPU is fine |
| Eval harness | Yes | Run overnight-ish; cache model load |
| Fancy multi-agent 7B | No | One tool + RAG is enough |

**Default model strategy on Colab:**

1. Keep your finetuned 7B for the **hero demo** (identity Q&A).
2. Do **fast iteration / ablations** on a smaller instruct model (e.g. 1B–3B) when Colab is tight.
3. Always report: model size, quant, VRAM peak, tokens/sec.

That last line alone signals engineering maturity.

---

## How to “prove you’re worth it” with zero budget

Hiring signal ≠ compute spent. Signal =

1. **You measured** — base vs SFT vs SFT+RAG on the same eval set  
2. **You grounded** — citations, fewer made-up facts  
3. **You found failure modes** — red-team table with fails included  
4. **You made it reproducible** — Colab notebook + Hub model + eval script  
5. **You wrote clearly** — 1–2 page report with tradeoffs  

Optional free amplifiers:

- Public GitHub + Hugging Face model card (limitations section filled in honestly)
- 2–3 minute demo video (screen record Colab+ngrok) so reviewers don’t need your live GPU
- Short blog / README section: “Building a personal LLM on a free T4”

---

## Current baseline

| Layer | Status |
|--------|--------|
| Training | QLoRA SFT notebook on personal bio Q&A |
| Serving | `serve/app.py` — FastAPI `/ask`, 4-bit, CUDA |
| UI | `ui/index.html` — history, stream, RAG, verify |
| Deploy | Colab + ngrok (your real path); Docker optional |
| Gaps | Live Colab may lag local code; fill eval tables; RAG ablation; safety writeup |

---

## Target layout (still Colab-friendly)

```text
avigpt/
  app.py               # thin entry → serve.app
  serve/               # FastAPI + streaming + RAG + safety
  eval/                # harness + metrics + run versioning
  train/               # QLoRA scripts
  data/raw|corpus|*.jsonl
  ui/                  # production chat client
  notebooks/           # Colab train / serve
  scripts/             # prepare_data, freeze/compare eval, docker
  reports/runs/        # frozen eval versions (before → after)
  docs/                # roadmap + Colab+ngrok guide
  tests/
  archive/             # superseded (e.g. old yo.html)
  README.md
```

Commit **reports and data splits** even when weights live on Hub. Reviewers read reports; they may never run your GPU.

---

## Upgrade path (priority under T4)

### 1. Eval harness (highest ROI — cheap on compute)

- 50–100 eval Qs + 20–30 red-team (CPU to score strings; GPU only for generation)
- Metrics: exact match / token F1 for facts; refusal rate for red-team
- Table: **base Llama-2-7B-chat 4-bit** vs **your SFT 4-bit** vs **SFT+RAG**
- Save JSON results in `reports/` so you never re-run unless needed

### 2. Fix generation (no extra VRAM)

- Proper Llama-2 chat template / stop tokens
- Drop the `[/]` + first-`.` hack
- Streaming SSE (same model, better UX)
- Multi-turn: keep last N turns truncated to token budget (e.g. 1024–2048)

### 3. RAG (best “smart” upgrade on free tier)

- Chunk bio/docs; embed on **CPU**; FAISS/Chroma
- Retrieve top-k → prompt → generate on T4
- Return `sources`
- Ablation table (this is your money shot without more GPUs)

### 4. Safety (writeup > fancy training)

- Fixed constitution / system rules
- Red-team pass rates
- Optional: small preference set; DPO only if T4 allows — else document “would run DPO next on …”

### 5. One tool (not a framework zoo)

- “Verify claim against my docs” → retrieve → yes/no + quote
- Show the tool trace in the UI

### 6. Chat UI + ngrok demo (last)

- History, streaming bubbles, RAG toggle
- Colab “Run all” + ngrok URL
- **Plus** a recorded demo video (Colab will die; video won’t)

### Skip / defer on free Colab

- Multi-GPU, continuous pretrain, huge DPO, 13B+, heavy agent stacks  
- Paid vector DBs, complex k8s  
- “20 providers” wrapper apps  

---

## Concrete 4-week plan (Colab T4 + ngrok only)

Assume ~a few Colab sessions per week. Prefer **eval + writeup** on days GPU is flaky.

### Week 1 — Measure + fix decoding + stream

| Task | Colab notes | Done when |
|------|-------------|-----------|
| W1.1 Split data | CPU: `data/train.jsonl`, `eval.jsonl`, `redteam.jsonl` | ~80 eval / ~25 red-team |
| W1.2 Eval script | Load 4-bit once; `max_new_tokens≤128`; save predictions | `reports/baseline.json` |
| W1.3 Baseline table | Base 4-bit vs your Hub SFT 4-bit | Markdown table in `reports/week1_baseline.md` |
| W1.4 Fix template | Correct `[INST]` / stops; no brittle post-parse | Spot-check 10 prompts |
| W1.5 Stream API | SSE `/ask/stream` | Works via ngrok |
| W1.6 Log VRAM | `torch.cuda.max_memory_allocated` in health/logs | README quotes peak GB |

**Deliverable:** baseline numbers + fixed generation + streaming over ngrok.

**T4 tip:** cache the model between eval batches; don’t reload per question.

---

### Week 2 — RAG + ablation (CPU embeddings + T4 generate)

| Task | Colab notes | Done when |
|------|-------------|-----------|
| W2.1 Corpus | Chunk `data/raw` / resume / notes | `data/corpus/` |
| W2.2 Index | Embed on CPU; save index to Drive/Hub | Rebuildable in one cell |
| W2.3 Grounded ask | `use_rag=true` → context + sources | API returns citations |
| W2.4 Ablation | Same eval for SFT / RAG / SFT+RAG | `reports/week2_rag.md` |
| W2.5 Hallucination probe | Ask facts **not** in docs | Show RAG refuses or says unknown |

**Deliverable:** RAG path + ablation — strongest free-tier portfolio artifact.

---

### Week 3 — Safety + reproducible Colab train

| Task | Colab notes | Done when |
|------|-------------|-----------|
| W3.1 Constitution | System rules in serve path | Documented + applied |
| W3.2 Red-team report | Run eval; include **failures** | `reports/week3_safety.md` |
| W3.3 Prefs (optional) | 50 pairs by hand | `data/prefs.jsonl` |
| W3.4 DPO decision | If T4 OOM → skip or use **1B–3B**; write why | Honest note in report |
| W3.5 Train as script | `train/sft.py` runnable from Colab | Hub push after run |
| W3.6 Model card | Limitations, bias, compute used (T4) | On Hugging Face |

**Deliverable:** safety report + reproducible train path + honest model card.

---

### Week 4 — Chat UI + durable proof

| Task | Colab notes | Done when |
|------|-------------|-----------|
| W4.1 Chat UI | History + streaming + RAG toggle | Static `ui/` is enough |
| W4.2 One tool | Doc-verify with visible trace | One happy-path demo |
| W4.3 Colab Serve notebook | Install → load → uvicorn → ngrok | `notebooks/serve_ngrok.ipynb` |
| W4.4 Demo video | 2–3 min screen record | Link in README |
| W4.5 Portfolio README | Problem → method → tables → failures → “only T4” | Skimmable in 5 min |

**Deliverable:** public GitHub + Hub model + Colab notebook + video (live ngrok optional).

---

## Session playbook (every Colab run)

```text
1. Runtime → GPU (T4)
2. Clone repo / mount Drive
3. pip install from requirements
4. nvidia-smi + log free VRAM
5. Do ONE job: train OR eval OR serve (don’t mix)
6. Push artifacts (adapter, reports JSON) before disconnect
7. If serving: ngrok → test /health → record clip if demo day
```

**VRAM hygiene on T4:**

- Always 4-bit for 7B  
- `max_new_tokens` 64–256 for eval/demo  
- Clear CUDA cache between experiments  
- Don’t run Streamlit + FastAPI + train in one session  

---

## Success criteria (zero-budget bar)

You’re “worth showing” when:

1. Eval table exists (base vs SFT vs SFT+RAG) with real numbers  
2. Decoding is correct + streaming works on ngrok  
3. RAG cites sources; ablation writeup committed  
4. Red-team report includes fails  
5. Colab notebooks reproduce train/serve/eval  
6. Demo video exists (independent of Colab uptime)  
7. README says clearly: **built and measured on free Colab T4**

That last point is a feature: it shows you ship under constraints.

---

## Role-matching (still valid without cluster)

| Target vibe | Lean on |
|-------------|---------|
| Applied / Product | Chat UX, streaming, ngrok demo, RAG citations |
| Inference / Platform | 4-bit, VRAM logs, latency, Colab packaging |
| Safety / Research-minded | Evals, red-team, honesty about failures, writeups |

For Anthropic-shaped storytelling, **evals + failure analysis + grounding** beat “I fine-tuned on more GPUs.”

---

## What not to do

- Wait for a paid GPU before starting evals/RAG (CPU/T4 is enough)  
- Spend weeks on UI gradients  
- Claim cluster-scale results you didn’t run  
- Hide Colab limits — **document them**  

---

## Implementation status (in-repo)

Shipped (CPU scaffolding; fill GPU numbers on Colab):

- `data/` splits + corpus + prefs + redteam
- `serve/` — prompts, 4-bit model, SSE stream, RAG, safety, `/verify`
- `eval/run_eval.py` + `reports/week*.md` stubs
- `train/sft.py` + YAML config
- `ui/index.html` — history, stream, RAG toggle, verify
- Colab cells: `notebooks/serve_colab.md`

### Your next Colab session

1. `python -m eval.run_eval --split eval --out reports/baseline.json`
2. Fill tables in `reports/week1_baseline.md`
3. Serve + ngrok → open `ui/index.html`
