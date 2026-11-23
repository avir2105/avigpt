# Interview prep: What SFT does, why it isn’t “truth,” and what is

Use this to answer calmly when someone says: *“If SFT alone memorizes vibes and doesn’t guarantee truth — then what does?”*

---

## 30-second answer (memorize this)

**SFT teaches the model *how to behave* on your data distribution — tone, format, which topics it should talk about, and average answers that look like your examples. It does *not* install a verified knowledge base.**

Nothing in a neural net “guarantees truth.” What you can do is **raise the chance of correct answers** by:

1. **Grounding** — retrieve documents at ask-time (RAG) and constrain the model to that context  
2. **Tools** — verify / lookup / calculate instead of free-generating facts  
3. **Measurement** — evals that catch lies (exact/F1/contains-gold, red-team)  
4. **Refusal** — say “I don’t know” when unsupported  

In AVIGPT terms: SFT made a personal assistant; **RAG + `/verify` + evals** are what push answers toward your actual bio facts.

---

## What SFT is (Supervised Fine-Tuning)

### Plain English

You show the model many pairs:

```text
Question → Answer you want
```

The training objective is: **predict the next tokens of the answer**, given the prompt.  
Over thousands of updates, the model’s weights shift so that *similar questions* tend to produce *similar-looking answers*.

That is **behavioral cloning** on your dataset — not “upload truth into the model.”

### What SFT is good at

| Skill | Example |
|-------|---------|
| Style / persona | “Answer as AVIGPT about Avinash…” |
| Format | Short answers, chat template `[INST]…[/INST]` |
| Task shape | Q&A, not random essay |
| Domain tilt | More likely to mention IIIT / family / skills that appeared in train data |
| Following instructions better | Obey system rules *more often* (still imperfect) |

### What SFT is *not*

| Myth | Reality |
|------|---------|
| “SFT stores a database of facts” | Weights are a **lossy compression** of patterns. Facts can be wrong, mixed, or outdated. |
| “If it was in the train set, it will always recall it” | Recall is probabilistic. Paraphrase the question → maybe fail. |
| “SFT guarantees truth” | SFT guarantees **likelihood under the fine-tune distribution**, not external correctness. |
| “More SFT epochs = more truth” | Often = more **overfit / parroting / hallucination confidence**. |

### Interview phrase

> “SFT optimizes for *looking like the training answers*, not for *being checkable against the world*.”

---

## Why people say “memorizes vibes”

“Vibes” means:

- The model learned **associations** (“father” ↔ “Rajput”, “born” ↔ “Maharashtra-ish”)  
- Not a crisp row in a table: `birthplace = Akhatwade, Taluka Pachora, Jalgaon`

So under chat pressure (“yes”, “tell me more”), it **continues the story** with plausible filler — Indore, random dates, extra relatives — because fluent continuation scored well in pretraining + SFT, while **strict factual grounding did not**.

Your own evals showed this:

| Mode | Contains gold (fact tokens present) |
|------|-------------------------------------|
| SFT only | ~0.19 |
| SFT + RAG | ~0.63 |

Same model weights. Different **ask-time** information. That’s the smoking gun for interviews.

---

## Who / what “guarantees” truth?

### Honest answer: **nothing fully guarantees it**

Not SFT, not RAG, not GPT-4, not a human without checking sources.

In production ML / safety interviews, the mature answer is:

> “We don’t claim guarantees. We claim **controls**: retrieval, tools, refusals, and evals — and we publish failure modes.”

### What *approximates* truth better (ranked)

#### 1. Grounding at inference (RAG / context injection) — strongest cheap lever

**Idea:** Don’t ask the model to remember. **Give it the paragraph** that contains the fact, then ask it to answer from that.

- Truth lives in **documents you control** (`data/corpus/facts.txt`)  
- Model’s job shrinks to: read + paraphrase + don’t contradict  

**Limits:** Bad retrieval → wrong doc → wrong answer. Garbage in, garbage out. Still not a guarantee — but it’s **auditable** (you can show `sources`).

**Interview line:**  
> “RAG doesn’t make the model truthful; it makes answers *attributable*. Attribution is what you can defend.”

#### 2. Tools (verify, search, SQL, calculator)

Your `/verify` endpoint: retrieve → judge SUPPORTED / CONTRADICTED / UNKNOWN + trace.

**Idea:** Separate **generation** from **checking**.  
Humans do this too (draft → look up).

**Limits:** Tool can be wrong; claim phrasing can confuse the judge model.

#### 3. Constrained decoding / schemes (advanced)

- Answer only with spans from context  
- JSON schema: `{ "answer": ..., "cite": ... }`  
- “If not in context → UNKNOWN”

These reduce creativity where creativity hurts.

#### 4. Preference optimization (DPO / RLHF) — helps honesty *sometimes*

Prefs teach: “chosen refusal / grounded answer” ≻ “invented prestige bio.”

**Limits:** Still no database. Can teach *style of honesty* without ensuring facts. You skipped 7B DPO on free T4 for compute reasons — say that openly.

#### 5. Better / cleaner SFT data — helps, doesn’t finish the job

Dedupe contradictions, one canonical birthplace, no conflicting Indore vs Jalgaon rows.  
SFT on clean data → fewer vibes. Still probabilistic.

#### 6. Evals — don’t create truth; they **detect** lies

Exact match, token F1, contains-gold, red-team pass rate, versioned `reports/runs/`.

**Interview line:**  
> “Truth isn’t guaranteed by training. It’s *monitored* by evaluation and *enforced* by grounding and refusals.”

---

## Mental model (draw this on a whiteboard)

```text
Pretraining     → general language + lots of world*approximate* knowledge
     ↓
SFT             → “act like my assistant / my answer style”
     ↓
(optional) DPO  → prefer safer / more preferred replies
     ↓
Serving stack   → RAG + tools + constitution + filters   ← where “truthfulness” is engineered
     ↓
Evals           → prove whether it worked
```

**SFT is necessary but not sufficient for factual personal Q&A.**

---

## How this maps to AVIGPT (use as your project story)

| Layer | Role re: truth |
|-------|----------------|
| QLoRA SFT (`avi2135/avigpt2`) | Learns to answer *as* a personal bio bot |
| `serve/prompts.py` + constitution | Instructs honesty / don’t invent |
| `serve/safety.py` hard filter | Blocks some unsafe / fabricate asks before generate |
| `serve/rag.py` | Injects corpus chunks at ask-time |
| `/verify` | Tool path with visible trace |
| `eval/run_eval.py` + `reports/runs/` | Before/after numbers (v002 vs v004) |

**Portfolio sentence:**  
> “I fine-tuned for persona and task format, then treated factuality as a *systems* problem: retrieve, cite, verify, refuse, and measure.”

That’s an Anthropic-shaped answer: not “I trained harder,” but “I instrumented honesty.”

---

## Tough interview questions & strong answers

### Q: Doesn’t the base model already know facts? Why SFT?

**A:** Base/chat models know *generic* patterns. SFT specializes to *your* schema and bio distribution. It still won’t reliably hold a private knowledge graph. For personal assistants, SFT + RAG beats either alone.

### Q: If RAG is the truth layer, why fine-tune at all?

**A:**  
- Better instruction following and style with short context  
- Less fighting the base model’s generic persona  
- Can answer *some* in-domain asks when retrieval misses  
- Demo quality / latency with smaller prompts  

RAG without any domain SFT often works; SFT without RAG often *sounds* right and isn’t. Together is the product.

### Q: Can you guarantee no hallucinations?

**A:** No. I can bound them: context-only prompts, refuse when empty retrieval, show sources, red-team, and keep a failure log (e.g. rt02 invent-fake-companies). Guarantees are for formal verification — LLMs aren’t that.

### Q: What metric do you trust for “truth”?

**A:** Depends on the claim type.  
- Closed bio facts: contains-gold / exact against a gold set  
- Open claims: human or tool verify against corpus  
- Safety: refusal rate on red-team, with fails listed  

I don’t trust loss alone or “the answer sounded good.”

### Q: Why did contains-gold jump with RAG but exact match stay modest?

**A:** Exact match is harsh (string equality after tokenization). RAG helps the model include the right entities (birthplace tokens) even when wording differs from gold. That’s expected — and I report both metrics so I’m not metric-hacking.

### Q: Isn’t TF-IDF RAG weak vs embeddings?

**A:** Yes, semantically weaker. On free Colab CPU it’s a deliberate tradeoff: simple, local, no extra GPU. I’d upgrade to sentence-transformers if paraphrase misses dominate the error analysis.

---

## One-page cheat sheet

| Term | One line |
|------|----------|
| **SFT** | Teach the model to imitate labeled answers (behavior/format/domain tilt) |
| **Truth** | Correspondence to verified sources — not a training loss |
| **Hallucination** | Fluent output unfaithful to sources / reality |
| **RAG** | Fetch sources at ask-time; shrink the generation problem |
| **Tool use** | Check or act instead of only sampling tokens |
| **Constitution / filter** | Policy layer: refuse or constrain |
| **Evals** | Empirical honesty — publish fails |

**Soundbite:**  
> “SFT shapes the assistant. Grounding and tools shape the facts. Evals keep me honest about both.”

---

## How to study this for interviews

1. Explain SFT objective in one sentence (next-token on instruction data).  
2. Draw the stack diagram above.  
3. Quote **your** numbers: SFT vs SFT+RAG contains-gold.  
4. Admit a failure (v001 PII invent / rt02) and the fix.  
5. End with: no guarantee — only systems + measurement.

If they push “who guarantees truth?” — smile and say:

> “Verified sources and process guarantee *accountability*. Models only approximate. My job is to wire the model to sources and prove it with evals.”
