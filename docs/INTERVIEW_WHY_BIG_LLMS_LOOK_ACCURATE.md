# Interview prep: Why big LLMs look accurate — and why my SFT model misses “easy” facts

Pair with [`INTERVIEW_SFT_TRUTH_AND_GROUNDING.md`](INTERVIEW_SFT_TRUTH_AND_GROUNDING.md).

---

## 30-second answer (memorize this)

> “Frontier models look accurate on **common world knowledge** because that knowledge was repeated massively in pretraining and then shaped by SFT + preference training + product layers (tools/search).  
> My AVIGPT problem is different: **rare personal facts** that exist only in a small dataset. SFT alone can’t act like a reliable database. That’s why I add **RAG, verify tools, refusals, and evals** — and why my numbers jump when RAG is on.”

If they push: *“But it was in your training data!”*

> “Being in the train set raises probability; it doesn’t create guaranteed recall. Slightly different wording, multi-turn ‘yes’, 4-bit 7B, and no retrieved context — and the model continues a *plausible story* instead of the row in my facts file.”

---

## The unfair comparison interviewers accidentally make

| What people test ChatGPT on | What people test *my* bot on |
|-----------------------------|------------------------------|
| Capitals, famous people, generic coding | Father’s name, village, niche internship |
| Facts repeated across the internet | Facts that barely exist outside my JSONL |
| Fresh chat, clear question | Follow-ups like “yes”, “tell me more” |
| Often with tools / browsing (product) | Raw 7B 4-bit + optional RAG |

**Line:**  
> “I’m not competing with Claude on Wikipedia. I’m building a personal knowledge assistant under Colab T4 constraints.”

---

## Why frontier LLMs *appear* close to accurate

### 1. Scale of pretraining (the big one)

- Trained on enormous, diverse text.
- Popular facts appear **thousands of times** → strong statistical attractors.
- Your birthplace / family details appear **dozens of times at best** in a tiny SFT set.

So “who is the president?” is easy for them; “exact wording of my bio fact #17” is hard for a 7B QLoRA adapter.

### 2. They use more than SFT

Typical stack:

```text
Pretrain → SFT → preference optimization (RLHF/DPO) → safety / eval loops → product (tools, retrieval, longer context)
```

You mostly shipped: **QLoRA SFT + serve + RAG + light safety + measured evals** on free T4.  
That’s the right portfolio story — don’t pretend you matched frontier post-training.

### 3. They’re strong at *generic* knowledge, weak at *your private* knowledge

Ask a frontier model your private details **without pasting documents** — it will also invent confidently.

**Demo for interviews:**  
> “Accuracy without sources is luck on common facts and hallucination on private facts — for every model.”

### 4. Product UX hides the raw model

Chat products may:

- Retrieve / search  
- Use tools  
- Apply stronger system policies  
- Use larger models / MoE / better decoding  

Your `ui/index.html` can call a **naked** 7B endpoint. Compare systems fairly.

### 5. Calibration and refusal (sometimes)

Better models more often say “I’m not sure” on unknowns *in some domains*.  
Small SFT models often become **confident storytellers**, especially after fine-tuning on Q&A that always had an answer.

---

## Why *my* model fails “simple” training questions

### 1. SFT optimizes imitation, not lookup

Training loss: predict answer tokens.  
Success = “sounds like training answers,” not “SELECT birthplace FROM facts.”

### 2. Personal facts are rare and easy to overwrite

Pretraining prior (“people are born in big cities like Indore”) can beat a few SFT rows about a village in Jalgaon — especially in free chat without RAG.

### 3. Data quality / contradictions

If `data/raw` / corpus has mixed 1st person vs 3rd person, paraphrases, or conflicting locations, SFT learns a **blur**.  
TF-IDF RAG can still surface the right chunk; the generative path alone may average conflicts.

### 4. Ask-time conditions differ from training

Failures spike when:

- RAG is **off**  
- User says **“yes”** / **“tell me more”** (no retrieveable fact string)  
- History drifts the topic  
- Decoding continues past a short factual span into filler  

### 5. Model size + 4-bit + short generations

`avi2135/avigpt2` on T4 in 4-bit is intentional under budget.  
Capacity ≪ frontier. Rare memorization is fragile under quantization.

### 6. “It was in train” ≠ “eval will pass”

Your metrics already showed this:

| Mode | Contains gold |
|------|----------------|
| SFT only (v002) | ~0.19 |
| SFT + RAG (v004) | ~0.63 |

Same weights. Different information at inference.  
That’s the core teaching point.

---

## What actually improved accuracy in *this* project

| Lever | Effect |
|-------|--------|
| RAG on (`use_rag=true`) | Large jump in contains-gold / F1 |
| Redeployed safety + constitution | Red-team 0.16 → 0.96 |
| `/verify` tool | Check claims against corpus with a trace |
| Versioned evals | Can show before/after instead of vibes |

**What I would do next if asked:**

1. Context-only prompting when RAG hits exist; else “I don’t know”  
2. Expand follow-up “yes” into a full query using last topic + retrieve again  
3. Canonical one-row-per-fact corpus (no conflicting birthplaces)  
4. Optional: embedding retrieval; small DPO for “don’t invent” (compute permitting)

---

## Whiteboard diagram

```text
Frontier chat product
  [huge pretrain] → [SFT] → [RLHF] → [tools/search] → accurate on COMMON facts

AVIGPT (Colab T4)
  [7B + QLoRA SFT] → good persona / format
       ↓
  without RAG → “vibes” on PERSONAL facts (fail mode)
       ↓
  with RAG + verify + refuse + evals → attributable answers (success mode)
```

---

## Tough Q&A

### Q: So is your fine-tune useless?

**A:** No. It teaches persona, chat format, and domain tilt. Factual reliability for a personal KB is a **systems** problem. Fine-tune alone was never the whole product.

### Q: Why not just use GPT-4 with your docs in the prompt?

**A:** Valid product choice. My goal was to show I can **train, serve 4-bit, RAG, evaluate, and harden safety under free T4 constraints** — the skill signal for applied LLM roles — not to outspend on API wrappers.

### Q: Didn’t the model see the answer in SFT? Isn’t that leakage?

**A:** Train and eval splits are separate (`data/train.jsonl` vs `eval.jsonl`). Even train items aren’t guaranteed at inference. Also eval measures generalization to held-out phrasings — that’s the point.

### Q: Your model gave a wrong birthplace in chat. Isn’t that a failed project?

**A:** It’s a documented failure mode of ungrounded generation. With RAG, birthplace retrieval worked in smoke tests and metrics rose. I treat wrong free-chat answers as a reason to **default RAG / refuse**, not to hide the failure.

### Q: How do you know big models aren’t also vibes?

**A:** They are — with better odds on common facts. Difference is **frequency in pretraining** + **product grounding**. I don’t claim magical truth; I claim measured grounding.

---

## “Why didn’t you just make the dataset 100× bigger / train more epochs?”

This is a common trap. They’re testing whether you chase **loss vibes** or **systems + measurement**.

### Argument structure (use this order)

**1. Acknowledge the idea (don’t be defensive)**  
> “I considered more SFT data and longer training. On a personal bio corpus, that’s rarely the bottleneck.”

**2. Say what bigger/duplicated data actually buys**  
> “100× repeating the same Q&A mostly improves memorization of fixed phrasings. It doesn’t fix rephrased questions, multi-turn ‘yes’ drift, or confident invention when context is missing. More epochs on the same rows usually means more overfitting, not more truth.”

**3. Say what VRAM/time looked like under my constraint**  
> “I was on free Colab T4 ~15 GB. Dataset size barely changes VRAM with QLoRA — time does. Burning quota on duplicate epochs would delay the experiments that actually moved metrics.”

**4. Point at the controlled evidence**  
> “I already ran an ablation: same 7B 4-bit weights, RAG off vs on. Contains-gold went from about 0.19 to 0.63. That gain came from grounding at inference, not from another SFT epoch. If the interviewer wanted ‘just train more,’ my counter-evidence is that ablation.”

**5. Reframe what ‘accurate’ meant for this project**  
> “My goal wasn’t leaderboard LM loss. It was a demoable personal assistant under constraints: attributable answers, safety refusals, and before/after evals. Accuracy here is a **retrieval + policy + measurement** problem after a reasonable SFT.”

**6. Close with what you *would* do with more compute (shows maturity)**  
> “With more budget I’d grow **diverse paraphrases**, clean contradictions, maybe embedding RAG or a small DPO for ‘don’t invent’ — not 100× duplicate the same file. I already seeded `prefs.jsonl` and documented skipping 7B DPO on T4 honestly.”

### One-liner if interrupted

> “I didn’t skip accuracy work — I prioritized the lever that measured better under T4: grounding and evals, not longer SFT on duplicated bio text.”

### What *not* to say

- “I didn’t have time” (sounds unfinished)  
- “SFT is useless” (too absolute)  
- “Bigger data never helps” (wrong — **diverse** data helps; **duplicates** don’t)

### If they insist “more data always helps”

> “More *information* helps. More *copies* of the same information mostly overfit. My corpus is a closed personal KB — the missing piece was bringing the right passage into the prompt, which RAG did.”

---

## Soundbites

- “Popular facts are overfitted by the internet; personal facts need retrieval.”  
- “SFT makes an assistant; RAG makes answers checkable.”  
- “I don’t guarantee truth — I guarantee sources, refusals, and evals.”  
- “Same model, RAG off vs on: contains-gold 0.19 → 0.63. That’s the interview.”  
- “I didn’t skip training harder — I refused to confuse duplicated epochs with factuality.”  

---

## How to practice

1. Say the 30-second answer out loud once.  
2. Draw the frontier vs AVIGPT diagram.  
3. Quote v002 vs v004 without looking.  
4. Admit one chat failure (wrong city / invented detail) and the fix (RAG / context-only / no bare “yes”).  
5. End on constraints: free T4, ngrok, versioned reports.
