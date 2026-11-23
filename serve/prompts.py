"""Llama-2 style chat prompting without brittle post-processing hacks."""

from __future__ import annotations

import re

from serve.safety import CONSTITUTION

_JUNK_LINE = re.compile(
    r"(?i)^(keywords?\s*:|<<\s*sys\s*>>|<<\s*syn\s*>>|\[/?inst\]|\[/?html\]|\[/?sys)"
)
_MARKUP = re.compile(
    r"(?i)</?s>|<<\s*/?\s*sys\s*>>|\[/?inst\]|\[/?html\]|\[/?sysy?\]|\[//?\]"
)


def _pairs_from_history(history: list[dict] | None, keep_last: int = 6) -> list[tuple[str, str]]:
    if not history:
        return []
    turns = history[-keep_last:]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.get("role") == "user":
            user = (t.get("content") or "").strip()
            assistant = ""
            if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                assistant = (turns[i + 1].get("content") or "").strip()
                i += 2
            else:
                i += 1
            if user and assistant:
                pairs.append((user, assistant))
        else:
            i += 1
    return pairs


def build_prompt(
    question: str,
    *,
    history: list[dict] | None = None,
    context: str | None = None,
    system: str | None = None,
) -> str:
    sys = system or CONSTITUTION
    if context:
        sys = (
            f"{sys}\n\nUse ONLY the following context when answering. "
            "If the context is insufficient, say you do not know.\n"
            f"<context>\n{context}\n</context>"
        )

    q = question.strip()
    pairs = _pairs_from_history(history)
    if not pairs:
        return f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{q} [/INST]"

    first_u, first_a = pairs[0]
    parts = [f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{first_u} [/INST] {first_a} </s>"]
    for u, a in pairs[1:]:
        parts.append(f"<s>[INST] {u} [/INST] {a} </s>")
    parts.append(f"<s>[INST] {q} [/INST]")
    return "".join(parts)


def extract_assistant_reply(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].replace("</s>", "").strip()
    marker = "[/INST]"
    if marker in full_text:
        return full_text.split(marker)[-1].replace("</s>", "").strip()
    return full_text.replace("</s>", "").strip()


def clean_answer(answer: str, question: str | None = None) -> str:
    """Strip prompt debris and accidental question echo from model output."""
    text = (answer or "").replace("<s>", "").replace("</s>", "").strip()
    text = _MARKUP.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept: list[str] = []
    q = (question or "").strip()
    for ln in lines:
        if _JUNK_LINE.search(ln):
            continue
        if q and ln.rstrip(" ?!.") == q.rstrip(" ?!."):
            continue
        if q and ln.lower().startswith(q.lower()[: min(40, len(q))].lower()) and len(ln) <= len(q) + 8:
            continue
        kept.append(ln)
    text = "\n".join(kept).strip() if kept else text
    if q:
        # Leading echo on first line: "Question?\nAnswer..."
        first, _, rest = text.partition("\n")
        if first.strip().rstrip(" ?!.") == q.rstrip(" ?!.") and rest.strip():
            text = rest.strip()
        elif text.lower().startswith(q.lower()) and len(text) > len(q) + 2:
            text = text[len(q) :].lstrip(" \n:-")
    return text.strip()
