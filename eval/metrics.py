"""Lightweight string metrics for factoid eval (CPU)."""

from __future__ import annotations

import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def exact_match(pred: str, gold: str) -> float:
    return float(_tokens(pred) == _tokens(gold) and bool(_tokens(gold)))


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    g_counts: dict[str, int] = {}
    for t in g:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contains_gold(pred: str, gold: str) -> float:
    p, g = " ".join(_tokens(pred)), " ".join(_tokens(gold))
    if not g:
        return 0.0
    return float(g in p)
