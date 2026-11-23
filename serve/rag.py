"""CPU-friendly keyword/TF-IDF RAG over local corpus (no GPU needed)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str


class RagIndex:
    def __init__(self, chunks: list[str], sources: list[str]):
        self.chunks = chunks
        self.sources = sources
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self.matrix = self.vectorizer.fit_transform(chunks) if chunks else None

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks or self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        out: list[RetrievedChunk] = []
        for i in idxs:
            if sims[i] <= 0:
                continue
            out.append(
                RetrievedChunk(
                    text=self.chunks[i],
                    score=float(sims[i]),
                    source=self.sources[i],
                )
            )
        return out


_INDEX: RagIndex | None = None


def _split_facts(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_index(corpus_dir: str | Path | None = None) -> RagIndex:
    global _INDEX
    root = Path(corpus_dir or os.getenv("CORPUS_DIR", Path(__file__).resolve().parents[1] / "data" / "corpus"))
    chunks: list[str] = []
    sources: list[str] = []

    facts = root / "facts.txt"
    if facts.exists():
        for i, chunk in enumerate(_split_facts(facts.read_text(encoding="utf-8", errors="ignore"))):
            chunks.append(chunk)
            sources.append(f"facts.txt#{i}")
    else:
        for path in sorted(root.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                chunks.append(text)
                sources.append(path.name)

    _INDEX = RagIndex(chunks, sources)
    return _INDEX


def get_index() -> RagIndex:
    global _INDEX
    if _INDEX is None:
        return build_index()
    return _INDEX


def retrieve_context(query: str, top_k: int = 4) -> tuple[str, list[dict]]:
    hits = get_index().search(query, top_k=top_k)
    if not hits:
        return "", []
    context = "\n\n".join(h.text for h in hits)
    sources = [{"source": h.source, "score": round(h.score, 4), "snippet": h.text[:240]} for h in hits]
    return context, sources
