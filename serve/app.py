"""FastAPI app: /ask, /ask/stream, /verify, /health — Colab T4 + tunnel ready."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from serve import model as model_mod
from serve.rag import build_index, retrieve_context
from serve.safety import REFUSAL_REPLY, should_refuse

app = FastAPI(title="AVIGPT", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = False
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_k: int = 4


class VerifyRequest(BaseModel):
    claim: str
    top_k: int = 4


@app.on_event("startup")
def startup_event() -> None:
    build_index()
    model_mod.load_model_and_tokenizer()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": model_mod.get_load_meta(),
        "vram": model_mod.vram_stats(),
    }


def _history_dicts(req: AskRequest) -> list[dict]:
    return [m.model_dump() for m in req.history]


def _run_ask(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    if should_refuse(req.question):
        return {
            "answer": REFUSAL_REPLY,
            "refused": True,
            "sources": [],
            "latency_s": 0.0,
            "vram": model_mod.vram_stats(),
        }

    sources: list[dict] = []
    context = None
    if req.use_rag:
        context, sources = retrieve_context(req.question, top_k=req.top_k)

    result = model_mod.generate_answer(
        req.question,
        history=_history_dicts(req),
        context=context,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {
        "answer": result["answer"],
        "refused": False,
        "sources": sources,
        "latency_s": result["latency_s"],
        "new_tokens": result["new_tokens"],
        "vram": result["vram"],
        "use_rag": req.use_rag,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    try:
        return _run_ask(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")

    def event_gen():
        try:
            if should_refuse(req.question):
                payload = {"type": "token", "text": REFUSAL_REPLY}
                yield f"data: {json.dumps(payload)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'refused': True, 'sources': []})}\n\n"
                return

            sources: list[dict] = []
            context = None
            if req.use_rag:
                context, sources = retrieve_context(req.question, top_k=req.top_k)
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            for token in model_mod.stream_answer(
                req.question,
                history=_history_dicts(req),
                context=context,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            ):
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'refused': False, 'sources': sources, 'vram': model_mod.vram_stats()})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/verify")
def verify_claim(req: VerifyRequest) -> dict[str, Any]:
    """Tool: check a claim against the local corpus and return evidence."""
    if not req.claim.strip():
        raise HTTPException(status_code=400, detail="claim is empty")

    context, sources = retrieve_context(req.claim, top_k=req.top_k)
    if not sources:
        return {
            "verdict": "unknown",
            "reason": "No relevant documents found in the corpus.",
            "sources": [],
            "trace": ["retrieve → 0 hits → unknown"],
        }

    prompt_q = (
        f"Claim: {req.claim}\n"
        "Based only on the context, answer with exactly one of: SUPPORTED, CONTRADICTED, UNKNOWN. "
        "Then give one short sentence of evidence."
    )
    result = model_mod.generate_answer(
        prompt_q,
        context=context,
        max_new_tokens=96,
        temperature=0.0,
    )
    answer = result["answer"]
    upper = answer.upper()
    if "SUPPORTED" in upper:
        verdict = "supported"
    elif "CONTRADICTED" in upper:
        verdict = "contradicted"
    else:
        verdict = "unknown"

    source_ids = [s.get("source", "?") for s in sources]
    return {
        "verdict": verdict,
        "reason": answer,
        "sources": sources,
        "trace": [
            "tool=verify_claim",
            f"retrieve → top_k={req.top_k} hits={len(sources)}",
            f"sources → {', '.join(source_ids) if source_ids else '(none)'}",
            "prompt → SUPPORTED | CONTRADICTED | UNKNOWN + evidence",
            f"generate → verdict={verdict} tokens≈{result.get('new_tokens')}",
        ],
        "latency_s": result["latency_s"],
        "vram": result["vram"],
    }


def main() -> None:
    import uvicorn

    uvicorn.run(
        "serve.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
