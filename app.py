import os

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


MODEL_NAME = os.getenv("MODEL_NAME", "avi2135/avigpt2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "80"))
USE_4BIT = os.getenv("USE_4BIT", "1") == "1"

model = None
tokenizer = None


def load_model_and_tokenizer() -> None:
    global model, tokenizer

    if model is not None and tokenizer is not None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-enabled GPU is required for this quantized model. "
            "Run the container with GPU support, for example: docker run --gpus all ..."
        )

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


@app.on_event("startup")
def startup_event() -> None:
    load_model_and_tokenizer()


def generate_response(user_input: str) -> str:
    if not user_input:
        return ""

    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded")

    input_device = next(model.parameters()).device
    inputs = tokenizer(user_input, return_tensors="pt").to(input_device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start_marker = "[/]"
    if start_marker in response:
        response = response.split(start_marker)[-1]
        response = response.split(".")[0]

    return response.strip()


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        answer = generate_response(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )