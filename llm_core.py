"""
Shared utilities to load Phi-1.5 and generate answers (CPU only).
"""

from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Official Hugging Face model id
MODEL_ID = "microsoft/phi-1_5"

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def get_model_and_tokenizer():
    """
    Load the tokenizer and model once (singleton pattern).
    Keeps them cached in memory to avoid reloading on every request.
    """
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # CPU friendly
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    _model.eval()
    return _tokenizer, _model


SYSTEM_PREFIX = (
    "You are an AI assistant that provides general health information only. "
    "You do NOT give medical advice or diagnosis. For emergencies, advise users to "
    "contact a licensed medical professional immediately.\n\n"
)


def generate_answer(question: str, max_new_tokens: int = 256) -> str:
    """
    Wrap the user question with an instruction, generate an answer,
    and clean up the output.
    """
    tokenizer, model = get_model_and_tokenizer()
    prompt = SYSTEM_PREFIX + "Answer the following question clearly and concisely:\n\n" + question.strip()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )[0]

    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Remove the prompt echo if present
    if text.startswith(SYSTEM_PREFIX):
        text = text[len(SYSTEM_PREFIX):].lstrip()

    # Remove unwanted markers
    for marker in ("(end of text)", ""):
        idx = text.find(marker)
        if idx != -1 and marker:
            text = text[:idx].strip()

    # Append disclaimer
    safe_footer = (
        "\n\n— Disclaimer: This is general information, not medical advice. "
        "Consult a qualified clinician for diagnosis or treatment."
    )
    return text.strip() + safe_footer
