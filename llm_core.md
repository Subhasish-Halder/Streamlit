# README — Phi-1.5 CPU Helper (Health-info demo)

This module provides two tiny, production-ish utilities to **load Microsoft’s `phi-1_5`** model once (CPU-only) and **generate answers** with a built-in, safety-focused system prompt for general health information.

---

## What’s in this file?

### 1) Constants & globals

* `MODEL_ID = "microsoft/phi-1_5"`
  The official Hugging Face model slug we’ll download on first use.

* `_tokenizer`, `_model`
  Module-level singletons (initialized to `None`). After the first load, they’re kept in memory so subsequent calls don’t re-download or re-instantiate the model.

### 2) `get_model_and_tokenizer()`

**Purpose:** Load and cache the tokenizer + model exactly once (singleton pattern).

**How it works:**

* If both `_tokenizer` and `_model` already exist, it returns them immediately (fast path).
* Otherwise:

  * Downloads/loads the tokenizer via `AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)`.
  * Downloads/loads the model via:

    ```python
    AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,      # forces CPU-friendly precision
        low_cpu_mem_usage=True,         # streaming weights to reduce RAM spikes
        trust_remote_code=True          # allow custom modeling code from repo
    )
    ```
  * Puts the model in eval mode: `_model.eval()`
* Returns both objects.

**Why `torch.float32`?**
For CPU inference, FP32 is the safe default. (Quantized or half-precision variants can be faster/smaller but are not covered here.)

**Why `low_cpu_mem_usage=True`?**
Helps avoid peak-RAM spikes during loading on CPU-only machines.

**About `trust_remote_code=True`:**
Some HF repos ship custom modeling code; this flag allows executing that code. Only enable for **trusted** sources (this repo is from Microsoft).

### 3) `SYSTEM_PREFIX`

A **safety/guard-rail instruction** prepended to every prompt:

> “You are an AI assistant that provides **general health information only**… For emergencies, advise users to contact a licensed medical professional…”

This steers the model away from diagnosis/treatment advice and frames outputs as educational.

### 4) `generate_answer(question: str, max_new_tokens: int = 256) -> str`

**Purpose:** Build a safe prompt, generate a completion, clean it up, and append a disclaimer.

**Steps:**

1. **Get cached objects**

   ```python
   tokenizer, model = get_model_and_tokenizer()
   ```

2. **Compose the full prompt**

   ```
   SYSTEM_PREFIX
   "Answer the following question clearly and concisely:\n\n"
   + question.strip()
   ```

3. **Tokenize**

   ```python
   inputs = tokenizer(prompt, return_tensors="pt")
   ```

4. **Generate (no gradients, CPU)**

   ```python
   with torch.no_grad():
       output_ids = model.generate(
           **inputs,
           max_new_tokens=max_new_tokens,   # length budget for new tokens
           do_sample=True,                  # use sampling (non-deterministic)
           top_k=50, top_p=0.9,             # nucleus/top-k sampling
           temperature=0.7,                 # smooth randomness
           eos_token_id=tokenizer.eos_token_id,  # stop when EOS produced
           pad_token_id=tokenizer.eos_token_id   # avoid pad-token warnings
       )[0]
   ```

   * **Sampling vs. greedy:** sampling yields more natural variation. For consistent outputs, you could set `do_sample=False`.

5. **Decode to text**

   ```python
   text = tokenizer.decode(output_ids, skip_special_tokens=True)
   ```

6. **Remove prompt echo (if model copies it)**
   If the returned text starts with `SYSTEM_PREFIX`, strip it.

7. **Strip unwanted markers**
   The loop removes certain substrings if present (e.g., “(end of text)”).
   *(Note: the loop includes an empty string `""`; because of the `if idx != -1 and marker` guard, only non-empty markers are removed.)*

8. **Attach a footer disclaimer**
   A final reminder that this is **not** medical advice:

   ```
   — Disclaimer: This is general information, not medical advice. 
   Consult a qualified clinician for diagnosis or treatment.
   ```

---

## Quick start

```bash
pip install torch transformers --upgrade
```

```python
from your_module import generate_answer

print(generate_answer("What are common symptoms of dehydration?"))
```

On first run, the tokenizer and model weights are downloaded from Hugging Face to your local cache (e.g., `~/.cache/huggingface/`).

---

## Design choices & rationale

* **CPU-only defaults**
  Many environments (servers, laptops, CI) don’t have GPUs. For simplicity and reliability, this code runs on CPU with FP32. If you later enable GPU:

  ```python
  model.to("cuda")
  inputs = {k: v.to("cuda") for k, v in inputs.items()}
  ```

  and consider half precision (`torch.float16` or `bfloat16`) if the model supports it.

* **Singleton pattern**
  Loading LLMs is expensive. Keeping a module-level `_model` and `_tokenizer` avoids repeated downloads/initializations per request in web apps (e.g., FastAPI/Streamlit).

* **Safe prompting**
  The `SYSTEM_PREFIX` plus the disclaimer at the end put clear guardrails around health content.

* **Stopping behavior**
  `eos_token_id` is set so generation halts when EOS is produced, preventing run-on texts.

* **Sampling knobs**

  * `temperature`: Controls randomness. Lower (e.g., `0.2`) = more focused, higher (e.g., `1.0`) = more creative.
  * `top_k`/`top_p`: Constrain the candidate pool; good defaults here are `top_k=50`, `top_p=0.9`.

---

## Common tweaks

* **Make outputs deterministic**

  ```python
  torch.manual_seed(0)
  output_ids = model.generate(
      **inputs, do_sample=False, num_beams=1, max_new_tokens=256,
      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
  )[0]
  ```

* **Shorter/longer answers**
  Increase/decrease `max_new_tokens`.

* **Stronger safety tone**
  Edit `SYSTEM_PREFIX` to be stricter (e.g., explicitly refuse conditions-specific instructions).

* **Add rate limiting / timeouts**
  For web apps, wrap `generate_answer` with request timeouts to prevent slow requests from piling up.

---

## Example: minimal Streamlit wrapper

```python
import streamlit as st
from helper_phi_cpu import generate_answer

st.title("Health Info (Phi-1.5 on CPU)")
q = st.text_area("Ask a health-related question")
if st.button("Answer") and q.strip():
    st.write(generate_answer(q))
```

> If VS Code or your app crashes after the first model download:
>
> * Make sure you have sufficient RAM and disk space.
> * Kill stray Python processes that may be holding the model in memory.
> * On Windows, long paths/firewall/AV can occasionally interfere with HF cache writes—try running as admin or moving the cache via `HF_HOME`.
> * Avoid loading the model multiple times across modules; reuse the singleton.

---

## Troubleshooting

* **It downloads every time I run the app.**
  Ensure you’re not starting a fresh process per request. In FastAPI/Streamlit, import this module once at startup so the singletons persist.

* **`pad_token_id` warning.**
  We set `pad_token_id=eos_token_id` during `generate` to avoid common warnings for causal LMs.

* **Slow responses on CPU.**
  That’s expected for larger LMs. Consider:

  * Smaller `max_new_tokens`
  * Lower `temperature` (often shorter outputs)
  * Quantized CPU model variants (e.g., `bitsandbytes` or GGUF via other runtimes)
  * Moving to GPU if available

* **Security note on `trust_remote_code=True`.**
  Only use with reputable, verified model repos. It executes repository-provided Python.

---

## API reference (tiny)

* `get_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForCausalLM]`
  Loads/caches tokenizer & model; returns both.

* `generate_answer(question: str, max_new_tokens: int = 256) -> str`
  Generates a safety-scoped, concise answer and appends a disclaimer.

---

## License & attribution

* Model weights: follow the license of `microsoft/phi-1_5` on Hugging Face.
* This helper code is MIT-style “do-whatever-you-want,” but please keep the safety prompt and disclaimer (or strengthen them) if you remain in the health domain.
