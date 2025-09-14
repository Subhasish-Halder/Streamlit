# README — Gradio App for “AI Health Assistance” (Phi-1.5)

This file explains the small Gradio UI that wraps your `generate_answer()` function from `llm_core.py` and exposes it as a simple web app.

---

## What this script does

* Spins up a **local web interface** (via [Gradio](https://www.gradio.app/)) where users can type a **health-related question**.
* For each question, it calls your **LLM helper** (`generate_answer`) which:

  * Loads/caches Microsoft’s `phi-1_5` model and tokenizer (CPU-only in your helper)
  * Prepends a **safety system prompt** (general info only; no medical advice)
  * Generates and returns a concise answer with a **disclaimer** appended
* Shows the model’s response in a textbox.

Run it with:

```bash
python gradio_app.py
```

By default, Gradio prints a local URL (e.g., `http://127.0.0.1:7860`) you can open in your browser.

---

## File layout & key pieces

```python
import gradio as gr
from llm_core import generate_answer
```

* **gradio**: UI library for quick ML demos.
* **generate\_answer**: your CPU helper from `llm_core.py` that does safe prompting + inference.

```python
def infer(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."
    return generate_answer(question)
```

* **infer** is the function Gradio will call when the user submits the form.
* It performs a **simple validation**:

  * If the textbox is empty or only whitespace → returns a friendly message.
  * Otherwise → delegates to `generate_answer(question)`.

```python
demo = gr.Interface(
    fn=infer,
    inputs=gr.Textbox(label="question", placeholder="What are the uses of Paracetamol tablets?"),
    outputs=gr.Textbox(label="output"),
    title="I am your AI Health Assistance 🏥",
    description="Ask general health-related questions to the AI Bot.",
    allow_flagging="never"
)
demo.launch()
```

* **Interface** constructor wires your `infer` function to a basic UI:

  * `inputs`: a single **Textbox** for the question

    * `label`: Shown above the box (“question”)
    * `placeholder`: Example prompt to guide the user
  * `outputs`: a **Textbox** where the model’s answer is displayed
  * `title`: Appears at the top of the page (contains an emoji hospital sign)
  * `description`: One-line explanation below the title
  * `allow_flagging="never"`: Hides Gradio’s default “Flag” button
* **launch()**: starts the local server and opens the app in your default browser (unless disabled).

---

## Prerequisites

Install dependencies (pins optional):

```bash
pip install gradio transformers torch
```

Also ensure your `llm_core.py` (the helper that loads `phi-1_5`) is in the **same directory** or importable on `PYTHONPATH`.

---

## How requests flow

1. User types a question in the **input textbox**.
2. Gradio calls `infer(question)`.
3. `infer` validates the input (non-empty), then calls `generate_answer(question)`.
4. `generate_answer` (from your helper):

   * Ensures the **tokenizer/model** are already loaded (singleton cache)
   * Builds a **safety-scoped prompt**
   * Generates a response (sampling settings)
   * Cleans up echoes/special markers
   * Appends a **disclaimer**
5. The final text is returned and shown in the **output textbox**.

---

## Configuration & useful flags

You can pass options to `launch()`:

```python
demo.launch(
    server_name="0.0.0.0",   # listen on all interfaces
    server_port=7860,        # custom port
    share=False,             # set True to get a public Gradio share URL
    inbrowser=True,          # auto-open browser tab
    auth=("user", "pass")    # basic auth if exposing beyond localhost
)
```

> If you **expose this app** to others, keep the safety prompt and disclaimer in `llm_core.py` intact, or strengthen them. This UI is intended for **general health information**, not diagnosis or treatment.

---

## Customization ideas

* **Rename labels**:

  ```python
  inputs=gr.Textbox(label="Your Question")
  outputs=gr.Textbox(label="AI Answer")
  ```
* **Longer/shorter answers**: add a slider and pass its value to `generate_answer(max_new_tokens=...)`.
* **Streaming output**: swap to a `ChatInterface` or use Gradio’s streaming features if you later add token streaming in your helper.
* **Multi-modal** (future): add file/image inputs; ensure your backend supports them.

---

## Troubleshooting

* **App starts but responses are slow**
  That’s normal for CPU inference. Try reducing tokens or running on a machine with more CPU/RAM. Consider adding `max_new_tokens` controls.
* **Model downloads every time**
  Make sure your `llm_core.get_model_and_tokenizer()` singletons are imported **once** and reused (they are in your current design). Avoid reloading in hot-reloading loops.
* **Emoji in title not showing on Windows console**
  It’s just the console print; the browser UI should render the emoji fine.
* **Firewall prompts**
  Gradio opens a local port; allow it or set a different `server_port`.

---

## Minimal example usage

```bash
python gradio_app.py
# Open the printed URL, e.g., http://127.0.0.1:7860
# Type: "What are common symptoms of dehydration?"
# Read the answer and disclaimer in the output box.
```

---

## Security & compliance notes

* Your backend already uses a **safety system prompt** and a **disclaimer**. Keep both.
* If you share the app publicly:

  * Add **basic auth** or IP restrictions.
  * Log requests/outputs responsibly; avoid storing PHI/PII.
  * Display a visible **Terms/Disclaimer** in the UI description if required by your org.

---

## Summary

This Gradio script is a thin, user-friendly wrapper around your `generate_answer()` helper. It provides a single-textbox interface to ask health-related questions and returns **concise, safety-scoped answers** powered by Phi-1.5, running on CPU by default.
