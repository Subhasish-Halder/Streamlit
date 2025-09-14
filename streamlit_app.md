# README — Streamlit App for “AI Health Assistance” (Phi-1.5)

This document explains the `streamlit_app.py` file, which provides a **Streamlit-based web UI** for your AI health assistant powered by Microsoft’s `phi-1_5` model.

The goal: a simple browser app where a user can type a health-related question and receive a concise, safety-scoped answer with a disclaimer.

---

## How to run

```bash
streamlit run streamlit_app.py
```

Streamlit will launch a local server and open your default browser with the app.

---

## File breakdown

```python
import streamlit as st
from llm_core import generate_answer
```

* **streamlit**: Python library for building interactive web apps with minimal code.
* **generate\_answer**: your helper function (defined in `llm_core.py`) that:

  * Loads/caches the Phi-1.5 model and tokenizer
  * Wraps input with a **safety system prompt**
  * Generates a CPU-friendly answer
  * Cleans and appends a disclaimer

---

### Page configuration

```python
st.set_page_config(
    page_title="AI Health Assistance 🏥",
    page_icon="🧑‍⚕️",
    layout="centered"
)
```

* **page\_title**: browser tab title
* **page\_icon**: favicon (emoji doctor icon 🧑‍⚕️)
* **layout="centered"**: centers the app content instead of wide layout

---

### App header

```python
st.title("I am your AI Health Assistance 🏥")
st.caption("Ask general health related questions to the AI Bot.")
```

* **title**: main heading on the page
* **caption**: short explanatory subheading

---

### Form for Q\&A

```python
with st.form("qa_form"):
    question = st.text_input("question", placeholder="What are the uses of Paracetamol tablets?")
    submitted = st.form_submit_button("Submit")
```

* A **form block** named `"qa_form"` groups the question input and submit button.
* **st.text\_input**:

  * Label: `"question"`
  * Placeholder: `"What are the uses of Paracetamol tablets?"`
* **st.form\_submit\_button**: triggers processing when clicked.

---

### Handling submission

```python
if submitted:
    if question.strip():
        with st.spinner("Thinking..."):
            answer = generate_answer(question)
        st.text_area("output", answer, height=320)
    else:
        st.warning("Please enter a question.")
```

* When **submitted**:

  * If question is not empty:

    * Shows a temporary spinner (“Thinking…”)
    * Calls `generate_answer(question)`
    * Displays the output in a **text\_area** box (multiline, height=320 px)
  * If empty:

    * Displays a warning prompt

---

### Clear button

```python
st.button("Clear", on_click=lambda: st.experimental_rerun())
```

* Adds a button labeled **Clear**.
* When clicked, it **reruns the script**, effectively resetting the form and clearing inputs/outputs.

---

## Request flow

1. User types a question into the **text\_input**.
2. Presses **Submit**.
3. `generate_answer(question)` is invoked.

   * Uses the cached Phi-1.5 model (singleton in `llm_core.py`).
   * Prepends the **system safety prefix**.
   * Generates an answer with sampling settings.
   * Strips echoes/unwanted markers.
   * Appends a **health disclaimer**.
4. Answer is shown in a **text\_area**.
5. Clicking **Clear** resets the app state.

---

## Dependencies

Install required packages:

```bash
pip install streamlit torch transformers
```

Also ensure `llm_core.py` (your helper) is in the same folder or on `PYTHONPATH`.

---

## Configuration & customization

* **Change placeholder example**
  In `st.text_input`, replace the `placeholder` string with another sample question.
* **Adjust output height**
  Change `height=320` in `st.text_area`.
* **Styling/layout**
  Replace `layout="centered"` with `"wide"` in `st.set_page_config` for wider content.
* **Persistent state**
  Use `st.session_state` to store Q\&A history if you want chat-like interactions.

---

## Example usage

```bash
streamlit run streamlit_app.py
# Browser opens automatically
# Type: "What are common causes of fever?"
# Click Submit → get answer + disclaimer
# Click Clear → reset the form
```

---

## Security & compliance

* The backend (`llm_core.py`) already enforces **guardrails** with:

  * A **system prefix** instructing “general health information only”
  * A **footer disclaimer** reminding users to consult clinicians
* Keep these intact (or strengthen them) if deploying beyond localhost.
* If exposing the app on the internet:

  * Consider `stauth` or a reverse proxy for authentication.
  * Add a Terms/Disclaimer section at the top for clarity.
  * Ensure logs do not store personally identifiable or sensitive health data.

---

## Summary

This Streamlit app provides a **simple, interactive UI** for your health assistant demo. Users can submit general health questions, see safe model-generated answers, and reset the interface with one click. It’s designed to run locally but can be shared on internal servers for demos.

It’s a clean wrapper around your `generate_answer()` helper, ensuring responses remain **informational, not diagnostic**.
