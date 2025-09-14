"""
Gradio interface for AI Health Assistance (Phi-1.5).
Run:  python gradio_app.py
"""

import gradio as gr
from llm_core import generate_answer


def infer(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."
    return generate_answer(question)


if __name__ == "__main__":
    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(label="question", placeholder="What are the uses of Paracetamol tablets?"),
        outputs=gr.Textbox(label="output"),
        title="I am your AI Health Assistance 🏥",
        description="Ask general health-related questions to the AI Bot.",
        allow_flagging="never"
    )
    demo.launch()
