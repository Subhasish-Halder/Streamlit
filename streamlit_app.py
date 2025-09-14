"""
Streamlit app for AI Health Assistance (Phi-1.5).
Run:  streamlit run streamlit_app.py
"""

import streamlit as st
from llm_core import generate_answer

st.set_page_config(
    page_title="AI Health Assistance 🏥",
    page_icon="🧑‍⚕️",
    layout="centered"
)

st.title("I am your AI Health Assistance 🏥")
st.caption("Ask general health related questions to the AI Bot.")

with st.form("qa_form"):
    question = st.text_input("question", placeholder="What are the uses of Paracetamol tablets?")
    submitted = st.form_submit_button("Submit")
    if submitted:
        if question.strip():
            with st.spinner("Thinking..."):
                answer = generate_answer(question)
            st.text_area("output", answer, height=320)
        else:
            st.warning("Please enter a question.")

st.button("Clear", on_click=lambda: st.experimental_rerun())
