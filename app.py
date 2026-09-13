"""
app.py — A small Streamlit web app over the RAG system.

Run with:  streamlit run app.py

This file is deliberately thin: all the logic lives in rag.py, and the database
was built once by ingest.py. The app just reads the database and renders a UI.
"""
import streamlit as st

from rag import ask_structured

st.set_page_config(page_title="Document Q&A", page_icon="📄")

st.title("📄 Document Q&A")
st.caption("Ask questions about your documents — answers are grounded in the source text.")

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Thinking..."):
        answer, sources = ask_structured(question)
    st.write(answer.answer)
    st.caption(f"Confidence: {answer.confidence:.0%}")
    st.caption("Sources: " + ", ".join(sources))
