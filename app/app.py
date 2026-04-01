import streamlit as st
import torch
from transformers import pipeline


@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model="./saved_model",
        device=device,
    )
    return summarizer


st.title("Legal Document Summarizer")
st.write("Paste a legal document below and click **Summarize** to generate a concise summary.")

text = st.text_area("Legal Text", height=300, placeholder="Paste your legal text here...")

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Generating summary..."):
            summarizer = load_model()
            result = summarizer(text, max_length=256, min_length=30, do_sample=False)
            summary = result[0]["summary_text"]
        st.subheader("Summary")
        st.write(summary)
