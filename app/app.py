import os
import streamlit as st
import torch
from transformers import pipeline
from rag_pipeline import build_vector_store, retrieve, format_context


# --- Page Config ---
st.set_page_config(page_title="Legal Document AI", layout="wide")


# --- Load Models (cached so they only load once) ---
@st.cache_resource
def load_summarizer():
    """Load the fine-tuned BART summarization model."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="./saved_model", device=device)


@st.cache_resource
def load_rag_index():
    """
    Build the RAG vector store from legal documents.
    This reads all judgement files, chunks them, embeds them,
    and stores them in ChromaDB for fast similarity search.
    """
    doc_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "test-data", "judgement"
    )
    persist_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "chroma_db"
    )
    collection, model = build_vector_store(doc_dir, persist_dir=persist_dir)
    return collection, model


# --- App Header ---
st.title("Legal Document AI")
st.write("Summarize legal text or ask questions across 100 court judgements using RAG.")

# --- Tabs for two features ---
tab1, tab2 = st.tabs(["Summarizer", "Q&A (RAG)"])


# ============================
# TAB 1: Summarization
# ============================
with tab1:
    st.header("Document Summarizer")
    st.write("Paste a legal document below to generate a concise summary using the fine-tuned BART model.")

    text = st.text_area(
        "Legal Text",
        height=300,
        placeholder="Paste your legal text here...",
        key="summarize_input",
    )

    if st.button("Summarize", key="btn_summarize"):
        if not text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                summarizer = load_summarizer()
                result = summarizer(
                    text, max_length=256, min_length=30, do_sample=False
                )
                summary = result[0]["summary_text"]
            st.subheader("Summary")
            st.write(summary)


# ============================
# TAB 2: RAG-based Q&A
# ============================
with tab2:
    st.header("Ask Questions Across Legal Documents")
    st.write(
        "This feature uses **RAG (Retrieval Augmented Generation)** to search "
        "through 100 court judgements and answer your question based on the "
        "most relevant passages found."
    )

    # Show how RAG works (expandable)
    with st.expander("How does RAG work?"):
        st.markdown("""
        **Step 1 - Retrieve:** Your question is converted into a vector (embedding).
        ChromaDB searches for the 5 most similar document chunks using cosine similarity.

        **Step 2 - Augment:** The retrieved chunks are combined into a context block.

        **Step 3 - Generate:** The context + your question are fed to the BART model,
        which generates an answer grounded in the actual legal documents.

        This approach prevents hallucination -- the AI only answers based on real documents.
        """)

    question = st.text_input(
        "Your Question",
        placeholder="e.g., What did the court rule about breach of contract?",
        key="rag_question",
    )

    if st.button("Search & Answer", key="btn_rag"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            # Step 1: Build/load the vector index
            with st.spinner("Loading document index (first time may take a minute)..."):
                collection, embed_model = load_rag_index()

            # Step 2: Retrieve relevant chunks
            with st.spinner("Searching documents..."):
                retrieved = retrieve(question, collection, embed_model)

            # Step 3: Show retrieved sources
            st.subheader("Retrieved Passages")
            for i, chunk in enumerate(retrieved, 1):
                with st.expander(
                    f"Source: {chunk['source']} (relevance: {1 - chunk['distance']:.2%})"
                ):
                    st.text(chunk["text"][:1000])

            # Step 4: Generate answer using BART
            with st.spinner("Generating answer..."):
                context = format_context(retrieved)
                prompt = (
                    f"Based on the following legal documents:\n\n"
                    f"{context}\n\n"
                    f"Answer this question: {question}"
                )
                # Use BART summarizer to condense the context into an answer
                summarizer = load_summarizer()
                # Truncate prompt if too long for BART (max 1024 tokens)
                max_chars = 3000
                if len(prompt) > max_chars:
                    prompt = prompt[:max_chars]
                result = summarizer(
                    prompt, max_length=256, min_length=30, do_sample=False
                )
                answer = result[0]["summary_text"]

            st.subheader("Answer")
            st.write(answer)

            # Show sources used
            sources = list(set(c["source"] for c in retrieved))
            st.caption(f"Based on: {', '.join(sources)}")
