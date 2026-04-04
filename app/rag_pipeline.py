"""
RAG (Retrieval Augmented Generation) Pipeline for Legal Documents

This module handles:
1. Loading legal documents from text files
2. Chunking them into smaller pieces
3. Storing chunks as embeddings in ChromaDB
4. Retrieving relevant chunks for a user query
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer


# --- Configuration ---
CHUNK_SIZE = 512          # number of characters per chunk
CHUNK_OVERLAP = 50        # overlap between consecutive chunks
TOP_K = 5                 # number of chunks to retrieve per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim embeddings
COLLECTION_NAME = "legal_documents"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a long document into overlapping chunks.

    Why overlapping? If a sentence sits right at the boundary between two
    chunks, the overlap ensures it appears in both, so we don't lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap      # step back by 'overlap' characters
    return chunks


def load_documents(doc_dir):
    """
    Read all .txt files from a directory.
    Returns a list of dicts: [{"filename": "...", "text": "..."}, ...]
    """
    documents = []
    if not os.path.isdir(doc_dir):
        return documents

    for fname in sorted(os.listdir(doc_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(doc_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            documents.append({"filename": fname, "text": text})
    return documents


def build_vector_store(doc_dir, persist_dir=None):
    """
    Main function: load documents, chunk them, embed them, store in ChromaDB.

    Steps:
      1. Load raw documents from disk
      2. Split each document into overlapping chunks
      3. Use SentenceTransformer to create embeddings
      4. Store everything in a ChromaDB collection

    Returns the ChromaDB collection and the embedding model.
    """
    # --- Step 1: Load documents ---
    documents = load_documents(doc_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {doc_dir}")

    # --- Step 2: Chunk ---
    all_chunks = []       # the text of each chunk
    all_metadata = []     # source filename + chunk index
    all_ids = []          # unique ID for each chunk

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": doc["filename"],
                "chunk_index": i,
            })
            all_ids.append(f"{doc['filename']}_chunk_{i}")

    # --- Step 3: Embed ---
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings_list = [emb.tolist() for emb in embeddings]

    # --- Step 4: Store in ChromaDB ---
    if persist_dir:
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.Client()

    # Delete old collection if it exists, then recreate
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Legal court judgement chunks"},
    )

    # ChromaDB has a batch limit; add in batches of 500
    batch_size = 500
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=all_ids[start:end],
            documents=all_chunks[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=all_metadata[start:end],
        )

    return collection, model


def retrieve(query, collection, model, top_k=TOP_K):
    """
    Given a user question, find the most relevant document chunks.

    Steps:
      1. Convert the query to an embedding vector
      2. Ask ChromaDB for the top-K closest chunks (cosine similarity)
      3. Return the chunks with their source filenames
    """
    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return retrieved


def format_context(retrieved_chunks):
    """
    Format retrieved chunks into a single context string
    that can be fed to the LLM as reference material.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source: {chunk['source']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)
