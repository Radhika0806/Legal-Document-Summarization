# RAG (Retrieval Augmented Generation) -- Concept Guide

## What is RAG? (Explain it like you're talking to a friend)

Imagine you're a lawyer. A client asks: "What did the court say about breach of contract in case X?"

**Without RAG:** You'd need to have memorized every single case. Impossible.

**With RAG:** You go to your filing cabinet, search for relevant cases, pull out the right files, read them quickly, and THEN answer the question using what you just read.

RAG works the same way for AI:
1. RETRIEVE -- search a database for relevant documents
2. AUGMENT -- attach those documents to the AI's prompt as context
3. GENERATE -- the AI reads the context and generates an answer

## Why RAG instead of fine-tuning?

| Approach | What it does | Downside |
|----------|-------------|----------|
| Fine-tuning | Retrains the model on your data | Expensive, slow, data gets "baked in" and can't be updated easily |
| RAG | Searches your data at query time and feeds it to the model | Cheaper, updatable, no retraining needed |

**Interview answer:** "Fine-tuning is like memorizing a textbook. RAG is like having the textbook open during the exam. For legal documents that change frequently, RAG is more practical."

## Key Components You Must Understand

### 1. Embeddings
- Text converted into numerical vectors (lists of numbers)
- Similar texts produce similar vectors
- Example: "breach of contract" and "contract violation" will have vectors close to each other
- Model used: `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality, 384 dimensions)

**Interview Q:** "What are embeddings?"
**Answer:** "Dense vector representations of text that capture semantic meaning. Unlike TF-IDF which matches exact words, embeddings understand that 'happy' and 'joyful' mean similar things."

### 2. Vector Database (ChromaDB)
- A database optimized for storing and searching vectors
- You store document chunks as vectors
- When user asks a question, their question is also converted to a vector
- The database finds the closest matching document chunks

**Why ChromaDB?** Lightweight, runs locally (no cloud needed), open source, Python-native.
**Alternatives:** Pinecone (cloud, paid), FAISS (Facebook, no metadata), Weaviate (cloud).

**Interview Q:** "Why not just use a regular database?"
**Answer:** "SQL databases search by exact keyword match. Vector databases search by meaning. If I search 'termination of employment', a vector DB will also find documents about 'firing an employee' -- a SQL WHERE clause won't."

### 3. Chunking
- Legal documents can be 10,000+ words
- LLMs have context limits (4K-128K tokens)
- So we split documents into smaller "chunks" (e.g., 512 tokens each)
- Overlap between chunks (e.g., 50 tokens) prevents losing context at boundaries

**Interview Q:** "How do you decide chunk size?"
**Answer:** "Too small and you lose context. Too large and the search becomes imprecise. 512 tokens with 50-token overlap is a good balance for legal text. I tested different sizes and measured retrieval quality."

### 4. Retrieval (Similarity Search)
- User asks a question
- Question is embedded into a vector
- ChromaDB finds the top-K most similar chunks using cosine similarity
- Those chunks become the "context" for the LLM

**Interview Q:** "What is cosine similarity?"
**Answer:** "It measures the angle between two vectors. If two vectors point in the same direction (similar meaning), cosine similarity is close to 1. If they're unrelated, it's close to 0."

### 5. Generation (LLM answering with context)
- The retrieved chunks are inserted into a prompt template
- The LLM reads the chunks and generates an answer grounded in the actual documents
- This prevents hallucination -- the AI only answers based on what's in the documents

## Architecture Diagram (draw this in interviews)

```
User Question
      |
      v
[Embedding Model] --> question vector
      |
      v
[ChromaDB] --> finds top 5 matching document chunks
      |
      v
[Prompt Template]
  "Based on these documents: {chunks}
   Answer this question: {question}"
      |
      v
[LLM (BART / any model)] --> generates answer
      |
      v
Answer displayed to user
```

## Common Interview Questions

1. "What happens if the relevant info spans two chunks?"
   --> "That's why we use overlapping chunks. The 50-token overlap ensures boundary information appears in both chunks."

2. "How do you evaluate RAG quality?"
   --> "I check retrieval quality (are the right chunks being found?) and generation quality (is the answer accurate?). Metrics: precision@k for retrieval, ROUGE/human eval for generation."

3. "What if the answer isn't in any document?"
   --> "The prompt instructs the model to say 'I could not find relevant information' rather than hallucinate. This is a key safety feature."

4. "Can RAG work with other types of documents?"
   --> "Yes. The same pipeline works for PDFs, contracts, medical records, research papers -- you just need to adjust the chunking strategy for each document type."

## How This Connects to Your Existing Project

Your project already has:
- BART model fine-tuned for summarization
- 100 legal judgement documents
- A Streamlit app

What we're adding:
- ChromaDB to store all 100 documents as searchable vectors
- An embedding model to convert text to vectors
- A Q&A interface where users can ask questions across ALL documents
- The existing BART model still handles summarization; RAG handles Q&A
