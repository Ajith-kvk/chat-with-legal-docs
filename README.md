# Chat with Legal Docs

A full-stack RAG (Retrieval-Augmented Generation) application. Upload a legal
document — a lease, an NDA, a terms-of-service PDF — and ask questions about
it in plain English. Answers are generated only from the uploaded document's
actual text, with citations back to the exact passage and page.

## How it works
## How it works

document (.pdf/.txt/.md)
│ PyPDFLoader / TextLoader
▼
raw text
│ RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
▼
chunks + metadata (filename, page number, doc_id)
│ sentence-transformers embeddings (local, free)
▼
ChromaDB (persisted vector store)

question ──► similarity search (top-k chunks) ──► Gemini (grounded prompt) ──► answer + [n] citations


## Stack

- **Backend:** Python, FastAPI, LangChain (document loaders, text splitter,
  Chroma vector store), ChromaDB, `sentence-transformers` for embeddings,
  Google Gemini for answer generation.
- **Frontend:** React (Vite), plain CSS.

## Setup

Requires Python 3.10+ and Node 18+.

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

python -m pip install -r requirements.txt
```

Create `backend/.env`:

GOOGLE_API_KEY=your-key-here

Get a free key at https://aistudio.google.com/apikey — no credit card required.

Run it:
```bash
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env`:

VITE_API_BASE_URL=http://localhost:8000/api


Run it:
```bash
npm run dev
```

Open http://localhost:5173. Both servers need to be running at the same time.

## Design decisions worth explaining

- **Embeddings run locally** (`sentence-transformers`) so document ingestion
  is free and works offline — only the final answer-generation step calls
  an external API.
- **Citations are structural, not decorative.** Every retrieved chunk keeps
  its filename and page number, gets numbered before being sent to the LLM,
  and the model is instructed to cite `[1]`, `[2]`, etc. — so citations in
  the UI point back to real, verifiable source text, not the model's
  paraphrase of where it got something.
- **Re-uploading a file replaces it** rather than duplicating it in the
  vector store, keyed by filename.
- **LLM calls retry with exponential backoff** and fail gracefully with a
  clear message if the provider is rate-limited or temporarily unavailable,
  rather than crashing the request.

## What I'd add next

- Structure-aware chunking for legal text (split on numbered clauses
  instead of a fixed character count, so a chunk never cuts an obligation
  in half)
- Conversation memory (the chat endpoint is currently stateless per question)
- Streaming answers instead of waiting for the full response
- Swap ChromaDB for Pinecone to compare a cloud-hosted vector DB