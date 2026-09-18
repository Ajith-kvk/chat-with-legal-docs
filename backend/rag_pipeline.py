"""
Core RAG logic, kept separate from FastAPI so it doesn't know anything
about HTTP. main.py calls these functions; these functions don't know
main.py exists.
"""

import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

CHROMA_DIR = "./chroma_db"
UPLOAD_DIR = Path("./uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

# These are created ONCE, when this module is first imported — not on
# every request. Loading the embedding model is slow; we don't want to
# pay that cost for every single API call.
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
_vectorstore = Chroma(
    collection_name="legal_docs",
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)
_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
_llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0, max_output_tokens=2048)

SYSTEM_PROMPT = """You are a legal-document assistant. Answer ONLY using the
numbered excerpts provided. Cite every claim with a bracket number like [1].
If the excerpts don't contain the answer, say so clearly instead of guessing."""


def _call_llm_with_retry(messages, max_attempts=3):
    """Retry the LLM call with increasing delays if it fails transiently
    (rate limits, temporary overload). This is a standard resilience
    pattern called 'exponential backoff' — wait longer after each
    failure, since immediately retrying a rate-limited request just
    gets rate-limited again."""
    delay = 5  # seconds
    for attempt in range(1, max_attempts + 1):
        try:
            return _llm.invoke(messages)
        except Exception as e:
            if attempt == max_attempts:
                raise  # out of retries, let the caller's except block handle it
            print(f"LLM call failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # back off further each time: 5s, 10s, 20s...


def ingest_document(file_path: str, filename: str) -> dict:
    """Load a file from disk, chunk it, embed it, and store it. Returns metadata."""
    # If a document with this exact filename already exists, remove it
    # first — otherwise re-uploading the same file just piles up
    # duplicate copies in the vector store.
    for existing in list_documents():
        if existing["filename"] == filename:
            delete_document(existing["doc_id"])

    doc_id = str(uuid.uuid4())  # a unique ID so we can find/delete this doc later
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    raw_docs = loader.load()
    chunks = _splitter.split_documents(raw_docs)

    if not chunks:
        raise ValueError("No text could be extracted from this file.")

    # Tag every chunk with which document and filename it came from —
    # this is what lets us show citations back to the right source later.
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["filename"] = filename
        # PyPDFLoader numbers pages starting at 0; humans expect page 1,
        # page 2, etc., so we shift it by one for display purposes.
        if "page" in chunk.metadata and chunk.metadata["page"] is not None:
            chunk.metadata["page"] = int(chunk.metadata["page"]) + 1

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    _vectorstore.add_documents(chunks, ids=ids)

    return {"doc_id": doc_id, "filename": filename, "num_chunks": len(chunks)}


def list_documents() -> list[dict]:
    """Rebuild a document list by looking at what's stored in Chroma."""
    data = _vectorstore.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in data["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": meta["filename"], "num_chunks": 0}
        seen[doc_id]["num_chunks"] += 1
    return list(seen.values())


def delete_document(doc_id: str) -> None:
    _vectorstore.delete(where={"doc_id": doc_id})


def answer_question(query: str, k: int = 4) -> dict:
    """Retrieve relevant chunks, then ask the LLM to answer using only those."""
    results = _vectorstore.similarity_search(query, k=k)

    if not results:
        return {"answer": "No documents have been uploaded yet.", "sources": []}

    numbered_context = ""
    sources = []
    for i, doc in enumerate(results, start=1):
        numbered_context += f"[{i}] {doc.page_content}\n\n"
        sources.append({
            "label": f"[{i}]",
            "filename": doc.metadata["filename"],
            "page": doc.metadata.get("page"),  # None for .txt/.md files, a number for PDFs
            "snippet": doc.page_content[:300],
        })

    try:
        response = _call_llm_with_retry([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{numbered_context}\nQuestion: {query}"},
        ])
    except Exception as e:
        # The LLM provider's servers can be temporarily overloaded or
        # unreachable. Don't crash the whole request — return a clear,
        # honest message so the frontend can show it to the user.
        return {
            "answer": f"The AI service is temporarily unavailable ({e.__class__.__name__}). "
                      "Please try asking again in a moment.",
            "sources": [],
        }

    if isinstance(response.content, str):
        answer_text = response.content
    else:
        answer_text = "".join(
            block["text"] for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    return {"answer": answer_text, "sources": sources}