import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import UPLOAD_DIR, answer_question, delete_document, ingest_document, list_documents

app = FastAPI(title="Chat With Legal Docs")

# Tell the browser it's OK for JavaScript running on the Vite dev server
# (a different "origin" from the backend's own port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


# Pydantic models define the SHAPE of request/response JSON.
# FastAPI uses these to validate incoming data automatically —
# if a request doesn't match this shape, it's rejected before your
# function even runs.
class ChatRequest(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def get_documents():
    return list_documents()


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save the uploaded file to disk so our loaders (which expect a
    # file PATH, not raw bytes) can read it.
    dest_path = UPLOAD_DIR / file.filename
    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        result = ingest_document(str(dest_path), file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    delete_document(doc_id)
    return {"status": "deleted"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return answer_question(req.query)