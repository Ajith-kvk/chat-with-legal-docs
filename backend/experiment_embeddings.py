from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Step 1: load and chunk (same as Phase 2) ---
loader = TextLoader("sample_docs/lease.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks\n")

# --- Step 2: load an embedding model ---
# This downloads a small model the first time, then caches it locally.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Let's actually LOOK at what an embedding is, before hiding it inside Chroma
sample_vector = embeddings.embed_query("When can the tenant end the lease?")
print(f"An embedding is a list of {len(sample_vector)} numbers.")
print(f"First 8 numbers: {sample_vector[:8]}")
print()

# --- Step 3: store all chunks in a Chroma vector database ---
# persist_directory means it saves to disk, not just memory
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_experiment",
)
print("Stored all chunks in ChromaDB.\n")

# --- Step 4: ask a question and see which chunks are "closest" in meaning ---
question = "When can the tenant end the lease?"
results = vectorstore.similarity_search(question, k=2)  # k=2 -> top 2 matches

print(f'Question: "{question}"')
print(f"Top {len(results)} matching chunks:\n")
for i, doc in enumerate(results, start=1):
    print(f"--- Match {i} ---")
    print(doc.page_content)
    print()