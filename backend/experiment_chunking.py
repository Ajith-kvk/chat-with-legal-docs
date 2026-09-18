from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load the raw document into memory
loader = TextLoader("sample_docs/lease.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print("---")
print(documents[0].page_content[:200])  # peek at the first 200 characters
print("---")

# Step 2: Split it into overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)

print(f"\nSplit into {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
    print()