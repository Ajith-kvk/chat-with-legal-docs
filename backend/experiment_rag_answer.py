import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Load ANTHROPIC_API_KEY from .env into the environment
load_dotenv()

# Wipe any previous test database so re-running this script doesn't
# add duplicate copies of the same chunks on top of old ones.
shutil.rmtree("./chroma_experiment", ignore_errors=True)

# --- Steps 1-3: same as before (load, chunk, embed, store) ---
loader = TextLoader("sample_docs/lease.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_experiment",
)

# --- Step 4: retrieve the relevant chunks for a question ---
question = "When can the tenant end the lease early, and what does it cost?"
results = vectorstore.similarity_search(question, k=2)

# --- Step 5: number the chunks so we can ask Claude to cite them ---
numbered_context = ""
for i, doc in enumerate(results, start=1):
    numbered_context += f"[{i}] {doc.page_content}\n\n"

print("=== Context being sent to Claude ===")
print(numbered_context)

# --- Step 6: build the prompt and call Claude ---
SYSTEM_PROMPT = """You are a legal-document assistant. Answer ONLY using the
numbered excerpts provided. Cite every claim with a bracket number like [1].
If the excerpts don't contain the answer, say so clearly instead of guessing."""

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, max_output_tokens=2048)

response = llm.invoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Excerpts:\n\n{numbered_context}\nQuestion: {question}"},
])

print("=== Gemini's answer ===")
if isinstance(response.content, str):
    answer_text = response.content
else:
    # Some providers (like Gemini) return content as a list of blocks
    # instead of a plain string. Pull out just the text parts.
    answer_text = "".join(
        block["text"] for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
print(answer_text)