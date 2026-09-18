import "./App.css";
import { useEffect, useState } from "react";

// Read from the environment instead of hardcoding — Vite exposes any
// variable prefixed with VITE_ to browser code via import.meta.env.
const API_BASE = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isAsking, setIsAsking] = useState(false);

  useEffect(() => {
    refreshDocuments();
  }, []);

  async function refreshDocuments() {
    const res = await fetch(`${API_BASE}/documents`);
    setDocuments(await res.json());
  }

  async function handleFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
    await refreshDocuments();
    setIsUploading(false);
    event.target.value = ""; // reset the file input so re-selecting the same file still fires onChange
  }

  async function handleDelete(docId) {
    await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
    await refreshDocuments();
  }

  async function handleAsk(event) {
    event.preventDefault();
    const query = input.trim();
    if (!query) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    setIsAsking(true);

    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();

    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: data.answer ?? "Something went wrong. Please try again.",
        sources: data.sources ?? [],
      },
    ]);
    setIsAsking(false);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>§ Document Library</h1>

        <label className="upload-btn">
          {isUploading ? "Processing…" : "Upload a document"}
          <input type="file" accept=".pdf,.txt,.md" onChange={handleFileChange} disabled={isUploading} hidden />
        </label>

        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.doc_id}>
              <span>{doc.filename}</span>
              <span className="doc-meta">{doc.num_chunks} chunks</span>
              <button className="delete-btn" onClick={() => handleDelete(doc.doc_id)} title="Delete">
                ×
              </button>
            </li>
          ))}
          {documents.length === 0 && <li className="empty">No documents yet.</li>}
        </ul>
      </aside>

      <main className="chat">
        <div className="thread">
          {messages.length === 0 && (
            <p className="placeholder">Ask a question about your uploaded documents.</p>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              <div className="bubble-role">{msg.role === "user" ? "You" : "Assistant"}</div>
              <p>{msg.content}</p>

              {msg.sources?.length > 0 && (
                <div className="citations">
                  {msg.sources.map((s) => (
                    <details key={s.label} className="citation">
                      <summary>
                        <span className="citation-label">{s.label}</span>
                        {s.filename}
                        {s.page ? `, p. ${s.page}` : ""}
                      </summary>
                      <blockquote>{s.snippet}…</blockquote>
                    </details>
                  ))}
                </div>
              )}
            </div>
          ))}

          {isAsking && <p className="thinking">Thinking…</p>}
        </div>

        <form className="composer" onSubmit={handleAsk}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question…"
            disabled={isAsking}
          />
          <button type="submit" disabled={isAsking || !input.trim()}>
            Ask
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;