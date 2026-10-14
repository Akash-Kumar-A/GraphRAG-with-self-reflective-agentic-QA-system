"""ChromaDB store for the confidential records. Embeddings come from Ollama and are passed explicitly (works on any chromadb version)."""
import chromadb

class Store:
    def __init__(s, ollama, path="chroma_db", name="confidential_records"):
        s.ol = ollama; s.col = chromadb.PersistentClient(path=path).get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    def count(s): return s.col.count()
    def reset(s):
        ids = s.col.get()["ids"]
        if ids: s.col.delete(ids=ids)
    def ingest(s, records, batch=64, progress=None):
        for i in range(0, len(records), batch):
            b = records[i:i + batch]
            s.col.add(ids=[f"rec{i + j}" for j in range(len(b))], documents=[r["text"] for r in b], embeddings=s.ol.embed([r["text"] for r in b]),
                      metadatas=[{"topic": r["topic"], "name": r["name"]} for r in b])
            if progress: progress(min(i + batch, len(records)) / len(records))
    def query(s, q, n):
        n = min(n, s.count()); r = s.col.query(query_embeddings=s.ol.embed([q]), n_results=n)
        return r["documents"][0], r["distances"][0]
