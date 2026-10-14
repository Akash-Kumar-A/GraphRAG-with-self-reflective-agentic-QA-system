"""Thin Ollama client: raw one-token completions (the 'voters'), full completions (plain RAG) and embeddings."""
import requests, concurrent.futures as cf

class Ollama:
    def __init__(s, host="http://localhost:11434", model="llama3.2:3b", embed_model="nomic-embed-text", workers=8):
        s.host, s.model, s.embed_model, s.workers = host.rstrip("/"), model, embed_model, workers
        s.http = requests.Session()
    def _post(s, path, payload, timeout=300):
        r = s.http.post(s.host + path, json=payload, timeout=timeout); r.raise_for_status(); return r.json()
    def models(s):
        r = s.http.get(s.host + "/api/tags", timeout=10); r.raise_for_status(); return [m["name"] for m in r.json().get("models", [])]
    def _gen(s, prompt, n, stop=None):
        o = {"temperature": 0.3, "seed": 0, "num_predict": n}
        if stop: o["stop"] = stop
        return s._post("/api/generate", {"model": s.model, "prompt": prompt, "raw": True, "stream": False, "options": o}).get("response", "")
    def next_piece(s, prompt): return s._gen(prompt, 1)                       # ONE token, greedy
    def next_pieces(s, prompts):                                              # all voters in parallel
        with cf.ThreadPoolExecutor(s.workers) as ex: return list(ex.map(s.next_piece, prompts))
    def complete(s, prompt, max_tokens=48): return s._gen(prompt, max_tokens, stop=["\n"]).strip()
    def embed(s, texts):
        out = []
        for i in range(0, len(texts), 32):
            b = texts[i:i + 32]
            try: out += s._post("/api/embed", {"model": s.embed_model, "input": b})["embeddings"]
            except Exception: out += [s._post("/api/embeddings", {"model": s.embed_model, "prompt": t})["embedding"] for t in b]
        return out
