"""Fake Ollama server for plumbing tests (no LLM). Voters copy from the record in the prompt, otherwise use a public trigram LM."""
import json, re, zlib, threading, sys, numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, ".")
from dprag import PublicLM, Doc, tok
D = json.load(open("prepared/data.json", encoding="utf-8")); LM = PublicLM([d["text"] for d in D["public"]] + D["format_text"])
def emb(t):
    v = np.zeros(128)
    for w in tok(t): v[zlib.crc32(w.encode()) % 128] += 1
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()
def nxt(prompt, n):
    q, pre = re.search(r"Question: (.*)\nAnswer in one short sentence:(.*)$", prompt, re.S).groups()
    docs = re.search(r"Records:\n(.*)\n\nQuestion:", prompt, re.S); d = Doc(docs.group(1).split("\n")[0]) if docs else None
    ctx, out = tok(q) + tok(pre), ""
    for _ in range(n):
        bg = tuple(ctx[-2:]); w = (d.idx.get(bg) if d else None) or LM.next(bg); ctx.append(w)
        out += w if w in ",." else " " + w
        if w == ".": break
    return out
class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def _send(s, o): b = json.dumps(o).encode(); s.send_response(200); s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s): s._send({"models": [{"name": "mock:latest"}, {"name": "nomic-embed-text:latest"}]})
    def do_POST(s):
        p = json.loads(s.rfile.read(int(s.headers["Content-Length"])))
        if s.path == "/api/embed": s._send({"embeddings": [emb(t) for t in p["input"]]})
        else: s._send({"response": nxt(p["prompt"], p["options"]["num_predict"])})
def start(port=11999):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H); threading.Thread(target=srv.serve_forever, daemon=True).start(); return srv
