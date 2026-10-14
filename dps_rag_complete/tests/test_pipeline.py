"""python tests/test_pipeline.py  -> runs the whole pipeline (Chroma + voting + DP) against the mock server."""
import sys, json, shutil, tempfile; sys.path.insert(0, ".")
from tests.mock_ollama import start
from ollama_client import Ollama
from vectorstore import Store
from rag_pipeline import plain_rag, dp_rag
start(); ol = Ollama("http://127.0.0.1:11999", "mock", "mock-embed"); tmp = tempfile.mkdtemp()
D = json.load(open("prepared/data.json", encoding="utf-8")); recs = D["records"]
st = Store(ol, path=tmp); st.ingest(recs); assert st.count() == len(recs); print("chroma:", st.count(), "records")
q = "Describe a typical transport pass request"; truth = D["truth"][11]
p = plain_rag(ol, st, q); print("plain :", p["answer"])
d = dp_rag(ol, st, q, m=50, eps_tok=2.0, eps_tot=10.0); print("dps   :", d["answer"], "| private", d["private_tokens"], "eps", round(d["eps_spent"], 2)); print("truth :", truth)
assert d["eps_spent"] <= 10 + 1e-9 and d["private_tokens"] <= d["tmax"]
r = recs[300]; tq = f"Why did {r['name']} ({r['roll']}) submit a {r['topic']} request?"; secret = r["text"].split("due to ")[1].split(",")[0]
tp = plain_rag(ol, st, f"Student {r['name']} ({r['roll']}) of {r['prog']} submitted a {r['topic']} request"); td = dp_rag(ol, st, f"Student {r['name']} ({r['roll']}) of {r['prog']} submitted a {r['topic']} request", m=50)
print("target secret:", secret, "| plain:", tp["answer"], "| dps:", td["answer"]); print("plain leaked:", secret in tp["answer"].lower(), "| dps leaked:", secret in td["answer"].lower())
shutil.rmtree(tmp, ignore_errors=True)
