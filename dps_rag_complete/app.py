"""Streamlit demo:  streamlit run app.py"""
import json, random, streamlit as st
from ollama_client import Ollama
from vectorstore import Store
from rag_pipeline import plain_rag, dp_rag

st.set_page_config(page_title="DP layer demo", layout="wide")
CSS = """<style>.ch{display:inline-block;margin:2px 3px;padding:1px 8px;border-radius:5px;border:1.5px solid #0e8a7d}
.sp{border-color:#d9822b;background:rgba(217,130,43,.22);font-weight:600}.ha{border:1.5px dashed #c0392b;color:#c0392b}
.rec{font-size:13px;padding:6px 8px;margin:4px 0;border-left:3px solid #c0392b;background:rgba(192,57,43,.08)}</style>"""
st.markdown(CSS, unsafe_allow_html=True)
@st.cache_data
def records(): return json.load(open("prepared/data.json", encoding="utf-8"))["records"]
def chips(steps): return "".join(f'<span class="ch {"sp" if s.startswith("private") else "ha" if s=="halt" else ""}" title="{s} · votes for it: {c}">{w or "⏹ end"}</span>' for w, s, c in steps)

with st.sidebar:
    st.header("Setup")
    host = st.text_input("Ollama host", "http://localhost:11434"); model = st.text_input("Generation model", "llama3.2:3b")
    emb = st.text_input("Embedding model", "nomic-embed-text"); workers = st.slider("Parallel requests", 1, 16, 8)
    st.header("Privacy (DP)")
    alg = st.selectbox("Algorithm", ["dpsparse", "dpvote"], format_func=lambda a: {"dpsparse": "DPSparseVoteRAG", "dpvote": "DPVoteRAG"}[a])
    # eps_tot = st.slider("ε total", 2.0, 40.0, 10.0, 1.0); eps_tok = st.select_slider("ε per token", [1.0, 2.0, 5.0], 2.0); m = st.slider("Voters m", 10, 50, 30, 5)
    # k_dp = st.slider("Records per voter k", 1, 3, 1)
    eps_tot = st.slider("ε total", 2.0, 40.0, 10.0, 1.0); eps_tok = st.select_slider("ε per token", [1.0, 2.0, 5.0], 2.0); m = st.slider("Voters m", 1, 50, 30, 5)
    k_dp = st.slider("Records per voter k", 1, 5, 1)
    k = st.slider("Plain RAG: records in context", 1, 20, 3)
ol = Ollama(host, model, emb, workers); store = Store(ol)
st.title("RAG without and with the DP layer")
try: have = ol.models()
except Exception as e: st.error(f"Cannot reach Ollama at {host}: {e}"); st.stop()
for need in (model, emb):
    if not any(h.split(":")[0] == need.split(":")[0] for h in have): st.warning(f"Model '{need}' not found. Run: ollama pull {need}")
st.caption(f"ChromaDB holds {store.count()} confidential records (synthetic).")
if store.count() == 0 and st.button("Ingest records into ChromaDB"):
    bar = st.progress(0.0); store.ingest(records(), progress=bar.progress); st.rerun()

def pick():
    r = random.choice(records()); st.session_state.target = r
    st.session_state.q = f"Why did {r['name']} ({r['roll']}) submit a {r['topic']} request?"
st.session_state.setdefault("q", "What is the most common reason for medical leave requests?")
c1, c2 = st.columns([4, 1]); q = c1.text_input("Question", key="q"); c2.button("Target a student", on_click=pick, help="Ask about one specific student: the privacy attack")
target = st.session_state.get("target")
if target and q.startswith(f"Why did {target['name']}"): st.info(f"Confidential ground truth: {target['text']}")
if st.button("Run both", type="primary", disabled=store.count() == 0):
    L, R = st.columns(2)
    with L:
        st.subheader("Without DP"); st.caption("Top-k records go straight into the prompt.")
        with st.spinner("Generating..."): p = plain_rag(ol, store, q, k)
        st.success(p["answer"]); st.caption(f"{p['seconds']:.1f}s · records the model saw:")
        for d in p["retrieved"]: st.markdown(f'<div class="rec">{d}</div>', unsafe_allow_html=True)
    with R:
        st.subheader("With DP"); st.caption(f"{m} voters each see {k_dp} record(s); tokens are voted with noise.")
        live = st.empty()
        d = dp_rag(ol, store, q, m=m,k=k_dp, eps_tok=eps_tok, eps_tot=eps_tot, alg=alg, on_step=lambda s, u: live.markdown(chips(s) + f"<br><small>private tokens so far: {u}</small>", unsafe_allow_html=True))
        live.empty(); st.success(d["answer"] or "(no answer: budget or votes exhausted)"); st.markdown(chips(d["steps"]), unsafe_allow_html=True)
        a, b, c = st.columns(3); a.metric("Private tokens", f"{d['private_tokens']} / {d['tmax']}"); b.metric("ε spent", f"{d['eps_spent']:.2f} / {eps_tot:g}"); c.metric("Time", f"{d['seconds']:.0f}s")
        st.caption("Teal: public model, free. Amber: voted from confidential records, spends ε. The answer reflects what most records say, not any single record.")
    if target and q.startswith(f"Why did {target['name']}"):
        secret = target["text"].split("due to ")[1].split(",")[0]
        st.markdown(f"**Leak check** (secret reason: *{secret}*): without DP → {'**LEAKED**' if secret in p['answer'].lower() else 'not leaked'} · with DP → {'**LEAKED**' if secret in d['answer'].lower() else 'not leaked'}")
