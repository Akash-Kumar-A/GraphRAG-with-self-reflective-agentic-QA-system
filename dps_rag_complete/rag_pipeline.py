"""Plain RAG  vs  RAG with the DPS layer (DPVoteRAG / DPSparseVoteRAG, Koga et al. 2025), on Ollama + ChromaDB."""
import time, numpy as np
from collections import Counter
from dprag import limited_domain, max_compositions, eps_spent

EOS = "<EOS>"
def build_prompt(question, docs, prefix=""):
    ctx = ("Records:\n" + "\n".join(docs) + "\n\n") if docs else ""
    return f"{ctx}Question: {question}\nAnswer in one short sentence:{prefix}"
def canon(piece):
    """Any 'end of answer' piece (empty, newline, lone full stop) becomes the single EOS symbol."""
    return EOS if piece == "" or "\n" in piece or piece.strip() == "." else piece

# ---------------- WITHOUT the DPS layer ----------------
def plain_rag(ol, store, question, k=3):
    t = time.time(); docs, _ = store.query(question, k)
    return dict(answer=ol.complete(build_prompt(question, docs)), retrieved=docs, seconds=time.time() - t)

# ---------------- WITH the DPS layer ----------------
def dp_rag(ol, store, question, m=30, eps_tok=2.0, eps_tot=10.0, alg="dpsparse", d_tok=1e-5, d_tot=1e-4, seed=0, max_tokens=40, on_step=None):
    """Retrieve m docs, give each of m voters ONE doc, vote token by token.
    dpsparse: ask the public model (no docs) first; only spend budget when voters disagree with it."""
    t0 = time.time(); rng = np.random.default_rng(seed)
    docs, _ = store.query(question, m); m = len(docs)
    parts = [docs[j] for j in rng.permutation(m)]                         # uniform random partition, k = 1
    tmax = max_compositions(eps_tok, d_tok, eps_tot, d_tot)
    T = min(max_tokens, tmax) if alg == "dpvote" else max_tokens
    if alg == "dpsparse": c, tau, eps_lap = tmax, m / 2, eps_tok / 2; tau_hat = tau + rng.laplace(0, 2 / eps_lap)
    y, steps, used = [], [], 0
    for t in range(T):
        prefix = "".join(y)
        pieces = [canon(p) for p in ol.next_pieces([build_prompt(question, [d], prefix) for d in parts] + [build_prompt(question, [], prefix)])]
        h, y_non = Counter(pieces[:-1]), pieces[-1]
        if alg == "dpvote":
            yt = limited_domain(h, eps_tok, d_tok, rng); used += 1; src = "private"
        else:
            a_t = h.get(y_non, 0)
            if a_t + rng.laplace(0, 4 / eps_lap) <= tau_hat:                # voters disagree with public model -> pay
                yt = limited_domain(h, eps_tok / 2, d_tok, rng); used += 1; c -= 1; src = "private"
                tau_hat = tau + rng.laplace(0, 2 / eps_lap)
            else: yt, src = y_non, "free"
        if yt is None: steps.append(("⊥ halt", "halt", 0)); break          # LimitedDomain returned the null token
        steps.append((("" if yt == EOS else yt), src if yt != EOS else src + "-eos", h.get(yt, 0)))
        if on_step: on_step(steps, used)
        if yt == EOS: break
        y.append(yt)
        if yt.strip().endswith(".") or (alg == "dpsparse" and c == 0): break
    return dict(answer="".join(y).strip(), steps=steps, private_tokens=used, tmax=tmax, eps_spent=eps_spent(used, eps_tok, d_tok, d_tot) if used else 0.0,
                retrieved=docs, seconds=time.time() - t0)
