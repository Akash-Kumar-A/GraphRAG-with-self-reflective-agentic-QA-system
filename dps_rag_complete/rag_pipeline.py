# """Plain RAG  vs  RAG with the DPS layer (DPVoteRAG / DPSparseVoteRAG, Koga et al. 2025), on Ollama + ChromaDB."""
# import time, numpy as np
# from collections import Counter
# from dprag import limited_domain, max_compositions, eps_spent

# EOS = "<EOS>"
# def build_prompt(question, docs, prefix=""):
#     ctx = ("Records:\n" + "\n".join(docs) + "\n\n") if docs else ""
#     return f"{ctx}Question: {question}\nAnswer in one short sentence:{prefix}"
# def canon(piece):
#     """Any 'end of answer' piece (empty, newline, lone full stop) becomes the single EOS symbol."""
#     return EOS if piece == "" or "\n" in piece or piece.strip() == "." else piece

# # ---------------- WITHOUT the DP layer ----------------
# def plain_rag(ol, store, question, k=3):
#     t = time.time(); docs, _ = store.query(question, k)
#     return dict(answer=ol.complete(build_prompt(question, docs)), retrieved=docs, seconds=time.time() - t)

# # ---------------- WITH the DP layer ----------------
# # def dp_rag(ol, store, question, m=30, eps_tok=2.0, eps_tot=10.0, alg="dpsparse", d_tok=1e-5, d_tot=1e-4, seed=0, max_tokens=40, on_step=None):
# def dp_rag(ol, store, question, m=30, k=1, eps_tok=2.0, eps_tot=10.0, alg="dpsparse", d_tok=1e-5, d_tot=1e-4, seed=0, max_tokens=40, on_step=None):
#     """Retrieve m docs, give each of m voters ONE doc, vote token by token.
#     dpsparse: ask the public model (no docs) first; only spend budget when voters disagree with it."""
#     t0 = time.time(); rng = np.random.default_rng(seed)
#     docs, _ = store.query(question, m); m = len(docs)
#     parts = [docs[j] for j in rng.permutation(m)]                         # uniform random partition, k = 1
    
#     tmax = max_compositions(eps_tok, d_tok, eps_tot, d_tot)
#     T = min(max_tokens, tmax) if alg == "dpvote" else max_tokens
#     if alg == "dpsparse": c, tau, eps_lap = tmax, m / 2, eps_tok / 2; tau_hat = tau + rng.laplace(0, 2 / eps_lap)
#     y, steps, used = [], [], 0
#     for t in range(T):
#         prefix = "".join(y)
#         pieces = [canon(p) for p in ol.next_pieces([build_prompt(question, [d], prefix) for d in parts] + [build_prompt(question, [], prefix)])]
#         h, y_non = Counter(pieces[:-1]), pieces[-1]
#         if alg == "dpvote":
#             yt = limited_domain(h, eps_tok, d_tok, rng); used += 1; src = "private"
#         else:
#             a_t = h.get(y_non, 0)
#             if a_t + rng.laplace(0, 4 / eps_lap) <= tau_hat:                # voters disagree with public model -> pay
#                 yt = limited_domain(h, eps_tok / 2, d_tok, rng); used += 1; c -= 1; src = "private"
#                 tau_hat = tau + rng.laplace(0, 2 / eps_lap)
#             else: yt, src = y_non, "free"
#         if yt is None: steps.append(("⊥ halt", "halt", 0)); break          # LimitedDomain returned the null token
#         steps.append((("" if yt == EOS else yt), src if yt != EOS else src + "-eos", h.get(yt, 0)))
#         if on_step: on_step(steps, used)
#         if yt == EOS: break
#         y.append(yt)
#         if yt.strip().endswith(".") or (alg == "dpsparse" and c == 0): break
#     return dict(answer="".join(y).strip(), steps=steps, private_tokens=used, tmax=tmax, eps_spent=eps_spent(used, eps_tok, d_tok, d_tot) if used else 0.0,
#                 retrieved=docs, seconds=time.time() - t0)

"""Plain RAG  vs  RAG with the DP layer (DPVoteRAG / DPSparseVoteRAG, Koga et al., arXiv:2412.04697), on Ollama + ChromaDB."""
import time, numpy as np
from collections import Counter
from dprag import limited_domain, max_compositions, eps_spent

EOS = "<EOS>"
ALGS = ("dpvote", "dpsparse")


def build_vote_prompt(question, docs, prefix=""):
    """DP side: prompt for ONE next token. Used by every voter (docs = its k records) and by the
    public model (docs = []). Same fixed template for all; only the records differ."""
    if docs:   # voter: report what ITS records say; the vote turns that into the majority answer
        ctx = ("Records:\n" + "\n".join(docs) + "\n\n"
               "Answer in a few words using what the records above say. ")
    else:      # public model: no records, no "Not mentioned" rule (otherwise nothing would ever be free)
        ctx = "Answer in a few words.\n"
    return f"{ctx}Question: {question}\nAnswer:{prefix}"


def build_plain_prompt(question, docs):
    """Plain RAG side (no DP): all top-k records in one prompt, one full answer."""
    ctx = "Records:\n" + "\n".join(docs) + "\n\n" if docs else ""
    return (f"{ctx}Answer the question using only the records above, in one short sentence. "
            f"Question: {question}\nAnswer:")


def canon(piece):
    """Any 'end of answer' piece (empty, newline, lone full stop) becomes the single EOS symbol."""
    return EOS if piece == "" or "\n" in piece or piece.strip() == "." else piece


# ---------------- WITHOUT the DP layer ----------------
def plain_rag(ol, store, question, k=3):
    t = time.time(); docs, _ = store.query(question, k)
    return dict(answer=ol.complete(build_plain_prompt(question, docs)), retrieved=docs, seconds=time.time() - t)


# ---------------- WITH the DP layer ----------------
def dp_rag(ol, store, question, m=30, k=1, eps_tok=2.0, eps_tot=10.0, alg="dpsparse", d_tok=1e-5, d_tot=1e-4,
           seed=None, max_tokens=100, on_step=None, debug=False, fallback_on_halt=False, stop_on_period=True):
    """Retrieve m*k docs, split them uniformly at random into m voters of k docs, vote token by token.
    dpsparse: ask the public model (no docs) first; only spend budget when voters disagree with it.

    Privacy: the guarantee is (eps_tot, d_tot)-DP per question, for the released tokens and trace only.
      * debug=False (default): result holds only what the guarantee covers. debug=True adds raw vote counts,
        retrieved docs and timing, which are NOT private: never show them to end users.
      * seed=None draws fresh OS entropy. A fixed seed is for reproducing experiments only and voids the guarantee.
      * fallback_on_halt=True: when LimitedDomain returns the null token, emit the public model's token
        instead of stopping. Still DP (post-processing with public data), but differs from the paper.
      * stop_on_period=False: don't stop on a token ending in "." (avoids cutting 'Dr. Megala M.'-style answers)."""
    # ---- validation (fix: unknown alg / bad sizes / zero budget)
    if alg not in ALGS: raise ValueError(f"alg must be one of {ALGS}, got {alg!r}")
    if m < 1 or k < 1 or max_tokens < 1: raise ValueError("m, k and max_tokens must all be >= 1")
    tmax = max_compositions(eps_tok, d_tok, eps_tot, d_tot)
    if tmax < 1: raise ValueError(f"eps_total={eps_tot} is too small for eps_token={eps_tok}: not even one private token fits")

    t0 = time.time(); rng = np.random.default_rng(seed)               # seed=None -> fresh OS entropy
    docs, _ = store.query(question, m * k); m = len(docs) // k; docs = docs[:m * k]   # paper Alg. 2, step 3
    if m < 1: raise RuntimeError(f"retriever returned fewer than k={k} records")
    perm = rng.permutation(m * k)                                         # paper Alg. 2, step 4: uniform random partition
    parts = [[docs[j] for j in perm[i * k:(i + 1) * k]] for i in range(m)]  # m voters, k records each

    T = min(max_tokens, tmax) if alg == "dpvote" else max_tokens
    if alg == "dpsparse": c, tau, eps_lap = tmax, m / 2, eps_tok / 2; tau_hat = tau + rng.laplace(0, 2 / eps_lap)
    need_public = alg == "dpsparse" or fallback_on_halt                   # fix: dpvote skips the unused public call
    y, steps, used = [], [], 0

    def emit():                                                           # what on_step / caller may see
        return steps if debug else [(w, s, None) for w, s, _ in steps]

    for t in range(T):
        prefix = "".join(y)
        prompts = [build_vote_prompt(question, d, prefix) for d in parts] + ([build_vote_prompt(question, [], prefix)] if need_public else [])
        pieces = [canon(p) for p in ol.next_pieces(prompts)]
        h = Counter(pieces[:m]); y_non = pieces[m] if need_public else None
        if alg == "dpvote":
            yt = limited_domain(h, eps_tok, d_tok, rng); used += 1; src = "private"
        else:
            a_t = h.get(y_non, 0)
            if a_t + rng.laplace(0, 4 / eps_lap) <= tau_hat:              # voters disagree with public model -> pay
                yt = limited_domain(h, eps_tok / 2, d_tok, rng); used += 1; c -= 1; src = "private"
                tau_hat = tau + rng.laplace(0, 2 / eps_lap)
            else: yt, src = y_non, "free"
        if yt is None:                                                    # LimitedDomain returned the null token (⊥)
            if not fallback_on_halt:
                steps.append(("⊥ halt", "halt", 0)); break
            yt, src = y_non, "fallback"                                   # public token: no extra privacy cost
        steps.append((("" if yt == EOS else yt), src if yt != EOS else src + "-eos", h.get(yt, 0)))
        if on_step: on_step(emit(), used)
        if yt == EOS: break
        y.append(yt)
        if (stop_on_period and yt.strip().endswith(".")) or (alg == "dpsparse" and c <= 0): break   # fix: c <= 0

    out = dict(answer="".join(y).strip(), steps=emit(), private_tokens=used, tmax=tmax,
               eps_guaranteed=eps_tot, delta_guaranteed=d_tot,          # fix: the real guarantee is the full pre-set budget
               eps_spent=eps_spent(used, eps_tok, d_tok, d_tot) if used else 0.0)   # informational only, NOT a privacy claim
    if debug: out.update(retrieved=docs, seconds=time.time() - t0)       # fix: non-private data only behind debug
    return out
