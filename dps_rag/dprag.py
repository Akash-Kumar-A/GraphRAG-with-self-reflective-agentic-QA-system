"""DPS layer: Differentially-Private RAG via sample-and-aggregate + Sparse Vector Technique
(Koga, Wu, Zhang, Chaudhuri - 'Privacy-Preserving RAG with Differential Privacy', arXiv:2412.04697v3).
Implements: Non-RAG, standard RAG, VoteRAG (non-private), DPVoteRAG (Alg.2), DPSparseVoteRAG (Alg.3),
LimitedDomain mechanism, AboveThreshold noise, and the composition accountant (sequential + advanced)."""
import re, numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

TOK = re.compile(r"[a-z0-9]+|[.,]")
EOS = "."
def tok(s): return TOK.findall(s.lower())

class PublicLM:
    """Stand-in for the pretrained LLM: trigram->bigram->unigram greedy LM trained ONLY on public text."""
    def __init__(s, texts):
        s.tri, s.bi, s.uni = defaultdict(Counter), defaultdict(Counter), Counter(); s.cache = {}
        for t in texts:
            w = tok(t); s.uni.update(w)
            for a, b, c in zip(w, w[1:], w[2:]): s.tri[(a, b)][c] += 1; s.bi[b][c] += 1
    def next(s, bg):
        if bg in s.cache: return s.cache[bg]
        pick = lambda c: sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]
        r = pick(s.tri[bg]) if s.tri.get(bg) else pick(s.bi[bg[1]]) if s.bi.get(bg[1]) else pick(s.uni)
        s.cache[bg] = r; return r

class Doc:
    """A record, indexed so that an in-context 'induction copy' is O(1): last-2-tokens -> next token."""
    def __init__(s, text):
        s.text, w = text, tok(text); s.idx = {(a, b): c for a, b, c in zip(w, w[1:], w[2:])}

class Retriever:
    def __init__(s, texts):
        s.v = TfidfVectorizer(sublinear_tf=True, token_pattern=r"[a-z0-9]+"); s.X = s.v.fit_transform(texts)
    def top(s, q, n):
        sims = (s.X @ s.v.transform([q]).T).toarray().ravel(); return list(np.argsort(-sims, kind="stable")[:n])

# ---------------- DP building blocks ----------------
def limited_domain(h, eps, delta, rng):
    """Durfee-Rogers LimitedDomain top-1 with k_bar = m (all observed tokens). None => null token (= EOS)."""
    toks = list(h); c = np.array([h[t] for t in toks], float)
    noisy = c + rng.gumbel(0, 1 / eps, len(c))
    thr = 1 + np.log(1 / delta) / eps                      # c_(k_bar+1) = 0
    i = int(noisy.argmax()); return toks[i] if noisy[i] > thr else None

def max_compositions(eps_tok, d_tok, eps_tot, d_tot):
    """Max # of (eps_tok,d_tok)-DP steps that fit (eps_tot,d_tot): max of sequential and advanced composition."""
    best = min(int(eps_tot / eps_tok + 1e-9), int(d_tot / d_tok + 1e-9))
    for T in range(1, 5000):
        dp = d_tot - T * d_tok
        if dp <= 0: break
        if eps_tok * np.sqrt(2 * T * np.log(1 / dp)) + T * eps_tok * (np.exp(eps_tok) - 1) <= eps_tot: best = max(best, T)
    return best

def eps_spent(c, eps_tok, d_tok, d_tot):
    if c == 0: return 0.0
    dp = max(d_tot - c * d_tok, 1e-12)
    return float(min(c * eps_tok, eps_tok * np.sqrt(2 * c * np.log(1 / dp)) + c * eps_tok * (np.exp(eps_tok) - 1)))

# ---------------- generation ----------------
def generate(alg, qt, docs, lm, rng, m=50, k=1, eps_tok=2.0, eps_tot=10.0, d_tok=1e-5, d_tot=1e-4, t_out=60, trace=False):
    """alg in nonrag | stdrag | vote | dpvote | dpsparse.  qt = question tokens, docs = ranked Doc list (>= m*k)."""
    y, steps, c_used = [], [], 0
    if alg in ("vote", "dpvote", "dpsparse"):
        perm = rng.permutation(m * k); parts = [[docs[j] for j in perm[i * k:(i + 1) * k]] for i in range(m)]
    tmax = max_compositions(eps_tok, d_tok, eps_tot, d_tot)
    if alg == "dpvote": t_out = min(t_out, tmax)
    if alg == "dpsparse":
        c = tmax; eps_lap, eps_ld = eps_tok / 2, eps_tok / 2; tau = m / 2
        tau_hat = tau + rng.laplace(0, 2 / eps_lap)
    for t in range(t_out):
        bg = tuple((qt + y)[-2:]); y_non = lm.next(bg); src, cnt = "public-LM", None
        if alg == "nonrag": yt = y_non
        elif alg == "stdrag":
            yt = next((d.idx[bg] for d in docs[:k] if bg in d.idx), y_non); src = "retrieved-doc"
        else:
            h = Counter(next((d.idx[bg] for d in p if bg in d.idx), y_non) for p in parts)
            if alg == "vote": yt, src = h.most_common(1)[0][0], "vote"
            elif alg == "dpvote":
                yt = limited_domain(h, eps_tok, d_tok, rng); src = "private-vote"; c_used += 1
                if yt is None: steps.append((EOS, "halt(⊥)", 0)); y.append(EOS); break
            else:  # DPSparseVoteRAG
                a_t = h.get(y_non, 0)
                if a_t + rng.laplace(0, 4 / eps_lap) <= tau_hat:             # voters disagree with public LM -> spend budget
                    yt = limited_domain(h, eps_ld, d_tok, rng); c -= 1; c_used += 1; src = "private-vote"
                    tau_hat = tau + rng.laplace(0, 2 / eps_lap)
                    if yt is None: steps.append((EOS, "halt(⊥)", 0)); y.append(EOS); break
                    cnt = h.get(yt, 0)
                else: yt = y_non; src = "public-LM(free)"; cnt = a_t
                if c == 0 and yt != EOS: y.append(yt); steps.append((yt, src, cnt)); break
        y.append(yt)
        steps.append((yt, src, cnt))
        if yt == EOS: break
    return dict(tokens=y, steps=steps, c_used=c_used, eps_spent=eps_spent(c_used, eps_tok, d_tok, d_tot) if alg in ("dpvote", "dpsparse") else None, tmax=tmax)

def slot_scores(tokens, truth):
    text = " " + " ".join(tokens) + " "
    return {k: float(f" {' '.join(tok(v))} " in text) for k, v in truth.items()}
