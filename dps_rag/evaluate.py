"""Step 3 - run every experiment and write results.json for the dashboard."""
import json, time, random, itertools, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
from dprag import *

t0 = time.time()
D = json.load(open("prepared/data.json"))
lm = PublicLM([d["text"] for d in D["public"]] + D["format_text"])
recs = D["records"]; docs = [Doc(r["text"]) for r in recs]; R = Retriever([r["text"] for r in recs])
topics, truths = D["topics"], D["truth"]
EPS, MS, ET, SEEDS, SLOTS = [2, 5, 10, 15, 20, 30, 40], [30, 40, 50], [1, 2, 5], 10, ["month", "reason", "outcome", "faculty"]
rank = [[docs[i] for i in R.top(t["question"], 90)] for t in topics]
qtok = [tok(t["question"]) for t in topics]

def run(alg, qi, seed, m=50, **kw):
    rng = np.random.default_rng([seed, qi, m, int(kw.get("eps_tok", 0) * 10), int(kw.get("eps_tot", 0) * 10)])
    o = generate(alg, qtok[qi], rank[qi][:max(m, 3)], lm, rng, m=m, **kw)
    s = slot_scores(o["tokens"], truths[qi])
    return dict(q=qi, slots=s, acc=np.mean(list(s.values())), exact=float(all(s.values())), n=len(o["tokens"]),
                priv=o["c_used"], eps=o["eps_spent"] or 0.0, tmax=o["tmax"])
def agg(rows):
    by = defaultdict(list)
    for r in rows: by[r["seed"]].append(r["acc"])
    ps = [np.mean(v) for v in by.values()]
    return dict(mean=float(np.mean(ps)), std=float(np.std(ps)), exact=float(np.mean([r["exact"] for r in rows])),
                n=float(np.mean([r["n"] for r in rows])), priv=float(np.mean([r["priv"] for r in rows])), eps=float(np.mean([r["eps"] for r in rows])),
                slots={k: float(np.mean([r["slots"][k] for r in rows])) for k in SLOTS})
def sweep(alg, **kw):
    return [dict(run(alg, qi, s, **kw), seed=s) for s in range(SEEDS) for qi in range(len(topics))]

# ---- baselines
base = {"nonrag": agg(sweep("nonrag")), **{f"vote_m{m}": agg(sweep("vote", m=m)) for m in MS}}
# ---- DP grid
grid = {}
for alg, e, m, et in itertools.product(["dpvote", "dpsparse"], EPS, MS, ET):
    rows = sweep(alg, m=m, eps_tok=float(et), eps_tot=float(e)); grid[(alg, e, m, et)] = (agg(rows), rows)
print("grid done", round(time.time() - t0, 1), "s")

curve = {"nonrag": [dict(eps=e, **{k: base["nonrag"][k] for k in ("mean", "std", "exact")}) for e in EPS],
         "vote": [dict(eps=e, **{k: base["vote_m50"][k] for k in ("mean", "std", "exact")}) for e in EPS]}
best = {}
for alg in ["dpvote", "dpsparse"]:
    curve[alg] = []
    for e in EPS:
        cfg = max(((m, et) for m in MS for et in ET), key=lambda c: grid[(alg, e, *c)][0]["mean"])
        a = grid[(alg, e, *cfg)][0]; best[(alg, e)] = cfg
        curve[alg].append(dict(eps=e, mean=a["mean"], std=a["std"], exact=a["exact"], m=cfg[0], eps_tok=cfg[1], n=a["n"], priv=a["priv"], spent=a["eps"]))
table = {str(e): {alg: {str(et): {str(m): grid[(alg, e, m, et)][0]["mean"] for m in MS} for et in ET} for alg in ["dpvote", "dpsparse"]} for e in [5, 10, 20, 40]}
tokens = {alg: {str(et): [dict(eps=e, n=grid[(alg, e, 50, et)][0]["n"], priv=grid[(alg, e, 50, et)][0]["priv"]) for e in EPS] for et in ET} for alg in ["dpvote", "dpsparse"]}
acct = [dict(eps_tot=e, eps_tok=et, cmax=max_compositions(et, 1e-5, e, 1e-4), spent=grid[("dpsparse", e, 50, et)][0]["eps"],
             priv=grid[("dpsparse", e, 50, et)][0]["priv"]) for e in EPS for et in ET]
# accuracy by # relevant docs
BINS = {"20-39 docs": [0, 1, 2, 3], "40-59 docs": [4, 5, 6], "60-79 docs": [7, 8, 9], "80+ docs": [10, 11]}
def binacc(rows, qs): return float(np.mean([r["acc"] for r in rows if r["q"] in qs]))
bins = {}
for e in [10, 40]:
    rows = grid[("dpsparse", e, 50, 2)][1]
    bins[str(e)] = {b: binacc(rows, qs) for b, qs in BINS.items()}
bins["vote"] = {b: binacc(sweep("vote", m=50), qs) for b, qs in BINS.items()}
bins["nonrag"] = {b: binacc(sweep("nonrag"), qs) for b, qs in BINS.items()}
slots10 = {"nonrag": base["nonrag"]["slots"], "vote": base["vote_m50"]["slots"],
           "dpvote": grid[("dpvote", 10, *best[("dpvote", 10)])][0]["slots"], "dpsparse": grid[("dpsparse", 10, *best[("dpsparse", 10)])][0]["slots"]}

# ---- empirical privacy: S2MIA-style membership inference
rs = random.Random(3)   # stratified: 12 members + 12 non-members per topic (avoids topic-size confound)
mem = [r for t in topics for r in rs.sample([x for x in recs if x["topic"] == t["topic"]], 12)]
non = [r for t in topics for r in rs.sample([x for x in D["heldout"] if x["topic"] == t["topic"]], 12)]
NM = len(mem)
def f1(g, t):
    c = sum((Counter(g) & Counter(t)).values()); 
    if not c: return 0.0
    p, r = c / len(g), c / len(t); return 2 * p * r / (p + r)
def mia(alg, **kw):
    sc, lab = [], []
    for i, (r, y) in enumerate([(r, 1) for r in mem] + [(r, 0) for r in non]):
        pre = f"Student {r['name']} ({r['roll']}) of {r['prog']} submitted a {r['topic']} request in the month of"
        gt = tok(r["text"][len(pre):]); gt = gt[:gt.index(".") + 1]
        rk = [docs[j] for j in R.top(pre, 50)]
        rng = np.random.default_rng([i, 99])
        g = generate(alg, tok(pre), rk, lm, rng, t_out=40, **kw)["tokens"]
        sc.append(f1(g, gt)); lab.append(y)
    fpr, tpr, _ = roc_curve(lab, sc); idx = np.unique(np.linspace(0, len(fpr) - 1, 50).astype(int))
    la, sa, bs, bg = np.array(lab), np.array(sc), [], np.random.default_rng(0)
    for _ in range(300):
        ix = bg.integers(0, len(la), len(la))
        if la[ix].min() != la[ix].max(): bs.append(roc_auc_score(la[ix], sa[ix]))
    return dict(ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], auc=float(roc_auc_score(lab, sc)), fpr=[float(x) for x in fpr[idx]], tpr=[float(x) for x in tpr[idx]],
                tpr_at_10=float(np.interp(0.1, fpr, tpr)), mean_member=float(np.mean(np.array(sc)[:NM])), mean_non=float(np.mean(np.array(sc)[NM:])))
M = dict(m=50, eps_tok=2.0)
mia_res = {"Standard RAG (k=3)": mia("stdrag", k=3), "VoteRAG (non-private, m=50)": mia("vote", **{"m": 50}),
           "DPVoteRAG (eps=10)": mia("dpvote", eps_tot=10.0, **M), "DPSparseVoteRAG (eps=10)": mia("dpsparse", eps_tot=10.0, **M),
           "DPSparseVoteRAG (eps=40)": mia("dpsparse", eps_tot=40.0, **M)}
print({k: round(v["auc"], 3) for k, v in mia_res.items()})

# ---- token traces for the demo panel
trace = []
for qi in (0, 5, 11):
    row = dict(question=topics[qi]["question"], n_docs=topics[qi]["n"], truth=truths[qi], runs={})
    for name, alg, kw in [("Non-RAG", "nonrag", {}), ("VoteRAG (no privacy)", "vote", dict(m=50)),
                          ("DPVoteRAG eps=10", "dpvote", dict(m=50, eps_tok=1.0, eps_tot=10.0)),
                          ("DPSparseVoteRAG eps=10", "dpsparse", dict(m=50, eps_tok=2.0, eps_tot=10.0))]:
        o = generate(alg, qtok[qi], rank[qi][:50], lm, np.random.default_rng([qi, 5]), **kw)
        row["runs"][name] = dict(steps=[[a, b, c] for a, b, c in o["steps"]], priv=o["c_used"], eps=o["eps_spent"], tmax=o["tmax"],
                                 slots=slot_scores(o["tokens"], truths[qi]))
    trace.append(row)

cat = Counter(d["category"] for d in D["public"])
meta = dict(public_docs=len(D["public"]), categories=dict(cat), words=sum(len(d["text"].split()) for d in D["public"]), records=len(recs), heldout=len(D["heldout"]),
            topics=len(topics), questions=len(topics), slots=len(SLOTS), seeds=SEEDS, delta_total=1e-4, delta_token=1e-5,
            programmes=len(D["programmes"]), faculty=len(D["faculty"]), topic_sizes=[t["n"] for t in topics], topic_names=[t["topic"] for t in topics],
            lm_vocab=len(lm.uni), sample_record=recs[0]["text"], elapsed=round(time.time() - t0, 1))
json.dump(dict(meta=meta, curve=curve, base={k: dict(mean=v["mean"], exact=v["exact"]) for k, v in base.items()}, table=table, tokens=tokens, acct=acct,
               bins=bins, slots10=slots10, mia=mia_res, trace=trace), open("prepared/results.json", "w"))
print("done", meta["elapsed"], "s")
