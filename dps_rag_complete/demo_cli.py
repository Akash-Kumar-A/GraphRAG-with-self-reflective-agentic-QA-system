"""Terminal demo:  python demo_cli.py --question "..." [--target]"""
import argparse, json, random
from ollama_client import Ollama
from vectorstore import Store
from rag_pipeline import plain_rag, dp_rag

ap = argparse.ArgumentParser()
ap.add_argument("--question", default="What is the most common reason for medical leave requests?")
ap.add_argument("--target", action="store_true", help="ask about one specific student (privacy attack)")
ap.add_argument("--model", default="llama3.2:3b"); ap.add_argument("--embed", default="nomic-embed-text"); ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--eps-total", type=float, default=10); ap.add_argument("--eps-token", type=float, default=2); ap.add_argument("--m", type=int, default=30)
ap.add_argument("--alg", default="dpsparse", choices=["dpsparse", "dpvote"]); a = ap.parse_args()

ol = Ollama(a.host, a.model, a.embed); store = Store(ol)
if store.count() == 0:
    recs = json.load(open("prepared/data.json", encoding="utf-8"))["records"]; print(f"Ingesting {len(recs)} records into ChromaDB ..."); store.ingest(recs)
q, target = a.question, None
if a.target:
    target = random.choice(json.load(open("prepared/data.json", encoding="utf-8"))["records"])
    q = f"Why did {target['name']} ({target['roll']}) submit a {target['topic']} request?"
print("\nQUESTION:", q)
if target: print("GROUND TRUTH (confidential):", target["text"])
p = plain_rag(ol, store, q)
print("\n=== WITHOUT DPS ===  retrieved", len(p["retrieved"]), "records\n  ", p["answer"], f"\n  ({p['seconds']:.1f}s)")
d = dp_rag(ol, store, q, m=a.m, eps_tok=a.eps_token, eps_tot=a.eps_total, alg=a.alg)
print(f"\n=== WITH DPS ({a.alg}, m={a.m}, eps_total={a.eps_total}, eps_token={a.eps_token}) ===\n  ", d["answer"])
print("   trace:", " ".join(f"[{w}|{s}]" for w, s, _ in d["steps"]))
print(f"   private tokens {d['private_tokens']} of max {d['tmax']} | eps spent {d['eps_spent']:.2f} of {a.eps_total} | {d['seconds']:.1f}s")
