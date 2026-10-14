"""Optional: plug a real HuggingFace causal LM (OPT-1.3B, Pythia-1.4B, Llama-3.1-8B...) into the voting layer.
Not run in the sandbox (no model download access) - tested only for interface compatibility.
The DP code in dprag.generate() only needs: next_token(question, docs, prefix) -> str."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
class HFVoter:
    def __init__(self, name="facebook/opt-1.3b", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.t = AutoTokenizer.from_pretrained(name); self.m = AutoModelForCausalLM.from_pretrained(name).to(device).eval(); self.dev = device
    @torch.no_grad()
    def next_token(self, question, docs, prefix):
        ctx = "\n".join(docs) + "\nQuestion: " + question + "\nAnswer:" + prefix      # docs=[] -> the public (non-RAG) token
        ids = self.t(ctx, return_tensors="pt").input_ids.to(self.dev)
        return self.t.decode(self.m(ids).logits[0, -1].argmax())
# To use: replace the dict lookup `d.idx[bg]` and `lm.next(bg)` inside dprag.generate() with
# voter.next_token(q, [d.text for d in part], decoded_prefix) and voter.next_token(q, [], decoded_prefix).
