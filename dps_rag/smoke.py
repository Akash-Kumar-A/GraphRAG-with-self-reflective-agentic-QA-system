import json, numpy as np
from dprag import *
D=json.load(open('prepared/data.json'))
lm=PublicLM([d['text'] for d in D['public']]+D['format_text'])
docs=[Doc(r['text']) for r in D['records']]; R=Retriever([r['text'] for r in D['records']])
for qi in (0,5,11):
    q=D['topics'][qi]['question']; qt=tok(q); rk=[docs[i] for i in R.top(q,50)]; tr=D['truth'][qi]
    print(q, tr)
    for alg,kw in [('nonrag',{}),('vote',{}),('dpvote',dict(eps_tok=1)),('dpsparse',dict(eps_tok=2)),('dpsparse',dict(eps_tok=1))]:
        o=generate(alg,qt,rk,lm,np.random.default_rng(1),**kw); print(' ',alg,kw,o['c_used'],o['tmax'],'|',' '.join(o['tokens']), slot_scores(o['tokens'],tr))
