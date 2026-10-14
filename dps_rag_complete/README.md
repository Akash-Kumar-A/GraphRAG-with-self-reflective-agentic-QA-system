# DPS layer: differentially private RAG on Ollama + ChromaDB

Paper: Koga, Wu, Zhang, Chaudhuri, arXiv:2412.04697v3 (DPVoteRAG, DPSparseVoteRAG).

## 1. One-time setup (Windows PowerShell)
    pip install -r requirements.txt
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    $env:OLLAMA_NUM_PARALLEL = "8"      # set BEFORE `ollama serve` / restart Ollama, so the voters run in parallel

## 2. Live demo (what you present)
    streamlit run app.py
1. Click "Ingest records into ChromaDB" (645 synthetic records, one time).
2. Ask "What is the most common reason for medical leave requests?" and press Run both.
3. Press "Target a student", then Run both. Without DPS the model repeats that student's confidential reason; with DPS the answer comes from the vote, and the leak check shows the difference.
Terminal version: `python demo_cli.py` or `python demo_cli.py --target`

## 3. Where the code is
| File | Role |
|---|---|
| vectorstore.py | ChromaDB: ingest records, similarity search (embeddings from Ollama) |
| ollama_client.py | one-token voters in parallel, full completions, embeddings |
| rag_pipeline.py | plain_rag() and dp_rag() (the DPS layer: partition, vote, sparse vector technique, LimitedDomain, accounting) |
| dprag.py | DP primitives + accountant, also the offline simulation used for the measured experiments |
| evaluate.py, build_dashboard.py | all metrics and the results dashboard (python evaluate.py; python build_dashboard.py) |
| tests/ | mock Ollama server + pipeline test: `python tests/test_pipeline.py` |

## 4. Tuning on a real model
- Each answer needs about (voters + 1) calls per token, so a 3B model with m=30 takes a few minutes on CPU. Lower m or the number of tokens if slow.
- epsilon per token = 1 needs m close to 50 (the vote must beat the noise threshold). If answers end early, raise epsilon per token or m.
- The private-token cap (c max) comes from the budget; epsilon total 10 with epsilon per token 2 allows 5 private tokens.
