# DPS layer - differentially private RAG (DPVoteRAG / DPSparseVoteRAG)
Paper: Koga, Wu, Zhang, Chaudhuri, arXiv:2412.04697v3.

    pip install numpy scikit-learn
    python data_prep.py          # clean public corpus + generate synthetic confidential records -> prepared/data.json
    python evaluate.py           # all experiments (about 10 s) -> prepared/results.json
    python build_dashboard.py    # -> dps_dashboard.html (open in a browser, needs internet for Chart.js + fonts)

Files: dprag.py (algorithms, LimitedDomain, accountant) | evaluate.py (metrics) | llm_backend.py (optional real LLM, untested)
