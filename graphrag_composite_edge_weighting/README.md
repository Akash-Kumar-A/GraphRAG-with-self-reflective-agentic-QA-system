# GraphRAG Composite Edge Weighting

A reproducible research scaffold to compare frequency, relationship strength,
and entity-description semantic similarity as edge-weight signals for Leiden
community detection in a GraphRAG-style knowledge graph.

## What is included

- LLM-based entity/relation extraction using the OpenAI API.
- Evidence validation: an extracted evidence string must occur verbatim in its source document.
- Conservative normalized-name entity resolution.
- Duplicate-evidence aggregation by source, target, and relation type.
- Features:
  - `F`: log-normalized distinct evidence frequency.
  - `S`: normalized mean LLM relationship strength.
  - `C`: entity-description cosine similarity transformed to `[0, 1]`.
- Six core weighting variants:
  1. Sum of relationship strengths (official-style baseline; verify against pinned GraphRAG release).
  2. Frequency only.
  3. Strength only.
  4. Similarity only.
  5. Frequency + strength.
  6. Composite: `0.4F + 0.4S + 0.2C`.
- Three component ablations.
- Weighted Leiden clustering with controlled seeds.
- NMI/ARI when labels are available, stability ARI, singleton share, connected components,
  isolated-node count, community sizes, and weighted modularity diagnostic.
- Corpus fractions and repeated document-level sampling.
- CSV and JSON outputs.
- Tiny example corpus and labels for a demonstration run.

## Important research caveats

This is an implementation scaffold, not a claim that the proposed method improves
GraphRAG. The result must be established experimentally.

The included `MOCK_EXTRACTION=1` mode is only a software demonstration. Its
rule-based extractor is intentionally incomplete. Do not use its output as
research evidence. For a formal study, use a fixed LLM extraction configuration,
audit a sample of entities/relations/strength scores, and freeze the extracted
graph before comparing weight variants.

The "official GraphRAG" variant currently implements sum-of-extracted-strengths
aggregation as described in the accompanying methodology. Pin the exact official
GraphRAG version you cite and verify its actual edge aggregation and extraction
configuration before describing this as an exact reproduction.

The supplied `labels.csv` is illustrative. Its labels are not a validated gold
standard. NMI and ARI are only meaningful when the labels correspond to the
nodes and communities being evaluated. The small included corpus is for checking
that the software pipeline runs, not for drawing scientific conclusions.

The implementation uses an undirected graph for Leiden and sums weights for
parallel/opposite-direction edges. This design choice is fixed across variants;
describe it explicitly in a paper.

## 1. Requirements

Recommended: Python 3.11.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

## 2. Configure the environment

```powershell
Copy-Item .env.example .env
```

For a no-API demonstration, keep `MOCK_EXTRACTION=1`.

For LLM extraction, put your API key in `.env` and set:

```text
OPENAI_API_KEY=your_key_here
MOCK_EXTRACTION=0
OPENAI_MODEL=gpt-4o-mini
```

Do not commit `.env` or publish your API key.

## 3. Run the demonstration

The first embedding run may download the sentence-transformer model.

```powershell
python run_experiment.py --fractions 0.5,1.0 --repeats 2
```

This uses the mock extractor if `MOCK_EXTRACTION=1` is set. To explicitly use
the mock extractor:

```powershell
python run_experiment.py --mock-extraction --fractions 0.5,1.0 --repeats 2
```

If you need to test the graph pipeline without downloading an embedding model:

```powershell
python run_experiment.py --mock-extraction --no-embeddings --fractions 1.0 --repeats 1
```

For the full suggested experiment:

```powershell
python run_experiment.py --fractions 0.25,0.50,0.75,1.0 --repeats 10
```

For a first API extraction run:

```powershell
python run_experiment.py --fractions 1.0 --repeats 1
```

Extraction outputs are cached by sampled-document signature under `outputs/cache/`.
Delete the cache only if you intentionally want to re-run extraction.

## 4. Output files

- `outputs/edge_weights.csv`: edge features, evidence provenance, and all variant weights.
- `outputs/experiment_results.csv`: one row per sample/variant with metrics.
- `outputs/summary_statistics.csv`: mean and standard deviation by variant and corpus fraction.
- `outputs/extraction_runs.csv`: extraction/sample metadata.
- `outputs/run_config.json`: parameters and formulas used.

## 5. CSV schema notes

Important columns in `edge_weights.csv` include:

- `source_key`, `target_key`, `relation_type`
- `evidence_ids`, `evidence_texts`, `source_paths`
- `evidence_count`, `strength_sum`, `mean_strength`
- `cosine_similarity_raw`, `similarity_unit`
- `F`, `S`, `C`
- `weight_*` columns for each baseline, proposed variant, and ablation

## 6. Method definitions

For edge `e`:

```text
F_e = log(1 + n_e) / log(1 + n_max)
S_e = (mean_strength_e - 1) / 9
C_e = (cosine_similarity_e + 1) / 2
W_e = 0.4 F_e + 0.4 S_e + 0.2 C_e
```

When `n_max <= 1`, the implementation sets `F=1` for every edge to avoid division
by zero. The semantic feature is an affine transformation of cosine similarity,
not a calibrated probability of relationship correctness.

## 7. Recommended research procedure

1. Pin the official GraphRAG repository release/commit and reproduce its edge weighting.
2. Use a corpus large enough for the research question and document deduplication.
3. Freeze extraction outputs so every variant is evaluated on the same graph.
4. Audit extraction quality and relationship-strength ratings.
5. Use a controlled synthetic corpus with known topics and weak/strong cross-topic bridges.
6. Use a real corpus with defensible node-level labels for NMI/ARI.
7. Compare all variants on matched samples and multiple seeds.
8. Run ablations and coefficient sensitivity analysis.
9. Report uncertainty, negative findings, and failure cases.
10. Separately evaluate community summaries and downstream GraphRAG answer quality.

## 8. Known limitations to address before publication

- The sample corpus is tiny and only illustrative.
- Mock extraction is not a substitute for an LLM.
- Simple normalized-name matching does not resolve all aliases.
- Entity-description similarity can downweight valid cross-topic relations.
- A single fixed set of coefficients is a heuristic; test alternatives.
- The current program evaluates graph/community structure, not downstream GraphRAG answer quality.
- The official baseline must be checked against the exact upstream version you cite.
