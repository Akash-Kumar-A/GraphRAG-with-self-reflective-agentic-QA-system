"""
Composite edge-weighting experiments for a GraphRAG-style knowledge graph.

Research implementation scaffold:
- Extract entities/relationships and evidence from text files.
- Resolve entity names conservatively and aggregate distinct evidence.
- Calculate frequency, strength, and semantic-similarity features.
- Compare six core weighting variants plus component ablations.
- Run weighted Leiden with controlled seeds.
- Calculate NMI/ARI where labels are available, partition stability,
  singleton share, connected components, and community diagnostics.
- Run document-level corpus subsampling and repeated experiments.
- Write extraction, edge-weight, per-run, and summary CSV outputs.

Important:
1. The "official_graphrag_sum_strength" variant implements the sum-of-strengths
   aggregation described in the project methodology. It is not a substitute for
   pinning and reproducing a specific official GraphRAG release. For a paper,
   document the exact upstream version and verify its extraction/aggregation code.
2. MOCK_EXTRACTION is only a runnable demonstration. Use the OpenAI extractor,
   audit the outputs, and freeze the extracted graph before formal experiments.
3. Entity semantic similarity here is cosine similarity between entity
   descriptions. It is not a direct measure of relation validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "corpus"
LABELS_PATH = ROOT / "data" / "labels.csv"
OUTPUT_DIR = ROOT / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("composite-edge-weighting")

RELATION_EXTRACTION_PROMPT = """You extract a knowledge graph from institutional policy text.
Return ONLY valid JSON with this exact shape:
{
  "entities": [
    {"name": "canonical entity name", "description": "short description"}
  ],
  "relationships": [
    {
      "source": "entity name",
      "target": "entity name",
      "relation_type": "short normalized verb phrase",
      "description": "concise statement of the relationship",
      "evidence": "verbatim supporting sentence from the input",
      "strength": 1
    }
  ]
}
Rules:
- Only extract entities and relations supported by the text.
- Use entity names consistently within this chunk.
- Evidence must be a sentence or substring present in the input.
- Strength must be an integer from 1 to 10, based on directness and defining
  importance of this specific relationship, not topic popularity:
  1-2 incidental/very weak; 3-4 weak; 5-6 moderate and explicit;
  7-8 strong and direct; 9-10 explicitly defining/central.
- Do not invent missing facts. Use an empty list if none.
- Prefer meaningful people, offices, policies, documents, processes and facilities.
"""

# Demonstration-only fallback. It is intentionally simple and is not intended
# to replace a research-quality LLM extractor.
MOCK_PATTERNS = [
    (r"medical leave.*medical certificate|medical certificate.*medical leave",
     "medical leave", "requires evidence", "medical certificate", 9),
    (r"Dean of Academic Affairs.*medical leave|medical leave.*Dean of Academic Affairs",
     "Dean of Academic Affairs", "approves", "medical leave", 8),
    (r"hostel residents.*hostel regulations|hostel regulations.*hostel residents",
     "hostel residents", "must follow", "hostel regulations", 8),
    (r"hostel mess.*hostel residents|hostel residents.*hostel mess",
     "hostel mess", "provides meals to", "hostel residents", 8),
    (r"warden.*hostel complaints|hostel complaints.*warden",
     "warden", "handles", "hostel complaints", 7),
    (r"examination office.*examination timetable|examination timetable.*examination office",
     "examination office", "publishes", "examination timetable", 8),
    (r"controller of examinations.*result publication|result publication.*controller of examinations",
     "controller of examinations", "oversees", "result publication", 8),
    (r"registrar.*enrollment records|enrollment records.*registrar",
     "registrar", "maintains", "enrollment records", 7),
    (r"department office.*course records|course records.*department office",
     "department office", "maintains", "course records", 7),
    (r"medical leave.*examination absence|examination absence.*medical leave",
     "medical leave", "may be considered for", "examination absence", 5),
]


@dataclass
class Document:
    doc_id: str
    path: str
    text: str


def stable_id(text: str, prefix: str = "") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def normalize_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s_-]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def load_documents(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    docs: list[Document] = []
    seen_hashes: set[str] = set()
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            LOG.warning("Skipping duplicate document: %s", path.name)
            continue
        seen_hashes.add(digest)
        docs.append(Document(
            doc_id=stable_id(text, "doc_"),
            path=str(path.relative_to(ROOT)),
            text=text,
        ))
    if not docs:
        raise FileNotFoundError(
            f"No .txt corpus documents found in {corpus_dir}. Add documents to data/corpus/."
        )
    return docs


def split_sentences(text: str) -> list[str]:
    # Basic sentence splitting for provenance/evidence matching.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def mock_extract(doc: Document) -> dict[str, Any]:
    entities: dict[str, dict[str, str]] = {}
    relationships: list[dict[str, Any]] = []
    sentences = split_sentences(doc.text)

    # Add entities by phrase matches, including names in patterns.
    candidate_entities = [
        "medical leave", "medical certificate", "Dean of Academic Affairs",
        "department office", "hostel residents", "hostel regulations",
        "hostel mess", "warden", "hostel complaints", "examination office",
        "examination timetable", "controller of examinations", "result publication",
        "registrar", "enrollment records", "course records", "examination absence",
        "revaluation request", "student", "students", "course instructor",
        "hostel office", "academic policy", "department heads",
    ]
    lower = doc.text.lower()
    for name in candidate_entities:
        if name.lower() in lower:
            entities[normalize_name(name)] = {
                "name": name,
                "description": f"Institutional concept or entity mentioned in {Path(doc.path).name}.",
            }

    for sent_idx, sentence in enumerate(sentences):
        lower_sentence = sentence.lower()
        for pattern, source, relation, target, strength in MOCK_PATTERNS:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                if source.lower() not in lower_sentence or target.lower() not in lower_sentence:
                    continue
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation_type": relation,
                    "description": f"{source} {relation} {target}",
                    "evidence": sentence,
                    "strength": strength,
                    "doc_id": doc.doc_id,
                    "chunk_id": f"{doc.doc_id}_chunk_0",
                    "evidence_id": stable_id(f"{doc.doc_id}|{sent_idx}|{sentence}", "ev_"),
                    "source_path": doc.path,
                })

    # A fallback makes the mock pipeline visibly useful for any corpus.
    if not relationships:
        names = list(entities.values())
        if len(names) >= 2:
            relationships.append({
                "source": names[0]["name"],
                "target": names[1]["name"],
                "relation_type": "mentioned with",
                "description": f"{names[0]['name']} is mentioned with {names[1]['name']}",
                "evidence": sentences[0] if sentences else doc.text[:300],
                "strength": 3,
                "doc_id": doc.doc_id,
                "chunk_id": f"{doc.doc_id}_chunk_0",
                "evidence_id": stable_id(f"{doc.doc_id}|fallback", "ev_"),
                "source_path": doc.path,
            })
    return {"entities": list(entities.values()), "relationships": relationships}


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    result = json.loads(content[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object.")
    result.setdefault("entities", [])
    result.setdefault("relationships", [])
    return result


def llm_extract(doc: Document, client: Any, model: str) -> dict[str, Any]:
    # A single document per request keeps provenance easy to trace. For very
    # large documents, split into chunks before extraction.
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RELATION_EXTRACTION_PROMPT},
            {"role": "user", "content": f"Document ID: {doc.doc_id}\nSource: {doc.path}\n\nTEXT:\n{doc.text}"},
        ],
    )
    data = parse_json_response(response.choices[0].message.content or "{}")
    entities = []
    for entity in data.get("entities", []):
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        entities.append({
            "name": str(entity["name"]).strip(),
            "description": str(entity.get("description", "")).strip(),
        })

    relationships = []
    sentences = split_sentences(doc.text)
    for idx, rel in enumerate(data.get("relationships", [])):
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        evidence = str(rel.get("evidence", "")).strip()
        if not source or not target or not evidence:
            continue
        # Reject evidence not found in source text to reduce unsupported edges.
        if evidence.casefold() not in doc.text.casefold():
            LOG.warning("Dropping relation with non-verbatim evidence in %s", doc.path)
            continue
        try:
            strength = int(round(float(rel.get("strength", 5))))
        except (TypeError, ValueError):
            strength = 5
        strength = max(1, min(10, strength))
        relationships.append({
            "source": source,
            "target": target,
            "relation_type": str(rel.get("relation_type", "related to")).strip() or "related to",
            "description": str(rel.get("description", "")).strip(),
            "evidence": evidence,
            "strength": strength,
            "doc_id": doc.doc_id,
            "chunk_id": f"{doc.doc_id}_chunk_0",
            "evidence_id": stable_id(f"{doc.doc_id}|{evidence.casefold()}", "ev_"),
            "source_path": doc.path,
        })
    return {"entities": entities, "relationships": relationships}


def extract_corpus(docs: list[Document], use_mock: bool, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = None
    if not use_mock:
        if OpenAI is None:
            raise ImportError("Install openai: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is missing. Copy .env.example to .env and set it.")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_entities: dict[str, dict[str, str]] = {}
    all_relations: list[dict[str, Any]] = []
    for doc in tqdm(docs, desc="Extracting documents"):
        extracted = mock_extract(doc) if use_mock else llm_extract(doc, client, model)
        for ent in extracted.get("entities", []):
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            key = normalize_name(name)
            if key not in all_entities:
                all_entities[key] = {"entity_key": key, "name": name, "description": str(ent.get("description", ""))}
            elif len(str(ent.get("description", ""))) > len(all_entities[key]["description"]):
                all_entities[key]["description"] = str(ent.get("description", ""))
        all_relations.extend(extracted.get("relationships", []))

    entities_df = pd.DataFrame(all_entities.values())
    relations_df = pd.DataFrame(all_relations)
    if relations_df.empty:
        raise RuntimeError("Extraction produced no relationships. Check the corpus or extractor.")
    # Canonicalize names with normalized entity keys; drop relations with missing nodes.
    name_lookup = {normalize_name(row["name"]): row["name"] for row in all_entities.values()}
    relations_df["source_key"] = relations_df["source"].map(normalize_name)
    relations_df["target_key"] = relations_df["target"].map(normalize_name)
    # Ensure relation endpoints exist even if the extractor omitted them in entities.
    for col in ["source", "target"]:
        key_col = f"{col}_key"
        for key, raw_name in zip(relations_df[key_col], relations_df[col]):
            if key and key not in all_entities:
                all_entities[key] = {"entity_key": key, "name": raw_name, "description": f"Entity mentioned in extracted relationship: {raw_name}"}
    entities_df = pd.DataFrame(all_entities.values())
    relations_df["source"] = relations_df["source_key"].map(lambda k: all_entities[k]["name"] if k in all_entities else k)
    relations_df["target"] = relations_df["target_key"].map(lambda k: all_entities[k]["name"] if k in all_entities else k)
    return entities_df, relations_df


def cosine_to_unit(cosine: float) -> float:
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def build_edge_table(
    entities_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    embedding_model_name: str,
    use_embeddings: bool = True,
) -> pd.DataFrame:
    """Aggregate distinct evidence by directed source/target/relation type."""
    entities = entities_df.copy()
    entities["entity_key"] = entities["entity_key"].map(normalize_name)
    description_by_key = {
        row["entity_key"]: (str(row.get("description", "")).strip() or str(row["name"]))
        for _, row in entities.iterrows()
    }
    if use_embeddings:
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers or run with --no-embeddings.")
        model = SentenceTransformer(embedding_model_name)
        keys = list(description_by_key)
        descriptions = [description_by_key[k] for k in keys]
        vectors = model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
        vector_by_key = {key: vectors[i] for i, key in enumerate(keys)}
    else:
        vector_by_key = {}

    rels = relations_df.copy()
    rels["source_key"] = rels["source"].map(normalize_name)
    rels["target_key"] = rels["target"].map(normalize_name)
    rels["relation_type"] = rels["relation_type"].fillna("related to").astype(str).str.lower().str.strip()
    rels["strength"] = pd.to_numeric(rels["strength"], errors="coerce").fillna(5).clip(1, 10)
    rels["evidence_id"] = rels["evidence_id"].fillna("").astype(str)
    rels["description"] = rels["description"].fillna("").astype(str)
    group_cols = ["source_key", "target_key", "relation_type"]
    records: list[dict[str, Any]] = []
    for (source_key, target_key, relation_type), group in rels.groupby(group_cols, dropna=False):
        evidence = group.drop_duplicates(subset=["evidence_id"]) if group["evidence_id"].ne("").any() else group.drop_duplicates(subset=["evidence"])
        strengths = evidence["strength"].astype(float).tolist()
        n = max(1, len(evidence))
        raw_strength_sum = float(sum(strengths))
        mean_strength = float(np.mean(strengths))
        if use_embeddings and source_key in vector_by_key and target_key in vector_by_key:
            cosine = float(np.dot(vector_by_key[source_key], vector_by_key[target_key]))
            similarity = cosine_to_unit(cosine)
        else:
            cosine, similarity = 0.0, 0.5
        records.append({
            "source_key": source_key,
            "target_key": target_key,
            "source": description_by_key.get(source_key, source_key),
            "target": description_by_key.get(target_key, target_key),
            "relation_type": relation_type,
            "descriptions": " || ".join(sorted(set(evidence["description"].astype(str))))[:4000],
            "evidence_ids": json.dumps(sorted(set(evidence["evidence_id"].astype(str)))),
            "evidence_count": n,
            "strength_sum": raw_strength_sum,
            "mean_strength": mean_strength,
            "frequency_raw": n,
            "cosine_similarity_raw": cosine,
            "similarity_unit": similarity,
            "evidence_texts": json.dumps(sorted(set(evidence["evidence"].astype(str)))),
            "source_paths": json.dumps(sorted(set(evidence["source_path"].astype(str)))) if "source_path" in evidence else "[]",
        })
    edge_df = pd.DataFrame(records)
    if edge_df.empty:
        raise RuntimeError("No aggregated edges available.")

    max_count = float(edge_df["evidence_count"].max())
    if max_count <= 1:
        edge_df["F"] = 1.0
    else:
        edge_df["F"] = np.log1p(edge_df["evidence_count"]) / math.log1p(max_count)
    edge_df["S"] = ((edge_df["mean_strength"] - 1.0) / 9.0).clip(0, 1)
    edge_df["C"] = edge_df["similarity_unit"].clip(0, 1)
    # Reproducible variants; official-style baseline uses the sum, as described
    # in the supplied methodology. It is not normalized to [0,1].
    edge_df["weight_official_graphrag_sum_strength"] = edge_df["strength_sum"].clip(lower=1e-9)
    edge_df["weight_frequency_only"] = edge_df["F"].clip(lower=1e-9)
    edge_df["weight_strength_only"] = edge_df["S"].clip(lower=1e-9)
    edge_df["weight_similarity_only"] = edge_df["C"].clip(lower=1e-9)
    edge_df["weight_frequency_strength"] = (0.5 * edge_df["F"] + 0.5 * edge_df["S"]).clip(lower=1e-9)
    edge_df["weight_composite"] = (0.4 * edge_df["F"] + 0.4 * edge_df["S"] + 0.2 * edge_df["C"]).clip(lower=1e-9)
    # Ablations: renormalize remaining coefficients.
    edge_df["weight_ablation_no_frequency"] = (0.4 / 0.6 * edge_df["S"] + 0.2 / 0.6 * edge_df["C"]).clip(lower=1e-9)
    edge_df["weight_ablation_no_strength"] = (0.4 / 0.6 * edge_df["F"] + 0.2 / 0.6 * edge_df["C"]).clip(lower=1e-9)
    edge_df["weight_ablation_no_similarity"] = (0.5 * edge_df["F"] + 0.5 * edge_df["S"]).clip(lower=1e-9)
    return edge_df


VARIANTS = {
    "A_official_graphrag_sum_strength": "weight_official_graphrag_sum_strength",
    "B_frequency_only": "weight_frequency_only",
    "C_strength_only": "weight_strength_only",
    "D_similarity_only": "weight_similarity_only",
    "E_frequency_plus_strength": "weight_frequency_strength",
    "F_composite_proposed": "weight_composite",
    "G_ablation_no_frequency": "weight_ablation_no_frequency",
    "H_ablation_no_strength": "weight_ablation_no_strength",
    "I_ablation_no_similarity": "weight_ablation_no_similarity",
}


def make_graph(edge_df: pd.DataFrame, weight_column: str) -> tuple[ig.Graph, list[str]]:
    nodes = sorted(set(edge_df["source_key"].astype(str)) | set(edge_df["target_key"].astype(str)))
    index = {name: i for i, name in enumerate(nodes)}
    # Leiden on undirected graph; combine opposite-direction and parallel relation edges.
    pair_weights: dict[tuple[int, int], float] = defaultdict(float)
    for _, row in edge_df.iterrows():
        a, b = index[str(row["source_key"])], index[str(row["target_key"])]
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        pair_weights[pair] += max(float(row[weight_column]), 1e-9)
    graph = ig.Graph(n=len(nodes), directed=False)
    if pair_weights:
        graph.add_edges(list(pair_weights.keys()))
        graph.es["weight"] = list(pair_weights.values())
    else:
        graph.es["weight"] = []
    graph.vs["name"] = nodes
    return graph, nodes


def run_leiden(graph: ig.Graph, seed: int, resolution: float) -> list[int]:
    if graph.vcount() == 0:
        return []
    if graph.ecount() == 0:
        return list(range(graph.vcount()))
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        seed=int(seed),
    )
    membership = list(partition.membership)
    # Isolates may be assigned to singleton communities by the partitioner;
    # this remapping simply makes labels compact and deterministic.
    mapping: dict[int, int] = {}
    return [mapping.setdefault(label, len(mapping)) for label in membership]


def load_labels(path: Path = LABELS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"entity", "topic"}.issubset(df.columns):
        LOG.warning("labels.csv must have columns entity,topic; ignoring labels.")
        return {}
    return {
        normalize_name(row["entity"]): str(row["topic"])
        for _, row in df.dropna(subset=["entity", "topic"]).iterrows()
    }


def partition_metrics(
    graph: ig.Graph,
    nodes: list[str],
    membership: list[int],
    labels: dict[str, str],
    stability_membership: list[int] | None = None,
) -> dict[str, Any]:
    sizes = Counter(membership)
    singleton_nodes = sum(size for size in sizes.values() if size == 1)
    n = len(nodes)
    labelled = [(i, labels[node]) for i, node in enumerate(nodes) if node in labels]
    if len(labelled) >= 2 and len({label for _, label in labelled}) >= 2:
        y_true = [label for _, label in labelled]
        y_pred = [membership[i] for i, _ in labelled]
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        ari = float(adjusted_rand_score(y_true, y_pred))
    else:
        nmi, ari = np.nan, np.nan

    if graph.ecount():
        components = len(graph.connected_components(mode="weak"))
        weighted_modularity = float(graph.modularity(membership, weights=graph.es["weight"]))
    else:
        components = graph.vcount()
        weighted_modularity = np.nan

    result = {
        "n_nodes": n,
        "n_edges": graph.ecount(),
        "n_communities": len(sizes),
        "singleton_share": singleton_nodes / n if n else np.nan,
        "connected_components": components,
        "isolated_nodes": sum(1 for degree in graph.degree() if degree == 0),
        "min_community_size": min(sizes.values()) if sizes else 0,
        "max_community_size": max(sizes.values()) if sizes else 0,
        "mean_community_size": float(np.mean(list(sizes.values()))) if sizes else np.nan,
        "weighted_modularity_diagnostic": weighted_modularity,
        "n_labelled_nodes": len(labelled),
        "NMI": nmi,
        "ARI": ari,
    }
    if stability_membership is not None and len(stability_membership) == len(membership):
        result["stability_ARI_vs_previous_seed"] = float(adjusted_rand_score(stability_membership, membership))
    else:
        result["stability_ARI_vs_previous_seed"] = np.nan
    return result


def sample_documents(docs: list[Document], fraction: float, seed: int) -> list[Document]:
    if fraction >= 0.999999:
        return list(docs)
    count = max(1, int(math.ceil(len(docs) * fraction)))
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(docs), size=min(count, len(docs)), replace=False).tolist())
    return [docs[i] for i in selected]


def extract_to_cache(
    docs: list[Document],
    cache_dir: Path,
    use_mock: bool,
    model: str,
    cache_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ent_path = cache_dir / f"entities_{cache_tag}.csv"
    rel_path = cache_dir / f"relationships_{cache_tag}.csv"
    if ent_path.exists() and rel_path.exists():
        LOG.info("Using extraction cache: %s", cache_tag)
        return pd.read_csv(ent_path), pd.read_csv(rel_path)
    ent, rel = extract_corpus(docs, use_mock=use_mock, model=model)
    ent.to_csv(ent_path, index=False)
    rel.to_csv(rel_path, index=False)
    return ent, rel


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "NMI", "ARI", "stability_ARI_vs_previous_seed", "singleton_share",
        "connected_components", "isolated_nodes", "n_communities",
        "weighted_modularity_diagnostic", "n_nodes", "n_edges",
    ]
    rows = []
    group_cols = ["variant", "corpus_fraction"]
    for keys, group in results.groupby(group_cols, dropna=False):
        row = {"variant": keys[0], "corpus_fraction": keys[1], "runs": len(group)}
        for metric in metric_columns:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_DIR / "cache"
    docs = load_documents(Path(args.corpus_dir))
    labels = load_labels(Path(args.labels))

    use_mock = args.mock_extraction
    model_name = args.openai_model
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    if any(f <= 0 or f > 1 for f in fractions):
        raise ValueError("--fractions values must be in (0, 1].")

    all_results: list[dict[str, Any]] = []
    all_edge_tables: list[pd.DataFrame] = []
    extraction_log: list[dict[str, Any]] = []
    run_counter = 0

    for fraction_index, fraction in enumerate(fractions):
        for repeat in range(args.repeats):
            seed = args.seed + fraction_index * 10000 + repeat
            sampled_docs = sample_documents(docs, fraction, seed)
            sample_signature = stable_id("|".join(sorted(d.doc_id for d in sampled_docs)), "sample_")
            LOG.info("Fraction %.2f | repeat %d/%d | docs=%d", fraction, repeat + 1, args.repeats, len(sampled_docs))

            entities_df, relations_df = extract_to_cache(
                sampled_docs, cache_dir, use_mock, model_name, sample_signature
            )
            extraction_log.append({
                "sample_id": sample_signature,
                "corpus_fraction": fraction,
                "repeat": repeat,
                "n_documents": len(sampled_docs),
                "n_entities": len(entities_df),
                "n_relationship_instances": len(relations_df),
                "extractor": "mock" if use_mock else model_name,
            })

            edge_df = build_edge_table(
                entities_df,
                relations_df,
                embedding_model_name=args.embedding_model,
                use_embeddings=not args.no_embeddings,
            )
            edge_df.insert(0, "sample_id", sample_signature)
            edge_df.insert(1, "corpus_fraction", fraction)
            edge_df.insert(2, "repeat", repeat)
            all_edge_tables.append(edge_df.copy())

            previous_memberships: dict[str, list[int]] = {}
            for variant, weight_column in VARIANTS.items():
                graph, nodes = make_graph(edge_df, weight_column)
                # One fixed seed per sample for matched primary comparisons.
                membership = run_leiden(graph, seed=seed, resolution=args.resolution)
                # Estimate stability by comparing with an adjacent seed on the same graph.
                membership_next = run_leiden(graph, seed=seed + 1, resolution=args.resolution)
                metrics = partition_metrics(graph, nodes, membership, labels)
                metrics["stability_ARI_vs_adjacent_seed"] = (
                    float(adjusted_rand_score(membership, membership_next))
                    if len(membership) == len(membership_next) and membership else np.nan
                )
                all_results.append({
                    "run_id": run_counter,
                    "sample_id": sample_signature,
                    "corpus_fraction": fraction,
                    "repeat": repeat,
                    "seed": seed,
                    "variant": variant,
                    "weight_column": weight_column,
                    "resolution": args.resolution,
                    **metrics,
                })
                run_counter += 1

    results_df = pd.DataFrame(all_results)
    edge_tables_df = pd.concat(all_edge_tables, ignore_index=True) if all_edge_tables else pd.DataFrame()
    summary_df = summarize_results(results_df) if not results_df.empty else pd.DataFrame()
    results_df.to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    edge_tables_df.to_csv(OUTPUT_DIR / "edge_weights.csv", index=False)
    pd.DataFrame(extraction_log).to_csv(OUTPUT_DIR / "extraction_runs.csv", index=False)

    # Persist run configuration for reproducibility.
    config = {
        "seed": args.seed,
        "repeats": args.repeats,
        "fractions": fractions,
        "resolution": args.resolution,
        "mock_extraction": use_mock,
        "openai_model": model_name if not use_mock else None,
        "embedding_model": args.embedding_model,
        "embeddings_enabled": not args.no_embeddings,
        "variants": VARIANTS,
        "weight_formulas": {
            "F": "log(1 + distinct_evidence_count) / log(1 + max_count); if max_count <= 1, F=1",
            "S": "(mean_strength - 1) / 9",
            "C": "(cosine_similarity + 1) / 2",
            "composite": "0.4F + 0.4S + 0.2C",
            "frequency_strength": "0.5F + 0.5S",
            "official_baseline_note": "Sum of extracted strength scores; verify against a pinned official GraphRAG release.",
        },
        "created_at_unix": time.time(),
    }
    (OUTPUT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    LOG.info("Finished. Outputs written to %s", OUTPUT_DIR)


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="GraphRAG composite edge-weighting experiment runner")
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--outputs", default=str(OUTPUT_DIR), help="Reserved for future output-directory override.")
    parser.add_argument("--fractions", default=os.getenv("CORPUS_FRACTIONS", "0.25,0.50,0.75,1.0"))
    parser.add_argument("--repeats", type=int, default=int(os.getenv("N_REPEATS", "10")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--resolution", type=float, default=float(os.getenv("LEIDEN_RESOLUTION", "1.0")))
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    parser.add_argument(
        "--mock-extraction", action="store_true",
        default=os.getenv("MOCK_EXTRACTION", "0").lower() in {"1", "true", "yes"},
        help="Use the rule-based demonstration extractor instead of an LLM.",
    )
    parser.add_argument("--no-embeddings", action="store_true", help="Use neutral similarity C=0.5; mainly for offline smoke tests.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        LOG.exception("Experiment failed: %s", exc)
        sys.exit(1)
