#!/usr/bin/env python3
"""
cluster_themes.py — Unsupervised thematic clustering of scholarly papers.

Loads search_results.csv (produced by search_papers.py), generates MiniLM
embeddings, clusters papers by semantic similarity, extracts keywords and
theme labels, and saves results to CSV and JSON.

Adaptive mode (default)
-----------------------
When fewer than 50 papers are detected, the script automatically runs three
methods (NMF, hierarchical, fixed k=4) and produces consensus themes for
greater robustness.

Clustering modes (manual override)
----------------------------------
  auto         – silhouette-score–based selection of k (KMeans)
  fixed        – user-specified cluster count (--clusters N)
  hierarchical – Agglomerative clustering with automatic cut

Topic modeling (manual override)
--------------------------------
  tfidf  – keyword extraction via TF-IDF centroids (default)
  nmf    – Non-Negative Matrix Factorization on TF-IDF
  lda    – Latent Dirichlet Allocation on TF-IDF

Usage
-----
    python cluster_themes.py [--input search_results.csv]
                             [--output-prefix clustered_papers]
                             [--mode auto|fixed|hierarchical]
                             [--clusters N]
                             [--topic-modeling tfidf|nmf|lda]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cluster_themes")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CLUSTERS = 2
MAX_CLUSTERS = 15
DEFAULT_N_KEYWORDS = 5
SMALL_DATASET_THRESHOLD = 50
SMALL_DATASET_DEFAULT_K = 4
RANDOM_STATE = 42
CONSENSUS_SIMILARITY_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(input_path: str) -> pd.DataFrame:
    """Load search results CSV, deduplicate, and select relevant columns."""
    df = pd.read_csv(input_path)

    required = {"title", "abstract", "authors", "year"}
    missing = required - set(df.columns)
    if missing:
        log.warning("Missing columns: %s. Proceeding with available data.", missing)

    if "doi" in df.columns:
        df = df.drop_duplicates(subset="doi", keep="first")
    if "title" in df.columns:
        df = df.drop_duplicates(subset="title", keep="first")

    log.info("Loaded %d unique papers from %s", len(df), input_path)
    return df


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combine title + abstract into a single 'text' column for analysis."""
    texts: List[str] = []
    for _, row in df.iterrows():
        title = _clean(row.get("title", ""))
        abstract = _clean(row.get("abstract", ""))
        combined = f"{title}. {abstract}" if abstract else title
        texts.append(combined)
    df = df.copy()
    df["text"] = texts
    return df


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------
def generate_embeddings(
    texts: List[str], model_name: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Generate sentence embeddings using the MiniLM model."""
    log.info("Loading model: %s", model_name)
    model = SentenceTransformer(model_name)
    log.info("Generating embeddings for %d texts...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = normalize(embeddings, axis=1)
    log.info("Embedding shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# Cluster count selection
# ---------------------------------------------------------------------------
def _estimate_clusters(
    embeddings: np.ndarray,
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
) -> Tuple[int, float, List[Dict]]:
    """Use silhouette score to pick best k. Returns (best_k, best_score, trial_log)."""
    n_samples = embeddings.shape[0]
    if n_samples <= min_clusters:
        return min_clusters, 0.0, []

    max_k = min(max_clusters, n_samples - 1)
    best_k = min_clusters
    best_score = -1.0
    trial_log: List[Dict] = []

    for k in range(MIN_CLUSTERS, max_k + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        labels = km.fit_predict(embeddings)
        n_unique = len(set(labels))
        if n_unique < 2:
            trial_log.append({"k": k, "silhouette": None, "reason": "fewer than 2 non-empty clusters"})
            continue
        score = silhouette_score(embeddings, labels)
        trial_log.append({"k": k, "silhouette": round(float(score), 4), "reason": "valid"})
        if score > best_score:
            best_score = score
            best_k = k

    log.info("Estimated optimal clusters: %d (silhouette: %.3f)", best_k, best_score)
    return best_k, best_score, trial_log


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _run_kmeans(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, Any]:
    log.info("KMeans clustering with k=%d", k)
    model = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_STATE, batch_size=256, n_init="auto",
    )
    labels = model.fit_predict(embeddings)
    return labels, model


def _run_agglomerative(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, Any]:
    log.info("Agglomerative clustering with k=%d", k)
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = model.fit_predict(embeddings)
    return labels, model


def _cluster_size_summary(labels: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def cluster_papers(
    embeddings: np.ndarray,
    mode: str = "auto",
    n_clusters: Optional[int] = None,
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
) -> Tuple[np.ndarray, Any, Dict]:
    """
    Cluster embeddings according to *mode*.

    Returns (labels, model, info_dict) where info_dict contains
    selection metadata for the report.
    """
    n_samples = embeddings.shape[0]
    info: Dict[str, Any] = {"mode": mode, "n_samples": n_samples}

    if mode == "fixed":
        k = n_clusters if n_clusters else min_clusters
        if k >= n_samples:
            k = max(min_clusters, n_samples - 1)
            log.warning("Clamping k to %d (n_samples - 1)", k)
        info["selection_method"] = "user-specified"
        info["n_clusters"] = k
        labels, model = _run_kmeans(embeddings, k)
        info["trial_log"] = []

    elif mode == "hierarchical":
        max_k = min(max_clusters, n_samples - 1)
        if n_clusters:
            k = n_clusters
        else:
            k = min(max_k, SMALL_DATASET_DEFAULT_K)
        if k >= n_samples:
            k = max(min_clusters, n_samples - 1)
        info["selection_method"] = "user-specified" if n_clusters else "default"
        info["n_clusters"] = k
        labels, model = _run_agglomerative(embeddings, k)
        info["trial_log"] = []

    else:
        k, score, trial_log = _estimate_clusters(embeddings, min_clusters, max_clusters)
        info["selection_method"] = "silhouette-optimisation"
        info["n_clusters"] = k
        info["best_silhouette"] = round(score, 4)
        info["trial_log"] = trial_log
        labels, model = _run_kmeans(embeddings, k)

    info["cluster_sizes"] = _cluster_size_summary(labels)
    return labels, model, info


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def compute_quality_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """Return silhouette score and Davies–Bouldin score."""
    metrics: Dict[str, Any] = {}
    n_unique = len(set(labels))
    if n_unique >= 2:
        metrics["silhouette_score"] = round(float(silhouette_score(embeddings, labels)), 4)
        metrics["davies_bouldin_score"] = round(float(davies_bouldin_score(embeddings, labels)), 4)
    else:
        metrics["silhouette_score"] = None
        metrics["davies_bouldin_score"] = None
    return metrics


# ---------------------------------------------------------------------------
# Keyword extraction (TF-IDF centroids)
# ---------------------------------------------------------------------------
def extract_keywords_tfidf(
    texts: List[str], labels: Optional[np.ndarray] = None,
    n_keywords: int = DEFAULT_N_KEYWORDS,
) -> Dict[int, List[str]]:
    """Extract representative keywords per cluster via TF-IDF centroids."""
    vectorizer = CountVectorizer(
        max_df=0.85, min_df=1, stop_words="english", ngram_range=(1, 2), max_features=5000,
    )
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    if labels is None:
        labels_arr = np.zeros(len(texts), dtype=int)
    else:
        labels_arr = labels

    keywords: Dict[int, List[str]] = {}
    for label in sorted(set(labels_arr)):
        mask = labels_arr == label
        centroid = dtm[mask].mean(axis=0).A1
        top_idx = centroid.argsort()[::-1][:n_keywords]
        keywords[label] = [feature_names[i] for i in top_idx]
    return keywords


# ---------------------------------------------------------------------------
# Topic modeling: NMF
# ---------------------------------------------------------------------------
def extract_keywords_nmf(
    texts: List[str], n_topics: int, n_keywords: int = DEFAULT_N_KEYWORDS
) -> Tuple[Dict[int, List[str]], np.ndarray, Any]:
    """Topic keywords via Non-Negative Matrix Factorization on TF-IDF."""
    vectorizer = TfidfVectorizer(
        max_df=0.85, min_df=2, stop_words="english", ngram_range=(1, 2), max_features=5000,
    )
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = min(n_topics, dtm.shape[1] - 1, dtm.shape[0] - 1)
    n_topics = max(2, n_topics)

    model = NMF(n_components=n_topics, random_state=RANDOM_STATE, init="nndsvda")
    W = model.fit_transform(dtm)
    H = model.components_

    keywords: Dict[int, List[str]] = {}
    for topic_id in range(n_topics):
        top_idx = H[topic_id].argsort()[::-1][:n_keywords]
        keywords[topic_id] = [feature_names[i] for i in top_idx]
    return keywords, W, model


# ---------------------------------------------------------------------------
# Topic modeling: LDA
# ---------------------------------------------------------------------------
def extract_keywords_lda(
    texts: List[str], n_topics: int, n_keywords: int = DEFAULT_N_KEYWORDS
) -> Tuple[Dict[int, List[str]], Any]:
    """Topic keywords via Latent Dirichlet Allocation on raw counts."""
    vectorizer = CountVectorizer(
        max_df=0.85, min_df=2, stop_words="english", max_features=5000,
    )
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = min(n_topics, dtm.shape[1] - 1, dtm.shape[0] - 1)
    n_topics = max(2, n_topics)

    model = LatentDirichletAllocation(
        n_components=n_topics, random_state=RANDOM_STATE, max_iter=100,
        learning_method="batch",
    )
    model.fit(dtm)

    keywords: Dict[int, List[str]] = {}
    for topic_id in range(n_topics):
        top_idx = model.components_[topic_id].argsort()[::-1][:n_keywords]
        keywords[topic_id] = [feature_names[i] for i in top_idx]
    return keywords, model


# ---------------------------------------------------------------------------
# Theme label generation
# ---------------------------------------------------------------------------
ARTIFACTS: Set[str] = {
    "nan", "jats", "null", "none", "article", "study", "na", "n a",
    "et al", "unavailable", "unknown", "abstract", "introduction",
    "method", "result", "conclusion", "background", "objective",
    "purpose", "approach", "analysis", "data", "based", "using",
    "also", "however", "thus", "therefore", "well", "within",
    "across", "among", "although", "because", "between", "both",
    "each", "more", "most", "much", "other", "some", "such", "than",
    "that", "these", "this", "very", "many",
}


def clean_keywords(keywords_list: List[str]) -> List[str]:
    """Remove artifacts and stopwords from a keyword list."""
    cleaned = []
    for kw in keywords_list:
        kw_clean = kw.strip().lower().replace("_", " ").replace("-", " ")
        if not kw_clean or kw_clean in ARTIFACTS:
            continue
        if len(kw_clean) <= 2:
            continue
        cleaned.append(kw_clean.title())
    return cleaned


def _generate_qwen_theme_label(keywords_list: List[str]) -> str:
    """Use Qwen 2.5 to generate a human-readable research theme from keywords."""
    try:
        from src.agents.qwen_adapter import QwenAdapter
        qwen = QwenAdapter()
        kw_str = ", ".join(keywords_list[:8])
        prompt = (
            f"Given these research keywords: {kw_str}\n\n"
            "Generate one concise, human-readable research theme label (max 15 words). "
            "Return ONLY the label text, no explanation, no punctuation at end."
        )
        result = qwen._call_model(prompt, max_new_tokens=40)
        if result:
            label = result.strip().strip('"').strip("'")
            if len(label) > 10 and len(label.split()) <= 20:
                return label
    except Exception:
        pass
    return ", ".join(keywords_list[:5])


def generate_theme_labels(
    keywords: Dict[int, List[str]], n_keywords: int = DEFAULT_N_KEYWORDS
) -> Dict[int, str]:
    """Create a human-readable research theme label from cluster keywords.

    Cleans artifacts, then uses Qwen 2.5 to summarize into a coherent
    research theme. Falls back to cleaned keyword list if Qwen unavailable.
    """
    labels: Dict[int, str] = {}
    for cluster_id, words in keywords.items():
        cleaned = clean_keywords(words)
        if not cleaned:
            labels[cluster_id] = "Unnamed Theme"
            continue
        label = _generate_qwen_theme_label(cleaned[:n_keywords])
        labels[cluster_id] = label
    return labels


# ---------------------------------------------------------------------------
# Representative papers
# ---------------------------------------------------------------------------
def _find_representatives(
    embeddings: np.ndarray, labels: np.ndarray, df: pd.DataFrame
) -> Dict[int, List[Dict]]:
    """Return papers closest to each cluster centroid."""
    reps: Dict[int, List[Dict]] = {}
    for label in sorted(set(labels)):
        mask = labels == label
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        distances = cdist([centroid], cluster_embs, metric="cosine").flatten()
        closest = distances.argsort()[:3]
        papers = []
        for idx in closest:
            row = df.iloc[mask].iloc[idx]
            papers.append(
                {
                    "title": row.get("title", ""),
                    "authors": row.get("authors", ""),
                    "year": row.get("year", 0),
                    "doi": row.get("doi", ""),
                }
            )
        reps[label] = papers
    return reps


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    cluster_info: Dict,
    quality_metrics: Dict,
    n_papers: int,
) -> str:
    """Produce a plain-text report explaining cluster selection decisions."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("CLUSTERING REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Papers analysed:       {n_papers}")
    lines.append(f"Clustering mode:       {cluster_info.get('mode', 'auto')}")
    lines.append(f"Selection method:      {cluster_info.get('selection_method', 'N/A')}")
    lines.append(f"Number of clusters:    {cluster_info.get('n_clusters', 'N/A')}")
    lines.append("")
    lines.append("--- Quality Metrics ---")
    for key, val in quality_metrics.items():
        if val is not None:
            lines.append(f"  {key:<25} {val}")
        else:
            lines.append(f"  {key:<25} N/A (fewer than 2 clusters)")
    lines.append("")
    lines.append("--- Cluster sizes ---")
    sizes = cluster_info.get("cluster_sizes", {})
    for cid in sorted(sizes):
        lines.append(f"  Cluster {cid:<4} {sizes[cid]} papers")
    lines.append("")

    method = cluster_info.get("selection_method", "")
    if method == "silhouette-optimisation":
        lines.append("--- How k was chosen ---")
        lines.append(
            "KMeans was run for k = 2 .. 15 (or n-1). The k with the highest "
            "silhouette score (range -1 to 1, higher = better separation) "
            "was selected."
        )
        best_sil = cluster_info.get("best_silhouette")
        if best_sil is not None:
            lines.append(f"Best silhouette score: {best_sil}")
        trial_log = cluster_info.get("trial_log", [])
        if trial_log:
            lines.append("")
            lines.append("Trial log (k -> silhouette):")
            for trial in trial_log:
                s = trial.get("silhouette")
                if s is not None:
                    lines.append(f"  k={trial['k']:<2}  {s}")
                else:
                    lines.append(f"  k={trial['k']:<2}  {trial.get('reason', 'N/A')}")
    elif method == "user-specified":
        lines.append("--- How k was chosen ---")
        lines.append(
            f"k was set to {cluster_info.get('n_clusters')} by the user "
            "(--clusters / --mode fixed)."
        )
    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Save helpers (standard pipeline)
# ---------------------------------------------------------------------------
def save_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    theme_labels: Dict[int, str],
    keywords: Dict[int, List[str]],
    representatives: Dict[int, List[Dict]],
    quality_metrics: Dict,
    cluster_info: Dict,
    prefix: str = "clustered_papers",
) -> None:
    """Save clustered papers, theme summary, quality metrics, and report."""
    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out["theme"] = df_out["cluster_id"].map(theme_labels)
    df_out = df_out.drop(columns=["text"], errors="ignore")

    csv_path = f"{prefix}.csv"
    json_path = f"{prefix}.json"
    df_out.to_csv(csv_path, index=False)
    log.info("Saved %d clustered papers -> %s", len(df_out), csv_path)
    with open(json_path, "w") as f:
        json.dump(df_out.to_dict(orient="records"), f, indent=2, default=str)
    log.info("Saved %d clustered papers -> %s", len(df_out), json_path)

    theme_prefix = "theme_summary"
    summary_rows = []
    for label in sorted(set(labels)):
        count = int((labels == label).sum())
        summary_rows.append(
            {
                "cluster_id": label,
                "theme": theme_labels[label],
                "count": count,
                "keywords": ", ".join(keywords[label]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{theme_prefix}.csv", index=False)
    log.info("Theme summary -> %s", f"{theme_prefix}.csv")
    with open(f"{theme_prefix}.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    log.info("Theme summary -> %s", f"{theme_prefix}.json")

    metrics_path = "quality_metrics.json"
    metrics_payload = {
        **quality_metrics,
        "cluster_sizes": cluster_info.get("cluster_sizes", {}),
        "selection_method": cluster_info.get("selection_method", ""),
        "n_clusters": cluster_info.get("n_clusters", 0),
        "clustering_mode": cluster_info.get("mode", ""),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("Quality metrics -> %s", metrics_path)

    report = generate_report(cluster_info, quality_metrics, len(df_out))
    report_path = "clustering_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Clustering report -> %s", report_path)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def display_summary(
    theme_labels: Dict[int, str],
    keywords: Dict[int, List[str]],
    labels: np.ndarray,
) -> None:
    """Print theme summary table to terminal."""
    print("\n" + "=" * 100)
    print(f"{'Theme':<40} {'Papers':<8} {'Keywords'}")
    print("=" * 100)
    for label in sorted(set(labels)):
        count = int((labels == label).sum())
        kw = ", ".join(keywords[label][:3])
        theme = theme_labels[label][:37]
        print(f"{theme:<40} {count:<8} {kw}")
    print("=" * 100)
    print()


def print_statistics(labels: np.ndarray) -> None:
    """Print aggregate statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    print("--- Clustering Statistics ---")
    print(f"  Total papers analysed:   {len(labels)}")
    print(f"  Themes discovered:       {len(unique)}")
    print(f"  Largest theme:           {counts.max()} papers")
    print(f"  Smallest theme:          {counts.min()} papers")
    print()


# ===================================================================
# SMALL CORPUS ADAPTIVE ANALYSIS
# ===================================================================

def _method_output_path(method_name: str) -> str:
    """Map method name to output CSV file."""
    mapping = {
        "nmf": "nmf_themes.csv",
        "hierarchical": "hierarchical_themes.csv",
        "fixed4": "fixed4_themes.csv",
    }
    return mapping.get(method_name, f"{method_name}_themes.csv")


def _save_method_output(
    df: pd.DataFrame,
    labels: np.ndarray,
    theme_labels: Dict[int, str],
    keywords: Dict[int, List[str]],
    filepath: str,
) -> None:
    """Save one method's results to CSV."""
    out = df.copy()
    out["cluster_id"] = labels
    out["theme"] = out["cluster_id"].map(theme_labels)
    out = out.drop(columns=["text"], errors="ignore")
    out.to_csv(filepath, index=False)
    log.info("Saved (%s) -> %s", len(out), filepath)


def _run_single_nmf(
    texts: List[str], df: pd.DataFrame, n_topics: int, args: Any, embeddings: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, List[str]]]:
    """Run NMF topic modeling and return (labels, theme_labels, keywords)."""
    n_topics = min(n_topics, len(texts) - 1)
    n_topics = max(2, n_topics)
    keywords, W, _ = extract_keywords_nmf(texts, n_topics, n_keywords=args.keywords)
    theme_labels = generate_theme_labels(keywords, n_keywords=args.keywords)
    labels = W.argmax(axis=1)
    return labels, theme_labels, keywords


def _run_single_hierarchical(
    texts: List[str], df: pd.DataFrame, embeddings: np.ndarray, args: Any,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, List[str]]]:
    """Run hierarchical clustering and return (labels, theme_labels, keywords)."""
    k = min(SMALL_DATASET_DEFAULT_K, len(texts) - 1)
    labels, _ = _run_agglomerative(embeddings, k)
    keywords = extract_keywords_tfidf(texts, labels, n_keywords=args.keywords)
    theme_labels = generate_theme_labels(keywords, n_keywords=args.keywords)
    return labels, theme_labels, keywords


def _run_single_fixed4(
    texts: List[str], df: pd.DataFrame, embeddings: np.ndarray, args: Any,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, List[str]]]:
    """Run fixed k=4 KMeans and return (labels, theme_labels, keywords)."""
    k = min(4, len(texts) - 1)
    labels, _ = _run_kmeans(embeddings, k)
    keywords = extract_keywords_tfidf(texts, labels, n_keywords=args.keywords)
    theme_labels = generate_theme_labels(keywords, n_keywords=args.keywords)
    return labels, theme_labels, keywords


def _build_consensus(
    methods_results: List[Dict],
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Build consensus themes from multiple method results using MiniLM
    semantic similarity between theme embeddings.

    Each element in *methods_results* has:
      - name: str
      - labels: np.ndarray
      - theme_labels: Dict[int, str]
      - keywords: Dict[int, List[str]]

    Returns (consensus_themes, consensus_labels_for_each_paper).
    """
    n_papers = len(df)
    total_methods = len(methods_results)

    # Compute theme embedding for each (method, cluster_id) as mean of paper embeddings
    all_themes: List[Dict] = []
    for res in methods_results:
        mname = res["name"]
        for cid, kws in res["keywords"].items():
            mask = res["labels"] == cid
            theme_emb = embeddings[mask].mean(axis=0)
            all_themes.append({
                "method": mname,
                "cluster_id": cid,
                "label": res["theme_labels"][cid],
                "keywords": kws,
                "embedding": theme_emb,
            })

    # Merge themes using MiniLM cosine similarity
    merged: List[Dict] = []
    assigned = [False] * len(all_themes)

    for i in range(len(all_themes)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        emb_i = all_themes[i]["embedding"]
        for j in range(i + 1, len(all_themes)):
            if assigned[j]:
                continue
            emb_j = all_themes[j]["embedding"]
            sim = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-10))
            sim = max(-1.0, min(1.0, sim))
            if sim >= CONSENSUS_SIMILARITY_THRESHOLD:
                group.append(j)
                assigned[j] = True

        # Merge group into one consensus theme
        merged_kw: List[str] = []
        merged_methods: Set[str] = set()
        merged_cluster_ids: List[tuple] = []
        group_embeddings: List[np.ndarray] = []
        for idx in group:
            t = all_themes[idx]
            merged_kw.extend(t["keywords"])
            merged_methods.add(t["method"])
            merged_cluster_ids.append((t["method"], t["cluster_id"]))
            group_embeddings.append(t["embedding"])

        seen_kw: Set[str] = set()
        ordered_kw: List[str] = []
        for kw in merged_kw:
            kw_clean = kw.strip().lower()
            if kw_clean not in seen_kw and kw_clean not in ARTIFACTS:
                seen_kw.add(kw_clean)
                ordered_kw.append(kw.strip().title())

        label = _generate_qwen_theme_label(ordered_kw[:DEFAULT_N_KEYWORDS])
        theme_keywords = ordered_kw[:DEFAULT_N_KEYWORDS]

        # average_theme_strength: mean pairwise cosine sim among merged theme embeddings
        if len(group_embeddings) >= 2:
            pairwise_sims = []
            for a in range(len(group_embeddings)):
                for b in range(a + 1, len(group_embeddings)):
                    ea = group_embeddings[a]
                    eb = group_embeddings[b]
                    s = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-10))
                    pairwise_sims.append(max(-1.0, min(1.0, s)))
            avg_strength = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.0
        else:
            avg_strength = 1.0

        method_ratio = len(merged_methods) / total_methods
        confidence = method_ratio * avg_strength

        merged.append({
            "consensus_id": len(merged),
            "label": label,
            "keywords": ordered_kw[:DEFAULT_N_KEYWORDS],
            "methods": sorted(merged_methods),
            "confidence": round(confidence, 2),
            "average_theme_strength": round(avg_strength, 4),
            "method_cluster_ids": merged_cluster_ids,
            "is_exploratory": len(merged_methods) == 1,
        })

    # Build direct mapping: (method_name, cluster_id) -> consensus_id
    method_to_consensus: Dict[Tuple[str, int], int] = {}
    for ct in merged:
        for mc, cc in ct["method_cluster_ids"]:
            method_to_consensus[(mc, cc)] = ct["consensus_id"]

    # Determine consensus label per paper by majority vote
    paper_votes: List[List[int]] = [[] for _ in range(n_papers)]
    for res in methods_results:
        for pid in range(n_papers):
            cid = int(res["labels"][pid])
            key = (res["name"], cid)
            if key in method_to_consensus:
                paper_votes[pid].append(method_to_consensus[key])

    consensus_labels_arr = np.zeros(n_papers, dtype=int)
    for pid in range(n_papers):
        if paper_votes[pid]:
            counter = Counter(paper_votes[pid])
            consensus_labels_arr[pid] = counter.most_common(1)[0][0]
        else:
            consensus_labels_arr[pid] = -1

    # Filter out consensus themes with zero assigned papers
    assigned_counts: Counter = Counter(consensus_labels_arr)
    active_ids = {cid for cid in assigned_counts if cid >= 0}
    merged = [ct for ct in merged if ct["consensus_id"] in active_ids]

    # Re-number consensus IDs sequentially
    id_remap = {old: new for new, old in enumerate(sorted(active_ids))}
    for ct in merged:
        ct["consensus_id"] = id_remap[ct["consensus_id"]]
    for pid in range(n_papers):
        if consensus_labels_arr[pid] >= 0:
            consensus_labels_arr[pid] = id_remap[consensus_labels_arr[pid]]

    return merged, consensus_labels_arr


def _save_consensus_outputs(
    consensus_themes: List[Dict],
    consensus_labels: np.ndarray,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    """Save consensus_themes.csv, consensus_themes.json, theme_analysis_report.md."""
    # --- consensus_themes.csv / .json ---
    out = df.copy().drop(columns=["text"], errors="ignore")
    out["consensus_theme_id"] = consensus_labels

    label_map = {ct["consensus_id"]: ct["label"] for ct in consensus_themes}
    out["consensus_theme"] = out["consensus_theme_id"].map(label_map)

    out.to_csv("consensus_themes.csv", index=False)
    log.info("Saved %d consensus papers -> consensus_themes.csv", len(out))

    consensus_json = []
    for _, row in out.iterrows():
        consensus_json.append(row.to_dict())
    with open("consensus_themes.json", "w") as f:
        json.dump(consensus_json, f, indent=2, default=str)
    log.info("Saved %d consensus papers -> consensus_themes.json", len(out))

    # --- consensus_metadata.json ---
    sorted_by_id = sorted(consensus_themes, key=lambda ct: ct["consensus_id"])
    meta_list = []
    for ct in sorted_by_id:
        cid = ct["consensus_id"]
        count = int((consensus_labels == cid).sum())
        meta_list.append({
            "consensus_id": cid,
            "label": ct["label"],
            "keywords": ct["keywords"],
            "methods": ct["methods"],
            "confidence": ct["confidence"],
            "average_theme_strength": ct.get("average_theme_strength", 0.0),
            "is_exploratory": ct["is_exploratory"],
            "paper_count": count,
        })
    with open("consensus_metadata.json", "w") as f:
        json.dump(meta_list, f, indent=2)
    log.info("Consensus metadata -> consensus_metadata.json")

    # --- Representatives per consensus theme ---
    rep_map: Dict[int, List[Dict]] = {}
    for ct in consensus_themes:
        cid = ct["consensus_id"]
        mask = consensus_labels == cid
        if mask.sum() == 0:
            rep_map[cid] = []
            continue
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        distances = cdist([centroid], cluster_embs, metric="cosine").flatten()
        closest = distances.argsort()[:3]
        papers = []
        for idx in closest:
            row = df.iloc[mask].iloc[idx]
            papers.append({
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "year": row.get("year", 0),
                "doi": row.get("doi", ""),
            })
        rep_map[cid] = papers

    # --- theme_analysis_report.md ---
    report_lines: List[str] = []
    report_lines.append("# Theme Analysis Report")
    report_lines.append("")
    report_lines.append(f"**Corpus size:** {len(df)} papers")
    report_lines.append("")
    report_lines.append("## Methods Used")
    report_lines.append("")
    report_lines.append("| Method | Description |")
    report_lines.append("|--------|-------------|")
    report_lines.append("| NMF | Non-Negative Matrix Factorization on TF-IDF |")
    report_lines.append("| Hierarchical | Agglomerative clustering (cosine distance, average linkage) |")
    report_lines.append("| Fixed k=4 | KMeans clustering with 4 clusters |")
    report_lines.append("")
    report_lines.append("## Theme Summaries")
    report_lines.append("")
    report_lines.append(f"**Total themes discovered:** {len(consensus_themes)}")
    report_lines.append("")
    report_lines.append("| Theme | Papers | Confidence | Methods | Keywords | Exploratory |")
    report_lines.append("|-------|--------|------------|---------|----------|-------------|")

    sorted_themes = sorted(consensus_themes, key=lambda t: t["confidence"], reverse=True)
    for ct in sorted_themes:
        cid = ct["consensus_id"]
        count = int((consensus_labels == cid).sum())
        conf_pct = f"{ct['confidence'] * 100:.0f}%"
        methods_str = ", ".join(ct["methods"])
        kw_str = ", ".join(ct["keywords"])
        expl = "Yes" if ct["is_exploratory"] else "No"
        label_esc = ct["label"][:50]
        report_lines.append(f"| {label_esc} | {count} | {conf_pct} | {methods_str} | {kw_str} | {expl} |")

    report_lines.append("")
    report_lines.append("## Confidence Rankings")
    report_lines.append("")
    for rank, ct in enumerate(sorted_themes, 1):
        cid = ct["consensus_id"]
        count = int((consensus_labels == cid).sum())
        conf_pct = f"{ct['confidence'] * 100:.0f}%"
        methods_str = ", ".join(ct["methods"])
        report_lines.append(f"**{rank}. {ct['label']}** — Confidence: {conf_pct}, Papers: {count}, Methods: {methods_str}")
    report_lines.append("")

    report_lines.append("## Representative Papers")
    report_lines.append("")
    for ct in sorted_themes:
        cid = ct["consensus_id"]
        papers = rep_map.get(cid, [])
        report_lines.append(f"### {ct['label']}")
        report_lines.append("")
        for p in papers:
            authors = p.get("authors", "") or "N/A"
            year = p.get("year", "") or ""
            doi = p.get("doi", "") or ""
            report_lines.append(f"- **{p['title']}** ({year}) — {authors}")
            if doi:
                report_lines.append(f"  DOI: {doi}")
        report_lines.append("")

    report_lines.append("## Potential Research Gaps")
    report_lines.append("")
    report_lines.append("The following themes are marked as exploratory (identified by only one method):")
    report_lines.append("")
    exploratory = [ct for ct in sorted_themes if ct["is_exploratory"]]
    if exploratory:
        for ct in exploratory:
            report_lines.append(f"- **{ct['label']}** — identified only by {ct['methods'][0]}")
    else:
        report_lines.append("All themes were identified by multiple methods, increasing confidence.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Generated by cluster_themes.py (small corpus adaptive mode)*")

    report_text = "\n".join(report_lines)
    with open("theme_analysis_report.md", "w") as f:
        f.write(report_text)
    log.info("Theme analysis report -> theme_analysis_report.md")


def _display_consensus_summary(
    consensus_themes: List[Dict],
    consensus_labels: np.ndarray,
) -> None:
    """Print consensus theme summary to terminal."""
    sorted_themes = sorted(consensus_themes, key=lambda t: t["confidence"], reverse=True)

    print("\n" + "=" * 120)
    header = f"{'Consensus Theme':<45} {'Papers':<8} {'Conf':<8} {'Methods':<20} {'Exploratory':<12}"
    print(header)
    print("=" * 120)
    for ct in sorted_themes:
        cid = ct["consensus_id"]
        count = int((consensus_labels == cid).sum())
        conf_pct = f"{ct['confidence'] * 100:.0f}%"
        methods_str = ", ".join(ct["methods"])[:18]
        expl = "Yes" if ct["is_exploratory"] else "No"
        label = ct["label"][:42]
        print(f"{label:<45} {count:<8} {conf_pct:<8} {methods_str:<20} {expl:<12}")
    print("=" * 120)
    print()


# ===================================================================
# SMALL CORPUS PIPELINE
# ===================================================================

def run_small_corpus_pipeline(df: pd.DataFrame, embeddings: np.ndarray, args: Any) -> None:
    """Execute multi-method small corpus analysis and build consensus."""
    texts = df["text"].tolist()
    n_papers = len(texts)
    n_topics = min(SMALL_DATASET_DEFAULT_K, n_papers - 1)

    print("\nSmall corpus detected (<50 papers). Standard embedding clustering "
          "may produce unstable themes. Running enhanced small corpus analysis.")
    print("Small corpus mode activated. Running multi-method theme discovery "
          "for improved robustness.\n")

    # --- Method A: NMF ---
    log.info("=== Method A: NMF Topic Modeling ===")
    labels_nmf, themes_nmf, kw_nmf = _run_single_nmf(texts, df, n_topics, args, embeddings)
    _save_method_output(df, labels_nmf, themes_nmf, kw_nmf, "nmf_themes.csv")

    # --- Method B: Hierarchical ---
    log.info("=== Method B: Hierarchical Clustering ===")
    labels_hier, themes_hier, kw_hier = _run_single_hierarchical(texts, df, embeddings, args)
    _save_method_output(df, labels_hier, themes_hier, kw_hier, "hierarchical_themes.csv")

    # --- Method C: Fixed k=4 ---
    log.info("=== Method C: Fixed KMeans (k=4) ===")
    labels_fixed, themes_fixed, kw_fixed = _run_single_fixed4(texts, df, embeddings, args)
    _save_method_output(df, labels_fixed, themes_fixed, kw_fixed, "fixed4_themes.csv")

    # --- Build consensus ---
    log.info("=== Building consensus themes ===")
    methods_results = [
        {"name": "nmf", "labels": labels_nmf, "theme_labels": themes_nmf, "keywords": kw_nmf},
        {"name": "hierarchical", "labels": labels_hier, "theme_labels": themes_hier, "keywords": kw_hier},
        {"name": "fixed4", "labels": labels_fixed, "theme_labels": themes_fixed, "keywords": kw_fixed},
    ]
    consensus_themes, consensus_labels = _build_consensus(methods_results, df, embeddings)

    _save_consensus_outputs(consensus_themes, consensus_labels, df, embeddings)

    print("\n--- Small Corpus Mode Complete ---")
    print(f"  Papers analysed:     {n_papers}")
    print(f"  Methods used:        3 (NMF, Hierarchical, Fixed k=4)")
    print(f"  Consensus themes:    {len(consensus_themes)}")
    print(f"  Output files:")
    print(f"    nmf_themes.csv, hierarchical_themes.csv, fixed4_themes.csv")
    print(f"    consensus_themes.csv, consensus_themes.json")
    print(f"    theme_analysis_report.md\n")

    _display_consensus_summary(consensus_themes, consensus_labels)

    # Automatically run report generation
    log.info("Running generate_reports.py...")
    try:
        subprocess.run(
            [sys.executable, "generate_reports.py"],
            check=False,
            capture_output=False,
            timeout=120,
        )
    except FileNotFoundError:
        log.warning("generate_reports.py not found; skipping report generation.")
    except subprocess.TimeoutExpired:
        log.warning("generate_reports.py timed out; skipping.")
    log.info("Report generation complete.")

    # Automatically run theme evolution analysis
    log.info("Running theme_evolution.py...")
    try:
        subprocess.run(
            [sys.executable, "theme_evolution.py"],
            check=False,
            capture_output=False,
            timeout=120,
        )
    except FileNotFoundError:
        log.warning("theme_evolution.py not found; skipping theme evolution.")
    except subprocess.TimeoutExpired:
        log.warning("theme_evolution.py timed out; skipping.")
    log.info("Theme evolution analysis complete.")


# ===================================================================
# STANDARD PIPELINE (>= 50 papers)
# ===================================================================

def run_standard_pipeline(df: pd.DataFrame, embeddings: np.ndarray, args: Any) -> None:
    """Run standard single-method clustering pipeline."""
    n_papers = len(df)

    log.info("Running standard clustering pipeline (%d papers)", n_papers)

    labels, cluster_model, cluster_info = cluster_papers(
        embeddings, mode=args.mode, n_clusters=args.clusters,
        min_clusters=args.min_clusters, max_clusters=args.max_clusters,
    )

    quality_metrics = compute_quality_metrics(embeddings, labels)
    log.info("Quality metrics: %s", quality_metrics)

    n_topics = cluster_info["n_clusters"]
    if args.topic_modeling == "nmf":
        keywords, _, _ = extract_keywords_nmf(
            df["text"].tolist(), n_topics, n_keywords=args.keywords,
        )
    elif args.topic_modeling == "lda":
        keywords, _ = extract_keywords_lda(
            df["text"].tolist(), n_topics, n_keywords=args.keywords,
        )
    else:
        keywords = extract_keywords_tfidf(
            df["text"].tolist(), labels, n_keywords=args.keywords,
        )

    theme_labels = generate_theme_labels(keywords, n_keywords=args.keywords)
    representatives = _find_representatives(embeddings, labels, df)

    save_results(
        df, labels, theme_labels, keywords, representatives,
        quality_metrics, cluster_info, prefix=args.output_prefix,
    )

    print_statistics(labels)
    display_summary(theme_labels, keywords, labels)
    print("See clustering_report.txt for details on cluster selection.\n")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised thematic clustering of scholarly papers."
    )
    parser.add_argument(
        "--input", type=str, default="search_results.csv",
        help="Input CSV from search_papers.py (default: search_results.csv)",
    )
    parser.add_argument(
        "--output-prefix", type=str, default="clustered_papers",
        help="Prefix for output files (default: clustered_papers)",
    )
    parser.add_argument(
        "--clusters", type=int, default=None,
        help="Number of clusters (used with --mode fixed)",
    )
    parser.add_argument(
        "--keywords", type=int, default=DEFAULT_N_KEYWORDS,
        help=f"Keywords per cluster (default: {DEFAULT_N_KEYWORDS})",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["auto", "fixed", "hierarchical"],
        help="Clustering mode (default: adaptive — auto for >=50, multi-method for <50)",
    )
    parser.add_argument(
        "--topic-modeling", type=str, default="tfidf",
        choices=["tfidf", "nmf", "lda"],
        help="Topic modeling method for keyword extraction (default: tfidf)",
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="SentenceTransformer model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--min-clusters", type=int, default=MIN_CLUSTERS,
        help="Minimum clusters for auto mode",
    )
    parser.add_argument(
        "--max-clusters", type=int, default=MAX_CLUSTERS,
        help="Maximum clusters for auto mode",
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="Disable adaptive small corpus mode; always run single-method pipeline",
    )
    args = parser.parse_args()

    input_path = args.input
    if not Path(input_path).exists():
        log.error("Input file not found: %s", input_path)
        return

    df = load_data(input_path)
    df = preprocess_text(df)
    n_papers = len(df)

    embeddings = generate_embeddings(df["text"].tolist(), model_name=args.model)

    # Determine pipeline
    use_adaptive = (
        not args.no_adaptive
        and args.mode is None
        and n_papers < SMALL_DATASET_THRESHOLD
    )

    if use_adaptive:
        run_small_corpus_pipeline(df, embeddings, args)
    else:
        if args.mode is None:
            args.mode = "auto"
        run_standard_pipeline(df, embeddings, args)


if __name__ == "__main__":
    main()
