#!/usr/bin/env python3
"""
generate_reports.py — Generate executive summary, research gaps, literature map,
knowledge base, and RAG preparation from consensus analysis outputs.

Usage
-----
    python generate_reports.py
        [--consensus consensus_themes.csv]
        [--papers search_results.csv]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("generate_reports")

SMALL_CORPUS_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
DIRS = {
    "themes": Path("outputs") / "themes",
    "reports": Path("outputs") / "reports",
    "knowledge": Path("outputs") / "knowledge_base",
}


def _ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    log.info("Created directory structure under outputs/")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_csv(path: str, required: bool = True) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        if required:
            log.error("File not found: %s", path)
            return None
        log.warning("File not found (skipping): %s", path)
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Theme ranking
# ---------------------------------------------------------------------------
def _compute_theme_rankings(
    df_consensus: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Rank themes by paper count."""
    theme_groups = df_consensus.groupby("consensus_theme")
    rankings: List[Dict[str, Any]] = []

    for theme_name, group in theme_groups:
        rankings.append({
            "theme": theme_name,
            "paper_count": len(group),
            "papers": group["title"].tolist(),
        })

    rankings.sort(key=lambda r: r["paper_count"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1
    return rankings


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------
def generate_executive_summary(
    df_consensus: pd.DataFrame,
    df_papers: Optional[pd.DataFrame],
    rankings: List[Dict],
) -> str:
    """Generate executive_summary.md."""
    n_papers = len(df_consensus)
    n_themes = df_consensus["consensus_theme"].nunique()
    small_corpus = n_papers < SMALL_CORPUS_THRESHOLD

    lines: List[str] = []
    lines.append("# Executive Summary")
    lines.append("")

    if small_corpus:
        lines.append(
            "> **Note:** Theme structure derived from a small corpus and should "
            "be interpreted as exploratory rather than definitive."
        )
        lines.append("")

    lines.append("## Corpus Overview")
    lines.append("")
    lines.append(f"- **Total papers:** {n_papers}")
    lines.append(f"- **Themes discovered:** {n_themes}")
    if df_papers is not None:
        sources = df_papers["source"].value_counts().to_dict()
        lines.append(f"- **Sources:** {', '.join(f'{k} ({v})' for k, v in sources.items())}")
        year_min = df_papers["year"].min()
        year_max = df_papers["year"].max()
        if pd.notna(year_min) and pd.notna(year_max):
            lines.append(f"- **Year range:** {int(year_min)} – {int(year_max)}")
    lines.append("")

    lines.append("## Top Themes")
    lines.append("")
    for r in rankings[:5]:
        lines.append(f"**{r['rank']}. {r['theme']}** — {r['paper_count']} papers")
        lines.append("")
    lines.append("")

    lines.append("## Emerging Themes")
    lines.append("")
    emerging = [r for r in rankings if 1 <= r["paper_count"] <= 3]
    if emerging:
        for r in emerging:
            lines.append(f"- **{r['theme']}** ({r['paper_count']} papers)")
    else:
        lines.append("No clearly emerging themes identified; all themes are well-established.")
    lines.append("")

    lines.append("## Weak Themes")
    lines.append("")
    weak = [r for r in rankings if r["paper_count"] == 1]
    if weak:
        for r in weak:
            lines.append(f"- **{r['theme']}** — only {r['paper_count']} paper, limited evidence")
    else:
        lines.append("All themes have multiple supporting papers.")
    lines.append("")

    lines.append("## Most Representative Papers")
    lines.append("")
    for r in rankings[:3]:
        titles = r["papers"][:2]
        for t in titles:
            if t and str(t).strip():
                lines.append(f"- **{t}** — ({r['theme']})")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by generate_reports.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Research gaps report
# ---------------------------------------------------------------------------
def generate_research_gaps(
    df_consensus: pd.DataFrame,
    df_papers: Optional[pd.DataFrame],
    rankings: List[Dict],
) -> str:
    """Generate research_gaps.md."""
    n_papers = len(df_consensus)
    small_corpus = n_papers < SMALL_CORPUS_THRESHOLD

    lines: List[str] = []
    lines.append("# Research Gaps Report")
    lines.append("")

    if small_corpus:
        lines.append(
            "> **Note:** Theme structure derived from a small corpus and should "
            "be interpreted as exploratory rather than definitive."
        )
        lines.append("")

    lines.append("## Sparse Themes (Few Papers)")
    lines.append("")
    small = [r for r in rankings if r["paper_count"] <= 2]
    if small:
        for r in small:
            lines.append(f"- **{r['theme']}** ({r['paper_count']} papers) — "
                         f"limited evidence; conclusions should be drawn cautiously")
    else:
        lines.append("All themes have at least 3 supporting papers.")
    lines.append("")

    lines.append("## Understudied Areas")
    lines.append("")
    meta_path = Path("consensus_metadata.json")
    exploratory = []
    if meta_path.exists():
        with open(meta_path) as f:
            for entry in json.load(f):
                if entry.get("is_exploratory"):
                    exploratory.append({
                        "theme": entry["label"],
                        "method": entry["methods"][0] if entry.get("methods") else "N/A",
                    })
        if exploratory:
            for e in exploratory:
                lines.append(f"- **{e['theme']}** — identified only by {e['method']}")
        else:
            lines.append("All consensus themes were identified by multiple methods.")
    else:
        lines.append("(consensus_metadata.json not available.)")
    lines.append("")

    lines.append("## Weakly Connected Topics")
    lines.append("")
    weakly = [r for r in rankings if r["paper_count"] == 1]
    if weakly:
        for r in weakly:
            lines.append(f"- **{r['theme']}** — only 1 paper, poorly connected to rest of corpus")
    else:
        lines.append("All themes have good coverage.")
    lines.append("")

    lines.append("## Potential Future Directions")
    lines.append("")
    candidates = small + [{"theme": e["theme"], "paper_count": 0} for e in exploratory]
    seen_directions: Set[str] = set()
    for c in candidates:
        theme = c["theme"]
        if theme and theme not in seen_directions:
            lines.append(f"- Investigate **{theme}** further with targeted literature searches and empirical studies")
            seen_directions.add(theme)
    if not seen_directions:
        lines.append("All themes are well-supported; consider broadening the search query to discover new directions.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    if small_corpus:
        lines.append("- **Expand corpus:** Search additional databases and refine keywords to increase coverage.")
    lines.append("- **Deepen analysis:** For high-confidence themes, conduct systematic review and citation chaining.")
    lines.append("- **Validate findings:** Cross-reference with recent publications and domain experts.")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by generate_reports.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Literature map
# ---------------------------------------------------------------------------
def generate_literature_map(
    df_consensus: pd.DataFrame,
    rankings: List[Dict],
    consensus_meta: List[Dict],
) -> pd.DataFrame:
    """Generate literature_map.csv."""
    meta_by_label = {m["label"]: m for m in consensus_meta}

    rows: List[Dict] = []
    for r in rankings:
        theme = r["theme"]
        group = df_consensus[df_consensus["consensus_theme"] == theme]
        keywords = [w.strip().title() for w in theme.split(",") if w.strip()]
        rep_titles = group["title"].dropna().head(3).tolist()
        meta = meta_by_label.get(theme, {})
        confidence = meta.get("confidence", 0)

        rows.append({
            "Theme": theme,
            "Keywords": ", ".join(keywords[:5]),
            "Paper_Count": r["paper_count"],
            "Representative_Papers": " | ".join(rep_titles),
            "Confidence": confidence,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
def generate_knowledge_base(
    df_consensus: pd.DataFrame,
    df_papers: Optional[pd.DataFrame],
    rankings: List[Dict],
    consensus_meta: List[Dict],
) -> Dict:
    """Generate structured knowledge_base.json with themes/papers/authors/keywords/citations."""
    meta_by_label = {m["label"]: m for m in consensus_meta}

    # Papers
    papers_list: List[Dict] = []
    authors_set: Set[str] = set()
    keywords_set: Set[str] = set()
    for _, row in df_consensus.iterrows():
        papers_list.append({
            "title": row.get("title", ""),
            "authors": row.get("authors", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "abstract": str(row.get("abstract", ""))[:500],
            "doi": row.get("doi", ""),
            "venue": row.get("venue", ""),
            "source": row.get("source", ""),
            "url": row.get("url", ""),
            "citation_count": int(row["citation_count"]) if pd.notna(row.get("citation_count")) else 0,
            "consensus_theme": row.get("consensus_theme", ""),
            "consensus_theme_id": int(row["consensus_theme_id"]) if pd.notna(row.get("consensus_theme_id")) else -1,
        })
        for a in str(row.get("authors", "")).split(","):
            a = a.strip()
            if a:
                authors_set.add(a)

    # Themes
    theme_entries: List[Dict] = []
    for r in rankings:
        group = df_consensus[df_consensus["consensus_theme"] == r["theme"]]
        paper_titles = group["title"].dropna().tolist()
        kw = [w.strip().title() for w in r["theme"].split(",") if w.strip()]
        keywords_set.update(k.lower() for k in kw)
        meta = meta_by_label.get(r["theme"], {})
        theme_entries.append({
            "theme": r["theme"],
            "keywords": kw,
            "paper_count": r["paper_count"],
            "papers": paper_titles,
            "confidence_rank": r["rank"],
            "confidence": meta.get("confidence", 0),
            "average_theme_strength": meta.get("average_theme_strength", 0),
        })

    # Citations
    citations = []
    for p in papers_list:
        if p.get("doi"):
            citations.append({
                "title": p["title"],
                "doi": p["doi"],
                "citation_count": p["citation_count"],
            })

    return {
        "corpus_size": len(df_consensus),
        "theme_count": len(rankings),
        "themes": theme_entries,
        "papers": papers_list,
        "authors": sorted(authors_set),
        "keywords": sorted(keywords_set),
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# RAG preparation
# ---------------------------------------------------------------------------
def prepare_rag(
    df_consensus: pd.DataFrame,
    consensus_meta: List[Dict],
) -> None:
    """Generate rag_chunks.json and embeddings.pkl for RAG indexing."""
    meta_by_label = {m["label"]: m for m in consensus_meta}

    chunks: List[Dict] = []
    chunk_texts: List[str] = []
    for _, row in df_consensus.iterrows():
        theme = row.get("consensus_theme", "")
        meta = meta_by_label.get(theme, {})
        kw_str = ", ".join(meta.get("keywords", []))

        chunk = {
            "title": row.get("title", ""),
            "abstract": str(row.get("abstract", ""))[:1000],
            "theme": theme,
            "keywords": kw_str,
            "source": row.get("source", ""),
            "doi": row.get("doi", ""),
        }
        chunks.append(chunk)

        text = f"{chunk['title']}. {chunk['abstract']} {chunk['theme']} {kw_str}"
        chunk_texts.append(text)

    # Save rag_chunks.json
    chunks_path = DIRS["knowledge"] / "rag_chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2, default=str)
    log.info("Saved %d RAG chunks -> %s", len(chunks), chunks_path)

    # Generate MiniLM embeddings
    log.info("Generating MiniLM embeddings for RAG chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    log.info("Embeddings shape: %s", embeddings.shape)

    # Save embeddings.pkl
    pkl_path = DIRS["knowledge"] / "embeddings.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(embeddings, f)
    log.info("Saved embeddings -> %s", pkl_path)


# ---------------------------------------------------------------------------
# File organization
# ---------------------------------------------------------------------------
def _copy_to_outputs(src: str, dst_dir: Path) -> None:
    src_path = Path(src)
    if src_path.exists():
        dst = dst_dir / src_path.name
        shutil.copy2(str(src_path), str(dst))
        log.info("Copied %s -> %s", src, dst)


def organize_outputs() -> None:
    """Copy generated files into the proper output directories."""
    theme_files = [
        "nmf_themes.csv",
        "hierarchical_themes.csv",
        "fixed4_themes.csv",
        "consensus_themes.csv",
        "consensus_themes.json",
        "consensus_metadata.json",
    ]
    report_files = [
        "theme_analysis_report.md",
        "executive_summary.md",
        "research_gaps.md",
        "clustering_report.txt",
        "literature_map.csv",
    ]
    knowledge_files = [
        "knowledge_base.json",
    ]

    for f in theme_files:
        _copy_to_outputs(f, DIRS["themes"])
    for f in report_files:
        _copy_to_outputs(f, DIRS["reports"])
    for f in knowledge_files:
        _copy_to_outputs(f, DIRS["knowledge"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reports from consensus clustering outputs."
    )
    parser.add_argument(
        "--consensus", type=str, default="consensus_themes.csv",
        help="Consensus themes CSV (default: consensus_themes.csv)",
    )
    parser.add_argument(
        "--papers", type=str, default="search_results.csv",
        help="Original paper metadata CSV (default: search_results.csv)",
    )
    args = parser.parse_args()

    df_consensus = _load_csv(args.consensus)
    if df_consensus is None:
        return

    df_papers = _load_csv(args.papers, required=False)

    if "consensus_theme" not in df_consensus.columns:
        log.error("consensus_themes.csv missing 'consensus_theme' column")
        return

    # Load consensus metadata
    consensus_meta: List[Dict] = []
    meta_path = Path("consensus_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            consensus_meta = json.load(f)

    rankings = _compute_theme_rankings(df_consensus)
    _ensure_dirs()

    # 1. Executive summary
    log.info("Generating executive_summary.md...")
    exec_summary = generate_executive_summary(df_consensus, df_papers, rankings)
    with open("executive_summary.md", "w") as f:
        f.write(exec_summary)
    log.info("Saved -> executive_summary.md")

    # 2. Research gaps
    log.info("Generating research_gaps.md...")
    gaps = generate_research_gaps(df_consensus, df_papers, rankings)
    with open("research_gaps.md", "w") as f:
        f.write(gaps)
    log.info("Saved -> research_gaps.md")

    # 3. Literature map
    log.info("Generating literature_map.csv...")
    df_map = generate_literature_map(df_consensus, rankings, consensus_meta)
    df_map.to_csv("literature_map.csv", index=False)
    log.info("Saved -> literature_map.csv")

    # 4. Knowledge base
    log.info("Generating knowledge_base.json...")
    kb = generate_knowledge_base(df_consensus, df_papers, rankings, consensus_meta)
    with open("knowledge_base.json", "w") as f:
        json.dump(kb, f, indent=2, default=str)
    log.info("Saved -> knowledge_base.json")

    # 5. RAG preparation
    log.info("Preparing RAG chunks and embeddings...")
    prepare_rag(df_consensus, consensus_meta)

    # 6. Organize into output directories
    organize_outputs()
    log.info("All outputs organized under outputs/")

    # Print summary
    n_papers = len(df_consensus)
    n_themes = len(rankings)
    print("\n--- Report Generation Complete ---")
    print(f"  Corpus:              {n_papers} papers")
    print(f"  Themes:              {n_themes}")
    print(f"  Executive summary:   outputs/reports/executive_summary.md")
    print(f"  Research gaps:       outputs/reports/research_gaps.md")
    print(f"  Literature map:      outputs/reports/literature_map.csv")
    print(f"  Knowledge base:      outputs/knowledge_base/knowledge_base.json")
    print(f"  RAG chunks:          outputs/knowledge_base/rag_chunks.json")
    print(f"  Embeddings:          outputs/knowledge_base/embeddings.pkl")
    if n_papers < SMALL_CORPUS_THRESHOLD:
        print(
            "  Note: Theme structure derived from a small corpus and should "
            "be interpreted as exploratory rather than definitive."
        )
    print()


if __name__ == "__main__":
    main()
