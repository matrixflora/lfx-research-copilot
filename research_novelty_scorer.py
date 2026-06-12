#!/usr/bin/env python3
"""
research_novelty_scorer.py — Estimate the novelty of each research theme
based on topic saturation, publication density, citation density, emerging
concepts, and similarity to existing literature.

Outputs
-------
outputs/reports/novelty_scores.csv
outputs/reports/novelty_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("novelty_scorer")

MODEL_NAME = "all-MiniLM-L6-v2"

EMERGING_TERMS = [
    "artificial intelligence", "machine learning", "deep learning", "neural",
    "blockchain", "digital twin", "internet of things", "iot", "big data",
    "generative", "transformer", "large language model", "llm",
    "precision agriculture", "climate resilience", "sustainable",
    "equity", "inclusion", "digital divide", "capacity building",
]


def score_novelty(kb_path: str = "knowledge_base.json",
                  papers_path: str = "search_results.csv") -> pd.DataFrame:
    kb = json.load(open(kb_path)) if Path(kb_path).exists() else {}
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()

    if not kb.get("themes"):
        log.warning("No themes in knowledge base")
        return pd.DataFrame()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    year_col = "year"

    rows = []
    for t in kb["themes"]:
        label = t.get("theme", "")
        pc = t.get("paper_count", 0)
        conf = t.get("confidence", 0.5)

        # Get papers in this theme
        theme_titles = set(t.get("papers", []))
        theme_df = df[df["title"].isin(theme_titles)] if not theme_titles.issuperset({""}) else pd.DataFrame()

        # Publication density: papers per year
        if not theme_df.empty and year_col in theme_df.columns:
            years = theme_df[year_col].dropna()
            if len(years) > 0:
                year_range = years.max() - years.min()
                pub_density = pc / max(year_range, 1)
            else:
                pub_density = 0
        else:
            pub_density = 0

        # Citation density
        if not theme_df.empty and "citation_count" in theme_df.columns:
            cites = theme_df["citation_count"].dropna()
            total_cites = cites.sum()
            citation_density = total_cites / max(pc, 1)
        else:
            citation_density = 0

        # Topic saturation: MiniLM similarity among theme papers
        theme_texts = []
        for _, row in theme_df.iterrows():
            t_text = f"{row.get('title', '')} {row.get('abstract', '')}"
            theme_texts.append(t_text)
        if len(theme_texts) >= 2:
            embs = model.encode(theme_texts, show_progress_bar=False)
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            sim_matrix = np.dot(embs, embs.T)
            # Upper triangle average similarity (excluding diagonal)
            upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            topic_saturation = float(np.mean(upper))
        else:
            topic_saturation = 0.3  # default moderate

        # Emerging concepts in this theme
        keyword_text = " ".join(t.get("keywords", [])).lower()
        emerging = sum(1 for term in EMERGING_TERMS if term in keyword_text)
        emerging_score = min(emerging / 3.0, 1.0)

        # Inverse saturation: HIGH saturation = LOW novelty
        inv_saturation = 1.0 - min(topic_saturation, 1.0)
        inv_density = 1.0 - min(pub_density / 5.0, 1.0)
        inv_citation = 1.0 - min(citation_density / 100.0, 1.0)

        # Recency bonus
        years_list = []
        if not theme_df.empty and year_col in theme_df.columns:
            years_list = [int(y) for y in theme_df[year_col].dropna() if y != 0]
        recency_bonus = 0
        if years_list:
            latest = max(years_list)
            recency_bonus = min((latest - 2020) / 5.0, 0.15)

        novelty = (
            0.25 * inv_saturation
            + 0.20 * inv_density
            + 0.15 * inv_citation
            + 0.25 * emerging_score
            + 0.15 * (1.0 - conf)  # low confidence = novel area
            + recency_bonus
        )

        if novelty >= 0.65:
            level = "Highly Novel"
        elif novelty >= 0.45:
            level = "Moderately Novel"
        elif novelty >= 0.25:
            level = "Incremental"
        else:
            level = "Highly Saturated"

        rows.append({
            "theme": label,
            "paper_count": pc,
            "pub_density": round(pub_density, 3),
            "citation_density": round(citation_density, 1),
            "topic_saturation": round(topic_saturation, 3),
            "emerging_score": round(emerging_score, 3),
            "recency_bonus": round(recency_bonus, 3),
            "novelty_score": round(novelty, 3),
            "classification": level,
        })

    return pd.DataFrame(rows)


def _generate_report(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Novelty Assessment Report\n")
    lines.append(f"- **Themes scored:** {len(df)}")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for level in ["Highly Novel", "Moderately Novel", "Incremental", "Highly Saturated"]:
        subset = df[df["classification"] == level]
        if subset.empty:
            continue
        lines.append(f"## {level}\n")
        for _, row in subset.iterrows():
            lines.append(f"### {row['theme']}")
            lines.append(f"- Novelty score: {row['novelty_score']:.2f}")
            lines.append(f"- Topic saturation: {row['topic_saturation']:.2f}")
            lines.append(f"- Publication density: {row['pub_density']:.2f} papers/year")
            lines.append(f"- Citation density: {row['citation_density']:.0f} cites/paper")
            lines.append(f"- Emerging concept score: {row['emerging_score']:.2f}\n")

    lines.append("---\n*Generated by research_novelty_scorer.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score research novelty.")
    parser.add_argument("--knowledge-base", type=str, default="knowledge_base.json")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    df = score_novelty(args.knowledge_base, args.papers)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(out_dir / "novelty_scores.csv", index=False)
        log.info("Saved -> %s", out_dir / "novelty_scores.csv")

    report = _generate_report(df)
    with open(out_dir / "novelty_report.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "novelty_report.md")

    print(f"\n--- Novelty Scoring Complete ---")
    print(f"  Themes scored: {len(df)}")
    for level in ["Highly Novel", "Moderately Novel", "Incremental", "Highly Saturated"]:
        c = len(df[df["classification"] == level])
        if c:
            print(f"    {level}: {c}")
    print()


if __name__ == "__main__":
    main()
