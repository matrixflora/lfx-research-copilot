#!/usr/bin/env python3
"""
theme_evolution.py — Classify themes by evolutionary stage (redundant, developing,
trending, future) based on publication trends, citation momentum, novelty, and
saturation. Extends knowledge_base.json with classifications.

Usage
-----
    python theme_evolution.py
        [--consensus consensus_themes.csv]
        [--papers search_results.csv]
        [--knowledge-base knowledge_base.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("theme_evolution")

SMALL_CORPUS_THRESHOLD = 50
RECENT_WINDOW = 3  # years for "recent" window


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
# Theme-level statistics
# ---------------------------------------------------------------------------
def _compute_theme_stats(
    df_consensus: pd.DataFrame,
    current_year: int = 2026,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-theme statistics: size, trend, growth, citation momentum,
    novelty, and saturation."""
    stats: Dict[str, Dict[str, Any]] = {}

    for theme_name, group in df_consensus.groupby("consensus_theme"):
        years = group["year"].dropna().astype(int)
        citations = group["citation_count"].dropna().astype(float)

        n_papers = len(group)
        n_papers_in_theme = n_papers

        # Papers per year
        year_counts = years.value_counts().to_dict()
        all_years = list(range(int(years.min()) if len(years) > 0 else current_year,
                               current_year + 1))
        papers_per_year = {y: year_counts.get(y, 0) for y in all_years}

        # Growth rate: recent_3_year_average / older_average
        recent_years = [y for y in all_years if y > current_year - RECENT_WINDOW]
        older_years = [y for y in all_years if y <= current_year - RECENT_WINDOW]

        recent_avg = sum(papers_per_year.get(y, 0) for y in recent_years) / max(len(recent_years), 1)
        older_avg = sum(papers_per_year.get(y, 0) for y in older_years) / max(len(older_years), 1)
        growth_rate = recent_avg / max(older_avg, 0.01)

        # Citation momentum
        recent_cites = sum(
            citations[years == y].sum() for y in recent_years
            if y in years.values
        )
        historical_cites = sum(
            citations[years == y].sum() for y in older_years
            if y in years.values
        )
        citation_momentum = recent_cites / max(historical_cites, 0.01)

        # Average publication year (normalised to 0-1 within corpus range)
        avg_year = years.mean() if len(years) > 0 else current_year

        # Novelty score: based on recency + low historical volume
        # Higher avg_year = more novel. Lower older_avg = more novel.
        recency_factor = (avg_year - (current_year - 10)) / 10.0  # 0-1 scale
        recency_factor = max(0.0, min(1.0, recency_factor))
        historical_rarity = 1.0 - min(older_avg / max(recent_avg, 0.01), 1.0)
        novelty = (recency_factor * 0.6 + historical_rarity * 0.4)

        # Saturation score: large volume + low novelty + declining growth
        volume_factor = min(n_papers_in_theme / 10.0, 1.0)  # normalize, cap at 10
        declining = max(0.0, 1.0 - growth_rate / 2.0)  # low growth = more saturated
        saturation = (volume_factor * 0.4 + (1.0 - novelty) * 0.3 + declining * 0.3)

        stats[theme_name] = {
            "theme": theme_name,
            "paper_count": n_papers_in_theme,
            "papers_per_year": papers_per_year,
            "growth_rate": round(growth_rate, 2),
            "citation_momentum": round(citation_momentum, 2),
            "avg_year": round(avg_year, 1),
            "novelty_score": round(novelty, 3),
            "saturation_score": round(saturation, 3),
            "avg_citations": round(citations.mean(), 1) if len(citations) > 0 else 0.0,
        }

    return stats


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
def _classify_themes(
    stats: Dict[str, Dict],
    n_papers: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Classify themes into redundant, developing, trending, future categories
    and return all theme classifications."""

    all_classifications: List[Dict] = []
    redundant: List[Dict] = []
    developing: List[Dict] = []
    trending: List[Dict] = []
    future: List[Dict] = []

    is_small = n_papers < SMALL_CORPUS_THRESHOLD

    for theme_name, s in stats.items():
        # Default classification is "developing"
        cls = "developing"
        reasons: List[str] = []

        # Redundant: high saturation, low novelty, low growth
        is_redundant = (
            s["saturation_score"] > 0.5
            and s["novelty_score"] < 0.4
            and s["growth_rate"] < 1.5
        )

        # Trending: strong growth + citation momentum
        is_trending = (
            s["growth_rate"] > 1.5
            and s["citation_momentum"] > 1.2
            and s["paper_count"] >= 2
        )

        # Future: high novelty, small but growing
        is_future = (
            s["novelty_score"] > 0.5
            and s["paper_count"] <= 3
            and s["growth_rate"] >= 1.0
        )

        # Developing: moderate volume, positive growth
        is_developing = (
            s["paper_count"] >= 2
            and s["growth_rate"] >= 0.8
            and not is_redundant
            and not is_trending
            and not is_future
        )

        if is_redundant:
            cls = "redundant"
            reasons.append("high saturation")
            reasons.append("low novelty")
        if is_trending:
            cls = "trending"
            reasons.append("strong growth")
            reasons.append("high citation momentum")
        if is_future:
            cls = "future"
            reasons.append("high novelty")
            reasons.append("emerging terminology")
        if is_developing and cls == "developing":
            reasons.append("moderate volume")
            reasons.append("positive growth")

        # Fallback
        if not is_redundant and not is_trending and not is_future and not is_developing:
            cls = "developing"
            reasons.append("default classification")

        entry = {
            "theme": theme_name,
            "classification": cls,
            "paper_count": s["paper_count"],
            "growth_rate": s["growth_rate"],
            "citation_momentum": s["citation_momentum"],
            "novelty_score": s["novelty_score"],
            "saturation_score": s["saturation_score"],
            "avg_year": s["avg_year"],
            "reasons": reasons,
            "preliminary": is_small,
        }

        all_classifications.append(entry)

        if cls == "redundant":
            redundant.append(entry)
        elif cls == "trending":
            trending.append(entry)
        elif cls == "future":
            future.append(entry)
        else:
            developing.append(entry)

    return all_classifications, redundant, developing, trending, future


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
def _generate_timeline(
    df_consensus: pd.DataFrame,
    stats: Dict[str, Dict],
) -> pd.DataFrame:
    """Generate theme_timeline.csv with yearly data per theme."""
    rows: List[Dict] = []
    for theme_name, group in df_consensus.groupby("consensus_theme"):
        s = stats[theme_name]
        for year, count in s["papers_per_year"].items():
            # Citation count for this year
            year_cites = group[
                group["year"].dropna().astype(int) == year
            ]["citation_count"].sum()
            rows.append({
                "Theme": theme_name,
                "Year": year,
                "Paper_Count": count,
                "Growth_Rate": s["growth_rate"],
                "Citation_Count": int(year_cites),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Representative papers per theme
# ---------------------------------------------------------------------------
def _get_rep_papers(df_consensus: pd.DataFrame, theme_name: str, n: int = 3) -> str:
    group = df_consensus[df_consensus["consensus_theme"] == theme_name]
    titles = group["title"].dropna().head(n).tolist()
    return " | ".join(titles)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def _save_classification_csv(
    entries: List[Dict], path: str, extra_fields: List[str],
) -> None:
    if not entries:
        pd.DataFrame(columns=["Theme"] + extra_fields).to_csv(path, index=False)
        log.info("Saved (empty) -> %s", path)
        return

    rows = []
    for e in entries:
        row = {
            "Theme": e["theme"],
            "Paper_Count": e["paper_count"],
        }
        for f in extra_fields:
            row[f] = e.get(f, "")
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Saved %d records -> %s", len(rows), path)


# ---------------------------------------------------------------------------
# Evolution report
# ---------------------------------------------------------------------------
def _generate_report(
    all_classifications: List[Dict],
    redundant: List[Dict],
    developing: List[Dict],
    trending: List[Dict],
    future: List[Dict],
    stats: Dict[str, Dict],
    n_papers: int,
    df_consensus: pd.DataFrame,
    df_papers: Optional[pd.DataFrame],
) -> str:
    """Generate theme_evolution_report.md."""
    is_small = n_papers < SMALL_CORPUS_THRESHOLD
    prelim = (
        "Preliminary assessment due to limited literature sample."
        if is_small else ""
    )

    lines: List[str] = []
    lines.append("# Theme Evolution Report")
    lines.append("")

    if prelim:
        lines.append(f"> **Note:** {prelim}")
        lines.append("")

    # 1. Domain Overview
    lines.append("## 1. Domain Overview")
    lines.append("")
    lines.append(f"- **Corpus size:** {n_papers} papers")
    lines.append(f"- **Total themes:** {len(stats)}")
    if df_papers is not None:
        valid_years = df_papers["year"][(df_papers["year"] > 1900) & (df_papers["year"].notna())]
        yr_min = int(valid_years.min()) if len(valid_years) > 0 else "N/A"
        yr_max = int(valid_years.max()) if len(valid_years) > 0 else "N/A"
        lines.append(f"- **Year range:** {yr_min} – {yr_max}")
        srcs = df_papers["source"].value_counts().to_dict()
        lines.append(f"- **Sources:** {', '.join(f'{k} ({v})' for k, v in srcs.items())}")
    lines.append("")

    # Classifications summary
    counts = defaultdict(int)
    for c in all_classifications:
        counts[c["classification"]] += 1
    lines.append("| Classification | Count |")
    lines.append("|---------------|-------|")
    for label in ["developing", "redundant", "trending", "future"]:
        lines.append(f"| {label.capitalize()} | {counts.get(label, 0)} |")
    lines.append("")

    # 2. Mature Research Themes
    lines.append("## 2. Mature Research Themes")
    lines.append("")
    lines.append("Themes with substantial publication history and established literature.")
    lines.append("")
    mature = sorted(
        [s for s in stats.values() if s["saturation_score"] > 0.4],
        key=lambda x: x["paper_count"], reverse=True,
    )
    if mature:
        for m in mature:
            rep = _get_rep_papers(df_consensus, m["theme"])
            lines.append(f"- **{m['theme']}** — {m['paper_count']} papers, "
                         f"saturation {m['saturation_score']:.2f}")
            if rep:
                lines.append(f"  - Representative: {rep[:120]}")
    else:
        lines.append("No mature themes identified; corpus may be too small.")
    lines.append("")

    # 3. Redundant Research Themes
    lines.append("## 3. Redundant Research Themes")
    lines.append("")
    if prelim:
        lines.append(f"> **Note:** {prelim}")
        lines.append("")
    if redundant:
        for r in redundant:
            rep = _get_rep_papers(df_consensus, r["theme"])
            lines.append(f"- **{r['theme']}** — saturation {r['saturation_score']:.2f}, "
                         f"novelty {r['novelty_score']:.2f}")
            if rep:
                lines.append(f"  - Papers: {rep[:120]}")
    else:
        lines.append("No redundant themes identified.")
    lines.append("")

    # 4. Developing Research Themes
    lines.append("## 4. Developing Research Themes")
    lines.append("")
    if prelim:
        lines.append(f"> **Note:** {prelim}")
        lines.append("")
    if developing:
        for d in developing:
            rep = _get_rep_papers(df_consensus, d["theme"])
            lines.append(f"- **{d['theme']}** — {d['paper_count']} papers, "
                         f"growth {d['growth_rate']:.2f}")
            if rep:
                lines.append(f"  - Papers: {rep[:120]}")
    else:
        lines.append("No developing themes identified.")
    lines.append("")

    # 5. Trending Research Themes
    lines.append("## 5. Trending Research Themes")
    lines.append("")
    if prelim:
        lines.append(f"> **Note:** {prelim}")
        lines.append("")
    if trending:
        for t in trending:
            rep = _get_rep_papers(df_consensus, t["theme"])
            lines.append(f"- **{t['theme']}** — growth {t['growth_rate']:.2f}, "
                         f"citation momentum {t['citation_momentum']:.2f}")
            if rep:
                lines.append(f"  - Papers: {rep[:120]}")
    else:
        lines.append("No trending themes identified.")
    lines.append("")

    # 6. Future Research Themes
    lines.append("## 6. Future Research Themes")
    lines.append("")
    if prelim:
        lines.append(f"> **Note:** {prelim}")
        lines.append("")
    if future:
        for f in future:
            rep = _get_rep_papers(df_consensus, f["theme"])
            lines.append(f"- **{f['theme']}** — novelty {f['novelty_score']:.2f}, "
                         f"growth {f['growth_rate']:.2f}")
            if rep:
                lines.append(f"  - Papers: {rep[:120]}")
    else:
        lines.append("No future themes identified.")
    lines.append("")

    # 7. Research Gaps
    lines.append("## 7. Research Gaps")
    lines.append("")
    gaps: List[str] = []
    for s in stats.values():
        if s["paper_count"] == 1:
            gaps.append(f"- **{s['theme']}** — only 1 paper, requires further investigation")
    if gaps:
        lines.extend(gaps)
    else:
        lines.append("No major research gaps identified within the current corpus.")
    lines.append("")

    # 8. Strategic Opportunities
    lines.append("## 8. Strategic Opportunities")
    lines.append("")
    if future:
        lines.append("The following future themes represent strategic research opportunities:")
        for f in future:
            lines.append(f"- **{f['theme']}** — {f['novelty_score']:.2f} novelty, "
                         f"{f['growth_rate']:.2f} growth rate")
    elif trending:
        lines.append("Invest in trending themes to capitalise on current momentum:")
        for t in trending:
            lines.append(f"- **{t['theme']}** — growth {t['growth_rate']:.2f}")
    else:
        lines.append("Expand the corpus to identify strategic opportunities.")
    lines.append("")

    # 9. Suggested Future Research Questions
    lines.append("## 9. Suggested Future Research Questions")
    lines.append("")
    questions: List[str] = []
    for f_cls in [future, trending, developing]:
        for t in f_cls:
            kw = t["theme"].split(",")[0].strip().lower()
            questions.append(
                f"- What are the emerging methodologies and frameworks in **{t['theme']}**?"
            )
            questions.append(
                f"- How does **{t['theme']}** intersect with other domains in this corpus?"
            )
    if not questions:
        questions.append("- Expand the corpus to generate targeted research questions.")
    for q in questions[:8]:
        lines.append(q)
    lines.append("")

    lines.append("---")
    lines.append("*Generated by theme_evolution.py*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extend knowledge base
# ---------------------------------------------------------------------------
def _extend_knowledge_base(
    kb_path: str,
    redundant: List[Dict],
    developing: List[Dict],
    trending: List[Dict],
    future: List[Dict],
) -> None:
    """Add theme classifications to knowledge_base.json."""
    p = Path(kb_path)
    if not p.exists():
        log.warning("knowledge_base.json not found; skipping extension.")
        return

    with open(p) as f:
        kb = json.load(f)

    def _clean(e: List[Dict]) -> List[Dict]:
        return [
            {
                "theme": x["theme"],
                "paper_count": x["paper_count"],
                "growth_rate": x["growth_rate"],
                "citation_momentum": x["citation_momentum"],
                "novelty_score": x["novelty_score"],
                "saturation_score": x["saturation_score"],
                "preliminary": x.get("preliminary", False),
            }
            for x in e
        ]

    kb["redundant_themes"] = _clean(redundant)
    kb["developing_themes"] = _clean(developing)
    kb["trending_themes"] = _clean(trending)
    kb["future_themes"] = _clean(future)

    with open(p, "w") as f:
        json.dump(kb, f, indent=2, default=str)
    log.info("Extended knowledge_base.json with theme classifications.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify themes by evolutionary stage."
    )
    parser.add_argument(
        "--consensus", type=str, default="consensus_themes.csv",
        help="Consensus themes CSV (default: consensus_themes.csv)",
    )
    parser.add_argument(
        "--papers", type=str, default="search_results.csv",
        help="Original paper metadata CSV (default: search_results.csv)",
    )
    parser.add_argument(
        "--knowledge-base", type=str, default="outputs/knowledge_base/knowledge_base.json",
        help="Path to knowledge_base.json (default: outputs/knowledge_base/knowledge_base.json)",
    )
    args = parser.parse_args()

    df_consensus = _load_csv(args.consensus)
    if df_consensus is None:
        return
    df_papers = _load_csv(args.papers, required=False)

    n_papers = len(df_consensus)
    is_small = n_papers < SMALL_CORPUS_THRESHOLD

    # Compute theme statistics
    stats = _compute_theme_stats(df_consensus)

    # Classify themes
    all_cls, redundant, developing, trending, future = _classify_themes(stats, n_papers)

    # Ensure output directories exist
    reports_dir = Path("outputs") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save redundant_themes.csv
    _save_classification_csv(
        redundant, str(reports_dir / "redundant_themes.csv"),
        ["saturation_score", "novelty_score"],
    )

    # Save developing_themes.csv
    _save_classification_csv(
        developing, str(reports_dir / "developing_themes.csv"),
        ["growth_rate", "novelty_score"],
    )

    # Save trending_themes.csv
    _save_classification_csv(
        trending, str(reports_dir / "trending_themes.csv"),
        ["growth_rate", "citation_momentum"],
    )

    # Save future_themes.csv
    _save_classification_csv(
        future, str(reports_dir / "future_themes.csv"),
        ["novelty_score", "growth_rate", "saturation_score"],
    )

    # Save theme_timeline.csv
    timeline = _generate_timeline(df_consensus, stats)
    timeline.to_csv(str(reports_dir / "theme_timeline.csv"), index=False)
    log.info("Saved %d rows -> %s", len(timeline), reports_dir / "theme_timeline.csv")

    # Generate evolution report
    report = _generate_report(
        all_cls, redundant, developing, trending, future,
        stats, n_papers, df_consensus, df_papers,
    )
    report_path = reports_dir / "theme_evolution_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Saved -> %s", report_path)

    # Extend knowledge base
    _extend_knowledge_base(args.knowledge_base, redundant, developing, trending, future)

    # Print summary
    print("\n--- Theme Evolution Analysis Complete ---")
    print(f"  Themes classified:     {len(stats)}")
    print(f"  Redundant:             {len(redundant)}")
    print(f"  Developing:            {len(developing)}")
    print(f"  Trending:              {len(trending)}")
    print(f"  Future:                {len(future)}")
    print(f"  Evolution report:      outputs/reports/theme_evolution_report.md")
    print(f"  Timeline:              outputs/reports/theme_timeline.csv")
    if is_small:
        print("  Note: Preliminary assessment due to limited literature sample.")
    print()


if __name__ == "__main__":
    main()
