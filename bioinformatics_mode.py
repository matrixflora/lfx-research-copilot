#!/usr/bin/env python3
"""
bioinformatics_mode.py — Specialised support for life science research:
discover omics datasets, recommend pathway analysis, identify biomarker
candidates, and propose multi-omics study designs.

Outputs
-------
outputs/bioinformatics/bioinformatics_report.md
outputs/bioinformatics/datasets.csv
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bioinformatics_mode")

OMICS_KEYWORDS = {
    "genomics": ["genome", "genomic", "WGS", "whole genome", "exome", "sequencing", "DNA-seq", "SNP", "variant"],
    "transcriptomics": ["transcriptome", "transcriptomic", "RNA-seq", "RNA sequencing", "microarray", "gene expression"],
    "proteomics": ["proteome", "proteomic", "mass spec", "MS/MS", "LC-MS", "protein expression"],
    "metabolomics": ["metabolome", "metabolomic", "metabolite", "NMR", "GC-MS"],
    "epigenomics": ["epigenome", "epigenomic", "ChIP-seq", "ATAC-seq", "methylation", "histone"],
    "metagenomics": ["metagenome", "metagenomic", "16S", "microbiome", "amplicon"],
}

REPOSITORY_PATTERNS = {
    "GEO": r"(?:GSE\d+|GEO|Gene Expression Omnibus)",
    "SRA": r"(?:SRR\d+|SRA|Sequence Read Archive|PRJNA\d+)",
    "ArrayExpress": r"(?:ArrayExpress|E-MTAB-\d+|E-GEOD-\d+)",
    "ProteomeXchange": r"(?:PXD\d+|ProteomeXchange|PRIDE)",
    "MetaboLights": r"(?:MTBLS\d+|MetaboLights)",
    "dbGaP": r"(?:dbGaP|phs\d+)",
}


def discover_omics_datasets(papers_path: str = "search_results.csv") -> pd.DataFrame:
    df = pd.read_csv(papers_path) if Path(papers_path).exists() else pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        doi = str(row.get("doi", ""))
        abstract = str(row.get("abstract", ""))
        if not title or title.lower() in ("nan", ""):
            continue
        text = f"{title} {abstract}"
        data_types = []
        for omic, keywords in OMICS_KEYWORDS.items():
            if any(k.lower() in text.lower() for k in keywords):
                data_types.append(omic)
        repositories = []
        for repo, pat in REPOSITORY_PATTERNS.items():
            if re.search(pat, text):
                repositories.append(repo)
        if data_types or repositories:
            rows.append({
                "paper_title": title[:120],
                "doi": doi,
                "data_types": "; ".join(data_types),
                "repositories": "; ".join(repositories),
                "potential_analysis": "; ".join([f"{d} analysis" for d in data_types]),
            })
    return pd.DataFrame(rows)


def recommend_pathway_analysis(omics_types: List[str]) -> Dict:
    tools = {
        "transcriptomics": ["DESeq2", "limma", "edgeR", "clusterProfiler", "GSEA", "Enrichr"],
        "proteomics": ["DEP", "MSstats", "STRING", "Cytoscape", "DAVID"],
        "metabolomics": ["xcms", "MetaboAnalyst", "mzMine", "GNPS"],
        "genomics": ["MAGMA", "FUMA", "VEP", "ANNOVAR"],
        "epigenomics": ["DiffBind", "MACS2", "HOMER", "ChIPseeker"],
    }
    recommended = {}
    for omic in omics_types:
        if omic in tools:
            recommended[omic] = tools[omic]
    return recommended


def identify_biomarkers(method: str = "transcriptomics") -> List[str]:
    """Template-based biomarker identification workflow steps."""
    steps = {
        "transcriptomics": [
            "Differential expression analysis (|log2FC| > 1, FDR < 0.05)",
            "ROC curve analysis for candidate markers",
            "Validation in independent cohort",
            "Multivariate classifier (LASSO, Random Forest)",
            "Pathway enrichment for biological interpretation",
        ],
        "proteomics": [
            "Protein quantification and differential abundance",
            "Feature selection via SVM or Random Forest",
            "Correlation with clinical phenotype",
            "ELISA or MRM validation in independent set",
        ],
        "metabolomics": [
            "Peak picking and alignment",
            "Multivariate modelling (PCA, PLS-DA)",
            "Metabolite identification (HMDB, METLIN)",
            "Biomarker panel selection (AUC > 0.80)",
        ],
    }
    return steps.get(method, steps["transcriptomics"])


def recommend_multiomics_designs() -> List[str]:
    return [
        "Collect matched samples for genomics + transcriptomics + proteomics",
        "Integrate via MOFA or iClusterBayes",
        "Use mediation analysis to link genetic variation → gene expression → protein → phenotype",
        "Validate key findings with targeted assays (qPCR, ELISA, MRM)",
        "Apply network fusion for patient stratification",
    ]


def _generate_report(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Bioinformatics Report\n")
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Omics Datasets Identified\n")
    if not df.empty:
        lines.append(df[["paper_title", "data_types", "repositories"]].to_markdown(index=False))
        lines.append("")

    lines.append("## Recommended Pathway Analysis Tools\n")
    omics_found = set()
    for dt in df["data_types"]:
        for o in str(dt).split("; "):
            if o.strip():
                omics_found.add(o.strip())
    tools = recommend_pathway_analysis(list(omics_found) if omics_found else ["transcriptomics"])
    for omic, tool_list in tools.items():
        lines.append(f"- **{omic.title()}:** {', '.join(tool_list)}")

    lines.append("\n## Biomarker Discovery Workflow\n")
    for step in identify_biomarkers("transcriptomics"):
        lines.append(f"- {step}")

    lines.append("\n## Multi-Omics Design Recommendations\n")
    for rec in recommend_multiomics_designs():
        lines.append(f"- {rec}")

    lines.append("\n---\n*Generated by bioinformatics_mode.py*")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bioinformatics mode for life science research.")
    parser.add_argument("--papers", type=str, default="search_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/bioinformatics")
    args = parser.parse_args()

    df = discover_omics_datasets(args.papers)
    report = _generate_report(df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(out_dir / "datasets.csv", index=False)
        log.info("Saved -> %s", out_dir / "datasets.csv")

    with open(out_dir / "bioinformatics_report.md", "w") as f:
        f.write(report)
    log.info("Saved -> %s", out_dir / "bioinformatics_report.md")

    print(f"\n--- Bioinformatics Mode Complete ---")
    print(f"  Omics datasets: {len(df)}")
    print()


if __name__ == "__main__":
    main()
