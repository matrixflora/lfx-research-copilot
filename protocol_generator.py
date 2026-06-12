#!/usr/bin/env python3
"""
protocol_generator.py — Generate experimental and computational protocols
with materials, methods, workflow steps, and quality control procedures.

Outputs
-------
outputs/protocols/protocol.md
outputs/protocols/protocol_checklist.csv
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

logging.basicConfig(level=logging.INFO, format="%(asynchronously)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("protocol_generator")

LAB_TEMPLATES = {
    "pcr": {
        "title": "PCR Amplification Protocol",
        "materials": ["DNA template", "Forward primer (10 µM)", "Reverse primer (10 µM)",
                      "dNTPs (10 mM each)", "DNA polymerase", "PCR buffer (10×)", "Nuclease-free water"],
        "steps": [
            "Prepare master mix: 2.5 µL buffer, 0.5 µL dNTPs, 0.5 µL each primer, 0.125 µL polymerase, 1 µL template, water to 25 µL",
            "Initial denaturation: 95°C for 3 min",
            "35 cycles: 95°C for 30 s, 55°C for 30 s, 72°C for 30 s/kb",
            "Final extension: 72°C for 5 min",
            "Hold at 4°C",
        ],
        "qc": ["Agarose gel electrophoresis to verify amplicon size", "Quantify DNA concentration (NanoDrop/Qubit)"],
    },
    "western": {
        "title": "Western Blot Protocol",
        "materials": ["Protein lysate", "SDS-PAGE gel", "PVDF membrane", "Primary antibody",
                      "HRP-conjugated secondary antibody", "ECL substrate", "TBST buffer"],
        "steps": [
            "Separate proteins by SDS-PAGE (120 V, 90 min)",
            "Transfer to PVDF membrane (100 V, 60 min, 4°C)",
            "Block in 5% BSA/TBST for 1 h at RT",
            "Incubate primary antibody (1:1000 in 5% BSA/TBST) overnight at 4°C",
            "Wash 3× 10 min TBST",
            "Incubate secondary antibody (1:5000 in 5% BSA/TBST) for 1 h at RT",
            "Wash 3× 10 min TBST",
            "Develop with ECL substrate and image",
        ],
        "qc": ["Ponceau S staining to confirm transfer", "Load equal protein amounts (BCA assay)"],
    },
}

BIOINFO_TEMPLATES = {
    "rna_seq": {
        "title": "RNA-Seq Analysis Pipeline",
        "materials": ["FASTQ files", "Reference genome (FASTA)", "Annotation (GTF/GFF)",
                      "FASTQC", "STAR aligner", "featureCounts", "DESeq2"],
        "steps": [
            "Quality control: FASTQC on raw reads",
            "Trimming: cutadapt/Trimmomatic for adapter removal",
            "Alignment: STAR --runThreadN 8 --genomeDir /path/to/index --readFilesIn sample_R1.fastq",
            "Quantification: featureCounts -a annotation.gtf -o counts.txt aligned.bam",
            "Differential expression: DESeq2 in R",
            "Visualisation: PCA plot, heatmap, volcano plot",
        ],
        "qc": ["FASTQC report (per-base quality, GC content, adapter contamination)",
               "STAR alignment rate (>70% uniquely mapped)", "MultiQC summary report"],
    },
    "variant_calling": {
        "title": "Germline Variant Calling Pipeline",
        "materials": ["WGS FASTQ files", "Reference genome (hg38/GRCh38)",
                      "BWA-MEM", "GATK4", "Picard", "bcftools"],
        "steps": [
            "Map reads: BWA-MEM -t 8 ref.fa sample_R1.fastq sample_R2.fastq > aligned.sam",
            "Sort & index: samtools sort aligned.sam -o aligned.bam && samtools index aligned.bam",
            "Mark duplicates: picard MarkDuplicates I=aligned.bam O=dedup.bam M=metrics.txt",
            "Base recalibration: GATK BaseRecalibrator & ApplyBQSR",
            "Variant calling: GATK HaplotypeCaller -R ref.fa -I recal.bam -O variants.vcf",
            "Filter: GATK VariantFiltration (QD < 2.0, FS > 60.0, MQ < 40.0)",
        ],
        "qc": ["Coverage metrics (mosdepth)", "Transition/transversion ratio", "Genotype concordance"],
    },
}


def generate_lab_protocol(protocol_type: str = "pcr") -> Dict:
    tpl = LAB_TEMPLATES.get(protocol_type, LAB_TEMPLATES["pcr"])
    return {
        "type": "lab",
        "protocol_type": protocol_type,
        **tpl,
        "generated_at": datetime.now().isoformat(),
    }


def generate_bioinformatics_protocol(protocol_type: str = "rna_seq") -> Dict:
    tpl = BIOINFO_TEMPLATES.get(protocol_type, BIOINFO_TEMPLATES["rna_seq"])
    return {
        "type": "bioinformatics",
        "protocol_type": protocol_type,
        **tpl,
        "generated_at": datetime.now().isoformat(),
    }


def generate_validation_plan(study_type: str = "experimental") -> Dict:
    plans = {
        "experimental": {
            "title": "Experimental Validation Plan",
            "steps": ["Power analysis for sample size determination",
                      "Randomisation and blinding protocol",
                      "Pre-registration of study design and analysis plan",
                      "Pilot study (n=10 per group)",
                      "Full data collection with interim monitoring",
                      "Pre-specified primary and secondary analyses",
                      "Sensitivity and subgroup analyses"],
            "qc": ["Independent replication by second researcher",
                   "Cross-validation or bootstrap for model validation",
                   "Outlier detection and influence analysis"],
        },
        "computational": {
            "title": "Computational Validation Plan",
            "steps": ["Train/test split (80/20) or k-fold cross-validation",
                      "Hyperparameter tuning on validation set",
                      "Evaluation on held-out test set",
                      "Comparison against baseline methods",
                      "Ablation studies for key components",
                      "Statistical significance testing (permutation test)"],
            "qc": ["Reproducibility: fix random seeds, containerise environment",
                   "Code review and documentation",
                   "Unit tests for core functions"],
        },
    }
    plan = plans.get(study_type, plans["experimental"])
    return {"type": "validation", "study_type": study_type, **plan, "generated_at": datetime.now().isoformat()}


def _protocol_to_md(protocol: Dict) -> str:
    lines: List[str] = []
    lines.append(f"# {protocol.get('title', 'Protocol')}\n")
    lines.append(f"- **Type:** {protocol.get('type', protocol.get('protocol_type', 'unknown'))}")
    lines.append(f"- **Generated:** {protocol.get('generated_at', '')}\n")

    if protocol.get("materials"):
        lines.append("## Materials\n")
        for m in protocol["materials"]:
            lines.append(f"- {m}")
        lines.append("")

    if protocol.get("steps"):
        lines.append("## Workflow\n")
        for i, step in enumerate(protocol["steps"], 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    if protocol.get("qc"):
        lines.append("## Quality Control\n")
        for q in protocol["qc"]:
            lines.append(f"- [ ] {q}")
        lines.append("")

    lines.append("---\n*Generated by protocol_generator.py*")
    return "\n".join(lines)


def _protocol_to_checklist(protocol: Dict) -> pd.DataFrame:
    rows = []
    for step in protocol.get("steps", []):
        rows.append({"category": "Step", "item": step, "completed": False})
    for qc in protocol.get("qc", []):
        rows.append({"category": "QC", "item": qc, "completed": False})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate protocols.")
    parser.add_argument("--lab-protocol", type=str, default=None,
                        choices=list(LAB_TEMPLATES.keys()) + [None],
                        help="Lab protocol type")
    parser.add_argument("--bio-protocol", type=str, default=None,
                        choices=list(BIOINFO_TEMPLATES.keys()) + [None],
                        help="Bioinformatics protocol type")
    parser.add_argument("--validation", type=str, default=None,
                        choices=["experimental", "computational", None],
                        help="Validation plan type")
    parser.add_argument("--output-dir", type=str, default="outputs/protocols")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocols: List[Dict] = []
    if args.lab_protocol:
        protocols.append(generate_lab_protocol(args.lab_protocol))
    if args.bio_protocol:
        protocols.append(generate_bioinformatics_protocol(args.bio_protocol))
    if args.validation:
        protocols.append(generate_validation_plan(args.validation))
    if not protocols:
        log.info("No protocol type specified; generating default PCR protocol")
        protocols.append(generate_lab_protocol("pcr"))

    for p in protocols:
        md = _protocol_to_md(p)
        fname = p.get("protocol_type", p.get("study_type", "protocol"))
        md_path = out_dir / f"protocol_{fname}.md"
        with open(md_path, "w") as f:
            f.write(md)
        log.info("Saved -> %s", md_path)

        check_df = _protocol_to_checklist(p)
        csv_path = out_dir / f"checklist_{fname}.csv"
        check_df.to_csv(csv_path, index=False)
        log.info("Saved -> %s", csv_path)

    print(f"\n--- Protocol Generation Complete ---")
    print(f"  Protocols: {len(protocols)}")
    print()


if __name__ == "__main__":
    main()
