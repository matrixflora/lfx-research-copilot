# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-11

### Added

- **Literature retrieval** — Multi-source paper search across Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, and CORE with DOI and semantic-title deduplication.
- **Citation intelligence** — OpenAlex-based directed citation graphs with PageRank and HITS ranking, foundational paper and hidden-gem detection.
- **Knowledge graph generation** — Scientific claim graph with supporting and contradictory evidence edges, suitable for RAG applications.
- **Theme analysis** — Adaptive clustering with small-corpus multi-method consensus (NMF, hierarchical, fixed-k), evolution classification (redundant/developing/trending/future), and strength scoring across five dimensions.
- **Evidence synthesis** — Structured extraction of objectives, methods, results, limitations from abstracts; cross-study comparison with consensus and disagreement detection.
- **Hypothesis generation** — Template-based, reproducible hypothesis banks with priority scoring, IV/DV specification, control variables, and methodology suggestions.
- **Manuscript generation** — Draft introduction, literature review, methods, and discussion sections with APA-formatted inline citations.
- **Research gap validation** — Semantic evidence search to validate claimed gaps with confidence scoring (Confirmed / Uncertain / Not Supported).
- **Contradiction detection** — Semantic opposition analysis across 15 claim-type pairs to identify conflicting conclusions.
- **Question optimization** — Template-based generation of 20 research questions per topic ranked by novelty, feasibility, funding potential, and translational impact.
- **Study design advisor** — Design-type recommendation based on theme maturity and evidence strength.
- **Bioinformatics mode** — Omics data-type detection (genomics, transcriptomics, proteomics, metabolomics, epigenomics, metagenomics) with repository mapping and pathway-tool suggestions.
- **Statistical consultant** — Test selection for 6 design types, normal-approximation sample size estimation, post-hoc power computation.
- **Reproducibility auditing** — Six-dimension paper scoring (data, code, sample size, statistics, validation, controls).
- **Meta-analysis readiness** — Feasibility assessment per theme based on outcome, intervention, and measurement consistency.
- **Novelty scoring** — Saturation, publication density, and emerging-concept analysis with 4-tier classification.
- **Grant proposal generation** — Concept, specific aims, and project summary from gaps and opportunity rankings.
- **Reviewer simulation** — Automated evaluation of manuscript drafts for weak arguments, missing citations, unsupported claims, and methodological concerns.
- **Protocol generation** — Lab protocols (PCR, Western blot) and bioinformatics pipelines (RNA-seq, variant calling) with QC checklists.
- **PDF management** — SHA-256 deduplication, metadata extraction, open-access URL flagging.
- **Figure and table extraction** — Searchable catalogs of figures, tables, and supplementary references from abstracts and PDFs.
- **Citation network analysis** — Seminal paper ranking, hub/authority/peripheral cluster detection, year-over-year embedding shift tracing.
- **Semantic alert system** — Cross-snapshot detection of new themes, theme shifts, and confidence changes.
- **Explainability engine** — Evidence source, confidence score, supporting papers, alternative interpretations, and limitations for every output.
- **Research dashboard** — Aggregated single-page view of active projects, themes, gaps, datasets, manuscripts, grants, and alerts.
- **Research memory** — Persistent JSON history of topics, papers, themes, gaps, hypotheses, and manuscripts across sessions.
- **Project manager** — JSON project database tracking hypotheses, manuscripts, datasets, and grants.
- **Living knowledge base** — Incremental merging of all pipeline outputs into a unified snapshot.
- **42-stage pipeline** — Dependency-ordered orchestration with `--quick` (17 modules), `--life-science` (9 additional modules), `--skip`, and `--until` flags.
- **Life-science mode** — Automatically enables evidence extraction, gap validation, study design, statistical, bioinformatics, meta-analysis, novelty, dataset discovery, and protocol modules.
