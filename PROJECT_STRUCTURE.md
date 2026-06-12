# Project Structure

## Folder Hierarchy

```
lfx-research-copilot/
├── config/                    # Configuration files
│   ├── __init__.py
│   └── config.yaml            # Default pipeline settings
├── data/                      # Data directory (placeholder)
├── docs/                      # Supplementary documentation
├── examples/                  # Example queries
│   └── example_queries.txt
├── outputs/                   # Generated outputs
│   └── LFX_Research_Copilot/  # All pipeline outputs
├── src/                       # Source modules (43 files)
├── tests/                     # Unit tests (5 test files)
├── validation/                # Researcher validation materials
│   ├── LFX_researcher_feedback_form.md
│   ├── LFX_validation_checklist.md
│   └── LFX_validation_protocol.md
├── .github/workflows/test.yml # CI configuration
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── PROJECT_STRUCTURE.md
├── README.md
├── generate_requirements.py   # Dependency export script
├── requirements.txt           # Python dependencies
├── run_validation.py          # Quick-start validation script
└── setup.py                   # Package installer
```

## Module Index

Every Python module in `src/` can be run standalone. Modules communicate through CSV, JSON, and Markdown files on disk.

| Module | Purpose | Reads | Writes |
|--------|---------|-------|--------|
| `search_papers.py` | Multi-source literature search (Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, CORE) | query string | `search_results.csv`, `search_results.json` |
| `pdf_manager.py` | SHA-256 deduplication, metadata extraction, open-access flagging for PDFs | `search_results.csv`, PDF directory | `pdf_library_index.csv` |
| `full_text_evidence_extraction.py` | Extract objectives, methods, results, limitations, conclusions from abstracts | `search_results.csv` | `evidence_matrix.csv`, `evidence_summary.md` |
| `figure_table_extractor.py` | Catalog figures, tables, supplementary references from abstracts and PDFs | `search_results.csv`, PDF library | `figure_catalog.csv`, `table_catalog.csv` |
| `evidence_synthesis.py` | Cross-study comparison within themes; consensus and disagreement detection | `evidence_matrix.csv` | `evidence_synthesis_report.md` |
| `cluster_themes.py` | Adaptive clustering with small-corpus consensus (NMF + hierarchical + fixed-k) | `search_results.csv` | `consensus_themes.csv`, `consensus_metadata.json`, clustering reports |
| `theme_evolution.py` | Classify themes as redundant/developing/trending/future | `knowledge_base.json` | `theme_evolution.json` |
| `systematic_review.py` | PRISMA screening flow, evidence tables, risk-of-bias assessment | `search_results.csv`, themes | `sr_flow.md`, `evidence_tables.csv` |
| `citation_intelligence.py` | OpenAlex citation graph, PageRank/HITS, foundational papers, hidden gems | `search_results.csv`, themes | `citation_metrics.csv`, `foundational_papers.csv`, citation cache |
| `author_intelligence.py` | h-index, emerging authors, collaboration graph, institutional leaders | `search_results.csv`, themes | `author_intelligence.csv` |
| `journal_intelligence.py` | Core/emerging journal ranking, theme-journal mapping | `search_results.csv`, themes | `journal_ranking.csv` |
| `methodology_mining.py` | Extract study designs, statistical tests, data collection types, ML methods | `search_results.csv`, themes | `methods_database.csv`, `methods_summary.md` |
| `contradiction_detector.py` | Semantic opposition analysis across 15 claim-type pairs | `search_results.csv`, themes | `contradictory_findings.md` |
| `evidence_strength.py` | Score themes on study count, citations, recency, method diversity, consistency | `search_results.csv`, themes | `evidence_strength.csv`, `evidence_strength_report.md` |
| `research_gap_validator.py` | Validate gaps via MiniLM semantic search; confidence scoring | `search_results.csv`, evidence, gaps | `gap_confidence_scores.csv`, `validated_research_gaps.md` |
| `study_design_advisor.py` | Recommend designs, controls, sample sizes, tests, validation strategies | knowledge base, evidence strength | `study_design_report.md` |
| `statistical_consultant.py` | Test selection, sample size estimation, post-hoc power analysis | CLI parameters | `statistical_report.md`, `sample_size_estimates.csv` |
| `bioinformatics_mode.py` | Omics data-type detection, repository mapping, pathway tool recommendations | `search_results.csv` | `bioinformatics_report.md` |
| `meta_analysis_readiness.py` | Assess feasibility of meta-analysis per theme | `search_results.csv`, knowledge base | `meta_analysis_readiness.csv` |
| `novelty_scoring.py` | Estimate saturation, publication density, emerging concepts per theme | knowledge base, papers | `novelty_scores.csv` |
| `citation_network_analysis.py` | Seminal paper ranking, hub/authority/peripheral clusters, embedding shift | `search_results.csv`, citation metrics | `citation_report.md`, `citation_network.csv` |
| `hypothesis_generator.py` | Template-based hypothesis bank with priority scores | papers, themes, gaps | `hypothesis_bank.csv`, `hypothesis_bank.md` |
| `opportunity_ranking.py` | Rank themes by novelty, growth, gap importance, citation momentum | knowledge base, gaps, papers | `research_opportunities.csv` |
| `funding_alignment.py` | Map themes to strategic priority areas | knowledge base, gaps | `funding_alignment_report.md` |
| `research_roadmap.py` | Generate 1yr/3yr/5yr roadmaps per theme | knowledge base, themes | `research_roadmap.md` |
| `question_optimizer.py` | Generate 20 ranked research questions per topic | topic string | `optimized_questions.csv` |
| `dataset_discovery.py` | Identify datasets from abstracts; classify by repository, accessibility, type | `search_results.csv` | `available_datasets.csv`, `dataset_landscape.md` |
| `knowledge_base_update.py` | Unify all pipeline output into single JSON snapshot | (internal module) | `knowledge_base.json` |
| `grant_proposal_copilot.py` | Draft grant concept, specific aims, project summary | knowledge base, gaps, opportunities | `grant_proposal.md` |
| `manuscript_copilot.py` | Draft introduction, literature review, methods, discussion with APA refs | papers, knowledge base, evidence, claim refs | `manuscript_draft.md` |
| `reviewer_simulator.py` | Evaluate manuscript for weak arguments, missing citations, unsupported claims | manuscript, evidence | `review_comments.md` |
| `reproducibility_auditor.py` | Score papers on data/code availability, sample size, stats, validation, controls | `search_results.csv` | `reproducibility_scores.csv` |
| `protocol_generator.py` | Generate lab and bioinformatics protocols with QC checklists | CLI parameters | `protocol_*.md`, `checklist_*.csv` |
| `support_claims_with_references.py` | Enrich reports with APA inline citations and full reference lists | papers, knowledge base, evidence, metrics | `claim_support_references.csv`, enriched `*_with_refs.md` |
| `scientific_claim_graph.py` | Build directed evidence graph of supporting/contradictory claims | `search_results.csv` | `claim_graph.json`, `claim_graph_summary.md` |
| `semantic_alert_system.py` | Compare KB snapshots; detect new themes, shifts, confidence changes | knowledge base, gaps | `semantic_alerts.md`, snapshots |
| `explainability_engine.py` | Trace every output to evidence sources with confidence and limitations | knowledge base, evidence, gaps, novelty | `explainability_report.md` |
| `living_knowledge_base.py` | Final unified snapshot merging all pipeline stages | all upstream outputs | `living_knowledge_base.json` |
| `project_manager.py` | Track projects, hypotheses, manuscripts, datasets, grants | hypotheses, manuscripts, datasets, grants | `project_database.json`, `project_dashboard.md` |
| `research_brief.py` | Comprehensive unified report aggregating all outputs | living KB, papers | `research_brief.md` |
| `research_dashboard.py` | Single-page aggregated dashboard | all outputs | `research_dashboard.md` |
| `research_memory.py` | Persistent cross-session history of all pipeline runs | all outputs | `research_history.json`, `research_dashboard.md` |
| `pipeline.py` | 42-stage orchestrator with --quick, --life-science, --skip, --until | all upstream outputs | pipeline summary |
| `__init__.py` | Package marker | — | — |

## Input/Output Flow

```
search query
    │
    ▼
search_papers.py ─────────► search_results.csv
    │
    ├──► cluster_themes.py ──► consensus_themes.csv ──► generate_reports.py ──► knowledge_base.json
    │                                                      │
    ├──► full_text_evidence_extraction.py ──► evidence_matrix.csv
    │                                                      │
    │                              ┌────────────────────────┘
    │                              ▼
    │                   evidence_synthesis.py, contradiction_detector.py,
    │                   evidence_strength.py, research_gap_validator.py,
    │                   hypothesis_generator.py, opportunity_ranking.py ...
    │                              │
    │                              ▼
    │                   manuscript_copilot.py, grant_proposal_copilot.py,
    │                   protocol_generator.py, reviewer_simulator.py ...
    │                              │
    │                              ▼
    │                   support_claims_with_references.py
    │                   scientific_claim_graph.py
    │                   explainability_engine.py
    │                              │
    │                              ▼
    │                   research_brief.py, research_dashboard.py,
    │                   research_memory.py, living_knowledge_base.py
    │
    └──► pdf_manager.py ──► pdf_library/
    └──► figure_table_extractor.py ──► figure_catalog.csv, table_catalog.csv
```

## Output Directory

All generated files are written to `outputs/LFX_Research_Copilot/`:

| Directory | Contents |
|-----------|----------|
| `reports/` | Executive summaries, research briefs, gap analyses, hypothesis banks, novelty scores, opportunity rankings, validated gaps, study design reports, reproducibility scores, review comments |
| `evidence/` | Evidence matrices, evidence summaries, synthesis reports |
| `knowledge_base/` | Machine-readable JSON snapshots, claim graphs |
| `references/` | APA citation support reference files |
| `dashboard/` | Aggregated research dashboard, research memory history |
| `manuscript/` | Generated manuscript drafts |
| `grants/` | Generated grant proposal drafts |
| `protocols/` | Lab and bioinformatics protocol documents and checklists |
| `statistics/` | Statistical test recommendations, sample size estimates |
| `bioinformatics/` | Omics dataset detection reports |
| `citation_network/` | Network analysis reports and CSV exports |
| `pdf_library/` | PDF library indexes |
| `figures/` | Figure reference catalogs |
| `tables/` | Table reference catalogs |
| `alerts/` | Semantic change detection reports and snapshots |
| `explainability/` | Evidence trace reports |
| `projects/` | Project tracking databases and dashboards |
