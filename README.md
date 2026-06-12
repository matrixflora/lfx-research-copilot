# LFX Research Copilot

A local AI-powered research intelligence assistant for literature discovery, evidence synthesis, research gap analysis, hypothesis generation, manuscript development, and research planning.

## Key Features

- **Multi-Source Literature Retrieval** — Search papers across Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, and CORE in parallel. Deduplicates by DOI and semantic title similarity.
- **Adaptive Theme Discovery** — Clusters papers into research themes using MiniLM embeddings. Falls back to multi-method consensus (NMF + hierarchical + fixed-k) for small corpora.
- **Evidence Extraction & Synthesis** — Extracts objectives, methods, results, limitations from abstracts. Compares findings across studies to detect consensus and disagreement.
- **Citation Intelligence** — Builds directed citation graphs via OpenAlex API. Computes PageRank, HITS, identifies foundational papers and hidden gems.
- **Research Gap Validation** — Validates claimed gaps by searching the corpus with semantic similarity. Assigns confidence scores (Confirmed / Uncertain / Not Supported).
- **Contradiction Detection** — Detects opposing claims across the literature using 15 semantic opposition pairs.
- **Hypothesis Generation** — Produces structured, reproducible hypothesis banks with priority scoring, IV/DV specification, and methodology suggestions.
- **Research Question Optimization** — Generates 20 research questions per topic ranked by novelty, feasibility, funding potential, and translational impact.
- **Manuscript Generation** — Drafts introduction, literature review, methods, and discussion sections with APA-formatted inline citations.
- **Reviewer Simulation** — Evaluates manuscript drafts for weak arguments, missing citations, unsupported claims, and methodological concerns.
- **Reproducibility Auditing** — Scores each paper across 6 dimensions: data availability, code availability, sample size, statistical rigor, validation strategy, and controls.
- **Meta-Analysis Readiness** — Assesses whether the corpus contains comparable studies suitable for quantitative synthesis.
- **Novelty Scoring** — Estimates topic saturation, publication density, and emerging concept presence to classify themes from Highly Novel to Highly Saturated.
- **Scientific Claim Graph** — Extracts claims from abstracts and builds a directed evidence graph (supporting / contradictory) for RAG applications.
- **Study Design Advisor** — Recommends experimental designs, controls, sample sizes, statistical tests, and validation strategies based on theme maturity.
- **Bioinformatics Mode** — Detects omics data types (genomics, transcriptomics, proteomics, metabolomics, epigenomics, metagenomics), maps to repositories (GEO, SRA, ArrayExpress, ProteomeXchange, MetaboLights), and recommends pathway tools.
- **Statistical Consultant** — Recommends statistical tests for 6 design types, estimates sample size via normal approximation, computes post-hoc power.
- **Protocol Generation** — Generates lab protocols (PCR, Western blot) and bioinformatics pipelines (RNA-seq, variant calling) with QC checklists.
- **Grant Proposal Generation** — Drafts grant concepts, specific aims, and project summaries from research gaps and opportunity rankings.
- **Explainability** — Every output includes evidence source, confidence score, supporting papers, alternative interpretations, and limitations.
- **Semantic Alerts** — Compares knowledge base snapshots between runs to detect new themes, theme shifts, and confidence changes.
- **Research Dashboard** — Aggregates active projects, themes, gaps, datasets, manuscripts, grants, and alerts into a single-page overview.
- **42-Stage Pipeline** — Orchestrates all modules in dependency order with `--quick` (17 priority modules), `--life-science` (adds bioinformatics modules), `--skip`, and `--until` flags.

## System Requirements

- **Python 3.10+**
- **8 GB RAM minimum** (16 GB recommended)
- ~2 GB disk for the sentence-transformers model cache

## Installation

```bash
git clone https://github.com/dpikaArya/lfx-research-copilot.git
cd lfx-research-copilot

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python run_validation.py
```

This runs the 17 priority pipeline modules on a default biomedical query. Outputs are written to `outputs/LFX_Research_Copilot/`.

### Search for Papers

```bash
python src/search_papers.py "deep learning drug discovery" --max 30
```

The search query is a positional argument. Results are saved to `search_results.csv` in the current directory. Supported sources: Crossref, OpenAlex, Semantic Scholar, PubMed, arXiv, CORE.

### Run the Full Pipeline

```bash
python src/pipeline.py
```

Executes all 42 stages in order. On a 21-paper corpus this completes in ~5 minutes depending on API call latency.

### Pipeline Modes

```bash
# Quick mode — 17 high-value modules (no paper retrieval, PDF, or full-document stages)
python src/pipeline.py --quick

# Life-science mode — enables bioinformatics, study design, statistical, and protocol modules
python src/pipeline.py --life-science

# Skip specific stages
python src/pipeline.py --skip pdf_manager citation_network_analysis

# Run up to a specific stage
python src/pipeline.py --until hypothesis_generator
```

### Individual Modules

Every module can be run independently:

```bash
python src/citation_intelligence.py
python src/contradiction_detector.py
python src/hypothesis_generator.py
python src/manuscript_copilot.py
python src/research_brief.py
```

Most modules accept `--papers`, `--consensus`, `--knowledge-base` or similar arguments to specify input files. Run any module with `--help` to see its options.

## Outputs

All generated files are organized under `outputs/LFX_Research_Copilot/`:

| Directory | Contents |
|-----------|----------|
| `reports/` | Executive summaries, research briefs, gap reports, hypothesis banks |
| `evidence/` | Evidence matrices, synthesis reports |
| `knowledge_base/` | Machine-readable JSON snapshots, claim graphs |
| `references/` | APA citation support files |
| `dashboard/` | Aggregated research dashboard, research memory |
| `manuscript/` | Generated manuscript drafts |
| `grants/` | Grant proposal drafts |
| `protocols/` | Lab and bioinformatics protocol checklists |
| `statistics/` | Sample size estimates, power analyses |
| `bioinformatics/` | Omics dataset reports |
| `citation_network/` | Network analysis reports |
| `pdf_library/` | PDF library indexes |
| `figures/` | Figure reference catalogs |
| `tables/` | Table reference catalogs |
| `alerts/` | Semantic change detection reports |
| `explainability/` | Evidence trace reports |
| `projects/` | Project tracking databases |

## Major Modules

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `search_papers.py` | Multi-source literature retrieval | Query string | `search_results.csv` |
| `cluster_themes.py` | Unsupervised theme discovery | `search_results.csv` | `consensus_themes.csv`, clustering reports |
| `generate_reports.py` | Executive summary, gaps, knowledge base | `consensus_themes.csv`, embeddings | Reports, `knowledge_base.json`, RAG chunks |
| `citation_intelligence.py` | Citation graph, PageRank, HITS | `search_results.csv` | Citation metrics, foundational papers |
| `hypothesis_generator.py` | Structured hypothesis bank | Knowledge base, gaps | `hypothesis_bank.csv` |
| `manuscript_copilot.py` | Draft manuscript sections | Evidence matrix, knowledge base | `manuscript_draft.md` |
| `contradiction_detector.py` | Cross-paper claim contradictions | `search_results.csv`, themes | `contradictory_findings.md` |
| `research_gap_validator.py` | Gap validation with confidence | Papers, evidence, existing gaps | `gap_confidence_scores.csv` |
| `study_design_advisor.py` | Design recommendations | Knowledge base, evidence strength | `study_design_report.md` |
| `bioinformatics_mode.py` | Omics data detection | `search_results.csv` | `bioinformatics_report.md` |
| `statistical_consultant.py` | Test selection, power analysis | CLI parameters | `statistical_report.md`, sample size estimates |
| `pipeline.py` | 42-stage orchestrator | All upstream outputs | Pipeline summary |

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data processing and CSV I/O |
| numpy | Numerical computing |
| scikit-learn | Clustering (NMF, hierarchical), metrics |
| sentence-transformers | MiniLM text embeddings for semantic similarity |
| networkx | Citation graph construction and analysis |
| scipy | Spatial distance computations |
| requests | HTTP API calls to Crossref, OpenAlex, Semantic Scholar |
| tqdm | Progress bars for API-heavy operations |

PDF backends (`pypdf`, `pdfplumber`, `pymupdf`) are optional and only needed for PDF figure/table extraction and PDF management.

## Use Cases

- **Literature review automation** — Search, cluster, and synthesize papers on any research topic
- **Research gap identification** — Detect and validate underexplored areas with confidence scoring
- **Manuscript preparation** — Generate drafts with inline citations and peer-review simulation
- **Grant writing** — Produce proposal components from gap and opportunity analyses
- **Bioinformatics exploration** — Identify omics datasets and recommend analysis pipelines
- **Reproducibility assessment** — Audit papers for data/code availability and statistical rigor

## License

MIT
## Citation

If you use this software in your research, teaching, or publications, 
please cite:

Deepika. (2026). LFX Research Copilot: A Local AI Powered Research 
Intelligence System for Literature Discovery, Evidence Synthesis, and 
Scientific Reasoning. GitHub Repository.

### BibTeX

@software{deepika2026lfx,
  author = {Deepika},
  title = {LFX Research Copilot: A Local AI Powered Research Intelligence 
System for Literature Discovery, Evidence Synthesis, and Scientific 
Reasoning},
  year = {2026},
  version = {1.0},
  url = {https://github.com/YOUR_USERNAME/YOUR_REPOSITORY}
}
