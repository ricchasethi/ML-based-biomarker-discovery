# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A 6-week graduate course curriculum on AI/ML-based biomarker discovery using miRNA expression data in Alzheimer's disease. Audience: wet-lab biologists with little prior coding experience. Each week has a lecture write-up (`.md`) and a standalone lab R script (`.R`). Week 4 also has a Python component (Jupyter notebook).

## Running the Scripts

All R scripts are run interactively in **RStudio** by highlighting sections and pressing `Ctrl+Enter` (Linux/Windows) or `Cmd+Enter` (Mac). There is no build system or test suite — these are teaching scripts, not a software package.

**Week 1 (first-time setup only):**
```r
# In RStudio, open and run Week1_Setup_Template.R
# Installs all Bioconductor and CRAN packages needed for all 6 weeks
# Takes 5–15 minutes; subsequent runs skip already-installed packages
```

**Weeks 2–6 (run in order each week):**
```r
# Set working directory to the course root before running any script:
setwd("/path/to/ML-based-biomarker-discovery")
# Each script reads from data/processed/ and writes to results/
```

**Python component (Lab 4B):**
```bash
# Requires: numpy, pandas, scikit-learn, matplotlib, seaborn, umap-learn
pip install scikit-learn umap-learn shap xgboost matplotlib seaborn
# Open Week4_ML_Classifier.ipynb in JupyterLab
```

## Required R Version and Packages

- R ≥ 4.3.0, Bioconductor ≥ 3.18
- **Bioconductor:** GEOquery, DESeq2, limma, edgeR, multiMiR, miRBaseConverter, clusterProfiler, org.Hs.eg.db, affy, oligo, sva, STRINGdb, igraph
- **CRAN:** ggplot2, tidyverse, pheatmap, reshape2, RColorBrewer, ggrepel, gridExtra, cluster, factoextra, car, MASS, pROC, knitr, rmarkdown

Install everything via `Week1_Setup_Template.R` Section 3.

## Project Directory Structure (Expected at Runtime)

Scripts create and read from this layout relative to the working directory:
```
data/
  raw/          — GEO downloads (never modified after download)
  processed/    — clean RDS and CSV files passed between weeks
qc_reports/     — QC plots and sample exclusion logs
results/        — all output plots and tables (Weeks 3–6)
  Week6/        — Week 6 creates its own subdirectory
```

## Week-by-Week Data Flow

Each script consumes outputs from the previous week. The critical files passed between weeks:

| File | Written by | Read by |
|------|-----------|---------|
| `data/processed/GSE120584_expr_clean.rds` | Week 2 | Week 3, 4 |
| `data/processed/GSE120584_metadata_clean.rds` | Week 2 | Weeks 3–6 |
| `data/processed/GSE120584_counts_filtered.rds` | Week 2 | Week 4 |
| `data/processed/GSE46579_expr_rma.rds` | Week 2 | Weeks 4–5 |
| `data/processed/GSE120584_expr_varianceFiltered.rds` | Week 3 | Weeks 4–5 |
| `data/processed/GSE120584_expr_vf.csv` | Week 3 | Lab 3B (Python) |
| `data/processed/GSE120584_expr_forML.csv` | Week 4 | Lab 4B (Python) |
| `results/consensus_features_Week4.csv` | Week 4 | Week 5 |
| Harmonized RDS objects | Week 5 | Week 6 |

## Script Architecture

**`Week2_DataAcquisition_QC.R`** — Downloads GSE120584 (RNA-seq) and GSE46579 (microarray) from GEO, runs full QC pipelines, normalises (DESeq2 VST / RMA), detects hemolysis and batch effects (ComBat), saves clean matrices.

**`Week3_EDA.R`** — Loads Week 2 outputs; computes per-miRNA statistics; runs PCA (prcomp), hierarchical clustering (Ward.D2), gap statistic, silhouette, k-means, pheatmap; quantifies confounder effects via partial R²; exports variance-filtered CSV for Python t-SNE/UMAP.

**`Week4_DE_FeatureSelection.R`** — Runs DESeq2 on GSE120584 (three comparisons: AD vs Control, MCI vs Control, AD vs MCI) with lfcShrink; runs limma on GSE46579; computes cross-dataset overlap; applies Mann-Whitney U filter; exports `expr_forML.csv` for Python Lab 4B.

**`Week5_Validation.R`** — Harmonizes miRNA names across miRBase versions (miRBaseConverter/MIMAT accessions), finds feature intersection between platforms, applies z-score standardisation, exports harmonized matrices for Python, runs DeLong AUC test (pROC), generates calibration plots.

**`Week6_Interpretation.R`** — Integrates SHAP + DE results into composite ranking, queries multiMiR (14 databases), runs KEGG/GO enrichment (clusterProfiler), builds STRINGdb PPI network, generates the paper-style forest plot, simulates qPCR ΔΔCt validation.

## Key Coding Conventions

- Scripts are written for **copy-paste-and-run** accessibility; sections are delimited by `# === SECTION N ===` banners.
- `GROUP_COLOURS` is defined at the top of every Week 3+ script and must stay consistent: `Control="#4575B4"`, `MCI="#FEE090"`, `AD="#D73027"`.
- All `ggsave()` calls write to `results/` with `dpi=150`.
- `set.seed(42)` is used wherever randomness exists (PCA initialisation, k-means, gap statistic bootstrap, train/test split).
- DESeq2 always uses `relevel(dds$group, ref = "Control")` so fold changes are expressed as treatment vs Control.
- LFC shrinkage: use `type="apeglm"` for single coefficients, `type="ashr"` for arbitrary contrasts.
- `scale.=TRUE` is always passed to `prcomp()` to prevent high-expression miRNAs from dominating PCA.
- The `expr` matrix convention: **rows = miRNAs, columns = samples**. Transpose before PCA and distance calculations.

## Primary GEO Datasets

- **GSE120584** — serum small RNA-seq; 3 groups: AD / MCI / Control (~148 samples); primary training dataset
- **GSE46579** — whole blood Affymetrix microarray; AD / Control; external validation dataset
