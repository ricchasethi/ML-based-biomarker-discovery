# Week 2: Data Acquisition & Quality Control
## AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease

---

## Learning Objectives

By the end of Week 2, you will be able to:
1. Navigate NCBI GEO to identify, evaluate, and select appropriate miRNA datasets for AD biomarker analysis
2. Download and import GEO datasets programmatically into R using the `GEOquery` package
3. Understand the structure of raw microarray (CEL) and RNA-seq (count matrix) miRNA data
4. Extract and organize sample metadata (clinical phenotypes, covariates) from GEO records
5. Perform systematic quality control for both microarray and RNA-seq miRNA data
6. Apply appropriate normalization methods and understand the rationale behind each choice
7. Detect and correct batch effects using PCA, RLE plots, and ComBat/limma
8. Produce a clean, analysis-ready expression matrix with documented QC decisions

---

## Conceptual Overview: Why QC and Normalization Matter

Imagine you are comparing blood glucose levels across 200 patients, but half the samples were measured with a calibrated analyzer and half with a cheap glucometer that reads 20% too high. Any "biological" difference you find between patient groups is contaminated by instrument artifact. The same problem exists in genomic data — at a much larger scale.

In a typical GEO dataset, samples may have been:
- Processed in different laboratories or at different times (**batch effects**)
- Extracted with different RNA isolation kits (**extraction efficiency variation**)
- Hybridized to arrays on different days with different reagent lots (**technical variation**)
- Collected from patients of different ages, sexes, or medication histories (**biological confounders**)

**Quality control** identifies samples that have failed technically.  
**Normalization** removes systematic technical variation while preserving biological signal.  
**Batch correction** removes variation attributable to processing groups rather than biology.

Done well, these steps produce an expression matrix where sample-to-sample differences reflect **true biological differences** between AD patients and controls — the signal we want our ML models to learn from.

---

## MODULE 2.1 — Understanding NCBI GEO

### 2.1.1 GEO Data Architecture

NCBI GEO (Gene Expression Omnibus) organizes data in a hierarchical structure. Understanding this hierarchy is essential for navigating the database efficiently.

```
GEO Repository
│
├── GPL (Platform)
│     └── Describes the array or sequencer used
│           e.g., GPL16384 = Affymetrix Human Gene 2.1 ST Array
│
├── GSM (Sample)
│     └── One biological sample; contains raw and/or processed data
│           e.g., GSM1234567 = serum from AD patient #001
│
├── GSE (Series)
│     └── A complete study; links multiple GSMs and their GPL
│           e.g., GSE120584 = "Serum miRNA profiling in AD patients and controls"
│           Contains: study description, publication link, all GSM records, processed data files
│
└── GDS (Dataset) — optional
      └── Curated, analysis-ready subset created by NCBI staff
            Not all GSEs have a GDS; use GSE directly when GDS unavailable
```

**Key things to check in a GSE record before downloading:**

| Field | What to Look For |
|-------|------------------|
| **Summary** | Study aims, disease, sample types, N per group |
| **Overall Design** | Experimental design, controls used, covariates measured |
| **Contributor** | Corresponding authors (helps assess study quality) |
| **Platform (GPL)** | Array type or sequencing platform; determines preprocessing workflow |
| **Samples (GSM)** | Number of samples; click individual GSMs to check metadata completeness |
| **Supplementary Files** | Raw data (CEL files), count matrices, processed expression tables |
| **Linked Publications** | PubMed IDs — always read the associated paper |

---

### 2.1.2 Selecting a Dataset: Evaluation Criteria

Not all GEO datasets are equally suitable for our purposes. Use these criteria to evaluate datasets before committing to download:

**Scientific criteria:**
- [ ] Disease: Alzheimer's disease (confirmed diagnosis, not just "dementia")
- [ ] Biomarker type: miRNA (not mRNA, protein, or methylation)
- [ ] Sample type: Blood-derived (serum, plasma, whole blood, PBMCs)
- [ ] Has both AD patients AND healthy controls in the same study
- [ ] Sample size: ≥ 20 per group (ideally ≥ 40 per group for stable ML training)
- [ ] Metadata available: Age, sex, clinical stage (MCI vs clinical AD) are highly desirable

**Technical criteria:**
- [ ] Platform is well-supported by Bioconductor (Affymetrix arrays, Illumina)
- [ ] Normalization method is documented in associated paper
- [ ] Spike-in controls or reference miRNAs documented (for RNA quantity normalization)

**Red flags:**
- No control group (cannot perform differential expression)
- Only processed/normalized data deposited with no raw data
- Very small N (< 10 per group) — insufficient for ML

---

### 2.1.3 Our Working Datasets

For this course, we will work with the following AD miRNA datasets, selected for data quality, sample size, and biological relevance:

**Primary Dataset: GSE120584**
- **Title:** Identification of serum miRNA biomarkers for Alzheimer's disease
- **Platform:** GPL19117 (Illumina HiSeq 2500 — small RNA-seq)
- **Sample type:** Serum
- **Groups:** AD (n=48), MCI (n=50), healthy controls (n=50)
- **Why selected:** Large sample size; includes MCI class; RNA-seq format; linked to peer-reviewed publication

**Validation Dataset: GSE46579**
- **Platform:** GPL16384 (Affymetrix GeneChip miRNA 3.0)
- **Sample type:** Whole blood
- **Groups:** AD (n=35), controls (n=30)
- **Why selected:** Independent cohort for cross-validation (Week 5); different platform tests generalizability

> **Course convention:** We will preprocess and clean each dataset independently, then use GSE120584 for model training and GSE46579 for external validation in Week 5.

---

## MODULE 2.2 — Downloading GEO Data in R

### 2.2.1 The GEOquery Package

`GEOquery` is a Bioconductor package that provides programmatic access to all GEO records directly from R. It downloads the data, parses the SOFT file format, and returns structured R objects.

```r
# Install if not already done (from Week 1 setup)
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GEOquery")

# Load the package
library(GEOquery)
```

### 2.2.2 Downloading a GSE Record

```r
# Download the full GSE series
# destdir: where to save the files locally (avoids re-downloading)
# GSEMatrix: download the processed expression matrix (TRUE) or raw only (FALSE)

gse <- getGEO("GSE120584", 
               destdir = "./data/raw/",
               GSEMatrix = TRUE,
               AnnotGPL = TRUE)   # include gene/probe annotations

# GEO often returns a list (one element per platform GPL)
# Check how many platforms are in this study
length(gse)
names(gse)

# Extract the first (usually only) element
gse_data <- gse[[1]]

# Inspect the object type
class(gse_data)  # Should be "ExpressionSet"
```

**What is an ExpressionSet?**

An `ExpressionSet` is a standardized Bioconductor data container with three linked components:

```
ExpressionSet
├── exprs(gse_data)        — Expression matrix: rows = miRNAs, columns = samples
├── pData(gse_data)        — Phenotype data: rows = samples, columns = metadata fields
└── fData(gse_data)        — Feature data: rows = miRNAs, columns = probe annotations
```

This linked structure ensures that when you subset samples, all three components stay synchronized — a critical safety feature when working with metadata.

### 2.2.3 Extracting the Expression Matrix

```r
# Extract the expression matrix
expr_matrix <- exprs(gse_data)

# Basic inspection
dim(expr_matrix)          # [rows = number of miRNA probes, cols = number of samples]
expr_matrix[1:5, 1:5]    # Preview first 5 miRNAs × 5 samples

# For a microarray dataset, values are typically:
# - Raw: fluorescence intensity (positive integers, wide range)
# - Processed: log2-transformed, normalized values (typically 2–15 range)

# Check value range to understand preprocessing state
range(expr_matrix)
summary(as.vector(expr_matrix))
```

---

### 2.2.4 Extracting Sample Metadata

The phenotype data (`pData`) is often the most valuable and most poorly documented part of a GEO submission. Extracting it correctly is essential.

```r
# Extract metadata table
metadata <- pData(gse_data)

# Show all available metadata columns
colnames(metadata)

# Common columns you will find in GEO:
# geo_accession    — GSM accession number
# title            — Sample name/identifier
# source_name_ch1  — Tissue/sample type (e.g., "serum")
# characteristics_ch1, characteristics_ch1.1, ... — Study-specific fields
# description      — Free-text sample description
# data_processing  — Normalization/processing steps applied

# View key columns for study design
head(metadata[, c("geo_accession", "title", "characteristics_ch1", 
                   "characteristics_ch1.1", "characteristics_ch1.2")])
```

**Parsing the characteristics columns:** GEO encodes clinical metadata as free-text key:value pairs that need manual parsing. This is one of the most common sources of confusion for new users:

```r
# Example: characteristics_ch1 might contain "disease state: Alzheimer's Disease"
# We need to extract just the value

# View unique values to understand the encoding
unique(metadata$characteristics_ch1)
# Output might be:
# [1] "disease state: Alzheimer's Disease"
# [2] "disease state: Mild Cognitive Impairment"  
# [3] "disease state: Control"

# Extract the group label
metadata$group <- gsub("disease state: ", "", metadata$characteristics_ch1)
metadata$group <- factor(metadata$group, 
                         levels = c("Control", "Mild Cognitive Impairment", "Alzheimer's Disease"))

# Similarly extract age and sex if available
metadata$age <- as.numeric(gsub("age: ", "", metadata$characteristics_ch1.1))
metadata$sex  <- gsub("sex: ", "", metadata$characteristics_ch1.2)

# Summary of your cohort
table(metadata$group)
table(metadata$sex, metadata$group)
summary(metadata$age)
```

> **Biological check:** Before any analysis, verify that the cohort composition makes sense. Expected age distribution for AD: typically 65–90 years. Expected sex distribution: slightly more females in AD cohorts (reflecting population demographics). Obvious anomalies (e.g., mean age 35 in an AD cohort) signal a metadata parsing error.

---

### 2.2.5 Downloading Raw Data (CEL Files for Microarray)

For microarray studies, the processed matrix in GEO has already been normalized by the data depositor — and we may not agree with their choices. Always download and re-normalize from raw CEL files when available.

```r
# Download supplementary files (raw data)
getGEOSuppFiles("GSE46579", 
                makeDirectory = TRUE,
                baseDir = "./data/raw/")

# This downloads to ./data/raw/GSE46579/

# List downloaded files
list.files("./data/raw/GSE46579/")
# Expect: a .tar archive containing individual .CEL files (one per sample)

# Untar the archive
untar("./data/raw/GSE46579/GSE46579_RAW.tar", 
      exdir = "./data/raw/GSE46579/CEL_files/")

# List the CEL files
cel_files <- list.files("./data/raw/GSE46579/CEL_files/", 
                        pattern = "\\.CEL\\.gz$", 
                        full.names = TRUE)
length(cel_files)  # Should match the number of samples in the study
```

---

## MODULE 2.3 — Raw Data Formats

Understanding data formats prevents misinterpretation. This module explains what is actually inside the files you download.

### 2.3.1 Affymetrix CEL Files

CEL files are binary files storing raw fluorescence intensities from each physical probe on an Affymetrix array. They contain:
- **PM (Perfect Match) probe intensities** — the primary signal
- **MM (Mismatch) probe intensities** — used historically for background correction (deprecated in modern methods)
- **Spatial coordinates** of each probe on the chip
- **Quality metrics** (outline of chip image, standard deviations)

One CEL file = one sample. Each Affymetrix miRNA array chip contains millions of probes, but most map to a much smaller number of miRNA features after summarization.

```
File structure (conceptual):
CEL file
├── Header: chip type, date, parameters
├── Intensity data: probe_id → intensity value
└── Quality data: standard deviation, number of pixels per probe
```

### 2.3.2 Count Matrix Files (RNA-seq)

For small RNA-seq data deposited in GEO, raw FASTQ files are usually hosted on SRA (Sequence Read Archive) — downloading and aligning them requires a high-performance computing environment beyond this course scope. Instead, GEO depositors typically also provide **processed count matrices**: tab-delimited text files where:
- Rows = miRNA names (e.g., `hsa-miR-21-5p`)
- Columns = sample identifiers
- Values = integer read counts (how many sequencing reads mapped to each miRNA)

```
Example count matrix (first 4 rows, 4 samples):
                    GSM3047001  GSM3047002  GSM3047003  GSM3047004
hsa-let-7a-5p           45230       41890       38920       52340
hsa-let-7b-5p           12450       13201        9870       14320
hsa-miR-21-5p           89302       92145       78432       95120
hsa-miR-29a-3p            234         198         301         276
```

Key distinction from microarray: count data are **non-negative integers** with a characteristic **overdispersion** (variance > mean) that requires specific statistical methods (negative binomial models in DESeq2/edgeR) rather than methods assuming normally distributed data.

### 2.3.3 SOFT Files

GEO SOFT (Simple Omnibus Format in Text) files are the primary metadata format for GEO entries. They contain:
- Platform (GPL) description: probe sequences and annotations
- Sample (GSM) records: metadata and processed values for each sample
- Series (GSE) record: study summary and design

`GEOquery` parses SOFT files automatically — you rarely need to handle them directly.

---

## MODULE 2.4 — Quality Control for Microarray Data

Quality control for microarray data aims to identify samples with technical failures: poor RNA quality, insufficient hybridization, physical damage to the chip, or pipetting errors.

### 2.4.1 Loading CEL Files into R

```r
library(affy)           # For 3' IVT arrays (older Affymetrix design)
library(oligo)          # For Gene ST arrays and miRNA arrays (newer design)
library(affyQCReport)   # QC report generation

# Load all CEL files into an AffyBatch object
raw_data <- read.celfiles(cel_files)

# Basic inspection
raw_data
dim(exprs(raw_data))  # probes × samples
sampleNames(raw_data)
```

### 2.4.2 QC Metric 1 — Raw Signal Intensity Distribution

**What it measures:** The distribution of raw fluorescence intensities across all probes for each sample.

**What to look for:**
- All samples should have similar distributions (similar median, similar shape)
- A sample with dramatically shifted distribution = possible hybridization failure
- A sample with truncated upper tail = possible chip scanning issue

```r
# Box plot of raw intensities (log2-scale)
boxplot(raw_data, 
        main = "Raw Probe Intensities (log2)",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,         # rotate x-axis labels
        ylab = "log2 Intensity",
        cex.axis = 0.6)

# Density plot (smoother view of distribution shape)
hist(raw_data, 
     main  = "Density of Raw Intensities",
     col   = rainbow(ncol(exprs(raw_data))),
     lty   = 1)
```

**Interpretation:**
- Boxes (or density peaks) at very different positions → technical variation; normalization needed
- One box clearly lower than all others → failed sample; consider exclusion
- Bimodal density distributions in many samples → possible: low RNA quality or high background

---

### 2.4.3 QC Metric 2 — RLE (Relative Log Expression) Plot

**What it measures:** For each probe in each sample, RLE = log2(probe intensity) − median(log2(probe intensity) across all samples). Measures deviation of each sample from the "typical" sample.

**What to look for:**
- Median of each box should be at or very near zero
- Box width (IQR) should be similar across samples
- Samples with median far from zero or very wide boxes are outliers

```r
library(arrayQualityMetrics)

# Generate automated QC report (saves HTML report to directory)
arrayQualityMetrics(expressionset = raw_data,
                    outdir        = "./qc_reports/raw_QC/",
                    force         = TRUE,
                    do.logtransform = TRUE)

# Manual RLE computation and plotting
log_expr <- log2(exprs(raw_data) + 1)
row_medians <- apply(log_expr, 1, median)  # median across samples per probe
rle <- sweep(log_expr, 1, row_medians, "-")  # subtract row median

# Box plot of RLE values
boxplot(rle,
        main = "RLE Plot — Raw Data",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,
        ylab = "RLE",
        ylim = c(-2, 2))
abline(h = 0, col = "red", lty = 2)
```

---

### 2.4.4 QC Metric 3 — NUSE (Normalized Unscaled Standard Error)

**What it measures:** For Affymetrix arrays, each miRNA feature is represented by multiple probes. NUSE measures the standard error of probe-level estimates, normalized to the median across samples. A sample with high NUSE = inconsistent probe signals = likely poor hybridization.

```r
library(affyPLM)  # For PLM (probe-level model fitting)

# Fit probe-level model (needed for NUSE and RLE at probe level)
plm_fit <- fitPLM(raw_data)

# NUSE plot
NUSE(plm_fit,
     main = "NUSE Plot",
     col  = as.numeric(factor(metadata$group)) + 1,
     las  = 2)
abline(h = 1.10, col = "red", lty = 2)  # Flag samples with median NUSE > 1.10
```

**Rule of thumb:** Samples with median NUSE > 1.10 or median RLE > 0.10 should be carefully reviewed and potentially excluded.

---

### 2.4.5 QC Metric 4 — Sample-to-Sample Correlation Heatmap

**What it measures:** Pearson or Spearman correlation between all pairs of samples based on their global expression profiles. Biologically similar samples (same group) should be more correlated with each other than with different groups — but all samples should show reasonably high correlation (typically >0.90 for same-platform data).

```r
library(pheatmap)
library(RColorBrewer)

# Compute pairwise correlations
cor_matrix <- cor(log2(exprs(raw_data) + 1), method = "pearson")

# Annotation bar showing sample group
annotation_col <- data.frame(Group = metadata$group,
                              Sex   = metadata$sex,
                              row.names = colnames(cor_matrix))

# Heatmap
pheatmap(cor_matrix,
         annotation_col  = annotation_col,
         color           = colorRampPalette(rev(brewer.pal(9, "RdBu")))(100),
         breaks          = seq(0.85, 1.0, length.out = 101),
         main            = "Sample-to-Sample Pearson Correlation",
         fontsize_row    = 6,
         fontsize_col    = 6,
         show_rownames   = FALSE)
```

**What to look for:**
- Clusters of samples from the same group (confirms biological signal exists)
- Any sample with uniformly low correlation to all others (<0.90) → likely failed sample
- Strong clustering by sex, age, or processing batch rather than disease group → confounders need addressing

---

### 2.4.6 Flagging and Removing Failed Samples

```r
# Track QC decisions in a data frame
qc_decisions <- data.frame(
    sample        = colnames(raw_data),
    group         = metadata$group,
    nuse_median   = apply(NUSE(plm_fit, type = "values"), 2, median),
    rle_iqr       = apply(rle, 2, IQR),
    pass_qc       = TRUE,
    exclude_reason = ""
)

# Flag samples failing thresholds
qc_decisions$pass_qc[qc_decisions$nuse_median > 1.10] <- FALSE
qc_decisions$exclude_reason[qc_decisions$nuse_median > 1.10] <- "NUSE > 1.10"

# How many samples pass?
table(qc_decisions$pass_qc, qc_decisions$group)

# Save QC decisions for documentation
write.csv(qc_decisions, "./qc_reports/sample_qc_decisions.csv", row.names = FALSE)

# Subset to passing samples only
passing_samples <- qc_decisions$sample[qc_decisions$pass_qc]
raw_data_filtered <- raw_data[, passing_samples]
metadata_filtered <- metadata[passing_samples, ]
```

> **Documentation principle:** Every sample removal decision must be documented with its reason. In a published paper or thesis, you will need to report: "N samples were excluded due to [reason]. Final analysis included N AD patients, N MCI patients, and N controls."

---

## MODULE 2.5 — Normalization of Microarray Data

### 2.5.1 What Does Normalization Do?

Normalization is a mathematical transformation that removes technical variation between samples while preserving biological differences. For miRNA microarray data, technical variation arises from:
- Differences in total RNA input between samples
- Differences in RNA labeling efficiency
- Differences in hybridization conditions (temperature fluctuations, reagent lot)
- Differences in scanner calibration

**The central assumption** of most normalization methods: the **majority of miRNAs are not differentially expressed** between groups. Therefore, differences in the bulk distribution of intensities are technical, not biological. This assumption is generally valid for blood miRNA in AD (most miRNAs do not change; only a subset are dysregulated).

> **Important caveat:** If you are studying a biological condition where **global** expression changes are expected (e.g., comparing cells where you've knocked out a transcription factor that regulates most genes), standard normalization can erase true signal. This is not a concern for our blood miRNA vs AD study.

---

### 2.5.2 Method 1: RMA (Robust Multi-array Average) — Recommended for Affymetrix

RMA is the gold-standard normalization method for Affymetrix arrays. It performs three steps:

1. **Background correction:** Removes optical noise and non-specific hybridization signal using a convolution model. Unlike older MAS5 background correction, RMA does not use mismatch probes.

2. **Quantile normalization:** Forces the distribution of intensities across all arrays to be identical — same minimum, same maximum, same median, same IQR. After quantile normalization, a boxplot of all samples should be completely flat.

3. **Summarization:** Combines the multiple probe intensities for each miRNA feature into a single value using a robust median polish algorithm (resistant to outlier probes).

```r
library(oligo)  # For modern Affymetrix arrays

# Apply RMA normalization
rma_normalized <- rma(raw_data_filtered)

# Extract normalized expression matrix
expr_rma <- exprs(rma_normalized)

# Verify normalization: all boxes should now be at same height
boxplot(expr_rma,
        main = "Post-RMA Normalized Intensities",
        col  = as.numeric(factor(metadata_filtered$group)) + 1,
        las  = 2,
        ylab = "RMA-normalized log2 Intensity")

# RLE plot should now show boxes centered at zero with narrow IQR
log_rma <- expr_rma  # Already log2-transformed by RMA
row_medians_rma <- apply(log_rma, 1, median)
rle_rma <- sweep(log_rma, 1, row_medians_rma, "-")
boxplot(rle_rma,
        main = "RLE Plot — Post-RMA Normalization",
        col  = as.numeric(factor(metadata_filtered$group)) + 1,
        las  = 2,
        ylab = "RLE",
        ylim = c(-2, 2))
abline(h = 0, col = "red", lty = 2)
```

---

### 2.5.3 Method 2: Reference-Gene Normalization

Reference-gene-based (RGB) normalization uses **stably expressed reference miRNAs** (analogous to housekeeping genes like GAPDH in RT-qPCR) as internal controls. Each sample's expression values are scaled relative to its reference miRNA levels.

Based on an article retrieved from PubMed, Wang et al. (2015) *Molecular BioSystems* [(DOI: 10.1039/c4mb00711e)](https://doi.org/10.1039/c4mb00711e) systematically compared normalization methods for miRNA microarray data — including quantile, variance stabilization, robust spline, global scaling, and reference-gene approaches. Their key finding: **reference-gene normalization generally outperforms global methods**, particularly in biological conditions with large shifts in miRNA expression patterns, because it avoids "flattening" genuine large-scale differences.

**Common reference miRNAs for blood-based studies:**
- **miR-93-5p** — frequently stable in serum
- **miR-191-5p** — commonly used blood reference
- **miR-16-5p** — platelet-derived; stable in plasma but unstable in serum if platelet contamination varies
- **Spike-in controls (cel-miR-39, cel-miR-54)** — exogenous *C. elegans* miRNAs spiked in at a defined concentration during RNA extraction; best normalization control when available

```r
# Example: normalize to miR-93-5p as reference
ref_miR <- "hsa-miR-93-5p"

# Find row index of reference miRNA
ref_idx <- which(rownames(expr_rma) == ref_miR)

# Compute scaling factor for each sample (relative to mean reference across all samples)
ref_values     <- expr_rma[ref_idx, ]
mean_ref       <- mean(ref_values)
scaling_factors <- ref_values - mean_ref  # log2 scale: subtraction = division

# Apply normalization
expr_rgb <- sweep(expr_rma, 2, scaling_factors, "-")

# Verify: reference miRNA should now be constant across all samples
boxplot(expr_rgb[ref_idx, ] ~ metadata_filtered$group,
        main = paste("Reference miRNA:", ref_miR, "post-normalization"),
        ylab = "Normalized log2 expression")
```

---

### 2.5.4 Special Case: Hemolysis Correction for Blood miRNA

A critical pre-analytical variable in blood miRNA studies is **hemolysis** — the lysis of red blood cells during or after blood collection, which releases miRNAs that are highly abundant in erythrocytes (particularly miR-451a and miR-23a-3p) and contaminates the serum/plasma miRNA profile.

Based on articles retrieved from PubMed, Murray et al. (2018) *Cancer Epidemiology, Biomarkers & Prevention* [(DOI: 10.1158/1055-9965.EPI-17-0657)](https://doi.org/10.1158/1055-9965.EPI-17-0657) systematically characterized how pre-analytical variables including hemolysis and blood storage time affect circulating miRNA levels. Their key findings:
- Levels of housekeeping miRNAs gradually increase over 14 days of storage at room temperature, in parallel with the hemolysis marker **hsa-miR-451a**
- Normalizing to miR-451a can stabilize these storage-induced changes
- Serum prepared with a low-speed centrifugation step is more suitable for miRNA quantification than plasma prepared for ctDNA extraction

**Hemolysis detection and correction:**

```r
# miR-451a and miR-23a-3p are released preferentially from red blood cells
# The ratio miR-451a / miR-23a-3p serves as a hemolysis index

mir451a_idx  <- which(rownames(expr_rma) == "hsa-miR-451a")
mir23a_idx   <- which(rownames(expr_rma) == "hsa-miR-23a-3p")

# Compute hemolysis index (in log2 space: subtraction = log2 ratio)
hemolysis_index <- expr_rma[mir451a_idx, ] - expr_rma[mir23a_idx, ]

# Flag hemolyzed samples (threshold: hemolysis_index > 7 in some protocols)
# Exact threshold depends on platform; consult original paper methods
metadata_filtered$hemolysis_index  <- hemolysis_index
metadata_filtered$hemolyzed        <- hemolysis_index > 7

# Remove or flag hemolyzed samples
table(metadata_filtered$hemolyzed, metadata_filtered$group)
```

> **Why this matters clinically:** If you build an ML model on data where AD patients happen to have slightly more hemolyzed samples than controls (due to sample handling differences), your model may be learning hemolysis signal, not disease biology. This would be a spurious biomarker that fails in prospective clinical validation.

---

## MODULE 2.6 — Quality Control for RNA-seq Count Data

For RNA-seq datasets from GEO, we typically work with pre-aligned count matrices (since aligning FASTQ files requires HPC resources). QC of count data differs from microarray QC.

### 2.6.1 Loading Count Matrix Data

```r
library(DESeq2)
library(edgeR)
library(readr)

# Load the count matrix (typically a tab-separated text file)
# Row names = miRNA names, Column names = sample IDs
count_matrix <- read.table("./data/raw/GSE120584/GSE120584_counts.txt",
                            header      = TRUE,
                            row.names   = 1,
                            sep         = "\t",
                            check.names = FALSE)

# Inspect dimensions
dim(count_matrix)       # Should be: n_miRNAs × n_samples
count_matrix[1:5, 1:5] # Preview

# Ensure sample order matches metadata
all(colnames(count_matrix) == metadata$geo_accession)  # Should be TRUE
# If not, reorder:
count_matrix <- count_matrix[, metadata$geo_accession]
```

### 2.6.2 QC Metric 1 — Library Size (Total Read Count per Sample)

```r
# Total counts per sample = library size
library_sizes <- colSums(count_matrix)

# Bar plot of library sizes
barplot(library_sizes,
        main = "Library Sizes per Sample",
        ylab = "Total Read Counts",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,
        cex.names = 0.6)
abline(h = mean(library_sizes) * 0.5, col = "red", lty = 2)  # Flag if < 50% of mean

# Expected range for serum small RNA-seq: ~1–10 million reads per sample
summary(library_sizes)
```

**What to look for:**
- Samples with very low library size (< 500,000 reads) → likely insufficient RNA or poor sequencing
- Samples with library size < 50% of the cohort mean → flag for possible exclusion
- Library size variation > 5-fold across samples → strong normalization required

---

### 2.6.3 QC Metric 2 — Detected miRNA Count

```r
# How many miRNAs have at least 1 count in each sample?
detected_per_sample <- colSums(count_matrix > 0)

# Samples with very few detected miRNAs may have failed
barplot(detected_per_sample,
        main = "Number of Detected miRNAs per Sample",
        ylab = "Count of miRNAs with > 0 reads",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2)

# How many miRNAs are detected in at least N samples?
# (Useful for filtering: only keep miRNAs detected in most samples)
min_samples <- 0.8 * ncol(count_matrix)  # detected in ≥ 80% of samples
expressed_miRNAs <- rowSums(count_matrix > 0) >= min_samples
table(expressed_miRNAs)
```

---

### 2.6.4 Low-Count Filtering

Low-count miRNAs introduce noise without informative signal and inflate the multiple testing burden. They should be removed before normalization.

```r
# Strategy 1: Minimum count threshold
# Keep miRNAs with at least 10 reads in at least 80% of samples in any group

keep <- filterByExpr(count_matrix,
                     group     = metadata$group,
                     min.count = 10,
                     min.total.count = 15)

# filterByExpr is from edgeR; it applies the group-aware filter
count_filtered <- count_matrix[keep, ]

cat("miRNAs before filtering:", nrow(count_matrix), "\n")
cat("miRNAs after filtering:", nrow(count_filtered), "\n")

# Strategy 2: Variance-based filter (use after normalization)
# Keep top 75% most variable miRNAs (by IQR across samples)
# Applied after normalization — see Module 2.7
```

> **Rule:** Never filter based on differential expression status (e.g., "keep only miRNAs with p < 0.1"). This introduces selection bias. Filter based on expression level and detection rate only.

---

### 2.6.5 QC Metric 3 — Count Distribution

```r
# Visualize count distribution (raw counts are highly skewed; log-transform for visualization)
# Add 0.5 pseudocount before log to handle zeros
log_counts <- log2(count_filtered + 0.5)

# Box plots
boxplot(log_counts,
        main = "log2(count + 0.5) Distribution per Sample",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,
        ylab = "log2(count + 0.5)",
        cex.axis = 0.6)
```

---

## MODULE 2.7 — Normalization of RNA-seq Count Data

### 2.7.1 Why Standard Normalization Doesn't Directly Apply to Counts

Count data differs from microarray intensity data in important ways:
- Counts are **non-negative integers** (cannot be negative; many zeros)
- Count variability scales with expression level (**mean-variance relationship**)
- Counts have **overdispersion**: variance > mean (negative binomial distribution fits best)
- Library size (total reads) dominates technical variation

Methods designed for normally distributed data (like quantile normalization used for arrays) are **not appropriate** for raw count data. Instead, we use count-aware normalization methods.

---

### 2.7.2 Method 1: TMM (Trimmed Mean of M-values) — edgeR

TMM normalizes by computing, for each sample, a **scaling factor** that accounts for differences in RNA composition between samples. It trims away the most highly and lowly expressed genes before computing the normalization factor, making it robust to the presence of a few highly expressed miRNAs that would otherwise dominate the calculation.

```r
library(edgeR)

# Create a DGEList object (edgeR's data container)
dge <- DGEList(counts = count_filtered,
               group  = metadata$group)

# Compute TMM normalization factors
dge <- calcNormFactors(dge, method = "TMM")

# View the normalization factors (should be close to 1.0 for most samples)
dge$samples$norm.factors

# Extract TMM-normalized CPM (Counts Per Million) values
cpm_tmm <- cpm(dge, normalized.lib.sizes = TRUE, log = TRUE, prior.count = 0.5)
# log = TRUE gives log2-transformed CPM; prior.count = 0.5 handles zeros

# Check post-normalization distributions
boxplot(cpm_tmm,
        main = "TMM-normalized log2 CPM Distribution",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,
        ylab = "TMM log2 CPM")
```

---

### 2.7.3 Method 2: DESeq2 Median-of-Ratios

DESeq2 uses a **median-of-ratios** normalization that computes a size factor for each sample by:
1. Calculating a geometric mean expression level for each miRNA across all samples
2. Dividing each sample's counts by those geometric means
3. Taking the median of these ratios as the sample's **size factor**

This approach is robust to outlier miRNAs (a single very abundant miRNA doesn't distort the size factor) and works well with count data.

```r
library(DESeq2)

# Create DESeqDataSet object
# Design formula includes group; add covariates if needed: ~ sex + age + group
dds <- DESeqDataSetFromMatrix(
    countData = count_filtered,
    colData   = metadata,
    design    = ~ group
)

# Estimate size factors (normalization factors)
dds <- estimateSizeFactors(dds)
sizeFactors(dds)  # Should be close to 1.0 for most samples

# Extract normalized counts (divided by size factors)
norm_counts <- counts(dds, normalized = TRUE)
log_norm    <- log2(norm_counts + 0.5)  # log2-transform for visualization

# Post-normalization box plot
boxplot(log_norm,
        main = "DESeq2-normalized log2 Count Distribution",
        col  = as.numeric(factor(metadata$group)) + 1,
        las  = 2,
        ylab = "log2(normalized count + 0.5)")
```

---

### 2.7.4 CPM, RPKM, TPM — What Not to Use (and Why)

Students sometimes see these metrics in papers and want to use them. Here is a brief clarification:

| Metric | Formula | Use Case | Appropriate for miRNA? |
|--------|---------|----------|------------------------|
| **CPM** (Counts Per Million) | count / lib_size × 1e6 | Cross-sample comparison of detection rates | Yes (library size correction only) |
| **RPKM/FPKM** | CPM / gene_length_kb | mRNA; accounts for gene length | **No** — miRNAs are all ~22 nt; length normalization is meaningless |
| **TPM** | RPKM / sum(RPKM) × 1e6 | mRNA; sum-normalized | **No** — same reason as RPKM |
| **TMM log-CPM** | edgeR calcNormFactors → cpm() | Differential expression | **Yes — recommended** |
| **DESeq2 rlog/VST** | Regularized log transformation | Visualization, PCA, clustering | **Yes — recommended** |

**For our course:** Use **TMM-normalized log2 CPM** (via edgeR) or **rlog/VST-transformed values** (via DESeq2) for visualization, PCA, and ML feature engineering. Raw counts go into the DESeq2/edgeR differential expression models directly.

```r
# Variance-Stabilizing Transformation (VST) — preferred for visualization/ML
vst_data <- vst(dds, blind = TRUE)  # blind=TRUE: don't use design info (unbiased QC)
expr_vst <- assay(vst_data)

# Or: regularized log (rlog) — better for small sample sizes (N < 30)
rlog_data <- rlog(dds, blind = TRUE)
expr_rlog <- assay(rlog_data)
```

---

## MODULE 2.8 — Batch Effect Detection and Correction

### 2.8.1 What is a Batch Effect?

A batch effect is **systematic technical variation** introduced by processing samples in different groups (batches). Common batch sources in miRNA studies:

- **Date of RNA extraction:** RNA degradation enzymes in lab air; reagent lot differences
- **Date of library preparation or array hybridization:** Operator skill variation, reagent aging
- **Sequencing run:** Lane-to-lane variation on sequencing instruments
- **Processing site:** Multi-site studies with different laboratory protocols
- **Freeze-thaw cycles:** Samples thawed different numbers of times

Batch effects are insidious because they can **mimic biological signals** if batches are confounded with biological groups — for example, if all AD samples were extracted in January and all controls in June.

---

### 2.8.2 Detecting Batch Effects — PCA

Principal Component Analysis (PCA) is the primary tool for batch effect visualization. It reduces the high-dimensional expression matrix to a small number of "principal components" that capture the most variance in the data, then we plot samples in 2D colored by both group and batch to see which explains more of the variance.

```r
library(ggplot2)

# Compute PCA on transposed expression matrix (samples as rows)
pca_result <- prcomp(t(expr_vst), scale. = TRUE)

# Variance explained by each PC
var_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100

# Build data frame for plotting
pca_df <- data.frame(
    PC1   = pca_result$x[, 1],
    PC2   = pca_result$x[, 2],
    Group = metadata$group,
    Batch = metadata$batch,       # Must be in your metadata
    Sex   = metadata$sex,
    Age   = metadata$age
)

# PCA colored by GROUP (biology)
p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Group, shape = Group)) +
    geom_point(size = 3, alpha = 0.8) +
    labs(title = "PCA: Colored by Disease Group",
         x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
         y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")) +
    theme_bw() +
    scale_color_manual(values = c("steelblue", "orange", "firebrick"))

# PCA colored by BATCH (technical)
p2 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Batch, shape = Group)) +
    geom_point(size = 3, alpha = 0.8) +
    labs(title = "PCA: Colored by Processing Batch",
         x = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
         y = paste0("PC2 (", round(var_explained[2], 1), "% variance)")) +
    theme_bw()

# Print both plots side by side
library(gridExtra)
grid.arrange(p1, p2, ncol = 2)
```

**Interpretation guide:**

| What you see | What it means |
|-------------|---------------|
| PC1 separates groups (AD vs control) | Strong biological signal — good! |
| PC1 separates batches, not groups | Batch effect dominates — must correct |
| PC1 separates groups AND batches | Confounded — difficult; correction with caution |
| Random scatter regardless of group | No biological signal detected (at this level) |
| Outlier samples far from cluster | Sample failed QC; confirm with NUSE/RLE |

---

### 2.8.3 Detecting Batch Effects — RLE Plot Across Batches

After normalization, RLE plots stratified by batch reveal whether batch-specific shifts remain.

```r
# Recompute RLE after normalization
row_medians_vst <- apply(expr_vst, 1, median)
rle_vst <- sweep(expr_vst, 1, row_medians_vst, "-")

# Color by batch
batch_colors <- as.numeric(factor(metadata$batch))
boxplot(rle_vst,
        col  = batch_colors,
        main = "RLE Post-Normalization (colored by batch)",
        ylab = "RLE",
        las  = 2,
        ylim = c(-2, 2))
abline(h = 0, col = "red", lty = 2)
legend("topright", legend = levels(factor(metadata$batch)),
       fill = unique(batch_colors), title = "Batch")
```

If boxes within the same batch are systematically shifted (all above or below zero relative to other batches), a batch correction is needed.

---

### 2.8.4 Batch Correction — ComBat (sva package)

ComBat (Johnson et al., 2007) uses an empirical Bayes approach to estimate batch-specific mean and variance parameters for each miRNA, then adjusts the data to remove these batch-specific effects. It is currently the most widely used batch correction method in genomics.

```r
library(sva)

# ComBat requires:
# - expression matrix (miRNAs × samples) — already normalized (VST or RMA)
# - batch vector (factor identifying which batch each sample belongs to)
# - optional: biological covariates to PRESERVE (mod matrix)

# Create model matrix preserving the biological variable (group)
# This tells ComBat: "remove batch effects, but keep group differences intact"
mod  <- model.matrix(~ group, data = metadata)
mod0 <- model.matrix(~ 1, data = metadata)  # null model (intercept only)

# Apply ComBat
expr_combat <- ComBat(dat    = expr_vst,       # normalized expression matrix
                      batch  = metadata$batch,  # batch labels
                      mod    = mod,              # model preserving biology
                      par.prior = TRUE,          # parametric empirical Bayes (recommended)
                      prior.plots = FALSE)

# Verify batch correction: PCA should no longer separate by batch
pca_combat <- prcomp(t(expr_combat), scale. = TRUE)
pca_df_combat <- data.frame(
    PC1   = pca_combat$x[, 1],
    PC2   = pca_combat$x[, 2],
    Group = metadata$group,
    Batch = metadata$batch
)

ggplot(pca_df_combat, aes(x = PC1, y = PC2, color = Group, shape = Batch)) +
    geom_point(size = 3, alpha = 0.8) +
    labs(title = "PCA After ComBat Batch Correction",
         x = "PC1", y = "PC2") +
    theme_bw()
```

---

### 2.8.5 Batch Correction — limma::removeBatchEffect

A simpler, linear model-based alternative to ComBat. Appropriate when batch is a simple blocking factor (e.g., extraction date) and the data are approximately normally distributed (post-normalization microarray or VST-transformed RNA-seq).

```r
library(limma)

# Remove batch effect using linear regression approach
expr_batch_corrected <- removeBatchEffect(
    x     = expr_rma,           # normalized microarray expression matrix
    batch = metadata$batch,     # batch factor
    design = model.matrix(~ group, data = metadata)  # preserve group
)

# This is equivalent to fitting: expression ~ batch + group
# and returning the residuals + group term
```

**When to use ComBat vs removeBatchEffect:**

| Scenario | Recommendation |
|---------|----------------|
| Multiple batches, large study, RNA-seq | ComBat (more robust empirical Bayes approach) |
| Two batches, microarray, small N | `limma::removeBatchEffect` (simpler, fewer assumptions) |
| Batch perfectly confounded with group | **Neither — cannot correct.** Flag this as a study limitation. |
| Batch not documented in metadata | Use surrogate variable analysis (SVA) to estimate hidden batch |

---

## MODULE 2.9 — The Clean Data Checkpoint

Before moving to Week 3, your data should meet all of the following criteria:

### 2.9.1 Pre-Analysis Data Audit Checklist

**Sample integrity:**
- [ ] All samples pass NUSE/RLE thresholds (or exclusions are documented)
- [ ] Hemolyzed samples are identified and excluded (if blood-based data)
- [ ] Library sizes are within acceptable range (RNA-seq)
- [ ] Sample metadata is complete: group, age, sex, any known covariates

**Expression matrix:**
- [ ] miRNAs with low counts filtered (RNA-seq) or probes QC-flagged (array)
- [ ] Data is normalized (RMA for array; TMM/DESeq2 for RNA-seq)
- [ ] Log2-transformation applied (RMA already does this; apply to CPM for RNA-seq)
- [ ] Batch effects assessed via PCA and RLE
- [ ] Batch correction applied if needed and documented

**Metadata alignment:**
- [ ] Column order of expression matrix matches row order of metadata table
- [ ] Group labels are factored with correct reference level (Control as reference)
- [ ] All covariates for downstream regression (age, sex, MMSE) are correctly typed (numeric vs factor)

**Documentation:**
- [ ] QC decisions saved to `sample_qc_decisions.csv`
- [ ] Final sample counts per group recorded
- [ ] Processing steps recorded in R script (reproducible)

```r
# Final save of clean data objects for Week 3
saveRDS(expr_combat,       "./data/processed/expr_matrix_clean.rds")
saveRDS(metadata_filtered, "./data/processed/metadata_clean.rds")

# Record final cohort composition
cat("=== FINAL COHORT AFTER QC ===\n")
print(table(metadata_filtered$group))
cat("\nSex distribution:\n")
print(table(metadata_filtered$sex, metadata_filtered$group))
cat("\nAge summary per group:\n")
print(tapply(metadata_filtered$age, metadata_filtered$group, summary))
```

---

### 2.9.2 What a Clean Expression Matrix Should Look Like

A well-preprocessed miRNA expression matrix from blood:
- **Rows:** 100–500 miRNAs (after low-expression filtering)
- **Columns:** Samples (annotated with group, age, sex, batch)
- **Values:** log2-transformed, normalized (typically ranging from 2 to 15 for microarray; -2 to 12 for log2 CPM RNA-seq)
- **Distribution:** Box plots across samples should be flat (similar medians and IQR)
- **PCA:** PC1/PC2 should show some separation by disease group (not necessarily clean — that's what ML is for)

---

## WEEK 2 LAB SESSION

### Lab 2A — Navigating GEO and Downloading Data (45 min)

**Task:** Find, evaluate, and download GSE120584 using both the GEO web interface and `GEOquery`.

Step-by-step:
1. Go to [https://www.ncbi.nlm.nih.gov/geo/](https://www.ncbi.nlm.nih.gov/geo/)
2. Search: `GSE120584`
3. On the GSE page, answer these questions (write them down):
   - What sample types were used?
   - What platform (GPL) was used?
   - How many samples in each group?
   - Is raw data (FASTQ or CEL) deposited, or only processed counts?
   - What publication is linked?
4. Click on 3 individual GSM records and note what metadata fields are available
5. Download using `GEOquery` as shown in Module 2.2

**Deliverable:** A completed cohort description table (sample type, N per group, age range, sex distribution)

---

### Lab 2B — Quality Control Pipeline (60 min)

Using the provided pre-built notebook `Week2_QC_Pipeline.ipynb`:

1. Load the expression matrix from GEO
2. Generate raw intensity box plot and density plot
3. Compute and plot RLE values
4. Generate sample-to-sample correlation heatmap
5. Flag any samples failing QC thresholds
6. Apply RMA normalization (array) or TMM normalization (RNA-seq)
7. Generate post-normalization QC plots
8. Save final `expr_matrix_clean.rds` and `metadata_clean.rds`

**Questions to answer:**
- How many samples (if any) were excluded and why?
- What is the median library size across samples?
- Does the PCA show any separation by disease group before batch correction?
- Were batch effects detected? If so, what correction was applied?

---

## WEEK 2 ASSIGNMENTS

### Reading Assignment
1. **Murray et al. (2018)** — *"Future-Proofing" Blood Processing for Measurement of Circulating miRNAs* [(DOI: 10.1158/1055-9965.EPI-17-0657)](https://doi.org/10.1158/1055-9965.EPI-17-0657)  
   Focus on: Table 1 (pre-analytical variables), hemolysis detection method, serum vs plasma comparison

2. **Wang et al. (2015)** — *Optimal consistency in microRNA expression analysis using reference-gene-based normalization* [(DOI: 10.1039/c4mb00711e)](https://doi.org/10.1039/c4mb00711e)  
   Focus on: Figure 2 (comparison of normalization methods), criteria used to evaluate methods, recommendation

### Reflection Questions
1. Why is it dangerous to apply quantile normalization to miRNA data from a disease condition where global upregulation or downregulation is expected? How does reference-gene normalization address this?
2. You receive a GEO dataset where all AD samples were processed in batch 1 and all controls in batch 2. Can batch correction rescue this dataset for biomarker discovery? Why or why not?
3. A collaborator sends you a count matrix where some samples have 200,000 total reads and others have 8,000,000 total reads. What normalization approach would you use, and which samples (if any) might you exclude before normalization?

### Practical Exercise
Using the QC pipeline from Lab 2B, deliberately skip the normalization step and proceed directly to a box plot comparison of AD vs control samples. Then repeat with normalization. Write 2–3 sentences describing what changes and why this matters for the ML analysis in Week 4.

---

## WEEK 2 GLOSSARY

| Term | Definition |
|------|------------|
| **CEL file** | Raw Affymetrix array data file; contains probe-level fluorescence intensities for one sample |
| **RMA** | Robust Multi-array Average; standard 3-step normalization for Affymetrix arrays (background correction + quantile normalization + summarization) |
| **RLE plot** | Relative Log Expression plot; diagnostic showing each sample's deviation from the cohort median; boxes should center on zero |
| **NUSE** | Normalized Unscaled Standard Error; probe-level QC metric; samples with median NUSE > 1.10 may have failed hybridization |
| **DGEList** | edgeR data container for RNA-seq count data; holds count matrix, sample information, and normalization factors |
| **DESeqDataSet** | DESeq2 data container; stores counts, sample metadata, and experimental design formula |
| **TMM** | Trimmed Mean of M-values; edgeR normalization method robust to outlier expression genes |
| **VST** | Variance-Stabilizing Transformation; DESeq2 method that stabilizes variance across the expression range; preferred for visualization and ML |
| **rlog** | Regularized log transformation; DESeq2 alternative to VST; better for very small sample sizes |
| **CPM** | Counts Per Million; count divided by library size × 1,000,000; accounts for sequencing depth |
| **Size factor** | DESeq2 per-sample normalization coefficient; accounts for library size and RNA composition |
| **Batch effect** | Systematic technical variation between groups of samples processed at different times or locations |
| **ComBat** | Empirical Bayes batch correction method from the `sva` R package; adjusts batch-specific mean and variance per feature |
| **SVA** | Surrogate Variable Analysis; method for estimating hidden (unannotated) batch variables |
| **Hemolysis** | Lysis of red blood cells releasing cell-type-specific miRNAs (notably miR-451a) that contaminate serum/plasma miRNA profiles |
| **Library size** | Total number of sequencing reads in one RNA-seq sample; primary source of technical variation |
| **filterByExpr** | edgeR function for filtering low-count features in a group-aware manner |
| **Spike-in control** | Synthetic RNA of defined sequence and amount (e.g., *C. elegans* cel-miR-39) added to samples during extraction for normalization control |
| **pData** | phenoData; metadata slot in an ExpressionSet containing sample-level clinical and technical information |
| **ExpressionSet** | Bioconductor data container that links expression matrix, sample metadata, and feature annotations |

---

## KEY REFERENCES (Week 2)

All references retrieved from PubMed.

1. Murray MJ et al. (2018). "Future-Proofing" Blood Processing for Measurement of Circulating miRNAs in Samples from Biobanks and Prospective Clinical Trials. *Cancer Epidemiol Biomarkers Prev* 27(2):208–218. [DOI: 10.1158/1055-9965.EPI-17-0657](https://doi.org/10.1158/1055-9965.EPI-17-0657)

2. Wang X, Gardiner EJ, Cairns MJ (2015). Optimal consistency in microRNA expression analysis using reference-gene-based normalization. *Mol Biosyst* 11(5):1235–1240. [DOI: 10.1039/c4mb00711e](https://doi.org/10.1039/c4mb00711e)

3. Jiang X et al. (2025). Integrative bulk and single-cell transcriptomic profiling identifies core gene networks in glioma [demonstrates ComBat batch correction workflow on GEO datasets]. *BMC Cancer* 26(1):84. [DOI: 10.1186/s12885-025-15454-5](https://doi.org/10.1186/s12885-025-15454-5)

**Software and Methods References (key methods papers cited for tools used):**

4. Irizarry RA et al. (2003). Exploration, normalization, and summaries of high density oligonucleotide array probe level data. *Biostatistics* 4(2):249–264. [DOI: 10.1093/biostatistics/4.2.249](https://doi.org/10.1093/biostatistics/4.2.249) — *RMA normalization*

5. Robinson MD, McCarthy DJ, Smyth GK (2010). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics* 26(1):139–140. [DOI: 10.1093/bioinformatics/btp616](https://doi.org/10.1093/bioinformatics/btp616) — *edgeR / TMM normalization*

6. Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biol* 15:550. [DOI: 10.1186/s13059-014-0550-8](https://doi.org/10.1186/s13059-014-0550-8) — *DESeq2 / size factor normalization*

7. Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* 8(1):118–127. [DOI: 10.1093/biostatistics/kxj037](https://doi.org/10.1093/biostatistics/kxj037) — *ComBat batch correction*

8. Ritchie ME et al. (2015). limma powers differential expression analyses for RNA-sequencing and microarray studies. *Nucleic Acids Res* 43(7):e47. [DOI: 10.1093/nar/gkv007](https://doi.org/10.1093/nar/gkv007) — *limma / removeBatchEffect*

---

*Next Week: Exploratory Data Analysis — We will apply PCA, t-SNE, UMAP, clustering, and visualization to understand the structure of the clean data before any supervised machine learning.*
