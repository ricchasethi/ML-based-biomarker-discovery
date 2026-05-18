# Week 3: Exploratory Data Analysis
## AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease

---

## Learning Objectives

By the end of Week 3, you will be able to:
1. Compute and interpret descriptive statistics for miRNA expression data, including coefficient of variation, sparsity, and zero-inflation, and explain what these statistics reveal about data quality and biological signal
2. Perform Principal Component Analysis on a miRNA expression matrix, interpret scree plots and loadings, and explain in biological terms what each principal component captures
3. Generate and critically interpret t-SNE and UMAP dimensionality reduction plots, selecting appropriate hyperparameters and recognizing common interpretation pitfalls specific to nonlinear methods
4. Execute hierarchical and k-means clustering, evaluate optimal cluster number using gap statistic and silhouette width, and compute cluster purity against known disease groups
5. Produce and read a publication-quality annotated heatmap of miRNA expression, interpreting row and column structure in terms of disease biology
6. Identify and quantify the contribution of biological confounders (age, sex) to principal components using correlation tests and partial R², and describe a strategy to account for these in downstream ML modeling

---

## Conceptual Overview: Why Explore Before You Model?

When a clinician sees a new patient for the first time, they do not immediately prescribe a treatment. They take a history, examine the patient, order basic tests, and build up a picture before making decisions. The same principle applies in machine learning.

**Exploratory Data Analysis (EDA)** is the systematic examination of a dataset — its structure, its variation, its distributions, its outliers, and its relationships — before any supervised modeling. In the context of our miRNA Alzheimer's disease dataset, EDA serves four essential purposes:

**1. Catching problems that QC missed.** Week 2 removed technically failed samples. EDA catches subtler issues: miRNAs that are zero in >80% of samples, outlier samples that passed QC metrics but sit far from all biological clusters in PCA space, or confounders (age, sex) that account for more variance than disease group.

**2. Building biological intuition.** Before you ask a machine learning model "which miRNAs discriminate AD from controls?", you should have a personal sense of the data. Which miRNAs are most variable? Do the AD and control samples look visually separable? Which samples are the most unusual? EDA builds the biological intuition that lets you critically evaluate ML outputs rather than passively accept them.

**3. Informing modeling decisions.** The dimensionality of your data (148 samples × 300 miRNAs), the degree of class separation visible in PCA, the presence of confounders — all of these findings from EDA directly determine choices in Week 4: how many features to carry into ML, whether to include age/sex as covariates, which cross-validation strategy to use.

**4. Generating preliminary biological hypotheses.** A miRNA that appears in the top cluster-defining rows of a heatmap, and whose cluster separates AD from control perfectly, is a candidate worth investigating mechanistically — even before the formal differential expression analysis.

> **The biology-first principle applies here more than anywhere else in the course.** Every plot you generate this week should prompt a biological question. A t-SNE cluster is not just a visual grouping; it represents a set of patients whose miRNA profiles are similar — and asking *why* they are similar is the beginning of a discovery.

---

## MODULE 3.1 — Descriptive Statistics for Expression Data

### 3.1.1 What Are We Measuring?

Before applying any sophisticated algorithm, we characterize the basic statistical properties of our miRNA expression matrix. Our working matrix, loaded from Week 2 outputs, has the following structure:

```
Rows    = miRNAs (e.g., hsa-miR-21-5p, hsa-let-7a-5p, ...)  — typically 100–400 after QC filtering
Columns = Samples (e.g., GSM3047001, GSM3047002, ...)        — approximately 148 samples (48 AD, 50 MCI, 50 Control)
Values  = VST-transformed log2-scale expression              — continuous, typically ranging ~2–15
```

For each miRNA (each row), we compute a set of statistics that describes its behavior across all samples.

---

### 3.1.2 Per-miRNA Descriptive Statistics

**Mean expression:** The average expression of a miRNA across all samples. Biologically, a miRNA with very low mean expression may not be reliably detected in serum and may be contributing more noise than signal.

**Standard deviation (SD):** Measures spread around the mean. A miRNA with high SD is more variable across samples — potentially because it changes between disease groups (interesting) or because it is noisy (less interesting).

**Coefficient of Variation (CV):** CV = SD / mean × 100%. CV expresses variability *relative to the mean*, enabling comparison of variability across miRNAs at very different expression levels. A miRNA with mean expression of 10 and SD of 2 has a CV of 20%; another with mean of 3 and SD of 1 has a CV of 33%. Without the CV, the second miRNA would appear less variable (SD of 1 < SD of 2), but relative to its expression level it is actually more variable.

**Interquartile Range (IQR):** The range from the 25th to the 75th percentile. IQR is more robust than SD to outlier samples. We use IQR-based variance filtering (keep top 75% by IQR) before most analyses.

**Percentage of zeros (% zeros):** In RNA-seq data, miRNAs with many zero counts — even after normalization — were not detected in many samples. A miRNA with 80% zeros is poorly expressed and will not contribute meaningful biological signal.

```r
# ============================================================
# Load clean data from Week 2
# ============================================================
expr   <- readRDS("data/processed/GSE120584_expr_clean.rds")
meta   <- readRDS("data/processed/GSE120584_metadata_clean.rds")

GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",
  "Mild Cognitive Impairment" = "#FEE090",
  "Alzheimer's Disease"       = "#D73027"
)

# ============================================================
# Compute per-miRNA descriptive statistics
# ============================================================
mirna_stats <- data.frame(
  mirna       = rownames(expr),
  mean_expr   = rowMeans(expr),
  sd_expr     = apply(expr, 1, sd),
  median_expr = apply(expr, 1, median),
  iqr_expr    = apply(expr, 1, IQR),
  pct_zeros   = rowSums(expr == 0) / ncol(expr) * 100,
  stringsAsFactors = FALSE
)

mirna_stats$cv <- mirna_stats$sd_expr / mirna_stats$mean_expr * 100

# Sort by CV (most variable first)
mirna_stats <- mirna_stats[order(mirna_stats$cv, decreasing = TRUE), ]

# Preview most variable miRNAs
cat("=== Top 10 Most Variable miRNAs (by CV) ===\n")
print(head(mirna_stats[, c("mirna", "mean_expr", "sd_expr", "cv", "pct_zeros")], 10))

# Preview most stable miRNAs (lowest CV)
cat("\n=== Top 10 Most Stable miRNAs (by CV) ===\n")
print(tail(mirna_stats[, c("mirna", "mean_expr", "sd_expr", "cv", "pct_zeros")], 10))
```

> **Biological sidebar — What do "stable" miRNAs mean in blood?**
> miRNAs with very low CV across all disease groups (AD + MCI + Control) are behaving like biological housekeeping genes in blood. If a miRNA has high and stable expression regardless of AD status, it may reflect constitutive biological processes in blood cells (cell maintenance, basic metabolic regulation) rather than disease-specific signals. These stable miRNAs are candidates for use as reference normalization controls — analogous to the role of GAPDH in RT-qPCR. The miRNAs with high CV, on the other hand, are the ones worth investigating: their variation *might* be disease-related, though it could also be noise.

---

### 3.1.3 Zero-Inflated Distributions in miRNA-seq Data

RNA-seq count data — even after VST transformation — frequently shows **zero inflation**: a higher proportion of zero values than would be expected from a simple Gaussian or negative binomial distribution. This arises for two distinct reasons:

**1. Structural zeros:** The miRNA is genuinely not expressed in a given sample. For example, a neuron-enriched miRNA (like miR-9-5p) may be undetectable in serum from some healthy individuals but detectable in AD patients whose neuronal EV release is increased.

**2. Sampling zeros:** The miRNA is expressed at very low levels, but by chance no reads mapping to it were captured in that library. This is a technical artifact.

Distinguishing these two types is important: structural zeros carry biological information (the miRNA is truly absent in that sample), while sampling zeros are noise. Unfortunately, they are mathematically indistinguishable in a single experiment.

```r
# Visualize the zero fraction distribution across miRNAs
library(ggplot2)

ggplot(mirna_stats, aes(x = pct_zeros)) +
  geom_histogram(bins = 40, fill = "#4575B4", colour = "white", alpha = 0.8) +
  geom_vline(xintercept = 20, colour = "red", linetype = "dashed") +
  labs(
    title   = "Distribution of Zero Percentage Across miRNAs",
    x       = "% Samples with Zero Expression",
    y       = "Number of miRNAs",
    caption = "Red dashed line: 20% zero threshold"
  ) +
  theme_bw(base_size = 12)

# How many miRNAs have >20% zeros?
cat("miRNAs with > 20% zeros:", sum(mirna_stats$pct_zeros > 20), "\n")
cat("miRNAs with > 50% zeros:", sum(mirna_stats$pct_zeros > 50), "\n")
```

> **Practical decision:** miRNAs with > 50% zeros across the dataset rarely contribute useful signal to ML models. The Week 2 `filterByExpr` step should have removed most of these, but it is worth verifying here. If >10% of your remaining miRNAs have zero fractions above 50%, consider tightening the filtering threshold.

---

### 3.1.4 Distribution Shape: Is the Data Well-Normalized?

After VST transformation, expression values should be approximately normally distributed within each sample — this is one of the goals of VST. Checking this with density plots confirms whether normalization succeeded.

```r
# Density plots for three randomly selected samples (one per group)
set.seed(42)
idx_ctrl <- sample(which(meta$group == "Control"), 1)
idx_mci  <- sample(which(meta$group == "Mild Cognitive Impairment"), 1)
idx_ad   <- sample(which(meta$group == "Alzheimer's Disease"), 1)

# Build a long-format data frame for ggplot
density_df <- data.frame(
  expression = c(expr[, idx_ctrl], expr[, idx_mci], expr[, idx_ad]),
  group      = rep(c("Control", "Mild Cognitive Impairment", "Alzheimer's Disease"),
                   each = nrow(expr))
)

ggplot(density_df, aes(x = expression, colour = group)) +
  geom_density(linewidth = 1.2) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title  = "Expression Value Distribution (Three Representative Samples)",
    x      = "VST-transformed Expression",
    y      = "Density",
    colour = "Group"
  ) +
  theme_bw(base_size = 12)
```

**What to look for:**
- All three curves should have similar overall shape (approximately unimodal and bell-shaped after VST)
- Curves shifted dramatically relative to each other suggest normalization is incomplete
- A bimodal distribution (two humps) can indicate: two populations of samples mixed together, or a subset of miRNAs with very low expression pulling down one tail

---

## MODULE 3.2 — Dimensionality Reduction: PCA in Depth

### 3.2.1 The Problem PCA Solves

Our miRNA expression matrix has approximately 300 miRNAs and 148 samples. Visualizing the data directly would require a 300-dimensional plot — impossible for the human visual system. We need a way to **reduce** those 300 dimensions into 2 or 3 that capture the most important structure in the data.

Principal Component Analysis (PCA) does exactly this — and it does so in a mathematically principled way that tells us *which* combinations of miRNAs drive the largest sources of variation in the dataset.

---

### 3.2.2 Mathematical Intuition: The Covariance Analogy for Wet-Lab Biologists

Imagine you measured two proteins in 100 serum samples: Protein A (total tau) and Protein B (phosphorylated tau / p-tau). If these two proteins behave similarly across samples — samples with high total tau tend to also have high p-tau — their measurements are **correlated**. If you plotted all 100 samples as points in a 2D scatter plot (Protein A on x-axis, Protein B on y-axis), the points would form an elongated ellipse along a diagonal line.

That diagonal line is the **first principal component (PC1)** of this 2-protein dataset. It is the axis of maximum variance — the direction along which samples differ the most. A single number (the coordinate of each sample along PC1) captures most of the information that was spread across two original measurements.

PCA generalizes this to 300 dimensions simultaneously. Each principal component is a **linear combination** of all 300 miRNAs, chosen to capture a different dimension of variation. PC1 captures the largest source of variation; PC2 captures the second largest (and is mathematically perpendicular to PC1); and so on.

**The covariance matrix:** Computationally, PCA starts by calculating the **covariance matrix** of the data — a 300×300 matrix where each entry measures how much two miRNAs co-vary across samples. miRNAs that increase together in AD samples (like miR-21 and miR-146a, both elevated in neuroinflammation) will have high positive covariance. miRNAs that move in opposite directions (like miR-29a, downregulated in AD, and miR-146a, upregulated) will have negative covariance. PCA finds the axes that best summarize this entire covariance structure.

> **Key insight for biologists:** If PC1 separates AD samples from controls along a single axis, it means that a coordinated program of miRNA changes — a biological module — is the dominant source of variation in the dataset. The miRNAs that contribute most to PC1 (the "loadings") are the miRNAs most responsible for separating the groups. These are your first-pass candidate biomarkers — before any formal differential expression analysis.

---

### 3.2.3 Running PCA in R

```r
library(ggplot2)
library(ggrepel)

# ============================================================
# PCA — prcomp() function
# ============================================================
# Important: prcomp expects samples as ROWS and features as COLUMNS.
# Our expression matrix has miRNAs as rows and samples as columns.
# So we transpose: t(expr) = samples x miRNAs.
#
# scale. = TRUE: standardize each miRNA to mean 0 and SD 1 before PCA.
# This prevents miRNAs with high absolute expression from dominating PCA
# simply because they have larger absolute values.

pca_result <- prcomp(t(expr), scale. = TRUE)

# Variance explained by each PC
var_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100
cumulative_var <- cumsum(var_explained)

cat("=== Variance Explained by First 10 PCs ===\n")
for (i in 1:10) {
  cat(sprintf("PC%d: %5.1f%%   (cumulative: %5.1f%%)\n",
              i, var_explained[i], cumulative_var[i]))
}
```

---

### 3.2.4 The Scree Plot: How Many PCs Matter?

A **scree plot** shows the variance explained by each principal component. Its name comes from geology: "scree" is the loose rock debris at the base of a cliff, and the plot often shows a steep drop followed by a flat scree of small components.

```r
# ---- Scree plot ----
scree_df <- data.frame(
  PC          = factor(1:20, levels = 1:20),
  variance    = var_explained[1:20],
  cumulative  = cumulative_var[1:20]
)

p_scree <- ggplot(scree_df, aes(x = PC, y = variance)) +
  geom_col(fill = "#4575B4", width = 0.7, alpha = 0.85) +
  geom_line(aes(y = cumulative, group = 1), colour = "#D73027",
            linewidth = 1, linetype = "solid") +
  geom_point(aes(y = cumulative), colour = "#D73027", size = 2.5) +
  geom_hline(yintercept = 80, colour = "grey40", linetype = "dashed") +
  labs(
    title   = "Scree Plot — PCA on GSE120584 miRNA Expression",
    x       = "Principal Component",
    y       = "% Variance Explained",
    caption = "Red line: cumulative variance. Dashed line: 80% cumulative threshold."
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

print(p_scree)
ggsave("results/pca_scree_plot.png", p_scree, width = 8, height = 5, dpi = 150)
```

**How to interpret the scree plot:**
- A steep drop from PC1 to PC2 means one dominant source of variation exists (ideal: this is disease)
- A gradual decline means variation is spread across many PCs — the dataset has complex structure with multiple sources of variation (disease, age, sex, batch, other biology)
- The "elbow" in the curve (where the steep drop becomes flat) guides how many PCs to include in downstream analyses
- The cumulative variance line tells you: to retain 80% of total data variance, you need the first N PCs

> **Biological interpretation sidebar:** In a well-preprocessed AD miRNA dataset, you might expect PC1 to explain 15–30% of total variance and to correlate strongly with disease group (Control vs AD). PC2 and PC3 might capture age-related variation, sex differences, or miRNA co-expression modules tied to immune cell composition. If PC1 explains >50% of variance and does not correspond to any known biological variable, this suggests residual batch effects or a strongly dominant technical artifact that ComBat did not fully remove.

---

### 3.2.5 PC1/PC2 Scatter Plot: The Core Visualization

```r
# ---- Build PCA data frame for plotting ----
pca_df <- data.frame(
  PC1   = pca_result$x[, 1],
  PC2   = pca_result$x[, 2],
  PC3   = pca_result$x[, 3],
  PC4   = pca_result$x[, 4],
  Group = meta$group,
  Sex   = meta$sex,
  Age   = meta$age,
  row.names = rownames(pca_result$x)
)

# ---- PC1 vs PC2 coloured by disease group ----
p_pc12 <- ggplot(pca_df, aes(x = PC1, y = PC2,
                              colour = Group, shape = Group)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 17, 15)) +  # circle, triangle, square
  stat_ellipse(aes(group = Group, colour = Group),
               type = "norm", level = 0.80, linetype = "dashed",
               linewidth = 0.8) +
  labs(
    title  = "PCA: PC1 vs PC2 — Coloured by Disease Group",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour = "Group", shape = "Group"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "right")

print(p_pc12)
ggsave("results/pca_pc1_pc2_group.png", p_pc12, width = 8, height = 6, dpi = 150)
```

**What to look for in this plot:**
- AD samples shifted to one end of PC1 and control samples at the other → PC1 represents a disease axis — excellent
- MCI samples positioned between AD and control → the miRNA signature tracks disease progression — biologically compelling
- Groups completely overlapping → no linear separation; harder ML task, consider nonlinear methods (t-SNE/UMAP in Module 3.3)
- Outlier points far from their group's cluster → possible sample swaps, extreme cases, or misdiagnosed samples

---

### 3.2.6 Biplot: Which miRNAs Drive PC1?

A **biplot** overlays both the sample scores and the miRNA loadings on the same PCA axes. Arrows represent miRNAs; their direction and length indicate how much each miRNA contributes to each PC and in which direction.

```r
library(ggrepel)

# Extract loadings for top contributing miRNAs on PC1 and PC2
loadings <- as.data.frame(pca_result$rotation[, 1:2])
loadings$mirna <- rownames(loadings)
loadings$importance <- sqrt(loadings$PC1^2 + loadings$PC2^2)

# Keep top 20 most important miRNAs for biplot clarity
top_loadings <- loadings[order(loadings$importance, decreasing = TRUE)[1:20], ]

# Scale loadings to sample score range for plotting
scale_factor <- max(abs(c(pca_df$PC1, pca_df$PC2))) /
                max(abs(c(top_loadings$PC1, top_loadings$PC2))) * 0.6

p_biplot <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Group)) +
  geom_point(size = 2, alpha = 0.6) +
  scale_colour_manual(values = GROUP_COLOURS) +
  geom_segment(data = top_loadings,
               aes(x = 0, y = 0,
                   xend = PC1 * scale_factor,
                   yend = PC2 * scale_factor),
               arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
               colour = "grey30", linewidth = 0.5, inherit.aes = FALSE) +
  geom_text_repel(data = top_loadings,
                  aes(x = PC1 * scale_factor,
                      y = PC2 * scale_factor,
                      label = mirna),
                  size = 2.8, colour = "grey20",
                  max.overlaps = 20, inherit.aes = FALSE) +
  labs(
    title  = "PCA Biplot — Top 20 miRNA Loadings",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

print(p_biplot)
ggsave("results/pca_biplot.png", p_biplot, width = 9, height = 7, dpi = 150)
```

---

### 3.2.7 Loadings Table: The miRNA Contributions to PC1

The loadings table is often more informative than the biplot for identifying specific candidate miRNAs. A positive loading on PC1 means the miRNA increases along PC1; if AD samples are at the positive end of PC1, this miRNA is upregulated in AD.

```r
# Detailed loadings table for PC1
pc1_loadings <- data.frame(
  mirna   = rownames(pca_result$rotation),
  loading = pca_result$rotation[, 1],
  stringsAsFactors = FALSE
)
pc1_loadings <- pc1_loadings[order(abs(pc1_loadings$loading), decreasing = TRUE), ]

cat("=== Top 20 miRNA Loadings on PC1 ===\n")
cat("(Positive loading = higher expression in the positive-PC1 direction)\n\n")
print(head(pc1_loadings, 20))

# Save loadings table
write.csv(pc1_loadings, "results/pca_pc1_loadings.csv", row.names = FALSE)
cat("\nFull PC1 loadings saved to results/pca_pc1_loadings.csv\n")
```

> **Biological sidebar — Interpreting PC1 Loadings as a Disease Axis:**
> If PC1 separates AD from control samples, then the miRNAs with the largest absolute loadings on PC1 are the miRNAs most responsible for that separation. A miRNA like miR-21-5p with a strong positive loading on PC1, combined with AD samples sitting at the positive end of PC1, implies miR-21-5p is elevated in AD. Cross-reference this with miRBase and miRTarBase: if this miRNA targets PTEN (a tumor suppressor that also plays roles in neuronal survival) and PDCD4, the biology is consistent with the reported anti-apoptotic activity of miR-21 in AD. This cross-referencing step — going from a computational loading number to a biological mechanism — is exactly what distinguishes a computational biologist from a bioinformatician who only reports p-values.

---

### 3.2.8 PC3 and PC4 Check

Checking PC3 and PC4 often reveals secondary sources of variation — confounders or biological sub-groups.

```r
# PC3 vs PC4
p_pc34 <- ggplot(pca_df, aes(x = PC3, y = PC4,
                              colour = Group, shape = Group)) +
  geom_point(size = 3, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 17, 15)) +
  labs(
    title  = "PCA: PC3 vs PC4 — Check for Secondary Structure",
    x      = paste0("PC3 (", round(var_explained[3], 1), "% variance)"),
    y      = paste0("PC4 (", round(var_explained[4], 1), "% variance)")
  ) +
  theme_bw(base_size = 12)

# PC1 vs PC2 coloured by sex (confounder check)
p_pc12_sex <- ggplot(pca_df, aes(x = PC1, y = PC2,
                                  colour = Sex, shape = Group)) +
  geom_point(size = 3, alpha = 0.85) +
  labs(
    title  = "PCA: PC1 vs PC2 — Coloured by Sex",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)")
  ) +
  theme_bw(base_size = 12)

library(gridExtra)
grid.arrange(p_pc34, p_pc12_sex, ncol = 2)
ggsave("results/pca_pc34_sex_check.png",
       arrangeGrob(p_pc34, p_pc12_sex, ncol = 2),
       width = 14, height = 6, dpi = 150)
```

---

## MODULE 3.3 — t-SNE and UMAP: Nonlinear Dimensionality Reduction

### 3.3.1 Why PCA Is Not Enough

PCA is a **linear** method. It finds the best linear combinations of miRNAs to capture variance. But biological data is often organized in **nonlinear manifolds** — the shape of the data in high-dimensional space is curved, not flat.

Imagine you have three groups of patients (AD, MCI, Control) whose miRNA profiles differ from each other, but the differences are a combination of continuous disease severity and discrete biological sub-types. In high-dimensional space, the data might sit on a curved surface where PCA's flat planes capture the gross structure but miss the subtle local neighborhood relationships.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** and **UMAP (Uniform Manifold Approximation and Projection)** are nonlinear methods that preserve **local neighborhood structure** — samples that are similar to each other in high-dimensional space are placed near each other in 2D. This often reveals cluster structures that PCA cannot show.

---

### 3.3.2 t-SNE: How It Works (Conceptual Explanation)

t-SNE was introduced by van der Maaten and Hinton in 2008. It works in two steps:

**Step 1 (high-dimensional space):** For each sample, compute probabilities that reflect how similar it is to every other sample. Samples that are close together in high-dimensional expression space have high probability of being "neighbors."

**Step 2 (low-dimensional space):** Place all samples in a random 2D arrangement, then iteratively adjust their positions so that the neighborhood probabilities in 2D match those in the original high-dimensional space as closely as possible. Samples that were neighbors in 300D should end up near each other in 2D.

The result: samples from the same type (e.g., all AD samples with similar miRNA profiles) clump together into visible clusters in 2D, even if PCA failed to separate them clearly.

**The perplexity parameter** controls the effective number of neighbors considered for each sample. Think of it as a scale parameter: low perplexity (5–10) focuses on very local structure (tight, small clusters); high perplexity (50–100) captures broader structure. For most biological datasets with 100–500 samples, perplexity = 30 is a reasonable starting point.

---

### 3.3.3 t-SNE in Python

t-SNE is computationally expensive; for datasets above a few thousand samples, use Barnes-Hut t-SNE or UMAP instead. For our dataset (148 samples), standard t-SNE runs in seconds.

```python
# ============================================================
# t-SNE in Python using scikit-learn
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load data (exported from R as CSV in Week 3 R script, Section 9)
expr   = pd.read_csv("data/processed/GSE120584_expr_vf.csv", index_col=0)
meta   = pd.read_csv("data/processed/GSE120584_metadata_clean.csv", index_col=0)

# Transpose so samples are rows
X = expr.T.values     # shape: (n_samples, n_features)

# Standardize: mean 0, unit variance per feature
# VST already stabilises variance, but StandardScaler ensures equal weight
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- t-SNE with default perplexity ----
tsne = TSNE(
    n_components   = 2,
    perplexity     = 30,       # key hyperparameter (try 15, 30, 50)
    learning_rate  = "auto",   # recommended in sklearn >= 1.2
    n_iter         = 1000,
    random_state   = 42,       # set seed for reproducibility
    init           = "pca"     # initialise from PCA (more stable than random)
)
X_tsne = tsne.fit_transform(X_scaled)

# Build results dataframe
tsne_df = pd.DataFrame({
    "tSNE1"  : X_tsne[:, 0],
    "tSNE2"  : X_tsne[:, 1],
    "Group"  : meta["group"].values
}, index=meta.index)

# Plot
GROUP_COLORS = {
    "Control"                   : "#4575B4",
    "Mild Cognitive Impairment" : "#FEE090",
    "Alzheimer's Disease"       : "#D73027"
}

fig, ax = plt.subplots(figsize=(8, 6))
for group, colour in GROUP_COLORS.items():
    mask = tsne_df["Group"] == group
    ax.scatter(tsne_df.loc[mask, "tSNE1"],
               tsne_df.loc[mask, "tSNE2"],
               c=colour, label=group, s=50, alpha=0.85, edgecolors="none")
ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
ax.set_title("t-SNE (perplexity=30) — GSE120584 miRNA Expression", fontweight="bold")
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig("results/tsne_perplexity30.png", dpi=150)
plt.show()

# ---- Perplexity sensitivity check ----
# A responsible t-SNE analysis always checks multiple perplexity values.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, perp in zip(axes, [10, 30, 50]):
    ts = TSNE(n_components=2, perplexity=perp,
              learning_rate="auto", n_iter=1000,
              random_state=42, init="pca")
    coords = ts.fit_transform(X_scaled)
    for group, colour in GROUP_COLORS.items():
        mask = meta["group"].values == group
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, label=group, s=40, alpha=0.8, edgecolors="none")
    ax.set_title(f"t-SNE perplexity={perp}", fontweight="bold")
    ax.set_xlabel("tSNE1")
    ax.set_ylabel("tSNE2")
    handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, frameon=True)
plt.suptitle("t-SNE Sensitivity to Perplexity — GSE120584", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("results/tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

> **Biological sidebar — Reading a t-SNE Plot:**
> When you see AD samples clustered tightly together and control samples in a separate cluster, this means the AD samples share a miRNA expression pattern that is internally similar AND distinct from controls. This is what you hope to see — it suggests a consistent, reproducible AD miRNA signature exists in this dataset. If the AD cluster is diffuse and scattered across the plot, it may mean: (a) AD is biologically heterogeneous (true — AD has multiple subtypes), (b) the miRNA signal is weak relative to noise, or (c) confounders like age and sex are dominating the structure. The perplexity sensitivity check tells you whether the clusters you see are real: if a tight AD cluster appears at perplexity 10, 30, and 50, it is robust. If it only appears at one perplexity, it may be an artifact.

---

### 3.3.4 t-SNE Interpretation Pitfalls

t-SNE is one of the most commonly misinterpreted visualizations in computational biology. Here are the cardinal rules:

**Rule 1: Cluster sizes in t-SNE are meaningless.**
t-SNE distorts global distances to reveal local structure. A large cluster in t-SNE does not mean that group has more samples or greater biological variability. All clusters may appear similar in size even if the corresponding groups are biologically very different.

**Rule 2: Distances between clusters are meaningless.**
The distance between the AD cluster and the Control cluster in t-SNE 2D space cannot be interpreted as a measure of how different AD and control miRNA profiles are. Use PC1 separation or fold changes for that.

**Rule 3: Shapes within clusters are not interpretable.**
A crescent-shaped cluster, a circle, or a line — these shapes emerge from the t-SNE optimization algorithm and do not have direct biological meaning.

**Rule 4: t-SNE is not reproducible without a fixed random seed.**
Running t-SNE twice with different random seeds will produce different plots. Always set `random_state=42` (or any fixed value) and report it in your methods section.

**How to tell if t-SNE is overfit:**
- Run t-SNE with very low perplexity (5) and very high iterations (5000): if you get many tiny clusters of 1–3 samples each, the algorithm is fitting individual noise rather than real structure
- Compare with PCA: if t-SNE shows 8 clusters but PCA shows 3 loose clouds, the additional t-SNE clusters are likely artificial
- The number of t-SNE clusters should roughly match your biological expectations

---

### 3.3.5 UMAP: A Modern Alternative

UMAP (McInnes et al., 2018) addresses several t-SNE limitations:
- It is **much faster** (especially for large datasets)
- It **better preserves global structure** — clusters that are far apart in UMAP space truly are more dissimilar than clusters that are close
- It is more **stable** across different hyperparameter settings

The key hyperparameter is `n_neighbors`: the number of neighboring samples considered when constructing the local structure. Like perplexity in t-SNE, it controls the balance between local detail and global structure. Values of 10–30 are typical.

```python
# ============================================================
# UMAP in Python using umap-learn
# ============================================================
import umap

# Standard UMAP run
reducer = umap.UMAP(
    n_components  = 2,
    n_neighbors   = 15,        # key hyperparameter (try 5, 15, 30)
    min_dist      = 0.1,       # controls how tightly points are packed in 2D
    metric        = "euclidean",
    random_state  = 42
)
X_umap = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame({
    "UMAP1" : X_umap[:, 0],
    "UMAP2" : X_umap[:, 1],
    "Group" : meta["group"].values,
    "Age"   : meta["age"].values,
    "Sex"   : meta["sex"].values
}, index=meta.index)

# ---- UMAP coloured by group, age, and sex ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Group
for group, colour in GROUP_COLORS.items():
    mask = umap_df["Group"] == group
    axes[0].scatter(umap_df.loc[mask, "UMAP1"], umap_df.loc[mask, "UMAP2"],
                    c=colour, label=group, s=50, alpha=0.85, edgecolors="none")
axes[0].set_title("UMAP — Disease Group", fontweight="bold")
axes[0].legend(fontsize=9)

# Panel 2: Age (continuous colour scale)
sc = axes[1].scatter(umap_df["UMAP1"], umap_df["UMAP2"],
                     c=umap_df["Age"], cmap="RdYlBu_r", s=50, alpha=0.85)
plt.colorbar(sc, ax=axes[1], label="Age (years)")
axes[1].set_title("UMAP — Age (Continuous)", fontweight="bold")

# Panel 3: Sex
sex_colors = {"Male": "#2166AC", "Female": "#B2182B", "M": "#2166AC", "F": "#B2182B"}
for sex_label in umap_df["Sex"].unique():
    mask = umap_df["Sex"] == sex_label
    col = sex_colors.get(sex_label, "#999999")
    axes[2].scatter(umap_df.loc[mask, "UMAP1"], umap_df.loc[mask, "UMAP2"],
                    c=col, label=sex_label, s=50, alpha=0.85, edgecolors="none")
axes[2].set_title("UMAP — Sex", fontweight="bold")
axes[2].legend(fontsize=9)

for ax in axes:
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

plt.suptitle("UMAP (n_neighbors=15) — GSE120584 miRNA Expression",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/umap_group_age_sex.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- n_neighbors sensitivity check ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, nn in zip(axes, [5, 15, 30]):
    u = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1,
                  random_state=42)
    coords = u.fit_transform(X_scaled)
    for group, colour in GROUP_COLORS.items():
        mask = meta["group"].values == group
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, label=group, s=40, alpha=0.8, edgecolors="none")
    ax.set_title(f"UMAP n_neighbors={nn}", fontweight="bold")
handles = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
axes[0].legend(handles=handles, fontsize=9)
plt.suptitle("UMAP Sensitivity to n_neighbors — GSE120584")
plt.tight_layout()
plt.savefig("results/umap_nneighbors_comparison.png", dpi=150)
plt.show()
```

---

### 3.3.6 When to Use Each Method

| Situation | Recommended Method | Why |
|-----------|-------------------|-----|
| First exploratory look at data | **PCA** | Fast, interpretable loadings, reproducible |
| Searching for subtle cluster structure | **UMAP** | Better global structure preservation; faster than t-SNE |
| Confirming cluster structure for publication | **Both t-SNE and UMAP** | Clusters seen in both methods are more trustworthy |
| Dataset > 10,000 samples | **UMAP** | t-SNE becomes very slow at large N |
| Need to interpret which features drive clusters | **PCA** | Loadings are directly interpretable |
| Want to show disease trajectory (preclinical → MCI → AD) | **UMAP** | Better at capturing continuous manifold structure |

> **Critical note for biologists:** Neither t-SNE nor UMAP produces output that can be used as input to a statistical test. You cannot compute a p-value from a t-SNE cluster. You cannot say "the AD cluster is significantly different from the control cluster because they are far apart in t-SNE." These plots are visualizations for hypothesis generation, not statistical tests. The formal statistical comparisons come in Week 4 (differential expression analysis).

---

## MODULE 3.4 — Unsupervised Clustering

### 3.4.1 What Clustering Tells Us (and What It Does Not)

Unsupervised clustering asks: if we group samples based purely on their miRNA expression profiles, with no knowledge of clinical labels, do the resulting groups correspond to our known disease categories?

If they do — if a purely data-driven algorithm re-discovers the AD/Control/MCI grouping — this is strong evidence that the miRNA expression patterns genuinely differ between groups. It suggests that the AD signal is not just a collection of small effects scattered across hundreds of miRNAs, but a coherent coordinated pattern that is large enough to drive global sample clustering.

If they do not — if expression-based clusters split AD patients into two sub-groups or mix AD and MCI samples — this too is informative. It may indicate biological heterogeneity (AD sub-types), or that the clinical labels themselves are imprecise (some MCI patients may be further along the AD trajectory than their MMSE score suggests).

---

### 3.4.2 Hierarchical Clustering

Hierarchical clustering builds a tree (dendrogram) by iteratively merging the two most similar samples into a cluster, then merging clusters, until all samples are in one tree. The height at which branches merge indicates the dissimilarity between groups.

**Linkage method — Ward.D2:** Among the several available linkage methods (single, complete, average, Ward.D2), **Ward.D2 linkage** (Ward's minimum variance method with squared Euclidean distances) consistently performs best for gene expression data clustering. It minimizes the total within-cluster variance at each merge step, producing compact, well-separated clusters. This is why pheatmap uses Ward.D2 by default.

**Distance metric:** We use **Euclidean distance** between samples. In the VST-transformed log2 space, Euclidean distance is appropriate because expression differences are on a comparable scale across miRNAs (after scaling). Alternative: 1 minus Pearson correlation (correlation distance) is sometimes used when relative patterns matter more than absolute levels.

```r
# ============================================================
# Hierarchical clustering — hclust with Ward.D2
# ============================================================
library(cluster)
library(factoextra)

# Compute Euclidean distance matrix between samples
# t(expr): samples as rows, miRNAs as columns
dist_matrix <- dist(t(expr), method = "euclidean")

# Perform hierarchical clustering
hc <- hclust(dist_matrix, method = "ward.D2")

# Basic dendrogram plot
par(mar = c(5, 4, 4, 2))
plot(hc,
     main   = "Hierarchical Clustering Dendrogram — Ward.D2",
     labels = meta$group,
     cex    = 0.5,
     xlab   = "Samples",
     ylab   = "Height (Ward.D2 dissimilarity)")

# Add group-coloured rectangle at k = 3
rect.hclust(hc, k = 3, border = c("#4575B4", "#FEE090", "#D73027"))
```

> **Biological sidebar — Reading a Dendrogram:**
> In a hierarchical clustering dendrogram, samples that merge at a **low height** are very similar to each other. Samples that only merge at a high height (near the top of the tree) are very different. If the dendrogram naturally separates into three branches at height h, and those three branches correspond to your three clinical groups (Control, MCI, AD), this is strong evidence that the miRNA expression signature tracks disease stage. When you see AD and MCI samples interleaved in the dendrogram — mixing between branches — consider two biological interpretations: (1) some MCI patients are at an advanced stage similar to AD, or (2) some AD patients have a relatively mild molecular phenotype. Both are biologically plausible and worth noting in your results.

---

### 3.4.3 Determining Optimal Number of Clusters: Gap Statistic and Silhouette

The number of clusters k is not given to us by the algorithm — we must choose it. Two complementary methods help:

**Gap statistic (Tibshirani et al., 2001):** Compares the observed within-cluster variation with that expected under a null distribution of no clustering. The optimal k is where the gap statistic is highest, or where adding another cluster stops improving the gap by a meaningful amount. The gap statistic accounts for both the quality of clustering and the expected behavior by chance — it penalizes overfitting.

**Silhouette width:** For each sample, the silhouette measures how similar it is to its own cluster compared to the nearest neighboring cluster. Values range from -1 (wrong cluster) to +1 (perfectly assigned). The average silhouette width across all samples reaches a maximum at the optimal k.

```r
# ---- Gap statistic ----
set.seed(42)
gap_stat <- clusGap(
  t(expr),
  FUN     = hcut,           # hierarchical clustering function
  K.max   = 8,              # try up to 8 clusters
  B       = 50,             # number of bootstrap resamples (increase to 500 for publication)
  nstart  = 25
)

p_gap <- fviz_gap_stat(gap_stat) +
  labs(title = "Gap Statistic for Optimal Number of Clusters") +
  theme_bw(base_size = 12)
print(p_gap)
ggsave("results/gap_statistic.png", p_gap, width = 7, height = 5, dpi = 150)

cat("Optimal k by gap statistic:", which.max(gap_stat$Tab[, "gap"]), "\n")

# ---- Silhouette width ----
sil_widths <- numeric(7)
for (k in 2:8) {
  clusters  <- cutree(hc, k = k)
  sil       <- silhouette(clusters, dist_matrix)
  sil_widths[k - 1] <- mean(sil[, "sil_width"])
}

sil_df <- data.frame(k = 2:8, sil_width = sil_widths)

p_sil <- ggplot(sil_df, aes(x = k, y = sil_width)) +
  geom_line(colour = "#D73027", linewidth = 1.2) +
  geom_point(size = 3.5, colour = "#D73027") +
  geom_vline(xintercept = sil_df$k[which.max(sil_df$sil_width)],
             linetype = "dashed", colour = "grey40") +
  labs(
    title = "Silhouette Width vs Number of Clusters",
    x     = "Number of Clusters (k)",
    y     = "Average Silhouette Width"
  ) +
  theme_bw(base_size = 12)
print(p_sil)
ggsave("results/silhouette_width.png", p_sil, width = 7, height = 5, dpi = 150)

cat("Optimal k by silhouette:", sil_df$k[which.max(sil_df$sil_width)], "\n")
```

---

### 3.4.4 Cluster Purity Analysis

Once we have chosen k and cut the dendrogram, we evaluate how well the expression-based clusters match the known clinical labels — this is **cluster purity**.

```r
# Cut dendrogram at k = 3 (assuming this is optimal)
k_optimal <- 3
cluster_labels <- cutree(hc, k = k_optimal)

# Cross-tabulation: expression clusters vs clinical groups
purity_table <- table(
  Cluster        = factor(cluster_labels, labels = paste0("Cluster_", 1:k_optimal)),
  Clinical_Group = meta$group
)

cat("=== Cluster Purity Table ===\n")
cat("(Rows = expression-based clusters; Columns = known clinical labels)\n\n")
print(purity_table)

# Compute purity per cluster (dominant group / total in cluster)
purity_per_cluster <- apply(purity_table, 1, function(row) max(row) / sum(row))
overall_purity     <- sum(apply(purity_table, 1, max)) / sum(purity_table)

cat("\nPurity per cluster:\n")
print(round(purity_per_cluster, 3))
cat("\nOverall cluster purity:", round(overall_purity, 3), "\n")
cat("(1.0 = perfect; > 0.70 = acceptable for unsupervised discovery)\n")

write.csv(as.data.frame.matrix(purity_table),
          "results/cluster_purity_table.csv")
```

> **Biological interpretation:** An overall cluster purity of 0.75 means that 75% of samples are in the cluster dominated by their known clinical group. For an unsupervised analysis — which uses no label information — this is a strong result. It means the miRNA expression patterns are coherent enough within clinical groups that a label-blind algorithm recovers roughly the right groupings. A purity of 0.50 (random) would suggest the miRNA profiles do not differ systematically between groups — concerning for the downstream ML analysis.

---

### 3.4.5 k-means Clustering

k-means clustering assigns each sample to one of k clusters by iteratively minimizing the sum of squared distances from each sample to its cluster centroid. Unlike hierarchical clustering, k-means does not produce a dendrogram; it directly assigns cluster memberships.

k-means is faster and scales better to large datasets than hierarchical clustering. It is also more sensitive to the initial random placement of centroids, which is why running it multiple times (nstart) with different starting conditions is essential.

```r
# ---- k-means with k=3 ----
set.seed(42)
km <- kmeans(t(expr), centers = 3, nstart = 50, iter.max = 200)

# Cross-tabulation with clinical groups
km_purity <- table(
  kmeans_cluster = factor(km$cluster, labels = paste0("kCluster_", 1:3)),
  Clinical_Group  = meta$group
)
cat("=== k-means Cluster Purity ===\n")
print(km_purity)

km_overall_purity <- sum(apply(km_purity, 1, max)) / sum(km_purity)
cat("\nk-means overall purity:", round(km_overall_purity, 3), "\n")

# Visualize k-means result on PCA
pca_df$kmeans_cluster <- factor(km$cluster)

p_km_pca <- ggplot(pca_df, aes(x = PC1, y = PC2,
                                 colour = Group, shape = kmeans_cluster)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title  = "k-means Clustering (k=3) Overlaid on PCA",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour = "Clinical Group", shape = "k-means Cluster"
  ) +
  theme_bw(base_size = 12)
print(p_km_pca)
ggsave("results/kmeans_pca_overlay.png", p_km_pca, width = 8, height = 6, dpi = 150)
```

---

## MODULE 3.5 — Heatmap Visualization

### 3.5.1 The Heatmap as a Biological Reading Tool

A heatmap is a matrix where cells are colored by value — red for high expression, blue for low, white for average — and rows and columns are reordered by hierarchical clustering so that similar miRNAs and similar samples are adjacent. For miRNA expression data in AD research, a well-constructed heatmap is often the single most informative visualization in a paper.

Reading a heatmap requires understanding three layers simultaneously:
- **The color of each cell:** How is this miRNA expressed in this sample relative to average?
- **The row clustering:** miRNAs clustered together have correlated expression patterns — they may be co-regulated or involved in the same biological pathway
- **The column clustering:** Samples clustered together have similar overall miRNA profiles — they may belong to the same disease group, be the same sex, or have another shared characteristic

---

### 3.5.2 Building a Publication-Quality Heatmap in R

```r
library(pheatmap)
library(RColorBrewer)

# ============================================================
# Select top 50 most variable miRNAs for heatmap
# ============================================================
# Using IQR (robust to outliers) rather than SD
iqr_per_mirna   <- apply(expr, 1, IQR)
top50_mirnas    <- names(sort(iqr_per_mirna, decreasing = TRUE))[1:50]
expr_top50      <- expr[top50_mirnas, ]

# ============================================================
# Annotation tracks (sidebar bars on top of heatmap columns)
# ============================================================
# Prepare annotation data frame (one row per sample, matching column order)
annotation_col <- data.frame(
  Group = meta$group,
  row.names = colnames(expr_top50)
)

if ("sex" %in% colnames(meta)) {
  annotation_col$Sex <- factor(meta$sex)
}

if ("age" %in% colnames(meta) && !all(is.na(meta$age))) {
  annotation_col$Age_Group <- cut(
    meta$age,
    breaks = c(59, 69, 74, 79, 100),
    labels = c("60-69", "70-74", "75-79", "80+"),
    include.lowest = TRUE
  )
}

# Define colours for annotation tracks
ann_colours <- list(
  Group = GROUP_COLOURS,
  Sex   = c("Male"   = "#2166AC", "Female" = "#B2182B",
            "M"      = "#2166AC", "F"      = "#B2182B"),
  Age_Group = c("60-69" = "#EFF3FF", "70-74" = "#BDD7E7",
                "75-79" = "#6BAED6", "80+"   = "#2171B5")
)
ann_colours <- ann_colours[intersect(names(ann_colours), colnames(annotation_col))]

# ============================================================
# Generate the heatmap
# ============================================================
pheatmap(
  expr_top50,
  scale              = "row",            # z-score each miRNA across samples
  color              = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
  breaks             = seq(-3, 3, length.out = 101),  # clamp at +/- 3 SD
  clustering_method  = "ward.D2",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  annotation_col     = annotation_col,
  annotation_colors  = ann_colours,
  show_rownames      = TRUE,
  show_colnames      = FALSE,
  fontsize_row       = 7,
  main               = "Top 50 Most Variable miRNAs — GSE120584\n(row-scaled, Ward.D2 clustering)",
  filename           = "results/heatmap_top50_miRNAs.png",
  width              = 12,
  height             = 14
)
cat("Heatmap saved to results/heatmap_top50_miRNAs.png\n")
```

---

### 3.5.3 Row and Column Scaling — Why It Matters

**Row scaling** (subtracting each miRNA's mean and dividing by its SD across samples) is the standard approach for expression heatmaps. It ensures that the color in each cell reflects *relative* expression of that miRNA in that sample — is it higher or lower than this miRNA's average? Without row scaling, a single highly-expressed miRNA (like miR-21-5p with mean expression 12) would have all cells colored the same intense red and would visually dominate the heatmap, making it impossible to see the biological pattern in lower-expressed miRNAs.

**Column scaling** (scaling each sample rather than each miRNA) is rarely appropriate for expression heatmaps — it makes every sample look equally "variable," which destroys information about samples that are genuinely more aberrant than others.

**No scaling** is sometimes used when you want to show absolute expression levels — for example, when comparing one particular miRNA's absolute level across clinical groups for a clinical cutoff analysis. This is not typical for exploratory heatmaps.

---

### 3.5.4 How to Read a Heatmap Biologically

Step through these interpretive questions every time you look at a heatmap:

**1. Do the columns cluster by clinical group?**
If the column dendrogram separates AD samples on one side and control samples on the other, the miRNA expression signatures are sufficiently different to drive clustering. If samples are mixed, the signal may be too weak, or confounders are dominating.

**2. What do the top row clusters look like?**
The top rows in a Ward.D2 clustered heatmap (those that form the largest row clusters) represent the miRNAs with the most correlated expression patterns. A cluster of 5–10 miRNAs all showing high expression in AD samples (red column block) represents a co-regulated gene module — potentially all regulated by the same transcription factor, or all released from the same cell type in AD.

**3. Are there annotation track patterns?**
A perfect alignment between the Group annotation bar and column clusters is ideal. But watch the other tracks: if the Sex bar correlates perfectly with the column clusters (all females in one cluster, all males in another), sex may be confounding the disease signal. Similarly for Age_Group — if older patients cluster together regardless of disease status, age is a major source of variance.

**4. Which miRNAs are cluster-defining?**
Extract the row names in each main row cluster. For each cluster, ask: are these known AD-associated miRNAs? Look up their targets in miRTarBase. A cluster containing miR-29a, miR-107, and miR-132 — all known to target BACE1 or tau kinases — is biologically coherent and strongly suggests this is a real biological signal, not a noise artifact.

> **Specific miRNAs to look for in top cluster-defining rows:**
> - **miR-29 family (miR-29a-3p, miR-29b-3p, miR-29c-3p):** All target BACE1; expect downregulation in AD cluster rows
> - **miR-132-3p / miR-212-3p:** Neuroprotective; target tau kinases CDK5 and GSK3-beta; expect downregulation in AD
> - **miR-146a-5p:** Neuroinflammation master regulator via NF-kB; expect upregulation in AD
> - **miR-21-5p:** Anti-apoptotic; targets PTEN; expect upregulation in AD
> - **miR-107:** Targets BACE1 and Cofilin; one of earliest changed miRNAs; expect downregulation
> Seeing these specific miRNAs dominating the row clusters is a positive signal that the computational results align with the published biology — an important sanity check before building ML models.

---

## MODULE 3.6 — Identifying Confounders and Outliers

### 3.6.1 What Is a Confounder?

A **confounder** is a variable that is associated with both the exposure (disease group) and the outcome (miRNA expression), potentially creating a spurious or inflated association. In a blood miRNA AD study, the primary confounders are:

- **Age:** AD patients are older on average than controls; age independently affects miRNA expression in blood. If miR-34a increases with age, and AD patients are older than controls, a finding of "miR-34a upregulated in AD" might simply reflect age differences, not disease.
- **Sex:** AD disproportionately affects women (approximately 2:1 female:male), and sex influences many circulating miRNA levels. A miRNA found consistently elevated in women (regardless of disease status) could appear "AD-elevated" if the cohort has more female AD patients than female controls.
- **Cell-type composition:** Blood is a mixture of immune cells (monocytes, T-cells, B-cells, NK cells, platelets). If the immune cell proportions differ between AD patients and controls — which they do, due to AD-associated chronic inflammation — then miRNA changes attributed to AD may actually reflect changes in immune cell composition.

---

### 3.6.2 Quantifying Confounder Effects on Principal Components

The correlation between each principal component and a potential confounder variable tells us how much of the variance captured by that PC is attributable to the confounder rather than disease.

```r
# ============================================================
# Correlate each PC with age and sex
# ============================================================

# Build PC score matrix (samples x PCs)
pc_scores <- pca_result$x[, 1:10]  # first 10 PCs

# Age correlation (continuous variable — Pearson correlation)
age_cor_results <- data.frame(
  PC      = paste0("PC", 1:10),
  r_age   = NA_real_,
  p_age   = NA_real_,
  stringsAsFactors = FALSE
)

if ("age" %in% colnames(meta) && !all(is.na(meta$age))) {
  for (i in 1:10) {
    ct <- cor.test(pc_scores[, i], meta$age, method = "pearson")
    age_cor_results$r_age[i] <- ct$estimate
    age_cor_results$p_age[i] <- ct$p.value
  }
}

# Sex correlation (binary variable — point-biserial correlation via cor.test)
sex_cor_results <- data.frame(
  PC      = paste0("PC", 1:10),
  r_sex   = NA_real_,
  p_sex   = NA_real_,
  stringsAsFactors = FALSE
)

if ("sex" %in% colnames(meta)) {
  sex_binary <- as.numeric(factor(meta$sex)) - 1  # 0/1 encoding
  for (i in 1:10) {
    ct <- cor.test(pc_scores[, i], sex_binary, method = "pearson")
    sex_cor_results$r_sex[i] <- ct$estimate
    sex_cor_results$p_sex[i] <- ct$p.value
  }
}

# Combine into a single summary table
confounder_table <- data.frame(
  PC           = paste0("PC", 1:10),
  Var_Exp_pct  = round(var_explained[1:10], 2),
  r_age        = age_cor_results$r_age,
  p_age        = age_cor_results$p_age,
  r_sex        = sex_cor_results$r_sex,
  p_sex        = sex_cor_results$p_sex
)

cat("=== Confounder Correlation with Principal Components ===\n")
print(confounder_table)

write.csv(confounder_table, "results/pc_confounder_correlations.csv", row.names = FALSE)
```

---

### 3.6.3 Partial R²: Partitioning Variance Between Biology and Confounders

Pearson correlation tells us the direction and strength of association, but **partial R²** tells us the *proportion* of variance in each PC explained by each variable, holding other variables constant. This allows us to say, for example: "PC2 is 35% explained by age, 8% explained by sex, and 12% explained by disease group."

```r
# Partial R² using linear regression
# Model: PC1 ~ group + age + sex

if (all(c("age", "sex") %in% colnames(meta))) {

  model_df <- data.frame(
    PC1   = pc_scores[, 1],
    PC2   = pc_scores[, 2],
    group = meta$group,
    age   = meta$age,
    sex   = meta$sex
  )
  model_df <- na.omit(model_df)

  fit_pc1 <- lm(PC1 ~ group + age + sex, data = model_df)
  fit_pc2 <- lm(PC2 ~ group + age + sex, data = model_df)

  library(car)
  anova_pc1 <- Anova(fit_pc1, type = "II")
  anova_pc2 <- Anova(fit_pc2, type = "II")

  ss_total_pc1    <- sum(anova_pc1$"Sum Sq")
  partial_r2_pc1  <- anova_pc1$"Sum Sq" / ss_total_pc1

  ss_total_pc2    <- sum(anova_pc2$"Sum Sq")
  partial_r2_pc2  <- anova_pc2$"Sum Sq" / ss_total_pc2

  cat("=== Partial R-squared for PC1 ===\n")
  cat(sprintf("  Disease group: %.3f (%.1f%%)\n",
              partial_r2_pc1[1], partial_r2_pc1[1] * 100))
  cat(sprintf("  Age:           %.3f (%.1f%%)\n",
              partial_r2_pc1[2], partial_r2_pc1[2] * 100))
  cat(sprintf("  Sex:           %.3f (%.1f%%)\n\n",
              partial_r2_pc1[3], partial_r2_pc1[3] * 100))

  cat("=== Partial R-squared for PC2 ===\n")
  cat(sprintf("  Disease group: %.3f (%.1f%%)\n",
              partial_r2_pc2[1], partial_r2_pc2[1] * 100))
  cat(sprintf("  Age:           %.3f (%.1f%%)\n",
              partial_r2_pc2[2], partial_r2_pc2[2] * 100))
  cat(sprintf("  Sex:           %.3f (%.1f%%)\n",
              partial_r2_pc2[3], partial_r2_pc2[3] * 100))
}
```

> **Biological sidebar — How to respond to confounder findings:**
> If the partial R² analysis reveals that age explains 25% of PC1 and disease group explains only 15%, this is a serious finding with direct consequences for Week 4 modeling. It means that a significant portion of the miRNA variance your ML model will be trained on is driven by age differences, not disease biology. The appropriate response is: include age as a covariate in the DESeq2 design formula (`~ sex + age + group`), consider matching AD patients with controls on age distribution, and report this finding explicitly in your paper's methods section. This is not a failure of the study — confounders are present in virtually every real clinical dataset. Acknowledging and accounting for them is the mark of rigorous science.

---

### 3.6.4 Identifying Sample Outliers

Beyond formal QC metrics (NUSE, library size), EDA sometimes reveals outlier samples that passed Week 2 QC but are biologically anomalous — extreme samples in PCA/UMAP space that sit far from all other samples in their group.

```r
# ============================================================
# Identify outlier samples in PCA space using Mahalanobis distance
# ============================================================
library(MASS)

identify_pca_outliers <- function(pca_df_input, group_col, threshold_pct = 97.5) {
  outlier_flags <- rep(FALSE, nrow(pca_df_input))
  groups        <- unique(pca_df_input[[group_col]])

  for (g in groups) {
    idx    <- which(pca_df_input[[group_col]] == g)
    coords <- as.matrix(pca_df_input[idx, c("PC1", "PC2")])
    if (nrow(coords) < 4) next

    tryCatch({
      cov_matrix <- cov(coords)
      centroid   <- colMeans(coords)
      mah_dist   <- mahalanobis(coords, centroid, cov_matrix)
      threshold  <- qchisq(threshold_pct / 100, df = 2)
      outlier_flags[idx[mah_dist > threshold]] <- TRUE
    }, error = function(e) NULL)
  }
  outlier_flags
}

pca_df$is_outlier <- identify_pca_outliers(pca_df, "Group")

cat("Potential outlier samples:\n")
print(rownames(pca_df)[pca_df$is_outlier])

# Visualize outliers on PCA
p_outliers <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Group,
                                   shape = is_outlier)) +
  geom_point(aes(size = is_outlier), alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 4)) +
  scale_size_manual(values = c(2.5, 5)) +
  geom_text_repel(data = pca_df[pca_df$is_outlier, ],
                  aes(label = rownames(pca_df)[pca_df$is_outlier]),
                  size = 3, colour = "black") +
  labs(
    title  = "PCA — Mahalanobis Outlier Detection",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour = "Clinical Group"
  ) +
  theme_bw(base_size = 12)
print(p_outliers)
ggsave("results/pca_outlier_detection.png", p_outliers, width = 8, height = 6, dpi = 150)

# BIOLOGICAL DECISION RULE:
# An extreme AD sample with all known AD miRNA signatures elevated is NOT an error —
# it may be a severe case worth retaining.
# An outlier sample with a completely inverted signature (AD sample with control-like
# profile) warrants individual review: check metadata, re-examine clinical diagnosis,
# check for labeling errors before any exclusion decision.
```

---

## WEEK 3 LAB SESSIONS

### Lab 3A — R-Based EDA (90 minutes)

**Objective:** Generate a complete EDA report for GSE120584 in R, covering descriptive statistics, PCA, hierarchical clustering, and a publication-quality heatmap.

**Setup:**
```r
# Load clean data from Week 2 (confirm this file exists)
expr <- readRDS("data/processed/GSE120584_expr_clean.rds")
meta <- readRDS("data/processed/GSE120584_metadata_clean.rds")
cat("Expression matrix:", nrow(expr), "miRNAs x", ncol(expr), "samples\n")
cat("Groups:\n")
print(table(meta$group))
```

**Tasks (complete in order):**

1. **Descriptive statistics table (20 min):**
   - Run the per-miRNA statistics code from Module 3.1.2
   - Identify the top 5 most variable miRNAs (by CV) and look each one up in miRBase — what does each one do biologically?
   - How many miRNAs have >30% zeros? Does this concern you, and why?

2. **PCA and scree plot (25 min):**
   - Run `prcomp()` and generate the scree plot
   - Generate the PC1/PC2 scatter colored by group
   - How much variance does PC1 explain? Does it separate the three groups?
   - Print the top 10 PC1 loadings. Look up the number 1 loading miRNA in miRTarBase and identify its top 3 validated targets. Are any of these targets relevant to AD pathology?

3. **Hierarchical clustering (20 min):**
   - Run `hclust()` with Ward.D2 linkage
   - Run gap statistic and silhouette analysis (B=20 is sufficient for the lab session)
   - Cut the tree at k=3 and compute cluster purity
   - Does the purity suggest the expression profiles are separating by clinical group?

4. **Heatmap (25 min):**
   - Run the pheatmap code from Module 3.5.2
   - Examine the output: identify one row cluster that shows high expression in the AD column block and low expression in the Control block
   - List the miRNA names in that cluster and annotate them with their known AD biology

**Deliverable:** A brief written interpretation (1 paragraph per analysis, 4 paragraphs total) answering the biological questions raised above.

---

### Lab 3B — Python-Based t-SNE and UMAP (75 minutes)

**Objective:** Generate and critically evaluate t-SNE and UMAP visualizations for the same dataset, learning to distinguish real biological structure from algorithmic artifacts.

**Setup (in JupyterLab):**
```python
# The variance-filtered expression matrix is exported from R in Week 3 R script Section 9
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

expr = pd.read_csv("data/processed/GSE120584_expr_vf.csv", index_col=0)
meta = pd.read_csv("data/processed/GSE120584_metadata_clean.csv", index_col=0)

print(f"Expression matrix: {expr.shape[0]} miRNAs x {expr.shape[1]} samples")
print(f"\nGroup counts:\n{meta['group'].value_counts()}")
```

**Tasks (complete in order):**

1. **PCA in Python for comparison (10 min):**
   ```python
   from sklearn.decomposition import PCA

   X = expr.T.values
   X_scaled = StandardScaler().fit_transform(X)
   pca = PCA(n_components=10)
   X_pca = pca.fit_transform(X_scaled)
   print("Variance explained by PC1:", round(pca.explained_variance_ratio_[0] * 100, 1), "%")
   ```
   Does Python's sklearn PCA give the same PC1 variance explained as R's prcomp? (It should, approximately.)

2. **t-SNE sensitivity analysis (25 min):**
   - Run t-SNE at perplexity = 10, 30, and 50 (use the comparison code from Module 3.3.3)
   - For each perplexity: Does a distinct AD cluster emerge? Is it consistent?
   - At which perplexity does the clustering look most consistent with your PCA result?
   - Record: are there any individual samples that appear as isolated points far from their group's cluster at all perplexity values? These are the strongest outlier candidates.

3. **UMAP analysis (25 min):**
   - Run UMAP with n_neighbors = 5, 15, and 30
   - Color the same UMAP plot three times: by Group, by Age, by Sex
   - Which variable (Group, Age, Sex) seems to organize the UMAP structure most strongly?
   - Compare the UMAP with the t-SNE plot at perplexity=30: do you see the same clusters in both plots? If yes, the structure is more likely to be real.

4. **Critical evaluation (15 min):**
   - Identify one cluster boundary in your t-SNE plot that disappears or changes shape when you change perplexity. Explain why this boundary is unreliable.
   - Identify one cluster boundary that remains stable across all three perplexity values. This is a more trustworthy structure.
   - Write one sentence describing what you would conclude about the AD miRNA signature from these t-SNE and UMAP plots — what can you say and what can you not say?

**Deliverable:** A 4-panel figure (saved as PNG) showing: t-SNE at perplexity=30, UMAP colored by group, UMAP colored by age, UMAP colored by sex. Include a one-paragraph interpretation.

---

## WEEK 3 ASSIGNMENTS

### Reading Assignment (Required)
1. **Ringnér M (2008)** — *What is principal component analysis?* *Nature Biotechnology* 26(3):303–304. [DOI: 10.1038/nbt0308-303](https://doi.org/10.1038/nbt0308-303)  
   Focus on: Figure 1 (the geometric interpretation of PCA); the explanation of eigenvalues and loadings. This short commentary is the clearest non-mathematical explanation of PCA in the bioinformatics literature.

2. **Becht E et al. (2019)** — *Dimensionality reduction for visualizing single-cell data using UMAP* *Nature Biotechnology* 37:38–44. [DOI: 10.1038/nbt.4314](https://doi.org/10.1038/nbt.4314)  
   Focus on: The comparison between t-SNE and UMAP in Figure 2; the section on global structure preservation; the discussion of biological interpretation pitfalls.

### Reflection Questions (Discuss in Week 4 opening)
1. You run PCA and find that PC1 explains 22% of variance and corresponds to disease group, while PC2 explains 18% and corresponds strongly to age (partial R² = 0.38 for age). What does this tell you about the relative contribution of disease biology vs aging to the miRNA expression pattern? How would you account for this in your Week 4 ML model design?

2. Your t-SNE plot shows a tight, distinct cluster of 8 AD samples that sit far from the main AD cluster. In the heatmap, these same 8 samples form a separate block with a distinct miRNA expression pattern. Propose two biological explanations and one technical explanation for this sub-cluster. What experiments or metadata checks could help distinguish between them?

3. The cluster purity analysis returns an overall purity of 0.68. A colleague says this is "too low to be useful." What would you say in response? What is the appropriate comparison baseline, and what does purity of 0.68 actually tell us about the dataset?

### Practical Exercise
Using the heatmap generated in Lab 3A, extract the miRNA names from the largest cluster that shows high expression in AD samples and low expression in controls (a "high-in-AD" cluster). Look up these miRNAs in miRTarBase (https://mirtarbase.cuhk.edu.cn). Write a 3–5 sentence biological interpretation: What known targets do these miRNAs share? What biological pathways are they likely co-regulating? Is the pattern consistent with known AD biology?

---

## WEEK 3 GLOSSARY

| Term | Definition |
|------|------------|
| **Principal Component (PC)** | A linear combination of all features (miRNAs) that captures a specific dimension of variance; PCs are orthogonal (perpendicular) to each other and ordered by decreasing variance explained |
| **Loading** | The coefficient of each original variable (miRNA) in a principal component; a large absolute loading means that miRNA contributes strongly to that PC's variance |
| **Scree plot** | A bar chart of variance explained by each principal component in order; the "elbow" indicates the optimal number of PCs to retain |
| **Biplot** | A PCA visualization showing both sample scores and feature loadings (as arrows) on the same axes |
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding; nonlinear dimensionality reduction that preserves local neighborhood structure; cluster distances and sizes are not interpretable |
| **Perplexity** | The key hyperparameter of t-SNE; controls the effective number of neighbors; values of 10–50 are typical for biological datasets of 100–1000 samples |
| **UMAP** | Uniform Manifold Approximation and Projection; nonlinear dimensionality reduction that is faster than t-SNE and better preserves global data structure |
| **n_neighbors** | Key hyperparameter of UMAP; controls the balance between local and global structure; analogous to perplexity in t-SNE |
| **Hierarchical clustering** | An agglomerative algorithm that iteratively merges samples into a tree (dendrogram) based on pairwise distances; no k needs to be specified in advance |
| **Ward.D2 linkage** | A hierarchical clustering linkage method that minimizes within-cluster variance at each merge; produces compact, well-separated clusters; recommended for expression data |
| **Dendrogram** | The tree diagram output of hierarchical clustering; branch height represents dissimilarity between merged groups |
| **Gap statistic** | A method to determine optimal cluster number k by comparing observed within-cluster variation to that expected under a null distribution of no clustering |
| **Silhouette width** | Per-sample metric (range -1 to +1) measuring how well a sample fits its assigned cluster vs the nearest other cluster; average across samples maximizes at the optimal k |
| **Cluster purity** | The fraction of samples in each cluster that belong to the dominant clinical group; measures how well unsupervised clusters recover known labels |
| **Row scaling (z-score)** | Subtracting a miRNA's mean and dividing by its SD across all samples before heatmap plotting; ensures color reflects relative rather than absolute expression |
| **Coefficient of Variation (CV)** | SD / mean × 100%; a scale-free measure of relative variability; allows comparison of variability between miRNAs at different expression levels |
| **Confounder** | A variable associated with both the exposure (disease group) and the outcome (miRNA expression) that can create spurious or inflated associations |
| **Partial R-squared** | The proportion of variance in a response variable (e.g., PC1 score) explained by one predictor variable, holding all other predictors constant; used to partition variance between disease, age, and sex |
| **Mahalanobis distance** | A multivariate distance measure that accounts for correlations between dimensions; used to identify outlier samples in PCA space |
| **Zero inflation** | The phenomenon of having more zero values in a dataset than expected from the underlying statistical distribution; common in RNA-seq count data |

---

## KEY REFERENCES (Week 3)

All references retrieved from PubMed.

1. Jolliffe IT, Cadima J (2016). Principal component analysis: a review and recent developments. *Philos Trans R Soc A Math Phys Eng Sci* 374(2065):20150202. [DOI: 10.1098/rsta.2015.0202](https://doi.org/10.1098/rsta.2015.0202)

2. Ringnér M (2008). What is principal component analysis? *Nat Biotechnol* 26(3):303–304. [DOI: 10.1038/nbt0308-303](https://doi.org/10.1038/nbt0308-303)

3. van der Maaten L, Hinton G (2008). Visualizing data using t-SNE. *J Mach Learn Res* 9:2579–2605. Available at: http://www.jmlr.org/papers/v9/vandermaaten08a.html

4. McInnes L, Healy J, Melville J (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv* [stat.ML]. [DOI: 10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)

5. Becht E et al. (2019). Dimensionality reduction for visualizing single-cell data using UMAP. *Nat Biotechnol* 37:38–44. [DOI: 10.1038/nbt.4314](https://doi.org/10.1038/nbt.4314)

6. Tibshirani R, Walther G, Hastie T (2001). Estimating the number of clusters in a data set via the gap statistic. *J R Stat Soc Series B Stat Methodol* 63(2):411–423. [DOI: 10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293)

7. Rousseeuw PJ (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J Comput Appl Math* 20:53–65. [DOI: 10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

8. Huber W et al. (2015). Orchestrating high-throughput genomic analysis with Bioconductor. *Nat Methods* 12(2):115–121. [DOI: 10.1038/nmeth.3252](https://doi.org/10.1038/nmeth.3252)

9. Kolde R (2019). pheatmap: Pretty Heatmaps. R package version 1.0.12. Available: https://CRAN.R-project.org/package=pheatmap

10. Leek JT, Storey JD (2007). Capturing heterogeneity in gene expression studies by surrogate variable analysis. *PLoS Genet* 3(9):e161. [DOI: 10.1371/journal.pgen.0030161](https://doi.org/10.1371/journal.pgen.0030161)

---

## NEXT WEEK PREVIEW

**Week 4: Feature Selection and Classical Machine Learning**

Having explored the structure of the data, we are now ready to build the first supervised machine learning models for AD classification.

In Week 4 you will:
- Run formal differential expression analysis (DESeq2 and limma) to identify miRNAs that are statistically significantly different between AD, MCI, and Control
- Understand the multiple testing problem and apply Benjamini-Hochberg FDR correction — what an adjusted p-value of 0.05 actually means when testing 300 miRNAs simultaneously
- Build your first ML classifiers: Logistic Regression, Linear SVM, and Random Forest
- Evaluate model performance with ROC curves, AUC, sensitivity, and specificity
- Implement cross-validation to get honest estimates of out-of-sample performance
- Apply filter-based (variance, fold-change) and embedded (LASSO) feature selection to reduce the 300-miRNA space to a focused biomarker panel of 10–30 miRNAs

The exploratory analyses you completed this week — particularly the PC1 loadings table and the heatmap cluster-defining miRNAs — will serve as a biological reference point for evaluating the Week 4 ML results. If your random forest identifies miR-29a and miR-146a as top features, and you already know from this week's heatmap that these miRNAs show strong group-dependent patterns, the biological coherence of the ML result is confirmed. If the ML identifies miRNAs you have never seen in the EDA, that is a red flag worth investigating carefully.

**Data object you will take into Week 4:**
`data/processed/GSE120584_expr_varianceFiltered.rds` — the variance-filtered expression matrix (top 75% most variable miRNAs by IQR), ready for differential expression and ML feature engineering.
