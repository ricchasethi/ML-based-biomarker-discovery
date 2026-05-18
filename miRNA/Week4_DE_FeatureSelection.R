# =============================================================================
# Week 4: Differential Expression & Feature Selection
# AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease
# =============================================================================
#
# COURSE CONTEXT:
#   This script performs the R-side work for Week 4: formal differential
#   expression (DE) analysis to rank miRNAs by statistical evidence for
#   AD-associated changes, followed by filter-based feature selection and
#   export of the feature matrix for Python-based ML classifiers (Lab 4B).
#
# TWO DATASETS:
#   GSE120584 — serum small RNA-seq (primary dataset; 148 samples)
#               Analysed with DESeq2 (negative binomial model for counts)
#   GSE46579  — whole blood Affymetrix microarray (validation dataset)
#               Analysed with limma (linear model on RMA-normalised values)
#
# THREE COMPARISONS on GSE120584:
#   1. Alzheimer's Disease (AD) vs Control
#   2. Mild Cognitive Impairment (MCI) vs Control
#   3. AD vs MCI
#
# FEATURE SELECTION (R-side, univariate filter):
#   Mann-Whitney U test (Wilcoxon rank-sum) per miRNA
#   Benjamini-Hochberg FDR correction
#   Export ranked feature matrix for Python ML (Lab 4B)
#
# PREREQUISITES (run Weeks 2 & 3 scripts first):
#   data/processed/GSE120584_counts_filtered.rds
#   data/processed/GSE120584_metadata_clean.rds
#   data/processed/GSE120584_expr_varianceFiltered.rds
#   data/processed/GSE46579_expr_rma.rds
#   data/processed/GSE46579_metadata_clean.rds
#
# OUTPUTS (written to results/):
#   volcano_deseq2_AD_vs_Control.png
#   volcano_deseq2_MCI_vs_Control.png
#   volcano_deseq2_AD_vs_MCI.png
#   ma_plot_deseq2_AD_vs_Control.png
#   volcano_limma_AD_vs_Control.png
#   ma_plot_limma_AD_vs_Control.png
#   de_results_deseq2_AD_vs_Control.csv  (+ _MCI_vs_Control, _AD_vs_MCI)
#   de_results_limma_AD_vs_Control.csv
#   overlap_deseq2_limma_AD.csv
#   venn_de_overlap.png
#   mwu_filter_features.csv
#   consensus_features_Week4.csv
#
# Exported for Python ML (Lab 4B):
#   data/processed/GSE120584_expr_forML.csv
#   data/processed/GSE120584_labels_binary.csv  (AD=1, Control=0)
#
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Package loading
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(DESeq2)       # negative binomial DE for RNA-seq count data
  library(limma)        # linear models for DE (microarray and RNA-seq)
  library(edgeR)        # DGEList and filterByExpr helper functions
  library(ggplot2)      # publication-quality plots
  library(ggrepel)      # non-overlapping labels on volcano plots
  library(dplyr)        # data manipulation (filter, arrange, mutate)
})

# Create results directory if it does not exist
dir.create("results", showWarnings = FALSE)

# Shared colour palette
GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",
  "Mild Cognitive Impairment" = "#FEE090",
  "Alzheimer's Disease"       = "#D73027"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 4 — Differential Expression & Feature Selection\n")
cat("====================================================\n\n")

# GSE120584: RNA-seq — use raw filtered counts for DESeq2
counts_filtered <- readRDS("data/processed/GSE120584_counts_filtered.rds")
metadata        <- readRDS("data/processed/GSE120584_metadata_clean.rds")
expr_vf         <- readRDS("data/processed/GSE120584_expr_varianceFiltered.rds")

cat("GSE120584 (RNA-seq):\n")
cat("  Filtered count matrix:", nrow(counts_filtered), "miRNAs ×",
    ncol(counts_filtered), "samples\n")
cat("  Variance-filtered VST matrix:", nrow(expr_vf), "miRNAs ×",
    ncol(expr_vf), "samples\n")
cat("  Groups:\n")
print(table(metadata$group))

# GSE46579: microarray — use RMA-normalised matrix for limma
rma_available <- file.exists("data/processed/GSE46579_expr_rma.rds") &&
                 file.exists("data/processed/GSE46579_metadata_clean.rds")

if (rma_available) {
  expr_rma  <- readRDS("data/processed/GSE46579_expr_rma.rds")
  meta_46   <- readRDS("data/processed/GSE46579_metadata_clean.rds")
  cat("\nGSE46579 (microarray):\n")
  cat("  RMA-normalised matrix:", nrow(expr_rma), "probes ×",
      ncol(expr_rma), "samples\n")
  cat("  Groups:\n")
  print(table(meta_46$group))
} else {
  cat("\nGSE46579 not found — skipping limma validation analysis.\n")
  cat("  (Run Week 2 script first to generate GSE46579 processed files)\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DESeq2 setup: DESeqDataSet construction
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY DESeq2 for RNA-seq:
#   Raw counts follow a negative binomial distribution (variance > mean due
#   to overdispersion from biological variation between samples).
#   DESeq2 models this explicitly, unlike a simple t-test which assumes
#   normality.
#
# DESIGN FORMULA:
#   ~ group           — basic model: test effect of disease group
#   ~ sex + age + group — recommended if age/sex confounding detected in Week 3
#
# RELEVEL: Control is set as the reference level so all comparisons are
#   "vs Control" by default, which is the biologically meaningful direction.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 2: DESeq2 setup ─────────────────────────────────────────────\n\n")

metadata$group <- factor(metadata$group,
                          levels = c("Control",
                                     "Mild Cognitive Impairment",
                                     "Alzheimer's Disease"))

# Build DESeq2 design: extend to ~ sex + age + group if covariates are available
if (all(c("age", "sex") %in% colnames(metadata)) &&
    !all(is.na(metadata$age)) && !all(is.na(metadata$sex))) {
  design_formula <- ~ sex + age + group
  cat("Using design: ~ sex + age + group (covariates included)\n")
} else {
  design_formula <- ~ group
  cat("Using design: ~ group (no covariates available)\n")
}

dds <- DESeqDataSetFromMatrix(
  countData = counts_filtered,
  colData   = metadata,
  design    = design_formula
)

# Control is the reference — all comparisons will be relative to Control
dds$group <- relevel(dds$group, ref = "Control")

cat("DESeqDataSet constructed:\n")
cat("  Samples:", ncol(dds), "\n")
cat("  miRNAs:", nrow(dds), "\n")
cat("  Reference level:", levels(dds$group)[1], "\n\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Run DESeq2
# ─────────────────────────────────────────────────────────────────────────────
#
# DESeq() performs three steps:
#   (1) estimateSizeFactors — median-of-ratios normalisation
#   (2) estimateDispersions — empirical Bayes dispersion estimation across miRNAs
#   (3) nbinomWaldTest     — fit negative binomial GLM and run Wald tests
#
# This may take 2–10 minutes depending on the machine.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 3: Running DESeq2 — please wait... ─────────────────────────\n\n")

dds <- DESeq(dds)

cat("DESeq2 run complete.\n")
cat("Estimated model coefficients:\n")
print(resultsNames(dds))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Extract DE results with LFC shrinkage
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY SHRINKAGE (lfcShrink):
#   Low-count miRNAs have very noisy fold change estimates.  A miRNA with
#   1 count in one group and 3 counts in another looks like a 3-fold change,
#   but this is almost certainly noise.  lfcShrink pulls extreme fold changes
#   from noisy features toward zero while preserving reliable estimates from
#   well-detected features.  This prevents false positives and is required
#   for ranked feature lists used in ML.
#
# METHOD CHOICE:
#   apeglm — best shrinkage for single coefficients (AD vs Control, MCI vs Control)
#   ashr   — required for arbitrary contrasts (AD vs MCI, which is not a direct
#            coefficient in our Control-reference design)
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 4: Extracting DE results with LFC shrinkage ─────────────────\n\n")

# ── 4a: AD vs Control ────────────────────────────────────────────────────
cat("Computing: AD vs Control...\n")
coef_ad_ctrl <- grep("Alzheimer", resultsNames(dds), value = TRUE)
if (length(coef_ad_ctrl) == 0) coef_ad_ctrl <- resultsNames(dds)[3]

res_AD_vs_Control <- lfcShrink(dds, coef = coef_ad_ctrl, type = "apeglm")
res_AD_df <- as.data.frame(res_AD_vs_Control)
res_AD_df$miRNA <- rownames(res_AD_df)
res_AD_df <- res_AD_df[order(res_AD_df$padj, na.last = TRUE), ]

sig_AD <- res_AD_df[!is.na(res_AD_df$padj) &
                    res_AD_df$padj < 0.05 &
                    abs(res_AD_df$log2FoldChange) > 0.5, ]

cat("=== AD vs Control ===\n")
cat("Total miRNAs tested:", nrow(res_AD_df), "\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.5):", nrow(sig_AD), "\n")
cat("  Upregulated in AD:   ", sum(sig_AD$log2FoldChange > 0), "\n")
cat("  Downregulated in AD: ", sum(sig_AD$log2FoldChange < 0), "\n")
cat("\nTop 15 DE miRNAs (AD vs Control):\n")
print(head(sig_AD[, c("miRNA", "log2FoldChange", "lfcSE", "pvalue", "padj")], 15))

write.csv(res_AD_df, "results/de_results_deseq2_AD_vs_Control.csv",
          row.names = FALSE)

# ── 4b: MCI vs Control ───────────────────────────────────────────────────
cat("\nComputing: MCI vs Control...\n")
coef_mci_ctrl <- grep("Impairment|MCI", resultsNames(dds), value = TRUE)
if (length(coef_mci_ctrl) == 0) coef_mci_ctrl <- resultsNames(dds)[2]

res_MCI_vs_Control <- lfcShrink(dds, coef = coef_mci_ctrl, type = "apeglm")
res_MCI_df <- as.data.frame(res_MCI_vs_Control)
res_MCI_df$miRNA <- rownames(res_MCI_df)
res_MCI_df <- res_MCI_df[order(res_MCI_df$padj, na.last = TRUE), ]

sig_MCI <- res_MCI_df[!is.na(res_MCI_df$padj) &
                      res_MCI_df$padj < 0.05 &
                      abs(res_MCI_df$log2FoldChange) > 0.5, ]

cat("=== MCI vs Control ===\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.5):", nrow(sig_MCI), "\n")
write.csv(res_MCI_df, "results/de_results_deseq2_MCI_vs_Control.csv",
          row.names = FALSE)

# ── 4c: AD vs MCI ────────────────────────────────────────────────────────
# Not a direct coefficient (Control-referenced design); use contrast + ashr
cat("\nComputing: AD vs MCI (contrast-based, using ashr shrinkage)...\n")
res_AD_vs_MCI <- lfcShrink(
  dds,
  contrast = c("group", "Alzheimer's Disease", "Mild Cognitive Impairment"),
  type     = "ashr"
)
res_AD_MCI_df <- as.data.frame(res_AD_vs_MCI)
res_AD_MCI_df$miRNA <- rownames(res_AD_MCI_df)
res_AD_MCI_df <- res_AD_MCI_df[order(res_AD_MCI_df$padj, na.last = TRUE), ]

sig_AD_MCI <- res_AD_MCI_df[!is.na(res_AD_MCI_df$padj) &
                              res_AD_MCI_df$padj < 0.05 &
                              abs(res_AD_MCI_df$log2FoldChange) > 0.5, ]

cat("=== AD vs MCI ===\n")
cat("Significant (FDR < 0.05, |log2FC| > 0.5):", nrow(sig_AD_MCI), "\n")
write.csv(res_AD_MCI_df, "results/de_results_deseq2_AD_vs_MCI.csv",
          row.names = FALSE)

# Three-way summary
cat("\n=== Three-Way DE Summary ===\n")
cat("  MCI vs Control (early-detection candidates): ", nrow(sig_MCI), "\n")
cat("  AD vs Control  (disease-stage markers):      ", nrow(sig_AD), "\n")
cat("  AD vs MCI      (progression markers):        ", nrow(sig_AD_MCI), "\n")

# Progressive markers: same direction in both MCI vs Ctrl AND AD vs Ctrl
if (nrow(sig_MCI) > 0 && nrow(sig_AD) > 0) {
  common_both  <- intersect(sig_MCI$miRNA, sig_AD$miRNA)
  # Keep only those with same directionality in both comparisons
  common_df    <- sig_AD[sig_AD$miRNA %in% common_both, ]
  mci_dir      <- sign(sig_MCI$log2FoldChange[match(common_df$miRNA, sig_MCI$miRNA)])
  ad_dir       <- sign(common_df$log2FoldChange)
  progressive  <- common_df$miRNA[mci_dir == ad_dir & !is.na(mci_dir)]
  cat("\n  Progressive (same-direction DE in both MCI and AD):", length(progressive), "\n")
  if (length(progressive) > 0) {
    cat("  Progressive markers:", paste(progressive, collapse = ", "), "\n")
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Volcano and MA plots (DESeq2)
# ─────────────────────────────────────────────────────────────────────────────
#
# VOLCANO PLOT: log2FC (x-axis) vs -log10(p-value) (y-axis)
#   Points in upper-right: strongly upregulated AND highly significant in AD
#   Points in upper-left: strongly downregulated AND highly significant in AD
#   Key convention: colour by significance + direction; label top 15 by FDR
#
# MA PLOT: average expression (A) vs log2FC (M)
#   A well-normalised dataset shows the cloud of points centred at M = 0
#   across all expression levels.  A systematic trend at low expression
#   indicates a normalisation artefact.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 5: Volcano and MA plots (DESeq2) ────────────────────────────\n\n")

make_volcano <- function(de_df, fdr_col, lfc_col, title, outfile) {
  plot_df <- de_df
  plot_df$miRNA <- if ("miRNA" %in% colnames(plot_df)) plot_df$miRNA else
                   rownames(plot_df)
  plot_df$significance <- "Not Significant"
  plot_df$significance[!is.na(plot_df[[fdr_col]]) &
                       plot_df[[fdr_col]] < 0.05 &
                       plot_df[[lfc_col]] >  0.5] <- "Up in AD"
  plot_df$significance[!is.na(plot_df[[fdr_col]]) &
                       plot_df[[fdr_col]] < 0.05 &
                       plot_df[[lfc_col]] < -0.5] <- "Down in AD"
  plot_df$significance <- factor(plot_df$significance,
                                 levels = c("Not Significant",
                                            "Up in AD", "Down in AD"))

  # Top 15 for labelling (by smallest FDR)
  label_df <- plot_df[!is.na(plot_df[[fdr_col]]), ]
  label_df <- head(label_df[order(label_df[[fdr_col]]), ], 15)

  # Use pvalue column for y-axis (not padj — keeps the shape informative)
  p_col <- if ("pvalue" %in% colnames(plot_df)) "pvalue" else "P.Value"

  p <- ggplot(plot_df[!is.na(plot_df[[p_col]]), ],
              aes(x = .data[[lfc_col]],
                  y = -log10(.data[[p_col]] + 1e-300),
                  colour = significance)) +
    geom_point(alpha = 0.55, size = 1.5) +
    geom_point(data = label_df, size = 2.5, alpha = 0.9) +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    geom_text_repel(data = label_df,
                    aes(label = miRNA), size = 2.8,
                    max.overlaps = 20, box.padding = 0.4,
                    colour = "black") +
    scale_colour_manual(
      values = c("Not Significant" = "grey70",
                 "Up in AD"        = "#D73027",
                 "Down in AD"      = "#4575B4")) +
    labs(title   = title,
         x       = "log2 Fold Change (AD / Control)",
         y       = "-log10(p-value)",
         colour  = NULL,
         caption = "Dashed lines: |log2FC| > 0.5 and p < 0.05 (uncorrected)") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"), legend.position = "top")

  ggsave(outfile, p, width = 8, height = 6, dpi = 150)
  cat("Saved:", outfile, "\n")
  invisible(p)
}

make_volcano(res_AD_df,
             fdr_col = "padj", lfc_col = "log2FoldChange",
             title   = "Volcano Plot: AD vs Control (DESeq2, GSE120584)",
             outfile = "results/volcano_deseq2_AD_vs_Control.png")

make_volcano(res_MCI_df,
             fdr_col = "padj", lfc_col = "log2FoldChange",
             title   = "Volcano Plot: MCI vs Control (DESeq2, GSE120584)",
             outfile = "results/volcano_deseq2_MCI_vs_Control.png")

make_volcano(res_AD_MCI_df,
             fdr_col = "padj", lfc_col = "log2FoldChange",
             title   = "Volcano Plot: AD vs MCI (DESeq2, GSE120584)",
             outfile = "results/volcano_deseq2_AD_vs_MCI.png")

# MA plot for AD vs Control
ma_df <- res_AD_df
ma_df$significance <- "Not Significant"
ma_df$significance[!is.na(ma_df$padj) & ma_df$padj < 0.05 &
                   ma_df$log2FoldChange >  0.5] <- "Up in AD"
ma_df$significance[!is.na(ma_df$padj) & ma_df$padj < 0.05 &
                   ma_df$log2FoldChange < -0.5] <- "Down in AD"
ma_df$significance <- factor(ma_df$significance,
                              levels = c("Not Significant", "Up in AD", "Down in AD"))

p_ma <- ggplot(ma_df[!is.na(ma_df$baseMean) & ma_df$baseMean > 0, ],
               aes(x = log2(baseMean + 1), y = log2FoldChange,
                   colour = significance)) +
  geom_point(alpha = 0.5, size = 1.2) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.5) +
  geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed",
             colour = "grey40", linewidth = 0.4) +
  scale_colour_manual(values = c("Not Significant" = "grey70",
                                 "Up in AD"        = "#D73027",
                                 "Down in AD"      = "#4575B4")) +
  labs(title   = "MA Plot: AD vs Control (DESeq2, GSE120584)",
       x       = "log2(Mean Normalised Count + 1)  [A]",
       y       = "log2 Fold Change (AD/Control)  [M]",
       colour  = NULL,
       caption = "Centred cloud at M=0 across all A values = good normalisation") +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "top")
ggsave("results/ma_plot_deseq2_AD_vs_Control.png", p_ma, width = 8, height = 5, dpi = 150)
cat("MA plot saved to results/ma_plot_deseq2_AD_vs_Control.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — limma-voom pipeline for GSE46579 (microarray validation)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY LIMMA FOR MICROARRAY:
#   Affymetrix microarray data is already on a continuous log2 scale after
#   RMA normalisation (Week 2), so we do NOT use the voom transformation
#   (which is for RNA-seq counts → log2-CPM).  We use limma directly on the
#   RMA matrix.
#
# eBayes moderation: "borrows strength" across all miRNAs to stabilise
#   variance estimates.  Each miRNA's variance is moderated toward a global
#   prior estimate, dramatically improving power with small N.
# ─────────────────────────────────────────────────────────────────────────────

if (rma_available) {
  cat("\n─── Section 6: limma pipeline (GSE46579 microarray) ────────────────────\n\n")

  # Restrict to AD and Control for the binary comparison
  keep_46 <- meta_46$group %in% c("Control", "Alzheimer's Disease")
  expr_rma_bin <- expr_rma[, keep_46]
  meta_46_bin  <- meta_46[keep_46, ]

  meta_46_bin$group <- factor(meta_46_bin$group,
                               levels = c("Control", "Alzheimer's Disease"))

  # Design matrix (no intercept — makes contrasts cleaner)
  design_46 <- model.matrix(~ 0 + group, data = meta_46_bin)
  colnames(design_46) <- levels(meta_46_bin$group)

  cat("Design matrix dimensions:", nrow(design_46), "samples ×",
      ncol(design_46), "group columns\n")

  # Fit linear models to every miRNA simultaneously
  fit_46 <- lmFit(expr_rma_bin, design_46)

  # Define contrast: AD - Control
  contrast_matrix <- makeContrasts(
    AD_vs_Control = "Alzheimer's Disease" - Control,
    levels = design_46
  )
  fit2_46 <- contrasts.fit(fit_46, contrast_matrix)

  # eBayes: moderate variance estimates across all miRNAs
  # trend = TRUE models the mean-variance relationship (recommended for arrays)
  fit2_46 <- eBayes(fit2_46, trend = TRUE)

  # Extract ranked results table (all miRNAs, sorted by FDR)
  results_limma <- topTable(
    fit2_46,
    coef    = "AD_vs_Control",
    number  = Inf,
    adjust  = "BH",
    sort.by = "P"
  )
  results_limma$miRNA <- rownames(results_limma)

  sig_limma <- results_limma[results_limma$adj.P.Val < 0.05 &
                               abs(results_limma$logFC) > 0.5, ]

  cat("=== GSE46579 limma Results: AD vs Control ===\n")
  cat("Total miRNAs tested:", nrow(results_limma), "\n")
  cat("Significant (FDR < 0.05, |logFC| > 0.5):", nrow(sig_limma), "\n")
  cat("  Upregulated in AD:  ", sum(sig_limma$logFC > 0), "\n")
  cat("  Downregulated in AD:", sum(sig_limma$logFC < 0), "\n")
  cat("\nTop 15 DE miRNAs (limma, GSE46579):\n")
  print(head(sig_limma[, c("miRNA", "logFC", "AveExpr", "t",
                            "P.Value", "adj.P.Val")], 15))

  write.csv(results_limma, "results/de_results_limma_AD_vs_Control.csv",
            row.names = FALSE)
  cat("\nFull limma results saved to results/de_results_limma_AD_vs_Control.csv\n")

  # Volcano plot — limma
  make_volcano_limma <- function(de_df, title, outfile) {
    plot_df <- de_df
    plot_df$significance <- "Not Significant"
    plot_df$significance[plot_df$adj.P.Val < 0.05 & plot_df$logFC >  0.5] <- "Up in AD"
    plot_df$significance[plot_df$adj.P.Val < 0.05 & plot_df$logFC < -0.5] <- "Down in AD"
    plot_df$significance <- factor(plot_df$significance,
                                   levels = c("Not Significant",
                                              "Up in AD", "Down in AD"))
    label_df <- head(plot_df[order(plot_df$adj.P.Val), ], 15)

    p <- ggplot(plot_df, aes(x = logFC, y = -log10(P.Value + 1e-300),
                              colour = significance)) +
      geom_point(alpha = 0.55, size = 1.5) +
      geom_point(data = label_df, size = 2.5) +
      geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed",
                 colour = "grey40", linewidth = 0.4) +
      geom_hline(yintercept = -log10(0.05), linetype = "dashed",
                 colour = "grey40", linewidth = 0.4) +
      geom_text_repel(data = label_df, aes(label = miRNA), size = 2.8,
                      max.overlaps = 20, box.padding = 0.4, colour = "black") +
      scale_colour_manual(values = c("Not Significant" = "grey70",
                                     "Up in AD"        = "#D73027",
                                     "Down in AD"      = "#4575B4")) +
      labs(title = title, x = "log2 Fold Change (AD / Control)",
           y = "-log10(P-value)", colour = NULL) +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(face = "bold"), legend.position = "top")
    ggsave(outfile, p, width = 8, height = 6, dpi = 150)
    cat("Saved:", outfile, "\n")
  }

  make_volcano_limma(
    results_limma,
    title   = "Volcano Plot: AD vs Control (limma, GSE46579 microarray)",
    outfile = "results/volcano_limma_AD_vs_Control.png"
  )

  # MA plot — limma
  p_ma_lm <- ggplot(results_limma,
                    aes(x = AveExpr, y = logFC,
                        colour = ifelse(adj.P.Val < 0.05 & abs(logFC) > 0.5,
                                        ifelse(logFC > 0, "Up in AD",
                                               "Down in AD"),
                                        "Not Significant"))) +
    geom_point(alpha = 0.5, size = 1.2) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.5) +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dashed",
               colour = "grey40", linewidth = 0.4) +
    scale_colour_manual(values = c("Not Significant" = "grey70",
                                   "Up in AD"        = "#D73027",
                                   "Down in AD"      = "#4575B4")) +
    labs(title   = "MA Plot: AD vs Control (limma, GSE46579)",
         x       = "Average log2 Expression (A)",
         y       = "log2 Fold Change (M)",
         colour  = NULL) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"), legend.position = "top")
  ggsave("results/ma_plot_limma_AD_vs_Control.png",
         p_ma_lm, width = 8, height = 5, dpi = 150)
  cat("limma MA plot saved to results/ma_plot_limma_AD_vs_Control.png\n")

} else {
  cat("\n─── Section 6: Skipped (GSE46579 files not found) ───────────────────────\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Cross-dataset overlap analysis
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: miRNAs that are independently significant in BOTH the RNA-seq (GSE120584)
#   and the microarray (GSE46579) validation dataset are far more trustworthy
#   than those significant in only one dataset.  Dataset-specific findings can
#   reflect platform artefacts, cohort-specific confounding, or batch effects.
#   The overlap is the most conservative and most credible feature list.
# ─────────────────────────────────────────────────────────────────────────────

if (rma_available && exists("sig_limma")) {
  cat("\n─── Section 7: Cross-dataset overlap analysis ───────────────────────────\n\n")

  sig_deseq2_names <- sig_AD$miRNA
  sig_limma_names  <- sig_limma$miRNA

  overlap_both <- intersect(sig_deseq2_names, sig_limma_names)
  only_deseq2  <- setdiff(sig_deseq2_names, sig_limma_names)
  only_limma   <- setdiff(sig_limma_names, sig_deseq2_names)

  cat("=== Cross-Dataset DE Overlap: AD vs Control ===\n")
  cat("DESeq2 significant (GSE120584):", length(sig_deseq2_names), "\n")
  cat("limma significant (GSE46579): ", length(sig_limma_names), "\n")
  cat("Overlap (both datasets):       ", length(overlap_both), "\n")
  cat("Only in DESeq2:                ", length(only_deseq2), "\n")
  cat("Only in limma:                 ", length(only_limma), "\n")

  if (length(overlap_both) > 0) {
    cat("\nOverlapping miRNAs:\n")
    print(overlap_both)

    # Save overlap with FC from both datasets
    overlap_df <- merge(
      sig_AD[sig_AD$miRNA %in% overlap_both,
             c("miRNA", "log2FoldChange", "padj")],
      sig_limma[sig_limma$miRNA %in% overlap_both,
                c("miRNA", "logFC", "adj.P.Val")],
      by = "miRNA"
    )
    colnames(overlap_df)[2:5] <- c("log2FC_DESeq2", "padj_DESeq2",
                                    "logFC_limma",   "padj_limma")
    overlap_df$direction_consistent <-
      sign(overlap_df$log2FC_DESeq2) == sign(overlap_df$logFC_limma)

    write.csv(overlap_df, "results/overlap_deseq2_limma_AD.csv", row.names = FALSE)
    cat("\nOverlap table saved to results/overlap_deseq2_limma_AD.csv\n")
  }

  # Simple Venn diagram using base R text output
  cat("\n=== Venn Diagram (text) ===\n")
  cat("┌──────────────────────────────────────────────┐\n")
  cat("│ DESeq2 only │  Both  │   limma only           │\n")
  cat(sprintf("│ %-12d│  %-5d │   %-20d│\n",
              length(only_deseq2), length(overlap_both), length(only_limma)))
  cat("└──────────────────────────────────────────────┘\n")

} else {
  cat("\n─── Section 7: Skipped (GSE46579 not available) ─────────────────────────\n")
  overlap_both <- character(0)
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Univariate filter feature selection (Mann-Whitney U in R)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Before exporting the feature matrix for Python ML, we apply a fast
#   univariate filter to remove the most obviously uninformative miRNAs.
#   The Mann-Whitney U test (= Wilcoxon rank-sum test) is non-parametric:
#   it does not assume normally distributed expression values.
#
# The result: a ranked miRNA list based on p-value, to be imported into
#   the Python ML notebook (Lab 4B) for further processing.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 8: Mann-Whitney U filter feature selection ──────────────────\n\n")

# Use VST variance-filtered matrix; restrict to AD vs Control binary comparison
ad_ctrl_mask <- metadata$group %in% c("Control", "Alzheimer's Disease")
expr_bin     <- expr_vf[, ad_ctrl_mask]
meta_bin     <- metadata[ad_ctrl_mask, ]

cat("Binary comparison subset: AD vs Control\n")
cat("Samples:", ncol(expr_bin), "—",
    sum(meta_bin$group == "Alzheimer's Disease"), "AD,",
    sum(meta_bin$group == "Control"), "Control\n")
cat("Running Mann-Whitney U test for each of", nrow(expr_bin), "miRNAs...\n")

# Compute Mann-Whitney U p-value for every miRNA
mw_results <- data.frame(
  miRNA  = rownames(expr_bin),
  W      = NA_real_,
  pvalue = NA_real_,
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(expr_bin))) {
  group0 <- expr_bin[i, meta_bin$group == "Control"]
  group1 <- expr_bin[i, meta_bin$group == "Alzheimer's Disease"]
  wt     <- wilcox.test(group0, group1, exact = FALSE)
  mw_results$W[i]      <- wt$statistic
  mw_results$pvalue[i] <- wt$p.value
}

# Benjamini-Hochberg FDR correction
mw_results$padj <- p.adjust(mw_results$pvalue, method = "BH")
mw_results <- mw_results[order(mw_results$pvalue), ]

# Per-miRNA mean expression in each group (for direction annotation)
mw_results$mean_AD   <- rowMeans(expr_bin[mw_results$miRNA,
                                           meta_bin$group == "Alzheimer's Disease"])
mw_results$mean_Ctrl <- rowMeans(expr_bin[mw_results$miRNA,
                                           meta_bin$group == "Control"])
mw_results$direction  <- ifelse(mw_results$mean_AD > mw_results$mean_Ctrl,
                                "Up in AD", "Down in AD")

cat("\n=== Top 20 miRNAs by Mann-Whitney U p-value ===\n")
print(head(mw_results[, c("miRNA", "pvalue", "padj", "direction")], 20))
cat("\nMiRNAs with FDR < 0.05:", sum(mw_results$padj < 0.05, na.rm = TRUE), "\n")
cat("miRNAs with FDR < 0.20:", sum(mw_results$padj < 0.20, na.rm = TRUE), "\n")

write.csv(mw_results, "results/mwu_filter_features.csv", row.names = FALSE)
cat("Mann-Whitney U ranked feature list saved to results/mwu_filter_features.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Consensus feature table (DESeq2 + MWU)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: A miRNA that appears in the top features from MULTIPLE independent
#   ranking methods is far more likely to represent a true biological signal.
#   This consensus list is the starting point for Week 5's nested CV and the
#   most credible input for the ML classifiers in Lab 4B.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 9: Consensus feature table ─────────────────────────────────\n\n")

# Top 50 from each method
top50_deseq2 <- head(res_AD_df$miRNA[!is.na(res_AD_df$padj)], 50)
top50_mwu    <- head(mw_results$miRNA, 50)

# Assign appearance counts
all_features <- unique(c(top50_deseq2, top50_mwu))
consensus_df <- data.frame(
  miRNA         = all_features,
  in_DESeq2     = all_features %in% top50_deseq2,
  in_MWU        = all_features %in% top50_mwu,
  stringsAsFactors = FALSE
)
consensus_df$n_methods <- as.integer(consensus_df$in_DESeq2) +
                          as.integer(consensus_df$in_MWU)

# Add cross-dataset overlap flag if available
if (length(overlap_both) > 0) {
  consensus_df$in_limma_overlap <- all_features %in% overlap_both
  consensus_df$n_methods <- consensus_df$n_methods +
                            as.integer(consensus_df$in_limma_overlap)
}

# Add DESeq2 fold change and FDR
consensus_df <- merge(
  consensus_df,
  res_AD_df[, c("miRNA", "log2FoldChange", "padj")],
  by = "miRNA", all.x = TRUE
)
colnames(consensus_df)[colnames(consensus_df) == "log2FoldChange"] <- "log2FC_DESeq2"
colnames(consensus_df)[colnames(consensus_df) == "padj"]           <- "padj_DESeq2"

consensus_df <- consensus_df[order(-consensus_df$n_methods,
                                    consensus_df$padj_DESeq2,
                                    na.last = TRUE), ]

cat("=== Consensus Feature Summary ===\n")
cat("In top 50 of BOTH DESeq2 and MWU:",
    sum(consensus_df$n_methods >= 2, na.rm = TRUE), "miRNAs\n")
cat("In top 50 of only one method:    ",
    sum(consensus_df$n_methods == 1, na.rm = TRUE), "miRNAs\n")
cat("\nTop 20 consensus features (appearing in most methods):\n")
print(head(consensus_df[, c("miRNA", "n_methods", "log2FC_DESeq2", "padj_DESeq2",
                              "in_DESeq2", "in_MWU")], 20))

write.csv(consensus_df, "results/consensus_features_Week4.csv", row.names = FALSE)
cat("\nFull consensus feature table saved to results/consensus_features_Week4.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Export feature matrix for Python ML (Lab 4B)
# ─────────────────────────────────────────────────────────────────────────────
#
# The Python Jupyter notebook (Week4_ML_Classifier.ipynb) needs:
#   (1) Feature matrix: samples × miRNAs  (CSV, miRNAs as columns)
#   (2) Sample labels: binary AD=1 / Control=0
#
# STRATEGY: Export the VST variance-filtered matrix for the AD vs Control
#   binary subset.  The Python notebook applies its own Mann-Whitney filter
#   (reproducing Section 8 logic) before ML training — giving the notebook
#   the same starting data but the flexibility to adjust the feature filter.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 10: Export feature matrix for Python ML ─────────────────────\n\n")

# Transpose: Python expects samples as rows, features as columns
expr_forML <- as.data.frame(t(expr_bin))  # nrow = samples, ncol = miRNAs
labels_binary <- data.frame(
  sample = rownames(meta_bin),
  group  = meta_bin$group,
  label  = as.integer(meta_bin$group == "Alzheimer's Disease"),
  row.names = rownames(meta_bin)
)

write.csv(expr_forML,    "data/processed/GSE120584_expr_forML.csv",
          row.names = TRUE)
write.csv(labels_binary, "data/processed/GSE120584_labels_binary.csv",
          row.names = FALSE)

cat("Feature matrix exported:\n")
cat("  data/processed/GSE120584_expr_forML.csv\n")
cat(sprintf("  Dimensions: %d samples × %d miRNAs\n",
            nrow(expr_forML), ncol(expr_forML)))
cat("\nBinary labels exported:\n")
cat("  data/processed/GSE120584_labels_binary.csv\n")
cat(sprintf("  AD: %d, Control: %d\n",
            sum(labels_binary$label == 1), sum(labels_binary$label == 0)))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Session summary
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 4 — DE & Feature Selection Summary\n")
cat("====================================================\n\n")

cat("DESeq2 results (GSE120584):\n")
cat(sprintf("  AD vs Control  — significant: %d (up: %d, down: %d)\n",
            nrow(sig_AD),
            sum(sig_AD$log2FoldChange > 0), sum(sig_AD$log2FoldChange < 0)))
cat(sprintf("  MCI vs Control — significant: %d\n", nrow(sig_MCI)))
cat(sprintf("  AD vs MCI      — significant: %d\n", nrow(sig_AD_MCI)))

if (rma_available && exists("sig_limma")) {
  cat(sprintf("\nlimma results (GSE46579):\n"))
  cat(sprintf("  AD vs Control  — significant: %d\n", nrow(sig_limma)))
  cat(sprintf("  Cross-dataset overlap:        %d\n", length(overlap_both)))
}

cat(sprintf("\nMann-Whitney U filter (AD vs Control):\n"))
cat(sprintf("  FDR < 0.05: %d  |  FDR < 0.20: %d\n",
            sum(mw_results$padj < 0.05, na.rm = TRUE),
            sum(mw_results$padj < 0.20, na.rm = TRUE)))

cat(sprintf("\nConsensus features (in ≥ 2 methods): %d\n",
            sum(consensus_df$n_methods >= 2, na.rm = TRUE)))

cat("\nFiles written to results/:\n")
cat("  de_results_deseq2_AD_vs_Control.csv\n")
cat("  de_results_deseq2_MCI_vs_Control.csv\n")
cat("  de_results_deseq2_AD_vs_MCI.csv\n")
cat("  volcano_deseq2_*.png  |  ma_plot_deseq2_AD_vs_Control.png\n")
if (rma_available && exists("sig_limma")) {
  cat("  de_results_limma_AD_vs_Control.csv\n")
  cat("  volcano_limma_*.png   |  ma_plot_limma_AD_vs_Control.png\n")
  cat("  overlap_deseq2_limma_AD.csv\n")
}
cat("  mwu_filter_features.csv\n")
cat("  consensus_features_Week4.csv\n")

cat("\nFiles for Python ML (Lab 4B):\n")
cat("  data/processed/GSE120584_expr_forML.csv\n")
cat("  data/processed/GSE120584_labels_binary.csv\n")

cat("\n─────────────────────────────────────────────────────\n")
cat("PROCEED TO:\n")
cat("  Lab 4B — Open Week4_ML_Classifier.ipynb in JupyterLab\n")
cat("  Week 5  — Open Week5_Validation.R in RStudio\n")
cat("─────────────────────────────────────────────────────\n\n")

sessionInfo()
