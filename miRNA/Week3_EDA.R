# =============================================================================
# Week 3: Exploratory Data Analysis (EDA)
# AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease
# =============================================================================
#
# COURSE CONTEXT:
#   This script performs comprehensive EDA on the QC-filtered, batch-corrected
#   miRNA expression matrix produced in Week 2.  All analyses are done with
#   no knowledge of class labels during the computational steps — this is
#   UNSUPERVISED exploration.  We then overlay known clinical labels on the
#   outputs to interpret what the data-driven structure means biologically.
#
# PREREQUISITES (run Week 2 script first):
#   data/processed/GSE120584_expr_clean.rds     — VST-normalised expr matrix
#   data/processed/GSE120584_metadata_clean.rds — sample metadata data.frame
#   data/processed/GSE120584_counts_filtered.rds — raw filtered counts (for
#                                                    variance-filtered export)
#
# OUTPUTS (written to results/):
#   pca_scree_plot.png, pca_pc1_pc2_group.png, pca_biplot.png
#   pca_pc34_sex_check.png, pca_outlier_detection.png
#   pca_pc1_loadings.csv
#   gap_statistic.png, silhouette_width.png
#   heatmap_top50_miRNAs.png
#   kmeans_pca_overlay.png
#   cluster_purity_table.csv
#   pc_confounder_correlations.csv
#   zero_inflation_histogram.png
#   density_plot_representative.png
#
# Also exported for Python (t-SNE / UMAP in Lab 3B):
#   data/processed/GSE120584_expr_vf.csv        — variance-filtered expr (CSV)
#   data/processed/GSE120584_metadata_clean.csv — metadata (CSV)
#   data/processed/GSE120584_expr_varianceFiltered.rds
#
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Package loading and directory setup
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ggplot2)       # publication-quality plots
  library(ggrepel)       # non-overlapping text labels on plots
  library(gridExtra)     # arrange multiple ggplot panels
  library(pheatmap)      # annotated heatmap
  library(RColorBrewer)  # colour palettes
  library(cluster)       # silhouette() and clusGap()
  library(factoextra)    # fviz_gap_stat()
  library(car)           # Anova() for partial R² (Type II SS)
  library(MASS)          # mahalanobis()
})

# Create results directory if it does not exist
dir.create("results", showWarnings = FALSE)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load clean data from Week 2
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 3 EDA — Loading data from Week 2 outputs\n")
cat("====================================================\n\n")

expr <- readRDS("data/processed/GSE120584_expr_clean.rds")
meta <- readRDS("data/processed/GSE120584_metadata_clean.rds")

# Quick sanity check
cat("Expression matrix dimensions:", nrow(expr), "miRNAs x", ncol(expr), "samples\n")
cat("Sample groups:\n")
print(table(meta$group))

# Verify that the sample order in expr columns matches meta rows
stopifnot(all(colnames(expr) == rownames(meta)))
cat("\nSample order in expression matrix and metadata: OK\n")

# ── Shared colour palette (used throughout all plots) ──────────────────────
GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",
  "Mild Cognitive Impairment" = "#FEE090",
  "Alzheimer's Disease"       = "#D73027"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Per-miRNA descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Before applying any algorithm, characterise the basic statistical
#      properties of every miRNA across all samples.
#
# KEY METRICS:
#   Mean   — average expression level
#   SD     — absolute variability
#   CV     — SD/mean × 100 %: scale-free relative variability
#   IQR    — robust to outlier samples
#   %zeros — fraction of samples where this miRNA was undetected
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 2: Per-miRNA descriptive statistics ───────────────────────\n\n")

mirna_stats <- data.frame(
  mirna       = rownames(expr),
  mean_expr   = rowMeans(expr),
  sd_expr     = apply(expr, 1, sd),
  median_expr = apply(expr, 1, median),
  iqr_expr    = apply(expr, 1, IQR),
  pct_zeros   = rowSums(expr == 0) / ncol(expr) * 100,
  stringsAsFactors = FALSE
)

# Coefficient of Variation (CV): SD / mean × 100%
mirna_stats$cv <- mirna_stats$sd_expr / mirna_stats$mean_expr * 100

# Sort by CV (most variable first)
mirna_stats <- mirna_stats[order(mirna_stats$cv, decreasing = TRUE), ]

cat("=== Top 10 Most Variable miRNAs (by CV — relative variability) ===\n")
print(head(mirna_stats[, c("mirna", "mean_expr", "sd_expr", "cv", "pct_zeros")], 10))

cat("\n=== Top 10 Most Stable miRNAs (by CV) ===\n")
cat("(candidates for reference normalization controls)\n\n")
print(tail(mirna_stats[, c("mirna", "mean_expr", "sd_expr", "cv", "pct_zeros")], 10))

# Save the full statistics table
write.csv(mirna_stats, "results/mirna_descriptive_stats.csv", row.names = FALSE)
cat("\nFull statistics table saved to results/mirna_descriptive_stats.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Zero-inflation visualisation
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: RNA-seq data has two types of zeros:
#   (1) Structural zeros — miRNA genuinely absent in that sample
#   (2) Sampling zeros   — miRNA expressed at very low levels, missed by chance
# miRNAs with >50% zeros rarely carry useful ML signal.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 3: Zero-inflation analysis ────────────────────────────────\n\n")

cat("miRNAs with >20% zeros:", sum(mirna_stats$pct_zeros > 20), "\n")
cat("miRNAs with >50% zeros:", sum(mirna_stats$pct_zeros > 50), "\n")
cat("(If >10% of remaining miRNAs have zero fraction > 50%, tighten Week 2 filter)\n\n")

p_zero <- ggplot(mirna_stats, aes(x = pct_zeros)) +
  geom_histogram(bins = 40, fill = "#4575B4", colour = "white", alpha = 0.8) +
  geom_vline(xintercept = 20, colour = "red", linetype = "dashed", linewidth = 0.8) +
  geom_vline(xintercept = 50, colour = "darkred", linetype = "dashed", linewidth = 0.8) +
  labs(
    title   = "Distribution of Zero Percentage Across miRNAs",
    x       = "% Samples with Zero Expression",
    y       = "Number of miRNAs",
    caption = "Red dashed lines: 20% and 50% zero thresholds"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/zero_inflation_histogram.png", p_zero, width = 7, height = 5, dpi = 150)
cat("Zero-inflation histogram saved to results/zero_inflation_histogram.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Distribution shape: density plots
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: After VST, expression values should be approximately normally distributed
#      within each sample.  All three density curves should have similar shape.
#      Dramatic shifts indicate incomplete normalisation.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 4: Distribution shape (density plots) ─────────────────────\n\n")

set.seed(42)
idx_ctrl <- sample(which(meta$group == "Control"), 1)
idx_mci  <- sample(which(meta$group == "Mild Cognitive Impairment"), 1)
idx_ad   <- sample(which(meta$group == "Alzheimer's Disease"), 1)

density_df <- data.frame(
  expression = c(expr[, idx_ctrl], expr[, idx_mci], expr[, idx_ad]),
  group      = rep(
    c("Control", "Mild Cognitive Impairment", "Alzheimer's Disease"),
    each = nrow(expr)
  )
)

p_density <- ggplot(density_df, aes(x = expression, colour = group)) +
  geom_density(linewidth = 1.2) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title   = "Expression Value Distribution (Three Representative Samples)",
    x       = "VST-transformed Expression",
    y       = "Density",
    colour  = "Group",
    caption = "Curves should have similar shape after VST normalisation"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/density_plot_representative.png", p_density, width = 8, height = 5, dpi = 150)
cat("Density plot saved to results/density_plot_representative.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Variance-filtered expression matrix
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Retaining only the top 75% most variable miRNAs (by IQR) removes
#      low-variance features that contribute noise but not biological signal.
#      IQR is robust to outlier samples; SD can be inflated by a single extreme
#      sample.  This filtered matrix is the input for all downstream analyses.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 5: Variance filtering (top 75% by IQR) ────────────────────\n\n")

iqr_per_mirna <- apply(expr, 1, IQR)
iqr_threshold <- quantile(iqr_per_mirna, 0.25)   # bottom 25% removed
expr_vf       <- expr[iqr_per_mirna >= iqr_threshold, ]

cat("Original matrix:", nrow(expr), "miRNAs\n")
cat("After variance filtering (IQR ≥ 25th percentile):", nrow(expr_vf), "miRNAs\n\n")

# Save variance-filtered matrix as both RDS (for R) and CSV (for Python)
saveRDS(expr_vf, "data/processed/GSE120584_expr_varianceFiltered.rds")
write.csv(expr_vf,
          "data/processed/GSE120584_expr_vf.csv",
          row.names = TRUE)
write.csv(as.data.frame(t(meta)),
          "data/processed/GSE120584_metadata_clean.csv",
          row.names = TRUE)

cat("Variance-filtered matrix saved:\n")
cat("  RDS: data/processed/GSE120584_expr_varianceFiltered.rds\n")
cat("  CSV: data/processed/GSE120584_expr_vf.csv\n")
cat("Metadata CSV: data/processed/GSE120584_metadata_clean.csv\n")
cat("  (Python t-SNE/UMAP scripts in Lab 3B read these CSVs)\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Principal Component Analysis (PCA)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: PCA is the first dimensionality reduction step.  It finds linear
#      combinations of miRNAs that explain the most variance.  PC1 loadings
#      tell us which miRNAs drive the largest source of variation in the data
#      (ideally: disease group).
#
# prcomp() notes:
#   • Input must be transposed: samples as rows, miRNAs as columns
#   • scale. = TRUE: standardises each miRNA to mean 0, SD 1 before PCA.
#     This prevents highly-expressed miRNAs from dominating PCA purely
#     because they have larger absolute values.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 6: Principal Component Analysis ────────────────────────────\n\n")

# Use the variance-filtered matrix for PCA
pca_result  <- prcomp(t(expr_vf), scale. = TRUE)
var_explained  <- (pca_result$sdev^2) / sum(pca_result$sdev^2) * 100
cumulative_var <- cumsum(var_explained)

cat("=== Variance Explained by First 10 PCs ===\n")
for (i in 1:10) {
  cat(sprintf("PC%d: %5.1f%%   (cumulative: %5.1f%%)\n",
              i, var_explained[i], cumulative_var[i]))
}
cat("\n")

# ── 6a: Scree plot ────────────────────────────────────────────────────────
scree_df <- data.frame(
  PC         = factor(1:20, levels = 1:20),
  variance   = var_explained[1:20],
  cumulative = cumulative_var[1:20]
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

ggsave("results/pca_scree_plot.png", p_scree, width = 8, height = 5, dpi = 150)
cat("Scree plot saved to results/pca_scree_plot.png\n")

# ── 6b: Build PCA data frame for all scatter plots ───────────────────────
pca_df <- data.frame(
  PC1   = pca_result$x[, 1],
  PC2   = pca_result$x[, 2],
  PC3   = pca_result$x[, 3],
  PC4   = pca_result$x[, 4],
  Group = meta$group,
  Sex   = if ("sex" %in% colnames(meta)) meta$sex else NA_character_,
  Age   = if ("age" %in% colnames(meta)) meta$age else NA_real_,
  row.names = rownames(pca_result$x)
)

# ── 6c: PC1 vs PC2 coloured by disease group ─────────────────────────────
p_pc12 <- ggplot(pca_df, aes(x = PC1, y = PC2,
                               colour = Group, shape = Group)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(16, 17, 15)) +
  stat_ellipse(aes(group = Group, colour = Group),
               type = "norm", level = 0.80, linetype = "dashed",
               linewidth = 0.8) +
  labs(
    title  = "PCA: PC1 vs PC2 — Coloured by Disease Group",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour = "Group", shape = "Group",
    caption = "80% confidence ellipses per group (dashed)"
  ) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "right")

ggsave("results/pca_pc1_pc2_group.png", p_pc12, width = 8, height = 6, dpi = 150)
cat("PC1/PC2 plot saved to results/pca_pc1_pc2_group.png\n")

# ── 6d: Biplot — top 20 miRNA loadings overlaid on sample scores ─────────
loadings <- as.data.frame(pca_result$rotation[, 1:2])
loadings$mirna      <- rownames(loadings)
loadings$importance <- sqrt(loadings$PC1^2 + loadings$PC2^2)

top_loadings <- loadings[order(loadings$importance, decreasing = TRUE)[1:20], ]

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
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    caption = "Arrows: miRNA loading vectors (direction = contribution to each PC)"
  ) +
  theme_bw(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave("results/pca_biplot.png", p_biplot, width = 9, height = 7, dpi = 150)
cat("Biplot saved to results/pca_biplot.png\n")

# ── 6e: Loadings table for PC1 ───────────────────────────────────────────
#
# Positive loading on PC1: miRNA increases along PC1.
# If AD samples are at the positive end of PC1, this miRNA is UP in AD.
# If AD samples are at the negative end, this miRNA is DOWN in AD.

pc1_loadings <- data.frame(
  mirna   = rownames(pca_result$rotation),
  loading = pca_result$rotation[, 1],
  stringsAsFactors = FALSE
)
pc1_loadings <- pc1_loadings[order(abs(pc1_loadings$loading), decreasing = TRUE), ]

cat("\n=== Top 20 miRNA Loadings on PC1 ===\n")
cat("(Large |loading| = miRNA strongly drives PC1 variation)\n\n")
print(head(pc1_loadings, 20))

write.csv(pc1_loadings, "results/pca_pc1_loadings.csv", row.names = FALSE)
cat("\nFull PC1 loadings saved to results/pca_pc1_loadings.csv\n")

# ── 6f: PC3 vs PC4 and Sex confounder check ──────────────────────────────
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

p_pc12_sex <- ggplot(pca_df, aes(x = PC1, y = PC2,
                                   colour = Sex, shape = Group)) +
  geom_point(size = 3, alpha = 0.85) +
  labs(
    title   = "PCA: PC1 vs PC2 — Coloured by Sex",
    x       = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y       = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    caption = "If colour aligns with PC axis, sex may be confounding the data"
  ) +
  theme_bw(base_size = 12)

ggsave("results/pca_pc34_sex_check.png",
       arrangeGrob(p_pc34, p_pc12_sex, ncol = 2),
       width = 14, height = 6, dpi = 150)
cat("PC3/PC4 and sex check saved to results/pca_pc34_sex_check.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Hierarchical clustering
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: If a label-blind clustering algorithm re-discovers the clinical groups,
#      the miRNA signatures are coherent and strong enough for supervised ML.
#
# METHOD: Ward.D2 linkage minimises within-cluster variance at each merge step.
#         It produces the most compact, well-separated clusters for expression
#         data and is the recommended default.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 7: Hierarchical clustering (Ward.D2) ───────────────────────\n\n")

dist_matrix <- dist(t(expr_vf), method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# Save dendrogram as PNG
png("results/dendrogram_ward_k3.png", width = 1200, height = 600, res = 120)
par(mar = c(5, 4, 4, 2))
plot(hc,
     main   = "Hierarchical Clustering Dendrogram — Ward.D2",
     labels = meta$group,
     cex    = 0.5,
     xlab   = "Samples (labelled by clinical group)",
     ylab   = "Height (Ward.D2 dissimilarity)")
rect.hclust(hc, k = 3,
            border = c("#4575B4", "#FEE090", "#D73027"))
dev.off()
cat("Dendrogram saved to results/dendrogram_ward_k3.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Optimal cluster number: Gap statistic and Silhouette width
# ─────────────────────────────────────────────────────────────────────────────
#
# GAP STATISTIC: Compares observed within-cluster variation to that expected
#   from a null distribution of no clustering.  Optimal k = largest gap.
#   (Use B=500 for publication; B=50 is fine for the lab session.)
#
# SILHOUETTE: Measures how well each sample fits its cluster vs the nearest
#   other cluster.  Average silhouette maximises at optimal k.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 8: Gap statistic and silhouette width ──────────────────────\n\n")
cat("Running gap statistic (B=50 bootstrap resamples) — please wait...\n")

set.seed(42)
gap_stat <- clusGap(
  t(expr_vf),
  FUN   = hcut,
  K.max = 8,
  B     = 50,     # increase to 500 for publication-quality analysis
  nstart = 25
)

p_gap <- fviz_gap_stat(gap_stat) +
  labs(title = "Gap Statistic for Optimal Number of Clusters") +
  theme_bw(base_size = 12)
ggsave("results/gap_statistic.png", p_gap, width = 7, height = 5, dpi = 150)
cat("Gap statistic: optimal k =", which.max(gap_stat$Tab[, "gap"]), "\n")

# Silhouette width for k = 2 to 8
sil_widths <- numeric(7)
for (k in 2:8) {
  clusters      <- cutree(hc, k = k)
  sil           <- silhouette(clusters, dist_matrix)
  sil_widths[k - 1] <- mean(sil[, "sil_width"])
}

sil_df <- data.frame(k = 2:8, sil_width = sil_widths)
optimal_k_sil <- sil_df$k[which.max(sil_df$sil_width)]

p_sil <- ggplot(sil_df, aes(x = k, y = sil_width)) +
  geom_line(colour = "#D73027", linewidth = 1.2) +
  geom_point(size = 3.5, colour = "#D73027") +
  geom_vline(xintercept = optimal_k_sil,
             linetype = "dashed", colour = "grey40") +
  labs(
    title = "Silhouette Width vs Number of Clusters",
    x     = "Number of Clusters (k)",
    y     = "Average Silhouette Width",
    caption = paste("Dashed line: optimal k =", optimal_k_sil, "by silhouette")
  ) +
  theme_bw(base_size = 12)
ggsave("results/silhouette_width.png", p_sil, width = 7, height = 5, dpi = 150)

cat("Silhouette: optimal k =", optimal_k_sil, "\n")
cat("Gap statistic plot saved to results/gap_statistic.png\n")
cat("Silhouette plot saved to results/silhouette_width.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Cluster purity analysis
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Cluster purity measures how well expression-based clusters recover
#      known clinical labels.  No label information is used during clustering.
#
# INTERPRETATION:
#   Purity = 1.0: perfect recovery of clinical labels
#   Purity > 0.70: acceptable for unsupervised biomarker discovery
#   Purity ≈ 0.50: no better than random — miRNA profiles do not systematically
#                  differ between clinical groups
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 9: Cluster purity analysis ─────────────────────────────────\n\n")

# We use k=3 to match the three clinical groups
k_optimal <- 3
cluster_labels <- cutree(hc, k = k_optimal)

purity_table <- table(
  Cluster        = factor(cluster_labels, labels = paste0("Cluster_", 1:k_optimal)),
  Clinical_Group = meta$group
)

cat("=== Cluster Purity Table (k =", k_optimal, ") ===\n")
cat("Rows = expression-based clusters; Columns = known clinical labels\n\n")
print(purity_table)

purity_per_cluster <- apply(purity_table, 1, function(row) max(row) / sum(row))
overall_purity     <- sum(apply(purity_table, 1, max)) / sum(purity_table)

cat("\nPurity per cluster:\n")
print(round(purity_per_cluster, 3))
cat("\nOverall cluster purity:", round(overall_purity, 3), "\n")
cat("(>0.70 = acceptable; 0.50 = no better than random)\n")

write.csv(as.data.frame.matrix(purity_table),
          "results/cluster_purity_table.csv")
cat("Cluster purity table saved to results/cluster_purity_table.csv\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — k-means clustering
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: k-means is a complementary clustering method.  If both hierarchical and
#      k-means recover similar groupings, the cluster structure is robust.
#
# nstart=50: run 50 times with random starting centroids; keep the best result.
#            This avoids the local optima problem of k-means.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 10: k-means clustering (k=3) ────────────────────────────────\n\n")

set.seed(42)
km <- kmeans(t(expr_vf), centers = 3, nstart = 50, iter.max = 200)

km_purity <- table(
  kmeans_cluster = factor(km$cluster, labels = paste0("kCluster_", 1:3)),
  Clinical_Group  = meta$group
)
cat("=== k-means Cluster Purity Table ===\n")
print(km_purity)

km_overall_purity <- sum(apply(km_purity, 1, max)) / sum(km_purity)
cat("\nk-means overall purity:", round(km_overall_purity, 3), "\n")
cat("Hierarchical purity:   ", round(overall_purity, 3), "\n")

# Overlay k-means clusters on PCA plot
pca_df$kmeans_cluster <- factor(km$cluster, labels = paste0("kC", 1:3))

p_km_pca <- ggplot(pca_df, aes(x = PC1, y = PC2,
                                 colour = Group, shape = kmeans_cluster)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  labs(
    title  = "k-means Clustering (k=3) Overlaid on PCA",
    x      = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y      = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour = "Clinical Group", shape = "k-means Cluster",
    caption = "Shape = k-means cluster; Colour = known clinical label"
  ) +
  theme_bw(base_size = 12)
ggsave("results/kmeans_pca_overlay.png", p_km_pca, width = 8, height = 6, dpi = 150)
cat("k-means PCA overlay saved to results/kmeans_pca_overlay.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Publication-quality heatmap
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: The heatmap is often the single most informative visualisation in an
#      miRNA biomarker paper.  Reading a heatmap requires three layers:
#        (1) Cell colour: relative expression of this miRNA in this sample
#        (2) Row clusters: co-regulated miRNA modules
#        (3) Column clusters: samples with similar overall miRNA profiles
#
# Annotation tracks (colour bars at top) map clinical metadata onto the
# column clustering — allowing visual comparison of cluster alignment with
# disease group, sex, and age.
#
# ROW SCALING (z-score): ensures colour reflects relative not absolute
# expression, so all miRNAs contribute equally to the visual.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 11: Publication-quality heatmap (top 50 miRNAs) ────────────\n\n")

# Top 50 most variable miRNAs by IQR
top50_mirnas <- names(sort(apply(expr_vf, 1, IQR), decreasing = TRUE))[1:50]
expr_top50   <- expr_vf[top50_mirnas, ]

# Annotation data frame: one row per sample, matching column order
annotation_col <- data.frame(
  Group = factor(meta$group),
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

ann_colours <- list(
  Group = GROUP_COLOURS,
  Sex   = c("Male" = "#2166AC", "Female" = "#B2182B",
            "M"    = "#2166AC", "F"      = "#B2182B"),
  Age_Group = c("60-69" = "#EFF3FF", "70-74" = "#BDD7E7",
                "75-79" = "#6BAED6", "80+"   = "#2171B5")
)
# Keep only annotation tracks that actually exist in the data
ann_colours <- ann_colours[intersect(names(ann_colours), colnames(annotation_col))]

# Generate heatmap (saved directly to file by pheatmap)
pheatmap(
  expr_top50,
  scale                    = "row",
  color                    = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100),
  breaks                   = seq(-3, 3, length.out = 101),
  clustering_method        = "ward.D2",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  annotation_col           = annotation_col,
  annotation_colors        = ann_colours,
  show_rownames            = TRUE,
  show_colnames            = FALSE,
  fontsize_row             = 7,
  main                     = "Top 50 Most Variable miRNAs — GSE120584\n(row z-score, Ward.D2 clustering)",
  filename                 = "results/heatmap_top50_miRNAs.png",
  width                    = 12,
  height                   = 14
)
cat("Heatmap saved to results/heatmap_top50_miRNAs.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Confounder analysis: PC correlation and partial R²
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: A confounder is associated with both disease group AND miRNA expression.
#   Age and sex independently affect circulating miRNA levels.
#   If age explains more of PC1 than disease group, miRNA differences
#   attributed to AD may actually reflect age differences.
#
# PARTIAL R²: Partitions variance in each PC between disease, age, and sex,
#   holding all other variables constant.  Used to inform Week 4 model design
#   (e.g., include age as covariate in DESeq2 design formula).
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 12: Confounder analysis (PC correlations + partial R²) ─────\n\n")

pc_scores <- pca_result$x[, 1:10]

# Initialise result data frames
age_cor_results <- data.frame(PC = paste0("PC", 1:10),
                               r_age = NA_real_, p_age = NA_real_,
                               stringsAsFactors = FALSE)
sex_cor_results <- data.frame(PC = paste0("PC", 1:10),
                               r_sex = NA_real_, p_sex = NA_real_,
                               stringsAsFactors = FALSE)

# Age: Pearson correlation (continuous variable)
if ("age" %in% colnames(meta) && !all(is.na(meta$age))) {
  for (i in 1:10) {
    ct <- cor.test(pc_scores[, i], meta$age, method = "pearson")
    age_cor_results$r_age[i] <- ct$estimate
    age_cor_results$p_age[i] <- ct$p.value
  }
  cat("Age correlations computed.\n")
} else {
  cat("Age not available in metadata — skipping age correlation.\n")
}

# Sex: point-biserial correlation (binary variable encoded as 0/1)
if ("sex" %in% colnames(meta) && !all(is.na(meta$sex))) {
  sex_binary <- as.numeric(factor(meta$sex)) - 1
  for (i in 1:10) {
    ct <- cor.test(pc_scores[, i], sex_binary, method = "pearson")
    sex_cor_results$r_sex[i] <- ct$estimate
    sex_cor_results$p_sex[i] <- ct$p.value
  }
  cat("Sex correlations computed.\n")
} else {
  cat("Sex not available in metadata — skipping sex correlation.\n")
}

confounder_table <- data.frame(
  PC          = paste0("PC", 1:10),
  Var_Exp_pct = round(var_explained[1:10], 2),
  r_age       = age_cor_results$r_age,
  p_age       = age_cor_results$p_age,
  r_sex       = sex_cor_results$r_sex,
  p_sex       = sex_cor_results$p_sex
)

cat("\n=== Confounder Correlation with Principal Components ===\n\n")
print(confounder_table)
write.csv(confounder_table, "results/pc_confounder_correlations.csv", row.names = FALSE)
cat("\nConfounder table saved to results/pc_confounder_correlations.csv\n")

# ── Partial R² via Type II ANOVA ─────────────────────────────────────────
if (all(c("age", "sex") %in% colnames(meta)) &&
    !all(is.na(meta$age)) && !all(is.na(meta$sex))) {

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

  anova_pc1 <- car::Anova(fit_pc1, type = "II")
  anova_pc2 <- car::Anova(fit_pc2, type = "II")

  ss_total_pc1   <- sum(anova_pc1$"Sum Sq")
  partial_r2_pc1 <- anova_pc1$"Sum Sq" / ss_total_pc1

  ss_total_pc2   <- sum(anova_pc2$"Sum Sq")
  partial_r2_pc2 <- anova_pc2$"Sum Sq" / ss_total_pc2

  cat("\n=== Partial R-squared for PC1 ===\n")
  cat("(How much of PC1 variance is explained by each variable?)\n\n")
  cat(sprintf("  Disease group: %.3f (%.1f%%)\n",
              partial_r2_pc1[1], partial_r2_pc1[1] * 100))
  cat(sprintf("  Age:           %.3f (%.1f%%)\n",
              partial_r2_pc1[2], partial_r2_pc1[2] * 100))
  cat(sprintf("  Sex:           %.3f (%.1f%%)\n",
              partial_r2_pc1[3], partial_r2_pc1[3] * 100))

  cat("\n=== Partial R-squared for PC2 ===\n\n")
  cat(sprintf("  Disease group: %.3f (%.1f%%)\n",
              partial_r2_pc2[1], partial_r2_pc2[1] * 100))
  cat(sprintf("  Age:           %.3f (%.1f%%)\n",
              partial_r2_pc2[2], partial_r2_pc2[2] * 100))
  cat(sprintf("  Sex:           %.3f (%.1f%%)\n\n",
              partial_r2_pc2[3], partial_r2_pc2[3] * 100))

  cat("DECISION RULE:\n")
  cat("  If age partial R² on PC1 > disease partial R², include age as a covariate\n")
  cat("  in the DESeq2 design formula (Week 4): design = ~ sex + age + group\n")
} else {
  cat("\nAge and/or sex not fully available — skipping partial R² analysis.\n")
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — Outlier detection via Mahalanobis distance
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Some samples may pass Week 2 QC metrics but still sit far from all
#      other samples in their group in PCA space — indicative of technical
#      or biological anomalies not captured by library-size-based QC.
#
# Mahalanobis distance accounts for the correlation between PC1 and PC2.
# Threshold: chi-squared distribution at 97.5% with df=2.
#
# IMPORTANT: An extreme AD sample with a strongly elevated AD miRNA signature
# is NOT an error — it may be a severe case.  Investigate metadata before
# any exclusion decision.
# ─────────────────────────────────────────────────────────────────────────────

cat("\n─── Section 13: Mahalanobis outlier detection ───────────────────────────\n\n")

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

n_outliers <- sum(pca_df$is_outlier)
cat("Potential outlier samples detected:", n_outliers, "\n")
if (n_outliers > 0) {
  cat("Outlier sample IDs:\n")
  print(rownames(pca_df)[pca_df$is_outlier])
}

p_outliers <- ggplot(pca_df, aes(x = PC1, y = PC2, colour = Group,
                                   shape = is_outlier)) +
  geom_point(aes(size = is_outlier), alpha = 0.85) +
  scale_colour_manual(values = GROUP_COLOURS) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 4)) +
  scale_size_manual(values  = c(`FALSE` = 2.5, `TRUE` = 5)) +
  geom_text_repel(data    = pca_df[pca_df$is_outlier, , drop = FALSE],
                  aes(label = rownames(pca_df)[pca_df$is_outlier]),
                  size = 3, colour = "black", inherit.aes = FALSE,
                  mapping = aes(x = PC1, y = PC2)) +
  labs(
    title   = "PCA — Mahalanobis Outlier Detection (97.5% threshold)",
    x       = paste0("PC1 (", round(var_explained[1], 1), "% variance)"),
    y       = paste0("PC2 (", round(var_explained[2], 1), "% variance)"),
    colour  = "Clinical Group",
    caption = "X markers: samples with Mahalanobis distance > chi² 97.5th percentile within group"
  ) +
  theme_bw(base_size = 12)
ggsave("results/pca_outlier_detection.png", p_outliers, width = 8, height = 6, dpi = 150)
cat("Outlier PCA plot saved to results/pca_outlier_detection.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Session summary and data export
# ─────────────────────────────────────────────────────────────────────────────

cat("\n====================================================\n")
cat("  Week 3 EDA — Session Summary\n")
cat("====================================================\n\n")

cat("Input data:\n")
cat("  Expression matrix:", nrow(expr_vf), "miRNAs ×", ncol(expr_vf), "samples\n")
cat("  Groups:", paste(names(table(meta$group)),
                       as.integer(table(meta$group)), sep = "=", collapse = ", "), "\n\n")

cat("Key EDA results:\n")
cat(sprintf("  PC1 variance explained:        %.1f%%\n", var_explained[1]))
cat(sprintf("  PC2 variance explained:        %.1f%%\n", var_explained[2]))
cat(sprintf("  Optimal k (gap statistic):     %d\n",    which.max(gap_stat$Tab[, "gap"])))
cat(sprintf("  Optimal k (silhouette):        %d\n",    optimal_k_sil))
cat(sprintf("  Hierarchical cluster purity:   %.3f\n",  overall_purity))
cat(sprintf("  k-means cluster purity:        %.3f\n",  km_overall_purity))
cat(sprintf("  Potential outlier samples:     %d\n\n",  n_outliers))

cat("Files exported to results/:\n")
cat("  pca_scree_plot.png\n  pca_pc1_pc2_group.png\n  pca_biplot.png\n")
cat("  pca_pc34_sex_check.png\n  pca_outlier_detection.png\n")
cat("  pca_pc1_loadings.csv\n  mirna_descriptive_stats.csv\n")
cat("  zero_inflation_histogram.png\n  density_plot_representative.png\n")
cat("  dendrogram_ward_k3.png\n  gap_statistic.png\n  silhouette_width.png\n")
cat("  heatmap_top50_miRNAs.png\n  kmeans_pca_overlay.png\n")
cat("  cluster_purity_table.csv\n  pc_confounder_correlations.csv\n\n")

cat("Files exported for Python (Lab 3B — t-SNE / UMAP):\n")
cat("  data/processed/GSE120584_expr_vf.csv\n")
cat("  data/processed/GSE120584_metadata_clean.csv\n")
cat("  data/processed/GSE120584_expr_varianceFiltered.rds\n\n")

cat("─────────────────────────────────────────────────────\n")
cat("PROCEED TO:\n")
cat("  Lab 3B — Open Week3_Lab3B_tSNE_UMAP.ipynb in JupyterLab\n")
cat("  Week 4  — Open Week4_DE_FeatureSelection.R in RStudio\n")
cat("─────────────────────────────────────────────────────\n\n")

sessionInfo()
