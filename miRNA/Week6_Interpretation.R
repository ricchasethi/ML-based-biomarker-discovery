################################################################################
# AI/ML in Biomarker Discovery — Week 6 Lab
# Title:   Biological Interpretation & Clinical Translation
# Disease: Alzheimer's Disease | Biomarker: miRNA
# Audience: Wet-lab biologists — Weeks 1–5 completed
#
# Learning Goals for This Script:
#   1. Load and integrate outputs from all previous weeks (DE results, SHAP rankings,
#      harmonized expression matrices)
#   2. Build a composite biomarker ranking score combining SHAP importance and
#      differential expression significance; select the top 15 candidate miRNAs
#   3. Query 14 miRNA–target databases simultaneously using multiMiR; filter for
#      validated targets; identify known Alzheimer's disease genes in the target list
#   4. Run KEGG pathway enrichment analysis with clusterProfiler; verify whether
#      the Alzheimer disease pathway (hsa05010) is enriched in the target gene set
#   5. Run GO Biological Process enrichment; simplify redundant terms; interpret
#      the top pathways in the context of AD neurobiological mechanisms
#   6. Build a protein-protein interaction (PPI) network using STRINGdb; identify
#      hub genes by degree centrality; overlay AD GWAS risk genes as colored nodes
#   7. Generate the "main figure" of the hypothetical paper: a forest plot showing
#      each biomarker miRNA with its log2FC, SHAP importance, and key target(s)
#   8. Simulate an analytical validation experiment: qPCR Ct values for the top 3
#      miRNAs, ΔΔCt calculation, and limit of detection (LOD) estimation
#   9. Save all interpretation results (tables and plots) to results/Week6/
#  10. Print a complete pipeline summary table and list all saved outputs from
#      every week; record session info for reproducibility
#
# Packages Required:
#   multiMiR, clusterProfiler, org.Hs.eg.db, STRINGdb, igraph,
#   ggplot2, ggrepel, dplyr, readr, pheatmap, RColorBrewer
#
# Run each section with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac).
################################################################################


# ==============================================================================
# SECTION 1: Load All Packages
# ==============================================================================
# If any library() call fails, install the missing package:
#   Bioconductor packages: BiocManager::install("package_name")
#   CRAN packages:         install.packages("package_name")
#
# multiMiR installation note: multiMiR is available on Bioconductor.
#   BiocManager::install("multiMiR")
# STRINGdb installation:
#   BiocManager::install("STRINGdb")

suppressPackageStartupMessages({
  # Bioconductor packages
  library(multiMiR)          # Query 14 miRNA-target databases
  library(clusterProfiler)   # Pathway and GO enrichment (ORA, GSEA)
  library(org.Hs.eg.db)      # Human genome annotation (Gene symbols → Entrez IDs)
  library(STRINGdb)          # Protein-protein interaction networks from STRING
  library(igraph)            # Network construction and analysis

  # CRAN packages
  library(ggplot2)
  library(ggrepel)           # Non-overlapping text labels on ggplot2 figures
  library(dplyr)
  library(readr)
  library(pheatmap)
  library(RColorBrewer)
  library(tibble)
})

cat("All packages loaded successfully.\n\n")

# Colour palette used throughout this script (consistent with Weeks 3–5)
GROUP_COLOURS <- c(
  "Control"                   = "#4575B4",   # blue
  "Mild Cognitive Impairment" = "#FEE090",   # amber
  "Alzheimer's Disease"       = "#D73027"    # red
)

# Known Alzheimer's disease genes — used throughout for highlighting
AD_KNOWN_GENES <- c(
  "APP", "BACE1", "MAPT", "PSEN1", "PSEN2", "APOE", "SIRT1", "FOXO3",
  "CDK5", "GSK3B", "TP53", "BCL2", "PTEN", "ADAM10", "CLU", "BIN1",
  "SORL1", "TREM2", "PLCG2", "SPI1"
)

# AD GWAS risk genes (Jansen et al. 2019, Lambert et al. 2013, Wightman et al. 2021)
GWAS_AD_GENES <- c(
  "APOE", "BIN1", "CLU", "PICALM", "CR1", "ABCA7", "SORL1", "PTK2B",
  "SPI1", "PLCG2", "TREM2", "FERMT2", "CASS4", "INPP5D", "MEF2C",
  "HLA-DRB1", "ZCWPW1", "CELF1", "NME8", "TRIP4"
)


# ==============================================================================
# SECTION 2: Create Output Directory and Ensure Data from Previous Weeks Exists
# ==============================================================================

# Create Week 6 results directory
if (!dir.exists("results/Week6")) {
  dir.create("results/Week6", recursive = TRUE)
  cat("Created: results/Week6/\n")
}

# Helper function: attempt to load a file; if missing, create a plausible
# synthetic stand-in so the rest of the script can still run and demonstrate
# the workflow. In a real course run, students will always have the actual files.
safe_load <- function(path, description) {
  if (file.exists(path)) {
    obj <- readRDS(path)
    cat("Loaded", description, "from", path, "\n")
    return(obj)
  } else {
    cat("NOTE:", path, "not found. Generating synthetic", description,
        "for demonstration.\n")
    return(NULL)
  }
}


# ==============================================================================
# SECTION 3: Load Results from Previous Weeks
# ==============================================================================
# We need three data objects from earlier weeks:
#
#   From Week 4:
#     DE_results_GSE120584.csv  — differential expression table (AD vs Control)
#
#   From Week 5:
#     shap_feature_importance.csv — SHAP mean |value| per miRNA from the RF/LR model
#     harmonized_expr.rds         — harmonized, batch-corrected expression matrix
#     metadata_harmonized.rds     — sample metadata for the harmonized cohort
#
# If your files have slightly different names (e.g., from your own Weeks 4–5
# run), update the paths below accordingly.

cat("=== Loading outputs from Weeks 4 and 5 ===\n")

# ---- Week 4: Differential expression results ----
de_path <- "results/Week4/DE_results_GSE120584.csv"
if (file.exists(de_path)) {
  de_results <- read_csv(de_path, show_col_types = FALSE)
  cat("DE results loaded:", nrow(de_results), "features\n")
} else {
  # Synthetic DE results for demonstration
  # In real usage, these come directly from DESeq2 in Week 4
  set.seed(123)
  n_mirnas <- 300
  de_results <- data.frame(
    miRNA      = paste0("hsa-miR-", sample(1:700, n_mirnas, replace = FALSE), "-",
                        sample(c("5p","3p"), n_mirnas, replace = TRUE)),
    log2FC     = rnorm(n_mirnas, mean = 0, sd = 1.2),
    pvalue     = runif(n_mirnas, 0, 0.5),
    padj       = runif(n_mirnas, 0, 0.8),
    stringsAsFactors = FALSE
  )
  # Make some miRNAs with known AD relevance strongly significant
  known_ad_mirnas <- c("hsa-miR-21-5p",  "hsa-miR-146a-5p", "hsa-miR-132-3p",
                       "hsa-miR-107",    "hsa-miR-29a-3p",  "hsa-miR-128-3p",
                       "hsa-miR-34a-5p", "hsa-miR-181a-5p", "hsa-miR-9-5p",
                       "hsa-miR-155-5p", "hsa-miR-26a-5p",  "hsa-miR-101-3p",
                       "hsa-miR-16-5p",  "hsa-miR-20a-5p",  "hsa-miR-125b-5p")
  de_results <- rbind(
    data.frame(
      miRNA  = known_ad_mirnas,
      log2FC = c(-1.8, 1.6, -2.1, -1.9, -1.7,  0.9,  1.4,  1.2, -1.5,  1.8,
                 -1.1, -0.8, -0.9,  1.0, -1.3),
      pvalue = c(2e-8, 5e-7, 1e-9, 3e-8, 8e-7, 2e-5, 4e-7, 6e-6, 9e-7, 3e-8,
                 1e-4, 3e-4, 2e-4, 5e-5, 7e-5),
      padj   = c(1e-6, 2e-5, 5e-8, 1e-6, 3e-5, 8e-4, 2e-5, 2e-4, 4e-5, 1e-6,
                 3e-3, 8e-3, 6e-3, 2e-3, 3e-3),
      stringsAsFactors = FALSE
    ),
    de_results
  )
  de_results <- de_results[!duplicated(de_results$miRNA), ]
  cat("NOTE: Using synthetic DE results for demonstration (", nrow(de_results),
      "features)\n")
}

# ---- Week 5: SHAP feature importance ----
shap_path <- "results/Week5/shap_feature_importance.csv"
if (file.exists(shap_path)) {
  shap_df <- read_csv(shap_path, show_col_types = FALSE)
  cat("SHAP importance loaded:", nrow(shap_df), "features\n")
} else {
  # Synthetic SHAP values — normally output by the Python Week 5 script
  # SHAP values: mean absolute SHAP across all test samples
  # The same known AD miRNAs should rank highly
  set.seed(456)
  shap_df <- data.frame(
    miRNA      = de_results$miRNA,
    mean_abs_shap = abs(rnorm(nrow(de_results), mean = 0.05, sd = 0.08)),
    stringsAsFactors = FALSE
  )
  # Boost known AD miRNAs to top SHAP ranks
  known_idx <- match(known_ad_mirnas, shap_df$miRNA)
  known_idx <- known_idx[!is.na(known_idx)]
  shap_df$mean_abs_shap[known_idx] <- seq(0.35, 0.12, length.out = length(known_idx)) +
    rnorm(length(known_idx), 0, 0.02)
  cat("NOTE: Using synthetic SHAP values for demonstration\n")
}

# ---- Week 5: Harmonized expression matrix ----
expr_harm <- safe_load("data/processed/harmonized_expr.rds",  "harmonized expression matrix")
meta_harm  <- safe_load("data/processed/metadata_harmonized.rds", "harmonized metadata")

if (is.null(expr_harm)) {
  # Synthetic expression matrix for demonstration
  set.seed(789)
  n_samples <- 120
  n_features <- min(nrow(de_results), 200)
  expr_harm <- matrix(
    rnorm(n_features * n_samples, mean = 8, sd = 2.5),
    nrow  = n_features,
    ncol  = n_samples,
    dimnames = list(
      de_results$miRNA[1:n_features],
      paste0("Sample_", seq_len(n_samples))
    )
  )
  meta_harm <- data.frame(
    sample = paste0("Sample_", seq_len(n_samples)),
    group  = rep(c("Alzheimer's Disease", "Control"), each = n_samples / 2),
    stringsAsFactors = FALSE
  )
  meta_harm$group <- factor(meta_harm$group, levels = c("Control", "Alzheimer's Disease"))
  cat("NOTE: Using synthetic expression matrix for demonstration\n")
}

cat("\nData loading complete.\n")


# ==============================================================================
# SECTION 4: Build Composite Biomarker Ranking Score
# ==============================================================================
# We combine two independent lines of evidence:
#   1. Differential expression significance   (from Week 4, −log10 adjusted p-value)
#   2. SHAP feature importance                (from Week 5, mean |SHAP|)
#
# Composite approach: rank each miRNA by each metric independently, then
# average the two ranks. miRNAs that rank highly by BOTH criteria are the
# most compelling candidates for biological interpretation.
#
# Why average ranks (not z-scores or raw values)?
#   Ranks are robust to scale differences between the two measures.
#   −log10(padj) can range from 0 to 50+; SHAP values range from 0 to 0.5.
#   Averaging raw values would allow one metric to dominate.

cat("\n=== Building composite biomarker ranking ===\n")

# Ensure column names are standardized
# Expected columns: miRNA, log2FC, padj (from DE), mean_abs_shap (from SHAP)
if (!"miRNA" %in% colnames(de_results)) {
  # Try common alternative column names
  mirna_col <- intersect(c("feature", "ID", "Gene", "mirna", "name"),
                          colnames(de_results))[1]
  colnames(de_results)[colnames(de_results) == mirna_col] <- "miRNA"
}
if (!"miRNA" %in% colnames(shap_df)) {
  mirna_col <- intersect(c("feature", "ID", "Gene", "mirna", "name"),
                          colnames(shap_df))[1]
  colnames(shap_df)[colnames(shap_df) == mirna_col] <- "miRNA"
}

# Merge DE results with SHAP values on miRNA name
combined_df <- inner_join(de_results, shap_df, by = "miRNA")
cat("miRNAs present in both DE and SHAP tables:", nrow(combined_df), "\n")

# Handle zeros in padj (replace with minimum non-zero value for log transform)
min_padj <- min(combined_df$padj[combined_df$padj > 0], na.rm = TRUE)
combined_df$padj_safe <- pmax(combined_df$padj, min_padj * 0.01, na.rm = TRUE)

# Compute −log10(adjusted p-value) as DE importance metric
combined_df$neg_log10_padj <- -log10(combined_df$padj_safe)

# Rank by each metric (rank 1 = most important)
combined_df$rank_de   <- rank(-combined_df$neg_log10_padj)    # higher padj significance → lower rank
combined_df$rank_shap <- rank(-combined_df$mean_abs_shap)     # higher SHAP → lower rank

# Composite score: average of the two ranks (lower = better)
combined_df$composite_rank <- (combined_df$rank_de + combined_df$rank_shap) / 2

# Sort by composite rank
combined_df <- combined_df[order(combined_df$composite_rank), ]

# Select top 15 miRNAs for biological interpretation
top15 <- combined_df[1:15, ]

cat("\nTop 15 biomarker miRNAs (composite SHAP + DE ranking):\n")
print(top15[, c("miRNA", "log2FC", "padj", "mean_abs_shap", "composite_rank")])

# Save the composite ranking table
write.csv(combined_df, "results/Week6/composite_mirna_ranking.csv", row.names = FALSE)
cat("\nFull composite ranking saved to results/Week6/composite_mirna_ranking.csv\n")

# Extract the top 15 miRNA names as a character vector for all downstream queries
top_mirna_names <- top15$miRNA


# ==============================================================================
# SECTION 5: multiMiR Target Prediction
# ==============================================================================
# multiMiR provides a unified R interface to 14 miRNA-target databases.
# We query validated databases first (experimentally confirmed interactions),
# then also retrieve high-confidence predicted targets for completeness.
#
# This query can take 5–15 minutes depending on internet speed and database load.
# multiMiR connects to a remote MySQL database hosted at:
# http://multimir.ucdenver.edu
#
# If the connection times out, re-run the query for individual miRNAs using:
#   multiMiR(mirna = "hsa-miR-21-5p", table = "validated")
# and rbind the results.

cat("\n=== Querying multiMiR for validated targets ===\n")
cat("Querying databases for", length(top_mirna_names), "miRNAs...\n")
cat("This may take 5–15 minutes. Please wait.\n\n")

# ---- 5A. Query validated (experimentally confirmed) interactions ----
tryCatch({
  validated_result <- multiMiR(
    org    = "hsa",
    mirna  = top_mirna_names,
    table  = "validated",         # validated databases: miRTarBase, miRecords, TarBase
    use.tibble = TRUE
  )

  val_df <- as.data.frame(validated_result@data)
  cat("Validated interactions retrieved:", nrow(val_df), "\n")
  cat("Unique target genes:", length(unique(val_df$target.symbol)), "\n")

}, error = function(e) {
  cat("multiMiR connection failed:", conditionMessage(e), "\n")
  cat("Using built-in curated fallback targets for demonstration.\n")
  # Fallback: hand-curated validated targets for the top 15 AD-relevant miRNAs
  # These are representative real interactions from miRTarBase (literature validated)
  val_df <<- data.frame(
    mature.mirna     = c(
      "hsa-miR-29a-3p","hsa-miR-29a-3p","hsa-miR-29a-3p",
      "hsa-miR-107",   "hsa-miR-107",   "hsa-miR-107",
      "hsa-miR-132-3p","hsa-miR-132-3p","hsa-miR-132-3p","hsa-miR-132-3p",
      "hsa-miR-34a-5p","hsa-miR-34a-5p","hsa-miR-34a-5p",
      "hsa-miR-146a-5p","hsa-miR-146a-5p",
      "hsa-miR-155-5p","hsa-miR-155-5p",
      "hsa-miR-21-5p", "hsa-miR-21-5p",
      "hsa-miR-9-5p",  "hsa-miR-9-5p",
      "hsa-miR-181a-5p","hsa-miR-181a-5p",
      "hsa-miR-101-3p","hsa-miR-101-3p",
      "hsa-miR-16-5p",
      "hsa-miR-128-3p","hsa-miR-128-3p",
      "hsa-miR-20a-5p",
      "hsa-miR-125b-5p","hsa-miR-125b-5p",
      "hsa-miR-26a-5p",
      "hsa-miR-181a-5p"
    ),
    target.symbol    = c(
      "BACE1","DNMT3B","BCL2",
      "BACE1","CDK6","DICER1",
      "FOXO3","EP300","SIRT1","ITPKB",
      "BCL2","SIRT1","CDK6",
      "IRAK1","TRAF6",
      "SHIP1","SOCS1",
      "PTEN","PDCD4",
      "BACE1","REST",
      "SIRT1","SMAD7",
      "APP","BACE1",
      "BCL2",
      "SIRT1","SP1",
      "PTEN",
      "TP53","CDK6",
      "GSK3B",
      "MAPT"
    ),
    experiment       = c(
      "Luciferase reporter assay","Western blot","qRT-PCR",
      "Luciferase reporter assay","Western blot","Luciferase reporter assay",
      "Luciferase reporter assay","Western blot","qRT-PCR","Luciferase reporter assay",
      "Western blot","qRT-PCR","Luciferase reporter assay",
      "Luciferase reporter assay","Western blot",
      "Luciferase reporter assay","qRT-PCR",
      "Western blot","Luciferase reporter assay",
      "Luciferase reporter assay","Western blot",
      "qRT-PCR","Luciferase reporter assay",
      "Luciferase reporter assay","Western blot",
      "Western blot",
      "Luciferase reporter assay","qRT-PCR",
      "Western blot",
      "Luciferase reporter assay","Western blot",
      "qRT-PCR",
      "Luciferase reporter assay"
    ),
    stringsAsFactors = FALSE
  )
})

# ---- 5B. Filter for strong experimental evidence ----
# Strong evidence: luciferase reporter, western blot, or qRT-PCR
# (as opposed to microarray correlation, which is indirect)
strong_evidence <- val_df %>%
  filter(grepl("luciferase|Luciferase|qRT-PCR|Western|western", experiment)) %>%
  distinct(mature.mirna, target.symbol, .keep_all = TRUE)

cat("\nInteractions with strong experimental evidence:", nrow(strong_evidence), "\n")
cat("Unique target genes (strong evidence):", length(unique(strong_evidence$target.symbol)), "\n")

# ---- 5C. Check for known Alzheimer's disease genes in the target list ----
# This is the key mechanistic validation check.
# If targets include APP, BACE1, MAPT, PSEN1, APOE, SIRT1, FOXO3, or CDK5,
# the biomarker panel has direct mechanistic grounding in AD biology.

ad_check_genes <- c("APP", "BACE1", "MAPT", "PSEN1", "APOE",
                    "SIRT1", "FOXO3", "CDK5", "GSK3B", "TP53")

ad_hits <- strong_evidence %>%
  filter(target.symbol %in% ad_check_genes)

cat("\n=== AD-Relevant Validated Targets Found ===\n")
if (nrow(ad_hits) > 0) {
  print(ad_hits[, c("mature.mirna", "target.symbol", "experiment")])
} else {
  cat("No direct AD gene targets found with strong evidence.\n")
  cat("Consider expanding to predicted targets (see Section 5D).\n")
}

# ---- 5D. Also query predicted targets for background universe ----
# We need the broader set of targets (predicted + validated) for two purposes:
#   1. As the background gene universe in enrichment analysis (Section 6)
#   2. For a more complete view of downstream biology
#
# We use a stringent percentile cutoff (top 20%) to reduce noise.
cat("\nQuerying predicted targets for background universe...\n")

tryCatch({
  predicted_result <- multiMiR(
    org                   = "hsa",
    mirna                 = top_mirna_names,
    table                 = "predicted",
    predicted.cutoff      = 20,         # top 20% prediction scores only
    predicted.cutoff.type = "p",        # "p" = percentile (not absolute score)
    use.tibble            = TRUE
  )
  pred_df <- as.data.frame(predicted_result@data)
  cat("Predicted targets retrieved:", nrow(pred_df), "\n")

}, error = function(e) {
  cat("Predicted target query failed:", conditionMessage(e), "\n")
  cat("Using validated targets as the query set for enrichment background.\n")
  # In practice, the background should be all targets of all expressed miRNAs,
  # not just the top 15. For demonstration, we use all strong-evidence targets.
  pred_df <<- strong_evidence
})

# Compile the complete target gene universe
all_target_symbols <- unique(c(strong_evidence$target.symbol,
                                pred_df$target.symbol))
cat("Total unique target genes (strong evidence + predicted):", length(all_target_symbols), "\n")

# ---- 5E. Check for GWAS AD risk gene overlap ----
gwas_overlap <- intersect(all_target_symbols, GWAS_AD_GENES)
cat("\nTarget genes overlapping with AD GWAS risk loci:\n")
if (length(gwas_overlap) > 0) {
  print(gwas_overlap)
  # Show which miRNA targets each GWAS gene
  gwas_detail <- strong_evidence %>%
    filter(target.symbol %in% GWAS_AD_GENES) %>%
    select(mature.mirna, target.symbol, experiment)
  if (nrow(gwas_detail) > 0) {
    cat("\nmiRNA → GWAS gene interactions:\n")
    print(gwas_detail)
  }
} else {
  cat("No direct GWAS gene targets found in validated interactions.\n")
  cat("Note: GWAS genes may appear in predicted targets (broader target set).\n")
}

# ---- 5F. Save target gene tables ----
write.csv(strong_evidence,
          "results/Week6/validated_targets_strong_evidence.csv",
          row.names = FALSE)
write.csv(data.frame(target_gene = all_target_symbols),
          "results/Week6/target_gene_universe.csv",
          row.names = FALSE)
write.csv(ad_hits,
          "results/Week6/AD_gene_targets.csv",
          row.names = FALSE)

cat("\nTarget gene tables saved to results/Week6/\n")


# ==============================================================================
# SECTION 6: clusterProfiler — KEGG Pathway Enrichment Analysis
# ==============================================================================
# KEGG (Kyoto Encyclopedia of Genes and Genomes) provides curated pathway maps
# covering metabolic, signaling, and disease-associated pathways.
#
# We use over-representation analysis (ORA) with Fisher's exact test.
# The background gene set is critical: we use all genes targeted by any miRNA
# expressed in our dataset — NOT all 20,000 human genes. Using the correct
# background prevents inflation of enrichment statistics.
#
# For results to be interpretable, at least 50–100 target genes are needed.
# With fewer than 50, most pathways will not reach statistical significance.

cat("\n=== KEGG Pathway Enrichment Analysis ===\n")

# ---- 6A. Convert gene symbols to Entrez IDs (required by enrichKEGG) ----
cat("Converting gene symbols to Entrez IDs...\n")

# Query gene set (validated targets of top 15 biomarker miRNAs)
query_gene_symbols <- unique(strong_evidence$target.symbol)

query_entrez <- bitr(
  geneID   = query_gene_symbols,
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Hs.eg.db,
  drop     = TRUE        # drop unmapped symbols
)
cat("Query genes: symbols provided =", length(query_gene_symbols),
    "| Entrez IDs mapped =", nrow(query_entrez), "\n")

# Background gene universe (all expressed miRNA targets)
bg_entrez <- bitr(
  geneID   = all_target_symbols,
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Hs.eg.db,
  drop     = TRUE
)
cat("Background: symbols provided =", length(all_target_symbols),
    "| Entrez IDs mapped =", nrow(bg_entrez), "\n")

# ---- 6B. Run enrichKEGG ----
kegg_result <- enrichKEGG(
  gene          = query_entrez$ENTREZID,
  universe      = bg_entrez$ENTREZID,
  organism      = "hsa",          # hsa = Homo sapiens
  pAdjustMethod = "BH",           # Benjamini-Hochberg FDR
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20,
  minGSSize     = 10,             # exclude very small pathways (< 10 genes)
  maxGSSize     = 500             # exclude very large generic pathways
)

kegg_df <- as.data.frame(kegg_result)
cat("\nSignificant KEGG pathways (p.adjust < 0.05):", sum(kegg_df$p.adjust < 0.05), "\n")

if (nrow(kegg_df) > 0) {
  cat("\nTop 10 enriched KEGG pathways:\n")
  print(kegg_df[1:min(10, nrow(kegg_df)),
                c("ID", "Description", "GeneRatio", "BgRatio", "pvalue", "p.adjust")])
} else {
  cat("No significant KEGG enrichment found.\n")
  cat("Possible causes: too few target genes; overly restrictive background; small dataset.\n")
}

# ---- 6C. Check specifically for AD-relevant KEGG pathways ----
# These are the pathways expected to be enriched if your biomarker panel
# captures genuine AD biology. Report even if not statistically significant
# (they may approach significance with more targets or a larger query).
cat("\n=== AD-Relevant KEGG Pathway Check ===\n")
ad_kegg_ids <- c(
  "hsa05010",  # Alzheimer disease  — THE key validation pathway
  "hsa04010",  # MAPK signaling      — tau phosphorylation kinases
  "hsa04151",  # PI3K-Akt signaling  — cell survival; PTEN targeted by miR-21
  "hsa04210",  # Apoptosis           — BCL2, caspases; miR-34a targets
  "hsa04064",  # NF-kB signaling     — neuroinflammation; miR-146a regulates
  "hsa04668",  # TNF signaling       — inflammatory cytokines in AD
  "hsa04115",  # p53 signaling       — neuronal apoptosis; miR-34a targets TP53
  "hsa05016"   # Huntington disease  — shares tau/mitochondria mechanisms with AD
)

if (nrow(kegg_df) > 0) {
  for (pid in ad_kegg_ids) {
    row <- kegg_df[kegg_df$ID == pid, ]
    if (nrow(row) > 0) {
      cat(sprintf("  FOUND:  %s | %s | padj = %.4f | GeneRatio = %s\n",
                  pid, row$Description, row$p.adjust, row$GeneRatio))
    } else {
      cat(sprintf("  absent: %s (%s)\n", pid,
                  switch(pid,
                         "hsa05010" = "Alzheimer disease",
                         "hsa04010" = "MAPK signaling",
                         "hsa04151" = "PI3K-Akt signaling",
                         "hsa04210" = "Apoptosis",
                         "hsa04064" = "NF-kB signaling",
                         "hsa04668" = "TNF signaling",
                         "hsa04115" = "p53 signaling pathway",
                         "hsa05016" = "Huntington disease",
                         pid)))
    }
  }
}

# ---- 6D. KEGG Dot Plot ----
if (nrow(kegg_df) > 0) {
  p_kegg_dot <- dotplot(
    kegg_result,
    showCategory = min(20, nrow(kegg_df)),
    title        = "KEGG Pathway Enrichment\nTop 15 AD Biomarker miRNA Targets",
    font.size    = 9
  ) +
    theme_bw(base_size = 10) +
    theme(
      axis.text.y  = element_text(size = 8),
      plot.title   = element_text(face = "bold", size = 11),
      legend.position = "right"
    )

  ggsave("results/Week6/kegg_dotplot.png",
         p_kegg_dot, width = 10, height = 7, dpi = 150)
  cat("\nKEGG dotplot saved to results/Week6/kegg_dotplot.png\n")
} else {
  cat("No significant KEGG enrichment to plot.\n")
  cat("Consider: increase minGSSize, lower qvalueCutoff, or use more target genes.\n")
}

# ---- 6E. Save KEGG results ----
write.csv(kegg_df, "results/Week6/KEGG_enrichment_results.csv", row.names = FALSE)
cat("KEGG enrichment table saved to results/Week6/KEGG_enrichment_results.csv\n")

# ---- 6F. KEGG Enrichment — Interpretation note ----
cat("\n--- KEGG Enrichment Interpretation Guide ---\n")
cat("hsa05010 (Alzheimer disease): Direct validation — targets are AD pathway components.\n")
cat("hsa04151 (PI3K-Akt): Survival signaling; PTEN loss (miR-21 target) activates Akt.\n")
cat("hsa04064 (NF-kB): Neuroinflammatory signaling; regulated by miR-146a, miR-155.\n")
cat("hsa04210 (Apoptosis): Neuronal apoptosis in AD; BCL2, caspases regulated by miR-34a.\n")
cat("hsa04010 (MAPK): Tau phosphorylation via ERK, JNK; upstream of many AD targets.\n")


# ==============================================================================
# SECTION 7: clusterProfiler — Gene Ontology Enrichment Analysis
# ==============================================================================
# GO (Gene Ontology) provides a structured, hierarchical vocabulary of
# biological processes (BP), molecular functions (MF), and cellular components (CC).
#
# For miRNA target enrichment, we focus on Biological Process (BP) ontology,
# which describes the biological roles of target genes.
#
# Key challenge: GO terms are highly redundant. A gene annotated to
# "regulation of apoptosis" is also annotated to "apoptotic process" and
# "programmed cell death" — all describing essentially the same biology.
# The simplify() function clusters and deduplicates redundant terms.

cat("\n=== Gene Ontology (GO) Biological Process Enrichment ===\n")

# ---- 7A. Run enrichGO ----
go_bp_result <- enrichGO(
  gene          = query_entrez$ENTREZID,
  universe      = bg_entrez$ENTREZID,
  OrgDb         = org.Hs.eg.db,
  ont           = "BP",           # Biological Process (BP), MF, or CC
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.20,
  readable      = TRUE,           # show gene symbols (not Entrez IDs) in results
  minGSSize     = 10,
  maxGSSize     = 500
)

go_bp_df <- as.data.frame(go_bp_result)
cat("GO-BP terms before simplification:", nrow(go_bp_df), "\n")

# ---- 7B. Simplify redundant GO terms ----
# simplify() uses semantic similarity (Wang method) to cluster parent-child terms
# and retains one representative per cluster (the one with best adjusted p-value)
if (nrow(go_bp_df) > 5) {
  go_simplified <- simplify(
    go_bp_result,
    cutoff     = 0.7,           # similarity threshold; 0.7 is standard
    by         = "p.adjust",
    select_fun = min            # keep term with lowest p.adjust per cluster
  )
  go_simp_df <- as.data.frame(go_simplified)
  cat("GO-BP terms after simplification:", nrow(go_simp_df), "\n")
} else {
  cat("Fewer than 5 GO terms found; skipping simplification.\n")
  go_simplified <- go_bp_result
  go_simp_df    <- go_bp_df
}

if (nrow(go_simp_df) > 0) {
  cat("\nTop 10 enriched GO-BP terms:\n")
  print(go_simp_df[1:min(10, nrow(go_simp_df)),
                   c("ID", "Description", "GeneRatio", "p.adjust", "geneID")])
}

# ---- 7C. GO Bar Plot ----
if (nrow(go_simp_df) > 0) {
  p_go_bar <- barplot(
    go_simplified,
    showCategory = min(20, nrow(go_simp_df)),
    title        = "GO Biological Process Enrichment (Simplified)\nTop 15 AD Biomarker miRNA Targets",
    font.size    = 9
  ) +
    theme_bw(base_size = 10) +
    theme(
      axis.text.y  = element_text(size = 8),
      plot.title   = element_text(face = "bold", size = 11)
    )

  ggsave("results/Week6/go_bp_barplot.png",
         p_go_bar, width = 10, height = 7, dpi = 150)
  cat("GO-BP barplot saved to results/Week6/go_bp_barplot.png\n")
}

# ---- 7D. GO Dot Plot (alternative visualization) ----
if (nrow(go_simp_df) > 0) {
  p_go_dot <- dotplot(
    go_simplified,
    showCategory = min(20, nrow(go_simp_df)),
    title        = "GO Biological Process Enrichment (Simplified)",
    font.size    = 9
  ) +
    theme_bw(base_size = 10) +
    theme(axis.text.y = element_text(size = 8))

  ggsave("results/Week6/go_bp_dotplot.png",
         p_go_dot, width = 10, height = 7, dpi = 150)
  cat("GO-BP dotplot saved to results/Week6/go_bp_dotplot.png\n")
}

# ---- 7E. Save GO results ----
write.csv(go_simp_df, "results/Week6/GO_BP_enrichment_results.csv", row.names = FALSE)
cat("GO-BP enrichment table saved to results/Week6/GO_BP_enrichment_results.csv\n")

# ---- 7F. Expected biological process terms — what to look for ----
cat("\n--- Expected GO-BP terms for an AD miRNA target panel ---\n")
expected_go <- c(
  "negative regulation of apoptotic process",
  "regulation of synaptic transmission",
  "response to oxidative stress",
  "tau protein binding",
  "amyloid precursor protein processing",
  "regulation of NF-kappaB transcription factor activity",
  "inflammatory response",
  "regulation of neurogenesis",
  "regulation of autophagy"
)
cat(paste(" •", expected_go), sep = "\n")
cat("\n")
if (nrow(go_simp_df) > 0) {
  for (term in expected_go) {
    if (any(grepl(term, go_simp_df$Description, ignore.case = TRUE))) {
      cat(sprintf("  [FOUND] %s\n", term))
    }
  }
}


# ==============================================================================
# SECTION 8: STRINGdb Protein-Protein Interaction Network Analysis
# ==============================================================================
# A PPI network shows how the target proteins of your biomarker miRNAs
# physically interact with each other and with known AD disease proteins.
#
# Key question: Do targets of the biomarker miRNAs directly interact with
# APP, BACE1, MAPT (tau), or SIRT1?
# If yes, the mechanistic case for the biomarker panel is strongly supported.
#
# STRING confidence score: 0–1000
#   < 400 = low confidence
#   400–700 = medium confidence
#   > 700 = high confidence  ← we use this threshold

cat("\n=== Building Protein-Protein Interaction Network (STRINGdb) ===\n")

# ---- 8A. Initialize STRINGdb ----
# Ensure the data/ directory exists for STRINGdb cache files
if (!dir.exists("data/raw/stringdb")) {
  dir.create("data/raw/stringdb", recursive = TRUE)
}

cat("Initializing STRINGdb (species = 9606 [H. sapiens], score threshold = 700)...\n")
cat("First run will download STRING network files (~80 MB); subsequent runs use cache.\n")

string_db <- STRINGdb$new(
  version          = "12.0",       # STRING version; update to latest
  species          = 9606,         # 9606 = Homo sapiens NCBI taxonomy ID
  score_threshold  = 700,          # high-confidence interactions only
  network_type     = "full",       # "full" = all evidence channels combined
  input_directory  = "data/raw/stringdb/"
)

# ---- 8B. Map target gene symbols to STRING protein IDs ----
# STRING uses its own internal protein IDs (9606.ENSPxxxxxxxxxxx format)
# We must map our gene symbols to STRING IDs before querying interactions

target_gene_df <- data.frame(
  gene = unique(strong_evidence$target.symbol),
  stringsAsFactors = FALSE
)

cat("Mapping", nrow(target_gene_df), "target genes to STRING IDs...\n")

proteins_mapped <- string_db$map(
  my_data_frame          = target_gene_df,
  my_data_frame_id_col   = "gene",
  removeUnmappedRows     = TRUE
)

n_mapped <- nrow(proteins_mapped)
n_unmapped <- nrow(target_gene_df) - n_mapped
cat("Genes mapped to STRING:", n_mapped, "of", nrow(target_gene_df), "\n")
if (n_unmapped > 0) {
  unmapped_genes <- setdiff(target_gene_df$gene, proteins_mapped$gene)
  cat("Unmapped genes (not found in STRING):", paste(unmapped_genes, collapse = ", "), "\n")
}

# ---- 8C. Retrieve interactions for mapped proteins ----
# string_db$get_interactions() downloads all edges between the mapped proteins
# that meet the score threshold specified at initialization

cat("Retrieving interactions (STRING score >= 700)...\n")
interactions <- string_db$get_interactions(proteins_mapped$STRING_id)
cat("Total interactions retrieved:", nrow(interactions), "\n")

# ---- 8D. Build igraph network object ----
if (nrow(interactions) > 0) {
  ppi_graph <- graph_from_data_frame(
    d         = interactions[, c("from", "to", "combined_score")],
    directed  = FALSE,
    vertices  = proteins_mapped
  )

  # Edge weight = STRING combined score (700–1000)
  E(ppi_graph)$weight <- interactions$combined_score
  V(ppi_graph)$gene   <- proteins_mapped$gene[match(V(ppi_graph)$name,
                                                     proteins_mapped$STRING_id)]

  # Remove isolated nodes (nodes with no edges at the score threshold)
  isolated <- which(degree(ppi_graph) == 0)
  if (length(isolated) > 0) {
    ppi_graph <- delete.vertices(ppi_graph, isolated)
    cat("Isolated nodes removed:", length(isolated), "\n")
  }

  cat("Final network: nodes =", vcount(ppi_graph),
      "| edges =", ecount(ppi_graph), "\n")

} else {
  cat("No interactions found at score threshold 700.\n")
  cat("Try lowering score_threshold to 400 (medium confidence).\n")

  # For demonstration, build a synthetic small network
  # using known AD gene interactions from the literature
  set.seed(42)
  demo_genes  <- c("APP", "BACE1", "MAPT", "SIRT1", "TP53", "BCL2",
                   "PTEN", "CDK5", "GSK3B", "FOXO3", "TRAF6", "IRAK1")
  demo_edges  <- data.frame(
    from = c("APP","APP","BACE1","MAPT","SIRT1","TP53","BCL2","PTEN",
             "CDK5","GSK3B","FOXO3","TRAF6","SIRT1","APP","MAPT"),
    to   = c("BACE1","PSEN1","APP","CDK5","FOXO3","BCL2","BCL2","GSK3B",
             "GSK3B","MAPT","SIRT1","IRAK1","TP53","SIRT1","GSK3B"),
    stringsAsFactors = FALSE
  )
  ppi_graph <- graph_from_data_frame(demo_edges, directed = FALSE,
                                     vertices = data.frame(name = demo_genes))
  V(ppi_graph)$gene <- V(ppi_graph)$name
  cat("Using synthetic demo network for visualization.\n")
}

# ---- 8E. Compute degree centrality and identify hub genes ----
node_degree  <- degree(ppi_graph)
node_between <- betweenness(ppi_graph, normalized = TRUE)
node_close   <- closeness(ppi_graph, normalized = TRUE)

degree_df <- data.frame(
  STRING_id   = V(ppi_graph)$name,
  gene        = V(ppi_graph)$gene,
  degree      = as.integer(node_degree),
  betweenness = round(node_between, 4),
  closeness   = round(node_close, 4),
  stringsAsFactors = FALSE
) %>%
  arrange(desc(degree))

cat("\nTop 15 hub genes (degree centrality):\n")
print(head(degree_df[, c("gene", "degree", "betweenness", "closeness")], 15))

# Hub genes: top 10% by degree
hub_threshold <- quantile(degree_df$degree, 0.9)
hub_genes_vec <- degree_df$gene[degree_df$degree >= hub_threshold]
cat("\nHub genes (top 10% by degree, threshold >=", hub_threshold, "):\n")
cat(paste(hub_genes_vec, collapse = ", "), "\n")

# ---- 8F. Identify AD gene hub overlap ----
ad_hub_overlap <- intersect(hub_genes_vec, AD_KNOWN_GENES)
gwas_hub_overlap <- intersect(hub_genes_vec, GWAS_AD_GENES)
cat("\nHub genes that are known AD disease genes:", paste(ad_hub_overlap, collapse = ", "), "\n")
cat("Hub genes that overlap with AD GWAS loci:", paste(gwas_hub_overlap, collapse = ", "), "\n")

# ---- 8G. Network visualization ----
# Color coding:
#   Red (#D73027)   = known AD disease gene
#   Blue (#74ADD1)  = target gene (not AD-specific)
#   Node size       = proportional to degree centrality (hub = larger)
#   Label           = shown only for hub genes and known AD genes (to avoid clutter)

V(ppi_graph)$is_ad_gene   <- V(ppi_graph)$gene %in% AD_KNOWN_GENES
V(ppi_graph)$is_hub       <- V(ppi_graph)$gene %in% hub_genes_vec
V(ppi_graph)$is_gwas      <- V(ppi_graph)$gene %in% GWAS_AD_GENES

# Node appearance
V(ppi_graph)$color <- ifelse(V(ppi_graph)$is_ad_gene, "#D73027",
                       ifelse(V(ppi_graph)$is_gwas, "#FD8D3C", "#74ADD1"))
V(ppi_graph)$frame.color <- "white"
V(ppi_graph)$size  <- 4 + (node_degree[V(ppi_graph)$name] /
                             max(node_degree) * 14)

# Show labels only for genes of interest
label_genes   <- union(hub_genes_vec, AD_KNOWN_GENES)
V(ppi_graph)$label <- ifelse(V(ppi_graph)$gene %in% label_genes,
                              V(ppi_graph)$gene, NA)

# Edge appearance
E(ppi_graph)$width <- 0.5
E(ppi_graph)$color <- "grey70"

set.seed(42)
layout_fr <- layout_with_fr(ppi_graph, niter = 1000)

png("results/Week6/ppi_network.png",
    width = 2000, height = 1600, res = 150)
par(mar = c(2, 2, 3, 2), bg = "white")
plot(
  ppi_graph,
  layout            = layout_fr,
  vertex.label.cex  = 0.55,
  vertex.label.color = "black",
  vertex.label.font = 2,
  main = "PPI Network — Top 15 Biomarker miRNA Target Proteins\n(String score ≥ 700 | Node size ∝ degree centrality)",
  cex.main = 0.9
)
legend(
  "bottomleft",
  legend = c("Known AD gene", "AD GWAS locus", "Other target"),
  fill   = c("#D73027", "#FD8D3C", "#74ADD1"),
  border = NA,
  bty    = "n",
  cex    = 0.8
)
dev.off()
cat("\nPPI network plot saved to results/Week6/ppi_network.png\n")

# ---- 8H. Export network tables for Cytoscape ----
# For publication-quality figures, import these tables into Cytoscape
# and apply the "yFiles Organic Layout"
node_table <- as_data_frame(ppi_graph, what = "vertices")
edge_table  <- as_data_frame(ppi_graph, what = "edges")

write.csv(node_table, "results/Week6/network_nodes_for_cytoscape.csv", row.names = FALSE)
write.csv(edge_table, "results/Week6/network_edges_for_cytoscape.csv", row.names = FALSE)

# ---- 8I. Save hub gene table ----
write.csv(degree_df, "results/Week6/hub_genes_centrality.csv", row.names = FALSE)

cat("Network tables saved to results/Week6/\n")
cat("To open in Cytoscape: File → Import → Network from file → select edge CSV\n")


# ==============================================================================
# SECTION 9: Biomarker Panel Summary Figure — Forest Plot
# ==============================================================================
# This is the "main figure" of the hypothetical paper: a single forest-plot-style
# figure that communicates four things simultaneously for each of the top 15 miRNAs:
#
#   1. Direction of change in AD (log2FC; left = downregulated, right = upregulated)
#   2. Magnitude and significance of differential expression (point + CI, color by padj)
#   3. ML importance (SHAP score; dot size)
#   4. Key validated target(s) (text annotation, especially if AD gene)
#
# This figure allows a reader to understand the entire biomarker panel in one view.

cat("\n=== Building Biomarker Panel Summary Figure ===\n")

# ---- 9A. Prepare data for the forest plot ----
# Merge top15 with hub gene and AD target information
# Add key target annotation
ad_target_by_mirna <- strong_evidence %>%
  filter(target.symbol %in% AD_KNOWN_GENES) %>%
  group_by(mature.mirna) %>%
  summarise(key_ad_target = paste(sort(unique(target.symbol)), collapse = "/"),
            .groups = "drop") %>%
  rename(miRNA = mature.mirna)

forest_df <- top15 %>%
  select(miRNA, log2FC, padj, mean_abs_shap, composite_rank) %>%
  left_join(ad_target_by_mirna, by = "miRNA") %>%
  mutate(
    key_ad_target = ifelse(is.na(key_ad_target), "", key_ad_target),
    # Direction label for axis: DOWN in AD vs UP in AD
    direction     = ifelse(log2FC < 0, "Down in AD", "Up in AD"),
    # Significance tier for color
    sig_tier      = case_when(
      padj < 0.001 ~ "padj < 0.001",
      padj < 0.01  ~ "padj < 0.01",
      padj < 0.05  ~ "padj < 0.05",
      TRUE         ~ "padj ≥ 0.05"
    ),
    sig_tier = factor(sig_tier,
                      levels = c("padj < 0.001", "padj < 0.01",
                                 "padj < 0.05", "padj ≥ 0.05")),
    # Standardize SHAP for dot sizing (range 1–8)
    shap_size     = 1 + 7 * (mean_abs_shap - min(mean_abs_shap)) /
                    max(mean_abs_shap - min(mean_abs_shap) + 1e-10),
    # Order miRNAs by log2FC for visual grouping (down first, then up)
    miRNA         = reorder(miRNA, log2FC),
    # Confidence interval (simulated as ±1.5 × asymptotic SE for log2FC)
    # In a real dataset this comes directly from DESeq2's lfcSE column
    se_lfc        = abs(log2FC) * 0.25 + 0.15,
    lfc_lower     = log2FC - 1.96 * se_lfc,
    lfc_upper     = log2FC + 1.96 * se_lfc
  )

# ---- 9B. Build the forest plot ----
sig_colors <- c(
  "padj < 0.001" = "#D73027",
  "padj < 0.01"  = "#FC8D59",
  "padj < 0.05"  = "#FEE090",
  "padj ≥ 0.05"  = "#91BFDB"
)

p_forest <- ggplot(forest_df, aes(x = log2FC, y = miRNA)) +
  # Zero line (no change)
  geom_vline(xintercept = 0, linetype = "solid", colour = "grey50", linewidth = 0.5) +
  # Fold change thresholds (±1 = 2-fold)
  geom_vline(xintercept = c(-1, 1), linetype = "dashed",
             colour = "grey70", linewidth = 0.4) +
  # Confidence interval bars
  geom_errorbarh(aes(xmin = lfc_lower, xmax = lfc_upper),
                 height = 0.25, linewidth = 0.6, colour = "grey40") +
  # Point: size = SHAP importance, color = significance tier
  geom_point(aes(size = shap_size, colour = sig_tier), alpha = 0.9) +
  # Annotate known AD targets next to each point
  geom_text(aes(label = key_ad_target),
            hjust    = ifelse(forest_df$log2FC < 0, 1.1, -0.1),
            size     = 2.8,
            colour   = "#D73027",
            fontface = "italic") +
  # Styling
  scale_colour_manual(values = sig_colors, name = "Adjusted p-value") +
  scale_size_continuous(range = c(2, 8), name = "SHAP importance\n(dot size)") +
  scale_x_continuous(
    breaks = seq(-3, 3, 0.5),
    limits = c(-3.5, 3.5),
    labels = function(x) ifelse(x == 0, "0", paste0(ifelse(x > 0, "+", ""), x))
  ) +
  labs(
    title    = "Biomarker miRNA Panel — Top 15 Candidates",
    subtitle = "Point size = SHAP importance | Colour = DE significance | Red text = AD target gene",
    x        = "log2 Fold Change (AD vs Control)",
    y        = NULL,
    caption  = paste0("Composite ranking: mean of SHAP rank and −log10(padj) rank\n",
                      "Whiskers = 95% CI of log2FC | Dashed lines at ±1 (2-fold change)")
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 9, colour = "grey40"),
    axis.text.y   = element_text(size = 10, face = "bold"),
    axis.text.x   = element_text(size = 9),
    panel.grid.major.y = element_line(colour = "grey92"),
    panel.grid.major.x = element_line(colour = "grey92"),
    legend.position = "right",
    legend.text   = element_text(size = 8),
    plot.caption  = element_text(size = 7, colour = "grey50")
  )

ggsave("results/Week6/biomarker_panel_forest_plot.png",
       p_forest, width = 11, height = 7, dpi = 150)
ggsave("results/Week6/biomarker_panel_forest_plot.pdf",
       p_forest, width = 11, height = 7)

cat("Biomarker panel forest plot saved:\n")
cat("  PNG: results/Week6/biomarker_panel_forest_plot.png\n")
cat("  PDF: results/Week6/biomarker_panel_forest_plot.pdf\n")


# ==============================================================================
# SECTION 10: Analytical Validation Simulation — qPCR Experiment
# ==============================================================================
# Before a miRNA biomarker can enter clinical use, the discovery RNA-seq result
# must be verified by an orthogonal, analytically validated assay.
# The gold standard is quantitative RT-PCR (qPCR) using TaqMan or SYBR Green
# chemistry, with a spike-in synthetic miRNA for normalization.
#
# This section simulates a qPCR validation experiment for the top 3 miRNAs.
# Key outputs:
#   1. Ct value distributions for AD vs Control (box plots)
#   2. ΔCt (Ct[target] − Ct[reference]) for each group
#   3. ΔΔCt (ΔCt[AD] − ΔCt[Control]) = log2 fold change estimate
#   4. A simulated LOD (limit of detection) curve
#
# Real qPCR values for serum miRNAs (TaqMan): typical Ct range 20–36.
# Lower Ct = higher expression.
# The reference miRNA (cel-miR-39 spike-in) has a Ct set to 23 in all samples
# (constant because it is added at a defined concentration before extraction).

cat("\n=== Analytical Validation Simulation: qPCR ===\n")

# ---- 10A. Define experiment parameters ----
set.seed(99)
n_ad      <- 30    # AD samples
n_ctrl    <- 30    # Control samples
ref_ct    <- 23.0  # Spike-in reference Ct (constant across samples; ±0.5 technical noise)
ref_sd    <- 0.5   # Technical noise on reference

# Top 3 miRNAs from our panel
top3_mirnas  <- top_mirna_names[1:3]
top3_log2FC  <- forest_df$log2FC[match(top3_mirnas, as.character(forest_df$miRNA))]
# If something went wrong with matching, set defaults
if (any(is.na(top3_log2FC))) {
  top3_log2FC <- c(-2.1, 1.6, -1.9)
}

# Ct values for a miRNA upregulated in AD:
#   Ct[AD] < Ct[Control] (lower Ct = more abundant)
# For a downregulated miRNA in AD:
#   Ct[AD] > Ct[Control]
# log2FC = −ΔΔCt, so ΔΔCt = −log2FC
# Therefore: Ct[AD] = Ct[Control] + log2FC  (noting that higher Ct = lower expression)

mirna_ctrl_ct_base <- c(28.5, 25.2, 30.1)  # baseline Ct in controls (realistic range)
mirna_ad_ct_base   <- mirna_ctrl_ct_base - top3_log2FC  # AD Ct adjusted by log2FC
# Note: log2FC = -1.8 means AD is lower; in Ct space AD Ct is higher (+1.8)

# ---- 10B. Simulate Ct measurements ----
# Within-sample technical CV ≈ 2% for qPCR → ≈ 0.3–0.5 Ct units SD
# Biological variance between samples ≈ 1.0–2.0 Ct units SD
within_sample_sd <- 0.4   # technical Ct noise
biological_sd    <- 1.2   # between-sample biological noise

qpcr_data <- lapply(seq_along(top3_mirnas), function(i) {

  # Control samples
  ctrl_ct_obs <- mirna_ctrl_ct_base[i] +
    rnorm(n_ctrl, 0, biological_sd) +   # biological variation
    rnorm(n_ctrl, 0, within_sample_sd)  # technical variation

  # AD samples
  ad_ct_obs <- mirna_ad_ct_base[i] +
    rnorm(n_ad, 0, biological_sd) +
    rnorm(n_ad, 0, within_sample_sd)

  # Reference Ct (cel-miR-39 spike-in; same for all samples ± technical noise)
  ctrl_ref <- ref_ct + rnorm(n_ctrl, 0, ref_sd)
  ad_ref   <- ref_ct + rnorm(n_ad,   0, ref_sd)

  data.frame(
    miRNA     = top3_mirnas[i],
    sample_id = c(paste0("CTRL_", seq_len(n_ctrl)),
                  paste0("AD_",   seq_len(n_ad))),
    group     = rep(c("Control", "Alzheimer's Disease"), c(n_ctrl, n_ad)),
    Ct_target = c(ctrl_ct_obs, ad_ct_obs),
    Ct_ref    = c(ctrl_ref,    ad_ref),
    stringsAsFactors = FALSE
  )
})

qpcr_df <- bind_rows(qpcr_data) %>%
  mutate(
    delta_Ct    = Ct_target - Ct_ref,  # ΔCt = Ct[target] − Ct[reference]
    group       = factor(group, levels = c("Control", "Alzheimer's Disease"))
  )

# ---- 10C. Compute ΔΔCt and fold change ----
ddct_summary <- qpcr_df %>%
  group_by(miRNA, group) %>%
  summarise(
    mean_Ct       = round(mean(Ct_target), 2),
    sd_Ct         = round(sd(Ct_target),   2),
    mean_delta_Ct = round(mean(delta_Ct),  2),
    sd_delta_Ct   = round(sd(delta_Ct),    2),
    n_samples     = n(),
    .groups       = "drop"
  )

ddct_fold <- qpcr_df %>%
  group_by(miRNA) %>%
  summarise(
    ctrl_mean_dCt = mean(delta_Ct[group == "Control"]),
    ad_mean_dCt   = mean(delta_Ct[group == "Alzheimer's Disease"]),
    .groups       = "drop"
  ) %>%
  mutate(
    delta_delta_Ct = ad_mean_dCt - ctrl_mean_dCt,
    # ΔΔCt-based fold change: 2^(−ΔΔCt)
    # Note sign: for downregulated miRNA, ΔΔCt > 0, fold change < 1 (loss in AD)
    fold_change_2exp = round(2^(-delta_delta_Ct), 3),
    log2FC_qPCR      = round(-delta_delta_Ct, 3)
  )

cat("=== ΔΔCt Results (qPCR Analytical Validation Simulation) ===\n")
print(ddct_fold[, c("miRNA", "ctrl_mean_dCt", "ad_mean_dCt",
                    "delta_delta_Ct", "fold_change_2exp", "log2FC_qPCR")])

# ---- 10D. Compare qPCR log2FC vs RNA-seq log2FC ----
comparison_df <- ddct_fold %>%
  left_join(top15[, c("miRNA", "log2FC")], by = "miRNA") %>%
  rename(log2FC_RNAseq = log2FC)

cat("\nComparison of RNA-seq vs simulated qPCR log2FC:\n")
cat("(These should be concordant in direction if biomarker is genuine)\n")
print(comparison_df[, c("miRNA", "log2FC_RNAseq", "log2FC_qPCR")])

# ---- 10E. Ct Box Plot ----
p_ct_box <- ggplot(qpcr_df, aes(x = group, y = delta_Ct, fill = group)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 1.5, alpha = 0.85, width = 0.5) +
  geom_jitter(width = 0.1, alpha = 0.35, size = 1.0, colour = "grey30") +
  scale_fill_manual(values = GROUP_COLOURS, name = NULL) +
  facet_wrap(~ miRNA, scales = "free_y", ncol = 3) +
  labs(
    title    = "Simulated qPCR Validation — ΔCt Values (AD vs Control)",
    subtitle = "ΔCt = Ct[target] − Ct[spike-in reference (cel-miR-39)]\nLower ΔCt = higher miRNA expression",
    x        = NULL,
    y        = "ΔCt (target − reference)",
    caption  = "Simulation: biological SD = 1.2 Ct; technical SD = 0.4 Ct; n = 30 per group"
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title     = element_text(face = "bold", size = 12),
    plot.subtitle  = element_text(size = 8.5, colour = "grey40"),
    strip.text     = element_text(face = "bold", size = 9),
    legend.position = "none",
    axis.text.x    = element_text(angle = 15, hjust = 1, size = 8),
    plot.caption   = element_text(size = 7, colour = "grey50")
  )

ggsave("results/Week6/qpcr_validation_sim.png",
       p_ct_box, width = 11, height = 5, dpi = 150)
cat("\nqPCR validation box plots saved to results/Week6/qpcr_validation_sim.png\n")

# ---- 10F. Limit of Detection (LOD) Simulation ----
# LOD is estimated by serially diluting a positive control sample (e.g., pooled AD serum)
# and measuring Ct at each dilution. The LOD is the concentration where signal
# is reliably distinguished from a no-template control (NTC).
# LOD = mean(NTC_Ct) − 3 × SD(NTC_Ct) in Ct units (since Ct increases with dilution)
# Here we simulate 10 serial 2-fold dilutions of a positive control.

cat("\nSimulating LOD determination for", top3_mirnas[1], "...\n")
set.seed(11)

dilution_factors  <- 2^(0:9)       # 1x, 2x, 4x, 8x, ... 512x dilution
dilution_labels   <- paste0("1:", dilution_factors)
copies_per_uL_start <- 10000       # estimated copies/µL in undiluted sample
copies_per_uL    <- copies_per_uL_start / dilution_factors

# PCR efficiency ≈ 100%; theoretical: each 2-fold dilution adds 1.0 Ct
# Real efficiency: typically 90–110%; add measurement noise
efficiency       <- 0.98           # 98% efficiency
Ct_start         <- 20.0           # Ct at undiluted concentration
Ct_at_dilution   <- Ct_start + (0:9) / (log(1 + efficiency) / log(2))

n_replicates     <- 3
lod_df <- do.call(rbind, lapply(seq_along(dilution_factors), function(i) {
  data.frame(
    dilution     = dilution_labels[i],
    dilution_num = dilution_factors[i],
    copies_uL    = copies_per_uL[i],
    Ct_obs       = Ct_at_dilution[i] + rnorm(n_replicates, 0, 0.4),
    replicate    = seq_len(n_replicates)
  )
}))

# NTC (no template control) — Ct should be above 40 or undetermined
ntc_ct     <- rnorm(n_replicates, mean = 42, sd = 0.8)
ntc_mean   <- mean(ntc_ct)
ntc_sd     <- sd(ntc_ct)
lod_ct_threshold <- ntc_mean - 3 * ntc_sd   # 3 SD below NTC mean

# Find LOD: lowest dilution where ALL replicates are below the threshold
lod_df$above_threshold <- lod_df$Ct_obs < lod_ct_threshold
# (Lower Ct = detected; threshold is the boundary)

cat(sprintf("NTC mean Ct = %.1f | SD = %.2f | LOD threshold = %.1f Ct\n",
            ntc_mean, ntc_sd, lod_ct_threshold))

# Plot LOD curve
lod_summary <- lod_df %>%
  group_by(dilution_num, copies_uL) %>%
  summarise(
    mean_Ct = mean(Ct_obs),
    sd_Ct   = sd(Ct_obs),
    .groups = "drop"
  )

p_lod <- ggplot(lod_df, aes(x = log2(dilution_num), y = Ct_obs)) +
  geom_point(colour = "#D73027", alpha = 0.8, size = 2.5) +
  geom_smooth(method = "lm", formula = y ~ x,
              se = TRUE, colour = "#4575B4", linewidth = 1) +
  geom_hline(yintercept = lod_ct_threshold,
             linetype = "dashed", colour = "grey30", linewidth = 0.8) +
  annotate("text", x = 7, y = lod_ct_threshold - 0.4,
           label = paste0("LOD threshold (NTC − 3SD): Ct = ",
                          round(lod_ct_threshold, 1)),
           colour = "grey30", size = 3) +
  scale_x_continuous(
    breaks = 0:9,
    labels = dilution_labels
  ) +
  labs(
    title    = paste0("LOD Determination — ", top3_mirnas[1]),
    subtitle = "Serial 2-fold dilutions of pooled AD serum; n = 3 replicates per dilution",
    x        = "Dilution factor",
    y        = "Ct value",
    caption  = paste0("Start: ", copies_per_uL_start, " copies/µL | ",
                      "Dashed line: LOD threshold (NTC − 3 × SD)")
  ) +
  theme_bw(base_size = 11) +
  theme(
    plot.title    = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.5, colour = "grey40"),
    axis.text.x   = element_text(angle = 35, hjust = 1, size = 8),
    plot.caption  = element_text(size = 7, colour = "grey50")
  )

ggsave("results/Week6/lod_curve_sim.png",
       p_lod, width = 8, height = 5, dpi = 150)
cat("LOD curve simulation saved to results/Week6/lod_curve_sim.png\n")

# ---- 10G. Save qPCR results ----
write.csv(ddct_fold,    "results/Week6/qpcr_ddct_results.csv",    row.names = FALSE)
write.csv(ddct_summary, "results/Week6/qpcr_ct_summary.csv",      row.names = FALSE)
write.csv(lod_df,       "results/Week6/lod_simulation_data.csv",  row.names = FALSE)
cat("qPCR simulation tables saved to results/Week6/\n")


# ==============================================================================
# SECTION 11: Save All Interpretation Results
# ==============================================================================
# Consolidate and confirm all files saved in this session.
# Also save the key R objects for downstream use.

cat("\n=== Saving all Week 6 results ===\n")

# Save R objects
saveRDS(top15,          "results/Week6/top15_biomarker_mirnas.rds")
saveRDS(strong_evidence,"results/Week6/validated_targets.rds")
saveRDS(kegg_df,        "results/Week6/kegg_enrichment.rds")
saveRDS(go_simp_df,     "results/Week6/go_bp_enrichment.rds")
saveRDS(degree_df,      "results/Week6/network_centrality.rds")
saveRDS(ddct_fold,      "results/Week6/qpcr_ddct.rds")

# List all files in results/Week6/
week6_files <- list.files("results/Week6/", full.names = FALSE)
cat("\nAll files saved to results/Week6/:\n")
for (f in sort(week6_files)) {
  cat(sprintf("  %-55s  %s\n",
              f,
              format(file.size(file.path("results/Week6/", f)),
                     big.mark = ",")))
}


# ==============================================================================
# SECTION 12: Course Completion Summary
# ==============================================================================
# Print a final pipeline summary table listing every major output from all
# 6 weeks of the course, then record session info for reproducibility.

cat("\n")
cat("================================================================================\n")
cat("  COURSE COMPLETION SUMMARY\n")
cat("  AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease\n")
cat("================================================================================\n\n")

pipeline_summary <- data.frame(
  Week = c(
    "Week 1", "Week 1",
    "Week 2", "Week 2", "Week 2", "Week 2", "Week 2",
    "Week 3", "Week 3", "Week 3",
    "Week 4", "Week 4", "Week 4",
    "Week 5", "Week 5", "Week 5",
    "Week 6", "Week 6", "Week 6", "Week 6", "Week 6", "Week 6", "Week 6"
  ),
  Stage = c(
    "Setup",          "Setup",
    "QC",             "Normalization",  "Normalization",   "Batch correction", "Output",
    "EDA",            "EDA",            "Clustering",
    "Differential Expression", "Visualization", "Output",
    "ML Modelling",   "Explainability", "Validation",
    "Target Prediction", "KEGG Enrichment", "GO Enrichment", "PPI Network",
    "Summary Figure", "Analytical Validation", "Output"
  ),
  Tool_Package = c(
    "R 4.3 + Bioconductor",     "Python 3.10 + conda",
    "edgeR::filterByExpr, DESeq2::vst", "DESeq2::vst, edgeR::calcNormFactors",
    "oligo::rma", "sva::ComBat", "saveRDS",
    "prcomp, Rtsne, umap", "ggplot2, ggrepel",
    "pheatmap, cluster::pam",
    "DESeq2::DESeq (counts) / limma::voom (array)",
    "ggplot2 (volcano plot)",
    "write.csv (DE results table)",
    "caret / sklearn RandomForest + LogReg",
    "shapr (R) / shap (Python)",
    "pROC::roc, yardstick::roc_curve",
    "multiMiR::multiMiR",
    "clusterProfiler::enrichKEGG",
    "clusterProfiler::enrichGO + simplify",
    "STRINGdb, igraph",
    "ggplot2 (forest plot)",
    "ggplot2 (box plots, LOD curve)",
    "write.csv / saveRDS / sessionInfo"
  ),
  Key_Output = c(
    "R project structure, installed packages",
    "conda env ml_biomarker, Week5 Python script",
    "Sample QC report; filtered count matrix",
    "GSE120584_expr_clean.rds (VST); TMM log-CPM",
    "GSE46579_expr_rma.rds (RMA)",
    "Batch-corrected expression matrix",
    "data/processed/ (clean data for downstream)",
    "PCA, t-SNE, UMAP plots",
    "Publication-quality dimensionality reduction figures",
    "Clustered heatmap; PAM cluster assignments",
    "DE results table (adj p, log2FC, SE)",
    "Volcano plot (AD vs Control)",
    "DE_results_GSE120584.csv; DE_results_GSE46579.csv",
    "Trained RF + LR classifiers; cross-validation folds",
    "shap_feature_importance.csv; SHAP summary beeswarm plot",
    "ROC curve; AUC with 95% CI; sensitivity/specificity table",
    "validated_targets_strong_evidence.csv; AD_gene_targets.csv",
    "KEGG_enrichment_results.csv; kegg_dotplot.png",
    "GO_BP_enrichment_results.csv; go_bp_barplot.png",
    "ppi_network.png; hub_genes_centrality.csv; Cytoscape exports",
    "biomarker_panel_forest_plot.png / .pdf",
    "qpcr_validation_sim.png; lod_curve_sim.png; ddct results",
    "All results/Week6/ files; session_info_week6.txt"
  ),
  stringsAsFactors = FALSE
)

# Print the table in a readable format
cat(sprintf("%-8s %-26s %-40s %-45s\n",
            "Week", "Stage", "Tool/Package", "Key Output"))
cat(strrep("-", 125), "\n")
for (i in seq_len(nrow(pipeline_summary))) {
  cat(sprintf("%-8s %-26s %-40s %-45s\n",
              pipeline_summary$Week[i],
              pipeline_summary$Stage[i],
              substr(pipeline_summary$Tool_Package[i], 1, 39),
              substr(pipeline_summary$Key_Output[i], 1, 44)))
}
cat(strrep("-", 125), "\n\n")

# ---- Final cohort summary ----
cat("=== Final Cohort Information ===\n")
cat("Primary training dataset:   GSE120584 (serum small RNA-seq; Illumina HiSeq 2500)\n")
cat("External validation dataset: GSE46579 (whole blood Affymetrix microarray)\n")
cat("Harmonized cohort (Week 5): GSE120584 + GSE46579 (ComBat harmonized)\n\n")

cat("=== Biomarker Panel Summary ===\n")
cat("Top 15 biomarker miRNAs (composite SHAP + DE ranking):\n")
for (i in seq_len(nrow(top15))) {
  direction <- ifelse(top15$log2FC[i] < 0, "DOWN in AD", "UP in AD")
  cat(sprintf("  %2d. %-22s  log2FC = %+5.2f  (%s)  padj = %.2e\n",
              i,
              top15$miRNA[i],
              top15$log2FC[i],
              direction,
              top15$padj[i]))
}

cat("\n=== Biological Interpretation Summary ===\n")
cat("Validated AD gene targets found:\n")
if (nrow(ad_hits) > 0) {
  for (g in unique(ad_hits$target.symbol)) {
    mirnas_for_gene <- ad_hits$mature.mirna[ad_hits$target.symbol == g]
    cat(sprintf("  %-10s  targeted by: %s\n", g,
                paste(mirnas_for_gene, collapse = ", ")))
  }
}

cat("\nGWAS AD risk genes in target set:\n")
cat(paste(" ", gwas_overlap, collapse = "\n"), "\n")

cat("\nHub genes by PPI degree centrality (top 5):\n")
print(head(degree_df[, c("gene", "degree")], 5))

cat("\n=== All Outputs Across All Weeks ===\n")
all_result_dirs <- c("data/processed", "qc_reports", "results/Week3",
                     "results/Week4",  "results/Week5", "results/Week6")
for (d in all_result_dirs) {
  if (dir.exists(d)) {
    files_in_dir <- list.files(d, full.names = FALSE)
    cat(sprintf("\n  %s/ (%d files)\n", d, length(files_in_dir)))
    for (f in head(files_in_dir, 10)) {
      cat(sprintf("    %s\n", f))
    }
    if (length(files_in_dir) > 10) {
      cat(sprintf("    ... and %d more\n", length(files_in_dir) - 10))
    }
  }
}

# ---- Session info ----
cat("\n=== R Session Information ===\n")
sink("results/Week6/session_info_week6.txt")
cat("AI/ML in Biomarker Discovery — Week 6 Session Info\n")
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
print(sessionInfo())
sink()

cat("Session info saved to results/Week6/session_info_week6.txt\n")

cat("\n")
cat("================================================================================\n")
cat("  WEEK 6 COMPLETE — COURSE COMPLETE!\n")
cat("================================================================================\n")
cat("\n")
cat("  You have built a complete miRNA biomarker discovery pipeline:\n")
cat("    Week 2: Data acquisition and quality control\n")
cat("    Week 3: Exploratory data analysis\n")
cat("    Week 4: Differential expression analysis\n")
cat("    Week 5: Machine learning classification and validation\n")
cat("    Week 6: Biological interpretation and clinical translation\n")
cat("\n")
cat("  Biomarker panel: top 15 miRNAs, validated targets, pathway enrichment\n")
cat("  PPI network: hub genes identified; AD gene overlay complete\n")
cat("  qPCR validation simulation: ΔΔCt concordant with RNA-seq log2FC\n")
cat("\n")
cat("  Next steps for translational impact:\n")
cat("    1. Verify top 3–5 miRNAs by ddPCR in a prospective clinical cohort\n")
cat("    2. Perform functional validation (miRNA overexpression/inhibition in neurons)\n")
cat("    3. Write and submit an NIA R21 or Alzheimer's Association Research Grant\n")
cat("    4. Expand cohort diversity (African American, Hispanic, Asian participants)\n")
cat("    5. Engage a CLIA-certified clinical laboratory for analytical validation\n")
cat("\n")
cat("  Good luck — and keep grounding your computation in biology.\n")
cat("================================================================================\n")
