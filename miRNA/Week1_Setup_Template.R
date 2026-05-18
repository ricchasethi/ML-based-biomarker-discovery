################################################################################
# AI/ML in Biomarker Discovery — Week 1 Lab
# Title:   R Environment Setup & Orientation
# Disease: Alzheimer's Disease | Biomarker: miRNA
# Audience: Wet-lab biologists — no prior R experience required
#
# Learning Goals for This Script:
#   1. Verify your R and RStudio installation
#   2. Install all Bioconductor and CRAN packages needed for the course
#   3. Confirm every package loads without error
#   4. Practice basic R syntax using miRNA biology examples
#   5. Generate your first plot (miRNA expression bar chart)
#
# How to use:
#   - Run each section by highlighting it and pressing Ctrl+Enter (Windows/Linux)
#     or Cmd+Enter (Mac), OR click the "Run" button in RStudio.
#   - A section is separated by the ---- dividers below.
#   - Read every comment (lines starting with #) before running the code.
################################################################################


# ==============================================================================
# SECTION 1: Check Your R Version
# ==============================================================================
# R changes over time. This course requires R >= 4.3.0.
# The line below prints your current R version to the Console.

R.version.string    # Should print something like "R version 4.4.x (20xx-xx-xx)"

# If your version is older than 4.3.0, download the latest R from:
#   https://cran.r-project.org/
# Then restart RStudio before continuing.


# ==============================================================================
# SECTION 2: Install BiocManager (Once Only)
# ==============================================================================
# Bioconductor is a repository of R packages built specifically for
# genomics and bioinformatics. BiocManager is the package that lets
# you install Bioconductor packages. We install it from CRAN first.

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Confirm the Bioconductor version (should be 3.18 or later)
BiocManager::version()


# ==============================================================================
# SECTION 3: Install All Course Packages
# ==============================================================================
# This block installs every package needed across all 6 weeks.
# It will take 5–15 minutes the first time — this is normal.
# Packages already installed are skipped automatically.
#
# What each package is for:
#   GEOquery      — Download datasets directly from NCBI GEO (Week 2)
#   DESeq2        — Differential expression for RNA-seq count data (Week 4)
#   limma         — Differential expression for microarray data (Week 4)
#   edgeR         — Alternative differential expression (negative binomial) (Week 4)
#   multiMiR      — Query 14 miRNA-target interaction databases (Week 6)
#   miRBaseConverter — Convert old miRNA names to current miRBase format (Weeks 2-6)
#   clusterProfiler  — Gene Ontology and KEGG pathway enrichment (Week 6)
#   org.Hs.eg.db  — Human gene annotation database (Week 6)
#   pheatmap      — Heatmap visualization (Week 3)
#   ggplot2       — Publication-quality plotting (Weeks 3-6)
#   tidyverse     — Data manipulation toolkit (dplyr, tidyr, readr, stringr)
#   affy          — Load and process Affymetrix CEL files (Week 2)
#   oligo         — Modern alternative to affy for CEL files (Week 2)
#   sva           — Surrogate Variable Analysis: batch correction (Week 2)
#   reshape2      — Reshape data between wide and long format (Week 3)

bioc_packages <- c(
  "GEOquery",
  "DESeq2",
  "limma",
  "edgeR",
  "multiMiR",
  "miRBaseConverter",
  "clusterProfiler",
  "org.Hs.eg.db",
  "affy",
  "oligo",
  "sva"
)

cran_packages <- c(
  "ggplot2",
  "tidyverse",
  "pheatmap",
  "reshape2",
  "RColorBrewer",
  "ggrepel",
  "gridExtra",
  "knitr",
  "rmarkdown"
)

# Install Bioconductor packages
BiocManager::install(bioc_packages, ask = FALSE, update = FALSE)

# Install CRAN packages
install.packages(cran_packages, repos = "https://cloud.r-project.org")

cat("\n===> Package installation complete. Proceed to Section 4.\n")


# ==============================================================================
# SECTION 4: Verify All Packages Load Successfully
# ==============================================================================
# This is your installation health check. Each library() call loads a package
# into your R session. If you see an error here, Section 7 explains how to fix it.

cat("--- Loading Bioconductor packages ---\n")
library(GEOquery)
library(DESeq2)
library(limma)
library(edgeR)
library(multiMiR)
library(miRBaseConverter)
library(clusterProfiler)
library(org.Hs.eg.db)
library(affy)
library(sva)

cat("--- Loading CRAN packages ---\n")
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(pheatmap)
library(reshape2)
library(RColorBrewer)

cat("\n===> All packages loaded successfully! Environment is ready.\n")
sessionInfo()    # Prints full session details — save this output for troubleshooting


# ==============================================================================
# SECTION 5: R Basics — Concepts You Will Use Every Week
# ==============================================================================
# No need to memorise all of this now. Run each block and read what prints
# in the Console. Come back here when you encounter these constructs later.

# --- 5.1 Variables ---
# A variable stores a value. Use <- to assign.
patient_count <- 120
disease_label <- "Alzheimer's Disease"
cat("Study has", patient_count, "patients with", disease_label, "\n")

# --- 5.2 Vectors — the fundamental data structure in R ---
# A vector is a list of values of the same type.
# Here we store expression values (log2-transformed) for 6 miRNAs in one AD patient.
mirna_names  <- c("hsa-miR-21-5p", "hsa-miR-29a-3p", "hsa-miR-107",
                  "hsa-miR-132-3p", "hsa-miR-146a-5p", "hsa-miR-155-5p")

log2_expr    <- c(7.3, 5.1, 4.8, 3.9, 8.2, 6.7)

# Name the vector so each value knows which miRNA it belongs to
names(log2_expr) <- mirna_names
print(log2_expr)

# Access a single value by name
log2_expr["hsa-miR-21-5p"]

# Access multiple values by position
log2_expr[1:3]

# --- 5.3 Data Frames — a table with rows and columns ---
# In real analyses, you will work with expression matrices:
#   rows    = miRNAs (features)
#   columns = samples (patients)
# A data frame is R's equivalent of an Excel spreadsheet.

mirna_df <- data.frame(
  mirna       = mirna_names,
  log2_expr   = log2_expr,
  direction   = c("up", "down", "down", "down", "up", "up"),
  key_targets = c("PTEN, PDCD4", "BACE1, APP", "BACE1, Cofilin-1",
                  "FOXO3a, tau kinases", "TRAF6, IRAK1", "SHIP1, C/EBPβ"),
  stringsAsFactors = FALSE
)

print(mirna_df)

# Access a single column with $
mirna_df$mirna
mirna_df$log2_expr

# Filter rows using dplyr (which you loaded in Section 4)
mirna_df |>
  filter(direction == "up") |>
  select(mirna, log2_expr)

# --- 5.4 Functions ---
# You call a function by writing its name followed by () with arguments inside.
mean(log2_expr)           # average expression
sd(log2_expr)             # standard deviation
range(log2_expr)          # min and max
which.max(log2_expr)      # which miRNA has the highest expression?
which.min(log2_expr)      # which has the lowest?


# ==============================================================================
# SECTION 6: Your First Plot — miRNA Expression Bar Chart
# ==============================================================================
# ggplot2 builds plots in layers. Read each + line as "then add this layer."
# This will create a bar chart of miRNA expression for these 6 AD-relevant miRNAs.

# Add a colour column based on up/down regulation direction
mirna_df$colour <- ifelse(mirna_df$direction == "up", "#D73027", "#4575B4")
# Red = upregulated in AD (potentially harmful / inflammatory)
# Blue = downregulated in AD (potentially protective / lost)

# Reorder miRNAs by expression for a cleaner chart
mirna_df$mirna <- factor(mirna_df$mirna,
                          levels = mirna_df$mirna[order(mirna_df$log2_expr)])

p <- ggplot(mirna_df, aes(x = mirna, y = log2_expr, fill = direction)) +
  geom_col(width = 0.65, colour = "white", linewidth = 0.3) +
  scale_fill_manual(
    values  = c("up" = "#D73027", "down" = "#4575B4"),
    labels  = c("up" = "Upregulated in AD", "down" = "Downregulated in AD"),
    name    = "Direction"
  ) +
  coord_flip() +
  labs(
    title    = "Week 1 Demo: AD-Relevant miRNA Expression",
    subtitle = "Simulated log2-expression values for 6 key miRNAs in Alzheimer's Disease",
    x        = "miRNA",
    y        = "Expression (log2)",
    caption  = "Source: Illustrative data — see Ludwig et al. 2019, Fattahi et al. 2024"
  ) +
  theme_bw(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(colour = "grey40"),
    legend.position = "top"
  )

print(p)

# Save to disk in your working directory
ggsave("Week1_miRNA_expression_demo.png", plot = p, width = 8, height = 5, dpi = 150)
cat("Plot saved as Week1_miRNA_expression_demo.png\n")


# ==============================================================================
# SECTION 7: Preview GEOquery — What We Will Use in Week 2
# ==============================================================================
# GEOquery lets you download any public GEO dataset directly into R.
# This week we just confirm it works; downloading will happen in Week 2.
#
# The key GEO datasets for this course:
#   GSE46579  — AD whole blood microarray (Affymetrix)
#   GSE120584 — AD serum small RNA-seq (Illumina)
#
# Run the line below to check GEOquery can reach NCBI servers.
# It downloads only a tiny metadata record (not the full dataset).

cat("Testing GEOquery connection to NCBI...\n")
gse_info <- tryCatch(
  {
    getGEO("GSE46579", GSEMatrix = FALSE, getGPL = FALSE)
  },
  error = function(e) {
    cat("Connection test failed:", conditionMessage(e), "\n")
    cat("This is OK for now — check your internet connection and try again.\n")
    NULL
  }
)

if (!is.null(gse_info)) {
  cat("GEOquery is working. GSE46579 metadata retrieved.\n")
  cat("Study title:", Meta(gse_info)$title, "\n")
}


# ==============================================================================
# SECTION 8: Key miRNA Biology Quick Reference
# ==============================================================================
# This section creates a reference data frame of the 12 most studied
# AD-relevant miRNAs from Module 1.2.4. You can query it at any time.

ad_mirna_ref <- data.frame(
  mirna       = c("hsa-miR-29a/b", "hsa-miR-107", "hsa-miR-9",
                  "hsa-miR-132/212", "hsa-miR-34a", "hsa-miR-146a",
                  "hsa-miR-155", "hsa-miR-21-5p", "hsa-miR-26a/26b-5p",
                  "hsa-miR-532-5p", "hsa-miR-128", "hsa-miR-181"),
  direction   = c("down", "down", "down", "down", "up", "up",
                  "up", "up", "down", "variable", "down", "up"),
  key_targets = c("BACE1, APP", "BACE1, Cofilin-1", "NFkB, SIRT1",
                  "FOXO3a, tau kinases", "SIRT1, BCL2", "TRAF6, IRAK1",
                  "SHIP1, C/EBPb", "PTEN, PDCD4", "PTEN, CDK5",
                  "Multiple", "PPARg, Bax", "SIRT1, GRP78"),
  pathway     = c("Amyloid production", "Amyloid production", "Neuroinflammation",
                  "Tau phosphorylation", "Apoptosis / Aging", "Neuroinflammation",
                  "Neuroinflammation", "Cell survival", "Cognitive decline",
                  "General AD", "Neuroprotection", "Tau / ER stress"),
  stringsAsFactors = FALSE
)

# Print the full reference table
print(ad_mirna_ref)

# Filter to see only upregulated miRNAs
cat("\nUpregulated miRNAs in AD blood:\n")
print(ad_mirna_ref[ad_mirna_ref$direction == "up", c("mirna", "key_targets", "pathway")])

# Filter by pathway keyword
cat("\nmiRNAs linked to Neuroinflammation:\n")
print(ad_mirna_ref[grepl("Neuroinflammation", ad_mirna_ref$pathway), ])


# ==============================================================================
# SECTION 9: Troubleshooting Guide
# ==============================================================================
# Common errors and how to fix them:
#
# ERROR: "there is no package called 'X'"
#   CAUSE:  The package was not installed, or installation failed silently.
#   FIX:    Run: install.packages("X")  OR  BiocManager::install("X")
#           Then try library(X) again.
#
# ERROR: "package 'X' was built under R version Y.Z"
#   CAUSE:  Minor version mismatch — usually harmless.
#   FIX:    Ignore the warning unless you see actual errors downstream.
#
# ERROR: "Error in getGEO ... could not resolve host"
#   CAUSE:  No internet connection or NCBI is temporarily unavailable.
#   FIX:    Check your internet connection. Try again in a few minutes.
#           On institutional networks, a VPN or proxy may be needed.
#
# ERROR: "'BiocManager' is not available for R version X.Y.Z"
#   CAUSE:  Your R version is too old.
#   FIX:    Download the latest R from https://cran.r-project.org/
#           Reinstall RStudio after upgrading R.
#
# GENERAL TIP: If you see a red error message in the Console, copy it and
#   paste it into the course discussion forum or share it with the instructor.
#   Include the output of sessionInfo() so we can reproduce your environment.


# ==============================================================================
# SECTION 10: Save Your Session & Next Steps
# ==============================================================================

cat("\n")
cat("=============================================================\n")
cat(" Week 1 Setup Complete!\n")
cat("=============================================================\n")
cat(" R version:       ", R.version.string, "\n")
cat(" Working dir:     ", getwd(), "\n")
cat(" Packages ready:  GEOquery, DESeq2, limma, edgeR,\n")
cat("                  multiMiR, clusterProfiler, ggplot2,\n")
cat("                  tidyverse, pheatmap, sva, affy, oligo\n")
cat("=============================================================\n")
cat("\n NEXT WEEK (Week 2):\n")
cat("   - Navigate NCBI GEO and identify AD miRNA datasets\n")
cat("   - Download GSE46579 (whole blood microarray) using GEOquery\n")
cat("   - Inspect CEL files and sample metadata\n")
cat("   - Run quality control: NUSE, RLE, and correlation heatmaps\n")
cat("   - Normalize with RMA (microarray) or DESeq2 (RNA-seq)\n")
cat("\n READING ASSIGNMENT:\n")
cat("   1. Fattahi et al. (2024) DOI: 10.1007/s11011-024-01431-7\n")
cat("   2. Ludwig et al. (2019) DOI: 10.1016/j.gpb.2019.09.004\n")
cat("=============================================================\n")

# Save the workspace so you can pick up where you left off
# (This saves all variables to a .RData file in your working directory)
save.image(file = "Week1_session.RData")
cat("Session saved to Week1_session.RData\n")
