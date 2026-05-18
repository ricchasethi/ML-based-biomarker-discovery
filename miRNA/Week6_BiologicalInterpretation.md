# Week 6: Biological Interpretation & Clinical Translation
## AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease

---

## Learning Objectives

By the end of Week 6, you will be able to:
1. Cross-reference ML-derived feature importance (SHAP values) with the published literature to distinguish statistical biomarkers from mechanistically grounded biomarkers
2. Query miRNA–target databases (miRTarBase, TargetScan, miRDB, multiMiR) to build evidence-tiered target gene lists for a panel of candidate biomarker miRNAs
3. Perform over-representation analysis (ORA) using clusterProfiler, and interpret enriched KEGG pathways and GO terms in the context of Alzheimer's disease biology
4. Navigate the five-stage biomarker development roadmap from discovery to clinical implementation, and identify what analytical and clinical validation experiments are required

---

## Conceptual Overview: The End of the Pipeline Is the Beginning of Biology

Over the past five weeks, you have built a complete computational pipeline: you downloaded and quality-controlled miRNA expression data from public repositories, explored its structure with dimensionality reduction and clustering, identified differentially expressed miRNAs, trained and tuned machine learning classifiers, and validated their performance on an independent cohort. You have a list of miRNAs. You have SHAP values. You have AUC curves.

But what does it mean biologically?

This is the question that separates a data analysis exercise from a scientific contribution. A list of miRNAs ranked by SHAP importance is like a photograph of a crime scene — it shows you something happened, but it does not tell you why, or what to do about it. Biological interpretation is the investigation that follows. It asks: Do these miRNAs have known functions in the brain or in neurodegeneration? Do they regulate genes that are already implicated in AD? Do the pathways they collectively modulate align with what we know about amyloid processing, tau phosphorylation, neuroinflammation, or synaptic loss?

This week also shifts the perspective from retrospective (what did the model learn from a GEO dataset?) to prospective (what would it take to translate this into a clinical blood test for AD?). That translation is long, difficult, regulated, and ethically non-trivial. Understanding the pathway is essential for anyone who wants to do more than publish a preprint.

The philosophy of this week: **the computational result is your hypothesis; the biological interpretation is your science.**

---

## MODULE 6.1 — Connecting ML Features to Biology

### 6.1.1 Revisiting the Biomarker Panel from Week 5

In Week 5, you trained a random forest and/or logistic regression classifier on the harmonized expression matrix (GSE120584 + GSE46579), validated it on a held-out external cohort, and computed SHAP values to explain the model's predictions. The SHAP analysis produced a ranked list of miRNAs: those at the top contributed most to separating AD from control samples.

Before biological interpretation begins, it is worth pausing to understand exactly what that ranking means — and what it does not mean.

**What SHAP importance tells you:**
- Which miRNAs, when their expression changes from the cohort average, most reliably shift the model's predicted probability toward AD or toward control
- The direction of each miRNA's effect (high expression pushes toward AD, or pushes away)
- The magnitude of the effect across the full test cohort (mean |SHAP| across all samples)

**What SHAP importance does not tell you:**
- Whether a miRNA is causally involved in AD pathogenesis
- Whether the association is with AD specifically, or with aging, medication use, or some other correlated variable
- Whether the association will replicate in an independent, prospectively collected cohort
- Whether a miRNA is technically measurable with sufficient precision for clinical use

Keep these limitations in mind throughout this week. Every biological interpretation is a hypothesis, not a conclusion.

---

### 6.1.2 Cross-Referencing SHAP Rankings with the Differential Expression Results

Your biomarker panel should reflect evidence from two independent analyses:
1. **Differential expression (Week 4):** miRNAs with statistically significant differences in mean expression between AD and control (adjusted p < 0.05; |log2FC| > 0.5)
2. **SHAP feature importance (Week 5):** miRNAs that contributed most to correct model predictions

These two lists will partially overlap but will not be identical. A miRNA can be highly differentially expressed but contribute little to the ML model (because it is correlated with many other features); conversely, a miRNA that is only modestly dysregulated can have high SHAP importance if it captures a distinct axis of variation.

**Constructing a composite ranking score:**

The simplest approach is to rank all miRNAs by their DE significance (−log10 adjusted p-value) and separately by SHAP importance (mean |SHAP| across samples), then compute the average of the two ranks. miRNAs ranked highly by both criteria are the most compelling candidates. This is formalized in Module 6.2 and in the Week 6 R script.

---

### 6.1.3 The Difference Between a Statistical Biomarker and a Mechanistic Biomarker

These two categories represent very different levels of scientific confidence, and understanding the distinction is essential when writing grants, papers, or clinical validation proposals.

**Statistical biomarker (association-only):**
- Defined entirely by the observed association between a molecular measurement and a disease state
- Requires no mechanistic understanding
- Example: "miR-X is significantly downregulated in AD serum (fold change = 3.2, adjusted p = 0.001)"
- Risk: the association may reflect a confounder (medication, comorbidity, hemolysis, batch effect) rather than AD biology

**Mechanistic biomarker (biologically grounded):**
- The molecular entity has a known or proposed biological function that plausibly connects it to the disease process
- Example: "miR-X targets BACE1 mRNA; its downregulation in AD serum could reflect loss of BACE1 suppression in peripheral immune cells, consistent with elevated amyloid production observed in this cohort"
- Stronger basis for clinical translation: if the mechanism is right, the biomarker is more likely to be specific to AD rather than to confounders

The goal of Modules 6.2–6.3 is to build mechanistic grounding for the top biomarker candidates.

---

### 6.1.4 Why Mechanistic Grounding Matters for Clinical Translation

Regulatory agencies and clinical laboratories do not just ask "does this test work in your dataset?" They ask: "why should we believe this test will work in the next patient?" Mechanistic grounding provides the scientific rationale for extrapolation.

Practically, this means:
- A biomarker with a known mechanism can be tested with orthogonal assays (e.g., measure the target gene's protein level; check if it correlates with the miRNA)
- A mechanistic biomarker can be tested in disease models (cell lines, animal models) to support causality
- Funding agencies (NIA, Alzheimer's Association) prioritize biomarkers with mechanistic hypotheses over purely data-driven associations
- A mechanistic biomarker panel with convergent evidence (multiple miRNAs all pointing to the same pathway) is more convincing than a panel of unrelated miRNAs

**The three-tier evidence pyramid for your biomarker panel:**

```
                  Tier 3 — Mechanistic
               (miRNA regulates known AD gene;
                pathway enrichment aligns with AD;
                protein network connects to APP/MAPT/BACE1)

           Tier 2 — Replication
         (DE significant in multiple datasets;
          replicates in external validation cohort)

     Tier 1 — Statistical Discovery
   (significant DE in primary dataset;
    high SHAP importance in ML model)
```

A Tier 1 finding is publishable as a preliminary result. A Tier 2 finding is publishable as a full biomarker paper. A Tier 3 finding provides the foundation for a grant application and clinical translation effort.

---

## MODULE 6.2 — miRNA Target Prediction

### 6.2.1 How miRNAs Regulate Gene Expression — A Refresher

Before predicting targets, recall the mechanism: miRNAs exert their regulatory effect primarily through imperfect base-pairing between their **seed sequence** (nucleotides 2–8 from the 5' end) and complementary sequences in the **3' untranslated region (3' UTR)** of target mRNAs. This interaction recruits the RISC (RNA-induced silencing complex), leading to mRNA degradation or translational repression. Each miRNA typically has hundreds of potential target genes; each gene can be regulated by multiple miRNAs. This redundancy and pleiotropy is why pathway-level analysis (Module 6.3) is often more interpretable than individual target analysis.

**Important biological caveats:**
- Predicted targets are not necessarily regulated in every cell type or tissue; context matters
- Blood-based miRNAs may not directly regulate brain gene expression — the connection may be indirect (e.g., peripheral immune cells producing miRNAs that reflect brain state)
- The fact that a miRNA can target a gene does not mean it does so physiologically at relevant concentrations

---

### 6.2.2 Three Tiers of Target Evidence

When predicting miRNA targets, evidence quality varies enormously. Always stratify by evidence tier:

**Tier 1 — Experimental validation (gold standard):**
Direct biochemical evidence that a miRNA binds to and regulates a specific target mRNA. Methods include:
- **Luciferase reporter assays:** 3' UTR of target gene cloned downstream of luciferase; miRNA overexpression reduces luminescence
- **AGO2 CLIP-seq / HITS-CLIP / PAR-CLIP:** Crosslinking immunoprecipitation of the Argonaute-2 protein pulls down miRNA:mRNA complexes; sequencing identifies binding sites genome-wide
- **Western blot or RT-qPCR validation:** After miRNA overexpression or inhibition, direct measurement of target protein or mRNA

Validated interactions are curated in **miRTarBase** (Huang et al., 2022).

**Tier 2 — Computational predictions (support):**
Algorithms that score the likelihood of functional targeting based on:
- Seed sequence complementarity (TargetScan; Agarwal et al., 2015)
- Seed match conservation across species (conserved targeting more likely to be functional)
- Site accessibility (local mRNA secondary structure affects RISC access)
- Multiple site synergy (two binding sites in same 3' UTR increase repression)

Major tools: **TargetScan** (context++ score), **miRDB** (probability score), **DIANA-microT**, **PicTar**, **miRanda**

**Tier 3 — Predicted with low confidence:**
Databases that aggregate all predictions without stringent filtering. Useful for generating a comprehensive list but require downstream filtering by evidence tier.

---

### 6.2.3 The multiMiR R Package — Querying 14 Databases Simultaneously

The `multiMiR` package (Ru et al., 2014) provides a unified R interface to 14 validated and predicted miRNA–target databases. Instead of querying each database individually through web interfaces, one function call retrieves everything.

**Databases queried by multiMiR:**

| Category | Databases |
|----------|-----------|
| Validated (experimental) | miRTarBase, miRecords, TarBase |
| Predicted | TargetScan, miRDB, DIANA-microT-CDS, PicTar, PITA, RNA22, miRanda, miRWalk, miRmap, STarMir, Microcosm |

**Key function arguments:**

```r
multiMiR(
  org         = "hsa",      # organism: "hsa" = human
  mirna       = "hsa-miR-21-5p",   # one or more miRNA names
  table       = "validated",       # "validated", "predicted", "all", or specific DB name
  predicted.cutoff = 20,   # percentile cutoff for predicted databases (20 = top 20%)
  predicted.cutoff.type = "p",
  use.tibble  = TRUE
)
```

---

### 6.2.4 Generating Target Gene Lists for the Top 10 Biomarker miRNAs

The following code block illustrates the complete workflow. Full working code is in the Week 6 R script.

```r
library(multiMiR)
library(dplyr)

# Assume top_mirnas is a character vector of your top 10 biomarker miRNAs
# e.g., from your composite SHAP + DE ranking in the R script
top_mirnas <- c("hsa-miR-21-5p", "hsa-miR-146a-5p", "hsa-miR-132-3p",
                "hsa-miR-107",   "hsa-miR-29a-3p",  "hsa-miR-128-3p",
                "hsa-miR-34a-5p","hsa-miR-181a-5p", "hsa-miR-9-5p",
                "hsa-miR-155-5p")

# Query validated interactions for all top miRNAs
validated_targets <- multiMiR(
  org    = "hsa",
  mirna  = top_mirnas,
  table  = "validated",
  use.tibble = TRUE
)

# Extract the result data frame
val_df <- validated_targets@data

# How many validated interactions were found?
cat("Total validated miRNA-target interactions:", nrow(val_df), "\n")
cat("Unique target genes:", length(unique(val_df$target.symbol)), "\n")

# Filter for strong evidence only
# Strong evidence: interactions with experiment type "Luciferase reporter assay"
# or "qRT-PCR" or "Western blot" — not just microarray/sequencing correlations
strong_evidence <- val_df %>%
  filter(grepl("luciferase|Luciferase|qRT-PCR|Western", experiment)) %>%
  distinct(mature.mirna, target.symbol, .keep_all = TRUE)

cat("Interactions with strong experimental evidence:", nrow(strong_evidence), "\n")

# Check for known AD genes in the target list
ad_genes <- c("APP", "BACE1", "MAPT", "PSEN1", "PSEN2", "APOE", 
              "SIRT1", "FOXO3", "CDK5", "ADAM10", "CLU", "BIN1")
ad_hits <- strong_evidence %>%
  filter(target.symbol %in% ad_genes)

cat("\n=== AD-relevant targets found ===\n")
print(ad_hits[, c("mature.mirna", "target.symbol", "experiment")])
```

**Expected biological findings — what to look for:**

Several miRNAs that frequently emerge as AD biomarkers have well-characterized targets in AD-relevant biology:

| miRNA | Key Validated Targets | AD Relevance |
|-------|----------------------|--------------|
| **hsa-miR-29a/b-3p** | BACE1, DNMT3b, BIM | Downregulated in AD; suppresses BACE1 (β-secretase); loss leads to increased Aβ production |
| **hsa-miR-107** | BACE1, CDK6, DICER1 | Downregulated in early AD; early loss may contribute to amyloid cascade initiation |
| **hsa-miR-132-3p** | FOXO3a, EP300, ITPKB, SIRT1 | Strongly downregulated in AD brain and blood; regulates tau phosphorylation via GSK-3β axis |
| **hsa-miR-146a-5p** | IRAK1, TRAF6, CFH | Upregulated in AD; modulates NF-κB neuroinflammatory signaling |
| **hsa-miR-155-5p** | SHIP1, SOCS1, PU.1 | Elevated in AD; promotes microglial activation and neuroinflammation |
| **hsa-miR-34a-5p** | BCL2, SIRT1, CDK6 | Upregulated in AD; promotes apoptosis; targets SIRT1 (NAD+ metabolism, tau deacetylation) |
| **hsa-miR-9-5p** | BACE1, REST, NFκB1 | Context-dependent; downregulated in AD neurons; regulates neurogenesis |
| **hsa-miR-21-5p** | PTEN, PDCD4, FASLG | Upregulated in AD; anti-apoptotic and pro-inflammatory roles |

> **Biological note for wet-lab scientists:** The presence of miR-29a/b and miR-107 as top biomarkers in your model is one of the most replicated findings in the blood miRNA AD literature, originally described by Hébert et al. (2008) for brain tissue and extended to blood in multiple subsequent studies. If your model identifies these miRNAs, that is a strong internal validation signal — it means your pipeline is detecting real biology, not noise.

---

### 6.2.5 miRWalk — Distinguishing 3' UTR from CDS and 5' UTR Targeting

Most miRNA targeting occurs through the 3' UTR, but binding within the coding sequence (CDS) and 5' UTR also occurs, with somewhat different functional consequences. miRWalk (Sticht et al., 2018) catalogs predicted and validated binding sites across all three regions.

For biomarker interpretation, focus primarily on 3' UTR interactions (most functionally validated). CDS binding can provide additional confidence when a 3' UTR interaction already exists.

---

## MODULE 6.3 — Gene Ontology and Pathway Enrichment Analysis

### 6.3.1 The Logic of Enrichment Analysis

You now have a list of target genes for your top biomarker miRNAs — perhaps several hundred unique genes. Interpreting hundreds of individual genes is impossible without a framework. Enrichment analysis provides that framework: it asks whether genes from particular biological processes or pathways are **overrepresented** in your target list compared to what you would expect by chance.

**Analogy:** Imagine you draw 300 names from a phone book of 20,000 people. You notice that 40 of them share the surname "Kim," whereas only 60 people named Kim are in the full phone book. Kim represents 40/300 = 13.3% of your list, but only 60/20,000 = 0.3% of the full book. This enrichment is extremely unlikely by chance. The same logic applies: if 30 of your 300 target genes are involved in "tau protein binding" (a GO term), and only 50 of 20,000 human genes belong to that term, the enrichment is statistically significant.

---

### 6.3.2 Over-Representation Analysis (ORA) vs Gene Set Enrichment Analysis (GSEA)

These are the two principal approaches, and they answer subtly different questions:

**Over-Representation Analysis (ORA):**
- Input: a binary list — genes in your target set vs all other genes (background)
- Test: Fisher's exact test (or hypergeometric test) at each pathway term
- Question: "Is pathway X overrepresented in my target gene list compared to the background?"
- Limitation: treats all genes in the list as equally important; ignores continuous expression values

**Gene Set Enrichment Analysis (GSEA):**
- Input: a ranked list of ALL genes, ranked by a score (e.g., −log10(p) × sign(log2FC)); no hard threshold
- Test: Kolmogorov-Smirnov-based test; compares the distribution of pathway genes across the ranked list
- Question: "Do genes in pathway X tend to cluster at the top or bottom of the ranked list?"
- Advantage: uses the full expression ranking; no arbitrary cutoff needed
- Disadvantage: computationally intensive; harder to interpret; requires a ranked gene list from DE analysis

**For our use case (miRNA target enrichment):** We primarily use ORA, because we have a discrete list of miRNA targets (not a ranked gene expression list). GSEA is more appropriate for mRNA expression data directly.

---

### 6.3.3 Background Gene Set — A Critical Choice

One of the most common methodological errors in enrichment analysis is using the wrong background gene set.

**Incorrect background (too large):** Using all human genes (~20,000) as background when your assay only measured 500 miRNAs that target a fraction of those genes. This artificially inflates enrichment of any term.

**Correct background:** All genes that **could have been in your target list** — i.e., all genes targeted by any miRNA expressed in your dataset (not just your top biomarker miRNAs). This is the "expressed miRNA targetome."

```r
# Correct approach:
# 1. Get all miRNAs expressed in your dataset (above detection threshold)
# 2. Query their targets (validated + high-confidence predicted)
# 3. Use those target genes as background for ORA

# This ensures the statistical test is: 
# "Among all genes targeted by expressed miRNAs, are AD-pathway genes 
#  enriched in the targets of the top biomarker miRNAs?"
```

---

### 6.3.4 clusterProfiler — enrichKEGG

`clusterProfiler` (Yu et al., 2012) is the standard Bioconductor package for ORA and GSEA. The `enrichKEGG` function queries the KEGG pathway database.

```r
library(clusterProfiler)
library(org.Hs.eg.db)
library(ggplot2)

# Convert gene symbols to Entrez IDs (required by enrichKEGG)
# Use bitr() from clusterProfiler
target_entrez <- bitr(
  geneID   = unique(strong_evidence$target.symbol),
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Hs.eg.db
)

background_entrez <- bitr(
  geneID   = unique(all_targets$target.symbol),  # all expressed miRNA targets
  fromType = "SYMBOL",
  toType   = "ENTREZID",
  OrgDb    = org.Hs.eg.db
)

# Run KEGG enrichment
kegg_result <- enrichKEGG(
  gene          = target_entrez$ENTREZID,
  universe      = background_entrez$ENTREZID,
  organism      = "hsa",     # hsa = Homo sapiens
  pAdjustMethod = "BH",      # Benjamini-Hochberg FDR correction
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.2,
  minGSSize     = 10,        # minimum pathway size
  maxGSSize     = 500        # exclude very large generic pathways
)

# View significant pathways
kegg_df <- as.data.frame(kegg_result)
cat("Significant KEGG pathways (p.adjust < 0.05):\n")
print(kegg_df[kegg_df$p.adjust < 0.05, c("ID", "Description", "GeneRatio", "p.adjust")])

# Check specifically for AD-relevant KEGG pathways
ad_pathways <- c("hsa05010",  # Alzheimer disease
                 "hsa05014",  # Amyotrophic lateral sclerosis
                 "hsa05016",  # Huntington disease
                 "hsa04010",  # MAPK signaling
                 "hsa04151",  # PI3K-Akt signaling  
                 "hsa04210",  # Apoptosis
                 "hsa04668",  # TNF signaling (neuroinflammation)
                 "hsa04064")  # NF-kB signaling

for (pid in ad_pathways) {
  if (pid %in% kegg_df$ID) {
    row <- kegg_df[kegg_df$ID == pid, ]
    cat(sprintf("FOUND: %s — %s (padj=%.4f)\n", pid, row$Description, row$p.adjust))
  }
}

# Dot plot — standard visualization for KEGG enrichment
dotplot(kegg_result,
        showCategory = 20,
        title        = "KEGG Pathway Enrichment — Top Biomarker miRNA Targets",
        font.size    = 10) +
  theme(axis.text.y = element_text(size = 9))

ggsave("results/Week6/kegg_dotplot.png", width = 10, height = 7, dpi = 150)
```

---

### 6.3.5 Expected KEGG Pathways in an AD miRNA Biomarker Study

If your biomarker panel has biological validity, you expect to see enrichment of these KEGG pathways:

| KEGG ID | Pathway Name | Why Expected |
|---------|-------------|--------------|
| **hsa05010** | Alzheimer disease | Direct AD pathway; contains APP, BACE1, PSEN1/2, MAPT, caspases |
| **hsa04010** | MAPK signaling pathway | Tau hyperphosphorylation involves MAP kinases; neuroinflammatory signaling |
| **hsa04151** | PI3K-Akt signaling | Cell survival; inhibition contributes to neuronal apoptosis; PTEN is a target of miR-21 |
| **hsa04210** | Apoptosis | Neuronal apoptosis is a late-stage feature; Bcl-2 family regulated by miR-34a, miR-132 |
| **hsa04064** | NF-κB signaling pathway | Neuroinflammation; microglial activation; regulated by miR-146a, miR-155 |
| **hsa04668** | TNF signaling pathway | Cytokine signaling in AD-associated neuroinflammation |
| **hsa04115** | p53 signaling pathway | DNA damage response; p53 activation in AD neurons; regulated by miR-34a |
| **hsa03030** | DNA replication | Less expected; if enriched, may reflect cell cycle dysregulation in AD |

**Interpretation principle:** The presence of hsa05010 ("Alzheimer disease") is an internal validation — it tells you your target genes are directly annotated to the disease process. The presence of additional pathways (PI3K-Akt, apoptosis, NF-κB) tells you your biomarker panel captures upstream regulatory mechanisms.

> **Biological insight:** The enrichment of neuroinflammation pathways (NF-κB, TNF signaling) is consistent with the emerging understanding that AD is not purely a disease of neurons, but involves extensive glial activation. Blood miRNAs that reflect neuroinflammatory signaling may be released from activated macrophages/monocytes that mirror brain microglial activity. This is one mechanistic hypothesis for why blood miRNA can reflect brain pathology.

---

### 6.3.6 clusterProfiler — enrichGO

Gene Ontology (GO) enrichment provides finer-grained biological process annotations than KEGG pathways.

```r
# GO Biological Process enrichment
go_bp_result <- enrichGO(
  gene          = target_entrez$ENTREZID,
  universe      = background_entrez$ENTREZID,
  OrgDb         = org.Hs.eg.db,
  ont           = "BP",           # BP = Biological Process; also "MF", "CC"
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.2,
  readable      = TRUE,           # convert Entrez IDs back to gene symbols
  minGSSize     = 10,
  maxGSSize     = 500
)

# Simplify redundant GO terms (parent-child GO terms overlap substantially)
go_simplified <- simplify(
  go_bp_result,
  cutoff    = 0.7,    # similarity cutoff; 0.7 = remove terms >70% similar
  by        = "p.adjust",
  select_fun = min   # keep the term with best p.adjust in each cluster
)

cat("GO terms before simplification:", nrow(as.data.frame(go_bp_result)), "\n")
cat("GO terms after simplification:", nrow(as.data.frame(go_simplified)), "\n")

# Bar plot of top 20 GO terms
barplot(go_simplified,
        showCategory = 20,
        title        = "GO Biological Process Enrichment (Simplified)",
        font.size    = 9) +
  theme(axis.text.y = element_text(size = 8))

ggsave("results/Week6/go_bp_barplot.png", width = 10, height = 7, dpi = 150)
```

**Expected GO Biological Process terms for AD miRNA target enrichment:**

- **Negative regulation of apoptotic process** — targets include BCL2, FOXO3a, SIRT1
- **Regulation of synaptic transmission** — reflects targets involved in synaptic plasticity loss
- **Response to oxidative stress** — mitochondrial dysfunction and ROS in AD neurons
- **Regulation of neurogenesis** — adult hippocampal neurogenesis impaired in AD
- **Tau protein binding** — direct link to neurofibrillary tangle biology
- **Amyloid precursor protein processing** — direct link to Aβ production
- **Regulation of NF-κB transcription factor activity** — neuroinflammation
- **Inflammatory response** — broad; consistent with peripheral immune activation in AD

---

### 6.3.7 Multiple Testing in Enrichment Analysis

This deserves special attention. With thousands of GO terms and hundreds of KEGG pathways tested simultaneously, uncorrected p-values are meaningless. Always report:
- **Adjusted p-value (p.adjust)** using Benjamini-Hochberg FDR correction (the default in clusterProfiler)
- The **q-value** (a stricter FDR estimate based on the distribution of all p-values)
- The **GeneRatio** (overlap genes / total query genes) and **BgRatio** (pathway size / background size) to contextualize effect size

A pathway with p.adjust = 0.04 but GeneRatio = 2/300 (2 genes out of 300 drive the enrichment) is a much weaker result than p.adjust = 0.04 with GeneRatio = 30/300.

---

## MODULE 6.5 — Integrating Multi-Omics Context

### 6.5.1 How Blood miRNA Biomarkers Relate to Brain Transcriptomic Changes

A fundamental question about blood miRNA biomarkers for a brain disease: why does the blood reflect what is happening in the brain? There are several non-exclusive mechanistic hypotheses:

1. **Neuronal injury-derived release:** Damaged neurons in the AD brain release intracellular contents, including miRNAs, into the cerebrospinal fluid (CSF) and eventually into the bloodstream. This is analogous to cardiac troponin release after myocardial injury.

2. **Peripheral immune cell response:** Microglia in the brain and monocytes/macrophages in the blood are functionally related cells of the myeloid lineage. Neuroinflammatory signals in the AD brain may program circulating monocytes to adopt altered miRNA expression profiles that mirror brain microglial states.

3. **Exosomal transfer:** Neurons, astrocytes, and microglia secrete exosomes (small extracellular vesicles) containing miRNAs. These can cross the blood-brain barrier or reach systemic circulation through the glymphatic system, where their miRNA content reflects the cell type and biological state of origin.

4. **Systemic metabolic alterations:** AD is associated with systemic metabolic changes (insulin resistance, mitochondrial dysfunction, oxidative stress) that affect peripheral tissues and may alter blood miRNA profiles independently of direct brain-to-blood communication.

**Cross-referencing with the AMP-AD Transcriptome:**

The Accelerating Medicines Partnership for Alzheimer's Disease (AMP-AD) consortium has generated large-scale multi-region brain transcriptomics data for hundreds of AD and control subjects (available on Synapse.org: syn2580853). You can ask: are the target genes of your blood miRNA panel also differentially expressed in AMP-AD brain tissue?

This cross-referencing strengthens the biological coherence of your biomarker: if miR-132 is downregulated in blood AND its target FOXO3a is upregulated in AMP-AD temporal cortex, the two datasets tell a consistent biological story.

---

### 6.5.2 Cross-Referencing with GWAS Hits

The GWAS catalog (ebi.ac.uk/gwas) and the AD GWAS literature provide a list of genetic loci associated with AD risk. Key replicated GWAS loci include: APOE, BIN1, CLU, PICALM, CR1, ABCA7, SORL1, PTK2B, SPI1, PLCG2, TREM2.

For each of your top biomarker miRNAs' target genes, check whether they overlap with GWAS-implicated genes. Overlap indicates that your miRNA panel captures variation in a genomic region already known to modify AD risk — a strong mechanistic argument.

```r
# GWAS overlap check
gwas_ad_genes <- c("APOE", "BIN1", "CLU", "PICALM", "CR1", "ABCA7", 
                   "SORL1", "PTK2B", "SPI1", "PLCG2", "TREM2", "FERMT2",
                   "CASS4", "INPP5D", "MEF2C", "HLA-DRB1", "ZCWPW1",
                   "CELF1", "NME8", "TRIP4")

gwas_overlap <- intersect(unique(strong_evidence$target.symbol), gwas_ad_genes)
cat("Target genes overlapping with AD GWAS hits:\n")
print(gwas_overlap)

# Show which miRNA targets which GWAS gene
gwas_detail <- strong_evidence %>%
  filter(target.symbol %in% gwas_ad_genes) %>%
  select(mature.mirna, target.symbol, experiment)

print(gwas_detail)
```

---

### 6.5.3 The miRNA-eQTL Concept

An **expression quantitative trait locus (eQTL)** is a genetic variant (SNP) that affects the expression level of a gene. A **miRNA-eQTL** is a genetic variant that affects either:
- The expression level of a miRNA (a **miR-eQTL**)
- The expression of a target gene in a miRNA-dependent manner (a **trans-eQTL** mediated by a miRNA)

The relevance to biomarker development: if a GWAS-identified AD risk SNP is also a miRNA-eQTL that affects one of your biomarker miRNAs, that provides a direct genetic link between the risk variant, the miRNA level, and the disease. This level of evidence — genetic causality — is the gold standard in biomarker biology.

**Practical resource:** The blood eQTL dataset from GTEx (v8; gtexportal.org) includes miRNA expression quantitative trait analyses for multiple blood and brain tissues. The BRAINEAC database provides brain-specific eQTL data.

---

### 6.5.4 Literature Triangulation — Building Confidence in a Biomarker

Before reporting a biomarker, systematically search PubMed for each candidate miRNA:

```
Search strategy for miR-132 as an example:
  PubMed: ("hsa-miR-132" OR "miR-132-3p") AND ("Alzheimer" OR "dementia")
  Focus on: Independent replication studies (not from your dataset's original paper)
  Key metrics: Same direction of change? Similar fold change? Same sample type?
```

Build a replication table:

| miRNA | Your Dataset | Literature Study 1 | Literature Study 2 | Replication Score |
|-------|-------------|-------------------|-------------------|-------------------|
| miR-132-3p | Downregulated, FC=−2.1 | Hernandez-Rapp et al. 2016: down | Fransquet et al. 2018: down | Strong |
| miR-146a-5p | Upregulated, FC=+1.8 | Lukiw et al. 2010: up in brain | Mitchell et al. 2010: up in blood | Moderate |

miRNAs with consistent direction and magnitude across multiple independent studies are the strongest candidates.

---

## MODULE 6.6 — Biomarker Validation Roadmap

### 6.6.1 The Five Stages of Biomarker Development

The FDA-NIH Biomarker Working Group (BEST Glossary, 2016) defines a structured progression for biomarker development from initial discovery to clinical use:

**Stage 1 — Discovery**
- What happens: High-throughput profiling (microarray, RNA-seq) on case-control samples; statistical methods identify candidate biomarkers
- Data: Retrospective, cross-sectional
- Output: A ranked list of candidate biomarker molecules
- Limitation: High false discovery rate; many candidates will not replicate
- Where we are in this course: **End of Week 5 / Beginning of Week 6**

**Stage 2 — Qualification**
- What happens: Analytical characterization of measurement; initial replication in one or more independent cohorts using different technology (e.g., qPCR after miRNA-seq discovery)
- Key questions: Does the biomarker measure what it claims to? Does it replicate in a different population?
- Output: Narrowed panel (3–15 miRNAs); qPCR assay developed
- Typical timeline: 2–3 years; 2–5 independent studies

**Stage 3 — Verification**
- What happens: Blinded testing in prospectively collected samples with standardized pre-analytical procedures; rigorous comparison against established clinical measures (MMSE, amyloid PET)
- Key questions: What is the sensitivity and specificity against the clinical gold standard? What is the optimal cutoff?
- Output: ROC curve in prospective cohort; sensitivity/specificity at operating point
- Typical timeline: 3–5 years; requires large prospective biobank

**Stage 4 — Clinical Validation**
- What happens: Multi-site prospective clinical trial; biomarker tested as intended for clinical use; comparison against clinical endpoints (progression to AD dementia)
- Key questions: Does the biomarker predict clinical outcomes? Does it add value beyond existing clinical tools?
- CLIA requirements apply if performed in clinical laboratory
- Output: FDA/regulatory submission data package
- Typical timeline: 5–10 years

**Stage 5 — Clinical Implementation**
- What happens: Regulatory approval (FDA LDT, FDA PMA for IVD); reimbursement coverage (CMS); clinical guideline inclusion; widespread laboratory adoption
- Output: Commercially available clinical test
- Typical timeline: 3–5 years post-validation

**Critical insight for aspiring biomarker researchers:** Most biomarkers discovered in Stage 1 never reach Stage 5. The primary reasons for failure are: (1) lack of replication in independent prospective cohorts, (2) insufficient analytical performance (too much CV, too high LOD), and (3) inadequate specificity (biomarker elevated in multiple diseases, not specific to AD). Understanding these failure modes is essential for designing studies that have a chance of succeeding.

---

### 6.6.2 Analytical Validation Parameters

Before clinical validation, the measurement assay must be analytically characterized:

**Precision (reproducibility):**
- **Within-run precision (repeatability):** CV (coefficient of variation) of repeated measurements of the same sample in the same run. Target: CV < 10% for plasma miRNA by qPCR.
- **Between-run precision (intermediate precision):** CV across different runs, operators, and days. Target: CV < 15%.
- **Between-laboratory precision:** CV across different laboratories. Critical for multi-site studies.

**Accuracy:**
- Recovery of spiked-in synthetic miRNA at known concentrations
- Typically assessed at 3 concentration levels: low, mid, high

**Limit of Detection (LOD):**
- The lowest concentration at which the signal is reliably distinguished from background noise
- For miRNA qPCR: typically expressed as number of copies per volume (e.g., copies/µL serum) or Ct threshold
- LOD = mean(blank) + 3 × SD(blank)

**Limit of Quantification (LOQ):**
- The lowest concentration at which measurements meet precision criteria (CV < 20%)
- LOQ > LOD; always specify both
- All samples with concentrations below the LOQ should be reported as "below LOQ," not as the measured value

**Linearity:**
- Serial dilution of a high-concentration sample should produce proportional decreases in signal
- Assessed over the expected clinical concentration range

**Specificity:**
- Demonstrate that the assay detects the intended miRNA and not closely related family members (miR-21-3p vs miR-21-5p; miR-29a vs miR-29b)
- Mature miRNA sequences differ by only 1–3 nucleotides within families — primer design is critical

---

### 6.6.3 Clinical Validation Parameters

**Sensitivity and Specificity:**
- **Sensitivity (true positive rate):** Proportion of AD patients correctly identified as positive. For a screening test, sensitivity should be high (ideally > 90%).
- **Specificity (true negative rate):** Proportion of controls correctly identified as negative. For a confirmatory test, specificity should be high.
- These are inversely related — setting a threshold changes both. ROC curves visualize this tradeoff across all possible thresholds.

**ROC Analysis in an Independent Prospective Cohort:**
- The ROC curve and AUC must be reported from an independent cohort not used in biomarker discovery or model training
- This is the external validation dataset (GSE46579 in our course; a truly prospective validation would require a new biobank study)
- Report: AUC with 95% CI (ideally from bootstrap); optimal sensitivity/specificity at Youden index

**Comparison to Clinical Gold Standard:**
- Compare biomarker performance to existing clinical measures: MMSE (global cognition), amyloid PET, CSF Aβ42/tau
- For a blood test to have clinical utility, it must either (a) match or exceed existing tests, or (b) provide complementary information at lower cost/invasiveness

---

### 6.6.4 CLIA Requirements

If a miRNA biomarker test is to be used in patient care, the laboratory performing it must comply with the **Clinical Laboratory Improvement Amendments (CLIA)** regulations, which establish minimum standards for accuracy, reliability, and timeliness of laboratory testing in the United States.

Key CLIA requirements for a laboratory-developed test (LDT):
- Laboratory must hold a CLIA certificate appropriate for test complexity (high-complexity for molecular diagnostics)
- Validation studies must demonstrate: accuracy, precision, analytical sensitivity, analytical specificity, reportable range, reference range
- Proficiency testing (external quality assurance) must be performed at least twice per year
- Personnel qualifications are specified (laboratory director, testing personnel, supervisors)

**Practical implication:** A research laboratory in a university cannot simply run a miRNA qPCR assay on patient samples and report results for clinical decision-making, even if the assay has been extensively validated in research settings. A CLIA-certified clinical laboratory must be involved.

---

## MODULE 6.7 — Regulatory and Ethical Considerations

### 6.7.1 FDA Regulatory Pathways for Molecular Diagnostics

**Laboratory-Developed Test (LDT) Pathway:**
An LDT is a test developed, validated, and used within a single laboratory. Until recently, LDTs were regulated primarily by CLIA and were not subject to FDA premarket review. The FDA's final rule on LDTs (published 2024) phases in FDA oversight for LDTs over several years, requiring high-risk LDTs (including those for serious and life-threatening conditions like AD) to follow medical device regulations.

For an AD miRNA blood test:
- **High-risk classification** (Class III device): Tests intended to diagnose AD are likely to be classified as high-risk because the result directly affects patient management and there are few alternatives
- **PMA (Premarket Approval):** Class III devices require a PMA application demonstrating safety and effectiveness through valid clinical evidence
- **De Novo pathway:** If a predicate device exists (another approved IVD in the same space), a 510(k) submission may be possible; for truly novel biomarkers, De Novo classification creates a new device category

**IVD (In Vitro Diagnostic) Pathway:**
An IVD is a medical device intended for use in diagnosis of disease in humans. FDA IVD approval requires:
- Analytical validation data
- Clinical validation data from prospective studies
- Labeling that accurately describes intended use, performance characteristics, and limitations

**Practical insight:** The FDA has approved several blood-based AD biomarker tests in recent years, including tests for amyloid beta ratios. The precedent is being established, but the evidentiary bar is high.

---

### 6.7.2 CE-IVD Marking in Europe

In the European Union, in vitro diagnostic devices are regulated under the **IVDR (In Vitro Diagnostic Regulation, EU 2017/746)**, which came into force in 2022 to replace the older IVDD. The CE marking indicates that the device meets EU safety and performance requirements.

Under IVDR, AD diagnostics would fall into **Class D** (highest risk; poses high individual and public health risk) or **Class C**, depending on the intended use and the state of the art. Class D devices require involvement of an EU Notified Body for conformity assessment.

Key differences from FDA:
- CE marking is a conformity assessment, not a product-by-product approval
- Clinical evidence requirements under IVDR are stricter than under the old IVDD
- Performance studies must include a Post-Market Performance Follow-Up plan

---

### 6.7.3 Ethical Issues: AD Diagnosis Without Treatment

A profound ethical dimension of AD biomarker research that is absent from most computational papers but must be confronted in clinical translation:

**The disclosure dilemma:** A blood test that can predict AD 10–15 years before symptom onset, if it were validated, would allow early identification of at-risk individuals. But currently, there is no disease-modifying therapy that can halt or reverse AD progression at the preclinical stage. Lecanemab and donanemab slow progression in early symptomatic AD; they have not been proven to prevent AD in presymptomatic individuals. Disclosing a positive biomarker result to a cognitively normal individual could cause severe psychological harm — anxiety, depression, changes in employment, insurance, and relationships — without offering a therapeutic benefit.

This is not a hypothetical concern. Studies with participants in prevention trials (e.g., the DIAN observational study) have documented the psychological impact of disclosing amyloid status.

**Ethical frameworks for disclosure:**
- **Autonomy-based:** Individuals have the right to know their health status, including biomarker results, if they have provided informed consent for disclosure
- **Beneficence-based:** Results should only be disclosed when clinical benefit (or benefit to decision-making for life planning) outweighs harm
- **Justice-based:** Disclosure practices should be equitable; wealthy individuals should not have preferred access to early diagnostic information

**Current consensus (as of 2025):** The Alzheimer's Association and the Global Alzheimer's Association Roundtable recommend that presymptomatic AD biomarker disclosure should occur within a structured support framework including genetic counseling, neuropsychological support, and clear communication of the biomarker's predictive limitations.

---

### 6.7.4 Genetic Privacy and Insurance Discrimination

Genetic information protection laws (GINA in the US; GDPR Article 9 in the EU) provide some protections, but blood-based molecular biomarkers like miRNA levels fall into a regulatory gray zone — they are not genetic information in the traditional sense, but they may be predictive of future disease with similar implications.

Long-term care insurance and life insurance remain unprotected under GINA in the US. An individual who discloses a positive AD miRNA biomarker result could potentially face premium increases or coverage denial.

This concern affects study design: informed consent for biomarker studies must explicitly address what happens to results and whether they will be disclosed to participants, insurers, or employers.

---

### 6.7.5 Equitable Access to Diagnostics

Even a perfectly validated AD blood test could exacerbate health disparities if:
- The test cost is prohibitive (miRNA sequencing-based tests currently cost $200–$500/sample)
- Insurance coverage is variable or absent
- The test was validated exclusively in non-Hispanic white European-ancestry cohorts (common in AD biomarker literature) and performs differently in African American, Hispanic, or Asian populations

**ADNI (Alzheimer's Disease Neuroimaging Initiative)**, the largest AD longitudinal cohort used for biomarker validation, was historically predominantly white. AMP-AD has made diversity a priority, but the field has significant ground to cover.

For your research proposal (Module 6.8), address diversity in the study design — this is increasingly required by NIA and the Alzheimer's Association.

---

### 6.7.6 Consent for Data Use in ML Studies and GDPR

When building ML models on GEO data, you are working with data that was collected from human participants under specific informed consent agreements. Key considerations:

- **GEO data re-use:** Most GEO datasets were deposited under consent forms that permit broad re-use for research. However, if you plan to integrate clinical outcomes data (e.g., from a hospital biobank), separate data sharing agreements may be required.
- **GDPR (EU):** Clinical data from EU participants cannot be transferred to non-EU servers without adequate safeguards. If your GEO dataset contains European patient data, this applies even for academic research.
- **Federated learning:** An emerging approach that keeps patient data within local clinical sites and only shares model gradients or trained model parameters. Relevant for multi-institutional AD biomarker studies.
- **De-identification:** Raw miRNA expression values from an individual, combined with age and sex, may be sufficient to re-identify patients. Data de-identification and access controls are essential.

---

## MODULE 6.8 — Course Synthesis: Building Your Research Proposal

### 6.8.1 The One-Page Research Proposal Framework

Translating your course work into a research proposal is a practical goal of this module. A one-page summary research proposal (which you can expand into a full grant application) has five components:

**1. Title (1 sentence)**
Specific, informative. Example: "A circulating miRNA panel for early detection of Alzheimer's disease: discovery, validation, and mechanistic characterization"

**2. Background and Significance (2–3 sentences)**
What is the clinical problem? Why is current diagnostics inadequate? What is the opportunity?

Example: "Alzheimer's disease affects over 55 million people worldwide, yet current blood-based biomarkers (Aβ42/40 ratio, phospho-tau217) have not been integrated into routine clinical practice. Non-invasive, inexpensive circulating miRNA biomarkers offer a complementary approach. Preliminary evidence suggests that a panel of 10–15 blood miRNAs can distinguish AD patients from cognitively normal controls with AUC > 0.85, but prospective validation and mechanistic characterization remain incomplete."

**3. Innovation (1–2 sentences)**
What is new about your approach? How does it advance the field?

Example: "This study is innovative in (1) applying SHAP-explainable random forest models to harmonized multi-cohort miRNA data to identify a mechanistically grounded biomarker panel, and (2) prospectively validating the panel in a diverse cohort that includes participants of African American and Hispanic ancestry."

**4. Specific Aims (2–3 bullet points)**
Concrete, testable, achievable within the grant period. Each aim should have a clear deliverable.

**Aim 1:** Identify and computationally validate a panel of ≤15 blood miRNA biomarkers for AD using integrative analysis of 5 public GEO datasets (n > 500 total) and SHAP-based feature importance.
**Aim 2:** Verify the miRNA panel in a prospective cohort of 200 AD patients and 200 age/sex-matched controls using droplet digital PCR (ddPCR), with pre-specified sensitivity and specificity targets.
**Aim 3:** Elucidate the mechanistic basis of the top biomarker miRNAs using target prediction, pathway enrichment, and correlation with AMP-AD brain transcriptomics data.

**5. Expected Outcomes and Impact (1–2 sentences)**
What will you have at the end? Why does it matter?

---

### 6.8.2 Relevant Funding Sources

**National Institute on Aging (NIA) — part of NIH:**
- **R01:** Standard research project grant; 4–5 years; up to $500K direct costs/year
  - Relevant program announcement: PAR-22-126 "Biomarker Development for Alzheimer's Disease and Alzheimer's Disease-Related Dementias (AD/ADRD)"
- **R21:** Exploratory/developmental research; 2 years; up to $275K total direct costs
  - Good for a first computational biomarker validation study before committing to a prospective cohort
- **K99/R00:** Career development award for postdocs transitioning to independent faculty positions

**Alzheimer's Association:**
- **Research Grant (RG):** Up to $150K over 3 years; competitive; requires preliminary data
- **New Investigator Research Grant (NIRG):** For investigators early in their career
- **Part the Cloud Translational Research Funding:** For studies explicitly focused on clinical translation (Stage 2–3 on the biomarker development roadmap)

**Michael J. Fox Foundation / Chan Zuckerberg Initiative:**
- Less relevant for AD but useful if your work has implications for other neurodegenerative diseases

**Key sections in an NIH R01 grant:**
- **Specific Aims (1 page):** This is what reviewers read first and remember longest
- **Significance:** Why does this matter? What gap does it fill?
- **Innovation:** What is genuinely new?
- **Approach:** Detailed experimental plan; preliminary data; power calculation
- **Human Subjects / Data Management and Sharing Plan**

---

## WEEK 6 LAB SESSION

### Lab 6A — Target Prediction and Pathway Enrichment in R (75 min)

**Objective:** Starting from the SHAP-ranked miRNA list generated in Week 5, identify target genes and perform pathway enrichment analysis.

**Step 1 (15 min):** Load Week 5 output files.
- Load `results/Week5/shap_feature_importance.csv` (or equivalent Python output)
- Load `results/Week4/DE_results_GSE120584.csv`
- Create a composite ranking (average of SHAP rank and DE significance rank)
- Select top 15 miRNAs

**Step 2 (20 min):** multiMiR target query.
- Query multiMiR for all 15 miRNAs with `table = "validated"`
- Filter to strong evidence only
- Check for known AD genes in the target list (APP, BACE1, MAPT, SIRT1, FOXO3, CDK5)
- Print: how many validated targets were found? How many are AD-relevant?

**Step 3 (20 min):** KEGG enrichment.
- Build background gene universe (all expressed miRNA targets)
- Run `enrichKEGG` with BH correction
- Make dotplot of top 20 KEGG pathways
- Identify whether hsa05010 (Alzheimer disease) is enriched

**Step 4 (20 min):** GO enrichment.
- Run `enrichGO` on BP ontology
- Apply `simplify()` to remove redundant terms
- Make barplot of top 20 GO terms
- Write 3 sentences interpreting the top 5 GO terms in AD biology context

**Deliverables:**
- `results/Week6/kegg_dotplot.png`
- `results/Week6/go_bp_barplot.png`
- `results/Week6/validated_targets.csv`
- Written interpretation: 1 paragraph describing what the enrichment results say about the biology of your biomarker miRNAs


## FINAL COURSE ASSIGNMENTS

### Assignment 6.1 — Literature Replication Table (Individual)

For each of the top 5 miRNAs from your biomarker panel, conduct a systematic PubMed literature search and complete the following table:

| miRNA | Direction in Your Study | Study 1 (citation, direction, sample type) | Study 2 (citation, direction) | Consistent? | Notes |
|-------|------------------------|-------------------------------------------|------------------------------|-------------|-------|
| ... | ... | ... | ... | ... | ... |

**Search strategy:**
- PubMed: ("hsa-miR-XXX" OR "miR-XXX-3p") AND ("Alzheimer" OR "Alzheimer's disease") AND ("biomarker" OR "blood" OR "serum" OR "plasma")
- Limit to: human studies; English language; 2010–present
- Include studies using any sample type (blood, CSF, brain)

Submit as a 1-page table with a brief (1-paragraph) conclusion about the replication status of your panel.

**Due:** End of Week 6

---

### Assignment 6.2 — Pathway Enrichment Interpretation (Individual)

After running enrichKEGG and enrichGO in Lab 6A, write a structured biological interpretation report (500–800 words) covering:

1. Which KEGG pathways were significantly enriched (list with adjusted p-values)?
2. Do the enriched pathways align with current understanding of AD pathomechanisms? Explain why each enriched pathway is or is not biologically plausible.
3. Which GO Biological Process terms were most enriched? What do they tell you about the regulatory functions of your biomarker miRNAs?
4. Were there any unexpected pathways? If so, propose a hypothesis for why they appeared.
5. What are the limitations of this enrichment analysis, and how might they affect your interpretation?

**Due:** End of Week 6

---

### Capstone Project — "Build and Present a Complete miRNA Biomarker Discovery Pipeline"

This is the culminating assignment of the course, designed to integrate all methods from Weeks 1–6.

**Overview:**
Working individually or in pairs, you will build and present a complete miRNA biomarker discovery pipeline applied to a real GEO dataset of your choice (not GSE120584 or GSE46579, which we used in class). The final product is a 15-minute oral presentation plus a written report.

**Dataset requirements:**
- Any miRNA profiling dataset from GEO with blood-derived samples
- Disease: AD, Parkinson's disease, mild cognitive impairment, or any neurodegenerative disease
- Minimum N: 20 per group; must include a disease group and a healthy control group
- Must be a dataset not used in any of the class lectures or labs

**Required pipeline components:**

| Week | Component | Required Deliverable |
|------|-----------|---------------------|
| Week 2 | Data acquisition & QC | Normalized expression matrix; QC report with ≥ 3 QC plots; documented exclusions |
| Week 3 | Exploratory analysis | PCA plot (colored by group); unsupervised clustering heatmap; 1-paragraph interpretation |
| Week 4 | Differential expression | Volcano plot; DE table with adjusted p-values; top 20 DE miRNAs |
| Week 5 | ML classification | ROC curve; SHAP summary plot; performance metrics (AUC, sensitivity, specificity) |
| Week 6 | Biological interpretation | Target gene table; KEGG dotplot; GO barplot; biomarker panel summary figure |

**Written report (10–15 pages, single-spaced, 11pt):**
1. Introduction (1–2 pages): Biological background of disease and miRNAs; rationale for the study
2. Methods (3–4 pages): Detailed description of each pipeline step; R and Python packages used; parameter choices justified
3. Results (4–6 pages): All required deliverables with biological interpretation
4. Discussion (2–3 pages): Limitations; comparison to existing literature; biomarker validation roadmap (what experiments would you do next?)
5. References (minimum 15 citations; all peer-reviewed)

**Oral presentation (15 min + 5 min Q&A):**
- Slide 1: Title, dataset, biological question
- Slides 2–3: Background and rationale
- Slides 4–6: Methods overview (one slide per week's key methods)
- Slides 7–10: Results (one slide per week's key deliverable)
- Slides 11–12: Biological interpretation and discussion
- Slide 13: Proposed validation roadmap (what would you do next?)
- Slide 14: Limitations and conclusions

**Grading rubric:**
- Scientific quality of pipeline (reproducible R code; correct statistical methods): 40%
- Biological interpretation (mechanistic grounding; use of target databases and enrichment): 30%
- Presentation clarity and scientific communication: 20%
- Literature engagement (appropriate citations; correct characterization of prior work): 10%

**Due:** 2 weeks after Week 6 session

---

## WEEK 6 GLOSSARY

| Term | Definition |
|------|------------|
| **Over-Representation Analysis (ORA)** | Statistical test (Fisher's exact) determining whether a predefined gene set (pathway or GO term) is enriched in a query gene list relative to a background |
| **GSEA** | Gene Set Enrichment Analysis; ranks all genes by a continuous score and tests whether pathway genes cluster at the extremes of the ranked list; no hard threshold required |
| **FDR** | False Discovery Rate; the expected proportion of significant test results that are false positives; controlled using Benjamini-Hochberg procedure in enrichment analysis |
| **GeneRatio** | In enrichment analysis, the fraction of query genes belonging to a pathway; numerator/denominator format (e.g., 12/300) |
| **Hub gene** | A highly connected node in a protein-protein interaction network; high degree centrality indicates many direct interaction partners |
| **Degree centrality** | Network metric; number of direct edges (interactions) a node has; simple measure of network importance |
| **Betweenness centrality** | Network metric; proportion of shortest paths between all node pairs that pass through a given node; identifies bridges between network modules |
| **miRTarBase** | Database of experimentally validated miRNA–target interactions with evidence type curated from the published literature (Huang et al., 2022) |
| **TargetScan** | Computational miRNA target prediction tool using seed sequence complementarity and conservation; provides context++ score |
| **STRING** | Search Tool for Retrieval of Interacting Genes/Proteins; integrates experimental, computational, and text-mining evidence for protein-protein interactions |
| **multiMiR** | R package providing a unified interface to query 14 miRNA-target databases simultaneously |
| **eQTL** | Expression Quantitative Trait Locus; a genomic variant that affects the expression level of a gene; miRNA-eQTLs link genetic risk to miRNA regulation |
| **LDT** | Laboratory-Developed Test; a diagnostic test developed and used within a single clinical laboratory; subject to CLIA and (increasingly) FDA oversight |
| **IVD** | In Vitro Diagnostic; a device used to perform tests on samples taken from the human body; regulated by FDA (US) and IVDR (EU) |
| **CLIA** | Clinical Laboratory Improvement Amendments; US federal standards for laboratory testing that ensure accuracy, reliability, and timeliness of patient testing |
| **CE-IVD** | Conformité Européenne marking for in vitro diagnostic devices; indicates compliance with EU IVDR requirements |
| **LOD** | Limit of Detection; lowest analyte concentration distinguishable from blank signal; LOD = mean(blank) + 3 × SD(blank) |
| **LOQ** | Limit of Quantification; lowest concentration at which measurement meets precision criteria (CV < 20%); always ≥ LOD |
| **BEST Glossary** | Biomarkers, EndpointS, and other Tools; FDA-NIH resource defining standardized terminology for biomarkers in drug development (Biomarker Working Group, 2016) |
| **SHAP** | SHapley Additive exPlanations; game-theoretic framework for explaining individual ML model predictions; used in Week 5 to rank miRNA feature importance |
| **AMP-AD** | Accelerating Medicines Partnership for Alzheimer's Disease; NIH-industry consortium providing large-scale multi-omics data from AD brain tissue (Synapse.org) |
| **Analytical validation** | Confirmation that an assay measures what it claims, at the required precision and accuracy, within its stated operating range |
| **Clinical validation** | Demonstration that an assay correctly identifies or predicts a clinical condition in the intended use population |

---

## KEY REFERENCES (Week 6)

All references retrieved from PubMed or official regulatory sources.

1. **Huang H-Y et al. (2022).** miRTarBase 2022: an informatics resource for experimentally validated miRNA–target interactions. *Nucleic Acids Res* 50(D1):D222–D230. [DOI: 10.1093/nar/gkab1079](https://doi.org/10.1093/nar/gkab1079) — *miRTarBase experimental validation database*

2. **Agarwal V et al. (2015).** Predicting effective microRNA target sites in mammalian mRNAs. *eLife* 4:e05005. [DOI: 10.7554/eLife.05005](https://doi.org/10.7554/eLife.05005) — *TargetScan context++ score; gold-standard computational target prediction*

3. **Szklarczyk D et al. (2023).** The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any of 12535 organisms. *Nucleic Acids Res* 51(D1):D638–D646. [DOI: 10.1093/nar/gkac1000](https://doi.org/10.1093/nar/gkac1000) — *STRING PPI database; version 12.0*

4. **Yu G et al. (2012).** clusterProfiler: an R Package for Comparing Biological Themes Among Gene Clusters. *OMICS* 16(5):284–287. [DOI: 10.1089/omi.2011.0118](https://doi.org/10.1089/omi.2011.0118) — *clusterProfiler; enrichKEGG, enrichGO, GSEA*

5. **Collins GS et al. (2015).** Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ* 350:g7594. [DOI: 10.1136/bmj.g7594](https://doi.org/10.1136/bmj.g7594) — *Reporting standards for prediction model studies; essential for publication*

6. **Biomarker Working Group (2016).** BEST (Biomarkers, EndpointS, and other Tools) Resource. FDA-NIH Joint Leadership Council. Available at: [https://www.ncbi.nlm.nih.gov/books/NBK326791/](https://www.ncbi.nlm.nih.gov/books/NBK326791/) — *Definitive glossary of biomarker terminology; Stage 1–5 biomarker development framework*

7. **Sticht C, De La Torre C, Parveen A, Gretz N (2018).** miRWalk: An online resource for prediction of microRNA binding sites. *PLoS ONE* 13(10):e0205239. [DOI: 10.1371/journal.pone.0205239](https://doi.org/10.1371/journal.pone.0205239) — *miRWalk database for 3'UTR/CDS/5'UTR target prediction*

8. **Hébert SS et al. (2008).** Loss of microRNA cluster miR-29a/b-1 in sporadic Alzheimer's disease correlates with increased BACE1/β-secretase expression. *Proc Natl Acad Sci USA* 105(17):6415–6420. [DOI: 10.1073/pnas.0710263105](https://doi.org/10.1073/pnas.0710263105) — *Landmark study linking miR-29 loss to BACE1 upregulation in AD*

9. **Hernandez-Rapp J et al. (2016).** microRNA-132/212 deficiency enhances Aβ production and senile plaque deposition in Alzheimer's disease triple transgenic mice. *Sci Rep* 6:30953. [DOI: 10.1038/srep30953](https://doi.org/10.1038/srep30953) — *Functional validation of miR-132 in AD animal model*

10. **Fransquet PD et al. (2018).** Blood-based small non-coding RNA as potential biomarkers of late-onset Alzheimer's disease. *J Alzheimers Dis* 66(4):1479–1496. [DOI: 10.3233/JAD-180562](https://doi.org/10.3233/JAD-180562) — *Systematic review of blood miRNA biomarkers in AD; miR-132 replication*

11. **FDA (2024).** Laboratory Developed Tests; Final Rule. *Federal Register* 89(82):37286. — *FDA regulatory framework for LDTs; phase-in schedule for enforcement*

12. **Ru Y et al. (2014).** The multiMiR R package and database: integration of microRNA–target interactions along with their disease and drug associations. *Nucleic Acids Res* 42(17):e133. [DOI: 10.1093/nar/gku631](https://doi.org/10.1093/nar/gku631) — *multiMiR R package; 14-database integration*

13. **Mattsson N et al. (2020).** Plasma tau in Alzheimer disease. *Neurology* 87(17):1827–1835. [DOI: 10.1212/WNL.0000000000006359](https://doi.org/10.1212/WNL.0000000000006359) — *Context for blood-based AD biomarker validation standards*

14. **Golde TE, Schneider LS, Koo EH (2011).** Anti-aβ therapeutics in Alzheimer's disease: the need for a paradigm shift. *Neuron* 69(2):203–213. [DOI: 10.1016/j.neuron.2011.01.002](https://doi.org/10.1016/j.neuron.2011.01.002) — *Biological context: why early detection matters; therapeutic window concept*

15. **Petersen RC et al. (2018).** Practice guideline update summary: Mild cognitive impairment. *Neurology* 90(3):126–135. [DOI: 10.1212/WNL.0000000000004826](https://doi.org/10.1212/WNL.0000000000004826) — *Clinical context for MCI classification; relevant for biomarker endpoint definition*

---

## APPENDIX: Complete Pipeline Summary — Weeks 1–6

*A reference table of every tool, package, dataset, and output from the entire course.*

| Week | Stage | Tool / Package | Input | Output | Location |
|------|-------|---------------|-------|--------|----------|
| 1 | Setup | R ≥ 4.3, Bioconductor 3.18 | — | Configured environment | Local |
| 1 | Setup | Python 3.10, conda | — | conda environment `ml_biomarker` | Local |
| 1 | Setup | RStudio | — | Project structure created | Local |
| 2 | Data acquisition | `GEOquery::getGEO()` | GSE120584 accession | ExpressionSet object | `data/raw/` |
| 2 | Data acquisition | `GEOquery::getGEOSuppFiles()` | GSE120584 | Count matrix (`.txt.gz`) | `data/raw/GSE120584/` |
| 2 | Data acquisition | `GEOquery::getGEO()` | GSE46579 accession | ExpressionSet + CEL files | `data/raw/GSE46579/` |
| 2 | QC | `edgeR::filterByExpr()` | Raw count matrix | Filtered count matrix | In memory |
| 2 | Normalization | `DESeq2::vst()` | Count matrix | VST expression matrix | `data/processed/GSE120584_expr_clean.rds` |
| 2 | Normalization | `edgeR::calcNormFactors()` | Count matrix | TMM-normalized CPM | `data/processed/` |
| 2 | Normalization | `oligo::rma()` | CEL files | RMA expression matrix | `data/processed/GSE46579_expr_rma.rds` |
| 2 | QC | `pheatmap::pheatmap()` | Correlation matrix | Heatmap PNG | `qc_reports/` |
| 2 | Batch correction | `sva::ComBat()` | VST matrix | Batch-corrected matrix | `data/processed/` |
| 2 | Output | `saveRDS()` | Clean matrices + metadata | `.rds` files | `data/processed/` |
| 3 | EDA | `stats::prcomp()` | VST matrix | PCA scores | In memory |
| 3 | EDA | `Rtsne::Rtsne()` | VST matrix | t-SNE embedding | In memory |
| 3 | EDA | `umap::umap()` | VST matrix | UMAP embedding | In memory |
| 3 | Clustering | `pheatmap::pheatmap()` | Top variable miRNAs | Clustered heatmap PNG | `results/Week3/` |
| 3 | Clustering | `cluster::pam()` | Distance matrix | Cluster assignments | `results/Week3/` |
| 3 | Output | `ggplot2::ggsave()` | PCA/t-SNE/UMAP plots | PNG files | `results/Week3/` |
| 4 | DE | `DESeq2::DESeq()` | Count matrix + design | DE statistics | In memory |
| 4 | DE | `limma::voom() + lmFit()` | CPM + design | DE statistics (microarray) | In memory |
| 4 | Visualization | `ggplot2` (volcano plot) | DE results | Volcano plot PNG | `results/Week4/` |
| 4 | Output | `write.csv()` | DE results table | `DE_results_GSE120584.csv` | `results/Week4/` |
| 4 | Output | `write.csv()` | DE results table | `DE_results_GSE46579.csv` | `results/Week4/` |
| 5 | Harmonization | `sva::ComBat()` | Both VST matrices | Harmonized matrix | `data/processed/harmonized_expr.rds` |
| 5 | ML | `caret::train()` / sklearn | Harmonized matrix | RF/LR model | `results/Week5/` |
| 5 | ML | SHAP (shapr / Python shap) | Trained model | SHAP values per miRNA | `results/Week5/shap_feature_importance.csv` |
| 5 | Validation | `pROC::roc()` | Model predictions | ROC curve, AUC | `results/Week5/` |
| 5 | Output | `ggplot2::ggsave()` | ROC + SHAP plots | PNG files | `results/Week5/` |
| 6 | Target prediction | `multiMiR::multiMiR()` | Top 15 miRNA list | Validated target table | `results/Week6/validated_targets.csv` |
| 6 | Enrichment | `clusterProfiler::enrichKEGG()` | Target Entrez IDs | KEGG result object | In memory |
| 6 | Enrichment | `clusterProfiler::enrichGO()` | Target Entrez IDs | GO-BP result object | In memory |
| 6 | Visualization | `clusterProfiler::dotplot()` | KEGG result | Dotplot PNG | `results/Week6/kegg_dotplot.png` |
| 6 | Visualization | `clusterProfiler::barplot()` | GO result | Barplot PNG | `results/Week6/go_bp_barplot.png` |
| 6 | Network | `STRINGdb$new()` | Target gene list | STRING network | In memory |
| 6 | Network | `igraph::graph_from_data_frame()` | STRING interactions | igraph object | In memory |
| 6 | Network | `igraph::degree()` | PPI graph | Hub gene table | `results/Week6/hub_genes.csv` |
| 6 | Visualization | `igraph::plot()` | PPI graph | Network PNG | `results/Week6/ppi_network.png` |
| 6 | Summary figure | `ggplot2` (forest plot) | DE + SHAP + targets | Biomarker panel figure | `results/Week6/biomarker_panel_forest_plot.png` |
| 6 | Validation sim | `ggplot2` (box plots) | Simulated qPCR Ct | qPCR validation plots | `results/Week6/qpcr_validation_sim.png` |
| 6 | Session info | `sessionInfo()` | R session | Session log | `results/Week6/session_info_week6.txt` |

**Datasets used throughout the course:**

| Dataset | GEO Accession | Sample Type | N (AD / MCI / Control) | Platform | Role |
|---------|--------------|-------------|------------------------|----------|------|
| GSE120584 | Primary | Serum | 48 / 50 / 50 | Illumina HiSeq 2500 (small RNA-seq) | Training + DE |
| GSE46579 | Validation | Whole blood | 35 / — / 30 | Affymetrix GeneChip miRNA 3.0 | External validation |

---

## Congratulations and Next Steps

You have completed the 6-week AI/ML in Biomarker Discovery course. This is not a trivial achievement. In six weeks, you have built the entire computational infrastructure of a miRNA biomarker discovery study — from raw GEO data to a biologically interpreted, ML-validated candidate panel. Most researchers spend months or years reaching this point.

**What you have built:**

- A reproducible, documented data processing pipeline covering two independent datasets
- A normalized, QC-filtered, batch-corrected expression matrix ready for analysis
- An exploratory data analysis suite (PCA, t-SNE, UMAP, clustering heatmaps) for understanding data structure
- A full differential expression analysis using appropriate count-based statistical models
- A trained and validated machine learning classifier with SHAP-based explainability
- A biologically grounded interpretation connecting computational results to known AD molecular pathology
- A network analysis identifying hub proteins that connect your biomarker miRNAs to the core AD proteome
- A biomarker validation roadmap from GEO dataset to a clinical blood test

**The gap that remains:**

Everything you have built in this course is **Stage 1 discovery** on the five-stage biomarker development roadmap. Real clinical translation requires:

- Independent analytical validation with a clinical-grade assay (digital PCR, next-generation sequencing with spike-in controls, validated RT-qPCR)
- Prospective cohort validation with pre-specified endpoints and powered sample sizes
- Multi-site replication with diverse populations
- Mechanistic studies (cell-based and animal model) to test the biology your targets suggest
- A regulatory strategy and clinical development plan

None of these steps are purely computational. They require wet-lab experiments, clinical partnerships, and regulatory expertise. But you are now equipped to design them intelligently, to know which biological hypotheses are most worth testing, and to speak the language of both computation and biology fluently.

**Where to go from here:**

If you are a wet-lab biologist, the most valuable next step is to collaborate with a clinical partner. Find a neurologist or geriatric psychiatrist who has access to a patient biobank. Propose a small prospective sample collection study (20–30 patients per group) as a qPCR validation of your top 3–5 miRNAs. This transforms your Stage 1 discovery into Stage 2 verification.

If you are moving toward a research career, use the capstone project as preliminary data for your first grant application. An R21 from the NIA or a New Investigator Research Grant from the Alzheimer's Association are appropriate targets for a well-framed, preliminary-data-supported proposal.

If this course sparked an interest in computational biology more broadly, the skills you have built — R programming, statistical inference, machine learning, biological database navigation, experimental design — transfer directly to cancer genomics, infectious disease, pharmacogenomics, and virtually every field of modern quantitative biology.

The computational tools in this course will evolve. New sequencing technologies, new ML architectures, and new databases will emerge. But the underlying principles — rigorous QC, appropriate statistical methods, reproducible documentation, biological grounding, and ethical awareness — are permanent.

Thank you for bringing curiosity, persistence, and biological intuition to this course. The most important biomarker discoveries of the next decade will be made by scientists who understand both the biology and the computation. You are now one of them.

---

*Course: AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease*  
*Week 6 of 6 — Biological Interpretation & Clinical Translation*
