# AI/ML in Biomarker Discovery
## 6-Week Intensive Program — miRNA-Based Biomarker Discovery in Alzheimer's Disease

---

# COURSE OVERVIEW

**Target Audience:** Wet-lab biologists with strong biomedical background and little to no coding experience  
**Duration:** 6 weeks, (lectures + hands-on labs)  
**Disease Focus:** Alzheimer's Disease (AD)  
**Biomarker Class:** MicroRNA (miRNA)  
**Computational Environment:** Python (Jupyter notebooks) + R (Bioconductor)  
**Data Source:** Publicly available GEO datasets (blood-based miRNA profiling)

---

## Program Philosophy

This course prioritizes **biological understanding first, computation second.** Every dataset, every algorithm, and every result will be interpreted through the lens of disease biology. You will not become a software engineer; you will become a biologist who can confidently use AI/ML tools and critically evaluate computational analyses in your own research.

---

## 6-Week Syllabus at a Glance

| Week | Title | Core Topics |
|------|-------|-------------|
| **1** | Foundations: Biology & Analytical Tools | Alzheimer's disease biology, miRNA biology, literature review, measurement technologies, data repositories, tool setup |
| **2** | Data Acquisition & Quality Control | Searching GEO, downloading datasets, sample metadata, raw data formats, preprocessing & normalization |
| **3** | Exploratory Data Analysis | Descriptive statistics, dimensionality reduction (PCA, t-SNE, UMAP), clustering, visualization, batch effects |
| **4** | Feature Selection & Classical ML | Differential expression, feature selection methods, logistic regression, SVM, random forest, model evaluation |
| **5** | Advanced ML & Validation | Cross-validation strategies, ensemble methods, deep learning introduction, external validation, overfitting |
| **6** | Biological Interpretation & Clinical Translation | Target prediction, pathway enrichment, network analysis, biomarker validation roadmap, regulatory considerations |

---

## Practical Workflow (Runs Across All 6 Weeks)

```
Literature Review → Dataset Selection → Data Download → 
Quality Control → Normalization → Exploratory Analysis → 
Differential Expression → Feature Selection → ML Modeling → 
Biological Interpretation → Clinical Contextualization
```

---

---

# WEEK 1: Foundations in Biomedical Science & Analytical Tools

## Learning Objectives

By the end of Week 1, you will be able to:
1. Describe the molecular pathology of Alzheimer's disease and explain why current diagnostics are insufficient
2. Explain miRNA biogenesis, function, and why they are attractive biomarker candidates
3. Critically read and discuss primary literature on miRNA biomarkers in AD
4. Distinguish between microarray and next-generation sequencing (NGS) technologies for miRNA profiling
5. Identify major platforms and databases for obtaining miRNA expression data
6. Navigate the key computational tools that will be used throughout the course

---

## MODULE 1.1 — Alzheimer's Disease: Biology and the Diagnostic Gap

### 1.1.1 What is Alzheimer's Disease?

Alzheimer's disease (AD) is the most common form of dementia, accounting for 60–80% of all dementia cases. It is a progressive, irreversible neurodegenerative disorder characterized by:

- **Amyloid-β (Aβ) plaques** — extracellular aggregates of misfolded amyloid-β peptides derived from the amyloid precursor protein (APP) by sequential cleavage by β-secretase (BACE1) and γ-secretase
- **Neurofibrillary tangles (NFTs)** — intracellular aggregates of hyperphosphorylated tau protein (encoded by the *MAPT* gene), which disrupt microtubule assembly and axonal transport
- **Neuroinflammation** — chronic activation of microglia and astrocytes, driven in part by Aβ accumulation
- **Synaptic loss and neurodegeneration** — ultimately leading to cognitive decline, memory impairment, and loss of function

**Key genetic risk factors:**
- *APOE ε4* allele — most common genetic risk factor for late-onset AD (increases risk 3–12×)
- *APP*, *PSEN1*, *PSEN2* mutations — cause rare early-onset familial AD (<1% of cases)

**Disease staging:**
- **Preclinical AD:** Aβ deposits begin accumulating ~15–20 years before symptoms; no cognitive impairment
- **Mild Cognitive Impairment (MCI):** Subjective and objective memory complaints; not yet dementia
- **Clinical AD:** Progressive dementia affecting daily life

> **Why this matters for biomarkers:** The pathological changes in AD begin **decades before** clinical symptoms appear. Any effective biomarker strategy must detect disease at the preclinical or MCI stage, when disease-modifying interventions would have the greatest impact.

---

### 1.1.2 The Diagnostic Challenge

Current gold-standard AD biomarkers include:

| Biomarker | Sample Type | Clinical Issue |
|-----------|-------------|----------------|
| Aβ42, total tau, p-tau181 | Cerebrospinal fluid (CSF) | Invasive lumbar puncture; patient reluctance |
| Amyloid PET, tau PET | Brain imaging | Expensive (~$3,000–5,000/scan); limited availability |
| Clinical cognitive tests (MMSE) | Behavioral | Subjective; only detects established disease |
| Plasma Aβ42/40, p-tau217 | Blood | Emerging; limited clinical infrastructure |

**The unmet need:** A **non-invasive, blood-based biomarker** that can detect AD at preclinical or prodromal stages, is cost-effective, and can be widely deployed in primary care settings. This is precisely the space where **circulating miRNAs** have shown enormous promise.

---

## MODULE 1.2 — MicroRNA Biology: From Genome to Biomarker

### 1.2.1 What are MicroRNAs?

MicroRNAs (miRNAs) are a class of small (~18–25 nucleotide), single-stranded, non-coding RNA molecules that regulate gene expression at the post-transcriptional level. They were first discovered in 1993 (*lin-4* in *C. elegans*; Lee et al., Science) and recognized as a major regulatory class in 2001.

**Key facts:**
- The human genome encodes **~2,656 mature miRNAs** (miRBase v22)
- Each miRNA can regulate **hundreds to thousands** of target genes
- Together, miRNAs are estimated to regulate **>60% of all human protein-coding genes**
- Named using a standardized nomenclature: e.g., **hsa-miR-21-5p** (hsa = *Homo sapiens*, 21 = gene number, 5p = derived from 5' arm of precursor)

---

### 1.2.2 miRNA Biogenesis Pathway

Understanding the biogenesis pathway helps you understand why miRNA expression can go wrong in disease:

```
NUCLEUS:
Gene (DNA) → RNA Pol II transcription → Primary miRNA (pri-miRNA)
                                              ↓
                               Drosha/DGCR8 complex (Microprocessor)
                                              ↓
                             ~70 nt hairpin: Precursor miRNA (pre-miRNA)
                                              ↓
                                      Exportin-5 (nuclear export)
                                              ↓
CYTOPLASM:
                                         pre-miRNA
                                              ↓
                                     Dicer + TRBP complex
                                              ↓
                             ~22 bp RNA duplex (miRNA/miRNA* duplex)
                                              ↓
                         Strand selection: one strand (guide) → mature miRNA
                                           other strand (passenger) → degraded
                                              ↓
                          RISC (RNA-Induced Silencing Complex) + AGO2 protein
                                              ↓
               Target mRNA recognition (partial complementarity, 3' UTR "seed" matching)
                                              ↓
              mRNA degradation OR translational repression → reduced protein output
```

**Biological consequence:** miRNAs act as **rheostats** (fine-tuners), not on/off switches. A single miRNA rarely completely silences a gene; instead, it reduces output by 30–80%, enabling precise modulation of complex biological networks.

---

### 1.2.3 Why Are miRNAs ideal biomarker candidates?

miRNAs possess several properties that make them exceptionally attractive as clinical biomarkers:

**1. Stability in biofluids**
Unlike mRNA (which degrades rapidly in blood due to RNases), miRNAs are remarkably stable in serum, plasma, CSF, urine, and saliva. This stability arises from:
- Encapsulation within **extracellular vesicles (EVs)** / exosomes
- Binding to **Argonaute 2 (AGO2) protein** (protein-protected form)
- Association with **HDL/LDL lipoproteins**
- Binding to **RNA-binding proteins** (e.g., nucleophosmin)

**2. Tissue specificity**
Many miRNAs are preferentially expressed in specific tissues (e.g., miR-9 and miR-132 are highly enriched in neurons), which means their presence in blood can reflect pathology in the originating tissue.

**3. Disease-responsive changes**
miRNA expression profiles change measurably in response to disease, preceding protein-level or structural changes in many cases.

**4. Brain-to-blood communication via EVs**
Neurons and glial cells release EVs that carry miRNAs into the bloodstream. Some brain-derived EVs can cross or release miRNAs across the blood-brain barrier (BBB), allowing **non-invasive sampling of brain miRNA profiles through blood draws.**

**5. Measurability at scale**
With current NGS and microarray platforms, hundreds to thousands of miRNAs can be profiled simultaneously from small volumes of biofluids (as little as 200 µL serum).

**6. Clinical deployability**
qPCR-based miRNA assays can be run on standard clinical laboratory equipment, making validated miRNA biomarkers potentially deployable in any hospital or primary care setting.

---

### 1.2.4 miRNAs Most Implicated in Alzheimer's Disease Biology

The following miRNAs have been repeatedly identified across AD studies. As a biologist, understanding their *targets* and *pathways* is essential for interpreting computational results:

| miRNA | Direction in AD | Key Targets | Biological Relevance |
|-------|-----------------|-------------|----------------------|
| **miR-29a/b** | ↓ Downregulated | BACE1, APP | Decreased miR-29 → increased BACE1 → more Aβ production |
| **miR-107** | ↓ Downregulated | BACE1, Cofilin-1 | One of the earliest changed miRNAs in AD |
| **miR-9** | ↓ Downregulated (brain) | NFκB, SIRT1 | Neuronal miRNA; regulates neuroinflammation |
| **miR-132/212** | ↓ Downregulated | FOXO3a, tau kinases | Neuroprotective; loss linked to tau hyperphosphorylation |
| **miR-34a** | ↑ Upregulated | SIRT1, BCL2 | Promotes apoptosis, linked to aging |
| **miR-146a** | ↑ Upregulated | TRAF6, IRAK1 | Master regulator of neuroinflammation via TLR/NF-κB |
| **miR-155** | ↑ Upregulated | SHIP1, C/EBPβ | Neuroinflammatory; elevated in microglia |
| **miR-21-5p** | ↑ Upregulated | PTEN, PDCD4 | Anti-apoptotic; elevated in AD serum |
| **miR-26a/26b-5p** | ↓ Downregulated | PTEN, CDK5 | Strongly correlated with MMSE cognitive scores |
| **miR-532-5p** | Variable | Multiple | Highest statistical significance in Ludwig et al. 2019 |
| **miR-128** | ↓ Downregulated | PPARγ, Bax | Neuroprotective; regulates insulin signaling |
| **miR-181** | ↑ Upregulated | SIRT1, GRP78 | Linked to tau phosphorylation and ER stress |

> **Key concept for wet-lab biologists:** When a computational analysis identifies a differentially expressed miRNA, your job is to ask: *Which of my target's proteins change? What pathway is perturbed? Is this biologically plausible for AD pathology?* This biological validation thinking is what separates a good bioinformatics study from a great one.

---

## MODULE 1.3 — Literature Review: miRNA Biomarkers in Alzheimer's Disease

*The following literature review is based on articles retrieved from PubMed. All citations include DOI links.*

### 1.3.1 Early Biomarker Discovery and the Case for Blood-Based Approaches

The quest for minimally invasive AD biomarkers has driven significant innovation. An influential early review by Zafari et al. (2015) *Gerontology* [(DOI: 10.1159/000375236)](https://doi.org/10.1159/000375236) outlined the landscape of circulating biomarker panels in AD, contrasting established CSF markers (Aβ42, tau) with emerging blood-based approaches. The authors highlighted the critical need for high-throughput profiling technologies — microarrays and next-generation sequencing — to discover miRNA signatures capable of detecting AD non-invasively.

Zetterberg and Burnham (2019) *Molecular Brain* [(DOI: 10.1186/s13041-019-0448-1)](https://doi.org/10.1186/s13041-019-0448-1) provided a comprehensive review of blood-based molecular biomarkers for AD, situating miRNAs within a broader landscape that includes plasma Aβ, tau, and neurofilament light chain (NfL). They emphasized that **blood-based biomarkers are critical for primary care screening** — the clinical setting where most patients first present — and that a scalable, cost-effective blood test for AD represents one of the field's most pressing unmet needs.

### 1.3.2 CSF miRNA Profiling

Cerebrospinal fluid, being in direct contact with the brain parenchyma, was among the first biofluids interrogated for miRNA biomarkers. Müller et al. (2015) *PLoS One* [(DOI: 10.1371/journal.pone.0126423)](https://doi.org/10.1371/journal.pone.0126423) applied **OpenArray technology** — a medium-throughput qPCR-based approach capable of profiling 1,178 unique miRNAs simultaneously — to CSF from AD patients and controls. This landmark study identified several miRNAs specifically dysregulated in AD CSF, including members of the let-7 family and miR-29 family, and demonstrated that **CSF miRNA signatures can distinguish AD from other dementias.** However, the invasiveness of lumbar puncture limits the clinical applicability of CSF-based approaches.

### 1.3.3 Blood-Based miRNA Biomarker Discovery

The majority of recent biomarker discovery work has shifted to blood (serum/plasma), which can be obtained via routine venipuncture.

A landmark systematic review by Fattahi et al. (2024) *Metabolic Brain Disease* [(DOI: 10.1007/s11011-024-01431-7)](https://doi.org/10.1007/s11011-024-01431-7) synthesized findings from 48 studies comprising **4,001 AD patients and 3,886 healthy controls.** Their key findings:
- **83 miRNAs were consistently upregulated** in AD blood
- **66 miRNAs were consistently downregulated** in AD blood
- Whole blood (39.6%) and serum (27.1%) were the most common sample types
- No single miRNA emerged as universally consistent — **panel-based approaches** (multiple miRNAs together) consistently outperform single miRNA biomarkers
- RT-qPCR was the most commonly used validation method

Wang et al. (2023) *Int J Mol Sci* [(DOI: 10.3390/ijms242216259)](https://doi.org/10.3390/ijms242216259) provided a mechanistic review connecting circulating miRNA profiles to core AD pathology. They catalogued miRNAs that directly regulate:
- **MAPT** (tau gene): miR-9, miR-132, miR-212, miR-34a regulate tau phosphorylation and splicing
- **APP** (amyloid precursor protein): miR-101, miR-16, miR-153 suppress APP translation
- **BACE1** (β-secretase, the enzyme that cleaves APP to produce Aβ): miR-29 family, miR-107, miR-339-5p — **arguably the most therapeutically relevant miRNA-target axis in AD**

This mechanistic grounding is essential: differentially expressed miRNAs in blood are not merely correlation signals; they often reflect **real causal biology** happening in the brain.

### 1.3.4 Extracellular Vesicles: A Bridge Between Brain and Blood

A crucial recent advance concerns **neuronal-derived extracellular vesicles (NDEVs).** Reho et al. (2025) *Alzheimer's & Dementia* [(DOI: 10.1002/alz.70050)](https://doi.org/10.1002/alz.70050) conducted a transcriptome-wide association study of NDEV-derived miRNAs in serum from 46 clinical AD patients, 14 preclinical AD patients, and 60 healthy controls — a multiethnic cohort.

**Key findings:**
- **14 miRNAs** were significantly associated with AD risk
- Preclinical AD individuals showed **more pronounced transcriptional alterations** than clinical AD individuals — suggesting NDEV miRNAs may be particularly powerful for early detection
- Key target genes identified: **SNCA** (alpha-synuclein), **CYCS** (cytochrome c), and **MAPT** (tau) — all central to neurodegeneration

This study illustrates the **power of combining EV biology with miRNA profiling**: by isolating neuron-specific EVs, researchers can enrich for brain-derived signals in a peripheral blood sample, effectively giving us a "liquid biopsy" of the brain.

> **Liquid biopsy concept:** A review by Malhotra et al. (2023) *Cells* [(DOI: 10.3390/cells12141911)](https://doi.org/10.3390/cells12141911) defines liquid biopsy as the analysis of disease-relevant biological molecules — including miRNAs, cell-free DNA, and EVs — from non-solid biofluids (blood, urine, CSF, saliva). In neurological diseases, liquid biopsy is emerging as a transformative approach precisely because traditional tissue biopsy of the brain is not feasible in living patients.

### 1.3.5 Machine Learning Applied to miRNA Biomarker Discovery

The application of ML to miRNA biomarker panels has substantially improved diagnostic accuracy over individual miRNAs. This is the central technical focus of our course.

**Ludwig et al. (2019)** *Genomics, Proteomics & Bioinformatics* [(DOI: 10.1016/j.gpb.2019.09.004)](https://doi.org/10.1016/j.gpb.2019.09.004) is a landmark study. Starting from high-throughput sequencing-derived miRNA signatures in US and German cohorts, the authors validated 21 circulating miRNAs in **465 individuals** (AD patients + controls) using RT-qPCR. A machine learning model achieved an **AUC of 87.6%** for AD vs control classification — a clinically meaningful performance threshold. Critically, **miR-26a/26b-5p** showed significant correlation with MMSE scores (cognitive severity), and **miR-532-5p** showed the highest statistical significance in disease association. miRNAs downregulated in AD were enriched in monocytes and T-helper cells; those upregulated were enriched in serum exosomes, suggesting cell-type-specific release mechanisms.

**Zhao et al. (2020)** *J Applied Lab Medicine* [(DOI: 10.1373/jalm.2019.029595)](https://doi.org/10.1373/jalm.2019.029595) profiled >500 miRNAs in 96 serum samples using multiplex RT-qPCR (OPTIMA cohort). A **random forest** classifier identified a **12-miRNA signature** achieving 76–85.7% accuracy for AD diagnosis. Notably, the signature accuracy improved substantially when validated on post-mortem histology-confirmed AD cases (85.7%), highlighting the challenge of clinical diagnosis variability.

**Xu et al. (2022)** *J Alzheimer's Disease* [(DOI: 10.3233/JAD-215502)](https://doi.org/10.3233/JAD-215502) built ML models using not just miRNA expression values but also **target gene descriptors and pathway features** derived from known miRNA-target interactions. Their best serum-based model achieved **92.0% accuracy** and their plasma-based model achieved **90.9% accuracy.** This study demonstrates a critical concept: incorporating **biological knowledge** (target gene networks, pathways) into ML feature engineering can substantially improve model performance — a strategy we will implement in Week 4 of this course.

**Viswambharan et al. (2017)** *Progress in Molecular Biology and Translational Science* [(DOI: 10.1016/bs.pmbts.2016.12.013)](https://doi.org/10.1016/bs.pmbts.2016.12.013) reviewed miRNAs as peripheral biomarkers in aging and age-related diseases, contextualizing AD miRNA findings within the broader landscape of aging biology and discussing the stabilizing mechanisms (exosomal packaging, protein binding) that make circulating miRNAs measurable.

### 1.3.6 Summary: What the Literature Tells Us

| Finding | Implication for Course |
|---------|------------------------|
| No single miRNA biomarker is sufficient | We will build **panel-based ML classifiers**, not single-marker tests |
| Panel performance peaks at 10–30 miRNAs | **Feature selection** (Week 4) is critical to avoid overfitting |
| Performance varies by sample type (serum vs plasma vs whole blood) | **Metadata management** and batch correction (Week 2) are essential |
| Biological knowledge improves ML performance | We will use **target gene and pathway features** alongside expression values |
| Preclinical detection is the frontier | Our ML models will include **MCI samples** as a distinct class |
| External validation is rarely done | We will implement **cross-cohort validation** using multiple GEO datasets |

---

## MODULE 1.4 — Measurement Technologies: How miRNA Data is Generated

Understanding how your data was generated is fundamental to interpreting it correctly. Two major technologies dominate miRNA profiling: **microarray** and **next-generation sequencing (small RNA-seq).**

### 1.4.1 Technology 1: Microarray-Based miRNA Profiling

**Principle:**  
Microarrays use **hybridization** — the property of complementary nucleic acid strands to bind to each other. Probes (short oligonucleotides complementary to known miRNA sequences) are spotted or synthesized in fixed positions on a glass or silicon chip. Labeled sample RNA is hybridized to the chip; signal intensity at each probe position reflects the abundance of the corresponding miRNA.

**Experimental workflow:**
```
Biofluid collection (serum/plasma) 
     ↓
RNA extraction (Qiagen miRNeasy, Thermo TRIzol-LS)
     ↓
RNA quality check (Bioanalyzer or TapeStation)
     ↓
Sample labeling (Cy3/Cy5 fluorescent dyes, or biotin)
     ↓
Hybridization to microarray chip (12–18 hours, 55°C)
     ↓
Chip washing & scanning (laser scanner)
     ↓
Image analysis → signal intensity matrix
     ↓
Normalization → Expression matrix (samples × miRNAs)
```

**Major microarray platforms for miRNA:**

| Platform | Company | # miRNAs | Notes |
|----------|---------|-----------|-------|
| GeneChip miRNA 4.0 Array | Affymetrix/Thermo Fisher | ~2,578 human miRNAs | Gold standard for sensitivity; also detects pre-miRNAs |
| SurePrint G3 Human miRNA 8x60K | Agilent Technologies | ~2,006 human miRNAs | High dynamic range; dual-color design |
| Human miRNA Expression BeadChip | Illumina | ~1,532 human miRNAs | Bead-based; high reproducibility |
| TaqMan Array Human MicroRNA A+B cards | Applied Biosystems | 754 miRNAs/card | qPCR-based array; highest sensitivity |
| OpenArray Human miRNA Panel | Applied Biosystems | 754 or 1,178 assays | Used in Müller et al. 2015 CSF study |

**Strengths of microarray:**
- Well-established; large body of published data in GEO for cross-study comparison
- Cost-effective for large sample sizes (n > 50)
- Quantification is rapid (~24 hours)
- Simpler bioinformatic pipeline
- No need for library preparation

**Limitations of microarray:**
- **Closed vocabulary:** Can only detect miRNAs with existing probes; novel miRNAs are missed
- **Cross-hybridization:** Probes with imperfect complementarity can bind related miRNAs (false signals)
- **Lower dynamic range** compared to sequencing
- **Probe design bias:** Detection efficiency varies by miRNA sequence composition
- **GC-content effects** on hybridization efficiency require careful normalization

**Data output:** A matrix of fluorescence intensity values (or log2-transformed ratios) for each miRNA × sample combination. Typical output: hundreds to thousands of miRNA probes × N samples.

---

### 1.4.2 Technology 2: Small RNA Sequencing (miRNA-seq)

**Principle:**  
Small RNA sequencing uses **next-generation sequencing (NGS)** to directly sequence small RNA molecules from a sample. Unlike microarrays, it is not limited to known sequences — any small RNA present in sufficient abundance will be detected and quantified.

**Experimental workflow:**
```
Biofluid collection
     ↓
RNA extraction (specialized small RNA extraction: Qiagen miRNeasy, 
               NEB Monarch, or exosome isolation first)
     ↓
Size selection (gel electrophoresis or bead-based: select 18–30 nt fragments)
     ↓
3' adapter ligation (RNA ligase)
     ↓
5' adapter ligation
     ↓
Reverse transcription (cDNA synthesis)
     ↓
PCR amplification (add sample index/barcode for multiplexing)
     ↓
Library quality check (Bioanalyzer: ~145 bp expected product)
     ↓
Pooling & sequencing (Illumina short-read platform, 50 bp single-end)
     ↓
Raw FASTQ files (millions of short reads per sample)
     ↓
Adapter trimming → Alignment to genome/miRBase → 
Read counting → Count matrix (samples × miRNAs)
```

**Major sequencing platforms used for miRNA-seq:**

| Platform | Company | Read Length | Throughput | Common Use |
|----------|---------|-------------|------------|------------|
| NovaSeq 6000/X | Illumina | 50–150 bp | Very high (6 Tb) | Population-scale studies |
| HiSeq 2500/4000 | Illumina | 50–150 bp | High | Standard research; many GEO datasets |
| MiSeq | Illumina | 150–300 bp | Low | Smaller studies, clinical validation |
| NextSeq 500/550 | Illumina | 75–150 bp | Medium | Mid-scale studies |
| Ion Torrent S5 | Thermo Fisher | Variable | Medium | Alternative to Illumina |

> **Why 50 bp single-end reads?** miRNAs are ~22 nt; with 3' adapter sequence (~30 nt), total insert size is ~52 bp. Single-end 50 bp reads are sufficient and cost-effective. Paired-end sequencing offers no advantage for miRNA-seq.

**Key bioinformatics steps for miRNA-seq (conceptual overview for Week 1):**

1. **Adapter trimming:** Remove the 3' adapter sequence added during library prep (using tools like Trim Galore or Cutadapt)
2. **Alignment:** Map trimmed reads to the human genome (hg38) or directly to miRBase hairpin sequences (using tools like miRDeep2, STAR, or Bowtie)
3. **Quantification:** Count reads mapping to each annotated miRNA (using featureCounts or miRDeep2)
4. **Output:** A **count matrix** — integer counts of reads per miRNA per sample

**Strengths of small RNA-seq:**
- **Open discovery:** Can detect novel miRNAs and isomiRs (miRNA sequence variants)
- **Higher dynamic range** — can detect very low and very high abundance miRNAs simultaneously
- **Single-base resolution** — can distinguish highly similar miRNA family members
- **Simultaneous profiling** of other small RNAs (piRNAs, snoRNAs, tRFs) from the same data
- Unbiased — no probe design needed

**Limitations of small RNA-seq:**
- **More expensive** per sample (library prep + sequencing costs)
- **More complex bioinformatics** — requires alignment, reference genomes, multiple QC steps
- **Ligation bias** — adapter ligation efficiency varies by miRNA sequence, affecting quantification accuracy
- **Longer turnaround time**
- **Requires more input RNA** quality consciousness (low integrity RNA causes bias)

---

### 1.4.3 Technology 3: RT-qPCR (Validation Gold Standard)

While not a primary discovery platform, RT-qPCR remains the **gold standard for validation** of candidate biomarkers identified by microarray or sequencing.

**Principle:** Reverse transcription of miRNA into cDNA, followed by quantitative PCR amplification using miRNA-specific primers. Signal is quantified as cycle threshold (Ct) — lower Ct = higher expression.

**Major platforms:**
- **TaqMan miRNA Assays (Applied Biosystems):** Stem-loop RT primer captures specific miRNA; TaqMan probe provides high specificity. Industry standard.
- **miRCURY LNA miRNA PCR Assays (Qiagen):** LNA-modified primers improve hybridization; no separate RT step needed for each miRNA.
- **SYBR Green-based assays:** Lower cost but less specific.

**When used in a biomarker pipeline:**
- **Discovery phase:** Microarray or miRNA-seq (high-throughput, discovery)
- **Screening phase:** OpenArray or TaqMan Array Cards (medium-throughput, 96–754 miRNAs)
- **Validation phase:** RT-qPCR on individual candidates (low-throughput, highest confidence)

---

### 1.4.4 Comparison Summary

| Feature | Microarray | Small RNA-seq | RT-qPCR |
|---------|-----------|---------------|---------|
| **Throughput** | High (1,000s miRNAs) | High (all known + novel) | Low (1–96/run) |
| **Dynamic range** | Limited (~3 logs) | Wide (~5 logs) | Wide (~5–6 logs) |
| **Novel miRNA detection** | No | Yes | No |
| **Sensitivity** | Moderate | High | Highest |
| **Specificity** | Moderate (cross-hybridization risk) | High (sequence-based) | Very high |
| **Cost per sample** | Moderate | High | Low (per target) |
| **Bioinformatics complexity** | Low | High | Very low |
| **Data in GEO for AD** | Abundant | Growing | Limited (summary stats) |
| **Typical use** | Discovery | Discovery / Deep profiling | Validation |

> **Course data:** We will primarily work with **GEO-deposited microarray and RNA-seq datasets** from AD studies. Understanding these technologies helps you critically evaluate study design, understand normalization choices, and correctly interpret QC metrics.

---

## MODULE 1.5 — Data Repositories & Platforms for miRNA Data

### 1.5.1 Primary Data Repositories

**1. NCBI GEO (Gene Expression Omnibus)**  
Website: [https://www.ncbi.nlm.nih.gov/geo/](https://www.ncbi.nlm.nih.gov/geo/)  
- World's largest public repository for functional genomics data
- Contains both raw and processed data for microarray and NGS experiments
- Searchable by disease, tissue, organism, platform, and publication
- Data is organized into:
  - **GSE** (Series) — a complete study submission
  - **GSM** (Samples) — individual sample records
  - **GPL** (Platforms) — the array or sequencing platform used
  - **GDS** (Datasets) — curated, analysis-ready subsets

**Key AD miRNA datasets in GEO for our course:**

| GEO Accession | Study Focus | Sample Type | Platform | N Samples |
|---------------|-------------|-------------|----------|-----------|
| GSE46579 | AD blood miRNA | Whole blood | Affymetrix | ~65 |
| GSE67491 | AD serum miRNA | Serum | Affymetrix | ~150 |
| GSE63501 | AD plasma miRNA | Plasma | Affymetrix | ~80 |
| GSE120584 | AD serum miRNA (sequencing) | Serum | Illumina | ~200 |
| GSE57353 | AD brain miRNA | Brain tissue | Illumina | ~100 |

*We will work with one or more of these datasets across Weeks 2–5.*

**2. EBI ArrayExpress**  
Website: [https://www.ebi.ac.uk/arrayexpress/](https://www.ebi.ac.uk/arrayexpress/)  
- European counterpart to GEO; many studies cross-deposited in both
- Strict MIAME (Minimum Information About a Microarray Experiment) compliance
- Good source for validation datasets when looking for independent cohorts

**3. miRBase — The miRNA Registry**  
Website: [https://www.mirbase.org/](https://www.mirbase.org/)  
- **Official repository for miRNA sequences, annotations, and nomenclature**
- Current release: v22.1 (2588 hairpin precursors, 2656 mature sequences for *H. sapiens*)
- Every miRNA referenced in the literature has a miRBase accession number (e.g., MI0000077 for hsa-miR-21)
- Essential for: looking up sequences, understanding biogenesis (5p vs 3p arms), finding synonyms from older nomenclature

**4. ADNI (Alzheimer's Disease Neuroimaging Initiative)**  
Website: [http://adni.loni.usc.edu/](http://adni.loni.usc.edu/)  
- Multi-site longitudinal study with deep clinical, imaging, and biofluid data
- Includes blood-based omics data with corresponding clinical phenotypes (MMSE, CDR, amyloid PET)
- Requires data access application (free for academic researchers)
- Invaluable for linking miRNA changes to cognitive severity and disease progression

**5. AMP-AD Knowledge Portal (via Synapse)**  
Website: [https://adknowledgeportal.synapse.org/](https://adknowledgeportal.synapse.org/)  
- Data from the Accelerating Medicines Partnership — Alzheimer's Disease program
- Includes ROSMAP (Religious Orders Study and Memory and Aging Project): deep multi-omic profiling of brain tissue from ~1,000 individuals, including miRNA data
- Also includes Mount Sinai Brain Bank and Mayo Clinic RNAseq Study data

---

### 1.5.2 Supporting Databases for Biological Interpretation

These databases will be used in Weeks 5–6 for interpreting computational findings:

| Database | URL | Purpose |
|----------|-----|---------|
| **miRTarBase** | https://mirtarbase.cuhk.edu.cn | Experimentally validated miRNA-target interactions |
| **TargetScan** | https://www.targetscan.org | Computational miRNA target predictions |
| **miRDB** | https://mirdb.org | Machine learning-based target predictions |
| **miRNet** | https://www.mirnet.ca | miRNA-target network visualization and enrichment |
| **DIANA-miRPath** | https://diana.e-ce.uth.gr/mirpathv3 | KEGG/GO pathway analysis for miRNA sets |
| **miRWalk** | https://mirwalk.umm.uni-heidelberg.de | Comprehensive target database (3' UTR, CDS, 5' UTR) |
| **KEGG** | https://www.kegg.jp | Pathway maps for interpreting target gene lists |
| **STRING** | https://string-db.org | Protein-protein interaction networks |

---

## MODULE 1.6 — Computational Tools Overview

This section introduces the tools we will use throughout the course. No coding is expected this week — the goal is to become familiar with the landscape and understand what each tool is for.

### 1.6.1 Programming Languages

**Python**  
- General-purpose language dominant in ML/AI applications
- We use it for: data manipulation, machine learning models, visualization
- Key libraries:
  - `pandas` — data tables (like Excel, but for code)
  - `numpy` — numerical operations
  - `scikit-learn` — machine learning algorithms
  - `matplotlib` / `seaborn` — plotting and visualization
  - `scipy` — statistical tests
  - `biopython` — biological sequence tools

**R**  
- Statistical computing language dominant in genomics/bioinformatics
- We use it for: differential expression analysis, Bioconductor packages
- Key packages (from Bioconductor):
  - `DESeq2` — differential expression for count data (RNA-seq)
  - `limma` — differential expression for microarray or voom-transformed RNA-seq
  - `edgeR` — differential expression using negative binomial models
  - `ggplot2` — publication-quality visualization

**Why both?** The field uses both languages. Most ML pipelines are Python-first; most genomics preprocessing pipelines are R-first. We will use each where it excels and provide templates so you are never writing code from scratch.

---

### 1.6.2 Development Environment

**Jupyter Notebook / JupyterLab**  
- Interactive computational notebook where code, results, and explanatory text coexist
- Every code cell can be run individually — excellent for step-by-step learning
- We will provide **pre-built notebooks** for each week's lab session

**RStudio**  
- Integrated development environment (IDE) for R
- Contains a script editor, console, variable inspector, and plot viewer
- We will use R Markdown documents — similar to Jupyter but for R

---

### 1.6.3 R Packages for miRNA-Specific Analysis

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `GEOquery` | Download data directly from NCBI GEO | `BiocManager::install("GEOquery")` |
| `DESeq2` | Differential expression for RNA-seq count data | `BiocManager::install("DESeq2")` |
| `limma` | Differential expression for microarray data | `BiocManager::install("limma")` |
| `multiMiR` | Integrates 14 miRNA-target databases | `BiocManager::install("multiMiR")` |
| `miRBaseConverter` | Convert between miRNA name versions | `BiocManager::install("miRBaseConverter")` |
| `clusterProfiler` | Gene ontology and KEGG enrichment | `BiocManager::install("clusterProfiler")` |
| `pheatmap` | Heatmap visualization | `install.packages("pheatmap")` |
| `ggplot2` | Publication-quality plots | `install.packages("ggplot2")` |

---

### 1.6.4 Python Packages for ML Analysis

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `pandas` | Data manipulation and analysis | `pip install pandas` |
| `numpy` | Numerical operations | `pip install numpy` |
| `scikit-learn` | ML algorithms (SVM, RF, logistic regression) | `pip install scikit-learn` |
| `matplotlib` | Base plotting library | `pip install matplotlib` |
| `seaborn` | Statistical visualization | `pip install seaborn` |
| `scipy` | Statistical tests (Mann-Whitney, t-test) | `pip install scipy` |
| `umap-learn` | UMAP dimensionality reduction | `pip install umap-learn` |
| `shap` | SHAP values for ML model interpretation | `pip install shap` |
| `statsmodels` | Statistical modeling | `pip install statsmodels` |

---

### 1.6.5 The Full Computational Workflow (Preview)

```
Week 2: R
GEOquery → Download raw data (CEL files or count matrix)
limma / DESeq2 → Normalization and quality control

Week 3: Python + R
pandas/numpy → Data structuring and exploration  
ggplot2/seaborn → Visualization (boxplots, PCA, heatmaps)
umap-learn → Dimensionality reduction

Week 4: Python
scikit-learn → Differential expression + feature selection
scikit-learn → Logistic Regression, SVM, Random Forest
ROC curves, AUC, cross-validation → Model evaluation

Week 5: Python
sklearn.ensemble → Advanced ensemble methods
SHAP → Feature importance interpretation
External dataset → Cross-cohort validation

Week 6: R + Python
multiMiR → miRNA-target lookup
clusterProfiler → Pathway enrichment analysis
miRNet / Cytoscape → Network visualization
```

---

## WEEK 1 LAB SESSION: Environment Setup

**Objective:** By end of lab, every student has a working Python and R environment capable of running the course notebooks.

### Lab 1A — Setting Up Python (Anaconda)

1. Download and install **Anaconda** from https://www.anaconda.com/download
2. Create a dedicated course environment:
   ```bash
   conda create -n biomarker_ml python=3.11
   conda activate biomarker_ml
   pip install pandas numpy scikit-learn matplotlib seaborn scipy umap-learn shap statsmodels jupyterlab
   ```
3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
4. Run the verification notebook provided (verifies all packages installed correctly)

### Lab 1B — Setting Up R and RStudio

1. Download and install **R** from https://cran.r-project.org/
2. Download and install **RStudio** from https://posit.co/download/rstudio-desktop/
3. Open RStudio and run:
   ```r
   install.packages("BiocManager")
   BiocManager::install(c("GEOquery", "DESeq2", "limma", "edgeR", 
                          "multiMiR", "clusterProfiler", "pheatmap"))
   install.packages(c("ggplot2", "tidyverse", "readr", "dplyr"))
   ```
4. Verify installation:
   ```r
   library(GEOquery)
   library(DESeq2)
   library(ggplot2)
   cat("All packages loaded successfully!\n")
   ```

---

## WEEK 1 ASSIGNMENTS

### Reading Assignment (Required)
1. **Fattahi et al. (2024)** — *Blood-based microRNAs as potential biomarkers for Alzheimer's disease: systematic review* [(DOI: 10.1007/s11011-024-01431-7)](https://doi.org/10.1007/s11011-024-01431-7)  
   Focus on: Table of dysregulated miRNAs, sample type comparison, Figure summarizing consistency across studies
   
2. **Ludwig et al. (2019)** — *Machine Learning to Detect Alzheimer's Disease from Circulating Non-coding RNAs* [(DOI: 10.1016/j.gpb.2019.09.004)](https://doi.org/10.1016/j.gpb.2019.09.004)  
   Focus on: Study design, which miRNAs were validated, how the ML model was built and evaluated

### Reflection Questions (Discuss in Week 2 opening session)
1. Why do you think no single miRNA has emerged as a reliable standalone AD diagnostic? What biological and technical factors contribute to this?
2. The Ludwig et al. study used RT-qPCR for validation rather than sequencing. What advantages does this offer for translating the findings to a clinical test?
3. If you were designing a miRNA biomarker study for AD today, would you use serum or plasma? What are the key considerations?

### Optional Deep Dive
- Reho et al. (2025) [(DOI: 10.1002/alz.70050)](https://doi.org/10.1002/alz.70050) — *Preclinical Alzheimer's disease shows alterations in circulating neuronal-derived extracellular vesicle microRNAs*  
  This is cutting-edge 2025 work combining EV biology with miRNA profiling — an excellent preview of where the field is heading.

---

## WEEK 1 GLOSSARY

| Term | Definition |
|------|------------|
| **miRNA** | MicroRNA; small (~22 nt) non-coding RNA that regulates gene expression post-transcriptionally |
| **pri-miRNA** | Primary miRNA transcript; first product of miRNA gene transcription |
| **pre-miRNA** | Precursor miRNA; ~70 nt hairpin produced by Drosha/DGCR8 cleavage |
| **RISC** | RNA-Induced Silencing Complex; protein complex (with AGO2) that uses mature miRNA to find and silence target mRNAs |
| **Isomer (isomiR)** | Sequence variant of a canonical miRNA; same general sequence but with 1–3 nt differences at 5' or 3' end |
| **Extracellular vesicle (EV)** | Membrane-enclosed particle (30–5,000 nm) released by cells into biofluids; carries proteins, nucleic acids, lipids |
| **Exosome** | Small EV (30–150 nm) of endosomal origin; carries high miRNA cargo |
| **Liquid biopsy** | Sampling and analysis of disease biomarkers (miRNA, ctDNA, CTCs) from body fluids |
| **MCI** | Mild Cognitive Impairment; prodromal stage of AD with objective memory impairment but preserved daily function |
| **BACE1** | Beta-site APP Cleaving Enzyme 1; the β-secretase that initiates Aβ production from APP |
| **MMSE** | Mini-Mental State Examination; standard clinical cognitive scoring (0–30; <24 indicates impairment) |
| **GEO** | Gene Expression Omnibus; NCBI database for genomic expression data |
| **AUC** | Area Under the (ROC) Curve; measure of classifier performance (0.5 = random, 1.0 = perfect) |
| **Ct value** | Cycle threshold in qPCR; the PCR cycle at which signal crosses threshold; lower Ct = higher expression |
| **Normalization** | Mathematical procedure to remove systematic technical variation between samples before biological comparison |
| **CEL file** | Raw data file format for Affymetrix microarrays; contains probe-level fluorescence intensities |
| **FASTQ** | Raw data format for NGS; text file containing read sequences and per-base quality scores |

---

## KEY REFERENCES (Week 1)

All references retrieved from PubMed.

1. Ludwig N et al. (2019). Machine Learning to Detect Alzheimer's Disease from Circulating Non-coding RNAs. *Genomics Proteomics Bioinformatics* 17(4):430–440. [DOI: 10.1016/j.gpb.2019.09.004](https://doi.org/10.1016/j.gpb.2019.09.004)

2. Zhao X et al. (2020). A Machine Learning Approach to Identify a Circulating MicroRNA Signature for Alzheimer Disease. *J Appl Lab Med* 5(1):15–28. [DOI: 10.1373/jalm.2019.029595](https://doi.org/10.1373/jalm.2019.029595)

3. Xu A et al. (2022). Alzheimer's Disease Diagnostics Using miRNA Biomarkers and Machine Learning. *J Alzheimers Dis* 86(2):841–859. [DOI: 10.3233/JAD-215502](https://doi.org/10.3233/JAD-215502)

4. Reho P et al. (2025). Preclinical Alzheimer's disease shows alterations in circulating neuronal-derived extracellular vesicle microRNAs in a multiethnic cohort. *Alzheimers Dement* 21(3):e70050. [DOI: 10.1002/alz.70050](https://doi.org/10.1002/alz.70050)

5. Fattahi F et al. (2024). Blood-based microRNAs as the potential biomarkers for Alzheimer's disease: evidence from a systematic review. *Metab Brain Dis* 40(1):44. [DOI: 10.1007/s11011-024-01431-7](https://doi.org/10.1007/s11011-024-01431-7)

6. Wang L et al. (2023). Potential Implications of miRNAs in the Pathogenesis, Diagnosis, and Therapeutics of Alzheimer's Disease. *Int J Mol Sci* 24(22). [DOI: 10.3390/ijms242216259](https://doi.org/10.3390/ijms242216259)

7. Malhotra S et al. (2023). Liquid Biopsy in Neurological Diseases. *Cells* 12(14):1911. [DOI: 10.3390/cells12141911](https://doi.org/10.3390/cells12141911)

8. Zetterberg H, Burnham SC (2019). Blood-based molecular biomarkers for Alzheimer's disease. *Mol Brain* 12(1):26. [DOI: 10.1186/s13041-019-0448-1](https://doi.org/10.1186/s13041-019-0448-1)

9. Zafari S et al. (2015). Circulating Biomarker Panels in Alzheimer's Disease. *Gerontology* 61(6):497–503. [DOI: 10.1159/000375236](https://doi.org/10.1159/000375236)

10. Müller M et al. (2015). MicroRNA Profiling of CSF Reveals Potential Biomarkers to Detect Alzheimer's Disease. *PLoS One* 10(5):e0126423. [DOI: 10.1371/journal.pone.0126423](https://doi.org/10.1371/journal.pone.0126423)

11. Viswambharan V et al. (2017). MicroRNAs as Peripheral Biomarkers in Aging and Age-Related Diseases. *Prog Mol Biol Transl Sci* 146:151–184. [DOI: 10.1016/bs.pmbts.2016.12.013](https://doi.org/10.1016/bs.pmbts.2016.12.013)

12. Biagioli M et al. (2015). miFRame: analysis and visualization of miRNA sequencing data in neurological disorders. *J Transl Med* 13:185. [DOI: 10.1186/s12967-015-0594-x](https://doi.org/10.1186/s12967-015-0594-x)

13. Jaberi SA et al. (2021). Novel Plasma miRNAs as Biomarkers and Therapeutic Targets of Alzheimer's Disease at the Prodromal Stage. *J Alzheimers Dis* 83(2):1063–1080. [DOI: 10.3233/JAD-210307](https://doi.org/10.3233/JAD-210307)

---

*Next Week: Data Acquisition & Quality Control — We will download a real AD miRNA dataset from GEO, examine its structure, perform quality control, and normalize the data for downstream analysis.*
