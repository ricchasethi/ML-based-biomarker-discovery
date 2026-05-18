# Week 5: Advanced Machine Learning & External Validation
## AI/ML in Biomarker Discovery — miRNA in Alzheimer's Disease

---

## Learning Objectives

By the end of Week 5, you will be able to:
1. Explain the bias-variance tradeoff and identify overfitting in ML model results, including the five most common forms of data leakage in bioinformatics
2. Design and implement nested cross-validation to obtain unbiased estimates of model performance and hyperparameter selection
3. Build gradient boosting models (XGBoost/LightGBM) and tune their key hyperparameters including learning rate, n_estimators, and early stopping
4. Compute and interpret SHAP values to explain model predictions to a biological audience, connecting computational importance scores to known AD miRNA biology
5. Extend binary classifiers to the three-class AD/MCI/Control problem and critically evaluate performance metrics appropriate for multiclass imbalanced data
6. Conduct a formal external validation analysis using GSE46579 as an independent cohort, applying cross-platform harmonization via miRBaseConverter
7. Report ML-based biomarker results according to TRIPOD and STARD guidelines, including confidence intervals on AUC and DeLong's test for AUC comparison

---

## Conceptual Overview: Why Weeks 1–4 Are Not Enough

In Weeks 3 and 4, you built your first machine learning classifiers — logistic regression, random forest — and saw training and test set AUC values. Perhaps your random forest achieved AUC = 0.91 on the held-out test set. That is an exciting number. Before you write a paper and send a press release, however, three questions should give you pause:

**Question 1:** Was your model performance estimate actually unbiased? In high-dimensional biological data — where you have perhaps 800 miRNA features and only 148 patients — many sources of optimistic bias can inflate apparent performance even on a "test set."

**Question 2:** Can you explain *which* miRNAs drove the predictions, and do those miRNAs make biological sense? A black-box model that cannot be interpreted is unlikely to generate clinical hypotheses or survive peer review.

**Question 3:** Does your model work on a completely different cohort, sequenced on a different platform, in a different country? A model that cannot replicate externally is a statistical artifact, not a biomarker.

Week 5 addresses all three questions. We introduce the rigorous statistical machinery that separates publishable biomarker studies from the many that cannot be reproduced. The methods in this week are demanding — but they are exactly what separates good science from overfit models.

---

## MODULE 5.1 — The Overfitting Problem

### 5.1.1 The Bias-Variance Tradeoff

Every supervised ML model makes a tradeoff between two types of error.

**Bias** is the error that comes from a model being too simple to capture the true relationship in the data. A logistic regression with no interaction terms applied to data with complex non-linear biology has high bias. The model cannot represent the true signal, so it performs poorly on both training and test data. This is called *underfitting*.

**Variance** is the error that comes from a model being too sensitive to the specific samples in the training data. A decision tree with no depth limit has low bias (it can fit any pattern) but very high variance: change 5 samples and you get a completely different tree. This is called *overfitting*.

The ideal model has moderate, balanced bias and variance. In practice, as model complexity increases:
- Training error decreases monotonically (the model "memorizes" the data)
- Test error first decreases (the model captures real biology), then increases (it captures noise)

The minimum of the test error curve is where you want to be. Cross-validation, regularization, and ensemble methods are all techniques for finding and staying near that minimum.

```
Error
  |
  |  \           Test error (U-shaped)
  |   \         /
  |    \       /
  |     \_____/
  |           \
  |            \  Training error (decreasing)
  |_____________\_______
                    Model Complexity
```

**In miRNA biomarker discovery:** You have roughly 800 detected miRNAs (features) and 148 patients (samples) in GSE120584. Your feature-to-sample ratio is approximately 5:1. This is a regime where overfitting is the primary threat. Every model that can freely adjust 800 parameters with only 148 examples will overfit if not carefully constrained.

### 5.1.2 The Curse of Dimensionality

The "curse of dimensionality" refers to a family of problems that emerge as the number of features grows relative to the number of samples.

**Sparsity:** In high-dimensional space, all samples become approximately equidistant from each other. Nearest-neighbor classifiers and distance-based clustering stop working well because the concept of "close" loses meaning.

**Exponential sample requirements:** To densely sample a 1-dimensional space with N points requires N samples. To densely sample a 2-dimensional space requires N^2. For a d-dimensional space: N^d. With 800 miRNA features, you would theoretically need an astronomical number of samples to densely sample the feature space — but you have 148 patients.

**Practical consequences for miRNA ML:**
- Many machine learning models will appear to perform extremely well on training data simply by finding coincidental correlations among 800 features that happen to separate the 148 training samples
- These correlations will not replicate in new patients
- The solution is: feature selection/reduction before modeling, strong regularization, and rigorous cross-validation

> **Biological intuition:** If you randomly pick 100 patients and flip a coin to assign them "AD" or "Control" (completely random labels), a random forest with 800 features will still achieve approximately 60–70% accuracy on the training set by memorizing accidental patterns. This is not biology. This is overfitting. This is why you always evaluate on held-out data.

### 5.1.3 Why a 99% Training Accuracy Is Meaningless

Consider this scenario: you train a random forest on 148 miRNA expression samples (AD vs Control). You evaluate performance on the same 148 samples the model was trained on. You report "training accuracy = 99%."

A decision tree with no depth limit will achieve exactly 100% training accuracy on any dataset, every time, by definition. It simply memorizes every sample. This tells you nothing about whether the model learned biology.

**The only number that matters is performance on samples the model has never seen.**

In practice, because we have small N, we cannot afford to permanently hold out a large test set. Instead, we use **cross-validation** — systematically rotating which samples are held out — to get an unbiased estimate of generalization performance. Module 5.2 covers this in detail.

### 5.1.4 Data Leakage: What It Is and Its Five Most Common Forms

Data leakage occurs when information from the test set "leaks" into the training process in ways that inflate apparent performance but are not available in real clinical deployment. Leakage is the single most common reason published biomarker studies fail to replicate.

**Definition:** Any flow of information from the evaluation set (the data used to estimate performance) into the model training process constitutes leakage. Even subtle leakage can inflate AUC by 0.05–0.30, making a mediocre model appear excellent.

The five most common forms of data leakage in bioinformatics:

**Form 1: Feature selection before data splitting**
The most common mistake. You compute differential expression across all 148 patients to select the top 50 miRNAs, then use those 50 miRNAs in a cross-validation loop. Problem: the feature selection step used all 148 samples, including the eventual test fold. The selected features were chosen because they separate *your exact samples*, not because they generalize.

*Example:* In a 5-fold CV, if you select features using all 148 samples and then cross-validate, your "test fold" samples influenced which features were selected. The features are therefore optimized for those test samples. Your reported AUC can be 0.10–0.25 higher than the true generalizable value.

*Correct approach:* Feature selection must happen INSIDE the cross-validation loop, applied only to the training fold.

**Form 2: Normalization using global statistics**
You compute mean and standard deviation across all 148 samples, then z-score normalize, then split into folds. Problem: the test fold samples contributed to the normalization parameters. In deployment, you would not have access to future patients when normalizing training data.

*Correct approach:* Fit normalization parameters (mean, SD, scaler) on the training fold only. Apply those parameters to the test fold without refitting.

**Form 3: Duplicate/correlated samples across train and test**
If a study contains technical replicates (the same patient's RNA run twice), or if two studies share patient samples (which happens more often than you might think), splitting randomly places replicates in both train and test. The model then "sees" essentially the same sample in training and test.

*Correct approach:* Split by patient, not by sample. Verify no shared patient IDs between your training and external validation datasets.

**Form 4: Target-informed imputation**
If you have missing values in your expression matrix and impute them using global statistics computed across all samples including the test fold, you have leaked label information into the imputed values.

*Correct approach:* Impute missing values using only training fold statistics. Apply training fold imputation parameters to the test fold.

**Form 5: Post-hoc threshold selection**
You train your model, generate predicted probabilities on the test set, then choose the classification threshold (e.g., 0.45 instead of the default 0.50) that maximizes accuracy on that same test set. The threshold was selected using the test set. If you then report accuracy at that threshold, you have overfit the threshold to the test set.

*Correct approach:* Select the operating threshold on a validation fold or through the training CV loop. Report test set performance at a fixed or clinically pre-specified threshold.

### 5.1.5 Checklist to Prevent Data Leakage

Before finalizing any analysis, verify each of the following:

**Pipeline construction:**
- [ ] Feature selection (DESeq2, variance filter, or any other) occurs INSIDE the CV loop, applied only to training data
- [ ] Normalization parameters (mean, SD, scaler) are fit ONLY on training data and applied to test data
- [ ] Hyperparameter tuning uses ONLY training data (nested CV or a dedicated validation fold)
- [ ] The test fold is touched EXACTLY ONCE — at the end, to report performance

**Sample integrity:**
- [ ] Training and external validation patients are confirmed non-overlapping
- [ ] No technical replicates of the same patient appear in both train and test
- [ ] Paired samples (e.g., baseline and follow-up from the same patient) are kept together in the same fold

**Reporting:**
- [ ] All reported metrics are from held-out data, not training data
- [ ] If thresholds were tuned, performance is reported at the TRAINING-fold-selected threshold, not the test-fold-optimal threshold
- [ ] The number of times the test set was evaluated is tracked (ideally: once)

> **Biologist's perspective:** Data leakage in genomics is analogous to a clinical trial where the control group reads the case report forms before their evaluation, so they know what answers the study is looking for. The "result" of that trial is meaningless. Leakage produces a meaningless model — one that appears to work on paper but will fail the moment you apply it to a new patient.

---

## MODULE 5.2 — Cross-Validation Strategies

### 5.2.1 k-Fold Cross-Validation

k-Fold cross-validation is the standard method for estimating model performance when the dataset is too small to afford a permanent held-out test set.

**How it works:**
1. Randomly divide the N samples into k equal-sized groups ("folds")
2. For each fold i from 1 to k:
   a. Train the model on all samples NOT in fold i (N - N/k training samples)
   b. Evaluate on fold i (N/k test samples)
   c. Record the performance metric (AUC, accuracy, etc.)
3. Average performance across all k folds → estimated generalization performance

**k = 5** is the most common choice. k = 10 is more computationally expensive but produces a slightly lower-variance estimate. For a dataset of 148 samples:
- k = 5: each fold has ~30 test samples; trains on ~118
- k = 10: each fold has ~15 test samples; trains on ~133

**Important:** In our course, k-fold CV is used for performance estimation. When it is also used for hyperparameter tuning without nesting, it becomes biased (see Section 5.2.4).

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Load preprocessed training data (from Week 2/3 pipeline)
expr_train = pd.read_csv("data/processed/GSE120584_harmonized.csv", index_col=0)
meta_train = pd.read_csv("data/processed/GSE120584_metadata_harmonized.csv", index_col=0)

# Binary classification: AD vs Control (drop MCI for now — covered in Module 5.5)
mask = meta_train["group"].isin(["Alzheimer's Disease", "Control"])
X = expr_train.loc[mask].values          # samples × miRNAs
y = (meta_train.loc[mask, "group"] == "Alzheimer's Disease").astype(int).values

print(f"Training set: {X.shape[0]} samples, {X.shape[1]} miRNA features")
print(f"Class distribution: {np.bincount(y)} (Control=0, AD=1)")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# cross_val_score performs k-fold with AUC scoring
# NOTE: This is a "flat" CV — no inner loop for hyperparameters
#       This is acceptable for fixed-hyperparameter evaluation
#       but biased for hyperparameter selection (see Section 5.2.4)
cv_aucs = cross_val_score(rf, X, y, cv=kf, scoring="roc_auc")

print(f"\n5-Fold CV AUC: {cv_aucs}")
print(f"Mean AUC:      {cv_aucs.mean():.4f}")
print(f"Std AUC:       {cv_aucs.std():.4f}")
print(f"95% CI (approx): [{cv_aucs.mean() - 1.96*cv_aucs.std():.4f}, "
      f"{cv_aucs.mean() + 1.96*cv_aucs.std():.4f}]")
```

### 5.2.2 Stratified k-Fold: Preserving Class Balance Across Folds

**The problem with standard k-fold on imbalanced data:** If 40% of your samples are AD, 40% MCI, and 20% Control, random fold assignment can produce a fold where one class has very few or zero representatives. Training on a fold that has no Control samples, or evaluating on a fold with only 2 Control samples, produces highly variable and unreliable performance estimates.

**Stratified k-fold** ensures that each fold has the same class proportions as the full dataset. If the dataset is 40% AD, every fold will also be approximately 40% AD.

```python
from sklearn.model_selection import StratifiedKFold

# Stratified k-fold: class proportions preserved in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Always use StratifiedKFold for classification in bioinformatics
# Standard KFold is appropriate only for regression
cv_aucs_stratified = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")

print(f"Stratified 5-Fold CV AUC: {cv_aucs_stratified.mean():.4f} "
      f"± {cv_aucs_stratified.std():.4f}")

# Verify stratification: check class balance in each fold
print("\nClass distribution in each fold (verification):")
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    fold_count = np.bincount(y[test_idx])
    fold_pct   = fold_count / fold_count.sum() * 100
    print(f"  Fold {fold_idx+1}: Control={fold_count[0]} ({fold_pct[0]:.0f}%), "
          f"AD={fold_count[1]} ({fold_pct[1]:.0f}%)")
```

> **Why this matters biologically:** Alzheimer's disease datasets frequently have unequal group sizes because MCI patients are harder to recruit and controls are often over-represented in convenience samples. If your test fold happens to contain all MCI patients and no controls, your AUC for that fold is measuring something completely different from the other folds — the average across folds will be meaningless.

### 5.2.3 Leave-One-Out Cross-Validation (LOOCV)

LOOCV is the extreme case of k-fold where k = N (number of samples). Each iteration trains on N-1 samples and tests on exactly one sample. This is repeated N times, once for each sample.

**When to use LOOCV:**
- Very small datasets (N < 50): minimizes the amount of data withheld per iteration
- When you need every sample in training to avoid high-bias estimates
- Simple, computationally fast models (LOOCV with random forest on large datasets is slow)

**Limitations of LOOCV:**
- High variance: each fold shares N-1 training samples with every other fold, making fold results highly correlated
- For AUC estimation, this high correlation means the variance of the LOOCV-AUC estimate is typically larger than for k=10 CV despite using more data per fold
- Not appropriate for hyperparameter tuning (same nesting issue as flat k-fold)

```python
from sklearn.model_selection import LeaveOneOut

# LOOCV — use for small datasets (N < 50)
# With N = 148, LOOCV is computationally feasible but 5-fold or 10-fold is preferred

loo = LeaveOneOut()

# Collect predictions for ROC curve construction
# With LOOCV, each test set has one sample, so we collect probabilities iteratively
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Use logistic regression for LOOCV (faster than random forest)
pipeline_loocv = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

y_prob_loo = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_train_loo, X_test_loo = X[train_idx], X[test_idx]
    y_train_loo              = y[train_idx]
    pipeline_loocv.fit(X_train_loo, y_train_loo)
    y_prob_loo[test_idx] = pipeline_loocv.predict_proba(X_test_loo)[:, 1]

loo_auc = roc_auc_score(y, y_prob_loo)
print(f"LOOCV AUC (Logistic Regression): {loo_auc:.4f}")
print("Note: LOOCV AUC uses all predicted probabilities to build one ROC curve")
print("      This is NOT the same as averaging per-fold AUC (which is undefined for N=1 test)")
```

**The right tool for the right sample size:**

| Sample size (N) | Recommended CV strategy |
|-----------------|-------------------------|
| N < 50 | LOOCV |
| 50 ≤ N < 100 | 10-fold stratified CV |
| N ≥ 100 | 5-fold or 10-fold stratified CV |
| N ≥ 500 | Fixed train/validation/test split (70/15/15) |

### 5.2.4 Nested Cross-Validation for Unbiased Performance Estimation

This is the most important concept in this week. Misunderstanding this point is responsible for a large fraction of inflated performance estimates in the published biomarker literature.

**The problem:** Suppose you want to:
1. Select the best hyperparameters for your random forest (e.g., `n_estimators`, `max_features`, `min_samples_leaf`)
2. Estimate the AUC of the best model on held-out data

If you use the same CV loop for both purposes — selecting the best hyperparameters AND estimating performance — your performance estimate is **biased upward**. The test folds influenced which hyperparameters were selected. The selected hyperparameters are optimal for your specific dataset, including the test folds.

**Nested cross-validation** solves this by using two loops:
- **Outer loop:** Evaluates model performance (produces the reported AUC)
- **Inner loop:** Selects hyperparameters using only the outer training fold

The test fold in the outer loop is never touched during hyperparameter selection.

```
Outer loop (k=5):
  Outer fold 1: Test = samples 1-30  |  Outer train = samples 31-148
    Inner loop (k=5 within outer train):
      Inner fold 1.1: test = 31-53,   train = 54-148
      Inner fold 1.2: test = 54-76,   train = 31-53 + 77-148
      ...
      → Select best hyperparameters based on inner CV AUC
    → Retrain on outer train (31-148) with best hyperparameters
    → Evaluate on outer test (1-30) → outer AUC for fold 1

  Outer fold 2: Test = samples 31-60  |  Outer train = samples 1-30 + 61-148
    ...

Final reported AUC = mean of 5 outer fold AUCs
(Test samples were NEVER used in any hyperparameter selection)
```

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# ============================================================
# Nested cross-validation implementation
# ============================================================

# Define the model pipeline
# Scaling before any model is good practice even for tree-based models
# (it does not hurt trees and is required if you switch to logistic regression)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf",     RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Hyperparameter grid for the inner loop search
param_grid = {
    "rf__n_estimators":    [100, 200],
    "rf__max_features":    ["sqrt", "log2"],
    "rf__min_samples_leaf": [1, 3, 5],
    "rf__max_depth":       [None, 10, 20]
}

# Outer CV: 5-fold stratified (produces unbiased AUC estimate)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: 5-fold stratified (used for hyperparameter selection)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# GridSearchCV wraps the inner CV loop
grid_search = GridSearchCV(
    estimator  = pipeline,
    param_grid = param_grid,
    cv         = inner_cv,
    scoring    = "roc_auc",
    n_jobs     = -1,
    refit      = True       # refit on the full outer training fold with best params
)

# Run nested CV: outer loop with GridSearchCV as the estimator
# Each outer fold: fits inner GridSearchCV on outer train, evaluates on outer test
nested_cv_aucs = cross_val_score(
    grid_search,
    X, y,
    cv      = outer_cv,
    scoring = "roc_auc",
    n_jobs  = 1            # inner GridSearchCV already uses n_jobs=-1
)

print("Nested CV Results (5-outer x 5-inner):")
print(f"  Per-fold outer AUC: {np.round(nested_cv_aucs, 4)}")
print(f"  Mean outer AUC:     {nested_cv_aucs.mean():.4f}")
print(f"  Std outer AUC:      {nested_cv_aucs.std():.4f}")
print(f"\n  95% CI: [{nested_cv_aucs.mean() - 1.96*nested_cv_aucs.std():.4f}, "
      f"{nested_cv_aucs.mean() + 1.96*nested_cv_aucs.std():.4f}]")

# For comparison: flat CV AUC (biased if hyperparameters were tuned using same data)
flat_cv_auc = cross_val_score(
    RandomForestClassifier(n_estimators=200, max_features="sqrt",
                           min_samples_leaf=1, random_state=42, n_jobs=-1),
    X, y, cv=outer_cv, scoring="roc_auc"
)
print(f"\nFlat CV AUC (fixed hyperparameters): {flat_cv_auc.mean():.4f}")
print("(If nested CV AUC is notably lower, hyperparameter tuning was overfitting the folds)")
```

### 5.2.5 Visualizing Nested CV vs Flat CV

Understanding the architectural difference between these two approaches is essential for designing valid experiments.

```
FLAT CV (biased when hyperparameters are tuned):
┌──────────────────────────────────────────────────────┐
│  ALL 148 samples                                     │
│                                                      │
│  GridSearchCV finds best hyperparameters             │
│  using all 148 samples ← LEAKAGE                     │
│                                                      │
│  Same 148 samples used to evaluate performance       │
│  using k-fold ← Biased: test folds saw HP tuning     │
└──────────────────────────────────────────────────────┘

NESTED CV (unbiased):
┌──────────────────────────────────────────────────────┐
│  OUTER FOLD 1 TEST: samples 1–30 (NEVER touched in   │
│  HP tuning — completely held out)                    │
│                                                      │
│  OUTER FOLD 1 TRAIN: samples 31–148                 │
│  ┌────────────────────────────────────────────────┐  │
│  │ INNER CV on samples 31–148 only:               │  │
│  │   Inner fold 1: train 66–148, test 31–65       │  │
│  │   Inner fold 2: train 31–65+97–148, test 66–96 │  │
│  │   ...                                          │  │
│  │   → Best hyperparameters selected here         │  │
│  └────────────────────────────────────────────────┘  │
│  Retrain with best HPs on all 31–148                │
│  Evaluate on 1–30 → unbiased AUC for fold 1         │
└──────────────────────────────────────────────────────┘
Repeat for all 5 outer folds → 5 unbiased AUC estimates
```

> **Key insight for biologists:** The difference between nested and flat CV typically amounts to 0.02–0.08 AUC units in miRNA studies. In a field where AUC 0.75 vs 0.83 is the difference between a mediocre and a strong biomarker, this gap matters enormously. Use nested CV for all published results.

---

## MODULE 5.3 — Ensemble Methods: Gradient Boosting

### 5.3.1 Bagging vs Boosting: Two Philosophies of Ensemble Learning

In Week 4 you learned Random Forest, which is a **bagging** method. Gradient boosting is a **boosting** method. Both build ensembles of decision trees, but in fundamentally different ways.

**Bagging (Bootstrap Aggregating — Random Forest):**
- Trains many trees INDEPENDENTLY and IN PARALLEL
- Each tree sees a random bootstrap sample of the data (sampling with replacement)
- Each tree sees a random subset of features at each split
- Final prediction = average (or majority vote) across all trees
- Strong because: individual trees have high variance; averaging reduces variance

**Boosting (XGBoost, LightGBM, AdaBoost):**
- Trains trees SEQUENTIALLY
- Each new tree corrects the errors of the previous trees
- The ensemble progressively focuses on the "hard" samples that earlier trees misclassified
- Final prediction = weighted sum of all trees
- Strong because: individual trees have high bias; boosting reduces bias iteratively

**The key difference:** Random Forest reduces variance by averaging independent trees. Gradient Boosting reduces bias by iteratively correcting residual errors. On structured tabular data (like our miRNA expression matrix), boosting typically outperforms bagging because:
1. The residual-correction mechanism efficiently exploits patterns that single trees miss
2. Boosting can fit interactions between features that are invisible to independent trees
3. With L1/L2 regularization built in, boosting is less prone to overfitting than naive boosting

### 5.3.2 How Gradient Boosting Works

At iteration 0, start with a simple prediction (the class mean or log-odds of the training labels). At each subsequent iteration t:

1. Compute the **residuals** (or negative gradients of the loss function) — how wrong are the current predictions?
2. Fit a new small decision tree to predict those residuals
3. Add a scaled version of this tree to the ensemble: `F_t(x) = F_{t-1}(x) + learning_rate × tree_t(x)`

The **learning rate** (also called shrinkage, typically 0.01–0.3) controls how much each new tree contributes. Smaller learning rate = more conservative updates = needs more trees = slower training but better generalization.

The **number of trees** (n_estimators) controls how long boosting continues. More trees can improve performance but risk overfitting. **Early stopping** monitors performance on a held-out validation set and stops adding trees when performance stops improving — a practical and effective overfitting prevention strategy.

### 5.3.3 XGBoost for miRNA Classification

XGBoost (Extreme Gradient Boosting) adds L1 and L2 regularization on tree leaf weights, column (feature) subsampling similar to random forest, and other optimizations that make it highly competitive on tabular data.

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# XGBoost Classifier for AD vs Control miRNA data
# ============================================================

# Data preparation (X and y from Section 5.2)
# Scale features (not required for XGBoost but harmless and good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define XGBoost model
# Key hyperparameters explained:
#   n_estimators:    maximum number of boosting rounds (trees)
#   learning_rate:   shrinkage factor per tree (eta); smaller = more conservative
#   max_depth:       maximum depth of each tree; lower = less complex individual trees
#   min_child_weight: minimum sum of weights in a leaf; higher = more conservative splits
#   subsample:       fraction of samples used per tree (like bootstrap in RF)
#   colsample_bytree: fraction of features sampled per tree (like max_features in RF)
#   gamma:           minimum loss reduction to create a new split (regularization)
#   reg_alpha:       L1 regularization on leaf weights (drives sparsity)
#   reg_lambda:      L2 regularization on leaf weights (penalizes large weights)
#   scale_pos_weight: handles class imbalance = n_negative / n_positive

n_positive = y.sum()
n_negative = (y == 0).sum()
scale_pw   = n_negative / n_positive

xgb_model = xgb.XGBClassifier(
    n_estimators       = 300,
    learning_rate      = 0.05,
    max_depth          = 4,
    min_child_weight   = 3,
    subsample          = 0.8,
    colsample_bytree   = 0.6,
    gamma              = 0.1,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    scale_pos_weight   = scale_pw,
    use_label_encoder  = False,
    eval_metric        = "auc",
    random_state       = 42,
    n_jobs             = -1
)

# 5-fold nested CV for XGBoost
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_aucs = cross_val_score(xgb_model, X_scaled, y, cv=outer_cv, scoring="roc_auc")

print(f"XGBoost 5-Fold CV AUC: {xgb_cv_aucs.mean():.4f} ± {xgb_cv_aucs.std():.4f}")
```

### 5.3.4 Early Stopping to Prevent Overfitting

Early stopping monitors validation performance during training and halts when it stops improving. This is more principled than manually setting `n_estimators`.

```python
from sklearn.model_selection import train_test_split

# Split data for early stopping demonstration
# In a full nested CV, this split would be the inner train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost with early stopping
xgb_early = xgb.XGBClassifier(
    n_estimators       = 1000,    # high ceiling; early stopping will find optimal N
    learning_rate      = 0.05,
    max_depth          = 4,
    subsample          = 0.8,
    colsample_bytree   = 0.6,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    use_label_encoder  = False,
    eval_metric        = "auc",
    early_stopping_rounds = 30,  # stop if no improvement for 30 consecutive rounds
    random_state       = 42,
    n_jobs             = -1
)

xgb_early.fit(
    X_tr, y_tr,
    eval_set   = [(X_val, y_val)],
    verbose    = 50              # print training progress every 50 rounds
)

print(f"\nBest iteration: {xgb_early.best_iteration}")
print(f"Best validation AUC: {xgb_early.best_score:.4f}")

# Plot learning curve (training vs validation AUC per round)
results = xgb_early.evals_result()
train_auc = results["validation_0"]["auc"]

plt.figure(figsize=(9, 4))
plt.plot(train_auc, label="Validation AUC", color="steelblue")
plt.axvline(x=xgb_early.best_iteration, color="firebrick", linestyle="--",
            label=f"Best iteration ({xgb_early.best_iteration})")
plt.xlabel("Boosting round (n_estimators)")
plt.ylabel("AUC")
plt.title("XGBoost Early Stopping Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/xgb_learning_curve.png", dpi=150)
plt.show()
```

### 5.3.5 Feature Importance from Gradient Boosting

XGBoost provides three built-in feature importance measures:

- **weight:** Number of times a feature is used across all trees for splits
- **gain:** Average improvement in the loss function from each split involving this feature
- **cover:** Average number of samples affected by splits on this feature

**Gain** is the most biologically meaningful metric because it directly measures how much predictive power a miRNA contributes.

```python
# Feature importance: gain-based (recommended)
feature_names = [f"miRNA_{i}" for i in range(X.shape[1])]
# Replace with actual miRNA names from your expression matrix
# feature_names = list(expr_train.columns)

# Refit on full training data (for illustration)
xgb_final = xgb.XGBClassifier(
    n_estimators     = xgb_early.best_iteration + 1,
    learning_rate    = 0.05,
    max_depth        = 4,
    subsample        = 0.8,
    colsample_bytree = 0.6,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    use_label_encoder = False,
    eval_metric      = "auc",
    random_state     = 42,
    n_jobs           = -1
)
xgb_final.fit(X_scaled, y)

# Extract importance scores
importance_df = pd.DataFrame({
    "miRNA":     feature_names,
    "gain":      xgb_final.feature_importances_,  # default is 'weight'; use booster for gain
})

# For gain importance specifically:
booster    = xgb_final.get_booster()
gain_scores = booster.get_score(importance_type="gain")

importance_gain = pd.DataFrame({
    "miRNA":    list(gain_scores.keys()),
    "gain":     list(gain_scores.values())
}).sort_values("gain", ascending=False)

print("Top 20 miRNAs by XGBoost gain importance:")
print(importance_gain.head(20).to_string())

# Plot top 20 features
top20 = importance_gain.head(20)
plt.figure(figsize=(8, 6))
plt.barh(top20["miRNA"][::-1], top20["gain"][::-1], color="steelblue")
plt.xlabel("Gain (mean improvement in AUC per split)")
plt.title("XGBoost Feature Importance — Top 20 miRNAs (Gain)")
plt.tight_layout()
plt.savefig("results/xgb_feature_importance_gain.png", dpi=150)
plt.show()
```

> **Biological sidebar — why does gradient boosting often outperform random forest on miRNA data?**
>
> Random forests are "democratized" ensembles: every tree gets an equal vote regardless of quality, and each tree sees only a random subset of features. For miRNA data, where perhaps 5–10 miRNAs carry most of the disease signal while hundreds are noise, random forests can waste many trees on uninformative splits.
>
> Gradient boosting is "focused": it continuously asks "where am I making the most mistakes?" and builds the next tree specifically to correct those mistakes. After a few rounds, boosting homes in on the most informative miRNAs. The result is a more efficient use of the modeling capacity — fewer trees, better features, higher AUC.
>
> The caveat: boosting is more sensitive to hyperparameters and overfits faster if not regularized. Early stopping and the regularization parameters (`reg_alpha`, `reg_lambda`, `gamma`) are essential safeguards.

---

## MODULE 5.4 — SHAP Values for Model Interpretation

### 5.4.1 Why Feature Importance Alone Is Not Enough

The gain-based feature importance from Section 5.3.5 tells you which miRNAs are most frequently used by the model and contribute most to prediction accuracy. However, it does not tell you:
- Does high expression of this miRNA predict AD, or low expression?
- Is this miRNA's effect consistent across all patients, or does it only matter for a subset?
- For patient X specifically, which miRNAs drove the prediction toward AD?

**SHAP (SHapley Additive exPlanations)** values answer all of these questions. They provide a theoretically grounded, interpretable decomposition of every model prediction into contributions from each feature.

### 5.4.2 What Is a SHAP Value?

SHAP values are derived from cooperative game theory. In game theory, a Shapley value answers: "If multiple players contribute cooperatively to produce an outcome, what is each player's fair share of that outcome?"

In ML, the "players" are the model features (miRNAs), and the "outcome" is the model's prediction for a single patient. A SHAP value for feature j in patient i represents:

**"How much did miRNA j shift the model's prediction for patient i away from the average prediction across all patients?"**

Specifically, for a predicted log-odds output:
- SHAP value = 0: this miRNA had no effect on this patient's prediction
- SHAP value = +1.2: this miRNA pushed the prediction toward AD (increased log-odds by 1.2)
- SHAP value = -0.8: this miRNA pushed the prediction toward Control (decreased log-odds by 0.8)

The SHAP values for all features in a patient sum to (prediction - base value), where the base value is the model's average prediction across the training set.

### 5.4.3 Computing SHAP Values for XGBoost

```python
import shap
import matplotlib.pyplot as plt
import numpy as np

# Compute SHAP values for the XGBoost model
# TreeExplainer is the fast, exact implementation for tree-based models
explainer = shap.TreeExplainer(xgb_final)

# Compute SHAP values for all samples
# shap_values is a matrix: n_samples × n_features
# Each entry: SHAP contribution of that feature for that sample
shap_values = explainer.shap_values(X_scaled)

print(f"SHAP values matrix shape: {shap_values.shape}")
print(f"  (rows = samples, columns = features/miRNAs)")
print(f"\nBase value (average model output): {explainer.expected_value:.4f}")
print(f"Example patient 0 — top 3 contributing miRNAs:")

# Sort features by |SHAP value| for patient 0
patient_shap  = shap_values[0]
sorted_idx    = np.argsort(np.abs(patient_shap))[::-1]
for i in range(min(3, len(sorted_idx))):
    idx = sorted_idx[i]
    print(f"  {feature_names[idx]}: SHAP = {patient_shap[idx]:+.4f}")
```

### 5.4.4 The Beeswarm Plot: Global Model Interpretation

The beeswarm plot (also called the SHAP summary plot) is the single most informative visualization for understanding a model's overall behavior.

**How to read it:**
- Each dot represents one patient-miRNA combination
- The x-axis shows the SHAP value (positive = pushes toward AD, negative = pushes toward Control)
- The y-axis lists features ordered by mean |SHAP value| (most important at top)
- The color encodes the actual expression level of that miRNA in that patient (red = high expression, blue = low expression)

```python
# Beeswarm (summary) plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names = feature_names,
    max_display   = 20,
    show          = False,
    plot_type     = "dot"         # "dot" = beeswarm; "bar" = mean |SHAP| bar chart
)
plt.title("SHAP Beeswarm Plot — XGBoost AD vs Control\n"
          "(Top 20 miRNAs by mean |SHAP value|)", fontsize=12)
plt.tight_layout()
plt.savefig("results/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# Bar chart version (mean |SHAP| — simpler for publication)
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values,
    X_scaled,
    feature_names = feature_names,
    max_display   = 20,
    show          = False,
    plot_type     = "bar"
)
plt.title("SHAP Feature Importance (mean |SHAP value|) — Top 20 miRNAs")
plt.tight_layout()
plt.savefig("results/shap_bar_importance.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 5.4.5 Force Plot: Individual Patient Explanation

The force plot shows, for a single patient, exactly which miRNAs pushed the prediction toward AD (red arrows, positive SHAP) and which pushed it toward Control (blue arrows, negative SHAP).

```python
# Force plot for a single patient
# Choose a patient with high predicted probability of AD (interesting case)
predicted_probs = xgb_final.predict_proba(X_scaled)[:, 1]
high_ad_idx     = np.argmax(predicted_probs)

print(f"Patient {high_ad_idx}: predicted P(AD) = {predicted_probs[high_ad_idx]:.4f}, "
      f"true label = {'AD' if y[high_ad_idx]==1 else 'Control'}")

# SHAP force plot (interactive in Jupyter; static image here)
shap.initjs()   # initialize JavaScript for interactive display in Jupyter

# Force plot for one patient
force_plot = shap.force_plot(
    base_value    = explainer.expected_value,
    shap_values   = shap_values[high_ad_idx],
    features      = X_scaled[high_ad_idx],
    feature_names = feature_names,
    matplotlib    = True,   # static matplotlib version
    show          = False
)
plt.title(f"SHAP Force Plot — Patient {high_ad_idx} "
          f"(Predicted P(AD) = {predicted_probs[high_ad_idx]:.3f})")
plt.tight_layout()
plt.savefig("results/shap_force_plot_patient.png", dpi=150, bbox_inches="tight")
plt.show()

# SHAP dependence plot: how does one miRNA's SHAP value vary with its expression?
# Interaction coloring automatically finds the feature with strongest interaction
target_mirna = feature_names[sorted_idx[0]]   # top feature by mean |SHAP|
shap.dependence_plot(
    ind           = feature_names.index(target_mirna),
    shap_values   = shap_values,
    features      = X_scaled,
    feature_names = feature_names,
    show          = False
)
plt.title(f"SHAP Dependence Plot: {target_mirna}")
plt.tight_layout()
plt.savefig("results/shap_dependence_top_feature.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 5.4.6 SHAP vs Gain-Based Feature Importance: Which Should You Report?

| Property | Gain importance | SHAP values |
|----------|-----------------|-------------|
| Accounts for feature interactions | No | Yes |
| Direction of effect (up vs down in AD) | No | Yes |
| Patient-level explanations | No | Yes |
| Consistent across model types | No | Yes (SHAP is model-agnostic in principle) |
| Computationally expensive | Fast | Moderate (TreeExplainer is fast for tree models) |
| Appropriate for peer-reviewed biomarker papers | Acceptable | Preferred |

**Recommendation:** Report SHAP values in published work. Use gain importance for internal exploration and sanity checks.

> **Biological sidebar — What does a high SHAP value for miR-29b mean clinically?**
>
> Suppose your beeswarm plot shows that hsa-miR-29b-3p has the largest mean |SHAP value|, and the color gradient shows that LOW expression (blue) is associated with POSITIVE SHAP values (pushes toward AD prediction). This tells you:
>
> Low miR-29b-3p expression in serum predicts AD in your model.
>
> This is biologically plausible. miR-29b is a well-studied neuronal miRNA that regulates BACE1 (the beta-secretase enzyme responsible for amyloid precursor protein cleavage — the first step in Abeta production). Multiple studies have reported downregulation of miR-29 family members in blood and brain tissue of AD patients (Hebert et al., 2008 PNAS; Cogswell et al., 2008 J Alzheimers Dis). Low miR-29b → less repression of BACE1 → more Abeta production → AD pathology.
>
> The SHAP value connects a computational importance score to a specific mechanistic hypothesis. This is the kind of interpretation that makes ML-based biomarker studies scientifically meaningful rather than purely predictive.

---

## MODULE 5.5 — Three-Class Classification: AD vs MCI vs Control

### 5.5.1 Why MCI Makes Classification Harder

In Week 4, we simplified the problem by training binary classifiers (AD vs Control). In clinical reality, MCI (Mild Cognitive Impairment) is a critical class: these are the patients most likely to benefit from early intervention, and they are the population where biomarker-guided treatment decisions would have the greatest impact.

MCI is biologically ambiguous because:
- MCI represents a heterogeneous prodromal stage: some MCI patients progress to AD; others remain stable; a few revert to normal cognition
- The miRNA profiles of MCI patients partially overlap with both AD and Control
- "Amnestic MCI" (most likely to progress to AD) has a different profile from "non-amnestic MCI"

**Expected consequence:** Your model will have lower performance for MCI samples than for AD vs Control. This is not a failure of the model. It reflects genuine biological uncertainty. A model that reports 90% accuracy for MCI is almost certainly overfitting.

### 5.5.2 Multiclass Classification Approaches

**One-vs-Rest (OvR):** Train three binary classifiers: (AD vs not-AD), (MCI vs not-MCI), (Control vs not-Control). For a new patient, apply all three classifiers and assign the class with the highest predicted probability.

**Multinomial (Softmax):** Train a single model that directly outputs probabilities for all three classes simultaneously, with the constraint that probabilities sum to 1. XGBoost and most sklearn estimators support this natively with `objective="multi:softprob"`.

For miRNA data, both approaches perform similarly. Multinomial is generally preferred because it uses all class information simultaneously during training.

```python
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Three-class classification: AD, MCI, Control
# ============================================================

# Include MCI samples
mask_3class = meta_train["group"].isin(
    ["Alzheimer's Disease", "Mild Cognitive Impairment", "Control"]
)
X_3class = expr_train.loc[mask_3class].values
groups_3  = meta_train.loc[mask_3class, "group"].values

# Encode labels to integers
le = LabelEncoder()
le.fit(["Control", "Mild Cognitive Impairment", "Alzheimer's Disease"])
y_3class = le.transform(groups_3)
# Control=0, Mild Cognitive Impairment=1, Alzheimer's Disease=2

print("Three-class label distribution:")
for cls, n in zip(le.classes_, np.bincount(y_3class)):
    print(f"  {cls}: n={n}")

# XGBoost multiclass classifier
xgb_3class = xgb.XGBClassifier(
    n_estimators       = 300,
    learning_rate      = 0.05,
    max_depth          = 4,
    subsample          = 0.8,
    colsample_bytree   = 0.6,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    objective          = "multi:softprob",
    num_class          = 3,
    eval_metric        = "mlogloss",
    use_label_encoder  = False,
    random_state       = 42,
    n_jobs             = -1
)

# 5-fold stratified CV for three-class problem
skf_3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For multiclass AUC, we need to collect predictions manually
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

scaler_3    = StandardScaler()
X_3_scaled  = scaler_3.fit_transform(X_3class)

all_y_test  = []
all_y_prob  = []

for fold_idx, (train_idx, test_idx) in enumerate(skf_3.split(X_3_scaled, y_3class)):
    X_tr3 = X_3_scaled[train_idx]
    y_tr3 = y_3class[train_idx]
    X_te3 = X_3_scaled[test_idx]
    y_te3 = y_3class[test_idx]

    xgb_3class.fit(X_tr3, y_tr3)
    y_prob3 = xgb_3class.predict_proba(X_te3)    # shape: n_test × 3

    all_y_test.append(y_te3)
    all_y_prob.append(y_prob3)

y_test_all = np.concatenate(all_y_test)
y_prob_all = np.vstack(all_y_prob)

# Macro AUC: average AUC across all 3 classes (treats all classes equally)
macro_auc = roc_auc_score(
    label_binarize(y_test_all, classes=[0, 1, 2]),
    y_prob_all,
    multi_class = "ovr",
    average     = "macro"
)

# Weighted AUC: weights each class's AUC by its sample size
weighted_auc = roc_auc_score(
    label_binarize(y_test_all, classes=[0, 1, 2]),
    y_prob_all,
    multi_class = "ovr",
    average     = "weighted"
)

print(f"\nThree-class XGBoost — 5-fold CV:")
print(f"  Macro AUC (unweighted):   {macro_auc:.4f}")
print(f"  Weighted AUC:             {weighted_auc:.4f}")
print(f"\nNote: Macro AUC gives equal weight to MCI even if MCI N is small.")
print(f"      Weighted AUC is less affected by small classes.")
print(f"      Report BOTH in papers alongside per-class AUC.")
```

### 5.5.3 Confusion Matrix for Three Classes

```python
# Confusion matrix
y_pred_3class = np.argmax(y_prob_all, axis=1)

cm = confusion_matrix(y_test_all, y_pred_3class)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

# Plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)\nAD vs MCI vs Control")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Percentage
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1])
axes[1].set_title("Confusion Matrix (Row %)\nAD vs MCI vs Control")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig("results/confusion_matrix_3class.png", dpi=150)
plt.show()

# Per-class classification report
print("\nPer-class classification report:")
print(classification_report(y_test_all, y_pred_3class,
                             target_names=le.classes_))
```

> **Biological interpretation of MCI misclassification:** In a typical confusion matrix for this three-class problem, you will see that MCI samples are misclassified as either AD or Control at much higher rates than AD or Control samples are misclassified as each other. This is not a model deficiency — it accurately reflects biological reality. MCI patients are biologically intermediate: some have early amyloid pathology indistinguishable from AD; others have mild, non-specific decline. A model that correctly classifies MCI 55% of the time but AD and Control 85–90% of the time is performing appropriately given the biology. The MCI classification performance tells you something about the biological distinctiveness of MCI as a molecular state — and in this case, the answer is that miRNA alone captures only part of the MCI signature. Combined biomarkers (miRNA + CSF Abeta/tau, or miRNA + PET imaging) would likely improve MCI classification substantially.

---

## MODULE 5.6 — External Validation: Why It Matters

### 5.6.1 The Replication Crisis in Biomarker ML

A sobering statistic: a systematic review by Bzdok & Ioannidis (2017) estimated that more than 85% of published "predictive biomarker" studies based on omics data do not include external validation on an independent cohort. Among studies that do attempt external validation, a significant fraction show AUC drops of 0.10–0.25 relative to the training/internal validation estimate.

Why does this happen? Several compounding factors:
- **Overfitting** to the training cohort's specific demographics, sample processing pipeline, and platform characteristics
- **Platform-specific technical artifacts** mistaken for biological signal
- **Cohort-specific confounders** (e.g., medication profiles, geographic ancestry, comorbidities) that differ between cohorts
- **Small training N** producing high-variance models that fitted noise

External validation is not just "a good idea" — it is the minimum standard of evidence required for clinical translation of a biomarker. The FDA's Biomarker Qualification Guidance explicitly requires independent validation before a biomarker can be used in drug development.

### 5.6.2 Training AUC vs Validation AUC: Interpreting the Gap

When you apply your model trained on GSE120584 to the independent cohort GSE46579, you will observe an AUC gap:

```
Training CV AUC (GSE120584):    e.g., 0.88
External Validation AUC (GSE46579): e.g., 0.73
Gap:                            0.15
```

How should you interpret the size of this gap?

| Gap size | Interpretation |
|----------|---------------|
| < 0.05 | Excellent generalizability. Model likely learned true biology. Publishable with appropriate caveats. |
| 0.05–0.10 | Acceptable. Some platform/cohort effects present but model retains predictive value. |
| 0.10–0.20 | Moderate degradation. Likely a combination of overfitting and platform/cohort differences. Report honestly. Investigate what drives the loss. |
| > 0.20 | Severe degradation. Model has likely overfit to training cohort. Feature harmonization and retraining may help. |
| Validation AUC ≤ 0.55 | Model has failed external validation. Report as a negative result and investigate root cause. |

> **Biological sidebar — What does a 15% AUC drop in external validation mean biologically?**
>
> Suppose your model achieves AUC = 0.88 on GSE120584 (serum RNA-seq) but drops to AUC = 0.73 on GSE46579 (whole blood microarray). Several biological and technical explanations are possible:
>
> 1. **Platform difference:** Serum versus whole blood captures different miRNA compartments. Serum is depleted of cellular miRNAs and enriched for exosome-packaged miRNAs. Whole blood includes miRNAs from leukocytes. Some miRNAs differentially expressed in serum-AD may not be differentially expressed (or may not be detected) in whole blood.
>
> 2. **Cohort heterogeneity:** GSE46579 was collected in a different country, with different AD diagnostic criteria (perhaps including more mild-stage patients), different comorbidity profiles, and different blood processing protocols.
>
> 3. **Overfitting in training:** Your model learned some features that are informative in the training cohort but are technical artifacts (e.g., platform-specific background noise patterns that correlate with disease group due to batch effects that were imperfectly corrected).
>
> 4. **True biological difference:** Different AD subtypes may have different miRNA profiles, and the cohorts may have different proportions of subtypes.
>
> The most productive response to a 15% drop is not to dismiss the result but to investigate it: examine which features have the highest SHAP values in the training set, check whether those same miRNAs are detected in the validation platform, and test whether a model retrained only on features present in both platforms narrows the gap.

### 5.6.3 The GSE46579 Validation Dataset

**GSE46579 characteristics:**
- Platform: GPL16384 (Affymetrix GeneChip miRNA 3.0 Array)
- Sample type: Whole blood
- Groups: AD (n=35), Controls (n=30)
- Total samples: 65
- Key difference from GSE120584: microarray vs RNA-seq; whole blood vs serum; different N

**The cross-platform challenge:** Affymetrix miRNA 3.0 arrays use probe names derived from the miRBase version current at the time of array design (approximately miRBase v14–v16). GSE120584 RNA-seq data uses miRNA names from the alignment reference used (often miRBase v20–v22). Many miRNA names changed between versions — hyphens, strand designations (*-3p vs * -5p), and mature vs precursor designations all changed.

Before any cross-platform comparison, miRNA names must be harmonized to the same version. Module 5.7 covers this in detail.

---

## MODULE 5.7 — Harmonizing Datasets for Cross-Platform Validation

### 5.7.1 The miRNA Name Problem

miRBase is the authoritative registry for miRNA names and sequences. It has gone through 22 major versions since 2002. Each version changed names for a subset of miRNAs:
- Version 6 → 12: Major restructuring of naming conventions
- Version 14 → 16: Strand designation introduced (-3p and -5p suffixes replaced * notation)
- Version 17 → 21: Many isomiR and precursor changes
- Version 21 → 22: Removal of some miRNAs later identified as non-canonical

**Practical consequence:** The miRNA named "hsa-miR-21*" in an older array annotation corresponds to "hsa-miR-21-3p" in miRBase v22. If you try to intersect features across your two datasets without harmonization, you will lose a significant fraction of your shared miRNAs — not because they are absent from one platform, but because they are named differently.

The R package `miRBaseConverter` resolves this by converting miRNA names between any two miRBase versions using the miRBase accession number (MIMAT) as a stable identifier.

### 5.7.2 miRBaseConverter Name Harmonization in R

The full harmonization workflow is implemented in the companion R script (`Week5_Validation.R`). Here is the conceptual walkthrough:

```r
# Conceptual R code — full implementation in Week5_Validation.R
library(miRBaseConverter)

# 1. Convert all miRNA names in GSE120584 to miRBase v22
# checkMiRNAVersion() tells you the likely version of your name format
version_check <- checkMiRNAVersion(rownames(expr_gse120584), verbose = TRUE)

# 2. Convert names
result_120584 <- miRNA_NameToAccession(rownames(expr_gse120584),
                                        version = "v22")
# Returns: original name → MIMAT accession → v22 canonical name

# 3. Same for GSE46579
result_46579 <- miRNA_NameToAccession(rownames(expr_gse46579),
                                       version = "v22")

# 4. Find the intersection of MIMAT accessions (version-independent)
common_MIMAT <- intersect(result_120584$Accession,
                          result_46579$Accession)
cat("miRNAs in common after harmonization:", length(common_MIMAT), "\n")

# 5. Subset both expression matrices to the intersection
expr_120584_sub <- expr_gse120584[result_120584$Accession %in% common_MIMAT, ]
expr_46579_sub  <- expr_gse46579[result_46579$Accession  %in% common_MIMAT, ]

# 6. Rename rows to v22 canonical names
rownames(expr_120584_sub) <- result_120584$v22Name[
    result_120584$Accession %in% common_MIMAT]
rownames(expr_46579_sub)  <- result_46579$v22Name[
    result_46579$Accession  %in% common_MIMAT]
```

### 5.7.3 Per-Dataset Z-Score Standardization

After finding the intersection, the expression values are on completely different scales: GSE120584 is VST-transformed RNA-seq counts; GSE46579 is RMA-normalized microarray intensities. You cannot train on one and test on the other without removing this scale difference.

**Z-score standardization per dataset** resolves this by converting each dataset independently to have mean = 0 and standard deviation = 1 per miRNA:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# Z-score standardization per dataset
# (Done separately for training and validation datasets)
# ============================================================

# Load harmonized matrices exported from R
X_train_raw = pd.read_csv("data/processed/GSE120584_harmonized.csv", index_col=0)
X_val_raw   = pd.read_csv("data/processed/GSE46579_harmonized.csv",  index_col=0)

meta_train_h = pd.read_csv("data/processed/GSE120584_metadata_harmonized.csv", index_col=0)
meta_val_h   = pd.read_csv("data/processed/GSE46579_metadata_harmonized.csv",  index_col=0)

# Verify the same features appear in both datasets
assert list(X_train_raw.columns) == list(X_val_raw.columns), \
    "Feature names do not match — check harmonization step!"
print(f"Shared miRNA features: {X_train_raw.shape[1]}")

# Z-score EACH DATASET INDEPENDENTLY
# Rationale: we do not want to shift the validation data
#            toward the training data's mean — that would be
#            a form of data leakage. Each dataset is standardized
#            to have mean=0, SD=1 per miRNA WITHIN that dataset.
#            The biological signal (relative differences between
#            AD and Control within each dataset) is preserved.
#            The platform-specific absolute scale is removed.

scaler_train = StandardScaler()
X_train_z    = scaler_train.fit_transform(X_train_raw.values)
X_train_z    = pd.DataFrame(X_train_z,
                             index   = X_train_raw.index,
                             columns = X_train_raw.columns)

scaler_val   = StandardScaler()   # SEPARATE scaler for validation
X_val_z      = scaler_val.fit_transform(X_val_raw.values)
X_val_z      = pd.DataFrame(X_val_z,
                             index   = X_val_raw.index,
                             columns = X_val_raw.columns)

print(f"\nTraining matrix (z-scored): {X_train_z.shape}")
print(f"  Mean per miRNA: {X_train_z.values.mean():.4f} (should be ~0)")
print(f"  SD per miRNA:   {X_train_z.values.std():.4f} (should be ~1)")

print(f"\nValidation matrix (z-scored): {X_val_z.shape}")
print(f"  Mean per miRNA: {X_val_z.values.mean():.4f} (should be ~0)")
print(f"  SD per miRNA:   {X_val_z.values.std():.4f} (should be ~1)")
```

> **Why standardize each dataset separately?** Consider: miR-21-5p might have a mean VST expression of 12.4 in GSE120584 but a mean RMA intensity of 8.7 in GSE46579. These numbers are not comparable — they are on completely different scales due to the different measurement technologies. After per-dataset z-scoring, both datasets will express miR-21-5p as deviations from its own dataset's mean. A sample with high miR-21-5p expression relative to the rest of GSE120584 will have a positive z-score, as will a sample with high miR-21-5p relative to the rest of GSE46579. Relative (within-dataset) differences are preserved; absolute (across-dataset) scale differences are removed.

### 5.7.4 External Validation: Train on GSE120584, Apply to GSE46579

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# Prepare training labels
y_train_ext = (meta_train_h["group"] == "Alzheimer's Disease").astype(int).values

# Prepare validation labels (binary: AD vs Control for external validation)
mask_val_binary = meta_val_h["group"].isin(["Alzheimer's Disease", "Control"])
X_val_binary    = X_val_z.loc[mask_val_binary].values
y_val_binary    = (meta_val_h.loc[mask_val_binary, "group"] ==
                   "Alzheimer's Disease").astype(int).values

print(f"Training samples: {X_train_z.shape[0]}")
print(f"Validation samples (binary): {X_val_binary.shape[0]}")

# Train final model on ALL training data
n_pos_tr = y_train_ext.sum()
n_neg_tr = (y_train_ext == 0).sum()

xgb_final_ext = xgb.XGBClassifier(
    n_estimators       = 300,
    learning_rate      = 0.05,
    max_depth          = 4,
    subsample          = 0.8,
    colsample_bytree   = 0.6,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    scale_pos_weight   = n_neg_tr / n_pos_tr,
    use_label_encoder  = False,
    eval_metric        = "auc",
    random_state       = 42,
    n_jobs             = -1
)
xgb_final_ext.fit(X_train_z.values, y_train_ext)

# Predict on external validation cohort
y_val_prob = xgb_final_ext.predict_proba(X_val_binary)[:, 1]

# External validation AUC
val_auc = roc_auc_score(y_val_binary, y_val_prob)
print(f"\nExternal Validation AUC (GSE46579): {val_auc:.4f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_val_binary, y_val_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="firebrick", lw=2,
         label=f"External Validation (AUC = {val_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — External Validation on GSE46579\n"
          "(Model trained on GSE120584)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_external_validation.png", dpi=150)
plt.show()

# Compare training CV AUC vs external validation AUC
print("\n=== AUC Comparison ===")
print(f"Training CV AUC (GSE120584, nested):  [from Section 5.2.4]")
print(f"External Validation AUC (GSE46579):   {val_auc:.4f}")
print(f"AUC gap:                              [Training_AUC - {val_auc:.4f}]")
```

### 5.7.5 ComBat-seq for Batch Correction Between Studies

For datasets where the sample types are compatible (e.g., both serum), ComBat-seq can be applied to remove cross-study batch effects before combining datasets. This is more aggressive than per-dataset z-scoring and should be used cautiously.

The full ComBat-seq workflow is implemented in `Week5_Validation.R`, Section 7. Key considerations:
- ComBat-seq requires raw count data (not VST/log-transformed)
- The `group` variable (AD vs Control) must be specified as a biological covariate to protect it from removal
- After ComBat-seq, re-run VST transformation and re-check PCA for remaining batch structure
- If the two datasets differ in sample type (serum vs whole blood), ComBat-seq will correct technical differences but cannot resolve the genuine biological difference in miRNA composition between sample types

---

## MODULE 5.8 — Reporting ML Results: What Makes a Good Paper

### 5.8.1 TRIPOD: Transparent Reporting of a Multivariable Prediction Model

TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) is an international reporting guideline specifically designed for studies that develop or validate prediction models. It was published in the Annals of Internal Medicine (Collins et al., 2015) and has since been adopted by most major clinical journals.

**Key TRIPOD items for miRNA ML biomarker studies:**

| TRIPOD Item | What to Report |
|-------------|---------------|
| 4a | Source of data (GEO accession, sample type, collection dates) |
| 4b | Eligibility criteria for participants (diagnosis method, exclusions) |
| 5 | Outcome to be predicted (AD diagnosis, conversion from MCI) |
| 7 | Sample size justification (power calculation or minimum N for stable AUC) |
| 8 | How missing data were handled |
| 10a | Predictors used and how they were selected/transformed |
| 10b | How features were handled (normalization, filtering) |
| 12 | Model development details: algorithm, hyperparameters, selection method |
| 13b | Internal validation method: k-fold, LOOCV, nested CV |
| 15 | Model performance: discrimination (AUC), calibration, decision curve analysis |
| 16 | For validation studies: description of validation data, sources of difference from development data |
| 17 | Comparison of development vs validation performance |
| 22 | Limitations: potential bias, sample size, generalizability |

### 5.8.2 What Metrics to Report

A complete reporting of a miRNA biomarker ML model should include:

**Discrimination:**
- AUC (with 95% confidence interval)
- Sensitivity and specificity at the operating threshold
- Positive Predictive Value (PPV) and Negative Predictive Value (NPV) at the operating threshold
- Youden index (optimal threshold selection)

**Calibration:**
- Calibration plot (observed vs predicted probability)
- Hosmer-Lemeshow test (for binary classifiers)

**Validation:**
- Internal CV AUC (from nested CV)
- External validation AUC (if independent cohort available)
- DeLong's test p-value for comparing AUCs (see Section 5.8.3)

**Feature reporting:**
- Top features (miRNAs) with effect direction (SHAP values, not just importance ranks)
- Biological annotation of top features

### 5.8.3 Confidence Intervals on AUC and DeLong's Test

A single AUC value without a confidence interval is incomplete. The most common method for CI computation is **bootstrap resampling**. For comparing two AUC values (e.g., training CV vs external validation, or Model A vs Model B), **DeLong's method** provides an analytic test based on the covariance of correlated ROC curves.

```python
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd

# ============================================================
# Bootstrap confidence interval for AUC
# ============================================================

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, ci_level=0.95, seed=42):
    """Compute bootstrap confidence interval for AUC."""
    rng      = np.random.default_rng(seed)
    n        = len(y_true)
    boot_aucs = []

    for _ in range(n_bootstrap):
        idx       = rng.integers(0, n, size=n)   # sample with replacement
        y_boot    = y_true[idx]
        p_boot    = y_prob[idx]

        # Need both classes represented
        if len(np.unique(y_boot)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_boot, p_boot))

    boot_aucs = np.array(boot_aucs)
    alpha     = (1 - ci_level) / 2

    ci_lower = np.percentile(boot_aucs, alpha * 100)
    ci_upper = np.percentile(boot_aucs, (1 - alpha) * 100)
    point_est = roc_auc_score(y_true, y_prob)

    return point_est, ci_lower, ci_upper, boot_aucs

# Compute CI for external validation AUC
auc_val, ci_lo, ci_hi, boot_dist = bootstrap_auc_ci(y_val_binary, y_val_prob)
print(f"External Validation AUC: {auc_val:.4f} (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")

# Visualize bootstrap distribution
plt.figure(figsize=(7, 4))
plt.hist(boot_dist, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
plt.axvline(auc_val, color="firebrick", lw=2, label=f"Observed AUC = {auc_val:.4f}")
plt.axvline(ci_lo, color="orange", lw=1.5, linestyle="--",
            label=f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
plt.axvline(ci_hi, color="orange", lw=1.5, linestyle="--")
plt.xlabel("Bootstrap AUC")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of External Validation AUC")
plt.legend()
plt.tight_layout()
plt.savefig("results/bootstrap_auc_distribution.png", dpi=150)
plt.show()
```

**DeLong's test** for comparing two AUCs is implemented in the companion R script (`Week5_Validation.R`, Section 6) using the `pROC` package. The DeLong method accounts for the fact that two AUCs computed on the same patients are correlated — a naive comparison of independent proportions would be incorrect.

```r
# DeLong AUC comparison in R — see Week5_Validation.R for full implementation
library(pROC)

# Read in predicted probabilities (saved from Python)
pred_train <- read.csv("results/training_cv_predictions.csv")
pred_val   <- read.csv("results/validation_predictions.csv")

roc_train <- roc(pred_train$true_label, pred_train$prob_AD)
roc_val   <- roc(pred_val$true_label,   pred_val$prob_AD)

# DeLong test (for overlapping vs non-overlapping cohorts, method differs)
# For non-overlapping independent cohorts (our case):
delong_result <- roc.test(roc_train, roc_val, method = "delong")
print(delong_result)
# If p < 0.05: the two AUCs are statistically different
# If p ≥ 0.05: cannot conclude the model performs differently on the two cohorts
```

### 5.8.4 Calibration Plots

A classifier with high AUC but poor calibration is clinically dangerous. Calibration measures whether predicted probabilities are accurate: if the model says P(AD) = 0.8 for a group of patients, approximately 80% of those patients should truly have AD.

Calibration is assessed with a **calibration plot** (reliability diagram): group patients by their predicted probability into deciles, then plot mean predicted probability (x-axis) vs observed event rate (y-axis). A perfectly calibrated model would fall on the diagonal.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Calibration plot for external validation
prob_true, prob_pred = calibration_curve(
    y_val_binary, y_val_prob,
    n_bins   = 10,       # 10 bins along the probability axis
    strategy = "quantile"  # equal-frequency bins
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, "o-", color="steelblue", lw=2,
         label=f"XGBoost (External Val.)")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
plt.xlabel("Mean Predicted Probability (P(AD))")
plt.ylabel("Observed Event Rate (Fraction True AD)")
plt.title("Calibration Plot — External Validation (GSE46579)")
plt.legend()
plt.tight_layout()
plt.savefig("results/calibration_plot_external.png", dpi=150)
plt.show()

# Quantitative calibration metric (Brier score)
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_val_binary, y_val_prob)
print(f"Brier Score (external validation): {brier:.4f}")
print("(Lower is better; 0 = perfect; 0.25 = uninformative model)")
```

### 5.8.5 Decision Curve Analysis

Decision Curve Analysis (DCA) evaluates whether a model provides clinical benefit compared to the baseline strategies of "treat everyone" or "treat no one." It accounts for the relative cost of false positives vs false negatives (the threshold probability below which you would treat).

While full DCA implementation is beyond this module's scope, the key principle is: a model with AUC = 0.85 that is poorly calibrated may provide LESS clinical benefit than a simpler model with AUC = 0.78 but excellent calibration. DCA plots net benefit as a function of threshold probability.

### 5.8.6 The STARD Checklist

STARD (Standards for Reporting of Diagnostic Accuracy) is the companion guideline to TRIPOD specifically for diagnostic test accuracy studies. It includes 30 items organized in four sections: title/abstract, introduction, methods, and results. For miRNA biomarker studies that include validation against a clinical diagnostic standard (CDR, MMSE, amyloid PET), STARD compliance is required by most diagnostics-focused journals.

Key STARD items for miRNA biomarker reporting:
- How was the reference standard (AD diagnosis) established? (Clinical criteria, biomarker-confirmed, post-mortem pathology)
- Were the index test (miRNA model) and reference standard interpreted without knowledge of each other? (Avoidance of review bias)
- Was there a pre-specified threshold or was it data-derived? (If data-derived, report the derivation process)
- What is the time interval between sample collection and clinical diagnosis?

---

## WEEK 5 LAB SESSIONS

### Lab 5A: Nested Cross-Validation and SHAP Analysis in Python (90 min)

**Objective:** Implement a complete, leakage-free nested CV pipeline with SHAP interpretation.

**Step-by-step:**

1. **Setup (10 min):** Load the harmonized GSE120584 training data
   ```python
   X = pd.read_csv("data/processed/GSE120584_harmonized.csv", index_col=0)
   meta = pd.read_csv("data/processed/GSE120584_metadata_harmonized.csv", index_col=0)
   # Subset to AD vs Control binary
   mask = meta["group"].isin(["Alzheimer's Disease", "Control"])
   X_bin = X.loc[mask].values
   y_bin = (meta.loc[mask, "group"] == "Alzheimer's Disease").astype(int).values
   ```

2. **Flat CV baseline (15 min):** Run a 5-fold stratified CV with fixed hyperparameters. Record AUC.

3. **Nested CV (30 min):** Implement nested CV using GridSearchCV as the inner estimator. Use at least 2 hyperparameters in the grid. Compare nested CV AUC to flat CV AUC. Are they different? By how much?

4. **XGBoost with early stopping (15 min):** Split off a 20% validation set from the training data. Train XGBoost with `early_stopping_rounds=30`. Record the optimal number of rounds.

5. **SHAP analysis (20 min):**
   - Refit the best model on all training data
   - Compute SHAP values with `shap.TreeExplainer`
   - Produce a beeswarm plot
   - Identify the top 5 miRNAs by mean |SHAP value|
   - For each: note the direction of effect (high expression → AD or Control?)
   - Look up each miRNA in miRBase (https://www.mirbase.org) and note its target gene families

**Deliverable:** A table with columns: `miRNA`, `mean_abs_SHAP`, `direction_in_AD`, `known_targets`, `biological_plausibility`.

**Reflection questions:**
- Was the nested CV AUC higher or lower than flat CV? Is this the expected direction?
- Which miRNA has the highest SHAP value, and does its direction of effect match published literature?
- What would a SHAP force plot for a patient classified as AD with high confidence look like, versus a patient classified near the decision boundary?

---

### Lab 5B: External Validation with GSE46579 (90 min)

**Objective:** Conduct a complete external validation analysis including harmonization, standardization, and AUC comparison.

**Part 1 — R: Data harmonization (45 min)**

Run `Week5_Validation.R` sections 1–5:
1. Load both datasets from `data/processed/`
2. Run miRBaseConverter harmonization. Note: how many miRNA names changed? How many features were lost at the intersection step?
3. Check that key AD-associated miRNAs (miR-21-5p, miR-29b-3p, miR-155-5p, miR-9-5p) are present in the intersection
4. Apply per-dataset z-score standardization
5. Export harmonized CSVs to `data/processed/`

**Part 2 — Python: Cross-platform prediction (45 min)**

1. Load the harmonized CSVs from Part 1
2. Confirm feature alignment between train and validation datasets
3. Train the best XGBoost model (from Lab 5A) on all GSE120584 harmonized data
4. Apply to GSE46579 harmonized data
5. Compute external validation AUC with 95% bootstrap CI
6. Plot ROC curve comparing training CV AUC vs validation AUC
7. Compute the calibration plot for external validation

**Deliverable:** A summary table:
| Model | Training CV AUC (95% CI) | Ext. Validation AUC (95% CI) | AUC Gap |
|-------|--------------------------|------------------------------|---------|
| XGBoost | ? ± ? | ? ± ? | ? |
| Random Forest | ? ± ? | ? ± ? | ? |
| Logistic Regression | ? ± ? | ? ± ? | ? |

**Reflection questions:**
- Which model has the smallest AUC gap between training and external validation? Is this the same model with the highest training CV AUC?
- What fraction of miRNA features were retained in the intersection? Which biologically important AD miRNAs were lost (if any)?
- If you were preparing this analysis for a journal submission, what would you report in the limitations section about the cross-platform validation?

---

## WEEK 5 ASSIGNMENTS

### Reading Assignment

1. **Collins GS et al. (2015)** — TRIPOD Statement: A Set of Recommendations for Reporting of Studies Developing, Validating, or Updating a Multivariable Clinical Prediction Model. *Ann Intern Med* 162(1):55–63. [DOI: 10.7326/M14-0698](https://doi.org/10.7326/M14-0698)
   Focus on: TRIPOD checklist items 10–17 (model development and validation sections)

2. **Bossuyt PM et al. (2015)** — STARD 2015: An Updated List of Essential Items for Reporting Diagnostic Accuracy Studies. *BMJ* 351:h5527. [DOI: 10.1136/bmj.h5527](https://doi.org/10.1136/bmj.h5527)
   Focus on: items related to reference standard and index test definition

3. **Xu Z et al. (2022)** — Machine learning and complex biological data. *Genome Biol* 23:197. [DOI: 10.1186/s13059-022-02754-3](https://doi.org/10.1186/s13059-022-02754-3)
   Focus on: the discussion of overfitting, evaluation strategies, and reproducibility challenges in omics ML

### Reflection Questions

1. You run a nested CV and get mean outer AUC = 0.82. Your collaborator runs a flat CV (hyperparameters tuned on the same data) and reports AUC = 0.91. How would you explain the discrepancy to a clinician who is excited about the 0.91 number? What specific leakage form does flat CV introduce?

2. Your model's SHAP analysis reveals that the top predictive miRNA is hsa-miR-486-5p, which is known to be highly abundant in red blood cells. Your training data comes from serum (depleted of RBCs), but the model assigns it high importance. What does this suggest about the data, and what should you investigate?

3. Your external validation AUC drops from 0.85 to 0.62 (a 0.23 gap). You examine the feature list and find that 40% of the top-30 model features are not present in the validation platform. What are two strategies you could employ to improve generalizability, and what are the biological trade-offs of each?

4. A colleague argues that calibration is irrelevant for a screening test — "all that matters is AUC." Construct a biological scenario where a high-AUC, poorly-calibrated model would lead to patient harm in an MCI screening context.

### Practical Exercise

Using the summary results table from Lab 5B, write a "Results" paragraph (4–6 sentences) as you would in a journal paper. Include: the training CV AUC with 95% CI, the external validation AUC with 95% CI, the DeLong test result comparing the two, the top-3 miRNAs by SHAP value with their effect directions, and one sentence about calibration. Follow TRIPOD item 15 reporting standards.

---

## WEEK 5 GLOSSARY

| Term | Definition |
|------|------------|
| **Overfitting** | When a model captures the noise patterns specific to the training data and performs substantially worse on new data |
| **Bias-variance tradeoff** | The inverse relationship between a model's error from overly rigid assumptions (bias) and its error from sensitivity to training data specifics (variance) |
| **Data leakage** | Any flow of information from the test or validation set into model training or feature selection, causing inflated performance estimates |
| **Nested cross-validation** | A CV design with two loops: the outer loop estimates unbiased performance; the inner loop selects hyperparameters using only the outer training fold |
| **Stratified k-fold CV** | A k-fold CV variant that ensures each fold contains the same class proportions as the full dataset; required for imbalanced classification problems |
| **LOOCV** | Leave-One-Out Cross-Validation; k = N; each sample serves as a test set exactly once; preferred for very small datasets |
| **Gradient boosting** | An ensemble method that builds trees sequentially, each correcting the residuals of the previous; tends to outperform random forests on tabular biological data |
| **Learning rate** | In gradient boosting, the shrinkage factor applied to each tree's contribution; lower rate requires more trees but reduces overfitting |
| **Early stopping** | A regularization technique that halts model training when performance on a held-out validation set stops improving |
| **SHAP values** | SHapley Additive exPlanations; assigns each feature a contribution to each individual prediction, grounded in cooperative game theory |
| **Beeswarm plot** | A SHAP summary visualization showing each sample-feature combination as a dot, colored by feature value and positioned by SHAP value |
| **Macro AUC** | Average AUC across all classes with equal weighting; used in multiclass classification; sensitive to performance on small classes |
| **Weighted AUC** | Average AUC across classes weighted by class sample size; less affected by small-class performance |
| **External validation** | Evaluation of a trained model on a completely independent dataset, ideally from a different cohort, site, or platform |
| **TRIPOD** | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis; international reporting guideline for biomarker ML studies |
| **STARD** | Standards for Reporting of Diagnostic Accuracy; reporting guideline for studies validating diagnostic tests against a clinical reference standard |
| **DeLong's method** | A statistical test comparing two AUC values that accounts for their correlation when computed on the same patients |
| **Calibration plot** | A reliability diagram plotting mean predicted probability against observed event rate; a perfectly calibrated model falls on the diagonal |
| **miRBaseConverter** | R package for converting miRNA names between different versions of the miRBase registry using stable MIMAT accession numbers |
| **Per-dataset z-scoring** | Standardizing each dataset's expression values independently to mean=0, SD=1 per feature before cross-platform comparison |

---

## KEY REFERENCES (Week 5)

1. Collins GS, Reitsma JB, Altman DG, Moons KGM (2015). Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *Ann Intern Med* 162(1):55–63. [DOI: 10.7326/M14-0698](https://doi.org/10.7326/M14-0698) — *Primary reference for prediction model reporting*

2. Bossuyt PM et al. (2015). STARD 2015: An Updated List of Essential Items for Reporting Diagnostic Accuracy Studies. *BMJ* 351:h5527. [DOI: 10.1136/bmj.h5527](https://doi.org/10.1136/bmj.h5527) — *Reporting standard for diagnostic accuracy studies*

3. DeLong ER, DeLong DM, Clarke-Pearson DL (1988). Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics* 44(3):837–845. [DOI: 10.2307/2531595](https://doi.org/10.2307/2531595) — *Statistical method for AUC comparison*

4. Ludwig N et al. (2019). Distribution of miRNA expression across human tissues. *Nucleic Acids Res* 47(1):e3. [DOI: 10.1093/nar/gky1151](https://doi.org/10.1093/nar/gky1151) — *Context for interpreting tissue-specific and blood-based miRNA profiles*

5. Xu Z, Marchionni L, Wang JT, et al. (2022). Machine learning and complex biological data. *Genome Biol* 23:197. [DOI: 10.1186/s13059-022-02754-3](https://doi.org/10.1186/s13059-022-02754-3) — *Review of overfitting and reproducibility in omics ML*

6. Varoquaux G (2018). Cross-validation failure: Small sample sizes lead to large error bars. *Neuroimage* 180(Pt A):68–77. [DOI: 10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061) — *Quantification of CV variance in small-N studies*

7. Lundberg SM, Lee SI (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems* 30:4765–4774. [arXiv: 1705.07874](https://arxiv.org/abs/1705.07874) — *Original SHAP paper*

8. Chen T, Guestrin C (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*:785–794. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) — *Original XGBoost paper*

9. Hebert SS et al. (2008). Loss of microRNA cluster miR-29a/b-1 in sporadic Alzheimer's disease correlates with increased BACE1/beta-secretase expression. *Proc Natl Acad Sci USA* 105(17):6415–6420. [DOI: 10.1073/pnas.0710263105](https://doi.org/10.1073/pnas.0710263105) — *Biological validation for miR-29 SHAP interpretation*

10. Cogswell JP et al. (2008). Identification of miRNA Changes in Alzheimer's Disease Brain and CSF Yields Putative Biomarkers and Insights into Disease Pathways. *J Alzheimers Dis* 14(1):27–41. [DOI: 10.3233/JAD-2008-14103](https://doi.org/10.3233/JAD-2008-14103) — *Context for AD miRNA biology and biomarker interpretation*

11. Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics* 8(1):118–127. [DOI: 10.1093/biostatistics/kxj037](https://doi.org/10.1093/biostatistics/kxj037) — *ComBat methodology for cross-study harmonization*

12. Robin X et al. (2011). pROC: an open-source package for R and S+ to analyze and compare ROC curves. *BMC Bioinformatics* 12:77. [DOI: 10.1186/1471-2105-12-77](https://doi.org/10.1186/1471-2105-12-77) — *DeLong test implementation in R*

---

## Next Week Preview (Week 6)

**Week 6: From Discovery to Clinical Translation**

Next week, we bridge the gap between a validated computational model and clinical deployment. We will cover:

- **Biomarker qualification frameworks:** FDA Biomarker Qualification Roadmap; EMA guidance on biomarker-driven trials; the IVD (In Vitro Diagnostic) regulatory pathway
- **From miRNA panel to clinical assay:** How a serum miRNA panel discovered by ML is translated to an RT-qPCR multiplex assay; what changes between discovery and clinical formats
- **Prospective validation design:** How to design a prospective validation study (sample size calculation, pre-specified thresholds, blinded evaluation)
- **Biological follow-up experiments:** How to use your SHAP-identified top miRNAs as entry points for functional validation (luciferase reporter assays, miRNA mimic/inhibitor transfection, pathway enrichment analysis)
- **Integrative biomarkers:** Combining miRNA with clinical variables (MMSE, age, APOE genotype) in a multi-modal model; multi-omics integration conceptual overview
- **Course synthesis:** Building your final analysis portfolio; writing a complete computational methods section for a biomarker paper; peer review simulation
