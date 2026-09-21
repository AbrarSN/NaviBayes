# NaviBayes
# MAGIC Gamma Telescope Classification

This repository contains data preprocessing, exploratory analysis, feature scaling, and machine learning classification models (Gaussian Naive Bayes and K-Nearest Neighbors) applied to the MAGIC Gamma Telescope dataset[cite: 3].

---

## 📌 Dataset Overview

* **Total Samples**: 19,020 entries[cite: 3]
* **Target Variable**: `class`[cite: 3]
  * `g` (Gamma rays) mapped to `1`[cite: 3]
  * `h` (Hadron noise) mapped to `0`[cite: 3]
* **Numerical Features (10)**: `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`[cite: 3]

---

## ⚙️ Data Preprocessing Pipeline

1. **Train/Val/Test Split**: Split into Training (60%), Validation (20%), and Testing (20%) sets[cite: 3].
2. **Feature Scaling**: Scaled feature distributions using `StandardScaler`[cite: 3].
3. **Class Balancing**: Balanced training class distributions using `RandomOverSampler`[cite: 3].

---

## 📊 Model Training & Performance

### 1. Gaussian Naive Bayes (`GaussianNB`)
* **Test Accuracy**: 71% (evaluated on 3,804 test samples)[cite: 3]
* **Classification Breakdown**:
  * **Class 0.0 (Hadron)**: Precision: 0.67 | Recall: 0.38 | F1-Score: 0.49[cite: 3]
  * **Class 1.0 (Gamma)**: Precision: 0.72 | Recall: 0.89 | F1-Score: 0.80[cite: 3]

### 2. K-Nearest Neighbors (`KNeighborsClassifier`)
* Initialized with `n_neighbors=3` for classification comparisons[cite: 3].

---

## 🚀 Setup & Execution

### Dependencies
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
