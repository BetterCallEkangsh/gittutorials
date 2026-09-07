# 📊 Logistic Regression Deep Dive

A hands-on exploration of Logistic Regression — from binary classification and hyperparameter tuning to multiclass problems, imbalanced datasets, and ROC-AUC evaluation. Built entirely on synthetic data (`sklearn.datasets.make_classification`) to isolate and understand the mechanics of the algorithm without dataset-specific noise.

## 📁 Repository Structure

| Notebook | Focus |
|---|---|
| `Logisticregression.ipynb` | Binary classification, GridSearchCV & RandomizedSearchCV hyperparameter tuning |
| `Multiclass_Logistic_Regression.ipynb` | 3-class classification + severely imbalanced (99:1) binary data with `class_weight` tuning |
| `ROC_AUC.ipynb` | ROC curves, AUC scoring, and threshold analysis vs. a dummy baseline |

---

## 1️⃣ Binary Classification + Hyperparameter Tuning

Dataset: 1,000 samples, 10 features, 2 balanced classes.

**Baseline `LogisticRegression()`**

| Metric | Class 0 | Class 1 | Overall |
|---|---|---|---|
| Precision | 0.80 | 0.88 | — |
| Recall | 0.87 | 0.82 | — |
| F1-score | 0.83 | 0.85 | — |
| **Accuracy** | | | **0.84** |

### 🔧 Tuners Compared

| Tuner | Search Space | Best Params Found | Best CV Score |
|---|---|---|---|
| **GridSearchCV** | `penalty` × `C` × `solver` (75 combos, 5-fold `StratifiedKFold`) — exhaustive | `{'C': 0.01, 'penalty': 'l2', 'solver': 'sag'}` | 0.877 |
| **RandomizedSearchCV** | Same space, randomly sampled subset | `{'solver': 'saga', 'penalty': 'elasticnet', 'C': 0.01}` | 0.877 |

Both tuners converged to the same CV score, with **RandomizedSearchCV reaching it exploring far fewer combinations** — the practical case for random search when the grid is large.

> `penalty`, `C`, and `solver` must be compatible (e.g. `sag`/`newton-cg`/`lbfgs` only support `l2`; only `saga` supports `elasticnet`) — GridSearchCV logs these as failed fits rather than crashing, which is expected behavior worth understanding rather than a bug.

---

## 2️⃣ Multiclass & Imbalanced Classification

**Multiclass (3 balanced classes, 10k samples)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.88 | 0.93 | 0.91 |
| 1 | 0.96 | 0.97 | 0.96 |
| 2 | 0.95 | 0.89 | 0.92 |
| **Accuracy** | | | **0.93** |

**Imbalanced Binary (99:1 class ratio, 10k samples)** — this is where accuracy lies to you:

| Model | Accuracy | Minority Class Recall | Minority Class F1 |
|---|---|---|---|
| Default `LogisticRegression` | **0.99** | 0.47 | 0.64 |
| `GridSearchCV` + tuned `class_weight` | 0.85 | **0.74** | 0.14 |

This is the **accuracy paradox** in action: the untouched model looks great at 99% accuracy but misses over half the minority class. Tuning `class_weight` (grid searched across `{0: 1–100, 1: 1–100}` combinations) trades overall accuracy for dramatically better recall on the class that actually matters — a deliberate precision/recall tradeoff, not a "worse" model.

---

## 3️⃣ ROC Curve & AUC Evaluation

Dataset: 10,000 samples, binary, evaluated against a **dummy (no-skill) classifier** as a baseline.

| Model | ROC-AUC Score |
|---|---|
| Dummy classifier | 0.500 |
| Logistic Regression | **0.953** |

<p align="center">
  <img src="images/roc_auc_curve.png" width="600" alt="ROC Curve comparing Logistic Regression to a dummy classifier">
</p>

The dashed diagonal is the dummy baseline (random guessing); the solid curve is the tuned model — the further it bows toward the top-left corner, the better the model separates the two classes at every threshold. The notebook also annotates each point on the curve with its decision threshold, making it easy to see the tradeoff between true positive rate and false positive rate as the cutoff shifts.

---

## 🔑 Key Concepts Covered

- **Binary & Multiclass Classification** using `LogisticRegression`
- **Hyperparameter Tuning** — `GridSearchCV` vs. `RandomizedSearchCV`, `StratifiedKFold` cross-validation
- **Imbalanced Data Handling** — simulating 99:1 imbalance, tuning `class_weight` to recover minority-class recall
- **Model Evaluation** — accuracy, confusion matrix, classification reports, ROC-AUC, threshold analysis
- **The Accuracy Paradox** — why high accuracy can hide a useless model on imbalanced data

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn`

## 🚀 Getting Started

```bash
git clone <your-repo-url>
cd <repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Run in order: `Logisticregression.ipynb` → `Multiclass_Logistic_Regression.ipynb` → `ROC_AUC.ipynb`

## 📌 Why This Project

Rather than jumping straight to a real-world dataset, this project isolates Logistic Regression fundamentals using controlled synthetic data — making it easy to see exactly how hyperparameters, class imbalance, and decision thresholds affect model behavior, before applying these lessons to messier, real-world data.

---
⭐ If you found this useful, consider starring the repo!
