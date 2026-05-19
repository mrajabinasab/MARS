# MARS — Multiple-classifier Accuracy Ranking Suite

MARS is a Python tool for comparing classifier performance across multiple datasets.

It takes a CSV file containing classifier scores and produces:

- Statistical significance tests
- Pairwise classifier comparisons
- Weighted classifier rankings (MARS rankings)
- Critical difference diagrams as PDF files
- CSV reports with all results

---

# Installation

```bash
git clone https://github.com/your-username/mars-suite.git
cd mars-suite
pip install pandas numpy matplotlib scipy networkx
```

Python 3.8+ is required.

---

# Input Format

The input must be a CSV file with these columns:

| Column | Description |
|---|---|
| `dataset_name` | Dataset identifier |
| `classifier_name` | Classifier identifier |
| `accuracy` | Performance score |

Example:

```csv
dataset_name,classifier_name,accuracy
Dataset_01,RandomForest,0.8731
Dataset_01,XGBoost,0.9012
Dataset_02,RandomForest,0.7645
Dataset_02,XGBoost,0.7901
```

Every classifier must appear for every dataset.

---

# Usage

Run MARS mode:

```bash
python mars.py --input results.csv
```

Run both standard and MARS analysis:

```bash
python mars.py --input results.csv --standard --mars
```

Use colour diagrams:

```bash
python mars.py --input results.csv --color
```

---

# How the Code Works

## 1. Load Data

The CSV file is loaded and converted into a dataset × classifier matrix.

---

## 2. Global Statistical Test

MARS runs a Friedman test to check whether classifier performance differences are statistically significant.

An optional permutation test can also be performed for additional robustness.

---

## 3. Pairwise Comparisons

If the Friedman test is significant, MARS performs pairwise Wilcoxon signed-rank tests between classifiers.

Holm correction is applied to control for multiple comparisons.

---

## 4. MARS Weighted Ranking

Instead of using only ordinal ranks, MARS weights classifier ranks based on performance gaps.

Large wins contribute more strongly than very small wins.

The final average weighted rank becomes the MARS score.

Lower scores are better.

---

## 5. Critical Difference Diagram

MARS generates publication-ready critical difference diagrams.

Classifiers that are not significantly different are connected with horizontal bars.

PDF diagrams are saved automatically.

---

# Output Files

Typical outputs:

| File | Description |
|---|---|
| `results_mars.pdf` | MARS critical difference diagram |
| `results_std.pdf` | Standard critical difference diagram |
| `results_mars_report.csv` | MARS statistics report |
| `results_std_report.csv` | Standard statistics report |

---

# Project Structure

```text
mars-suite/
├── mars.py
├── README.md
├── stress_test_20clfs.csv
└── examples/
```

---

# Citation

```bibtex
CITATION WILL BE PROVIDED UPON PUBLICATION
```

---

# Acknowledgements

The critical difference diagram implementation is inspired by:

- https://github.com/hfawaz/cd-diagram

---
