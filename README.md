# MARS — Magnitude-Aware Rank Statistics

MARS is a Python tool for comparing the performance of different machine learning solutions across multiple observations and settings (e.g., datasets). 
The software suite supports both Standard and MARS rank statistics.

It takes a CSV file containing classifier scores and produces:

---

# Input Format

The input must be a CSV file with these columns:

| Column | Description |
|---|---|
| `observation` | Observation (e.g., Dataset) |
| `subject` | Subject (e.g., Classifier) |
| `value` | Value (e.g., Accuracy) |

Example:

```csv
observation,subject,value
Dataset_01,RandomForest,0.8731
Dataset_01,XGBoost,0.9012
Dataset_02,RandomForest,0.7645
Dataset_02,XGBoost,0.7901
```
---
# Command Line Arguments

| Argument | Description |
|---|---|
| `--input` | Path to the input CSV file |
| `--standard` | Generate standard average-rank analysis |
| `--mars` | Generate MARS weighted-rank analysis |
| `--color` | Produce colour diagrams |
| `--bw` | Produce black-and-white diagrams |
| `--alpha` | Statistical significance threshold |
| `--name` | Custom experiment name for figures |
| `--perms` | Number of permutations for permutation testing |
| `--n-jobs` | Number of CPU workers for parallel computation |

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

# Citation

```bibtex
@misc{rajabinasab2026mars,
      title={MARS: Magnitude-Aware Rank Statistics}, 
      author={Muhammad Rajabinasab and Afsaneh M. Nejad and Arthur Zimek},
      year={2026},
      eprint={2605.23563},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.23563}, 
}
```

---

# Acknowledgements

The critical difference diagram visualization implementation is inspired by:

- https://github.com/hfawaz/cd-diagram

---
