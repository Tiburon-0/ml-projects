# Machine Learning Projects

Portfolio of supervised and unsupervised learning projects built with skills and knowledge obtained via the University of Denver. Each notebook follows a full end-to-end pipeline: exploratory data analysis → data preparation → baseline modeling → hyperparameter tuning → model selection.

> **Note:** GitHub's notebook renderer is currently experiencing issues. Use the nbviewer links below for a fully rendered view of each project.

---

## Projects

### [Assignment 2 — Diamond Price Prediction](https://nbviewer.org/github/Tiburon-0/ml-projects/blob/main/ml_assignment_2.ipynb)
**Task:** Regression · **Dataset:** Seaborn Diamonds (~54,000 records)

Predicts diamond price from cut, color, clarity, and physical dimensions. Includes zero-value imputation using the depth formula, outlier removal, and parallel evaluation of ordinal vs. one-hot encoding across four regressors. Champion: **Decision Tree** with ordinal encoding and tuned `min_samples_leaf` (RMSE ≈ 624).

---

### [Assignment 3 — Titanic Survival Classification](https://nbviewer.org/github/Tiburon-0/ml-projects/blob/main/ml_assignment_3.ipynb)
**Task:** Binary Classification · **Dataset:** Seaborn Titanic (891 records)

Predicts passenger survival using class, sex, age, fare, and embarkation point. Features stratified train/test split, class-stratified median age imputation, and cross-validated confusion matrices and ROC-AUC scoring. Champion: **tuned SVC** (C ≈ 49.4, γ ≈ 0.030, ~84.8% CV accuracy).

---

### [Assignment 4 — Bike Share Usage Regression](https://nbviewer.org/github/Tiburon-0/ml-projects/blob/main/ml_assignment_4.ipynb)
**Task:** Regression · **Dataset:** Bike Share Hour (17,379 records)

Predicts hourly bike rental counts from weather, time-of-day, and calendar features. Includes eight EDA visualizations, leakage analysis, and permutation feature importance on the held-out test set. Champion: **tuned Random Forest** (`max_features=0.9`, `n_estimators=1000`, RMSE ≈ 56.80).

---

## Stack

Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn
