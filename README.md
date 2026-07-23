
# A05: Quality Escape Prevention

## MindForgeAI Internship 2026 - Week 7

### Project Overview

This project focuses on **Quality Escape Prevention** using the **APS Failure at Scania Trucks** dataset. The goal is to build a machine learning model that predicts failures in the Air Pressure System (APS) of heavy-duty trucks based on sensor data.

Early prediction of failures helps manufacturers reduce quality escapes, minimize maintenance costs, and improve vehicle reliability.

---

## Problem Statement

Develop a binary classification model that predicts whether an APS component will fail based on sensor measurements.

- **Positive (pos):** APS Failure
- **Negative (neg):** Normal Operation

---

## Dataset

- **Dataset:** APS Failure at Scania Trucks
- **Source:** UCI Machine Learning Repository
- **Domain:** Manufacturing
- **Task:** Binary Classification
- **Target Column:** `class`

---

## Project Structure

```
A05_Quality_Escape_Prevention/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── quality_escape_prevention.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── train/
│   ├── evaluate/
│   └── predict/
│
├── tests/
│   └── test_placeholder.py
│
├── requirements.txt
├── DATA_CARD.md
├── README.md
└── LICENSE
```

---

## Machine Learning Workflow

1. Load Dataset
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Missing Value Handling
6. Feature Engineering
7. Data Splitting
8. Feature Scaling
9. Train Machine Learning Models
10. Hyperparameter Tuning
11. Model Evaluation
12. Model Comparison
13. Save Best Model

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook
- Pytest

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Notebook

Open Jupyter Notebook and execute all cells in:

```
notebooks/quality_escape_prevention.ipynb
```

---

## Project Outcome

The trained machine learning model predicts APS failures using sensor data, enabling proactive maintenance and reducing quality escapes in manufacturing.

---

## Author

**Shweta Vibhute**

MindForgeAI Internship 2026