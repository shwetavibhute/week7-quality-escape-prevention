
# Data Card

## Dataset Name

APS Failure at Scania Trucks

---

## Domain

Manufacturing

---

## Project

A05: Quality Escape Prevention

---

## Source

UCI Machine Learning Repository

---

## Dataset Description

The APS Failure at Scania Trucks dataset contains sensor measurements collected from heavy Scania trucks. The objective is to predict failures in the Air Pressure System (APS) using machine learning techniques.

---

## Machine Learning Task

Binary Classification

---

## Target Variable

**class**

| Value | Meaning |
|-------|---------|
| neg | Normal Operation |
| pos | APS Failure |

---

## Features

- Total Features: 170
- Sensor measurements collected from truck components
- Numerical attributes
- Missing values are present in several features

---

## Data Preprocessing

The following preprocessing steps were performed:

- Missing value handling
- Feature scaling
- Train-test split
- Feature preparation for machine learning

---

## Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Hyperparameter tuning was performed to optimize the final Random Forest model.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Objective

The objective of this project is to predict APS failures before they occur so that manufacturers can reduce quality escapes, improve maintenance planning, and minimize operational costs.

---

## Applications

- Predictive Maintenance
- Manufacturing Quality Control
- Industrial Machine Learning
- Failure Prediction
- Quality Assurance

---

## Author

**Shweta Vibhute**

MindForgeAI Internship 2026 – Week 7