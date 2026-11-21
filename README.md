# ML Lab Work

Machine Learning lab notebooks & projects : **Lakshay** (AP24110010677), B.Tech CSE (AI/ML).

---

## 📌 End-Semester Project

### [Credit Card Fraud Detection using Machine Learning](https://github.com/lakshay8toic/Credit-Card-Fraud-Detection)

- **Repository:** [https://github.com/lakshay8toic/Credit-Card-Fraud-Detection](https://github.com/lakshay8toic/Credit-Card-Fraud-Detection)
- **Institution:** SRM University – AP · Department of Computer Science & Engineering
- **Supervisor:** Mr. Ajay Dilip Kumar Marapatla
- **Team:** Lakshay (AP24110010677), Akshita Modi, Mohana Krishna Reddy Pittu, Nadimpally Dheva Harshit

#### Overview & Key Highlights
- **Problem Statement:** Real-time detection of fraudulent credit card transactions under extreme class imbalance (only 492 frauds out of 284,807 transactions, ~0.172%).
- **Curriculum Alignment:** Implemented across the complete 5-unit Machine Learning syllabus covering data exploration, preprocessing, supervised classification, ensemble learning, neural networks, dimensionality reduction (PCA/LDA), and cost-sensitive evaluation.
- **Models Benchmarked:** 12+ machine learning algorithms trained & evaluated on an identical stratified split:
  - *Decision Trees (ID3)*, *k-Nearest Neighbours (k-NN)*, *Gaussian Naive Bayes*
  - *Support Vector Machines (SVM - RBF Kernel)*
  - *Artificial Neural Networks (Perceptron & Multilayer Perceptron via Backpropagation)*
  - *Ensembles (Random Forest, Bagging, AdaBoost, XGBoost, Stacking & Voting)*
  - *Unsupervised Learning (K-Means Clustering)*
- **Key Findings:** Accuracy is misleading due to severe imbalance (a naive classifier achieves 99.83% accuracy while catching zero fraud). Evaluated strictly on **Precision, Recall, F1-Score, and PR-AUC**.
- **Top Performer:** **XGBoost** achieved the best trade-off with **F1-Score: 0.8788** (Precision: 0.9508, Recall: 0.8169) — catching 116 of 142 frauds in the test set with only 6 false positives.
- **Interactive Web Demo:** Includes an interactive [Streamlit demo app](https://github.com/lakshay8toic/Credit-Card-Fraud-Detection#live-demo) allowing real-time single-transaction evaluation, decision path explainability, threshold tuning, and live confusion matrix simulation.

---

## 🧪 Lab Notebooks & Assignments

| # | Topic | Notebook / Resource |
|---|---|---|
| 1 | NumPy : arrays to image analysis | [01_NumPy](01_NumPy) |
| 2 | Pandas : Series & DataFrames | [02_Pandas](02_Pandas) |
| 3 | Matplotlib & Seaborn visualisation | [03_Matplotlib_Seaborn](03_Matplotlib_Seaborn) |
| 4 | k-Nearest Neighbours on Iris (seed 77) | [04_KNN_Iris](04_KNN_Iris) |
| 5 | Decision Tree (ID3) : PlayTennis | [05_Decision_Tree_ID3](05_Decision_Tree_ID3) |
| 6 | Covariance & Correlation (pair + matrices) and PCA : small random dataset, 10 students (seed 77) | [Lab6_Covariance_Correlation_PCA.ipynb](Lab6_Covariance_Correlation_PCA.ipynb) |
| 7 | Simple Linear Regression : hours studied vs marks (hand calculation + sklearn, R² / RMSE / MAE) | [Lab7_Simple_Linear_Regression.ipynb](Lab7_Simple_Linear_Regression.ipynb) |
| 8 | Multiple Linear Regression : study + sleep hours vs marks, β = (XᵀX)⁻¹Xᵀy (calculation + sklearn) | [Lab8_Multiple_Linear_Regression.ipynb](Lab8_Multiple_Linear_Regression.ipynb) |
| 9 | Polynomial Regression : 2nd-order fit on hours vs grade points (normal equations + sklearn) | [Lab9_Polynomial_Regression.ipynb](Lab9_Polynomial_Regression.ipynb) |
| 10 | PCA : Wine Quality dataset, 13 features → 2 (from scratch + sklearn, lab questions) | [Lab10_PCA](Lab10_PCA) |
| CLA-3 | Case Study : Sentiment Analysis of Social Media Using Machine Learning (Report & Presentation) | [CLA-3_AP24110010677_CaseStudy](CLA-3_AP24110010677_CaseStudy) |
| **Project** | **End-Semester Project : Credit Card Fraud Detection** | **[Credit-Card-Fraud-Detection](https://github.com/lakshay8toic/Credit-Card-Fraud-Detection)** |

---

Open any `.ipynb` directly on GitHub or in Google Colab.
