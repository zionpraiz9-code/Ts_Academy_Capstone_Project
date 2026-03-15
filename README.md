# 🛡️ Advanced Fraud Detection Analysis in Mobile Money Systems
**TS Academy Capstone Project | Group 12**

## 1. Project Overview

This project addresses the growing financial security challenges within mobile money ecosystems. Using a high-fidelity synthetic simulator (PaySim), we analyzed over **5.4 million transactions** to identify behavioral fingerprints of fraud.

The objective was to transition from a **linear baseline model** to an **advanced ensemble architecture** in order to achieve maximum detection sensitivity (**Recall**) while effectively handling high-dimensional financial data.

## 2. Data Sourcing & Justification

**Dataset:** PaySim (Aggregated Version)  
**Scale:** 5,420,481 rows | 27 columns  
**Target:** `isFraud` (Highly imbalanced: <1% Fraud)

**Justification**

This dataset was selected because it reflects **real-world financial transaction behavior**. It captures complex relationships between account balances, transaction behavior, and rapid account liquidations—patterns commonly associated with fraud.

The dataset also presents a **class imbalance scenario**, where fraudulent transactions represent **less than 1% of total transactions**, making it a realistic and challenging problem for machine learning systems.

**Dataset Source**

The dataset comes from the PaySim mobile money simulator, which generates **synthetic financial transaction data based on aggregated statistics from real mobile money systems** while preserving privacy.

🔗 Dataset Link  
https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

## 3. Technical Methodology

### A. Data Preprocessing

To ensure efficient processing of the large dataset, the following preprocessing steps were performed:

**Cleaning & Audit**

- Verified **zero missing values**
- Checked for possible **data leakage**

**Feature Encoding**

Categorical variables such as:

- `transaction_type` (CASH_OUT, TRANSFER, etc.)
- `week_group`

were converted into numerical format using **Label Encoding**.

**Normalization**

A **StandardScaler** was applied to:

- `amount`
- balance-related features

This prevented large transaction values from dominating the model’s coefficients.

**Memory Optimization**

Numeric data types were **downcasted** (for example `float64 → float32`), reducing the dataset’s **memory footprint by over 50%**.

### B. Feature Selection & Engineering

Instead of using all raw columns, the analysis focused on features with the **highest predictive value**.

**Aggregated Velocity Features**

Behavioral indicators were created by calculating **transaction counts over 7-day and 30-day windows**, allowing the model to detect bursts of suspicious activity.

**Gini Importance Ranking**

A **Random Forest–based feature importance analysis** was used to remove noise and retain the **top 10 predictive features**.

## 4. Machine Learning Implementation

### Baseline Model: Logistic Regression (LR)

**Purpose**

To establish a performance baseline and test whether the data could be separated using a **linear classification model**.

**Mechanism**

The model estimates the **probability of fraud** based on weighted combinations of input features.

### Advanced Model: Random Forest (RF)

**Purpose**

To capture **complex, non-linear relationships** between transaction time, amount, and historical account behavior.

**Mechanism**

Random Forest is an **ensemble learning algorithm** consisting of multiple decision trees. Each tree produces a classification, and the final prediction is determined through **majority voting**, improving robustness and reducing variance.

## 5. Evaluation Strategy

A structured validation framework was used to ensure reliable performance.

**Train-Test Split**

- **80% Training Data**
- **20% Testing Data**

The original class distribution was preserved.

**Cross-Validation**

Cross-validation ensured that the model did not **overfit specific portions of the dataset**.

**Primary Evaluation Metric**

**Recall (Sensitivity)** was prioritized because, in fraud detection systems, **missing fraudulent transactions (False Negatives)** is far more costly than generating **false alarms (False Positives)**.

## 6. Model Performance & Evolution

### Performance Comparison

| Metric | Logistic Regression (Baseline) | Random Forest (Final Model) |
|------|------|------|
| Accuracy | 99.46% | 99.13% |
| Recall (Catch Rate) | 99.99% | 97.50% |
| Precision | 0.68 | 0.56 |
| F1 Score | 0.81 | 0.71 |

### Elite Feature Discovery (Top Predictors)

Using **Gini Importance**, the Random Forest model identified the following **top predictors**:

1. **Transaction Amount (33.5%)** – The strongest indicator of suspicious activity.  
2. **Tx Count Last 7 Days (21.5%)** – Detects bursts of transaction activity.  
3. **Avg Amount Last 30 Days (17.6%)** – Captures deviations from normal spending patterns.  
4. **Transaction Type (9.9%)** – Certain channels, particularly transfers, carry higher fraud risk.

### Final Confusion Matrix (Test Data)

- **True Positives (Fraud Caught):** 11,830  
- **False Negatives (Missed Fraud):** 303  
- **True Negatives (Legitimate Verified):** 1,062,790  
- **False Positives (False Alarms):** 9,174

## 7. Conclusion

The **TS Academy Capstone Project – Group 12** analysis demonstrates that while **Logistic Regression** achieves extremely high sensitivity, the **Random Forest model** provides a more nuanced understanding of fraudulent behavior.

By prioritizing **behavioral velocity indicators** (transaction counts) alongside **transaction amounts**, the developed system is capable of monitoring **over 5 million transactions** while maintaining **97.5% fraud detection recall**.

## 8. Acknowledgments & License

**Tutor:** Hart Ofigwe  
**Institution:** TS Academy  

## 9. References

### Data Source
- Lopez-Rojas, E. A., Elmir, A., & Axelson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. In 28th European Modeling and Simulation Symposium (EMSS).  
- chendoytshman. (2023). *Fraud Detection - PaySim (with aggregated) [Data set]*. Kaggle. https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

### Software & Libraries
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,

**License:** Apache License 2.0  
**Copyright:** © 2026 TS Academy Capstone Project – Group 12
