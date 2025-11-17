# 📘 Telecom Customer Churn Prediction — End-to-End Analysis (EDA + Machine Learning)

## 🧠 Project Overview

This project focuses on understanding and predicting **customer churn** in a telecom company using data analytics and machine learning techniques. The goal is to identify **key drivers of churn**, understand **customer behavior patterns**, and **predict high-risk customers** to help the business improve retention.

The project combines:

* 📊 **Exploratory Data Analysis (EDA)** for insights and trends
* 🤖 **Machine Learning (Random Forest Classifier)** for churn prediction
* 🎨 **Visualizations (Power BI + Python)** for business storytelling

---

## 🧾 Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [EDA: Exploratory Data Analysis](#eda-exploratory-data-analysis)
3. [Power BI Dashboard](#power-bi-dashboard)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Model Performance](#model-performance)
6. [Model Visualizations](#model-visualizations)
7. [Key Insights](#key-insights)
8. [Business Application](#business-application)
9. [Tools & Technologies](#tools--technologies)
10. [Project Files](#project-files)
11. [Conclusion](#conclusion)

---

## 📂 Dataset Overview

* **Dataset Name:** Telecom Customer Churn Dataset
* **Records:** ~7,000 customers
* **Target Variable:** `Customer Status` — {Stayed, Churned, Joined}
* **Objective:** Predict which customers are most likely to churn

**Key Features:**

* Demographics (Gender, Age, Dependents)
* Services (Internet Type, Security, Backup, Tech Support)
* Billing (Contract Type, Payment Method, Paperless Billing)
* Financials (Monthly Charges, Total Charges)
* Tenure (Months with the company)

---

## 🔍 EDA: Exploratory Data Analysis

### 🧹 Data Cleaning & Preparation

* Removed irrelevant columns: `Churn Reason`, `Churn Category`.
* Handled missing values — categorical: filled with **mode**, numeric: filled with **median**.
* Renamed columns and standardized column names.
* Created derived metrics: **Churn Rate**, **Total Revenue Lost**, **Monthly Revenue Lost**.

### 📊 Key Observations from EDA

1. **Churn Distribution:** About 26% of customers have churned.
2. **Contract Type:** Month-to-Month contracts show the highest churn (>40%).
3. **Tenure:** Churn rate decreases steadily with longer tenure.
4. **Internet Type:** Fiber optic users are more likely to churn than DSL users.
5. **Billing & Payment:** Paperless billing and electronic payment methods are linked to higher churn.

📸 **Power BI Dashboard Visualization:**

![POWER\_BI\_DASHBOARD](POWER_BI_DASHBOARD.png)

### 💰 Insights from Dashboard

* **Total Revenue Lost:** 2.86M
* **Total Churned Customers:** 1,869
* **Total Monthly Revenue Lost:** 137.09K
* Churn rate trends are shown by **Tenure**, **Contract Type**, and **Internet Type**.
* Dynamic slicers available for demographic and service-level analysis.

---



---
## 🤖 Machine Learning Summary Steps  

## 🧠 Overview  
Customer churn is one of the biggest profitability challenges in telecom. This project applies a **Random Forest Classifier** to predict which customers are most likely to **churn (attrite)**, combining **Python (scikit-learn)** for modeling and **Power BI** for business intelligence dashboards.

---

## ⚙️  Data Preprocessing

| Step | Description |
|------|--------------|
| **1.1 Data Cleaning** | Removed irrelevant columns (`Churn Reason`, `Churn Category`). |
| **1.2 Missing Values** | Replaced missing categorical values with the **mode** and numerical values with the **median**. |
| **1.3 Feature Encoding** | Used `pd.get_dummies()` for categorical variables. |
| **1.4 Target Encoding** | Applied `LabelEncoder` to `Customer Status` → {Stayed, Churned, Joined} → {0, 1, 2}. |
| **1.5 Train-Test Split** | 80% training / 20% testing with `stratify=y` for balanced classes. |
| **1.6 Feature Scaling** | Standardized numeric columns with `StandardScaler`. |
| **1.7 Imbalance Handling** | Balanced target classes using **SMOTE** (Synthetic Minority Oversampling Technique). |

---

## 🧩  Model Training

| Model | Key Parameters | Purpose |
|--------|----------------|----------|
| **Random Forest (baseline)** | `class_weight='balanced', random_state=0` | Handles imbalance & non-linearity |
| **Logistic Regression** | `max_iter=1000, random_state=0` | Linear benchmark |
| **Decision Tree** | `random_state=0` | Interpretable baseline |
| **Naive Bayes** | Default params | Probabilistic baseline |

All models were trained on **resampled + scaled** data (`x_train_resample`, `y_train_resample`).

---

##  📊 Model Performance (Final Results)


```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=0
)
```

**Performance:**
- Train Accuracy: 0.951
- Test Accuracy: 0.833
- ROC-AUC: 0.936
- Precision: 0.836
- Recall: 0.833
- F1 Score: 0.833
- ROC AUC: 0.936


✅ **Interpretation**  
- Fine-tuning improved **generalization** and reduced overfitting.  
- **AUC = 0.936** shows strong discrimination across classes (*Stayed / Churned / Joined*).  
- Balanced metrics confirm deployment-ready stability.


---

## 📈 Model Insights

### 🔹 Key Churn Drivers

- Tenure, billing type, and contract length drive churn risk.
- Fiber-optic internet & paperless billing → higher churn probability.
- Auto-pay + long-term contracts → better retention.

### 🔹 Top 10 Feature Importances

1. Tenure
2. Monthly Charges
3. Contract Type
4. Internet Service
5. Payment Method
6. Paperless Billing
7. Online Security
8. Device Protection
9. Dependents
10. Tech Support

---

## 📉  ROC Curve, Probability Distribution & Thresholds

| Parameter | Value |
|-----------|-------|
| Optimal Churn Probability Threshold | 0.295 |
| ROC-AUC Score | 0.936 |

### 🧭 Interpretation
- Customers with Churn Probability ≥ 0.295 are high-risk attriters.
- The 0.295 threshold balances sensitivity (True Positive Rate) and specificity (1 – False Positive Rate).
- ROC curve and probability distribution visuals confirm robust separation between churned and non-churned customers.

---

## 🏁  Business Application

### 🎯 Deployment Strategy
- Retrain and deploy monthly to detect emerging churn patterns.
- Integrate predictions into Power BI dashboards for real-time business decisions.

### Customer Segmentation

| Probability Range | Segment | Action |
|-------------------|---------|--------|
| P(Churned) ≥ 0.295 | High-Risk | Retention offers & personalized discounts |
| 0.25 ≤ P(Churned) < 0.295 | Medium-Risk | Customer support & plan improvements |
| < 0.25 | Safe | Regular loyalty programs & engagement |

---

##  Technical Stack
- **Python** (scikit-learn, pandas, numpy)
- **SMOTE** for imbalance handling
- **Random Forest Classifier** (primary model)
- **Power BI** for dashboards
- **StandardScaler** for feature scaling
- **LabelEncoder** for target encoding

---

## 📌 Project Status
✅ **Production-Ready** — Model demonstrates balanced performance with strong generalization capabilities and is ready for deployment in real-world telecom churn prediction scenarios.

By acting on churn predictions and customer insights, telecom companies can:
- Reduce churn by 20–30%
- Increase customer lifetime value
- Optimize marketing and retention strategies

---

**Author:** Sourav Mondal, Email: souravmondal5f@gamil.com 
**Tools:** Python | Power BI | Excel | SQL
**Keywords:** Telecom, Churn Prediction, Machine Learning, Data Visualization, Customer Retention
