# 📊 Telecom Customer Churn Analysis

## 🧭 Project Overview

This project analyzes telecom customer data to uncover key factors influencing **customer churn** and builds a **machine learning model** to predict whether a customer will **Stay**, **Churn**, or **Join**.

It combines **Exploratory Data Analysis (EDA)** with **predictive modeling** (Random Forest, Logistic Regression, Decision Tree, Naive Bayes) to provide actionable business insights.

---

## 🧰 Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn
- **Environment:** Jupyter Notebook / Google Colab

---

## 📂 Dataset Overview

**Records:** 7043  
**Target Column:** `Customer Status` → { Stayed | Churned | Joined }

| Feature Type | Examples |
|---------------|-----------|
| Demographic | Gender · Age · Married · Dependents |
| Account Info | Tenure · Contract · Payment Method · Paperless Billing |
| Services | Internet Type · Online Backup · Device Protection |
| Financial | Monthly Charge · Total Charges · Total Revenue |

---

## 🧼 Data Pre-Processing

1️⃣ Removed irrelevant columns (`Latitude`, `Longitude`, `Zip Code`, `City`, `Churn Reason`, `Churn Category`).  
2️⃣ Filled missing values (mode for categorical, median for numeric).  
3️⃣ Converted categorical features via `pd.get_dummies()`.  
4️⃣ Encoded `Customer Status` with `LabelEncoder`.  
5️⃣ Scaled numerical columns using `StandardScaler`.  
6️⃣ Handled class imbalance with `SMOTE`.

---

## 📈 Exploratory Data Analysis (EDA)

### 🔹 Churn by Contract Type

The visualization below shows that customers on **month-to-month contracts** have the highest churn rates, while those on **1-year** and **2-year** contracts are much more likely to stay.

Encouraging long-term contracts helps reduce churn significantly.

![Churn by Contract Type](./contract%20type.png)

---

### 🔹 Top 10 Most Important Features That Cause Churn

These features have the greatest influence on predicting customer churn and retention:

![Top Features](./Screenshot%202025-10-21%20213419.png)

| Rank | Feature | Insight |
|------|----------|----------|
| 1️⃣ | Total Charges | High spenders tend to stay |
| 2️⃣ | Total Revenue | Correlates with customer value |
| 3️⃣ | Tenure in Months | Short tenure → higher churn |
| 4️⃣ | Total Long Distance Charges | Reflects engagement |
| 5️⃣ | Contract (Two Year) | Long contracts reduce churn |
| 6️⃣ | Monthly Charge | High bills drive churn |
| 7️⃣ | Contract (One Year) | Improves retention |
| 8️⃣ | Multiple Lines | More services → higher retention |
| 9️⃣ | Married | Married customers churn less |
| 🔟 | Streaming TV | Bundled services increase loyalty |

---

## 🤖 Machine Learning Pipeline

| Step | Description |
|------|--------------|
| **1️⃣** | Encoding features and target |
| **2️⃣** | Train/test split (80 / 20) |
| **3️⃣** | Scaling numeric features |
| **4️⃣** | Oversampling with SMOTE |
| **5️⃣** | Model training (Random Forest, Logistic Regression, Decision Tree, Naive Bayes) |
| **6️⃣** | Evaluation (Accuracy, Recall, F1) |
| **7️⃣** | Feature importance visualization |

---

## 🧠 Model Performance Summary

| Model | Accuracy | Weighted Recall | Weighted F1 |
|--------|-----------|----------------|--------------|
| **Random Forest** | ≈ 84% | 0.84 | 0.83 |
| Logistic Regression | ≈ 77% | 0.78 | 0.74 |
| Decision Tree | ≈ 81% | 0.81 | 0.81 |
| Naive Bayes | ≈ 12% | 0.12 | 0.06 |

✅ **Best Model:** Tuned Random Forest Classifier
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

---

## 🧮 Confusion Matrix — Random Forest

The confusion matrix shows that the model performs best for "Stayed" customers, with slight overlap between "Churned" and "Joined."

![Confusion Matrix](./Screenshot%202025-10-21%20213427.png)
---

## 🏁 Conclusion

This end-to-end Telecom Customer Churn Analysis project demonstrates how EDA and Machine Learning can predict churn and identify the key drivers of customer retention.

By acting on churn predictions and customer insights, telecom companies can:

- Reduce churn by 20–30%
- Increase customer lifetime value
- Optimize marketing and retention strategies

---


## 📊 Key Insights

- **Month-to-month contracts** are the biggest churn risk factor
- **Long-term contracts** (1-2 years) significantly improve retention
- **Tenure** and **total charges** are strong predictors of customer loyalty
- **Service bundling** (multiple lines, streaming) increases customer stickiness
- **High monthly charges** correlate with increased churn probability

---


