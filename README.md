# 📊 Telecom Customer Churn Prediction with AI-Powered Retention Strategy

> **End-to-end analytics and machine learning system with intelligent retention recommendations**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](https://colab.research.google.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Enabled-green.svg)](https://www.crewai.com/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow.svg)](https://powerbi.microsoft.com/)

---

## 🎯 Project Overview

Customer churn represents one of the most significant profitability challenges in the telecommunications industry. This project delivers a **production-ready churn prediction and retention system** that combines:

- **Machine Learning** for accurate churn probability prediction
- **Exploratory Data Analysis** to identify churn drivers
- **AI-Powered Decision Layer** for targeted retention recommendations
- **Power BI Dashboards** for business intelligence visualization

### Key Innovation

Unlike traditional churn prediction systems that stop at probability scores, this project includes an **AI Retention Strategy Module** that converts predictions into **actionable business decisions** using CrewAI and Google Gemini.

---

## 📁 Dataset Overview

| Attribute | Details |
|-----------|---------|
| **Total Customers** | 7,043 |
| **Target Variable** | Customer Status (Stayed, Churned, Joined) |
| **Feature Categories** | Demographics, Account Info, Services, Financial |
| **Data Quality** | No major missing values, handled outliers |

### Feature Categories

- **Demographics:** Gender, Age, Marital Status, Dependents
- **Account Information:** Tenure, Contract Type, Payment Method, Paperless Billing
- **Services:** Internet, Phone, Streaming, Security, Backup
- **Financial:** Monthly Charges, Total Charges, Total Revenue

---

## 🔍 Key Findings from Exploratory Data Analysis

### 📆 Tenure & Customer Loyalty

- **Critical Insight:** Churn rate is highest within the first 12 months of service
- Average tenure: ~30 months
- Customers with tenure ≥ 36 months show very low churn rates

> **Business Takeaway:** The first year is the critical retention window

### 💸 Billing & Payment Behavior

| Factor | Observation | Business Impact |
|--------|-------------|-----------------|
| **Monthly Charges** | Higher charges correlate with higher churn | Review premium pricing strategy |
| **Contract Type** | Month-to-month contracts have highest churn | Incentivize annual/biennial contracts |
| **Payment Method** | Bank withdrawal users churn more | Investigate billing experience |
| **Paperless Billing** | Associated with higher churn | Price-sensitive customer indicator |

### 🌐 Service Usage Patterns

- **Fiber-optic users** show higher churn rates (pricing/reliability concerns)
- Customers **without security/backup services** churn more frequently
- Streaming services show minimal retention impact

### 💰 Revenue Impact Analysis

- Churned customers have higher average monthly charges (~$73)
- **High-value customer retention delivers highest ROI**

### 🚦 Churn Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| **Stayed** | 4,720 | 67% |
| **Churned** | 1,869 | 26% |
| **Joined** | 454 | 6% |

---

## 🤖 Machine Learning Pipeline

### ⚙️ Data Preprocessing

```
1. Data Cleaning
   ├── Remove leakage-prone columns (Churn Reason, Churn Category)
   ├── Fill missing values (mode for categorical, median for numerical)
   └── Handle class imbalance using SMOTE

2. Feature Engineering
   ├── Encode categorical features (pd.get_dummies)
   ├── Encode target variable (LabelEncoder)
   └── Scale numeric features (StandardScaler)

3. Train-Test Split
   └── 80/20 stratified split
```

### 🏆 Model Selection & Performance

**Final Model:** Gradient Boosting Classifier

```python
GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=0
)
```

### 📊 Per-Class Performance

| Class | Precision | Recall | F1-Score | ROC-AUC | Support |
|-------|-----------|--------|----------|---------|---------|
| **Churned** | 0.736 | 0.618 | 0.672 | 0.897 | 374 |
| **Joined** | 0.588 | 0.879 | 0.705 | 0.981 | 91 |
| **Stayed** | 0.909 | 0.924 | 0.916 | 0.952 | 944 |

### 🎯 Confusion Matrix

```
                Predicted
              Churned  Joined  Stayed
Actual Churned    231      56      87
       Joined      11      80       0
       Stayed      72       0     872
```

**Key Insights:**
- ✅ Excellent performance on "Stayed" class (92.4% recall)
- ⚠️ "Churned" class has moderate recall (61.8%) - some churners misclassified
- ✅ "Joined" class shows high recall (87.9%) despite smaller sample size
- 💡 Model is conservative in predicting churn, reducing false alarms

### 🔝 Top 10 Predictive Features

1. Tenure in Months
2. Number of Referrals
3. Contract_Two Year
4. Monthly Charge
5. Contract_One Year
6. Number of Dependents
7. Payment Method_Credit Card
8. Paperless Billing_Yes
9. Age
10. Total Charges

---

## 🎯 Probability Thresholding & Risk Segmentation

### Optimal Churn Threshold: **0.23**

Selected using **Youden's J statistic** for optimal sensitivity-specificity balance.

### Customer Risk Segments

| Churn Probability | Risk Level | Business Action |
|-------------------|------------|-----------------|
| **≥ 0.23** | 🔴 High Risk | Immediate retention offers & personalized discounts |
| **0.15 – 0.23** | 🟡 Medium Risk | Proactive customer support & engagement |
| **< 0.15** | 🟢 Low Risk | Standard loyalty programs |

---

## 🤖 AI Retention Strategy Module

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 MACHINE LEARNING LAYER                  │
│  • Predicts churn probability for new customers         │
│  • Assigns risk segment (Low/Medium/High)               │
│  • Uses optimal threshold (0.23)                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ├── Low Risk → No AI Invocation
                     │
                     ├── Medium Risk ──┐
                     │                 │
                     └── High Risk ────┤
                                       │
                     ┌─────────────────┴──────────────────┐
                     │      AI DECISION LAYER             │
                     │  • CrewAI orchestration            │
                     │  • Gemini 2.5 Flash reasoning      │
                     │  • Generates retention strategy    │
                     └────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | ML predicts, AI recommends |
| **Conditional Gating** | AI only for Medium/High risk |
| **No Model Override** | AI cannot change predictions |
| **Cost Efficiency** | Reduced unnecessary API calls |
| **Production Alignment** | Mirrors real-world workflows |

### Why Conditional AI Invocation?

**AI is NOT called for all customers.** Here's why:

- **Low Risk Customers (< 0.15):** Stable, no intervention needed
- **Medium/High Risk (≥ 0.15):** Require strategic retention actions

This approach:
- ✅ Reduces operational costs
- ✅ Avoids over-intervention
- ✅ Focuses resources on high-impact decisions
- ✅ Reflects real telecom retention practices

### CrewAI Agent Configuration

```python
Agent: Telecom Retention Strategist
├── Inputs: Churn Probability, Risk Segment
├── LLM: Google Gemini 2.5 Flash
├── Goal: Generate concise retention recommendations
└── Output: Business-oriented action plan
```

### Error-Resilient Design

```python
try:
    recommendation = crew.kickoff()
except Exception as e:
    recommendation = "AI service temporarily unavailable. Please retry later."
```

**Handles Gemini API 503 errors gracefully:**
- Pipeline continues without crashing
- Fallback recommendations provided
- Business logic remains intact
- Results remain auditable

> **Note:** Gemini 503 errors occur due to temporary cloud service overloads during peak usage. This is an infrastructure-level issue, not a code error.

---

## 📊 Power BI Dashboard

The Power BI dashboard provides visual insights into:
- 📈 Customer churn trends and patterns
- 🎯 High-risk customer segment analysis
- 💰 Revenue impact visualization
- 🔧 Service-wise churn breakdown
- 📋 Actionable retention strategy insights

---

## 🛠️ Technical Stack

### Core Technologies

| Category | Technology |
|----------|------------|
| **Platform** | Google Colab |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, Gradient Boosting |
| **Data Balancing** | SMOTE |
| **Feature Scaling** | StandardScaler |
| **Encoding** | LabelEncoder, pd.get_dummies |
| **Model Interpretability** | SHAP |
| **AI Orchestration** | CrewAI |
| **LLM** | Google Gemini 2.5 Flash |
| **Visualization** | Power BI |
| **Development** | Python 3.8+ |

---

## 🚀 Getting Started

### Running in Google Colab

```python
# 1. Open Google Colab
# Visit: https://colab.research.google.com/

# 2. Install required packages in Colab
!pip install crewai
!pip install google-generativeai
!pip install imbalanced-learn

# 3. Set up Gemini API Key
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"

# 4. Upload your dataset to Colab
from google.colab import files
uploaded = files.upload()

# 5. Run the notebooks in order:
# - 01_eda.ipynb (Exploratory Data Analysis)
# - 02_model_training.ipynb (Model Training & Evaluation)
# - 03_ai_retention.ipynb (AI Recommendation System)
```

### Key Implementation Steps

```python
# Step 1: Load and preprocess data
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Step 2: Train Gradient Boosting Model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=0
)

# Step 3: Generate predictions with risk segments
predictions = model.predict_proba(X_test)
risk_segments = assign_risk_segments(predictions, threshold=0.23)

# Step 4: Get AI recommendations for Medium/High risk customers
from crewai import Agent, Task, Crew
retention_agent = Agent(
    role='Telecom Retention Strategist',
    goal='Generate actionable retention recommendations',
    llm='gemini/gemini-2.0-flash-exp'
)
```

---

## 📈 Business Impact

### Quantifiable Benefits

- 🎯 **84% prediction accuracy** for churn identification
- 🔍 **0.94 ROC-AUC score** for risk ranking
- 💰 **Focus on high-value churners** maximizes retention ROI
- ⚡ **Proactive intervention** in critical first-year window
- 🤖 **AI-powered recommendations** for strategic retention

### Production Deployment Features

- ✅ Monthly model retraining capability
- ✅ Power BI dashboard integration
- ✅ Probability-based decision making
- ✅ Scalable AI recommendation system
- ✅ Error-resilient architecture

---

## 🔄 Model Maintenance

### Recommended Schedule

- **Weekly:** Monitor prediction accuracy and drift
- **Monthly:** Retrain model with new data
- **Quarterly:** Review feature importance and threshold optimization
- **Annual:** Comprehensive model evaluation and strategy reassessment

---

## 📊 Project Structure

```
telecom-churn-prediction/
├── Teleco_Chustomer_Churn_Analysis.ipynb    # Main Jupyter notebook (EDA + ML)
├── teleco_chustomer_churn_analysis.py       # Python script version
├── telecom_customer_churn.csv               # Raw dataset
├── teleco_churn_clean_data (1).xlsx        # Cleaned dataset
├── Churn dashboard.pbix                     # Power BI dashboard
├── powerbidash.png                          # Dashboard screenshot
└── README.md                                # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset source: Telecom Customer Churn Dataset
- Inspired by real-world telecom retention challenges
- Built with modern ML and AI best practices
- Developed using Google Colab for accessibility and reproducibility

---

## 📌 Project Status

### ✅ Production-Ready

This project delivers a **well-validated, production-ready churn prediction and retention system** with:

- Interpretable machine learning insights
- AI-powered decision support
- Strong business alignment
- Real-world deployment capability

**Ready for telecom industry deployment.**

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>
