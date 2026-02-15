# 🚀 Customer Churn Prediction & Revenue Protection System

## 📌 Business Problem
Customer churn is one of the largest revenue threats for SaaS companies. Acquiring a new customer can cost **5x more** than retaining an existing one.

This project builds an **end-to-end machine learning intelligence system** that predicts:

- Which customers are likely to churn  
- When they are likely to churn  
- How much revenue is at risk  
- What factors are driving churn  

---

## 🧠 Solution Overview

This production-style ML system combines:

- Predictive Modeling  
- Survival Analysis  
- Customer Segmentation  
- Explainable AI (SHAP)  
- Executive Dashboard  

to enable **data-driven retention strategies**.

---

## ⭐ Key Features

- Synthetic SaaS dataset with realistic churn behavior  
- Advanced Feature Engineering (RFM, Engagement Score, CLV)  
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)  
- Survival analysis for time-to-churn prediction  
- SHAP explainability for model transparency  
- Revenue-at-risk estimation  
- Interactive Streamlit dashboard  

---

## 🏗 Architecture

```
Data Generation → Feature Engineering → Model Training →
Risk Scoring → Explainability → Dashboard
```

---

## 📊 Dashboard Preview

Add screenshots of:
- Executive KPIs
- Churn risk distribution
- Survival curve
- SHAP feature importance

---

## ⚙ Tech Stack

**Languages & ML**
- Python
- Scikit-learn
- XGBoost
- Lifelines

**Visualization**
- Plotly
- Streamlit

**Database**
- SQLite

**Explainability**
- SHAP

---

## 📈 Business Impact

The system enables organizations to:

- Identify high-risk customers early  
- Launch targeted retention campaigns  
- Protect recurring revenue  
- Understand churn drivers  

**Sample Output:**  
> Annual Revenue At Risk Identified: $XXX,XXX

---

## ▶ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Generate data
```bash
python src/data_generator.py
```

### 3️⃣ Train models
```bash
python main.py
```

### 4️⃣ Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔮 Future Enhancements

- Real-time churn scoring API  
- Automated retraining pipeline  
- Cloud deployment (AWS/GCP)  
- Customer intervention simulator  

---

## 👤 Author

**Koyalkar Sri Harsha**  
MBA – Business Analytics  
Aspiring Data Scientist | Machine Learning & Analytics
