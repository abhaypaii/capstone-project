
# 🧠 Fitbit Behavioral Clustering & Regression Modeling

This project leverages machine learning techniques to segment Fitbit users into behavioral groups using unsupervised learning and address missing health data using supervised regression. The end goal is to generate actionable consumer insights to support targeted marketing strategies — mimicking a real-world consumer packaged goods (CPG) application.

## 📌 Objective

To identify meaningful customer segments from health tracker data (sleep, activity, weight, etc.) and impute missing values using machine learning, thereby enhancing the quality of insights for data-driven marketing decisions.

---

## 🚀 Key Highlights

- ✅ **Regression Modeling**: Applied XGBoost Regression to impute missing values in BMI, sleep, and weight, preserving non-linear feature relationships.
- ✅ **Unsupervised Clustering**: Used K-Means clustering to group users into 4 behavior-based segments for downstream marketing strategy alignment.
- ✅ **Time-Series Feature Engineering**: Extracted and aggregated temporal patterns (sleep patterns, activity trends) from minute-by-minute data.
- ✅ **Business Relevance**: Derived user personas to support marketing decisions for a CPG-like product strategy.
- ✅ **End-to-End Pipeline**: From data wrangling and feature engineering to model training, validation, and dashboard-ready insights.

---

## 🧮 Techniques & Tools

| Category               | Approach/Tool                                    |
|------------------------|--------------------------------------------------|
| Data Processing        | `pandas`, `numpy`, `datetime`                    |
| Feature Engineering    | Time-based aggregation, missing value treatment  |
| Regression Modeling    | `XGBoost` for supervised learning                |
| Clustering             | `KMeans` from `sklearn.cluster`                  |
| Visualization          | `matplotlib`, `seaborn`, `plotly`               |
| Dashboard/Presentation | `Jupyter Notebook`, Python-based visual reports  |

---

## 📊 Cluster Profiles (Sample Summary)

| Cluster | Characteristics                             | Marketing Insight                                 |
|---------|----------------------------------------------|---------------------------------------------------|
| 0       | Low activity, irregular sleep                | Promote habit-building programs                   |
| 1       | High activity, consistent sleep              | Cross-sell performance accessories or subscriptions |
| 2       | Moderate metrics but high variability        | Educate on healthy routines and consistency       |
| 3       | Restless sleepers, low calories burned       | Target with sleep-focused wellness campaigns      |

---

## 📁 Project Structure

