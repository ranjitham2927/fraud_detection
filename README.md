# 🛡️ Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using **machine learning** techniques.  
We use the **Credit Card Fraud Detection Dataset** from Kaggle, which contains transactions made by European cardholders in September 2013.  
The dataset is highly imbalanced, with only **0.172%** of transactions being fraudulent.

---

## 📂 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows:** 284,807  
- **Columns:** 31 (including `Class` label)  
- **Target Variable:**  
  - `0` → Non-Fraud  
  - `1` → Fraud  

**Note:** The dataset has been transformed using **PCA** for confidentiality, except for `Time` and `Amount`.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas** for data handling
- **NumPy** for numerical operations
- **Matplotlib / Seaborn** for visualization
- **Scikit-learn** for machine learning models

---

## 📊 Steps Performed

### 1️⃣ Data Exploration (EDA)
- Checked dataset shape, missing values, and class distribution  
- Visualized fraud vs non-fraud transactions  
- Created correlation heatmaps for features

### 2️⃣ Data Preprocessing
- Scaled the `Amount` feature using **StandardScaler**
- Split dataset into train/test sets using **stratified sampling** to handle imbalance

### 3️⃣ Model Building
We implemented two models:
- **Logistic Regression**
- **Random Forest Classifier**

### 4️⃣ Model Evaluation
Metrics used:
- Accuracy
- Precision, Recall, F1-score
- ROC AUC Score
- Confusion Matrix

---

## 📈 Results

| Model                | Accuracy | ROC AUC |
|----------------------|----------|---------|
| Logistic Regression  | ~99.9%   | 0.95+   |
| Random Forest        | ~99.9%   | 0.99+   |

✅ **Random Forest** performed slightly better and handled imbalance well.

---

## 📌 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
