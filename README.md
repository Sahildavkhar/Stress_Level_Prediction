# 🧠 Stress Level Prediction

This project predicts a person's **stress level** based on multiple lifestyle and psychological factors using **Machine Learning** models.  
It compares the performance of **Random Forest**, **SVM**, and **XGBoost** using **GridSearchCV**, and visualizes the results in a **Streamlit web app**.

---

## 📋 Project Overview

This system analyzes inputs such as:
- Age  
- Gender  
- Work Pressure  
- Sleep Duration  
- Financial Stress  

and predicts the **Stress Level (Depression)** of an individual.

The project is divided into two main parts:
1. **Model Training (`stress_model.py`)** — trains and compares ML models using GridSearchCV.  
2. **Streamlit App (`app.py`)** — interactive web interface for real-time predictions.

---

## 🧰 Technologies Used

- **Python 3.10+**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Joblib**
- **Streamlit**

---

## 📊 GridSearchCV Parameters

| Model | Tuned Parameters |
|--------|------------------|
| **Random Forest** | `n_estimators=[100]`, `max_depth=[10]` |
| **SVM** | `C=[1]`, `kernel=['linear']` |
| **XGBoost** | `n_estimators=[100]`, `max_depth=[5]`, `learning_rate=[0.1]` |

The best parameters and performance metrics are automatically saved in `model_comparison.csv`.

---

## 🚀 How to Run

### 🖥️ Step 1 — Clone Repository
```bash
git clone https://github.com/Sahildavkhar/Stress_Level_Prediction.git
cd Stress_Level_Prediction
```

### 🖥️ Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
If requirements.txt is not available, manually install:
pip install pandas scikit-learn xgboost streamlit joblib
```

### 🖥️ Step 3 — Train Models
```bash
python stress_model.py
```

### 🖥️ Step 4 — Run Streamlit App
```bash
streamlit run app.py
```

📜 License
This project is open-source and available under the MIT License.



