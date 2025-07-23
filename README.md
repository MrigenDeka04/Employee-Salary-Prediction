#  Employee Salary Prediction - AI Internship Project

This project is a machine learning-based salary prediction app developed as part of the **Artificial Intelligence Virtual Internship** offered by **Edunet Foundation** in collaboration with **IBM SkillsBuild** and **AICTE**.

---

##  Project Overview

This project predicts whether an employee earns more than 50K or not based on various personal and professional attributes using classic ML models.

The final application is deployed using **Streamlit**, and it includes:
- Data cleaning & preprocessing
- Feature encoding and scaling
- ML model training & comparison
- Best model selection
- Streamlit interface for predictions

---

##  Dataset

We used the **UCI Adult Dataset** for income classification.

📁 File: [`adult.data`](./adult.data)  
📚 Source: [UCI ML Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

---

##  Algorithms Used

- Logistic Regression ✅ (Best model)
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes

---

##  Final Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression ✅ | 0.8571   | 0.7498    | 0.6301 | 0.6847   |
| Random Forest         | 0.8554   | 0.7409    | 0.6351 | 0.6839   |
| SVM                   | 0.8479   | 0.7462    | 0.5795 | 0.6524   |

---

##  How to Run the Project Locally

###  Prerequisites
- Python 3.10+
- pip
- Jupyter Notebook
- Streamlit


---

##  Repository Structure

```
Employee-Salary-Prediction/
│
├── app.py                     # Streamlit app script
├── adult.data                 # Dataset file
├── scaler.pkl                 # Saved scaler
├── logistic_model.pkl         # Trained model
├── feature_columns.pkl        # List of features used
├── requirements.txt           # Python dependencies
├── Employee-Salary-Prediction.ipynb  # Jupyter notebook
└── README.md
```


## 🏁 Acknowledgements

Special thanks to:
- **IBM SkillsBuild**
- **Edunet Foundation**
- **AICTE**
- All mentors who guided during this AI Internship

---

> 🙏 Thank you for visiting my project. Do ⭐ the repo if you find it useful or inspiring!
