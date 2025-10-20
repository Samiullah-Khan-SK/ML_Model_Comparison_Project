#  Sales Performance Prediction & Model Comparison Dashboard

##  Overview
This project demonstrates how to compare multiple **Machine Learning classification models** using a real-world sales dataset and visualize their performance.  
It also includes a **Streamlit web application** to test predictions interactively using the trained models.

---

## Key Features
✅ Train and evaluate **4 ML models**:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

✅ Automatically compute metrics:
- Accuracy, Precision, Recall, F1-Score, and Training Time  

✅ Visualize:
- Model Accuracy Comparison  
- Precision / Recall / F1 Comparison  
- Confusion Matrices  
- ROC Curves  

✅ Save all trained models using `joblib`  
✅ Streamlit app for interactive model testing  
✅ Performance metrics stored in JSON for quick reference  

---

##  Tech Stack
| Category | Tools |
|-----------|--------|
| Programming | Python |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, joblib |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Model Storage | Joblib, JSON |

---

##  Project Structure

```
SalesPerformance/
│
├── data/
│   └── Data.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── model_metrics.json
│
├── train_and_compare.py   
├── app.py                 
├── requirements.txt       
└── README.md              
```

---

##  Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/Sales-Performance-ML-Models.git
cd Sales-Performance-ML-Models
```

### 2️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Model Training Script
```bash
python train_and_compare.py
```

### 4️⃣ Run Streamlit App
```bash
streamlit run app.py
```

---

##  Example Metrics Output
| Model | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|--------|-----------|------------|----------|------------|----------------|
| Random Forest | 0.92 | 0.91 | 0.93 | 0.92 | 0.023 |
| Logistic Regression | 0.88 | 0.87 | 0.88 | 0.87 | 0.004 |
| Decision Tree | 0.86 | 0.85 | 0.86 | 0.85 | 0.006 |
| SVM | 0.89 | 0.88 | 0.90 | 0.89 | 0.021 |

---

##  Visual Outputs
- **Bar charts** for accuracy and precision/recall/F1
- **Confusion matrix** for each model
- **ROC curve** comparison


##  Streamlit App Preview
The Streamlit app allows users to:
- Input new data (Region, Rep, Item, Units, UnitCost)
- Select which trained model to use
- Predict whether the sales are **High ** or **Low **
- View all model performance metrics directly in the app

---

## 🪄 How Model Storage Works
All models are saved using `joblib`:
```python
joblib.dump(model, 'models/random_forest_model.pkl')
```

And later loaded in Streamlit:
```python
model = joblib.load('models/random_forest_model.pkl')
```

Performance metrics are stored in JSON format:
```python
with open('models/model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)
``
