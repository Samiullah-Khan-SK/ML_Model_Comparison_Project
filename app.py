import streamlit as st
import pandas as pd
import joblib
import json

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl")
}

# Load metrics
with open("models/model_metrics.json") as f:
    metrics = json.load(f)

st.title("📊 Sales Performance Prediction App")

st.sidebar.header("Enter Input Data")
region = st.number_input("Region (encoded)", 0)
rep = st.number_input("Rep (encoded)", 0)
item = st.number_input("Item (encoded)", 0)
units = st.number_input("Units Sold", 1)
unit_cost = st.number_input("Unit Cost", 1.0)

input_data = pd.DataFrame([[region, rep, item, units, unit_cost]],
                          columns=['Region', 'Rep', 'Item', 'Units', 'UnitCost'])

st.subheader("Select Model for Prediction")
selected_model = st.selectbox("Choose Model", list(models.keys()))

if st.button("Predict Sales Category"):
    model = models[selected_model]
    prediction = model.predict(input_data)[0]
    category = "💰 High Sales" if prediction == 1 else "📉 Low Sales"
    st.success(f"Prediction: {category}")

st.markdown("---")
st.subheader("📈 Model Performance Metrics")

st.dataframe(pd.DataFrame(metrics))
