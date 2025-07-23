import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Loading model
model = joblib.load("best_model.pkl")

# Mapping features values as we have encoded these values while model training
education_map = {
    "HS-grad":0, "Some-college":1, "Bachelors":2, "Masters":3,
    "Assoc-voc":4, "11th":5, "Assoc-acdm":6, "10th":7, "7th-8th":8,
    "Prof-school":9, "9th":10, "12th":11, "Doctorate":12
}

workclass_list = ["Private","Self-emp-not-inc","Local-gov","State-gov","Federal-gov"]
marital_list = ["Married-civ-spouse","Divorced","Never-married","Separated","Widowed"]
occupation_list = ["Tech-support","Craft-repair","Other-service","Sales",
                 "Exec-managerial","Prof-specialty","Handlers-cleaners",
                 "Machine-op-inspct","Adm-clerical","Farming-fishing",
                 "Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"]
relationship_list = ["Husband","Wife","Not-in-family","Other-relative"]
race_list = ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"]
country_list = ["United-States","Mexico","Philippines","Germany","Canada"]

# Encoders
workclass_enc = LabelEncoder().fit(workclass_list)
marital_enc = LabelEncoder().fit(marital_list)
occupation_enc = LabelEncoder().fit(occupation_list)
relationship_enc = LabelEncoder().fit(relationship_list)
race_enc = LabelEncoder().fit(race_list)
country_enc = LabelEncoder().fit(country_list)


st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="centered")
st.title("💰 Income Predictor")
st.write("Predict whether annual income exceeds $50K")

# Sidebar
with st.sidebar:
    st.header("Personal Details")
    age = st.slider("Age", 18, 65, 30)
    workclass = st.selectbox("Work Class", workclass_list)
    education = st.selectbox("Education Level", list(education_map.keys()))
    marital = st.selectbox("Marital Status", marital_list)
    occupation = st.selectbox("Occupation", occupation_list)
    relationship = st.selectbox("Relationship", relationship_list)
    race = st.selectbox("Race", race_list)
    gender = st.selectbox("Gender", ["Male","Female"])
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 10000, 0)
    hours = st.slider("Hours per Week", 1, 100, 40)
    country = st.selectbox("Native Country", country_list)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000)

# Main Screen
st.subheader("Your Input Summary")
input_summary = pd.DataFrame({
    "Feature": ["Age", "Work Class", "Education", "Marital Status", "Occupation",
               "Relationship", "Race", "Gender", "Capital Gain", "Capital Loss",
               "Hours/Week", "Country", "Final Weight"],
    "Value": [age, workclass, education, marital, occupation,
             relationship, race, gender, capital_gain, capital_loss,
             hours, country, fnlwgt]
})
st.table(input_summary)

# Prediction
if st.button("Predict Income"):
    input_data = {
        'age': age,
        'workclass': workclass_enc.transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education': education_map[education],
        'marital-status': marital_enc.transform([marital])[0],
        'occupation': occupation_enc.transform([occupation])[0],
        'relationship': relationship_enc.transform([relationship])[0],
        'race': race_enc.transform([race])[0],
        'gender': 1 if gender == "Male" else 0,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours,
        'native-country': country_enc.transform([country])[0]
    }
    
    prediction = model.predict(pd.DataFrame([input_data]))
    proba = model.predict_proba(pd.DataFrame([input_data]))[0]
    
    st.subheader("Prediction Result")
    if prediction[0] == ">50K":
        st.success(f"✅ High Earner (>$50K) with {proba[1]*100:.1f}% confidence")
    else:
        st.info(f"💼 Moderate Earner (≤$50K) with {proba[0]*100:.1f}% confidence")

st.caption("Model accuracy: ~86.7%")
