import streamlit as st
import pandas as pd
import pickle

# Load Model and Scaler
model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

st.set_page_config(page_title="Diabetic Patient Prediction", page_icon="🩺", layout="wide")

st.title("🩺 Diabetic Patient Prediction")

# Input Columns
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.slider("Pregnancies", 0, 17, 1)
    BloodPressure = st.slider("Blood Pressure", 40, 140, 72)
    Insulin = st.slider("Insulin", 15, 300, 80)
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.05,
        max_value=3.0,
        value=0.47,
        step=0.001,
        format="%.3f"
    )

with col2:
    Glucose = st.slider("Glucose", 50, 200, 120)
    SkinThickness = st.slider("Skin Thickness", 7, 99, 20)
    BMI = st.slider("BMI", 18.0, 50.0, 32.0, step=0.1)
    Age = st.slider("Age", 21, 81, 33)

# Prediction
if st.button("Predict"):

    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    data = [[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]]

    df = pd.DataFrame(data, columns=columns)
    df = pd.DataFrame(sc.transform(df), columns=columns)

    prediction = model.predict(df)

    st.divider()

    if prediction[0] == 1:
        st.error("🩺 Patient is Diabetic")

        c1, c2 = st.columns(2)

        with c1:
            st.info("💊 Take Medicine")
            st.info("🥗 Healthy Diet")

        with c2:
            st.info("🏃 Exercise")
            st.info("👨‍⚕️ Consult Doctor")

    else:
        st.balloons()
        st.success("🎉 Patient is Not Diabetic")

        c1, c2 = st.columns(2)

        with c1:
            st.success("🍬 Enjoy Sweet")

        with c2:
            st.success("😊 Stay Healthy")