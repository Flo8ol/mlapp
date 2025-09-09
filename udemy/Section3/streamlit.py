import streamlit as st
import joblib

chemin = "udemy/"


model = joblib.load(chemin + 'linear_regression_model.pk1')
scaler = joblib.load(chemin + 'scaler.pk1')
st.title("Titre")
st.write("Entrer le nombre of hours")

hours = st.number_input("Hours:", min_value=0.0, step=1.0)

if st.button("Predict"):
    try :
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write(f"Prediction: {prediction[0]}")
    except Exception as e :
        st.error(e)
