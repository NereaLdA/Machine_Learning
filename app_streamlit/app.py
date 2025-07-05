import streamlit as st
import numpy as np
import pickle
import os
import streamlit as st


st.write("La app está arrancando...")

ruta = "../models/M4B_XGBoost.pkl"

st.write("Directorio actual:", os.getcwd())
st.write("Ruta completa del modelo:", os.path.abspath(ruta))
if not os.path.exists(ruta):
    st.error("El archivo del modelo no se encuentra en esa ruta.")
else:
    # Abrir y cargar el modelo desde el archivo pickle
    with open(ruta, 'rb') as f:
        model = pickle.load(f)

    st.title("¿Vas a perder el pelo? 🤔")

    st.markdown("Introduce tus datos para saber si tienes riesgo de pérdida de cabello.")

    # Entradas del usuario
    Stress = st.selectbox("Nivel de estrés", ["Bajo", "Medio", "Alto"])
    Age = st.slider("Edad", 10, 80, 30)
    Genetics_encoder = st.selectbox("¿Tiene factores genéticos?", ["Sí", "No"])
    Hormones_encoder = st.selectbox("¿Tiene problemas hormonales?", ["Sí", "No"])
    HairCare_encoder = st.selectbox("¿Se cuida poco el cabello?", ["Sí", "No"])
    Smoking_encoder = st.selectbox("¿Fuma?", ["Sí", "No"])
    Weight_encoder = st.selectbox("¿Ha perdido peso?", ["Sí", "No"])

    # Funciones para convertir entradas
    def binarize(resp):
        return 1 if resp == "Sí" else 0

    def Stress_to_num(level):
        mapping = {"Bajo": 0, "Medio": 1, "Alto": 2}
        return mapping[level]

    Stress = Stress_to_num(Stress)
    Genetics_encoder = binarize(Genetics_encoder)
    Hormones_encoder = binarize(Hormones_encoder)
    HairCare_encoder = binarize(HairCare_encoder)
    Smoking_encoder = binarize(Smoking_encoder)
    Weight_encoder = binarize(Weight_encoder)

    input_data = np.array([[Stress, Age, Genetics_encoder, Hormones_encoder, HairCare_encoder, Smoking_encoder, Weight_encoder]])

    if st.button("Predecir"):
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("😬 Riesgo de pérdida de cabello")
        else:
            st.success("😊 Sin riesgo aparente de pérdida de cabello")

#poner esto en terminal: python -m streamlit run app.py

