import streamlit as st
import numpy as np
import pickle
import os

st.write("La app está arrancando...")

# Rutas de los archivos (ajusta si hace falta)
modelo_path = "../models/M2B_KNN.pkl"
scaler_path = "../models/M2B_minmax_scaler.pkl"

st.write("Directorio actual:", os.getcwd())
st.write("Ruta del modelo:", os.path.abspath(modelo_path))
st.write("Ruta del scaler:", os.path.abspath(scaler_path))

# Comprobación de archivos
if not os.path.exists(modelo_path) or not os.path.exists(scaler_path):
    st.error("No se encuentra el modelo o el scaler.")
else:
    # Cargar modelo y scaler
    with open(modelo_path, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    st.title("¿Sufres riesgo de caída del cabello?")
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

    # Convertir entradas
    Stress = Stress_to_num(Stress)
    Genetics_encoder = binarize(Genetics_encoder)
    Hormones_encoder = binarize(Hormones_encoder)
    HairCare_encoder = binarize(HairCare_encoder)
    Smoking_encoder = binarize(Smoking_encoder)
    Weight_encoder = binarize(Weight_encoder)

    # Crear input no escalado
    input_data = np.array([[Stress, Age, Genetics_encoder, Hormones_encoder,
                            HairCare_encoder, Smoking_encoder, Weight_encoder]])

    # Escalar input completo
    input_scaled = scaler.transform(input_data)

    # Predicción
    if st.button("Predecir"):
        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            st.error("Riesgo de pérdida de cabello")
        else:
            st.success("Sin riesgo aparente de pérdida de cabello")



#poner esto en terminal: python -m streamlit run appKNN.py