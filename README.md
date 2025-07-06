# Proyecto de Machine Learning: Predicción de la Pérdida de Cabello

## 🧠 Objetivo
Desarrollar un modelo de clasificación binaria que prediga si una persona sufrirá pérdida de cabello (`Hair Loss`) en función de múltiples factores personales, médicos y de estilo de vida.

## 📊 Dataset
El conjunto de datos utilizado incluye información como:
- Edad
- Genética
- Cambios hormonales
- Cuidados capilares
- Entorno
- Tabaquismo
- Peso
- Estrés 
- Condiciones médicas
- Tratamientos médicos
- Deficiencias nutricionales

**Target:** `Hair Loss` (0 = No, 1 = Sí)

El dataset está equilibrado en cuanto a la variable objetivo.

## 🛠️ Preprocesamiento
- Conversión de valores categóricos `Yes/No` a booleanos o 1/0.
- Normalización de la variable `Age` usando `StandardScaler`.
- Separación en conjuntos de entrenamiento y test (`train_test_split` con 20% para test).
- Se han creado dos csv: uno completo y otro reducido 

## 📦 Modelos Entrenados
Se entrenaron y compararon varios modelos supervisados:

### Modelos:
- `Logistic Regression`
- `K-Nearest Neighbors (KNN)`
- `Random Forest`
- `XGBoost`
- `Support Vector Machine (SVM)`

Además, se exploró el uso de:
- `PCA` para reducción de dimensionalidad
- `KMeans` como modelo no supervisado para clusterización exploratoria

## 🔍 Evaluación
Se evaluaron los modelos usando:
- Matriz de confusión
- Accuracy
- Precision
- Recall
- F1-score


## 🧪 Resultados
Ningún modelo superó significativamente el 55-60% de accuracy debido a:
- Muy baja correlación entre las variables y el target
- Datos muy equilibrados pero sin variables predictivas fuertes
- Información médica con muchas variables binarias con efecto marginal

Se probó la combinación de PCA + KNN sin mejoras notables.

## 💾 Modelos Guardados (`.pkl`)
Se guardaron los modelos entrenados y los `scalers` correspondientes con `pickle` para su posterior carga y uso, tanto los obtenidos con lso datos completos como los simplificados, indicados con una "B".


## 📁 Estructura del Proyecto

Proyecto_ML_Hair_Loss/
├── app_streamlit/
│   ├── appKNN.py
│
├── data/
│   ├── processed/
│   │   └── limpio1.csv    
│   │   └── simple.csv   
│   └── raw/
│       └── Predict Hair Fall.csv      
│
├── docs/
│   ├── presentacion_tecnica.pdf
│   └── presentación_negocio.pdf    
│
├── models/
│   ├── final/
│   │   └── M4B_XGBoost.pkl
│   ├── M1_LR.pkl
│   ├── M1_scaler.pkl
│   ├── M1B_LR.pkl
│   ├── M1B_scaler.pkl
│   ├── M2_KNN.pkl.py
│   ├── M2_minmax_scaler.pkl
│   ├── M2B_KNN.pkl
│   ├── M2B_minmax_scaler.pkl
│   ├── M3_RF.pkl
│   ├── M3B_RF.pkl
│   ├── M4_XGBoost.pkl
│   ├── M4B_XGBoost.pkl
│   ├── M5_SVM.pkl
│   ├── M5_scaler.pkl
│   ├── M5B_SVM.pkl
│   ├── M5B_scaler.pkl
│   ├── M6_knn.pkl
│   ├── M6_PCA.pkl
│   ├── M6B_knn.pkl
│   ├── M6B_PCA.pkl
│   ├── M7_kmeans.pkl
│   ├── M7_PCA.pkl
│   ├── M7B_kmeans.pkl
│   └── M7B_PCA.pkl
│
├── notebooks/
│   ├── KNN-variables_raras.ipynb
│   ├── _pruebas3.ipynb
│   ├── Limpieza.ipynb
│   ├── M1-LogRegr.ipynb
│   ├── M2-KNN.ipynb
│   ├── M3-RF.ipynb
│   ├── M4-XGBoost.ipynb
│   ├── M5-SVM.ipynb
│   ├── M6-PCA_y_KNN.ipynb
│   └── M7-KMeans.ipynb
├── notebooks simplificado/
│   ├── M1-LogRegr.ipynb
│   ├── M2-KNN.ipynb
│   ├── M3-RF.ipynb
│   ├── M4-XGBoost.ipynb
│   ├── M5-SVM.ipynb
│   ├── M6-PCA_y_KNN.ipynb
│   └── M7-KMeans.ipynb
│
└── README.md


## 🧠 Conclusión
Aunque los modelos no alcanzaron una gran precisión, el proyecto demuestra el flujo completo de un caso de clasificación en Machine Learning, desde la limpieza de datos hasta la evaluación, uso de PCA, y almacenamiento de modelos. 



