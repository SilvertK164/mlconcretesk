import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar los modelos y el scaler
modelF = pickle.load(open('modelF.pkl', 'rb'))
modelS = pickle.load(open('modelS.pkl', 'rb'))
modelT = pickle.load(open('modelT.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Título y descripción
st.title('Resistencia a la Compresión del Concreto - ML 🤖🏗️')
st.write('Esta aplicación fue creada a partir de tres modelos de Machine Learning')
st.write('by: Silvert Kevin Quispe Pacompia')

# Expander para mostrar datos
with st.expander('Data'):
    st.write('**Data de Concreto 🏗️**')
    df = pd.read_csv('Concrete_Data_New.csv')
    st.dataframe(df)

# Sidebar para la entrada de insumos
with st.sidebar:
    cemento = st.slider("Cemento [kg]", 0, 100, 50)
    escoria = st.slider("Escoria [kg]", 0, 100, 50)
    ceniza = st.slider("Ceniza [kg]", 0, 100, 50)
    agua = st.slider("Agua [kg]", 0, 100, 50)
    superplastificante = st.slider("Superplastificante [kg]", 0, 100, 50)
    ag_grueso = st.slider("Agregado Grueso [kg]", 0, 100, 50)
    ag_fino = st.slider("Agregado Fino [kg]", 0, 100, 50)

# Mostrar los inputs seleccionados
data = {
    'Cemento': cemento,
    'EscoriaAltoHorno': escoria,
    'CenizaVolante': ceniza,
    'Agua': agua,
    'Superplastificante': superplastificante,
    'Agregado Grueso': ag_grueso,
    'Agregado Fino': ag_fino
}
input_df = pd.DataFrame(data, index=[0])
st.write("**Inputs seleccionados**")
st.dataframe(input_df)

# Preparar los datos de entrada para la predicción
datos = np.array([[cemento, escoria, ceniza, agua, superplastificante, ag_grueso, ag_fino]])

# Lista de edades a evaluar
edades = [7, 14, 21, 28]
prediccionesF = []
prediccionesS = []
prediccionesT = []

# Predecir la resistencia para cada edad
for edad in edades:
    # Normalizar el registro (la suma se realiza sobre todos los insumos)
    nuevo_registro_normalizado = datos / np.sum(datos)
    # Agregar la edad como columna (se concatena horizontalmente)
    nuevo_registro_normalizado = np.hstack((nuevo_registro_normalizado, np.array([[edad]])))
    # Escalar los datos
    nuevo_registro_scaled = scaler.transform(nuevo_registro_normalizado)
    # Predecir con cada modelo
    predF = modelF.predict(nuevo_registro_scaled)[0]
    predS = modelS.predict(nuevo_registro_scaled)[0]
    predT = modelT.predict(nuevo_registro_scaled)[0]
    
    prediccionesF.append(predF)
    prediccionesS.append(predS)
    prediccionesT.append(predT)

