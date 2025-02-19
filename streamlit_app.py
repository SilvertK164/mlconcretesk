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
    cemento = st.number_input("Cemento [kg]", value=50, step=1)
    escoria = st.number_input("Escoria [kg]", value=50, step=1)
    ceniza = st.number_input("Ceniza [kg]", value=50, step=1)
    agua = st.number_input("Agua [kg]", value=50, step=1)
    superplastificante = st.number_input("Superplastificante [kg]", value=50, step=1)
    ag_grueso = st.number_input("Agregado Grueso [kg]", value=50, step=1)
    ag_fino = st.number_input("Agregado Fino [kg]", value=50, step=1)

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

# Crear DataFrames para mostrar las predicciones, usando la edad como índice
dfF = pd.DataFrame({'Edad': edades, 'Resistencia Modelo F': prediccionesF}).set_index('Edad')
dfS = pd.DataFrame({'Edad': edades, 'Resistencia Modelo S': prediccionesS}).set_index('Edad')
dfT = pd.DataFrame({'Edad': edades, 'Resistencia Modelo T': prediccionesT}).set_index('Edad')

# Mostrar los gráficos utilizando las funciones integradas de Streamlit
st.subheader("Evolución de la Resistencia (MPa) - Modelo F")
st.line_chart(dfF)

st.subheader("Evolución de la Resistencia (MPa) - Modelo S")
st.line_chart(dfS)

st.subheader("Evolución de la Resistencia (MPa) - Modelo T")
st.line_chart(dfT)
