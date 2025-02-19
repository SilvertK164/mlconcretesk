import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configurar la página para un layout amplio
st.set_page_config(layout="wide")

# Inyectar CSS para personalizar la apariencia con la paleta de colores
st.markdown(
    """
    <style>
    /* Cambiar el color de fondo general y el texto */
    body {
        color: #3A3A3A;
        background-color: #FFFFFF;
    }
    /* Títulos y encabezados */
    h1 {
        color: #003459;
    }
    h2, h3 {
        color: #007EA7;
    }
    /* Personalizar el sidebar (fondo y texto) */
    .css-1d391kg {
        background-color: #FF8C42;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
    st.image("aecode-logo.webp", width=150)
    st.write('**✍️ ENTRADA**')
    cemento = st.number_input("Cemento [kg]", value=181.4, step=0.01, format="%.2f")
    escoria = st.number_input("Escoria [kg]", value=0.0, step=0.01, format="%.2f")
    ceniza = st.number_input("Ceniza [kg]", value=167.0, step=0.01, format="%.2f")
    agua = st.number_input("Agua [kg]", value=169.6, step=0.01, format="%.2f")
    superplastificante = st.number_input("Superplastificante [kg]", value=7.6, step=0.01, format="%.2f")
    ag_grueso = st.number_input("Agregado Grueso [kg]", value=1055.6, step=0.01, format="%.2f")
    ag_fino = st.number_input("Agregado Fino [kg]", value=777.8, step=0.01, format="%.2f")

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

# Crear un DataFrame unificado usando la edad como índice
df_all = pd.DataFrame({
    'RandomForestRegressor': prediccionesF,
    'GradientBoostingRegressor': prediccionesS,
    'BaggingRegressor': prediccionesT
}, index=edades)

# Mostrar el gráfico combinado
st.subheader("Evolución de la Resistencia (MPa) - Todos los Modelos")
st.line_chart(df_all)

# Mostrar la tabla con las predicciones
st.subheader("Tabla de Predicciones")
st.dataframe(df_all)
