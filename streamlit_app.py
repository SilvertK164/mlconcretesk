import streamlit as st
import pandas as pd
import numpy as np
import pickle

modelF = pickle.load(open('modelF.pkl', 'rb'))
modelS = pickle.load(open('modelS.pkl', 'rb'))
modelT = pickle.load(open('modelT.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title('Resistencia a la Compresión del Concreto - ML 🤖🏗️')

st.write('Esta aplicación fue creada a partir de tres modelo de Machine Learning')
st.write('by: Silvert Kevin Quispe Pacompia')

with st.expander('Data'):
  st.write('**Data de Concreto 🏗️**')
  df = pd.read_csv('Concrete_Data_New.csv')
  df
  
  #st.write('**Inputs**')
  #X = df.drop(['EdadDias','ResistenciaCompresion'],axis=1)
  #X

with st.sidebar:
  cemento = st.slider("Cemento [kg]",0 ,100, 50)
  escoria = st.slider("Escoria [kg]",0 ,100, 50)
  ceniza = st.slider("Ceniza [kg]",0 ,100, 50)
  agua = st.slider("Agua [kg]",0 ,100, 50)
  superplastificante = st.slider("Superplastificante [kg]",0 ,100, 50)
  ag_grueso = st.slider("Agregado Grueso [kg]",0 ,100, 50)
  ag_fino = st.slider("Agregado Fino [kg]",0 ,100, 50)
  
  #DATAFRAME PARA INPUTS
  data = {'Cemento': cemento,
       'EscoriaAltoHorno': escoria,
       'CenizaVolante' : ceniza,
       'Agua': agua,
       'Superplastificante': superplastificante,
       'Agregado Grueso': ag_grueso,
       'Agregado Fino': ag_fino}
  input_df = pd.DataFrame(data, index=[0])
  #input_concrete = pd.concat([input_df, X], axis=0)

input_df
  
  #DATOS DE ENTRADA
datos = [[cemento, escoria, ceniza, agua, superplastificante, ag_grueso, ag_fino]]
# Lista de edades a evaluar
edades = [7, 14, 21, 28]
prediccionesF = []
prediccionesS = []
prediccionesT = []

for edad in edades:
  nuevo_registro_normalizado = datos / np.sum(datos)
  nuevo_registro_normalizado = np.hstack((nuevo_registro_normalizado, np.array([[edad]])))
  nuevo_registro_scaled = scaler.transform(nuevo_registro_normalizado)
  predF = modelF.predict(nuevo_registro_scaled)[0]
  predS = modelS.predict(nuevo_registro_scaled)[0]
  predT = modelT.predict(nuevo_registro_scaled)[0]
  prediccionesF.append(predF)
  prediccionesS.append(predS)
  prediccionesT.append(predT)
  
# Crear DataFrames para cada conjunto de predicciones
dfF = pd.DataFrame({'Edad': edades, 'Resistencia Modelo F': prediccionesF}).set_index('Edad')
dfS = pd.DataFrame({'Edad': edades, 'Resistencia Modelo S': prediccionesS}).set_index('Edad')
dfT = pd.DataFrame({'Edad': edades, 'Resistencia Modelo T': prediccionesT}).set_index('Edad')

# Mostrar los gráficos usando las funciones integradas de Streamlit
st.subheader("Evolución de la Resistencia (MPa) - Modelo F")
st.line_chart(dfF)

st.subheader("Evolución de la Resistencia (MPa) - Modelo S")
st.line_chart(dfS)

st.subheader("Evolución de la Resistencia (MPa) - Modelo T")
st.line_chart(dfT)





