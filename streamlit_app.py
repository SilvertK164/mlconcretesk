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
  
edades_np = np.array(edades)
prediccionesF_np = np.array(prediccionesF)
prediccionesS_np = np.array(prediccionesS)
prediccionesT_np = np.array(prediccionesT)

# Generar y mostrar los gráficos en Streamlit
st.subheader("Evolución de la Resistencia (MPa)")

# Gráfico para el Modelo F
figF, axF = plt.subplots(figsize=(8, 6))
axF.plot(edades_np, prediccionesF_np, marker='o', linestyle='-', color='blue')
axF.set_title("Resistencia Modelo F")
axF.set_xlabel("Edad (días)")
axF.set_ylabel("Resistencia (MPa)")
axF.grid(True)
st.pyplot(figF)

# Gráfico para el Modelo S
figS, axS = plt.subplots(figsize=(8, 6))
axS.plot(edades_np, prediccionesS_np, marker='o', linestyle='-', color='green')
axS.set_title("Resistencia Modelo S")
axS.set_xlabel("Edad (días)")
axS.set_ylabel("Resistencia (MPa)")
axS.grid(True)
st.pyplot(figS)

# Gráfico para el Modelo T
figT, axT = plt.subplots(figsize=(8, 6))
axT.plot(edades_np, prediccionesT_np, marker='o', linestyle='-', color='black')
axT.set_title("Resistencia Modelo T")
axT.set_xlabel("Edad (días)")
axT.set_ylabel("Resistencia (MPa)")
axT.grid(True)
st.pyplot(figT)
  





