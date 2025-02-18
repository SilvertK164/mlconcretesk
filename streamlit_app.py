import streamlit as st
import pandas as pd

st.title('Resistencia a la Compresión del Concreto - ML 🤖🏗️')

st.write('Esta aplicación fue creada a partir de tres modelo de Machine Learning')
st.write('by: Silvert Kevin Quispe Pacompia')

with st.expander('Data'):
  st.write('**Data de Concreto 🏗️**')
  df = pd.read_csv('Concrete_Data_New.csv')
  df

with st.sidebar:
  cemento = st.slider("Cemento [kg]",0 ,100, 50)
  escoria = st.slider("Escoria [kg]",0 ,100, 50)
  ceniza = st.slider("Ceniza [kg]",0 ,100, 50)
  agua = st.slider("Agua [kg]",0 ,100, 50)
  superplastificante = st.slider("Superplastificante [kg]",0 ,100, 50)
  ag_grueso = st.slider("Agregado Grueso [kg]",0 ,100, 50)
  ag_fino = st.slider("Agregado Fino [kg]",0 ,100, 50)
  #DATOS DE ENTRADA
  datos = [[cemento, escoria, ceniza, agua, superplastificante, ag_grueso, ag_fino]]
