import streamlit as st
import pandas as pd

st.title('Resistencia a la Compresión del Concreto - ML 🤖🏗️')

st.write('Esta aplicación fue creada a partir de tres modelo de Machine Learning')
st.write('by: Silvert Kevin Quispe Pacompia')

with st.expander('Data'):
  st.write('**Data de Concreto 🏗️')
  df = pd.read_csv('https://github.com/SilvertK164/mlconcretesk/blob/d7e3691d4c60807f35064cb4f2f7bd267c228bf4/Concrete_Data.csv')
  df
