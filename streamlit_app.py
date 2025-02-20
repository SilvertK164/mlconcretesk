import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Configurar la página para un layout amplio
st.set_page_config(layout="wide")

# Cargar los modelos y el scaler
modelF = pickle.load(open('modelF.pkl', 'rb'))
modelS = pickle.load(open('modelS.pkl', 'rb'))
modelT = pickle.load(open('modelT.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Título y descripción
st.markdown(
    "<h1 style='color: #007EA7;'> ML - RESISTENCIA A LA COMPRESIÓN DEL CONCRETO 🤖🏗️</h1>",
    unsafe_allow_html=True
)

st.subheader("Acerca de la Aplicación")
st.markdown(
    """
    Esta aplicación utiliza tres modelos de Machine Learning (RandomForestRegressor, 
    GradientBoostingRegressor y BaggingRegressor) para predecir la resistencia a la 
    compresión del concreto. Los insumos ingresados se normalizan y escalonan antes de 
    generar las predicciones para diferentes edades (en días). Además, se calcula un 
    promedio de las predicciones para brindar una visión consolidada.
    """
)

# Expander para mostrar datos
with st.expander('Data'):
    st.write('**Data de Concreto 🏗️**')
    df = pd.read_csv('Concrete_Data_New.csv')
    st.dataframe(df)

# Sidebar para la entrada de insumos
with st.sidebar:
    st.image("aecode-logo.webp", width=150)
    st.sidebar.title("✍️ENTRADA")
    cemento = st.number_input("Cemento [kg]", value=181.4, step=0.01, format="%.2f")
    escoria = st.number_input("Escoria [kg]", value=0.0, step=0.01, format="%.2f")
    ceniza = st.number_input("Ceniza [kg]", value=167.0, step=0.01, format="%.2f")
    agua = st.number_input("Agua [kg]", value=169.6, step=0.01, format="%.2f")
    superplastificante = st.number_input("Superplastificante [kg]", value=7.6, step=0.01, format="%.2f")
    ag_grueso = st.number_input("Agregado Grueso [kg]", value=1055.6, step=0.01, format="%.2f")
    ag_fino = st.number_input("Agregado Fino [kg]", value=777.8, step=0.01, format="%.2f")

########################################################################
# Diccionario de ingredientes y colores
ingredientes = {
    "Cemento": (cemento, "red"),
    "Escoria": (escoria, "blue"),
    "Ceniza": (ceniza, "green"),
    "Agua": (agua, "cyan"),
    "Superplastificante": (superplastificante, "magenta"),
    "Agregado Grueso": (ag_grueso, "orange"),
    "Agregado Fino": (ag_fino, "brown")
}

# Normalizar las alturas para visualización
total = sum([v[0] for v in ingredientes.values()])
alturas = {k: v[0] / total * 10 for k, v in ingredientes.items()}  # Altura proporcional

# Parámetros del cilindro
num_slices = len(ingredientes)
theta = np.linspace(0, 2 * np.pi, num_slices + 1)
z_top = np.cumsum(list(alturas.values()))
z_base = np.insert(z_top[:-1], 0, 0)

# Crear la figura
fig = go.Figure()

for i, (ingrediente, (valor, color)) in enumerate(ingredientes.items()):
    x = np.cos(theta[i:i+2])
    y = np.sin(theta[i:i+2])

    # Crear sección coloreada
    fig.add_trace(go.Mesh3d(
        x=[x[0], x[1], x[1], x[0], x[0], x[1], x[1], x[0]],
        y=[y[0], y[1], y[1], y[0], y[0], y[1], y[1], y[0]],
        z=[z_base[i], z_base[i], z_top[i], z_top[i], z_base[i], z_base[i], z_top[i], z_top[i]],
        color=color,
        name=ingrediente
    ))

    # Agregar etiquetas
    fig.add_trace(go.Scatter3d(
        x=[np.mean(x)],
        y=[np.mean(y)],
        z=[z_top[i] + 0.2],
        text=[f"{ingrediente}: {valor:.2f} kg/m³"],
        mode="text",
        showlegend=False
    ))

# Configurar el layout
fig.update_layout(
    title="🌀 Representación 3D de Ingredientes del Concreto",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Cantidad (kg/m³)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Mostrar en Streamlit
st.plotly_chart(fig, use_container_width=True)
########################################################################

    st.markdown("---")
    st.sidebar.header("Sígueme")
    st.sidebar.markdown(
        """
        <a href="https://www.linkedin.com/in/silvertq/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="24" alt="LinkedIn" style="vertical-align: middle;">LinkedIn
        </a><br>
        <a href="https://www.tiktok.com/@silvertk164" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/en/a/a9/TikTok_logo.svg" width="24" alt="TikTok" style="vertical-align: middle; filter: brightness(0) invert(1);">TikTok
        </a><br>
        <a href="https://github.com/SilvertK164" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="24" alt="GitHub" style="vertical-align: middle; filter: brightness(0) invert(1);">GitHub
        </a>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown("---")
    st.write('by: Silvert Kevin Quispe Pacompia')

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

st.subheader("Grafico de Resultados")
st.markdown("**Eje x:** Días |  **Eje y:** Resistencia (MPa)")
# Preparar los datos de entrada para la predicción
# Lista de edades a evaluar
edades = [7, 14, 21, 28]
prediccionesF = []
prediccionesS = []
prediccionesT = []

# Predecir la resistencia para cada edad
for edad in edades:
    datos = np.array([[cemento, escoria, ceniza, agua, superplastificante, ag_grueso, ag_fino]])
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


# Crear tres columnas con anchos relativos: izquierda, centro y derecha
col1, col2 = st.columns([0.25, 0.6])

# En la columna izquierda, definimos los checkboxes y construimos la lista de columnas seleccionadas
with col1:
    mostrar_rf = st.checkbox("RandomForestRegressor", value=True)
    mostrar_gb = st.checkbox("GradientBoostingRegressor", value=True)
    mostrar_bg = st.checkbox("BaggingRegressor", value=True)
    
    columnas_seleccionadas = []
    if mostrar_rf:
        columnas_seleccionadas.append("RandomForestRegressor")
    if mostrar_gb:
        columnas_seleccionadas.append("GradientBoostingRegressor")
    if mostrar_bg:
        columnas_seleccionadas.append("BaggingRegressor")

# En la columna central, mostramos el gráfico sólo si hay al menos una columna seleccionada
with col2:
    if columnas_seleccionadas:
        st.line_chart(df_all[columnas_seleccionadas])
    else:
        st.write("Por favor, seleccione al menos un modelo para mostrar el gráfico.")


# Calcular la columna 'Promedio'
df_all['Promedio'] = df_all.mean(axis=1)

# Resaltar las celdas de la columna 'Promedio' con un fondo verde claro (por ejemplo, #90EE90)
styled_df = df_all.style.applymap(lambda x: 'background-color: #90EE90; color: black;', subset=['Promedio'])

# Mostrar la tabla estilizada
st.info("El área resaltada en la tabla muestra el promedio de las predicciones de los modelos.")
st.subheader("Tabla de Predicciones con Promedio Resaltado")
st.dataframe(styled_df)

csv = df_all.to_csv().encode('utf-8')
st.download_button(
    "Descargar Predicciones",
    csv,
    "predicciones.csv",
    "text/csv",
    key='download-csv'
)
