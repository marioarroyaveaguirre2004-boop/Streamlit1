import streamlit as st
import pandas as pd
import plotly.express as px

from data_generator import generar_historial
from analysis import calcular_estadisticas, predecir_proximo_titulo

st.set_page_config(
    page_title="Análisis Independiente Medellín",
    layout="wide"
)

st.title("🔴🔵 Análisis Histórico - Independiente Medellín")

st.write(
"""
Esta aplicación simula el desempeño histórico del DIM y realiza
análisis estadísticos junto con una estimación de cuándo podría
volver a ganar una Liga Colombiana.
"""
)

# --------------------------
# Simulación
# --------------------------

df = generar_historial()

st.subheader("Datos simulados")

st.dataframe(df)

# --------------------------
# Estadísticas
# --------------------------

stats = calcular_estadisticas(df)

col1,col2,col3,col4 = st.columns(4)

col1.metric("Temporadas", stats["temporadas"])
col2.metric("Victorias", stats["victorias"])
col3.metric("Títulos", stats["titulos"])
col4.metric("Promedio de puntos", round(stats["promedio"],1))

# --------------------------
# Gráficos
# --------------------------

st.subheader("Puntos por temporada")

fig = px.line(
    df,
    x="Temporada",
    y="Puntos",
    markers=True
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("Posición final")

fig2 = px.bar(
    df,
    x="Temporada",
    y="Posicion"
)

st.plotly_chart(fig2,use_container_width=True)

# --------------------------
# Interacción
# --------------------------

st.subheader("Predicción")

años = st.slider(
    "Horizonte de simulación",
    1,
    15,
    5
)

pred = predecir_proximo_titulo(df,años)

st.success(
    f"Según la simulación, el DIM tendría mayor probabilidad de ganar nuevamente en el año **{pred}**."
)

st.info(
"""
Este resultado es únicamente una simulación basada en tendencias
estadísticas y NO representa una predicción deportiva real.
"""
)
