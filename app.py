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

import numpy as np

def generar_historial():

    np.random.seed(7)

    temporadas = list(range(2005,2026))

    puntos = np.random.normal(35,8,len(temporadas)).astype(int)
    victorias = np.random.randint(6,18,len(temporadas))
    posicion = np.random.randint(1,20,len(temporadas))

    titulos = []

    for año in temporadas:

        if año in [2009,2016]:
            titulos.append(1)
        else:
            titulos.append(0)

    df = pd.DataFrame({
        "Temporada":temporadas,
        "Puntos":puntos,
        "Victorias":victorias,
        "Posicion":posicion,
        "Titulo":titulos
    })

    return df

import numpy as np
from sklearn.linear_model import LinearRegression

def calcular_estadisticas(df):

    return {

        "temporadas":len(df),

        "victorias":df["Victorias"].sum(),

        "titulos":df["Titulo"].sum(),

        "promedio":df["Puntos"].mean()

    }


def predecir_proximo_titulo(df,horizonte):

    años = np.array(df["Temporada"]).reshape(-1,1)

    puntos = np.array(df["Puntos"])

    modelo = LinearRegression()

    modelo.fit(años,puntos)

    futuros = np.arange(
        df["Temporada"].max()+1,
        df["Temporada"].max()+horizonte+1
    )

    pred = modelo.predict(futuros.reshape(-1,1))

    indice = pred.argmax()

    return int(futuros[indice])
