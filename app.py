import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Análisis Histórico - Independiente Medellín",
    page_icon="🔴",
    layout="wide"
)

# =====================================================
# GENERACIÓN DE DATOS SIMULADOS
# =====================================================

@st.cache_data
def generar_historial():

    np.random.seed(42)

    temporadas = np.arange(2005, 2026)

    puntos = np.random.randint(24, 48, len(temporadas))
    victorias = np.random.randint(6, 18, len(temporadas))
    empates = np.random.randint(4, 12, len(temporadas))
    derrotas = np.random.randint(2, 12, len(temporadas))

    posiciones = []

    for p in puntos:
        if p > 42:
            posiciones.append(np.random.randint(1,4))
        elif p > 36:
            posiciones.append(np.random.randint(4,8))
        elif p > 30:
            posiciones.append(np.random.randint(8,12))
        else:
            posiciones.append(np.random.randint(12,20))

    titulos = []

    for año in temporadas:
        if año in [2009, 2016]:
            titulos.append(1)
        else:
            titulos.append(0)

    goles_favor = np.random.randint(20, 48, len(temporadas))
    goles_contra = np.random.randint(18, 40, len(temporadas))

    df = pd.DataFrame({
        "Temporada": temporadas,
        "Puntos": puntos,
        "Victorias": victorias,
        "Empates": empates,
        "Derrotas": derrotas,
        "Posición": posiciones,
        "Goles a Favor": goles_favor,
        "Goles en Contra": goles_contra,
        "Título": titulos
    })

    return df

df = generar_historial()

# =====================================================
# ESTADÍSTICAS
# =====================================================

def estadisticas(df):

    return {
        "Temporadas": len(df),
        "Victorias": int(df["Victorias"].sum()),
        "Empates": int(df["Empates"].sum()),
        "Derrotas": int(df["Derrotas"].sum()),
        "Títulos": int(df["Título"].sum()),
        "Promedio Puntos": df["Puntos"].mean(),
        "Promedio Posición": df["Posición"].mean()
    }

stats = estadisticas(df)

# =====================================================
# PREDICCIÓN
# =====================================================

def predecir_titulo(df, horizonte):

    X = df["Temporada"].values.reshape(-1,1)
    y = df["Puntos"].values

    modelo = LinearRegression()
    modelo.fit(X,y)

    futuros = np.arange(
        df["Temporada"].max()+1,
        df["Temporada"].max()+horizonte+1
    )

    pred = modelo.predict(futuros.reshape(-1,1))

    mejor = futuros[np.argmax(pred)]

    return int(mejor), pred

# =====================================================
# INTERFAZ
# =====================================================

st.title("🔴🔵 Análisis del Independiente Medellín")

st.markdown("""
Esta aplicación **simula** el rendimiento histórico del Independiente Medellín,
muestra estadísticas, gráficos y estima en qué año tendría mayor probabilidad
de volver a ganar una Liga Colombiana.
""")

st.divider()

# =====================================================
# MÉTRICAS
# =====================================================

c1,c2,c3,c4 = st.columns(4)

c1.metric("Temporadas", stats["Temporadas"])
c2.metric("Victorias", stats["Victorias"])
c3.metric("Títulos", stats["Títulos"])
c4.metric("Promedio Puntos", f"{stats['Promedio Puntos']:.1f}")

st.divider()

# =====================================================
# TABLA
# =====================================================

st.subheader("Datos Simulados")

st.dataframe(df, use_container_width=True)

# =====================================================
# GRÁFICOS
# =====================================================

st.subheader("Puntos por Temporada")

fig1 = px.line(
    df,
    x="Temporada",
    y="Puntos",
    markers=True,
    title="Evolución de Puntos"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------------------------------

st.subheader("Posición Final")

fig2 = px.bar(
    df,
    x="Temporada",
    y="Posición",
    title="Posición por Temporada"
)

fig2.update_yaxes(autorange="reversed")

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------

st.subheader("Victorias por Temporada")

fig3 = px.bar(
    df,
    x="Temporada",
    y="Victorias",
    color="Victorias",
    title="Victorias"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------

st.subheader("Distribución de Puntos")

fig4 = px.histogram(
    df,
    x="Puntos",
    nbins=10,
    title="Distribución de Puntos"
)

st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# INTERACCIÓN
# =====================================================

st.divider()

st.subheader("🔮 Predicción del Próximo Título")

horizonte = st.slider(
    "Años a simular",
    1,
    20,
    8
)

año_predicho, valores = predecir_titulo(df, horizonte)

futuro = pd.DataFrame({
    "Temporada": np.arange(
        df["Temporada"].max()+1,
        df["Temporada"].max()+horizonte+1
    ),
    "Puntos Esperados": valores
})

fig5 = px.line(
    futuro,
    x="Temporada",
    y="Puntos Esperados",
    markers=True,
    title="Proyección de Rendimiento"
)

st.plotly_chart(fig5, use_container_width=True)

st.success(
    f"📈 Según esta simulación, el Independiente Medellín tendría su mejor oportunidad de ganar una liga en el año **{año_predicho}**."
)

st.info("""
⚠️ Esta predicción es únicamente una simulación basada en una tendencia lineal de los datos generados aleatoriamente.
No representa una predicción deportiva real.
""")

# =====================================================
# SIMULACIÓN DE TÍTULOS FUTUROS
# =====================================================

st.subheader("🔮 Simulación de títulos futuros")

años = st.slider(
    "¿Cuántos años desea simular?",
    5,
    50,
    20
)

# Probabilidad base de ser campeón cada año
prob_base = st.slider(
    "Probabilidad anual de ser campeón (%)",
    5,
    40,
    12
) / 100

años_futuros = []
campeon = []

for año in range(df["Temporada"].max()+1, df["Temporada"].max()+años+1):

    titulo = np.random.rand() < prob_base

    años_futuros.append(año)
    campeon.append(1 if titulo else 0)

pred_df = pd.DataFrame({
    "Temporada": años_futuros,
    "Título": campeon
})

titulos_futuros = pred_df["Título"].sum()

# Mostrar resultados

st.metric(
    "🏆 Ligas ganadas en la simulación",
    int(titulos_futuros)
)

if titulos_futuros > 0:

    años_campeon = pred_df[pred_df["Título"]==1]["Temporada"].tolist()

    st.success(
        "El Medellín sería campeón en:\n\n" +
        ", ".join(map(str,años_campeon))
    )

else:

    st.error(
        "En esta simulación el Medellín no ganó ninguna liga."
    )

# Gráfico

fig6 = px.bar(
    pred_df,
    x="Temporada",
    y="Título",
    title="Ligas simuladas"
)

st.plotly_chart(fig6, use_container_width=True)

# =====================================================
# RESUMEN
# =====================================================

st.divider()

st.subheader("Resumen Estadístico")

st.write(f"• Total de temporadas analizadas: **{stats['Temporadas']}**")
st.write(f"• Total de victorias: **{stats['Victorias']}**")
st.write(f"• Total de empates: **{stats['Empates']}**")
st.write(f"• Total de derrotas: **{stats['Derrotas']}**")
st.write(f"• Títulos obtenidos: **{stats['Títulos']}**")
st.write(f"• Promedio de puntos: **{stats['Promedio Puntos']:.2f}**")
st.write(f"• Promedio de posición: **{stats['Promedio Posición']:.2f}**")

st.caption("Proyecto de analítica deportiva desarrollado con Streamlit, Pandas, Plotly y Scikit-Learn.")
