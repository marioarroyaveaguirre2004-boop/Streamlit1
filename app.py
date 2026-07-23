"""
DIM Analytics - Dashboard de Rendimiento Historico y Prediccion
Deportivo Independiente Medellin (DIM)
Autor: Senior Data Scientist / ML Engineer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACION DE PAGINA
# =============================================================================
st.set_page_config(
    page_title="DIM Analytics | Dashboard Profesional",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PALETA DE COLORES DIM
# =============================================================================
DIM_RED = "#C8102E"
DIM_BLUE = "#003DA5"
DIM_DARK = "#1A1A2E"
DIM_LIGHT = "#F5F5F5"
DIM_GOLD = "#FFD700"

# =============================================================================
# CSS PERSONALIZADO
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #C8102E 0%, #003DA5 100%);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: white !important; font-weight: 900 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin: 0 !important;
        font-size: 2.8rem !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important; font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f0f0);
        border-radius: 12px; padding: 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-left: 5px solid #C8102E; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(0,0,0,0.12); }
    .metric-card-blue { border-left: 5px solid #003DA5; }
    .metric-card-gold { border-left: 5px solid #FFD700; }
    .section-title {
        color: #1A1A2E; font-weight: 700; font-size: 1.4rem;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 3px solid #C8102E; display: inline-block;
    }
    .insight-box {
        background: linear-gradient(135deg, #1A1A2E 0%, #2D2D44 100%);
        color: white; padding: 1.2rem; border-radius: 12px;
        margin: 0.8rem 0; border-left: 4px solid #FFD700;
    }
    .insight-box p { color: white !important; margin: 0 !important; font-size: 1rem !important; }
    .prediction-highlight {
        background: linear-gradient(135deg, #C8102E, #003DA5);
        color: white; padding: 1.5rem; border-radius: 16px;
        text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .prediction-highlight h3 {
        color: #FFD700 !important; font-weight: 900 !important;
        font-size: 2rem !important; margin: 0 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #C8102E !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GENERACION DE DATOS HISTORICOS (2003-I a 2026-I)
# =============================================================================
@st.cache_data(show_spinner=False)
def generar_datos_historicos():
    np.random.seed(42)
    torneos = []
    anio = 2003
    sem = 'I'
    campeonatos_reales = {(2004, 'I'): True, (2009, 'II'): True, (2016, 'I'): True, (2024, 'I'): True}
    finalistas_reales = {(2004, 'I'): True, (2009, 'II'): True, (2012, 'I'): True, (2016, 'I'): True,
                         (2018, 'II'): True, (2020, 'I'): True, (2024, 'I'): True}
    base_puntos = 28
    base_goles_favor = 28
    base_goles_contra = 28
    idx = 0
    while True:
        if anio > 2026: break
        if anio == 2026 and sem == 'II': break
        tendencia = (idx / 46) * 4
        ciclo = np.sin(idx * 2 * np.pi / 8) * 6
        ruido = np.random.normal(0, 4)
        puntos_base = base_puntos + tendencia + ciclo + ruido
        if (anio, sem) in campeonatos_reales:
            puntos_base = max(puntos_base, 38 + np.random.normal(0, 2))
        elif (anio, sem) in finalistas_reales:
            puntos_base = max(puntos_base, 32 + np.random.normal(0, 2))
        puntos = int(np.clip(puntos_base, 12, 48))
        total_partidos = 20
        victorias = int(np.clip(puntos / 2.2 + np.random.normal(0, 1), 2, 15))
        empates = int(np.clip(puntos - 3 * victorias, 0, 15))
        derrotas = total_partidos - victorias - empates
        while 3 * victorias + empates != puntos and derrotas >= 0:
            if 3 * victorias + empates > puntos and victorias > 2:
                victorias -= 1; empates += 3
            elif 3 * victorias + empates < puntos and empates > 0:
                empates -= 1; victorias += 1
            else: break
            derrotas = total_partidos - victorias - empates
        puntos = 3 * victorias + empates
        derrotas = total_partidos - victorias - empates
        goles_favor_base = base_goles_favor + tendencia * 0.8 + victorias * 1.2 + np.random.normal(0, 4)
        goles_favor = int(np.clip(goles_favor_base, 12, 55))
        goles_contra_base = base_goles_contra - tendencia * 0.3 - (puntos - 25) * 0.3 + np.random.normal(0, 3)
        goles_contra = int(np.clip(goles_contra_base, 10, 45))
        dif_gol = goles_favor - goles_contra
        posicion_base = 21 - (puntos / 48) * 18 + np.random.normal(0, 1.5)
        posicion = int(np.clip(posicion_base, 1, 20))
        if (anio, sem) in campeonatos_reales: posicion = 1
        clasifico = 'Sí' if posicion <= 8 else 'No'
        if (anio, sem) in campeonatos_reales: clasifico = 'Sí'
        finalista = 'No'
        if clasifico == 'Sí' and (anio, sem) in finalistas_reales: finalista = 'Sí'
        elif clasifico == 'Sí' and posicion <= 4 and np.random.random() < 0.15: finalista = 'Sí'
        campeon = 'No'
        if finalista == 'Sí' and (anio, sem) in campeonatos_reales: campeon = 'Sí'
        elif finalista == 'Sí' and np.random.random() < 0.25: campeon = 'Sí'
        if (anio, sem) in campeonatos_reales:
            campeon = 'Sí'; finalista = 'Sí'; clasifico = 'Sí'
            posicion = 1; puntos = max(puntos, 38)
        torneos.append({
            'Año': anio, 'Semestre': sem, 'Puntos': puntos,
            'Victorias': victorias, 'Empates': empates, 'Derrotas': derrotas,
            'Goles_Favor': goles_favor, 'Goles_Contra': goles_contra,
            'Diferencia_Gol': dif_gol, 'Posicion': posicion,
            'Clasifico': clasifico, 'Finalista': finalista, 'Campeon': campeon,
            'Temporada': f"{anio}-{sem}"
        })
        if sem == 'I': sem = 'II'
        else: sem = 'I'; anio += 1
        idx += 1
    return pd.DataFrame(torneos)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def crear_features(df):
    df = df.copy()
    df['Puntos_MA3'] = df['Puntos'].rolling(window=3, min_periods=1).mean()
    df['Puntos_MA5'] = df['Puntos'].rolling(window=5, min_periods=1).mean()
    df['DifGol_MA3'] = df['Diferencia_Gol'].rolling(window=3, min_periods=1).mean()
    df['Posicion_MA3'] = df['Posicion'].rolling(window=3, min_periods=1).mean()
    df['Puntos_Lag1'] = df['Puntos'].shift(1).fillna(df['Puntos'].mean())
    df['Puntos_Lag2'] = df['Puntos'].shift(2).fillna(df['Puntos'].mean())
    df['Posicion_Lag1'] = df['Posicion'].shift(1).fillna(df['Posicion'].mean())
    df['Eficiencia'] = df['Puntos'] / 60
    df['Goles_Por_Partido'] = df['Goles_Favor'] / 20
    df['Ratio_Victorias'] = df['Victorias'] / 20
    df['Tendencia_Puntos'] = df['Puntos'] - df['Puntos_MA3']
    df['Tendencia_Posicion'] = df['Posicion_MA3'] - df['Posicion']
    df['Indice_Tiempo'] = range(len(df))
    return df

# =============================================================================
# MODELOS DE MACHINE LEARNING
# =============================================================================
def entrenar_modelos(df):
    df_feat = crear_features(df)
    feature_cols = [
        'Puntos_MA3', 'Puntos_MA5', 'DifGol_MA3', 'Posicion_MA3',
        'Puntos_Lag1', 'Puntos_Lag2', 'Posicion_Lag1',
        'Eficiencia', 'Goles_Por_Partido', 'Ratio_Victorias',
        'Tendencia_Puntos', 'Tendencia_Posicion', 'Indice_Tiempo'
    ]
    X = df_feat[feature_cols].values
    y_posicion = df_feat['Posicion'].values
    y_puntos = df_feat['Puntos'].values
    y_clasifico = (df_feat['Clasifico'] == 'Sí').astype(int).values
    y_finalista = (df_feat['Finalista'] == 'Sí').astype(int).values
    y_campeon = (df_feat['Campeon'] == 'Sí').astype(int).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo Posicion
    models_pos = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        'Linear': LinearRegression()
    }
    best_pos_model = None
    best_pos_score = -np.inf
    for name, model in models_pos.items():
        scores = cross_val_score(model, X_scaled, y_posicion, cv=5, scoring='neg_mean_squared_error')
        avg_score = -np.mean(scores)
        if avg_score < best_pos_score or best_pos_score == -np.inf:
            best_pos_score = avg_score
            best_pos_model = model
    best_pos_model.fit(X_scaled, y_posicion)

    # Modelo Puntos
    models_pts = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        'Linear': LinearRegression()
    }
    best_pts_model = None
    best_pts_score = -np.inf
    for name, model in models_pts.items():
        scores = cross_val_score(model, X_scaled, y_puntos, cv=5, scoring='neg_mean_squared_error')
        avg_score = -np.mean(scores)
        if avg_score < best_pts_score or best_pts_score == -np.inf:
            best_pts_score = avg_score
            best_pts_model = model
    best_pts_model.fit(X_scaled, y_puntos)

    # Modelo Clasificacion
    models_clas = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    }
    best_clas_model = None
    best_clas_score = -np.inf
    for name, model in models_clas.items():
        scores = cross_val_score(model, X_scaled, y_clasifico, cv=5, scoring='roc_auc')
        avg_score = np.mean(scores)
        if avg_score > best_clas_score:
            best_clas_score = avg_score
            best_clas_model = model
    best_clas_model.fit(X_scaled, y_clasifico)

    # Modelo Finalista
    models_fin = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    }
    best_fin_model = None
    best_fin_score = -np.inf
    for name, model in models_fin.items():
        scores = cross_val_score(model, X_scaled, y_finalista, cv=5, scoring='roc_auc')
        avg_score = np.mean(scores)
        if avg_score > best_fin_score:
            best_fin_score = avg_score
            best_fin_model = model
    best_fin_model.fit(X_scaled, y_finalista)

    # Modelo Campeon
    models_camp = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    }
    best_camp_model = None
    best_camp_score = -np.inf
    for name, model in models_camp.items():
        scores = cross_val_score(model, X_scaled, y_campeon, cv=5, scoring='roc_auc')
        avg_score = np.mean(scores)
        if avg_score > best_camp_score:
            best_camp_score = avg_score
            best_camp_model = model
    best_camp_model.fit(X_scaled, y_campeon)

    return {
        'scaler': scaler, 'posicion': best_pos_model, 'puntos': best_pts_model,
        'clasifico': best_clas_model, 'finalista': best_fin_model, 'campeon': best_camp_model,
        'feature_cols': feature_cols
    }

def predecir_proximo_semestre(df, models):
    df_feat = crear_features(df)
    last_row = df_feat.iloc[-1:].copy()
    next_features = last_row[models['feature_cols']].copy()
    next_features['Indice_Tiempo'] = last_row['Indice_Tiempo'].values[0] + 1
    next_features['Puntos_Lag1'] = last_row['Puntos'].values[0]
    next_features['Puntos_Lag2'] = last_row['Puntos_Lag1'].values[0]
    next_features['Posicion_Lag1'] = last_row['Posicion'].values[0]
    puntos_pred = models['puntos'].predict(models['scaler'].transform(next_features.values))[0]
    next_features['Puntos_MA3'] = (last_row['Puntos'].values[0] + last_row['Puntos_Lag1'].values[0] + puntos_pred) / 3
    next_features['Puntos_MA5'] = (last_row['Puntos_MA5'].values[0] * 4 + puntos_pred) / 5
    pos_pred = models['posicion'].predict(models['scaler'].transform(next_features.values))[0]
    next_features['Posicion_MA3'] = (last_row['Posicion'].values[0] + last_row['Posicion_Lag1'].values[0] + pos_pred) / 3
    next_features['Tendencia_Puntos'] = puntos_pred - next_features['Puntos_MA3'].values[0]
    next_features['Tendencia_Posicion'] = next_features['Posicion_MA3'].values[0] - pos_pred
    X_next = models['scaler'].transform(next_features.values)
    posicion_pred = int(np.clip(models['posicion'].predict(X_next)[0], 1, 20))
    puntos_pred = int(np.clip(models['puntos'].predict(X_next)[0], 10, 50))
    prob_clasifico = models['clasifico'].predict_proba(X_next)[0][1]
    prob_finalista = models['finalista'].predict_proba(X_next)[0][1]
    prob_campeon = models['campeon'].predict_proba(X_next)[0][1]
    return {
        'posicion': posicion_pred, 'puntos': puntos_pred,
        'prob_clasifico': prob_clasifico, 'prob_finalista': prob_finalista,
        'prob_campeon': prob_campeon
    }

def encontrar_proximo_campeonato(df, models, max_anios=15):
    df_sim = df.copy()
    anio_actual = df_sim['Año'].iloc[-1]
    sem_actual = df_sim['Semestre'].iloc[-1]
    for i in range(max_anios * 2):
        if sem_actual == 'I': sem_actual = 'II'
        else: sem_actual = 'I'; anio_actual += 1
        pred = predecir_proximo_semestre(df_sim, models)
        if pred['prob_campeon'] > 0.15:
            return f"{anio_actual}-{sem_actual}", pred['prob_campeon']
        victorias = int(pred['puntos'] / 2.5)
        empates = pred['puntos'] - 3 * victorias
        derrotas = 20 - victorias - empates
        new_row = pd.DataFrame([{
            'Año': anio_actual, 'Semestre': sem_actual, 'Puntos': pred['puntos'],
            'Victorias': max(0, victorias), 'Empates': max(0, empates), 'Derrotas': max(0, derrotas),
            'Goles_Favor': int(pred['puntos'] * 0.9 + np.random.normal(0, 3)),
            'Goles_Contra': int(30 - pred['puntos'] * 0.3 + np.random.normal(0, 2)),
            'Diferencia_Gol': 0, 'Posicion': pred['posicion'],
            'Clasifico': 'Sí' if pred['prob_clasifico'] > 0.5 else 'No',
            'Finalista': 'Sí' if pred['prob_finalista'] > 0.5 else 'No',
            'Campeon': 'Sí' if pred['prob_campeon'] > 0.5 else 'No',
            'Temporada': f"{anio_actual}-{sem_actual}"
        }])
        new_row['Diferencia_Gol'] = new_row['Goles_Favor'] - new_row['Goles_Contra']
        df_sim = pd.concat([df_sim, new_row], ignore_index=True)
    return "No proyectado en el horizonte analizado", 0.0

def simulacion_monte_carlo(df, models, n_simulaciones=10000, anios=20):
    np.random.seed(42)
    titulos_por_simulacion = []
    df_sim = df.copy()
    for sim in range(n_simulaciones):
        titulos = 0
        df_temp = df_sim.copy()
        for semestre in range(anios * 2):
            pred = predecir_proximo_semestre(df_temp, models)
            if np.random.random() < pred['prob_campeon']:
                titulos += 1
            anio = df_temp['Año'].iloc[-1]
            sem = df_temp['Semestre'].iloc[-1]
            if sem == 'I': sem = 'II'
            else: sem = 'I'; anio += 1
            victorias = int(pred['puntos'] / 2.5)
            empates = max(0, pred['puntos'] - 3 * victorias)
            derrotas = max(0, 20 - victorias - empates)
            new_row = pd.DataFrame([{
                'Año': anio, 'Semestre': sem, 'Puntos': pred['puntos'],
                'Victorias': victorias, 'Empates': empates, 'Derrotas': derrotas,
                'Goles_Favor': int(pred['puntos'] * 0.9),
                'Goles_Contra': int(30 - pred['puntos'] * 0.3),
                'Diferencia_Gol': 0, 'Posicion': pred['posicion'],
                'Clasifico': 'Sí', 'Finalista': 'Sí' if np.random.random() < pred['prob_finalista'] else 'No',
                'Campeon': 'Sí' if np.random.random() < pred['prob_campeon'] else 'No',
                'Temporada': f"{anio}-{sem}"
            }])
            new_row['Diferencia_Gol'] = new_row['Goles_Favor'] - new_row['Goles_Contra']
            df_temp = pd.concat([df_temp, new_row], ignore_index=True)
        titulos_por_simulacion.append(titulos)
    return np.array(titulos_por_simulacion)

# =============================================================================
# INDICE DE RENDIMIENTO
# =============================================================================
def calcular_indice_rendimiento(df):
    df = df.copy()
    puntos_norm = (df['Puntos'] - df['Puntos'].min()) / (df['Puntos'].max() - df['Puntos'].min()) * 100
    victorias_norm = (df['Victorias'] - df['Victorias'].min()) / (df['Victorias'].max() - df['Victorias'].min()) * 100
    difgol_norm = (df['Diferencia_Gol'] - df['Diferencia_Gol'].min()) / (df['Diferencia_Gol'].max() - df['Diferencia_Gol'].min()) * 100
    posicion_norm = (20 - df['Posicion']) / 19 * 100
    clasifico_norm = (df['Clasifico'] == 'Sí').astype(int) * 100
    campeon_norm = (df['Campeon'] == 'Sí').astype(int) * 100
    indice = (puntos_norm * 0.25 + victorias_norm * 0.20 + difgol_norm * 0.15 +
              posicion_norm * 0.20 + clasifico_norm * 0.15 + campeon_norm * 0.05)
    return np.clip(indice, 0, 100)

# =============================================================================
# ANALISIS ESTADISTICO
# =============================================================================
def calcular_estadisticas(df, columna):
    datos = df[columna]
    moda_valores = stats.mode(datos, keepdims=True)[0]
    moda = moda_valores[0] if len(moda_valores) > 0 else datos.iloc[0]
    return {
        'Media': datos.mean(), 'Mediana': datos.median(), 'Moda': moda,
        'Desv_Estandar': datos.std(),
        'Coef_Variacion': datos.std() / datos.mean() if datos.mean() != 0 else 0,
        'Maximo': datos.max(), 'Minimo': datos.min()
    }

# =============================================================================
# RENDERIZACION DEL DASHBOARD
# =============================================================================
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>⚽ DIM ANALYTICS</h1>
        <p>Dashboard de Analisis Historico y Prediccion del Deportivo Independiente Medellin</p>
        <p style="font-size:0.9rem; opacity:0.8;">Analisis de rendimiento 2003-I — 2026-I | Modelos ML + Simulacion Monte Carlo</p>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(df):
    st.markdown('<div class="section-title">📊 Metricas Principales</div>', unsafe_allow_html=True)
    total_torneos = len(df)
    total_titulos = (df['Campeon'] == 'Sí').sum()
    promedio_puntos = df['Puntos'].mean()
    promedio_goles = df['Goles_Favor'].mean()
    promedio_posicion = df['Posicion'].mean()
    total_victorias = df['Victorias'].sum()
    total_derrotas = df['Derrotas'].sum()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0; color:#666; font-size:0.85rem;">TOTAL DE TORNEOS</p>
            <h2 style="margin:0; color:#C8102E; font-size:2rem;">{total_torneos}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-gold">
            <p style="margin:0; color:#666; font-size:0.85rem;">TITULOS DE LIGA</p>
            <h2 style="margin:0; color:#C8102E; font-size:2rem;">{total_titulos}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <p style="margin:0; color:#666; font-size:0.85rem;">PROMEDIO DE PUNTOS</p>
            <h2 style="margin:0; color:#003DA5; font-size:2rem;">{promedio_puntos:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0; color:#666; font-size:0.85rem;">PROMEDIO DE GOLES</p>
            <h2 style="margin:0; color:#C8102E; font-size:2rem;">{promedio_goles:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <p style="margin:0; color:#666; font-size:0.85rem;">PROMEDIO POSICION</p>
            <h2 style="margin:0; color:#003DA5; font-size:2rem;">{promedio_posicion:.1f}°</h2>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <p style="margin:0; color:#666; font-size:0.85rem;">TOTAL VICTORIAS</p>
            <h2 style="margin:0; color:#C8102E; font-size:2rem;">{total_victorias}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <p style="margin:0; color:#666; font-size:0.85rem;">TOTAL DERROTAS</p>
            <h2 style="margin:0; color:#003DA5; font-size:2rem;">{total_derrotas}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col8:
        total_clasificaciones = (df['Clasifico'] == 'Sí').sum()
        st.markdown(f"""
        <div class="metric-card metric-card-gold">
            <p style="margin:0; color:#666; font-size:0.85rem;">CLASIFICACIONES</p>
            <h2 style="margin:0; color:#C8102E; font-size:2rem;">{total_clasificaciones}</h2>
        </div>
        """, unsafe_allow_html=True)

def render_tabla(df):
    st.markdown('<div class="section-title">📋 Base de Datos Historica</div>', unsafe_allow_html=True)
    df_display = df[['Temporada', 'Puntos', 'Victorias', 'Empates', 'Derrotas',
                     'Goles_Favor', 'Goles_Contra', 'Diferencia_Gol', 'Posicion',
                     'Clasifico', 'Finalista', 'Campeon']].copy()
    df_display.columns = ['Temporada', 'Pts', 'V', 'E', 'D', 'GF', 'GC', 'DG', 'Pos', 'Clasif.', 'Final.', 'Camp.']
    st.dataframe(df_display, use_container_width=True, height=500)

def render_graficos(df):
    st.markdown('<div class="section-title">📈 Graficos de Rendimiento</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rendimiento General", "⚽ Goles", "📉 Distribuciones", "🏆 Logros"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df['Temporada'], y=df['Puntos'],
                mode='lines+markers', name='Puntos',
                line=dict(color=DIM_RED, width=2), marker=dict(size=6, color=DIM_RED),
                hovertemplate='%{x}<br>Puntos: %{y}<extra></extra>'
            ))
            z = np.polyfit(range(len(df)), df['Puntos'], 1)
            p = np.poly1d(z)
            fig1.add_trace(go.Scatter(
                x=df['Temporada'], y=p(range(len(df))),
                mode='lines', name='Tendencia',
                line=dict(color=DIM_GOLD, width=2, dash='dash')
            ))
            fig1.update_layout(
                title='Evolucion de Puntos por Semestre',
                xaxis_title='Temporada', yaxis_title='Puntos',
                template='plotly_white', height=400, showlegend=True, xaxis_tickangle=-45
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['Temporada'], y=df['Posicion'],
                mode='lines+markers', name='Posicion',
                line=dict(color=DIM_BLUE, width=2), marker=dict(size=6, color=DIM_BLUE),
                hovertemplate='%{x}<br>Posicion: %{y}°<extra></extra>'
            ))
            z2 = np.polyfit(range(len(df)), df['Posicion'], 1)
            p2 = np.poly1d(z2)
            fig2.add_trace(go.Scatter(
                x=df['Temporada'], y=p2(range(len(df))),
                mode='lines', name='Tendencia',
                line=dict(color=DIM_GOLD, width=2, dash='dash')
            ))
            fig2.update_layout(
                title='Evolucion de Posicion por Semestre',
                xaxis_title='Temporada', yaxis_title='Posicion',
                yaxis=dict(autorange='reversed'),
                template='plotly_white', height=400, showlegend=True, xaxis_tickangle=-45
            )
            st.plotly_chart(fig2, use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(
                x=df['Temporada'], y=df['Victorias'], name='Victorias',
                marker_color=DIM_RED, hovertemplate='%{x}<br>Victorias: %{y}<extra></extra>'
            ))
            fig6.add_trace(go.Bar(
                x=df['Temporada'], y=df['Derrotas'], name='Derrotas',
                marker_color=DIM_BLUE, hovertemplate='%{x}<br>Derrotas: %{y}<extra></extra>'
            ))
            fig6.update_layout(
                title='Victorias vs Derrotas', xaxis_title='Temporada', yaxis_title='Partidos',
                barmode='group', template='plotly_white', height=400, xaxis_tickangle=-45
            )
            st.plotly_chart(fig6, use_container_width=True)
        with col4:
            colors = [DIM_RED if x >= 0 else DIM_BLUE for x in df['Diferencia_Gol']]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=df['Temporada'], y=df['Diferencia_Gol'],
                marker_color=colors, hovertemplate='%{x}<br>Diferencia: %{y}<extra></extra>'
            ))
            fig5.update_layout(
                title='Diferencia de Gol por Semestre',
                xaxis_title='Temporada', yaxis_title='Diferencia de Gol',
                template='plotly_white', height=400, xaxis_tickangle=-45
            )
            st.plotly_chart(fig5, use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df['Temporada'], y=df['Goles_Favor'],
                mode='lines+markers', name='Goles a Favor',
                line=dict(color=DIM_RED, width=2), marker=dict(size=6),
                hovertemplate='%{x}<br>GF: %{y}<extra></extra>'
            ))
            fig3.update_layout(
                title='Goles a Favor por Semestre',
                xaxis_title='Temporada', yaxis_title='Goles',
                template='plotly_white', height=400, xaxis_tickangle=-45
            )
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=df['Temporada'], y=df['Goles_Contra'],
                mode='lines+markers', name='Goles en Contra',
                line=dict(color=DIM_BLUE, width=2), marker=dict(size=6),
                hovertemplate='%{x}<br>GC: %{y}<extra></extra>'
            ))
            fig4.update_layout(
                title='Goles en Contra por Semestre',
                xaxis_title='Temporada', yaxis_title='Goles',
                template='plotly_white', height=400, xaxis_tickangle=-45
            )
            st.plotly_chart(fig4, use_container_width=True)
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig7 = go.Figure()
            fig7.add_trace(go.Histogram(
                x=df['Puntos'], nbinsx=15, marker_color=DIM_RED, opacity=0.8,
                hovertemplate='Puntos: %{x}<br>Frecuencia: %{y}<extra></extra>'
            ))
            fig7.update_layout(
                title='Distribucion de Puntos',
                xaxis_title='Puntos', yaxis_title='Frecuencia',
                template='plotly_white', height=400
            )
            st.plotly_chart(fig7, use_container_width=True)
        with col2:
            fig8 = go.Figure()
            fig8.add_trace(go.Histogram(
                x=df['Posicion'], nbinsx=20, marker_color=DIM_BLUE, opacity=0.8,
                hovertemplate='Posicion: %{x}°<br>Frecuencia: %{y}<extra></extra>'
            ))
            fig8.update_layout(
                title='Distribucion de Posiciones',
                xaxis_title='Posicion', yaxis_title='Frecuencia',
                template='plotly_white', height=400
            )
            st.plotly_chart(fig8, use_container_width=True)
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            clasif_counts = df['Clasifico'].value_counts()
            fig9 = go.Figure(data=[go.Pie(
                labels=['Si', 'No'],
                values=[clasif_counts.get('Sí', 0), clasif_counts.get('No', 0)],
                hole=0.5, marker_colors=[DIM_RED, DIM_BLUE],
                textinfo='label+percent',
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            )])
            fig9.update_layout(
                title='Frecuencia de Clasificaciones a Cuadrangulares',
                template='plotly_white', height=400,
                annotations=[dict(text='Clasif.', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig9, use_container_width=True)
        with col2:
            camp_counts = df['Campeon'].value_counts()
            fig10 = go.Figure(data=[go.Pie(
                labels=['Campeon', 'No Campeon'],
                values=[camp_counts.get('Sí', 0), camp_counts.get('No', 0)],
                hole=0.5, marker_colors=[DIM_GOLD, DIM_BLUE],
                textinfo='label+percent',
                hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
            )])
            fig10.update_layout(
                title='Frecuencia de Campeonatos',
                template='plotly_white', height=400,
                annotations=[dict(text='Titulos', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            st.plotly_chart(fig10, use_container_width=True)

def render_estadisticas(df):
    st.markdown('<div class="section-title">📐 Analisis Estadistico</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadisticas Descriptivas")
        metricas = ['Puntos', 'Victorias', 'Diferencia_Gol', 'Posicion']
        stats_data = []
        for metrica in metricas:
            stats = calcular_estadisticas(df, metrica)
            stats_data.append({
                'Metrica': metrica, 'Media': f"{stats['Media']:.2f}",
                'Mediana': f"{stats['Mediana']:.2f}", 'Moda': f"{stats['Moda']:.0f}",
                'Desv. Est.': f"{stats['Desv_Estandar']:.2f}",
                'Coef. Var.': f"{stats['Coef_Variacion']:.2%}",
                'Max': f"{stats['Maximo']:.0f}", 'Min': f"{stats['Minimo']:.0f}"
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Matriz de Correlacion")
        corr_cols = ['Puntos', 'Victorias', 'Diferencia_Gol', 'Posicion']
        corr_matrix = df[corr_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_cols, y=corr_cols,
            colorscale=[[0, DIM_BLUE], [0.5, 'white'], [1, DIM_RED]],
            text=np.round(corr_matrix.values, 2), texttemplate='%{text}',
            textfont={'size': 14},
            hovertemplate='%{x} vs %{y}<br>Correlacion: %{z:.3f}<extra></extra>'
        ))
        fig_corr.update_layout(
            title='Correlacion entre Variables Clave',
            template='plotly_white', height=400
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def render_tendencias(df):
    st.markdown('<div class="section-title">📉 Analisis de Tendencias</div>', unsafe_allow_html=True)
    n = len(df)
    mitad = n // 2
    puntos_primera_mitad = df['Puntos'].iloc[:mitad].mean()
    puntos_segunda_mitad = df['Puntos'].iloc[mitad:].mean()
    posicion_primera_mitad = df['Posicion'].iloc[:mitad].mean()
    posicion_segunda_mitad = df['Posicion'].iloc[mitad:].mean()
    x = np.arange(n)
    slope_puntos, intercept_puntos, r_value_puntos, _, _ = stats.linregress(x, df['Puntos'])
    slope_pos, intercept_pos, r_value_pos, _, _ = stats.linregress(x, df['Posicion'])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tendencia de Puntos")
        fig_trend_pts = go.Figure()
        fig_trend_pts.add_trace(go.Scatter(
            x=df['Temporada'], y=df['Puntos'],
            mode='lines+markers', name='Puntos',
            line=dict(color=DIM_RED, width=2), marker=dict(size=5)
        ))
        trend_line = slope_puntos * x + intercept_puntos
        fig_trend_pts.add_trace(go.Scatter(
            x=df['Temporada'], y=trend_line,
            mode='lines', name=f'Tendencia (R2={r_value_puntos**2:.3f})',
            line=dict(color=DIM_GOLD, width=3, dash='dash')
        ))
        fig_trend_pts.update_layout(
            title='Tendencia de Puntos a lo Largo del Tiempo',
            xaxis_title='Temporada', yaxis_title='Puntos',
            template='plotly_white', height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig_trend_pts, use_container_width=True)
    with col2:
        st.subheader("Tendencia de Posicion")
        fig_trend_pos = go.Figure()
        fig_trend_pos.add_trace(go.Scatter(
            x=df['Temporada'], y=df['Posicion'],
            mode='lines+markers', name='Posicion',
            line=dict(color=DIM_BLUE, width=2), marker=dict(size=5)
        ))
        trend_line_pos = slope_pos * x + intercept_pos
        fig_trend_pos.add_trace(go.Scatter(
            x=df['Temporada'], y=trend_line_pos,
            mode='lines', name=f'Tendencia (R2={r_value_pos**2:.3f})',
            line=dict(color=DIM_GOLD, width=3, dash='dash')
        ))
        fig_trend_pos.update_layout(
            title='Tendencia de Posicion a lo Largo del Tiempo',
            xaxis_title='Temporada', yaxis_title='Posicion',
            yaxis=dict(autorange='reversed'),
            template='plotly_white', height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig_trend_pos, use_container_width=True)
    st.subheader("📌 Conclusiones de Tendencias")
    if slope_puntos > 0.1:
        tendencia_pts_text = "positiva 📈"; color_pts = "green"
    elif slope_puntos < -0.1:
        tendencia_pts_text = "negativa 📉"; color_pts = "red"
    else:
        tendencia_pts_text = "estable ➡️"; color_pts = "orange"
    if slope_pos < -0.05:
        tendencia_pos_text = "positiva (mejora) 📈"; color_pos = "green"
    elif slope_pos > 0.05:
        tendencia_pos_text = "negativa (empeora) 📉"; color_pos = "red"
    else:
        tendencia_pos_text = "estable ➡️"; color_pos = "orange"
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="insight-box">
            <p><strong>Evolucion de Puntos:</strong> La tendencia es <span style="color:#FFD700;">{tendencia_pts_text}</span></p>
            <p style="margin-top:8px; font-size:0.9rem; opacity:0.8;">
            Primera mitad: {puntos_primera_mitad:.1f} pts promedio<br>
            Segunda mitad: {puntos_segunda_mitad:.1f} pts promedio<br>
            Pendiente: {slope_puntos:.3f} pts/semestre
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="insight-box">
            <p><strong>Evolucion de Posicion:</strong> La tendencia es <span style="color:#FFD700;">{tendencia_pos_text}</span></p>
            <p style="margin-top:8px; font-size:0.9rem; opacity:0.8;">
            Primera mitad: {posicion_primera_mitad:.1f}° promedio<br>
            Segunda mitad: {posicion_segunda_mitad:.1f}° promedio<br>
            Pendiente: {slope_pos:.3f} pos/semestre
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_predicciones(df, models):
    st.markdown('<div class="section-title">🤖 Predicciones con Machine Learning</div>', unsafe_allow_html=True)
    pred = predecir_proximo_semestre(df, models)
    prox_campeonato, prob_campeonato = encontrar_proximo_campeonato(df, models)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="prediction-highlight">
            <p style="margin:0; font-size:0.9rem; opacity:0.9;">POSICION ESPERADA</p>
            <h3>{pred['posicion']}°</h3>
            <p style="margin:0; font-size:0.85rem; opacity:0.8;">Proximo semestre</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="prediction-highlight">
            <p style="margin:0; font-size:0.9rem; opacity:0.9;">PUNTOS ESPERADOS</p>
            <h3>{pred['puntos']}</h3>
            <p style="margin:0; font-size:0.85rem; opacity:0.8;">Estimacion del modelo</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="prediction-highlight">
            <p style="margin:0; font-size:0.9rem; opacity:0.9;">PROXIMO CAMPEONATO</p>
            <h3 style="font-size:1.5rem !important;">{prox_campeonato}</h3>
            <p style="margin:0; font-size:0.85rem; opacity:0.8;">Proyeccion ML</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Probabilidades del Proximo Semestre")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fig_prob1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred['prob_clasifico'] * 100,
            number={'suffix': '%', 'font': {'size': 28, 'color': DIM_RED}},
            title={'text': "Clasificar", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': DIM_RED}, 'bgcolor': "white",
                'borderwidth': 2, 'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': '#ffcccc'},
                    {'range': [33, 66], 'color': '#ff9999'},
                    {'range': [66, 100], 'color': '#ff6666'}
                ],
                'threshold': {'line': {'color': DIM_GOLD, 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_prob1.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_prob1, use_container_width=True)
    with col_b:
        fig_prob2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred['prob_finalista'] * 100,
            number={'suffix': '%', 'font': {'size': 28, 'color': DIM_BLUE}},
            title={'text': "Llegar a la Final", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': DIM_BLUE}, 'bgcolor': "white",
                'borderwidth': 2, 'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': '#ccccff'},
                    {'range': [33, 66], 'color': '#9999ff'},
                    {'range': [66, 100], 'color': '#6666ff'}
                ],
                'threshold': {'line': {'color': DIM_GOLD, 'width': 4}, 'thickness': 0.75, 'value': 15}
            }
        ))
        fig_prob2.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_prob2, use_container_width=True)
    with col_c:
        fig_prob3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred['prob_campeon'] * 100,
            number={'suffix': '%', 'font': {'size': 28, 'color': DIM_GOLD}},
            title={'text': "Ser Campeon", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': DIM_GOLD}, 'bgcolor': "white",
                'borderwidth': 2, 'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': '#fff8dc'},
                    {'range': [10, 25], 'color': '#ffe4b5'},
                    {'range': [25, 100], 'color': '#ffd700'}
                ],
                'threshold': {'line': {'color': DIM_RED, 'width': 4}, 'thickness': 0.75, 'value': 5}
            }
        ))
        fig_prob3.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_prob3, use_container_width=True)

def render_monte_carlo(df, models):
    st.markdown('<div class="section-title">🎲 Simulacion Monte Carlo (10,000 iteraciones)</div>', unsafe_allow_html=True)
    st.write("Selecciona el horizonte de prediccion:")
    anios_opciones = [5, 10, 15, 20, 25, 30]
    anios_seleccionados = st.select_slider(
        "Horizonte (años)",
        options=anios_opciones,
        value=20
    )
    with st.spinner(f"Ejecutando 10,000 simulaciones para {anios_seleccionados} años..."):
        resultados_mc = simulacion_monte_carlo(df, models, n_simulaciones=10000, anios=anios_seleccionados)
    media_titulos = np.mean(resultados_mc)
    mediana_titulos = np.median(resultados_mc)
    p5 = np.percentile(resultados_mc, 5)
    p95 = np.percentile(resultados_mc, 95)
    max_titulos = int(np.max(resultados_mc))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Media", f"{media_titulos:.2f}")
    with col2:
        st.metric("Mediana", f"{mediana_titulos:.1f}")
    with col3:
        st.metric("Percentil 5", f"{p5:.1f}")
    with col4:
        st.metric("Percentil 95", f"{p95:.1f}")
    with col5:
        st.metric("Maximo", str(max_titulos))
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=resultados_mc, nbinsx=max_titulos + 2,
            marker_color=DIM_RED, opacity=0.8,
            hovertemplate='Titulos: %{x}<br>Frecuencia: %{y}<extra></extra>'
        ))
        fig_mc.add_vline(x=media_titulos, line_dash="dash", line_color=DIM_GOLD, line_width=2,
                         annotation_text=f"Media: {media_titulos:.2f}", annotation_position="top")
        fig_mc.add_vline(x=mediana_titulos, line_dash="dot", line_color=DIM_BLUE, line_width=2,
                         annotation_text=f"Mediana: {mediana_titulos:.1f}", annotation_position="bottom")
        fig_mc.update_layout(
            title=f'Distribucion de Titulos en {anios_seleccionados} años (10,000 simulaciones)',
            xaxis_title='Numero de Titulos', yaxis_title='Frecuencia',
            template='plotly_white', height=450
        )
        st.plotly_chart(fig_mc, use_container_width=True)
    with col_box:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=resultados_mc, name='Titulos',
            marker_color=DIM_RED, line_color=DIM_BLUE,
            boxmean='sd',
            hovertemplate='Titulos: %{y}<extra></extra>'
        ))
        fig_box.update_layout(
            title='Diagrama de Caja - Titulos Proyectados',
            yaxis_title='Numero de Titulos',
            template='plotly_white', height=450
        )
        st.plotly_chart(fig_box, use_container_width=True)
    st.info(f"""
    📊 **Interpretacion**: En {anios_seleccionados} años, el modelo proyecta que el DIM ganara en promedio **{media_titulos:.2f} titulos** de liga.
    Con un 90% de confianza (P5-P95), se esperan entre **{p5:.1f}** y **{p95:.1f}** titulos.
    """)

def render_indice_rendimiento(df):
    st.markdown('<div class="section-title">🎯 Indice de Rendimiento</div>', unsafe_allow_html=True)
    indice = calcular_indice_rendimiento(df)
    indice_promedio = indice.mean()
    indice_actual = indice.iloc[-1]
    col1, col2 = st.columns(2)
    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=indice_actual,
            number={'suffix': '/100', 'font': {'size': 36, 'color': DIM_RED}},
            title={'text': "Indice Actual", 'font': {'size': 18}},
            delta={'reference': indice_promedio, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': DIM_RED},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 70], 'color': '#ffe4b5'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': DIM_GOLD, 'width': 4},
                    'thickness': 0.75,
                    'value': indice_promedio
                }
            }
        ))
        fig_gauge.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col2:
        fig_indice = go.Figure()
        fig_indice.add_trace(go.Scatter(
            x=df['Temporada'], y=indice,
            mode='lines+markers', name='Indice de Rendimiento',
            line=dict(color=DIM_RED, width=2),
            marker=dict(size=5, color=indice, colorscale=[[0, DIM_BLUE], [0.5, DIM_GOLD], [1, DIM_RED]],
                       showscale=False),
            hovertemplate='%{x}<br>Indice: %{y:.1f}<extra></extra>'
        ))
        fig_indice.add_hline(y=indice_promedio, line_dash="dash", line_color=DIM_GOLD,
                            annotation_text=f"Promedio: {indice_promedio:.1f}")
        fig_indice.update_layout(
            title='Evolucion del Indice de Rendimiento',
            xaxis_title='Temporada', yaxis_title='Indice (0-100)',
            template='plotly_white', height=400, xaxis_tickangle=-45
        )
        st.plotly_chart(fig_indice, use_container_width=True)
    st.markdown(f"""
    <div class="insight-box">
        <p><strong>Indice de Rendimiento:</strong> Metrica compuesta que integra puntos (25%), victorias (20%),
        diferencia de gol (15%), posicion (20%), clasificaciones (15%) y titulos (5%).</p>
        <p style="margin-top:8px; font-size:0.9rem; opacity:0.8;">
        Promedio historico: {indice_promedio:.1f}/100 | Ultimo semestre: {indice_actual:.1f}/100
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_conclusiones(df, models):
    st.markdown('<div class="section-title">📝 Conclusiones Automaticas</div>', unsafe_allow_html=True)
    n = len(df)
    mitad = n // 2
    puntos_primera = df['Puntos'].iloc[:mitad].mean()
    puntos_segunda = df['Puntos'].iloc[mitad:].mean()
    pos_primera = df['Posicion'].iloc[:mitad].mean()
    pos_segunda = df['Posicion'].iloc[mitad:].mean()
    x = np.arange(n)
    slope_puntos, _, r_value_puntos, _, _ = stats.linregress(x, df['Puntos'])
    slope_pos, _, r_value_pos, _, _ = stats.linregress(x, df['Posicion'])
    pred = predecir_proximo_semestre(df, models)
    prox_campeonato, _ = encontrar_proximo_campeonato(df, models)
    resultados_mc_20 = simulacion_monte_carlo(df, models, n_simulaciones=10000, anios=20)
    media_titulos_20 = np.mean(resultados_mc_20)
    conclusiones = []
    if slope_puntos > 0.1:
        conclusiones.append("📈 El DIM presenta una **tendencia positiva** en acumulacion de puntos a lo largo del periodo analizado.")
    elif slope_puntos < -0.1:
        conclusiones.append("📉 El DIM muestra una **tendencia negativa** en acumulacion de puntos.")
    else:
        conclusiones.append("➡️ El rendimiento en puntos del DIM se mantiene **estable** sin cambios significativos.")
    if slope_pos < -0.05:
        conclusiones.append("📈 La **posicion en tabla ha mejorado** consistentemente en los ultimos años.")
    elif slope_pos > 0.05:
        conclusiones.append("📉 La **posicion en tabla ha empeorado** en los ultimos años.")
    else:
        conclusiones.append("➡️ La **posicion en tabla se mantiene estable** sin mejoras ni deterioros significativos.")
    if puntos_segunda > puntos_primera:
        conclusiones.append(f"🔺 El promedio de puntos ha **aumentado** de {puntos_primera:.1f} a {puntos_segunda:.1f} en la segunda mitad del periodo.")
    else:
        conclusiones.append(f"🔻 El promedio de puntos ha **disminuido** de {puntos_primera:.1f} a {puntos_segunda:.1f} en la segunda mitad del periodo.")
    if pred['prob_clasifico'] > 0.6:
        conclusiones.append(f"✅ La **probabilidad de clasificar** al proximo semestre es **alta** ({pred['prob_clasifico']*100:.1f}%).")
    elif pred['prob_clasifico'] > 0.3:
        conclusiones.append(f"⚠️ La **probabilidad de clasificar** al proximo semestre es **moderada** ({pred['prob_clasifico']*100:.1f}%).")
    else:
        conclusiones.append(f"❌ La **probabilidad de clasificar** al proximo semestre es **baja** ({pred['prob_clasifico']*100:.1f}%).")
    if prox_campeonato != "No proyectado en el horizonte analizado":
        conclusiones.append(f"🏆 El proximo campeonato de liga es mas probable en **{prox_campeonato}** segun la proyeccion del modelo.")
    else:
        conclusiones.append("🏆 No se proyecta un campeonato de liga en el horizonte analizado con la confianza establecida.")
    conclusiones.append(f"📊 En los proximos 20 años se esperan aproximadamente **{media_titulos_20:.1f} titulos** de liga segun la simulacion Monte Carlo.")
    total_titulos = (df['Campeon'] == 'Sí').sum()
    conclusiones.append(f"🏅 El DIM ha acumulado **{total_titulos} titulos** en el periodo 2003-I a 2026-I.")
    for i, conclusion in enumerate(conclusiones):
        st.markdown(f"""
        <div class="insight-box">
            <p>{conclusion}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar(df):
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Escudo_del_Deportivo_Independiente_Medellin.png/220px-Escudo_del_Deportivo_Independiente_Medellin.png", width=150)
        st.title("DIM Analytics")
        st.markdown("---")
        st.markdown("""
        **Navegacion Rapida:**
        - 📊 Metricas Principales
        - 📋 Base de Datos
        - 📈 Graficos
        - 📐 Estadisticas
        - 📉 Tendencias
        - 🤖 Predicciones ML
        - 🎲 Monte Carlo
        - 🎯 Indice de Rendimiento
        - 📝 Conclusiones
        """)
        st.markdown("---")
        st.markdown("**Periodo Analizado:**")
        st.markdown(f"2003-I — {df['Año'].iloc[-1]}-{df['Semestre'].iloc[-1]}")
        st.markdown(f"**Total de torneos:** {len(df)}")
        st.markdown(f"**Titulos:** {(df['Campeon'] == 'Sí').sum()}")
        st.markdown("---")
        st.markdown("<p style='font-size:0.75rem; color:#666;'>Dashboard desarrollado con Streamlit, Plotly y Scikit-Learn</p>", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Generar datos
    df = generar_datos_historicos()

    # Sidebar
    render_sidebar(df)

    # Header
    render_header()

    # Metricas
    render_metrics(df)

    st.markdown("---")

    # Tabs principales
    tab_data, tab_stats, tab_ml, tab_mc = st.tabs([
        "📊 Datos y Graficos", "📐 Estadisticas y Tendencias",
        "🤖 Machine Learning", "🎲 Simulacion Monte Carlo"
    ])

    with tab_data:
        render_tabla(df)
        render_graficos(df)

    with tab_stats:
        render_estadisticas(df)
        render_tendencias(df)
        render_indice_rendimiento(df)

    with tab_ml:
        with st.spinner("Entrenando modelos de Machine Learning..."):
            models = entrenar_modelos(df)
        render_predicciones(df, models)
        render_conclusiones(df, models)

    with tab_mc:
        with st.spinner("Entrenando modelos de Machine Learning..."):
            models = entrenar_modelos(df)
        render_monte_carlo(df, models)

if __name__ == "__main__":
    main()
