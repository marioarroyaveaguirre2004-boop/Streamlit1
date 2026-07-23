import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(page_title="DIM - Analítica y Predicción", layout="wide", page_icon="🔴")

st.markdown('''
    <style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #001e60; }
    .metric-card {
        background-color: #e3001b; 
        color: white; 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }
    .metric-card h3 { color: white; margin-bottom: 0px; }
    .metric-card h2 { color: white; margin-top: 5px; }
    </style>
''', unsafe_allow_html=True)

@st.cache_data
def generate_data():
    np.random.seed(42)
    years = []
    semesters = []
    for y in range(2003, 2026):
        years.extend([y, y])
        semesters.extend(['I', 'II'])
    years.append(2026)
    semesters.append('I')
    
    n = len(years)
    base_strength = np.sin(np.linspace(0, 10, n)) * 5 + 50 
    
    data = []
    for i in range(n):
        y = years[i]
        s = semesters[i]
        matches = 20
        
        form = base_strength[i] + np.random.normal(0, 5)
        if y == 2004 and s == 'I': form += 25
        if y == 2009 and s == 'II': form += 25
        if y == 2016 and s == 'I': form += 25
        
        win_prob = np.clip(form / 120, 0.15, 0.65)
        draw_prob = np.clip((100 - form) / 150, 0.15, 0.35)
        
        victorias = int(round(matches * win_prob))
        empates = int(round(matches * draw_prob))
        derrotas = matches - victorias - empates
        
        if derrotas < 0:
            derrotas = 0
            victorias = matches - empates
        while victorias + empates + derrotas != matches:
            if victorias + empates + derrotas < matches: empates += 1
            else:
                if empates > 0: empates -= 1
                else: derrotas -= 1
        
        puntos = victorias * 3 + empates
        
        gf = victorias * 2 + empates * 1 + np.random.randint(0, 5)
        gc = derrotas * 2 + empates * 1 + np.random.randint(0, 5)
        dg = gf - gc
        
        pos = int(np.clip(21 - (puntos / 45 * 20) + np.random.normal(0, 2), 1, 20))
        
        clasifico = 'Sí' if pos <= 8 else 'No'
        finalista = 'No'
        campeon = 'No'
        
        if clasifico == 'Sí':
            if pos <= 4 or puntos > 32:
                if np.random.rand() > 0.6: finalista = 'Sí'
        if finalista == 'Sí':
            if np.random.rand() > 0.5: campeon = 'Sí'
            
        if (y == 2004 and s == 'I') or (y == 2009 and s == 'II') or (y == 2016 and s == 'I'):
            pos = max(1, pos - 3)
            clasifico = 'Sí'
            finalista = 'Sí'
            campeon = 'Sí'
            
        data.append([f"{y}-{s}", y, s, puntos, victorias, empates, derrotas, gf, gc, dg, pos, clasifico, finalista, campeon])
        
    df = pd.DataFrame(data, columns=[
        'Torneo', 'Año', 'Semestre', 'Puntos', 'Victorias', 'Empates', 'Derrotas', 
        'Goles a favor', 'Goles en contra', 'Diferencia de gol', 'Posición', 'Clasificó a cuadrangulares', 'Finalista', 'Campeón'
    ])
    return df

df = generate_data()

df['Clasificó_num'] = df['Clasificó a cuadrangulares'].map({'Sí': 1, 'No': 0})
df['Finalista_num'] = df['Finalista'].map({'Sí': 1, 'No': 0})
df['Campeón_num'] = df['Campeón'].map({'Sí': 1, 'No': 0})
df['Periodo'] = np.arange(len(df))

# Encabezado
col_img, col_title = st.columns([1, 5])
with col_title:
    st.title("🔴🔵 DIM: Dashboard de Analítica Deportiva")
    st.markdown("Análisis histórico y predicción del rendimiento del Deportivo Independiente Medellín mediante Machine Learning.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen Ejecutivo", "Datos Históricos", "Análisis y Tendencias", "Modelos ML y Predicción", "Proyecciones y Monte Carlo"])

with tab1:
    st.header("Métricas Globales (2003 - 2026)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h3>Total Torneos</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card' style='background-color:#001e60;'><h3>Total Títulos</h3><h2>{df['Campeón_num'].sum()}</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Promedio Puntos</h3><h2>{df['Puntos'].mean():.1f}</h2></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card' style='background-color:#001e60;'><h3>Promedio Posición</h3><h2>{df['Posición'].mean():.1f}</h2></div>", unsafe_allow_html=True)
        
    st.write("---")
    c5, c6, c7, c8 = st.columns(4)
    with c5: st.metric("Total Victorias", df['Victorias'].sum())
    with c6: st.metric("Total Derrotas", df['Derrotas'].sum())
    with c7: st.metric("Promedio Goles a Favor", f"{df['Goles a favor'].mean():.1f}")
    with c8: st.metric("Promedio Goles en Contra", f"{df['Goles en contra'].mean():.1f}")
        
    st.subheader("Índice de Rendimiento")
    score_pts = df['Puntos'].mean() / 45 * 100
    score_pos = (21 - df['Posición'].mean()) / 20 * 100
    score_vic = df['Victorias'].mean() / 20 * 100
    idx_val = (score_pts * 0.4 + score_pos * 0.4 + score_vic * 0.2)
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number", value = idx_val, title = {'text': "Índice de Rendimiento (0-100)"},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#e3001b"}}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.header("Base de Datos Histórica")
    st.dataframe(df.drop(columns=['Clasificó_num', 'Finalista_num', 'Campeón_num', 'Periodo']), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(df, x='Torneo', y='Puntos', title='Puntos por semestre', markers=True).update_traces(line_color='#e3001b'), use_container_width=True)
        st.plotly_chart(px.bar(df, x='Torneo', y=['Goles a favor', 'Goles en contra'], title='Goles a Favor vs En Contra', barmode='group'), use_container_width=True)
        st.plotly_chart(px.line(df, x='Torneo', y='Victorias', title='Victorias por semestre', markers=True).update_traces(line_color='#001e60'), use_container_width=True)
        st.plotly_chart(px.histogram(df, x='Puntos', nbins=15, title='Histograma de puntos', color_discrete_sequence=['#e3001b']), use_container_width=True)
    with c2:
        st.plotly_chart(px.line(df, x='Torneo', y='Posición', title='Posición por semestre', markers=True).update_layout(yaxis=dict(autorange="reversed")).update_traces(line_color='#001e60'), use_container_width=True)
        st.plotly_chart(px.bar(df, x='Torneo', y='Diferencia de gol', title='Diferencia de gol por semestre').update_traces(marker_color='#e3001b'), use_container_width=True)
        st.plotly_chart(px.pie(df, names='Clasificó a cuadrangulares', title='Frecuencia de clasificaciones', color_discrete_sequence=['#e3001b', '#001e60']), use_container_width=True)
        st.plotly_chart(px.histogram(df, x='Posición', nbins=20, title='Histograma de posiciones', color_discrete_sequence=['#001e60']), use_container_width=True)

with tab3:
    st.header("Análisis Estadístico Descriptivo")
    stats = df[['Puntos', 'Victorias', 'Diferencia de gol', 'Posición']].agg(['mean', 'median', 'std', 'min', 'max'])
    stats.loc['moda'] = df[['Puntos', 'Victorias', 'Diferencia de gol', 'Posición']].mode().iloc[0]
    stats.loc['cv'] = stats.loc['std'] / stats.loc['mean']
    st.dataframe(stats.T)
    
    st.subheader("Matriz de Correlación")
    corr = df[['Puntos', 'Victorias', 'Diferencia de gol', 'Posición']].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
    
    st.subheader("Análisis de Tendencias")
    X = df[['Periodo']]
    y = df['Puntos']
    model_trend = LinearRegression().fit(X, y)
    trend_slope = model_trend.coef_[0]
    trend_msg = "El Medellín presenta una tendencia histórica positiva." if trend_slope > 0 else "El Medellín presenta una tendencia histórica negativa (ha empeorado)."
    st.info(trend_msg)
    
    fig_trend = px.scatter(df, x='Periodo', y='Puntos', title="Evolución de puntos con línea de tendencia")
    fig_trend.add_traces(go.Scatter(x=df['Periodo'], y=model_trend.predict(X), mode='lines', name='Tendencia'))
    st.plotly_chart(fig_trend, use_container_width=True)

with tab4:
    st.header("Predicción del Semestre Actual (2026-I)")
    
    train_df = df.iloc[:-1]
    X_train = train_df[['Periodo']]
    
    reg_pts = RandomForestRegressor(random_state=42).fit(X_train, train_df['Puntos'])
    reg_pos = RandomForestRegressor(random_state=42).fit(X_train, train_df['Posición'])
    
    pred_pts = reg_pts.predict([[len(df)-1]])[0]
    pred_pos = max(1, min(20, round(reg_pos.predict([[len(df)-1]])[0])))
    
    features = ['Puntos', 'Victorias', 'Diferencia de gol', 'Posición']
    X_clf = train_df[features]
    
    clf_clasif = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_clf, train_df['Clasificó_num'])
    clf_final = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_clf, train_df['Finalista_num'])
    clf_camp = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_clf, train_df['Campeón_num'])
    
    pred_vict = pred_pts / 3 * 0.8
    pred_dg = pred_pts - 25
    curr_feat = [[pred_pts, pred_vict, pred_dg, pred_pos]]
    
    p_clasif = clf_clasif.predict_proba(curr_feat)[0][1]
    p_final = clf_final.predict_proba(curr_feat)[0][1]
    p_camp = clf_camp.predict_proba(curr_feat)[0][1]
    
    ca, cb = st.columns(2)
    with ca:
        st.metric("Posición esperada", f"{pred_pos:.0f}")
        st.metric("Puntos esperados", f"{pred_pts:.1f}")
    with cb:
        st.progress(p_clasif, text=f"Probabilidad de clasificar: {p_clasif*100:.1f}%")
        st.progress(p_final, text=f"Probabilidad de llegar a la final: {p_final*100:.1f}%")
        st.progress(p_camp, text=f"Probabilidad de ser campeón: {p_camp*100:.1f}%")

with tab5:
    st.header("Predicción del próximo campeonato")
    future_periods = np.arange(len(df), len(df)+30).reshape(-1, 1)
    future_pts = reg_pts.predict(future_periods)
    future_pos = reg_pos.predict(future_periods)
    
    idx_camp = -1
    for i in range(len(future_periods)):
        p_c = clf_camp.predict_proba([[future_pts[i], future_pts[i]/3*0.8, future_pts[i]-25, future_pos[i]]])[0][1]
        if p_c + np.random.normal(0, 0.05) > 0.35:
            idx_camp = i
            break
            
    if idx_camp != -1:
        sem_add = (idx_camp + 1) % 2
        y_add = (idx_camp + 1) // 2
        f_y = 2026 + y_add
        f_s = 'I' if (1 + sem_add) % 2 == 1 else 'II'
        st.success(f"### ¿Cuándo volverá a ganar una liga el Medellín?: {f_y}-{f_s}")
    
    st.write("---")
    st.subheader("Predicción de títulos futuros (Monte Carlo)")
    anos_proj = st.slider("Intervalo de años", 5, 30, 10, step=5)
    
    prob_base = df['Campeón_num'].mean() * (1.1 if trend_slope > 0 else 0.9)
    simulaciones = np.random.binomial(n=anos_proj*2, p=prob_base, size=10000)
    
    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Número esperado de títulos (Media)", f"{simulaciones.mean():.1f}")
    cs2.metric("Mediana", f"{np.median(simulaciones):.0f}")
    cs3.metric("Percentiles (5 - 95)", f"{int(np.percentile(simulaciones, 5))} - {int(np.percentile(simulaciones, 95))}")
    
    fig_mc = px.histogram(simulaciones, title=f'Distribución de Títulos en {anos_proj} años (10,000 simulaciones)', labels={'value':'Títulos'})
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.subheader("Conclusiones automáticas")
    st.write(f"- {trend_msg}")
    st.write(f"- La probabilidad de clasificar ha {'aumentado' if p_clasif > df['Clasificó_num'].mean() else 'disminuido'}.")
    if idx_camp != -1:
        st.write(f"- El siguiente campeonato es más probable en {f_y}-{f_s}.")
    st.write(f"- En los próximos {anos_proj} años se esperan aproximadamente {simulaciones.mean():.1f} títulos.")
