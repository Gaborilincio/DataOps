import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from PIL import Image
from sqlalchemy import create_engine
import os

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Caffé Ops",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #f59e0b;
    }
    .stMetric label { font-size: 0.85rem !important; color: #94a3b8 !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def cargar_datos():
    """Carga desde PostgreSQL, con fallback a CSV procesado."""
    DB_HOST = os.getenv("DB_HOST", "db")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "cafedataops")
    DB_USER = os.getenv("DB_USER", "cafe_user")
    DB_PASS = os.getenv("DB_PASSWORD", "cafe_pass")

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        df = pd.read_sql("SELECT * FROM ventas_cafeteria", engine)
        fuente = "PostgreSQL"
    except Exception:
        fallback = "data/processed/ventas_limpias.csv"
        if os.path.exists(fallback):
            df = pd.read_csv(fallback, parse_dates=["Transaction Date"])
        else:
            st.error("No se encontró fuente de datos. Ejecuta el pipeline primero.")
            st.stop()
        fuente = "CSV (fallback)"

    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    return df, fuente

df, fuente = cargar_datos()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/twitter/376/hot-beverage_2615.png", width=60)
    st.title("Caffé Ops")
    st.caption(f"Fuente de datos: **{fuente}**")
    st.divider()

    items_disponibles = sorted(df["Item"].unique())
    items_sel = st.multiselect("Filtrar por producto", items_disponibles, default=items_disponibles)

    fecha_min = df["Transaction Date"].min().date()
    fecha_max = df["Transaction Date"].max().date()
    rango = st.date_input("Rango de fechas", value=(fecha_min, fecha_max),
                          min_value=fecha_min, max_value=fecha_max)

    metodos = sorted(df["Payment Method"].unique())
    metodos_sel = st.multiselect("Método de pago", metodos, default=metodos)

    st.divider()
    if st.button("Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

# ── Filtrado ─────────────────────────────────────────────────────────────────
if len(rango) == 2:
    df_f = df[
        (df["Item"].isin(items_sel)) &
        (df["Transaction Date"].dt.date >= rango[0]) &
        (df["Transaction Date"].dt.date <= rango[1]) &
        (df["Payment Method"].isin(metodos_sel))
    ]
else:
    df_f = df[df["Item"].isin(items_sel) & df["Payment Method"].isin(metodos_sel)]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Caffé Ops — Dashboard de Ventas")
st.caption("Pipeline DataOps automatizado · datos limpios y validados")
st.divider()

# ── KPIs principales ─────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total transacciones", f"{len(df_f):,}")
k2.metric("Ingresos totales", f"${df_f['Total Spent'].sum():,.2f}")
k3.metric("Ticket promedio", f"${df_f['Total Spent'].mean():.2f}")
k4.metric("Producto top", df_f.groupby("Item")["Total Spent"].sum().idxmax() if not df_f.empty else "-")
k5.metric("Días con datos", df_f["Transaction Date"].dt.date.nunique())

st.divider()

# ── Fila 1: Ventas por producto + Método de pago ─────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Ingresos por producto")
    ventas_item = (
        df_f.groupby("Item")["Total Spent"]
        .sum().sort_values(ascending=True).reset_index()
    )
    fig = px.bar(ventas_item, x="Total Spent", y="Item", orientation="h",
                 color="Total Spent", color_continuous_scale="Oranges",
                 labels={"Total Spent": "Ingresos ($)", "Item": ""})
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=0, b=0), height=320)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Método de pago")
    pay = df_f["Payment Method"].value_counts().reset_index()
    pay.columns = ["Método", "Cantidad"]
    fig2 = px.pie(pay, names="Método", values="Cantidad",
                  color_discrete_sequence=px.colors.sequential.Oranges_r,
                  hole=0.45)
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=320,
                       legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig2, use_container_width=True)

# ── Fila 2: Ventas en el tiempo + Ubicación ───────────────────────────────────
col3, col4 = st.columns([3, 2])

with col3:
    st.subheader("Evolución de ventas diarias")
    ventas_dia = df_f.groupby(df_f["Transaction Date"].dt.date)["Total Spent"].sum().reset_index()
    ventas_dia.columns = ["Fecha", "Ingresos"]
    fig3 = px.area(ventas_dia, x="Fecha", y="Ingresos",
                   color_discrete_sequence=["#f59e0b"],
                   labels={"Ingresos": "Ingresos ($)", "Fecha": ""})
    fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Ventas por ubicación")
    loc = df_f.groupby("Location")["Total Spent"].sum().reset_index()
    fig4 = px.bar(loc, x="Location", y="Total Spent",
                  color="Location",
                  color_discrete_sequence=["#f59e0b", "#fb923c", "#fcd34d"],
                  labels={"Total Spent": "Ingresos ($)", "Location": ""})
    fig4.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=300)
    st.plotly_chart(fig4, use_container_width=True)

# ── Fila 3: Heatmap día × producto ──────────────────────────────────────────
st.subheader("Volumen de ventas: día de semana × producto")
orden_dias = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_data = df_f.groupby(["DayOfWeek","Item"])["Quantity"].sum().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex([d for d in orden_dias if d in heatmap_data.index])
fig5 = px.imshow(heatmap_data, color_continuous_scale="Oranges",
                 labels=dict(x="Producto", y="Día", color="Unidades"),
                 aspect="auto")
fig5.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
st.plotly_chart(fig5, use_container_width=True)

# ── Tabla de datos limpios ────────────────────────────────────────────────────
with st.expander("Ver datos limpios y validados"):
    st.dataframe(
        df_f[["Transaction ID","Item","Quantity","Price Per Unit",
              "Total Spent","Payment Method","Location","Transaction Date"]]
        .sort_values("Transaction Date", ascending=False),
        use_container_width=True, height=300
    )
    st.download_button(
        "Descargar CSV limpio",
        df_f.to_csv(index=False).encode("utf-8"),
        "ventas_limpias.csv", "text/csv"
    )


st.divider()
st.header("🤖 Modelo IA — Predicción de Venta Alta/Baja")
st.caption("Clasificación binaria entrenada sobre datos limpios del pipeline · Regresión Logística")
 
# ── Cargar reporte del modelo ─────────────────────────────────────────────────
RUTA_REPORTE = "output/modelo/reporte_modelo.json"
 
if not os.path.exists(RUTA_REPORTE):
    st.warning("⚠️ No se encontró el reporte del modelo. Ejecuta primero: `python modelo_ia.py`")
else:
    with open(RUTA_REPORTE, "r", encoding="utf-8") as f:
        reporte = json.load(f)
 
    mejor     = reporte["modelo_seleccionado"]
    metricas  = reporte["metricas"][mejor]
    cm        = metricas["confusion_matrix"]
 
    # ── KPIs del modelo ───────────────────────────────────────────────────────
    st.subheader(f"Modelo seleccionado: {mejor}")
    st.caption(reporte.get("justificacion", ""))
 
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Accuracy",  f"{metricas['accuracy']*100:.1f}%")
    k2.metric("Precision", f"{metricas['precision']*100:.1f}%")
    k3.metric("Recall",    f"{metricas['recall']*100:.1f}%")
    k4.metric("F1 Score",  f"{metricas['f1_score']*100:.1f}%")
    k5.metric("AUC-ROC",   f"{metricas['auc_roc']:.3f}")
    k6.metric("Gini",      f"{metricas['gini']:.3f}")
 
    st.divider()
 
    # ── Interpretación de métricas ────────────────────────────────────────────
    with st.expander("📖 Interpretación de las métricas"):
        st.markdown(f"""
        | Métrica | Valor | Interpretación |
        |---|---|---|
        | **Accuracy** | {metricas['accuracy']*100:.1f}% | El modelo acierta el {metricas['accuracy']*100:.1f}% de las predicciones |
        | **Precision** | {metricas['precision']*100:.1f}% | Cuando predice venta alta, acierta el {metricas['precision']*100:.1f}% de las veces |
        | **Recall** | {metricas['recall']*100:.1f}% | Detecta el {metricas['recall']*100:.1f}% de las ventas altas reales |
        | **F1 Score** | {metricas['f1_score']*100:.1f}% | Balance entre precision y recall |
        | **AUC-ROC** | {metricas['auc_roc']:.3f} | Capacidad discriminativa del modelo (0.5 = azar, 1.0 = perfecto) |
        | **Gini** | {metricas['gini']:.3f} | Poder predictivo sobre clasificador aleatorio (0 = nulo, 1 = perfecto) |
        """)
        st.info(
            f"El modelo supera al clasificador aleatorio (50%) con un accuracy de "
            f"{metricas['accuracy']*100:.1f}%. El Gini de {metricas['gini']:.3f} indica "
            f"capacidad predictiva moderada, esperada para un dataset de 464 registros "
            f"con variables contextuales de comportamiento de compra."
        )
 
    # ── Matriz de confusión ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Matriz de Confusión")
        df_cm = pd.DataFrame({
            "Predicho: Baja (0)": [cm["TN"], cm["FN"]],
            "Predicho: Alta (1)": [cm["FP"], cm["TP"]],
        }, index=["Real: Baja (0)", "Real: Alta (1)"])
        st.dataframe(df_cm, use_container_width=True)
 
        st.markdown(f"""
        - **TP={cm['TP']}** — Ventas altas correctamente identificadas
        - **TN={cm['TN']}** — Ventas bajas correctamente identificadas
        - **FP={cm['FP']}** — Ventas bajas clasificadas erróneamente como altas
        - **FN={cm['FN']}** — Ventas altas que el modelo no detectó
        """)
 
    with col2:
        st.subheader("Comparación de modelos")
        modelos_nombres = list(reporte["metricas"].keys())
        df_comp = pd.DataFrame({
            nombre: {
                "Accuracy":  reporte["metricas"][nombre]["accuracy"],
                "Precision": reporte["metricas"][nombre]["precision"],
                "Recall":    reporte["metricas"][nombre]["recall"],
                "F1 Score":  reporte["metricas"][nombre]["f1_score"],
                "AUC-ROC":   reporte["metricas"][nombre]["auc_roc"],
                "Gini":      reporte["metricas"][nombre]["gini"],
            }
            for nombre in modelos_nombres
        }).T
        st.dataframe(df_comp.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
        st.caption("Verde = mejor valor por métrica")
 
    st.divider()
 
    # ── Gráficos del modelo ───────────────────────────────────────────────────
    st.subheader("Gráficos del modelo")
    graficos_modelo = {
        "Matriz de Confusión":      "output/modelo/01_matriz_confusion.png",
        "Curva ROC":                "output/modelo/02_curva_roc.png",
        "Comparación de métricas":  "output/modelo/03_comparacion_metricas.png",
    }
 
    col_g1, col_g2 = st.columns(2)
    for i, (titulo, ruta) in enumerate(graficos_modelo.items()):
        col = col_g1 if i % 2 == 0 else col_g2
        if os.path.exists(ruta):
            col.markdown(f"**{titulo}**")
            col.image(ruta, use_column_width=True)
        else:
            col.warning(f"Gráfico no encontrado: {ruta}")
 
    # ── Partición de datos ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Partición del dataset")
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Total registros", "464")

st.caption("Caffé Ops 2026 · Jaime Barrales · Gabriel Méndez · Iván Álvarez")
