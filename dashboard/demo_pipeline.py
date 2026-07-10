"""
demo_pipeline.py — CafféData Ops
Demo interactiva del pipeline ETL para presentación Parcial 3.
Ejecutar con: streamlit run demo_pipeline.py
"""

import time
import os
import json
import uuid
import datetime
import pandas as pd
import joblib
import streamlit as st
from sqlalchemy import create_engine, text

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CafféData Ops — Demo Pipeline",
    page_icon="☕",
    layout="wide"
)

# ── Estilos (CSS básico para mejorar la interfaz) ─────────────────────────────
st.markdown("""
<style>
.etapa-header {
    background: #f0f2f6;
    border-left: 4px solid #1f77b4;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 12px;
    font-weight: 600;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes y Configuración de BD ──────────────────────────────────────────
RUTA_CSV = "data/raw/dirty_cafe_sales.csv"

# Usamos variables de entorno con valores por defecto (buena práctica de seguridad básica)
DB_HOST  = os.getenv("DB_HOST", "localhost")
DB_PORT  = os.getenv("DB_PORT", "5432")
DB_NAME  = os.getenv("DB_NAME", "cafedataops")
DB_USER  = os.getenv("DB_USER", "cafe_user")
DB_PASS  = os.getenv("DB_PASSWORD", "cafe_pass")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Funciones del Pipeline (Separación de responsabilidades) ──────────────────

def etapa_ingesta(ruta: str) -> pd.DataFrame:
    """Lee el CSV crudo sin alterar el archivo original."""
    if not os.path.exists(ruta):
        st.error(f"Archivo no encontrado: {ruta}")
        st.stop()
    
    # Leemos todo como texto inicialmente para evitar errores de parseo automáticos
    return pd.read_csv(ruta, dtype=str)

def etapa_limpieza(df: pd.DataFrame) -> tuple:
    """Aplica transformaciones básicas de limpieza de datos."""
    log = {}
    df = df.copy()

    # 1. Eliminar duplicados exactos
    antes = len(df)
    df.drop_duplicates(inplace=True)
    log["duplicados_eliminados"] = antes - len(df)

    # 2. Limpiar la columna 'Total Spent' y recalcular donde dice 'ERROR'
    df["Total Spent"] = df["Total Spent"].astype(str).str.strip().str.upper()
    mask_error = df["Total Spent"] == "ERROR"
    
    # Convertimos temporalmente a object para poder mezclar strings y números
    df["Total Spent"] = df["Total Spent"].astype(object)
    
    recuperados = 0
    # Iteramos solo sobre las filas con error para recalcular (enfoque simple y legible)
    for idx in df[mask_error].index:
        qty = df.at[idx, "Quantity"]
        price = df.at[idx, "Price Per Unit"]
        if pd.notna(qty) and pd.notna(price):
            try:
                df.at[idx, "Total Spent"] = round(float(qty) * float(price), 2)
                recuperados += 1
            except ValueError:
                pass
    log["total_spent_recalculados"] = recuperados

    # 3. Eliminar registros críticos incompletos (huérfanos)
    antes = len(df)
    df.dropna(subset=["Transaction ID", "Item"], inplace=True)
    log["registros_huerfanos"] = antes - len(df)

    # 4. Imputar valores 'UNKNOWN' o 'ERROR' en columnas categóricas
    imputados = 0
    for col in ["Payment Method", "Location"]:
        mascara = df[col].astype(str).str.upper().isin(["UNKNOWN", "ERROR", "NAN", ""])
        imputados += int(mascara.sum())
        df.loc[mascara, col] = "No Especificado"
    log["imputados_unknown"] = imputados

    # 5. Normalizar textos (quitar espacios extra y capitalizar)
    for col in ["Item", "Payment Method", "Location"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 6. Estandarizar fechas
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    fechas_corruptas = int(df["Transaction Date"].isna().sum())
    df.dropna(subset=["Transaction Date"], inplace=True)
    log["fechas_corruptas"] = fechas_corruptas

    # 7. Derivar columnas de tiempo para el modelo de datos final
    df["Hour"]      = df["Transaction Date"].dt.hour
    df["Month"]     = df["Transaction Date"].dt.month
    df["DayOfWeek"] = df["Transaction Date"].dt.day_name()

    return df, log

def etapa_validacion(df: pd.DataFrame, total_crudos: int) -> tuple:
    """Valida tipos de datos y reglas de negocio. Separa registros válidos de rechazados."""
    metricas = {"validaciones": {}}
    rechazados = []

    # 1. Asegurar tipos numéricos
    df["Quantity"]       = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df["Total Spent"]    = pd.to_numeric(df["Total Spent"], errors="coerce")
    
    mask_tipo = df[["Quantity", "Price Per Unit", "Total Spent"]].isna().any(axis=1)
    if mask_tipo.any():
        rechazados.append(df[mask_tipo].copy().assign(motivo_rechazo="Tipo de dato numérico inválido"))
        df = df[~mask_tipo]
    metricas["validaciones"]["tipo_invalido"] = int(mask_tipo.sum())

    # 2. Reglas de negocio: Valores positivos
    mask_neg = (df["Quantity"] <= 0) | (df["Price Per Unit"] <= 0)
    if mask_neg.any():
        rechazados.append(df[mask_neg].copy().assign(motivo_rechazo="Valor negativo o cero"))
        df = df[~mask_neg]
    metricas["validaciones"]["valores_negativos"] = int(mask_neg.sum())

    # 3. Consistencia matemática básica
    total_esperado  = (df["Quantity"] * df["Price Per Unit"]).round(2)
    total_real      = df["Total Spent"].round(2)
    mask_incons     = (total_esperado - total_real).abs() > 0.05
    if mask_incons.any():
        rechazados.append(df[mask_incons].copy().assign(motivo_rechazo="Inconsistencia matemática"))
        df = df[~mask_incons]
    metricas["validaciones"]["inconsistencia_matematica"] = int(mask_incons.sum())

    # Calcular métricas finales
    registros_validos   = len(df)
    registros_rechazados = sum(len(r) for r in rechazados)
    completitud         = round((registros_validos / total_crudos) * 100, 1) if total_crudos > 0 else 0

    metricas.update({
        "total_crudos": total_crudos,
        "registros_validos": registros_validos,
        "registros_rechazados": registros_rechazados,
        "completitud_pct": completitud,
        "supera_umbral": completitud >= 90.0
    })

    df_rechazados = pd.concat(rechazados, ignore_index=True) if rechazados else pd.DataFrame()
    return df, metricas, df_rechazados

def etapa_carga(df: pd.DataFrame) -> dict:
    """Intenta guardar en BD (PostgreSQL). Si falla, hace fallback a CSV local."""
    resultado = {}
    inicio = time.time()
    
    try:
        engine = create_engine(DATABASE_URL)
        # Probamos la conexión antes de intentar cargar
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        df.to_sql("ventas_cafeteria", con=engine, if_exists="replace", index=False)
        resultado.update({"estado": "ok", "registros": len(df), "motor": "PostgreSQL"})
        
    except Exception as e:
        # Fallback local seguro si la BD no está disponible
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/ventas_limpias.csv", index=False)
        resultado.update({
            "estado": "fallback",
            "error": str(e),
            "registros": len(df),
            "motor": "CSV local"
        })
        
    resultado["latencia_seg"] = round(time.time() - inicio, 3)
    return resultado


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("☕ CafféData Ops — Demo Pipeline ETL")
st.caption("Demostración del flujo completo: CSV crudo → limpieza → validación → base de datos")
st.divider()

# ── Inicializar estado de sesión ──────────────────────────────────────────────
variables_estado = [
    "df_crudo", "df_limpio", "log_limpieza",
    "df_valido", "metricas", "df_rechazados", "resultado_carga"
]
for var in variables_estado:
    if var not in st.session_state:
        st.session_state[var] = None

# ── ETAPA 1: Ingesta ──────────────────────────────────────────────────────────
st.markdown('<div class="etapa-header">📥 Etapa 1 — Ingesta de datos</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶ Ejecutar ingesta", use_container_width=True, type="primary"):
        with st.spinner("Leyendo CSV..."):
            time.sleep(0.5) # Simulación breve de latencia
            st.session_state.df_crudo = etapa_ingesta(RUTA_CSV)

if st.session_state.df_crudo is not None:
    df_c = st.session_state.df_crudo
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros crudos", len(df_c))
    c2.metric("Columnas detectadas", len(df_c.columns))
    c3.metric("Tamaño en memoria", f"{df_c.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with st.expander("🔍 Ver muestra del CSV crudo"):
        st.dataframe(df_c.head(), use_container_width=True)

st.divider()

# ── ETAPA 2: Limpieza ─────────────────────────────────────────────────────────
st.markdown('<div class="etapa-header">🧹 Etapa 2 — Limpieza y transformación</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶ Ejecutar limpieza", use_container_width=True, type="primary", disabled=(st.session_state.df_crudo is None)):
        with st.spinner("Aplicando transformaciones..."):
            df_l, log_l = etapa_limpieza(st.session_state.df_crudo)
            st.session_state.df_limpio = df_l
            st.session_state.log_limpieza = log_l

if st.session_state.df_limpio is not None:
    log = st.session_state.log_limpieza
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicados eliminados", log.get("duplicados_eliminados", 0))
    c2.metric("Precios recalculados", log.get("total_spent_recalculados", 0))
    c3.metric("Registros huérfanos", log.get("registros_huerfanos", 0))
    c4.metric("UNKNOWN imputados", log.get("imputados_unknown", 0))

st.divider()

# ── ETAPA 3: Validación ───────────────────────────────────────────────────────
st.markdown('<div class="etapa-header">✅ Etapa 3 — Validación estructural y semántica</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶ Ejecutar validación", use_container_width=True, type="primary", disabled=(st.session_state.df_limpio is None)):
        with st.spinner("Ejecutando compuertas de calidad..."):
            df_v, met, df_rech = etapa_validacion(st.session_state.df_limpio, len(st.session_state.df_crudo))
            st.session_state.df_valido = df_v
            st.session_state.metricas = met
            st.session_state.df_rechazados = df_rech

if st.session_state.metricas is not None:
    met = st.session_state.metricas
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros válidos", met['registros_validos'])
    c2.metric("Registros rechazados", met['registros_rechazados'])
    c3.metric("Completitud", f"{met['completitud_pct']}%")

    if met["supera_umbral"]:
        st.success("✅ Completitud supera el umbral del 90%. Pipeline autorizado para cargar.")
    else:
        st.error("🚨 Completitud bajo el umbral del 90%. Se requiere revisión.")

st.divider()

# ── ETAPA 4: Carga ────────────────────────────────────────────────────────────
st.markdown('<div class="etapa-header">🗄️ Etapa 4 — Persistencia (Base de Datos)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    puede_cargar = st.session_state.df_valido is not None and st.session_state.metricas.get("supera_umbral", False)
    if st.button("▶ Ejecutar carga", use_container_width=True, type="primary", disabled=not puede_cargar):
        with st.spinner("Conectando y guardando datos..."):
            st.session_state.resultado_carga = etapa_carga(st.session_state.df_valido)

if st.session_state.resultado_carga is not None:
    res = st.session_state.resultado_carga
    if res["estado"] == "ok":
        st.success(f"✅ {res['registros']} registros guardados en PostgreSQL.")
    else:
        st.warning(f"⚠️ PostgreSQL falló. Backup guardado en CSV local. Error: {res.get('error','')[:60]}...")

st.divider()

# ── ETAPA 5: Ingreso Manual ───────────────────────────────────────────────────
st.markdown('<div class="etapa-header">✍️ Etapa 5 — Ingreso de transacción en vivo</div>', unsafe_allow_html=True)

ITEMS_DISPONIBLES = ["Coffee", "Latte", "Cappuccino", "Espresso", "Tea", "Juice", "Cake", "Muffin", "Sandwich", "Cookie"]
METODOS_PAGO = ["Cash", "Credit Card", "Debit Card", "Mobile Pay"]
UBICACIONES = ["In-store", "Takeaway", "Online"]

with st.form("form_nueva_transaccion", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        item = st.selectbox("Producto", ITEMS_DISPONIBLES)
        cantidad = st.number_input("Cantidad", min_value=1, max_value=50, value=1)
        precio = st.number_input("Precio unitario ($)", min_value=0.01, value=3.50)
    with c2:
        metodo = st.selectbox("Método de pago", METODOS_PAGO)
        ubicacion = st.selectbox("Ubicación", UBICACIONES)
        fecha = st.date_input("Fecha")

    if st.form_submit_button("➕ Insertar transacción", type="primary", use_container_width=True):
        
        # Validaciones simples antes de crear el registro
        errores = []
        if cantidad <= 0: errores.append("Cantidad debe ser mayor a 0")
        if precio <= 0: errores.append("Precio debe ser mayor a 0")
        
        if errores:
            for e in errores: st.error(f"❌ {e}")
        else:
            fecha_dt = datetime.datetime.combine(fecha, datetime.time(12, 0))
            nuevo_registro = {
                "Transaction ID": f"TXN-{str(uuid.uuid4())[:8].upper()}",
                "Item": item,
                "Quantity": int(cantidad),
                "Price Per Unit": float(precio),
                "Total Spent": round(cantidad * precio, 2),
                "Payment Method": metodo,
                "Location": ubicacion,
                "Transaction Date": fecha_dt,
                "Hour": 12,
                "Month": fecha_dt.month,
                "DayOfWeek": fecha_dt.strftime("%A"),
            }
            
            st.success("✅ Registro validado.")
            try:
                engine = create_engine(DATABASE_URL)
                pd.DataFrame([nuevo_registro]).to_sql("ventas_cafeteria", con=engine, if_exists="append", index=False)
                st.success("🗄️ Transacción insertada en la BD correctamente.")
            except Exception as e:
                st.warning("⚠️ Sin conexión a BD. Modo Fallback (No se guardó el registro individual en CSV por simplicidad del demo).")

st.divider()

# ── ETAPA 6: Predicción en vivo con Modelo IA ─────────────────────────────────
st.markdown('<div class="etapa-header">🤖 Etapa 6 — Predicción en vivo (Modelo IA)</div>', unsafe_allow_html=True)
st.caption("Usa el modelo real entrenado en la Fase 3 (Regresión Logística / Random Forest) para predecir "
           "si una transacción será de venta ALTA o BAJA respecto a la mediana histórica.")

RUTA_MODELO   = "output/modelo/modelo.pkl"
RUTA_FEATURES = "output/modelo/feature_columns.json"


@st.cache_resource
def cargar_modelo():
    """Carga el modelo persistido y las columnas de features que espera (one-hot encoding)."""
    if not os.path.exists(RUTA_MODELO) or not os.path.exists(RUTA_FEATURES):
        return None, None
    paquete = joblib.load(RUTA_MODELO)
    with open(RUTA_FEATURES, "r", encoding="utf-8") as f:
        columnas = json.load(f)
    return paquete, columnas


paquete_modelo, columnas_modelo = cargar_modelo()

if paquete_modelo is None:
    st.warning(
        "⚠️ No se encontró el modelo entrenado en `output/modelo/modelo.pkl`. "
        "Ejecuta primero: `python modelo/modelo_ia.py`"
    )
else:
    with st.form("form_prediccion"):
        c1, c2 = st.columns(2)
        with c1:
            item_pred    = st.selectbox("Producto", ITEMS_DISPONIBLES, key="item_pred")
            metodo_pred  = st.selectbox("Método de pago", METODOS_PAGO, key="metodo_pred")
        with c2:
            ubicacion_pred = st.selectbox("Ubicación", UBICACIONES, key="ubicacion_pred")
            fecha_pred     = st.date_input("Fecha de la venta", key="fecha_pred")
        hora_pred = st.slider("Hora del día", 0, 23, 12)

        if st.form_submit_button("🔮 Predecir venta alta/baja", type="primary", use_container_width=True):
            # Construimos la fila cruda con las mismas columnas que vio el modelo antes del one-hot
            fila = pd.DataFrame([{
                "Item":           item_pred,
                "Payment Method": metodo_pred,
                "Location":       ubicacion_pred,
                "Hour":           hora_pred,
                "Month":          fecha_pred.month,
                "DayOfWeek":      fecha_pred.strftime("%A"),
            }])

            # Mismo one-hot encoding que se usó en el entrenamiento (eda_modelo.py)
            fila_encoded = pd.get_dummies(fila, columns=["Item", "Payment Method", "Location", "DayOfWeek"])
            # Reindexamos a las columnas EXACTAS del entrenamiento; categorías no vistas quedan en 0
            fila_encoded = fila_encoded.reindex(columns=columnas_modelo, fill_value=0)

            modelo = paquete_modelo["modelo"]
            scaler = paquete_modelo.get("scaler")
            X_pred = scaler.transform(fila_encoded) if scaler is not None else fila_encoded

            pred  = modelo.predict(X_pred)[0]
            proba = modelo.predict_proba(X_pred)[0][1]

            if pred == 1:
                st.success(f"🔼 **Venta ALTA** — probabilidad estimada: {proba*100:.1f}%")
            else:
                st.info(f"🔽 **Venta BAJA** — probabilidad de ser alta: {proba*100:.1f}%")

            st.caption(
                "Predicción generada por el modelo persistido en `output/modelo/modelo.pkl` "
                "(mismo modelo entrenado y evaluado con F1/AUC-ROC/Gini en la Fase 3)."
            )