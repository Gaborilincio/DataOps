import time
import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text

from ingestion.lectura_csv import leer_datos_csv
from procesamiento.transformacion import generar_transformaciones
from data_quality.validacion import ejecutar_validaciones

# ── Configuración de logging ─────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)

# ── Configuración de base de datos ───────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cafedataops")
DB_USER = os.getenv("DB_USER", "cafe_user")
DB_PASS = os.getenv("DB_PASSWORD", "cafe_pass")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def run_pipeline():
    print("=" * 60)
    print("Caffé Ops — Pipeline ETL")
    print("=" * 60)
    inicio = time.time()
    logging.info("Pipeline iniciado")

    # ── A. INGESTA ───────────────────────────────────────────────────────────
    df_crudo = leer_datos_csv("data/raw/dirty_cafe_sales.csv")
    total_crudos = len(df_crudo)

    # ── B. TRANSFORMACIÓN ────────────────────────────────────────────────────
    df_limpio = generar_transformaciones(df_crudo)

    # ── C. VALIDACIÓN ────────────────────────────────────────────────────────
    df_valido = ejecutar_validaciones(df_limpio, total_crudos)

    # ── D. PERSISTENCIA ──────────────────────────────────────────────────────
    print(f"\n[CARGA] Conectando a PostgreSQL en {DB_HOST}:{DB_PORT}...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  ✓ Conexión exitosa")

        df_valido.to_sql(
            name="ventas_cafeteria",
            con=engine,
            if_exists="replace",
            index=False
        )
        print(f"  ✓ {len(df_valido)} registros cargados en tabla 'ventas_cafeteria'")
        logging.info(f"Carga completada: {len(df_valido)} registros en PostgreSQL")

    except Exception as e:
        print(f"  ✗ Error al conectar a PostgreSQL: {e}")
        logging.error(f"Error en carga PostgreSQL: {e}")
        # Fallback: guardar CSV procesado localmente
        os.makedirs("data/processed", exist_ok=True)
        df_valido.to_csv("data/processed/ventas_limpias.csv", index=False)
        print("  → Fallback: datos guardados en data/processed/ventas_limpias.csv")
        logging.info("Fallback activado: datos guardados en CSV procesado")

    # ── Métricas finales ─────────────────────────────────────────────────────
    latencia = round(time.time() - inicio, 2)
    completitud = round(len(df_valido) / total_crudos * 100, 1)

    print("\n" + "=" * 60)
    print(f"Pipeline completado en {latencia}s")
    print(f"Completitud final: {completitud}%")
    print(f"Registros ingeridos: {total_crudos}")
    print(f"Registros cargados: {len(df_valido)}")
    print("=" * 60)

    logging.info(f"Pipeline completado en {latencia}s | Completitud: {completitud}%")
    return df_valido


if __name__ == "__main__":
    run_pipeline()
