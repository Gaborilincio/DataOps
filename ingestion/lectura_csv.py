import pandas as pd
import logging
import os

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Columnas que el CSV de entrada DEBE tener
COLUMNAS_ESPERADAS = {
    "Transaction ID",
    "Item",
    "Quantity",
    "Price Per Unit",
    "Total Spent",
    "Payment Method",
    "Location",
    "Transaction Date",
}


def leer_datos_csv(ruta: str = "data/raw/dirty_cafe_sales.csv") -> pd.DataFrame:
    """
    Ingesta del archivo plano CSV.
    """
    if not os.path.exists(ruta):
        logging.error(f"Archivo no encontrado: {ruta}")
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    df = pd.read_csv(ruta, dtype=str)  

    # ── Validación de schema ─────────────────────────────────────────────────
    columnas_presentes = set(df.columns)
    columnas_faltantes = COLUMNAS_ESPERADAS - columnas_presentes
    columnas_extra = columnas_presentes - COLUMNAS_ESPERADAS

    if columnas_faltantes:
        msg = f"Schema inválido — columnas faltantes: {columnas_faltantes}"
        logging.critical(msg)
        raise ValueError(msg)

    if columnas_extra:
        logging.warning(f"Columnas extra ignoradas en el CSV: {columnas_extra}")
        print(f"Columnas extra detectadas (se ignorarán): {columnas_extra}")

    print(f"Schema validado: todas las columnas requeridas presentes")
    logging.info("Schema validado correctamente")

    total = len(df)
    logging.info(f"Ingesta completada: {total} registros crudos leídos desde '{ruta}'")
    print(f"[INGESTA] {total} registros leídos desde '{ruta}'")

    return df