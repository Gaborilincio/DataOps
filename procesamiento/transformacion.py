import pandas as pd
import logging

def generar_transformaciones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y transformación del DataFrame crudo.
    Aplica correcciones sobre anomalías conocidas del dataset de ventas.
    """
    df = df.copy()
    total_inicial = len(df)
    print(f"\n[TRANSFORMACIÓN] Iniciando con {total_inicial} registros...")

    # 1. Eliminar duplicados exactos
    antes = len(df)
    df.drop_duplicates(inplace=True)
    duplicados = antes - len(df)
    print(f"  Duplicados eliminados: {duplicados}")
    logging.info(f"Duplicados eliminados: {duplicados}")

    # 2. Castear tipos numericos base
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")

    # 3. Recalcular Total Spent donde dice ERROR
    total_series = df["Total Spent"].astype(str).str.strip().str.upper()
    mask_error = total_series == "ERROR"
    recuperados = int(mask_error.sum())

    # Convertir columna a object para poder mezclar tipos
    df["Total Spent"] = df["Total Spent"].astype(object)
    for idx in df[mask_error].index:
        qty = df.at[idx, "Quantity"]
        price = df.at[idx, "Price Per Unit"]
        if pd.notna(qty) and pd.notna(price):
            df.at[idx, "Total Spent"] = round(float(qty) * float(price), 2)

    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")
    print(f"  Total Spent recalculados (eran ERROR): {recuperados}")
    logging.info(f"Total Spent recalculados: {recuperados}")

    # 4. Eliminar filas sin Transaction ID o Item
    antes = len(df)
    df.dropna(subset=["Transaction ID", "Item"], inplace=True)
    huerfanos = antes - len(df)
    print(f"  Registros huerfanos eliminados (sin ID o Item): {huerfanos}")
    logging.info(f"Registros huerfanos eliminados: {huerfanos}")

    # 5. Normalizar texto: strip + title case
    for col in ["Item", "Payment Method", "Location"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 6. Imputar UNKNOWN / ERROR en Payment Method y Location
    for col in ["Payment Method", "Location"]:
        mascara = df[col].str.upper().isin(["UNKNOWN", "ERROR", "NAN", ""])
        imputados = int(mascara.sum())
        df.loc[mascara, col] = "No Especificado"
        print(f"  '{col}' imputados como 'No Especificado': {imputados}")

    # 7. Estandarizar Transaction Date a ISO 8601
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    fechas_corruptas = int(df["Transaction Date"].isna().sum())
    df.dropna(subset=["Transaction Date"], inplace=True)
    print(f"  Fechas corruptas descartadas: {fechas_corruptas}")
    logging.info(f"Fechas corruptas descartadas: {fechas_corruptas}")

    # 8. Columnas de tiempo para dashboard
    df["Hour"] = df["Transaction Date"].dt.hour
    df["Month"] = df["Transaction Date"].dt.month
    df["DayOfWeek"] = df["Transaction Date"].dt.day_name()

    print(f"\n[TRANSFORMACION] Completada. Registros resultantes: {len(df)}")
    logging.info(f"Transformacion completada: {len(df)} registros limpios")
    return df
