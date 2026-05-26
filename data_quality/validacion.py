import pandas as pd
import logging
import json
import os
import time

UMBRAL_COMPLETITUD = 0.90  # 90% mínimo aceptable
UMBRAL_OUTLIER_STD = 3.0   # desviaciones estándar máximas permitidas


def ejecutar_validaciones(df: pd.DataFrame, total_crudos: int) -> pd.DataFrame:
    """
    Quality Gate: valida tipos, reglas semánticas, consistencia matemática
    y outliers estadísticos. Registros inválidos van a logs/rechazados.csv
    para auditoría. Genera logs/reporte_calidad.json con métricas completas.
    """
    df = df.copy()
    rechazados = []
    metricas = {"total_crudos": total_crudos, "validaciones": {}}
    inicio = time.time()

    print(f"\n[VALIDACIÓN] Evaluando {len(df)} registros...")

    # ── 1. Validación de tipos de datos ─────────────────────────────────────
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price Per Unit"] = pd.to_numeric(df["Price Per Unit"], errors="coerce")
    df["Total Spent"] = pd.to_numeric(df["Total Spent"], errors="coerce")

    mask_tipo = df[["Quantity", "Price Per Unit", "Total Spent"]].isna().any(axis=1)
    rechazados.append(df[mask_tipo].copy().assign(motivo_rechazo="Tipo de dato inválido"))
    df = df[~mask_tipo]
    metricas["validaciones"]["tipo_invalido"] = int(mask_tipo.sum())
    print(f"  ✓ Rechazados por tipo inválido: {mask_tipo.sum()}")

    # ── 2. Regla semántica: valores positivos ────────────────────────────────
    mask_neg = (df["Quantity"] <= 0) | (df["Price Per Unit"] <= 0)
    rechazados.append(df[mask_neg].copy().assign(motivo_rechazo="Valor negativo o cero"))
    df = df[~mask_neg]
    metricas["validaciones"]["valores_negativos"] = int(mask_neg.sum())
    print(f"  ✓ Rechazados por valores negativos/cero: {mask_neg.sum()}")

    # ── 3. Consistencia matemática: Total ≈ Qty × Price ──────────────────────
    total_esperado = (df["Quantity"] * df["Price Per Unit"]).round(2)
    total_real = df["Total Spent"].round(2)
    mask_inconsistente = (total_esperado - total_real).abs() > 0.05
    rechazados.append(df[mask_inconsistente].copy().assign(motivo_rechazo="Inconsistencia matemática"))
    df = df[~mask_inconsistente]
    metricas["validaciones"]["inconsistencia_matematica"] = int(mask_inconsistente.sum())
    print(f"  ✓ Rechazados por inconsistencia matemática: {mask_inconsistente.sum()}")

    # ── 4. Detección de outliers estadísticos en Total Spent ─────────────────
    media = df["Total Spent"].mean()
    std = df["Total Spent"].std()
    limite_superior = media + UMBRAL_OUTLIER_STD * std
    limite_inferior = media - UMBRAL_OUTLIER_STD * std
    mask_outlier = (df["Total Spent"] > limite_superior) | (df["Total Spent"] < limite_inferior)
    rechazados.append(df[mask_outlier].copy().assign(motivo_rechazo=f"Outlier estadístico (±{UMBRAL_OUTLIER_STD}σ)"))
    df = df[~mask_outlier]
    metricas["validaciones"]["outliers_estadisticos"] = int(mask_outlier.sum())
    metricas["validaciones"]["outlier_limite_superior"] = round(limite_superior, 2)
    metricas["validaciones"]["outlier_limite_inferior"] = round(limite_inferior, 2)
    print(f"  ✓ Rechazados por outlier estadístico (±{UMBRAL_OUTLIER_STD}σ): {mask_outlier.sum()}")
    print(f"    Rango aceptado: ${limite_inferior:.2f} — ${limite_superior:.2f}")
    logging.info(f"Outliers detectados: {mask_outlier.sum()} (rango ±{UMBRAL_OUTLIER_STD}σ: ${limite_inferior:.2f}–${limite_superior:.2f})")

    # ── 5. Guardar rechazados para auditoría ─────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    df_rechazados = pd.concat(rechazados, ignore_index=True)
    total_rechazados = len(df_rechazados)
    if not df_rechazados.empty:
        df_rechazados.to_csv("logs/rechazados.csv", index=False)
        logging.warning(f"Registros rechazados: {total_rechazados} → logs/rechazados.csv")

    # ── 6. KPI de Completitud ────────────────────────────────────────────────
    completitud = len(df) / total_crudos
    print(f"\nKPI Completitud: {completitud*100:.1f}% ({len(df)}/{total_crudos})")
    logging.info(f"KPI Completitud: {completitud*100:.1f}%")

    if completitud < UMBRAL_COMPLETITUD:
        msg = f"ALERTA: Completitud {completitud*100:.1f}% < umbral {UMBRAL_COMPLETITUD*100}%"
        logging.critical(msg)
        print(f"{msg}")
    else:
        print(f"Completitud dentro del umbral aceptable (≥ {UMBRAL_COMPLETITUD*100:.0f}%)")

    # ── 7. Generar reporte de calidad JSON ───────────────────────────────────
    metricas["registros_validos"] = len(df)
    metricas["registros_rechazados"] = total_rechazados
    metricas["completitud_pct"] = round(completitud * 100, 1)
    metricas["umbral_completitud_pct"] = UMBRAL_COMPLETITUD * 100
    metricas["supera_umbral"] = completitud >= UMBRAL_COMPLETITUD
    metricas["latencia_validacion_seg"] = round(time.time() - inicio, 3)

    reporte_path = "logs/reporte_calidad.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"\nReporte de calidad guardado en {reporte_path}")
    logging.info(f"Reporte de calidad generado: {reporte_path}")

    print(f"\n[VALIDACIÓN] Completada. Registros válidos: {len(df)}")
    logging.info(f"Validación completada: {len(df)} registros válidos")
    return df