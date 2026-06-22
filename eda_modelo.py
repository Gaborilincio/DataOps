"""
eda_modelo.py — CafféData Ops
Prepara y analiza los datos para entrenar el modelo de IA.
Ejecutar con: python eda_modelo.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cafedataops")
DB_USER = os.getenv("DB_USER", "cafe_user")
DB_PASS = os.getenv("DB_PASSWORD", "cafe_pass")

os.makedirs("output/eda", exist_ok=True)


def cargar_datos():
    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        df = pd.read_sql("SELECT * FROM ventas_cafeteria", engine)
        print(f"✓ Datos cargados desde PostgreSQL ({len(df)} registros)")
    except Exception:
        fallback = "data/processed/ventas_limpias.csv"
        df = pd.read_csv(fallback, parse_dates=["Transaction Date"])
        print(f"✓ Datos cargados desde CSV fallback ({len(df)} registros)")

    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Hour"]      = df["Transaction Date"].dt.hour
    df["Month"]     = df["Transaction Date"].dt.month
    df["DayOfWeek"] = df["Transaction Date"].dt.day_name()
    return df


def crear_variable_objetivo(df):
    umbral = df["Total Spent"].median()
    df["venta_alta"] = (df["Total Spent"] > umbral).astype(int)
    print(f"\n── Variable objetivo ──────────────────────────")
    print(f"Umbral (mediana): ${umbral:.2f}")
    print(f"Clase 1 (alta) : {df['venta_alta'].sum()} ({df['venta_alta'].mean()*100:.1f}%)")
    print(f"Clase 0 (baja) : {(df['venta_alta']==0).sum()} ({(1-df['venta_alta'].mean())*100:.1f}%)")
    return df, umbral


def estadisticas(df):
    print(f"\n── Estadísticas numéricas ─────────────────────")
    cols = ["Total Spent", "Quantity", "Price Per Unit"]
    print(df[cols].describe().round(2).to_string())

    print(f"\n── Frecuencias categóricas ────────────────────")
    for col in ["Item", "Payment Method", "Location"]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())

    print(f"\n── Correlación con venta_alta ─────────────────")
    cols_num = ["Quantity", "Price Per Unit", "Total Spent", "Hour", "Month"]
    corr = df[cols_num + ["venta_alta"]].corr()["venta_alta"].drop("venta_alta").round(3)
    print(corr.sort_values(ascending=False).to_string())


def graficos(df, umbral):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("CafféData Ops — EDA para modelo IA", fontsize=14, fontweight="bold")

    # 1. Distribución Total Spent
    axes[0,0].hist(df["Total Spent"], bins=25, color="#f59e0b", edgecolor="white")
    axes[0,0].axvline(umbral, color="red", linestyle="--", label=f"Umbral ${umbral:.2f}")
    axes[0,0].set_title("Distribución Total Spent")
    axes[0,0].set_xlabel("Total Spent ($)")
    axes[0,0].legend(fontsize=8)

    # 2. Balance de clases
    conteo = df["venta_alta"].value_counts().sort_index()
    bars = axes[0,1].bar(["Baja (0)", "Alta (1)"], conteo.values,
                          color=["#e74c3c", "#2ecc71"], edgecolor="white")
    axes[0,1].bar_label(bars, padding=3)
    axes[0,1].set_title("Balance de clases")
    axes[0,1].set_ylabel("Registros")

    # 3. % ventas altas por producto
    prop = df.groupby("Item")["venta_alta"].mean().sort_values(ascending=False)
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in prop.values]
    axes[0,2].bar(prop.index, prop.values * 100, color=colors, edgecolor="white")
    axes[0,2].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[0,2].set_title("% ventas altas por producto")
    axes[0,2].set_ylabel("%")
    axes[0,2].tick_params(axis="x", rotation=30)

    # 4. Matriz de correlación
    cols_num = ["Quantity", "Price Per Unit", "Total Spent", "Hour", "Month", "venta_alta"]
    corr = df[cols_num].corr().round(2)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=axes[1,0], cbar=False)
    axes[1,0].set_title("Matriz de correlación")

    # 5. Total Spent por producto (boxplot)
    items = df["Item"].unique()
    data  = [df[df["Item"] == i]["Total Spent"].values for i in items]
    axes[1,1].boxplot(data, labels=items, patch_artist=True)
    axes[1,1].set_title("Total Spent por producto")
    axes[1,1].tick_params(axis="x", rotation=30)

    # 6. Ventas por día de semana
    orden = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    freq  = df["DayOfWeek"].value_counts().reindex(orden).fillna(0)
    axes[1,2].bar(freq.index, freq.values, color="#f59e0b", edgecolor="white")
    axes[1,2].set_title("Ventas por día")
    axes[1,2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    ruta = "output/eda/eda_completo.png"
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n✓ Gráfico guardado en {ruta}")


def guardar_dataset_modelo(df):
    df_modelo = df[[
        "Item", "Quantity", "Price Per Unit",
        "Payment Method", "Location",
        "Hour", "Month", "DayOfWeek",
        "venta_alta"
    ]].copy()

    df_modelo = pd.get_dummies(
        df_modelo,
        columns=["Item", "Payment Method", "Location", "DayOfWeek"],
        drop_first=False
    )

    ruta = "output/eda/dataset_modelo.csv"
    df_modelo.to_csv(ruta, index=False)
    print(f"✓ Dataset para modelo guardado en {ruta}")
    print(f"  Shape: {df_modelo.shape[0]} filas x {df_modelo.shape[1]} columnas")
    return df_modelo


if __name__ == "__main__":
    print("="*50)
    print("CafféData Ops — EDA para modelo IA")
    print("="*50)

    df            = cargar_datos()
    df, umbral    = crear_variable_objetivo(df)
    estadisticas(df)
    graficos(df, umbral)
    df_modelo     = guardar_dataset_modelo(df)

    print("\n" + "="*50)
    print("EDA completado")
    print(f"Registros listos para entrenar: {len(df_modelo)}")
    print(f"Umbral venta alta: ${umbral:.2f}")
    print("="*50)