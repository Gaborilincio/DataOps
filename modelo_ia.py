"""
modelo_ia.py — CafféData Ops
Entrenamiento del modelo de clasificación: predice si una venta será alta o baja.
Fase 3 — Evaluación Parcial N°3 | ITY1101 Gestión de Datos para IA

Ejecutar con: python modelo_ia.py
Los resultados se guardan en: output/modelo/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

warnings.filterwarnings("ignore")
os.makedirs("output/modelo", exist_ok=True)

RUTA_DATASET = "output/eda/dataset_modelo.csv"
RANDOM_STATE = 42


# ═════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y PARTICIÓN
# ═════════════════════════════════════════════════════════════════════════════

def cargar_y_partir():
    print("="*55)
    print("CafféData Ops — Entrenamiento Modelo IA")
    print("="*55)

    df = pd.read_csv(RUTA_DATASET)
    print(f"\n✓ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Separar features y variable objetivo
    columnas_trampa = [
        "venta_alta", 
        "Quantity", 
        "Price Per Unit"
    ]

    X = df.drop(columns=columnas_trampa)
    y = df["venta_alta"]

    # Partición 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n── Partición de datos ─────────────────────────")
    print(f"  Total registros : {len(df)}")
    print(f"  Entrenamiento   : {len(X_train)} ({len(X_train)/len(df)*100:.0f}%)")
    print(f"  Prueba          : {len(X_test)} ({len(X_test)/len(df)*100:.0f}%)")
    print(f"  Features        : {X.shape[1]}")
    print(f"  Clase 1 en train: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  Clase 1 en test : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO — DOS MODELOS PARA COMPARAR
# ═════════════════════════════════════════════════════════════════════════════

def entrenar_modelos(X_train, X_test, y_train, y_test):
    print(f"\n── Entrenamiento ──────────────────────────────")

    # Escalar para Regresión Logística
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Regresión Logística": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        if nombre == "Regresión Logística":
            modelo.fit(X_train_sc, y_train)
            y_pred       = modelo.predict(X_test_sc)
            y_prob       = modelo.predict_proba(X_test_sc)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred       = modelo.predict(X_test)
            y_prob       = modelo.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        gini = 2 * auc - 1
        cm   = confusion_matrix(y_test, y_pred)

        resultados[nombre] = {
            "modelo":    modelo,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "auc_roc":   round(auc,  4),
            "gini":      round(gini, 4),
            "confusion_matrix": cm,
            "scaler":    scaler if nombre == "Regresión Logística" else None,
        }

        print(f"\n  {nombre}:")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    AUC-ROC   : {auc:.4f}")
        print(f"    Gini      : {gini:.4f}")

    return resultados


# ═════════════════════════════════════════════════════════════════════════════
# 3. SELECCIÓN DEL MEJOR MODELO
# ═════════════════════════════════════════════════════════════════════════════

def seleccionar_mejor(resultados):
    mejor_nombre = max(resultados, key=lambda k: resultados[k]["f1"])
    mejor        = resultados[mejor_nombre]

    print(f"\n── Modelo seleccionado ────────────────────────")
    print(f"  → {mejor_nombre}")
    print(f"  Justificación: mayor F1 Score ({mejor['f1']:.4f})")
    print(f"  El F1 balancea precision y recall, ideal para")
    print(f"  datasets con posible desbalance de clases.")

    return mejor_nombre, mejor


# ═════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS DE MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def grafico_matriz_confusion(cm, nombre_modelo, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Baja (0)", "Alta (1)"],
        yticklabels=["Baja (0)", "Alta (1)"],
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Matriz de Confusión\n{nombre_modelo}", fontsize=13)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")

    # Anotar TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.15,
            f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            ha="center", transform=ax.transAxes, fontsize=9, color="gray")

    plt.tight_layout()
    ruta = "output/modelo/01_matriz_confusion.png"
    plt.savefig(ruta, dpi=150); plt.close()
    print(f"  ✓ {ruta}")


def grafico_curva_roc(resultados, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colores = ["#f59e0b", "#1f77b4"]

    for (nombre, res), color in zip(resultados.items(), colores):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{nombre} (AUC={res['auc_roc']:.3f}, Gini={res['gini']:.3f})")

    ax.plot([0,1], [0,1], "k--", linewidth=1, label="Clasificador aleatorio")
    ax.fill_between([0,1], [0,1], alpha=0.05, color="gray")
    ax.set_title("Curva ROC — Comparación de modelos", fontsize=13)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    ruta = "output/modelo/02_curva_roc.png"
    plt.savefig(ruta, dpi=150); plt.close()
    print(f"  ✓ {ruta}")


def grafico_comparacion_metricas(resultados):
    metricas   = ["accuracy", "precision", "recall", "f1", "auc_roc", "gini"]
    nombres    = list(resultados.keys())
    valores_rf = [resultados[nombres[0]][m] for m in metricas]
    valores_lr = [resultados[nombres[1]][m] for m in metricas]

    x     = np.arange(len(metricas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, valores_rf, width, label=nombres[0],
                color="#f59e0b", edgecolor="white")
    b2 = ax.bar(x + width/2, valores_lr, width, label=nombres[1],
                color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy","Precision","Recall","F1","AUC-ROC","Gini"])
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparación de métricas entre modelos", fontsize=13)
    ax.set_ylabel("Valor")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    ruta = "output/modelo/03_comparacion_metricas.png"
    plt.savefig(ruta, dpi=150); plt.close()
    print(f"  ✓ {ruta}")


def grafico_importancia_features(mejor_nombre, mejor, feature_names):
    if mejor_nombre != "Random Forest":
        print("  (Importancia de features solo disponible para Random Forest)")
        return

    importancias = mejor["modelo"].feature_importances_
    df_imp = pd.DataFrame({
        "feature":     feature_names,
        "importancia": importancias
    }).sort_values("importancia", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(df_imp["feature"], df_imp["importancia"],
                   color="#f59e0b", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Top 15 variables más importantes\nRandom Forest", fontsize=13)
    ax.set_xlabel("Importancia")

    plt.tight_layout()
    ruta = "output/modelo/04_importancia_features.png"
    plt.savefig(ruta, dpi=150); plt.close()
    print(f"  ✓ {ruta}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. GUARDAR REPORTE JSON
# ═════════════════════════════════════════════════════════════════════════════

def guardar_reporte(resultados, mejor_nombre):
    reporte = {
        "modelo_seleccionado": mejor_nombre,
        "justificacion": "Mayor F1 Score — balancea precision y recall",
        "metricas": {}
    }

    for nombre, res in resultados.items():
        cm = res["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()
        reporte["metricas"][nombre] = {
            "accuracy":         res["accuracy"],
            "precision":        res["precision"],
            "recall":           res["recall"],
            "f1_score":         res["f1"],
            "auc_roc":          res["auc_roc"],
            "gini":             res["gini"],
            "confusion_matrix": {
                "TN": int(tn), "FP": int(fp),
                "FN": int(fn), "TP": int(tp)
            }
        }

    ruta = "output/modelo/reporte_modelo.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Reporte guardado en {ruta}")

    return reporte


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Cargar y partir datos
    X_train, X_test, y_train, y_test, features = cargar_y_partir()

    # 2. Entrenar ambos modelos
    resultados = entrenar_modelos(X_train, X_test, y_train, y_test)

    # 3. Seleccionar el mejor
    mejor_nombre, mejor = seleccionar_mejor(resultados)

    # 4. Generar gráficos
    print(f"\n── Generando gráficos ─────────────────────────")
    grafico_matriz_confusion(mejor["confusion_matrix"], mejor_nombre, y_test)
    grafico_curva_roc(resultados, y_test)
    grafico_comparacion_metricas(resultados)
    grafico_importancia_features(mejor_nombre, mejor, features)

    # 5. Guardar reporte
    reporte = guardar_reporte(resultados, mejor_nombre)

    # Resumen final
    m = reporte["metricas"][mejor_nombre]
    print(f"\n{'='*55}")
    print(f"Modelo entrenado: {mejor_nombre}")
    print(f"  Accuracy  : {m['accuracy']}")
    print(f"  Precision : {m['precision']}")
    print(f"  Recall    : {m['recall']}")
    print(f"  F1 Score  : {m['f1_score']}")
    print(f"  AUC-ROC   : {m['auc_roc']}")
    print(f"  Gini      : {m['gini']}")
    print(f"  TP={m['confusion_matrix']['TP']}  TN={m['confusion_matrix']['TN']}  "
          f"FP={m['confusion_matrix']['FP']}  FN={m['confusion_matrix']['FN']}")
    print(f"{'='*55}")
    print(f"Archivos en output/modelo/:")
    print(f"  01_matriz_confusion.png")
    print(f"  02_curva_roc.png")
    print(f"  03_comparacion_metricas.png")
    print(f"  04_importancia_features.png")
    print(f"  reporte_modelo.json")