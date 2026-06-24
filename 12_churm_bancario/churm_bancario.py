# -*- coding: utf-8 -*-
"""
Predicción de Churn (Abandono de Clientes) en Banca/Fintech
Datos sintéticos generados con Faker, modelo con XGBoost.

Autor: Alejandro Ugarte Mendoza
"""

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
import xgboost as xgb
import matplotlib.pyplot as plt

np.random.seed(42)
fake = Faker('es_MX')
Faker.seed(42)

# ============================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================
N = 5000

data = pd.DataFrame({
    'cliente_id': [fake.uuid4()[:8] for _ in range(N)],
    'antiguedad_meses': np.random.randint(1, 96, N),
    'num_productos': np.random.randint(1, 6, N),
    'saldo_promedio': np.random.lognormal(mean=9.0, sigma=1.0, size=N).round(2),
    'num_transacciones_mes': np.random.poisson(8, N),
    'usa_app_movil': np.random.choice([0, 1], N, p=[0.35, 0.65]),
    'tickets_soporte_6m': np.random.poisson(0.8, N),
    'tiene_tarjeta_credito': np.random.choice([0, 1], N, p=[0.4, 0.6]),
    'edad': np.random.randint(18, 80, N),
    'cambio_saldo_3m_pct': np.round(np.random.normal(0, 0.15, N), 3),
})

# Variable objetivo: churn, construida con lógica de negocio realista.
# Estandarizamos variables continuas para que cada coeficiente tenga
# un efecto comparable.
z_antiguedad = (data['antiguedad_meses'] - data['antiguedad_meses'].mean()) / data['antiguedad_meses'].std()
z_transacciones = (data['num_transacciones_mes'] - data['num_transacciones_mes'].mean()) / data['num_transacciones_mes'].std()
z_cambio_saldo = (data['cambio_saldo_3m_pct'] - data['cambio_saldo_3m_pct'].mean()) / data['cambio_saldo_3m_pct'].std()

logit = (
    -1.5
    - 0.8 * z_antiguedad                      # clientes más antiguos, menos churn
    - 0.6 * z_transacciones                   # menos uso, más churn (signo invertido abajo)
    - 0.5 * data['usa_app_movil']             # usar la app reduce churn
    + 0.45 * data['tickets_soporte_6m']        # más quejas, más churn
    - 0.25 * data['num_productos']             # más productos, más "atado" al banco
    - 0.7 * z_cambio_saldo                     # saldo cayendo fuerte, más churn
    + np.random.normal(0, 0.4, N)
)
prob_churn_real = 1 / (1 + np.exp(-logit))
data['churn'] = (np.random.uniform(0, 1, N) < prob_churn_real).astype(int)

print("=== Resumen del dataset ===")
print(f"Tasa de churn observada: {data['churn'].mean():.2%}")
print(data.describe().round(2))

# ============================================================
# 2. PREPARACIÓN PARA EL MODELO
# ============================================================
features = ['antiguedad_meses', 'num_productos', 'saldo_promedio',
            'num_transacciones_mes', 'usa_app_movil', 'tickets_soporte_6m',
            'tiene_tarjeta_credito', 'edad', 'cambio_saldo_3m_pct']

X = data[features]
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Tasa de churn en train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

# ============================================================
# 3. ENTRENAMIENTO DEL MODELO (XGBoost)
# ============================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

modelo = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    random_state=42
)

modelo.fit(X_train, y_train)

# ============================================================
# 4. EVALUACIÓN
# ============================================================
y_pred_proba = modelo.predict_proba(X_test)[:, 1]
y_pred = modelo.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n=== Evaluación del modelo ===")
print(f"AUC-ROC: {auc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['No churn', 'Churn']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de confusión:\n{cm}")

# ============================================================
# 5. IMPORTANCIA DE VARIABLES
# ============================================================
importancias = pd.DataFrame({
    'variable': features,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False)

print("\n=== Importancia de variables ===")
print(importancias.to_string(index=False))

# ============================================================
# 6. SEGMENTACIÓN DE CLIENTES EN RIESGO
# ============================================================
data_test = X_test.copy()
data_test['prob_churn'] = y_pred_proba
data_test['churn_real'] = y_test.values

def clasificar_riesgo(p):
    if p >= 0.7:
        return 'Alto riesgo'
    elif p >= 0.4:
        return 'Riesgo medio'
    else:
        return 'Bajo riesgo'

data_test['segmento_riesgo'] = data_test['prob_churn'].apply(clasificar_riesgo)
print("\n=== Distribución de clientes por segmento de riesgo ===")
print(data_test['segmento_riesgo'].value_counts())

# ============================================================
# 7. VISUALIZACIÓN
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='steelblue', linewidth=2)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Azar')
axes[0].set_title('Curva ROC')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].barh(importancias['variable'], importancias['importancia'], color='darkorange')
axes[1].set_title('Importancia de Variables (XGBoost)')
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3, axis='x')

segmentos_count = data_test['segmento_riesgo'].value_counts()
colores_seg = {'Alto riesgo': 'red', 'Riesgo medio': 'orange', 'Bajo riesgo': 'green'}
axes[2].bar(segmentos_count.index, segmentos_count.values,
            color=[colores_seg[s] for s in segmentos_count.index])
axes[2].set_title('Clientes por Segmento de Riesgo de Churn')
axes[2].set_ylabel('Número de clientes')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('churn_resultados.png', dpi=150)
print("\nGráficas guardadas en: churn_resultados.png")
plt.show()
