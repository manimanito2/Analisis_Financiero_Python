# -*- coding: utf-8 -*-
"""
Modelo de Scoring de Crédito (Probabilidad de Impago)
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
    'edad': np.random.randint(18, 75, N),
    'ingreso_mensual': np.random.lognormal(mean=9.5, sigma=0.5, size=N).round(2),
    'antiguedad_laboral_meses': np.random.randint(0, 360, N),
    'num_creditos_activos': np.random.poisson(1.5, N),
    'historial_atrasos_12m': np.random.poisson(0.3, N),
    'monto_solicitado': np.random.lognormal(mean=9.0, sigma=0.7, size=N).round(2),
    'plazo_meses': np.random.choice([12, 24, 36, 48, 60], N),
    'tiene_vivienda_propia': np.random.choice([0, 1], N, p=[0.6, 0.4]),
    'razon_deuda_ingreso': np.round(np.random.beta(2, 5, N) * 1.2, 3),
})

# Variable objetivo: probabilidad de impago, construida con una lógica
# realista (no aleatoria pura) para que el modelo tenga señal real que aprender.
# Estandarizamos las variables continuas antes de combinarlas, así cada
# coeficiente tiene un efecto comparable y controlado sobre el resultado.
z_deuda_ingreso = (data['razon_deuda_ingreso'] - data['razon_deuda_ingreso'].mean()) / data['razon_deuda_ingreso'].std()
z_antiguedad = (data['antiguedad_laboral_meses'] - data['antiguedad_laboral_meses'].mean()) / data['antiguedad_laboral_meses'].std()
z_ingreso = (data['ingreso_mensual'] - data['ingreso_mensual'].mean()) / data['ingreso_mensual'].std()

logit = (
    -1.8
    + 0.55 * data['historial_atrasos_12m']
    + 1.6 * z_deuda_ingreso
    - 0.7 * data['tiene_vivienda_propia']
    - 0.9 * z_antiguedad
    + 0.35 * data['num_creditos_activos']
    - 0.5 * z_ingreso
    + np.random.normal(0, 0.4, N)  # ruido moderado, deja señal aprendible
)
prob_impago_real = 1 / (1 + np.exp(-logit))
data['impago'] = (np.random.uniform(0, 1, N) < prob_impago_real).astype(int)

print("=== Resumen del dataset ===")
print(f"Tasa de impago observada: {data['impago'].mean():.2%}")
print(data.describe().round(2))

# ============================================================
# 2. PREPARACIÓN PARA EL MODELO
# ============================================================
features = ['edad', 'ingreso_mensual', 'antiguedad_laboral_meses',
            'num_creditos_activos', 'historial_atrasos_12m', 'monto_solicitado',
            'plazo_meses', 'tiene_vivienda_propia', 'razon_deuda_ingreso']

X = data[features]
y = data['impago']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Tasa de impago en train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

# ============================================================
# 3. ENTRENAMIENTO DEL MODELO (XGBoost)
# ============================================================
# scale_pos_weight compensa el desbalance de clases (hay muchos menos
# impagos que no-impagos, como en la realidad)
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
print(classification_report(y_test, y_pred, target_names=['No impago', 'Impago']))

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
# 6. VISUALIZACIÓN
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='steelblue', linewidth=2)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Azar')
axes[0].set_title('Curva ROC')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Importancia de variables
axes[1].barh(importancias['variable'], importancias['importancia'], color='darkorange')
axes[1].set_title('Importancia de Variables (XGBoost)')
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3, axis='x')

# Distribución de score por clase real
axes[2].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='No impago (real)', color='green')
axes[2].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Impago (real)', color='red')
axes[2].set_title('Distribución del Score de Riesgo por Clase Real')
axes[2].set_xlabel('Probabilidad de impago predicha')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scoring_credito_resultados.png', dpi=150)
print("\nGráficas guardadas en: scoring_credito_resultados.png")
plt.show()
