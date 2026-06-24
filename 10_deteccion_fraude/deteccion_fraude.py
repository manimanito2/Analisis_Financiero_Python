# -*- coding: utf-8 -*-
"""
Detección de Fraude Transaccional
Datos sintéticos generados con Faker, modelo con XGBoost.

Autor: Alejandro Ugarte Mendoza
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve)
import xgboost as xgb
import matplotlib.pyplot as plt

np.random.seed(42)
fake = Faker('es_MX')
Faker.seed(42)

# ============================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================
# El fraude es un problema MUY desbalanceado en la realidad (a menudo
# <1% de las transacciones). Lo simulamos así a propósito.
N = 20000
TASA_FRAUDE_OBJETIVO = 0.015  # ~1.5%, realista para este tipo de problema

categorias_comercio = ['Supermercado', 'Restaurante', 'Gasolina', 'Entretenimiento',
                        'Electrónica', 'Ropa', 'Servicios', 'Joyería', 'Viajes', 'Online']

fecha_inicio = datetime(2025, 1, 1)
horas = np.random.randint(0, 24, N)
fechas = [fecha_inicio + timedelta(days=int(d), hours=int(h))
          for d, h in zip(np.random.randint(0, 180, N), horas)]

data = pd.DataFrame({
    'transaccion_id': [fake.uuid4()[:10] for _ in range(N)],
    'fecha': fechas,
    'hora': horas,
    'monto': np.round(np.random.lognormal(mean=4.5, sigma=1.2, size=N), 2),
    'categoria_comercio': np.random.choice(categorias_comercio, N),
    'distancia_casa_km': np.round(np.abs(np.random.exponential(15, N)), 2),
    'num_transacciones_dia': np.random.poisson(2.5, N),
    'es_internacional': np.random.choice([0, 1], N, p=[0.92, 0.08]),
    'antiguedad_cuenta_dias': np.random.randint(1, 3650, N),
})

# Variable objetivo: fraude. Construido con lógica de negocio: el fraude
# tiende a ocurrir de madrugada, en montos altos, lejos de casa, en
# transacciones internacionales, y en cuentas nuevas.
es_madrugada = ((data['hora'] >= 0) & (data['hora'] <= 5)).astype(int)
z_monto = (np.log1p(data['monto']) - np.log1p(data['monto']).mean()) / np.log1p(data['monto']).std()
z_distancia = (data['distancia_casa_km'] - data['distancia_casa_km'].mean()) / data['distancia_casa_km'].std()
z_antiguedad = (data['antiguedad_cuenta_dias'] - data['antiguedad_cuenta_dias'].mean()) / data['antiguedad_cuenta_dias'].std()

logit = (
    -5.5
    + 1.8 * es_madrugada
    + 1.1 * z_monto
    + 0.9 * z_distancia
    + 1.6 * data['es_internacional']
    - 0.8 * z_antiguedad
    + 0.3 * data['num_transacciones_dia']
    + np.random.normal(0, 0.6, N)
)
prob_fraude_real = 1 / (1 + np.exp(-logit))

# Ajustamos el umbral para acercarnos a la tasa de fraude objetivo
umbral = np.percentile(prob_fraude_real, 100 * (1 - TASA_FRAUDE_OBJETIVO))
data['fraude'] = (prob_fraude_real >= umbral).astype(int)

print("=== Resumen del dataset ===")
print(f"Tasa de fraude observada: {data['fraude'].mean():.3%}")
print(f"Transacciones fraudulentas: {data['fraude'].sum()} de {N}")

# ============================================================
# 2. PREPARACIÓN PARA EL MODELO
# ============================================================
data['es_madrugada'] = es_madrugada
data_encoded = pd.get_dummies(data, columns=['categoria_comercio'], drop_first=True)

features = [c for c in data_encoded.columns if c not in
            ['transaccion_id', 'fecha', 'fraude']]

X = data_encoded[features]
y = data_encoded['fraude']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Fraudes en train: {y_train.sum()} | test: {y_test.sum()}")

# ============================================================
# 3. ENTRENAMIENTO DEL MODELO (XGBoost)
# ============================================================
# Con un desbalance tan fuerte, scale_pos_weight es crítico para que
# el modelo no colapse a "predecir siempre no-fraude".
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

modelo = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',  # AUC de precision-recall, más informativo en datos desbalanceados
    random_state=42
)

modelo.fit(X_train, y_train)

# ============================================================
# 4. EVALUACIÓN
# ============================================================
y_pred_proba = modelo.predict_proba(X_test)[:, 1]
y_pred = modelo.predict(X_test)

auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"\n=== Evaluación del modelo ===")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude'], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print(f"\nMatriz de confusión:\n{cm}")
print("(filas: real, columnas: predicho — [0,0]=legítima OK, [1,1]=fraude detectado)")

# ============================================================
# 5. IMPORTANCIA DE VARIABLES
# ============================================================
importancias = pd.DataFrame({
    'variable': features,
    'importancia': modelo.feature_importances_
}).sort_values('importancia', ascending=False).head(10)

print("\n=== Top 10 variables más importantes ===")
print(importancias.to_string(index=False))

# ============================================================
# 6. VISUALIZACIÓN
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.3f}', color='steelblue', linewidth=2)
axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Azar')
axes[0].set_title('Curva ROC')
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].legend()
axes[0].grid(alpha=0.3)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(recall, precision, color='darkorange', linewidth=2)
axes[1].set_title('Curva Precision-Recall\n(más informativa que ROC en datos desbalanceados)')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].grid(alpha=0.3)

axes[2].barh(importancias['variable'][::-1], importancias['importancia'][::-1], color='crimson')
axes[2].set_title('Top 10 Variables Más Importantes')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('fraude_resultados.png', dpi=150)
print("\nGráficas guardadas en: fraude_resultados.png")
plt.show()
