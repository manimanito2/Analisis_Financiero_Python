# -*- coding: utf-8 -*-
"""
Predicción del Costo de Adquisición de Clientes (CAC) por Campaña
Datos sintéticos generados con Faker, modelo de regresión con XGBoost.

Autor: Alejandro Ugarte Mendoza
"""

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

np.random.seed(42)
fake = Faker('es_MX')
Faker.seed(42)

# ============================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================
# A diferencia de los proyectos anteriores (clasificación), aquí el
# objetivo es un número continuo: el CAC de cada campaña.
N = 3000

canales = ['Google Ads', 'Meta Ads', 'TikTok Ads', 'Email', 'Referidos', 'SEO Orgánico']
segmentos = ['18-24', '25-34', '35-44', '45-54', '55+']

data = pd.DataFrame({
    'campana_id': [fake.uuid4()[:8] for _ in range(N)],
    'canal': np.random.choice(canales, N),
    'segmento_edad': np.random.choice(segmentos, N),
    'presupuesto_diario': np.round(np.random.lognormal(mean=6.0, sigma=0.8, size=N), 2),
    'duracion_dias': np.random.randint(3, 60, N),
    'ctr_pct': np.round(np.random.beta(2, 30, N) * 100, 3),       # click-through rate
    'tasa_conversion_pct': np.round(np.random.beta(2, 50, N) * 100, 3),
    'competencia_indice': np.round(np.random.uniform(0.2, 1.0, N), 3),  # qué tan saturado está el canal
    'dia_semana_lanzamiento': np.random.choice(
        ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'], N),
})

# Costo por clic base, distinto por canal (refleja realidad: algunos
# canales son estructuralmente más caros que otros)
costo_clic_base = {
    'Google Ads': 8.5, 'Meta Ads': 5.0, 'TikTok Ads': 3.5,
    'Email': 0.5, 'Referidos': 1.0, 'SEO Orgánico': 0.8
}
data['costo_clic_base'] = data['canal'].map(costo_clic_base)

# === Variable objetivo: CAC ===
# Lógica: el CAC sube con costo por clic y competencia, y baja con
# mejor CTR y tasa de conversión (campaña más eficiente = más clientes
# por el mismo gasto = menor CAC).
clics_estimados = data['presupuesto_diario'] * data['duracion_dias'] / data['costo_clic_base']
clientes_estimados = clics_estimados * (data['tasa_conversion_pct'] / 100)
clientes_estimados = clientes_estimados.clip(lower=1)  # evitar división por cero

gasto_total = data['presupuesto_diario'] * data['duracion_dias']

ruido_multiplicativo = np.random.lognormal(mean=0, sigma=0.25, size=N)
ajuste_competencia = 1 + 0.6 * data['competencia_indice']

data['cac'] = np.round((gasto_total / clientes_estimados) * ajuste_competencia * ruido_multiplicativo, 2)

# Recortamos outliers extremos (campañas con casi cero conversión)
# que distorsionarían el modelo sin aportar señal real de negocio
cac_p99 = data['cac'].quantile(0.99)
data = data[data['cac'] <= cac_p99].reset_index(drop=True)

print("=== Resumen del dataset ===")
print(f"CAC promedio: ${data['cac'].mean():,.2f}")
print(f"CAC mediana: ${data['cac'].median():,.2f}")
print(f"\nCAC promedio por canal:")
print(data.groupby('canal')['cac'].mean().sort_values().round(2))

# ============================================================
# 2. PREPARACIÓN PARA EL MODELO
# ============================================================
data_encoded = pd.get_dummies(
    data, columns=['canal', 'segmento_edad', 'dia_semana_lanzamiento'], drop_first=True
)

features = [c for c in data_encoded.columns if c not in ['campana_id', 'cac']]

X = data_encoded[features]
y = data_encoded['cac']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# 3. ENTRENAMIENTO DEL MODELO (XGBoost Regressor)
# ============================================================
modelo = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

modelo.fit(X_train, y_train)

# ============================================================
# 4. EVALUACIÓN
# ============================================================
y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n=== Evaluación del modelo ===")
print(f"MAE  (Error Absoluto Medio): ${mae:,.2f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): ${rmse:,.2f}")
print(f"MAPE (Error Porcentual Absoluto Medio): {mape:.2f}%")
print(f"R²: {r2:.4f}")

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

axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([0, max_val], [0, max_val], linestyle='--', color='red', label='Predicción perfecta')
axes[0].set_title(f'CAC Real vs. Predicho (R² = {r2:.3f})')
axes[0].set_xlabel('CAC Real ($)')
axes[0].set_ylabel('CAC Predicho ($)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].barh(importancias['variable'][::-1], importancias['importancia'][::-1], color='darkorange')
axes[1].set_title('Top 10 Variables Más Importantes')
axes[1].grid(alpha=0.3, axis='x')

cac_por_canal = data.groupby('canal')['cac'].mean().sort_values()
axes[2].barh(cac_por_canal.index, cac_por_canal.values, color='seagreen')
axes[2].set_title('CAC Promedio por Canal')
axes[2].set_xlabel('CAC promedio ($)')
axes[2].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('cac_resultados.png', dpi=150)
print("\nGráficas guardadas en: cac_resultados.png")
plt.show()
