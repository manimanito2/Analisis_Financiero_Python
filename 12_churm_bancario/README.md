# Predicción de Churn (Abandono de Clientes) en Banca/Fintech

Modelo de clasificación que predice la probabilidad de que un cliente bancario abandone la institución, con segmentación de clientes por nivel de riesgo.

## Cómo funciona

1. **Generación de datos sintéticos:** se simulan ~5,000 clientes (con `Faker`) con variables de comportamiento: antigüedad, número de productos, saldo promedio, transacciones mensuales, uso de app móvil, tickets de soporte, tenencia de tarjeta de crédito y variación reciente del saldo.
2. **Construcción de la variable objetivo:** el churn se modela con una función logística donde la antigüedad y el uso de la app reducen el riesgo, mientras que los tickets de soporte y una caída fuerte de saldo lo aumentan — replicando patrones de comportamiento conocidos en retención de clientes bancarios.
3. **Entrenamiento con XGBoost:** clasificador con ajuste por desbalance de clases.
4. **Evaluación:** AUC-ROC, matriz de confusión, reporte de clasificación.
5. **Segmentación de riesgo:** los clientes del set de prueba se agrupan en "Alto riesgo", "Riesgo medio" y "Bajo riesgo" según su probabilidad predicha de churn — el tipo de output que un equipo de retención usaría directamente para priorizar a quién contactar primero.

## Resultado obtenido

- **AUC-ROC: 0.77**
- Las variables más relevantes son antigüedad, número de transacciones mensuales y variación reciente del saldo — consistente con la intuición de negocio de que un cliente que deja de usar sus productos es la señal de alerta más temprana.

## Por qué importa

Predecir churn no es solo clasificar "se va / no se va" — el valor real está en la segmentación de riesgo, que permite a un equipo de retención enfocar recursos limitados (llamadas, ofertas, descuentos) en los clientes con mayor probabilidad de irse, en lugar de tratar a toda la base por igual.

## Stack técnico

`pandas`, `numpy`, `Faker`, `scikit-learn`, `xgboost`, `matplotlib`

## Cómo ejecutarlo

```bash
pip install pandas numpy faker scikit-learn xgboost matplotlib
python churn_bancario.py
```
