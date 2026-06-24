# Detección de Fraude Transaccional

Modelo de clasificación para identificar transacciones fraudulentas en un conjunto de datos fuertemente desbalanceado (~1.5% de fraude), una situación típica en este tipo de problema.

## Cómo funciona

1. **Generación de datos sintéticos:** se simulan 20,000 transacciones (con `Faker`) con variables como monto, categoría de comercio, hora del día, distancia respecto al domicilio del cliente, si la transacción es internacional, número de transacciones en el día y antigüedad de la cuenta.
2. **Construcción de la variable objetivo:** el fraude se modela con una función logística que incorpora patrones conocidos en la literatura de detección de fraude: transacciones de madrugada, montos altos, lejos del domicilio habitual, internacionales, y en cuentas relativamente nuevas tienen mayor probabilidad de ser fraudulentas. Se calibra un umbral para obtener una tasa de fraude realista (~1.5%).
3. **Entrenamiento con XGBoost:** clasificador con `scale_pos_weight` muy alto (~66x) dado el fuerte desbalance de clases — sin este ajuste, el modelo tendería a predecir "no fraude" siempre y aun así tener accuracy alta, sin ser útil.
4. **Evaluación con métricas apropiadas para desbalance:** además de AUC-ROC, se usa la curva Precision-Recall, que es más informativa cuando la clase positiva es rara.

## Resultado obtenido

- **AUC-ROC: 0.99**
- **Recall de fraude: 0.80** (detecta 8 de cada 10 fraudes reales)
- **Precision de fraude: 0.55** (de cada 100 alertas emitidas, ~55 son fraude real y ~45 son falsas alarmas)
- Las variables más relevantes son distancia al domicilio, monto, hora de madrugada y si la transacción es internacional.

## Por qué importa

Un AUC alto puede ser engañoso en problemas desbalanceados — accuracy y AUC-ROC se ven bien incluso con modelos poco útiles, porque la clase mayoritaria domina la métrica. El verdadero trade-off de negocio está en el balance precision/recall: subir el umbral de decisión reduce falsas alarmas pero deja pasar más fraude, y viceversa. Este proyecto reporta explícitamente ambas métricas para reflejar esa decisión real que enfrentaría un equipo de riesgo.

## Stack técnico

`pandas`, `numpy`, `Faker`, `scikit-learn`, `xgboost`, `matplotlib`

## Cómo ejecutarlo

```bash
pip install pandas numpy faker scikit-learn xgboost matplotlib
python deteccion_fraude.py
```
