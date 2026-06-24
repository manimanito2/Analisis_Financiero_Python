# Modelo de Scoring de Crédito (Probabilidad de Impago)

Modelo de clasificación que estima la probabilidad de que un solicitante de crédito incurra en impago, usando XGBoost sobre variables financieras y de comportamiento.

## Cómo funciona

1. **Generación de datos sintéticos:** se simulan ~5,000 solicitantes de crédito (con `Faker`) con variables como edad, ingreso mensual, antigüedad laboral, número de créditos activos, historial de atrasos, monto solicitado, plazo, vivienda propia y razón deuda/ingreso.
2. **Construcción de la variable objetivo:** el impago no se asigna al azar — se construye con una función logística que combina las variables de forma realista (más razón deuda/ingreso y atrasos previos aumentan el riesgo; más antigüedad laboral y vivienda propia lo reducen), con un componente de ruido aleatorio para que no sea perfectamente determinista.
3. **Entrenamiento con XGBoost:** se entrena un clasificador con `scale_pos_weight` ajustado para compensar el desbalance natural entre clientes que pagan y los que no.
4. **Evaluación:** AUC-ROC, matriz de confusión, reporte de precisión/recall, y curva ROC.
5. **Importancia de variables:** identifica qué factores pesan más en la predicción del modelo.

## Resultado obtenido

- **AUC-ROC: 0.86**
- La variable más importante es la razón deuda/ingreso, seguida de antigüedad laboral y tenencia de vivienda propia — coherente con cómo funciona el scoring de crédito en la práctica.

## Por qué importa

El scoring de crédito es una de las aplicaciones más extendidas de machine learning en banca. Este proyecto no solo entrena un modelo, sino que documenta el proceso de calibrar la señal de los datos sintéticos (ver nota abajo) para que el resultado sea representativo de un problema real, en lugar de ruido puro.

## Nota sobre los datos sintéticos

La primera versión de la variable objetivo usaba demasiado ruido aleatorio, lo que resultaba en un AUC mediocre (~0.65) — un modelo casi inútil. Se corrigió estandarizando las variables continuas antes de combinarlas en la función logística (para que cada una tuviera un efecto comparable) y reduciendo el ruido relativo, lo que llevó el AUC a 0.86. Esto ilustra un punto importante al trabajar con datos sintéticos: la señal que se "inyecta" en los datos debe ser deliberada y calibrada, no solo plausible a simple vista.

## Stack técnico

`pandas`, `numpy`, `Faker`, `scikit-learn`, `xgboost`, `matplotlib`

## Cómo ejecutarlo

```bash
pip install pandas numpy faker scikit-learn xgboost matplotlib
python scoring_credito.py
```
