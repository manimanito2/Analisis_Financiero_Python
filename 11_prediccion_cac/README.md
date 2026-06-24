# Predicción del Costo de Adquisición de Clientes (CAC) por Campaña

Modelo de regresión que predice el CAC esperado de una campaña de marketing digital, a partir de sus características de configuración (canal, presupuesto, duración, segmento, métricas de desempeño).

> **Nota de alcance:** este proyecto es de analítica de marketing/growth, no de mercados financieros. Se documenta aquí junto a los proyectos cuantitativos por usar el mismo stack técnico (Python + XGBoost + datos sintéticos), pero conceptualmente pertenece a un dominio distinto.

## Cómo funciona

1. **Generación de datos sintéticos:** se simulan 3,000 campañas (con `Faker`) sobre 6 canales (Google Ads, Meta Ads, TikTok Ads, Email, Referidos, SEO Orgánico), con presupuesto diario, duración, CTR, tasa de conversión, índice de competencia y día de lanzamiento.
2. **Construcción de la variable objetivo:** el CAC se deriva de un modelo de negocio explícito — gasto total dividido entre clientes estimados (a partir de clics y tasa de conversión), ajustado por un índice de competencia del canal y un componente de ruido multiplicativo. Cada canal tiene un costo por clic base distinto, reflejando que estructuralmente unos canales son más caros que otros.
3. **Entrenamiento con XGBoost (regresión):** a diferencia de los otros 3 proyectos (clasificación), aquí el objetivo es un valor continuo en dinero.
4. **Evaluación:** MAE, RMSE, MAPE y R², junto con un gráfico de valores reales vs. predichos.

## Resultado obtenido

- **R²: 0.89**
- **MAPE: ~30%**
- El CAC promedio por canal sigue un orden coherente con la realidad del marketing digital: Email y SEO Orgánico son los más baratos, Google Ads el más caro.
- Las variables más relevantes son el canal utilizado, el costo por clic base y la tasa de conversión.

## Por qué importa

Predecir el CAC esperado de una campaña antes de lanzarla (o con datos parciales en sus primeros días) permite a un equipo de growth decidir si vale la pena escalar el presupuesto o pausarla, en lugar de esperar a que termine para evaluar su eficiencia.

## Stack técnico

`pandas`, `numpy`, `Faker`, `scikit-learn`, `xgboost`, `matplotlib`

## Cómo ejecutarlo

```bash
pip install pandas numpy faker scikit-learn xgboost matplotlib
python prediccion_cac.py
```
