# Indicador Propio de Anticipación de Crisis Financieras

Sistema que combina varias técnicas de series de tiempo para etiquetar regímenes de mercado (laminar / turbulento / recuperación) y construir un indicador propio — el **"número de Reynolds financiero"** — inspirado en la mecánica de fluidos, diseñado para anticipar episodios de crisis antes de que se vuelvan evidentes en el precio.

## Cómo funciona

1. **Descarga y resampleo:** datos semanales de S&P 500 y VIX desde 1990.
2. **Etiquetado de regímenes (filtro HP):** se aplica el filtro Hodrick-Prescott al logaritmo del S&P 500 para separar tendencia de ciclo; el componente cíclico por debajo de un percentil define episodios "turbulentos", y se filtran episodios cortos (ruido).
3. **Número de Reynolds financiero:** un indicador propio calculado como `momentum × volumen_normalizado / VIX_normalizado`, winsorizado y normalizado, que busca capturar cuándo las condiciones de mercado se vuelven "turbulentas" antes de que el precio lo refleje.
4. **Modelo de régimen con HMM:** un Hidden Markov Model entrenado sobre las mismas features para detectar régimen de forma independiente al filtro HP, como validación cruzada del etiquetado.
5. **Estimación de duración con Kalman:** un filtro de Kalman estima cuánto puede durar un episodio de crisis una vez detectado, con bandas de incertidumbre.
6. **Backtesting:** se evalúa el indicador contra 8 crisis históricas reales (México 1994, Puntocom 2000, 2008, Deuda Europa 2011, Corrección 2018, COVID 2020, Inflación 2022, Corrección 2025), midiendo cuántas semanas antes de cada crisis el indicador mostró una señal de alerta.

## Por qué importa

No es un modelo de caja negra: cada componente (HP, Reynolds, HMM, Kalman) aporta una perspectiva distinta sobre el mismo fenómeno, y el backtesting contra crisis reales permite cuantificar qué tan útil es el indicador en la práctica, en lugar de solo mostrar que "se ve bien" en una gráfica.

## Stack técnico

`statsmodels` (filtro HP), `hmmlearn`, `scipy`, `pandas`, `numpy`, `yfinance`, `matplotlib`

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy scipy statsmodels hmmlearn matplotlib
python predictor_crisis.py
```

El script descarga los datos automáticamente, genera las visualizaciones de cada etapa y termina con una tabla de backtesting mostrando la anticipación lograda para cada crisis histórica.
