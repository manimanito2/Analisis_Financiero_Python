# Sistema de Detección de Regímenes de Mercado (HMM + LSTM)

Modelo híbrido que combina **Hidden Markov Models** (HMM) con una red **LSTM** (PyTorch) para detectar el régimen de mercado vigente y predecir la dirección del precio de un activo.

## Cómo funciona

1. **Construcción de features:** retornos a 1, 5 y 20 días, volatilidad rolling, RSI, MACD normalizado, ratio de volumen y VIX normalizado.
2. **Selección del número de regímenes:** se entrena un HMM Gaussiano para distintos números de estados (2 a 4) y se selecciona el óptimo según el criterio BIC (Bayesian Information Criterion), que balancea ajuste del modelo contra complejidad.
3. **Predicción de dirección:** una red LSTM toma una ventana de 60 días de features y predice si el precio subirá o bajará, usando el régimen detectado por el HMM como contexto adicional.
4. **Validación:**
   - Matriz de confusión y reporte de clasificación (precisión/recall por clase)
   - Accuracy desagregada por régimen de mercado y por año
   - Comparación de Sharpe ratio de la estrategia contra buy-and-hold
   - Equity curve y máximo drawdown
5. **Salida:** exporta resultados a un archivo Excel con formato (resumen ejecutivo, regímenes por día, predicciones, matriz de transición entre regímenes).

## Por qué importa

La mayoría de modelos de predicción de precio ignoran que el mercado cambia de "comportamiento" (régimen) a lo largo del tiempo. Este proyecto detecta esos cambios de régimen explícitamente y mide qué tan bien predice el modelo *dentro de cada régimen*, no solo en promedio — lo que da una imagen más honesta de cuándo el modelo funciona y cuándo no.

## Stack técnico

`hmmlearn` (GaussianHMM), `torch` (LSTM), `scikit-learn` (StandardScaler, métricas), `pandas`, `numpy`, `yfinance`, `openpyxl`

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy torch hmmlearn scikit-learn matplotlib openpyxl tqdm
python hmm_lstm_hibrido.py
```

El script pedirá un ticker (ej. `AAPL`, `TSLA`) y generará las gráficas de validación más un archivo Excel con los resultados detallados.
