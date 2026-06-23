# Optimización de Portafolios y Análisis de Riesgo/Rendimiento

Conjunto de scripts para optimización de portafolios de inversión y clasificación de activos por su perfil de riesgo/rendimiento.

## Cómo funciona

**Optimización de portafolios (Markowitz):**
- Descarga precios históricos de una lista de tickers y calcula retornos logarítmicos, matriz de covarianza y correlación.
- Optimiza tres portafolios distintos: máximo Sharpe ratio, mínima varianza, y máximo Sortino ratio (que penaliza solo la volatilidad a la baja).
- Calcula CVaR (Conditional Value at Risk) al 5% y máximo drawdown para cada portafolio.
- Construye la frontera eficiente y simula 50,000 portafolios aleatorios vía Monte Carlo para visualizar el espacio riesgo/retorno completo.

**Análisis de riesgo/rendimiento:**
- Clasifica un universo de activos en cuadrantes (rendimiento alto/bajo × riesgo alto/bajo) comparando contra la mediana del conjunto.
- Genera un mapa de calor / gráfica de dispersión para identificar visualmente los activos con mejor perfil ajustado al riesgo.
- Incluye también análisis técnico complementario (medias móviles, RSI, MACD, estocástico, filtro Hodrick-Prescott) para contexto adicional sobre cada activo.

## Por qué importa

La optimización de Markowitz clásica solo usa varianza como medida de riesgo, lo cual penaliza por igual subidas y bajadas. Este proyecto añade Sortino (que distingue volatilidad "buena" de "mala") y CVaR (que se enfoca en el peor escenario) para dar una imagen de riesgo más completa que solo el Sharpe ratio.

## Stack técnico

`scipy.optimize` (optimización), `numpy`, `pandas`, `yfinance`, `matplotlib`, `statsmodels` (filtro HP)

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy scipy matplotlib statsmodels
python optimizacion_portafolios.py
```

Los tickers, rango de fechas y número de simulaciones Monte Carlo se configuran directamente en las variables al inicio del script.
