# Análisis de Riesgo / Rendimiento con Filtro de Liquidez y Ranking

Script de Python que analiza un universo de activos (acciones de EE.UU. y México) por su perfil de riesgo/rendimiento, filtra por liquidez real, detecta tendencias recientes y genera un ranking compuesto exportado a Excel.

## Cómo funciona

1. **Descarga y limpieza:** descarga precios y volumen histórico (desde 2015) y reciente (últimos ~6 meses) de un universo de ~35 tickers; descarta automáticamente los que tienen más del 20% de datos faltantes.
2. **Métricas de riesgo/rendimiento:** calcula rendimiento y volatilidad anualizados, Sharpe ratio (ajustado por tasa libre de riesgo) y Sortino ratio (penaliza solo volatilidad a la baja), tanto históricos como de los últimos 6 meses.
3. **Tendencia reciente:** compara el rendimiento reciente contra el histórico para detectar si un activo está "mejorando" o "deteriorando" su desempeño.
4. **Filtro de liquidez:** calcula el volumen promedio en dólares de cada activo y excluye del ranking final a los que no cumplen un mínimo diario, para evitar recomendar activos poco negociables.
5. **Clasificación en cuadrantes:** categoriza cada activo en una matriz de riesgo/rendimiento, tanto contra la mediana del propio universo (dinámica) como contra umbrales fijos definidos de antemano.
6. **Ranking compuesto:** combina Sharpe, Sortino, tendencia reciente y liquidez (con pesos definidos) en un score único, y muestra el Top 5 y Bottom 5 de activos líquidos.
7. **Visualización:** 4 gráficas — clasificación por cuadrantes (dinámica y fija), ranking top/bottom, tendencia reciente por activo, y comparación de Sharpe histórico vs. reciente.
8. **Exportación a Excel:** genera un archivo con 3 hojas (clasificación dinámica, clasificación fija, ranking) con formato condicional por categoría.

## Por qué importa

Un ranking de "mejores acciones" que solo mire rendimiento histórico puede recomendar activos que ya no tienen buen desempeño, o que son tan poco negociados que comprarlos/venderlos movería el precio. Este script corrige ambos problemas: pondera qué tan reciente es el buen desempeño y filtra por liquidez real antes de recomendar nada.

## Stack técnico

`pandas`, `numpy`, `yfinance`, `matplotlib`, `openpyxl` (con formato condicional)

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy matplotlib openpyxl
python analisis_riesgo_rendimiento.py
```

Los tickers, fechas y umbrales se configuran directamente en las variables al inicio del script. El reporte en Excel se guarda automáticamente en la misma carpeta donde corre el script.

