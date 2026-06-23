# Radar de Sentimiento de Mercado en Tiempo Real

Aplicación de escritorio que monitorea sentimiento de mercado en vivo sobre un universo de ~30 tickers, agregando múltiples fuentes de información en un solo score por acción.

## Cómo funciona

1. **Fuentes agregadas:** noticias de Finnhub (análisis de sentimiento con TextBlob), mensajes de StockTwits, noticias de Yahoo Finance, actividad de insiders, tendencias de búsqueda en Google Trends, y un ajuste por "spillover" de mercados asiáticos (Nikkei, Hang Seng, Shanghai).
2. **Estado del mercado:** detecta automáticamente si el mercado está abierto, en pre-market, after-hours o cerrado, y ajusta la frecuencia de actualización en consecuencia (más frecuente cuando el mercado está activo).
3. **Score consolidado:** combina las fuentes en un score final por ticker, con una señal categorizada (fuerte compra, compra, neutral, precaución, evitar).
4. **Visualización interactiva:** mapa de dispersión (sentimiento vs. momentum, tamaño = volumen inusual) con tooltips al pasar el cursor mostrando el detalle de cada fuente.
5. **Exportación:** resultados exportables a Excel (con formato condicional por señal) y CSV.

## Por qué importa

Las señales de sentimiento son ruidosas si se usan de una sola fuente. Este proyecto trata el sentimiento como un problema de **agregación de señales débiles** — cada fuente individual aporta poco, pero combinadas (y ponderadas por contexto de mercado) dan una imagen más estable.

## Stack técnico

`customtkinter` (interfaz), `requests` (consumo de APIs), `textblob` (NLP de sentimiento), `pytrends` (Google Trends), `yfinance`, `matplotlib` (embebido en Tkinter), `openpyxl`

## Cómo ejecutarlo

```bash
pip install customtkinter requests yfinance matplotlib textblob pytrends pytz openpyxl
python radar_sentimiento.py
```

**Nota:** requiere una API key de Finnhub (gratuita en [finnhub.io](https://finnhub.io)) configurada en el script.
