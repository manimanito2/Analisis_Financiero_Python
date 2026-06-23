# Utilidades

Scripts de soporte usados en otros proyectos del repositorio.

## `verificador_tickers.py`

Verifica en paralelo (usando `ThreadPoolExecutor`) si una lista de tickers existe y en qué mercado/exchange cotiza, usando la API de Yahoo Finance vía `yfinance`. Útil como paso de validación antes de correr un análisis sobre una lista grande de tickers, para descartar símbolos inválidos o mal escritos.

Exporta los resultados (existe / no existe, mercado, nombre de la empresa) a un archivo Excel.

### Cómo ejecutarlo

```bash
pip install yfinance pandas openpyxl
python verificador_tickers.py
```
