# Filtro de Kalman para Predicción de Precios

Implementación de un filtro de Kalman para estimar precio y "velocidad" (tendencia de corto plazo) de un activo financiero, con detección de régimen de volatilidad y ajuste por señales de mercado externas.

## Cómo funciona

1. **Modelo de estado:** se modela el precio como un sistema lineal con dos variables de estado — posición (precio) y velocidad (tasa de cambio) — actualizadas en cada paso con el filtro de Kalman clásico.
2. **Calibración automática:** los parámetros del filtro (umbral de régimen, ventana de confirmación, sensibilidad) se calibran automáticamente según la volatilidad histórica del activo, en lugar de fijarse a mano.
3. **Detección de régimen:** se distingue entre periodos "tranquilos" y de "crisis" según la varianza móvil del precio, ajustando dinámicamente cuánto confía el filtro en cada nueva observación (matrices Q y R).
4. **Ajuste por contexto de mercado:** la velocidad estimada se ajusta por el momentum del S&P 500, su correlación con el activo, y el nivel del VIX — para que la predicción no ignore lo que está pasando en el mercado en general.
5. **Predicción con incertidumbre:** proyecta el precio varios días hacia adelante con bandas de confianza de 1, 2 y 3 desviaciones estándar.
6. **Señales de momentum:** detecta cambios de tendencia (posibles pisos o techos) a partir de la velocidad estimada, con filtros para evitar señales repetidas demasiado seguido.

## Por qué importa

Un filtro de Kalman "de libro" usa parámetros fijos sin importar qué tan volátil esté el mercado. Esta versión ajusta su propia confianza según el régimen de volatilidad detectado, lo que lo hace más realista en mercados que alternan entre calma y turbulencia.

## Stack técnico

`numpy`, `pandas`, `yfinance`, `scikit-learn` (métricas de error), `matplotlib`

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy scikit-learn matplotlib
python filtro_kalman.py
```
