# Valuación de Empresas por Flujos de Caja Descontados (DCF)

Implementación de un modelo de valuación por **Flujos de Caja Descontados (DCF)** sobre una empresa real que cotiza en bolsa, usando datos financieros públicos. El objetivo es calcular un "precio justo" por acción desde primeros principios y compararlo contra el precio real de mercado.

**Caso de estudio:** The Coca-Cola Company (KO)

## Cómo funciona

1. **Flujo de Caja Libre histórico:** se descargan los últimos 4 años de Flujo de Caja Libre (FCF) reportado, verificado manualmente como Flujo de Caja Operativo menos CapEx.
2. **Elección de la base de proyección:** se evalúa el CAGR del periodo completo, pero se descarta al ser financieramente implausible (distorsionado por una caída atípica de FCF en un año puntual). En su lugar, se usa el promedio de los últimos 3 años como base más representativa, y una tasa de crecimiento conservadora justificada (decreciente de 5% a 2.5% en 5 años, propia de una empresa madura en un mercado de bebidas consolidado).
3. **Costo de Capital Propio (CAPM):** `Tasa libre de riesgo + Beta × Prima de riesgo de mercado`.
4. **Costo de la Deuda (después de impuestos):** gasto en intereses sobre deuda total, ajustado por el escudo fiscal de la deuda.
5. **WACC:** promedio ponderado de ambos costos, según el peso real de deuda y capital en la estructura de la empresa.
6. **Proyección a 5 años** del FCF con tasa de crecimiento decreciente.
7. **Valor Terminal** (perpetuidad de Gordon): captura el valor de todos los flujos después del año 5, asumiendo crecimiento constante de largo plazo.
8. **Descuento a valor presente** de los flujos proyectados y del Valor Terminal, usando el WACC.
9. **De Enterprise Value a precio por acción:** se resta la deuda neta para obtener el Equity Value, y se divide entre las acciones en circulación.
10. **Comparación contra el precio real de mercado.**

## Resultado obtenido (KO)

| Métrica | Valor |
|---|---|
| WACC | 4.42% |
| Precio justo según DCF | $78.69 |
| Precio de mercado | $80.31 |
| Diferencia | -2.02% |

## Hallazgo relevante del proceso

La primera versión del modelo, usando el FCF del año más reciente como base, arrojó una valuación 23% por debajo del precio de mercado. Al investigar la causa, se identificó que ese año estaba distorsionado por una caída atípica de FCF respecto a años anteriores. Ajustando la base a un promedio de 3 años (más representativo del FCF "normal" de la empresa), el resultado se corrigió a solo 2% de diferencia contra el precio real — una demostración directa de qué tan sensible es un modelo DCF a la elección del periodo base, no solo a la tasa de descuento o de crecimiento.

**Nota de honestidad analítica:** un resultado tan cercano al precio de mercado (-2%) es inusual para un DCF construido con supuestos simples — en la práctica, el precio de mercado también incorpora expectativas y riesgos que un modelo así no captura del todo. Este resultado es alentador, pero no se debe interpretar como validación definitiva del modelo; con otra empresa o otro periodo, la diferencia podría ser mucho mayor.

## Por qué importa

Un DCF no es solo aplicar una fórmula — requiere decisiones de criterio financiero en cada paso (qué tasa de crecimiento usar, qué periodo base, qué tan agresivo ser con los supuestos). Este proyecto documenta esas decisiones explícitamente, incluyendo un error de modelado real (la elección inicial de base) y cómo se diagnosticó y corrigió.

## Stack técnico

`yfinance` (datos financieros reales), `pandas`, `numpy`

## Cómo ejecutarlo

```bash
pip install yfinance pandas numpy
python valuacion_dcf.py
```

El ticker, las tasas de crecimiento y los supuestos de mercado (tasa libre de riesgo, prima de riesgo) se configuran directamente en las variables al inicio del script.
