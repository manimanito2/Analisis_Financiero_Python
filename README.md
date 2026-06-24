# Análisis Financiero con Python

Colección de proyectos de análisis cuantitativo de mercados financieros, desarrollados en Python. Combino series de tiempo, machine learning, deep learning y optimización de portafolios para construir herramientas de análisis, predicción y gestión de riesgo.

**Autor:** Alejandro Ugarte Mendoza — Ingeniero Industrial (UNAM), Maestría en Optimización Financiera (UNAM)
[LinkedIn](https://www.linkedin.com/in/ing-alejandro-ugarte-ab5b43207)

---

## Proyectos

| Proyecto | Descripción breve | Técnicas principales |
|---|---|---|
| [`01_hmm_lstm_regimenes/`](./01_hmm_lstm_regimenes) | Detección de régimen de mercado y predicción de dirección de precio | HMM, LSTM (PyTorch), selección por BIC |
| [`02_predictor_crisis_reynolds/`](./02_predictor_crisis_reynolds) | Indicador propio de anticipación de crisis financieras, validado contra 8 crisis históricas | Filtro HP, HMM, Kalman, indicador propio |
| [`03_radar_sentimiento/`](./03_radar_sentimiento) | Aplicación de escritorio que agrega sentimiento de mercado en tiempo real | APIs en tiempo real, CustomTkinter |
| [`04_optimizacion_portafolios/`](./04_optimizacion_portafolios) | Optimización de portafolios | Markowitz, Sharpe, Sortino, CVaR, Monte Carlo |
| [`05_filtro_kalman/`](./05_filtro_kalman) | Predicción de precios con detección de régimen y ajuste por señales externas | Filtro de Kalman |
| [`06_pipeline_datos_bancarios/`](./06_pipeline_datos_bancarios) | Limpieza de datos bancarios sintéticos y dashboard en Power BI | pandas, Excel, Power BI |
| [`07_utilidades/`](./07_utilidades) | Scripts de soporte (verificación de tickers, etc.) | yfinance, concurrencia |
| [`08_riesgo_rendimiento/`](./08_riesgo_rendimiento) | Análisis de riesgo/rendimiento con filtro de liquidez y ranking compuesto | Sharpe, Sortino, tendencia reciente, Excel |
| [`09_valuacion_dcf/`](./09_valuacion_dcf) | Valuación de empresas por flujos de caja descontados, validada contra precio real de mercado | DCF, WACC, CAPM |
| [`10_scoring_credito/`](./10_scoring_credito) | Probabilidad de impago en solicitudes de crédito | XGBoost, AUC-ROC |
| [`11_churn_bancario/`](./11_churn_bancario) | Predicción de abandono de clientes con segmentación de riesgo | XGBoost, AUC-ROC |
| [`12_deteccion_fraude/`](./12_deteccion_fraude) | Detección de fraude transaccional en datos desbalanceados | XGBoost, Precision-Recall |

---

## Stack técnico

**Lenguajes y librerías:** Python, pandas, numpy, scipy, scikit-learn, statsmodels, PyTorch, hmmlearn, XGBoost, yfinance, matplotlib, openpyxl, Faker

**Métodos:** Hidden Markov Models, redes LSTM, filtros de Kalman y Hodrick-Prescott, optimización de portafolios (Markowitz), valuación por flujos de caja descontados (DCF, WACC, CAPM), simulación Monte Carlo, análisis técnico (RSI, MACD, estocástico), modelos de clasificación con XGBoost (riesgo crediticio, churn, fraude), backtesting

**Otras herramientas:** Power BI, Excel, APIs externas en tiempo real (Finnhub, StockTwits, Google Trends)

---

## Nota

Estos son proyectos personales de aprendizaje y portafolio. El código se comparte con fines demostrativos; ningún proyecto constituye asesoría de inversión, crédito o riesgo.


