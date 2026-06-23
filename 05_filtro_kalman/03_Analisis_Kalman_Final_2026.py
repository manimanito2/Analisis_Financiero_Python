import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====================
# Activar modo interactivo
# ====================
plt.ion()

# === Parámetros base ===
ticker_symbol = "ARQQ"
CRYPTO_SUFFIXES = ("-USD", "-BTC", "-ETH", "-USDT")
es_cripto = any(ticker_symbol.upper().endswith(s) for s in CRYPTO_SUFFIXES)
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d") # o ce puede cambiar a fecha "2026-06-06"
n_pred = 5
window_regimen = 20

def calibrar_parametros(ticker_symbol, start_date, end_date):
    # Descarga rápida para calibrar
    data_cal = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data_cal.empty:
        # Si no hay datos usar valores conservadores por defecto
        return 3.5, 25, 3, 1.5, 0.15

    precios_cal = np.asarray(data_cal['Close'].values, dtype=float).flatten()
    retornos = np.diff(precios_cal) / precios_cal[:-1] * 100  # retornos diarios en %

    vol_diaria = np.std(retornos)  # volatilidad diaria promedio del activo
    vol_segura = max(vol_diaria, 0.01)
    # Escalar parámetros en base a la volatilidad
    umbral_reg    = round(max(3.5, min(7.0, vol_diaria * 2.5)), 1)
    min_gap_cal   = int(max(12, min(35, vol_diaria * 7)))
    confirm_cal   = int(max(3, min(6, round(vol_diaria * 1.5))))
    cambio_pct_cal = round(max(0.3, min(6.0, vol_diaria * 0.4)), 1)
    vel_factor_cal = round(max(0.15, min(0.4, 0.15 * (1 / vol_segura) * 2)), 2)

    print(f"\n📊 Calibración automática para {ticker_symbol}:")
    print(f"   Volatilidad diaria promedio : {vol_diaria:.2f}%")
    print(f"   umbral_regimen              : {umbral_reg}")
    print(f"   min_gap                     : {min_gap_cal} días")
    print(f"   confirmacion                : {confirm_cal} días")
    print(f"   cambio_pct umbral           : {cambio_pct_cal}%")
    print(f"   vel_factor                  : {vel_factor_cal}\n")

    return umbral_reg, min_gap_cal, confirm_cal, cambio_pct_cal, vel_factor_cal

umbral_regimen, min_gap, confirmacion, cambio_pct_umbral, vel_factor = calibrar_parametros(
    ticker_symbol, start_date, end_date
)

# === Precio real conocido automático posterior al end_date ===
data_post = yf.download(ticker_symbol, start=end_date, end=datetime.today().date() + timedelta(days=1))
if data_post.empty:
    print("⚠️ No se pudo obtener el precio real posterior a la fecha de entrenamiento.")
    precio_real = None
    fecha_real = None
else:
    precio_real = float(data_post['Close'].iloc[0].item())
    fecha_real = data_post.index[0].date()
#se cambio -1 por 0 para obtener el precio del primer día posterior al end_date

# === Descargar precios históricos para entrenamiento ===
data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False)
if data.empty:
    raise ValueError(f"No se encontraron datos para {ticker_symbol} entre {start_date} y {end_date}.")
prices = np.asarray(data['Close'].values, dtype=float).flatten()
prices = pd.Series(prices).ffill().bfill().values
dates = data.index

# === Descargar S&P500 y VIX como señales externas ===
data_sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=False, progress=False)
data_vix   = yf.download("^VIX",  start=start_date, end=end_date, auto_adjust=False, progress=False)

sp500_prices = np.asarray(data_sp500['Close'].values, dtype=float).flatten() if not data_sp500.empty else None
vix_prices   = np.asarray(data_vix['Close'].values,   dtype=float).flatten() if not data_vix.empty else None

# VIX más reciente disponible
vix_actual = float(vix_prices[-1]) if vix_prices is not None and len(vix_prices) > 0 else 20.0
sp500_actual = sp500_prices[-1] if sp500_prices is not None and len(sp500_prices) > 5 else None

print(f"📡 VIX actual: {vix_actual:.2f} | S&P500 último cierre: {sp500_actual:.2f}" if sp500_actual else f"📡 VIX actual: {vix_actual:.2f}")

# === Matrices base ===
dt = 1
F = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]])

# === Calibrar Q y R dinámicamente con varianza móvil (ventana 20) ===
window = 20
price_var_moving = pd.Series(prices.flatten()).rolling(window=window, min_periods=1).var().values
price_var_moving = np.nan_to_num(price_var_moving, nan=1e-8)
price_var_moving = np.maximum(price_var_moving, 1e-8)

# === Filtro de Kalman ===
def apply_kalman(prices, F, H, price_var_moving, x_init, P_init, umbral_regimen):
    x = x_init.copy()
    P = P_init.copy()
    estimations = []
    velocities = []
    regimenes = []

    var_promedio = np.mean(price_var_moving)

    for i, z in enumerate(prices):
        z = np.array([[z]])

        # Detectar régimen
        if price_var_moving[i] > var_promedio * umbral_regimen:
            regimen = 'crisis'
            Q = np.array([[1e-4, 0],
                          [0, 1e-3]]) * price_var_moving[i]   # más reactivo
            R = np.array([[1]]) * (price_var_moving[i] * 3)   # confía menos en el modelo
        else:
            regimen = 'tranquilo'
            Q = np.array([[1e-5, 0],
                          [0, 1e-4]]) * price_var_moving[i]   # más suave
            R = np.array([[1]]) * (price_var_moving[i] * 10)  # confía más en el modelo

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        y = z - H @ x_pred
        x = (x_pred + K @ y).reshape(2, 1)
        P = (np.eye(2) - K @ H) @ P_pred

        estimations.append(x[0, 0])
        velocities.append(x[1, 0])
        regimenes.append(regimen)

    return np.array(estimations), np.array(velocities), np.array(regimenes), x, P

# === Predicción futura ===
def predict_future_states(x_last, P_last, F, base_Q, steps, last_date, es_cripto=False):
    future_preds = []
    future_dates = []
    bounds_1σ, bounds_2σ, bounds_3σ = [], [], []

    x_pred = x_last.copy()
    P_pred = P_last.copy()
    for i in range(steps):
        P_pred = F @ P_pred @ F.T + base_Q
        x_pred = F @ x_pred

        future_preds.append(x_pred[0, 0])

        # Días calendario para cripto, días hábiles para mercados tradicionales
        next_date = last_date + timedelta(days=i + 1)
        if not es_cripto:
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
        future_dates.append(next_date)

        std_dev = np.sqrt(P_pred[0, 0])
        bounds_1σ.append((x_pred[0, 0] - 1 * std_dev, x_pred[0, 0] + 1 * std_dev))
        bounds_2σ.append((x_pred[0, 0] - 2 * std_dev, x_pred[0, 0] + 2 * std_dev))
        bounds_3σ.append((x_pred[0, 0] - 3 * std_dev, x_pred[0, 0] + 3 * std_dev))

    return future_preds, future_dates, bounds_1σ, bounds_2σ, bounds_3σ

# === Inicialización Kalman ===
x0 = np.vstack(([prices[0]], [0.0]))
P0 = np.eye(2)

# === Aplicar filtro ===
#estimations, velocities, x_final, P_final = apply_kalman(prices, F, H, price_var_moving, x0, P0)
estimations, velocities, regimenes, x_final, P_final = apply_kalman(prices, F, H, price_var_moving, x0, P0, umbral_regimen)
# === Predicción futura ===
# === Ajuste por activos correlacionados ===
# Momentum S&P500: promedio de los últimos 5 días vs 20 días
if sp500_prices is not None and len(sp500_prices) >= 20:
    sp500_mom_corto = np.mean(sp500_prices[-5:])
    sp500_mom_largo = np.mean(sp500_prices[-20:])
    ratio_sp500 = (sp500_mom_corto - sp500_mom_largo) / sp500_mom_largo
else:
    ratio_sp500 = 0.0


# Correlación activo vs SP500
if sp500_prices is not None and len(sp500_prices) > 30:

    min_len = min(len(prices), len(sp500_prices))

    ret_activo = np.diff(prices[-min_len:]) / prices[-min_len:-1]
    ret_spx = np.diff(sp500_prices[-min_len:]) / sp500_prices[-min_len:-1]

    corr_sp500 = np.corrcoef(ret_activo, ret_spx)[0, 1]

    if np.isnan(corr_sp500):
        corr_sp500 = 0.0

else:
    corr_sp500 = 0.0


# Ajuste por VIX
factor_vix = np.clip(
    20 / max(vix_actual, 1),
    0.3,
    1.2
)


# Ajuste SP500
if not es_cripto:
    factor_sp500 = 1.0 + (ratio_sp500 * corr_sp500 * 0.5)
    factor_sp500 = np.clip(factor_sp500, 0.7, 1.3)
else:
    factor_sp500 = 1.0
    
x_final[1, 0] = x_final[1, 0] * 0.3 * factor_sp500 * factor_vix

print(
    f"🔧 Corr SP500: {corr_sp500:.3f}"
    f" | Ajuste S&P500: {factor_sp500:.3f}"
    f" | Ajuste VIX ({vix_actual:.1f}): {factor_vix:.2f}"
)

# Para predecir siempre usamos parámetros conservadores ignorando el periodo reciente volátil
avg_var = np.mean(price_var_moving[:-window])
base_Q = np.array([[1e-5, 0],
                   [0, 1e-4]]) * avg_var
future_preds, future_dates, bounds_1σ, bounds_2σ, bounds_3σ = predict_future_states(
    x_final, P_final, F, base_Q, n_pred, dates[-1], es_cripto)


# === Errores ===
mae = mean_absolute_error(prices, estimations)
rmse = np.sqrt(mean_squared_error(prices, estimations))

# Comparación real vs predicción (out-of-sample, solo si existe precio real)
if precio_real is not None:
    pred_día1 = future_preds[0]
    error_real = abs(precio_real - pred_día1)
    pct_error_real = (error_real / precio_real) * 100
else:
    error_real = None
    pct_error_real = None
    
# === Señales de momentum ===
momentum_signals = []
vel_umbral = np.std(velocities) * vel_factor
last_signal_idx = -min_gap

for i in range(confirmacion, len(velocities)):
    if (i - last_signal_idx) >= min_gap:
        tendencia_alcista = all(velocities[i - j] > velocities[i - j - 1] for j in range(confirmacion - 1))
        tendencia_bajista = all(velocities[i - j] < velocities[i - j - 1] for j in range(confirmacion - 1))

        cambio_pct = (estimations[i] - estimations[i - confirmacion]) / estimations[i - confirmacion] * 100

        if velocities[i] < -vel_umbral and tendencia_alcista and cambio_pct > cambio_pct_umbral:
            momentum_signals.append((dates[i], estimations[i], 'compra, posible piso'))
            last_signal_idx = i
        elif velocities[i] > vel_umbral and tendencia_bajista and cambio_pct < -cambio_pct_umbral:
            momentum_signals.append((dates[i], estimations[i], 'venta, posible techo'))
            last_signal_idx = i

# === Gráficas ===
plt.figure(figsize=(14, 10))

# Subplot 1: Precio y predicción
plt.subplot(2, 1, 1)
# Sombrear periodos de crisis
en_crisis = False
inicio_crisis = None
for i in range(len(regimenes)):
    if regimenes[i] == 'crisis' and not en_crisis:
        inicio_crisis = dates[i]
        en_crisis = True
    elif regimenes[i] == 'tranquilo' and en_crisis:
        plt.axvspan(inicio_crisis, dates[i], color='red', alpha=0.1)
        en_crisis = False
# Cerrar último periodo si termina en crisis
if en_crisis:
    plt.axvspan(inicio_crisis, dates[-1], color='red', alpha=0.1)
plt.plot(dates, prices, label='Precio observado', color='blue')
plt.plot(dates, estimations, label='Estimación Kalman', linestyle='--', color='red')
plt.plot(future_dates, future_preds, label='Predicción (futuro)', linestyle='-.', color='green', marker='o')

# Intervalos
lower_1σ, upper_1σ = zip(*bounds_1σ)
lower_2σ, upper_2σ = zip(*bounds_2σ)
lower_3σ, upper_3σ = zip(*bounds_3σ)
plt.fill_between(future_dates, lower_3σ, upper_3σ, color='green', alpha=0.1, label='±3σ (~99.7%)')
plt.fill_between(future_dates, lower_2σ, upper_2σ, color='lime', alpha=0.2, label='±2σ (~95%)')
plt.fill_between(future_dates, lower_1σ, upper_1σ, color='forestgreen', alpha=0.3, label='±1σ (~68%)')

# Anotar predicciones
for i, val in enumerate(future_preds):
    plt.annotate(f'{val:.2f}', (future_dates[i], val), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

# Señales de momentum
for date, price, tipo in momentum_signals:
    color = 'green' if 'compra' in tipo else 'red'
    plt.axvline(date, color=color, linestyle=':', alpha=0.6)
    plt.annotate(tipo.capitalize(), (date, price), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=7, color=color)

# Mostrar precio real conocido si existe
if precio_real is not None and fecha_real is not None:
    plt.axvline(fecha_real, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label='Precio real futuro')
    plt.plot(fecha_real, precio_real, marker='D', color='purple', markersize=6, label=f'Real: {precio_real:.2f}')
    plt.annotate(f'Real: {precio_real:.2f}', (fecha_real, precio_real), xytext=(0, 12),
                 textcoords='offset points', ha='center', fontsize=8, color='purple')

tipo_mercado = "Cripto 24/7" if es_cripto else "Mercado tradicional"
plt.title(f'Filtro de Kalman - Precio y Velocidad estimada ({ticker_symbol}) [{tipo_mercado}]')
plt.ylabel('Precio')
plt.legend()
plt.grid()

# Subplot 2: Velocidad estimada
plt.subplot(2, 1, 2)
plt.plot(dates, velocities, label='Velocidad estimada (cambio diario)', color='orange')
plt.axhline(0, linestyle='--', color='gray')
plt.ylabel('Velocidad')
plt.xlabel('Fecha')
plt.legend()
plt.grid()

# Métricas
#plt.figtext(0.5, 0.01, f"MAE: {mae:.4f} | RMSE: {rmse:.4f}", ha='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='black'))
extra = f" | Error real día 1: {error_real:.2f} ({pct_error_real:.2f}%)" if error_real is not None else ""
calibracion = f"  |  umbral_reg: {umbral_regimen} | min_gap: {min_gap} | confirm: {confirmacion} | Δ%: {cambio_pct_umbral}"
vix_info = f"  |  VIX: {vix_actual:.1f} | SP500 factor: {factor_sp500:.3f} | VIX factor: {factor_vix:.2f}"
plt.figtext(0.5, 0.01, f"MAE: {mae:.4f} | RMSE: {rmse:.4f}{extra}{calibracion}{vix_info}", ha='center', fontsize=9, bbox=dict(facecolor='white', edgecolor='black'))

plt.tight_layout()

# Fin del script
plt.ioff()
plt.show()