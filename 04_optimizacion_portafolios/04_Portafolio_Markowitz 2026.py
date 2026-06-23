import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================
# Activar modo interactivo
# ====================
plt.ion()

# =============================
# 1. CONFIGURACIÓN
# =============================
tickers = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'NFLX', 'NVDA', 'TSLA']

start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d") # o ce puede cambiar a fecha "2026-06-06"
risk_free_rate = 0.04 / 252      # 4% anual → diario
risk_free_annual = 0.04          # para cálculos anualizados
NUM_PORTFOLIOS = 50000            # portafolios Monte Carlo
CONFIDENCE_LEVEL = 0.05          # 5% para CVaR

# Portafolio actual del usuario (opcional) — pon tus pesos reales aquí
# Si no tienes portafolio actual, déjalo en None
current_weights = None
# Ejemplo: current_weights = np.array([0.20, 0.15, 0.15, 0.10, 0.10, 0.20, 0.10])

# =============================
# 2. DESCARGA Y LIMPIEZA DE DATOS
# =============================
print("📥 Descargando datos...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Extraer precios ajustados
data_adj = pd.DataFrame()
if isinstance(data.columns, pd.MultiIndex):
    for ticker in tickers:
        if ('Adj Close', ticker) in data.columns:
            data_adj[ticker] = data[('Adj Close', ticker)]
        elif ('Close', ticker) in data.columns:
            data_adj[ticker] = data[('Close', ticker)]
        else:
            print(f"⚠️ No se encontró columna válida para {ticker}, se omite.")
else:
    for ticker in tickers:
        if 'Adj Close' in data.columns:
            data_adj[ticker] = data['Adj Close']
        elif 'Close' in data.columns:
            data_adj[ticker] = data['Close']

# Eliminar tickers con demasiados NaN (más del 20% de datos faltantes)
threshold = 0.20
tickers_validos = [t for t in data_adj.columns if data_adj[t].isna().mean() < threshold]
tickers_removidos = [t for t in tickers if t not in tickers_validos]
if tickers_removidos:
    print(f"⚠️ Tickers removidos por datos insuficientes: {tickers_removidos}")

data_adj = data_adj[tickers_validos].ffill().bfill().dropna()
tickers = tickers_validos
num_assets = len(tickers)

print(f"✅ Tickers válidos: {tickers}")
print(f"✅ Datos: {len(data_adj)} días de {start_date} a {end_date}\n")

# =============================
# 3. RENDIMIENTOS Y ESTADÍSTICAS
# =============================
returns = np.log(data_adj / data_adj.shift(1)).dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
corr_matrix = returns.corr()

# =============================
# 4. FUNCIONES DE PORTAFOLIO
# =============================

def portfolio_performance(weights, mean_returns, cov_matrix, annual=False):

    ret = np.dot(weights, mean_returns)

    risk = np.sqrt(
        np.dot(weights.T,
               np.dot(cov_matrix, weights))
    )

    if annual:
        return ret * 252, risk * np.sqrt(252)

    return ret, risk

def sortino_ratio(weights, mean_returns, cov_matrix, rf=risk_free_annual):

    ret = np.dot(weights, mean_returns) * 252

    portfolio_returns = returns @ weights

    downside_returns = portfolio_returns[
        portfolio_returns < 0
    ]

    if len(downside_returns) == 0:
        return np.inf

    downside_dev = np.sqrt(
        np.mean(downside_returns**2)
    ) * np.sqrt(252)

    if downside_dev == 0:
        return np.inf

    return (ret - rf) / downside_dev

def sharpe_ratio(weights, mean_returns, cov_matrix, rf=risk_free_annual):

    ret, risk = portfolio_performance(
        weights,
        mean_returns,
        cov_matrix,
        annual=True
    )

    if risk == 0:
        return 0

    return (ret - rf) / risk

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf=risk_free_annual):
    """Negativo del Sharpe para minimización."""
    return -sharpe_ratio(weights, mean_returns, cov_matrix, rf)

def cvar(weights, confidence=CONFIDENCE_LEVEL):
    """
    Calcula el CVaR (Conditional Value at Risk) diario.
    Responde: en el peor X% de los días, ¿cuánto se pierde en promedio?
    """
    portfolio_returns = returns @ weights
    var_threshold = np.percentile(portfolio_returns, confidence * 100)
    cvar_value = portfolio_returns[portfolio_returns <= var_threshold].mean()
    return abs(cvar_value)

def portfolio_variance(weights, mean_returns, cov_matrix):
    """Varianza del portafolio."""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]**2

def negative_sortino(weights, mean_returns, cov_matrix, rf=risk_free_annual):
    """Negativo del Sortino para minimización."""
    return -sortino_ratio(weights, mean_returns, cov_matrix, rf)

def max_drawdown(weights):

    portfolio_returns = returns @ weights

    cumulative = np.exp(portfolio_returns.cumsum())

    running_max = cumulative.cummax()

    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()

# =============================
# 5. OPTIMIZACIÓN
# =============================
print("⚙️ Optimizando portafolios...")

constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = tuple((0.05, 0.35) for _ in range(num_assets))
initial_guess = np.array(num_assets * [1. / num_assets])

# Portafolio Máximo Sharpe
opt_sharpe = minimize(
    negative_sharpe_ratio,
    initial_guess,
    args=(mean_returns, cov_matrix, risk_free_annual),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
if not opt_sharpe.success:
    print("⚠️ Advertencia: La optimización de Máximo Sharpe no convergió perfectamente.")

# Portafolio Mínima Varianza
opt_min_var = minimize(
    portfolio_variance,
    initial_guess,
    args=(mean_returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
if not opt_min_var.success:
    print("⚠️ Advertencia: La optimización de Mínima Varianza no convergió perfectamente.")

# Portafolio Máximo Sortino
opt_sortino = minimize(
    negative_sortino,
    initial_guess,
    args=(mean_returns, cov_matrix, risk_free_annual),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
if not opt_sortino.success:
    print("⚠️ Advertencia: La optimización de Máximo Sortino no convergió perfectamente.")

weights_sharpe = opt_sharpe.x
weights_min_var = opt_min_var.x
weights_sortino = opt_sortino.x

ret_sharpe, risk_sharpe = portfolio_performance(weights_sharpe, mean_returns, cov_matrix, annual=True)
ret_min, risk_min = portfolio_performance(weights_min_var, mean_returns, cov_matrix, annual=True)
ret_sortino, risk_sortino = portfolio_performance(weights_sortino, mean_returns, cov_matrix, annual=True)

sharpe_sharpe = sharpe_ratio(weights_sharpe, mean_returns, cov_matrix)
sharpe_minvar = sharpe_ratio(weights_min_var, mean_returns, cov_matrix)
sharpe_sortino_p = sharpe_ratio(weights_sortino, mean_returns, cov_matrix)

sortino_sharpe_p = sortino_ratio(weights_sharpe, mean_returns, cov_matrix)
sortino_minvar_p = sortino_ratio(weights_min_var, mean_returns, cov_matrix)
sortino_sortino_p = sortino_ratio(weights_sortino, mean_returns, cov_matrix)

cvar_sharpe = cvar(weights_sharpe)
cvar_minvar = cvar(weights_min_var)
cvar_sortino = cvar(weights_sortino)

print("✅ Optimización completada.\n")

# =============================
# 6. FRONTERA EFICIENTE
# =============================
print("📈 Calculando frontera eficiente...")

def efficient_frontier(mean_returns, cov_matrix, n_points=50):
    """Calcula la frontera eficiente con n_points puntos."""
    results = []
    ret_min_var_val = portfolio_performance(weights_min_var, mean_returns, cov_matrix)[0]
    ret_max = mean_returns.max()
    returns_range = np.linspace(ret_min_var_val, ret_max, n_points)

    for target_ret in returns_range:
        constraints_ef = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x, r=target_ret: portfolio_performance(x, mean_returns, cov_matrix)[0] - r}
        )
        result = minimize(
            portfolio_variance,
            initial_guess,
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_ef
        )
        if result.success:
            results.append((target_ret, np.sqrt(result.fun)))
    return results

frontier = efficient_frontier(mean_returns, cov_matrix)
print("✅ Frontera eficiente calculada.\n")

# =============================
# 7. MONTE CARLO
# =============================
print(f"🎲 Simulando {NUM_PORTFOLIOS} portafolios aleatorios...")

mc_returns, mc_risks, mc_sharpes, mc_sortinos, mc_cvars = [], [], [], [], []
mc_weights_list = []

for _ in range(NUM_PORTFOLIOS):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    r, risk = portfolio_performance(w, mean_returns, cov_matrix, annual=True)
    sh = sharpe_ratio(w, mean_returns, cov_matrix)
    so = sortino_ratio(w, mean_returns, cov_matrix)
    cv = cvar(w)  # CVaR diario — no se anualiza multiplicando por 252
    mc_returns.append(r)
    mc_risks.append(risk)
    mc_sharpes.append(sh)
    mc_sortinos.append(so)
    mc_cvars.append(cv)
    mc_weights_list.append(w)

mc_returns = np.array(mc_returns)
mc_risks = np.array(mc_risks)
mc_sharpes = np.array(mc_sharpes)

print("✅ Monte Carlo completado.\n")

# =============================
# 8. REBALANCEO (si hay portafolio actual)
# =============================
rebalanceo_info = None
if current_weights is not None and len(current_weights) == num_assets:
    current_weights = np.array(current_weights)
    current_weights /= current_weights.sum()  # normalizar
    ret_current, risk_current = portfolio_performance(current_weights, mean_returns, cov_matrix, annual=True)
    sharpe_current = sharpe_ratio(current_weights, mean_returns, cov_matrix)
    delta_sharpe = weights_sharpe - current_weights
    delta_minvar = weights_min_var - current_weights
    rebalanceo_info = {
        'current': current_weights,
        'ret_current': ret_current,
        'risk_current': risk_current,
        'sharpe_current': sharpe_current,
        'delta_sharpe': delta_sharpe,
        'delta_minvar': delta_minvar
    }

# =============================
# 9. CONSOLA — RESUMEN COMPLETO
# =============================
print("=" * 60)
print("📊 OPTIMIZACIÓN DE PORTAFOLIO — MARKOWITZ")
print("=" * 60)

def print_portfolio(name, weights, ret, risk, sharpe, sortino, cvar_val):
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Rendimiento anual : {ret:.2%}")
    print(f"  Riesgo anual      : {risk:.2%}")
    print(f"  Ratio Sharpe      : {sharpe:.4f}")
    print(f"  Ratio Sortino     : {sortino:.4f}")
    print(f"  CVaR (5%)         : {cvar_val:.2%}")
    print(f"\n  Pesos óptimos:")
    for t, w in zip(tickers, weights):
        bar = '█' * int(w * 30)
        print(f"    {t:8s}: {w:.2%}  {bar}")

print_portfolio("★ Máximo Sharpe", weights_sharpe, ret_sharpe, risk_sharpe,
                sharpe_sharpe, sortino_sharpe_p, cvar_sharpe)
print_portfolio("● Mínima Varianza", weights_min_var, ret_min, risk_min,
                sharpe_minvar, sortino_minvar_p, cvar_minvar)
print_portfolio("▲ Máximo Sortino", weights_sortino, ret_sortino, risk_sortino,
                sharpe_sortino_p, sortino_sortino_p, cvar_sortino)

if rebalanceo_info:
    print(f"\n{'─'*40}")
    print(f"  🔄 Rebalanceo sugerido (vs Máx. Sharpe)")
    print(f"{'─'*40}")
    print(f"  Portafolio actual → Rend: {rebalanceo_info['ret_current']:.2%} | "
          f"Riesgo: {rebalanceo_info['risk_current']:.2%} | "
          f"Sharpe: {rebalanceo_info['sharpe_current']:.4f}")
    print(f"\n  Ajustes necesarios:")
    for t, d in zip(tickers, rebalanceo_info['delta_sharpe']):
        accion = "COMPRAR" if d > 0.01 else "VENDER" if d < -0.01 else "MANTENER"
        print(f"    {t:8s}: {accion:8s} {abs(d):.2%}")

print("\n" + "=" * 60)

# =============================
# 10. GRÁFICAS
# =============================

# ── Figura 1: Frontera Eficiente + Monte Carlo ──────────────────
plt.figure(figsize=(12, 7))
plt.suptitle("Optimización de Portafolio — Markowitz", fontsize=13, fontweight='bold')

sc = plt.scatter(mc_risks, mc_returns, c=mc_sharpes, cmap='viridis', alpha=0.4, s=15, label='Portafolios MC')
plt.colorbar(sc, label='Ratio Sharpe')

if frontier:
    plt.plot([r * np.sqrt(252) for _, r in frontier],
             [ret * 252 for ret, _ in frontier],
             'g--', linewidth=2, label='Frontera Eficiente')

plt.scatter(risk_sharpe, ret_sharpe, c='red', marker='*', s=300, zorder=5, label=f'Máx. Sharpe ({sharpe_sharpe:.2f})')
plt.scatter(risk_min, ret_min, c='blue', marker='o', s=200, zorder=5, label=f'Mín. Varianza ({sharpe_minvar:.2f})')
plt.scatter(risk_sortino, ret_sortino, c='orange', marker='^', s=200, zorder=5, label=f'Máx. Sortino ({sharpe_sortino_p:.2f})')

plt.xlabel("Riesgo anual (Volatilidad)")
plt.ylabel("Rendimiento anual esperado")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 2: Pesos de los 3 portafolios ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribución de Pesos por Portafolio", fontsize=13, fontweight='bold')

portfolios = [
    ("★ Máx. Sharpe", weights_sharpe, 'lightcoral'),
    ("● Mín. Varianza", weights_min_var, 'lightblue'),
    ("▲ Máx. Sortino", weights_sortino, 'lightyellow')
]

for ax, (name, weights, color) in zip(axes, portfolios):
    wedges, texts, autotexts = ax.pie(
        weights,
        labels=[f"{t}\n{w:.1%}" for t, w in zip(tickers, weights)],
        autopct='',
        colors=plt.cm.Set3(np.linspace(0, 1, num_assets)),
        startangle=90
    )
    ax.set_title(name, fontweight='bold')

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 3: Heatmap de correlaciones ──────────────────────────
plt.figure(figsize=(8, 6))
plt.suptitle("Mapa de Correlación entre Activos", fontsize=13, fontweight='bold')

im = plt.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, label='Correlación')
plt.xticks(range(num_assets), tickers, rotation=45, ha='right')
plt.yticks(range(num_assets), tickers)

for i in range(num_assets):
    for j in range(num_assets):
        val = corr_matrix.iloc[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 4: Comparativa de métricas ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Comparativa de Métricas por Portafolio", fontsize=13, fontweight='bold')

nombres = ['Máx. Sharpe', 'Mín. Varianza', 'Máx. Sortino']
colores = ['lightcoral', 'lightblue', 'lightyellow']

# Rendimiento vs Riesgo
rets_comp = [ret_sharpe, ret_min, ret_sortino]
risks_comp = [risk_sharpe, risk_min, risk_sortino]
x = np.arange(len(nombres))
width = 0.35
axes[0].bar(x - width/2, rets_comp, width, label='Rendimiento', color='green', alpha=0.7)
axes[0].bar(x + width/2, risks_comp, width, label='Riesgo', color='red', alpha=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(nombres, rotation=15, ha='right')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[0].set_title('Rendimiento vs Riesgo')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sharpe vs Sortino
sharpes_comp = [sharpe_sharpe, sharpe_minvar, sharpe_sortino_p]
sortinos_comp = [sortino_sharpe_p, sortino_minvar_p, sortino_sortino_p]
axes[1].bar(x - width/2, sharpes_comp, width, label='Sharpe', color='blue', alpha=0.7)
axes[1].bar(x + width/2, sortinos_comp, width, label='Sortino', color='purple', alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(nombres, rotation=15, ha='right')
axes[1].set_title('Sharpe vs Sortino')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# CVaR
cvars_comp = [cvar_sharpe, cvar_minvar, cvar_sortino]
bars = axes[2].bar(nombres, cvars_comp, color=colores, edgecolor='gray', alpha=0.9)
axes[2].set_xticklabels(nombres, rotation=15, ha='right')
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[2].set_title('CVaR 5% (menor es mejor)')
for bar, val in zip(bars, cvars_comp):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.2%}', ha='center', va='bottom', fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 5: Rebalanceo (si aplica) ───────────────────────────
if rebalanceo_info:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Rebalanceo sugerido: Actual vs Máximo Sharpe", fontsize=13, fontweight='bold')

    x = np.arange(num_assets)
    width = 0.35
    ax.bar(x - width/2, rebalanceo_info['current'], width, label='Portafolio Actual', color='gray', alpha=0.7)
    ax.bar(x + width/2, weights_sharpe, width, label='Máx. Sharpe', color='lightcoral', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
plt.pause(0.1)

# ── Figura 6: Resumen visual completo ───────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.suptitle("📊 Resumen — Optimización de Portafolio Markowitz", 
             fontsize=14, fontweight='bold', y=0.98)

portfolios_resumen = [
    ("★ Máximo Sharpe", weights_sharpe, ret_sharpe, risk_sharpe,
     sharpe_sharpe, sortino_sharpe_p, cvar_sharpe, 'lightcoral'),
    ("● Mínima Varianza", weights_min_var, ret_min, risk_min,
     sharpe_minvar, sortino_minvar_p, cvar_minvar, 'lightblue'),
    ("▲ Máximo Sortino", weights_sortino, ret_sortino, risk_sortino,
     sharpe_sortino_p, sortino_sortino_p, cvar_sortino, 'lightyellow'),
]

for col, (nombre, weights, ret, risk, sharpe, sortino, cvar_val, color) in enumerate(portfolios_resumen):
    ax = fig.add_subplot(1, 3, col + 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(color)
    fig.patch.set_facecolor('white')

    # Fondo del panel
    rect = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.4, transform=ax.transAxes, zorder=0)
    ax.add_patch(rect)

    # Título del portafolio
    ax.text(0.5, 0.96, nombre, ha='center', va='top', fontsize=13,
            fontweight='bold', transform=ax.transAxes)

    # Línea separadora
    ax.plot([0.05, 0.95], [0.91, 0.91], color='gray', linewidth=0.8, transform=ax.transAxes)

    # Métricas principales
    metricas = [
        ("Rendimiento anual", f"{ret:.2%}"),
        ("Riesgo anual",      f"{risk:.2%}"),
        ("Ratio Sharpe",      f"{sharpe:.4f}"),
        ("Ratio Sortino",     f"{sortino:.4f}"),
        ("CVaR (5%)",         f"{cvar_val:.2%}"),
    ]

    y_pos = 0.87
    for label, valor in metricas:
        ax.text(0.08, y_pos, label, ha='left', va='top', fontsize=10,
                transform=ax.transAxes, color='#333333')
        ax.text(0.92, y_pos, valor, ha='right', va='top', fontsize=10,
                fontweight='bold', transform=ax.transAxes, color='#111111')
        y_pos -= 0.07

    # Línea separadora pesos
    ax.plot([0.05, 0.95], [y_pos + 0.03, y_pos + 0.03], color='gray', linewidth=0.8, transform=ax.transAxes)
    y_pos -= 0.02

    ax.text(0.5, y_pos, "Pesos óptimos", ha='center', va='top', fontsize=10,
            fontweight='bold', transform=ax.transAxes, color='#333333')
    y_pos -= 0.06

    # Barras de pesos
    for ticker, w in zip(tickers, weights):
        bar_width = w * 0.7
        # Barra
        bar_rect = plt.Rectangle((0.08, y_pos - 0.025), bar_width, 0.03,
                                   color='steelblue', alpha=0.6, transform=ax.transAxes)
        ax.add_patch(bar_rect)
        # Ticker
        ax.text(0.08, y_pos, ticker, ha='left', va='center', fontsize=9,
                transform=ax.transAxes, color='#111111')
        # Porcentaje
        ax.text(0.92, y_pos, f"{w:.1%}", ha='right', va='center', fontsize=9,
                fontweight='bold', transform=ax.transAxes, color='#111111')
        y_pos -= 0.055

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show(block=False)
plt.pause(0.1)

print("\n✅ Análisis completado.")
plt.ioff()
input("\nPresiona Enter para cerrar todas las gráficas...")
plt.close('all')


#temporal 
print(f"DEBUG: rf diario={risk_free_rate:.6f}, rf anual={risk_free_annual:.4f}")

# Fin del script
plt.ioff()
plt.show()