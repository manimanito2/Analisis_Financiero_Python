import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

plt.ion()

# =============================
# 1. CONFIGURACIÓN
# =============================
tickers = ['AAPL', 'AMX', 'AVGO', 'CAT', 'CLF', 'FCX', 'FSLR', 'GCC', 'GE', 'GME',
           'GMEXICOB.MX', 'GOOGL', 'GRUMAB.MX', 'JPM', 'KOFUBL.MX', 'LIVEPOLC-1.MX',
           'LLY', 'MA', 'MFRISCOA-1.MX', 'MRK', 'NVDA', 'ORCL', 'PLTR', 'SOFI',
           'T', 'UBER', 'UNH', 'WFC', 'WMT', 'XOM', 'MARA','ZVRA', 'FLUT','MGNI','DKNG','UNH', 'ARQQ']

start_date       = '2015-01-01'
end_date         = datetime.today().strftime("%Y-%m-%d") # o ce puede cambiar a fecha "2026-06-06"
start_reciente   = '2025-11-01'   # últimos ~6 meses para tendencia reciente
trading_days     = 252
risk_free_rate   = 0.04           # 4% anual
umbral_return    = 0.08           # umbral rendimiento clasificación fija
umbral_vol       = 0.25           # umbral volatilidad clasificación fija
min_vol_millones = 1.0            # volumen mínimo diario promedio en millones USD
TOP_N            = 5              # mejores/peores a mostrar en ranking

# Ruta de salida relativa al directorio donde está el script
ruta_salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resultados_inversion.xlsx")

# =============================
# 2. DESCARGA Y LIMPIEZA
# =============================
print("📥 Descargando datos históricos...")
raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
raw_reciente = yf.download(tickers, start=start_reciente, end=end_date, progress=False)

def extraer_precios_volumen(raw_data, tickers_list):
    """Extrae precios de cierre y volumen manejando MultiIndex."""
    precios = pd.DataFrame()
    volumenes = pd.DataFrame()
    if isinstance(raw_data.columns, pd.MultiIndex):
        for t in tickers_list:
            if ('Close', t) in raw_data.columns:
                precios[t] = raw_data[('Close', t)]
            if ('Volume', t) in raw_data.columns:
                volumenes[t] = raw_data[('Volume', t)]
    else:
        if 'Close' in raw_data.columns:
            precios = raw_data[['Close']].copy()
        if 'Volume' in raw_data.columns:
            volumenes = raw_data[['Volume']].copy()
    return precios, volumenes

precios_hist, volumenes_hist = extraer_precios_volumen(raw, tickers)
precios_rec, _ = extraer_precios_volumen(raw_reciente, tickers)

# Detectar y remover tickers con demasiados NaN (>20%)
threshold = 0.20
tickers_validos = [t for t in precios_hist.columns
                   if precios_hist[t].isna().mean() < threshold]
tickers_removidos = [t for t in tickers if t not in tickers_validos]
if tickers_removidos:
    print(f"⚠️ Tickers removidos por datos insuficientes: {tickers_removidos}")

precios_hist  = precios_hist[tickers_validos].ffill().bfill()
precios_rec   = precios_rec[[t for t in tickers_validos if t in precios_rec.columns]].ffill().bfill()
volumenes_hist = volumenes_hist[[t for t in tickers_validos if t in volumenes_hist.columns]].ffill().bfill()
tickers = tickers_validos

print(f"✅ Tickers válidos: {len(tickers)} de {len(tickers_validos) + len(tickers_removidos)}")

# =============================
# 3. RENDIMIENTOS
# =============================
returns_hist = precios_hist.pct_change().dropna()
returns_rec  = precios_rec.pct_change().dropna()

# =============================
# 4. MÉTRICAS HISTÓRICAS
# =============================
mean_daily   = returns_hist.mean()
std_daily    = returns_hist.std()
mean_annual  = mean_daily * trading_days
std_annual   = std_daily * np.sqrt(trading_days)

# Sharpe corregido con tasa libre de riesgo
sharpe = (mean_annual - risk_free_rate) / std_annual

# Sortino corregido
def calc_sortino(returns_series, rf=risk_free_rate, days=trading_days):
    ret_annual = returns_series.mean() * days
    downside = returns_series[returns_series < 0]
    dd = np.sqrt((downside**2).mean()) * np.sqrt(days) if len(downside) > 0 else 1e-9
    return (ret_annual - rf) / dd

sortino = returns_hist.apply(calc_sortino)

# =============================
# 5. TENDENCIA RECIENTE (últimos ~6 meses)
# =============================
mean_rec_annual = returns_rec.mean() * trading_days
std_rec_annual  = returns_rec.std() * np.sqrt(trading_days)
sharpe_rec      = (mean_rec_annual - risk_free_rate) / std_rec_annual

tendencia = mean_rec_annual - mean_annual  # positivo = mejorando, negativo = deteriorando

# =============================
# 6. FILTRO DE LIQUIDEZ
# =============================
vol_promedio_usd = pd.Series(dtype=float)
for t in tickers:
    if t in volumenes_hist.columns and t in precios_hist.columns:
        vol_usd = (volumenes_hist[t] * precios_hist[t]).mean() / 1e6  # en millones
        vol_promedio_usd[t] = vol_usd
    else:
        vol_promedio_usd[t] = 0.0

tickers_liquidos   = vol_promedio_usd[vol_promedio_usd >= min_vol_millones].index.tolist()
tickers_iliquidos  = vol_promedio_usd[vol_promedio_usd < min_vol_millones].index.tolist()
if tickers_iliquidos:
    print(f"⚠️ Tickers con baja liquidez (<${min_vol_millones}M/día): {tickers_iliquidos}")

# =============================
# 7. DATAFRAME BASE
# =============================
df = pd.DataFrame({
    'Ticker':          tickers,
    'Rend Anual':      mean_annual.values,
    'Vol Anual':       std_annual.values,
    'Sharpe':          sharpe.values,
    'Sortino':         sortino.values,
    'Rend Reciente':   mean_rec_annual.reindex(tickers).values,
    'Vol Reciente':    std_rec_annual.reindex(tickers).values,
    'Sharpe Reciente': sharpe_rec.reindex(tickers).values,
    'Tendencia':       tendencia.reindex(tickers).values,
    'Vol USD (M)':     vol_promedio_usd.reindex(tickers).values,
    'Liquido':         [t in tickers_liquidos for t in tickers]
}).set_index('Ticker')

df = df.dropna(subset=['Rend Anual', 'Vol Anual', 'Sharpe'])

# =============================
# 8. CLASIFICACIÓN DINÁMICA
# =============================
med_ret = df['Rend Anual'].median()
med_vol = df['Vol Anual'].median()

def clasificar(row, ret_threshold, vol_threshold):
    if row['Rend Anual'] >= ret_threshold and row['Vol Anual'] <= vol_threshold:
        return '🟢 Alto Rend / Bajo Riesgo'
    elif row['Rend Anual'] < ret_threshold and row['Vol Anual'] <= vol_threshold:
        return '🟡 Bajo Rend / Bajo Riesgo'
    elif row['Rend Anual'] >= ret_threshold and row['Vol Anual'] > vol_threshold:
        return '🟠 Alto Rend / Alto Riesgo'
    else:
        return '🔴 Bajo Rend / Alto Riesgo'

df['Cat Dinámica'] = df.apply(lambda r: clasificar(r, med_ret, med_vol), axis=1)
df['Cat Fija']     = df.apply(lambda r: clasificar(r, umbral_return, umbral_vol), axis=1)

# =============================
# 9. RANKING
# =============================
df['Score'] = (
    df['Sharpe'].rank(pct=True) * 0.35 +
    df['Sortino'].rank(pct=True) * 0.25 +
    df['Tendencia'].rank(pct=True) * 0.25 +
    df['Vol USD (M)'].rank(pct=True) * 0.15
)

df_liquido = df[df['Liquido']].copy()
top5    = df_liquido.nlargest(TOP_N, 'Score')
bottom5 = df_liquido.nsmallest(TOP_N, 'Score')

# =============================
# 10. CONSOLA — RESUMEN
# =============================
colores_cat = {
    '🟢 Alto Rend / Bajo Riesgo': 'green',
    '🟡 Bajo Rend / Bajo Riesgo': 'gold',
    '🟠 Alto Rend / Alto Riesgo': 'orange',
    '🔴 Bajo Rend / Alto Riesgo': 'red'
}

print("\n" + "=" * 65)
print("📊 ANÁLISIS DE RIESGO / RENDIMIENTO")
print("=" * 65)
print(f"\n🏆 TOP {TOP_N} MEJORES (activos líquidos):")
print(f"{'Ticker':12s} {'Rend Anual':>12} {'Vol Anual':>10} {'Sharpe':>8} {'Tendencia':>10}")
print("─" * 60)
for t, row in top5.iterrows():
    tend = "📈" if row['Tendencia'] > 0 else "📉"
    print(f"{t:12s} {row['Rend Anual']:>11.2%} {row['Vol Anual']:>10.2%} "
          f"{row['Sharpe']:>8.4f} {tend} {row['Tendencia']:>+.2%}")

print(f"\n⚠️ TOP {TOP_N} PEORES:")
print(f"{'Ticker':12s} {'Rend Anual':>12} {'Vol Anual':>10} {'Sharpe':>8} {'Tendencia':>10}")
print("─" * 60)
for t, row in bottom5.iterrows():
    tend = "📈" if row['Tendencia'] > 0 else "📉"
    print(f"{t:12s} {row['Rend Anual']:>11.2%} {row['Vol Anual']:>10.2%} "
          f"{row['Sharpe']:>8.4f} {tend} {row['Tendencia']:>+.2%}")

print("\n" + "=" * 65)

# =============================
# 11. GRÁFICAS
# =============================

# ── Figura 1: Clasificación Dinámica y Fija ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Análisis de Riesgo / Rendimiento", fontsize=14, fontweight='bold')

for ax, col_cat, titulo, ret_line, vol_line in [
    (axes[0], 'Cat Dinámica', 'Clasificación Dinámica (vs mediana)', med_ret, med_vol),
    (axes[1], 'Cat Fija',     f'Clasificación Fija (Rend≥{umbral_return:.0%} / Vol≤{umbral_vol:.0%})', umbral_return, umbral_vol)
]:
    for cat in df[col_cat].unique():
        subset = df[df[col_cat] == cat]
        ax.scatter(subset['Vol Anual'], subset['Rend Anual'],
                   label=cat, s=120, alpha=0.8,
                   c=colores_cat[cat], edgecolors='k', linewidths=0.5)

    for ticker, row in df.iterrows():
        # Marcar ilíquidos con X
        marker = '✗' if not row['Liquido'] else ''
        ax.annotate(f"{ticker}{marker}",
                    (row['Vol Anual'], row['Rend Anual']),
                    fontsize=7.5, ha='right', va='bottom',
                    color='gray' if not row['Liquido'] else 'black')

    ax.axvline(vol_line, color='grey', linestyle='--', linewidth=1)
    ax.axhline(ret_line, color='grey', linestyle='--', linewidth=1)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel("Volatilidad Anualizada")
    ax.set_ylabel("Rendimiento Anualizado")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 2: Ranking Top/Bottom ────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(f"Ranking — Top {TOP_N} Mejores vs Peores (activos líquidos)", fontsize=13, fontweight='bold')

# Top 5
top5_sorted = top5.sort_values('Score')
bars = axes2[0].barh(top5_sorted.index, top5_sorted['Score'], color='green', alpha=0.7, edgecolor='k')
axes2[0].set_title(f"🏆 Top {TOP_N} Mejores", fontweight='bold')
axes2[0].set_xlabel("Score compuesto")
for bar, (t, row) in zip(bars, top5_sorted.iterrows()):
    axes2[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                  f"Sharpe: {row['Sharpe']:.2f} | Rend: {row['Rend Anual']:.1%}",
                  va='center', fontsize=8)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim(0, 1.3)

# Bottom 5
bot5_sorted = bottom5.sort_values('Score', ascending=False)
bars2 = axes2[1].barh(bot5_sorted.index, bot5_sorted['Score'], color='red', alpha=0.7, edgecolor='k')
axes2[1].set_title(f"⚠️ Top {TOP_N} Peores", fontweight='bold')
axes2[1].set_xlabel("Score compuesto")
for bar, (t, row) in zip(bars2, bot5_sorted.iterrows()):
    axes2[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                  f"Sharpe: {row['Sharpe']:.2f} | Rend: {row['Rend Anual']:.1%}",
                  va='center', fontsize=8)
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim(0, 1.3)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 3: Tendencia Reciente ────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 6))
fig3.suptitle("Tendencia Reciente (últimos 6 meses vs histórico)", fontsize=13, fontweight='bold')

df_sorted = df.sort_values('Tendencia', ascending=True)
colores_tend = ['green' if v > 0 else 'red' for v in df_sorted['Tendencia']]
bars3 = ax3.barh(df_sorted.index, df_sorted['Tendencia'] * 100,
                  color=colores_tend, alpha=0.75, edgecolor='k', linewidth=0.5)
ax3.axvline(0, color='black', linewidth=1)
ax3.set_xlabel("Cambio en Rendimiento Anual (puntos porcentuales)")
ax3.set_title("Verde = Mejorando | Rojo = Deteriorando")
for bar, val in zip(bars3, df_sorted['Tendencia'] * 100):
    x = bar.get_width()
    ax3.text(x + (0.3 if x >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
             f'{val:+.1f}%', va='center', ha='left' if x >= 0 else 'right', fontsize=7.5)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ── Figura 4: Sharpe Histórico vs Reciente ──────────────────────
fig4, ax4 = plt.subplots(figsize=(14, 6))
fig4.suptitle("Sharpe Histórico vs Reciente (6 meses)", fontsize=13, fontweight='bold')

df_sh = df[['Sharpe', 'Sharpe Reciente']].dropna().sort_values('Sharpe', ascending=False)
x = np.arange(len(df_sh))
width = 0.35
ax4.bar(x - width/2, df_sh['Sharpe'], width, label='Sharpe Histórico', color='steelblue', alpha=0.8)
ax4.bar(x + width/2, df_sh['Sharpe Reciente'], width, label='Sharpe Reciente', color='orange', alpha=0.8)
ax4.axhline(0, color='black', linewidth=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(df_sh.index, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel("Ratio de Sharpe")
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# =============================
# 12. EXPORTAR A EXCEL
# =============================
wb = Workbook()

fill_map = {
    '🟢 Alto Rend / Bajo Riesgo': PatternFill(start_color="00C851", end_color="00C851", fill_type="solid"),
    '🟡 Bajo Rend / Bajo Riesgo': PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid"),
    '🟠 Alto Rend / Alto Riesgo': PatternFill(start_color="FF8800", end_color="FF8800", fill_type="solid"),
    '🔴 Bajo Rend / Alto Riesgo': PatternFill(start_color="FF4444", end_color="FF4444", fill_type="solid"),
}

header_fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
header_font = Font(color="FFFFFF", bold=True)
border = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin')
)

def escribir_hoja(ws, df_data, col_cat, titulo):
    """Escribe una hoja de Excel con formato."""
    headers = ['Ticker', 'Categoría', 'Rend Anual', 'Vol Anual', 'Sharpe',
               'Sortino', 'Rend Reciente', 'Tendencia', 'Vol USD (M)', 'Líquido']
    ws.title = titulo

    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center')
        cell.border = border
        ws.column_dimensions[get_column_letter(col_idx)].width = 18

    for row_idx, (ticker, row) in enumerate(df_data.iterrows(), start=2):
        values = [
            ticker,
            row[col_cat],
            f"{row['Rend Anual']:.2%}",
            f"{row['Vol Anual']:.2%}",
            f"{row['Sharpe']:.4f}",
            f"{row['Sortino']:.4f}",
            f"{row['Rend Reciente']:.2%}",
            f"{row['Tendencia']:+.2%}",
            f"{row['Vol USD (M)']:.1f}",
            "✅" if row['Liquido'] else "❌"
        ]
        for col_idx, val in enumerate(values, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
            if col_idx == 2:
                cell.fill = fill_map.get(row[col_cat], PatternFill())

# Hoja Dinámica
ws_dyn = wb.active
escribir_hoja(ws_dyn, df, 'Cat Dinámica', 'Dinámica')

# Hoja Fija
ws_fix = wb.create_sheet()
escribir_hoja(ws_fix, df, 'Cat Fija', 'Inversión Real')

# Hoja Ranking
ws_rank = wb.create_sheet("Ranking")
rank_headers = ['Ranking', 'Ticker', 'Score', 'Sharpe', 'Sortino', 'Rend Anual', 'Tendencia']
for col_idx, h in enumerate(rank_headers, 1):
    cell = ws_rank.cell(row=1, column=col_idx, value=h)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')
    cell.border = border
    ws_rank.column_dimensions[get_column_letter(col_idx)].width = 16

df_rank = df_liquido.sort_values('Score', ascending=False).reset_index()
for row_idx, row in df_rank.iterrows():
    vals = [
        row_idx + 1,
        row['Ticker'],
        f"{row['Score']:.4f}",
        f"{row['Sharpe']:.4f}",
        f"{row['Sortino']:.4f}",
        f"{row['Rend Anual']:.2%}",
        f"{row['Tendencia']:+.2%}"
    ]
    color = "00C851" if row_idx < TOP_N else ("FF4444" if row_idx >= len(df_rank) - TOP_N else "FFFFFF")
    for col_idx, val in enumerate(vals, 1):
        cell = ws_rank.cell(row=row_idx + 2, column=col_idx, value=val)
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
        cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")

wb.save(ruta_salida)
print(f"\n✅ Resultados exportados a: {ruta_salida}")

# =============================
# 13. CIERRE
# =============================
print("\n✅ Análisis completado.")
plt.ioff()
input("\nPresiona Enter para cerrar todas las gráficas...")
plt.close('all')
