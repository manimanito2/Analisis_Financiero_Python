# -*- coding: utf-8 -*-
"""
Valuación de Empresas por Flujos de Caja Descontados (DCF)
Caso: The Coca-Cola Company (KO)

Autor: Alejandro Ugarte Mendoza
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# 1. DESCARGA DE DATOS
# ============================================================
ticker_symbol = "KO"
empresa = yf.Ticker(ticker_symbol)

cashflow = empresa.cashflow          # Flujo de efectivo
balance = empresa.balance_sheet      # Balance general
income_stmt = empresa.income_stmt    # Estado de resultados
info = empresa.info                  # Datos generales (precio, beta, acciones)

# ============================================================
# 2. FLUJO DE CAJA LIBRE HISTÓRICO
# ============================================================
fcf_historico = cashflow.loc['Free Cash Flow']
ocf_historico = cashflow.loc['Operating Cash Flow']
capex_historico = cashflow.loc['Capital Expenditure']  # ya viene negativo

print("=== Flujo de Caja Libre histórico (reportado por Yahoo) ===")
print(fcf_historico)

print("\n=== Verificación manual: OCF + CapEx ===")
print(ocf_historico + capex_historico)

# CAGR del periodo completo (referencia, NO se usa directo por el año atípico 2024)
n_años_hist = len(fcf_historico) - 1
cagr = (fcf_historico.iloc[0] / fcf_historico.iloc[-1]) ** (1 / n_años_hist) - 1
print(f"\nCAGR histórico (referencia, distorsionado por 2024): {cagr:.2%}")

# ============================================================
# 3. SUPUESTOS DE CRECIMIENTO (tasa conservadora justificada)
# ============================================================
# No usamos el CAGR de -17.8% (absurdo para una empresa madura como KO,
# distorsionado por la caída atípica de FCF en 2024). En su lugar,
# usamos una tasa conservadora típica de empresa de bebidas consolidada,
# que decrece gradualmente hacia un crecimiento de largo plazo.
#
# FCF BASE: en vez de usar solo el año más reciente (2025), que sigue
# "deprimido" por la caída atípica de 2024, usamos el promedio de los
# últimos 3 años. Esto da una base más representativa del FCF "normal"
# de la empresa, sin que un solo año raro distorsione toda la proyección.
fcf_base_ultimo_año = fcf_historico.iloc[0]                 # solo referencia
fcf_base_promedio_3a = fcf_historico.iloc[0:3].mean()       # 2025, 2024, 2023

fcf_base = fcf_base_promedio_3a

print(f"\n=== Elección de FCF base ===")
print(f"FCF último año (2025) — no usado como base: ${fcf_base_ultimo_año:,.0f}")
print(f"FCF promedio últimos 3 años — usado como base: ${fcf_base_promedio_3a:,.0f}")

g_inicial = 0.05    # 5% en el año 1 (recuperación moderada post-2024)
g_terminal = 0.025  # 2.5% a largo plazo (similar al crecimiento de la economía global)
n_años_proyeccion = 5

# ============================================================
# 4. COSTO DE CAPITAL PROPIO (CAPM)
# ============================================================
beta = info.get('beta', 0.6)
tasa_libre_riesgo = 0.045      # ~4.5%, bonos del Tesoro EE.UU. a 10 años
prima_riesgo_mercado = 0.055   # ~5.5%, prima histórica de mercado

costo_capital = tasa_libre_riesgo + beta * prima_riesgo_mercado

# ============================================================
# 5. COSTO DE LA DEUDA (después de impuestos)
# ============================================================
deuda_total = balance.loc['Total Debt'].iloc[0]
capital_contable = balance.loc['Stockholders Equity'].iloc[0]
acciones_circulacion = balance.loc['Ordinary Shares Number'].iloc[0]

gasto_intereses = income_stmt.loc['Interest Expense'].iloc[0]
tasa_fiscal = income_stmt.loc['Tax Rate For Calcs'].iloc[0]

costo_deuda_pretax = gasto_intereses / deuda_total
costo_deuda_despues_impuestos = costo_deuda_pretax * (1 - tasa_fiscal)

print(f"\n=== Datos del balance ===")
print(f"Deuda total: ${deuda_total:,.0f}")
print(f"Capital contable: ${capital_contable:,.0f}")
print(f"Acciones en circulación: {acciones_circulacion:,.0f}")
print(f"Beta: {beta:.2f}")
print(f"Costo de capital (CAPM): {costo_capital:.2%}")
print(f"Gasto en intereses: ${gasto_intereses:,.0f}")
print(f"Tasa fiscal: {tasa_fiscal:.2%}")
print(f"Costo de deuda (antes de impuestos): {costo_deuda_pretax:.2%}")
print(f"Costo de deuda (después de impuestos): {costo_deuda_despues_impuestos:.2%}")

# ============================================================
# 6. WACC (Costo Promedio Ponderado de Capital)
# ============================================================
valor_total = deuda_total + capital_contable
peso_deuda = deuda_total / valor_total
peso_capital = capital_contable / valor_total

wacc = peso_capital * costo_capital + peso_deuda * costo_deuda_despues_impuestos

print(f"\n=== WACC ===")
print(f"Peso de la deuda: {peso_deuda:.2%}")
print(f"Peso del capital: {peso_capital:.2%}")
print(f"WACC: {wacc:.2%}")

# ============================================================
# 7. PROYECCIÓN DE FCF A 5 AÑOS (crecimiento decreciente)
# ============================================================
tasas_crecimiento = np.linspace(g_inicial, g_terminal, n_años_proyeccion)

fcf_proyectado = []
fcf_actual = fcf_base
for g in tasas_crecimiento:
    fcf_actual = fcf_actual * (1 + g)
    fcf_proyectado.append(fcf_actual)

print(f"\n=== FCF proyectado por año ===")
for i, (fcf, g) in enumerate(zip(fcf_proyectado, tasas_crecimiento), 1):
    print(f"  Año {i}: ${fcf:,.0f}  (crecimiento: {g:.2%})")

# ============================================================
# 8. VALOR TERMINAL (perpetuidad de Gordon)
# ============================================================
fcf_año5 = fcf_proyectado[-1]
valor_terminal = (fcf_año5 * (1 + g_terminal)) / (wacc - g_terminal)

print(f"\nValor Terminal (al final del año 5): ${valor_terminal:,.0f}")

# ============================================================
# 9. DESCUENTO A VALOR PRESENTE
# ============================================================
valor_presente_fcf = []
for i, fcf in enumerate(fcf_proyectado, 1):
    vp = fcf / (1 + wacc) ** i
    valor_presente_fcf.append(vp)

valor_presente_terminal = valor_terminal / (1 + wacc) ** n_años_proyeccion
valor_empresa = sum(valor_presente_fcf) + valor_presente_terminal

print(f"\n=== Descuento a valor presente ===")
print(f"Valor presente de los flujos años 1-5: ${sum(valor_presente_fcf):,.0f}")
print(f"Valor presente del Valor Terminal: ${valor_presente_terminal:,.0f}")
print(f"VALOR DE LA EMPRESA (Enterprise Value): ${valor_empresa:,.0f}")

# ============================================================
# 10. DE ENTERPRISE VALUE A PRECIO POR ACCIÓN
# ============================================================
deuda_neta = balance.loc['Net Debt'].iloc[0]
equity_value = valor_empresa - deuda_neta

precio_justo_accion = equity_value / acciones_circulacion
precio_mercado_actual = info.get('currentPrice', info.get('regularMarketPrice'))

diferencia = (precio_justo_accion / precio_mercado_actual - 1)

print(f"\n=== Resultado Final ===")
print(f"Equity Value: ${equity_value:,.0f}")
print(f"PRECIO JUSTO SEGÚN DCF: ${precio_justo_accion:.2f}")
print(f"PRECIO ACTUAL DE MERCADO: ${precio_mercado_actual:.2f}")
print(f"Diferencia: {diferencia:+.2%}")
