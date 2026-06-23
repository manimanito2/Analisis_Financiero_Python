# -*- coding: utf-8 -*-
"""
Generador de Excel "sucio" con movimientos bancarios simulados.
Objetivo: practicar limpieza de datos con pandas antes de llevarlos a Power BI.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

fake = Faker('es_MX')
random.seed(42)
Faker.seed(42)

N = 1500  # número de movimientos a generar
RUTA_SALIDA = r'C:\Users\1552\Downloads'  # carpeta donde se guarda el Excel

categorias = ['Nómina', 'Renta', 'Supermercado', 'Restaurante', 'Transferencia',
              'Pago de Tarjeta', 'Servicios', 'Gasolina', 'Entretenimiento',
              'Salud', 'Educación', 'Comisión Bancaria', 'Inversión', 'Retiro Cajero']

# Variantes "sucias" de la misma categoría (mayúsculas, espacios, typos)
categorias_sucias = {
    'Nómina': ['Nómina', 'NOMINA', 'nomina ', 'Nomina'],
    'Renta': ['Renta', 'RENTA', ' renta', 'Renta '],
    'Supermercado': ['Supermercado', 'SUPERMERCADO', 'super mercado', 'Supermercado '],
    'Restaurante': ['Restaurante', 'restaurante', 'RESTAURANTE ', 'Restaurant'],
    'Transferencia': ['Transferencia', 'TRANSFERENCIA', 'transferencia ', 'Transf.'],
    'Pago de Tarjeta': ['Pago de Tarjeta', 'PAGO TARJETA', 'pago de tarjeta', 'Pago Tarjeta'],
    'Servicios': ['Servicios', 'SERVICIOS', ' servicios', 'Servicio'],
    'Gasolina': ['Gasolina', 'GASOLINA', 'gasolina ', 'Gas'],
    'Entretenimiento': ['Entretenimiento', 'ENTRETENIMIENTO', 'entretenimiento ', 'Entrenimiento'],
    'Salud': ['Salud', 'SALUD', ' salud', 'Salud '],
    'Educación': ['Educación', 'EDUCACION', 'educacion', 'Educación '],
    'Comisión Bancaria': ['Comisión Bancaria', 'COMISION BANCARIA', 'comision bancaria', 'Com. Bancaria'],
    'Inversión': ['Inversión', 'INVERSION', 'inversion ', 'Inversion'],
    'Retiro Cajero': ['Retiro Cajero', 'RETIRO CAJERO', 'retiro cajero', 'Retiro ATM'],
}

bancos = ['BBVA', 'Banorte', 'Santander', 'HSBC', 'Banamex', 'Scotiabank']
tipos_mov = ['Cargo', 'Abono', 'CARGO', 'abono', 'Cargo ', 'Abono ']
formatos_fecha = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y']

filas = []
cuentas = [fake.bban() for _ in range(8)]

fecha_inicio = datetime(2024, 1, 1)

for i in range(N):
    cat_real = random.choice(categorias)
    cat_sucia = random.choice(categorias_sucias[cat_real])

    fecha = fecha_inicio + timedelta(days=random.randint(0, 540))
    fmt = random.choice(formatos_fecha)
    fecha_str = fecha.strftime(fmt)

    # Monto: a veces negativo, a veces con texto, a veces con $ y comas
    monto = round(random.uniform(50, 25000), 2)
    es_cargo = 'Cargo' in cat_real or random.random() < 0.5
    if cat_real in ['Nómina', 'Transferencia', 'Inversión'] and random.random() < 0.6:
        es_cargo = False

    if es_cargo:
        monto = -abs(monto)

    # Ensuciar el formato del monto en distintas formas
    r = random.random()
    if r < 0.15:
        monto_str = f"${monto:,.2f}"
    elif r < 0.25:
        monto_str = f"{monto:.2f} MXN"
    elif r < 0.30:
        monto_str = "N/D"
    elif r < 0.33:
        monto_str = ""
    else:
        monto_str = monto

    descripcion = random.choice([
        fake.company(),
        fake.catch_phrase(),
        f"PAGO {fake.company().upper()}",
        f"SPEI {fake.first_name()} {fake.last_name()}",
    ])

    cuenta = random.choice(cuentas)
    # Ensuciar formato de cuenta (con espacios, guiones)
    if random.random() < 0.3:
        cuenta = cuenta.replace(" ", "")
    if random.random() < 0.2:
        cuenta = cuenta + "  "

    banco = random.choice(bancos)
    tipo = random.choice(tipos_mov)

    fila = {
        'Fecha': fecha_str,
        'Cuenta': cuenta,
        'Banco': banco,
        'Tipo': tipo,
        'Categoria': cat_sucia,
        'Descripcion': descripcion,
        'Monto': monto_str,
        'Saldo': round(random.uniform(-5000, 150000), 2),
    }
    filas.append(fila)

df = pd.DataFrame(filas)

# Inyectar duplicados exactos (~3%)
dup_idx = df.sample(frac=0.03, random_state=1).index
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# Inyectar nulos en varias columnas (~5%)
for col in ['Descripcion', 'Banco', 'Saldo']:
    null_idx = df.sample(frac=0.05, random_state=random.randint(1, 999)).index
    df.loc[null_idx, col] = None

# Inyectar filas completamente vacías (separadores accidentales de Excel)
filas_vacias = pd.DataFrame([{}] * 5, columns=df.columns)
df = pd.concat([df, filas_vacias], ignore_index=True)

# Inyectar algunos outliers absurdos en Saldo
df['Saldo'] = df['Saldo'].astype(object)
outlier_idx = df.sample(n=4, random_state=7).index
df.loc[outlier_idx, 'Saldo'] = [99999999, -99999999, 1e12, None]

# Mezclar el orden de las filas para que no se note el patrón de inyección
df = df.sample(frac=1, random_state=3).reset_index(drop=True)

# Mezclar el orden de las filas para que no se note el patrón de inyección
df = df.sample(frac=1, random_state=3).reset_index(drop=True)

os.makedirs(RUTA_SALIDA, exist_ok=True)
nombre_excel = f"Movimientos_Bancarios_Sucio_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
ruta_excel = os.path.join(RUTA_SALIDA, nombre_excel)

df.to_excel(ruta_excel, index=False, sheet_name='Movimientos')
print(f"Archivo guardado en: {ruta_excel}")
print(f"Filas generadas: {len(df)}")
print(df.head(10))
