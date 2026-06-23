# predictor_crisis_Reinods.py
# Experimento: Predictor de Crisis Financieras basado en dinámica de fluidos
# Autor: A. Ugarte (nombre temporal) 😄
# Módulo 1: Descarga de datos históricos SP500 + VIX desde 1990
# Módulo 2: Filtro HP para identificación y etiquetado de crisis

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import matplotlib.patches as mpatches
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import StandardScaler

# ============================================================
# MÓDULO 1: DESCARGA DE DATOS
# ============================================================

def descargar_datos(inicio="1990-01-01", fin=None):
    """Descarga SP500 y VIX histórico en frecuencia semanal"""
    print("Descargando datos históricos...")
    
    sp500_raw = yf.download("^GSPC", start=inicio, end=fin, auto_adjust=True)
    vix_raw   = yf.download("^VIX",  start=inicio, end=fin, auto_adjust=True)
    
    if isinstance(sp500_raw.columns, pd.MultiIndex):
        sp500 = sp500_raw["Close"]["^GSPC"]
        vix   = vix_raw["Close"]["^VIX"]
    else:
        sp500 = sp500_raw["Close"]
        vix   = vix_raw["Close"]

    # Resamplear a semanal — elimina el ruido diario
    sp500 = sp500.resample("W").last()
    vix   = vix.resample("W").mean()

    df = pd.DataFrame({"SP500": sp500, "VIX": vix}).dropna()
    print(f"Datos descargados: {len(df)} semanas desde {df.index[0].date()} hasta {df.index[-1].date()}")
    return df

# ============================================================
# MÓDULO 2: FILTRO HP — ETIQUETADO DE CRISIS
# ============================================================

# DESPUÉS — pon esto:
from statsmodels.tsa.filters.hp_filter import hpfilter

def filtro_hp(serie, lamb=1600000):
    """Filtro HP usando statsmodels — rápido"""
    ciclo, tendencia = hpfilter(serie.values, lamb=lamb)
    return tendencia, ciclo

def etiquetar_crisis(df, lamb=107000, umbral_percentil=20):
    """
    Usa el componente cíclico del HP para etiquetar regímenes:
    - Turbulento (crisis): ciclo por debajo del percentil umbral
    - Recuperación: ciclo subiendo desde mínimo local
    - Laminar: resto del tiempo
    """
    print("Aplicando filtro HP y etiquetando crisis...")
    
    tendencia, ciclo = filtro_hp(np.log(df["SP500"]), lamb=lamb)
    df["Tendencia_HP"] = np.exp(tendencia)
    df["Ciclo_HP"]     = ciclo
    
    # Suavizar el ciclo con media móvil de 21 días antes de etiquetar
    df["Ciclo_HP_suavizado"] = pd.Series(ciclo, index=df.index).rolling(window=21, center=False).mean()

    # Usar el ciclo suavizado para el umbral
    umbral = np.percentile(df["Ciclo_HP_suavizado"].dropna(), umbral_percentil)
    df["Regimen"] = 0
    df.loc[df["Ciclo_HP_suavizado"] < umbral, "Regimen"] = 1
    
    
    # Etiquetado inicial
    # 0 = Laminar, 1 = Turbulento, 2 = Recuperación
    df["Regimen"] = 0
    df.loc[df["Ciclo_HP"] < umbral, "Regimen"] = 1
    
    # Filtrar episodios turbulentos cortos (ruido)
    duracion_minima = 5  # días
    en_crisis = False
    inicio_crisis = None

    for i in range(len(df)):
        if df["Regimen"].iloc[i] == 1 and not en_crisis:
            en_crisis = True
            inicio_crisis = i
        elif df["Regimen"].iloc[i] != 1 and en_crisis:
            duracion = i - inicio_crisis
            if duracion < duracion_minima:
                # Era ruido, lo regresamos a laminar
                df.iloc[inicio_crisis:i, df.columns.get_loc("Regimen")] = 0
            en_crisis = False
            inicio_crisis = None
    
    # Identificar recuperación: saliendo de turbulencia pero ciclo aún negativo
    en_recuperacion = False
    for i in range(1, len(df)):
        if df["Regimen"].iloc[i-1] == 1 and df["Regimen"].iloc[i] == 0:
            en_recuperacion = True
        if en_recuperacion:
            if df["Ciclo_HP"].iloc[i] >= 0:
                en_recuperacion = False
            else:
                df.iloc[i, df.columns.get_loc("Regimen")] = 2
    
    return df, umbral

def visualizar_etiquetado(df, umbral):
    """Grafica SP500 con regímenes coloreados"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Predictor de Crisis Reinods — Etiquetado HP", fontsize=14, fontweight="bold")
    
    # Colores por régimen
    colores = {0: "lightgreen", 1: "salmon", 2: "lightyellow"}
    nombres  = {0: "Laminar", 1: "Turbulento", 2: "Recuperación"}
    
    # Grafica SP500 con fondo coloreado
    for regimen, color in colores.items():
        mask = df["Regimen"] == regimen
        ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                        where=mask, alpha=0.3, color=color, label=nombres[regimen])
    
    ax1.plot(df.index, df["SP500"], color="black", linewidth=0.8, label="SP500")
    ax1.plot(df.index, df["Tendencia_HP"], color="blue", linewidth=1.2,
             linestyle="--", label="Tendencia HP")
    ax1.set_ylabel("SP500")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Grafica componente cíclico
    ax2.plot(df.index, df["Ciclo_HP_suavizado"], color="purple", linewidth=0.8, label="Ciclo HP suavizado")
    ax2.axhline(umbral, color="red", linestyle="--", linewidth=1, label=f"Umbral crisis ({umbral:.4f})")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Componente Cíclico")
    ax2.set_xlabel("Fecha")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de etiquetado
    total = len(df)
    for r, nombre in nombres.items():
        cantidad = (df["Regimen"] == r).sum()
        print(f"{nombre}: {cantidad} días ({100*cantidad/total:.1f}%)")

# ============================================================
# MÓDULO 3: REYNOLDS FINANCIERO
# ============================================================

def calcular_reynolds(df, ventana_momentum=4):
    """
    Calcula el número de Reynolds financiero.
    Re = (momentum × volumen_normalizado) / VIX_normalizado
    """
    print("\n[3/X] Calculando Reynolds Financiero...")

    # Momentum: retorno porcentual en ventana de N semanas
    df["Momentum"] = df["SP500"].pct_change(ventana_momentum)

    # Volumen normalizado (0 a 1)
    # Descargamos volumen del SP500
    sp500_vol = yf.download("^GSPC", start=df.index[0], end=df.index[-1], 
                             auto_adjust=True)["Volume"]
    # DESPUÉS — pon esto:
    sp500_vol = sp500_vol.resample("W").sum()
    sp500_vol = sp500_vol.reindex(df.index, method="nearest")
    vol_ma = sp500_vol.rolling(window=52).mean()
    vol_relativo = (sp500_vol / vol_ma).fillna(1.0)
    df["Volumen"] = vol_relativo.values

    # VIX normalizado
    vix_norm = (df["VIX"] - df["VIX"].min()) / (df["VIX"].max() - df["VIX"].min())
    df["VIX_norm"] = vix_norm

    # Reynolds: evitar división por cero
    df["Reynolds"] = df["Momentum"].abs() * df["VIX_norm"] * df["Volumen"]
    # Winsorizar — cortar outliers extremos al percentil 1 y 99
    p1  = df["Reynolds"].quantile(0.01)
    p99 = df["Reynolds"].quantile(0.99)
    df["Reynolds"] = df["Reynolds"].clip(p1, p99)

    # Normalizar Reynolds final
    df["Reynolds"] = (df["Reynolds"] - df["Reynolds"].mean()) / df["Reynolds"].std()
    # Suavizar Reynolds con media móvil de 8 semanas
    df["Reynolds"] = df["Reynolds"].rolling(window=8, center=False).mean()

    # Umbral crítico de Reynolds
    umbral_reynolds = df["Reynolds"].quantile(0.80)
    df["Reynolds_alerta"] = (df["Reynolds"] > umbral_reynolds).astype(int)
    print(f"  Umbral crítico Reynolds: {umbral_reynolds:.3f}")

    # Alerta actual
    reynolds_actual = df["Reynolds"].dropna().iloc[-1]
    if reynolds_actual > umbral_reynolds:
        print(f"  ⚠️  ALERTA: Reynolds en zona de riesgo ({reynolds_actual:.3f} > {umbral_reynolds:.3f})")
    else:
        print(f"  ✅ Reynolds en zona segura ({reynolds_actual:.3f})")

    print("✓ Reynolds calculado")
    return df

def visualizar_reynolds(df):
    """Grafica Reynolds Financiero vs regímenes"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Predictor de Crisis Reinods — Reynolds Financiero", fontsize=14, fontweight="bold")

    colores = {0: "lightgreen", 1: "salmon", 2: "lightyellow"}
    nombres  = {0: "Laminar", 1: "Turbulento", 2: "Recuperación"}

    # SP500 con regímenes
    for regimen, color in colores.items():
        mask = df["Regimen"] == regimen
        ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                        where=mask, alpha=0.3, color=color, label=nombres[regimen])
    ax1.plot(df.index, df["SP500"], color="black", linewidth=0.8)
    ax1.set_ylabel("SP500")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Reynolds
    for regimen, color in colores.items():
        mask = df["Regimen"] == regimen
        ax2.fill_between(df.index, df["Reynolds"].min(), df["Reynolds"].max(),
                        where=mask, alpha=0.3, color=color)
    ax2.plot(df.index, df["Reynolds"], color="darkblue", linewidth=0.8, label="Reynolds")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Reynolds Financiero (normalizado)")
    ax2.set_xlabel("Fecha")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# MÓDULO 4: HMM — DETECCIÓN DE REGÍMENES
# ============================================================

from hmmlearn.hmm import GaussianHMM

def entrenar_hmm(df, n_estados=3):
    """
    Entrena HMM Gaussiano con 3 estados ocultos.
    Observaciones: Ciclo HP suavizado + VIX normalizado + Reynolds
    """
    print("\n[5/X] Entrenando HMM...")

    # Preparar observaciones — eliminar NaN
    features = ["Ciclo_HP_suavizado", "VIX_norm", "Reynolds"]
    datos = df[features].dropna()

    # Normalizar VIX para que esté en escala similar
    # DESPUÉS — pon esto:
    scaler = StandardScaler()
    X = scaler.fit_transform(datos.values)

    # Entrenar HMM
    # DESPUÉS — pon esto:
    # DESPUÉS:
    medias_iniciales = np.array([
        X[df.loc[datos.index, "Regimen"] == 0].mean(axis=0),  # Tranquilo
        X[(df.loc[datos.index, "Regimen"] == 1) | 
        (df.loc[datos.index, "Regimen"] == 2)].mean(axis=0), # Turbulento+Recuperación
    ])

    modelo = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        init_params="stc",
        params="stmc"
    )
    modelo.means_ = medias_iniciales
    modelo.fit(X)

    # Predecir estados
    estados = modelo.predict(X)

    # Asignar estados al dataframe
    df.loc[datos.index, "Estado_HMM"] = estados

    print(f"✓ HMM entrenado — Score: {modelo.score(X):.2f}")
    print(f"  Convergió: {modelo.monitor_.converged}")

    return df, modelo, datos.index

def visualizar_hmm(df):
    """Grafica estados HMM vs regímenes HP"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Predictor de Crisis Reinods — Estados HMM", fontsize=14, fontweight="bold")

    # SP500 con regímenes HP
    colores_hp = {0: "lightgreen", 1: "salmon", 2: "lightyellow"}
    nombres_hp = {0: "Laminar HP", 1: "Turbulento HP", 2: "Recuperación HP"}
    for regimen, color in colores_hp.items():
        mask = df["Regimen"] == regimen
        ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                        where=mask, alpha=0.3, color=color, label=nombres_hp[regimen])
    ax1.plot(df.index, df["SP500"], color="black", linewidth=0.8)
    ax1.set_ylabel("SP500")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Estados HMM
    colores_hmm = {0: "lightblue", 1: "orange", 2: "lightpink"}
    nombres_hmm = {0: "Estado HMM 0", 1: "Estado HMM 1", 2: "Estado HMM 2"}
    for estado, color in colores_hmm.items():
        mask = df["Estado_HMM"] == estado
        ax2.fill_between(df.index, 0, 1,
                        where=mask, alpha=0.5, color=color,
                        label=nombres_hmm[estado], transform=ax2.get_xaxis_transform())
    ax2.plot(df.index, df["Reynolds"], color="darkblue", linewidth=0.8, label="Reynolds", alpha=0.5)
    ax2.set_ylabel("Estados HMM")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# MÓDULO 5: KALMAN — ESTIMACIÓN DE DURACIÓN
# ============================================================

def estimar_duracion_kalman(df, idx_hmm):
    """
    Usa Kalman para estimar duración restante de la crisis actual.
    Estado oculto: duración esperada del régimen
    Observación: días transcurridos en régimen actual
    """
    print("\n[6/X] Estimando duración con Kalman...")

    # Calcular duración histórica de cada episodio turbulento (HP)
    duraciones = []
    en_crisis = False
    inicio = None

    for i, (idx, row) in enumerate(df.iterrows()):
        if row["Regimen"] == 1 and not en_crisis:
            en_crisis = True
            inicio = i
        elif row["Regimen"] != 1 and en_crisis:
            duraciones.append(i - inicio)
            en_crisis = False

    duraciones = np.array(duraciones)
    duracion_media  = duraciones.mean()
    duracion_std    = duraciones.std()

    print(f"  Duración media crisis histórica: {duracion_media:.1f} semanas")
    print(f"  Duración std: {duracion_std:.1f} semanas")
    print(f"  Min: {duraciones.min()} | Max: {duraciones.max()} semanas")

    # Kalman simple 1D
    # Estado: duración estimada restante
    # Observación: VIX actual normalizado como proxy de intensidad

    # Parámetros Kalman
    x = duracion_media      # estimación inicial
    P = duracion_std**2     # incertidumbre inicial
    Q = 0.5                 # ruido del proceso
    R = 2.0                 # ruido de observación
    H = 1.0                 # matriz de observación

    estimaciones = []
    incertidumbres = []

    for idx in df.loc[idx_hmm].index:
        row = df.loc[idx]

        # Predicción
        x_pred = x
        P_pred = P + Q

        # Observación: VIX como señal de intensidad de crisis
        z = row["VIX_norm"] * duracion_media * 2

        # Actualización
        K = P_pred * H / (H * P_pred * H + R)
        x = x_pred + K * (z - H * x_pred)
        P = (1 - K * H) * P_pred

        estimaciones.append(max(0, x))
        incertidumbres.append(np.sqrt(P))

    df.loc[idx_hmm, "Duracion_estimada"] = estimaciones
    df.loc[idx_hmm, "Incertidumbre"]     = incertidumbres

    # Régimen y duración actual
    regimen_actual = df["Regimen"].iloc[-1]
    estado_hmm     = df["Estado_HMM"].dropna().iloc[-1]
    duracion_est   = df["Duracion_estimada"].dropna().iloc[-1]
    incert         = df["Incertidumbre"].dropna().iloc[-1]

    print(f"\n{'='*50}")
    print(f"RÉGIMEN HP ACTUAL:  { {0:'🟢 LAMINAR', 1:'🔴 TURBULENTO', 2:'🟡 RECUPERACIÓN'}[regimen_actual]}")
    print(f"ESTADO HMM ACTUAL:  { {0:'🔵 TRANQUILO', 1:'🟠 ACTIVO'}[int(estado_hmm)] }")
    print(f"DURACIÓN ESTIMADA:  {duracion_est:.1f} semanas ± {incert:.1f}")
    print(f"VIX ACTUAL:         {df['VIX'].iloc[-1]:.2f}")
    # DESPUÉS:
    print(f"REYNOLDS ACTUAL:    {df['Reynolds'].dropna().iloc[-1]:.3f}")
    print(f"{'='*50}")

# Señal unificada
    regimen_hp  = df["Regimen"].iloc[-1]
    estado_hmm  = int(df["Estado_HMM"].dropna().iloc[-1])
    reynolds_ok = df["Reynolds"].dropna().iloc[-1] > df["Reynolds"].quantile(0.80)

    score = 0
    if regimen_hp == 1: score += 2      # HP turbulento — peso alto
    if regimen_hp == 2: score += 1      # HP recuperación
    if estado_hmm == 1: score += 1      # HMM activo
    if reynolds_ok:     score += 1      # Reynolds elevado

    niveles = {0: "🟢 CALMA TOTAL", 1: "🟡 PRECAUCIÓN", 
               2: "🟠 ALERTA MODERADA", 3: "🔴 TURBULENCIA", 
               4: "🚨 CRISIS ACTIVA"}
    print(f"SEÑAL UNIFICADA:    {niveles.get(score, '🚨 CRISIS ACTIVA')} (score: {score}/4)")

    return df, duracion_media, duraciones



def visualizar_kalman(df, duracion_media):
    """Grafica estimación de duración Kalman"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("Predictor de Crisis Reinods — Duración Kalman", fontsize=14, fontweight="bold")

    # SP500 con regímenes HP
    colores_hp = {0: "lightgreen", 1: "salmon", 2: "lightyellow"}
    for regimen, color in colores_hp.items():
        mask = df["Regimen"] == regimen
        ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                        where=mask, alpha=0.3, color=color)
    ax1.plot(df.index, df["SP500"], color="black", linewidth=0.8)
    ax1.set_ylabel("SP500")
    ax1.grid(True, alpha=0.3)

    # Duración estimada Kalman
    ax2.plot(df.index, df["Duracion_estimada"], color="darkred",
             linewidth=1.2, label="Duración estimada (semanas)")
    ax2.fill_between(df.index,
                     df["Duracion_estimada"] - df["Incertidumbre"],
                     df["Duracion_estimada"] + df["Incertidumbre"],
                     alpha=0.2, color="red", label="Incertidumbre ±1σ")
    ax2.axhline(duracion_media, color="blue", linestyle="--",
                linewidth=1, label=f"Media histórica ({duracion_media:.1f} sem)")

    for regimen, color in colores_hp.items():
        mask = df["Regimen"] == regimen
        ax2.fill_between(df.index, 0, df["Duracion_estimada"].max(),
                        where=mask, alpha=0.1, color=color)

    ax2.set_ylabel("Semanas estimadas")
    ax2.set_xlabel("Fecha")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# MÓDULO 6: DETECCIÓN DE RECUPERACIÓN
# ============================================================

def detectar_recuperacion(df):
    """
    Detecta fase de recuperación post-crisis:
    - Sale de régimen turbulento HP
    - Ciclo HP aún negativo pero subiendo
    - Reynolds bajando desde pico
    - HMM aún en estado activo
    """
    print("\n[7/X] Detectando fases de recuperación...")

    df["Recuperacion_Reinods"] = 0  # 0=No, 1=Recuperación activa

    en_recuperacion = False
    reynolds_pico   = None

    for i in range(2, len(df)):
        row      = df.iloc[i]
        row_prev = df.iloc[i-1]

        # Condición de entrada a recuperación:
        # HP sale de turbulento + ciclo aún negativo + Reynolds bajando

        salida_turbulencia = (row_prev["Regimen"] == 1 and row["Regimen"] == 0)
        ciclo_negativo     = row["Ciclo_HP"] < 0  # usar ciclo sin suavizar
        reynolds_val       = row["Reynolds"]

        if salida_turbulencia:
            en_recuperacion = True
            reynolds_pico   = reynolds_val

        if en_recuperacion:
            # Condición de salida: ciclo HP vuelve a positivo
            if en_recuperacion:
                if df["Ciclo_HP_suavizado"].iloc[i] >= 0.02:  # umbral más alto
                    en_recuperacion = False
                    reynolds_pico   = None
            else:
                df.iloc[i, df.columns.get_loc("Recuperacion_Reinods")] = 1
    
    salidas = 0
    for i in range(1, len(df)):
        if df["Regimen"].iloc[i-1] == 1 and df["Regimen"].iloc[i] == 0:
            salidas += 1
            ciclo_val = df["Ciclo_HP"].iloc[i]
            print(f"  Salida turbulencia en {df.index[i].date()} — Ciclo_HP: {ciclo_val:.4f}")
    print(f"  Total salidas de turbulencia: {salidas}")
   
    # Estadísticas
    total_recuperacion = df["Recuperacion_Reinods"].sum()
    print(f"  Semanas en recuperación detectadas: {total_recuperacion}")

    # Estado actual
    estado_rec = df["Recuperacion_Reinods"].iloc[-1]
    if estado_rec == 1:
        print("  🟡 ACTUALMENTE EN FASE DE RECUPERACIÓN")
    else:
        print("  ✅ No en recuperación activa")
# Al final de detectar_recuperacion, antes del return:
    print(f"  Ciclo HP actual: {df['Ciclo_HP'].iloc[-1]:.4f}")
    print(f"  Ciclo HP suavizado actual: {df['Ciclo_HP_suavizado'].iloc[-1]:.4f}")
    print(f"  Régimen últimas 8 semanas: {df['Regimen'].tail(8).tolist()}")
    
    return df

def visualizar_final(df, duracion_media):
    """Dashboard final unificado"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("Predictor de Crisis Reinods — Dashboard Final", 
                 fontsize=14, fontweight="bold")

    # Colores base
    colores = {0: "lightgreen", 1: "salmon", 2: "lightyellow"}

    # --- Panel 1: SP500 con todos los regímenes ---
    ax1 = axes[0]
    for reg, color in colores.items():
        mask = df["Regimen"] == reg
        ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                        where=mask, alpha=0.25, color=color)

    # Sombrear recuperación Reinods
    mask_rec = df["Recuperacion_Reinods"] == 1
    ax1.fill_between(df.index, df["SP500"].min(), df["SP500"].max(),
                    where=mask_rec, alpha=0.4, color="gold", label="Recuperación Reinods")

    ax1.plot(df.index, df["SP500"], color="black", linewidth=0.8, label="SP500")
    ax1.plot(df.index, df["Tendencia_HP"], color="blue", linewidth=1,
             linestyle="--", alpha=0.5, label="Tendencia HP")
    ax1.set_ylabel("SP500")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Reynolds + VIX ---
    ax2 = axes[1]
    ax2.plot(df.index, df["Reynolds"], color="darkblue", 
             linewidth=0.8, label="Reynolds", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)

    ax2b = ax2.twinx()
    ax2b.plot(df.index, df["VIX"], color="red", linewidth=0.8, 
              alpha=0.5, label="VIX")
    ax2b.set_ylabel("VIX", color="red")

    for reg, color in colores.items():
        mask = df["Regimen"] == reg
        ax2.fill_between(df.index, df["Reynolds"].min(), df["Reynolds"].max(),
                        where=mask, alpha=0.1, color=color)

    ax2.set_ylabel("Reynolds")
    ax2.legend(loc="upper left", fontsize=7)
    ax2b.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Duración estimada Kalman ---
    ax3 = axes[2]
    ax3.plot(df.index, df["Duracion_estimada"], color="darkred",
             linewidth=1, label="Duración estimada (sem)")
    ax3.fill_between(df.index,
                     df["Duracion_estimada"] - df["Incertidumbre"],
                     df["Duracion_estimada"] + df["Incertidumbre"],
                     alpha=0.2, color="red", label="±1σ")
    ax3.axhline(duracion_media, color="blue", linestyle="--",
                linewidth=1, label=f"Media histórica ({duracion_media:.1f} sem)")

    for reg, color in colores.items():
        mask = df["Regimen"] == reg
        ax3.fill_between(df.index, 0, df["Duracion_estimada"].max(),
                        where=mask, alpha=0.1, color=color)

    ax3.set_ylabel("Semanas estimadas")
    ax3.set_xlabel("Fecha")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Anotación estado actual
    regimen_actual = df["Regimen"].iloc[-1]
    rec_actual     = df["Recuperacion_Reinods"].iloc[-1]
    dur_actual     = df["Duracion_estimada"].dropna().iloc[-1]
    vix_actual     = df["VIX"].iloc[-1]

    estado_txt = {0: "LAMINAR", 1: "TURBULENTO", 2: "RECUPERACIÓN HP"}[regimen_actual]
    if rec_actual == 1:
        estado_txt += " | 🟡 RECUPERACIÓN REINODS"

    fig.text(0.5, 0.01,
             f"Estado actual: {estado_txt} | Duración est: {dur_actual:.1f} sem | VIX: {vix_actual:.2f}",
             ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


# ============================================================
# MÓDULO 8: BACKTESTING — ANTICIPACIÓN DEL REYNOLDS
# ============================================================

def backtesting_reynolds(df):
    """
    Verifica cuántas semanas antes anticipó el Reynolds
    cada crisis histórica conocida.
    """
    print("\n[8/8] Backtesting Reynolds...")

    crisis_conocidas = {
        "México 1994":       ("1994-01-01", "1995-06-01"),
        "Puntocom 2000":     ("2000-03-01", "2002-10-01"),
        "Crisis 2008":       ("2007-10-01", "2009-03-01"),
        "Deuda Europa 2011": ("2011-07-01", "2012-01-01"),
        "Corrección 2018":   ("2018-09-01", "2018-12-31"),
        "COVID 2020":        ("2020-02-01", "2020-04-01"),
        "Inflación 2022":    ("2022-01-01", "2022-10-01"),
        "Corrección 2025":   ("2025-02-01", "2026-04-01"),
    }

    resultados = []

    for nombre, (inicio, fin) in crisis_conocidas.items():
        inicio_dt = pd.to_datetime(inicio)
        fin_dt    = pd.to_datetime(fin)
        ventana   = df[(df.index >= inicio_dt - pd.Timedelta(weeks=8)) &
                       (df.index <= fin_dt)]

        if len(ventana) == 0:
            continue

        idx_pico = ventana["Reynolds"].idxmax()
        semanas  = (idx_pico - inicio_dt).days // 7

        # Anticipación del HMM — agrega aquí:
        alerta_hmm = df[(df.index < inicio_dt) & (df["Estado_HMM"] == 1)].tail(1)
        if len(alerta_hmm) > 0:
            semanas_hmm = (inicio_dt - alerta_hmm.index[0]).days // 7
        else:
            semanas_hmm = None

        resultados.append({
            "Crisis":            nombre,
            "Pico_Reynolds":     idx_pico.date(),
            "Semanas_vs_inicio": semanas,
            "Reynolds_max":      round(ventana["Reynolds"].max(), 3),
            "Semanas_HMM":       semanas_hmm
        })
    # Imprimir resultados
    print(f"\n{'='*62}")
    print(f"{'Crisis':<22} {'Pico Reynolds':<15} {'Re vs inicio':>12} {'Re max':>7} {'HMM antes':>9}")
    print(f"{'='*70}")
    for r in resultados:
        sem     = r['Semanas_vs_inicio']
        sem_hmm = f"{r['Semanas_HMM']} sem" if r['Semanas_HMM'] else "N/A"
        indicador = "⚠️ ANTES" if sem < 0 else "✅ DURANTE" if sem <= 4 else "❌ TARDE"
        print(f"{r['Crisis']:<22} {str(r['Pico_Reynolds']):<15} {sem:>8} sem  {r['Reynolds_max']:>6}  {sem_hmm:>8} HMM  {indicador}")
    print(f"{'='*70}")

    anticipaciones = [r["Semanas_vs_inicio"] for r in resultados if r["Semanas_vs_inicio"] < 0]
    print(f"\nCrisis anticipadas: {len(anticipaciones)}/{len(resultados)}")
    if anticipaciones:
        print(f"Anticipación promedio: {abs(np.mean(anticipaciones)):.1f} semanas antes")

    anticipaciones_hmm = [r["Semanas_HMM"] for r in resultados if r["Semanas_HMM"]]
    print(f"Crisis anticipadas por HMM:  {len(anticipaciones_hmm)}/{len(resultados)}")
    if anticipaciones_hmm:
        print(f"Anticipación promedio HMM:   {np.mean(anticipaciones_hmm):.1f} semanas antes")

    return pd.DataFrame(resultados)

# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("PREDICTOR DE CRISIS REINODS")
    print("=" * 50)
    
    print("\n[1/3] Descargando datos históricos...")
    df = descargar_datos(inicio="1990-01-01")
    print("✓ Datos listos")
    
    print("\n[2/3] Aplicando filtro HP (puede tardar ~2 min)...")
    df, umbral = etiquetar_crisis(df, lamb=270400, umbral_percentil=15)
    print("✓ Etiquetado listo")
    
    print("\n[3/3] Generando visualización...")
    visualizar_etiquetado(df, umbral)
    print("✓ Listo")

    print("\n[4/4] Calculando Reynolds Financiero...")
    df = calcular_reynolds(df)
    visualizar_reynolds(df)
    
    print("\n[5/5] Entrenando HMM...")
    df, modelo_hmm, idx_hmm = entrenar_hmm(df, n_estados=2)
    visualizar_hmm(df)
    
    print("\n[6/6] Estimando duración con Kalman...")
    df, duracion_media, duraciones = estimar_duracion_kalman(df, idx_hmm)
    visualizar_kalman(df, duracion_media)
    
    # Limpiar NaN
    df["Ciclo_HP_suavizado"] = df["Ciclo_HP_suavizado"].bfill().ffill()
    df["Momentum"]           = df["Momentum"].bfill().ffill()
    df["Reynolds"]           = df["Reynolds"].bfill().ffill()
    df["Estado_HMM"]         = df["Estado_HMM"].bfill().ffill()
    df["Duracion_estimada"]  = df["Duracion_estimada"].bfill().ffill()
    df["Incertidumbre"]      = df["Incertidumbre"].bfill().ffill()
    
    print("\n[7/7] Generando dashboard final...")
    df = detectar_recuperacion(df)
    visualizar_final(df, duracion_media)
    
    print("\n" + "=" * 50)
    print("RÉGIMEN ACTUAL:", {0: "🟢 LAMINAR", 1: "🔴 TURBULENTO", 2: "🟡 RECUPERACIÓN"}[df["Regimen"].iloc[-1]])
    print(f"VIX ACTUAL: {round(df['VIX'].iloc[-1], 2)}")
    print("=" * 50)
    
    print("\n[8/8] Ejecutando backtesting...")
    df_backtest = backtesting_reynolds(df)
    
    print("\n--- NaN por columna ---")
    print(df.isnull().sum())