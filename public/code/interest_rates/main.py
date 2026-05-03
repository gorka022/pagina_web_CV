"""
Pricing Interest Rate Derivatives and Underlying Construction
=============================================================

Seminario QfB 2025/2026 — Ejercicio 1

Apartado A: Valoración de un Interest Rate Swap Fixed/Floating (colateralizado, EUR).
Apartado B: Valoración de un Interest Rate Cap (modelo Normal / Bachelier).
Apartado C: Volatilidad implícita equivalente en modelo Shifted Log-normal (shift = 3%).

Autor: Gorka Crespo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from product.interest_rate.interest_rate_swap import InterestRateSwap
from underlying.interest_rate.interest_rate_curve import InterestRateCurve
from underlying.volatility.volatility_surface import VolatilitySurface
from product.interest_rate.interest_rate_cap import InterestRateCap
from basics.bussiness_calendar import apply_spot_lag


def limpiar_plazos_indice(label):
    """Convierte '1Yr' a 1.0 y '18Mo' a 1.5 para la superficie de volatilidad."""
    s = str(label).strip()
    if 'Yr' in s:
        return float(s.replace('Yr', ''))
    elif 'Mo' in s:
        return float(s.replace('Mo', '')) / 12.0
    return float(s)


if __name__ == "__main__":
    archivo_path = Path("data") / "Datos_Ejercicio_1.xlsx"
    valuation_date = datetime(2018, 12, 31)

    # Fecha spot: T+2 dias habiles TARGET
    start_date = apply_spot_lag(valuation_date, spot_lag_days=2)

    print(f"Fecha de valoracion: {valuation_date.strftime('%Y-%m-%d')}")
    print(f"Fecha spot (T+2):    {start_date.strftime('%Y-%m-%d')}")

    # ==========================================================================
    #  Carga de datos de mercado
    # ==========================================================================

    # Curva Euribor 6M (estimacion)
    df_euribor6m = pd.read_excel(archivo_path, header=2, usecols="B:H").dropna(how='all')
    dates_6m = pd.to_datetime(df_euribor6m.iloc[:, 0]).tolist()
    discounts_6m = df_euribor6m.iloc[:, 5].tolist()
    if valuation_date not in dates_6m:
        dates_6m.insert(0, valuation_date)
        discounts_6m.insert(0, 1.0)

    # Curva EUR OIS (descuento)
    df_eur_ois = pd.read_excel(archivo_path, header=2, usecols="Z:AG").dropna(how='all')
    dates_ois = pd.to_datetime(df_eur_ois.iloc[:, 1]).tolist()
    discounts_ois = df_eur_ois.iloc[:, 6].tolist()
    if valuation_date not in dates_ois:
        dates_ois.insert(0, valuation_date)
        discounts_ois.insert(0, 1.0)

    # Interpolacion log-linear para ambas curvas
    curve_6m = InterestRateCurve(
        start_date=valuation_date, end_dates=dates_6m, discount_factors=discounts_6m,
        interpolation='log-linear', day_count='Act/365'
    )
    curve_ois = InterestRateCurve(
        start_date=valuation_date, end_dates=dates_ois, discount_factors=discounts_ois,
        interpolation='log-linear', day_count='Act/365'
    )

    # ==========================================================================
    #  APARTADO A: Valoracion de Interest Rate Swap Fixed/Floating
    # ==========================================================================

    print("\n--- APARTADO A: Valoracion de Interest Rate Swap ---")

    swap = InterestRateSwap(
        start_date=start_date, maturity_years=20, notional=10_000_000,
        fixed_rate=0.024215, fixed_freq_months=12, fixed_day_count='30E360',
        float_freq_months=6, float_day_count='act360'
    )

    npv_swap = swap.npv(discount_curve=curve_ois, estimation_curve=curve_6m)
    print(f"NPV del Swap (Receptor pata fija): {npv_swap:,.2f} EUR")

    # ==========================================================================
    #  APARTADO B: Valoracion de Interest Rate Cap (modelo Normal)
    # ==========================================================================

    print("\n--- APARTADO B: Valoracion de Interest Rate Cap ---")

    # Superficie de Volatilidad Normal
    df_vol = pd.read_excel(archivo_path, sheet_name="Normal Volatility", header=3, usecols="D:S", index_col=0)
    if 'ATM' in df_vol.columns:
        df_vol = df_vol.drop(columns=['ATM'])
    df_vol.columns = pd.to_numeric(df_vol.columns)
    df_vol = df_vol / 10000.0
    df_vol.index = [limpiar_plazos_indice(i) for i in df_vol.index]
    surface = VolatilitySurface(df_vol)

    cap = InterestRateCap(
        start_date=start_date, maturity_years=20, notional=10_000_000,
        strike=0.015133, freq_months=6, day_count='act360'
    )

    npv_cap = cap.npv(
        valuation_date=valuation_date, discount_curve=curve_ois,
        estimation_curve=curve_6m, vol_surface=surface,
    )
    print(f"NPV del Cap (modelo Normal): {npv_cap:,.2f} EUR")

    # ==========================================================================
    #  APARTADO C: Volatilidad Implicita Shifted Log-normal (3%)
    # ==========================================================================

    print("\n--- APARTADO C: Volatilidad Implicita Shifted Log-normal ---")

    shift_cap = 0.03
    implied_vol = cap.implied_volatility(target_npv=npv_cap, shift=shift_cap)
    print(f"Volatilidad implicita Shifted LN (shift={shift_cap:.0%}): {implied_vol:.6%}")

    # Verificacion: recalcular NPV con la vol encontrada
    npv_shifted = sum(
        cap.shifted_caplet_price(
            info['forward_rate'], cap.strike, implied_vol,
            info['time_to_maturity'], info['df'], info['tau'], shift_cap
        ) * cap.notional
        for info in cap.caplets_info
    )
    print(f"NPV Normal:      {npv_cap:,.2f} EUR")
    print(f"NPV Shifted LN:  {npv_shifted:,.2f} EUR")
    print(f"Diferencia:      {abs(npv_cap - npv_shifted):.6f} EUR")

    # ==========================================================================
    #  GRAFICOS
    # ==========================================================================

    # Usar estilo por defecto de matplotlib (fondo blanco, mas limpio)
    plt.style.use('default')

    # -- Grafico 1: Curvas de Discount Factors --
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    years_6m = [t for t in curve_6m.delta_time if t >= 0]
    dfs_6m = [curve_6m.discount_factors[i] for i, t in enumerate(curve_6m.delta_time) if t >= 0]
    years_ois = [t for t in curve_ois.delta_time if t >= 0]
    dfs_ois = [curve_ois.discount_factors[i] for i, t in enumerate(curve_ois.delta_time) if t >= 0]

    ax1.plot(years_6m, dfs_6m, color='tab:blue', linewidth=2, label='Euribor 6M')
    ax1.plot(years_ois, dfs_ois, color='tab:orange', linewidth=2, label='EUR OIS')
    ax1.set_xlabel('Tiempo a vencimiento (años)')
    ax1.set_ylabel('Factor de Descuento')
    ax1.set_title('Curvas de Factores de Descuento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0)
    fig1.tight_layout()

    # -- Grafico 2: Forward Rates con Strike del Cap --
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fwd_starts = [c.start_date for c in swap.floating_leg]
    fwd_ends = [c.end_date for c in swap.floating_leg]
    fwd_rates = [curve_6m.forward_rate(s, e, 'act360') * 100 for s, e in zip(fwd_starts, fwd_ends)]
    fwd_years = [(s - valuation_date).days / 365.0 for s in fwd_starts]

    ax2.plot(fwd_years, fwd_rates, color='tab:blue', marker='o', markersize=4, label='Forward 6M')
    ax2.axhline(y=cap.strike * 100, color='tab:red', linestyle='--', label=f'Strike del Cap ({cap.strike:.2%})')
    ax2.axhline(y=swap.fixed_rate * 100, color='tab:green', linestyle='-.', label=f'Tipo Fijo Swap ({swap.fixed_rate:.2%})')
    ax2.fill_between(fwd_years, fwd_rates, cap.strike * 100,
                     where=[r > cap.strike * 100 for r in fwd_rates],
                     alpha=0.2, color='tab:red', label='Cap In-The-Money')
    ax2.set_xlabel('Tiempo a vencimiento (años)')
    ax2.set_ylabel('Tipo Forward (%)')
    ax2.set_title('Estructura Temporal de Tipos Forward (Euribor 6M)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # -- Grafico 3: Caplet NPV --
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    caplet_ttm = [info['time_to_maturity'] for info in cap.caplets_info]
    caplet_npvs = [info['caplet_npv'] / 1000 for info in cap.caplets_info]

    ax3.bar(caplet_ttm, caplet_npvs, width=0.4, color='tab:blue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Tiempo a vencimiento (años)')
    ax3.set_ylabel('NPV del Caplet (miles de EUR)')
    ax3.set_title('Valoración por Caplets (Interest Rate Cap)')
    ax3.text(0.95, 0.95, f'NPV Total: {npv_cap:,.0f} EUR', 
             transform=ax3.transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.tight_layout()

    plt.show()
