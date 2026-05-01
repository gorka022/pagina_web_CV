"""
Portfolio Management — Markowitz Mean-Variance & Max-Sharpe Optimization
Author: Gorka Crespo Bravo
Description: Construcción de carteras óptimas utilizando la teoría de Markowitz.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# ============================================================
# 1. Descarga de datos
# ============================================================
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "PFE", "JNJ"]
start_date = "2019-01-01"
end_date = "2024-01-01"

data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
returns = data.pct_change().dropna()

# ============================================================
# 2. Parámetros de la cartera
# ============================================================
mean_returns = returns.mean() * 252          # Rendimiento anualizado
cov_matrix = returns.cov() * 252             # Matriz de covarianzas anualizada
n_assets = len(tickers)
risk_free_rate = 0.04                        # Tasa libre de riesgo (4%)

# ============================================================
# 3. Funciones de optimización
# ============================================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calcula rendimiento y volatilidad de la cartera."""
    port_return = np.dot(weights, mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
    """Sharpe ratio negativo (para minimización)."""
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - rf) / p_vol


def minimize_volatility(weights, mean_returns, cov_matrix):
    """Función objetivo: minimizar volatilidad."""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


# ============================================================
# 4. Restricciones y límites
# ============================================================
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Pesos suman 1
bounds = tuple((0, 1) for _ in range(n_assets))                # Sin ventas en corto
initial_weights = np.array([1 / n_assets] * n_assets)

# ============================================================
# 5. Cartera de Mínima Varianza (MVP)
# ============================================================
min_var_result = minimize(
    minimize_volatility,
    initial_weights,
    args=(mean_returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
mvp_weights = min_var_result.x
mvp_return, mvp_vol = portfolio_performance(mvp_weights, mean_returns, cov_matrix)

print("=" * 60)
print("CARTERA DE MÍNIMA VARIANZA (MVP)")
print("=" * 60)
for ticker, w in zip(tickers, mvp_weights):
    if w > 0.001:
        print(f"  {ticker:>6s}: {w:.2%}")
print(f"\n  Rendimiento: {mvp_return:.2%}")
print(f"  Volatilidad: {mvp_vol:.2%}")
print(f"  Sharpe Ratio: {(mvp_return - risk_free_rate) / mvp_vol:.4f}")

# ============================================================
# 6. Cartera Max-Sharpe (Tangente)
# ============================================================
max_sharpe_result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
ms_weights = max_sharpe_result.x
ms_return, ms_vol = portfolio_performance(ms_weights, mean_returns, cov_matrix)

print("\n" + "=" * 60)
print("CARTERA MAX-SHARPE (TANGENTE)")
print("=" * 60)
for ticker, w in zip(tickers, ms_weights):
    if w > 0.001:
        print(f"  {ticker:>6s}: {w:.2%}")
print(f"\n  Rendimiento: {ms_return:.2%}")
print(f"  Volatilidad: {ms_vol:.2%}")
print(f"  Sharpe Ratio: {(ms_return - risk_free_rate) / ms_vol:.4f}")

# ============================================================
# 7. Frontera eficiente (Monte Carlo)
# ============================================================
n_portfolios = 10000
results = np.zeros((3, n_portfolios))

for i in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n_assets))
    p_ret, p_vol = portfolio_performance(w, mean_returns, cov_matrix)
    results[0, i] = p_vol
    results[1, i] = p_ret
    results[2, i] = (p_ret - risk_free_rate) / p_vol

# ============================================================
# 8. Visualización
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(
    results[0, :], results[1, :],
    c=results[2, :], cmap="viridis", marker="o", s=5, alpha=0.5
)
plt.colorbar(scatter, label="Sharpe Ratio")

ax.scatter(mvp_vol, mvp_return, c="red", marker="*", s=300,
           label=f"MVP (σ={mvp_vol:.2%}, μ={mvp_return:.2%})", zorder=5)
ax.scatter(ms_vol, ms_return, c="gold", marker="*", s=300,
           label=f"Max-Sharpe (σ={ms_vol:.2%}, μ={ms_return:.2%})", zorder=5)

ax.set_xlabel("Volatilidad (σ)", fontsize=12)
ax.set_ylabel("Rendimiento Esperado (μ)", fontsize=12)
ax.set_title("Frontera Eficiente — Markowitz Mean-Variance", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150)
plt.show()
