"""
Derivatives Valuation — Black-Scholes & Monte Carlo
Author: Gorka Crespo Bravo
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class BlackScholes:
    def __init__(self, S, K, T, r, sigma, q=0):
        self.S, self.K, self.T, self.r, self.sigma, self.q = S, K, T, r, sigma, q
        self.d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)

    def call_price(self):
        return self.S*np.exp(-self.q*self.T)*norm.cdf(self.d1) - \
               self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2)

    def put_price(self):
        return self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2) - \
               self.S*np.exp(-self.q*self.T)*norm.cdf(-self.d1)

    def delta(self, t="call"):
        d = np.exp(-self.q*self.T)*norm.cdf(self.d1)
        return d if t == "call" else d - np.exp(-self.q*self.T)

    def gamma(self):
        return np.exp(-self.q*self.T)*norm.pdf(self.d1) / \
               (self.S*self.sigma*np.sqrt(self.T))

    def vega(self):
        return self.S*np.exp(-self.q*self.T)*norm.pdf(self.d1)*np.sqrt(self.T)/100

    def theta(self, t="call"):
        t1 = -(self.S*self.sigma*np.exp(-self.q*self.T)*norm.pdf(self.d1))/(2*np.sqrt(self.T))
        if t == "call":
            return (t1 - self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2))/365
        return (t1 + self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2))/365

def monte_carlo_option(S, K, T, r, sigma, n=100000, opt="call"):
    np.random.seed(42)
    Z = np.random.standard_normal(n // 2)
    Z = np.concatenate([Z, -Z])
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0) if opt == "call" else np.maximum(K - ST, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    se = np.exp(-r*T) * np.std(payoff) / np.sqrt(n)
    return price, se

# Parámetros
S, K, T, r, sigma = 100, 105, 0.5, 0.04, 0.25
bs = BlackScholes(S, K, T, r, sigma)

print("=" * 55)
print("VALORACIÓN DE OPCIONES EUROPEAS")
print("=" * 55)
print(f"  S={S}  K={K}  T={T}  r={r:.2%}  σ={sigma:.2%}\n")
print(f"  BS Call: {bs.call_price():.4f}   Put: {bs.put_price():.4f}")

mc_c, se_c = monte_carlo_option(S, K, T, r, sigma, 100000, "call")
mc_p, se_p = monte_carlo_option(S, K, T, r, sigma, 100000, "put")
print(f"  MC Call: {mc_c:.4f}±{se_c:.4f}  Put: {mc_p:.4f}±{se_p:.4f}")
print(f"\n  Delta={bs.delta():.4f}  Gamma={bs.gamma():.4f}")
print(f"  Vega={bs.vega():.4f}   Theta={bs.theta():.4f}")

# Greeks sensitivity plot
spots = np.linspace(70, 140, 200)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, fn), color in zip(axes.flat,
    [("Delta", lambda s: BlackScholes(s,K,T,r,sigma).delta()),
     ("Gamma", lambda s: BlackScholes(s,K,T,r,sigma).gamma()),
     ("Vega",  lambda s: BlackScholes(s,K,T,r,sigma).vega()),
     ("Theta", lambda s: BlackScholes(s,K,T,r,sigma).theta())],
    ["#3b82f6","#8b5cf6","#10b981","#ef4444"]):
    ax.plot(spots, [fn(s) for s in spots], color=color, lw=2)
    ax.axvline(K, color="gray", ls="--", alpha=0.5)
    ax.set_title(name, fontweight="bold"); ax.grid(True, alpha=0.3)
fig.suptitle("Greeks Sensitivity — European Call", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.savefig("greeks.png", dpi=150); plt.show()
