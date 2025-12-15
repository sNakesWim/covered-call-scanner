# forward_cc_monte_carlo.py
# Simulate 1 year of weekly ATM covered-call rolls using current option-chain IV.
# Outputs distribution of annual returns and key percentiles.

import math, sys
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Black–Scholes call ----------
def _ncdf(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)

# ---------- inputs ----------
SYMBOL = input("Ticker: ").strip().upper() or "AAPL"
N_PATHS = int(input("Number of scenarios [e.g., 2000]: ") or "2000")
WEEKS = 52
DT = 1.0 / 52.0
R = 0.02  # annual risk-free proxy
STRIKE_STEP = 1.0  # nearest dollar
np.random.seed(42)

# ---------- pull spot and current ATM IV ----------
tkr = yf.Ticker(SYMBOL)
opts = tkr.options
if not opts:
    sys.exit("No option expiries found for this ticker.")

# pick the first expiry at least 7 days out if possible
def pick_expiry(expiries):
    for e in expiries:
        dte = (pd.to_datetime(e) - pd.Timestamp.today()).days
        if dte >= 7:
            return e
    return expiries[0]

exp = pick_expiry(opts)
chain = tkr.option_chain(exp).calls
spot = float(tkr.fast_info.last_price or tkr.history(period="1d", interval="1m")["Close"].dropna().iloc[-1])

# choose ATM by nearest strike, prefer >= spot
chain = chain.dropna(subset=["strike"]).copy()
chain["dist"] = chain["strike"] - spot
atm = chain[chain["dist"] >= 0].sort_values(["dist", "strike"]).head(1)
if atm.empty:
    atm = chain.reindex((chain["strike"] - spot).abs().sort_values().index).head(1)

iv0 = float(atm["impliedVolatility"].iloc[0]) if "impliedVolatility" in atm.columns else np.nan

# fallback IV from realized vol if IV missing
hist = tkr.history(period="3y", interval="1d")["Close"].dropna()
rv_annual = float(np.log(hist).diff().dropna().std() * np.sqrt(252)) if len(hist) > 30 else 0.25
SIGMA0 = iv0 if np.isfinite(iv0) and iv0 > 0 else rv_annual

print(f"Using spot={spot:.2f}, base IV={SIGMA0:.3f}, expiry={exp}")

# optional drift from history, else neutral drift
if len(hist) > 30:
    mu_hist = float(np.log(hist).diff().dropna().mean() * 252.0)
else:
    mu_hist = 0.0

# simple vol of vol to vary IV week to week
vol_of_vol = 0.20  # 20% relative noise around base IV

# ---------- simulate paths ----------
def nearest_strike(x, step):
    return round(x / step) * step

def run_path():
    S = spot
    eq = spot  # start with 1 share notionally for return math
    sigma = SIGMA0
    for _ in range(WEEKS):
        # sell 1 ATM call for next week at mid theoretical using sigma
        K = nearest_strike(S, STRIKE_STEP)
        prem = bs_call_price(S, K, R, sigma, DT)

        # simulate next week spot under GBM
        z = np.random.normal()
        S_next = S * math.exp((mu_hist - 0.5 * sigma * sigma) * DT + sigma * math.sqrt(DT) * z)

        # PnL for covered call per share
        stock_pnl = S_next - S
        call_pnl = prem - max(S_next - K, 0)
        pnl = stock_pnl + call_pnl
        eq += pnl

        # update for next week
        S = S_next
        # jitter IV within a band
        sigma = max(0.0001, SIGMA0 * (1.0 + vol_of_vol * np.random.normal(scale=0.5)))
    # annual return percent relative to initial equity
    return 100.0 * (eq / spot - 1.0)

rets = np.array([run_path() for _ in range(N_PATHS)])

# ---------- summarize ----------
p5, p50, p95 = np.percentile(rets, [5, 50, 95])
print("\n=== 1-year simulated distribution ===")
print(f"Mean return [%]:      {rets.mean():.2f}")
print(f"Median [%]:           {p50:.2f}")
print(f"5th percentile [%]:   {p5:.2f}")
print(f"95th percentile [%]:  {p95:.2f}")
print(f"Min [%]:              {rets.min():.2f}")
print(f"Max [%]:              {rets.max():.2f}")
print(f"Prob. of loss [%]:    {100.0 * (rets < 0).mean():.2f}")

ans = input("Save histogram? [y/N]: ").strip().lower()
if ans in ("y", "yes"):
    try:
        import matplotlib.pyplot as plt
        plt.hist(rets, bins=50)
        plt.title(f"{SYMBOL} weekly ATM CC, 1y Monte Carlo")
        plt.xlabel("Annual return [%]")
        plt.ylabel("Frequency")
        plt.tight_layout()
        out = f"{SYMBOL}_cc_forward_hist.png"
        plt.savefig(out, dpi=120)
        print(f"Saved {out}")
    except Exception as e:
        print("Plot unavailable. Install matplotlib with: pip install matplotlib")
        print(e)

