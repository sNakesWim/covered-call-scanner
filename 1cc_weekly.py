# covered_call_weekly.py
# Weekly covered-call backtest: sell ATM call each Friday, close next Friday, roll.
# Uses Black-Scholes with rolling realized volatility as an IV proxy.

import math
import pandas as pd
import yfinance as yf
from datetime import timedelta

# ---------- helpers ----------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def realized_vol_annualized(returns, win=21):
    if len(returns) < win:
        return None
    vol = returns[-win:].std()
    return vol * math.sqrt(252)

def nearest_strike(price, increment=1.0):
    return round(price / increment) * increment

# ---------- inputs ----------
SYMBOL = input("Enter stock ticker: ").strip().upper() or "AAPL"
PERIOD = "10y"   # change to "20y" if you wish and have data
RISK_FREE = 0.02 # 2% annual proxy; adjust as you like
VOL_WIN = 21     # use 1-month realized vol as proxy for IV
STRIKE_STEP = 1.0  # simple $1 increments
T_WEEK = 7.0 / 365.0
SLIPPAGE = 0.0   # per-share slippage
STOCK_FEE = 0.0  # per-share commission
OPT_FEE = 0.0    # per-share option commission (contract is 100 shares; we model per share)

print(f"Downloading {SYMBOL}...")

import numpy as np  # add once near top if not already

px = yf.download(
    SYMBOL, period=PERIOD, interval="1d",
    auto_adjust=False, group_by="column", progress=False
).rename(columns=str.title)

# keep only OHLCV
px = px[["Open","High","Low","Close","Volume"]].dropna()

# ensure Close is 1-D
close = px["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

# weekly schedule: Friday closes
week_close = px["Close"].resample("W-FRI").last().dropna()
# align to actual trading days for return calc
close_on = px["Close"]

weekly_rows = []
equity = None
equity_curve = []

# build daily log returns for realized vol
daily_ret = np.log(close).diff().dropna()

# iterate week-by-week
for i in range(len(week_close) - 1):
    date_sell = week_close.index[i]
    date_buy  = week_close.index[i+1]

    # map to actual close prices on those dates (already aligned by resample)
    if date_sell not in close_on.index or date_buy not in close_on.index:
        continue

    S0 = float(close_on.loc[date_sell])
    S1 = float(close_on.loc[date_buy])

    # realized vol proxy through the sell date
    hist = daily_ret[daily_ret.index <= date_sell]
    sigma = realized_vol_annualized(hist.values, VOL_WIN)
    if sigma is None or not math.isfinite(sigma):
        continue

    # sell ATM call at Friday close
    K = nearest_strike(S0, STRIKE_STEP)
    premium = bs_call_price(S0, K, RISK_FREE, sigma, T_WEEK)

    # one-week payoff per share for covered call
    # long stock + short call
    stock_pnl = (S1 - S0) - SLIPPAGE - STOCK_FEE
    call_pnl  = premium - max(S1 - K, 0) - SLIPPAGE - OPT_FEE
    pnl = stock_pnl + call_pnl

    # initialize equity as 1 share notional to make returns interpretable
    if equity is None:
        equity = S0

    weekly_ret = pnl / equity  # return on current equity base
    equity += pnl
    equity_curve.append({"date": date_buy, "equity": equity, "ret": weekly_ret})

    weekly_rows.append({
        "SellDate": date_sell.date(),
        "BuyDate": date_buy.date(),
        "S0": S0,
        "S1": S1,
        "K": K,
        "Sigma": sigma,
        "Premium": premium,
        "StockPnL": stock_pnl,
        "CallPnL": call_pnl,
        "TotalPnL": pnl,
        "WeeklyRet": weekly_ret,
        "ExpiredOTM": 1 if S1 <= K else 0
    })

if not weekly_rows:
    raise SystemExit("Not enough data to run weekly cycles. Try a shorter VOL_WIN or longer PERIOD.")

wk = pd.DataFrame(weekly_rows)
eq = pd.DataFrame(equity_curve).set_index("date")

# Extra metrics
initial_equity = wk.iloc[0]["S0"]
final_equity = equity

# average weekly premium as % of spot at sale
avg_premium_pct = 100.0 * (wk["Premium"] / wk["S0"]).mean()


# summary stats
weeks = len(wk)
wins = int((wk["TotalPnL"] > 0).sum())
win_rate = 100.0 * wins / weeks
total_return = 100.0 * ((equity / wk.iloc[0]["S0"]) - 1.0)
avg_premium = wk["Premium"].mean()
expired_otm = 100.0 * wk["ExpiredOTM"].mean()
trades = weeks  # one option cycle per week

# annualized return (CAGR) from weekly cycles
annual_return_pct = ((final_equity / initial_equity) ** (52.0 / weeks) - 1.0) * 100.0

# Sharpe on weekly returns
if eq["ret"].std() and eq["ret"].std() > 0:
    sharpe = (eq["ret"].mean() / eq["ret"].std()) * math.sqrt(52.0)
else:
    sharpe = float("nan")

print("\n=== Weekly Covered Call Results ===")
print(f"Symbol:           {SYMBOL}")
print(f"Weeks:            {weeks}")
print(f"Win Rate [%]:     {win_rate:.2f}")
print(f"Return [%]:       {total_return:.2f}")
print(f"Annual Return [%]: {annual_return_pct:.2f}")
print(f"Sharpe (weekly):  {sharpe:.2f}")
print(f"# Trades:         {trades}")
print(f"Avg Premium ($):  {avg_premium:.2f}")
print(f"Avg Premium [%]:  {avg_premium_pct:.2f}")
print(f"% Expired OTM:    {expired_otm:.2f}")

# save details
wk.to_csv(f"{SYMBOL}_covered_call_weekly.csv", index=False)
print(f"\nSaved trade log: {SYMBOL}_covered_call_weekly.csv")
