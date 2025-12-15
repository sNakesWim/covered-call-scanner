# cc_scanner.py
# Scan S&P 500 for highest % Premium Return from selling the nearest-expiry ATM call.
# % Premium Return = ((Strike - Spot) + Premium) / Spot * 100

import time
import math
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import yfinance as yf

import requests
YF_SESSION = requests.Session()
YF_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (cc-scanner/1.0; macOS)"})

# user-provided expiry (YYYY-MM-DD)
EXP_STR = input("ENTER EXPIRATION (YYYY-MM-DD): ").strip()
INCLUDE_EARN_BEF_EXP = input("Include tickers with earnings before expiry? (y/n): ").strip().lower().startswith("y")
EXP_TS = pd.to_datetime(EXP_STR, errors="coerce")
if pd.isna(EXP_TS):
    raise SystemExit("Invalid date. Use format YYYY-MM-DD, e.g., 2025-10-24")

OPTION_MODE = input("Which calls to scan? (atm/itm/both): ").strip().lower()
if OPTION_MODE not in {"atm", "itm", "both"}:
    OPTION_MODE = "atm"

# -------- settings --------
MIN_OI = 100          # open interest filter
MIN_VOL = 10          # options volume filter
SLEEP_BETWEEN = 0.15  # seconds between symbols to be gentle on Yahoo
ATM_TOL = 0.02  # 2%: require |K - S| / S <= 1%

SHOW_COLS = [
    "Symbol","Spot","Expiry","DTE","Strike","Bid","Ask","Mid",
    "OpenInterest","Volume","Premium Return [%]","Next Earnings","Earnings Before Expiry"
]

# -------- helpers --------
def fetch_sp500_tickers(max_retries=3, sleep_s=2) -> list:
    import io
    import pandas as pd
    import time

    # 1) yfinance helper
    try:
        if hasattr(yf, "tickers_sp500"):
            syms = yf.tickers_sp500()
            if syms:
                return sorted({s.replace(".", "-").strip().upper() for s in syms})
    except Exception:
        pass

    # 2) Wikipedia (primary)
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for _ in range(max_retries):
        try:
            html = YF_SESSION.get(wiki_url, timeout=20).text
            tables = pd.read_html(io.StringIO(html), flavor="lxml")
            df = tables[0]
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            syms = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .str.upper()
                .tolist()
            )
            out = [s for s in syms if s and s != "SYMBOL"]
            if out:
                return sorted(set(out))
        except Exception:
            time.sleep(sleep_s)

    # 3) CSV mirror backup (best-effort)
    gh_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    for _ in range(max_retries):
        try:
            csvtxt = YF_SESSION.get(gh_url, timeout=20).text
            df = pd.read_csv(io.StringIO(csvtxt))
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            syms = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .str.upper()
                .tolist()
            )
            if syms:
                return sorted(set(syms))
        except Exception:
            time.sleep(sleep_s)

    # 4) Last resort: small static subset so the script still runs
    print("Warning: could not fetch S&P 500 list. Using a small fallback set.")
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM",
        "UNH","V","JNJ","WMT","HD","MA","PG","LLY","AVGO","NFLX"
    ]

def mid_price(bid, ask, last):
    vals = []
    if pd.notna(bid) and bid > 0:
        vals.append(float(bid))
    if pd.notna(ask) and ask > 0:
        vals.append(float(ask))
    if pd.notna(last) and last > 0:
        vals.append(float(last))
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (float(bid) + float(ask))
    if vals:
        return max(vals)
    return 0.0

def pick_atm_row(calls_df: pd.DataFrame, spot: float) -> pd.Series | None:
    if calls_df is None or calls_df.empty:
        return None

    calls = calls_df.dropna(subset=["strike"]).copy()
    if calls.empty:
        return None

    # distance to spot
    calls["abs_dist"] = (calls["strike"] - spot).abs()
    calls["rel_dist"] = calls["abs_dist"] / spot

    # try within ATM_TOL first
    near = calls[calls["rel_dist"] <= ATM_TOL].sort_values(["abs_dist", "strike"])
    if near.empty:
        near = calls.sort_values(["abs_dist", "strike"])

    # 1) prefer nearest strike with decent liquidity + valid spread
    for _, r in near.iterrows():
        oi_ok = pd.notna(r.get("openInterest")) and r.get("openInterest") >= MIN_OI
        vol_ok = pd.notna(r.get("volume")) and r.get("volume") >= MIN_VOL
        bid = r.get("bid")
        ask = r.get("ask")
        spread_ok = pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid
        if oi_ok and vol_ok and spread_ok:
            return r

    # 2) fallback: just take the nearest strike that has any price at all
    for _, r in near.iterrows():
        bid = r.get("bid")
        ask = r.get("ask")
        last = r.get("lastPrice")
        if (
            (pd.notna(bid) and bid > 0)
            or (pd.notna(ask) and ask > 0)
            or (pd.notna(last) and last > 0)
        ):
            return r

    return None

def pick_itm_row(calls_df: pd.DataFrame, spot: float) -> pd.Series | None:
    """Pick the nearest in the money call (strike <= spot)."""
    if calls_df is None or calls_df.empty:
        return None

    calls = calls_df.dropna(subset=["strike"]).copy()
    if calls.empty:
        return None

    # only strikes at or below spot
    itm = calls[calls["strike"] <= spot].copy()
    if itm.empty:
        return None

    # distance under spot
    itm["under_dist"] = (spot - itm["strike"]).abs()
    itm = itm.sort_values(["under_dist", "strike"], ascending=[True, False])

    # 1) prefer liquid strikes first
    for _, r in itm.iterrows():
        oi_ok = pd.notna(r.get("openInterest")) and r.get("openInterest") >= MIN_OI
        vol_ok = pd.notna(r.get("volume")) and r.get("volume") >= MIN_VOL
        bid = r.get("bid"); ask = r.get("ask")
        spread_ok = pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid
        if oi_ok and vol_ok and spread_ok:
            return r

    # 2) fallback - any ITM strike that has some price info
    for _, r in itm.iterrows():
        bid = r.get("bid"); ask = r.get("ask"); last = r.get("lastPrice")
        if (
            (pd.notna(bid) and bid > 0)
            or (pd.notna(ask) and ask > 0)
            or (pd.notna(last) and last > 0)
        ):
            return r

    return None

def pick_call_row(calls_df: pd.DataFrame, spot: float, mode: str) -> pd.Series | None:
    """Pick a call row based on mode: atm, itm, or both."""
    if mode == "atm":
        return pick_atm_row(calls_df, spot)

    if mode == "itm":
        return pick_itm_row(calls_df, spot)

    if mode == "both":
        # try both, choose the one with higher premium return
        candidates = []
        r_atm = pick_atm_row(calls_df, spot)
        if r_atm is not None:
            candidates.append(r_atm)
        r_itm = pick_itm_row(calls_df, spot)
        if r_itm is not None:
            candidates.append(r_itm)

        if not candidates:
            return None

        best = None
        best_ret = -1e9
        for r in candidates:
            bid = float(r.get("bid") or float("nan"))
            ask = float(r.get("ask") or float("nan"))
            last = float(r.get("lastPrice") or float("nan"))
            mid = mid_price(bid, ask, last)
            if not np.isfinite(mid):
                continue
            strike = float(r["strike"])
            prem_ret_pct = ((strike - spot) + mid) / spot * 100.0
            if prem_ret_pct > best_ret:
                best_ret = prem_ret_pct
                best = r

        return best

    # fallback
    return pick_atm_row(calls_df, spot)

def get_universe() -> list:
    """
    Mode 1: S&P 500 (current scanner)
    Mode 2: Manual tickers until user types STOP
    """
    choice = input("Choose universe: 1 = S&P 500, 2 = Manual: ").strip()
    if choice == "2":
        print('Enter tickers one per line. Type "STOP" to finish.')
        items = []
        while True:
            t = input("Ticker: ").strip()
            if t.upper() == "STOP":
                break
            if not t:
                continue
            items.append(t.replace(".", "-").upper())
        # de-dup while keeping order
        seen, out = set(), []
        for t in items:
            if t not in seen:
                seen.add(t)
                out.append(t)
        if not out:
            raise SystemExit("No tickers entered.")
        return out
    # default to S&P 500
    return fetch_sp500_tickers()

# ---- helper: get spot price ----
def get_spot(tkr: yf.Ticker) -> float | None:
    # 1) fast_info last trade
    try:
        fi = tkr.fast_info
        p = getattr(fi, "last_price", None)
        if p is not None and np.isfinite(p) and p > 0:
            return float(p)
    except Exception:
        pass

    # 2) latest intraday close (may be empty after-hours)
    try:
        h = tkr.history(period="1d", interval="1m")
        if h is not None and not h.empty:
            v = float(h["Close"].dropna().iloc[-1])
            if np.isfinite(v) and v > 0:
                return v
    except Exception:
        pass

    # 3) fallback: last regular daily close (works off-hours)
    try:
        d = tkr.history(period="5d", interval="1d")
        if d is not None and not d.empty:
            v = float(d["Close"].dropna().iloc[-1])
            if np.isfinite(v) and v > 0:
                return v
    except Exception:
        pass

    return None

def resolve_expiry(tkr: yf.Ticker, target_str: str) -> str | None:
    """Return the exact target expiry if available, else the nearest future expiry,
    else the nearest overall. Input format YYYY-MM-DD."""
    exps = tkr.options or []
    if not exps:
        return None

    # fast path: exact match
    if target_str in exps:
        return target_str

    target_ts = pd.to_datetime(target_str, errors="coerce")
    if pd.isna(target_ts):
        return None

    exp_ts = pd.to_datetime(exps, errors="coerce")
    # future expiries relative to now
    now_utc = pd.Timestamp.utcnow().normalize()
    future_idx = [i for i, ts in enumerate(exp_ts) if pd.notna(ts) and ts >= now_utc]

    # choose nearest future to target if possible
    if future_idx:
        future = exp_ts[future_idx]
        deltas = (future - target_ts).abs()
        pick = future_idx[int(deltas.argmin())]
        return exps[pick]

    # else choose nearest overall to target
    deltas_all = (exp_ts - target_ts).abs()
    pick = int(deltas_all.argmin())
    return exps[pick]

def resolve_expiry(tkr: yf.Ticker, target_str: str) -> str | None:
    """Return exact target expiry if available, else the nearest future expiry, else nearest overall."""
    exps = tkr.options or []
    if not exps:
        return None
    if target_str in exps:
        return target_str

    target_ts = pd.to_datetime(target_str, errors="coerce")
    if pd.isna(target_ts):
        return None

    exp_ts = pd.to_datetime(exps, errors="coerce")
    now_utc = pd.Timestamp.utcnow().normalize()
    future_idx = [i for i, ts in enumerate(exp_ts) if pd.notna(ts) and ts >= now_utc]

    if future_idx:
        future = exp_ts[future_idx]
        pick = future_idx[int((future - target_ts).abs().argmin())]
        return exps[pick]

    pick = int((exp_ts - target_ts).abs().argmin())
    return exps[pick]

# ---- main scan for one symbol ----
def scan_symbol(symbol: str) -> dict | None:
    try:
        tkr = yf.Ticker(symbol)

        # only use the user-entered expiry; skip if this ticker does not have it
        exps = tkr.options or []
        if EXP_STR not in exps:
            return None
        exp_use = EXP_STR

        spot = get_spot(tkr)
        if spot is None or not np.isfinite(spot) or spot <= 0:
            return None
        
        chain = tkr.option_chain(exp_use).calls
        if chain is None or chain.empty:
            return None

        row = pick_call_row(chain, spot, OPTION_MODE)
        if row is None:
            return None
        
        # we have an ATM row; inspect pricing before premium check
        strike = float(row["strike"])
        bid = float(row.get("bid") or float("nan"))
        ask = float(row.get("ask") or float("nan"))
        last = float(row.get("lastPrice") or float("nan"))
        oi  = int(row.get("openInterest") or 0)
        vol = int(row.get("volume") or 0)
        premium_mid = mid_price(bid, ask, last)
        
        if not np.isfinite(premium_mid):
            return None


        prem_ret_pct = ((strike - spot) + premium_mid) / spot * 100.0

        try:
            dte_days = int((pd.to_datetime(exp_use) - pd.Timestamp.utcnow()).ceil("D").days)
        except Exception:
            dte_days = np.nan

        earn = get_next_earnings(tkr)
        earn_dt = pd.to_datetime(earn) if earn and earn != "N/A" else pd.NaT
        flag = "⚠️" if pd.notna(earn_dt) and pd.notna(pd.to_datetime(exp_use)) and earn_dt.date() <= pd.to_datetime(exp_use).date() else ""

        return {
            "Symbol": symbol,
            "Spot": round(spot, 2),
            "Expiry": exp_use,  # show the actual expiry used
            "DTE": dte_days,
            "Strike": round(strike, 2),
            "Bid": round(bid, 2) if np.isfinite(bid) else np.nan,
            "Ask": round(ask, 2) if np.isfinite(ask) else np.nan,
            "Mid": round(premium_mid, 2),
            "OpenInterest": oi,
            "Volume": vol,
            "Premium Return [%]": round(prem_ret_pct, 2),
            "Next Earnings": earn or "N/A",
            "Earnings Before Expiry": flag,
        }
    except Exception:
        return None

def get_next_earnings(tkr: yf.Ticker) -> str | None:
    """Return the next earnings date if available."""
    try:
        df = tkr.get_earnings_dates(limit=1)
        if df is not None and not df.empty:
            next_date = df.index[0]
            return pd.to_datetime(next_date).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def main():
    print("Fetching S&P 500 tickers...")
    tickers = get_universe()
    print(f"Scanning {len(tickers)} symbols for nearest-expiry ATM call premium...")

    rows = []
    for i, sym in enumerate(tickers, 1):
        res = scan_symbol(sym)
        if res:
            rows.append(res)
            print(f"[{i}/{len(tickers)}] {sym}  ->  {res['Premium Return [%]']}%")
        else:
            print(f"[{i}/{len(tickers)}] {sym}  ->  skipped")
        time.sleep(SLEEP_BETWEEN)

    if not rows:
        print("No results. Try again later or relax filters.")
        return

    valid_count = sum(res is not None for res in rows)
    print(f"\n{valid_count} tickers had the {EXP_STR} expiry available.")

    df = pd.DataFrame(rows)
    if not INCLUDE_EARN_BEF_EXP:
        df = df[df["Earnings Before Expiry"] != "⚠️"]
    df.sort_values("Premium Return [%]", ascending=False, inplace=True)
    print("\nTop 50 by % Premium Return (nearest-expiry ATM calls):")
    print(df[SHOW_COLS].head(50).to_string(index=False))

if __name__ == "__main__":
    main()
