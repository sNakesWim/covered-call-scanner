from flask import Flask, request, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np

import time  # if not already imported

MIN_OI = 100          # minimum open interest
MIN_VOL = 10          # minimum option volume
SLEEP_BETWEEN = 0.15  # pause between tickers, in seconds
ATM_TOL = 0.02        # 2% band: |K - S| / S <= 0.02

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Covered Call Scanner</title>
  <style>
    :root {
      --bg-main: #050608;
      --bg-card: #0b0f16;
      --bg-table-header: #111827;
      --bg-table-row: #020617;
      --bg-table-alt: #050b16;
      --accent: #22c55e;
      --accent-soft: rgba(34, 197, 94, 0.15);
      --accent-border: rgba(34, 197, 94, 0.6);
      --text-main: #e5e7eb;
      --text-dim: #9ca3af;
      --text-danger: #f97373;
      --border-soft: #1f2937;
      --input-bg: #020617;
    }

    * {
      box-sizing: border-box;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "Segoe UI", sans-serif;
    }

    body {
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at top, #020617 0%, #020617 40%, #000000 100%);
      color: var(--text-main);
    }

    .shell {
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 32px 16px;
    }

    .card {
      width: 100%;
      max-width: 1100px;
      background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000000 100%);
      border-radius: 18px;
      border: 1px solid var(--border-soft);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.7);
      padding: 24px 24px 32px;
    }

    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }

    .title-block h1 {
      font-size: 1.35rem;
      margin: 0;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .title-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      color: var(--accent);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .title-badge span.dot {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 8px rgba(34, 197, 94, 0.8);
    }

    .header p {
      margin: 4px 0 0;
      font-size: 0.8rem;
      color: var(--text-dim);
    }

    .pill {
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid var(--border-soft);
      font-size: 0.75rem;
      color: var(--text-dim);
    }

    form {
      margin-top: 12px;
      margin-bottom: 20px;
      padding: 14px 14px 10px;
      border-radius: 14px;
      border: 1px solid var(--border-soft);
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.9));
    }

    .form-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 10px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 130px;
      flex: 1;
    }

    .field label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-dim);
    }

    input[type="text"],
    textarea,
    select {
      border-radius: 10px;
      border: 1px solid var(--border-soft);
      padding: 7px 9px;
      font-size: 0.85rem;
      background: var(--input-bg);
      color: var(--text-main);
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }

    input[type="text"]:focus,
    textarea:focus,
    select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.5);
    }

    textarea {
      min-height: 70px;
      resize: vertical;
    }

    .radio-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .radio-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 9px;
      border-radius: 999px;
      border: 1px solid var(--border-soft);
      font-size: 0.78rem;
      cursor: pointer;
      color: var(--text-dim);
      background: rgba(15, 23, 42, 0.8);
      transition: border-color 0.12s ease, background 0.12s ease, color 0.12s ease;
    }

    .radio-chip input {
      accent-color: var(--accent);
    }

    .radio-chip.active {
      border-color: var(--accent);
      color: var(--accent);
      background: rgba(34, 197, 94, 0.08);
    }

    .actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      margin-top: 4px;
      gap: 10px;
    }

    .btn-main {
      border-radius: 999px;
      border: none;
      padding: 7px 18px;
      font-size: 0.85rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      background: radial-gradient(circle at 0% 0%, #4ade80 0%, #22c55e 30%, #16a34a 70%, #166534 100%);
      color: #001106;
      cursor: pointer;
      box-shadow: 0 10px 25px rgba(34, 197, 94, 0.35);
      transition: transform 0.08s ease, box-shadow 0.08s ease;
    }

    .btn-main:hover {
      transform: translateY(-1px);
      box-shadow: 0 14px 28px rgba(34, 197, 94, 0.45);
    }

    .btn-main:active {
      transform: translateY(0);
      box-shadow: 0 8px 18px rgba(34, 197, 94, 0.25);
    }

    .helper-chip {
      font-size: 0.75rem;
      color: var(--text-dim);
    }

    .error {
      margin-top: 10px;
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid rgba(239, 68, 68, 0.5);
      background: rgba(127, 29, 29, 0.35);
      color: #fecaca;
      font-size: 0.8rem;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      border-radius: 12px;
      overflow: hidden;
      font-size: 0.82rem;
    }

    thead {
      background: linear-gradient(135deg, var(--bg-table-header), #020617);
    }

    thead th {
      padding: 8px 10px;
      text-align: left;
      font-weight: 500;
      border-bottom: 1px solid var(--border-soft);
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
    }

    tbody tr {
      background: var(--bg-table-row);
      border-bottom: 1px solid rgba(15, 23, 42, 0.8);
    }

    tbody tr:nth-child(even) {
      background: var(--bg-table-alt);
    }

    tbody tr:hover {
      background: rgba(34, 197, 94, 0.05);
    }

    tbody td {
      padding: 7px 10px;
      white-space: nowrap;
    }

    .tag-earn {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid rgba(234, 179, 8, 0.5);
      background: rgba(234, 179, 8, 0.13);
      color: #facc15;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .tag-earn span.dot {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: #facc15;
    }

    .tag-ok {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid rgba(34, 197, 94, 0.5);
      background: rgba(22, 163, 74, 0.22);
      color: var(--accent);
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .cell-muted {
      color: var(--text-dim);
    }

    @media (max-width: 900px) {
      .card {
        padding: 20px 16px 24px;
      }
      .header {
        flex-direction: column;
        align-items: flex-start;
      }
      table {
        font-size: 0.78rem;
        overflow-x: auto;
        display: block;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <div class="header">
        <div class="title-block">
          <div class="title-badge">
            <span class="dot"></span>
            Covered Call Radar
          </div>
          <h1>Premium Collection Scanner</h1>
          <p>Scan S&amp;P 500 or custom symbols for short-dated covered calls.</p>
        </div>
        <div class="pill">Yahoo Finance · Options · Intraday spot</div>
      </div>

      <form method="post">
        <div class="form-row">
          <div class="field" style="max-width:180px;">
            <label>Expiry (YYYY-MM-DD)</label>
            <input type="text" name="expiration" value="{{ expiration }}" placeholder="2025-11-14" />
          </div>

          <div class="field">
            <label>Universe</label>
            <div class="radio-row">
              <label class="radio-chip {% if universe == 'manual' %}active{% endif %}">
                <input type="radio" name="universe" value="manual" {% if universe == 'manual' %}checked{% endif %}/>
                Manual tickers
              </label>
              <label class="radio-chip {% if universe == 'sp500' %}active{% endif %}">
                <input type="radio" name="universe" value="sp500" {% if universe == 'sp500' %}checked{% endif %}/>
                S&amp;P 500 list
              </label>
            </div>
          </div>

          <div class="field" style="max-width:180px;">
            <label>Strike mode</label>
            <select name="mode">
              <option value="atm" {% if mode == 'atm' %}selected{% endif %}>ATM</option>
              <option value="itm" {% if mode == 'itm' %}selected{% endif %}>ITM</option>
              <option value="both" {% if mode == 'both' %}selected{% endif %}>Best of ATM / ITM</option>
            </select>
          </div>

          <div class="field" style="max-width:220px;">
            <label>Earnings before expiry</label>
            <select name="include_earn">
              <option value="yes" {% if include_earn == 'yes' %}selected{% endif %}>Include but flag</option>
              <option value="no" {% if include_earn == 'no' %}selected{% endif %}>Exclude those names</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="field">
            <label>Tickers (one per line)</label>
            <textarea name="tickers" placeholder="TSLA
AAPL
NVDA">{{ tickers }}</textarea>
          </div>
        </div>

        <div class="actions">
          <div class="helper-chip">S&amp;P 500 scans may take time · be patient</div>
          <button type="submit" class="btn-main">Scan</button>
        </div>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
      </form>

      {% if results %}
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Spot</th>
            <th>Expiry</th>
            <th>Strike</th>
            <th>Mid</th>
            <th>Premium %</th>
            <th>Next Earnings</th>
            <th>Earnings&nbsp;Before&nbsp;Expiry</th>
          </tr>
        </thead>
        <tbody>
          {% for r in results %}
          <tr>
            <td>{{ r["Symbol"] }}</td>
            <td class="cell-muted">{{ "%.2f"|format(r["Spot"]) }}</td>
            <td class="cell-muted">{{ r["Expiry"] }}</td>
            <td>{{ "%.2f"|format(r["Strike"]) }}</td>
            <td>{{ "%.2f"|format(r["Mid"]) }}</td>
            <td>{{ "%.2f"|format(r["PremiumReturn"]) }}</td>
            <td class="cell-muted">{{ r["NextEarnings"] or "N/A" }}</td>
            <td>
              {% if r["EarningsBeforeExpiry"] %}
                <span class="tag-earn"><span class="dot"></span> Yes</span>
              {% else %}
                <span class="tag-ok">Clean</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

MIN_OI = 1       # you can tighten these once everything works as expected
MIN_VOL = 0
ATM_TOL = 0.05   # 5 percent window around spot


def mid_price(bid, ask, last):
    vals = []
    if pd.notna(bid) and bid > 0:
        vals.append(float(bid))
    if pd.notna(ask) and ask > 0:
        vals.append(float(ask))
    if pd.notna(last) and last > 0:
        vals.append(float(last))
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid:
        return 0.5 * (float(bid) + float(ask))
    if vals:
        return max(vals)
    return np.nan


def pick_atm_row(calls_df: pd.DataFrame, spot: float) -> pd.Series | None:
    if calls_df is None or calls_df.empty:
        return None

    calls = calls_df.dropna(subset=["strike"]).copy()
    if calls.empty:
        return None

    calls["abs_dist"] = (calls["strike"] - spot).abs()
    calls["rel_dist"] = calls["abs_dist"] / spot

    window = calls[calls["rel_dist"] <= ATM_TOL]
    if window.empty:
        window = calls

    for _, r in window.sort_values(["abs_dist", "strike"]).iterrows():
        oi_ok = pd.notna(r.get("openInterest")) and r.get("openInterest") >= MIN_OI
        vol_ok = pd.notna(r.get("volume")) and r.get("volume") >= MIN_VOL
        bid = r.get("bid"); ask = r.get("ask")
        spread_ok = pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid
        if oi_ok and vol_ok and spread_ok:
            return r

    for _, r in window.sort_values(["abs_dist", "strike"]).iterrows():
        bid = r.get("bid"); ask = r.get("ask"); last = r.get("lastPrice")
        if (
            (pd.notna(bid) and bid > 0)
            or (pd.notna(ask) and ask > 0)
            or (pd.notna(last) and last > 0)
        ):
            return r

    return None


def pick_itm_row(calls_df: pd.DataFrame, spot: float) -> pd.Series | None:
    if calls_df is None or calls_df.empty:
        return None

    calls = calls_df.dropna(subset=["strike"]).copy()
    if calls.empty:
        return None

    itm = calls[calls["strike"] <= spot].copy()
    if itm.empty:
        return None

    itm["under_dist"] = (spot - itm["strike"]).abs()
    itm = itm.sort_values(["under_dist", "strike"], ascending=[True, False])

    for _, r in itm.iterrows():
        oi_ok = pd.notna(r.get("openInterest")) and r.get("openInterest") >= MIN_OI
        vol_ok = pd.notna(r.get("volume")) and r.get("volume") >= MIN_VOL
        bid = r.get("bid"); ask = r.get("ask")
        spread_ok = pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid
        if oi_ok and vol_ok and spread_ok:
            return r

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
    if mode == "atm":
        return pick_atm_row(calls_df, spot)
    if mode == "itm":
        return pick_itm_row(calls_df, spot)
    if mode == "both":
        cand = []
        r1 = pick_atm_row(calls_df, spot)
        if r1 is not None:
            cand.append(r1)
        r2 = pick_itm_row(calls_df, spot)
        if r2 is not None:
            cand.append(r2)
        if not cand:
            return None
        best = None
        best_ret = -1e9
        for r in cand:
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
    return pick_atm_row(calls_df, spot)

def fetch_sp500_tickers() -> list:
    """Return a fixed list of S&P 500 tickers (Yahoo Finance format)."""
    return [
        "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","LLY","TSLA","UNH",
        "XOM","JPM","V","AVGO","PG","MA","HD","JNJ","COST","CVX","WMT",
        "MRK","ABBV","PEP","KO","BAC","NFLX","PFE","ORCL","DIS",
        "CRM","ABT","ACN","CSCO","TMO","MCD","LIN","AMD","WFC","INTU",
        "COP","RTX","ADBE","PM","NKE","NEE","TXN","UPS","BMY",
        "UNP","SPGI","CAT","AMAT","MS","LOW","HON","GS","IBM","INTC",
        "QCOM","GE","AXP","BLK","CVS","NOW","DE","MDT","SBUX","C",
        "AMGN","LMT","ISRG","ADI","MU","SYK","ELV","BKNG","GILD","PLD",
        "TJX","PGR","TGT","MMC","CB","PNC","CI","MO","BDX","ZTS",
        "DUK","SO","BSX","ADP","GM","EQIX","VRTX","MMC","SHW","AON",
        "USB","REGN","ITW","CSX","PXD","DHR","SLB","EOG","ETN","EMR",
        "NSC","CL","FDX","FIS","APD","SPG","COF","KDP","ROP",
        "MPC","MSI","EXC","AIG","F","WELL","OXY","MAR","NOC","CME",
        "CCI","MCO","D","VLO","ADM","ALL","TFC","ORLY","HUM","AEP",
        "CMG","PSA","HAL","AFL","TRV","WM","KR","SRE","LRCX","CTAS",
        "GIS","DLR","CNC","PH","ECL","PCAR","FTNT","PAYX","EW","HCA",
        "MNST","AMP","RSG","PSX","CTVA","KLAC","OKE","KMB","ROST",
        "IDXX","WMB","HPQ","CBRE","O","NEM","EA","A","PPG","AJG",
        "MSCI","EIX","AME","KMI","TDG","MKC","VICI","FAST","PEG","ILMN",
        "ODFL","VRSK","WEC","STZ","AVB","FISV","EXR","KEYS","CHTR","DLTR",
        "ANET","PAYC","FITB","DFS","HES","ED","DHI","TT","NUE","BLL",
        "ROK","AWK","MTB","YUM","MLM","CPRT","RF","WBD","OTIS","XYL",
        "IR","ZBH","PRU","DOW","TEL","VMC","LEN","GWW","DTE","WST",
        "NTRS","HBAN","NDAQ","HIG","STT","ES","CTSH","GLW","LH","CMS",
        "TSN","TRMB","EFX","CAG","ALB","CDNS","NTAP","HPE","RMD","LUV",
        "VTR","WRB","SWK","CINF","EQR","DG","MTD","STE","AVY",
        "PKG","DRI","HWM","ETR","OMC","WAT","WBA","AKAM","IP","FANG",
        "ROK","PPL","SYF","DVN","TTWO","HSY","AEE","CNP",
        "CMS","INCY","TROW","CARR","IFF","MOS","EXPD","COO",
        "HII","ZBRA","CMS","LYB","PWR","ATO","A","XYL","EXPE","MTB",
        "LNT","ED","IRM","TECH","APA","CHD","HRL","CMS","UAL","LKQ",
        "NVR","WRK","JNPR","WDC","CMA","HST","XRAY","BEN","KIM","NDSN",
        "KEY","BIO","RJF","J","MAS","POOL","GNRC","LKQ","HOLX",
        "TXT","HII","CLX","CF","LW","SWKS","DVA","CBOE","JKHY","CPT",
        "REG","CPB","ESS","L","GL","PKG","JCI","K","WRB","ATO",
        "PFG","FRT","SNA","MGM","NI","HII","HBAN","JKHY","AES",
        "ROL","CINF","PKI","EVRG","LW","WRK","FDS","TAP","BAX",
        "JKHY","LW","PKG","ROL","AIZ","MKTX","WYNN","BBY","KEYS","HAS",
        "SJM","IFF","MHK","AOS","ALLE","WRB","ROL","LW","BEN","BIO",
        "J","NRG","NDSN","TAP","BIO","ROL","LW","J","NDSN","AOS",
        "YINN","SOXL","SOXS","SQQQ","TQQQ","TSLL","TSLQ","VXX","AMDL","NVDL","BABA"
    ]


def get_next_earnings(tkr: yf.Ticker) -> str | None:
    """Return next earnings date as YYYY-MM-DD string if available."""
    try:
        df = tkr.get_earnings_dates(limit=1)
        if df is not None and not df.empty:
            next_date = pd.to_datetime(df.index[0])
            return next_date.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def scan_single(symbol, expiration, mode, include_earn_bef_exp):
    print(f"Scanning {symbol} for {expiration} ({mode})")
    try:
        t = yf.Ticker(symbol)

        # 1) Confirm expiration exists
        try:
            exps = list(t.options)
            print(f"{symbol} expirations: {exps[:10]} ... total={len(exps)}")
            if expiration not in exps:
                print(f"{symbol}: requested expiration {expiration} not in options list")
                return None
        except Exception as e:
            print(f"{symbol}: failed to get options list: {e}")
            return None

        # 2) Spot price
        hist = t.history(period="1d")
        if hist.empty:
            print(f"{symbol}: empty price history")
            return None
        spot = float(hist["Close"].iloc[-1])
        print(f"{symbol}: spot={spot}")

        # 3) Option chain
        chain = t.option_chain(expiration)
        calls = chain.calls
        print(f"{symbol}: {len(calls)} calls rows")

        # 4) Pick call row
        row = pick_call_row(calls, spot, mode)
        if row is None:
            print(f"{symbol}: no call row selected")
            return None

        bid = float(row.get("bid") or float("nan"))
        ask = float(row.get("ask") or float("nan"))
        last = float(row.get("lastPrice") or float("nan"))
        mid = mid_price(bid, ask, last)
        if not np.isfinite(mid):
            print(f"{symbol}: mid price not finite")
            return None

        strike = float(row["strike"])
        prem_ret_pct = ((strike - spot) + mid) / spot * 100.0

        # 5) Earnings info: always define defaults
        next_earn_str = "N/A"
        earn_before = False
        try:
            cal = t.get_earnings_dates(limit=4)
            if cal is not None and not cal.empty:
                next_earn_date = cal.index[0].to_pydatetime().date()
                next_earn_str = next_earn_date.isoformat()
                exp_date = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
                earn_before = next_earn_date <= exp_date

                # If user chose to exclude earnings before expiry, skip
                if not include_earn_bef_exp and earn_before:
                    print(f"{symbol}: earnings before expiry, skipping due to setting")
                    return None
        except Exception as e:
            print(f"{symbol}: earnings lookup error: {e}")

        # 6) Build result dict
        return {
            "Symbol": symbol,
            "Spot": round(spot, 2),
            "Expiry": expiration,
            "Strike": round(strike, 2),
            "Mid": round(mid, 2),
            "PremiumReturn": round(prem_ret_pct, 2),
            "NextEarnings": next_earn_str,
            "EarningsBeforeExpiry": earn_before,
        }

    except Exception as e:
        print(f"Error on {symbol}: {e}")
        return None


import time
SLEEP_BETWEEN = 0.15  # seconds between tickers

@app.route("/", methods=["GET", "POST"])
def index():
    expiration = ""
    mode = "atm"
    tickers_text = ""
    universe = "manual"
    include_earn = "yes"
    error = None
    results = []

    if request.method == "POST":
        expiration = request.form.get("expiration", "").strip()
        universe = request.form.get("universe", "manual")
        mode = request.form.get("mode", "atm").strip().lower()
        include_earn = request.form.get("include_earn", "yes")
        include_earn_bef_exp = (include_earn == "yes")
        tickers_text = request.form.get("tickers", "")

        if not expiration:
            error = "Expiration is required."
        else:
            if universe == "sp500":
                tickers = fetch_sp500_tickers()
            else:
                tickers = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]

            if not tickers:
                error = "Please enter at least one ticker or choose S and P 500."
            else:
                for sym in tickers:
                    res = scan_single(sym, expiration, mode, include_earn_bef_exp)
                    if res:
                        results.append(res)

                    # gentle delay between requests
                    time.sleep(SLEEP_BETWEEN)

                if not results:
                    if universe == "sp500":
                        error = "No valid options found for that date in the S and P 500."
                    else:
                        error = "No valid options found for the given date and tickers."
                else:
                    results.sort(key=lambda r: r["PremiumReturn"], reverse=True)

    return render_template_string(
        HTML,
        expiration=expiration,
        mode=mode,
        tickers=tickers_text,
        universe=universe,
        include_earn=include_earn,
        error=error,
        results=results,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
