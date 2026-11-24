"""4H EMA 20/50 + COT scanner.  Output: scanner_results.json (load it in the dashboard).

pip install yfinance pandas cot_reports
python scanner.py            # run on demand
Schedule it (cron / Task Scheduler) every 4 hours; COT only changes weekly (Fridays).
"""
import json, datetime as dt
import pandas as pd, yfinance as yf, cot_reports as cot

# symbol -> (market, yfinance ticker, COT market-name substring, invert COT?)
# invert=True when the futures contract is the foreign currency and the pair quotes USD first.
WATCH = {
    "EURUSD": ("Forex",   "EURUSD=X", "EURO FX - ",         False),
    "GBPUSD": ("Forex",   "GBPUSD=X", "BRITISH POUND",      False),
    "AUDUSD": ("Forex",   "AUDUSD=X", "AUSTRALIAN DOLLAR",  False),
    "NZDUSD": ("Forex",   "NZDUSD=X", "NZ DOLLAR",          False),
    "USDJPY": ("Forex",   "JPY=X",    "JAPANESE YEN",       True),
    "USDCAD": ("Forex",   "CAD=X",    "CANADIAN DOLLAR",    True),
    "USDCHF": ("Forex",   "CHF=X",    "SWISS FRANC",        True),
    "XAUUSD": ("Metals",  "GC=F",     "GOLD - COMMODITY",   False),
    "XAGUSD": ("Metals",  "SI=F",     "SILVER - COMMODITY", False),
    "HG":     ("Metals",  "HG=F",     "COPPER",             False),
    "ES":     ("Futures", "ES=F",     "E-MINI S&P 500",     False),
    "NQ":     ("Futures", "NQ=F",     "NASDAQ MINI",        False),
    "YM":     ("Futures", "YM=F",     "DJIA",               False),
    "RTY":    ("Futures", "RTY=F",    "RUSSELL E-MINI",     False),
    "ZN":     ("Futures", "ZN=F",     "UST 10Y NOTE",       False),
    "CL":     ("Futures", "CL=F",     "CRUDE OIL, LIGHT SWEET - NEW", False),
    "BTC":    ("Crypto",  "BTC-USD",  "BITCOIN - CHICAGO",  False),
}
CLOSED_CANDLES_ONLY = True   # True: ignore the 4H candle still forming, so every cross is confirmed
COT_NEUTRAL = 3.0   # net position under this % of open interest counts as neutral (a starting guess, worth testing)
LONG_PCT, SHORT_PCT = 60, 40   # COT percentile thresholds for a bullish / bearish bias


def load_cot(years=4):
    now = dt.date.today().year
    frames = [cot.cot_year(y, cot_report_type="legacy_fut") for y in range(now - years + 1, now + 1)]
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["As of Date in Form YYYY-MM-DD"])
    df["net"] = df["Noncommercial Positions-Long (All)"] - df["Noncommercial Positions-Short (All)"]
    df["oi"] = df["Open Interest (All)"]
    return df.sort_values("date")


def cot_stats(df, key, invert):
    names = df["Market and Exchange Names"].str.upper()
    d = df[names.str.startswith(key.upper()) & ~names.str.contains("CONSOLIDATED")]
    g = d.groupby("date")
    s, oi = g["net"].sum().tail(156), g["oi"].sum().tail(156)   # ~3 years of weekly data
    if len(s) < 20:
        return None
    net, prev = int(s.iloc[-1]), int(s.iloc[-2])
    pct = round(float((s <= net).mean() * 100))
    share = round(net / float(oi.iloc[-1]) * 100, 1)   # net position as % of open interest
    if invert:
        pct, net, prev, share = 100 - pct, -net, -prev, -share
    return {"net": net, "change": net - prev, "pct": pct, "share": share, "asof": str(s.index[-1].date())}


def ema_stats(ticker):
    h = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
    if h.empty:
        return None
    close = h["Close"].squeeze().resample("4h").last().dropna()
    if CLOSED_CANDLES_ONLY and close.index[-1] + pd.Timedelta(hours=4) > pd.Timestamp.now(tz=close.index.tz):
        close = close.iloc[:-1]   # drop the unfinished 4H candle
    fast, slow = close.ewm(span=20, adjust=False).mean(), close.ewm(span=50, adjust=False).mean()
    up = fast > slow
    flips = up.ne(up.shift()).iloc[1:]
    bars_since = int(len(up) - 1 - flips[flips].index.map(up.index.get_loc).max())
    return {"state": "bull" if up.iloc[-1] else "bear", "since": bars_since,
            "gap": round(float((fast.iloc[-1] - slow.iloc[-1]) / slow.iloc[-1] * 100), 3),
            "price": round(float(close.iloc[-1]), 5),
            "chart": [round(float(x), 5) for x in close.tail(120)],
            "ema20": [round(float(x), 5) for x in fast.tail(120)],
            "ema50": [round(float(x), 5) for x in slow.tail(120)],
            "ema200": [round(float(x), 5) for x in close.ewm(span=200, adjust=False).mean().tail(120)]}  # last 120 four-hour closes for the dashboard


# ---------- macro: rate differentials, real rates, data surprises ----------
# FRED series IDs (free, no key). I couldn't test them live: if one fails or stops updating,
# the script prints "rates skip" and you can swap in another ID from fred.stlouisfed.org.
RATES = {"USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "GBP": "IR3TIB01GBM156N",
         "JPY": "IR3TIB01JPM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
         "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N"}                      # 3-month rates
CPI = {"USD": ("CPIAUCSL", 12), "EUR": ("CP0000EZ19M086NEST", 12), "GBP": ("GBRCPIALLMINMEI", 12),
       "JPY": ("JPNCPIALLMINMEI", 12), "AUD": ("AUSCPIALLQINMEI", 4),   # AUD CPI is quarterly
       "CAD": ("CANCPIALLMINMEI", 12), "CHF": ("CHECPIALLMINMEI", 12), "NZD": ("NZLCPIALLQINMEI", 4)}
PAIRS = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
         "USDJPY": ("USD", "JPY"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF")}
clip = lambda x: max(-1.0, min(1.0, float(x)))


def fred(sid):
    d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0])
    return s.dropna()


def ccy_rates():
    """Nominal 3M rate and real rate (nominal minus CPI inflation, year on year) per currency."""
    out = {}
    for c, sid in RATES.items():
        try:
            nom = float(fred(sid).iloc[-1])
            cid, n = CPI[c]
            cpi = fred(cid)
            out[c] = {"nom": nom, "real": nom - float((cpi.iloc[-1] / cpi.iloc[-1 - n] - 1) * 100)}
        except Exception as ex:
            print(f"rates skip {c}: {ex}")
    return out


def us_real_yield():
    """US 10y TIPS real yield: latest level and change over ~20 trading days."""
    try:
        s = fred("DFII10")
        return float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-21])
    except Exception as ex:
        print(f"real yield skip: {ex}")


def surprises(path="releases.csv"):
    """Score data releases per currency, -1..1. Needs releases.csv (see template):
    date,currency,indicator,actual,forecast,direction   (direction: 1 if higher is bullish, -1 if bearish)
    Surprise = (actual - forecast) in units of that indicator's typical miss, fading with a 15-day half-life."""
    try:
        d = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return {}
    if d.empty:
        return {}
    d["err"] = d["actual"] - d["forecast"]
    sd = d.groupby("indicator")["err"].transform(lambda s: s.std() if len(s) >= 4 else float("nan"))
    scale = sd.fillna(d["forecast"].abs() * 0.1).replace(0, 0.1)
    age = (pd.Timestamp.today() - d["date"]).dt.days
    d["pts"] = d["direction"] * d["err"] / scale * 0.5 ** (age / 15)
    d = d[age <= 30]
    return {c: clip(g["pts"].sum() / 3) for c, g in d.groupby("currency")}


def macro(sym, rates, surp, tips):
    """Macro score -100..100 (positive = bullish for the instrument)."""
    if sym in PAIRS:
        b, q = PAIRS[sym]
        if b not in rates or q not in rates:
            return None
        rd, rr = rates[b]["nom"] - rates[q]["nom"], rates[b]["real"] - rates[q]["real"]
        s = 0.35 * clip(rd / 2) + 0.35 * clip(rr / 2) + 0.3 * clip(surp.get(b, 0) - surp.get(q, 0))
        return {"macro_score": round(100 * s), "rate_diff": round(rd, 2), "real_diff": round(rr, 2)}
    if tips is None or sym in ("CL", "HG", "BTC"):
        return None
    level, chg = tips
    us = surp.get("USD", 0)
    if sym in ("XAUUSD", "XAGUSD"):   # gold likes falling and low real yields, and a weak dollar
        s = 0.5 * clip(-chg / 0.3) + 0.3 * clip(-level / 2) + 0.2 * clip(-us)
    elif sym == "ZN":     # bond prices rise when real yields fall and US data softens
        s = 0.6 * clip(-chg / 0.3) + 0.4 * clip(-us)
    else:                 # equity index futures: falling real yields and firm US data help
        s = 0.6 * clip(-chg / 0.3) + 0.4 * clip(us)
    return {"macro_score": round(100 * s), "note": f"US real 10y {level:.2f}%, {chg:+.2f} in 20d"}


# ---------- currency strength (display only, not part of the score) ----------
STRENGTH_ORDER = ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]   # standard quoting order


def currency_strength():
    """Average price change of each currency against the other 7 (all 28 crosses), over 24 hours and 5 days."""
    pairs = [(a, b) for i, a in enumerate(STRENGTH_ORDER) for b in STRENGTH_ORDER[i + 1:]]
    tick = [f"{a}{b}=X" for a, b in pairs]
    px = yf.download(tick, period="30d", interval="1h", progress=False, auto_adjust=True)["Close"]
    px = px.resample("4h").last().dropna(how="all").ffill()
    if CLOSED_CANDLES_ONLY and px.index[-1] + pd.Timedelta(hours=4) > pd.Timestamp.now(tz=px.index.tz):
        px = px.iloc[:-1]   # drop the unfinished 4H candle
    out = {}
    for name, n in (("d1", 6), ("d5", 30)):      # 6 four-hour bars = 24h, 30 bars = about 5 trading days
        s = dict.fromkeys(STRENGTH_ORDER, 0.0)
        for (a, b), t in zip(pairs, tick):
            if t not in px:
                continue
            r = float(px[t].iloc[-1] / px[t].iloc[-1 - n] - 1) * 100
            if r == r:                            # skip missing data
                s[a] += r / 7
                s[b] -= r / 7
        out[name] = {c: round(v, 3) for c, v in s.items()}
    return out


def strength_gap(sym, st):
    """Places between the two currencies in the 24h ranking (positive: first currency ranks higher)."""
    if sym not in PAIRS or not st:
        return {}
    b, q = PAIRS[sym]
    rank = sorted(st["d1"], key=st["d1"].get, reverse=True)
    return {"strength_gap": rank.index(q) - rank.index(b)}


def main():
    cot_df, rows = load_cot(), []
    rates, surp, tips = ccy_rates(), surprises(), us_real_yield()
    try:
        st = currency_strength()
    except Exception as ex:
        print(f"strength skip: {ex}")
        st = {}
    for sym, (mkt, tick, key, inv) in WATCH.items():
        try:
            e, c = ema_stats(tick), cot_stats(cot_df, key, inv)
            if e and c:
                rows.append({"symbol": sym, "market": mkt, **e, **{"cot_" + k: v for k, v in c.items()},
                             **(macro(sym, rates, surp, tips) or {}), **strength_gap(sym, st)})
        except Exception as ex:
            print(f"skip {sym}: {ex}")
    out = {"generated": dt.datetime.utcnow().isoformat(timespec="minutes") + "Z",
           "thresholds": [SHORT_PCT, LONG_PCT], "closed_only": CLOSED_CANDLES_ONLY, "cot_neutral": COT_NEUTRAL, "strength": st, "rows": rows}
    json.dump(out, open("scanner_results.json", "w"), indent=1)
    print(f"{len(rows)} instruments written to scanner_results.json")


if __name__ == "__main__":
    main()
