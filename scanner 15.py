"""4H EMA 20/50 + COT scanner.  Output: scanner_results.json (load it in the dashboard).

pip install yfinance pandas cot_reports
python scanner.py            # run on demand
Schedule it (cron / Task Scheduler) every 4 hours; COT only changes weekly (Fridays).
"""
import io, json, os, time, requests, datetime as dt
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
# Each currency has a list of series to try in order; the first one that loads AND is recent is used (see first_fresh
# below). The OECD series on FRED sometimes stop or lag, so there are central-bank and growth-rate fallbacks.
# "rate:" means the series is already a year-on-year % change. The log prints "economy skip" for every one that fails.
RATE_IDS = {"USD": ["IR3TIB01USM156N", "DGS3MO", "FEDFUNDS"], "EUR": ["IR3TIB01EZM156N", "ECBDFR", "IRSTCI01EZM156N"],
            "GBP": ["IR3TIB01GBM156N", "IRSTCI01GBM156N"], "JPY": ["IR3TIB01JPM156N", "IRSTCI01JPM156N"],
            "AUD": ["IR3TIB01AUM156N", "IRSTCI01AUM156N"], "CAD": ["IR3TIB01CAM156N", "IRSTCI01CAM156N"],
            "CHF": ["IR3TIB01CHM156N", "IRSTCI01CHM156N"], "NZD": ["IR3TIB01NZM156N", "IRSTCI01NZM156N"]}
CPI_IDS = {"USD": ["CPIAUCSL"], "EUR": ["CP0000EZ19M086NEST", "CP0000EZ20M086NEST", "rate:CPALTT01EZM659N"],
           "GBP": ["GBRCPIALLMINMEI", "rate:CPALTT01GBM659N"], "JPY": ["JPNCPIALLMINMEI", "rate:CPALTT01JPM659N"],
           "AUD": ["AUSCPIALLQINMEI", "rate:CPALTT01AUQ659N"], "CAD": ["CANCPIALLMINMEI", "rate:CPALTT01CAM659N"],
           "CHF": ["CHECPIALLMINMEI", "rate:CPALTT01CHM659N"], "NZD": ["NZLCPIALLQINMEI", "rate:CPALTT01NZQ659N"]}
PAIRS = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
         "USDJPY": ("USD", "JPY"), "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF")}
# BIS (Bank for International Settlements) is tried LAST for every currency, after the FRED series above fail or are out of date.
# Policy rates: dataset WS_CBPOL. Inflation: dataset WS_LONG_CPI, code 771 = year-on-year % change.
BIS_AREA = {"USD": "US", "EUR": "XM", "GBP": "GB", "JPY": "JP", "AUD": "AU", "CAD": "CA", "CHF": "CH", "NZD": "NZ"}
for _c, _a in BIS_AREA.items():
    RATE_IDS[_c].append(f"bis:WS_CBPOL/M.{_a}")
    if _c not in ("USD", "EUR"):
        CPI_IDS[_c].append(f"rate:bis:WS_LONG_CPI/M.{_a}.771")
clip = lambda x: max(-1.0, min(1.0, float(x)))


def fred(sid):
    for attempt in range(3):   # FRED sometimes refuses or times out; wait a moment and try again
        try:
            d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0])
    return s.dropna()


def ccy_rates():
    """Short-term rate and real rate (rate minus CPI inflation, year on year) per currency, with freshness checks."""
    out = {}
    try:   # last good rates = the ones saved in the previous scanner_results.json
        saved = json.load(open("scanner_results.json")).get("rates", {})
    except Exception:
        saved = {}
    for c in RATE_IDS:
        try:
            r, k = first_fresh(RATE_IDS[c], f"{c} rate"), first_fresh(CPI_IDS[c], f"{c} inflation")
            if not r or not k:
                raise ValueError("no recent rate or inflation series found")
            rs, _, rid, _, rold = r
            cs, n, cid, crate, cold = k
            nom = float(rs.iloc[-1])
            infl = float(cs.iloc[-1]) if crate else float((cs.iloc[-1] / cs.iloc[-1 - n] - 1) * 100)
            out[c] = {"nom": nom, "real": nom - infl, "infl": round(infl, 2), "asof": str(rs.index[-1].date()),
                      "rate_series": rid, "cpi_series": cid, "old": bool(rold or cold)}
        except Exception as ex:
            print(f"rates skip {c}: {ex}")
            if c in saved:   # fetch failed: reuse the last good values and mark them as old
                out[c] = dict(saved[c], old=True, cached=True)
                print(f"rates {c}: using saved values from {saved[c].get('asof')}")
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
    Each release gets a score from -1 to +1: the miss (actual - forecast) in units of that indicator's typical miss,
    capped so one big miss cannot dominate. A currency's score is the average of its last 30 days of releases, newer ones
    counting more (15-day half-life), pulled toward 0 when there are only a few releases."""
    try:
        d = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError:
        return {}
    if d.empty:
        return {}
    d["err"] = d["actual"] - d["forecast"]
    sd = d.groupby("indicator")["err"].transform(lambda s: s.std() if len(s) >= 4 else float("nan"))
    scale = sd.fillna(d["forecast"].abs() * 0.1).clip(lower=0.2)   # typical miss; 10% of the forecast until an indicator has 4+ entries
    age = (pd.Timestamp.today() - d["date"]).dt.days
    d["w"] = 0.5 ** (age / 15)
    d["pts"] = (d["direction"] * d["err"] / scale).clip(-1, 1)
    d = d[age <= 30]
    return {c: clip((g["pts"] * g["w"]).sum() / (g["w"].sum() + 2)) for c, g in d.groupby("currency")}


# ---------- economy scorecard: growth, jobs and data surprises per currency ----------
# Each input has a list of FRED series to try in order. The first one that loads AND is up to date is used, so a
# dead or renamed series just falls through to the next. The OECD "MEI" series (e.g. euro area unemployment, industrial
# production) stopped updating on FRED, which is why the first version showed so many gaps. Growth is now real GDP.
# Not every ID below is confirmed live: the log prints "economy skip" for every one that fails.
ECON = {
    "USD": {"growth": ["GDPC1"], "jobs": ["UNRATE"]},
    "EUR": {"growth": ["CLVMNACSCAB1GQEA19", "CLVMNACSCAB1GQEA20", "NGDPRSAXDCEZQ"],
            "jobs": ["ecb:LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T", "LRHUTTTTEA20M156S"]},
    "GBP": {"growth": ["NGDPRSAXDCGBQ", "CLVMNACSCAB1GQUK"], "jobs": ["LRHUTTTTGBM156S", "LRUNTTTTGBM156S"]},
    "JPY": {"growth": ["NGDPRSAXDCJPQ", "JPNRGDPEXP"], "jobs": ["LRHUTTTTJPM156S", "LRUNTTTTJPM156S"]},
    "AUD": {"growth": ["NGDPRSAXDCAUQ"], "jobs": ["LRHUTTTTAUM156S", "LRUNTTTTAUM156S"]},
    "CAD": {"growth": ["NGDPRSAXDCCAQ"], "jobs": ["LRUNTTTTCAM156S", "LRHUTTTTCAM156S"]},
    "CHF": {"growth": ["NGDPRSAXDCCHQ", "CLVMNACSCAB1GQCH"], "jobs": ["LRUN74TTCHQ156S", "LRUN74TTCHQ156N", "LRUNTTTTCHQ156S", "LRHUTTTTCHQ156S", "LRHUTTTTCHM156S"]},
    "NZD": {"growth": ["NGDPRSAXDCNZQ", "rate:NZLGDPRQPSMEI"], "jobs": ["LRHUTTTTNZQ156S", "LRUNTTTTNZQ156S"]},
}
# Prefixes: "ecb:" loads from the European Central Bank's free data API instead of FRED (the FRED euro area unemployment
# series is frozen). "rate:" marks a series that is already a year-on-year growth rate, not an index level.
ECON_W = {"growth": 0.35, "jobs": 0.35, "momentum": 0.3, "surprise": 0.2}   # weights, re-balanced over whatever is available
ECON_RELEASES_FIRST = True   # True: a currency with releases in releases.csv (last 30 days) is scored from those alone; FRED growth/jobs/momentum
                             #       stays in the background and is used for any currency with no recent releases.
                             # False: always blend FRED growth, jobs and momentum with releases.csv.


def ecb(path):
    d = pd.read_csv(f"https://data-api.ecb.europa.eu/service/data/{path}?format=csvdata")
    s = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    s.index = pd.to_datetime(d["TIME_PERIOD"])
    return s.dropna()


def bis(path):
    r = requests.get(f"https://stats.bis.org/api/v1/data/{path}", params={"format": "csv", "startPeriod": "2023-01"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    s.index = pd.to_datetime(d["TIME_PERIOD"])
    return s.dropna().sort_index()


def first_fresh(ids, label):
    """Return (series, points per year, id, is_rate, old) for the first series that loads and isn't out of date."""
    for sid in ids:
        try:
            rate = sid.startswith("rate:")
            name = sid.split(":", 1)[1] if rate else sid
            s = bis(name[4:]) if name.startswith("bis:") else ecb(name[4:]) if name.startswith("ecb:") else fred(name)
            gap = s.index.to_series().diff().dt.days.median()
            n = 4 if gap > 60 else 12   # quarterly or monthly
            age = (pd.Timestamp.today() - s.index[-1]).days   # dates are period starts, so quarterly data looks older
            if age > (240 if name.startswith("bis:") else 420 if n == 4 else 150):   # BIS inflation can lag a few months (e.g. Australia)
                raise ValueError(f"last data {s.index[-1].date()}, out of date")
            return s, n, name, rate, age > (250 if n == 4 else 100)
        except Exception as ex:
            print(f"economy skip {label} {sid}: {ex}")
    return None


def economy(surp):
    """Per currency: growth, jobs and surprise scores (-1..1) and an overall score -100..100."""
    out = {}
    for c, ids in ECON.items():
        if ECON_RELEASES_FIRST and c in surp:   # releases take over: recent releases against forecast, -1..1 scaled to -100..100
            out[c] = {"surprise": {"score": round(float(surp[c]), 2)}, "n": 1, "score": round(100 * float(surp[c])), "source": "releases"}
            continue
        comp, mom, mom_scores = {}, {}, []
        r = first_fresh(ids["growth"], f"{c} growth")
        if r:
            s, n, sid, rate, old = r
            g = float(s.iloc[-1]) if rate else float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)
            comp["growth"] = {"value": round(g, 2), "score": round(clip(g / 3), 2), "asof": str(s.index[-1].date()),
                              "series": sid, "old": old}
            prev = float(s.iloc[-2]) if rate else float((s.iloc[-2] / s.iloc[-2 - n] - 1) * 100)
            accel = round(g - prev, 2)   # is growth speeding up or slowing down versus the previous reading?
            mom["growth_accel"] = accel
            mom_scores.append(clip(accel / 1.0))
        r = first_fresh(ids["jobs"], f"{c} jobs")
        if r:
            s, n, sid, rate, old = r
            u, chg = float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-1 - n])
            comp["jobs"] = {"value": round(u, 1), "change": round(chg, 2), "score": round(clip(-chg), 2),
                            "asof": str(s.index[-1].date()), "series": sid, "old": old}
            k = 3 if n == 12 else 1   # latest 3 months (monthly data) or latest quarter
            recent = round(u - float(s.iloc[-1 - k]), 2)
            mom["jobs_recent"] = recent
            mom_scores.append(clip(-recent / 0.5))
        if mom_scores:   # automatic stand-in for data surprises: is the economy improving or worsening lately?
            comp["momentum"] = {**mom, "score": round(sum(mom_scores) / len(mom_scores), 2)}
        if c in surp:
            comp["surprise"] = {"score": round(float(surp[c]), 2)}
        tw = sum(ECON_W[k] for k in comp)
        out[c] = {**comp, "n": len(comp), "source": "FRED",
                  "score": round(100 * sum(ECON_W[k] * v["score"] for k, v in comp.items()) / tw) if tw else None}
    return out


def macro(sym, rates, surp, tips, econ=None):
    """Macro score -100..100 (positive = bullish for the instrument)."""
    if sym in PAIRS:
        b, q = PAIRS[sym]
        if b not in rates or q not in rates:
            return None
        rd, rr = rates[b]["nom"] - rates[q]["nom"], rates[b]["real"] - rates[q]["real"]
        eb, eq = [((econ or {}).get(c) or {}).get("score") for c in (b, q)]
        if eb is not None and eq is not None:
            # economy score (growth, jobs, momentum and release surprises) difference, tied into the macro score
            ed = eb - eq
            s = 0.3 * clip(rd / 2) + 0.3 * clip(rr / 2) + 0.4 * clip(ed / 50)
            return {"macro_score": round(100 * s), "rate_diff": round(rd, 2), "real_diff": round(rr, 2), "econ_diff": round(ed)}
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
        econ = economy(surp)
    except Exception as ex:
        print(f"economy skip: {ex}")
        econ = {}
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
                             **(macro(sym, rates, surp, tips, econ) or {}), **strength_gap(sym, st)})
        except Exception as ex:
            print(f"skip {sym}: {ex}")
    try:
        dx = ema_stats("DX-Y.NYB")   # US dollar index, same 4H 20/50 EMA check
        dxy = {k: dx[k] for k in ("state", "since", "gap", "price")} if dx else None
    except Exception as ex:
        print(f"dxy skip: {ex}")
        dxy = None
    out = {"generated": dt.datetime.utcnow().isoformat(timespec="minutes") + "Z",
           "thresholds": [SHORT_PCT, LONG_PCT], "closed_only": CLOSED_CANDLES_ONLY, "cot_neutral": COT_NEUTRAL, "strength": st, "dxy": dxy, "economy": econ, "rates": {c: dict(d, real=round(d["real"], 2)) for c, d in rates.items()}, "rows": rows}
    json.dump(out, open("scanner_results.json", "w"), indent=1)
    print(f"{len(rows)} instruments written to scanner_results.json")


if __name__ == "__main__":
    main()
