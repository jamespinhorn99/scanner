"""COT scoring backtest.  Run:  python cot_backtest.py
Compares the OLD COT signal (long / neutral / short from net position only) with the NEW COT score
(position + momentum + crowding) and shows what price did over the following 1, 4 and 8 weeks.
Needs the same packages as scanner_18.py and an internet connection. Put it in the same folder as scanner_18.py.
It prints a summary and writes cot_backtest_details.csv (one row per instrument, week and signal)."""
import sys, datetime as dt
import numpy as np, pandas as pd

HORIZONS = (1, 4, 8)      # weeks after the COT release
MIN_SIGNAL = 0.25         # a score beyond +/-0.25 counts as a bullish / bearish call (the old signal is exactly -1, 0 or +1)
NEUTRAL = 3.0             # same as COT_NEUTRAL in the scanner


def cot_history(df, key, invert):
    """Weekly COT numbers as they would have looked on each report date, using only data up to that date."""
    names = df["Market and Exchange Names"].str.upper()
    d = df[names.str.startswith(key.upper()) & ~names.str.contains("CONSOLIDATED")]
    g = d.groupby("date")
    s, oi = g["net"].sum(), g["oi"].sum()
    if len(s) < 80:
        return None
    share = s / oi * 100
    out = pd.DataFrame({"net": s, "share": share})
    out["pct"] = s.rolling(156, min_periods=52).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
    for tag, k in (("1w", 1), ("1m", 4), ("3m", 13)):
        ch = share.diff(k)
        out["z_" + tag] = ch / ch.rolling(156, min_periods=52).std()
    out = out.dropna()
    if invert:
        out["share"], out["net"], out["pct"] = -out["share"], -out["net"], 100 - out["pct"]
        for tag in ("1w", "1m", "3m"):
            out["z_" + tag] = -out["z_" + tag]
    return out


def old_signal(r):
    return 1.0 if r.share >= NEUTRAL else -1.0 if r.share <= -NEUTRAL else 0.0


def new_signal(r):
    """Same maths as the dashboard: 0.6 x position + 0.4 x momentum, trimmed 30% when the position is crowded."""
    z = [r.z_1w, r.z_1m, r.z_3m]
    mom = .2 * np.clip(z[0] / 2, -1, 1) + .4 * np.clip(z[1] / 2, -1, 1) + .4 * np.clip(z[2] / 2, -1, 1)
    if not (all(v > 0 for v in z) or all(v < 0 for v in z)):
        mom *= .5                                    # the three periods disagree
    pos = .5 * np.tanh(r.share / 15) + .5 * (r.pct - 50) / 50
    s = .6 * pos + .4 * mom
    crowd = 1 if r.pct >= 90 else -1 if r.pct <= 10 else 0
    if crowd and s * crowd > 0:
        s *= .7                                      # crowded and still leaning the same way
    return float(np.clip(s, -1, 1))


def forward_returns(prices, release, weeks):
    """Price change from the first close on or after the Friday release to `weeks` later (no peeking at the report date)."""
    idx = prices.index
    i = idx.searchsorted(release)
    if i >= len(idx):
        return None
    out = {}
    for w in weeks:
        j = idx.searchsorted(idx[i] + pd.Timedelta(days=7 * w))
        if j >= len(idx) or (idx[j] - idx[i]).days > 7 * w + 6:
            out[w] = np.nan
        else:
            out[w] = float(prices.iloc[j] / prices.iloc[i] - 1)
    return out


def run(cot_df, price_map, watch):
    rows = []
    for sym, (mkt, tick, key, inv) in watch.items():
        h, p = cot_history(cot_df, key, inv), price_map.get(sym)
        if h is None or p is None or len(p) < 100:
            print("skipped", sym)
            continue
        for date, r in h.iterrows():
            fr = forward_returns(p, date + pd.Timedelta(days=3), HORIZONS)   # COT is published on the Friday after the Tuesday it describes
            if not fr:
                continue
            row = {"symbol": sym, "date": date, "old": old_signal(r), "new": new_signal(r)}
            row.update({"ret_%dw" % w: fr[w] for w in HORIZONS})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(d):
    out = []
    for w in HORIZONS:
        col = "ret_%dw" % w
        for name in ("old", "new"):
            x = d[[name, col]].dropna()
            calls = x[x[name].abs() >= MIN_SIGNAL]
            dirn = np.sign(calls[name]) * calls[col] * 100
            out.append({"horizon_weeks": w, "signal": name, "calls": len(calls), "calls_per_year": round(len(calls) / max(1, (d["date"].max() - d["date"].min()).days / 365), 1),
                        "hit_rate_%": round((dirn > 0).mean() * 100, 1) if len(calls) else np.nan,
                        "avg_return_in_signal_direction_%": round(dirn.mean(), 3) if len(calls) else np.nan,
                        "rank_correlation": round(x[name].corr(x[col], method="spearman"), 3)})
    return pd.DataFrame(out)


def main():
    import yfinance as yf
    import scanner_18 as sc
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print("Downloading %d years of COT data..." % years)
    cot_df = sc.load_cot(years)
    start = cot_df["date"].min() - pd.Timedelta(days=10)
    prices = {}
    for sym, (mkt, tick, key, inv) in sc.WATCH.items():
        try:
            c = yf.download(tick, start=start, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
            prices[sym] = c
        except Exception as ex:
            print("no prices for", sym, ex)
    d = run(cot_df, prices, sc.WATCH)
    d.to_csv("cot_backtest_details.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPooled over all instruments (%d weekly observations, %s to %s)" % (len(d), d["date"].min().date(), d["date"].max().date()))
    print(summarise(d).to_string(index=False))
    print("\nHow to read it: hit_rate_% is how often price moved the way the signal said. Above 50 is better than a coin flip.")
    print("avg_return_in_signal_direction_% is the average price move in the signal's direction (before costs and carry).")
    print("rank_correlation is how closely the score tracks later price changes (0 = no link). Weekly observations overlap at the longer")
    print("horizons, so treat small differences as noise. Only trust a result that is clearly better on most horizons.")


if __name__ == "__main__":
    main()
