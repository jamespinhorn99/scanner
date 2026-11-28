"""2-year government bond yield source test.
Run it once, then paste what it prints (it is also saved in yield_test_output.txt) back into the chat.
It only READS data from public websites. It changes nothing and needs no keys.
Needs: pandas and requests (you already have them because the scanner uses them).
"""
import csv, io, re
import pandas as pd, requests

HEAD = {"User-Agent": "Mozilla/5.0"}
TODAY = pd.Timestamp.today().normalize()
out = []


def get(url):
    r = requests.get(url, headers=HEAD, timeout=30)
    r.raise_for_status()
    return r.text


def series(values, dates):
    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").values,
                  index=pd.to_datetime(pd.Series(list(dates)), errors="coerce"))
    return s[s.index.notna()].dropna().sort_index()


# ---- one function per source: each returns a dated series of 2-year yields in % ----
def fred_us():
    d = pd.read_csv(io.StringIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2")))   # get() has a 30 second time limit
    return series(d.iloc[:, 1], d.iloc[:, 0])


def ecb_eur():   # euro area AAA government bonds (mostly German-like), daily
    d = pd.read_csv(io.StringIO(get("https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y?format=csvdata&lastNObservations=10")))
    return series(d["OBS_VALUE"], d["TIME_PERIOD"])


def boc_cad():   # Bank of Canada, 2-year benchmark bond
    j = requests.get("https://www.bankofcanada.ca/valet/observations/BD.CDN.2YR.DQ.YLD/json?recent=10", headers=HEAD, timeout=30).json()
    o = j["observations"]
    return series([x["BD.CDN.2YR.DQ.YLD"]["v"] for x in o], [x["d"] for x in o])


def mof_jpy():   # Japan Ministry of Finance, all JGB maturities, daily
    ls = get("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcm.csv").splitlines()
    i = next(k for k, l in enumerate(ls) if l.startswith("Date"))
    d = pd.read_csv(io.StringIO("\n".join(ls[i:])))
    return series(d["2Y"], d["Date"])


def rba_aud():   # Reserve Bank of Australia table F2, government bond yields
    rows = list(csv.reader(io.StringIO(get("https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"))))
    i = next(k for k, r in enumerate(rows) if r and r[0].strip().lower() == "title")
    j = next(c for c, v in enumerate(rows[i]) if re.search(r"(?<!\d)2[ -]year", v.lower()))
    data = [r for r in rows[i + 1:] if len(r) > j and pd.notna(pd.to_datetime(r[0], errors="coerce"))]
    return series([r[j] for r in data], [r[0] for r in data])


def snb_chf():   # Swiss National Bank, Swiss Confederation bond yields
    ls = get("https://data.snb.ch/api/cube/rendoblid/data/csv/en").splitlines()
    i = next(k for k, l in enumerate(ls) if l.replace('"', "").startswith("Date"))
    d = pd.read_csv(io.StringIO("\n".join(ls[i:])), sep=";")
    d = d[d["D0"] == "2J"]
    return series(d["Value"], d["Date"])


def stooq(sym):  # stooq.com daily history; the symbol names are a best guess, so both spellings get tried
    t = get(f"https://stooq.com/q/d/l/?s={sym}&i=d")
    d = pd.read_csv(io.StringIO(t))
    if "Close" not in d:
        raise ValueError("no data or blocked: " + t[:40].replace("\n", " "))
    return series(d["Close"], d["Date"])


def report(ccy, name, fn):
    try:
        s = fn()
        if s.empty:
            raise ValueError("no data returned")
        age = (TODAY - s.index[-1].normalize()).days
        line = f"{ccy:4} {name:24} {'GOOD' if age <= 5 else 'STALE':6} {float(s.iloc[-1]):7.3f}%  date {s.index[-1].date()}  ({age} days old)"
        ok = age <= 5
    except Exception as ex:
        line, ok = f"{ccy:4} {name:24} FAILED {str(ex)[:70]}", False
    print(line)
    out.append(line)
    return ok


def main():
    print("Testing 2-year yield sources (takes about a minute)...\n")
    for ccy, name, fn in (("USD", "FRED DGS2", fred_us), ("EUR", "ECB yield curve", ecb_eur),
                          ("CAD", "Bank of Canada", boc_cad), ("JPY", "Japan MOF", mof_jpy),
                          ("AUD", "RBA table F2", rba_aud), ("CHF", "SNB", snb_chf)):
        report(ccy, name, fn)
    print()
    for ccy, cc in (("GBP", "gb"), ("NZD", "nz"), ("USD", "us"), ("EUR", "de"), ("JPY", "jp"), ("AUD", "au"), ("CAD", "ca"), ("CHF", "ch")):
        for sym in (f"2{cc}y.b", f"2y{cc}y.b"):
            if report(ccy, f"stooq {sym}", lambda s=sym: stooq(s)):
                break
    open("yield_test_output.txt", "w").write("\n".join(out))
    print("\nDone. Copy everything above (or open yield_test_output.txt) and paste it into the chat.")


if __name__ == "__main__":
    main()
