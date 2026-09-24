"""
Regression test: the two charts this whole project was built from must be detected
exactly as they appear on screen.  Run it after ANY change to pattern.py:

    python3 -m nse_scanner.tests.test_exemplars

If a change to the detector breaks one of these, the detector no longer describes
the pattern we validated — do not ship it.
"""
import sys, json, urllib.request
import pandas as pd
sys.path.insert(0, ".")
from nse_scanner import pattern as P, config as C

C.MAX_SETUP_BARS = 400
C.TRIGGER_MODE = "rail_reclaim"

# The rail is the drift's floor.  Its exact value depends on where the drift is judged to end
# (the last drift bar vs the first breakdown bar's low) and can differ ~5% from the level drawn
# by hand on the chart.  The SHAKEOUT LOW and the ENTRY DATE are exact and are what drive the trade.
CASES = [
    # symbol, ignition date, expected rail, expected shakeout low, expected entry date, tolerance
    ("NIACL",      "2026-04-10", 154.00, 145.50, "2026-06-15", 0.05),
    ("LAMBODHARA", "2026-08-24", 127.37, 115.57, "2026-09-24", 0.03),
]

def fetch(sym):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}.NS?range=5y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    d = pd.DataFrame({"date": pd.to_datetime(pd.Series(r["timestamp"]), unit="s", utc=True)
                      .dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None),
                      "open": q["open"], "high": q["high"], "low": q["low"],
                      "close": q["close"], "volume": q["volume"]}).dropna().set_index("date")
    return d

def main():
    fails = 0
    for sym, ign, rail, sh_low, entry_date, tol in CASES:
        g = fetch(sym)
        out = P.find_setups(g, sym, last_only=False)
        hit = [s for s in out if s.pole_date == ign]
        if not hit:
            print(f"FAIL {sym}: no setup detected at the {ign} ignition"); fails += 1; continue
        s = hit[-1]
        checks = [("rail", s.rail, rail), ("shakeout low", s.sweep_low, sh_low)]
        ok = True
        for name, got, want in checks:
            good = abs(got - want) / want <= tol
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} {sym} {name}: got {got}  expected ≈{want}")
        good = str(s.entry_date)[:10] == entry_date
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {sym} entry date: got {str(s.entry_date)[:10]}  expected {entry_date}")
        print(f"       ({s.flag_bars} drift bars, ends {s.flag_end_date}, sweep {s.sweep_below_pct}% below rail, "
              f"risk {s.risk_pct}%, {s.reward_R}R, outcome {s.outcome or 'open'})")
        if not ok:
            fails += 1
        print()
    if fails:
        print(f"{fails} exemplar(s) FAILED — the detector no longer matches the validated pattern.")
        return 1
    print("both exemplars pass — the detector matches the charts it was built from.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
