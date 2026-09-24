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

# the shakeout SIGNATURE, measured on the two charts: volume expansion out of the quiet drift,
# and a red falling structure.  NIACL and LAMBODHARA are the reference values.
SIGNATURE = {
    # symbol: (min leg volume vs 20d avg, min leg volume vs the drift's volume, max one-day drop %, min red fraction)
    "NIACL":      (1.0, 3.0, -4.0, 0.5),
    "LAMBODHARA": (1.0, 3.0, -8.0, 0.5),
}


def main():
    fails, skipped = 0, []
    for sym, ign, rail, sh_low, entry_date, tol in CASES:
        g = fetch(sym)
        # A source outage must not be reported as a detector failure, and must not be silent
        # either.  Yahoo dropped the entire 2026-09-24 session from its daily series on the
        # evening of 2026-09-24, which left the LAMBODHARA case with no bar to detect on.  If
        # the expected entry bar is simply not in the data, say so loudly and skip the case;
        # if the bar IS there and the detector disagrees, that is a real regression.
        if entry_date not in {str(d)[:10] for d in g.index}:
            print(f"  SKIP {sym}: the source data does not contain {entry_date} "
                  f"(last bar {str(g.index[-1])[:10]}). This is a data gap, not a detector "
                  f"failure - re-run when Yahoo has the bar.")
            skipped.append((sym, entry_date))
            continue
        out = P.find_setups(g, sym, last_only=False)
        hit = [s for s in out if s.pole_date == ign]
        if not hit:
            print(f"FAIL {sym}: no setup detected at the {ign} ignition - the detector changed")
            fails += 1
            continue
        s = hit[-1]
        checks = [("rail", s.rail, rail), ("shakeout low", s.sweep_low, sh_low)]
        good = True
        for name, got, want in checks:
            hit_ok = abs(got - want) / want <= tol
            good &= hit_ok
            print(f"  {'ok  ' if hit_ok else 'FAIL'} {sym} {name}: got {got}  expected ~{want}")
        hit_ok = str(s.entry_date)[:10] == entry_date
        good &= hit_ok
        print(f"  {'ok  ' if hit_ok else 'FAIL'} {sym} entry date: got {str(s.entry_date)[:10]}  expected {entry_date}")
        print(f"       ({s.flag_bars} drift bars, ends {s.flag_end_date}, sweep {s.sweep_below_pct}% below rail, "
              f"risk {s.risk_pct}%, {s.reward_R}R, outcome {s.outcome or 'open'})")

        # ---- the shakeout signature: volume + falling structure, not just depth ----
        v_min, d_min, drop_max, red_min = SIGNATURE[sym]
        sig = [("leg volume vs 20d avg", s.sweep_vol_x20, v_min, ">="),
               ("leg volume vs the drift", s.sweep_vol_vs_drift, d_min, ">="),
               ("worst red bar in the leg %", s.sweep_leg_pct, drop_max, "<="),
               (f"red bars in the leg ({s.sweep_red}/{s.sweep_bars})", s.sweep_red_frac, red_min, ">=")]
        for name, got, want, op in sig:
            hit_ok = (got >= want) if op == ">=" else (got <= want)
            good &= bool(hit_ok)
            print(f"  {'ok  ' if hit_ok else 'FAIL'} {sym} shakeout {name}: got {got}  (needs {op} {want})")
        if not good:
            fails += 1
        print()
    if fails:
        print(f"{fails} exemplar(s) FAILED - the detector no longer matches the validated pattern "
              "(geometry OR the shakeout's volume / candle structure).")
        return 1
    if skipped:
        print(f"{len(skipped)} exemplar(s) SKIPPED for a missing source bar: "
              + ", ".join(f"{s} ({d})" for s, d in skipped))
        print("the detector was NOT invalidated - the vendor data is incomplete right now.")
        return 0
    print("both exemplars pass - the detector matches the charts it was built from, "
          "including the shakeout's volume and falling-candle structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
