"""
Regression test for the position lifecycle - the bug behind the GODREJIND alert.

The scanner reported:

    entry 1255.6 (2026-08-13)  stop 1188.54  last 1099.0 -> -2.34R  "in trade"

Two things are wrong with that line and both are asserted here:

  1. GODREJIND's stop (1188.54) was crossed on 2026-08-24.  The trade is CLOSED at -1.00R.
     Carrying it as an open position is what made a downtrend look like a live shakeout trade.
  2. A stop cannot manufacture a -2.34R mark.  No position may ever be displayed below -1R.

Run after any change to pattern.py, positions.py or live_scanner.py:

    python3 -m nse_scanner.tests.test_positions
"""
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from nse_scanner import pattern as P, positions as POS

# symbol, entry date, expected state, expected exit date, expected booked R
CASES = [
    ("GODREJIND", "2026-08-13", POS.STOPPED, "2026-08-24", -1.0),   # the reported bug
    ("LAMBODHARA", "2026-09-18", POS.OPEN, None, None),             # a real, still-open position
]


def fetch(sym: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}.NS?range=2y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(urllib.request.urlopen(req, timeout=40))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(r["timestamp"], unit="s").tz_localize("UTC")
                .tz_convert("Asia/Kolkata").normalize().tz_localize(None),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
    }).dropna().drop_duplicates("date").set_index("date")
    return df


def main() -> int:
    fails = 0
    for sym, entry_date, state, exit_date, exit_R in CASES:
        g = fetch(sym)
        setups = [s for s in P.find_setups(g, sym, last_only=False) if s.entry_date == entry_date]
        if not setups:
            print(f"  FAIL {sym}: the {entry_date} setup is no longer detected - either the "
                  f"pattern changed or the data moved")
            fails += 1
            continue
        s = setups[-1]
        info = POS.classify(g, s)
        ok = True
        if info["state"] != state:
            print(f"  FAIL {sym}: state {info['state']}, expected {state}")
            ok = False
        if exit_date and info["exit_date"] != exit_date:
            print(f"  FAIL {sym}: closed on {info['exit_date']}, expected {exit_date}")
            ok = False
        if exit_R is not None and abs(float(info["exit_R"]) - float(exit_R)) > 1e-9:
            print(f"  FAIL {sym}: booked {info['exit_R']}R, expected {exit_R}R")
            ok = False
        # the hard rule: no display, ever, below the loss the stop allows
        if np.isfinite(info["now_R"]) and info["now_R"] < -1.0001:
            print(f"  FAIL {sym}: shown at {info['now_R']}R - a stop cannot allow worse than -1R")
            ok = False
        if info["now_R"] is not None and np.isfinite(info["now_R"]) and info["state"] != POS.OPEN \
                and abs(float(info["now_R"]) - float(info["exit_R"])) > 1e-9:
            print(f"  FAIL {sym}: a closed trade must mark at its booked R, not {info['now_R']}")
            ok = False
        if ok:
            detail = (f"{info['state']} on {info['exit_date']} at {info['exit_R']:+.2f}R"
                      if info["state"] != POS.OPEN else f"OPEN, marked {info['now_R']:+.2f}R")
            print(f"  ok   {sym}: entry {s.entry_date} ₹{s.entry} stop ₹{s.stop} -> {detail}")
    print(f"\n  {'all position lifecycle checks passed' if not fails else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
