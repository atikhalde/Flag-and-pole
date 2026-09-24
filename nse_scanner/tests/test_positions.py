"""
Regression test for the two cases that were reported from live alerts.

CASE 1 - GODREJIND.NS, reported as:
        entry 1255.6 (2026-08-13)  stop 1188.54  last 1099.0  ->  -2.34R   "in trade"
    Its stop was crossed on 2026-08-24 and the whole base then broke down.  The 08-13 trigger
    was a bounce that was undercut, so it must be voided on the way down and the structure must
    end up DEAD - never carried as a position, and never displayed below -1R.

CASE 2 - LAMBODHARA.NS, reported as:
        entry 120.21 (2026-09-18)  stop 113.06
    The shakeout was not over on 09-18: four bars later the stock printed a LOWER low (115.57
    under 116.60) and closed below the earlier flush.  2026-09-23 is the valid entry, and the
    stop must re-base to the new shakeout low.

Run after any change to pattern.py / positions.py / live_scanner.py:

    python3 -m nse_scanner.tests.test_positions
"""
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from nse_scanner import pattern as P, positions as POS

FAILS = []


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


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


def voided(g: pd.DataFrame, sym: str, entry_date: str):
    v = [s for s in P.find_setups(g, sym, last_only=False) if s.entry_date == entry_date]
    return v[-1] if v else None


def main() -> int:
    # ---- CASE 1: GODREJIND
    g = fetch("GODREJIND")
    s = voided(g, "GODREJIND", "2026-08-13")
    check(s is not None and s.outcome == "voided",
          f"GODREJIND 2026-08-13 trigger is voided (got {s.outcome if s else 'nothing'}, "
          f"R {s.R if s else float('nan'):+.2f}) - it was a bounce that got undercut")
    live = P.find_setups(g, "GODREJIND", last_only=True)
    st = live[0].status if live else "NOTHING"
    check(st == "DEAD", f"GODREJIND is DEAD now, not a position (got {st}) - "
                        f"the scanner used to show it as 'in trade' at -2.34R")

    # ---- CASE 2: LAMBODHARA
    g = fetch("LAMBODHARA")
    s = voided(g, "LAMBODHARA", "2026-09-18")
    check(s is not None and s.outcome == "voided" and abs((s.R or 0) + 0.51) < 0.06,
          f"LAMBODHARA 2026-09-18 trigger is voided at ~-0.51R (got {s.outcome if s else 'nothing'}, "
          f"R {s.R if s else float('nan'):+.2f}) - the shakeout made a lower low on 09-22")
    live = P.find_setups(g, "LAMBODHARA", last_only=True)
    if not live:
        check(False, "LAMBODHARA has a live setup")
    else:
        v = live[0]
        check(v.entry_date == "2026-09-23",
              f"LAMBODHARA live entry is 2026-09-23 (got {v.entry_date})")
        check(abs(v.entry - 119.93) < 0.03, f"LAMBODHARA entry price 119.93 (got {v.entry})")
        check(abs(v.stop - 112.21) < 0.06, f"LAMBODHARA stop re-based to 112.21 (got {v.stop})")
        check(abs(v.sweep_low - 115.57) < 0.03 and v.sweep_date == "2026-09-22",
              f"LAMBODHARA shakeout low is 115.57 on 2026-09-22 (got {v.sweep_low} on {v.sweep_date})")
        info = POS.classify(g, v)
        check(info["state"] == POS.OPEN and info["now_R"] >= -1.0,
              f"LAMBODHARA is OPEN and never shown below -1R (got {info['state']} {info['now_R']:+.2f}R)")

    # ---- CASE 3: the digest card for a fired entry must print the date it fired.
    # Reported: LAMBODHARA's 2026-09-23 entry was displayed as "entry fired" with no date and
    # read as if it had fired the day of the digest.
    from nse_scanner import telegram_bot as TG
    card = TG.status_card(dict(status="BUY", symbol="LAMBODHARA.NS", shortName="LAMBODHARA TEXTILES",
                               entry_date="2026-09-23", entry=119.93, stop=112.21, risk_pct=6.44,
                               last_close=139.46, target=147.19, bars_since_entry=1,
                               pole_date="2026-08-24", pole_low=105.5, pole_high=147.19, pole_gain=0.395,
                               rail=127.37, flag_bars=13, sweep_date="2026-09-22", sweep_low=115.57))
    check("2026-09-23" in card, "the entry card prints the date the entry fired (no date = 'fired today')")
    check("entry ₹119.93" in card, "the entry card prints the entry price")

    # ---- CASE 4: a watchlist name may not be listed twice (once as a signal, once as watchlist)
    w = [dict(symbol="LAMBODHARA.NS", entry_date="2026-09-23", entry=119.93, stop=112.21,
              last_close=139.46, gate_note="drift only 13 bars < 15")]
    d = TG.digest([], "post-close digest", fresh=0, watch=w)
    check(d.count("LAMBODHARA.NS") == 1, f"a watchlist name appears exactly once in the digest "
                                          f"(found {d.count('LAMBODHARA.NS')})")

    print()
    if FAILS:
        print(f"  {len(FAILS)} FAILED - do not ship this detector")
        return 1
    print("  all position lifecycle checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
