"""
Regression test for the bug that silenced 2026-09-25.

`digest_last` used to be the IST CALENDAR date of the run, so any run after IST-midnight -
00:07, 03:58, whenever - stamped "today" and the real post-close digest was then skipped for the
whole day.  It happened twice in one night: run #15 committed digest_last = 2026-09-25 at
2026-09-24T18:37Z (= 00:07 IST on the 25th) and the market did not open for another nine hours.

The digest is now keyed on the trading SESSION: a run before the 09:15 open belongs to the previous
session and cannot consume the new day's slot.

    python3 -m nse_scanner.tests.test_digest_session
"""
import sys, datetime as dt
sys.path.insert(0, ".")
from nse_scanner import data as D

CASES = [
    # run time (IST)                     session it belongs to   why
    (dt.datetime(2026, 9, 25, 3, 58),     dt.date(2026, 9, 24),   "before the open -> the PREVIOUS session"),
    (dt.datetime(2026, 9, 25, 0, 7),      dt.date(2026, 9, 24),   "the run that actually caused the blackout"),
    (dt.datetime(2026, 9, 25, 9, 14),     dt.date(2026, 9, 24),   "one minute before the open"),
    (dt.datetime(2026, 9, 25, 9, 16),     dt.date(2026, 9, 25),   "just after the open"),
    (dt.datetime(2026, 9, 25, 13, 15),    dt.date(2026, 9, 25),   "intraday"),
    (dt.datetime(2026, 9, 25, 15, 40),    dt.date(2026, 9, 25),   "the post-close run"),
    (dt.datetime(2026, 9, 26, 9, 30),    dt.date(2026, 9, 25),   "Saturday -> Friday's session"),
    (dt.datetime(2026, 9, 27, 12, 0),    dt.date(2026, 9, 25),   "Sunday -> Friday's session"),
    (dt.datetime(2026, 9, 28, 8, 0),      dt.date(2026, 9, 25),   "Monday pre-open -> Friday's session"),
    (dt.datetime(2026, 9, 28, 15, 45),    dt.date(2026, 9, 28),   "Monday's post-close run"),
]


def main() -> int:
    fails = 0
    for t, want, why in CASES:
        got = D.session_date(t)
        ok = got == want
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {t:%a %Y-%m-%d %H:%M} IST -> session {got}  (want {want})  {why}")
    # the invariant that matters: an early-morning run can never key the day that is about to open
    early = D.session_date(dt.datetime(2026, 9, 25, 3, 58))
    if early == dt.date(2026, 9, 25):
        print("  FAIL an 03:58 IST run keyed the day that had not opened - the blackout is back")
        fails += 1
    else:
        print("  ok   03:58 IST cannot consume the coming day's digest slot")
    print()
    if fails:
        print(f"{fails} digest-session check(s) FAILED - a day can be silenced again.")
        return 1
    print("digest sessions all correct - one digest per trading session, and a missed session is "
          "delivered by the next morning's run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
