"""Position lifecycle - the answer to "is this alert still a live trade?"

The scanner used to label any setup whose entry was more than one bar old as "in trade"
and then never look at it again.  Nothing checked the stop, nothing checked the target,
so a name that died weeks ago kept being reported as an open position - and kept
compounding its R past the stop, which is impossible if the stop is honoured.

That is exactly how a downtrend ends up sitting on the board looking like a live
shakeout trade.  On the 2026-09-24 run the scanner was carrying 33 positions:
16 had already hit their stop, 9 had already reached their target, and only 8 were
genuinely open.  Nine were being shown at worse than -1R (GODREJIND at -2.34R when its
stop was crossed on 2026-08-24).

classify() walks the bars after the entry and returns one of:

    OPEN       the trade is still running - price is between the stop and the target
    STOPPED    the stop was touched, booked at -1.00R
    TARGET     the pole high was reached, booked at the reward that was on offer
    EXPIRED    MAX_HOLD_BARS went by with neither, booked at that close

Run it over the whole live book with:

    python3 -m nse_scanner.positions [--limit N]
"""
import sys
import numpy as np
import pandas as pd

from . import config as C

OPEN, STOPPED, TARGET, EXPIRED, INVALIDATED = "OPEN", "STOPPED", "TARGET", "EXPIRED", "INVALIDATED"


def _g(s, key, default=np.nan):
    """Read a field from either a Setup or a plain dict."""
    v = s.get(key, default) if isinstance(s, dict) else getattr(s, key, default)
    return v


def classify(g: pd.DataFrame, s) -> dict:
    """Walk the bars after the entry and decide whether this trade is still alive."""
    entry, stop, target, ei = _g(s, "entry"), _g(s, "stop"), _g(s, "target"), _g(s, "entry_idx")
    out = dict(state=OPEN, exit_date=None, exit_R=np.nan, now_R=np.nan, bars=0, mark=np.nan)
    if not (np.isfinite(entry) and np.isfinite(stop) and np.isfinite(target) and np.isfinite(ei)):
        return out
    risk = float(entry) - float(stop)
    if risk <= 0:
        return out
    i = int(ei)
    last_i = len(g) - 1
    out["bars"] = max(last_i - i, 0)
    out["mark"] = round(float(g["close"].iloc[i]), 2)

    for k in range(i + 1, last_i + 1):
        lo, hi = float(g["low"].iloc[k]), float(g["high"].iloc[k])
        if lo <= stop:                                    # the resting stop fills first
            out.update(state=STOPPED, exit_R=-1.0, exit_date=str(g.index[k].date()),
                       now_R=-1.0, bars=k - i, mark=float(stop))
            return out
        if hi >= target:
            out.update(state=TARGET, exit_R=round((float(target) - float(entry)) / risk, 2),
                       exit_date=str(g.index[k].date()), now_R=round((float(target) - float(entry)) / risk, 2),
                       bars=k - i, mark=float(target))
            return out
        if C.EXIT_ON_CLOSE_BELOW_SWEEP and np.isfinite(_g(s, "sweep_low")):
            # optional: the flush has failed - a close back under the shakeout low voids the setup
            if float(g["close"].iloc[k]) < float(_g(s, "sweep_low")):
                r = round((float(g["close"].iloc[k]) - float(entry)) / risk, 2)
                out.update(state=INVALIDATED, exit_R=r, exit_date=str(g.index[k].date()), now_R=r,
                           bars=k - i, mark=round(float(g["close"].iloc[k]), 2))
                return out
        if k - i >= C.MAX_HOLD_BARS:                      # the time stop
            r = round((float(g["close"].iloc[k]) - float(entry)) / risk, 2)
            out.update(state=EXPIRED, exit_R=r, exit_date=str(g.index[k].date()), now_R=r,
                       bars=k - i, mark=round(float(g["close"].iloc[k]), 2))
            return out

    # still running: mark it to the last close, and never beyond the stop
    last = float(g["close"].iloc[-1])
    out["now_R"] = round(max((last - float(entry)) / risk, -1.0), 2)
    out["mark"] = round(last, 2)
    return out


def describe(r: dict) -> str:
    """One line explaining why this name is no longer a position."""
    st, d, R = r.get("pos_state"), r.get("pos_exit_date") or "?", r.get("pos_exit_R")
    R = "-" if R is None or not np.isfinite(R) else f"{R:+.2f}R"
    if st == STOPPED:
        return f"stopped out {d} at {R} - this position is closed, not open"
    if st == TARGET:
        return f"target reached {d} at {R} - this position is closed, not open"
    if st == EXPIRED:
        return f"time stop after {r.get('pos_bars')} bars on {d} at {R} - this position is closed, not open"
    if st == INVALIDATED:
        return f"shakeout invalidated {d} at {R} (price closed back below the flush) - this position is closed"
    return "open position"


def summarize(rows: list[dict]) -> dict:
    def cnt(st):
        return sum(1 for r in rows if r.get("pos_state") == st)
    stopped = cnt(STOPPED)
    target = cnt(TARGET)
    expired = cnt(EXPIRED)
    return dict(total=len(rows), open=cnt(OPEN), stopped=stopped, target=target, expired=expired,
                invalidated=cnt(INVALIDATED),
                booked_R=round(sum(r["pos_exit_R"] for r in rows
                                   if r.get("pos_state") in (STOPPED, TARGET, EXPIRED, INVALIDATED)
                                   and r.get("pos_exit_R") is not None
                                   and np.isfinite(r["pos_exit_R"])), 2),
                open_R=round(sum(r["pos_now_R"] for r in rows
                                 if r.get("pos_state") == OPEN and r.get("pos_now_R") is not None
                                 and np.isfinite(r["pos_now_R"])), 2))


def main() -> int:
    from . import live_scanner as LS, pattern as P, data as D
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    frames, _ = LS.stage1(D.load_universe(), limit)
    rows = []
    for sym, g in frames.items():
        out = P.find_setups(g, sym, last_only=True)
        if not out:
            continue
        s = out[0]
        if s.status != "BUY" or not np.isfinite(s.bars_since_entry) or s.bars_since_entry <= 1:
            continue
        if not (np.isfinite(s.entry) and np.isfinite(s.stop) and np.isfinite(s.target)):
            continue
        info = classify(g, s)
        rows.append(dict(symbol=sym, entry_date=s.entry_date, entry=s.entry, stop=s.stop,
                         target=s.target, last=round(float(g["close"].iloc[-1]), 2),
                         pos_state=info["state"], pos_exit_date=info["exit_date"],
                         pos_exit_R=info["exit_R"], pos_now_R=info["now_R"], pos_bars=info["bars"]))
    rows.sort(key=lambda r: (r["pos_state"] != OPEN, r["pos_now_R"] if np.isfinite(r["pos_now_R"]) else 0))
    print(f"\n{'='*112}\nthe scanner is carrying {len(rows)} setup(s) as positions\n{'='*112}")
    print(f"  {'symbol':16s} {'entry':11s} {'entry ₹':>9s} {'stop ₹':>9s} {'last ₹':>9s} {'state':8s} "
          f"{'exit on':11s} {'booked':>7s} {'now':>7s}")
    for r in rows:
        booked = f"{r['pos_exit_R']:+.2f}" if np.isfinite(r["pos_exit_R"]) else "-"
        now = f"{r['pos_now_R']:+.2f}" if np.isfinite(r["pos_now_R"]) else "-"
        print(f"  {r['symbol']:16s} {r['entry_date']:11s} {r['entry']:9.2f} {r['stop']:9.2f} {r['last']:9.2f} "
              f"{r['pos_state']:8s} {str(r['pos_exit_date'] or '-'):11s} {booked:>7s} {now:>7s}")
    s = summarize(rows)
    print(f"\n  {s['open']} open  |  {s['stopped']} stopped  |  {s['target']} at target  |  "
          f"{s['expired']} expired  |  {s['invalidated']} invalidated")
    print(f"  booked so far {s['booked_R']:+.1f}R  ·  open positions marked {s['open_R']:+.1f}R")
    worst = [r for r in rows if r["pos_state"] == STOPPED and np.isfinite(r.get("pos_now_R"))
             and r["pos_now_R"] < -1.001]
    if worst:
        print(f"\n  {len(worst)} of them would have been displayed BELOW -1R by the old code "
              f"(a stop cannot allow that):")
        for r in worst:
            print(f"      {r['symbol']:16s} shown as {r['pos_now_R']:+.2f}R, actually "
                  f"{r['pos_exit_R']:+.2f}R on {r['pos_exit_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
