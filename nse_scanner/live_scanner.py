"""
live_scanner.py — entry point for the live (intraday) scanner.

  1. load universe (price > ₹100 AND mcap >= ₹1000 cr)
  2. batch-download daily bars for the whole universe
  3. stage-1 filter: find names with a pole/flag/sweep structure
  4. shortlist → pull INTRAday bars for just those names
  5. evaluate the trigger on the live price → PROVISIONAL alert
  6. after 15:35 IST, re-evaluate on the completed daily bar → CONFIRMED alert + daily digest
  7. de-duplicate everything through state/live_state.json (committed back by the workflow)

Usage:
  python3 -m nse_scanner.live_scanner                  # auto: provisional intraday or confirmed after 15:35
  python3 -m nse_scanner.live_scanner --force-provisional
  python3 -m nse_scanner.live_scanner --digest-only
  python3 -m nse_scanner.live_scanner --limit 200 --no-telegram   (testing)
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import pandas as pd

from . import config as C, data as D, pattern as P, telegram_bot as TG


def stage1(uni: pd.DataFrame, limit: int = 0) -> tuple[dict, dict]:
    """Daily download for the universe → {sym: frames}, {sym: setup} for interesting names."""
    syms = uni.symbol.tolist()
    print(f"[stage1] downloading daily bars for {len(syms)} symbols …")
    t0 = time.time()
    frames = D.download_daily(syms, label="stage1")
    print(f"[stage1] got {len(frames)} usable symbols in {time.time()-t0:.0f}s")

    interesting, all_states = {}, {}
    for sym, g in frames.items():
        try:
            out = P.find_setups(g, sym, last_only=True)
        except Exception as e:
            continue
        if not out:
            continue
        s = out[0]
        if s.status in ("FLAG_READY", "SWEPT", "BUY", "COILING"):
            all_states[sym] = s
        if s.status in ("FLAG_READY", "SWEPT", "BUY"):
            interesting[sym] = g
    print(f"[stage1] structure found in {len(interesting)} names "
          f"({sum(1 for s in all_states.values() if s.status=='BUY')} already BUY on daily)")
    return interesting, all_states


def stage2_live_prices(syms: list[str]) -> dict:
    """Today's live price / day low-high for the shortlist."""
    if not syms:
        return {}
    print(f"[stage2] intraday bars for {len(syms)} shortlisted symbols …")
    intra = D.download_intraday(syms, period="5d")
    out = {}
    today = D.now_ist().date()
    for sym, df in intra.items():
        df = df.copy()
        df["d"] = pd.to_datetime(df.index).date
        day = df[df.d == today]
        if day.empty:                       # market closed / holiday → use the last session available
            day = df[df.d == df.d.max()]
        if day.empty:
            continue
        out[sym] = dict(last_close=float(df["close"].iloc[-1]),
                        day_open=float(day["open"].iloc[0]),
                        day_high=float(day["high"].max()),
                        day_low=float(day["low"].min()),
                        day_vol=float(day["volume"].sum()),
                        bars=len(df))
    print(f"[stage2] live prices for {len(out)} symbols")
    return out


def evaluate_live(frames: dict, live: dict, confirmed: bool, regime: dict = None) -> list[dict]:
    regime = regime or {}
    """Re-run detection with today's live value grafted as the last bar, then decide the alert."""
    alerts = []
    for sym, g in frames.items():
        s0 = P.find_setups(g, sym, last_only=True)
        if not s0:
            continue
        s = s0[0]
        lv = live.get(sym)
        if lv is None:
            continue
        # ---- graft today's live print as a provisional daily bar
        today = pd.Timestamp(D.now_ist().date())
        gg = g.copy()
        if gg.index[-1].normalize() < today:
            gg.loc[today] = dict(open=lv["day_open"], high=lv["day_high"], low=lv["day_low"],
                                 close=lv["last_close"], volume=lv["day_vol"])
        else:
            gg.iloc[-1, gg.columns.get_loc("high")] = max(float(gg["high"].iloc[-1]), lv["day_high"])
            gg.iloc[-1, gg.columns.get_loc("low")] = min(float(gg["low"].iloc[-1]), lv["day_low"])
            gg.iloc[-1, gg.columns.get_loc("close")] = lv["last_close"]
        out = P.find_setups(gg, sym, last_only=True)
        if not out:
            continue
        st = out[0]

        d = dict(st.__dict__)
        d["shortName"] = lv.get("shortName")
        d["last_close"] = round(lv["last_close"], 2)

        if st.status == "BUY" and np.isfinite(st.rail):
            d["dist_to_rail_pct"] = round((st.rail / lv["last_close"] - 1) * 100, 2)
            # ---- quality gates (evidence-based, see backtest_report.pdf)
            reasons = []
            stale = np.isfinite(st.bars_since_entry) and st.bars_since_entry > 1
            if stale:
                reasons.append(f"entry is {int(st.bars_since_entry)} bars old (now an open position, not a new signal)")
            # ---- 1. the false-signal filter: a violent reclaim bar is exhaustion, not absorption
            if np.isfinite(st.vol_x20) and st.vol_x20 > C.EXCLUDE_VOL_SPIKE_X:
                reasons.append(f"entry-bar volume {st.vol_x20}x > {C.EXCLUDE_VOL_SPIKE_X}x "
                               f"(exhaustion - historically -0.05R)")
            # ---- 2. the quality profile (validated out-of-sample)
            if C.QUALITY_PROFILE == "strict":
                if np.isfinite(st.flag_bars) and st.flag_bars < C.QUALITY_MIN_DRIFT:
                    reasons.append(f"drift only {st.flag_bars} bars < {C.QUALITY_MIN_DRIFT}")
                if np.isfinite(st.sweep_below_pct) and abs(st.sweep_below_pct) < C.QUALITY_MIN_SWEEP:
                    reasons.append(f"shakeout only {st.sweep_below_pct}% < -{C.QUALITY_MIN_SWEEP}%")
            elif C.QUALITY_PROFILE == "balanced":
                if np.isfinite(st.flag_bars) and st.flag_bars < 15:
                    reasons.append(f"drift only {st.flag_bars} bars < 15")
            # ---- 3. optional regime gate (off by default - edge does not survive out-of-sample)
            if C.REGIME_FILTER == "weak_market" and regime.get("ok") and regime["bull"]:
                reasons.append("NIFTY is above its 200-DMA (weak-market filter)")
            if np.isfinite(st.reward_R) and st.reward_R < C.MIN_REWARD_R:
                reasons.append(f"reward {st.reward_R}R < MIN_REWARD_R {C.MIN_REWARD_R}")
            if reasons:
                # keep the rejected setup in the result set - otherwise "blocked" is always 0 and
                # there is no way to see WHY a name did not alert
                d["skip_reason"] = "; ".join(reasons)
            alerts.append(d)
        else:
            d["dist_to_rail_pct"] = (round((st.rail / lv["last_close"] - 1) * 100, 2)
                                     if np.isfinite(st.rail) else None)
            alerts.append(d)
    return alerts


def run(args) -> int:
    now = D.now_ist()
    confirmed = (D.is_confirmed_run(now) or args.confirmed) and not args.force_provisional
    mode = "CONFIRMED (post-close)" if confirmed else "PROVISIONAL (intraday)"
    print(f"=== NSE flag scanner | {now:%Y-%m-%d %H:%M:%S %Z} | {mode} | "
          f"market_open={D.is_market_open(now)} ===")
    print(f"    trigger={C.TRIGGER_MODE} | profile={C.QUALITY_PROFILE} | "
          f"exclude volume>{C.EXCLUDE_VOL_SPIKE_X}x | regime={C.REGIME_FILTER} | min_reward={C.MIN_REWARD_R}R")

    regime = D.market_regime()
    if C.REGIME_FILTER == "weak_market":
        if regime.get("ok"):
            print(f"[regime] NIFTY {regime['close']:.0f} vs 200-DMA {regime['ma200']:.0f} "
                  f"({regime['pct_vs_ma']:+.2f}%) → {'BULL (weak_market filter blocks alerts)' if regime['bull'] else 'WEAK (alerts allowed)'}")
        else:
            print("[regime] unavailable — weak_market filter will not block alerts")
    if not D.is_market_open(now) and not confirmed and not args.force_provisional:
        print("market is closed and this is not the post-close run — nothing to do "
              "(use --force-provisional to test).")
        return 0

    uni = D.load_universe(limit=args.limit)
    if len(uni) == 0:
        print("empty universe — check UNIVERSE_CSV / filters")
        return 1
    name_map = dict(zip(uni.symbol, uni.get("shortName", uni.symbol)))
    mcap_map = dict(zip(uni.symbol, uni.get("mcap_cr", [np.nan] * len(uni))))

    frames, states = stage1(uni, args.limit)
    shortlist = list(frames.keys())
    live = stage2_live_prices(shortlist)
    for k, v in live.items():
        v["shortName"] = name_map.get(k)

    results = evaluate_live(frames, live, confirmed, regime)
    state = D.load_state()
    fired = state.setdefault("alerts", {})

    buys = [r for r in results if r["status"] == "BUY" and not r.get("skip_reason")]
    in_trade = [r for r in results if r["status"] == "BUY" and r.get("skip_reason")
                and "bars old" in str(r.get("skip_reason"))]
    waiting = [r for r in results if r["status"] in ("SWEPT", "FLAG_READY", "COILING")
               and not r.get("skip_reason")]
    gated = [r for r in results if r.get("skip_reason") and r not in in_trade]
    for r in in_trade:
        r["status"] = "IN TRADE"

    print(f"[result] {len(buys)} fresh BUY signal(s) | {len(in_trade)} already in trade | "
          f"{len(gated)} blocked by quality gates | {len(waiting)} waiting")
    for g in gated[:10]:
        print(f"    - {g['symbol']}: {g['skip_reason']}")

    sent = 0
    for b in buys:
        key = f'{b["symbol"]}|{b["pole_date"]}|{"CONFIRMED" if confirmed else "PROVISIONAL"}'
        if key in fired:
            continue
        b["mcap_cr"] = mcap_map.get(b["symbol"])
        body = TG.alert_card(b, "CONFIRMED" if confirmed else "PROVISIONAL")
        if not args.no_telegram:
            ok = TG.send(body)
        else:
            print(body)
            ok = True
        if ok:
            fired[key] = now.isoformat()
            sent += 1

    # ---- daily digest (post-close run only, once per day)
    today = str(now.date())
    if confirmed and state.get("digest_last") != today and not args.no_digest:
        items = []
        for r in waiting + buys:
            r["mcap_cr"] = mcap_map.get(r["symbol"])
            items.append(r)
        items.sort(key=lambda x: {"SWEPT": 0, "BUY": 1, "IN TRADE": 2, "FLAG_READY": 3, "COILING": 4}.get(x["status"], 9))
        if not args.no_telegram:
            TG.send(TG.digest(items, "post-close digest"))
        else:
            print(TG.digest(items, "post-close digest"))
        state["digest_last"] = today

    D.save_state(state)
    print(f"[done] alerts sent {sent} | digest {'yes' if confirmed else 'no'} | state saved")
    return 0


def main():
    ap = argparse.ArgumentParser(description="NSE pole-flag-sweep-reclaim live scanner")
    ap.add_argument("--limit", type=int, default=0, help="cap the universe size (testing)")
    ap.add_argument("--confirmed", action="store_true", help="treat this run as the post-close run")
    ap.add_argument("--force-provisional", action="store_true", help="evaluate intraday even when closed")
    ap.add_argument("--digest-only", action="store_true")
    ap.add_argument("--no-telegram", action="store_true", help="print alerts instead of sending")
    ap.add_argument("--no-digest", action="store_true")
    args = ap.parse_args()
    if args.digest_only:
        args.no_telegram = False
    sys.exit(run(args))


if __name__ == "__main__":
    main()
