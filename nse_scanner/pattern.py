"""
pattern.py — the detection engine, shared by the live scanner and the back-tester.

Sequence:
  POLE   : locked/≥+9.8% bar on ≥3× volume (close near its high) → impulse to the pole high
  FLAG   : consolidation from the pole high until price has given back >=50% of the pole,
           pole low intact.  Lower rail = the flag's lowest low (level to reclaim).
  SWEEP  : a bar that trades below that rail (the shakeout)
  BUY    : first close back above the rail (default) — "rail_reclaim"
           or the first bullish candle after the sweep — "first_bullish"
  RISK   : stop = sweep low − 0.5×ATR     target = pole high     time stop = 60 bars
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Setup:
    symbol: str
    pole_date: str = ""
    pole_low: float = np.nan
    pole_high: float = np.nan
    pole_high_date: str = ""
    pole_gain: float = np.nan          # pole_high / pole_low - 1
    flag_bars: int = 0
    flag_end_date: str = ""
    rail: float = np.nan               # the level that must be reclaimed
    rail_slope_pct: float = np.nan     # informational: flag's lower-rail slope (%/bar)
    sweep_date: str = ""
    sweep_low: float = np.nan
    sweep_below_pct: float = np.nan    # how far the sweep undercut the rail
    entry_date: str = ""
    entry: float = np.nan
    stop: float = np.nan
    target: float = np.nan
    risk_pct: float = np.nan
    reward_R: float = np.nan
    atr: float = np.nan
    status: str = ""                   # "BUY" | "SWEPT" | "COILING" | "DONE" | "DEAD"
    bars_since_entry: int = np.nan
    vol_x20: float = np.nan            # volume on the trigger bar vs its 20-day average
    # ---- the shakeout leg's own character (what the user's charts actually show) ----
    sweep_bars: int = 0                # bars from the floor break to the shakeout low
    sweep_red: int = 0                 # how many of them closed below their open (falling structure)
    sweep_red_frac: float = np.nan     # sweep_red / sweep_bars
    sweep_vol_x20: float = np.nan      # PEAK volume on the shakeout leg vs the 20-day average
    sweep_vol_vs_drift: float = np.nan # that peak / the median volume of the quiet drift
    sweep_close_pos: float = np.nan    # where the shakeout bar closed in its range (0 = on the low)
    sweep_leg_pct: float = np.nan      # size of the shakeout's largest one-day drop (%)
    turnover_cr: float = np.nan
    # ---- what the trigger bar itself did (entry quality, no look-ahead) ----
    entry_gt_prior_high: bool = False   # close > the previous bar's high (it actually reclaimed ground)
    entry_gt_flush_high: bool = False   # close > the high of the bar that made the flush low
    entry_vs_rail_pct: float = np.nan   # entry close vs the drift floor
    rebased: int = 0                    # how many earlier triggers this entry replaced
    rebased_from: str = ""              # their dates
    R_voided: float = 0.0               # what those voided entries cost, booked at the void
    # outcome fields (backtest only)
    outcome: str = ""
    R: float = np.nan
    ret_horizon: float = np.nan        # return after RET_HORIZON bars
    mfe: float = np.nan
    mae: float = np.nan
    bars_held: int = np.nan
    # the structural exit: close back below the shakeout low = the flush failed = not a shakeout
    R_struct: float = np.nan
    outcome_struct: str = ""
    bars_held_struct: int = np.nan
    entry_idx: int = -1
    sweep_idx: int = -1
    flag_end_idx: int = -1
    pole_idx: int = -1


def _prep(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["ret"] = g["close"].pct_change()
    g["v20"] = g["volume"].rolling(21).mean().shift(1)
    tr = np.maximum(g["high"] - g["low"],
                    np.maximum((g["high"] - g["close"].shift()).abs(),
                               (g["low"] - g["close"].shift()).abs()))
    g["atr"] = tr.rolling(14).mean()
    g["turnover_cr"] = (g["close"] * g["volume"]).rolling(20).mean() / 1e7
    return g


def _ignition_mask(g: pd.DataFrame) -> pd.Series:
    near_hi = g["close"] >= C.POLE_CLOSE_NEAR_HI * g["high"]
    vol_ok = g["volume"] >= C.VOL_SURGE * g["v20"]
    if C.POLE_MODE == "locked":
        return (g["ret"] >= C.POLE_RET_MIN) & near_hi & vol_ok & (g["v20"] > 0)
    return (g["ret"] >= C.POLE_RET_ALT) & near_hi & vol_ok & (g["v20"] > 0)


def find_setups(raw: pd.DataFrame, symbol: str, last_only: bool = True,
                scan_from: int = 0) -> list[Setup]:
    """
    last_only=True  → return at most one Setup (the most recent pole) — used by the live scanner
    last_only=False → return every historical setup that produced an entry (back-test)
    """
    if raw is None or len(raw) < 80:
        return []
    g = _prep(raw)
    ign = _ignition_mask(g)
    idxs = [i for i in range(len(g)) if bool(ign.iloc[i])]
    if not idxs:
        return []

    setups: list[Setup] = []
    last_ok_i = -999
    order = idxs[::-1] if last_only else idxs
    for i in order:
        if i - last_ok_i < 30:
            continue
        if last_only and (len(g) - 1 - i) > C.MAX_SETUP_BARS:
            break              # most recent pole is too old — nothing live here
                               # (historical scans ignore the freshness limit)
        s = _build_setup(g, symbol, i, last_only)
        if s is None:
            continue
        last_ok_i = i
        setups.append(s)
        if not last_only:
            setups.extend(getattr(s, "voided_setups", []) or [])   # voids are real closed trades
        if last_only:
            break
    # for the back-test, only "completed" setups are tradeable
    if not last_only:
        setups = [s for s in setups if s.status in ("BUY", "DONE", "DEAD", "SWEPT", "REBASING")]
    return setups


def _voided_as_setup(g: pd.DataFrame, s: "Setup", a: int, b: int, stop_a: float, R_a: float) -> "Setup":
    """A trigger that was voided is a trade you would have taken and closed - report it as one."""
    v = Setup(symbol=s.symbol, pole_date=s.pole_date, pole_low=s.pole_low, pole_high=s.pole_high,
              pole_high_date=s.pole_high_date, pole_gain=s.pole_gain, flag_bars=s.flag_bars,
              flag_end_date=s.flag_end_date, rail=s.rail, rail_slope_pct=s.rail_slope_pct)
    v.sweep_idx, v.sweep_date, v.sweep_low = s.sweep_idx, s.sweep_date, s.sweep_low
    v.sweep_below_pct, v.sweep_bars, v.sweep_red = s.sweep_below_pct, s.sweep_bars, s.sweep_red
    v.sweep_red_frac, v.sweep_vol_x20, v.sweep_vol_vs_drift = s.sweep_red_frac, s.sweep_vol_x20, s.sweep_vol_vs_drift
    v.sweep_close_pos, v.sweep_leg_pct, v.turnover_cr = s.sweep_close_pos, s.sweep_leg_pct, s.turnover_cr
    v.entry_idx, v.entry_date = int(a), str(g.index[a].date())
    v.entry, v.stop, v.target = round(float(g["close"].iloc[a]), 2), round(float(stop_a), 2), s.pole_high
    v.risk_pct = round((v.entry - v.stop) / v.entry * 100, 2) if v.entry > v.stop else np.nan
    v.reward_R = round((v.target - v.entry) / (v.entry - v.stop), 2) if v.entry > v.stop else np.nan
    v.atr = round(float(g["atr"].iloc[a]), 2) if np.isfinite(g["atr"].iloc[a]) else np.nan
    v.vol_x20 = round(float(g["volume"].iloc[a]) / float(g["v20"].iloc[a]), 2) if float(g["v20"].iloc[a]) > 0 else np.nan
    v.status, v.outcome, v.R = "DONE", "voided", round(float(R_a), 2)
    v.bars_held = int(b - a)
    v.exit_date_void = str(g.index[b].date())
    h = min(a + C.MAX_HOLD_BARS, len(g) - 1)
    fwd = g.iloc[a + 1:h + 1]
    if len(fwd):
        v.mfe = round(float(fwd["high"].max()) / v.entry - 1, 4)
        v.mae = round(float(fwd["low"].min()) / v.entry - 1, 4)
    j = min(a + C.RET_HORIZON, len(g) - 1)
    v.ret_horizon = round(float(g["close"].iloc[j]) / v.entry - 1, 4)
    return v


def _build_setup(g: pd.DataFrame, symbol: str, i: int, last_only: bool) -> Setup | None:
    n = len(g)
    if i < C.BASE_LOOKBACK + 2 or i + 3 >= n:
        return None
    pole_low = float(g["low"].iloc[max(0, i - C.BASE_LOOKBACK):i].min())
    win = g.iloc[i + 1: min(i + 1 + C.IMPULSE_WINDOW, n)]
    if len(win) < 3:
        return None
    hv = win["high"].values.astype(float)
    pole_hi = float(np.nanmax(hv))
    hi_idx = i + 1 + int(np.nanargmax(hv))          # positional index into g
    if not np.isfinite(pole_hi) or pole_hi <= pole_low:
        return None
    pole_gain = pole_hi / pole_low - 1

    s = Setup(symbol=symbol,
              pole_date=str(g.index[i].date()), pole_low=round(pole_low, 2),
              pole_high=round(pole_hi, 2), pole_high_date=str(g.index[hi_idx].date()),
              pole_gain=round(pole_gain, 4), pole_idx=i, turnover_cr=round(float(g["turnover_cr"].iloc[-1]), 2))

    # ---------------------------- PHASE 2: THE DRIFT (the flag) ----------------------------
    # The drift runs from the pole high until the first CLOSE that breaks the drift's own floor
    # (the running minimum low).  That floor IS the rail the chart draws.
    run_min = float("inf")
    drift_end = None
    for k in range(hi_idx, min(hi_idx + 1 + C.MAX_DRIFT_BARS, n)):
        atr_k = float(g["atr"].iloc[k]) if np.isfinite(g["atr"].iloc[k]) else float(g["close"].iloc[k]) * 0.03
        margin = max(C.BREAK_ATR * atr_k, C.BREAK_MIN_PCT / 100.0 * float(g["close"].iloc[k]))
        # the drift ends only when a CLOSE breaks the floor AND the next bar does NOT reclaim it.
        # (a wick/poke that is immediately bought back is part of the drift, not the shakeout)
        if (k > hi_idx + C.MIN_FLAG_BARS and np.isfinite(run_min)
                and float(g["close"].iloc[k]) < run_min - margin
                and k + 1 < n and float(g["close"].iloc[k + 1]) < run_min):
            drift_end = k
            break
        run_min = min(run_min, float(g["low"].iloc[k]))
    if drift_end is None:
        if last_only:
            s.status = "COILING"
            s.rail = round(run_min, 2) if np.isfinite(run_min) else np.nan
            s.flag_bars = n - 1 - hi_idx
        return s if last_only else None

    rail = run_min                                        # the drift's floor = the level
    dr = g.iloc[hi_idx:drift_end]
    s.flag_bars, s.flag_end_idx, s.flag_end_date, s.rail = len(dr), drift_end, str(g.index[drift_end].date()), round(rail, 2)
    xs = np.arange(len(dr))
    s.rail_slope_pct = round(float(np.polyfit(xs, dr["low"].values, 1)[0]) / max(float(dr["close"].mean()), 1e-9) * 100, 3)
    if len(dr) < C.MIN_FLAG_BARS:
        return None

    # ---------------------------- PHASE 3: THE SHAKEOUT ----------------------------
    # the shakeout is the leg that trades BELOW the drift's floor
    sweep = None
    for k in range(drift_end, min(drift_end + 1 + C.SWEEP_MAX_BARS, n)):
        if float(g["low"].iloc[k]) < rail:
            sweep = k
            break
    if sweep is None:
        s.status = "FLAG_READY"                            # drift complete, floor not yet broken
        return s if last_only else None

    # walk forward to the shakeout's low (the running minimum until the entry)
    def _is_trigger(k):
        """the candle we are allowed to buy: a bullish close, in the top of its own range
        (or, in rail_reclaim mode, a close back above the drift floor and the prior 5-bar high)"""
        if C.TRIGGER_MODE == "first_bullish":
            c, o = float(g["close"].iloc[k]), float(g["open"].iloc[k])
            rng = float(g["high"].iloc[k]) - float(g["low"].iloc[k])
            pos = (c - float(g["low"].iloc[k])) / rng if rng > 0 else 0
            return c > o and pos >= (1 - C.BULLISH_CLOSE_POS)
        prior5 = float(g["high"].iloc[max(0, k - 5):k].max())
        return float(g["close"].iloc[k]) > max(rail, prior5)

    entry_idx = None
    sh_low, sh_low_idx = float(g["low"].iloc[sweep]), sweep
    for k in range(sweep + 1, min(sweep + 1 + C.SWEEP_MAX_BARS, n)):
        if float(g["low"].iloc[k]) < sh_low:
            sh_low, sh_low_idx = float(g["low"].iloc[k]), k
        if _is_trigger(k):
            entry_idx = k
            break

    # ---- RE-BASE: a trigger that is later undercut and closed below was a bounce inside the
    # shakeout, not the end of it.  Void it, move the shakeout low (and therefore the stop)
    # down to the new low, and take the next bullish candle.  Repeat.
    voided, voided_R = [], 0.0
    while C.REBASE_ON_NEW_LOW and entry_idx is not None:
        inval = None
        standing = sh_low          # the shakeout low THIS entry is anchored to - it must not move
        for k in range(entry_idx + 1, min(entry_idx + 1 + C.SWEEP_MAX_BARS, n)):
            if float(g["close"].iloc[k]) < standing:
                inval = k
                break
        if inval is None:
            break
        if inval is None:
            break
        nxt = None
        for k in range(inval + 1, min(inval + 1 + C.SWEEP_MAX_BARS, n)):
            if float(g["low"].iloc[k]) < sh_low:
                sh_low, sh_low_idx = float(g["low"].iloc[k]), k
            if _is_trigger(k):
                nxt = k
                break
        _seg = g["low"].iloc[entry_idx:inval + 1]
        if float(_seg.min()) < sh_low:
            sh_low, sh_low_idx = float(_seg.min()), int(entry_idx + int(np.nanargmin(_seg.values)))
        # what the voided entry would have cost you had you taken it (it did not fill a target:
        # it was closed at the invalidation close, so this is a real loss, not a free do-over)
        _low_at = float(g["low"].iloc[entry_idx:inval + 1].min())
        _atr = float(g["atr"].iloc[entry_idx]) if np.isfinite(g["atr"].iloc[entry_idx]) else 0.0
        _stop = min(_low_at, sh_low) - C.ATR_BUF * _atr
        _risk = float(g["close"].iloc[entry_idx]) - _stop
        _Rv = ((float(g["close"].iloc[inval]) - float(g["close"].iloc[entry_idx])) / _risk) if _risk > 0 else np.nan
        if np.isfinite(_Rv):
            voided_R += _Rv
        voided.append((int(entry_idx), int(inval), float(_stop), float(_Rv)))
        entry_idx = nxt
        if entry_idx is None:
            break
    if sh_low < pole_low * C.BASE_INTACT_TOL:              # the whole base is gone
        s.status = "DEAD"
        s.sweep_idx, s.sweep_date, s.sweep_low = sweep, str(g.index[sweep].date()), round(sh_low, 2)
        s.sweep_below_pct = round((sh_low / rail - 1) * 100, 2)
        s.rebased = int(len(voided))
        s.rebased_from = ",".join(str(g.index[a].date()) for a, *_ in voided)
        s.R_voided = round(voided_R, 2)
        if voided and not last_only:
            # the base is dead, but the entries that were voided on the way down are trades
            # that happened and were closed.  Dropping them is how this rule flatters itself.
            s.voided_setups = [_voided_as_setup(g, s, a, b, st, Rv) for a, b, st, Rv in voided
                               if np.isfinite(Rv)]
            return s
        return s if last_only else None
    s.sweep_idx, s.sweep_date, s.sweep_low = sweep, str(g.index[sh_low_idx].date()), round(sh_low, 2)
    s.sweep_below_pct = round((sh_low / rail - 1) * 100, 2)

    # ---- characterise the shakeout leg: how many bars, how red, and how much volume ----
    leg = g.iloc[sweep:sh_low_idx + 1]
    if len(leg):
        red = (leg["close"] < leg["open"]).sum()
        lg_vx = (leg["volume"] / leg["v20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        dr_vx = (dr["volume"] / dr["v20"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        s.sweep_bars = int(len(leg))
        s.sweep_red = int(red)
        s.sweep_red_frac = round(float(red) / len(leg), 2)
        s.sweep_vol_x20 = round(float(np.nanmax(lg_vx.values)), 2) if np.isfinite(np.nanmax(lg_vx.values)) else np.nan
        drift_med_vx = float(np.nanmedian(dr_vx.values)) if len(dr_vx) and np.isfinite(np.nanmedian(dr_vx.values)) else np.nan
        s.sweep_vol_vs_drift = (round(s.sweep_vol_x20 / drift_med_vx, 2)
                                if drift_med_vx and drift_med_vx > 0 and np.isfinite(s.sweep_vol_x20) else np.nan)
        bar = g.iloc[sh_low_idx]
        rng = float(bar["high"]) - float(bar["low"])
        s.sweep_close_pos = round((float(bar["close"]) - float(bar["low"])) / rng, 2) if rng > 0 else np.nan
        s.sweep_leg_pct = round(float(leg["ret"].min()) * 100, 2) if np.isfinite(leg["ret"].min()) else np.nan
    s.rebased = int(len(voided))
    s.rebased_from = ",".join(str(g.index[a].date()) for a, *_ in voided)
    s.R_voided = round(voided_R, 2)
    if entry_idx is None:
        s.status = "REBASING" if voided else "SWEPT"       # shakeout still running / waiting for a candle
        if voided and not last_only:
            # no new trigger has appeared yet, but the voided entries are still trades that
            # happened and were closed.  They must be counted - dropping them is how a
            # "wait for a cleaner entry" rule flatters itself.
            s.voided_setups = [_voided_as_setup(g, s, a, b, st, Rv) for a, b, st, Rv in voided
                               if np.isfinite(Rv)]
            return s
        return s if last_only else None

    entry = float(g["close"].iloc[entry_idx])
    atr = float(g["atr"].iloc[entry_idx]) if np.isfinite(g["atr"].iloc[entry_idx]) else entry * 0.03
    stop = min(sh_low, float(g["low"].iloc[sweep:entry_idx + 1].min())) - C.ATR_BUF * atr
    s.entry_idx, s.entry_date, s.entry, s.atr = entry_idx, str(g.index[entry_idx].date()), round(entry, 2), round(atr, 2)
    s.stop, s.target = round(stop, 2), round(pole_hi, 2)
    s.risk_pct = round((entry - stop) / entry * 100, 2)
    s.reward_R = round((pole_hi - entry) / (entry - stop), 2) if entry > stop else np.nan
    v20 = float(g["v20"].iloc[entry_idx])
    s.vol_x20 = round(float(g["volume"].iloc[entry_idx]) / v20, 2) if v20 > 0 else np.nan
    s.entry_gt_prior_high = bool(entry > float(g["high"].iloc[entry_idx - 1]))
    s.entry_gt_flush_high = bool(entry > float(g["high"].iloc[sh_low_idx])) if entry_idx > sh_low_idx else False
    s.entry_vs_rail_pct = round((entry / rail - 1) * 100, 2) if np.isfinite(rail) else np.nan

    # ---- outcome (walk forward) — meaningful for historical setups
    s.status = "BUY" if last_only else "DONE"
    end = min(entry_idx + C.MAX_HOLD_BARS, n - 1)

    def _walk(structural: bool):
        """Returns (outcome, R, exit bar). structural=True adds: a CLOSE below the
        shakeout low invalidates the setup - the flush failed, this is a downtrend."""
        for k in range(entry_idx + 1, end + 1):
            lo, hi = float(g["low"].iloc[k]), float(g["high"].iloc[k])
            if lo <= stop:                           # the resting stop fills first
                return "stopped", -1.0, k
            if structural and float(g["close"].iloc[k]) < sh_low:
                # the flush has failed and the stop was not touched: the setup is void
                return "invalidated", (float(g["close"].iloc[k]) - entry) / (entry - stop), k
            if hi >= pole_hi:
                return "target", (pole_hi - entry) / (entry - stop), k
        if n - 1 > entry_idx + C.MAX_HOLD_BARS:
            return "time", (float(g["close"].iloc[end]) - entry) / (entry - stop), end
        return "open", np.nan, end

    outcome, Rr, exit_k = _walk(False)
    _so, _sR, _sk = _walk(True)
    if s.status == "DONE":
        s.outcome, s.R, s.bars_held = outcome, round(float(Rr), 2) if np.isfinite(Rr) else np.nan, int(exit_k - entry_idx)
        s.outcome_struct = _so
        s.R_struct = round(float(_sR), 2) if np.isfinite(_sR) else np.nan
        s.bars_held_struct = int(_sk - entry_idx)
    if voided and not last_only:
        s.voided_setups = [_voided_as_setup(g, s, a, b, st, Rv) for a, b, st, Rv in voided
                           if np.isfinite(Rv)]
        h = min(entry_idx + C.MAX_HOLD_BARS, n - 1)
        fwd = g.iloc[entry_idx + 1:h + 1]
        if len(fwd):
            s.mfe = round(float(fwd["high"].max()) / entry - 1, 4)
            s.mae = round(float(fwd["low"].min()) / entry - 1, 4)
        j = min(entry_idx + C.RET_HORIZON, n - 1)
        s.ret_horizon = round(float(g["close"].iloc[j]) / entry - 1, 4)
    else:
        s.bars_since_entry = n - 1 - entry_idx
    return s


def setups_to_frame(setups: list[Setup]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in setups])
