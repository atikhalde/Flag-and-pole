"""
edge_study.py — accuracy, returns and a falsification test of the "edge".

Nothing here re-fits the pattern. It takes the trades the engine already found
(reports/backtest_trades.csv) and asks the uncomfortable questions:

  1. ACCURACY  — how often is a signal right, by gate, by year, by market regime
  2. RETURNS   — a capital-constrained equity curve at a realistic risk-per-trade,
                 with cost sensitivity and a benchmark comparison
  3. THE EDGE  — ablation (remove one gate at a time), a walk-forward by year,
                 a bootstrap confidence interval, and a permutation test that asks
                 "could this gate have looked this good by chance?"

Usage:
  python3 -m nse_scanner.edge_study --csv reports/backtest_trades.csv
  python3 -m nse_scanner.edge_study --csv reports/backtest_trades.csv --out reports --n-boot 5000 --n-perm 1000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

from . import config as C

MODE = "first_bullish"
SPLIT = "2022-01-01"


# ----------------------------------------------------------------------------- gates
def gate_masks(d: pd.DataFrame) -> dict[str, pd.Series]:
    """The named conditions. Each is a boolean mask over the trade frame."""
    idx = d.index
    return {
        "all signals":                pd.Series(True, index=idx),
        "quiet reclaim (vol<=2x)":    d.vol_x20 <= 2.0,
        "drift >= 15 bars":           d.flag_bars >= 15,
        "shakeout depth >= 8%":       d.sweep_below_pct <= -8.0,
        "shakeout vol >= 1x avg":     d.sweep_vol_x20 >= C.SWEEP_MIN_VOL_X20,
        "shakeout vol >= 3x drift":   d.sweep_vol_vs_drift >= C.SWEEP_MIN_VOL_VS_DRIFT,
        "red fall >= 4% in leg":      d.sweep_leg_pct <= -C.SWEEP_MIN_DROP_PCT,
        "leg more than half red":     d.sweep_red_frac >= C.SWEEP_MIN_RED_FRAC,
    }


VOLUME_GATE = ["quiet reclaim (vol<=2x)", "drift >= 15 bars", "shakeout vol >= 1x avg",
               "shakeout vol >= 3x drift", "red fall >= 4% in leg", "leg more than half red"]


def stats(x: pd.DataFrame) -> dict:
    x = x.dropna(subset=["R"])
    if len(x) < 5:
        return dict(n=int(len(x)))
    wins, losses = x.R[x.R > 0], x.R[x.R < 0]
    return dict(
        n=int(len(x)),
        win=round(float((x.R > 0).mean()), 3),
        avgR=round(float(x.R.mean()), 3),
        medR=round(float(x.R.median()), 3),
        med30=round(float(x.ret_horizon.median()), 4),
        avg30=round(float(x.ret_horizon.mean()), 4),
        target=round(float((x.outcome == "target").mean()), 3),
        stopped=round(float((x.outcome == "stopped").mean()), 3),
        payoff=round(float(abs(wins.mean() / losses.mean())), 2) if len(wins) and len(losses) else None,
        sdR=round(float(x.R.std()), 3),
        medRisk=round(float(x.risk_pct.median()), 2),
        totalR=round(float(x.R.sum()), 1),
    )


# ------------------------------------------------------------------- 1. accuracy
def accuracy(d: pd.DataFrame) -> dict:
    out = {}
    out["overall"] = stats(d)
    out["by_year"] = [dict(year=int(y), **stats(g)) for y, g in d.groupby(d.entry_date.str[:4])]
    _reg = d.dropna(subset=["nifty_bull"]).copy()
    _reg["nifty_bull"] = _reg.nifty_bull.astype(int)
    out["by_regime"] = [dict(regime=("NIFTY < 200DMA" if b == 0 else "NIFTY > 200DMA"), **stats(g))
                        for b, g in _reg.groupby("nifty_bull")]
    gm = gate_masks(d)
    out["by_gate"] = []
    for name, mask in gm.items():
        s = stats(d[mask])
        out["by_gate"].append(dict(gate=name, **s))
    # what a signal actually is: distribution of R
    q = d.R.dropna()
    out["R_distribution"] = dict(
        p05=round(float(q.quantile(0.05)), 2), p25=round(float(q.quantile(0.25)), 2),
        median=round(float(q.median()), 2), p75=round(float(q.quantile(0.75)), 2),
        p95=round(float(q.quantile(0.95)), 2),
        worst=round(float(q.min()), 2), best=round(float(q.max()), 2),
        pct_below_minus1=round(float((q < -1).mean()), 4))
    return out


# -------------------------------------------------------------------- 2. returns
def equity_curve(d: pd.DataFrame, risk_frac: float = 0.005, cost_R: float = 0.0,
                 start_capital: float = 1_000_000.0) -> dict:
    """Compound capital. Each trade risks `risk_frac` of equity; cost_R is subtracted per trade.

    Trades are walked in exit order (entry date + bars_held trading days) so overlapping
    positions are handled the way a real book would be.
    """
    x = d.dropna(subset=["R", "bars_held"]).copy()
    x["entry_dt"] = pd.to_datetime(x.entry_date)
    x["exit_dt"] = x.entry_dt + pd.to_timedelta((x.bars_held.astype(float) * 7 / 5).round(), unit="D")
    x = x.sort_values("exit_dt")
    eq = start_capital
    peak, max_dd = eq, 0.0
    curve, streaks, cur = [], 0, 0
    for r in x.itertuples():
        pnl_R = float(r.R) - cost_R
        eq *= (1 + risk_frac * pnl_R)
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
        cur = cur + 1 if pnl_R < 0 else 0
        streaks = max(streaks, cur)
        curve.append((r.exit_dt, eq))
    span = (x.exit_dt.max() - x.entry_dt.min()).days / 365.25 if len(x) else 1
    total = eq / start_capital - 1
    cagr = (eq / start_capital) ** (1 / span) - 1 if span > 0 else np.nan
    # per-trade Sharpe, annualised by signals per year
    per_year = len(x) / span if span > 0 else np.nan
    sd = x.R.std()
    sharpe = (x.R.mean() - cost_R) / sd * np.sqrt(per_year) if sd and per_year else np.nan
    return dict(trades=int(len(x)), risk_pct=risk_frac * 100, cost_R=cost_R,
                final=round(eq, 0), total_return=round(total, 3), cagr=round(cagr, 4),
                max_dd=round(max_dd, 3), worst_losing_streak=int(streaks),
                signals_per_year=round(per_year, 1), sharpe=round(float(sharpe), 2) if np.isfinite(sharpe) else None,
                years=round(span, 2))


def returns(d: pd.DataFrame, bench: dict = None) -> dict:
    out = {}
    x = d.dropna(subset=["R"])
    out["equal_weight_30d"] = dict(
        avg=round(float(x.ret_horizon.mean()), 4), median=round(float(x.ret_horizon.median()), 4),
        win=round(float((x.ret_horizon > 0).mean()), 4))
    out["equity"] = [equity_curve(d, 0.005, c) for c in (0.0, 0.1, 0.25, 0.5)]
    out["equity_at_1pct_risk"] = equity_curve(d, 0.01, 0.0)
    out["benchmark"] = bench or {}
    return out


def nifty_benchmark(start="2016-01-01") -> dict:
    try:
        import yfinance as yf
        n = yf.download("^NSEI", start=start, progress=False, auto_adjust=True)
        if not len(n):
            return {}
        c = n["Close"].squeeze()
        span = (c.index[-1] - c.index[0]).days / 365.25
        tot = float(c.iloc[-1] / c.iloc[0] - 1)
        dd = float(1 - (c / c.cummax()).min())
        return dict(name="NIFTY 50 buy & hold", from_=str(c.index[0].date()), to=str(c.index[-1].date()),
                    total_return=round(tot, 3), cagr=round((1 + tot) ** (1 / span) - 1, 4),
                    max_dd=round(dd, 3), years=round(span, 2))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------- 3. is it an edge?
def ablation(d: pd.DataFrame) -> list[dict]:
    """Remove one gate at a time from the shipped combination and see what breaks."""
    gm = gate_masks(d)
    full = pd.Series(True, index=d.index)
    for g in VOLUME_GATE:
        full &= gm[g]
    rows = [dict(configuration="shipped (all 6 conditions)", **split_stats(d[full]))]
    for g in VOLUME_GATE:
        m = pd.Series(True, index=d.index)
        for h in VOLUME_GATE:
            if h != g:
                m &= gm[h]
        rows.append(dict(configuration=f"without: {g}", **split_stats(d[m])))
    return rows


def split_stats(x: pd.DataFrame) -> dict:
    tr = x[x.entry_date < SPLIT]
    te = x[x.entry_date >= SPLIT]
    a, b = stats(x), stats(tr)
    c = stats(te)
    return dict(n=a.get("n"), win=a.get("win"), avgR=a.get("avgR"), med30=a.get("med30"),
                train_n=b.get("n"), train_avgR=b.get("avgR"),
                test_n=c.get("n"), test_avgR=c.get("test_avgR") if "test_avgR" in c else c.get("avgR"))


def subset_search(d: pd.DataFrame, min_n: int = 60) -> list[dict]:
    """Try every subset of the six conditions. Which combinations are simple AND stable?

    Reported with a multiple-testing caveat: 64 combinations are being scanned, so the
    single best row is not as impressive as it looks. What matters is whether a condition
    keeps appearing at the top.
    """
    import itertools
    gm = gate_masks(d)
    keys = VOLUME_GATE
    rows = []
    for k in range(1, len(keys) + 1):
        for combo in itertools.combinations(keys, k):
            m = pd.Series(True, index=d.index)
            for g in combo:
                m &= gm[g]
            x = d[m]
            if len(x) < min_n:
                continue
            st = stats(x)
            tr = stats(x[x.entry_date < SPLIT])
            te = stats(x[x.entry_date >= SPLIT])
            yrs = [st["avgR"] for _, g in x.groupby(x.entry_date.str[:4])
                   if (st := stats(g)).get("avgR") is not None and st.get("n", 0) >= 5]
            rows.append(dict(conditions=" + ".join(combo), k=k, n=st["n"], win=st["win"], avgR=st["avgR"],
                             med30=st["med30"],
                             train_avgR=tr.get("avgR"), train_n=tr.get("n"),
                             test_avgR=te.get("avgR"), test_n=te.get("n"),
                             worst_year=round(min(yrs), 2), years_negative=int(sum(1 for y in yrs if y < 0)),
                             years=len(yrs)))
    rows.sort(key=lambda r: (-(r["test_avgR"] or -9), r["k"]))
    return rows


def walk_forward(d: pd.DataFrame) -> list[dict]:
    """The shipped rule, frozen, applied year by year. No re-tuning anywhere."""
    gm = gate_masks(d)
    full = pd.Series(True, index=d.index)
    for g in VOLUME_GATE:
        full &= gm[g]
    rows = []
    for y, g in d.groupby(d.entry_date.str[:4]):
        rows.append(dict(year=int(y), all_signals=stats(g).get("avgR"),
                         shipped_n=stats(g[full.loc[g.index]]).get("n"),
                         shipped_avgR=stats(g[full.loc[g.index]]).get("avgR"),
                         shipped_win=stats(g[full.loc[g.index]]).get("win")))
    return rows


def bootstrap_ci(d: pd.DataFrame, n: int = 5000, seed: int = 7) -> dict:
    """Resample trades with replacement -> how uncertain is 'avgR'?"""
    gm = gate_masks(d)
    full = pd.Series(True, index=d.index)
    for g in VOLUME_GATE:
        full &= gm[g]
    rng = np.random.default_rng(seed)
    out = {}
    for label, mask in [("all signals", gm["all signals"]), ("shipped", full)]:
        r = d[mask].R.dropna().values
        if len(r) < 10:
            continue
        draws = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
        out[label] = dict(n=len(r), avgR=round(float(r.mean()), 3),
                          ci95=[round(float(np.percentile(draws, 2.5)), 3),
                                round(float(np.percentile(draws, 97.5)), 3)],
                          p_gt_zero=round(float((draws > 0).mean()), 4))
    return out


def permutation_test(d: pd.DataFrame, feature: str, threshold: float, direction: str = ">=",
                     n: int = 1000, seed: int = 11) -> dict:
    """Could this gate look this good by chance?

    Shuffle the feature across trades, apply the SAME threshold in the SAME direction, and
    count how often a random split produces a test-half edge at least as large as the real one.
    """
    def sel(vals):
        return (vals >= threshold) if direction == ">=" else (vals <= threshold)
    real_mask = pd.Series(sel(d[feature].values), index=d.index)
    real = stats(d[real_mask & (d.entry_date >= SPLIT)]).get("avgR", np.nan)
    rng = np.random.default_rng(seed)
    vals = d[feature].values.copy()
    beats = 0
    measured = 0
    for _ in range(n):
        rng.shuffle(vals)
        m = pd.Series(sel(vals), index=d.index)
        s = stats(d[m & (d.entry_date >= SPLIT)])
        if s.get("n", 0) < 20:
            continue
        measured += 1
        if s.get("avgR", -9) >= real:
            beats += 1
    return dict(feature=feature, threshold=threshold, direction=direction, real_test_avgR=real,
                permutations=measured, as_good_or_better=beats,
                p_value=round(beats / measured, 4) if measured else None)


# ------------------------------------------------------------------------- report
CANDIDATES = {
    "all signals": [],
    "6-condition (previous default)": VOLUME_GATE,
    "edge (3) - SHIPPED": ["drift >= 15 bars", "shakeout vol >= 3x drift", "red fall >= 4% in leg"],
    "edge (2)": ["shakeout vol >= 3x drift", "red fall >= 4% in leg"],
    "edge (1) - volume alone": ["shakeout vol >= 3x drift"],
}


def candidate_compare(d: pd.DataFrame) -> list[dict]:
    gm = gate_masks(d)
    rows = []
    for label, conds in CANDIDATES.items():
        m = pd.Series(True, index=d.index)
        for c in conds:
            m &= gm[c]
        x = d[m]
        st, tr, te = stats(x), stats(x[x.entry_date < SPLIT]), stats(x[x.entry_date >= SPLIT])
        eq10 = equity_curve(x, 0.005, 0.10)
        eq0 = equity_curve(x, 0.005, 0.0)
        yrs = [st2["avgR"] for _, g in x.groupby(x.entry_date.str[:4])
               if (st2 := stats(g)).get("avgR") is not None and st2.get("n", 0) >= 5]
        rows.append(dict(rule=label, n=st.get("n"), per_year=round((st.get("n") or 0) / 11.0, 1),
                         win=st.get("win"), avgR=st.get("avgR"), med30=st.get("med30"),
                         payoff=st.get("payoff"), train_avgR=tr.get("avgR"), train_n=tr.get("n"),
                         test_avgR=te.get("avgR"), test_n=te.get("n"),
                         losing_years=sum(1 for y in yrs if y < 0), years=len(yrs),
                         worst_year=round(min(yrs), 2) if yrs else None,
                         cagr_no_cost=eq0["cagr"], cagr_0p1R=eq10["cagr"],
                         maxdd=eq0["max_dd"], sharpe=eq0["sharpe"], total_R=st.get("totalR")))
    return rows


def risk_matched(d: pd.DataFrame, target_dd: float = 0.384, cost_R: float = 0.10) -> dict:
    """Scale the risk per trade until the drawdown matches the benchmark's, then compare CAGR.

    Comparing a 0.5%-risk book against a 100%-invested index is meaningless; this makes the
    two portfolios carry the same risk.
    """
    lo, hi, best = 0.001, 0.15, None
    for _ in range(40):
        mid = (lo + hi) / 2
        e = equity_curve(d, mid, cost_R)
        if e["max_dd"] < target_dd:
            lo = best = mid if best is not None else mid
            best = mid
            lo = mid
        else:
            hi = mid
    if best is None:
        return {}
    e = equity_curve(d, best, cost_R)
    return dict(target_dd=target_dd, risk_per_trade=round(best * 100, 2), cagr=round(e["cagr"], 4),
                max_dd=round(e["max_dd"], 3), sharpe=e["sharpe"], final=round(e["final"], 0),
                total_return=round(e["total_return"], 3), cost_R=cost_R)


def build(d: pd.DataFrame, args) -> dict:
    d = d[d.trigger_mode == MODE].dropna(subset=["R"]).copy()
    d["entry_date"] = d.entry_date.astype(str)
    d["nifty_bull"] = d.nifty_bull.map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    rep = dict(trigger=MODE, split=SPLIT, trades=int(len(d)))
    print(f"mode={MODE}  trades={len(d)}  train(<{SPLIT})={int((d.entry_date < SPLIT).sum())} "
          f"test(>={SPLIT})={int((d.entry_date >= SPLIT).sum())}\n")
    rep["accuracy"] = accuracy(d)
    rep["returns"] = returns(d, nifty_benchmark())
    rep["ablation"] = ablation(d)
    rep["subset_search"] = subset_search(d)
    rep["candidate_compare"] = candidate_compare(d)
    gm = gate_masks(d)
    _edge = pd.Series(True, index=d.index)
    for c in CANDIDATES["edge (3) - SHIPPED"]:
        _edge &= gm[c]
    rep["risk_matched"] = dict(
        edge3=risk_matched(d[_edge]),
        all_signals=risk_matched(d),
    )
    rep["walk_forward"] = walk_forward(d)
    rep["bootstrap"] = bootstrap_ci(d, args.n_boot)
    rep["permutation"] = [
        permutation_test(d, "sweep_vol_vs_drift", C.SWEEP_MIN_VOL_VS_DRIFT, ">=", args.n_perm),
        permutation_test(d, "sweep_vol_x20", C.SWEEP_MIN_VOL_X20, ">=", args.n_perm),
        permutation_test(d, "sweep_leg_pct", -C.SWEEP_MIN_DROP_PCT, "<=", args.n_perm),
        permutation_test(d, "sweep_red_frac", C.SWEEP_MIN_RED_FRAC, ">=", args.n_perm),
        permutation_test(d, "vol_x20", 2.0, "<=", args.n_perm),
        permutation_test(d, "flag_bars", 15.0, ">=", args.n_perm),
    ]

    # ---------------- print
    a = rep["accuracy"]
    print("=== 1. ACCURACY — every first_bullish signal ===")
    o = a["overall"]
    print(f"  trades {o['n']}  hit rate {o['win']*100:.1f}%  target {o['target']*100:.0f}%  "
          f"stopped {o['stopped']*100:.0f}%  median 30d {o['med30']*100:+.2f}%  avgR {o['avgR']:+.2f}  "
          f"payoff {o['payoff']}:1")
    r = a["R_distribution"]
    print(f"  R spread: p05 {r['p05']}  p25 {r['p25']}  median {r['median']}  p75 {r['p75']}  p95 {r['p95']}  "
          f"(worst {r['worst']}, best {r['best']}, {r['pct_below_minus1']*100:.1f}% worse than -1R)")
    print("\n  by gate (ranked by avgR):")
    for row in sorted(a["by_gate"], key=lambda x: -(x.get("avgR") or -9)):
        print(f"    {row['gate']:28s} n={row.get('n',0):4d}  hit {row.get('win',0)*100:4.1f}%  "
              f"avgR {row.get('avgR',0):+.2f}  med30 {row.get('med30',0)*100:+5.2f}%  payoff {row.get('payoff')}")
    print("\n  by year:")
    for row in a["by_year"]:
        print(f"    {row['year']}  n={row.get('n',0):4d}  hit {row.get('win',0)*100:4.1f}%  "
              f"avgR {row.get('avgR',0):+.2f}  med30 {row.get('med30',0)*100:+6.2f}%")
    print("\n  by regime:")
    for row in a["by_regime"]:
        print(f"    {row['regime']:16s} n={row.get('n',0):4d}  hit {row.get('win',0)*100:4.1f}%  "
              f"avgR {row.get('avgR',0):+.2f}  med30 {row.get('med30',0)*100:+6.2f}%")

    print("\n=== 2. RETURNS ===")
    rt = rep["returns"]
    eqs = {e["cost_R"]: e for e in rt["equity"]}
    print(f"  equal weight, {C.RET_HORIZON}-bar hold: avg {rt['equal_weight_30d']['avg']*100:+.2f}%  "
          f"median {rt['equal_weight_30d']['median']*100:+.2f}%  win {rt['equal_weight_30d']['win']*100:.1f}%")
    print("  capital-constrained equity (0.5% risk per trade, compounding):")
    for c in (0.0, 0.1, 0.25, 0.5):
        e = eqs[c]
        print(f"    costs {c}R/trade: final x{e['final']/1e6:.2f}  total {e['total_return']*100:+.1f}%  "
              f"CAGR {e['cagr']*100:+.1f}%  maxDD {e['max_dd']*100:.1f}%  "
              f"worst losing streak {e['worst_losing_streak']}  Sharpe {e['sharpe']}")
    e1 = rt["equity_at_1pct_risk"]
    print(f"    at 1% risk, no costs: final x{e1['final']/1e6:.2f}  CAGR {e1['cagr']*100:+.1f}%  "
          f"maxDD {e1['max_dd']*100:.1f}%")
    b = rt["benchmark"]
    if b and "error" not in b:
        print(f"  benchmark {b['name']}: {b['total_return']*100:+.1f}% total, CAGR {b['cagr']*100:+.1f}%, "
              f"maxDD {b['max_dd']*100:.1f}% ({b['from_']} → {b['to']})")

    print("\n=== 3. THE EDGE — ablation (remove one condition at a time) ===")
    for row in rep["ablation"]:
        print(f"  {row['configuration']:36s} n={row['n'] or 0:4d}  avgR {row['avgR'] or 0:+.2f}  "
              f"train {row['train_avgR'] or 0:+.2f} (n={row['train_n']})  "
              f"TEST {row['test_avgR'] or 0:+.2f} (n={row['test_n']})")
    print("\n  candidate rules compared (returns are 0.5% risk/trade compounding):")
    print(f"    {'rule':62s} {'n':>4} {'/yr':>4} {'hit':>5} {'avgR':>6} {'train':>6} {'TEST':>6} "
          f"{'loseYrs':>7} {'CAGR0':>6} {'CAGR.1R':>7} {'maxDD':>6} {'Sharpe':>6}")
    for row in rep["candidate_compare"]:
        print(f"    {row['rule']:62s} {row['n']:>4} {row['per_year']:>4} {row['win']*100:>4.1f}% "
              f"{row['avgR']:>+6.2f} {row['train_avgR']:>+6.2f} {row['test_avgR']:>+6.2f} "
              f"{row['losing_years']:>3}/{row['years']:<3} {row['cagr_no_cost']*100:>+5.1f}% "
              f"{row['cagr_0p1R']*100:>+6.1f}% {row['maxdd']*100:>5.1f}% {row['sharpe']:>6.2f}")

    rm = rep["risk_matched"]
    print("\n  RISK-MATCHED against the benchmark (same drawdown, then compare CAGR; 0.1R costs):")
    for k, v in rm.items():
        if v:
            print(f"    {k:14s} risk {v['risk_per_trade']:>5.2f}%/trade  CAGR {v['cagr']*100:+.1f}%  "
                  f"maxDD {v['max_dd']*100:.1f}%  Sharpe {v['sharpe']}")

    print("\n  subset search — every combination of the six conditions (train/test + year stability):")
    print(f"    {'conditions':64s} {'k':>2} {'n':>5} {'avgR':>6} {'train':>6} {'TEST':>6} {'neg yrs':>7} {'worst yr':>8}")
    for row in rep["subset_search"][:12]:
        print(f"    {row['conditions']:64s} {row['k']:>2} {row['n']:>5} {row['avgR']:>+6.2f} "
              f"{row['train_avgR']:>+6.2f} {row['test_avgR']:>+6.2f} {row['years_negative']:>4}/{row['years']:<2} "
              f"{row['worst_year']:>+8.2f}")

    print("\n  walk-forward (rule frozen, applied year by year):")
    for row in rep["walk_forward"]:
        s = f"{row['shipped_avgR']:+.2f}" if row["shipped_avgR"] is not None else "  -  "
        print(f"    {row['year']}  all signals {row['all_signals'] or 0:+.2f}   shipped {s} "
              f"(n={row['shipped_n'] or 0}, hit {(row['shipped_win'] or 0)*100:.0f}%)")
    print("\n  bootstrap 95% CI on avgR:")
    for k, v in rep["bootstrap"].items():
        print(f"    {k:12s} n={v['n']:4d}  avgR {v['avgR']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"P(edge>0) {v['p_gt_zero']*100:.1f}%")
    print("\n  permutation test (could the gate look this good by chance?):")
    for p in rep["permutation"]:
        print(f"    {p['feature']:22s} {p['direction']} {p['threshold']:<5} real test-half avgR {p['real_test_avgR']:+.3f}  "
              f"random shuffles as good or better: {p['as_good_or_better']}/{p['permutations']}  p={p['p_value']}")
    return rep


def main():
    ap = argparse.ArgumentParser(description="accuracy / returns / edge study")
    ap.add_argument("--csv", default="reports/backtest_trades.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--mode", default=MODE)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    globals()["MODE"] = a.mode
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.csv)
    rep = build(d, a)
    json.dump(rep, open(f"{a.out}/edge_study.json", "w"), indent=1, default=str)
    print(f"\nwrote {a.out}/edge_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
