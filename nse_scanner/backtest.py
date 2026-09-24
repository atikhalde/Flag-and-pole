"""
backtest.py — walk the whole filtered NSE universe through history and measure the edge.

  * universe = exchange NSI, price > ₹100, mcap ≥ ₹1000 cr  (same filters as the live scanner)
  * history  = BACKTEST_START → today, daily bars
  * for every detected pole → flag → sweep → reclaim sequence it simulates the trade
  * headline metric: return RET_HORIZON (30) bars after entry, plus R-multiple and outcome
  * writes  backtest_trades.csv, backtest_summary.json  and (optionally) a PDF report

Usage:
  python3 -m nse_scanner.backtest --limit 300                 # quick test
  python3 -m nse_scanner.backtest --pdf                       # full run + PDF report
  python3 -m nse_scanner.backtest --trigger both --pdf
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

from . import config as C, data as D, pattern as P


def run_backtest(uni: pd.DataFrame, trigger_modes=("rail_reclaim",), verbose=True):
    """Returns (trades_df, baseline_dict, open_trades_df).

    trades_df  = completed trades only (target / stopped / time) — every statistic uses this.
    open_trades_df = positions still running at the last bar (outcome == "open"), marked to market.
    Open trades used to be dropped silently, which is why LAMBODHARA never appeared in the list.
    """
    syms = uni.symbol.tolist()
    if verbose:
        print(f"[backtest] downloading {len(syms)} symbols from {C.BACKTEST_START} …")
    t0 = time.time()
    frames = D.download_daily(syms, start=C.BACKTEST_START, label="backtest")
    if verbose:
        print(f"[backtest] {len(frames)} symbols in {time.time()-t0:.0f}s")

    # ---- benchmark: every bar's forward RET_HORIZON return in this universe/period
    fwd_all = []
    for sym, g in frames.items():
        f = (g["close"].shift(-C.RET_HORIZON) / g["close"] - 1).dropna()
        if len(f):
            fwd_all.append(f.values)
    if fwd_all:
        b = np.concatenate(fwd_all)
        baseline = dict(median_ret=round(float(np.median(b)), 4), avg_ret=round(float(b.mean()), 4),
                        win_rate=round(float((b > 0).mean()), 4), samples=int(len(b)))
    else:
        baseline = {}

    rows, open_rows = [], []
    for mode in trigger_modes:
        C.TRIGGER_MODE = mode
        n_trades = n_open = 0
        for sym, g in frames.items():
            try:
                setups = P.find_setups(g, sym, last_only=False)
            except Exception:
                continue
            last_close = float(g["close"].iloc[-1]) if len(g) else np.nan
            for s in setups:
                if s.status != "DONE":
                    continue
                if not np.isfinite(s.R):
                    # ---- still open at the report date → record it, do not drop it
                    if s.entry_date and np.isfinite(s.entry):
                        d = s.__dict__.copy()
                        d["trigger_mode"] = mode
                        d["last_close"] = last_close
                        risk = s.entry - s.stop
                        d["R"] = (last_close - s.entry) / risk if risk and np.isfinite(risk) and risk > 0 else np.nan
                        d["pnl_pct"] = last_close / s.entry - 1 if s.entry else np.nan
                        d["bars_held"] = (len(g) - 1 - s.entry_idx) if s.entry_idx >= 0 else np.nan
                        d["outcome"] = "open"
                        open_rows.append(d)
                        n_open += 1
                    continue
                d = s.__dict__.copy()
                d["trigger_mode"] = mode
                rows.append(d)
                n_trades += 1
        if verbose:
            print(f"[backtest] trigger={mode}: {n_trades} completed trades | {n_open} still open")
    C.TRIGGER_MODE = trigger_modes[0]

    df = pd.DataFrame(rows)
    if len(df):
        cols = [c for c in ["symbol", "shortName", "mcap_cr", "watchlist"] if c in uni.columns]
        df = df.merge(uni[cols], on="symbol", how="left")
        if "watchlist" not in df.columns:
            df["watchlist"] = False
        df["watchlist"] = df["watchlist"].fillna(False)
        df["year"] = pd.to_datetime(df["entry_date"]).dt.year
        df = _add_regime(df)
        df = df.sort_values("entry_date").reset_index(drop=True)

    odf = pd.DataFrame(open_rows)
    if len(odf):
        cols = [c for c in ["symbol", "shortName", "mcap_cr", "watchlist"] if c in uni.columns]
        odf = odf.merge(uni[cols], on="symbol", how="left")
        if "watchlist" not in odf.columns:
            odf["watchlist"] = False
        odf["watchlist"] = odf["watchlist"].fillna(False)
        odf["year"] = pd.to_datetime(odf["entry_date"]).dt.year
        odf = _add_regime(odf)
        odf = odf.sort_values("entry_date").reset_index(drop=True)
    return df, baseline, odf


def _add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every trade with the market regime on its entry date (NIFTY 50 vs its 200-DMA)."""
    try:
        import yfinance as yf
        idx = yf.download("^NSEI", start=C.BACKTEST_START, interval="1d", progress=False, auto_adjust=False)
        if idx is None or idx.empty:
            return df
        idx.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in idx.columns]
        idx["ma200"] = idx["close"].rolling(200).mean()
        idx["nifty_bull"] = idx["close"] > idx["ma200"]
        reg = idx[["nifty_bull"]].reset_index()
        reg.columns = ["d", "nifty_bull"]
        df = df.merge(reg, left_on=pd.to_datetime(df.entry_date), right_on="d", how="left").drop(columns=["d"])
    except Exception as e:
        print(f"[backtest] regime tagging skipped: {type(e).__name__} {e}")
        df["nifty_bull"] = np.nan
    return df


def evidence_tables(df: pd.DataFrame) -> dict:
    """The tables that decide whether the edge is real: by year, by regime, by sweep depth."""
    out = {}
    d = df.dropna(subset=["R"])
    if not len(d):
        return out
    out["by_year"] = (d.groupby("year").agg(trades=("R", "size"), win=("R", lambda s: (s > 0).mean()),
                                            avgR=("R", "mean"), medR=("R", "median"),
                                            med_ret=("ret_horizon", "median"), target=("outcome", lambda s: (s == "target").mean()))
                      .round(3).reset_index().to_dict("records"))
    if d.nifty_bull.notna().any():
        out["by_regime"] = []
        for flag, lab in [(True, "NIFTY above 200-DMA"), (False, "NIFTY below 200-DMA")]:
            b = d[d.nifty_bull == flag]
            if len(b):
                out["by_regime"].append(dict(regime=lab, trades=int(len(b)), win=round(float((b.R > 0).mean()), 3),
                                             avgR=round(float(b.R.mean()), 3), medR=round(float(b.R.median()), 2),
                                             med_ret=round(float(b.ret_horizon.median()), 4),
                                             avg_ret=round(float(b.ret_horizon.mean()), 4),
                                             target=round(float((b.outcome == "target").mean()), 3)))
    d = d.copy()
    d["sweep_bucket"] = pd.cut(d.sweep_below_pct, [-99, -3, -1.5, -0.5, 0.01],
                               labels=["deeper than −3%", "−3% … −1.5%", "−1.5% … −0.5%", "shallower than −0.5%"])
    out["by_sweep"] = (d.groupby("sweep_bucket", observed=True)
                       .agg(trades=("R", "size"), win=("R", lambda s: (s > 0).mean()), avgR=("R", "mean"),
                            med_ret=("ret_horizon", "median"), target=("outcome", lambda s: (s == "target").mean()))
                       .round(3).reset_index().astype({"sweep_bucket": str}).to_dict("records"))
    def _cfg(x):
        return dict(trades=int(len(x)), win=round(float((x.R > 0).mean()), 3), avgR=round(float(x.R.mean()), 3),
                    medR=round(float(x.R.median()), 2), med_ret=round(float(x.ret_horizon.median()), 4),
                    avg_ret=round(float(x.ret_horizon.mean()), 4),
                    target=round(float((x.outcome == "target").mean()), 3),
                    stopped=round(float((x.outcome == "stopped").mean()), 3))
    out["best_config"] = []
    out["best_config"].append(dict(label="A . every signal (no gate)", **_cfg(d)))
    _strict = d[(d.vol_x20 <= 2.0) & (d.flag_bars >= C.QUALITY_MIN_DRIFT)
                & (d.sweep_below_pct <= -C.QUALITY_MIN_SWEEP)]
    if len(_strict):
        out["best_config"].append(dict(
            label=f"B . shipped profile: quiet reclaim + drift >= {C.QUALITY_MIN_DRIFT} bars + shakeout >= "
                  f"{C.QUALITY_MIN_SWEEP:.0f}%", **_cfg(_strict)))
        out["best_config"].append(dict(label="C . shipped profile + weak market",
                                       **_cfg(_strict[_strict.nifty_bull == False])))
    out["best_config"].append(dict(label="D . weak market only", **_cfg(d[d.nifty_bull == False])))
    # ---- the false-signal study: does the entry bar's volume separate winners from traps?
    d2 = d.copy()
    d2["vol_bucket"] = pd.cut(d2.vol_x20, [0, .5, .8, 1.2, 2, 3, 999],
                              labels=["<0.5x", "0.5-0.8x", "0.8-1.2x", "1.2-2x", "2-3x", ">3x"])
    out["by_volume"] = (d2.groupby("vol_bucket", observed=True)
                        .agg(trades=("R", "size"), win=("R", lambda s: (s > 0).mean()),
                             avgR=("R", "mean"), med_ret=("ret_horizon", "median"),
                             stopped=("outcome", lambda s: (s == "stopped").mean()))
                        .round(3).reset_index().astype({"vol_bucket": str}).to_dict("records"))
    out["false_signal"] = dict(
        spike=dict(**_cfg(d[d.vol_x20 > C.EXCLUDE_VOL_SPIKE_X])),
        ok=dict(**_cfg(d[d.vol_x20 <= C.EXCLUDE_VOL_SPIKE_X])) if len(d[d.vol_x20 <= C.EXCLUDE_VOL_SPIKE_X]) else {},
        threshold=C.EXCLUDE_VOL_SPIKE_X)

    # ---- the shipped quality profile, time-split so nobody is fooled by in-sample luck
    def _profile(x):
        tr = x[x.entry_date.astype(str) < "2022-01-01"]
        te = x[x.entry_date.astype(str) >= "2022-01-01"]
        return dict(**_cfg(x),
                    train=dict(**_cfg(tr)) if len(tr) else {},
                    test=dict(**_cfg(te)) if len(te) else {})
    _strict = d[(d.vol_x20 <= 2.0) & (d.flag_bars >= C.QUALITY_MIN_DRIFT)
                & (d.sweep_below_pct <= -C.QUALITY_MIN_SWEEP)]
    _balanced = d[(d.vol_x20 <= 2.0) & (d.flag_bars >= 15)]
    out["profiles"] = [dict(label="off - every signal", **_profile(d)),
                       dict(label="weak market only (NIFTY < 200-DMA)",
                            **_profile(d[d.nifty_bull == False])),
                       dict(label="balanced - quiet reclaim, drift >= 15 bars", **_profile(_balanced)),
                       dict(label=f"strict - quiet, drift >= {C.QUALITY_MIN_DRIFT}, "
                                  f"shakeout >= {C.QUALITY_MIN_SWEEP:.0f}%", **_profile(_strict))]

    out["by_reward"] = (d.assign(rb=pd.cut(d.reward_R, [0, 1, 1.5, 2, 3, 99],
                                           labels=["<1R", "1–1.5R", "1.5–2R", "2–3R", ">3R"]))
                        .groupby("rb", observed=True)
                        .agg(trades=("R", "size"), win=("R", lambda s: (s > 0).mean()), avgR=("R", "mean"),
                             med_ret=("ret_horizon", "median")).round(3).reset_index().astype({"rb": str}).to_dict("records"))
    return out


def summarise(df: pd.DataFrame, ret_h: int = None) -> dict:
    ret_h = ret_h or C.RET_HORIZON
    col = "ret_horizon"
    if df is None or not len(df):
        return {}
    x = df.dropna(subset=[col])
    wins, losses = df[df.R > 0], df[df.R <= 0]
    eq = df.sort_values("entry_date").R.fillna(0).cumsum().values
    dd = float(np.min(eq - np.maximum.accumulate(eq))) if len(eq) else 0.0
    s = dict(
        trades=int(len(df)),
        trades_with_return=int(len(x)),
        win_rate_ret=round(float((x[col] > 0).mean()), 4) if len(x) else None,
        avg_ret=round(float(x[col].mean()), 4) if len(x) else None,
        median_ret=round(float(x[col].median()), 4) if len(x) else None,
        p25_ret=round(float(x[col].quantile(.25)), 4) if len(x) else None,
        p75_ret=round(float(x[col].quantile(.75)), 4) if len(x) else None,
        target_pct=round(float((df.outcome == "target").mean()), 4),
        stopped_pct=round(float((df.outcome == "stopped").mean()), 4),
        time_pct=round(float((df.outcome == "time").mean()), 4),
        avg_R=round(float(df.R.mean()), 3),
        median_R=round(float(df.R.median()), 3),
        total_R=round(float(df.R.sum()), 1),
        avg_win_R=round(float(wins.R.mean()), 3) if len(wins) else None,
        avg_loss_R=round(float(losses.R.mean()), 3) if len(losses) else None,
        payoff=round(float(abs(wins.R.mean() / losses.R.mean())), 2) if len(losses) and losses.R.mean() != 0 else None,
        median_risk_pct=round(float(df.risk_pct.median()), 2),
        median_MFE=round(float(df.mfe.median()), 4),
        median_MAE=round(float(df.mae.median()), 4),
        max_drawdown_R=round(dd, 1),
        avg_hold_bars=round(float(df.bars_held.mean()), 1),
        median_pole_gain=round(float(df.pole_gain.median()), 4),
        median_flag_bars=round(float(df.flag_bars.median()), 1),
        horizon_bars=ret_h,
    )
    return s


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year").agg(trades=("symbol", "size"), win=("R", lambda s: (s > 0).mean()),
                               avgR=("R", "mean"), medR=("R", "median"),
                               avg_ret=("ret_horizon", "mean"), med_ret=("ret_horizon", "median"))
    return g.round(3)


def _run_summary(df, summary, open_df, main_mode, out_dir):
    """Write the job summary shown on the Actions run page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    sm = summary.get(main_mode, {})
    base = summary.get("_baseline", {}) or {}
    ev = summary.get("_evidence", {}) or {}
    L = ["## NSE back-test — run result", ""]
    L.append(f"Entry rule **{main_mode}** · {summary.get('_meta',{}).get('universe','?')} symbols · "
             f"history from {summary.get('_meta',{}).get('start','?')} · filters {summary.get('_meta',{}).get('filters','')}")
    L.append("")
    L.append("| metric | value | market baseline |")
    L.append("|---|---|---|")
    L.append(f"| completed trades | **{sm.get('trades','?')}** | — |")
    L.append(f"| return after {C.RET_HORIZON} bars — median | {sm.get('median_ret',0)*100:+.2f}% | "
             f"{base.get('median_ret',0)*100:+.2f}% |")
    L.append(f"| return after {C.RET_HORIZON} bars — average | {sm.get('avg_ret',0)*100:+.2f}% | "
             f"{base.get('avg_ret',0)*100:+.2f}% |")
    L.append(f"| win rate | {sm.get('win_rate_ret',0)*100:.1f}% | {base.get('win_rate',0)*100:.1f}% |")
    L.append(f"| expectancy | **{sm.get('avg_R',0):+.2f}R** (payoff {sm.get('payoff','?')}:1) | — |")
    L.append(f"| target / stopped / time | {sm.get('target_pct',0)*100:.0f}% / {sm.get('stopped_pct',0)*100:.0f}% / "
             f"{sm.get('time_pct',0)*100:.0f}% | — |")
    L.append(f"| total R / worst drawdown | {sm.get('total_R',0):+.1f}R / {sm.get('max_drawdown_R',0):.1f}R | — |")
    L.append("")
    if ev.get("profiles"):
        L.append("### Filter profiles, measured on a train/test split")
        L.append("")
        L.append("| profile | trades | avg R | median 30-bar | train R (<2022) | test R (2022+) |")
        L.append("|---|---|---|---|---|---|")
        for pr in ev["profiles"]:
            tr = pr.get("train") or {}; te = pr.get("test") or {}
            L.append(f"| {pr['label']} | {pr['trades']} | {pr['avgR']:+.2f} | {pr.get('med_ret',0)*100:+.2f}% | "
                     f"{tr.get('avgR', float('nan')):+.2f} | {te.get('avgR', float('nan')):+.2f} |")
        L.append("")
    if ev.get("false_signal"):
        fs = ev["false_signal"]
        L.append(f"**The false signal:** reclaim bars on more than {fs.get('threshold')}x average volume average "
                 f"{fs.get('spike',{}).get('avgR',0):+.2f}R, everything else {fs.get('ok',{}).get('avgR',0):+.2f}R.")
        L.append("")
    if open_df is not None and len(open_df):
        od = open_df[open_df.trigger_mode == main_mode]
        L.append(f"### Positions still open at the report date ({len(od)})")
        L.append("")
        L.append("| symbol | name | entry date | entry | last | R now |")
        L.append("|---|---|---|---|---|---|")
        for _, r in od.sort_values("R", ascending=False).head(15).iterrows():
            L.append(f"| `{r.symbol}` | {r.get('shortName','')} | {str(r.entry_date)[:10]} | {r.entry:.2f} | "
                     f"{r.get('last_close', float('nan')):.2f} | "
                     f"{(f'{r.R:+.2f}R' if r.R == r.R else '-')} |")
        L.append("")
    L.append("---")
    L.append(f"The full report — headline stats, the filter study, every position and the trade log with full company "
             f"names — is the **`backtest_report.pdf`** artifact of this run, and it is committed to "
             f"`{out_dir}/backtest_report.pdf` in the repo.")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("[summary] wrote the run summary to GITHUB_STEP_SUMMARY")
    except Exception as e:
        print(f"[summary] could not write the run summary: {type(e).__name__} {e}")


def main():
    ap = argparse.ArgumentParser(description="NSE pole-flag-sweep-reclaim back-test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--trigger", type=str, default="both", choices=["rail_reclaim", "first_bullish", "both"])
    ap.add_argument("--pdf", action="store_true", help="write a PDF report")
    ap.add_argument("--out", type=str, default="reports")
    ap.add_argument("--from-csv", type=str, default=None,
                    help="rebuild summary + PDF from an existing backtest_trades.csv (no downloads)")
    a = ap.parse_args()
    if a.start:
        C.BACKTEST_START = a.start

    if a.from_csv:
        df = pd.read_csv(a.from_csv)
        if "nifty_bull" not in df.columns:
            df = _add_regime(df)
        baseline_path = os.path.join(os.path.dirname(a.from_csv), "backtest_summary.json")
        baseline = {}
        if os.path.exists(baseline_path):
            try:
                baseline = json.load(open(baseline_path)).get("_baseline", {})
            except Exception:
                pass
        summary = {m: summarise(df[df.trigger_mode == m]) for m in df.trigger_mode.unique()}
        summary["_baseline"] = baseline
        _main = "first_bullish" if "first_bullish" in set(df.trigger_mode) else df.trigger_mode.iloc[0]
        summary["_evidence"] = evidence_tables(df[df.trigger_mode == _main])
        open_df = pd.DataFrame()
        op = os.path.join(os.path.dirname(a.from_csv), "backtest_open_trades.csv")
        if os.path.exists(op):
            open_df = pd.read_csv(op)
            if "nifty_bull" not in open_df.columns:
                open_df = _add_regime(open_df)
        summary["_meta"] = dict(universe="(from csv)", start=C.BACKTEST_START,
                                filters=f"price>₹{C.MIN_PRICE:.0f}, mcap≥₹{C.MIN_MCAP_CR:.0f}cr",
                                generated=str(D.now_ist())[:19], horizon=C.RET_HORIZON,
                                rule="pole(≥9.8% locked, 3x vol) → flag(≥50% give-back, base intact) → sweep of flag low → "
                                     "close back above the rail; stop sweep-low − 0.5ATR, target pole high, 60-bar time stop",
                                watchlist=D.load_watchlist(), open_trades=int(len(open_df)))
        json.dump(summary, open(f"{a.out}/backtest_summary.json", "w"), indent=1, default=str)
        from . import pdf_report
        path = pdf_report.build(df, summary, f"{a.out}/backtest_report.pdf", open_df=open_df)
        print(f"PDF report → {path}")
        _run_summary(df, summary, open_df, _main, a.out)
        return 0

    uni = D.load_universe(limit=a.limit)
    modes = ("rail_reclaim", "first_bullish") if a.trigger == "both" else (a.trigger,)
    df, baseline, open_df = run_backtest(uni, modes)
    os.makedirs(a.out, exist_ok=True)
    if not len(df):
        print("no trades found")
        return 1

    df.to_csv(f"{a.out}/backtest_trades.csv", index=False)
    if len(open_df):
        open_df.to_csv(f"{a.out}/backtest_open_trades.csv", index=False)
        print(f"\nstill-open positions at the report date ({len(open_df)}):")
        cols = [c for c in ["trigger_mode", "symbol", "shortName", "entry_date", "entry", "stop", "R", "pnl_pct", "bars_held"] if c in open_df.columns]
        print(open_df[cols].to_string(index=False))
    summary = {m: summarise(df[df.trigger_mode == m]) for m in df.trigger_mode.unique()}
    summary["_baseline"] = baseline
    _main = "first_bullish" if "first_bullish" in set(df.trigger_mode) else df.trigger_mode.iloc[0]
    summary["_evidence"] = evidence_tables(df[df.trigger_mode == _main])
    summary["_meta"] = dict(universe=len(uni), start=C.BACKTEST_START,
                            filters=f"price>₹{C.MIN_PRICE:.0f}, mcap≥₹{C.MIN_MCAP_CR:.0f}cr",
                            generated=str(D.now_ist())[:19], horizon=C.RET_HORIZON,
                            rule="pole(≥9.8% locked, 3x vol) → flag(≥50% give-back, base intact) → sweep of flag low → "
                                 "close back above the rail; stop sweep-low − 0.5ATR, target pole high, 60-bar time stop",
                            watchlist=D.load_watchlist(), open_trades=int(len(open_df)))
    json.dump(summary, open(f"{a.out}/backtest_summary.json", "w"), indent=1, default=str)

    for m, s in summary.items():
        if m.startswith("_") or not isinstance(s, dict) or "trades" not in s:
            continue
        print(f"\n===== {m}  ({s['trades']} trades) =====")
        print(f"  win rate (30-bar return > 0) : {s['win_rate_ret']*100:.1f}%")
        print(f"  avg / median 30-bar return   : {s['avg_ret']*100:+.2f}% / {s['median_ret']*100:+.2f}%")
        print(f"  target / stopped / time      : {s['target_pct']*100:.0f}% / {s['stopped_pct']*100:.0f}% / {s['time_pct']*100:.0f}%")
        print(f"  expectancy                   : {s['avg_R']:+.2f}R per trade  (payoff {s['payoff']}:1)")
        print(f"  total R / max drawdown       : {s['total_R']:+.1f}R / {s['max_drawdown_R']:.1f}R")
        print(f"  median risk / hold           : {s['median_risk_pct']}% / {s['avg_hold_bars']:.0f} bars")
        print(f"  median MFE / MAE             : {s['median_MFE']*100:+.1f}% / {s['median_MAE']*100:+.1f}%")

    if baseline:
        print(f"\n  BASELINE (any random bar, same universe): median {baseline['median_ret']*100:+.2f}%, "
              f"win {baseline['win_rate']*100:.1f}%  (n={baseline['samples']:,})")
    print("\nby year (rail_reclaim):")
    print(by_year(df[df.trigger_mode == "rail_reclaim"]).to_string())

    if a.pdf:
        from . import pdf_report
        path = pdf_report.build(df, summary, f"{a.out}/backtest_report.pdf", open_df=open_df)
        print(f"\nPDF report → {path}")
    _run_summary(df, summary, open_df, _main, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
