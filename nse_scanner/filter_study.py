#!/usr/bin/env python3
"""
filter_study.py — which features actually separate the winners from the false signals?

Method (deliberately anti-overfit):
  * split the trades by TIME:  train = entries before 2022-01-01,  test = 2022 onward
  * rank candidate filters on the TRAIN half only
  * report every candidate on the TEST half — that is the number that matters
  * a filter is only recommended if it holds up out-of-sample

Usage:
  python3 -m nse_scanner.filter_study --csv reports/backtest_trades.csv --mode first_bullish
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

SPLIT = "2022-01-01"

FEATURES = [
    ("sweep_below_pct", "how deep the shakeout undercut the rail", [-100, -12, -8, -5, -3, -1, 0.01]),
    ("risk_pct",        "stop distance as % of entry",              [0, 5, 8, 10, 12, 15, 20, 999]),
    ("reward_R",        "reward to pole high at entry",              [0, 1, 1.5, 2, 3, 5, 999]),
    ("pole_gain",       "pole size (low -> high)",                   [0, .2, .3, .4, .5, .7, 99]),
    ("flag_bars",       "drift length in bars",                      [0, 8, 12, 18, 25, 40, 999]),
    ("vol_x20",         "volume on the trigger bar vs 20-day avg",   [0, .5, .8, 1.2, 2, 3, 999]),
    ("mcap_cr",         "market cap (Rs cr)",                        [0, 2000, 5000, 15000, 50000, 1e9]),
    ("turnover_cr",     "20-day average traded value (Rs cr)",       [0, 1, 5, 20, 100, 1e9]),
    ("atr",             "ATR in rupees",                             [0, 5, 15, 40, 100, 1e9]),
    ("rail_slope_pct",  "slope of the drift floor (%/bar)",          [-99, -0.5, -0.2, 0, 0.2, 99]),
    ("sweep_vol_x20",   "volume on the shakeout leg vs its 20d avg",  [0, 0.5, 1.0, 2.0, 3.0, 999]),
    ("sweep_vol_vs_drift", "shakeout volume vs the quiet drift's",    [0, 1.5, 3.0, 6.0, 999]),
    ("sweep_leg_pct",   "worst single day inside the shakeout (%)",   [-99, -8, -6, -4, -2, 0]),
    ("sweep_red_frac",  "share of red candles in the shakeout leg",   [0, 0.34, 0.5, 0.67, 0.99, 1.01]),
]
# NOTE: bars_held / outcome / R / ret_horizon / mfe / mae are deliberately NOT here.
# They are only known AFTER the trade and would leak the answer into the filter.


def _stats(x: pd.DataFrame) -> dict:
    x = x.dropna(subset=["R"])
    if len(x) < 5:
        return {}
    w, l = x[x.R > 0], x[x.R <= 0]
    return dict(n=int(len(x)),
                win=round(float((x.R > 0).mean()), 3),
                avgR=round(float(x.R.mean()), 3),
                medR=round(float(x.R.median()), 2),
                med30=round(float(x.ret_horizon.median()), 4) if "ret_horizon" in x else None,
                avg30=round(float(x.ret_horizon.mean()), 4) if "ret_horizon" in x else None,
                target=round(float((x.outcome == "target").mean()), 3),
                stopped=round(float((x.outcome == "stopped").mean()), 3),
                payoff=round(float(abs(w.R.mean() / l.R.mean())), 2) if len(l) and l.R.mean() else None)


def scan(d: pd.DataFrame, guard: bool = False) -> pd.DataFrame:
    """One-condition-at-a-time bucket scan, evaluated separately on train and test."""
    rows = []
    for col, label, cuts in FEATURES:
        if col not in d.columns or sorted(cuts) != list(cuts):
            continue
        b = pd.cut(d[col], cuts)
        for bucket in b.cat.categories:
            m = b == bucket
            tr, te = d[m & (d.entry_date < SPLIT)], d[m & (d.entry_date >= SPLIT)]
            if len(tr) < 20:
                continue
            st = _stats(tr); se = _stats(te)
            rows.append(dict(feature=col, bucket=str(bucket), label=label,
                             **{f"train_{k}": v for k, v in st.items()},
                             **{f"test_{k}": v for k, v in se.items()}))
    return pd.DataFrame(rows)


def greedy(d: pd.DataFrame, max_terms: int = 3, min_train: int = 90, min_test: int = 30) -> list[dict]:
    """
    Grow a rule set by adding the single condition that most improves TRAIN expectancy,
    then re-score on TEST.  Stops when nothing helps or the sample gets too thin.
    """
    base_tr, base_te = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    mask = pd.Series(True, index=d.index)
    history = [dict(step=0, cond="(all signals)", **{f"train_{k}": v for k, v in _stats(base_tr).items()},
                    **{f"test_{k}": v for k, v in _stats(base_te).items()})]
    for step in range(1, max_terms + 1):
        best = None
        for col, label, cuts in FEATURES:
            if col not in d.columns or sorted(cuts) != list(cuts):
                continue
            b = pd.cut(d[col], cuts)
            for bucket in b.cat.categories:
                cand = mask & (b == bucket)
                tr = d[cand & (d.entry_date < SPLIT)]
                if len(tr) < min_train:
                    continue
                s = _stats(tr)
                if not s:
                    continue
                if best is None or s["avgR"] > best[1]["avgR"]:
                    best = (f"{col} in {bucket}", s, cand)
        if best is None:
            break
        # only accept if it improves train expectancy
        if best[1]["avgR"] <= history[-1]["train_avgR"]:
            break
        mask = best[2]
        te = d[mask & (d.entry_date >= SPLIT)]
        st = _stats(te)
        if not st or st["n"] < min_test:
            break
        history.append(dict(step=step, cond=best[0],
                            **{f"train_{k}": v for k, v in best[1].items()},
                            **{f"test_{k}": v for k, v in st.items()}))
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/backtest_trades.csv")
    ap.add_argument("--mode", default="first_bullish")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.read_csv(a.csv)
    d = d[d.trigger_mode == a.mode].dropna(subset=["R"]).copy()
    d["entry_date"] = d.entry_date.astype(str)
    if "nifty_bull" in d.columns:
        d["nifty_bull"] = d.nifty_bull.map({True: 1, False: 0, "True": 1, "False": 0})
    pd.set_option("display.width", 240)
    print(f"mode={a.mode}  trades={len(d)}   train(<{SPLIT})={ (d.entry_date<SPLIT).sum() }  "
          f"test(>={SPLIT})={ (d.entry_date>=SPLIT).sum() }")
    # `baseline` was referenced later without ever being defined, so this script died with a
    # NameError after printing its first lines - and the workflow had continue-on-error on, so
    # the study silently produced nothing run after run.
    baseline = dict(train=_stats(d[d.entry_date < SPLIT]), test=_stats(d[d.entry_date >= SPLIT]))
    print(f"baseline: train avgR {baseline['train']['avgR']:+.3f} | "
          f"test avgR {baseline['test']['avgR']:+.3f}")

    s = scan(d)
    if len(s):
        s["edge_train"] = (s.train_avgR - _stats(d[d.entry_date < SPLIT])["avgR"]).round(3)
        s["edge_test"] = (s.test_avgR - _stats(d[d.entry_date >= SPLIT])["avgR"]).round(3)
        s = s.sort_values("edge_test", ascending=False)
        s.to_csv(f"{a.out}/filter_scan.csv", index=False)
        cols = ["feature", "bucket", "train_n", "train_avgR", "train_win", "test_n", "test_avgR",
                "test_win", "test_med30", "test_target", "edge_train", "edge_test"]
        print("\n=== single-condition buckets, ranked by OUT-OF-SAMPLE edge ===")
        show = s[cols].head(18)
        print(show.to_string(index=False))

    # regime is a special case (boolean)
    if "nifty_bull" in d.columns:
        print("\n=== market regime (NIFTY vs 200-DMA on the entry date) ===")
        for v, lab in [(1, "NIFTY above 200-DMA"), (0, "NIFTY below 200-DMA")]:
            x = d[d.nifty_bull == v]
            tr, te = _stats(x[x.entry_date < SPLIT]), _stats(x[x.entry_date >= SPLIT])
            print(f"  {lab:22s} train n={tr.get('n',0):4d} avgR {tr.get('avgR',float('nan')):+.3f} "
                  f"| test n={te.get('n',0):4d} avgR {te.get('avgR',float('nan')):+.3f} "
                  f"med30 {te.get('med30',float('nan'))*100:+.2f}%")

    print("\n=== greedy rule search (grown on TRAIN, judged on TEST) ===")
    h = greedy(d)
    for r in h:
        print(f"  step {r['step']}  {r['cond']:38s} train n={r['train_n']:4d} avgR {r['train_avgR']:+.3f} | "
              f"TEST n={r['test_n']:4d} avgR {r['test_avgR']:+.3f} win {r['test_win']*100:4.1f}% "
              f"med30 {r['test_med30']*100:+.2f}% target {r['test_target']*100:.0f}%")
    report = dict(mode=a.mode, split=SPLIT, signals=int(len(d)), steps=h)
    if baseline:
        report["baseline"] = baseline

    # how much "false stock" removal does the final rule buy?
    if len(h) > 1:
        final_cond = h[-1]["cond"]
        col, _, bucket = final_cond.partition(" in ")
        # re-derive the mask by replaying the greedy chain
        mask = pd.Series(True, index=d.index)
        for r in h[1:]:
            c, _, bkt = r["cond"].partition(" in ")
            spec = dict((f[0], f[2]) for f in FEATURES).get(c)
            mask &= (pd.cut(d[c], spec).astype(str) == bkt)
        kept, dropped = d[mask], d[~mask]
        ks, ds2 = _stats(kept), _stats(dropped)
        print(f"\n=== what the {len(h)-1}-condition rule removes ===")
        print(f"  KEPT    n={ks['n']:4d} ({ks['n']/len(d)*100:.0f}% of signals)  avgR {ks['avgR']:+.3f}  "
              f"win {ks['win']*100:.0f}%  med30 {ks['med30']*100:+.2f}%  stopped {ks['stopped']*100:.0f}%")
        print(f"  DROPPED n={ds2['n']:4d} ({ds2['n']/len(d)*100:.0f}% of signals)  avgR {ds2['avgR']:+.3f}  "
              f"win {ds2['win']*100:.0f}%  med30 {ds2['med30']*100:+.2f}%  stopped {ds2['stopped']*100:.0f}%")
        print(f"  -> the rule keeps {ks['avgR']/ds2['avgR'] if ds2['avgR'] else float('inf'):.1f}x the expectancy "
              f"while discarding {ds2['n']/len(d)*100:.0f}% of the signals")
        # test-half only
        kt = _stats(kept[kept.entry_date >= SPLIT]); dt_ = _stats(dropped[dropped.entry_date >= SPLIT])
        print(f"  OUT-OF-SAMPLE: kept avgR {kt.get('avgR')} (n={kt.get('n')}) vs dropped avgR {dt_.get('avgR')} (n={dt_.get('n')})")
        report["kept"] = ks; report["dropped"] = ds2
        report["kept_test"] = kt; report["dropped_test"] = dt_
    json.dump(report, open(f"{a.out}/filter_study.json", "w"), indent=1, default=str)
    print(f"\nwrote {a.out}/filter_scan.csv and {a.out}/filter_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
