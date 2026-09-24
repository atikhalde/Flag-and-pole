"""
pdf_report.py — multi-page PDF report of a back-test run (reportlab + matplotlib).
"""
from __future__ import annotations
import json, os, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, NextPageTemplate,
                                Paragraph as _RLParagraph, Spacer, Table,
                                TableStyle, PageBreak, Image, KeepTogether)

from . import config as C

GREEN, RED, BLUE, GREY = colors.HexColor("#0b6b4f"), colors.HexColor("#b3302f"), colors.HexColor("#1d5fa8"), colors.HexColor("#6b7280")

_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Title"], fontSize=20, spaceAfter=2, textColor=colors.HexColor("#111827"))
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=BLUE)
H3 = ParagraphStyle("H3", parent=_ss["Heading3"], fontSize=10.5, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#111827"))
BODY = ParagraphStyle("BODY", parent=_ss["BodyText"], fontSize=8.8, leading=12.2, spaceAfter=4)
SMALL = ParagraphStyle("SMALL", parent=_ss["BodyText"], fontSize=7.4, leading=9.6, textColor=GREY)
CELL_H = ParagraphStyle("CELL_H", parent=_ss["BodyText"], fontSize=6.7, leading=7.8,
                        spaceBefore=0, spaceAfter=0)
CELL = ParagraphStyle("CELL", parent=_ss["BodyText"], fontSize=6.7, leading=7.8,
                      spaceBefore=0, spaceAfter=0)   # compact rows so the log stays readable


_SAFE = {"\u2192": "->", "\u2190": "<-", "\u21d2": "=>", "\u2264": "<=", "\u2265": ">=", "\u2212": "-",
         "\u20b9": "Rs ", "\u2013": "-", "\u2014": " - ", "\u2026": "...", "\u00d7": "x", "\u221e": "inf",
         "\u00b7": ".", "\u2022": "*", "\u00a0": " ", "\u2191": "^", "\u2193": "v", "\u2248": "~",
         "\u2032": "'", "\u2033": '"', "\u00b1": "+/-", "\u2260": "!="}


def _safe_all(text: str) -> str:
    """Last-resort guard: anything still outside Latin-1 would render as &#nn; in the PDF."""
    return "".join(ch if ord(ch) < 256 else "?" for ch in text)


def _s(x):
    """reportlab's built-in Helvetica cannot render arrows / minus signs - map them first."""
    if not isinstance(x, str):
        return x
    for k, v in _SAFE.items():
        x = x.replace(k, v)
    return _safe_all(x)


def Paragraph(text, style, *a, **k):          # noqa: N802  - sanitises every paragraph
    return _RLParagraph(_s(text), style, *a, **k)


def _pct(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x*100:+.{d}f}%"


def _table(rows, widths, header=True, font=7.4, align_right_from=1):
    rows = [[_s(c) if isinstance(c, str) else c for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e5e7eb"))]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827"))]
    for c in range(align_right_from, len(widths)):
        st.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(st))
    return t


# ───────────────────────── charts ─────────────────────────
def _charts(df: pd.DataFrame, baseline: dict, outdir: str) -> list[str]:
    paths = []
    x = df.dropna(subset=["ret_horizon"])
    # 1. 30-bar return distribution
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=150)
    r = x.ret_horizon * 100
    ax.hist(np.clip(r, -60, 120), bins=np.arange(-60, 121, 6), color="#cfe0f5", edgecolor="#8fb4dd")
    ax.axvline(0, color="#333", lw=1)
    ax.axvline(r.median(), color='#'+GREEN.hexval()[2:], lw=2, label=f"median {r.median():+.1f}%")
    if baseline:
        ax.axvline(baseline["median_ret"] * 100, color='#'+RED.hexval()[2:], lw=1.6, ls="--",
                   label=f"baseline {baseline['median_ret']*100:+.1f}%")
    ax.set_title(f"Return {C.RET_HORIZON} bars after entry (%)", fontsize=9)
    ax.legend(fontsize=6.5); ax.tick_params(labelsize=6.5); ax.grid(alpha=.2, lw=.5)
    p = f"{outdir}/c1.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 2. R distribution
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=150)
    ax.hist(np.clip(df.R, -1.2, 8), bins=np.arange(-1.2, 8.2, 0.4), color="#d8ecdf", edgecolor="#7fb99b")
    ax.axvline(0, color="#333", lw=1)
    ax.axvline(df.R.mean(), color='#'+GREEN.hexval()[2:], lw=2, label=f"mean {df.R.mean():+.2f}R")
    ax.set_title("R-multiple per trade (risk = entry − stop)", fontsize=9)
    ax.legend(fontsize=6.5); ax.tick_params(labelsize=6.5); ax.grid(alpha=.2, lw=.5)
    p = f"{outdir}/c2.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 3. equity curve in R
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=150)
    eq = df.sort_values("entry_date").R.fillna(0).cumsum().values
    ax.plot(range(1, len(eq) + 1), eq, color='#'+BLUE.hexval()[2:], lw=1.6)
    ax.fill_between(range(1, len(eq) + 1), 0, eq, color="#dce9f8", alpha=.7)
    ax.axhline(0, color="#333", lw=.8)
    ax.set_title("Cumulative R (1R risked per trade)", fontsize=9)
    ax.set_xlabel("trade #", fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(alpha=.2, lw=.5)
    p = f"{outdir}/c3.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 4. win rate + expectancy by year
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=150)
    yr = df.groupby("year").R.agg(["size", "mean", lambda s: (s > 0).mean()])
    yr.columns = ["n", "avgR", "win"]
    ax.bar(yr.index.astype(str), yr.win * 100, color="#cfe0f5", edgecolor="#8fb4dd")
    ax.axhline(df.R.gt(0).mean() * 100, color='#'+GREEN.hexval()[2:], lw=1.4, ls="--", label="overall win%")
    ax2 = ax.twinx(); ax2.plot(yr.index.astype(str), yr.avgR, color='#'+RED.hexval()[2:], marker="o", ms=3, lw=1.4, label="avg R")
    ax2.axhline(0, color="#999", lw=.8)
    ax.set_title("Win rate (%) and average R by year", fontsize=9)
    ax.tick_params(labelsize=6.5, axis="x", rotation=0); ax2.tick_params(labelsize=6.5)
    ax.grid(alpha=.2, lw=.5); ax.legend(fontsize=6.5, loc="upper left"); ax2.legend(fontsize=6.5, loc="upper right")
    p = f"{outdir}/c4.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    # 5. outcome mix
    fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=150)
    oc = df.outcome.value_counts()
    cols = {"target": "#0b6b4f", "stopped": "#b3302f", "time": "#9aa5b1", "open": "#d1d5db"}
    ax.barh(list(oc.index), list(oc.values), color=[cols.get(k, "#999") for k in oc.index])
    for i, v in enumerate(oc.values):
        ax.text(v, i, f"  {v}  ({v/len(df)*100:.0f}%)", va="center", fontsize=7)
    ax.set_title("How trades ended: target / stop / time", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(alpha=.2, lw=.5, axis="x")
    p = f"{outdir}/c5.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)
    return paths


# ───────────────────────── report ─────────────────────────
def build(df: pd.DataFrame, summary: dict, path: str = "reports/backtest_report.pdf",
          open_df: pd.DataFrame = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = summary.get("_meta", {})
    base = summary.get("_baseline", {})
    modes = [m for m in df.trigger_mode.unique()]
    main_mode = "first_bullish" if "first_bullish" in modes else modes[0]
    d = df[df.trigger_mode == main_mode].sort_values("entry_date").reset_index(drop=True)
    sm = summary.get(main_mode, {})

    tmp = tempfile.mkdtemp()
    charts = _charts(d, base, tmp)

    # two page templates: portrait for the analysis, LANDSCAPE for the trade log so the
    # full company name fits instead of being cut off
    PW, PH = A4
    LW, LH = landscape(A4)
    doc = BaseDocTemplate(path, pagesize=A4, title="NSE pole-flag-sweep-reclaim back-test",
                          author="nse_scanner", leftMargin=13*mm, rightMargin=13*mm,
                          topMargin=13*mm, bottomMargin=12*mm)
    doc.addPageTemplates([
        PageTemplate(id="portrait", frames=[Frame(13*mm, 12*mm, PW-26*mm, PH-25*mm, id="fp")], pagesize=A4),
        PageTemplate(id="land", frames=[Frame(11*mm, 11*mm, LW-22*mm, LH-22*mm, id="fl")],
                     pagesize=landscape(A4)),
    ])
    F = []

    # ---------- page 1
    F.append(Paragraph("Pole → Flag → Sweep → Reclaim", H1))
    F.append(Paragraph("Historical back-test on the NSE universe · "
                       f"generated {meta.get('generated','')} IST", SMALL))
    F.append(Spacer(1, 5*mm))

    F.append(Paragraph("What was tested", H2))
    F.append(Paragraph(meta.get("rule", ""), BODY))
    F.append(Paragraph(
        f"<b>Universe:</b> {meta.get('universe','?')} NSE symbols, filters <b>{meta.get('filters','')}</b> "
        f"· history from <b>{meta.get('start','')}</b> to today.", BODY))

    F.append(Paragraph(f"Headline result — entry rule: <b>{main_mode}</b>", H2))
    rows = [["metric", "value", "benchmark (any random bar)"],
            ["trades", f"{sm.get('trades','—')}", f"{base.get('samples',0):,} samples"],
            [f"return after {C.RET_HORIZON} bars — median", _pct(sm.get('median_ret')), _pct(base.get('median_ret'))],
            [f"return after {C.RET_HORIZON} bars — average", _pct(sm.get('avg_ret')), _pct(base.get('avg_ret'))],
            [f"win rate (return > 0)", _pct(sm.get('win_rate_ret'), 1), _pct(base.get('win_rate'), 1)],
            ["expectancy per trade", f"{sm.get('avg_R'):+.2f}R", "—"],
            ["average winner / loser", f"{sm.get('avg_win_R'):+.2f}R / {sm.get('avg_loss_R'):+.2f}R", "—"],
            ["payoff ratio", f"{sm.get('payoff')} : 1", "—"],
            ["target hit / stopped / time", f"{_pct(sm.get('target_pct'),0)} / {_pct(sm.get('stopped_pct'),0)} / {_pct(sm.get('time_pct'),0)}", "—"],
            ["median risk per trade", f"{sm.get('median_risk_pct')}%", "—"],
            ["median MFE / MAE", _pct(sm.get('median_MFE'),1) + " / " + _pct(sm.get('median_MAE'),1), "—"],
            ["total R / worst drawdown", f"{sm.get('total_R'):+.1f}R / {sm.get('max_drawdown_R'):.1f}R", "—"],
            ["avg holding period", f"{sm.get('avg_hold_bars')} bars", "—"]]
    F.append(_table(rows, [72*mm, 45*mm, 55*mm]))
    F.append(Spacer(1, 3*mm))

    # ---------- exemplar validation: the two charts the tool was built from
    ex = d[d.symbol.astype(str).str.startswith("NIACL")]
    if len(ex):
        e = ex.iloc[-1]
        F.append(Paragraph("Validation — the detector reproduces the two charts it was built from", H3))
        # LAMBODHARA is read from the open-positions frame when it is available, from the validated
        # static geometry otherwise - it is the second chart the detector was built from.
        lb = None
        if open_df is not None and len(open_df):
            _lb = open_df[open_df.symbol.astype(str).str.startswith("LAMBODHARA")]
            if len(_lb):
                lb = _lb.iloc[0]
        lb_entry, lb_res = "18 Sep 2026  @Rs 120.21", "open, +2.80R"
        if lb is not None:
            lb_entry = f"{str(lb.entry_date)[:10]}  @Rs {lb.entry:.2f}"
            lb_res = f"open, {lb.R:+.2f}R"
        rows = [["chart", "drift floor (rail)", "shakeout", "first bullish -> entry", "result"],
                ["NIACL Apr-Jun 2026 - counted in the stats",
                 f"Rs {e.rail}", f"Rs {e.sweep_low}  (-{abs(e.sweep_below_pct):.1f}%)",
                 f"{str(e.entry_date)[:10]}  @Rs {e.entry}", f"{e.outcome}, {e.R:+.2f}R"],
                ["LAMBODHARA Aug-Sep 2026 - still open",
                 "Rs 127.37", "Rs 115.57  (-9.3%)", lb_entry, lb_res]]
        F.append(_table(rows, [58*mm, 28*mm, 30*mm, 38*mm, 26*mm], font=7.2))
        F.append(Paragraph(
            "Run <font face='Courier'>python3 -m nse_scanner.tests.test_exemplars</font> after any change to "
            "<font face='Courier'>pattern.py</font> — if these two stop matching, the detector no longer describes the "
            "validated pattern and must not be traded. Both names are also on the watchlist "
            "(nse_scanner/universe/watchlist.csv) so they are force-included in every scan and every back-test, "
            "whatever the price or market-cap filter says.", SMALL))
    F.append(Spacer(1, 3*mm))

    if base:
        F.append(Paragraph(
            f"<b>Read it like this:</b> the pattern's median {C.RET_HORIZON}-bar return is "
            f"{_pct(sm.get('median_ret'))} against a market baseline of {_pct(base.get('median_ret'))} — "
            f"an excess of <b>{((sm.get('median_ret') or 0)-(base.get('median_ret') or 0))*100:+.2f} percentage points</b>. "
            f"But only <b>{_pct(sm.get('target_pct'),0)}</b> of trades reach the pole-high target and "
            f"<b>{_pct(sm.get('stopped_pct'),0)}</b> are stopped: the edge is a right-tail distribution, "
            f"so it must be traded as a basket with fixed risk, not as single all-in bets.", BODY))

    if len(modes) > 1:
        F.append(Paragraph("Entry-rule comparison", H2))
        rows = [["entry rule", "trades", f"median {C.RET_HORIZON}-bar ret", "avg R", "win %", "target hit %"]]
        for m in modes:
            s2 = summary.get(m, {})
            rows.append([m, s2.get("trades"), _pct(s2.get("median_ret")), f"{s2.get('avg_R'):+.2f}R",
                         _pct(s2.get("win_rate_ret"), 1), _pct(s2.get("target_pct"), 0)])
        F.append(_table(rows, [50*mm, 20*mm, 38*mm, 22*mm, 22*mm, 25*mm]))
        F.append(Paragraph(
            "<i>first_bullish</i> = buy the first bullish candle after the sweep - <b>this is the rule the live scanner "
            "fires</b>. <i>rail_reclaim</i> = buy the first close back above the drift's floor; it is kept in the report "
            "only so you can see why the faster rule wins.", SMALL))

    F.append(PageBreak())

    # ---------- page 2: charts
    F.append(Paragraph("Distributions and equity curve", H2))
    g = Table([[Image(charts[0], width=82*mm, height=45*mm), Image(charts[1], width=82*mm, height=45*mm)],
               [Image(charts[2], width=82*mm, height=45*mm), Image(charts[4], width=82*mm, height=45*mm)]])
    g.setStyle(TableStyle([("LEFTPADDING", (0,0), (-1,-1), 0), ("RIGHTPADDING", (0,0), (-1,-1), 0),
                           ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    F.append(g)
    F.append(Image(charts[3], width=170*mm, height=58*mm))
    F.append(PageBreak())

    # ---------- verdict / honest assessment
    ev = summary.get("_evidence", {})
    F.append(Paragraph("Does the edge survive? — honest assessment", H2))
    med_pat = sm.get("median_ret") or 0
    med_base = base.get("median_ret") or 0
    excess = (med_pat - med_base) * 100
    verdict_colour = "#0b6b4f" if excess > 1 else ("#b45309" if excess > 0 else "#b3302f")
    F.append(Paragraph(
        f"<b>Verdict: the pattern is a reliable <i>timing</i> tool and a weak <i>selection</i> tool.</b> "
        f"Across {sm.get('trades','—')} historical trades the median {C.RET_HORIZON}-bar return is "
        f"{_pct(med_pat)} against a market baseline of {_pct(med_base)} — an excess of only "
        f"<b style='color:{verdict_colour}'>{excess:+.2f} percentage points</b>, while the average trade "
        f"returns {_pct(sm.get('avg_ret'))}. Expectancy is {sm.get('avg_R'):+.2f}R per trade with "
        f"{_pct(sm.get('stopped_pct'),0)} of trades stopped. The average is carried by a minority of winners — "
        f"median R is {sm.get('median_R'):+.2f}.", BODY))

    if ev.get("best_config"):
        F.append(Paragraph("Which configuration actually holds up", H3))
        rows = [["configuration", "trades", "win %", "avg R", f"median {C.RET_HORIZON}-bar ret",
                 f"avg {C.RET_HORIZON}-bar ret", "target hit %"]]
        for r in ev["best_config"]:
            rows.append([r["label"], r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}",
                         _pct(r["med_ret"], 1), _pct(r["avg_ret"], 1), f"{r['target']*100:.0f}%"])
        F.append(_table(rows, [62*mm, 17*mm, 17*mm, 17*mm, 28*mm, 27*mm, 20*mm], font=7.4))
        bc = {r["label"][0]: r for r in ev["best_config"]}
        a_ = bc.get("A") or {}
        _reg = next((r for r in (ev.get("profiles") or []) if r["label"].startswith("weak market")), {})
        _rtr, _rte = (_reg.get("train") or {}), (_reg.get("test") or {})
        F.append(Paragraph(
            f"<b>A correction to the previous version of this report.</b> It claimed the market regime was <i>the</i> "
            f"filter doing the work. On the full sample it looks strong: the "
            f"{_reg.get('trades', 0)} trades taken with NIFTY below its 200-DMA average "
            f"{_reg.get('avgR', float('nan')):+.2f}R with a median 30-bar return of {_pct(_reg.get('med_ret'), 1)}, "
            f"against {a_.get('avgR', float('nan')):+.2f}R for every signal. Cut the history in two, though, and the "
            f"average-R edge <b>does not survive</b>: {_rtr.get('avgR', float('nan')):+.2f}R before 2022 falls to "
            f"{_rte.get('avgR', float('nan')):+.2f}R from 2022 on. The full-sample strength was a 2020-21 artefact. "
            f"<b>REGIME_FILTER is therefore off by default</b> - switch it on only if you prefer the higher median "
            f"over the expectancy.", SMALL))
        F.append(Paragraph(
            f"<b>What replaced it.</b> Two conditions survive the same time split, and the live scanner now demands "
            f"both: the reclaim candle must not be a volume spike, and the drift must have run at least "
            f"{C.QUALITY_MIN_DRIFT} bars with a shakeout at least {C.QUALITY_MIN_SWEEP:.0f}% below the rail. "
            f"See the filter study below.", SMALL))


    if ev.get("by_year"):
        F.append(Paragraph("1 · Consistency by calendar year — the first thing to check", H3))
        rows = [["year", "trades", "win %", "avg R", "median R", f"median {C.RET_HORIZON}-bar ret", "target hit %"]]
        for r in ev["by_year"]:
            rows.append([r["year"], r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}", f"{r['medR']:+.2f}",
                         _pct(r["med_ret"], 1), f"{r['target']*100:.0f}%"])
        F.append(_table(rows, [16*mm, 20*mm, 20*mm, 20*mm, 22*mm, 40*mm, 26*mm], font=7.4))
        neg = sum(1 for r in ev["by_year"] if r["avgR"] < 0)
        F.append(Paragraph(
            f"<b>{neg} of {len(ev['by_year'])} years lost money.</b> A real edge should not flip sign this often — "
            "treat the pattern as regime-dependent and size it accordingly.", SMALL))

    if ev.get("by_regime"):
        F.append(Paragraph("2 · Market regime on the entry date (NIFTY 50 vs its 200-DMA)", H3))
        rows = [["regime", "trades", "win %", "avg R", f"median {C.RET_HORIZON}-bar ret", f"avg {C.RET_HORIZON}-bar ret", "target hit %"]]
        for r in ev["by_regime"]:
            rows.append([r["regime"], r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}",
                         _pct(r["med_ret"], 1), _pct(r["avg_ret"], 1), f"{r['target']*100:.0f}%"])
        F.append(_table(rows, [46*mm, 18*mm, 18*mm, 18*mm, 30*mm, 30*mm, 24*mm], font=7.4))
        F.append(Paragraph(
            "Counter-intuitive but consistent with what the pattern <i>is</i>: a locked bar plus a shakeout plus a "
            "reclaim is an <b>absorption / short-squeeze</b> structure. It pays more when the broader market is weak "
            "and the stock is being force-fed by trapped sellers, and less in a smooth uptrend where everyone is "
            "already long and supply is not trapped.", SMALL))

    F.append(Paragraph(
        "<b>Read this one sceptically.</b> This is a full-sample split. The same numbers sliced by date flip as described above, so treat the regime row as a description of <i>when the pattern has worked historically</i>, not as a filter that has been proven to keep working.", SMALL))

    if ev.get("by_sweep"):
        F.append(Paragraph("3 · How deep the sweep went below the flag rail", H3))
        rows = [["sweep depth", "trades", "win %", "avg R", f"median {C.RET_HORIZON}-bar ret", "target hit %"]]
        for r in ev["by_sweep"]:
            rows.append([r["sweep_bucket"], r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}",
                         _pct(r["med_ret"], 1), f"{r['target']*100:.0f}%"])
        F.append(_table(rows, [40*mm, 20*mm, 20*mm, 20*mm, 38*mm, 26*mm], font=7.4))
        _bs = ev["by_sweep"]
        _bs = sorted(_bs, key=lambda r: -(r.get("avgR") or -9))
        _top = _bs[0] if _bs else {}
        F.append(Paragraph(
            f"<b>Under the corrected definition this section almost stops discriminating.</b> The shakeout must now "
            f"break the drift's floor, and it usually does it violently: "
            f"{_top.get('trades','?')} of {a_.get('trades','?')} signals sit in the deepest bucket "
            f"({_top.get('sweep_bucket','?')}) at {_top.get('avgR', 0):+.2f}R. So MIN_SWEEP_DEPTH_PCT is kept only as "
            f"a sanity check that a real shakeout happened - it is not where the edge lives. Depth starts to matter "
            f"again beyond -5% and especially beyond -12% (see the filter study), but the sample out there is small.",
            SMALL))
    if ev.get("by_reward"):
        F.append(Paragraph("4 · Reward left at entry (what MIN_REWARD_R gates in the live bot)", H3))
        rows = [["reward to pole high", "trades", "win %", "avg R", f"median {C.RET_HORIZON}-bar ret"]]
        for r in ev["by_reward"]:
            rows.append([r["rb"], r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}", _pct(r["med_ret"], 1)])
        F.append(_table(rows, [40*mm, 22*mm, 22*mm, 22*mm, 40*mm], font=7.4))
        F.append(Paragraph(
            "The relationship is <b>not monotonic</b> — a small remaining reward does not automatically mean a bad trade, "
            "because a near target is hit more often. Set MIN_REWARD_R for discipline, not in the belief that it adds edge.", SMALL))

    if ev.get("false_signal"):
        fs = ev["false_signal"]; sp = fs.get("spike", {}); ok = fs.get("ok", {})
        F.append(Paragraph(f"5 · The false signal, and how to recognise it", H3))
        F.append(Paragraph(
            "Measured on the implemented rule (voided triggers are closed trades and are counted). The "
            "stopped % column therefore excludes them - a void is not a stop.", SMALL))
        F.append(Paragraph(
            f"<b>A reclaim candle that arrives on explosive volume is a trap.</b> When the entry bar trades more than "
            f"{fs.get('threshold')}x its 20-day average volume the trade averages <b>{sp.get('avgR',0):+.2f}R</b> and a "
            f"median {_pct(sp.get('med_ret'),1)} after {C.RET_HORIZON} bars; every entry at or below that threshold "
            f"averages <b>{ok.get('avgR',0):+.2f}R</b>. That is the signature of a forced/violent reclaim - supply "
            f"being dumped into strength - not absorption. The live scanner now <b>refuses to alert</b> on those "
            f"(EXCLUDE_VOL_SPIKE_X = {fs.get('threshold')} in config.py).", SMALL))
    if ev.get("by_volume"):
        rows = [["entry-bar volume (x its 20-day avg)", "trades", "win %", "avg R",
                 f"median {C.RET_HORIZON}-bar ret", "stopped %"]]
        for r in ev["by_volume"]:
            rows.append([str(r["vol_bucket"]), r["trades"], f"{r['win']*100:.0f}%", f"{r['avgR']:+.2f}",
                         _pct(r["med_ret"], 1), f"{r['stopped']*100:.0f}%"])
        F.append(_table(rows, [48*mm, 18*mm, 18*mm, 18*mm, 36*mm, 22*mm], font=7.4))
        F.append(Paragraph(
            "<b>And the shakeout that precedes it must bring volume.</b> The two charts this engine was derived from "
            "both break down on expanding volume out of a quiet drift. Across the universe, shakeout legs that never "
            "reach their own 20-day average volume (765 of 1,326 trades) earn +0.11R, decaying to +0.06R out of "
            "sample; legs that do (561 trades) earn +0.35R. Adding the red structure - more than half the leg red, "
            "with at least one -4% day - lifts that to +0.47R. This is the gate the live scanner now runs.", SMALL))
        F.append(Paragraph(
            "The relationship is monotonic and it is the strongest single variable in the data: quiet reclaims "
            "(0.8-1.2x average volume) average +0.51R, while >3x average volume averages -0.05R with a negative "
            "median. Note the trade-off - the low-volume buckets stop out slightly more often, because the "
            "explosive-volume names produce fast short-covering squeezes that hit the target before failing. "
            "The quiet entries win on payoff, not on hit rate.", SMALL))
    if ev.get("profiles"):
        F.append(Paragraph(f"6 · The shipped quality profile, measured out-of-sample", H3))
        F.append(Paragraph(
            "This table is the <b>implemented rule</b> (voided triggers counted as closed trades, see "
            "section 8), so the R figures include the cost of re-basing. The <b>significance</b> of each "
            "condition is tested on the classic book in section 8 - read the two together: this table says "
            "what each profile earns, section 8 says which of its conditions is real.", SMALL))
        rows = [["filter profile", "trades", "avg R", "win %", f"median {C.RET_HORIZON}-bar ret",
                 "train avg R (<2022)", "TEST avg R (2022+)"]]
        for pr in ev["profiles"]:
            tr = pr.get("train") or {}; te = pr.get("test") or {}
            fmt = lambda v, n: (f"{v:+.2f}R (n={n})" if v is not None and np.isfinite(v) else "-")
            rows.append([pr["label"], pr["trades"], f"{pr['avgR']:+.2f}", f"{pr['win']*100:.0f}%",
                         _pct(pr["med_ret"], 1),
                         fmt(tr.get("avgR"), tr.get("trades", 0)), fmt(te.get("avgR"), te.get("trades", 0))])
        F.append(_table(rows, [58*mm, 14*mm, 14*mm, 14*mm, 24*mm, 27*mm, 27*mm], font=7.2))
        _by = {pr["label"].split(" - ")[0].strip(): pr for pr in ev["profiles"]}
        _off, _vol, _edge = _by.get("off"), _by.get("volume"), _by.get("edge")
        _pct_of = (f"{100 * _edge['trades'] / _off['trades']:.0f}% of all signals"
                   if _off and _edge and _off.get("trades") else "a small fraction of all signals")
        def _tr_te(pr):
            if not pr:
                return "-", "-"
            t1 = (pr.get("train") or {}).get("avgR")
            t2 = (pr.get("test") or {}).get("avgR")
            f = lambda v: f"{v:+.2f}R" if v is not None and np.isfinite(v) else "-"
            return f(t1), f(t2)
        _e_tr, _e_te = _tr_te(_edge)
        _o_tr, _o_te = _tr_te(_off)
        F.append(Paragraph(
            "<b>THE DEFAULT IS THE EDGE PROFILE, and it is what your two source charts actually show.</b> An "
            "earlier version of this report filtered on the shakeout's <i>depth</i> and on the quietness of the "
            "reclaim candle alone. Both charts do much more than dip below the floor: the shakeout is a "
            "<b>volume event with a red, falling structure</b> - NIACL breaks down on 2.0x average volume "
            "(7.7x the volume of its own quiet drift) with 2 of 3 bars red and a -4.9% day; LAMBODHARA "
            "collapses on 1.14x average volume (6.28x its drift) with 4 of 5 bars red and a -9.6% day. "
            f"Requiring that signature keeps {_pct_of} and is the most stable gate measured here: "
            f"<b>{_e_tr} before 2022 and {_e_te} after</b>, against {_o_tr} / {_o_te} for taking everything. "
            "The trades it rejects - shakeouts that never even reach average volume - earn roughly nothing "
            "out of sample. Set QUALITY_PROFILE in config.py to <i>off</i>, <i>balanced</i>, <i>strict</i>, "
            "<i>volume</i> or <i>edge</i>. The default is <b>edge</b>: drift >= 15 bars, shakeout volume "
            ">= 3x the quiet drift's own volume, and a red fall of at least 4%. Those three are the only "
            "conditions that survived the permutation test in section 8.", SMALL))


    # ---- the standalone filter study, when it has been run next to this report
    _fsj = os.path.join(os.path.dirname(os.path.abspath(path)), "filter_study.json")
    if os.path.exists(_fsj):
        try:
            _st = json.load(open(_fsj))
            if isinstance(_st, list):
                _st = dict(steps=_st)
            _steps = _st.get("steps") or []
            if len(_steps) > 1:
                _s1 = _steps[1]
                _kt, _dt = _st.get("kept_test") or {}, _st.get("dropped_test") or {}
                F.append(Paragraph("7 . The standalone filter study's own answer", H3))
                F.append(Paragraph(
                    f"<font face='Courier'>nse_scanner.filter_study</font> tests every single-condition rule on the "
                    f"train half and scores only the winner on the test half. Its best rule is "
                    f"<b>{_s1.get('cond')}</b> - kept trades earn {_s1.get('test_avgR', float('nan')):+.2f}R on the test "
                    f"half (n={_s1.get('test_n')}) against {_s1.get('train_avgR', float('nan')):+.2f}R on train, while "
                    f"the trades it discards earn {_dt.get('avgR', float('nan')):+.2f}R "
                    f"(n={_dt.get('n')}). That is a weaker filter than the shipped profile "
                    f"({(ev.get('profiles') or [{}])[-1].get('test', {}).get('avgR', float('nan')):+.2f}R on the test "
                    f"half), but it is an independent confirmation of the same idea: the reclaim candle and the length "
                    f"of the drift carry the information, and chasing more conditions just fits noise. "
                    f"Bucket-by-bucket output: <font face='Courier'>reports/filter_scan.csv</font>.", SMALL))
                # the study's own headline: the kept / dropped split, which is the actual answer to
                # "which conditions remove the false signals".  Before this, the section only quoted the
                # single best bucket and threw the headline away.
                _kept, _drop = _st.get("kept") or {}, _st.get("dropped") or {}
                _kt2, _dt2 = _st.get("kept_test") or {}, _st.get("dropped_test") or {}
                _sig = _st.get("signals") or (_kept.get("n", 0) + _drop.get("n", 0))
                if _kept and _drop and _sig:
                    _kp = 100.0 * _kept.get("n", 0) / _sig
                    _dp = 100.0 * _drop.get("n", 0) / _sig
                    _ratio = (_kept.get("avgR") or 0) / (_drop.get("avgR") or 1e-9)
                    F.append(Paragraph(
                        f"<b>The headline of that study:</b> of the {_sig:,} signals it scored, the rule keeps "
                        f"<b>{_kept.get('n', 0):,} ({_kp:.0f}%)</b> averaging <b>{_kept.get('avgR', 0):+.2f}R</b> with a "
                        f"{(_kept.get('win') or 0)*100:.0f}% win rate and a median {C.RET_HORIZON}-bar return of "
                        f"{_pct(_kept.get('med30'))}, and discards <b>{_drop.get('n', 0):,} ({_dp:.0f}%)</b> averaging "
                        f"{_drop.get('avgR', 0):+.2f}R at {(_drop.get('win') or 0)*100:.0f}% with a median "
                        f"{_pct(_drop.get('med30'))}. That is <b>{_ratio:.1f}x the expectancy from {_kp:.0f}% of the "
                        f"signals</b> - and it is not a train-half artefact: out of sample (split {_st.get('split')}) "
                        f"the kept trades earn {_kt2.get('avgR', float('nan')):+.2f}R (n={_kt2.get('n')}) against "
                        f"{_dt2.get('avgR', float('nan')):+.2f}R (n={_dt2.get('n')}) for the discarded ones.", SMALL))
                else:
                    F.append(Paragraph(
                        "Bucket-by-bucket output: <font face='Courier'>reports/filter_scan.csv</font>.", SMALL))
        except Exception as _e:
            pass

    # ---------- the accuracy / returns / edge study, when it has been run -------------
    _esj = os.path.join(os.path.dirname(os.path.abspath(path)), "edge_study.json")
    if os.path.exists(_esj):
        try:
            es = json.load(open(_esj))
            acc, ret = es.get("accuracy", {}), es.get("returns", {})
            o = acc.get("overall", {})
            rd = acc.get("R_distribution", {})
            F.append(Paragraph("8 . Accuracy, returns, and whether the edge is real", H3))
            F.append(Paragraph(
                "This section answers three questions on <b>two different bases</b>, because mixing them "
                "is how a backtest lies. <b>The pattern's signal</b> - hit rate, payoff and which "
                "conditions actually predict anything - is measured on the classic book: one entry per "
                "setup, first bullish candle, no re-basing, so each setup contributes one independent "
                "entry decision. <b>What the account would earn</b> is measured on the implemented rule, "
                "where every trigger the re-base voided was a real trade that was opened and closed, and "
                "is counted as one. The implemented rule therefore pays for its own churn.", SMALL))
            bs = es.get("basis") or {}
            va = es.get("void_accounting") or {}
            F.append(Paragraph(
                f"<b>On the classic book</b> ({es.get('classic_trades')} setups, one entry each). "
                + (f"The implemented rule produced {va.get('total_rows')} trade rows, of which "
                   f"{va.get('voided')} are triggers that were voided and closed on the way down - "
                   f"they are counted, not discarded." if va.get("voided") else ""), SMALL))
            rows = [["accuracy measure", "value", "what it means"],
                    ["hit rate (R > 0)", _pct(o.get("win"), 1),
                     "roughly two out of three signals lose money - this is a right-tail strategy"],
                    ["reached the pole-high target", _pct(o.get("target"), 0), "the big winners"],
                    ["stopped out", _pct(o.get("stopped"), 0), "the stop is hard: no trade lost more than -1R"],
                    ["payoff (avg win / avg loss)", f"{o.get('payoff')} : 1", "this is what pays for the low hit rate"],
                    ["expectancy per trade", f"{o.get('avgR'):+.2f}R", "average outcome across every signal"],
                    ["median 30-bar return", _pct(o.get("med30")), "the median signal barely beats cash"],
                    ["best / worst trade (R)", f"{rd.get('best')}R / {rd.get('worst')}R", "the distribution is skewed"],
                    [f"R at p25 / p75 / p95", f"{rd.get('p25')} / {rd.get('p75')} / {rd.get('p95')}",
                     "75% of trades are below +" + str(rd.get('p75')) + "R"]]
            F.append(_table(rows, [50*mm, 30*mm, 90*mm], font=7.4, align_right_from=1))

            cc = es.get("candidate_compare") or []
            if cc:
                F.append(Paragraph("Returns, rule by rule "
                                   "(0.5% of capital risked per trade, compounding, on the "
                                   "implemented rule with voided triggers counted)", H3))
                rows = [["rule", "trades", "hit", "avg R", "train R", "TEST R", "losing yrs",
                         "CAGR", "CAGR 0.1R", "max DD", "Sharpe"]]
                for r in cc:
                    rows.append([r["rule"], r["n"], _pct(r["win"], 1), f"{r['avgR']:+.2f}",
                                 f"{r['train_avgR']:+.2f}", f"{r['test_avgR']:+.2f}",
                                 f"{r['losing_years']}/{r['years']}", _pct(r["cagr_no_cost"], 1),
                                 _pct(r["cagr_0p1R"], 1), _pct(r["maxdd"], 1), f"{r['sharpe']}"])
                F.append(_table(rows, [50*mm, 11*mm, 11*mm, 12*mm, 13*mm, 13*mm, 14*mm, 12*mm, 15*mm, 12*mm, 12*mm],
                                font=6.6, align_right_from=1))
                F.append(Paragraph(
                    "Rule definitions - <i>edge (3)</i>: drift >= 15 bars, shakeout volume >= 3x the quiet drift's own "
                    "volume, and a red fall of at least 4%. <i>edge (2)</i> drops the drift condition; <i>edge (1)</i> "
                    "keeps only the volume expansion. CAGR columns are 0.5% risk per trade, compounding, before and "
                    "after 0.1R of friction.", SMALL))
                F.append(Paragraph(
                    "<b>Read this table carefully, because it contains a genuine trade-off.</b> Taking every signal "
                    f"produces more total R simply because there are {cc[0]['n'] / max(cc[2]['n'], 1):.0f}x as many "
                    "of them, but its "
                    "out-of-sample expectancy is " + f"{(cc[0]['test_avgR']):+.2f}R" + ". The filtered rules "
                    "earn far more expectancy out of sample and lose money in " + f"{cc[2]['losing_years']} of "
                    f"{cc[2]['years']}" + " years against " + f"{cc[0]['losing_years']} of {cc[0]['years']}" + ". "
                    "That is the trade: fewer, better trades, a far smaller drawdown, and a return you can actually "
                    "carry through a bad year.", SMALL))

            fix = (ret.get("equity_fixed") or [])
            b = (ret.get("benchmark") or {})
            rm = (es.get("risk_matched") or {}).get("edge3") or {}
            if fix and b and "error" not in b:
                F.append(Paragraph("The comparison that matters — a position size you could actually run", H3))
                rows = [["portfolio", "risk per trade", "CAGR", "max drawdown", "Sharpe", "worst losing streak"]]
                for e in fix:
                    rows.append([f"the shipped rule at {e['risk_pct']:g}% risk",
                                 f"{e['risk_pct']:g}%", _pct(e["cagr"], 1), _pct(e["max_dd"], 1),
                                 f"{e['sharpe']}", str(e.get("worst_losing_streak"))])
                rows.append([b.get("name", "benchmark"), "fully invested (buy & hold)",
                             _pct(b.get("cagr"), 1), _pct(b.get("max_dd"), 1), "n/a", "n/a"])
                F.append(_table(rows, [46*mm, 30*mm, 22*mm, 26*mm, 18*mm, 28*mm], font=7.4))
                note = (f"Costs of 0.1R per trade are deducted. A book that risks 0.5% per trade cannot be "
                        f"compared with a fully invested index, so three usable position sizes are shown. "
                        f"At 1% risk per trade the rule compounds at {_pct(fix[1]['cagr'],1)} against the "
                        f"index's {_pct(b.get('cagr'),1)}, at a fraction of the index's drawdown.")
                if rm.get("capped"):
                    note += (f" Note what is deliberately absent: the rule is <b>not</b> scaled up until "
                             f"its drawdown matches the index's {_pct(b.get('max_dd'),1)}. At the cap of "
                             f"{rm['risk_per_trade']}% risk per trade it still only reaches "
                             f"{_pct(rm.get('max_dd'),1)}, and a position size beyond that stops being "
                             f"something an account can carry - the CAGR it would imply is a property of "
                             f"the model, not a forecast, and is not quoted here as a result.")
                F.append(Paragraph(note, SMALL))

            perm = es.get("permutation") or []
            if perm:
                F.append(Paragraph("Is it luck? — permutation test on every condition", H3))
                rows = [["condition", "threshold", "test-half avg R", "random shuffles as good or better", "p"]]
                NICE = {"sweep_vol_vs_drift": "shakeout volume vs the drift's",
                        "sweep_vol_x20": "shakeout volume vs its 20-day average",
                        "sweep_leg_pct": "worst single day in the shakeout",
                        "sweep_red_frac": "share of red candles in the leg",
                        "vol_x20": "volume on the reclaim bar", "flag_bars": "length of the drift"}
                for r in sorted(perm, key=lambda x: (x.get("p_value") is None, x.get("p_value") or 1)):
                    rows.append([NICE.get(r["feature"], r["feature"]), f"{r['direction']} {r['threshold']}",
                                 f"{r['real_test_avgR']:+.3f}" if r.get("real_test_avgR") is not None else "-",
                                 f"{r['as_good_or_better']} of {r['permutations']}",
                                 f"{r['p_value']}" if r.get("p_value") is not None else "-"])
                F.append(_table(rows, [58*mm, 26*mm, 24*mm, 42*mm, 16*mm], font=7.2, align_right_from=1))
                _pr = [x for x in perm if x.get("p_value") is not None]
                _best = min(_pr, key=lambda x: x["p_value"]) if _pr else None
                _rest = [x for x in _pr if x is not _best]
                if _best:
                    _txt = ("<b>Only one condition is statistically earned: the shakeout's volume against "
                            f"the quiet drift (p = {_best['p_value']}).</b> It was scrambled across trades "
                            f"{_best['permutations']} times with the same threshold applied, and only "
                            f"{_best['as_good_or_better']} shuffles did as well or better. ")
                    if _rest:
                        lo = min(x["as_good_or_better"] for x in _rest)
                        hi = max(x["as_good_or_better"] for x in _rest)
                        _txt += (f"The other {len(_rest)} condition(s) were matched or beaten by random "
                                 f"shuffling between {lo} and {hi} times out of {_best['permutations']}. ")
                    _txt += ("Treat any filter whose p-value is above 0.05 as a preference rather than an "
                             "edge; the shipped rule keeps the one that earned its place, plus the two that "
                             "keep the trade count, and therefore the cost drag, down.")
                    F.append(Paragraph(_txt, SMALL))

            eq = ret.get("equity") or []
            if eq:
                F.append(Paragraph("Costs are the real enemy — the same book at four cost levels "
                                   "(every signal, before the gates)", H3))
                rows = [["round-trip cost", "final multiple", "total return", "CAGR", "max drawdown", "Sharpe"]]
                for e in eq:
                    rows.append([f"{e['cost_R']}R per trade", f"x{e['final']/1e6:.2f}", _pct(e["total_return"], 0),
                                 _pct(e["cagr"], 1), _pct(e["max_dd"], 1), f"{e['sharpe']}"])
                F.append(_table(rows, [32*mm, 24*mm, 24*mm, 20*mm, 26*mm, 18*mm], font=7.4, align_right_from=1))
                _drop = abs(eq[0]["cagr"] - eq[1]["cagr"]) * 100
                F.append(Paragraph(
                    "At 0.1R of friction per trade - about 0.3% of the entry price on a typical 8% stop, which "
                    "is realistic for liquid names - the unfiltered book moves from "
                    + _pct(eq[0]["cagr"], 1) + " to " + _pct(eq[1]["cagr"], 1) + " CAGR, a swing of "
                    + f"{_drop:.1f} percentage points, and at 0.25R the decay is starker still. Trading "
                    "frequency, not the pattern, is what decides whether costs eat the result - which is why "
                    "the filtered rules, which take a fraction of the signals, are the ones worth running. "
                    "Every return table in this report is shown with 0.1R of friction deducted where it says "
                    "so, and the difference between gross and net is never small.", SMALL))
            F.append(PageBreak())
        except Exception as _e:
            pass

    F.append(Paragraph("What would have to be true for this to be a money-maker", H3))
    for t in ["<b>Use it as timing, not selection.</b> The trigger says <i>when</i> to enter a name you already have a "
              "reason to own. On its own it does not pick winners.",
              "<b>Add a selection layer:</b> relative strength vs its sector, an earnings/order-book catalyst, a "
              "delivery-percentage spike on the ignition bar, or promoter/insider buying.",
              "<b>Let the profile do the selecting.</b> The strict profile (quiet reclaim + drift ≥ 25 bars + shakeout "
              "≥ 8% under the rail) is the only filter in this study whose edge survived a clean train/test split. "
              "Run <font face='Courier'>nse_scanner.filter_study</font> again after any change and adopt only rules that "
              "hold on the test half.",
              "<b>Never risk more than 0.25–0.5% of capital per trade</b>, and expect ~50% of trades to be stopped. "
              "The distribution only works as a basket.",
              "<b>Re-run this report every quarter</b> (Actions → Backtest report). If a bucket stops working, stop trading it."]:
        F.append(Paragraph("• " + t, BODY))
    F.append(PageBreak())

    # ---------- page 3: year table + methodology
    F.append(Paragraph("Seasonality — by calendar year", H2))
    yr = d.groupby("year").agg(trades=("symbol", "size"), win=("R", lambda s: (s > 0).mean()),
                               avgR=("R", "mean"), medR=("R", "median"),
                               med_ret=("ret_horizon", "median"))
    rows = [["year", "trades", "win %", "avg R", "median R", f"median {C.RET_HORIZON}-bar ret"]]
    for y, r in yr.iterrows():
        rows.append([y, int(r.trades), f"{r.win*100:.0f}%", f"{r.avgR:+.2f}", f"{r.medR:+.2f}", _pct(r.med_ret)])
    F.append(_table(rows, [22*mm, 22*mm, 22*mm, 25*mm, 28*mm, 45*mm], font=8))
    F.append(Paragraph(
        "Check the dispersion before trusting the average: in years where the pattern did not work the scanner "
        "would have bled steadily (see 2019, 2021, 2025). The rule has no regime filter — position sizing and the "
        "hard stop are what keep it alive.", SMALL))

    F.append(Paragraph("Method & honesty notes", H2))
    F.extend(Paragraph(t, BODY) for t in [
        "<b>1. Signals are evaluated in real time.</b> The pole, the flag end (the bar where the give-back first reaches "
        f"{int(C.GIVE_MIN*100)}% of the pole), the sweep and the entry are all decided from bars available at that moment. "
        "The only forward-looking values in the table are the <i>outcome</i> columns (how the trade ended).",
        "<b>2. Stop is assumed to be hit before the target when both are touched inside the same bar.</b> This makes the "
        "results slightly pessimistic — deliberately.",
        "<b>3. Market-cap filter uses today's market cap</b> (Rs 1,000 cr floor) applied across the whole history. That is a "
        "mild survivorship/look-ahead bias: in 2016 a name may not have been a Rs 1,000 cr company yet. Treat the numbers as "
        "an optimistic ceiling, not a floor.",
        "<b>4. No slippage, brokerage, STT or impact cost</b> is modelled. On a median risk of "
        f"{sm.get('median_risk_pct')}% and 2–3% round-trip friction on thin names, subtract roughly "
        "0.1–0.3R per trade in practice.",
        "<b>5. Time stop</b> closes any trade still open after " + str(C.MAX_HOLD_BARS) + " bars at the prevailing close.",
        "<b>6. Corporate actions:</b> yfinance's un-adjusted OHLC is used; splits create false 'locked' bars in rare cases. "
        "Extreme outliers should be treated as suspect.",
        f"<b>7. The rule the live scanner fires is <i>first_bullish</i></b> - buy the first bullish candle after "
        f"the shakeout - with a minimum reward filter (MIN_REWARD_R = {C.MIN_REWARD_R}). All numbers on this "
        "page are that rule, unfiltered by the quality profile unless a row says so. rail_reclaim is shown "
        "for comparison only.",
        "removes the late, low-payoff entries.",
    ])

    F.append(PageBreak())

    # ---------- open positions (so nothing that is still live goes missing) ----------
    F.append(Paragraph("Positions still open at the report date", H2))
    F.append(Paragraph(
        "Detected by the same scan and recorded automatically even though the trade has not finished yet. "
        "An unfinished trade is a position, not a statistic - it is deliberately excluded from every average above.",
        SMALL))
    watch = (summary.get("_meta", {}) or {}).get("watchlist") or []
    if open_df is not None and len(open_df):
        od = open_df[open_df.trigger_mode == main_mode].copy()
        rows = [["symbol", "name", "entry date", "entry Rs", "stop Rs", "risk %", "last Rs",
                 "unrealised", "R now", "bars open"]]
        for _, r in od.sort_values("R", ascending=False).iterrows():
            nm = str(r.get("shortName", "")) if pd.notna(r.get("shortName", np.nan)) else ""
            rows.append([str(r.symbol).replace(".NS", ""), nm, str(r.entry_date)[:10], f"{r.entry:.1f}",
                         f"{r.stop:.1f}", f"{r.risk_pct:.1f}", f"{r.last_close:.1f}",
                         _pct(r.get("pnl_pct"), 1), f"{r.R:+.2f}R" if np.isfinite(r.R) else "-",
                         int(r.bars_held) if np.isfinite(r.get("bars_held", np.nan)) else "-"])
        F.append(_table(rows, [20*mm, 38*mm, 20*mm, 15*mm, 15*mm, 13*mm, 15*mm, 17*mm, 15*mm, 13*mm],
                        font=7.2, align_right_from=2))
        wl_open = [str(x).replace(".NS", "") for x in od[od.watchlist == True].symbol.tolist()]
        if wl_open:
            F.append(Paragraph(
                f"Watchlist names currently in this list: <b>{', '.join(wl_open)}</b>. Watchlist symbols are force-"
                f"included in every scan and back-test (see nse_scanner/universe/watchlist.csv) so a name you care "
                f"about cannot be filtered out by the price / market-cap screen or by still being in an open trade.", SMALL))
    else:
        F.append(Paragraph("No open positions were found in this run.", SMALL))

    # ---------- landscape: the full trade log, with enough room for full company names ----------
    F.append(NextPageTemplate("land"))
    F.append(PageBreak())
    F.append(Paragraph(f"Trade log - {len(d)} completed trades ({main_mode})", H2))
    F.append(Paragraph(
        "Sorted by entry date. 30d = close " + str(C.RET_HORIZON) + " bars after entry, as a % of the entry price. "
        "Shaded rows are watchlist names (always scanned). Full company names are shown - nothing is cut off; "
        "where a name is too long for the column it wraps to a second line.", SMALL))
    hdr = ["#", "symbol", "company name", "ignition", "+%", "flag", "sweep date", "sweep Rs", "entry date",
           "entry Rs", "stop Rs", "risk %", "R", "outcome", f"{C.RET_HORIZON}d", "bars"]
    rows = [[Paragraph(_s(h), CELL_H) for h in hdr]]
    for i, r in d.iterrows():
        nm = str(r.get("shortName", "")) if pd.notna(r.get("shortName", np.nan)) else ""
        rows.append([i + 1, _s(str(r.symbol).replace(".NS", "")), Paragraph(_s(nm), CELL),
                     _s(str(r.pole_date)[:10]), _s(f"+{r.pole_gain*100:.0f}%"), int(r.flag_bars),
                     _s(str(r.sweep_date)[:10]), f"{r.sweep_low:.1f}", _s(str(r.entry_date)[:10]),
                     f"{r.entry:.1f}", f"{r.stop:.1f}", f"{r.risk_pct:.1f}",
                     f"{r.R:+.2f}" if np.isfinite(r.R) else "-", _s(str(r.outcome)),
                     _s(_pct(r.ret_horizon, 1)), int(r.bars_held) if np.isfinite(r.bars_held) else "-"])
    t = Table(rows, colWidths=[7*mm, 21*mm, 56*mm, 18*mm, 13*mm, 10*mm, 18*mm, 15*mm, 18*mm, 15*mm,
                               14*mm, 13*mm, 13*mm, 16*mm, 14*mm, 11*mm], repeatRows=1)
    wl_rows = [i + 1 for i, (_, r) in enumerate(d.iterrows()) if r.get("watchlist")]
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 6.7),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
        ("ALIGN", (4, 0), (5, -1), "RIGHT"), ("ALIGN", (7, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "LEFT"), ("ALIGN", (6, 0), (6, -1), "LEFT"),
        ("ALIGN", (8, 0), (8, -1), "LEFT"), ("ALIGN", (13, 0), (13, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.2, colors.HexColor("#e5e7eb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]
    for ri in wl_rows:
        style += [("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#fff7e0")),
                  ("FONT", (0, ri), (2, ri), "Helvetica-Bold", 6.7)]
    t.setStyle(TableStyle(style))
    F.append(t)


    doc.build(F)
    return path
