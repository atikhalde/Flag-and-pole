"""
telegram_bot.py — minimal, dependency-free Telegram sender (HTML parse mode).
"""
from __future__ import annotations
import html, time
import requests

from . import config as C

API = "https://api.telegram.org/bot{token}/sendMessage"


def _esc(s) -> str:
    return html.escape(str(s), quote=False)


def send(text: str, token: str = None, chat: str = None, disable_preview: bool = True,
         retries: int = 3) -> bool:
    token = token or C.TG_TOKEN
    chat = chat or C.TG_CHAT
    if not token or not chat:
        print("  [telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printing instead:\n" + text)
        return False
    for a in range(retries):
        try:
            r = requests.post(API.format(token=token),
                              data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                                    "disable_web_page_preview": disable_preview},
                              timeout=20)
            if r.status_code == 200:
                return True
            print(f"  [telegram] {r.status_code}: {r.text[:200]}")
            time.sleep(2 * (a + 1))
        except Exception as e:
            print(f"  [telegram] attempt {a+1} failed: {type(e).__name__} {e}")
            time.sleep(2 * (a + 1))
    return False


def alert_card(s: dict, kind: str) -> str:
    """Full card for a BUY signal.  kind = 'PROVISIONAL' | 'CONFIRMED'"""
    icon = "🟡" if kind == "PROVISIONAL" else "🟢"
    sym = _esc(s["symbol"])
    nx = s.get("shortName") or sym
    lines = [
        f"{icon} <b>{_esc(nx)}</b>  <code>{sym}</code>",
        f"<b>flag-rail reclaim — {kind}</b>",
        "",
        f"Entry      <b>₹{s['entry']}</b>   ({'current price' if kind=='PROVISIONAL' else 'close'})",
        f"Stop       ₹{s['stop']}   (risk {s['risk_pct']}%)",
        f"Target     ₹{s['target']}   pole high",
        f"Reward     <b>{s['reward_R']}R</b>",
        "",
        f"Pole       {s['pole_date']}   ₹{s['pole_low']} → ₹{s['pole_high']}  (+{round(s['pole_gain']*100,1)}%)",
        f"Flag       {s['flag_bars']} bars   rail ₹{s['rail']}",
        f"Sweep      {s['sweep_date']}   low ₹{s['sweep_low']}  ({s['sweep_below_pct']}% under rail)",
        (f"Shakeout   {s.get('sweep_red')}/{s.get('sweep_bars')} bars red  ·  vol "
         f"{s.get('sweep_vol_x20')}x average  ·  {s.get('sweep_vol_vs_drift')}x the drift  ·  "
         f"worst day {s.get('sweep_leg_pct')}%" if s.get("sweep_vol_x20") is not None else ""),
        f"Trigger    {s['entry_date']}   vol {s['vol_x20']}× 20-day avg",
        "",
        f"<i>mcap ₹{s.get('mcap_cr','?')} cr · 20d turnover ₹{s.get('turnover_cr','?')} cr</i>",
        f'<a href="{C.CHART_LINK.format(sym=sym.replace(".NS",""))}">open chart on TradingView</a>',
    ]
    if kind == "PROVISIONAL":
        lines += ["", "⚠️ <i>Intraday print — valid only if it CLOSES above ₹%s. A confirmed alert follows after 15:30 IST.</i>" % s["rail"]]
    return "\n".join(lines)


def status_card(s: dict) -> str:
    st = s["status"]
    # an open position must show its levels and where it stands - that is the whole point of the digest
    if st == "IN TRADE" and s.get("entry"):
        risk = (s.get("entry") or 0) - (s.get("stop") or 0)
        r_now = ((s.get("last_close") or 0) - (s.get("entry") or 0)) / risk if risk > 0 else None
        return "\n".join([
            f"<b>{_esc(s.get('shortName') or s['symbol'])}</b>  <code>{_esc(s['symbol'])}</code>",
            "\U0001F535 in trade - entry already fired",
            f"entry \u20b9{s['entry']}  ({s.get('entry_date')})",
            f"stop  \u20b9{s['stop']}  (risk {s.get('risk_pct')}%)",
            (f"last  \u20b9{s.get('last_close')}  \u2192 {r_now:+.2f}R" if r_now is not None else
             (f"last  \u20b9{s.get('last_close')}" if s.get("last_close") else "")),
            f"target \u20b9{s.get('target')}  (pole high)",
            (f"{int(s['bars_since_entry'])} bars in" if s.get("bars_since_entry") is not None else ""),
        ]).strip()
    head = {"SWEPT": "🟠 swept — waiting for the reclaim", "FLAG_READY": "🔵 flag built — waiting for the sweep",
            "COILING": "⚪ coiling", "BUY": "🟢 entry fired", "IN TRADE": "🔵 in trade (entry already fired)"}.get(st, st)
    return "\n".join([
        f"<b>{_esc(s['symbol'])}</b>",
        head,
        f"pole {s['pole_date']}  ₹{s['pole_low']} → ₹{s['pole_high']} (+{round(s['pole_gain']*100,1)}%)",
        (f"rail ₹{s['rail']} · flag {s['flag_bars']} bars" if s.get("rail") else ""),
        (f"sweep {s['sweep_date']} low ₹{s['sweep_low']}" if s.get("sweep_date") else ""),
        (f"last ₹{s.get('last_close')}  ·  dist to rail {s.get('dist_to_rail_pct')}%" if s.get("dist_to_rail_pct") is not None else ""),
    ])


def digest(items: list[dict], kind: str = "post-close", gated: list[dict] = None,
           fresh: int = None) -> str:
    """Post-close digest: open positions, fresh triggers and the names filtered out today."""
    gated = gated or []
    n_pos = sum(1 for s in items if s.get("status") == "IN TRADE")
    head = ["\U0001F4CA <b>NSE flag scanner - " + kind + "</b>"]
    if fresh is not None:
        head.append(f"{fresh} new trigger(s) / {len(items)} setup(s) on the board "
                    f"({n_pos} position(s) already running)")
    else:
        head.append(f"{len(items)} live setup(s)")
    head.append("")
    out = ["\n".join(head)]
    if not items:
        out.append("No live setups in the universe today.")
        out.append("")
    for s in items[:40]:
        out.append(status_card(s))
        out.append("")
    if gated:
        out.append("<b>Filtered out today</b> (these would have alerted with QUALITY_PROFILE=off)")
        for g in gated[:8]:
            out.append("\u2022 <code>" + _esc(g.get("symbol")) + "</code> - " + _esc(g.get("skip_reason")))
        out.append("")
    if len(items) > 40:
        out.append(f"...and {len(items)-40} more (full list in the Actions run summary).")
    return "\n".join(out)[:4000]