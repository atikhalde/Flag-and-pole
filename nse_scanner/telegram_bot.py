"""
telegram_bot.py — minimal, dependency-free Telegram sender (HTML parse mode).
"""
from __future__ import annotations
import html, time
import requests

from . import config as C

API = "https://api.telegram.org/bot{token}/sendMessage"
# Every message this process delivered, and every one it failed to deliver.  The run summary prints
# them, so "did it alert?" is answered from the Actions page instead of by assumption.
SENT: list = []
FAILED: list = []
BOT_USERNAME: str = ""   # filled in by the first getMe (whoami), so the summary can name the bot

API_ME = "https://api.telegram.org/bot{token}/getMe"
API_CHAT = "https://api.telegram.org/bot{token}/getChat"
API_UPD = "https://api.telegram.org/bot{token}/getUpdates"


def _where(chat: dict) -> str:
    """Human-readable name of the chat a message landed in."""
    if not isinstance(chat, dict):
        return "?"
    return (chat.get("title") or chat.get("username") or
            " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x) or "?")


def _mask(cid) -> str:
    """Chat ids are printed to a PUBLIC Actions log - never in full."""
    cid = str(cid)
    return cid if len(cid) <= 6 else f"{cid[:6]}{'*' * (len(cid) - 8)}{cid[-2:]}" if len(cid) > 8 else cid


def whoami(token: str = None, chat: str = None, updates: int = 20) -> dict:
    """Report WHICH bot and WHICH chat the alerts go to - so 'no alert arrived' can be answered
    from the log instead of guessed.  Prints no secret: the token is never echoed and chat ids are
    masked."""
    token = token or C.TG_TOKEN
    chat = chat or C.TG_CHAT
    out = {"bot": None, "chat": None, "updates": []}
    if not token:
        print("  [diag] TELEGRAM_BOT_TOKEN is NOT set - nothing can be sent. Add the secret.")
        return out
    try:
        me = requests.get(API_ME.format(token=token), timeout=20).json()
        res = me.get("result") or {}
        out["bot"] = res
        if res:
            globals()["BOT_USERNAME"] = res.get("username") or ""
            print(f"  [diag] bot: @{res.get('username')} ({res.get('first_name')!r}, id {_mask(res.get('id'))})")
        else:
            print(f"  [diag] getMe FAILED: {str(me)[:180]}  <-- the token is wrong or revoked")
    except Exception as e:
        print(f"  [diag] getMe error: {type(e).__name__} {e}")
    if not chat:
        print("  [diag] TELEGRAM_CHAT_ID is NOT set - the sender prints messages to the log instead.")
    else:
        try:
            ch = requests.get(API_CHAT.format(token=token), params={"chat_id": chat}, timeout=20).json()
            res = ch.get("result") or {}
            out["chat"] = res
            if res:
                print(f"  [diag] chat: {_where(res)!r} type={res.get('type')} id={_mask(res.get('id'))}")
            else:
                print(f"  [diag] getChat FAILED for the configured TELEGRAM_CHAT_ID "
                      f"({_mask(chat)}): {str(ch)[:180]}  <-- the chat id is wrong")
        except Exception as e:
            print(f"  [diag] getChat error: {type(e).__name__} {e}")
    try:
        up = requests.get(API_UPD.format(token=token), params={"limit": updates}, timeout=20).json()
        seen, rows = set(), []
        for u in (up.get("result") or []):
            m = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
            c = m.get("chat") or {}
            if c.get("id") is None or c["id"] in seen:
                continue
            seen.add(c["id"])
            rows.append((c.get("id"), _where(c), c.get("type"),
                         (m.get("text") or "")[:24]))
        out["updates"] = rows
        if rows:
            print(f"  [diag] {len(rows)} chat(s) have messaged this bot recently "
                  f"(set TELEGRAM_CHAT_ID to the right one):")
            for cid, name, typ, txt in rows:
                print(f"           id={_mask(cid)}  {name!r}  type={typ}  last: {txt!r}")
        else:
            print("  [diag] no recent messages TO this bot - send it a 'hi' from your phone, re-run "
                  "this, and the chat id appears here")
    except Exception as e:
        print(f"  [diag] getUpdates error: {type(e).__name__} {e}")
    return out


def _esc(s) -> str:
    return html.escape(str(s), quote=False)


def send(text: str, token: str = None, chat: str = None, disable_preview: bool = True,
         retries: int = 3) -> bool:
    token = token or C.TG_TOKEN
    chat = chat or C.TG_CHAT
    if not token or not chat:
        print("  [telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — printing instead:\n" + text)
        return False
    reason = ""
    for a in range(retries):
        try:
            r = requests.post(API.format(token=token),
                              data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                                    "disable_web_page_preview": disable_preview},
                              timeout=20)
            if r.status_code == 200:
                try:
                    m = (r.json().get("result") or {})
                    c = m.get("chat") or {}
                    rec = {"where": _where(c), "type": c.get("type"), "chat": _mask(c.get("id")),
                           "message_id": m.get("message_id"), "chars": len(text)}
                    SENT.append(rec)
                    print(f"  [telegram] delivered -> {rec['where']!r} (chat {rec['chat']}, "
                          f"type {rec['type']}, message_id {rec['message_id']}, {rec['chars']} chars)")
                except Exception:
                    SENT.append({"where": "?", "type": "?", "chat": _mask(chat), "message_id": None,
                                 "chars": len(text)})
                return True
            reason = f"HTTP {r.status_code} {r.text[:160]}"
            print(f"  [telegram] {reason}")
            time.sleep(2 * (a + 1))
        except Exception as e:
            reason = f"{type(e).__name__} {e}"
            print(f"  [telegram] attempt {a+1} failed: {reason}")
            time.sleep(2 * (a + 1))
    FAILED.append({"chat": _mask(chat), "reason": reason or "no response"})
    print(f"  [telegram] !!! NOT DELIVERED after {retries} attempt(s) to chat {_mask(chat)}: "
          f"{FAILED[-1]['reason']} - the run summary says so too, and the next run retries.")
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
    # a fired entry MUST carry its date: without it the card reads as "fired today" even when
    # the candle it belongs to is days old (the LAMBODHARA 2026-09-23 entry reported as today)
    if st == "BUY":
        bars = s.get("bars_since_entry")
        age = ("" if bars is None or int(bars) <= 0
               else f"  ({int(bars)} bar{'s' if int(bars) != 1 else ''} ago)")
        risk = (s.get("entry") or 0) - (s.get("stop") or 0)
        r_now = (((s.get("last_close") or 0) - (s.get("entry") or 0)) / risk) if risk > 0 else None
        return "\n".join([
            f"<b>{_esc(s.get('shortName') or s['symbol'])}</b>  <code>{_esc(s['symbol'])}</code>",
            f"🟢 entry fired <b>{_esc(str(s.get('entry_date')))}</b>{age}",
            f"entry ₹{s.get('entry')}   stop ₹{s.get('stop')}  (risk {s.get('risk_pct')}%)",
            (f"last ₹{s.get('last_close')}  → {r_now:+.2f}R" if r_now is not None
             else (f"last ₹{s.get('last_close')}" if s.get("last_close") else "")),
            f"target ₹{s.get('target')}  (pole high)",
            f"pole {s['pole_date']}  ₹{s['pole_low']} → ₹{s['pole_high']} (+{round(s['pole_gain']*100,1)}%)",
            (f"rail ₹{s['rail']} · flag {s['flag_bars']} bars" if s.get("rail") else ""),
            (f"sweep {s['sweep_date']} low ₹{s['sweep_low']}" if s.get("sweep_date") else ""),
        ]).strip()
    head = {"SWEPT": "🟠 swept — waiting for the reclaim", "FLAG_READY": "🔵 flag built — waiting for the sweep",
            "COILING": "⚪ coiling", "IN TRADE": "🔵 in trade (entry already fired)",
            "REBASING": "🔄 flush failed — waiting for the next bullish candle"}.get(st, st)
    return "\n".join([
        f"<b>{_esc(s['symbol'])}</b>",
        head,
        f"pole {s['pole_date']}  ₹{s['pole_low']} → ₹{s['pole_high']} (+{round(s['pole_gain']*100,1)}%)",
        (f"rail ₹{s['rail']} · flag {s['flag_bars']} bars" if s.get("rail") else ""),
        (f"sweep {s['sweep_date']} low ₹{s['sweep_low']}" if s.get("sweep_date") else ""),
        (f"last ₹{s.get('last_close')}  ·  dist to rail {s.get('dist_to_rail_pct')}%" if s.get("dist_to_rail_pct") is not None else ""),
    ])


def digest(items: list[dict], kind: str = "post-close", gated: list[dict] = None,
           fresh: int = None, newly_closed: list[dict] = None,
           watch: list[dict] = None, data_through: str = None) -> str:
    """Post-close digest: open positions, fresh triggers and the names filtered out today."""
    gated = gated or []
    newly_closed = newly_closed or []
    watch = watch or []
    n_pos = sum(1 for s in items if s.get("status") == "IN TRADE")
    head = ["\U0001F4CA <b>NSE flag scanner - " + kind + "</b>"]
    if fresh is not None:
        head.append(f"{fresh} new trigger(s) / {len(items)} setup(s) on the board "
                    f"({n_pos} position(s) already running)")
    else:
        head.append(f"{len(items)} live setup(s)")
    # the bar the scan is computed on: if the vendor lags, you need to know that when you read
    # a signal, not afterwards
    if data_through:
        head.append(f"<i>daily bars through {data_through}</i>")
    head.append("")
    out = ["\n".join(head)]
    if not items:
        out.append("No live setups in the universe today.")
        out.append("")
    for s in items[:40]:
        out.append(status_card(s))
        out.append("")
    if watch:
        out.append("<b>Watchlist</b> \u2014 always shown, below the gate, so no alert was sent")
        for w2 in watch[:8]:
            risk = (w2.get("entry") or 0) - (w2.get("stop") or 0)
            r_now = (((w2.get("last_close") or 0) - (w2.get("entry") or 0)) / risk) if risk > 0 else None
            line = ("\u2022 <code>" + _esc(w2.get("symbol")) + "</code> entry <b>" +
                    _esc(str(w2.get("entry_date"))) + "</b> @ " + _esc(str(w2.get("entry"))))
            if r_now is not None:
                line += f"  \u2192 {r_now:+.2f}R now"
            line += "\n    gate: " + _esc(w2.get("gate_note"))
            out.append(line)
        out.append("")
    if newly_closed:
        out.append("<b>Closed since the last digest</b> (no longer positions)")
        for c in newly_closed[:12]:
            out.append("\u2022 <code>" + _esc(c.get("symbol")) + "</code> - " + _esc(c.get("pos_reason")))
        out.append("")
    if gated:
        out.append("<b>Filtered out today</b> (these would have alerted with QUALITY_PROFILE=off)")
        for g in gated[:8]:
            out.append("\u2022 <code>" + _esc(g.get("symbol")) + "</code> - " + _esc(g.get("skip_reason")))
        out.append("")
    if len(items) > 40:
        out.append(f"...and {len(items)-40} more (full list in the Actions run summary).")
    return "\n".join(out)[:4000]