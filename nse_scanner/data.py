"""
data.py — universe handling, resilient batched downloads, market-cap cache.

yfinance from cloud IPs is rate-limited and occasionally flaky, so every download
is (a) batched, (b) retried with exponential back-off, (c) validated before use.
"""
from __future__ import annotations
import json, os, time, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

from . import config as C


# ─────────────────────────────── universe ───────────────────────────────
def refresh_universe(path: str = None) -> pd.DataFrame:
    """Rebuild the NSE universe + market caps from Yahoo's screener (exchange = NSI)."""
    path = path or C.UNIVERSE_CSV
    from yfinance import EquityQuery as EQ
    rows, off = [], 0
    while off < 3000:
        q = EQ("and", [EQ("eq", ["exchange", "NSI"]), EQ("gt", ["intradaymarketcap", 100_000_000])])
        try:
            res = yf.screen(q, size=250, offset=off, sortField="intradaymarketcap", sortAsc=False)
        except Exception as e:
            print(f"  [universe] screener stopped at offset {off}: {type(e).__name__} {e}")
            break
        qs = res.get("quotes", [])
        if not qs:
            break
        rows += qs
        off += 250
        if len(qs) < 250:
            break
        time.sleep(0.3)
    if not rows:
        raise RuntimeError("screener returned nothing — keeping the existing universe file")
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
    keep = [c for c in ["symbol", "shortName", "longName", "marketCap", "regularMarketPrice", "currency"] if c in df.columns]
    df = df[keep].copy()
    df["mcap_cr"] = (df.get("marketCap", np.nan) / 1e7).round(1)
    df = df[df.symbol.astype(str).str.endswith(".NS")].reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_watchlist(path: str = None) -> list[str]:
    """Symbols that must ALWAYS be scanned and back-tested, bypassing the price / mcap filters."""
    path = path or C.WATCHLIST_CSV
    if not os.path.exists(path):
        return []
    try:
        w = pd.read_csv(path)
        syms = [str(x).strip() for x in w.iloc[:, 0].dropna().tolist() if str(x).strip()]
        return [x if x.endswith(".NS") else x + ".NS" for x in syms]
    except Exception as e:
        print(f"  [watchlist] unreadable: {type(e).__name__} {e}")
        return []


def load_universe(limit: int = 0, apply_filters: bool = True) -> pd.DataFrame:
    """Load the universe csv, apply the price/mcap filters, then force-include the watchlist."""
    path = C.UNIVERSE_CSV
    if not os.path.exists(path):
        print("  [universe] file missing → refreshing from screener")
        refresh_universe(path)
    df = pd.read_csv(path)
    if "mcap_cr" not in df.columns and "marketCap" in df.columns:
        df["mcap_cr"] = df.marketCap / 1e7
    df = df.dropna(subset=["symbol"]).reset_index(drop=True)
    df["watchlist"] = False
    if apply_filters:
        n0 = len(df)
        keep = (df.regularMarketPrice > C.MIN_PRICE) & (df.mcap_cr >= C.MIN_MCAP_CR)
        df = df[keep].reset_index(drop=True)
        print(f"  [universe] filters price>₹{C.MIN_PRICE:.0f} & mcap>=₹{C.MIN_MCAP_CR:.0f}cr : "
              f"{n0} → {len(df)} names")
    wl = load_watchlist()
    if wl:
        have = set(df.symbol)
        missing = [x for x in wl if x not in have]
        if missing:
            # pull a row for each missing watchlist symbol from the raw universe file (or make a stub)
            raw = pd.read_csv(path)
            raw["mcap_cr"] = raw.get("mcap_cr", raw.get("marketCap", np.nan) / 1e7)
            add = raw[raw.symbol.isin(missing)].copy()
            for x in missing:
                if x not in set(add.symbol):
                    add = pd.concat([add, pd.DataFrame([dict(symbol=x, shortName=x, mcap_cr=np.nan,
                                                             regularMarketPrice=np.nan)])], ignore_index=True)
            add["watchlist"] = True
            df = pd.concat([df, add], ignore_index=True)
        df.loc[df.symbol.isin(wl), "watchlist"] = True
        print(f"  [universe] + watchlist force-included: "
              f"{', '.join(s.replace('.NS','') for s in wl)}  (total {len(df)})")
    if limit:
        forced = df[df.watchlist == True]
        head = df[df.watchlist != True].head(max(0, limit - len(forced)))
        df = pd.concat([head, forced], ignore_index=True)
    return df


# ─────────────────────────────── downloads ───────────────────────────────
def _clean(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 60:
        return None
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adj_close": "adj_close"})
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def download_daily(symbols: list[str], period: str = None, start: str = None,
                   batch: int = None, label: str = "") -> dict[str, pd.DataFrame]:
    """Batched daily download → {symbol: DataFrame}. Retries, pauses, and skips bad tickers."""
    period = period or C.DAILY_PERIOD
    batch = batch or C.BATCH_SIZE
    out: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for bi in range(0, total, batch):
        chunk = symbols[bi:bi + batch]
        kw = dict(group_by="ticker", threads=True, progress=False, auto_adjust=False, actions=False)
        df = None
        for attempt in range(C.DOWNLOAD_RETRIES):
            try:
                df = (yf.download(chunk, start=start, **kw) if start
                      else yf.download(chunk, period=period, **kw))
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print(f"  [{label}] batch {bi//batch} attempt {attempt+1} failed: {type(e).__name__} {e}")
            time.sleep(2 * (attempt + 1))
        if df is None or df.empty:
            print(f"  [{label}] batch {bi//batch} returned nothing — skipping {len(chunk)} tickers")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            for s in chunk:
                if s in set(lvl0):
                    c = _clean(df[s], s)
                    if c is not None:
                        out[s] = c
        else:  # single ticker
            c = _clean(df, chunk[0])
            if c is not None:
                out[chunk[0]] = c
        done = min(bi + batch, total)
        print(f"  [{label}] {done}/{total} tickers fetched ({len(out)} usable)", flush=True)
    return out


def download_intraday(symbols: list[str], interval: str = None, period: str = "5d") -> dict[str, pd.DataFrame]:
    """Intraday bars for a SHORT list (used to read today's live price)."""
    interval = interval or C.INTRADAY_INTERVAL
    if not symbols:
        return {}
    out = {}
    for bi in range(0, len(symbols), 100):
        chunk = symbols[bi:bi + 100]
        df = None
        for attempt in range(C.DOWNLOAD_RETRIES):
            try:
                df = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                                 threads=True, progress=False, auto_adjust=False, actions=False)
                if df is not None and not df.empty:
                    break
            except Exception as e:
                print(f"  [intraday] attempt {attempt+1} failed: {type(e).__name__} {e}")
            time.sleep(2 * (attempt + 1))
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for s in chunk:
                if s in set(df.columns.get_level_values(0)):
                    c = _clean(df[s], s)
                    if c is not None:
                        out[s] = c
        else:
            c = _clean(df, chunk[0])
            if c is not None:
                out[chunk[0]] = c
    return out


def market_regime(refresh: bool = False) -> dict:
    """Is NIFTY 50 above its 200-DMA?  Used by the REGIME_FILTER gate."""
    try:
        df = yf.download("^NSEI", period="2y", interval="1d", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return {"ok": False, "bull": None}
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        ma = df["close"].rolling(200).mean()
        return {"ok": True, "bull": bool(df["close"].iloc[-1] > ma.iloc[-1]),
                "close": float(df["close"].iloc[-1]), "ma200": float(ma.iloc[-1]),
                "pct_vs_ma": round(float(df["close"].iloc[-1] / ma.iloc[-1] - 1) * 100, 2)}
    except Exception as e:
        print(f"  [regime] unavailable: {type(e).__name__} {e}")
        return {"ok": False, "bull": None}


# ─────────────────────────────── misc helpers ───────────────────────────────
def now_ist() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))


def is_market_open(t: dt.datetime = None) -> bool:
    t = t or now_ist()
    if t.weekday() >= 5:
        return False
    o = t.replace(hour=C.MARKET_OPEN[0], minute=C.MARKET_OPEN[1], second=0, microsecond=0)
    c = t.replace(hour=C.MARKET_CLOSE[0], minute=C.MARKET_CLOSE[1], second=0, microsecond=0)
    return o <= t <= c


def is_confirmed_run(t: dt.datetime = None) -> bool:
    t = t or now_ist()
    return t.hour * 60 + t.minute >= C.CONFIRMED_AFTER[0] * 60 + C.CONFIRMED_AFTER[1]


def load_state(path: str = None) -> dict:
    path = path or C.STATE_JSON
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {"alerts": {}, "digest_last": None}


def save_state(state: dict, path: str = None) -> None:
    path = path or C.STATE_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(state, open(path, "w"), indent=1, default=str)
