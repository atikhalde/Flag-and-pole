"""
config.py — every tunable in one place.
"""
from __future__ import annotations
import os

# ──────────────────────────────── pattern definition ────────────────────────────────
POLE_MODE          = os.getenv("POLE_MODE", "locked")     # "locked" = +9.8% & close near high & 3x vol | "breakout" = looser
POLE_RET_MIN       = float(os.getenv("POLE_RET_MIN", 0.098))   # ignition bar return
POLE_RET_ALT       = float(os.getenv("POLE_RET_ALT", 0.06))    # used when POLE_MODE="breakout"
POLE_CLOSE_NEAR_HI = 0.98        # close must be >= 98% of that bar's high
VOL_SURGE          = float(os.getenv("VOL_SURGE", 3.0))        # vs prior 20-day average volume
IMPULSE_WINDOW     = 10          # bars after the ignition to find the pole high
BASE_LOOKBACK      = 10          # bars before the ignition to find the pole low
GIVE_MIN           = float(os.getenv("GIVE_MIN", 0.50))        # flag must retrace >=50% of the pole
MIN_FLAG_BARS      = int(os.getenv("MIN_FLAG_BARS", 5))
MAX_DRIFT_BARS     = int(os.getenv("MAX_DRIFT_BARS", 60))       # cap on the drift/flag length
BREAK_ATR          = float(os.getenv("BREAK_ATR", 0.5))          # floor break must exceed this many ATRs …
BREAK_MIN_PCT      = float(os.getenv("BREAK_MIN_PCT", 1.5))      # … or this % of price, whichever is larger
#   + the break must NOT be reclaimed on the very next bar (that separates a genuine
#     shakeout from a wick inside the drift — the two exemplars differ only by this)
BASE_INTACT_TOL    = 0.995       # pole low may not be broken harder than this

# trigger
TRIGGER_MODE       = os.getenv("TRIGGER_MODE", "first_bullish")
#   "first_bullish" = buy the first bullish candle after the shakeout  ← DEFAULT, best measured
#   "rail_reclaim"  = buy the first close back above the drift floor AND above the prior 5-bar high
#   Full universe, 2016-2026, with both quality gates on:
#     first_bullish : 329 trades | +0.369R | median 30-bar +5.13% | median risk  8.0% | payoff 2.32:1
#     rail_reclaim  : 175 trades | +0.301R | median 30-bar +3.75% | median risk 16.1% | payoff 1.06:1
#   first_bullish fires earlier (smaller stop, bigger payoff, lower hit-rate); rail_reclaim waits
#   for confirmation (higher hit-rate, but the stop sits ~16% away — the R-multiple suffers).
BULLISH_CLOSE_POS  = float(os.getenv("BULLISH_CLOSE_POS", 0.6)) # for first_bullish: close in top 40% of the bar's range
SWEEP_MAX_BARS     = int(os.getenv("SWEEP_MAX_BARS", 40))       # sweep must happen within N bars of the flag end
MAX_SETUP_BARS     = int(os.getenv("MAX_SETUP_BARS", 90))       # whole pole->now window

# risk / trade management
ATR_BUF            = float(os.getenv("ATR_BUF", 0.5))           # stop = sweep low − ATR_BUF × ATR
MAX_HOLD_BARS      = int(os.getenv("MAX_HOLD_BARS", 60))
RET_HORIZON        = int(os.getenv("RET_HORIZON", 30))          # headline "return after 30 days"
MIN_REWARD_R       = float(os.getenv("MIN_REWARD_R", 0.5))      # live alerts only: skip if reward/pole-target < this

# ──────────────────── validated quality gates (see backtest_report.pdf) ────────────────────
# Across 2,156 historical trades the pattern unfiltered is worth ~0 (median 30-bar return
# +0.01% vs a +1.03% baseline). These two gates isolate the only configuration with a stable
# edge: deep absorption + a weak tape (+0.40R, 61% win, +5.6% median 30-bar return, n=138).
MIN_SWEEP_DEPTH_PCT = float(os.getenv("MIN_SWEEP_DEPTH_PCT", 3.0))   # shakeout must undercut the drift floor by >= this %
#   evidence (corrected definition, full 1,275-name universe, 2016-2026): the median shakeout is
#   -8.4% deep, so this gate passes 828 of 838 signals — it is a cheap sanity check, NOT the edge.
#   Depth helps only mildly: <=-5% gives median 30-bar +1.4% vs +0.7% for all signals.
#   The gate that actually matters is REGIME_FILTER below (0.11R -> 0.28R on its own).
REGIME_FILTER       = os.getenv("REGIME_FILTER", "off")              # "weak_market" = require NIFTY < 200DMA | "off"
#   HONEST NOTE: in a full-sample test the weak-market split looked strong (+0.28R vs +0.06R) and it
#   does improve the MEDIAN outcome.  But on a time-split (train <2022 / test >=2022) the average-R
#   edge does NOT survive: above-200DMA +0.18 train -> +0.16 test, below-200DMA +0.73 train -> +0.16 test.
#   It is therefore OFF by default.  Turn it on if you prefer a higher hit-rate median over expectancy.

# ── the false-signal filter: this is the one that separates real reclaims from bull traps ──
EXCLUDE_VOL_SPIKE_X = float(os.getenv("EXCLUDE_VOL_SPIKE_X", 3.0))
#   entry-bar volume > 3x its 20-day average:
#     66 trades | avgR -0.048 | median 30-bar -2.19% | train -0.03 / test -0.07  (stable negative)
#   i.e. a violent-volume reclaim bar is EXHAUSTION, not absorption. The good entries are quiet:
#     volume 0.8-1.2x -> +0.51R | volume >3x -> -0.05R

# ── quality profile, validated out-of-sample (train <2022 / test >=2022) ──
QUALITY_PROFILE      = os.getenv("QUALITY_PROFILE", "volume")  # "off" | "balanced" | "strict" | "volume"
QUALITY_MIN_DRIFT    = int(os.getenv("QUALITY_MIN_DRIFT", 25))  # balanced=15, strict=25

# ── THE SHAKEOUT SIGNATURE (profile "volume") ────────────────────────────────────────────
# This is what the two source charts actually show and what the detector was ignoring: the
# shakeout is a VOLUME EVENT with a RED, FALLING structure, not just a dip below the floor.
#   NIACL      2026-06: break bar 2.00x average, 7.7x the drift's own volume, 2 of 3 bars red, worst day -4.9%
#   LAMBODHARA 2026-09: collapse bar 1.14x average, 6.3x the drift, 4 of 5 red (5 of 6 on the daily leg), -9.6%
# Measured over the whole universe (1,326 trades), train <2022 / test >=2022:
#   leg volume < 1.0x average  ->  765 trades | +0.11R | train +0.17 / test +0.06   (no edge at all)
#   leg volume >= 1.0x average ->  561 trades | +0.35R | train +0.43 / test +0.27
#   full signature             ->  178 trades | +0.54R | train +0.46 / test +0.59   <- most stable gate found
SWEEP_MIN_VOL_X20     = float(os.getenv("SWEEP_MIN_VOL_X20", 1.0))     # leg must trade >= its 20-day average
SWEEP_MIN_VOL_VS_DRIFT = float(os.getenv("SWEEP_MIN_VOL_VS_DRIFT", 3.0))  # ...and >= 3x the quiet drift's volume
SWEEP_MIN_DROP_PCT    = float(os.getenv("SWEEP_MIN_DROP_PCT", 4.0))     # a red bar of at least this much
SWEEP_MIN_RED_FRAC    = float(os.getenv("SWEEP_MIN_RED_FRAC", 0.5))     # more than half the leg must be red
QUALITY_MIN_SWEEP    = float(os.getenv("QUALITY_MIN_SWEEP", 8.0))  # strict: shakeout >= 8% below the rail
#   strict   : volume<=2x + drift>=25 + shakeout<=-8%  -> 149 trades | +0.51R | 42% win | median +2.9%
#              train +0.54 / test +0.49   <- best expectancy, but the WHOLE universe only yields
#              ~16 signals a year, i.e. roughly one alert every three weeks
#   balanced : volume<=2x + drift>=15                 -> 677 trades | +0.33R | train +0.46 / test +0.21
#              <- the shipped default: ~1 alert a week, still keeps the volume cap that removes the traps
#   Note both keep the >3x volume rejection (EXCLUDE_VOL_SPIKE_X): that is the false-signal filter.
#   Override per run from the Actions tab: workflow input "profile" sets this env var.

# ──────────────────────────────── universe filters ────────────────────────────────
MIN_PRICE          = float(os.getenv("MIN_PRICE", 100))         # ₹
MIN_MCAP_CR        = float(os.getenv("MIN_MCAP_CR", 1000))      # ₹ crore
MIN_TURNOVER_CR    = float(os.getenv("MIN_TURNOVER_CR", 0.5))   # 20-day avg traded value, ₹ crore
MAX_UNIVERSE       = int(os.getenv("MAX_UNIVERSE", 0))          # 0 = no cap (useful for testing)

# ──────────────────────────────── data plumbing ────────────────────────────────
DAILY_PERIOD       = os.getenv("DAILY_PERIOD", "1y")     # live scanner lookback
BACKTEST_START     = os.getenv("BACKTEST_START", "2016-01-01")
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", 200))
DOWNLOAD_RETRIES   = int(os.getenv("DOWNLOAD_RETRIES", 3))
INTRADAY_INTERVAL  = os.getenv("INTRADAY_INTERVAL", "15m")
UNIVERSE_CSV       = os.getenv("UNIVERSE_CSV", "nse_scanner/universe/nse_universe.csv")
WATCHLIST_CSV      = os.getenv("WATCHLIST_CSV", "nse_scanner/universe/watchlist.csv")
STATE_JSON         = os.getenv("STATE_JSON", "nse_scanner/state/live_state.json")

# ──────────────────────────────── telegram ────────────────────────────────
TG_TOKEN           = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT            = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────── misc ────────────────────────────────
IST                = "Asia/Kolkata"
MARKET_OPEN        = (9, 15)     # IST
MARKET_CLOSE       = (15, 30)
CONFIRMED_AFTER    = (15, 35)    # runs at/after this IST time are treated as the "confirmed close" run
CHART_LINK         = "https://www.tradingview.com/chart/?symbol=NSE:{sym}"
