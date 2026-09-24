# NSE Pole → Flag → Sweep → Reclaim scanner
### live Telegram alerts from GitHub Actions + a full-universe back-test that writes a PDF

---

## 1. What this does

Two things, from one shared detection engine:

| | **Live scanner** | **Back-test** |
|---|---|---|
| Runs on | GitHub Actions, every 15 min (08:30–16:15 IST, Mon–Fri) | GitHub Actions, on demand (Actions → *Backtest report* → Run workflow) |
| Universe | full NSE list, filtered **price > ₹100** and **market cap ≥ ₹1,000 cr** | same filters, history from 2016 |
| Output | Telegram alert the moment a trigger fires (+ one post-close digest) | `backtest_report.pdf` — every trade, 30-day returns, expectancy, equity curve |

**The pattern** (the rule the scanner fires) — four phases, in this order:

1. **POLE** — a locked/≥+9.8% bar on ≥3× its 20-day average volume, closing near its high, followed by the impulse.
2. **DRIFT (the flag)** — the sloping consolidation that follows the pole high. It ends when a **close breaks the drift's own floor** *and the next bar does not reclaim it* (a poke that is immediately bought back is part of the drift, not a shakeout). **That floor is the rail** — the level drawn on the chart.
3. **SHAKEOUT** — the leg that trades **below** the drift's floor. This is the phase that stops you out. The pole low must survive (base intact).
4. **BUY** — the first **close back above the rail *and* above the prior 5-bar high**. Stop = shakeout low − 0.5 ATR. Target = pole high. Time stop 60 bars.

> **Validated on the two charts this was built from** (run `python3 -m nse_scanner.tests.test_exemplars`):
>
> | | rail (drift floor) | shakeout | reclaim → entry | outcome |
> |---|---|---|---|---|
> | **NIACL** Apr-Jun 2026 | **154.00** | 145.50 (-5.5%) | **4 Jun @151.69** | target hit, **+3.22R** |
> | **LAMBODHARA** Aug-Sep 2026 | **127.37** | 115.57 (-9.3%) | **18 Sep @120.21** | open, **+2.80R** |
>
> **Full NSE universe, 1,276 names (1,275 filtered + 2 watchlist), 2016→2026** (`reports/backtest_report.pdf`):
>
> | configuration | trades | avg R | median 30-bar return | median risk | payoff |
> |---|---|---|---|---|---|
> | every signal (`rail_reclaim`) | 839 | +0.11R | +0.71% | 15.4% | 0.96:1 |
> | every signal (**`first_bullish`** - the live rule) | 1,307 | +0.22R | +1.41% | 7.5% | 2.33:1 |
> | **shipped profile: quiet reclaim, drift >= 25 bars, shakeout >= 8%** | **149** | **+0.51R** | **+2.9%** | **8.0%** | **2.4:1** |
> | shipped profile's train / test halves | 76 / 73 | +0.54R / **+0.49R** | - | - | - |
> | entries whose reclaim bar printed > 3x average volume (rejected) | 66 | **-0.05R** | **-2.2%** | - | - |
> | market baseline (any random bar) | 2,356,998 | - | +1.03% | - | - |
>
> Read that carefully, and note it **corrects an earlier version of this file**. The market-regime gate looked
> like the edge (+0.36R full sample) but it does **not** survive a time split - NIFTY below its 200-DMA earns
> +0.73R before 2022 and +0.16R after, exactly the same as taking every signal. So `REGIME_FILTER` defaults to
> **off**. What does survive is the *quality* of the reclaim candle and the *length* of the drift: the shipped
> profile is the only one that holds ~+0.5R on both halves of the history, and a reclaim bar on >3x average
> volume is the single clearest false-signal marker in the data. `nse_scanner/filter_study.py` reruns that
> whole test and writes `reports/filter_scan.csv`.


---

## 2. Repo layout

```
.github/workflows/
  live_scan.yml          # the 15-minute scanner + manual run button
  backtest_report.yml    # the PDF back-test job
nse_scanner/
  config.py              # ← every threshold lives here
  data.py                # universe + resilient yfinance downloads
  pattern.py             # the detection engine (shared by both jobs)
  live_scanner.py        # live entry point
  backtest.py            # back-test entry point
  pdf_report.py          # PDF builder (reportlab)
  telegram_bot.py        # alert cards + digest
  refresh_universe.py    # rebuilds the NSE list + market caps
  universe/nse_universe.csv   # committed symbol + market-cap snapshot
  state/live_state.json       # alert de-duplication (auto-committed)
  requirements.txt
reports/
  backtest_report.pdf    # produced by the back-test job
  backtest_trades.csv    # completed trades (all statistics use these)
  backtest_open_trades.csv    # positions still running - recorded, never dropped
  backtest_summary.json
  filter_scan.csv / filter_study.json   # output of the filter study
```

**The watchlist** (`nse_scanner/universe/watchlist.csv`) solves a real problem: the price > ₹100 and
market-cap > ₹1,000 cr filters, plus the rule that only *finished* trades were logged, meant the two charts this
whole engine was built from could go missing from the report. NIACL and LAMBODHARA are now force-included in every
scan and every back-test, and a trade that is still open is reported as a position (with live R) instead of being
silently dropped. Add any name you care about to that file, one symbol per line, and it can never be filtered out.

---

## 3. Setup (5 minutes)

1. **Copy this folder structure into your repo** (keep the paths; the workflows expect them).
2. **Add two repository secrets** — Settings → Secrets and variables → Actions → New repository secret:
   - `TELEGRAM_BOT_TOKEN` — from @BotFather
   - `TELEGRAM_CHAT_ID` — your chat/group id (message your bot, then open
     `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `result[].message.chat.id`)
3. **Enable Actions writes** — Settings → Actions → General → Workflow permissions → **Read and write permissions**.
   (The scanner commits `state/live_state.json` back so you never get the same alert twice.)
4. **Test without spam**: Actions → *Live flag scanner* → Run workflow → set `dry_run = true`.
   The alert text appears in the job log instead of Telegram.

> **First scheduled run:** GitHub fires the cron on the next matching slot. Imported/quiet repos have scheduled
> workflows disabled after 60 days of inactivity — the manual Run-workflow button always works.

---

## 4. What the alerts look like

```
🟢 LAMBODHARA TEXTILES  LAMBODHARA.NS
flag-rail reclaim — CONFIRMED

Entry      135.90   (close)
Stop       111.50   (risk 17.95%)
Target     147.19   pole high
Reward     0.46R

Pole       2026-08-24   ₹105.5 → ₹147.19  (+39.5%)
Flag       14 bars   rail ₹116.6
Sweep      2026-09-22   low ₹115.57  (-0.88% under rail)
Trigger    2026-09-24   vol 3.9× 20-day avg

mcap ₹157 cr · 20d turnover ₹0.42 cr
open chart on TradingView
```

* `PROVISIONAL` (intraday) = price is currently above the rail; it only counts if the day **closes** above it.
  A `CONFIRMED` alert follows after 15:30 IST.
* Every setup that is rejected is still printed with the reason, e.g.
  `RUBICON.NS: drift only 17 bars < 25; shakeout only -4.67% < -8.0%` — you always see why a name did not alert.
* Live gates, in order: the reclaim bar must not print more than `EXCLUDE_VOL_SPIKE_X` (3x) its 20-day average
  volume; the drift must be at least `QUALITY_MIN_DRIFT` (25) bars; the shakeout must undercut the rail by at
  least `QUALITY_MIN_SWEEP` (8%); reward to the pole high must be at least `MIN_REWARD_R` (0.5R); and a trade whose
  entry is more than one bar old is labelled **IN TRADE** instead of being re-alerted.
* One alert per setup **ever** (keyed on symbol + pole date + provisional/confirmed), plus one **post-close digest**
  listing every live setup (coiling / flag built / swept / in trade).

---

## 5. Tuning

Everything lives in `nse_scanner/config.py` (each is also an env var):

| knob | default | meaning |
|---|---|---|
| `POLE_MODE` | `locked` | `locked` = +9.8% & near-high & 3× vol · `breakout` = looser +6% |
| `VOL_SURGE` | `3.0` | ignition volume multiple |
| `BREAK_ATR` / `BREAK_MIN_PCT` | `0.5` / `1.5` | how decisive the floor break must be to end the drift |
| `TRIGGER_MODE` | `first_bullish` | `first_bullish` = buy the first bullish candle after the shakeout (**default** — fires earlier, stop half as wide, payoff 2.33:1) or `rail_reclaim` = wait for a close above the rail + prior 5-bar high (higher hit rate, but the stop sits ~16% away) |
| `MIN_SWEEP_DEPTH_PCT` | `3.0` | sanity check that a real shakeout happened (it passes ~98% of signals, so it is not the edge) |
| `QUALITY_PROFILE` | `strict` | `off` · `balanced` (quiet reclaim + drift ≥ 15) · `strict` (quiet + drift ≥ 25 + shakeout ≥ 8%) |
| `QUALITY_MIN_DRIFT` / `QUALITY_MIN_SWEEP` | `25` / `8.0` | the two strict-profile thresholds |
| `EXCLUDE_VOL_SPIKE_X` | `3.0` | refuse alerts whose reclaim bar trades more than this multiple of its 20-day average volume |
| `REGIME_FILTER` | `off` | require NIFTY < its 200-DMA. **Off by default** - the full-sample edge (+0.36R) decays to nothing on a time split (+0.73R → +0.16R) |
| `ATR_BUF` | `0.5` | stop buffer below the sweep low, in ATRs |
| `MAX_HOLD_BARS` | `60` | time stop |
| `RET_HORIZON` | `30` | the "return after 30 days" column in the report |
| `MIN_PRICE` / `MIN_MCAP_CR` | `100` / `1000` | the two filters you asked for |
| `MIN_REWARD_R` | `0.5` | live alerts only - skip if reward to pole high < this |
| `MAX_UNIVERSE` | `0` | cap the universe (handy for testing) |

---

## 6. Honest limitations — read before trading this

1. **GitHub cron is not a real-time ticker.** Scheduled runs fire every ~15 min but GitHub frequently delays
   them by 5–20 min under load, and can skip a slot entirely. Treat alerts as "within 20 minutes", not "same second".
2. **Intraday alerts are provisional.** The tested rule is a *daily close* above the rail. A `PROVISIONAL` alert
   can die before 15:30 — that is exactly what the `CONFIRMED` alert is for.
3. **yfinance is an unofficial, rate-limited source.** The code batches, retries with back-off and skips stale data,
   but occasionally Yahoo returns 429s from cloud IPs. If a run logs "got 0 usable symbols", just re-run it.
4. **The back-test's market-cap filter uses today's market cap** across the whole history — that is a mild
   look-ahead/survivorship bias. Real results would be somewhat worse.
5. **No costs are modelled.** Subtract roughly 0.1–0.3R per trade for brokerage, STT, slippage and impact.
6. **The stop-assumed-first rule** makes the back-test slightly pessimistic; keep it that way.
7. **Filters are measured, not assumed.** Every gate in this repo was tested on a train/test time split
   (`filter_study.py`, and the profile table in the PDF). Two earlier claims did **not** survive and were dropped:
   the market-regime filter (it decays from +0.73R to +0.16R out of sample) and the sweep-depth gate as "the edge"
   (it passes ~98% of signals). What survives is the quiet reclaim candle and the drift length. If a future run
   shows a gate no longer holding on the test half, turn it off.
8. **This is not investment advice.** It is a mechanical pattern scanner. Position sizing and the hard stop are the
   risk management.

---

## 7. Running it locally

```bash
pip install -r nse_scanner/requirements.txt

# refresh the NSE universe + market caps
python3 -m nse_scanner.refresh_universe

# live scan, no Telegram, first 150 symbols
python3 -m nse_scanner.live_scanner --limit 150 --no-telegram --force-provisional

# what would the scanner have said today? (market-closed safe)
python3 -m nse_scanner.live_scanner --limit 150 --no-telegram --confirmed

# quick back-test on 150 names, no PDF
python3 -m nse_scanner.backtest --limit 150 --trigger both

# full back-test + PDF
python3 -m nse_scanner.backtest --trigger both --pdf --out reports

# rebuild the summary + PDF from an existing trades csv (no downloads, ~4 seconds)
python3 -m nse_scanner.backtest --from-csv reports/backtest_trades.csv --out reports

# which filters actually remove the false signals? (train <2022 / test >=2022)
python3 -m nse_scanner.filter_study --csv reports/backtest_trades.csv --mode first_bullish

# the two charts the engine was built from — run after ANY change to pattern.py
python3 -m nse_scanner.tests.test_exemplars
```

---

## 8. Troubleshooting

| symptom | fix |
|---|---|
| "TELEGRAM_BOT_TOKEN not set — printing instead" | secrets not added, or the workflow wasn't given them |
| no alerts but the job succeeded | nothing triggered, or everything was filtered by `MIN_REWARD_R`; run with `dry_run` and `limit=50` to see the gate messages |
| "got 0 usable symbols" | Yahoo rate-limit — re-run, or raise `BATCH_SIZE` sleep (edit `data.py`) |
| scheduled runs stopped | repo inactive > 60 days → hit Run workflow once, or push a commit |
| duplicate alerts | check that the commit-state step ran (needs *Read and write* workflow permissions) |
