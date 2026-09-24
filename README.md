# NSE pole → flag → sweep → reclaim scanner

Live Telegram alerts (GitHub Actions, every 15 min) + a full-universe back-test that writes a PDF report.

**Start here → [nse_scanner/README.md](nse_scanner/README.md)**

* `nse_scanner/` — the engine: `pattern.py` (detection), `live_scanner.py`, `backtest.py`, `pdf_report.py`,
  `filter_study.py`, `config.py` (every threshold), `universe/watchlist.csv`
* `reports/backtest_report.pdf` — the generated report (open positions, filter study, 1,307-trade log with full
  company names)
* `.github/workflows/` — `live_scan.yml` (cron + manual) and `backtest_report.yml` (manual PDF job)

The two charts the engine was built from (NIACL, LAMBODHARA) are asserted by
`python3 -m nse_scanner.tests.test_exemplars` — run it after any change to `pattern.py`.
