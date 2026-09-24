"""
refresh_universe.py — rebuild universe/nse_universe.csv (all NSE symbols + market cap).

Run daily (the workflow does it automatically) or manually:
    python3 -m nse_scanner.refresh_universe
"""
from . import data as D

if __name__ == "__main__":
    df = D.refresh_universe()
    print(f"universe refreshed: {len(df)} NSE symbols")
    f = D.load_universe(apply_filters=True)
    print(f"after filters (price > ₹{D.C.MIN_PRICE:.0f} & mcap ≥ ₹{D.C.MIN_MCAP_CR:.0f} cr): {len(f)} names")
