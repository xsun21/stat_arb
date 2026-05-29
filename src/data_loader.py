"""
data_loader.py
--------------
Download and cache daily close prices from Yahoo Finance.
Train period : 2010-01-01 – 2022-12-31
Test period  : 2023-01-01 – 2023-12-31
"""

import os
import yfinance as yf
import pandas as pd

# ── pairs traded ──────────────────────────────────────────────────────────────
PAIRS = [
    ("GLD", "SLV"),   # gold / silver ETFs  
    ("KO",  "PEP"),   # Coca-Cola / Pepsi   
]

TRAIN_START = "2010-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2023-12-31"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _cache_path(ticker: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}.csv")


def fetch_prices(tickers: list[str],
                 start: str = TRAIN_START,
                 end: str   = TEST_END,
                 use_cache: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame of adjusted close prices, one column per ticker.
    Data is cached locally as CSV to avoid repeated downloads.
    """
    frames = {}
    for ticker in tickers:
        path = _cache_path(ticker)
        if use_cache and os.path.exists(path):
            series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
        else:
            raw = yf.download(ticker, start=start, end=end,
                              auto_adjust=True, progress=False)
            series = raw["Close"].dropna()
            series.to_csv(path, header=["Close"])
            series = series  
        frames[ticker] = series

    prices = pd.DataFrame(frames).dropna()
    prices.index = pd.to_datetime(prices.index)
    return prices


def train_test_split(prices: pd.DataFrame):
    """Split prices into train (2010-2022) and test (2023) DataFrames."""
    train = prices.loc[TRAIN_START:TRAIN_END]
    test  = prices.loc[TEST_START:TEST_END]
    return train, test



