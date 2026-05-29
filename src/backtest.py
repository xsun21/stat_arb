"""
backtest.py
-----------
Step 4: Bias-free backtesting engine.

"""

import numpy as np
import pandas as pd


TRANSACTION_COST = 0.001   # 0.1% per trade (open or close), one-way


def compute_daily_returns(spread: pd.Series,
                          position: pd.Series,
                          cost: float = TRANSACTION_COST) -> pd.Series:
    """
    Compute daily strategy returns after transaction costs.

    Parameters
    ----------
    spread   : spread series aligned with position
    position : +1 / 0 / -1 signal (lagged by 1 day inside this function)
    cost     : one-way transaction cost as a fraction

    Returns
    -------
    returns : daily return series (fractional, not %)
    """
    spread    = spread.reindex(position.index).dropna()
    position  = position.reindex(spread.index)

    # lag position by 1: act on yesterday's signal at today's open
    pos_lagged = position.shift(1).fillna(0)

    # spread daily change
    d_spread = spread.diff()

    # raw returns: position times change in spread, normalised by spread level
    # normalise by the absolute mean spread so returns are scale-independent
    spread_scale = spread.abs().mean()
    raw_returns  = pos_lagged * d_spread / spread_scale

    # transaction costs: deducted whenever position changes
    trades = pos_lagged.diff().abs()   # 0, 1, or 2 
    cost_series = trades * cost

    returns = raw_returns - cost_series
    returns.name = "strategy_returns"
    return returns


def run_backtest(prices_train: pd.DataFrame,
                 prices_test:  pd.DataFrame,
                 pair:         tuple[str, str],
                 beta:         float,
                 window:       int   = 60,
                 entry_z:      float = 2.0,
                 exit_z:       float = 0.5,
                 cost:         float = TRANSACTION_COST,
                 verbose:      bool  = True) -> dict:
    """
    Full backtest for a single pair.

    beta is estimated from training data (not re-estimated here).
    Z-score rolling stats are also computed on training data and frozen.

    Returns a dict with returns, cumulative PnL, and trade log.
    """
    from signals import compute_spread, compute_zscore, generate_signals

    # ------- training set -------
    spread_train = compute_spread(prices_train, pair, beta)
    zscore_train = compute_zscore(spread_train, window)
    pos_train    = generate_signals(zscore_train, entry_z, exit_z)
    ret_train    = compute_daily_returns(spread_train, pos_train, cost)

    # -------- test set --------
    # append train to test so the rolling window warms up correctly,
    # then slice to test dates only.
    spread_full  = compute_spread(pd.concat([prices_train, prices_test]), pair, beta)
    zscore_full  = compute_zscore(spread_full, window)
    pos_full     = generate_signals(zscore_full, entry_z, exit_z)

    test_idx     = prices_test.index
    spread_test  = spread_full.reindex(test_idx)
    pos_test     = pos_full.reindex(test_idx)
    ret_test     = compute_daily_returns(spread_test, pos_test, cost)

    # --------- trade log ---------
    trade_log = _build_trade_log(pos_train, ret_train, "train")

    if verbose:
        n_trades = (pos_train.diff().abs() > 0).sum()
        print(f"\n  Backtest [{pair[0]}/{pair[1]}]  train period")
        print(f"  Trades opened : {n_trades}")
        print(f"  Total return  : {ret_train.sum()*100:.2f}%")
        print(f"  Ann. Sharpe   : {_sharpe(ret_train):.3f}")

    return {
        "pair"        : pair,
        "beta"        : beta,
        "ret_train"   : ret_train,
        "ret_test"    : ret_test,
        "pos_train"   : pos_train,
        "pos_test"    : pos_test,
        "spread_train": spread_train,
        "spread_test" : spread_test,
        "trade_log"   : trade_log,
    }


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (rf = 0)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _build_trade_log(position: pd.Series,
                     returns:  pd.Series,
                     label:    str) -> pd.DataFrame:
    """
    Build a per-trade log: entry date, exit date, direction, PnL.
    """
    records = []
    in_trade  = False
    entry_idx = None
    direction = 0

    for date, pos in position.items():
        if not in_trade and pos != 0:
            in_trade  = True
            entry_idx = date
            direction = pos
        elif in_trade and pos == 0:
            pnl = returns.loc[entry_idx:date].sum()
            records.append({
                "entry"    : entry_idx,
                "exit"     : date,
                "direction": "long" if direction == 1 else "short",
                "pnl"      : round(pnl, 6),
                "set"      : label,
            })
            in_trade = False

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader   import fetch_prices, train_test_split, PAIRS
    from cointegration import run_cointegration_analysis

    all_tickers = list({t for pair in PAIRS for t in pair})
    prices = fetch_prices(all_tickers)
    train, test = train_test_split(prices)

    pair   = PAIRS[0]
    result = run_cointegration_analysis(train, pair, verbose=False)
    bt     = run_backtest(train, test, pair, result["beta"])
    print("\nFirst 5 trades:")
    print(bt["trade_log"].head())
