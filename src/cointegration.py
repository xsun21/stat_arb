"""
cointegration.py
----------------
Step 1 – Cointegration testing and hedge ratio estimation.

"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ------------------ helpers --------------------

def adf_summary(series: pd.Series, name: str = "") -> dict:
    """
    Run Augmented Dickey-Fuller test.
    H_0: series is non-stationary.
    Reject H_0 at p < 0.05: series is stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "series"    : name or series.name,
        "adf_stat"  : round(result[0], 4),
        "p_value"   : round(result[1], 4),
        "stationary": result[1] < 0.05,
        "lags_used" : result[2],
    }


def engle_granger_test(s_a: pd.Series, s_b: pd.Series) -> dict:
    """
    Engle-Granger cointegration test using statsmodels.coint.
    H_0: no cointegration.
    Returns test statistic, p-value, and critical values.
    """
    stat, p_value, crits = coint(s_a, s_b)
    return {
        "eg_stat"      : round(stat, 4),
        "p_value"      : round(p_value, 4),
        "cointegrated" : p_value < 0.05,
        "crit_1pct"    : round(crits[0], 4),
        "crit_5pct"    : round(crits[1], 4),
        "crit_10pct"   : round(crits[2], 4),
    }


def estimate_hedge_ratio(s_a: pd.Series, s_b: pd.Series) -> tuple[float, pd.Series]:
    """
    Estimate hedge ratio beta by OLS regression: P_A = alpha + beta P_B + epsilon.

    Returns
    -------
    beta   : float   - hedge ratio
    spread : Series  - residuals  S = P_A - beta P_B  (the mean-reverting spread)

    Note: OLS here gives a static beta over the full training window.
    """
    X = add_constant(s_b.values)
    model = OLS(s_a.values, X).fit()
    alpha, beta = model.params
    spread = s_a - beta * s_b
    spread.name = "spread"
    return beta, spread


def run_cointegration_analysis(prices: pd.DataFrame,
                               pair: tuple[str, str],
                               verbose: bool = True) -> dict:
    """
    Full cointegration analysis for a single pair on a price DataFrame.

    Returns a dict with hedge ratio, spread series, and all test results.
    """
    a, b = pair
    s_a, s_b = prices[a], prices[b]

    # confirm each series is I(1): ADF should NOT reject (p > 0.05)
    adf_a = adf_summary(s_a, a)
    adf_b = adf_summary(s_b, b)

    # Engle-Granger cointegration test
    eg = engle_granger_test(s_a, s_b)

    # estimate hedge ratio and build spread
    beta, spread = estimate_hedge_ratio(s_a, s_b)

    # 4) ADF on the spread — should be stationary if cointegrated
    adf_spread = adf_summary(spread, "spread")

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Pair: {a} / {b}")
        print(f"{'='*55}")
        print(f"  ADF {a:5s}: stat={adf_a['adf_stat']:7.3f}  p={adf_a['p_value']:.4f}"
              f"  stationary={adf_a['stationary']}")
        print(f"  ADF {b:5s}: stat={adf_b['adf_stat']:7.3f}  p={adf_b['p_value']:.4f}"
              f"  stationary={adf_b['stationary']}")
        print(f"  EG test    : stat={eg['eg_stat']:7.3f}  p={eg['p_value']:.4f}"
              f"  cointegrated={eg['cointegrated']}")
        print(f"  Hedge ratio = {beta:.4f}")
        print(f"  ADF spread : stat={adf_spread['adf_stat']:7.3f}  p={adf_spread['p_value']:.4f}"
              f"  stationary={adf_spread['stationary']}")

    return {
        "pair"      : pair,
        "beta"      : beta,
        "spread"    : spread,
        "adf_a"     : adf_a,
        "adf_b"     : adf_b,
        "eg"        : eg,
        "adf_spread": adf_spread,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_prices, train_test_split, PAIRS

    all_tickers = list({t for pair in PAIRS for t in pair})
    prices = fetch_prices(all_tickers)
    train, _ = train_test_split(prices)

    for pair in PAIRS:
        run_cointegration_analysis(train, pair)
