"""
failure_modes.py
----------------
Step 7 – Failure mode analysis.

Two main failure modes analysed here
------------------------------------
1. Hedge ratio drift (beta instability)
2. Regime shifts / non-stationarity of the spread
   - Rolling ADF p-value
   - Rolling Hurst exponent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ---------------- rolling hedge ratio ---------------------

def rolling_hedge_ratio(prices: pd.DataFrame,
                        pair: tuple[str, str],
                        window: int = 252) -> pd.Series:
    """
    Compute beta via rolling OLS regression over the last "window" days.
    """
    a, b = pair
    betas = {}

    for i in range(window, len(prices) + 1):
        chunk = prices.iloc[i - window : i]
        X = add_constant(chunk[b].values)
        model = OLS(chunk[a].values, X).fit()
        betas[prices.index[i - 1]] = model.params[1]

    return pd.Series(betas, name="rolling_beta")


def plot_rolling_beta(rolling_beta: pd.Series,
                      static_beta: float,
                      pair: tuple[str, str],
                      save_path: str = None):
    """Plot rolling beta vs static beta estimated on full training window."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_beta, color="#2c7bb6", linewidth=0.9, label="Rolling β (252-day window)")
    ax.axhline(static_beta, color="#d7191c", linestyle="--", linewidth=1.2,
               label=f"Static β = {static_beta:.4f}")
    ax.fill_between(rolling_beta.index,
                    rolling_beta - rolling_beta.std(),
                    rolling_beta + rolling_beta.std(),
                    alpha=0.15, color="#2c7bb6")
    ax.set_title(f"Rolling Hedge Ratio β — {pair[0]}/{pair[1]}\n"
                 "Drift here means the cointegration relationship is changing")
    ax.set_ylabel("β")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ------------------ rolling ADF p-value -----------------------

def rolling_adf_pvalue(spread: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling ADF test p-value.
    """
    pvals = {}
    for i in range(window, len(spread) + 1):
        chunk = spread.iloc[i - window : i].dropna()
        if len(chunk) < window // 2:
            continue
        try:
            _, p, *_ = adfuller(chunk, autolag="AIC")
        except Exception:
            p = np.nan
        pvals[spread.index[i - 1]] = p
    return pd.Series(pvals, name="rolling_adf_p")


# ------------------ Hurst exponent ---------------------
def hurst_exponent(series: pd.Series) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.
    """
    series = series.dropna().values
    n = len(series)
    if n < 20:
        return np.nan

    lags   = range(2, min(100, n // 2))
    tau    = []
    rs_vals = []

    for lag in lags:
        sub = series[:lag]
        mean_sub = np.mean(sub)
        deviations = np.cumsum(sub - mean_sub)
        R = deviations.max() - deviations.min()
        S = np.std(sub, ddof=1)
        if S == 0:
            continue
        tau.append(lag)
        rs_vals.append(R / S)

    if len(tau) < 2:
        return np.nan

    log_tau = np.log(tau)
    log_rs  = np.log(rs_vals)
    poly    = np.polyfit(log_tau, log_rs, 1)
    return poly[0]


def rolling_hurst(spread: pd.Series, window: int = 252) -> pd.Series:
    """Rolling Hurst exponent over the spread."""
    hursts = {}
    for i in range(window, len(spread) + 1):
        chunk = spread.iloc[i - window : i]
        #hursts[spread.index[i - 1]] = hurst_exponent(chunk)
        hursts[spread.index[i - 1]] = hurst_exponent(chunk.diff().dropna())
    return pd.Series(hursts, name="rolling_hurst")


# --------------- combined failure mode plot ------------------

def plot_failure_modes(spread: pd.Series,
                       rolling_beta: pd.Series,
                       static_beta: float,
                       pair: tuple[str, str],
                       save_path: str = None):
    """
    Three-panel failure mode dashboard:
      1. Rolling beta vs static beta
      2. Rolling ADF p-value with 0.05 threshold
      3. Rolling Hurst exponent with 0.5 threshold
    """
    adf_p  = rolling_adf_pvalue(spread)
    hurst  = rolling_hurst(spread)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f"Failure Mode Analysis — {pair[0]}/{pair[1]}", fontsize=14, fontweight="bold")

    # panel 1: rolling β
    axes[0].plot(rolling_beta, color="#2c7bb6", linewidth=0.9)
    axes[0].axhline(static_beta, color="#d7191c", linestyle="--", linewidth=1.0,
                    label=f"Static β = {static_beta:.4f}")
    axes[0].set_title("Rolling Hedge Ratio β  (drift = structural change)")
    axes[0].set_ylabel("β")
    axes[0].legend(fontsize=8)

    # panel 2: ADF p-value
    axes[1].plot(adf_p, color="#555", linewidth=0.8)
    axes[1].axhline(0.05, color="#d7191c", linestyle="--", linewidth=1.0,
                    label="p = 0.05 threshold")
    axes[1].fill_between(adf_p.index, adf_p, 0.05,
                         where=(adf_p > 0.05), color="#d7191c", alpha=0.2,
                         label="Non-stationary zone")
    axes[1].set_title("Rolling ADF p-value  (above 0.05 = danger zone)")
    axes[1].set_ylabel("p-value")
    axes[1].legend(fontsize=8)

    # panel 3: Hurst exponent
    axes[2].plot(hurst, color="#1a9641", linewidth=0.8)
    axes[2].axhline(0.5, color="#d7191c", linestyle="--", linewidth=1.0,
                    label="H = 0.5 (random walk)")
    axes[2].fill_between(hurst.index, hurst, 0.5,
                         where=(hurst > 0.5), color="#d7191c", alpha=0.2,
                         label="Trending regime")
    axes[2].set_title("Rolling Hurst Exponent  (> 0.5 = trending, stat arb breaks)")
    axes[2].set_ylabel("H")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


