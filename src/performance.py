"""
performance.py
--------------
Step 5 – Performance evaluation.

Metrics
-------
- Sharpe ratio   : risk-adjusted return (annualised, rf=0)
- Max drawdown   : largest peak-to-trough loss in cumulative PnL
- Return distribution : mean, std, skewness, kurtosis, fat-tail check
- Calmar ratio   : annualised return / max drawdown 

All metrics computed on daily returns series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


PERIODS_PER_YEAR = 252


# ------------------- individual metrics -----------------------

def sharpe_ratio(returns: pd.Series) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown of the cumulative return curve.
    Returns a negative number.
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return divided by absolute max drawdown."""
    ann_ret = returns.mean() * PERIODS_PER_YEAR
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return ann_ret / mdd


def return_distribution_stats(returns: pd.Series) -> dict:
    """
    Summary statistics of the return distribution.

    Skewness > 0:  right tail (occasional large gains)
    Excess kurtosis > 0:  fat tails (more extreme events than Gaussian)
    Jarque-Bera p < 0.05:  non-Gaussian distribution
    """
    clean = returns.dropna()
    jb_stat, jb_p = stats.jarque_bera(clean)
    return {
        "mean"            : float(clean.mean()),
        "std"             : float(clean.std()),
        "skewness"        : float(clean.skew()),
        "excess_kurtosis" : float(clean.kurtosis()),  # pandas kurtosis is excess
        "jb_stat"         : round(jb_stat, 4),
        "jb_p_value"      : round(jb_p, 4),
        "gaussian"        : jb_p >= 0.05,
        "ann_return_pct"  : round(clean.mean() * PERIODS_PER_YEAR * 100, 2),
    }


def summarise(returns: pd.Series, label: str = "") -> dict:
    """Compute and print all performance metrics."""
    dist = return_distribution_stats(returns)
    metrics = {
        "label"          : label,
        "sharpe"         : round(sharpe_ratio(returns), 3),
        "max_drawdown_pct": round(max_drawdown(returns) * 100, 2),
        "calmar"         : round(calmar_ratio(returns), 3),
        **dist,
    }
    if label:
        print(f"\n------------ Performance: {label} -------------")
        print(f"  Sharpe ratio      : {metrics['sharpe']}")
        print(f"  Max drawdown      : {metrics['max_drawdown_pct']}%")
        print(f"  Calmar ratio      : {metrics['calmar']}")
        print(f"  Ann. return       : {metrics['ann_return_pct']}%")
        print(f"  Skewness          : {metrics['skewness']:.3f}")
        print(f"  Excess kurtosis   : {metrics['excess_kurtosis']:.3f}")
        print(f"  Gaussian (JB test): {metrics['gaussian']}  (p={metrics['jb_p_value']})")
    return metrics


# ----------------- plots -------------------
def plot_performance(returns: pd.Series,
                     label: str = "Strategy",
                     save_path: str = None):
    """
    Four-panel performance dashboard:
      1. Cumulative return curve
      2. Drawdown curve
      3. Daily return distribution vs Gaussian fit
      4. Q-Q plot
    """
    cum_ret  = (1 + returns).cumprod() - 1
    roll_max = (1 + returns).cumprod().cummax()
    drawdown = ((1 + returns).cumprod() - roll_max) / roll_max

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Performance Dashboard - {label}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. cumulative return
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cum_ret * 100, color="#2c7bb6", linewidth=1.0)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_title("Cumulative Return (%)")
    ax1.set_ylabel("%")

    # 2. drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color="#d7191c", alpha=0.5)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("%")

    # 3. return distribution
    ax3 = fig.add_subplot(gs[1, 0])
    clean = returns.dropna()
    ax3.hist(clean, bins=60, density=True, color="#abd9e9", edgecolor="white", linewidth=0.3)
    x = np.linspace(clean.min(), clean.max(), 300)
    ax3.plot(x, stats.norm.pdf(x, clean.mean(), clean.std()),
             color="#d7191c", linewidth=1.2, label="Gaussian fit")
    ax3.set_title("Return Distribution")
    ax3.legend(fontsize=8)

    # 4. Q-Q plot
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(clean, dist="norm", plot=ax4)
    ax4.set_title("Q-Q Plot (vs Normal)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()



