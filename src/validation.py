"""
validation.py
-------------
Step 6 – Statistical validation.

Two complementary checks
------------------------
1. Shuffle test (permutation test)
2. Out-of-sample stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


PERIODS_PER_YEAR = 252


def _sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR))


# ------------------- shuffle test -----------------------
def shuffle_test(returns: pd.Series,
                 position: pd.Series,
                 n_shuffles: int = 1000,
                 seed: int = 42,
                 verbose: bool = True) -> dict:
    """
    Permutation test: shuffle trade entry dates and recompute Sharpe.

    Returns
    -------
    dict with null_sharpes array, real_sharpe, p_value, significant flag.
    """
    rng = np.random.default_rng(seed)
    real_sharpe = _sharpe(returns)
    null_sharpes = []

    ret_vals = returns.values
    pos_vals = position.values
    n = len(pos_vals)

    for _ in range(n_shuffles):
        shift = rng.integers(1, n)
        shuffled_pos = np.roll(pos_vals, shift)

        # recompute returns with shuffled position (same spread changes)
        # approximate: treat returns as position times daily change
        # since we can't perfectly reconstruct spread, we shift the position
        # and reuse the return magnitudes scaled by sign
        shuffled_returns = np.where(
            shuffled_pos != 0,
            np.abs(ret_vals) * np.sign(shuffled_pos),
            0.0
        )
        sr = pd.Series(shuffled_returns)
        null_sharpes.append(_sharpe(sr))

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= real_sharpe).mean()
    significant = p_value < 0.05

    if verbose:
        print(f"\n----------- Shuffle Test ------------")
        print(f"  Real Sharpe        : {real_sharpe:.3f}")
        print(f"  Null mean +/- std    : {null_sharpes.mean():.3f} +/- {null_sharpes.std():.3f}")
        print(f"  95th pct of null   : {np.percentile(null_sharpes, 95):.3f}")
        print(f"  p-value            : {p_value:.4f}")
        print(f"  Significant (p<.05): {significant}")

    return {
        "real_sharpe"  : real_sharpe,
        "null_sharpes" : null_sharpes,
        "p_value"      : p_value,
        "significant"  : significant,
    }


def plot_shuffle_test(result: dict, label: str = "", save_path: str = None):
    """Histogram of null Sharpe distribution with real Sharpe marked."""
    null   = result["null_sharpes"]
    real   = result["real_sharpe"]
    p_val  = result["p_value"]
    p95    = np.percentile(null, 95)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null, bins=50, color="#abd9e9", edgecolor="white", linewidth=0.3,
            density=True, label="Null distribution (shuffled)")
    ax.axvline(real, color="#d7191c", linewidth=2.0, label=f"Real Sharpe = {real:.3f}")
    ax.axvline(p95,  color="#fdae61", linewidth=1.5, linestyle="--",
               label=f"95th pct = {p95:.3f}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.set_title(f"Shuffle Test - {label}\np-value = {p_val:.4f}"
                 + ("  is significant" if result["significant"] else "   not significant"))
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ------------------- out-of-sample comparison --------------------

def oos_comparison(ret_train: pd.Series,
                   ret_test:  pd.Series,
                   label: str = "",
                   verbose: bool = True) -> dict:
    """
    Compare in-sample and out-of-sample performance metrics.
    """
    from performance import summarise, max_drawdown

    is_metrics  = summarise(ret_train, label=f"{label} — In-Sample"  if verbose else "")
    oos_metrics = summarise(ret_test,  label=f"{label} — Out-of-Sample" if verbose else "")

    if verbose:
        print(f"\n-------------- OOS vs IS comparison: {label} ----------------")
        for k in ["sharpe", "max_drawdown_pct", "ann_return_pct"]:
            print(f"  {k:20s}: IS = {is_metrics[k]:8.3f}   OOS = {oos_metrics[k]:8.3f}")

    return {"in_sample": is_metrics, "out_of_sample": oos_metrics}


def plot_oos_comparison(ret_train: pd.Series,
                        ret_test:  pd.Series,
                        label: str = "",
                        save_path: str = None):
    """Side-by-side cumulative return curves for IS and OOS."""
    cum_train = (1 + ret_train).cumprod() - 1
    cum_test  = (1 + ret_test ).cumprod() - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"In-Sample vs Out-of-Sample - {label}", fontsize=13, fontweight="bold")

    for ax, cum, title, color in [
        (axes[0], cum_train, "In-Sample (2010–2022)",    "#2c7bb6"),
        (axes[1], cum_test,  "Out-of-Sample (2023)",     "#1a9641"),
    ]:
        ax.plot(cum * 100, color=color, linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_ylabel("Cumulative Return (%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


