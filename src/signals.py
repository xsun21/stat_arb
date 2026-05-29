"""
signals.py
----------
Step 2 & 3 – Spread construction and z-score signal generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------ default thresholds ----------------
DEFAULT_WINDOW  = 60    # rolling lookback in trading days (~3 months)
DEFAULT_ENTRY_Z = 2.0   # open a position when |z| crosses this
DEFAULT_EXIT_Z  = 0.5   # close a position when |z| falls below this


def compute_spread(prices: pd.DataFrame,
                   pair: tuple[str, str],
                   beta: float) -> pd.Series:
    """Build spread  S = P_A - beta P_B  from a price DataFrame."""
    a, b = pair
    spread = prices[a] - beta * prices[b]
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, window: int = DEFAULT_WINDOW) -> pd.Series:
    """
    Rolling z-score of the spread.
    """
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std()
    zscore = (spread - roll_mean) / roll_std
    zscore.name = "zscore"
    return zscore


def generate_signals(zscore: pd.Series,
                     entry_z: float = DEFAULT_ENTRY_Z,
                     exit_z:  float = DEFAULT_EXIT_Z) -> pd.Series:
    """
    Generate position signals from the z-score.

    Returns a Series of integers:
        +1  :  long  the spread
        -1  :  short the spread
         0  :  flat (no position)
    """
    position = pd.Series(0, index=zscore.index, dtype=int, name="position")
    current  = 0  # current position: -1, 0, or +1

    for i, z in enumerate(zscore):
        if np.isnan(z):
            position.iloc[i] = 0
            continue

        if current == 0:
            # no position: check for entry
            if z < -entry_z:
                current = +1   # spread below mean -> long, expect upward reversion
            elif z > entry_z:
                current = -1   # spread above mean -> short, expect downward reversion
        elif current == +1:
            if abs(z) < exit_z:
                current = 0
        elif current == -1:
            if abs(z) < exit_z:
                current = 0

        position.iloc[i] = current

    return position


def plot_signals(spread: pd.Series,
                 zscore: pd.Series,
                 position: pd.Series,
                 pair: tuple[str, str],
                 save_path: str = None):
    """Three-panel plot: spread, z-score with bands, position."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    a, b = pair
    fig.suptitle(f"Signal Overview: {a} / {b}", fontsize=14, fontweight="bold")

    # panel 1: raw spread
    axes[0].plot(spread, color="#2c7bb6", linewidth=0.8)
    axes[0].axhline(spread.mean(), color="black", linestyle="--", linewidth=0.8, label="mean")
    axes[0].set_ylabel("Spread")
    axes[0].legend(fontsize=8)

    # panel 2: z-score with threshold bands
    axes[1].plot(zscore, color="#555", linewidth=0.8)
    for level, color in [(DEFAULT_ENTRY_Z, "#d7191c"), (-DEFAULT_ENTRY_Z, "#1a9641"),
                         (DEFAULT_EXIT_Z,  "#aaa"),    (-DEFAULT_EXIT_Z,  "#aaa")]:
        axes[1].axhline(level, linestyle="--", linewidth=0.8, color=color)
    axes[1].set_ylabel("Z-score")

    # panel 3: position
    axes[2].fill_between(position.index, position, 0,
                         where=position > 0, color="#1a9641", alpha=0.6, label="Long")
    axes[2].fill_between(position.index, position, 0,
                         where=position < 0, color="#d7191c", alpha=0.6, label="Short")
    axes[2].set_ylabel("Position")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


