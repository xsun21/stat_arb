"""
main.py
-------
Run the full stat arb pipeline for all pairs.
Results and plots saved to results/.

"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader   import fetch_prices, train_test_split, PAIRS
from cointegration import run_cointegration_analysis
from signals       import compute_spread, compute_zscore, generate_signals, plot_signals
from backtest      import run_backtest
from performance   import summarise, plot_performance
from validation    import shuffle_test, plot_shuffle_test, oos_comparison, plot_oos_comparison
from failure_modes import (rolling_hedge_ratio, rolling_hurst, hurst_exponent,
                           plot_failure_modes)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def run_pair(pair: tuple[str, str],
             train: pd.DataFrame,
             test:  pd.DataFrame,
             all_prices: pd.DataFrame):

    a, b = pair
    tag  = f"{a}_{b}"
    print(f"\n{'#'*60}")
    print(f"  PAIR: {a} / {b}")
    print(f"{'#'*60}")

    # -------------------- Step 1: cointegration ---------------------
    coint = run_cointegration_analysis(train, pair, verbose=True)
    if not coint["eg"]["cointegrated"]:
        print(f" !Pair {a}/{b} is NOT cointegrated at 5% level -- skipping.")
        return

    beta = coint["beta"]

    # -------------------- Step 2-3: signals ----------------------
    spread   = coint["spread"]
    zscore   = compute_zscore(spread)
    position = generate_signals(zscore)
    plot_signals(spread, zscore, position, pair,
                 save_path=save(f"{tag}_signals.png"))

    # -------------------- Step 4: backtest -----------------------
    bt = run_backtest(train, test, pair, beta, verbose=True)

    # -------------------- Step 5: performance ------------------------
    summarise(bt["ret_train"], label=f"{a}/{b} -- Train")
    plot_performance(bt["ret_train"], label=f"{a}/{b} -- Train",
                     save_path=save(f"{tag}_performance_train.png"))

    # -------------------- Step 6: validation --------------------------
    shuf = shuffle_test(bt["ret_train"], bt["pos_train"], verbose=True)
    plot_shuffle_test(shuf, label=f"{a}/{b}",
                      save_path=save(f"{tag}_shuffle_test.png"))

    oos_comparison(bt["ret_train"], bt["ret_test"],
                   label=f"{a}/{b}", verbose=True)
    plot_oos_comparison(bt["ret_train"], bt["ret_test"],
                        label=f"{a}/{b}",
                        save_path=save(f"{tag}_oos.png"))

    # -------------------- Step 7: failure modes --------------------------
    spread_full = all_prices[a] - beta * all_prices[b]
    rb = rolling_hedge_ratio(all_prices, pair)
    h  = hurst_exponent(spread)
    print(f"\n  Hurst exponent (training spread): {h:.4f}"
          + ("  is mean-reverting" if h < 0.5 else "  not mean-reverting"))

    plot_failure_modes(spread_full, rb, beta, pair,
                       save_path=save(f"{tag}_failure_modes.png"))

    print(f"\n   All plots saved to results/{tag}_*.png")


def main():
    print("Fetching price data …")
    all_tickers = list({t for pair in PAIRS for t in pair})
    all_prices  = fetch_prices(all_tickers)
    train, test = train_test_split(all_prices)

    print(f"Train: {train.index[0].date()} -> {train.index[-1].date()}")
    print(f"Test : {test.index[0].date()}  -> {test.index[-1].date()}")

    for pair in PAIRS:
        run_pair(pair, train, test, all_prices)

    print(f"\n{'='*60}")
    print("  Pipeline complete. Results saved to results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
