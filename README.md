# Statistical Arbitrage & Mean-Reversion Strategy

An implementation of a pairs trading strategy based on cointegration and z-score signals, with bias-free backtesting, statistical validation, and failure mode analysis.

---

## Project Structure

```
stat_arb/
├── data/                  # Raw price data
├── src/
│   ├── data_loader.py     # Yahoo Finance data fetching
│   ├── cointegration.py   # Engle-Granger tests, hedge ratio estimation
│   ├── signals.py         # Spread construction, z-score signals
│   ├── backtest.py        # Bias-free backtesting engine
│   ├── performance.py     # Sharpe, drawdown, return distribution
│   ├── validation.py      # Shuffle tests, out-of-sample checks
│   └── failure_modes.py   # Non-stationarity & regime shift analysis
├── results/               # Saved plots and metrics
├── main.py                # Run full pipeline
├── requirements.txt
└── README.md
```

---

## Strategy Overview

- **Universe:** GLD/SLV (gold/silver ETFs) and KO/PEP (Coca-Cola/Pepsi)
- **Data:** Daily close prices via "yfinance", 2010-2022 (train), 2023 (out-of-sample)
- **Signal:** Z-score of the cointegrated spread (rolling 60-day window)
- **Entry:** |z| > 2.0 - **Exit:** |z| < 0.5
- **Costs:** 0.1% transaction cost per trade


```




