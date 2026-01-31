# Portfolio Optimization with NSGA-II

A multi-objective portfolio optimization tool using the NSGA-II evolutionary algorithm with Walk-Forward Optimization (WFO) for robust backtesting. The system optimizes portfolio weights to maximize returns while minimizing risk, comparing results against the Markowitz efficient frontier and market indices.

## Features

- **Multi-Objective Optimization**: Find optimal trade-offs between return and risk using NSGA-II.
- **Walk-Forward Optimization (WFO)**: Robust backtesting methodology to mitigate look-ahead bias and simulate real-world rebalancing.
- **Dynamic Universe Filtering**: Filter assets based on historical liquidity (`--min-liquidity`) to address survivorship bias.
- **Transaction Costs**: Optional modeling of transaction costs to provide more realistic net returns.
- **Preprocessing Options**: Control data smoothing (`--use-smoothing`) and handling of missing market days (`--fill-missing`).
- **Multiple Risk Metrics**: Optimize for Volatility (std), Max Drawdown (MDD), or Sharpe Ratio.
- **Comprehensive Reporting**: Generates detailed factsheets (Equity Curve, Drawdown, Metrics Table) and animated GIFs of portfolio evolution.
- **Markowitz Comparison**: Automated baseline against analytical efficient frontiers.
- **Automated Data Fetching**: Integrated `yfinance` support for global assets.
- **Reproducibility**: Seed support for consistent experimental results.
- **Rich Visualization**: Automated generation of Pareto fronts and performance charts.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python main.py [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ticker-set` | Asset set (`WIG20`, `WIG_BROAD`, `US_TECH`, etc.) | `WIG_BROAD` |
| `--risk-metric` | Objective: `std`, `mdd`, or `sharpe` | `std` |
| `--seed` | Random seed for reproducibility | `None` |
| `--pop-size` | Population size | `100` |
| `--n-generations` | Number of generations | `50` |
| `--train-window` | Training window size in days (e.g. 1008 = 4 years) | `1008` |
| `--rebalance-freq` | Rebalancing frequency in days (e.g. 90 = 1 quarter) | `90` |
| `--start-date` | Data start date (YYYY-MM-DD) | `2010-01-01` |
| `--benchmark` | Benchmark ticker (optional) | Synthetic |
| `--use-smoothing` | Apply moving average smoothing to prices | `False` |
| `--fill-missing` | Interpolate prices for missing calendar days | `False` |
| `--transaction-cost` | Transaction cost per rebalance (e.g. 0.0025 = 0.25%%) | `0.0025` |
| `--min-liquidity` | Minimum average daily turnover (Price*Vol) to consider a stock valid | `500000.0` |
| `--quiet` | Suppress detailed output (e.g., progress bars) | `False` |

### Examples

**Reproducible US Tech run:**
```bash
uv run python main.py --ticker-set US_TECH --seed 42 --n-generations 100
```

**Crypto Max Drawdown optimization:**
```bash
uv run python main.py --ticker-set CRYPTO --risk-metric mdd --pop-size 300
```

**WIG Broad universe with liquidity filter and costs:**
```bash
uv run main.py --ticker-set WIG_BROAD --risk-metric sharpe --start-date 2010-01-01 --min-liquidity 1000000 --transaction-cost 0.005
```

## Output

Results saved to `plots/wfo-{TIMESTAMP}/`:
- `config.yaml`: Run parameters and seed.
- `wfo_factsheet.png`: Comprehensive factsheet including Equity Curve, Drawdown, and Metrics Table.
- `portfolio_evolution.gif`: Animated GIF showing portfolio composition changes over time.
- `final_portfolio_composition.png`: Final asset weights (at the end of backtest).
- `pareto_vs_markowitz.png`: Pareto front vs Efficient Frontier (from the last training window).

## Project Structure

```
├── main.py                 # CLI entry point
├── pyproject.toml          # Config & dependencies
├── src/
│   ├── crossovers.py       # Crossover operators
│   ├── data.py             # Data fetching
│   ├── evolution.py        # NSGA-II engine
│   ├── mutations.py        # Mutation operators
│   ├── plots.py            # Visualization
│   ├── selections.py       # Selection logic
│   ├── tickers.py          # Asset sets
│   └── utils.py            # Financial metrics & Markowitz
├── notebooks/              # Interactive analysis
└── plots/                  # Saved outputs
```

## Known Limitations

This tool provides a powerful framework for portfolio optimization. However, it's crucial to understand its inherent limitations, especially when interpreting backtesting results:

- **Survivorship Bias & Look-ahead Bias (Universe Selection)**: The biggest challenge for historical backtesting with free data sources like `yfinance`.
    -   We use a broad universe of currently existing tickers (`WIG_BROAD`). This means the algorithm implicitly "sees" companies that survived until today, ignoring those that delisted, went bankrupt, or were illiquid in the past. It does not account for companies that *existed* in the index historically but are no longer traded or whose data has been purged from `yfinance`.
    -   Our `min-liquidity` filter helps mitigate this by only considering truly tradable assets in each historical window, but it cannot reintroduce data for delisted companies.
    -   Therefore, the historical performance shown should be considered an **optimistic upper bound** of what might have been achievable.
- **Data Quality & Completeness (`yfinance`)**:
    -   `yfinance` data can be incomplete, especially for older periods or less liquid stocks. Merged or delisted companies often have their historical data removed or ticker symbols changed, making true survivorship-bias-free backtesting extremely difficult.
    -   Errors in dividend adjustments or splits can lead to artificial price spikes/drops, which the algorithm might exploit unrealistically.
- **Transaction Costs Model**: The current model applies a flat percentage cost (`--transaction-cost`) to the entire portfolio value at each rebalancing step. This is a simplification.
    -   A more granular model would calculate costs based on the actual *turnover* (percentage of portfolio rebalanced) and specific commission rates, plus slippage based on market depth.
- **Liquidity Filter (`--min-liquidity`)**: While effective at filtering out truly illiquid assets, it's a proxy for market capitalization and tradability. It might exclude valid but temporarily less liquid stocks or include those that were liquid but still poor performers.
- **Optimization Assumptions**: Evolutionary algorithms, like traditional Markowitz optimization, rely on historical data and assume that past relationships (returns, correlations, volatilities) will hold true to some extent in the future. This is rarely perfectly the case.
    -   They also assume a rational market and that historical risk-return profiles accurately represent future opportunities.
- **No Short Selling / Leverage**: The current model assumes a long-only portfolio with no leverage, which simplifies the problem but might not reflect all real-world trading strategies.
