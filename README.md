# Portfolio Optimization with NSGA-II

A multi-objective portfolio optimization tool using the NSGA-II evolutionary algorithm. The system optimizes portfolio weights to maximize returns while minimizing risk, comparing results against the Markowitz efficient frontier and market indices.

## Features

- **Multi-Objective Optimization**: Find optimal trade-offs between return and risk using NSGA-II.
- **Multiple Risk Metrics**: Optimize for Volatility (std), Max Drawdown (MDD), or Sharpe Ratio.
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
| `--ticker-set` | Asset set (`WIG20`, `US_TECH`, `US_DEFENSIVE`, `ETFS`, `CRYPTO`) | `WIG20` |
| `--risk-metric` | Objective: `std`, `mdd`, or `sharpe` | `std` |
| `--seed` | Random seed for reproducibility | `None` |
| `--pop-size` | Population size | `200` |
| `--n-generations` | Number of generations | `80` |
| `--start-date` | Data start date (YYYY-MM-DD) | `2002-01-01` |
| `--benchmark` | Benchmark ticker (optional) | Synthetic |
| `--no-plots` | Suppress plot display | `False` |

### Examples

**Reproducible US Tech run:**
```bash
uv run python main.py --ticker-set US_TECH --seed 42 --n-generations 100
```

**Crypto Max Drawdown optimization:**
```bash
uv run python main.py --ticker-set CRYPTO --risk-metric mdd --pop-size 300
```

## Output

Results saved to `plots/experiment-{TIMESTAMP}/`:
- `config.yaml`: Run parameters and seed.
- `pareto_vs_markowitz.png`: Pareto front vs Efficient Frontier.
- `Final_Portfolio_vs_Index.png`: Cumulative growth comparison.
- `final_portfolio_composition.png`: Final asset weights.

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
