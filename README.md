# Portfolio Optimization with NSGA-II

A multi-objective portfolio optimization tool using the NSGA-II evolutionary algorithm. The system optimizes portfolio weights to maximize returns while minimizing risk, comparing results against the Markowitz efficient frontier and market indices.

## Installation

Requires Python 3.14+. Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.14
uv sync
```

## Usage

### Basic Run

```bash
uv run main.py
```

This runs optimization with default parameters on WIG20 stocks.

### Command Line Options

```bash
uv run main.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--tickers` | Comma-separated ticker symbols | WIG20 constituents |
| `--start-date` | Historical data start date (YYYY-MM-DD) | `2002-01-01` |
| `--benchmark` | Benchmark index ticker | Synthetic equal-weight index |
| `--pop-size` | Population size | `200` |
| `--n-generations` | Number of generations | `80` |
| `--cxpb` | Crossover probability | `0.7` |
| `--mutpb` | Mutation probability | `0.6` |
| `--callback-interval` | Plot every N generations | `10` |
| `--no-plots` | Suppress plot display (still saves to disk) | `False` |

### Examples

**Quick test run:**
```bash
uv run main.py --pop-size 50 --n-generations 20
```

**Full optimization run:**
```bash
uv run main.py --pop-size 1500 --n-generations 1000 --callback-interval 100 --no-plots
```

**Custom tickers (S&P 500 diversified portfolio ~20 stocks):**
```bash
uv run main.py --tickers "JPM,BAC,WFC,XOM,CVX,MRK,JNJ,PFE,PG,KO,MCD,WMT,HD,BA,GE,MMM,CAT,IBM,LMT,HON" --benchmark "VOO"
```

## Output

Results are saved to `plots/experiment-YYYYMMDD-HHMMSS/`:

- `config.yaml` - Experiment parameters
- `Pareto_Front_vs_Markowitz.png` - Pareto front comparison with efficient frontier
- `Final_Portfolio_vs_Index.png` - Cumulative returns comparison
- `Final_Optimized_Portfolio.png` - Final portfolio weight distribution
- Intermediate plots at each callback interval

## Project Structure

```
├── main.py                 # CLI entry point
├── pyproject.toml          # Project configuration
├── src/
│   ├── crossovers.py       # Crossover operators
│   ├── data.py             # Data loading (yfinance)
│   ├── evolution.py        # DEAP setup & NSGA-II runner
│   ├── mutations.py        # Mutation operators
│   ├── plots.py            # Visualization functions
│   ├── selections.py       # Selection operators
│   └── utils.py            # Markowitz optimization & utilities
├── notebooks/
│   ├── data_analysis.ipynb
│   └── portfolio_optimization.ipynb
└── plots/                  # Generated experiment outputs
```
