"""Orchestrates NSGA-II optimization with periodic plotting."""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from deap import tools

from src.data import load_prices, process_returns
from src.evolution import setup_deap, run_nsga2
from src.plots import plot_pareto_vs_markowitz, plot_portfolio_vs_baseline, plot_final_portfolio
from src.utils import optimize_markowitz


# WIG20 default tickers
DEFAULT_TICKERS = [
	"PKN.WA",  # Orlen
	"PKO.WA",  # PKO BP
	"PZU.WA",  # PZU
	"PEO.WA",  # Bank Pekao
	"KGH.WA",  # KGHM
	"CDR.WA",  # CD Projekt
	"DNP.WA",  # Dino Polska
	"LPP.WA",  # LPP
	"ALE.WA",  # Allegro
	"CPS.WA",  # Cyfrowy Polsat
	"OPL.WA",  # Orange Polska
	"JSW.WA",  # JSW
	"CCC.WA",  # CCC
	"MBK.WA",  # mBank
	"SPL.WA",  # Santander Bank Polska
	"PGE.WA",  # PGE
	"TPE.WA",  # Tauron
	"KTY.WA",  # Kety
	"ACP.WA",  # Asseco Poland
	"LWB.WA",  # Bogdanka
]


def load_benchmark(ticker: str, start: str):
	if not ticker:
		return None
	try:
		benchmark = load_prices([ticker], start=start)
		if isinstance(benchmark, pd.DataFrame):
			benchmark = benchmark.iloc[:, 0]
		# Check if we got meaningful data (more than just a few rows)
		valid_count = benchmark.notna().sum()
		if valid_count < 50:
			print(f"[WARN] Benchmark {ticker} has only {valid_count} valid prices - skipping")
			return None
		return benchmark
	except Exception as exc:
		print(f"Could not load benchmark {ticker}: {exc}")
		return None


def create_synthetic_index(prices_df, method="equal"):
	"""Create a synthetic index from component stocks.
	
	Args:
		prices_df: DataFrame of stock prices (columns are tickers)
		method: 'equal' for equal-weight, 'price' for price-weighted
	
	Returns:
		pd.Series: Synthetic index price series
	"""
	if method == "equal": # Equal-weight: average daily return across all stocks, then cumulative
		returns = prices_df.pct_change().dropna()
		avg_returns = returns.mean(axis=1)
		synthetic = 100 * (1 + avg_returns).cumprod()
		
		first_date = prices_df.index[0]
		synthetic = pd.concat([pd.Series([100.0], index=[first_date]), synthetic])
	elif method == "price": # Price-weighted (like Dow Jones): sum of prices / divisor
		synthetic = prices_df.sum(axis=1)
		synthetic = 100 * synthetic / synthetic.iloc[0]
	else:
		raise ValueError(f"Unknown method: {method}")
	
	synthetic.name = "Synthetic Index"
	return synthetic


def make_callback(prices_df, stock_returns_m, stock_returns_s, p_m, p_s, stock_names, index_prices=None, output_dir=None, show_plots=True):
	def _callback(gen, pop, logbook):
		if show_plots: print(f"Generation {gen}: plotting Pareto front and portfolio performance")
		pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

		plot_pareto_vs_markowitz(pareto_front, stock_returns_m, stock_returns_s, p_m, p_s, output_dir=output_dir, show=show_plots)

		best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
		plot_portfolio_vs_baseline(
			prices_df[stock_names],
			np.array(best),
			index_prices=index_prices,
			title=f"{gen} portfolio vs index",
			output_dir=output_dir,
			show=show_plots,
		)

	return _callback


def parse_args():
	parser = argparse.ArgumentParser(description="Run NSGA-II portfolio optimization with periodic plots")
	parser.add_argument(
		"--tickers",
		type=str,
		help="Comma-separated tickers (default: built-in large-cap set)",
		default="",
	)
	parser.add_argument("--start-date", type=str, default="2002-01-01", help="Historical start date (YYYY-MM-DD)")
	parser.add_argument("--benchmark", type=str, default="", help="Optional benchmark ticker")
	parser.add_argument("--pop-size", type=int, default=200, help="Population size")
	parser.add_argument("--n-generations", type=int, default=80, help="Number of generations")
	parser.add_argument("--cxpb", type=float, default=0.7, help="Crossover probability")
	parser.add_argument("--mutpb", type=float, default=0.6, help="Mutation probability")
	parser.add_argument("--callback-interval", type=int, default=10, help="Plot every N generations")
	parser.add_argument("--no-plots", action="store_true", help="Don't display plots during evolution (still saves them)")
	return parser.parse_args()


def main():
	args = parse_args()
	
	# Generate experiment ID
	now = datetime.now()
	exp_id = f"experiment-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"
	output_dir = os.path.join("plots", exp_id)
	os.makedirs(output_dir, exist_ok=True)
	print(f"Saving plots to: {output_dir}")
	
	tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or DEFAULT_TICKERS
	start_date = args.start_date
	benchmark_ticker = args.benchmark
	
	# Save experiment parameters to YAML
	config = {
		"experiment_id": exp_id,
		"timestamp": now.isoformat(),
		"parameters": {
			"tickers": tickers,
			"start_date": start_date,
			"benchmark": benchmark_ticker,
			"pop_size": args.pop_size,
			"n_generations": args.n_generations,
			"crossover_prob": args.cxpb,
			"mutation_prob": args.mutpb,
			"callback_interval": args.callback_interval,
			"show_plots": not args.no_plots,
		}
	}
	config_path = os.path.join(output_dir, "config.yaml")
	with open(config_path, "w") as f:
		yaml.dump(config, f, default_flow_style=False, sort_keys=False)
	print(f"Saved configuration to: {config_path}")

	prices = load_prices(tickers, start=start_date)
	if isinstance(prices, pd.Series):
		prices = prices.to_frame(name=tickers[0])

	stats = process_returns(prices)
	stock_names = stats["valid_tickers"]
	prices = prices[stock_names]
	stock_returns_m = stats["returns_m"].loc[stock_names]
	stock_returns_s = stats["returns_s"].loc[stock_names]
	stock_covariances = stats["covariances"].loc[stock_names, stock_names]

	_, p_m, p_s = optimize_markowitz(stock_returns_m.values, stock_covariances.values, n_portfolios=500)

	toolbox = setup_deap(stock_names, stock_returns_m.values, stock_covariances.values)

	benchmark_prices = load_benchmark(benchmark_ticker, start_date)
	
	# If no valid benchmark, create synthetic index from component stocks
	if benchmark_prices is None:
		print("[INFO] Creating synthetic index from component stocks (equal-weight)")
		benchmark_prices = create_synthetic_index(prices, method="equal")
		print(f"[INFO] Synthetic index created with {len(benchmark_prices)} data points")
	else:
		valid_count = benchmark_prices.notna().sum()
		print(f"Loaded benchmark {benchmark_ticker}: {valid_count} valid prices")
		
	callback = make_callback(prices, stock_returns_m, stock_returns_s, p_m, p_s, stock_names, benchmark_prices, output_dir=output_dir, show_plots=not args.no_plots)

	pop_size = args.pop_size
	n_generations = args.n_generations
	cxpb = args.cxpb
	mutpb = args.mutpb
	callback_interval = args.callback_interval

	pop, logbook = run_nsga2(
		toolbox,
		pop_size=pop_size,
		n_generations=n_generations,
		cxpb=cxpb,
		mutpb=mutpb,
		callback=callback,
		callback_interval=callback_interval,
	)

	pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
	best = max(pareto_front, key=lambda ind: ind.fitness.values[0])
	print(f"Final best: return={best.fitness.values[0]:.4f}, risk={best.fitness.values[1]:.4f}")

	plot_pareto_vs_markowitz(pareto_front, stock_returns_m, stock_returns_s, p_m, p_s, output_dir=output_dir, show=not args.no_plots)
	plot_portfolio_vs_baseline(prices, np.array(best), index_prices=benchmark_prices, title="Final Portfolio vs Index", output_dir=output_dir, show=not args.no_plots)
	plot_final_portfolio(np.array(best), np.array(stock_names), title="Final Optimized Portfolio", output_dir=output_dir, show=not args.no_plots)


if __name__ == "__main__":
	main()
