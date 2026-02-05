"""Orchestrates NSGA-II portfolio optimization using Walk-Forward Optimization."""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from deap import tools
from loguru import logger
from tqdm import tqdm

from src.data import create_synthetic_index, load_benchmark, load_prices, process_returns
from src.evolution import run_nsga2, setup_deap
from src.plots import (
    create_evolution_gif,
    create_portfolio_gif,
    generate_wfo_factsheet,
    plot_hypervolume_evolution,
    plot_intermediate_pareto_front,
    plot_intermediate_portfolio_vs_benchmark,
)
from src.tickers import DEFAULT_TICKER_SET, TICKER_SETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II portfolio optimization with Walk-Forward Validation")
    parser.add_argument(
        "--ticker-set",
        type=str,
        choices=TICKER_SETS.keys(),
        default=DEFAULT_TICKER_SET,
        help=f"Predefined ticker set to use (default: {DEFAULT_TICKER_SET})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Data start date (YYYY-MM-DD).",
    )
    parser.add_argument("--benchmark", type=str, default="", help="Optional benchmark ticker")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size")
    parser.add_argument("--n-generations", type=int, default=20, help="Generations per rebalance")
    parser.add_argument(
        "--train-window", type=int, default=365 * 3, help="Training window size in days (e.g. 1008 = 4 years)"
    )
    parser.add_argument(
        "--rebalance-freq", type=int, default=90, help="Rebalancing frequency in days (e.g. 90 = 1 quarter)"
    )
    parser.add_argument(
        "--risk-metric",
        type=str,
        default="mdd",
        choices=["std", "mdd", "sharpe"],
        help="Risk metric: 'std' (volatility), 'mdd' (max drawdown), 'sharpe' (Sharpe ratio)",
    )
    parser.add_argument(
        "--crossover",
        type=str,
        default="arithmetic",
        choices=["arithmetic", "blend", "dirichlet"],
        help="Crossover method: 'arithmetic', 'blend', 'dirichlet'",
    )
    parser.add_argument(
        "--mutation",
        type=str,
        default="gaussian",
        choices=["gaussian", "transfer", "combined"],
        help="Mutation method: 'gaussian', 'transfer', 'combined'",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="tournament",
        choices=["tournament", "nsga2", "best", "worst"],
        help="Selection method for the genetic algorithm ('tournament', 'nsga2', 'best', 'worst')",
    )
    parser.add_argument(
        "--selection-tournsize",
        type=int,
        default=3,
        help="Tournament size for 'tournament' selection method",
    )
    parser.add_argument(
        "--callback-interval",
        type=int,
        default=1,
        help="Interval for saving intermediate Pareto front plots during evolution (e.g., 10 = save every 10 generations)",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.5,
        help="Duration of each frame in the evolution GIFs (seconds)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-smoothing", action="store_true", help="Apply moving average smoothing to prices")
    parser.add_argument("--fill-missing", action="store_true", help="Interpolate prices for missing calendar days")
    parser.add_argument(
        "--no-fill-weekends",
        action="store_true",
        help="If --fill-missing is enabled, do NOT interpolate for weekend days (only interpolate for weekdays).",
    )
    parser.add_argument(
        "--transaction-cost", type=float, default=0.0025, help="Transaction cost per rebalance (e.g. 0.0025 = 0.25%%)"
    )
    parser.add_argument(
        "--min-liquidity", type=float, default=500000.0, help="Minimum daily turnover (Price*Vol) to trade"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    def log_msg(msg: str):
        if not args.quiet:
            logger.info(msg)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    now = datetime.now()
    exp_id = f"wfo-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"
    output_dir = os.path.join("plots", exp_id)
    os.makedirs(output_dir, exist_ok=True)
    log_msg(f"Saving WFO results to: {output_dir}")

    ticker_config = TICKER_SETS[args.ticker_set]
    tickers = ticker_config["tickers"]
    default_benchmark = ticker_config["benchmark"]

    log_msg(f"Tickers: {tickers}")

    full_prices, full_volumes = load_prices(tickers, start=args.start_date)

    if isinstance(full_prices, pd.Series):
        full_prices = full_prices.to_frame(name=tickers[0])
        full_volumes = full_volumes.to_frame(name=tickers[0])

    benchmark_ticker = args.benchmark if args.benchmark else default_benchmark
    log_msg(f"[INFO] Using benchmark: {benchmark_ticker}")
    benchmark_prices = load_benchmark(benchmark_ticker, args.start_date)
    if benchmark_prices is None:
        log_msg("[INFO] Creating synthetic index from component stocks for benchmark.")
        benchmark_prices = create_synthetic_index(full_prices, method="equal")

    train_window_days = args.train_window
    rebalance_freq = args.rebalance_freq
    total_days = len(full_prices)

    if total_days < train_window_days + rebalance_freq:
        logger.error(
            f"[ERROR] Not enough data. Have {total_days} days, need at least {train_window_days + rebalance_freq}."
        )
        return

    log_msg("\n[INFO] Starting Walk-Forward Optimization")
    log_msg(f"       Window Strategy: Train on {train_window_days} days, Test/Hold for {rebalance_freq} days.")
    log_msg(f"       Risk Metric: {args.risk_metric}")
    log_msg(f"       Evolution: Crossover={args.crossover}, Mutation={args.mutation}")
    log_msg(f"       Preprocessing: Smoothing={args.use_smoothing}, Fill Missing={args.fill_missing}")
    log_msg(f"       Transaction Cost: {args.transaction_cost:.2%} per rebalance")
    log_msg(f"       Liquidity Filter: Min {args.min_liquidity:,.0f} daily")
    log_msg("-" * 60)

    current_idx = train_window_days
    last_population = None

    equity_curve_parts = {"Conservative": [], "Balanced": [], "Aggressive": []}
    portfolio_history = {"Conservative": [], "Balanced": [], "Aggressive": []}

    num_steps = (total_days - train_window_days) // rebalance_freq
    if (total_days - train_window_days) % rebalance_freq != 0:
        num_steps += 1

    wfo_progress_bar = tqdm(total=num_steps, desc="WFO Progress", dynamic_ncols=True, disable=args.quiet)

    step = 0
    while current_idx < total_days:
        step += 1

        train_start_idx = max(0, current_idx - train_window_days)
        train_prices = full_prices.iloc[train_start_idx:current_idx]
        train_volumes = full_volumes.iloc[train_start_idx:current_idx]

        test_end_idx = min(current_idx + rebalance_freq, total_days)
        test_prices = full_prices.iloc[current_idx:test_end_idx]

        if test_prices.empty:
            break

        period_start = train_prices.index[-1].date()
        period_end = test_prices.index[-1].date()

        wfo_progress_bar.set_description(
            f"WFO Progress (Step {step}: Train {train_prices.index[0].date()}->{period_start} | Test {test_prices.index[0].date()}->{period_end})"
        )

        try:
            stats = process_returns(
                train_prices,
                apply_smoothing=args.use_smoothing,
                fill_missing=args.fill_missing,
                fill_weekends=not args.no_fill_weekends,
                volume_df=train_volumes,
                min_liquidity=args.min_liquidity,
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"[WARN] Not enough valid data in this window. Skipping. Error: {e}")
            current_idx += rebalance_freq
            wfo_progress_bar.update(1)
            continue

        valid_tickers = stats["valid_tickers"]
        period_test_prices = test_prices[valid_tickers]

        stock_returns_m = stats["returns_m"].loc[valid_tickers]
        stock_covariances = stats["covariances"].loc[valid_tickers, valid_tickers]
        historical_returns = stats["historical_returns"].loc[:, valid_tickers].values

        if len(valid_tickers) < 2:
            log_msg(f"[WARN] Not enough valid tickers in window {step}. Skipping.")
            current_idx += rebalance_freq
            wfo_progress_bar.update(1)
            continue

        toolbox = setup_deap(
            valid_tickers,
            stock_returns_m.values,
            stock_covariances.values,
            historical_returns=historical_returns,
            risk_metric=args.risk_metric,
            crossover_method=args.crossover,
            mutation_method=args.mutation,
            selection_method=args.selection_method,
            selection_kwargs={"tournsize": args.selection_tournsize} if args.selection_method == "tournament" else None,
        )

        # Create output directory for intermediate plots
        step_output_dir = os.path.join(output_dir, f"step_{step:03d}_intermediate")

        # Prepare benchmark returns for the training period
        train_start_date = train_prices.index[0]
        train_end_date = train_prices.index[-1]
        benchmark_train = benchmark_prices.loc[train_start_date:train_end_date]
        benchmark_train_returns = None
        if not benchmark_train.empty and len(benchmark_train) > 1:
            benchmark_train_returns = benchmark_train.pct_change().fillna(0).values

        # Define callback to save intermediate Pareto fronts
        def save_pareto_callback(gen, pop, logbook):
            plot_intermediate_pareto_front(
                generation=gen,
                population=pop,
                output_dir=step_output_dir,
                logbook=logbook,
                risk_metric=args.risk_metric,
                step=step,
            )
            plot_intermediate_portfolio_vs_benchmark(
                generation=gen,
                population=pop,
                historical_returns=historical_returns,
                benchmark_returns=benchmark_train_returns,
                output_dir=step_output_dir,
                risk_metric=args.risk_metric,
                step=step,
            )

        pop, logbook = run_nsga2(
            toolbox,
            pop_size=args.pop_size,
            n_generations=args.n_generations,
            seed_population=last_population,
            callback=save_pareto_callback,
            callback_interval=args.callback_interval,
            verbose=False,
        )
        last_population = pop

        # Plot hypervolume evolution for this step
        plot_hypervolume_evolution(logbook, output_dir, step=step)

        # Create GIFs for this step's evolution
        pareto_gif_path = os.path.join(output_dir, f"step_{step:03d}_pareto_evolution.gif")
        create_evolution_gif(step_output_dir, pareto_gif_path, pattern="pareto_gen_*.png", duration=args.gif_duration)

        benchmark_gif_path = os.path.join(output_dir, f"step_{step:03d}_benchmark_evolution.gif")
        create_evolution_gif(
            step_output_dir, benchmark_gif_path, pattern="portfolio_vs_benchmark_gen_*.png", duration=args.gif_duration
        )

        # Cleanup intermediate frames
        import shutil

        shutil.rmtree(step_output_dir, ignore_errors=True)

        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        ind_aggressive = max(pareto_front, key=lambda ind: ind.fitness.values[0])

        if args.risk_metric == "sharpe":
            sorted_front = sorted(pareto_front, key=lambda ind: ind.fitness.values[0])
            ind_balanced = sorted_front[len(sorted_front) // 2]
            ind_conservative = min(pareto_front, key=lambda ind: ind.fitness.values[1])
        else:
            ind_conservative = min(pareto_front, key=lambda ind: ind.fitness.values[1])
            ind_balanced = max(pareto_front, key=lambda ind: ind.fitness.values[0] / (ind.fitness.values[1] + 1e-9))

        selected_inds = {"Conservative": ind_conservative, "Balanced": ind_balanced, "Aggressive": ind_aggressive}

        period_test_returns = period_test_prices.pct_change().fillna(0)

        for profile_name, ind in selected_inds.items():
            weights = np.array(ind)
            port_test_ret = period_test_returns.dot(weights)
            segment_equity = (1 + port_test_ret).cumprod()

            equity_curve_parts[profile_name].append(segment_equity)
            portfolio_history[profile_name].append({"date": period_start, "weights": weights, "tickers": valid_tickers})

        current_idx += rebalance_freq
        wfo_progress_bar.update(1)
    wfo_progress_bar.close()
    log_msg("\n[INFO] Optimization finished. Stitching performance...")

    if not equity_curve_parts["Balanced"]:
        logger.error("No results generated.")
        return

    for profile_name in ["Conservative", "Balanced", "Aggressive"]:
        log_msg(f"\nProcessing results for: {profile_name}")
        parts = equity_curve_parts[profile_name]

        full_equity = pd.Series(dtype=float)
        cumulative_factor = 1.0

        for i, segment in enumerate(parts):
            if i > 0:
                cumulative_factor *= 1 - args.transaction_cost

            scaled_segment = segment * cumulative_factor
            if full_equity.empty:
                full_equity = scaled_segment
            else:
                full_equity = pd.concat([full_equity, scaled_segment])
            cumulative_factor = scaled_segment.iloc[-1]

        start_date_test = full_equity.index[0]
        end_date_test = full_equity.index[-1]
        bench_test = benchmark_prices.loc[start_date_test:end_date_test]
        if not bench_test.empty:
            bench_test = bench_test / bench_test.iloc[0] * full_equity.iloc[0]

        profile_dir = os.path.join(output_dir, profile_name.lower())
        os.makedirs(profile_dir, exist_ok=True)

        generate_wfo_factsheet(
            full_equity,
            bench_test,
            args.risk_metric,
            args.train_window,
            args.rebalance_freq,
            portfolio_history[profile_name],
            ticker_set_name=f"{args.ticker_set} ({profile_name})",
            benchmark_name=benchmark_ticker,
            output_dir=profile_dir,
            show=False,
        )

        create_portfolio_gif(portfolio_history[profile_name], profile_dir)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    logger.info(f"Done! All results saved to {output_dir}")


if __name__ == "__main__":
    main()
