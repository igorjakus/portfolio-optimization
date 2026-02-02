import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import imageio.v2 as imageio
from IPython.display import clear_output
import os
from loguru import logger

import deap.tools as tools
from src.utils import (
    cagr,
    portfolio_std_dev,
    correlation_with_benchmark,
    calculate_turnover,
    sharpe_ratio,
    sortino_ratio,
    semivariance,
    calculate_rolling_annual_returns,
    max_drawdown_duration,
)


def plot_close_price(stock_name, prices):
    """Plots closing prices for a given stock."""
    ts = prices[stock_name].dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts.values, color="steelblue", linewidth=2)
    plt.title(f"Notowania: {stock_name}", fontsize=14)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Cena zamknięcia", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_returns_and_risk(stock_returns_m, stock_returns_s):
    """Plots expected returns and risk (variance) for stocks."""
    plt.figure(figsize=(14, 6))
    plt.bar(stock_returns_m.index, 100.0 * stock_returns_m, color="steelblue", alpha=0.8)
    plt.title("Expected Return Rate (%)", fontsize=14)
    plt.xlabel("Ticker", fontsize=12)
    plt.ylabel("Return (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(stock_returns_s.index, stock_returns_s**2, color="indianred", alpha=0.8)
    plt.title("Variance of Return Rate (Risk)", fontsize=14)
    plt.xlabel("Ticker", fontsize=12)
    plt.ylabel("Variance", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_correlations(stock_correlations):
    """Plots correlation heatmap."""
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(stock_correlations, cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_xticks(range(len(stock_correlations.columns)))
    ax.set_yticks(range(len(stock_correlations.columns)))
    ax.set_xticklabels(stock_correlations.columns, rotation=90)
    ax.set_yticklabels(stock_correlations.columns)
    ax.set_title("Correlation of Return Rates", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_dominance(stock_returns_m, stock_returns_s):
    """Plots dominated vs non-dominated stocks."""
    is_dominated = np.zeros(stock_returns_m.size)
    for i in range(stock_returns_m.size):
        for j in range(stock_returns_m.size):
            if (
                (i != j)
                and (stock_returns_m.iloc[i] <= stock_returns_m.iloc[j])
                and (stock_returns_s.iloc[i] > stock_returns_s.iloc[j])
            ):
                is_dominated[i] = 1
                break

    fig, ax = plt.subplots(figsize=(13, 8))

    ax.scatter(
        stock_returns_s[is_dominated == 1],
        100.0 * stock_returns_m[is_dominated == 1],
        color="#cccccc",
        alpha=0.6,
        label="Dominated stocks",
        s=120,
        edgecolors="black",
        linewidth=1,
    )

    ax.scatter(
        stock_returns_s[is_dominated == 0],
        100.0 * stock_returns_m[is_dominated == 0],
        color="#e74c3c",
        alpha=0.85,
        label="Non-dominated stocks",
        s=200,
        edgecolors="black",
        linewidth=1.5,
    )

    for i, txt in enumerate(stock_returns_m.index):
        ax.annotate(
            txt,
            (stock_returns_s.iloc[i], 100.0 * stock_returns_m.iloc[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.3, edgecolor="none"),
        )

    ax.set_title("WIG20: Risk vs Return Analysis", fontweight="bold")
    ax.set_xlabel("Standard Deviation of Return Rate (Risk)", fontweight="bold")
    ax.set_ylabel("Expected Return Rate (%)", fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_efficient_frontier(stock_returns_m, stock_returns_s, p_m, p_s):
    """Visualizes the Markowitz efficient frontier."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].scatter(
        stock_returns_s**2,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[0, 0].plot(p_s**2, p_m, color="#e74c3c", linewidth=3, label="Efficient frontier")
    axes[0, 0].set_xlabel("Variance of Return Rate")
    axes[0, 0].set_ylabel("Expected Return Rate")
    axes[0, 0].set_title("Efficient Frontier (Variance)", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].scatter(
        stock_returns_s**2,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[0, 1].plot(p_s**2, p_m, color="#e74c3c", linewidth=3, label="Efficient frontier")
    axes[0, 1].set_xlim([0, 0.0016])
    axes[0, 1].set_ylim([-0.01, 0.01])
    axes[0, 1].set_xlabel("Variance of Return Rate")
    axes[0, 1].set_ylabel("Expected Return Rate")
    axes[0, 1].set_title("Efficient Frontier (Zoomed - Variance)", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].scatter(
        stock_returns_s,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[1, 0].plot(p_s, p_m, color="#e74c3c", linewidth=3, label="Efficient frontier")
    axes[1, 0].set_xlabel("Standard Deviation of Return Rate (Volatility)")
    axes[1, 0].set_ylabel("Expected Return Rate")
    axes[1, 0].set_title("Efficient Frontier (Volatility)", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].scatter(
        stock_returns_s,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[1, 1].plot(p_s, p_m, color="#e74c3c", linewidth=3, label="Efficient frontier")
    axes[1, 1].set_xlim([0, 0.04])
    axes[1, 1].set_ylim([-0.01, 0.01])
    axes[1, 1].set_xlabel("Standard Deviation of Return Rate (Volatility)")
    axes[1, 1].set_ylabel("Expected Return Rate")
    axes[1, 1].set_title("Efficient Frontier (Zoomed - Volatility)", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(
        "Markowitz Efficient Frontier - Multiple Views",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()


def plot_portfolio_composition(weights, names, title="Skład portfela"):
    """Plots the composition of a single portfolio."""
    mask = np.abs(weights) > 0.01
    filtered_weights = weights[mask]
    filtered_names = names[mask]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("husl", len(filtered_weights))
    bars = ax.bar(
        filtered_names,
        filtered_weights * 100,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel("Stock (Ticker)", fontweight="bold")
    ax.set_ylabel("Weight (%)", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_final_portfolio(weights, names, title="Final Optimized Portfolio", output_dir=None, show=True):
    """Plots the composition of the final optimized portfolio."""
    mask = np.abs(weights) > 0.01
    filtered_weights = weights[mask]
    filtered_names = names[mask]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("husl", len(filtered_weights))
    bars = ax.bar(
        filtered_names,
        filtered_weights * 100,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xlabel("Stock (Ticker)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Weight (%)", fontweight="bold", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "final_portfolio_composition.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_summary(
    prices_df,
    portfolio_weights,
    stock_names,
    index_prices=None,
    title="Portfolio Summary",
    output_dir=None,
    show=True,
    filename="performance_summary.png",
    test_start_date=None,
):
    """
    Plots cumulative performance (Left) and portfolio composition (Right) side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    prices_df = pd.DataFrame(prices_df).copy()
    weights = np.asarray(portfolio_weights, dtype=float)

    returns = prices_df.pct_change().dropna()
    portfolio_curve = (1 + returns.values.dot(weights)).cumprod()

    curves = pd.DataFrame({"Portfolio": portfolio_curve}, index=returns.index)

    if index_prices is not None:
        index_series = pd.Series(index_prices)
        index_aligned = index_series.reindex(prices_df.index).ffill().bfill()
        index_returns = index_aligned.pct_change().reindex(returns.index).fillna(0)
        index_cumulative = (1 + index_returns).cumprod()
        curves["Index"] = index_cumulative.values

    for col in curves.columns:
        ax1.plot(curves.index, curves[col], linewidth=2, label=col)

    if test_start_date is not None and test_start_date in curves.index:
        ax1.axvline(test_start_date, color="red", linestyle="--", linewidth=1.5, label="Start of Test Period")
        ax1.axvspan(test_start_date, curves.index[-1], color="red", alpha=0.05)
        ax1.text(
            test_start_date,
            ax1.get_ylim()[1],
            " Out-of-Sample",
            color="red",
            ha="left",
            va="top",
            rotation=90,
            fontweight="bold",
        )

    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title("Cumulative Return vs Benchmark", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Growth of $1", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=12)

    mask = np.abs(weights) > 0.01
    filtered_weights = weights[mask]
    filtered_names = stock_names[mask]

    colors = sns.color_palette("husl", len(filtered_weights))
    bars = ax2.bar(
        filtered_names,
        filtered_weights * 100,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.3,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_title("Portfolio Composition (>1%)", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Asset", fontsize=12)
    ax2.set_ylabel("Weight (%)", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)

    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=11)

    plt.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_vs_markowitz(
    final_pop,
    stock_returns_m,
    stock_returns_s,
    p_m,
    p_s,
    output_dir=None,
    show=True,
    risk_metric="std",
    markowitz_custom_risk=None,
    covariances=None,
):
    """Visualizes the Pareto front found by NSGA-II against Markowitz efficient frontier."""

    pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
    pareto_returns = np.array([ind.fitness.values[0] for ind in pareto_front])
    pareto_fitness_risks = np.array([ind.fitness.values[1] for ind in pareto_front])

    sort_idx = np.argsort(pareto_returns)
    pareto_returns_sorted = pareto_returns[sort_idx]
    pareto_fitness_risks_sorted = pareto_fitness_risks[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if risk_metric == "std":
        axes[0].scatter(
            stock_returns_s,
            stock_returns_m,
            color="#3498db",
            s=100,
            alpha=0.6,
            edgecolors="black",
            label="Individual stocks",
        )
        axes[0].plot(p_s, p_m, color="#2ecc71", linewidth=2.5, label="Markowitz EF")
        axes[0].scatter(
            pareto_fitness_risks_sorted,
            pareto_returns_sorted,
            color="#e74c3c",
            s=150,
            alpha=0.8,
            edgecolors="darkred",
            label="NSGA-II Pareto Front",
        )
        axes[0].set_xlabel("Standard Deviation (Risk)", fontweight="bold")
        axes[0].set_ylabel("Expected Return", fontweight="bold")
        axes[0].set_title("NSGA-II vs Markowitz (Std Dev)", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(
            stock_returns_s,
            stock_returns_m,
            color="#3498db",
            s=100,
            alpha=0.6,
            edgecolors="black",
            label="Individual stocks",
        )
        axes[1].plot(p_s, p_m, color="#2ecc71", linewidth=2.5, label="Markowitz EF")
        axes[1].scatter(
            pareto_fitness_risks_sorted,
            pareto_returns_sorted,
            color="#e74c3c",
            s=150,
            alpha=0.8,
            edgecolors="darkred",
            label="NSGA-II Pareto Front",
        )
        min_x, max_x = min(pareto_fitness_risks_sorted), max(pareto_fitness_risks_sorted)
        min_y, max_y = min(pareto_returns_sorted), max(pareto_returns_sorted)
        margin_x = (max_x - min_x) * 0.5 if max_x != min_x else 0.01
        margin_y = (max_y - min_y) * 0.5 if max_y != min_y else 0.005

        axes[1].set_xlim([max(0, min_x - margin_x), max_x + margin_x])
        axes[1].set_ylim([min_y - margin_y, max_y + margin_y])

        axes[1].set_xlabel("Standard Deviation (Risk)", fontweight="bold")
        axes[1].set_ylabel("Expected Return", fontweight="bold")
        axes[1].set_title("NSGA-II vs Markowitz (Zoomed)", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    else:
        if covariances is None:
            logger.warning("[WARN] Covariances not provided, cannot plot Pareto on Std Dev axis correctly.")
            pareto_std_devs = pareto_fitness_risks_sorted
        else:
            pareto_std_devs = []
            for ind in pareto_front:
                w = np.array(ind)
                std_dev = np.sqrt(np.dot(w.T, np.dot(covariances, w)))
                pareto_std_devs.append(std_dev)

            pareto_std_devs = np.array(pareto_std_devs)[sort_idx]

        axes[0].scatter(
            stock_returns_s,
            stock_returns_m,
            color="#3498db",
            s=100,
            alpha=0.4,
            edgecolors="black",
            label="Stocks",
        )
        axes[0].plot(p_s, p_m, color="#2ecc71", linewidth=2.5, label="Markowitz EF")
        axes[0].scatter(
            pareto_std_devs,
            pareto_returns_sorted,
            color="#e74c3c",
            s=120,
            alpha=0.8,
            edgecolors="darkred",
            label="NSGA-II (Projected)",
        )
        axes[0].set_xlabel("Standard Deviation (Risk)", fontweight="bold")
        axes[0].set_ylabel("Expected Return", fontweight="bold")
        axes[0].set_title("Comparison on Std Dev (Markowitz Home Turf)", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if markowitz_custom_risk is not None:
            axes[1].plot(
                markowitz_custom_risk,
                p_m,
                color="#2ecc71",
                linewidth=2.5,
                linestyle="--",
                label="Markowitz EF (Projected)",
            )

        axes[1].scatter(
            pareto_fitness_risks_sorted,
            pareto_returns_sorted,
            color="#e74c3c",
            s=120,
            alpha=0.8,
            edgecolors="darkred",
            label=f"NSGA-II ({risk_metric.upper()})",
        )

        metric_label = "Max Drawdown" if risk_metric == "mdd" else risk_metric.upper()
        if risk_metric == "sharpe":
            metric_label = "Negative Sharpe Ratio"

        axes[1].set_xlabel(f"{metric_label}", fontweight="bold")
        axes[1].set_ylabel("Expected Return", fontweight="bold")
        axes[1].set_title(f"Comparison on {metric_label} (Target Metric)", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "pareto_vs_markowitz.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return pareto_front, pareto_returns_sorted, pareto_fitness_risks_sorted


def plot_pareto_portfolio_composition(portfolio_idx, pareto_front, stock_names, p_m, p_s):
    """Interactive viewer for Pareto front portfolios."""
    clear_output(wait=True)
    plt.close("all")

    portfolio = pareto_front[portfolio_idx]
    portfolio_return = portfolio.fitness.values[0]
    portfolio_risk = portfolio.fitness.values[1]

    mask = portfolio > 0.01
    filtered_weights = portfolio[mask]
    filtered_names = stock_names[mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = sns.color_palette("husl", len(filtered_weights))
    bars = ax1.bar(
        filtered_names,
        filtered_weights * 100,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )

    for bar in bars:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.2,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_title(
        f"Portfolio #{portfolio_idx} (Pareto Front)\nReturn: {portfolio_return * 100:.4f}% | Risk: {portfolio_risk * 100:.4f}%",
        fontweight="bold",
    )
    ax1.set_xlabel("Stock (Ticker)", fontweight="bold")
    ax1.set_ylabel("Weight (%)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    pareto_returns = [ind.fitness.values[0] for ind in pareto_front]
    pareto_risks = [ind.fitness.values[1] for ind in pareto_front]
    ax2.scatter(
        pareto_risks,
        pareto_returns,
        color="#e74c3c",
        s=100,
        alpha=0.6,
        edgecolors="darkred",
        label="Pareto Front",
    )
    ax2.scatter(
        portfolio_risk,
        portfolio_return,
        color="#f39c12",
        s=400,
        marker="*",
        edgecolors="black",
        label=f"Current #{portfolio_idx}",
    )
    ax2.plot(p_s, p_m, color="#2ecc71", linewidth=2, alpha=0.7, label="Markowitz EF")
    ax2.set_xlabel("Risk (Volatility)", fontweight="bold")
    ax2.set_ylabel("Return", fontweight="bold")
    ax2.set_title("Position on Pareto Front", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_portfolio_vs_baseline(
    prices_df,
    portfolio_weights,
    index_prices=None,
    title="Portfolio vs Index",
    output_dir=None,
    show=True,
    test_start_date=None,
):
    """Plots cumulative performance of a chosen portfolio versus an index benchmark.

    Args:
        prices_df (pd.DataFrame): Price history for the assets used in the portfolio (columns are tickers).
        portfolio_weights (array-like): Weights for the selected portfolio, aligned with ``prices_df`` columns.
        index_prices (pd.Series, optional): Optional benchmark index price history to compare against.
        title (str, optional): Plot title. Defaults to "Portfolio vs Index".
        output_dir (str, optional): Directory to save the plot. If provided, plot is saved as PNG.
        show (bool, optional): Whether to display the plot. Defaults to True.
        test_start_date (datetime/str, optional): Date where the test period begins (draws a vertical line).

    Returns:
        pd.DataFrame: Cumulative growth series for portfolio and benchmark (if provided).
    """

    prices_df = pd.DataFrame(prices_df).copy()
    weights = np.asarray(portfolio_weights, dtype=float)
    if prices_df.shape[1] != weights.size:
        raise ValueError("prices_df columns must align with portfolio_weights length")

    weights = weights / np.sum(weights)

    returns = prices_df.pct_change().dropna()
    portfolio_curve = (1 + returns.values.dot(weights)).cumprod()

    curves = pd.DataFrame(
        {
            "Portfolio": portfolio_curve,
        },
        index=returns.index,
    )

    if index_prices is not None:
        index_series = pd.Series(index_prices)
        index_aligned = index_series.reindex(prices_df.index).ffill().bfill()
        index_returns = index_aligned.pct_change().reindex(returns.index).fillna(0)
        index_cumulative = (1 + index_returns).cumprod()
        curves["Index"] = index_cumulative.values

    plt.figure(figsize=(12, 6))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], linewidth=2, label=col)

    if test_start_date is not None and test_start_date in curves.index:
        plt.axvline(test_start_date, color="red", linestyle="--", linewidth=1.5, label="Start of Test Period")
        plt.axvspan(test_start_date, curves.index[-1], color="red", alpha=0.05)
        plt.text(
            test_start_date,
            plt.ylim()[1],
            " Out-of-Sample",
            color="red",
            ha="left",
            va="top",
            rotation=90,
            fontweight="bold",
        )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.title(title, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return curves


def _calculate_portfolio_metrics(
    equity_series: pd.Series,
    returns_series: pd.Series,
    portfolio_history: list[dict] | None = None,
    aligned_returns_with_benchmark: pd.DataFrame | None = None,
    is_benchmark: bool = False,
) -> dict:
    """Calculates key performance metrics for a given equity curve.

    Args:
        equity_series: A pandas Series of cumulative returns.
        returns_series: A pandas Series of daily returns.
        portfolio_history: List of dicts with portfolio composition history (for turnover).
        aligned_returns_with_benchmark: DataFrame with aligned portfolio and benchmark returns,
                                        used for correlation if not a benchmark itself.
        is_benchmark: True if calculating metrics for the benchmark, False for portfolio.

    Returns:
        dict: A dictionary of calculated performance metrics.
    """
    metrics = {}

    if equity_series.empty or returns_series.empty:
        return {
            "total_ret": 0.0,
            "cagr": 0.0,
            "std": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "semi_var": 0.0,
            "downside_dev": 0.0,
            "best_year": 0.0,
            "worst_year": 0.0,
            "dd_duration": 0.0,
            "turnover": 0.0,
            "corr": 0.0,
        }

    metrics["total_ret"] = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    metrics["cagr"] = cagr(equity_series)
    metrics["std"] = portfolio_std_dev(equity_series)
    metrics["mdd"] = (equity_series / equity_series.cummax() - 1).min()

    float_returns = returns_series.values.astype(float)
    metrics["sharpe"] = sharpe_ratio(float_returns)
    metrics["sortino"] = sortino_ratio(float_returns)
    metrics["semi_var"] = semivariance(float_returns)
    metrics["downside_dev"] = np.sqrt(metrics["semi_var"]) * np.sqrt(252)

    metrics["best_year"], metrics["worst_year"] = calculate_rolling_annual_returns(equity_series)
    metrics["dd_duration"] = max_drawdown_duration(equity_series)

    if not is_benchmark and portfolio_history is not None:
        metrics["turnover"] = calculate_turnover(portfolio_history)
    else:
        metrics["turnover"] = 0.0

    if not is_benchmark and aligned_returns_with_benchmark is not None and not aligned_returns_with_benchmark.empty:
        metrics["corr"] = correlation_with_benchmark(
            aligned_returns_with_benchmark["portfolio"], aligned_returns_with_benchmark["benchmark"]
        )
    else:
        metrics["corr"] = 0.0

    return metrics


def generate_wfo_factsheet(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_metric: str,
    train_window: int,
    rebalance_freq: int,
    portfolio_history: list[dict],
    ticker_set_name: str,
    benchmark_name: str,
    output_dir: str | None = None,
    show: bool = True,
):
    """Generates a comprehensive factsheet for Walk-Forward Optimization results.

    The factsheet includes:
    1. Equity curve comparison (Portfolio vs Benchmark).
    2. Drawdown curve comparison.
    3. Table of key performance metrics for both Portfolio and Benchmark.

    Args:
        portfolio_equity: A pandas Series of portfolio cumulative returns.
        benchmark_equity: A pandas Series of benchmark cumulative returns.
        risk_metric: The risk metric used for optimization (e.g., 'mdd', 'sharpe').
        train_window: Training window size in days.
        rebalance_freq: Rebalancing frequency in days.
        portfolio_history: List of dicts with portfolio composition history (for turnover).
        ticker_set_name: Name of the ticker set used (e.g. 'WIG20').
        benchmark_name: Name of the benchmark ticker (e.g. '^WIG20').
        output_dir: Directory to save the factsheet.
        show: Whether to display the plot.
    """

    pf_returns = portfolio_equity.pct_change().dropna()
    bm_returns = benchmark_equity.pct_change().dropna()

    aligned_returns = pd.DataFrame({"portfolio": pf_returns, "benchmark": bm_returns}).dropna()

    port_metrics = _calculate_portfolio_metrics(
        portfolio_equity, pf_returns, portfolio_history, aligned_returns, is_benchmark=False
    )
    bench_metrics = _calculate_portfolio_metrics(
        benchmark_equity, bm_returns, aligned_returns_with_benchmark=aligned_returns, is_benchmark=True
    )

    logger.info("\n" + "=" * 50)
    logger.info(f" WFO RESULTS SUMMARY ({risk_metric.upper()})")
    logger.info(f" Tickers: {ticker_set_name} | Benchmark: {benchmark_name}")
    logger.info("=" * 50)
    logger.info(f"{'Metric':<30} | {'Portfolio':<15} | {'Benchmark':<15}")
    logger.info("-" * 50)
    logger.info(
        f"{'Total Return':<30} | {port_metrics['total_ret'] * 100:14.2f}% | {bench_metrics['total_ret'] * 100:14.2f}%"
    )
    logger.info(
        f"{'Compound Annual Growth Rate':<30} | {port_metrics['cagr'] * 100:14.2f}% | {bench_metrics['cagr'] * 100:14.2f}%"
    )
    logger.info(f"{'Annualized Volatility':<30} | {port_metrics['std'] * 100:14.2f}% | {bench_metrics['std'] * 100:14.2f}%")
    logger.info(f"{'Max Drawdown':<30} | {port_metrics['mdd'] * 100:14.2f}% | {bench_metrics['mdd'] * 100:14.2f}%")
    logger.info(
        f"{'Best 365 Days':<30} | {port_metrics['best_year'] * 100:14.2f}% | {bench_metrics['best_year'] * 100:14.2f}%"
    )
    logger.info(
        f"{'Worst 365 Days':<30} | {port_metrics['worst_year'] * 100:14.2f}% | {bench_metrics['worst_year'] * 100:14.2f}%"
    )
    logger.info(f"{'Sharpe Ratio':<30} | {port_metrics['sharpe']:14.2f} | {bench_metrics['sharpe']:14.2f}")
    logger.info(f"{'Sortino Ratio':<30} | {port_metrics['sortino']:14.2f} | {bench_metrics['sortino']:14.2f}")
    logger.info(f"{'Semivariance':<30} | {port_metrics['semi_var']:14.2f} | {bench_metrics['semi_var']:14.2f}")
    logger.info(f"{'Correlation':<30} | {port_metrics['corr']:14.2f} | {'N/A':>15}")
    logger.info(f"{'Avg Turnover per Rebalance':<30} | {port_metrics['turnover'] * 100:14.2f}% | {'N/A':>15}")
    logger.info("=" * 50 + "\n")

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1.2], hspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(
        portfolio_equity.index,
        portfolio_equity.values.astype(float),
        label="WFO Portfolio",
        color="#2c3e50",
        linewidth=2,
    )
    if not benchmark_equity.empty:
        ax0.plot(
            benchmark_equity.index,
            benchmark_equity.values.astype(float),
            label=f"Benchmark ({benchmark_name})",
            color="#95a5a6",
            linestyle="--",
        )
    ax0.set_title(f"Walk-Forward Optimization: Equity Curve ({risk_metric.upper()})", fontweight="bold", fontsize=14)
    ax0.set_ylabel("Growth of $1", fontsize=12)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=10)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    pf_drawdown = portfolio_equity / portfolio_equity.cummax() - 1
    bm_drawdown = benchmark_equity / benchmark_equity.cummax() - 1

    ax1.plot(
        pf_drawdown.index, pf_drawdown.values.astype(float), label="Portfolio Drawdown", color="#e74c3c", linewidth=2
    )
    if not benchmark_equity.empty:
        ax1.plot(
            bm_drawdown.index,
            bm_drawdown.values.astype(float),
            label="Benchmark Drawdown",
            color="#f39c12",
            linestyle="--",
        )
    ax1.fill_between(pf_drawdown.index, pf_drawdown.values.astype(float), 0, color="#e74c3c", alpha=0.1)
    ax1.fill_between(bm_drawdown.index, bm_drawdown.values.astype(float), 0, color="#f39c12", alpha=0.05)
    ax1.set_title("Drawdown", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Drawdown (%)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=10)
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_axis_off()

    metrics_data = [
        ["Total Return", f"{port_metrics['total_ret']:.2%}", f"{bench_metrics['total_ret']:.2%}"],
        ["Comp. Annual Growth Rate", f"{port_metrics['cagr']:.2%}", f"{bench_metrics['cagr']:.2%}"],
        ["Ann. Volatility", f"{port_metrics['std']:.2%}", f"{bench_metrics['std']:.2%}"],
        ["Ann. Downside Deviation", f"{port_metrics['downside_dev']:.2%}", f"{bench_metrics['downside_dev']:.2%}"],
        ["Max Drawdown", f"{port_metrics['mdd']:.2%}", f"{bench_metrics['mdd']:.2%}"],
        ["Max DD Duration (Days)", f"{port_metrics['dd_duration']:.0f}", f"{bench_metrics['dd_duration']:.0f}"],
        ["Best 365 Days", f"{port_metrics['best_year']:.2%}", f"{bench_metrics['best_year']:.2%}"],
        ["Worst 365 Days", f"{port_metrics['worst_year']:.2%}", f"{bench_metrics['worst_year']:.2%}"],
        ["Sharpe Ratio", f"{port_metrics['sharpe']:.2f}", f"{bench_metrics['sharpe']:.2f}"],
        ["Sortino Ratio", f"{port_metrics['sortino']:.2f}", f"{bench_metrics['sortino']:.2f}"],
        ["Correlation", f"{port_metrics['corr']:.2f}", "N/A"],
        ["Avg Turnover", f"{port_metrics['turnover']:.2%}", "N/A"],
    ]
    col_labels = ["Metric", "Portfolio", "Benchmark"]

    table = ax2.table(
        cellText=metrics_data, colLabels=col_labels, loc="center", cellLoc="center", bbox=(0.0, 0.0, 1.0, 1.0)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.suptitle(
        f"Walk-Forward Optimization Report\nTicker Set: {ticker_set_name} | Benchmark: {benchmark_name} | Metric: {risk_metric.upper()}\nWindow: {train_window} days | Rebalance: {rebalance_freq} days",
        fontweight="bold",
        fontsize=16,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "wfo_factsheet.png"), dpi=200)
        logger.info(f"Saved WFO factsheet to {output_dir}/wfo_factsheet.png")

    if show:
        plt.show()
    else:
        plt.close()


def create_portfolio_gif(portfolio_history: list, output_dir: str):
    """Generates an animated GIF of portfolio composition changes over time.

    Args:
        portfolio_history: List of dicts with keys 'date', 'weights', 'tickers'.
        output_dir: Directory to save the GIF.
    """
    logger.info("[INFO] Generating portfolio evolution GIF...")
    frames = []
    temp_frames_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)

    for i, step in enumerate(portfolio_history):
        d = step["date"]
        w = step["weights"]
        t = step["tickers"]

        mask = w > 0.001
        f_w = w[mask]
        f_t = np.array(t)[mask]

        idx = np.argsort(f_w)[::-1]
        f_w = f_w[idx]
        f_t = f_t[idx]

        plt.figure(figsize=(10, 6))
        plt.ylim(0, 100)
        sns.barplot(x=f_t, y=f_w * 100, hue=f_t, palette="viridis", legend=False)
        plt.title(f"Portfolio Composition: {d}", fontweight="bold", fontsize=16)
        plt.ylabel("Weight (%)", fontsize=12)
        plt.xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        frame_path = os.path.join(temp_frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100)
        plt.close()
        frames.append(frame_path)

    gif_path = os.path.join(output_dir, "portfolio_evolution.gif")
    with imageio.get_writer(gif_path, mode="I", duration=0.5) as writer:
        for filename in frames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in frames:
        os.remove(filename)
    os.rmdir(temp_frames_dir)
    logger.info(f"Saved GIF to {gif_path}")


def plot_rolling_composition_and_correlations(portfolio_history: list[dict], output_dir: str, max_plots: int = 5):
    """Generates a side-by-side plot of portfolio composition and correlations.

    Args:
        portfolio_history: List of dicts, each containing:
                           - 'date' (datetime/date)
                           - 'weights' (np.array)
                           - 'tickers' (list[str])
                           - 'correlations' (pd.DataFrame)
        output_dir: Directory to save the plot.
        max_plots: Maximum number of periods to visualize. If len(history) > max_plots,
                   selects evenly spaced periods.
    """
    n_periods = len(portfolio_history)
    if n_periods == 0:
        return

    if n_periods <= max_plots:
        indices = list(range(n_periods))
    else:
        indices = np.linspace(0, n_periods - 1, max_plots, dtype=int).tolist()

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5 * n_rows), constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])

    for i, idx in enumerate(indices):
        data = portfolio_history[idx]
        date_str = str(data["date"])
        weights = data["weights"]
        tickers = np.array(data["tickers"])
        corr_matrix = data.get("correlations", None)

        ax_comp = axes[i, 0]

        mask = weights > 0.01
        f_weights = weights[mask]
        f_tickers = tickers[mask]

        sort_idx = np.argsort(f_weights)[::-1]
        f_weights = f_weights[sort_idx]
        f_tickers = f_tickers[sort_idx]

        if len(f_weights) > 0:
            colors = sns.color_palette("viridis", len(f_weights))
            ax_comp.bar(f_tickers, f_weights * 100, color=colors)
            ax_comp.set_title(f"Composition: {date_str}", fontweight="bold")
            ax_comp.set_ylabel("Weight (%)")
            ax_comp.tick_params(axis="x", rotation=45)
            ax_comp.grid(axis="y", alpha=0.3)
        else:
            ax_comp.text(0.5, 0.5, "No significant positions", ha="center", va="center")

        ax_corr = axes[i, 1]

        if corr_matrix is not None and not corr_matrix.empty and len(f_tickers) > 1:
            subset_corr = corr_matrix.loc[f_tickers, f_tickers]

            sns.heatmap(
                subset_corr,
                ax=ax_corr,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                annot=True,
                fmt=".2f",
                cbar=True,
                annot_kws={"size": 8},
            )
            ax_corr.set_title(f"Correlations: {date_str} (Active Assets)", fontweight="bold")
        else:
            ax_corr.text(0.5, 0.5, "Insufficient assets/data for correlation", ha="center", va="center")
            ax_corr.set_title(f"Correlations: {date_str}")

    plt.suptitle("Portfolio Evolution: Composition vs Correlations", fontsize=20, fontweight="bold", y=1.02)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "evolution_composition_correlation.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved evolution plot to {save_path}")

    plt.close()
