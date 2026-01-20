import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import clear_output
import os

import deap.tools as tools


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
    # expected return
    plt.figure(figsize=(14, 6))
    plt.bar(stock_returns_m.index, 100.0 * stock_returns_m, color="steelblue", alpha=0.8)
    plt.title("Expected Return Rate (%)", fontsize=14)
    plt.xlabel("Ticker", fontsize=12)
    plt.ylabel("Return (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # variance
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

    # Dominated
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

    # Non-dominated
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

    # Adding stock labels
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

    # Top-left: Variance vs Return (full range)
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

    # Top-right: Variance vs Return (zoomed)
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

    # Bottom-left: Volatility vs Return (full range)
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

    # Bottom-right: Volatility vs Return (zoomed)
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

    plt.suptitle("Markowitz Efficient Frontier - Multiple Views", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.show()


def plot_portfolio_composition(weights, names, title="Skład portfela"):
    """Plots the composition of a single portfolio."""
    mask = np.abs(weights) > 0.01
    filtered_weights = weights[mask]
    filtered_names = names[mask]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = sns.color_palette("husl", len(filtered_weights))
    bars = ax.bar(filtered_names, filtered_weights * 100, color=colors, edgecolor="black", linewidth=1)

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
    bars = ax.bar(filtered_names, filtered_weights * 100, color=colors, edgecolor="black", linewidth=1.5)

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


def plot_pareto_vs_markowitz(final_pop, stock_returns_m, stock_returns_s, p_m, p_s, output_dir=None, show=True):
    """Visualizes the Pareto front found by NSGA-II against Markowitz efficient frontier."""

    pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
    pareto_returns = np.array([ind.fitness.values[0] for ind in pareto_front])
    pareto_risks = np.array([ind.fitness.values[1] for ind in pareto_front])

    sort_idx = np.argsort(pareto_returns)
    pareto_returns_sorted = pareto_returns[sort_idx]
    pareto_risks_sorted = pareto_risks[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === Left: Return vs Risk (Volatility) ===
    axes[0].scatter(
        stock_returns_s,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[0].plot(p_s, p_m, color="#2ecc71", linewidth=2.5, label="Markowitz Efficient Frontier")
    axes[0].scatter(
        pareto_risks_sorted,
        pareto_returns_sorted,
        color="#e74c3c",
        s=150,
        alpha=0.8,
        edgecolors="darkred",
        label="NSGA-II Pareto Front",
    )
    axes[0].set_xlabel("Standard Deviation of Return (Risk)", fontweight="bold")
    axes[0].set_ylabel("Expected Return Rate", fontweight="bold")
    axes[0].set_title("NSGA-II Pareto Front vs Markowitz EF", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # === Right: Zoomed view ===
    axes[1].scatter(
        stock_returns_s,
        stock_returns_m,
        color="#3498db",
        s=100,
        alpha=0.6,
        edgecolors="black",
        label="Individual stocks",
    )
    axes[1].plot(p_s, p_m, color="#2ecc71", linewidth=2.5, label="Markowitz Efficient Frontier")
    axes[1].scatter(
        pareto_risks_sorted,
        pareto_returns_sorted,
        color="#e74c3c",
        s=150,
        alpha=0.8,
        edgecolors="darkred",
        label="NSGA-II Pareto Front",
    )
    axes[1].set_xlim([0, 0.035])
    axes[1].set_ylim([-0.005, 0.008])
    axes[1].set_xlabel("Standard Deviation of Return (Risk)", fontweight="bold")
    axes[1].set_ylabel("Expected Return Rate", fontweight="bold")
    axes[1].set_title("NSGA-II vs Markowitz (Zoomed)", fontweight="bold")
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

    return pareto_front, pareto_returns_sorted, pareto_risks_sorted


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
    bars = ax1.bar(filtered_names, filtered_weights * 100, color=colors, edgecolor="black", linewidth=1)

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

    # Risk-Return profile
    pareto_returns = [ind.fitness.values[0] for ind in pareto_front]
    pareto_risks = [ind.fitness.values[1] for ind in pareto_front]
    ax2.scatter(
        pareto_risks, pareto_returns, color="#e74c3c", s=100, alpha=0.6, edgecolors="darkred", label="Pareto Front"
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


def plot_portfolio_vs_baseline(prices_df, portfolio_weights, index_prices=None, title="Portfolio vs Index", output_dir=None, show=True):
    """Plots cumulative performance of a chosen portfolio versus an index benchmark.

    Args:
        prices_df (pd.DataFrame): Price history for the assets used in the portfolio (columns are tickers).
        portfolio_weights (array-like): Weights for the selected portfolio, aligned with ``prices_df`` columns.
        index_prices (pd.Series, optional): Optional benchmark index price history to compare against.
        title (str, optional): Plot title. Defaults to "Portfolio vs Index".
        output_dir (str, optional): Directory to save the plot. If provided, plot is saved as PNG.
        show (bool, optional): Whether to display the plot. Defaults to True.

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
        # Align benchmark to the same date range and compute cumulative growth
        index_series = pd.Series(index_prices)
        # Reindex to portfolio dates, forward-fill missing
        index_aligned = index_series.reindex(prices_df.index).ffill().bfill()
        # Compute returns starting from the first valid return date
        index_returns = index_aligned.pct_change().reindex(returns.index).fillna(0)
        # Normalize to start at 1 (same as portfolio)
        index_cumulative = (1 + index_returns).cumprod()
        curves["Index"] = index_cumulative.values

    plt.figure(figsize=(12, 6))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], linewidth=2, label=col)

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
