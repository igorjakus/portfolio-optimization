import numpy as np
import pandas as pd


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalizes portfolio weights to sum to 1.

    Args:
        weights: Portfolio weights, shape (pop_size, n_assets).

    Returns:
        np.ndarray: Normalized portfolio weights, shape (pop_size, n_assets).

    Raises:
        ValueError: If sum of weights is zero.
    """
    total = np.sum(weights, axis=1, keepdims=True)
    if np.any(total == 0):
        raise ValueError("Sum of weights is zero, cannot normalize.")
    return weights / total


def is_valid_portfolio(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    max_cardinality: int | None = None,
    epsilon: float = 1e-6,
) -> bool:
    """Checks if the portfolio weights sum to 1 and satisfy all constraints.

    Args:
        weights: Portfolio weights.
        min_weight: Minimum allowed weight for non-zero assets.
        max_weight: Maximum allowed weight for each asset.
        max_cardinality: Maximum number of assets in portfolio.
        epsilon: Tolerance for treating weights as zero.

    Returns:
        bool: True if valid, False otherwise.

    Notes:
        Weights with |w| <= epsilon are treated as "not selected" and exempt from min/max checks.
        Only |weights| > epsilon are validated against min_weight and max_weight constraints.
    """
    nonzero_mask = np.abs(weights) > epsilon

    return bool(
        np.isclose(np.sum(weights), 1.0)
        and np.all(weights[nonzero_mask] >= min_weight)
        and np.all(weights[nonzero_mask] <= max_weight)
        and (max_cardinality is None or np.sum(nonzero_mask) <= max_cardinality)
    )


def initialize_population(pop_size: int, n_assets: int, method: str = "dirichlet") -> np.ndarray:
    """Initializes a population of portfolios with random weights.

    Args:
        pop_size: Population size.
        n_assets: Number of assets in the portfolio.
        method: Method to initialize weights ('uniform' or 'dirichlet').

    Returns:
        np.ndarray: Population of portfolios (shape: pop_size x n_assets).

    Raises:
        ValueError: If initialization method is unknown.
    """
    if method == "uniform":
        population = np.random.rand(pop_size, n_assets)
        population = normalize_weights(population)
        return population
    elif method == "dirichlet":
        population = np.random.dirichlet(np.ones(n_assets), size=pop_size)
        return population
    else:
        raise ValueError("Unknown initialization method.")


def calculate_portfolio_return(weights: np.ndarray, returns_m: np.ndarray) -> float:
    """Calculates expected return of the portfolio.

    Args:
        weights: Portfolio weights.
        returns_m: Expected returns of individual assets.

    Returns:
        float: Expected return of the portfolio.
    """
    return float(np.dot(weights, returns_m))


def optimize_markowitz(
    returns_m: np.ndarray, covariances: np.ndarray, n_portfolios: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the efficient frontier using the analytical Markowitz solution.

    Args:
        returns_m: Mean returns vector.
        covariances: Covariance matrix.
        n_portfolios: Number of portfolios to generate along the frontier.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - weights: Array of portfolio weights (n_assets, n_portfolios).
            - expected_returns: Array of expected returns for the portfolios.
            - volatilities: Array of volatilities (std dev) for the portfolios.
    """
    d = returns_m.size
    s_inv = np.linalg.inv(covariances)
    ones = np.ones(d)

    A = returns_m.T.dot(s_inv.dot(ones))
    B = returns_m.T.dot(s_inv.dot(returns_m))
    C = ones.T.dot(s_inv.dot(ones))
    D = B * C - A**2

    p1 = 1 / D * (B * s_inv.dot(ones) - A * s_inv.dot(returns_m))
    p2 = 1 / D * (C * s_inv.dot(returns_m) - A * s_inv.dot(ones))

    portfolios = np.array([p1 + 0.0001 * i * p2 for i in range(n_portfolios)]).T

    p_m = portfolios.T.dot(returns_m)
    p_s = np.sqrt(np.diag(portfolios.T.dot(covariances.dot(portfolios))))

    return portfolios, p_m, p_s


def semivariance(returns: np.ndarray) -> float:
    """Calculates semivariance (variance of negative returns).

    Args:
        returns: Return rates.

    Returns:
        float: Variance of negative deviations from the mean.
    """
    mean_return = np.mean(returns)
    negative_deviations = returns[returns < mean_return] - mean_return
    return float(np.var(negative_deviations))


def maximum_drawdown(returns: np.ndarray) -> float:
    """Calculates maximum drawdown.

    Args:
        returns: Return rates.

    Returns:
        float: Maximum drawdown value (negative or zero).
    """
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    return float(np.min(drawdowns))


def sharpe_ratio(returns: np.ndarray) -> float:
    """Calculates the Sharpe ratio of a portfolio.

    Args:
        returns: Return rates.

    Returns:
        float: Sharpe ratio.
    """
    return float(np.mean(returns) / np.std(returns))


def sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    """Calculates the Sortino ratio of a portfolio.

    Args:
        returns: Return rates.
        target_return: Target return or risk-free rate.

    Returns:
        float: Sortino ratio.
    """
    mean_return = np.mean(returns)
    downside_diff = returns - target_return
    downside_diff = downside_diff[downside_diff < 0]

    if len(downside_diff) == 0:
        return float(np.inf)

    downside_deviation = np.sqrt(np.mean(downside_diff**2))
    return float((mean_return - target_return) / downside_deviation)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Calculates Compound Annual Growth Rate (CAGR).

    Args:
        equity_curve: A pandas Series of cumulative returns (equity curve).
        periods_per_year: Number of trading periods per year (e.g., 252 for daily).

    Returns:
        float: The Compound Annual Growth Rate.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    total_returns = equity_curve.iloc[-1] / equity_curve.iloc[0]
    num_years = len(equity_curve) / periods_per_year
    return float((total_returns ** (1 / num_years)) - 1)


def portfolio_std_dev(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Calculates annualized standard deviation (volatility) of daily returns.

    Args:
        equity_curve: A pandas Series of cumulative returns (equity curve).
        periods_per_year: Number of trading periods per year (e.g., 252 for daily).

    Returns:
        float: Annualized standard deviation.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    daily_returns = equity_curve.pct_change().dropna()
    return float(np.std(daily_returns) * np.sqrt(periods_per_year))


def correlation_with_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculates Pearson correlation between two series of returns.

    Args:
        portfolio_returns: Series of daily returns for the portfolio.
        benchmark_returns: Series of daily returns for the benchmark.

    Returns:
        float: Pearson correlation coefficient, or 0.0 if not enough data.
    """
    if portfolio_returns.empty or benchmark_returns.empty or len(portfolio_returns) < 2:
        return 0.0

    aligned_returns = pd.DataFrame({"portfolio": portfolio_returns, "benchmark": benchmark_returns}).dropna()
    if len(aligned_returns) < 2:
        return 0.0

    return aligned_returns["portfolio"].corr(aligned_returns["benchmark"])


def calculate_turnover(portfolio_history: list[dict], total_portfolio_value: float = 1.0) -> float:
    """Calculates average portfolio turnover.

    Args:
        portfolio_history: List of dicts with keys 'date', 'weights', 'tickers' for each rebalance.
        total_portfolio_value: Assumed total value of the portfolio for calculation (e.g., $1).

    Returns:
        float: Average portfolio turnover percentage.
    """
    if len(portfolio_history) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(portfolio_history)):
        prev_weights_dict = {
            t: w for t, w in zip(portfolio_history[i - 1]["tickers"], portfolio_history[i - 1]["weights"])
        }
        curr_weights_dict = {t: w for t, w in zip(portfolio_history[i]["tickers"], portfolio_history[i]["weights"])}

        all_tickers = sorted(list(set(prev_weights_dict.keys()) | set(curr_weights_dict.keys())))
        prev_weights_aligned = np.array([prev_weights_dict.get(t, 0.0) for t in all_tickers])
        curr_weights_aligned = np.array([curr_weights_dict.get(t, 0.0) for t in all_tickers])

        turnover = 0.5 * np.sum(np.abs(curr_weights_aligned - prev_weights_aligned))
        turnovers.append(turnover)

    return np.mean(turnovers) if turnovers else 0.0


def calculate_rolling_annual_returns(equity_curve: pd.Series, periods_per_year: int = 252) -> tuple[float, float]:
    """Calculates the best and worst rolling 1-year returns.

    Args:
        equity_curve: A pandas Series of cumulative returns.
        periods_per_year: Number of trading periods per year.

    Returns:
        tuple[float, float]: (Best 1-Year Return, Worst 1-Year Return).
    """
    if len(equity_curve) < periods_per_year:
        return 0.0, 0.0

    rolling_returns = equity_curve.pct_change(periods=periods_per_year).dropna()

    if rolling_returns.empty:
        return 0.0, 0.0

    return rolling_returns.max(), rolling_returns.min()


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Calculates the maximum duration of a drawdown in trading days.

    Args:
        equity_curve: A pandas Series of cumulative returns.

    Returns:
        int: Maximum number of days in drawdown.
    """
    if equity_curve.empty:
        return 0

    high_water_mark = equity_curve.cummax()
    is_drawdown = equity_curve < high_water_mark

    if not is_drawdown.any():
        return 0

    groups = (is_drawdown != is_drawdown.shift()).cumsum()
    drawdown_groups = groups[is_drawdown]

    if drawdown_groups.empty:
        return 0

    durations = drawdown_groups.value_counts()
    return durations.max()
