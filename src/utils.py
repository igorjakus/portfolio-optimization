import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalizes portfolio weights to sum to 1.

    Args:
        weights (np.ndarray): Portfolio weights, shape (pop_size, n_assets)
    Returns:
        np.ndarray: Normalized portfolio weights, shape (pop_size, n_assets)
    """
    total = np.sum(weights, axis=1, keepdims=True)
    if np.any(total == 0):
        raise ValueError("Sum of weights is zero, cannot normalize.")
    return weights / total


def is_valid_portfolio(weights: np.ndarray, min_weight=0.0, max_weight=1.0, max_cardinality=None, epsilon=1e-6):
    """Checks if the portfolio weights sum to 1 and satisfy all constraints.

    Args:
        weights (np.ndarray): Portfolio weights
        min_weight (float, optional): Minimum allowed weight for non-zero assets. Defaults to 0.0.
        max_weight (float, optional): Maximum allowed weight for each asset. Defaults to 1.0.
        max_cardinality (int, optional): Maximum number of assets in portfolio. Defaults to None.
        epsilon (float, optional): Tolerance for treating weights as zero. Defaults to 1e-6.

    Returns:
        bool: True if valid, False otherwise

    Notes:
        - Weights with |w| <= epsilon are treated as "not selected" and exempt from min/max checks
        - Such weights may be slightly negative due to numerical errors - this is acceptable
        - Only |weights| > epsilon are validated against min_weight and max_weight constraints
    """
    nonzero_mask = np.abs(weights) > epsilon

    return (
        np.isclose(np.sum(weights), 1.0)
        and np.all(weights[nonzero_mask] >= min_weight)
        and np.all(weights[nonzero_mask] <= max_weight)
        and (max_cardinality is None or np.sum(nonzero_mask) <= max_cardinality)
    )


def initialize_population(pop_size: int, n_assets: int, method="dirichlet"):
    """Initializes a population of portfolios with random weights.

    Args:
        pop_size (int): Population size
        n_assets (int): Number of assets in the portfolio
        method (str): Method to initialize weights ('uniform' or 'dirichlet')
    Returns:
        np.ndarray: Population of portfolios (shape: pop_size x n_assets)
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


def calculate_portfolio_return(weights: np.ndarray, returns_m: np.ndarray):
    """Calculates expected return of the portfolio.

    Args:
        weights (np.ndarray): Portfolio weights
        returns_m (np.ndarray): Expected returns of individual assets
    Returns:
        float: Expected return of the portfolio
    """
    return np.dot(weights, returns_m)


def optimize_markowitz(returns_m, covariances, n_portfolios=1000):
    """
    Calculates the efficient frontier using the analytical Markowitz solution.
    Returns: (weights, expected_returns, volatilities)
    """
    d = returns_m.size
    Sinv = np.linalg.inv(covariances)
    ones = np.ones(d)

    A = returns_m.T.dot(Sinv.dot(ones))
    B = returns_m.T.dot(Sinv.dot(returns_m))
    C = ones.T.dot(Sinv.dot(ones))
    D = B * C - A**2

    p1 = 1 / D * (B * Sinv.dot(ones) - A * Sinv.dot(returns_m))
    p2 = 1 / D * (C * Sinv.dot(returns_m) - A * Sinv.dot(ones))

    # Generate portfolios along the frontier
    # 0.0001 is a step for the expected return multiplier
    portfolios = np.array([p1 + 0.0001 * i * p2 for i in range(n_portfolios)]).T

    p_m = portfolios.T.dot(returns_m)
    p_s = np.sqrt(np.diag(portfolios.T.dot(covariances.dot(portfolios))))

    return portfolios, p_m, p_s


def semivariance(returns: np.ndarray):
    """Calculates semivariance

    Args:
        returns (np.ndarray): Return rates
    Returns:
        float: Variance of negative deviations from the mean
    """
    mean_return = np.mean(returns)
    negative_deviations = returns[returns < mean_return] - mean_return
    return np.var(negative_deviations)
