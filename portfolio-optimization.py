import numpy as np


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


def calculate_portfolio_return(weights: np.ndarray, returns_m: np.ndarray):
    """Calculates expected return of the portfolio.

    Args:
        weights (np.ndarray): Portfolio weights
        returns_m (np.ndarray): Expected returns of individual assets
    Returns:
        float: Expected return of the portfolio
    """
    return np.dot(weights, returns_m)


def tournament_selection(fitnesses: np.ndarray, offspring_size: int, tournament_size=3):
    """Selects individuals from the population using tournament selection.

    Args:
        fitnesses (np.ndarray): Fitness values of individuals
        offspring_size (int): Number of individuals to select
        tournament_size (int, optional): Number of individuals in each tournament. Defaults to 3.
    Returns:
        np.ndarray: Selected individuals indices
    """
    pop_size = fitnesses.shape[0]

    # generate random indices for candidates in each tournament
    candidates_indices = np.random.randint(0, pop_size, (offspring_size, tournament_size))
    candidates_fitnesses = fitnesses[candidates_indices]

    # find the winner (index of the best candidate) in each tournament
    winners_local_indices = np.argmin(candidates_fitnesses, axis=1)

    # map local winner indices back to global population indices
    selected_indices = candidates_indices[np.arange(offspring_size), winners_local_indices]
    return selected_indices


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


def gaussian_mutation(population: np.ndarray, mutation_rate: float = 0.2, sigma: float = 0.05) -> np.ndarray:
    """Applies Gaussian mutation to portfolio weights.
    Adds small random Gaussian noise to weights, then renormalizes to maintain sum=1.

    Args:
        population (np.ndarray): Population of portfolios (shape: pop_size x n_assets)
        mutation_rate (float): Probability of mutation for each individual. Defaults to 0.2.
        sigma (float): Standard deviation of Gaussian noise. Defaults to 0.05.

    Returns:
        np.ndarray: Mutated population (new copy)

    Note:
        Preserves sum-to-one constraint through renormalization.
        Doesn't preserve min-weight, max-weight and max-cardinality contraints.
    """
    population = population.copy()
    pop_size, n_assets = population.shape

    # Determine which individuals will be mutated
    mutation_mask = np.random.rand(pop_size) < mutation_rate
    mutation_indices = np.where(mutation_mask)[0]

    if len(mutation_indices) == 0:
        return population

    # Add Gaussian noise to mutated individuals
    noise = np.random.normal(0, sigma, (len(mutation_indices), n_assets))
    population[mutation_indices] += noise

    # Ensure all weights stay non-negative
    population[mutation_indices] = np.maximum(population[mutation_indices], 0)

    # Renormalize to maintain sum=1
    population[mutation_indices] = normalize_weights(population[mutation_indices])

    return population


def swap_mutation(population: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Applies swap mutation to the population (vectorized).

    Swaps the weights of two randomly selected assets for individuals that are selected for mutation.
    Returns a new mutated population without modifying the input.

    Args:
        population (np.ndarray): Population of portfolios (shape: pop_size x n_assets)
        mutation_rate (float): Probability of mutation for each individual. Defaults to 0.1.

    Returns:
        np.ndarray: Mutated population (new copy)

    Note:
        Preserves the sum-to-one constraint (swapping doesn't change total sum).
        Preserves min/max bounds and cardinality constraints *only if the input already satisfies them*.
        If input violates these constraints, swap will not fix them.
    """
    population = population.copy()  # Work on a copy to avoid modifying input
    # TODO: check whether it's detrimental to performance

    pop_size, n_assets = population.shape

    # Determine which individuals will be mutated
    mutation_mask = np.random.rand(pop_size) < mutation_rate
    mutation_indices = np.where(mutation_mask)[0]
    n_mutations = len(mutation_indices)

    if n_mutations == 0:
        return population

    # Generate pairs of random assets to swap
    asset1_indices = np.random.randint(0, n_assets, size=n_mutations)
    asset2_indices = np.random.randint(0, n_assets, size=n_mutations)

    # Ensure asset1 != asset2
    while np.any(asset1_indices == asset2_indices):
        bad_mask = asset1_indices == asset2_indices
        asset2_indices[bad_mask] = np.random.randint(0, n_assets, size=np.sum(bad_mask))

    # Perform swaps
    temp = population[mutation_indices, asset1_indices].copy()
    population[mutation_indices, asset1_indices] = population[mutation_indices, asset2_indices]
    population[mutation_indices, asset2_indices] = temp

    return population


def arithmetic_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Performs arithmetic crossover between two parent portfolios.

    Args:
        parent1 (np.ndarray): Weights of the first parent portfolio
        parent2 (np.ndarray): Weights of the second parent portfolio
        alpha (float): Crossover parameter (0 <= alpha <= 1). Defaults to 0.5.

    Returns:
        (np.ndarray, np.ndarray): Weights of the two offspring portfolios

    Note:
        Maintains key constraints when parents satisfy them:
        - Sum-to-one: if both parents sum to 1, offspring will also (since α·1 + (1-α)·1 = 1)
        - Min/max bounds: if all parent weights are in [min_w, max_w], offspring weights will be too
        - Does NOT preserve cardinality: offspring may have more non-zero weights than max_cardinality
          even if both parents respect the limit (due to convex combinations creating new non-zero entries)
    """
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    return offspring1, offspring2
