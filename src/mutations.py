import numpy as np

from src.utils import normalize_weights


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

    mutation_mask = np.random.rand(pop_size) < mutation_rate
    mutation_indices = np.where(mutation_mask)[0]

    if len(mutation_indices) == 0:
        return population

    noise = np.random.normal(0, sigma, (len(mutation_indices), n_assets))
    population[mutation_indices] += noise
    population[mutation_indices] = np.maximum(population[mutation_indices], 0)
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
    population = population.copy()
    pop_size, n_assets = population.shape

    mutation_mask = np.random.rand(pop_size) < mutation_rate
    mutation_indices = np.where(mutation_mask)[0]
    n_mutations = len(mutation_indices)

    if n_mutations == 0:
        return population

    asset1_indices = np.random.randint(0, n_assets, size=n_mutations)
    asset2_indices = np.random.randint(0, n_assets, size=n_mutations)

    while np.any(asset1_indices == asset2_indices):
        bad_mask = asset1_indices == asset2_indices
        asset2_indices[bad_mask] = np.random.randint(0, n_assets, size=np.sum(bad_mask))

    temp = population[mutation_indices, asset1_indices].copy()
    population[mutation_indices, asset1_indices] = population[mutation_indices, asset2_indices]
    population[mutation_indices, asset2_indices] = temp

    return population


def transfer_mutation(population: np.ndarray, mutation_rate: float = 0.2, flow_amount: float = 0.05) -> np.ndarray:
    """Applies transfer mutation to portfolio weights.
    Transfers a small amount of weight from one asset to another.

    Args:
        population (np.ndarray): Population of portfolios (shape: pop_size x n_assets)
        mutation_rate (float): Probability of mutation for each individual. Defaults to 0.2.
        flow_amount (float): Amount of weight to transfer between assets. Defaults to 0.05.


    Returns:
        np.ndarray: Mutated population (new copy)

    Note:
        Preserves sum-to-one constraint.
        Doesn't preserve min-weight, max-weight and max-cardinality contraints.
    """
    population = population.copy()
    pop_size, n_assets = population.shape

    mutation_mask = np.random.rand(pop_size) < mutation_rate
    mutation_indices = np.where(mutation_mask)[0]
    n_mutations = len(mutation_indices)

    if n_mutations == 0:
        return population

    asset_from = np.random.randint(0, n_assets, size=n_mutations)
    asset_to = np.random.randint(0, n_assets, size=n_mutations)

    while np.any(asset_from == asset_to):
        bad_mask = asset_from == asset_to
        asset_to[bad_mask] = np.random.randint(0, n_assets, size=np.sum(bad_mask))

    actual_flow = np.minimum(flow_amount, population[mutation_indices, asset_from])
    population[mutation_indices, asset_from] -= actual_flow
    population[mutation_indices, asset_to] += actual_flow

    return population
