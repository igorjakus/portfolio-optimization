import numpy as np


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


def blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Performs blend crossover between two parent portfolios.

    Args:
        parent1 (np.ndarray): Weights of the first parent portfolio
        parent2 (np.ndarray): Weights of the second parent portfolio
        alpha (float): Blend parameter (alpha >= 0). Defaults to 0.5.

    Returns:
        (np.ndarray, np.ndarray): Weights of the two offspring portfolios

    Note:
        Does NOT guarantee preservation of key constraints:
        - Sum-to-one: offspring may not sum to 1 even if both parents do
        - Min/max bounds: offspring weights may exceed min/max bounds of parents
        - Cardinality: offspring may have more non-zero weights than max_cardinality
    """
    lower_bound = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2)
    upper_bound = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2)

    offspring1 = np.random.uniform(lower_bound, upper_bound)
    offspring2 = np.random.uniform(lower_bound, upper_bound)

    return offspring1, offspring2


def dirichlet_blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    concentration_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Performs a Dirichlet-based blend crossover.

    This method combines BLX-alpha exploration with the Dirichlet distribution's
    natural ability to satisfy the sum-to-one constraint.

    Args:
        parent1 (np.ndarray): Weights of the first parent portfolio
        parent2 (np.ndarray): Weights of the second parent portfolio
        alpha (float): BLX expansion parameter. Defaults to 0.5.
        concentration_power (float): Controls how much the sampling concentrates
                                     on the parents' range widths. Defaults to 1.0.

    Returns:
        (np.ndarray, np.ndarray): Weights of the two offspring portfolios (summing to 1)

    Note:
        Let $L$ be the lower bound vector and each component must receive at least $L_i$.
        Therefore $R = 1 - \sum L_i$ is the remaining budget to be allocated.

        Let $X$ be a sample from the Dirichlet distribution ($\sum X_i = 1$ by definition).

        The offspring $W$ is calculated as $W = L + X \cdot R$.

        The sum-to-one constraint is preserved mathematically:
            $\sum W_i = \sum L_i + R \cdot \sum X_i = \sum L_i + (1 - \sum L_i) \cdot 1 = 1$.
    """
    diff = np.abs(parent1 - parent2)
    lower_bound = np.maximum(0, np.minimum(parent1, parent2) - alpha * diff)
    upper_bound = np.maximum(0, np.maximum(parent1, parent2) + alpha * diff)

    widths = np.maximum(upper_bound - lower_bound, 1e-9)

    def sample_one():
        l_sum = np.sum(lower_bound)
        if l_sum >= 1.0:
            return lower_bound / l_sum

        remaining_budget = 1.0 - l_sum
        dirichlet_weights = np.random.dirichlet(widths * concentration_power)
        return lower_bound + dirichlet_weights * remaining_budget

    offspring1 = sample_one()
    offspring2 = sample_one()

    return offspring1, offspring2
