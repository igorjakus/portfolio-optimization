import numpy as np


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
