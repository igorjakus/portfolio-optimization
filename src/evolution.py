import numpy as np
from typing import Any, Iterable
from collections.abc import Callable
from deap import base, creator, tools, algorithms
from tqdm import tqdm
from numpy.typing import NDArray
from src.utils import maximum_drawdown, normalize_weights, sharpe_ratio
from src.mutations import gaussian_mutation, swap_mutation, transfer_mutation
from src.crossovers import arithmetic_crossover, blend_crossover, dirichlet_blend_crossover


class FitnessMulti(base.Fitness):
    """Custom fitness class with hashability support for DEAP."""

    weights = (1.0, -1.0)


def evaluate_portfolio(
    portfolio: np.ndarray,
    returns_m: np.ndarray,
    covariances: np.ndarray,
    hist_returns: np.ndarray,
    metric: str,
) -> tuple[float, float]:
    """Evaluates a portfolio based on return and a specified risk metric.

    Args:
        portfolio: Array of portfolio weights.
        returns_m: Array of mean returns for assets.
        covariances: Covariance matrix of asset returns.
        hist_returns: Matrix of historical returns (n_days, n_assets).
        metric: Risk metric to optimize ('std', 'mdd', 'sharpe').

    Returns:
        tuple[float, float]: A tuple containing (expected_return, risk_score).
        Note that for 'sharpe', the risk_score is negative Sharpe ratio (to be minimized).
    """
    if np.any(np.isnan(portfolio)) or np.any(np.isinf(portfolio)):
        return float("nan"), float("nan")

    weight_sum = np.sum(portfolio)
    if not np.isclose(weight_sum, 1.0):
        portfolio = portfolio / weight_sum

    portfolio_return = np.dot(portfolio, returns_m)

    if metric == "std":
        portfolio_variance = portfolio @ covariances @ portfolio
        risk_value = np.sqrt(max(0, portfolio_variance))
    elif metric == "mdd":
        portfolio_returns = hist_returns @ portfolio
        risk_value = -maximum_drawdown(portfolio_returns)
    elif metric == "sharpe":
        portfolio_returns = hist_returns @ portfolio
        sr = sharpe_ratio(portfolio_returns)
        risk_value = -sr
    else:
        raise ValueError(f"Unknown risk metric: {metric}")

    return portfolio_return, risk_value


def mutate_wrapper(
    individual: NDArray[np.float64],
    method: str = "gaussian",
    gaussian_rate: float = 0.3,
    gaussian_sigma: float = 0.08,
    swap_rate: float = 0.15,
    transfer_amount: float = 0.05,
) -> tuple[np.ndarray]:
    """Wrapper for mutation operations on an individual.

    Args:
        individual: The individual (portfolio weights) to mutate.
        method: Mutation method ('gaussian', 'transfer', 'combined').
        gaussian_rate: Probability of applying Gaussian mutation.
        gaussian_sigma: Standard deviation for Gaussian mutation.
        swap_rate: Probability of swapping weights between assets.
        transfer_amount: Amount to transfer in transfer mutation.

    Returns:
        tuple[np.ndarray]: Tuple containing the mutated individual.
    """
    mutated = individual.reshape(1, -1).copy()

    if method == "gaussian":
        mutated = gaussian_mutation(mutated, mutation_rate=gaussian_rate, sigma=gaussian_sigma)
        mutated = swap_mutation(mutated, mutation_rate=swap_rate)
    elif method == "transfer":
        mutated = transfer_mutation(mutated, mutation_rate=gaussian_rate, flow_amount=transfer_amount)
    elif method == "combined":
        if np.random.random() < 0.5:
            mutated = transfer_mutation(mutated, mutation_rate=gaussian_rate, flow_amount=transfer_amount)
        else:
            mutated = gaussian_mutation(mutated, mutation_rate=gaussian_rate, sigma=gaussian_sigma)

        mutated = swap_mutation(mutated, mutation_rate=swap_rate)
    else:
        raise ValueError(f"Unknown mutation method: {method}")

    individual[:] = mutated[0]
    if hasattr(individual.fitness, "values"):
        del individual.fitness.values
    return (individual,)


def crossover_wrapper(
    parent1: NDArray[np.float64], parent2: NDArray[np.float64], method: str = "arithmetic", alpha: float = 0.5
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wrapper for crossover operations between two parents.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.
        method: Crossover method ('arithmetic', 'blend', 'dirichlet').
        alpha: Mixing parameter.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two offspring individuals.
    """
    p1 = np.array(parent1)
    p2 = np.array(parent2)

    if method == "arithmetic":
        offspring1, offspring2 = arithmetic_crossover(p1, p2, alpha=alpha)
    elif method == "blend":
        offspring1, offspring2 = blend_crossover(p1, p2, alpha=alpha)
    elif method == "dirichlet":
        offspring1, offspring2 = dirichlet_blend_crossover(p1, p2, alpha=alpha, concentration_power=1.0)
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    offspring1 = normalize_weights(offspring1.reshape(1, -1))[0]
    offspring2 = normalize_weights(offspring2.reshape(1, -1))[0]

    parent1[:] = offspring1
    parent2[:] = offspring2
    if hasattr(parent1.fitness, "values"):
        del parent1.fitness.values
    if hasattr(parent2.fitness, "values"):
        del parent2.fitness.values
    return parent1, parent2


def setup_deap(
    stock_names: list[str],
    stock_returns_m: np.ndarray,
    stock_covariances: np.ndarray,
    historical_returns: np.ndarray | None = None,
    risk_metric: str = "std",
    crossover_method: str = "arithmetic",
    mutation_method: str = "gaussian",
    selection_method: str = "tournament",
    mutation_kwargs: dict[str, Any] | None = None,
    crossover_kwargs: dict[str, Any] | None = None,
    selection_kwargs: dict[str, Any] | None = None,
) -> base.Toolbox:
    """Initializes DEAP toolbox with multiobjective setup.

    Args:
        stock_names: List of stock ticker names.
        stock_returns_m: Mean returns for each stock.
        stock_covariances: Covariance matrix of stock returns.
        historical_returns: Historical returns array (n_days, n_assets). Required for 'mdd'/'sharpe'.
        risk_metric: Risk metric to use - 'std' (volatility), 'mdd' (max drawdown), or 'sharpe'.
        crossover_method: Crossover method name.
        mutation_method: Mutation method name.
        selection_method: Selection method name ('tournament', 'nsga2', 'best', 'worst').
        mutation_kwargs: Dictionary of kwargs for the mutation function.
        crossover_kwargs: Dictionary of kwargs for the crossover function.
        selection_kwargs: Dictionary of kwargs for the selection function.

    Returns:
        base.Toolbox: Configured DEAP toolbox ready for evolution.

    Raises:
        ValueError: If risk_metric requires historical_returns but none provided, or if an unknown
                    selection method is provided.
    """
    if risk_metric in ("mdd", "sharpe") and historical_returns is None:
        raise ValueError(f"historical_returns required for risk_metric='{risk_metric}'")

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    def create_individual(n_assets: int):
        ind = creator.Individual(np.random.dirichlet(np.ones(n_assets)))
        return ind

    toolbox.register("individual", create_individual, n_assets=len(stock_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate_portfolio,
        returns_m=stock_returns_m,
        covariances=stock_covariances,
        hist_returns=historical_returns,
        metric=risk_metric,
    )

    mut_kwargs = mutation_kwargs or {}
    mut_kwargs["method"] = mutation_method
    toolbox.register("mutate", mutate_wrapper, **mut_kwargs)

    cx_kwargs = crossover_kwargs or {}
    cx_kwargs["method"] = crossover_method
    toolbox.register("mate", crossover_wrapper, **cx_kwargs)

    SELECTION_METHODS: dict[str, Callable] = {
        "tournament": tools.selTournament,
        "nsga2": tools.selNSGA2,
        "best": tools.selBest,
        "worst": tools.selWorst,
    }

    if selection_method not in SELECTION_METHODS:
        raise ValueError(
            f"Unknown selection method: {selection_method}. Supported methods are {list(SELECTION_METHODS.keys())}"
        )

    selection_func = SELECTION_METHODS[selection_method]
    sel_kwargs = selection_kwargs or {}

    if selection_method == "tournament":
        sel_kwargs.setdefault("tournsize", 3)

    toolbox.register("select", selection_func, **sel_kwargs)

    return toolbox


def run_nsga2(
    toolbox: base.Toolbox,
    pop_size: int = 200,
    n_generations: int = 150,
    cxpb: float = 0.7,
    mutpb: float = 0.6,
    callback: Callable | None = None,
    callback_interval: int = 10,
    seed_population: list[Any] | None = None,
    verbose: bool = True,
) -> tuple[list[Any], tools.Logbook]:
    """Runs NSGA-II algorithm.

    Args:
        toolbox: DEAP toolbox with registered operators.
        pop_size: Size of the population.
        n_generations: Number of generations to run.
        cxpb: Probability of crossover.
        mutpb: Probability of mutation.
        callback: Optional callback function called every `callback_interval`.
        callback_interval: Interval for calling the callback.
        seed_population: Optional population from a previous run (Warm Start).
        verbose: Whether to show progress bar.

    Returns:
        tuple[list[Any], tools.Logbook]: The final population and the logbook with statistics.
    """
    if seed_population is None:
        pop = toolbox.population(n=pop_size)
    else:
        dummy = toolbox.individual()
        expected_size = len(dummy)

        valid_seed = []
        if len(seed_population) > 0 and len(seed_population[0]) == expected_size:
            valid_seed = [toolbox.clone(ind) for ind in seed_population]
            for ind in valid_seed:
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values
            if verbose:
                print(f"[INFO] Warm Start: Reusing {len(valid_seed)} individuals from previous period.")
        else:
            if len(seed_population) > 0 and verbose:
                print(
                    f"[WARN] Seed population dimension mismatch (Expected {expected_size}, got {len(seed_population[0])}). Starting fresh."
                )

        n_needed = pop_size - len(valid_seed)
        if n_needed > 0:
            new_inds = toolbox.population(n=n_needed)
            pop = valid_seed + new_inds
        else:
            pop = valid_seed[:pop_size]

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = tools.selNSGA2(pop, len(pop))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_return", lambda x: np.mean([f[0] for f in x]))
    stats.register("avg_risk", lambda x: np.mean([f[1] for f in x]))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg_return", "avg_risk"]

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    generations_iterable: Iterable[int] = range(1, n_generations + 1)
    if verbose:
        generations_iterable = tqdm(generations_iterable, desc="Generations", total=n_generations)

    for gen in generations_iterable:
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop = tools.selNSGA2(pop + offspring, pop_size)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if callback is not None and gen % callback_interval == 0:
            try:
                callback(gen, pop, logbook)
            except Exception as exc:
                print(f"Callback failed at gen {gen}: {exc}")

    return pop, logbook
