import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm
from src.utils import maximum_drawdown, normalize_weights, sharpe_ratio
from src.mutations import gaussian_mutation, swap_mutation
from src.crossovers import arithmetic_crossover


class FitnessMulti(base.Fitness):
    """Custom fitness class with hashability support for DEAP."""

    weights = (1.0, -1.0)

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return id(self) == id(other)


def evaluate_portfolio(
    portfolio: np.ndarray,
    returns_m: np.ndarray,
    covariances: np.ndarray,
    hist_returns: np.ndarray,
    metric: str,
    min_positions: int = 0,
    penalty_factor: float = 0.1,
    min_weight: float = 0.01,
):
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
        # Calculate portfolio returns over time and MDD
        portfolio_returns = hist_returns @ portfolio
        risk_value = -maximum_drawdown(portfolio_returns)  # MDD is negative, we want positive
    elif metric == "sharpe":
        # For sharpe, we want to maximize it, so we negate it (since fitness minimizes 2nd objective)
        portfolio_returns = hist_returns @ portfolio
        sr = sharpe_ratio(portfolio_returns)
        risk_value = -sr  # Negate so minimizing this = maximizing sharpe
    else:
        raise ValueError(f"Unknown risk metric: {metric}")

    # Add penalty if portfolio is too small
    active_positions = np.sum(portfolio > min_weight)
    if active_positions < min_positions:
        missing = min_positions - active_positions
        tax_rate = missing * penalty_factor
        penalty_amount = np.abs(portfolio_return) * tax_rate

        portfolio_return -= penalty_amount
        risk_value += (risk_value * tax_rate)

    return portfolio_return, risk_value


def mutate_wrapper(individual, gaussian_rate=0.3, gaussian_sigma=0.08, swap_rate=0.15):
    mutated = individual.reshape(1, -1).copy()
    mutated = gaussian_mutation(mutated, mutation_rate=gaussian_rate, sigma=gaussian_sigma)
    mutated = swap_mutation(mutated, mutation_rate=swap_rate)
    individual[:] = mutated[0]
    if hasattr(individual.fitness, "values"):
        del individual.fitness.values
    return (individual,)


def crossover_wrapper(parent1, parent2, alpha=0.5):
    offspring1, offspring2 = arithmetic_crossover(np.array(parent1), np.array(parent2), alpha=alpha)
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
    stock_names,
    stock_returns_m,
    stock_covariances,
    historical_returns=None,
    risk_metric="std",
    min_positions=0,
    min_weight=0.01,
    penalty_factor=0.1,
    mutation_kwargs=None,
    crossover_kwargs=None,
):
    """Initializes DEAP toolbox with multiobjective setup.

    Args:
        stock_names: List of stock ticker names
        stock_returns_m: Mean returns for each stock
        stock_covariances: Covariance matrix of stock returns
        historical_returns: Historical returns array (n_days, n_assets) - required for mdd/sharpe
        risk_metric: Risk metric to use - 'std' (volatility), 'mdd' (max drawdown), or 'sharpe'
        min_positions: Minimum number of assets to use for portfolio before adding penalty.
        min_weight: Minimum weight threshold for an asset to be considered an "active position" (e.g., 0.01 for 1% of portfolio).
        penalty_factor: The penalty multiplier applied to return and risk for each missing position (e.g., 0.1 means 10% penalty per missing asset).
        mutation_kwargs: Dictionary of kwargs for the mutation function
        crossover_kwargs: Dictionary of kwargs for the crossover function
    """
    if risk_metric in ("mdd", "sharpe") and historical_returns is None:
        raise ValueError(f"historical_returns required for risk_metric='{risk_metric}'")

    # Define fitness: maximize return, minimize risk (or maximize sharpe)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=FitnessMulti)

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
        min_positions=min_positions,
        penalty_factor=penalty_factor,
        min_weight=min_weight,
    )

    mut_kwargs = mutation_kwargs or {}
    toolbox.register("mutate", mutate_wrapper, **mut_kwargs)

    cx_kwargs = crossover_kwargs or {}
    toolbox.register("mate", crossover_wrapper, **cx_kwargs)

    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_nsga2(
    toolbox,
    pop_size=200,
    n_generations=150,
    cxpb=0.7,
    mutpb=0.6,
    callback=None,
    callback_interval=10,
):
    """Runs NSGA-II with optional callback every ``callback_interval`` generations."""

    pop = toolbox.population(n=pop_size)

    # Initial evaluation
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Ensure crowding values are set before the evolutionary loop
    pop = tools.selNSGA2(pop, len(pop))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_return", lambda x: np.mean([f[0] for f in x]))
    stats.register("avg_risk", lambda x: np.mean([f[1] for f in x]))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg_return", "avg_risk"]

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    for gen in tqdm(range(1, n_generations + 1), desc="Generations", total=n_generations):
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
