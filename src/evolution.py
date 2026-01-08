import numpy as np
from deap import base, creator, tools, algorithms
from src.utils import normalize_weights
from src.mutations import gaussian_mutation, swap_mutation
from src.crossovers import arithmetic_crossover


def setup_deap(stock_names, stock_returns_m, stock_covariances):
    """Initializes DEAP toolbox with multiobjective setup."""
    # Define fitness: maximize return, minimize risk
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

    def evaluate_portfolio(portfolio: np.ndarray, returns_m: np.ndarray, covariances: np.ndarray):
        if np.any(np.isnan(portfolio)) or np.any(np.isinf(portfolio)):
            return float("nan"), float("nan")

        weight_sum = np.sum(portfolio)
        if not np.isclose(weight_sum, 1.0):
            portfolio = portfolio / weight_sum

        portfolio_return = np.dot(portfolio, returns_m)
        portfolio_variance = portfolio @ covariances @ portfolio
        portfolio_volatility = np.sqrt(max(0, portfolio_variance))

        return portfolio_return, portfolio_volatility

    toolbox.register("evaluate", evaluate_portfolio, returns_m=stock_returns_m, covariances=stock_covariances)

    def mutate_wrapper(individual, gaussian_rate=0.3, gaussian_sigma=0.08, swap_rate=0.15):
        mutated = individual.reshape(1, -1).copy()
        mutated = gaussian_mutation(mutated, mutation_rate=gaussian_rate, sigma=gaussian_sigma)
        mutated = swap_mutation(mutated, mutation_rate=swap_rate)
        individual[:] = mutated[0]
        if hasattr(individual.fitness, "values"):
            del individual.fitness.values
        return (individual,)

    toolbox.register("mutate", mutate_wrapper)

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

    toolbox.register("mate", crossover_wrapper, alpha=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_nsga2(toolbox, pop_size=200, n_generations=150, cxpb=0.7, mutpb=0.6):
    """Runs NSGA-II algorithm."""
    pop = toolbox.population(n=pop_size)

    # Initial evaluation
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg_return", lambda x: np.mean([f[0] for f in x]))
    stats.register("avg_risk", lambda x: np.mean([f[1] for f in x]))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg_return", "avg_risk"]

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, verbose=True
    )

    return pop, logbook
