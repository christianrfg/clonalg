from typing import Callable

import numpy as np
from numpy.random import uniform

# Individual is represented as a tuple: (solution_vector: np.ndarray, affinity: float)
Individual = tuple[np.ndarray, float]


def evaluate(
    population: list[np.ndarray],
    objective_fn: Callable,
) -> list[Individual]:
    """Compute affinity for every member of the population and sort ascending.

    Lower affinity = better solution.
    """
    scored = [(p, objective_fn(p)) for p in population]
    return sorted(scored, key=lambda x: x[1])


def create_population(
    population_size: int,
    problem_size: int,
    b_lo: float,
    b_up: float,
) -> list[np.ndarray]:
    """Initialize a population of uniformly distributed random candidates."""
    return [
        uniform(low=b_lo, high=b_up, size=problem_size) for _ in range(population_size)
    ]


def clone(
    individual: Individual,
    rank: int,
    clone_rate: int,
) -> list[Individual]:
    """
    Clone an individual. Clone count is inversely proportional to rank (1 = best),
    following the standard CLONALG formulation: n_clones = floor(clone_rate / rank).
    """
    n_clones = max(1, int(clone_rate / rank))
    return [individual] * n_clones


def hypermutate(
    individual: Individual,
    mutation_rate: float,
    b_lo: float,
    b_up: float,
) -> Individual:
    """
    Apply hypermutation inversely proportional to affinity.
    Individuals with higher affinity (worse solutions) mutate more aggressively.
    """
    solution, affinity = individual
    mutation_prob = affinity / (mutation_rate * 100)

    if uniform() <= mutation_prob:
        mutated = np.array(
            [
                uniform(low=b_lo, high=b_up) if uniform() <= mutation_prob else gene
                for gene in solution
            ],
        )
        return (mutated, affinity)

    return individual


def select(
    population: list[Individual],
    clones: list[Individual],
    population_size: int,
    objective_fn: Callable,
) -> list[Individual]:
    """Re-evaluate clones and select the best individuals from the combined pool.

    Merges original population with mutated clones, returns top
    population_size by affinity.
    """
    re_evaluated_clones = [(sol, objective_fn(sol)) for sol, _ in clones]
    combined = population + re_evaluated_clones
    return sorted(combined, key=lambda x: x[1])[:population_size]


def replace(
    population: list[Individual],
    random_cells: list[Individual],
    population_size: int,
) -> list[Individual]:
    """Replace the worst individuals with random cells to maintain diversity."""
    combined = population + random_cells
    return sorted(combined, key=lambda x: x[1])[:population_size]
