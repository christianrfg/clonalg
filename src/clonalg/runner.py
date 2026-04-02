from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import (
    Individual,
    clone,
    create_population,
    evaluate,
    hypermutate,
    replace,
    select,
)


@dataclass
class CLONALGConfig:
    problem_size: int
    b_lo: float
    b_up: float
    population_size: int = 100
    selection_size: int = 10
    random_cells_num: int = 20
    clone_rate: int = 20
    mutation_rate: float = 0.2
    max_iterations: int = 1000


@dataclass
class CLONALGResult:
    best_individual: Individual
    history: list[list[Individual]] = field(default_factory=list)

    @property
    def best_solution(self) -> np.ndarray:
        return self.best_individual[0]

    @property
    def best_affinity(self) -> float:
        return self.best_individual[1]


def run(
    objective_fn: Callable,
    config: CLONALGConfig,
    on_iteration: Callable[[int, list[Individual]], None] | None = None,
) -> CLONALGResult:
    """
    Execute the CLONALG optimization loop.

    Parameters
    ----------
    objective_fn:
        Function to minimize. Receives a np.ndarray and returns a float.
        Lower values indicate better solutions.
    config:
        Algorithm hyperparameters.
    on_iteration:
        Optional callback invoked at the end of each iteration with
        (iteration_index, top5_individuals). Use for live progress plots.

    Returns
    -------
    CLONALGResult with the best individual found and per-iteration history.
    """
    population = create_population(
        config.population_size,
        config.problem_size,
        config.b_lo,
        config.b_up,
    )
    history: list[list[Individual]] = []

    for iteration in range(config.max_iterations):
        # Score and rank
        population_scored = evaluate(population, objective_fn)
        history.append(population_scored[:5])

        if on_iteration is not None:
            on_iteration(iteration, population_scored[:5])

        # Select top candidates
        selected = population_scored[: config.selection_size]

        # Clone (rank 1 = best → most clones)
        clones: list[Individual] = []
        for rank, individual in enumerate(selected, start=1):
            clones.extend(clone(individual, rank, config.clone_rate))

        # Hypermutate
        clones = [
            hypermutate(ind, config.mutation_rate, config.b_lo, config.b_up)
            for ind in clones
        ]

        # Select survivors from original + mutated clones
        population_scored = select(
            population_scored,
            clones,
            config.population_size,
            objective_fn,
        )

        # Inject random cells for diversity
        random_cells = create_population(
            config.random_cells_num,
            config.problem_size,
            config.b_lo,
            config.b_up,
        )
        random_cells_scored = evaluate(random_cells, objective_fn)
        population_scored = replace(
            population_scored,
            random_cells_scored,
            config.population_size,
        )

        population = [ind[0] for ind in population_scored]

    final_scored = evaluate(population, objective_fn)

    return CLONALGResult(
        best_individual=final_scored[0],
        history=history,
    )
