import numpy as np
from numpy.random import uniform

def affinity(p_i):
    """
    Description
    -----------
    Return the affinity of one subject.
    
    Parameters
    -----------
    p_i: numpy.array
        Subject of a population.
    
    Return
    -----------
    return: float
        Affinity of the subject passed as parameter.
    
    """
    return np.sum(np.power(p_i, 2))

def create_random_cells(population_size, problem_size, b_lo, b_up):
    population = [uniform(low=b_lo, high=b_up, size=problem_size) for x in range(population_size)]
    
    return population

def clone(p_i, clone_rate):
    clone_num = int(clone_rate / p_i[1])
    clones = [(p_i[0], p_i[1]) for x in range(clone_num)]
    
    return clones

def hypermutate(p_i, mutation_rate, b_lo, b_up):
    if uniform() <= p_i[1] / (mutation_rate * 100):
        ind_tmp = []
        for gen in p_i[0]:
            if uniform() <= p_i[1] / (mutation_rate * 100):
                ind_tmp.append(uniform(low=b_lo, high=b_up))
            else:
                ind_tmp.append(gen)
                
        return (np.array(ind_tmp), affinity(ind_tmp))
    else:
        return p_i

def select(pop, pop_clones, pop_size):
    population = pop + pop_clones
    
    population = sorted(population, key=lambda x: x[1])[:pop_size]
    
    return population

def replace(population, population_rand, population_size):
    population = population + population_rand
    population = sorted(population, key=lambda x: x[1])[:population_size]
    
    return population