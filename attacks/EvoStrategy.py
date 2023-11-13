from abc import ABC, abstractmethod
from utils import cap
import numpy as np
import random
import matplotlib.pyplot as plt


random.seed(40)


class EvoStrategy(ABC):
    """Abstract class for evolutionary search strategies

    This is an abstract class that provides a template for standard evolutionary
    search strategies through common functionality related to retrieving and
    generating new generations, computing fitness, and retrieving fittest individuals

    Attributes:
        generation_count: An integer count of the generations created so far.
        active_generation: A list of individuals of the same data type representing the
            most recent generation created by the evolutionary search strategy.
        fitness_scores: A list of floats representing the fitness scores of each
            individual in the current active generation.
        queries: An integer represeting all individuals explored so far by the
            evolutionary search strategy, i.e. the sum of all generation sizes so far.

    """

    def __init__(self):
        """Inits EvoStrategy with an empty active generation and no history."""
        self.generation_count = 0
        self.active_generation = []
        self.fitness_scores = []
        self.queries = 0
        pass

    def get_active_generation(self):
        """Retrieves the active generation."""
        return self.active_generation

    def get_best_candidate(self):
        """Retrieves the fittest individual from the active generation."""
        fitnesses = self.fitness_scores
        best_candidate_index = np.argmax(fitnesses)
        return self.active_generation[best_candidate_index]

    @abstractmethod
    def get_next_generation(self):
        """Retrieves next generation starting from the active one."""
        pass

    @abstractmethod
    def get_fitness_scores(self):
        """Retrieves fitness scores of each individual from the active generation."""
        pass

    def generate_next_generation(self):
        """Retrieves next generation starting from the active one and sets it to active."""
        new_generation = self.get_next_generation()
        self.active_generation = new_generation
        self.generation_count += 1
        self.fitness_scores = self.get_fitness_scores()
        self.queries += queries
