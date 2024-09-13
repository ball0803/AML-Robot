from abc import ABC, abstractmethod
import random
from typing import List, Callable, Optional, Dict
from encocding import Chromosome
from crossover import CrossoverStrategy
from selection import SelectionStrategy
from mutation import MutationStrategy


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        selection_strategy: SelectionStrategy,  # Accepts a SelectionStrategy object
        crossover_strategy: CrossoverStrategy,  # Accepts a CrossoverStrategy object
        mutation_strategy: MutationStrategy,  # Accepts a MutationStrategy object
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.population: List[Chromosome] = []
        self.fitness_scores: List[float] = []

    def initialize_population(self, chromosome_template: Chromosome) -> None:
        self.population = [
            chromosome_template.clone() for _ in range(self.population_size)
        ]

    def evaluate_population(self, args: Dict[str, any]) -> None:
        """Evaluate the population and store the fitness scores."""
        self.fitness_scores = [
            chromosome.evaluate(args)[0] for chromosome in self.population
        ]

    def select(self) -> Chromosome:
        """Select a chromosome using the selection strategy."""
        return self.selection_strategy.select(self.population, self.fitness_scores)

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Perform crossover using the crossover strategy."""
        return self.crossover_strategy.crossover(parent1, parent2)

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Mutate the chromosome using the mutation strategy."""
        return self.mutation_strategy.mutate(chromosome)

    def create_next_generation(self) -> None:
        """Create the next generation of chromosomes."""
        new_population = []

        while len(new_population) < self.population_size:
            parent1 = self.select()
            parent2 = self.select()

            if random.random() < self.crossover_rate:
                offspring = self.crossover(parent1, parent2)
            else:
                offspring = parent1.clone()

            if random.random() < self.mutation_rate:
                offspring = self.mutate(offspring)

            new_population.append(offspring)

        self.population = new_population
