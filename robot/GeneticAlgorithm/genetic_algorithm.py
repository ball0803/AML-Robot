from abc import ABC, abstractmethod
import random
from typing import List, Dict, Tuple
from .encoding import Genotype
from .crossover import CrossoverStrategy
from .selection import SelectionStrategy
from .mutation import MutationStrategy


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        selection_strategy: SelectionStrategy,  # Accepts a SelectionStrategy object
        crossover_strategy: CrossoverStrategy,  # Accepts a CrossoverStrategy object
        mutation_strategy: MutationStrategy,  # Accepts a MutationStrategy object
        elitism_percentage: float = 0.1,  # Percentage of the population to preserve as elites
    ) -> None:
        self.selection_strategy = selection_strategy()
        self.crossover_strategy = crossover_strategy()
        self.mutation_strategy = mutation_strategy
        self.elitism_percentage = elitism_percentage

        self.population_size: int = population_size
        self.population: List[Genotype] = []
        self.fitness_scores: List[float] = []

    def __len__(self) -> int:
        return self.population_size

    def initialize_population(self, genotype_template: Genotype) -> None:
        """Initialize the population with clones of a Genotype template."""
        for _ in range(len(self)):
            for chromosomes in genotype_template.chromosomes:
                for gene in chromosomes.rules_list:
                    gene.value = random.choice(gene.variant)
                for gene in chromosomes.returns_list:
                    gene.value = random.uniform(gene.variant[0], gene.variant[1])
            self.population.append(genotype_template.clone())

    def evaluate_population(self, args: Dict[str, any]) -> None:
        """Evaluate the population and store the fitness scores."""
        self.fitness_scores = [
            genotype.evaluate(args)[0] for genotype in self.population
        ]

    def select(self) -> Genotype:
        """Select a Genotype using the selection strategy."""
        return self.selection_strategy.select(
            population=self.population, fitness_scores=self.fitness_scores
        )

    def crossover(self, parent1: Genotype, parent2: Genotype) -> Tuple[Genotype]:
        """Perform crossover using the crossover strategy."""
        return self.crossover_strategy.crossover(parent1, parent2)

    def mutate(self, genotype: Genotype) -> Genotype:
        """Mutate the genotype using the mutation strategy."""
        return self.mutation_strategy.mutate(genotype)

    def elitism(self) -> List[Genotype]:
        """Preserve the top N% Genotypes based on the elitism percentage."""
        elite_count = max(1, int(len(self) * self.elitism_percentage))
        elite_indexes = sorted(
            range(len(self.fitness_scores)),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )[:elite_count]
        return [self.population[i].clone() for i in elite_indexes]

    def create_next_generation(self) -> None:
        """Create the next generation of genotypes."""
        # Get the elite genotypes
        new_population = self.elitism()

        # Continue creating new genotypes until the population is full
        while len(new_population) < len(self):
            # print(len(new_population), len(self))
            parent1 = self.select()
            parent2 = self.select()
            # print(parent1, parent2)

            offspring1, offspring2 = self.crossover(parent1, parent2)
            # print(offspring1)
            self.mutate(offspring1)
            # print(offspring1)

            new_population.append(offspring1)

        # Update the population with the new generation
        self.population = new_population

    def run(self, generations: int, args: Dict[str, any]) -> None:
        """Run the genetic algorithm for a specified number of generations."""
        for _ in range(generations):
            self.evaluate_population(args)
            self.create_next_generation()
