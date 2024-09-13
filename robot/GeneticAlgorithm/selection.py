from abc import ABC, abstractmethod
from encocding import Genotype
from typing import List
import random


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(
        self, population: List[Genotype], fitness_scores: List[float]
    ) -> Genotype:
        """Selects a Genotype from the population based on their fitness scores."""
        pass


class RouletteWheelSelection(SelectionStrategy):
    """Implements roulette wheel selection (fitness-proportionate selection)."""

    def select(
        self, population: List[Genotype], fitness_scores: List[float]
    ) -> Genotype:
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for Genotype, fitness in zip(population, fitness_scores):
            current += fitness
            if current > pick:
                return Genotype.clone()


class TournamentSelection(SelectionStrategy):
    """Implements tournament selection."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(
        self, population: List[Genotype], fitness_scores: List[float]
    ) -> Genotype:
        tournament_contestants = random.sample(
            list(zip(population, fitness_scores)), self.tournament_size
        )
        winner = max(
            tournament_contestants, key=lambda pair: pair[1]
        )  # Select the best fitness
        return winner[0].clone()


class RankBasedSelection(SelectionStrategy):
    """Implements rank-based selection where selection probability is based on rank, not raw fitness."""

    def select(
        self, population: List[Genotype], fitness_scores: List[float]
    ) -> Genotype:
        sorted_population = [
            Genotype
            for Genotype, _ in sorted(
                zip(population, fitness_scores), key=lambda pair: pair[1]
            )
        ]
        rank_weights = list(range(1, len(sorted_population) + 1))  # Create rank weights
        total_rank = sum(rank_weights)
        pick = random.uniform(0, total_rank)
        current = 0
        for Genotype, rank_weight in zip(sorted_population, rank_weights):
            current += rank_weight
            if current > pick:
                return Genotype.clone()


class RandomSelection(SelectionStrategy):
    """Implements random selection where Genotypes are selected purely by chance."""

    def select(
        self, population: List[Genotype], fitness_scores: List[float]
    ) -> Genotype:
        return random.choice(population).clone()
