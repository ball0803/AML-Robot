from abc import ABC, abstractmethod
from .encoding import Genotype, Chromosome
from typing import Tuple
import random
from fuzzy_logic import CombinedMembershipFunctions, MembershipFunction


class CrossoverStrategy(ABC):
    """Abstract base class for crossover strategies."""

    @abstractmethod
    def crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> Tuple[Genotype, Genotype]:
        """Performs crossover between two parent genotypes."""
        pass


class OnePointCrossover(CrossoverStrategy):
    """One-point crossover strategy at the gene level inside each chromosome."""

    def crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> Tuple[Genotype, Genotype]:
        offspring1 = Genotype()
        offspring2 = Genotype()
        for c1, c2 in zip(parent1.chromosomes, parent2.chromosomes):
            # Perform one-point crossover for each chromosome
            point = random.randint(0, len(c1.genes_list) - 1)
            new_c1 = c1.clone()
            new_c2 = c2.clone()

            for i in range(point, len(new_c1)):
                new_c1.set_gene_at_index(i, c2.genes_list[i].clone())
                new_c2.set_gene_at_index(i, c1.genes_list[i].clone())

            offspring1.add_chromosome(new_c1)
            offspring2.add_chromosome(new_c2)

        return offspring1, offspring2


class TwoPointCrossover(CrossoverStrategy):
    """Two-point crossover strategy at the gene level inside each chromosome."""

    def crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> Tuple[Genotype, Genotype]:
        offspring1 = Genotype()
        offspring2 = Genotype()

        for c1, c2 in zip(parent1.chromosomes, parent2.chromosomes):
            # Perform two-point crossover for each chromosome
            point1 = random.randint(0, len(c1.genes_list) - 2)
            point2 = random.randint(point1 + 1, len(c1.genes_list) - 1)
            new_c1 = c1.clone()
            new_c2 = c2.clone()

            for i in range(point1, point2):
                new_c1.set_gene_at_index(i, c2.genes_list[i].clone())
                new_c2.set_gene_at_index(i, c1.genes_list[i].clone())

            offspring1.add_chromosome(new_c1)
            offspring2.add_chromosome(new_c2)

        return offspring1, offspring2


class UniformCrossover(CrossoverStrategy):
    """Uniform crossover strategy at the gene level where each gene has a 50% chance to come from either parent."""

    def crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> Tuple[Genotype, Genotype]:
        offspring1 = Genotype()
        offspring2 = Genotype()

        for c1, c2 in zip(parent1.chromosomes, parent2.chromosomes):
            new_c1 = c1.clone()
            new_c2 = c2.clone()

            for i in range(len(c1.genes_list)):
                if random.random() < 0.5:
                    new_c1.set_gene_at_index(i, c2.genes_list[i].clone())
                    new_c2.set_gene_at_index(i, c1.genes_list[i].clone())

            offspring1.add_chromosome(new_c1)
            offspring2.add_chromosome(new_c2)

        return offspring1, offspring2


class NoCrossover(CrossoverStrategy):
    """No crossover strategy, returns clones of both parents."""

    def crossover(
        self, parent1: Genotype, parent2: Genotype
    ) -> Tuple[Genotype, Genotype]:
        return parent1.clone(), parent2.clone()


def main():
    dist_msf = CombinedMembershipFunctions()
    dist_msf.add_membership(
        "near", MembershipFunction.create(function="triangular", a=0, b=0, c=100)
    )
    dist_msf.add_membership(
        "far", MembershipFunction.create(function="triangular", a=0, b=100, c=100)
    )

    c1 = Chromosome()
    c2 = Chromosome()

    for _ in range(8):
        c1.add_rule_gene(
            value="_",
            mapping=dict(
                _=lambda **_: 1,
                near=lambda **args: dist_msf.fuzzify(args["x"])["near"],
                far=lambda **args: dist_msf.fuzzify(args["x"])["far"],
            ),
        )
        c2.add_rule_gene(
            value="far",
            mapping=dict(
                _=lambda **_: 1,
                near=lambda **args: dist_msf.fuzzify(args["x"])["near"],
                far=lambda **args: dist_msf.fuzzify(args["x"])["far"],
            ),
        )

    c1.add_return_gene(
        name="turn",
        value=180,
        func=lambda x: (x % 181) - 90,
    )
    c2.add_return_gene(
        name="turn",
        value=180,
        func=lambda x: (x % 181) - 90,
    )

    c1.add_return_gene(
        name="move",
        value=10,
        func=lambda x: (x % 21) - 10,
    )
    c2.add_return_gene(
        name="move",
        value=10,
        func=lambda x: (x % 21) - 10,
    )

    gn1 = Genotype()
    gn2 = Genotype()

    for _ in range(16):
        gn1.add_chromosome(c1.clone())

    for _ in range(16):
        gn2.add_chromosome(c2.clone())

    # print(gn2)
    cross_strat = OnePointCrossover()
    off1, off2 = cross_strat.crossover(gn1, gn2)
    print(off1, off2)


if __name__ == "__main__":
    main()
