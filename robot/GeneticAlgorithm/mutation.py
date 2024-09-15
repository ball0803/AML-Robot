from abc import ABC, abstractmethod
from typing import List, Callable
import random
from .encoding import Genotype, Chromosome

from fuzzy_logic import CombinedMembershipFunctions, MembershipFunction


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""

    def __init__(
        self,
        mutation_probability: float = 0.01,
        gene_type: List[str] = ["RuleGene", "ReturnGene"],
    ) -> None:
        self.mutation_probability = mutation_probability
        self.gene_type = gene_type

    @abstractmethod
    def mutate(self, genotype: Genotype) -> None:
        """Performs mutation on a genotype."""
        pass

    def _clone_and_apply(self, genotype: Genotype, mutate_func) -> None:
        for chromosome in genotype.chromosomes:
            offspring = chromosome.clone()
            mutate_func(offspring)
            genotype.chromosomes[genotype.chromosomes.index(chromosome)] = offspring

    def apply_mutation_probability(self) -> bool:
        # print(self.mutation_probability)
        """Helper method to check if mutation should be applied based on probability."""
        return random.random() < self.mutation_probability

    def apply_gene(self, gene_type: str):
        return gene_type in self.gene_type


class BitFlipMutation(MutationStrategy):
    """Simple bit-flip mutation for binary-encoded chromosomes."""

    def mutate(self, genotype: Genotype) -> None:
        def mutate_func(chromosome):
            if self.apply_gene("RuleGene"):
                for gene in chromosome.rules_list:
                    variant = gene.variant
                    if len(variant) == 2 and self.apply_mutation_probability():
                        gene.value = variant[1 - variant.index(gene.value)]

            if self.apply_gene("ReturnGene"):
                for gene in chromosome.returns_list:
                    if (
                        isinstance(gene.value, int)
                        and self.apply_mutation_probability()
                    ):
                        gene.value = 1 - gene.value

        self._clone_and_apply(genotype, mutate_func)


class RandomResetMutation(MutationStrategy):
    """Random reset mutation where a gene value is replaced with a random value within a specified range."""

    def mutate(self, genotype: Genotype) -> None:
        def mutate_func(chromosome):
            if self.apply_gene("RuleGene"):
                for gene in chromosome.rules_list:
                    if self.apply_mutation_probability():
                        gene.value = random.choice(gene.variant)

            if self.apply_gene("ReturnGene"):
                for gene in chromosome.returns_list:
                    if self.apply_mutation_probability():
                        gene.value = random.randrange(gene.variant[0], gene.variant[1])

        self._clone_and_apply(genotype, mutate_func)


class SwapMutation(MutationStrategy):
    """Swap mutation, swaps the values of two randomly selected genes."""

    def mutate(self, genotype: Genotype) -> None:
        def mutate_func(chromosome):
            if self.apply_gene("RuleGene"):
                if (
                    self.apply_mutation_probability()
                    and len(chromosome.rules_list) >= 2
                ):
                    i, j = random.sample(range(len(chromosome.rules_list)), 2)
                    chromosome.rules_list[i], chromosome.rules_list[j] = (
                        chromosome.rules_list[j],
                        chromosome.rules_list[i],
                    )

            if self.apply_gene("ReturnGene"):
                if (
                    self.apply_mutation_probability()
                    and len(chromosome.returns_list) >= 2
                ):
                    i, j = random.sample(range(len(chromosome.returns_list)), 2)
                    chromosome.returns_list[i], chromosome.returns_list[j] = (
                        chromosome.returns_list[j],
                        chromosome.returns_list[i],
                    )

        self._clone_and_apply(genotype, mutate_func)


class GaussianMutation(MutationStrategy):
    """Gaussian mutation, adds Gaussian noise to numeric genes (useful for float-based genes)."""

    def __init__(
        self, mutation_probability: float, mean: float = 0.0, stddev: float = 1.0
    ) -> None:
        super().__init__(mutation_probability)
        self.mean = mean
        self.stddev = stddev

    def mutate(self, genotype: Genotype) -> None:
        def mutate_func(chromosome):
            if self.apply_gene("ReturnGene"):
                for gene in chromosome.returns_list:
                    # print(isinstance(gene.value, float))
                    if (
                        isinstance(gene.value, float)
                        and self.apply_mutation_probability()
                    ):
                        noise = random.gauss(self.mean, self.stddev)
                        # print(noise)
                        gene.value += noise

        self._clone_and_apply(genotype, mutate_func)


class CompositeMutation(MutationStrategy):
    def __init__(self, strategies: List[MutationStrategy]) -> None:
        super().__init__()
        self.strategies = strategies

    def mutate(self, genotype: Genotype) -> None:
        for strategy in self.strategies:
            strategy.mutate(genotype)


def main():
    dist_msf = CombinedMembershipFunctions()
    dist_msf.add_membership(
        "near", MembershipFunction.create(function="triangular", a=0, b=0, c=100)
    )
    dist_msf.add_membership(
        "far", MembershipFunction.create(function="triangular", a=0, b=100, c=100)
    )
    c1 = Chromosome()

    for _ in range(8):
        c1.add_rule_gene(
            value="_",
            mapping=dict(
                _=lambda **_: 1,
                near=lambda **args: dist_msf.fuzzify(args["x"])["near"],
                far=lambda **args: dist_msf.fuzzify(args["x"])["far"],
            ),
        )

    c1.add_return_gene(
        name="turn",
        value=180.0,
        func=lambda x: (x % 181.0) - 90.0,
    )

    c1.add_return_gene(
        name="move",
        value=10.0,
        func=lambda x: (x % 21.0) - 10.0,
    )

    gn1 = Genotype()

    for _ in range(16):
        gn1.add_chromosome(c1.clone())

    mutation = RandomResetMutation(
        mutation_probability=0.80,
    )
    mutation = CompositeMutation(
        strategies=[
            RandomResetMutation(mutation_probability=0.1, gene_type=["RuleGene"]),
            GaussianMutation(mutation_probability=0.05, stddev=20.0),
        ]
    )
    mutation.mutate(gn1)

    # print(gn1)


if __name__ == "__main__":
    main()
