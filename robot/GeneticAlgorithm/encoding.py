from fuzzy_logic import CombinedMembershipFunctions, MembershipFunction
from typing import Tuple, Callable, List, Dict, Union
import copy
import random


class Gene:
    """Base class for a gene."""

    def __init__(self, name: Union[str, int], value: Union[int, float, str]) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"

    def clone(self) -> "Gene":
        """Creates a deep copy of the gene."""
        return copy.deepcopy(self)

    def evaluate(self, **args) -> float:
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class RuleGene(Gene):
    """Represents a rule gene that can have a value of str or int."""

    def __init__(
        self,
        name: Union[str, int],
        value: Union[str, int],
        mapping: Dict[Union[str, int], Callable[..., float]],
        variant: Tuple[str, int] = None,
    ) -> None:
        super().__init__(name=name, value=value)
        self.mapping: Dict[Union[str, int], Callable[..., float]] = mapping
        self.variant: Tuple[str, int] = (
            variant if variant else tuple(self.mapping.keys())
        )

    def __len__(self) -> int:
        return len(self.varaint())

    def clone(self) -> "RuleGene":
        """Creates a deep copy of the RuleGene."""
        return copy.deepcopy(self)

    def evaluate(self, **args) -> float:
        """Evaluates the value using the provided mapping and optional arguments."""
        if self.value in self.mapping:
            try:
                return self.mapping[self.value](**args)
            except Exception as e:
                raise RuntimeError(f"Error evaluating RuleGene '{self.name}': {e}")
        else:
            raise ValueError(f"Value {self.value} not found in mapping.")


class ReturnGene(Gene):
    """Represents a return gene that has a value of int or float and uses a function to compute results."""

    def __init__(
        self,
        name: Union[str, int],
        value: Union[int, float],
        func: Callable[..., float],
        variant: Tuple[int, float] = None,
    ) -> None:
        super().__init__(name=name, value=value)
        self.func: Callable[..., float] = func
        self.variant: Tuple[int, float] = variant if variant else tuple((0, 100))

    def __len__(self) -> int:
        return max(self.variant)

    def clone(self) -> "ReturnGene":
        """Creates a deep copy of the ReturnGene."""
        return copy.deepcopy(self)

    def evaluate(self, **args) -> float:
        """Evaluates the function by passing the value along with additional arguments."""
        try:
            return self.func(self.value, **args)
        except Exception as e:
            raise RuntimeError(f"Error evaluating ReturnGene '{self.name}': {e}")


class Chromosome:
    def __init__(
        self, rules_list: List[RuleGene] = None, returns_list: List[ReturnGene] = None
    ) -> None:
        self.rules_list: List[RuleGene] = rules_list if rules_list else []
        self.returns_list: List[ReturnGene] = returns_list if returns_list else []

    def __len__(self) -> int:
        return len(self.genes_list)

    def __str__(self) -> str:
        return "{ " + ", ".join(str(gene) for gene in self.genes_list) + " }"

    @property
    def genes_list(self) -> List[Gene]:
        return self.rules_list + self.returns_list

    @genes_list.setter
    def genes_list(self, new_genes):
        """Distributes new_genes back into rules_list and returns_list."""
        num_rules = len(self.rules_list)
        num_returns = len(self.returns_list)

        # Ensure the new genes list matches the expected length
        if len(new_genes) != num_rules + num_returns:
            raise ValueError(
                f"Expected {num_rules + num_returns} genes, got {len(new_genes)}."
            )

        # Set the first part of new_genes to rules_list
        self.rules_list = new_genes[:num_rules]

        # Set the remaining part of new_genes to returns_list
        self.returns_list = new_genes[num_rules:]

    def add_rule_gene(
        self,
        value: Union[str, int],
        mapping: Dict[Union[str, int], Callable[[], float]],
        name: Union[str, int] = None,
    ) -> None:
        if name is None:
            name = len(self.rules_list)
        self.rules_list.append(RuleGene(value=value, mapping=mapping, name=name))

    def add_return_gene(
        self,
        value: Union[int, float],
        func: Callable[[Union[int, float], dict], float],
        name: Union[str, int] = None,
    ) -> None:
        if name is None:
            name = len(self)
        self.returns_list.append(ReturnGene(value=value, func=func, name=name))

    def set_gene_at_index(self, index, gene):
        """Sets a specific gene at the given index, adjusting rules_list and returns_list as needed."""
        if not (0 <= index < len(self.genes_list)):
            raise IndexError("Index out of range.")

        num_rules = len(self.rules_list)

        if index < num_rules:
            self.rules_list[index] = gene
        else:
            self.returns_list[index - num_rules] = gene

    def clone(self) -> "Chromosome":
        """Creates a deep copy of the Chromosome."""
        return copy.deepcopy(self)

    def evaluate(self, args: Dict[str, any]) -> Tuple[float, ...]:
        rule_value: float = 1.0
        try:
            for rule in self.rules_list:
                rule_args = args.get(rule.name, {})
                rule_value *= rule.evaluate(**rule_args)

            returns_values: Tuple[float, ...] = tuple(
                rule_value * ret.evaluate(**args.get(ret.name, {}))
                for ret in self.returns_list
            )
        except Exception as e:
            raise RuntimeError(f"Error during Chromosome evaluation: {e}")

        return returns_values


class Genotype:
    def __init__(self, chromosomes: List[Chromosome] = None) -> None:
        self.chromosomes: List[Chromosome] = chromosomes if chromosomes else list()
        self.chromosome_return_length: int = (
            len(chromosomes[0].returns_list) if chromosomes else None
        )

    def __len__(self) -> int:
        return len(self.chromosomes)

    def __str__(self) -> str:
        return ", \n".join(
            f"{idx}: {str(chromosome)}"
            for idx, chromosome in enumerate(self.chromosomes)
        )

    def add_chromosome(self, chromosome: Chromosome) -> None:
        """Adds a chromosome after checking if its length matches the others."""
        if self.chromosome_return_length is None:
            # Set chromosome length based on the first chromosome
            self.chromosome_return_length = len(chromosome.returns_list)
        elif len(chromosome.returns_list) != self.chromosome_return_length:
            raise ValueError(
                f"Chromosome return gennes length {len(chromosome.returns_list)} does not match expected length {self.chromosome_return_length}"
            )

        self.chromosomes.append(chromosome)

    def clone(self) -> "Genotype":
        """Creates a deep copy of the Genotype."""
        return copy.deepcopy(self)

    def scamble(self):
        for chromosomes in self.chromosomes:
            for gene in chromosomes.rules_list:
                gene.value = random.choice(gene.variant)
            for gene in chromosomes.returns_list:
                gene.value = random.uniform(gene.variant[0], gene.variant[1])

    def evaluate(self, args: Dict[str, any]) -> Tuple[float, ...]:
        """Evaluates all chromosomes and sums their evaluation results."""
        try:
            if not self.chromosomes:
                raise ValueError("No chromosomes to evaluate.")

            # Initialize result with zeros based on the first chromosome's evaluation
            result = [0.0] * len(self.chromosomes[0].evaluate(args=args))

            for chromosome in self.chromosomes:
                chromosome_result = chromosome.evaluate(args=args)
                # print(chromosome_result)
                result = [r + cr for r, cr in zip(result, chromosome_result)]

            return tuple(result)

        except Exception as e:
            raise RuntimeError(f"Error during Genotype evaluation: {e}")


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
        gn2.add_chromosome(c2.clone())

    # print(gn1)
    # print(
    #     gn1.evaluate(
    #         args={
    #             0: {"x": 10},
    #             1: {"x": 10},
    #             2: {"x": 10},
    #             3: {"x": 10},
    #             4: {"x": 10},
    #             5: {"x": 10},
    #             6: {"x": 10},
    #             7: {"x": 10},
    #         }
    #     )
    # )
    # result = c1.evaluate(args={0: {"x": 50}})
    # print(result)
    # g1 = RuleGene(
    #     name="front",
    #     value="near",
    #     mapping=dict(
    #         _=lambda: 1,
    #         near=lambda x: dist_msf.fuzzify(x)["near"],
    #         far=lambda x: dist_msf.fuzzify(x)["far"],
    #     ),
    # )
    # print(g1.varaint())
    # gr1 = ReturnGene(name="turn", value=180, func=lambda x: (x % 181) - 90)
    # print(gr1)
    # print(gr1.evaluate())


if __name__ == "__main__":
    main()
