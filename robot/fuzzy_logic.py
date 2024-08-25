from abc import ABC, abstractmethod
import math
from collections import namedtuple
from typing import Dict, Callable
from scipy.integrate import quad
import json

Membership = namedtuple("Membership", ["name", "function"])
FuzzyVariable = Dict[str, float]


class MembershipFunction(ABC):
    @abstractmethod
    def compute(self, x: float) -> float:
        pass

    @staticmethod
    def create(function: str, **kwargs) -> "MembershipFunction":
        """
        Factory method to create various types of membership functions.

        Parameters:
        - function (str): The type of membership function to create.
                          Possible values are:
            - "trapezoidal": Requires `a`, `b`, `c`, and `d` parameters.
            - "gaussian": Requires `c` and `sigma` parameters.
            - "triangular": Requires `a`, `b`, and `c` parameters.
            - "sigmoidal": Requires `a` and `c` parameters.
            - "bell": Requires `a`, `b`, and `c` parameters.
            - "z": Requires `a` and `b` parameters.
            - "s": Requires `a` and `b` parameters.
        - kwargs: The specific parameters required for the selected membership function.

        Parameters for each function type:
        - TrapezoidalMembershipFunction:
            - `a` (float): The start of the left slope.
            - `b` (float): The start of the top plateau.
            - `c` (float): The end of the top plateau.
            - `d` (float): The end of the right slope.
        - GaussianMembershipFunction:
            - `c` (float): The center of the Gaussian curve.
            - `sigma` (float): The standard deviation, controlling the width of the curve.
        - TriangularMembershipFunction:
            - `a` (float): The start of the left slope.
            - `b` (float): The peak of the triangle.
            - `c` (float): The end of the right slope.
        - SigmoidalMembershipFunction:
            - `a` (float): Controls the slope of the sigmoid curve.
            - `c` (float): The center of the sigmoid curve.
        - BellMembershipFunction:
            - `a` (float): Controls the width of the bell shape.
            - `b` (float): Controls the slope of the bell shape.
            - `c` (float): The center of the bell curve.
        - ZMembershipFunction:
            - `a` (float): The start of the slope.
            - `b` (float): The end of the slope.
        - SMembershipFunction:
            - `a` (float): The start of the slope.
            - `b` (float): The end of the slope.

        Returns:
        - MembershipFunction: An instance of the specified membership function.

        Raises:
        - ValueError: If an unknown function type is provided.

        Example Usage:
        - Create a trapezoidal membership function:
        `trapezoidal_mf = create("trapezoidal", a=1.0, b=2.0, c=3.0, d=4.0)`
        - Create a Gaussian membership function:
        `gaussian_mf = create("gaussian", c=2.5, sigma=0.5)`
        """
        if function == "trapezoidal":
            return TrapezoidalMembershipFunction(
                kwargs["a"], kwargs["b"], kwargs["c"], kwargs["d"]
            )
        elif function == "gaussian":
            return GaussianMembershipFunction(kwargs["c"], kwargs["sigma"])
        elif function == "triangular":
            return TriangularMembershipFunction(kwargs["a"], kwargs["b"], kwargs["c"])
        elif function == "sigmoidal":
            return SigmoidalMembershipFunction(kwargs["a"], kwargs["c"])
        elif function == "bell":
            return BellMembershipFunction(kwargs["a"], kwargs["b"], kwargs["c"])
        elif function == "z":
            return ZMembershipFunction(kwargs["a"], kwargs["b"])
        elif function == "s":
            return SMembershipFunction(kwargs["a"], kwargs["b"])
        else:
            raise ValueError(f"Unknown function: {function}")


class TrapezoidalMembershipFunction(MembershipFunction):
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def compute(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x <= self.c:
            return 1.0
        elif self.c < x < self.d:
            return (self.d - x) / (self.d - self.c)

    def centroid(self) -> float:
        def integrand(x):
            return x * self.compute(x)

        def integral(x):
            return self.compute(x)

        # Integrate over the interval where the function is active
        area, _ = quad(integral, self.a, self.d)
        centroid, _ = quad(integrand, self.a, self.d)
        return centroid / area if area != 0 else float("nan")


class GaussianMembershipFunction(MembershipFunction):
    def __init__(self, c: float, sigma: float):
        self.c = c
        self.sigma = sigma

    def compute(self, x: float) -> float:
        return math.exp(-((x - self.c) ** 2) / (2 * self.sigma**2))

    def centroid(self) -> float:
        return self.c


class TriangularMembershipFunction(MembershipFunction):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def compute(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif self.b < x < self.c:
            return (self.c - x) / (self.c - self.b)

    def centroid(self) -> float:
        return (self.a + self.b + self.c) / 3


class SigmoidalMembershipFunction(MembershipFunction):
    def __init__(self, a: float, c: float):
        self.a = a
        self.c = c

    def compute(self, x: float) -> float:
        return 1 / (1 + math.exp(-self.a * (x - self.c)))

    def centroid(self) -> float:
        # Numerical integration may be required here
        return self.c  # Approximation


class BellMembershipFunction(MembershipFunction):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def compute(self, x: float) -> float:
        return 1 / (1 + abs((x - self.c) / self.a) ** (2 * self.b))

    def centroid(self) -> float:
        def integrand(x):
            return x * self.compute(x)

        def integral(x):
            return self.compute(x)

        # Integrate over a sufficiently wide interval
        area, _ = quad(integral, self.c - 10 * self.a, self.c + 10 * self.a)
        centroid, _ = quad(integrand, self.c - 10 * self.a, self.c + 10 * self.a)
        return centroid / area if area != 0 else float("nan")


class ZMembershipFunction(MembershipFunction):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def compute(self, x: float) -> float:
        if x <= self.a:
            return 1.0
        elif x >= self.b:
            return 0.0
        else:
            return 1 - 2 * ((x - self.a) / (self.b - self.a)) ** 2

    def centroid(self) -> float:
        def integrand(x):
            return x * self.compute(x)

        def integral(x):
            return self.compute(x)

        # Integrate over the interval where the function is active
        area, _ = quad(integral, self.a, self.b)
        centroid, _ = quad(integrand, self.a, self.b)
        return centroid / area if area != 0 else float("nan")


class SMembershipFunction(MembershipFunction):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def compute(self, x: float) -> float:
        if x <= self.a:
            return 0.0
        elif x >= self.b:
            return 1.0
        else:
            return 2 * ((x - self.a) / (self.b - self.a)) ** 2

    def centroid(self) -> float:
        def integrand(x):
            return x * self.compute(x)

        def integral(x):
            return self.compute(x)

        # Integrate over the interval where the function is active
        area, _ = quad(integral, self.a, self.b)
        centroid, _ = quad(integrand, self.a, self.b)
        return centroid / area if area != 0 else float("nan")


class CombinedMembershipFunctions:
    def __init__(self):
        self._memberships: Dict[str, MembershipFunction] = {}
        self._value: float = None
        self._results: Dict[str, float] = {}

    def add_membership(self, name: str, function: MembershipFunction):
        self._memberships[name] = function

    def add_memberships(self, functions: Dict[str, Membership]):
        for name, function in functions.items():
            self._memberships[name] = function

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value
        self._results = self.fuzzify(value)

    @property
    def results(self) -> FuzzyVariable:
        return self._results.copy()

    @property
    def memberships(self) -> Dict[str, MembershipFunction]:
        return self._memberships

    def fuzzify(self, value: float) -> FuzzyVariable:
        results = {}
        for name, function in self._memberships.items():
            results[name] = function.compute(value)
        return results

    def defuzzify(self, percentages: FuzzyVariable) -> float:
        centroids = {
            name: function.centroid() for name, function in self._memberships.items()
        }

        weighted_sum = sum(
            centroids.get(name, 0) * percentages.get(name, 0) for name in percentages
        )

        total_weight = sum(percentages.values())

        return weighted_sum / total_weight if total_weight != 0 else float("nan")


class FuzzyInterface:
    def __init__(
        self,
        input_mfs: Dict[str, CombinedMembershipFunctions] = {},
    ) -> None:
        self.rules = []
        self.input_mfs: Dict[str, CombinedMembershipFunctions] = input_mfs

    def add_rule(
        self, condition: Callable[[FuzzyVariable], float], output: float
    ) -> None:
        self.rules.append((condition, output))

    def add_rules(self, rules: list[(Callable[[FuzzyVariable], float], float)]) -> None:
        for rule in rules:
            self.rules.append(rule)

    def evaluate_rules(self, input_values: Dict[str, float]) -> float:
        # Create a dictionary to store the fuzzified input values
        fuzzy_values = {
            name: self.input_mfs[name].fuzzify(value)
            for name, value in input_values.items()
        }

        print(json.dumps(fuzzy_values, indent=4))
        strength = 0

        for condition, output in self.rules:
            strength += output * condition(fuzzy_values)

        return strength


def main():
    temp_msf = CombinedMembershipFunctions()
    temp_msf.add_memberships(
        {
            "Cold": MembershipFunction.create(
                function="trapezoidal", a=-10, b=-5, c=5, d=15
            ),
            "Warm": MembershipFunction.create(function="triangular", a=10, b=20, c=30),
            "Hot": MembershipFunction.create(
                function="trapezoidal", a=25, b=30, c=40, d=45
            ),
        }
    )

    weather_msf = CombinedMembershipFunctions()
    weather_msf.add_memberships(
        {
            "Rainy": MembershipFunction.create("trapezoidal", a=0, b=0, c=30, d=50),
            "Clear": MembershipFunction.create("triangular", a=40, b=60, c=80),
            "Sunny": MembershipFunction.create("trapezoidal", a=70, b=80, c=100, d=100),
        }
    )

    # Create speed membership functions
    speed_msf = CombinedMembershipFunctions()
    speed_msf.add_memberships(
        {
            "Slow": MembershipFunction.create(
                function="trapezoidal", a=0, b=20, c=30, d=40
            ),
            "Medium": MembershipFunction.create(
                function="triangular", a=30, b=50, c=70
            ),
            "Fast": MembershipFunction.create(
                function="trapezoidal", a=60, b=80, c=100, d=120
            ),
        }
    )

    # Initialize the FuzzyInterface with input and output memberships
    fuzzy_interface = FuzzyInterface(
        input_mfs={"Temperature": temp_msf, "Weather": weather_msf},
        output_mfs=speed_msf,
    )

    # Add rules
    fuzzy_interface.add_rule(
        lambda values: min(values["Temperature"]["Cold"], values["Weather"]["Rainy"]),
        "Slow",
    )
    fuzzy_interface.add_rule(lambda values: values["Temperature"]["Warm"], "Medium")
    fuzzy_interface.add_rule(
        lambda values: values["Temperature"]["Hot"] * values["Weather"]["Sunny"], "Fast"
    )

    # Evaluate the rules with some input
    input_values = {"Temperature": 20, "Weather": 40}
    resulting_speed = fuzzy_interface.evaluate_rules(input_values)

    print(f"Recommended driving speed: {resulting_speed}")


if __name__ == "__main__":
    main()
