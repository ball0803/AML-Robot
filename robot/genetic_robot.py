from typing import Callable, List
from kivy.logger import Logger
from pysimbotlib.core import Robot, PySimbotApp, Simbot
from pysimbotlib.core.Util import Util

from sensors import SensorData

from strategies import Move, Turn, GeneticMove, GeneticTurn
from fuzzy_logic import CombinedMembershipFunctions, MembershipFunction

from GeneticAlgorithm import (
    GeneticAlgorithm,
    Chromosome,
    Genotype,
    TournamentSelection,
    TwoPointCrossover,
    CompositeMutation,
    RandomResetMutation,
    GaussianMutation,
)

from config import REFRESH_INTERVAL
import random
import os, platform
import matplotlib.pyplot as plt

if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"

SIDE = [
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
]

dist_msf = CombinedMembershipFunctions()
dist_msf.add_membership(
    "near", MembershipFunction.create(function="triangular", a=0, b=0, c=100)
)
dist_msf.add_membership(
    "far", MembershipFunction.create(function="triangular", a=0, b=100, c=100)
)

smell_msf = CombinedMembershipFunctions()
smell_msf.add_membership(
    "smell_left",
    MembershipFunction.create(function="triangular", a=-45, b=-45, c=0),
)
smell_msf.add_membership(
    "smell_center",
    MembershipFunction.create(function="triangular", a=-45, b=0, c=45),
)
smell_msf.add_membership(
    "smell_right",
    MembershipFunction.create(function="triangular", a=0, b=45, c=45),
)

chromosome = Chromosome()

for side in SIDE:
    chromosome.add_rule_gene(
        name=side,
        value="_",
        mapping=dict(
            _=lambda **_: 1,
            __=lambda **_: 1,
            ___=lambda **_: 1,
            near=lambda **args: dist_msf.fuzzify(args["x"])["near"],
            far=lambda **args: dist_msf.fuzzify(args["x"])["far"],
        ),
    )

chromosome.add_rule_gene(
    name="smell_direction",
    value="_",
    mapping=dict(
        _=lambda **_: 1,
        __=lambda **_: 1,
        ___=lambda **_: 1,
        smell_left=lambda **args: smell_msf.fuzzify(args["x"])["smell_left"],
        smell_center=lambda **args: smell_msf.fuzzify(args["x"])["smell_center"],
        smell_right=lambda **args: smell_msf.fuzzify(args["x"])["smell_right"],
    ),
)

chromosome.add_return_gene(
    name="turn",
    value=180.0,
    func=lambda x: (x % 181.0) - 90.0,
)

chromosome.add_return_gene(
    name="move",
    value=10.0,
    func=lambda x: (x % 21.0) - 10.0,
)

genotype = Genotype()

for _ in range(10):
    genotype.add_chromosome(chromosome.clone())


genetic_algorithm = GeneticAlgorithm(
    population_size=50,
    elitism_percentage=0.1,
    selection_strategy=TournamentSelection,
    crossover_strategy=TwoPointCrossover,
    mutation_strategy=CompositeMutation(
        strategies=[
            RandomResetMutation(mutation_probability=0.1, gene_type=["RuleGene"]),
            GaussianMutation(mutation_probability=0.05, stddev=20.0),
        ]
    ),
)

avg_fitness_value_list = []
max_fitness_value_list = []


def before_simulation(simbot: Simbot):
    Logger.info("GA: initial population")
    if simbot.simulation_count == 0:
        genetic_algorithm.initialize_population(genotype_template=genotype)

    for i, robot in enumerate(simbot.robots):
        robot.genotype = genetic_algorithm.population[i].clone()
        robot.move_strategy = robot.create_move_strategy()
        robot.turn_strategy = robot.create_turn_strategy()
        robot.i = i

    # for simbot_robot in simbot.robots:
    #     print(simbot_robot.genotype)


def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")

    # Calculate fitness value for each robot
    fitness: List[float] = []

    for robot in simbot.robots:
        fitness_value = 0

        food_pos = simbot.objectives[0].pos
        robot_pos = robot.pos
        distance = Util.distance(food_pos, robot_pos)

        fitness_value = 1000 - int(distance) + int(robot.total_back_move)
        fitness_value -= robot.total_stop * 10
        fitness_value -= robot.collision_count * 5
        fitness_value -= robot.spin_count * 10
        fitness_value += (200 - robot.time_to_eat) * 5

        fitness.append(fitness_value)

    avg_fitness_value_list.append(sum(fitness) / len(fitness))
    max_fitness_value_list.append(max(fitness))

    genetic_algorithm.fitness_scores = fitness
    genetic_algorithm.create_next_generation()


class GeneticRobot(Robot):
    SAFE_DIST: float = 30.0
    CLOSE_DIST: float = 5.0
    HIT_DIST: float = 0.0
    MOVE_SPEED: float = 10.0
    TURN_SPEED: float = 20.0
    TURN_SHARP_SPEED: float = 50.0

    def __init__(self, **kwarg) -> None:
        super(GeneticRobot, self).__init__(**kwarg)
        self.sensor_data = self.sensor()
        self.genotype = genotype
        self.move_strategy: Move = self.create_move_strategy()
        self.turn_strategy: Turn = self.create_turn_strategy()

        self.total_back_move = 0
        self.spin_count = 0

        self.prev_pos = [self.pos, self.pos]
        self.prev_direction = [self._direction, self._direction]

        self.time = 0
        self.total_stop = 0
        self.time_to_eat = 200

    def create_move_strategy(self) -> Move:
        # print(self.genotype)
        return GeneticMove(sensor=self.sensor_data, genotype=self.genotype)

    def create_turn_strategy(self) -> Turn:
        return GeneticTurn(sensor=self.sensor_data, genotype=self.genotype)

    def update(self):
        try:
            if (
                Util.distance(self.prev_pos[1], self.pos) < 10
                and abs(self._direction - self.prev_direction[1]) > 100
            ):
                self.spin_count += 1
            else:
                self.spin_count = 0

            if self.just_eat and self.time_to_eat == 200:
                self.time_to_eat = self.time

            self.time += 1
            self.prev_direction[1] = self.prev_direction[0]
            self.prev_direction[0] = self._direction
            self.prev_pos[1] = self.prev_pos[0]
            self.prev_pos[0] = self.pos

            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            move_value: float = self.move_strategy.calculate()
            self.move(move_value)
            # Logger.debug(self.genotype)

            # Logger.debug(
            #     f"Computed move value:{self.i} {move_value}, turn value: {turn_value}"
            # )
            # Logger.info("Robot state updated successfully.")
        except Exception as e:
            Logger.error("Error during robot update:", exc_info=True)

    def sensor(self) -> SensorData:
        return SensorData(
            distances=super().distance,
            smell=super().smell,
            smell_nearest=super().smell_nearest,
            stuck=super().stuck,
            safe_dist=self.SAFE_DIST,
            close_dist=self.CLOSE_DIST,
            hit_dist=self.HIT_DIST,
        )


if __name__ == "__main__":
    app = PySimbotApp(
        robot_cls=GeneticRobot,
        num_robots=50,
        max_tick=200,
        interval=REFRESH_INTERVAL,
        simulation_forever=True,
        customfn_before_simulation=before_simulation,
        customfn_after_simulation=after_simulation,
        food_move_after_eat=False,
    )
    app.run()
    # Plot the average fitness values
    plt.figure()
    plt.plot(avg_fitness_value_list)
    plt.title("Average Fitness Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")

    # Plot the maximum fitness values
    plt.figure()
    plt.plot(max_fitness_value_list)
    plt.title("Maximum Fitness Over Time")
    plt.xlabel("Generation")
    plt.ylabel("Maximum Fitness")

    plt.show()
