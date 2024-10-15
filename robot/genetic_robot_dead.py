from typing import Callable, List
from kivy.logger import Logger
from pysimbotlib.core import Robot, PySimbotApp, Simbot
from pysimbotlib.core.Util import Util
from collections import deque
import numpy as np
from sensors import SensorData

from strategies import Move, Turn, GeneticMove, GeneticTurn
from fuzzy_logic import CombinedMembershipFunctions, MembershipFunction

from GeneticAlgorithm import (
    GeneticAlgorithm,
    Chromosome,
    Genotype,
    TournamentSelection,
    RankBasedSelection,
    OnePointCrossover,
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
    population_size=20,
    elitism_percentage=0.1,
    selection_strategy=TournamentSelection,
    crossover_strategy=OnePointCrossover,
    mutation_strategy=CompositeMutation(
        strategies=[
            RandomResetMutation(mutation_probability=0.1, gene_type=["RuleGene"]),
            GaussianMutation(mutation_probability=0.05, stddev=20.0),
        ]
    ),
)

death_counts = []
avg_fitness_value_list = []
max_fitness_value_list = []
current_tick = 0
TICK_INTERVAL = 40000


def before_simulation(simbot: Simbot):
    Logger.info("GA: initial population")


def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")

    # Calculate fitness value for each robot
    # fitness: List[float] = []

    for robot in simbot.robots:
        fitness_value = 0

        food_pos = simbot.objectives[0].pos
        robot_pos = robot.pos
        distance = Util.distance(food_pos, robot_pos)

        fitness_value = 1000 - int(distance) + int(robot.total_back_move)
        fitness_value -= robot.total_stop * 10
        # fitness_value -= robot.collision_count * 5
        # fitness_value -= robot.spin_count * 10
        fitness_value += (200 - robot.time_to_eat) * 5

        # fitness.append(fitness_value)

    # avg_fitness_value_list.append(sum(fitness) / len(fitness))
    # max_fitness_value_list.append(max(fitness))

    # genetic_algorithm.fitness_scores = fitness
    # genetic_algorithm.create_next_generation()



class GeneticRobot(Robot):
    SAFE_DIST: float = 30.0
    CLOSE_DIST: float = 5.0
    HIT_DIST: float = 0.0
    MOVE_SPEED: float = 10.0
    TURN_SPEED: float = 20.0
    TURN_SHARP_SPEED: float = 50.0
    MAX_ENERGY: int = 500
    WINDOW = 100

    def __init__(self, **kwarg) -> None:
        super(GeneticRobot, self).__init__(**kwarg)
        self.sensor_data = self.sensor()
        self.genotype = genotype
        genotype.scamble()
        self.move_strategy: Move = self.create_move_strategy()
        self.turn_strategy: Turn = self.create_turn_strategy()

        self.energy = self.MAX_ENERGY
        self.lazy_count = 0
        self.headache_count = 0
        self.total_back_move = 0
        self.time = 0
        self.total_stop = 0
        self.time_to_eat = 200
        self.death_count = 0
        self.previous_positions = deque(maxlen=50)
        self.just_hit = 0
        self.total_distance_moved = 0
        self.circular_movement_count = 0

    def calculate_fitness(self) -> float:
        fitness_value = 1000
        # food_pos = self._sm.objectives[0].pos
        # robot_pos = self.pos
        # distance = Util.distance(food_pos, robot_pos)

        distance = self.smell_nearest()

        fitness_value -= int(distance) * 20
        # print("1", fitness_value)
        fitness_value -= int(self.total_back_move) * 10
        # print("2", self.total_back_move, fitness_value)
        fitness_value -= self.total_stop * 10
        # print("3", self.total_stop, fitness_value)
        # fitness_value -= self.collision_count * 2
        # print("4", self.time)
        fitness_value += self.time * 1
        fitness_value -= self.circular_movement_count * 10
        # print("5", fitness_value)
        fitness_value += self.eat_count * 1000
        # print("6", fitness_value)

        return fitness_value

    def clear_stat(self) -> None:
        self.time = 0
        self.collision_count = 0
        self.total_stop = 0
        self.lazy_count = 0
        self.headache_count = 0
        self.total_back_move = 0
        self.circular_movement_count = 0
        self.eat_count = 0
        self.time_to_eat = 200
        self.total_distance_moved = 0
        self.previous_positions.clear()

    def create_move_strategy(self) -> Move:
        return GeneticMove(sensor=self.sensor_data, genotype=self.genotype)

    def create_turn_strategy(self) -> Turn:
        return GeneticTurn(sensor=self.sensor_data, genotype=self.genotype)

    def change_color(self) -> None:
        if self.energy < 100:
            self.set_color(255, 255, 0, 1)
        elif self.energy < 300:
            self.set_color(255, 0, 0, 1)
        elif self.energy < 500:
            self.set_color(0, 255, 0, 1)
        elif self.energy < 700:
            self.set_color(0, 0, 255, 1)
        elif self.energy < 900:
            self.set_color(255, 0, 255, 1)
        elif self.energy < 1200:
            self.set_color(0, 0, 0, 1)

    def is_moving_in_circle(self) -> bool:
        if len(self.previous_positions) < 50:
            return False

        # Convert positions to numpy array for easier calculations
        positions = np.array(self.previous_positions)

        # Calculate the center of the positions
        center = np.mean(positions, axis=0)

        # Calculate the distances from each point to the center
        distances = np.linalg.norm(positions - center, axis=1)

        # Calculate the standard deviation of the distances
        std_dev = np.std(distances)

        # If the standard deviation is low, the points are likely in a circle
        # You may need to adjust this threshold based on your specific scenario
        return std_dev < 5
 

    def update(self):
        global death_counts, current_tick
        fitness: List[float] = []
        try:

            # if self.just_eat and self.time_to_eat == 200:
            #     self.time_to_eat = self.time
            self.time += 1

            self.change_color()

            current_pos = self.pos
            self.previous_positions.append(current_pos)

            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            move_value: float = self.move_strategy.calculate()
            self.move(move_value)

            if move_value < 0:
                self.total_back_move -= move_value

            if move_value < 1:
                self.total_stop += 1

            self.energy -= 1

            # Calculate distance moved
            if len(self.previous_positions) >= 2:
                self.total_distance_moved += Util.distance(self.previous_positions[-2], current_pos)

            # Check for circular movement
            if len(self.previous_positions) == 50:
                if self.is_moving_in_circle():
                    self.circular_movement_count += 1
                    self.energy -= 15 

            # if self.stuck:
            #     # print("stuck")
            #     self.energy -= 10

            if int(move_value) == 0 and int(turn_value) == 0:
                self.lazy_count += 1

            if int(move_value) < 0 or abs(turn_value) > 30:
                self.headache_count += 1

            # if the robot is lazy
            if self.lazy_count:
                self.energy -= 25

            if self.just_hit :
                self.energy -= 60

            # if the robot headache
            if self.headache_count:
                self.energy -= 20   

            if self.just_eat:
                self.energy += 500

            if self.is_dead():
                self.death_count += 1
                death_counts.append(current_tick)
                
                genetic_algorithm.population = [
                    robot.genotype for robot in self._sm.robots
                ]
                genetic_algorithm.fitness_scores = [
                    robot.calculate_fitness() for robot in self._sm.robots
                    # fitness.append(robot.calculate_fitness())
                ]
                self.genotype = genetic_algorithm.create_new_genotype()
                self.move_strategy: Move = self.create_move_strategy()
                self.turn_strategy: Turn = self.create_turn_strategy()
                self.clear_stat()
                self.energy = 400
                
                
            
            self.time += 1
            current_tick += 1

        

        except Exception as e:
            Logger.error("Error during robot update:", exc_info=True)

    def is_dead(self) -> bool:
        return self.energy < 0

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

def plot_death_counts():
    global current_tick 

    last_tick = current_tick
    num_intervals = (last_tick // TICK_INTERVAL) + 1
    aggregated_counts = [0] * num_intervals

    for death_tick in death_counts:
        interval_index = death_tick // TICK_INTERVAL
        if interval_index < num_intervals: 
            aggregated_counts[interval_index] += 1

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_intervals), aggregated_counts, align='center')
    plt.title(f'Death Counts per 2000Ticks')
    plt.xlabel(f'Tick Interval')
    plt.ylabel('Number of Deaths')
    
    x_ticks = range(0, num_intervals, max(1, num_intervals // 10))
    plt.xticks(x_ticks, [f'{i*2000}' for i in x_ticks], rotation=45)
    
    plt.tight_layout()
    plt.show()

    print(f"Simulation ran for {last_tick} ticks. Death counts plot has been displayed.")

if __name__ == "__main__":
    app = PySimbotApp(
        robot_cls=GeneticRobot,
        num_robots=20,
        num_objectives=4,
        max_tick=100000,
        interval=REFRESH_INTERVAL,
        theme="default",
        map="default_map2",
        simulation_forever=False,
        food_move_after_eat=True,
        robot_see_each_other=True,
        customfn_before_simulation=before_simulation,
        customfn_after_simulation=after_simulation,
    )
    app.run()

    plot_death_counts()
    print("Death counts plot has been displayed. Close the plot window to end the program.")

        # Plot Dead value
    # plt.figure()
    # plt.plot(death_value_list)
    # plt.title("Death Vale Overtime")
    # plt.xlabel("Time")
    # plt.ylabel("Death")

    # plt.show()