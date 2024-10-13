#!/usr/bin/python3

import os, platform
import random

if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"

from pysimbotlib.core import PySimbotApp, Robot, Simbot
from kivy.logger import Logger
from kivy.config import Config

from pysimbotlib.core.Util import Util
import csv
import matplotlib.pyplot as plt

# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set("kivy", "log_level", "info")

REFRESH_INTERVAL = 1

next_gen_robots = []  # Define the next_gen_robots list
avg_fitness_value_list = []  # Define the avg_fitness_value_list list
max_fitness_value_list = []  # Define the max_fitness_value_list list
# Initialize a list to store the best fitness values of the last few generations

not_change = 0

# Define the number of generations over which to check for changes in fitness
FITNESS_CHECK_GENERATIONS = 50

# Define the threshold for the change in fitness
FITNESS_CHANGE_THRESHOLD = 10


def write_rule(robot, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(robot.RULES)


def read_rule(robot, filename):
    pass
    # simbot.robot[0].RULES = list(csv.reader(open("best_1.csv")))


def before_simulation(simbot: Simbot):
    Logger.info("GA: initial population")
    for robot in simbot.robots:
        # random RULES value for the first generation
        if simbot.simulation_count == 0:
            for i, RULE in enumerate(robot.RULES):
                for k in range(len(RULE)):
                    robot.RULES[i][k] = random.randrange(256)
        # used the calculated RULES value from the previous generation
        else:
            # Logger.info("GA: copy the rules from previous generation")
            for simbot_robot, robot_from_last_gen in zip(
                simbot.robots, next_gen_robots
            ):
                simbot_robot.RULES = robot_from_last_gen.RULES


def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")
    global not_change
    for robot in simbot.robots:
        print(robot.collision_count)
        food_pos = simbot.objectives[0].pos
        robot_pos = robot.pos
        distance = Util.distance(food_pos, robot_pos)
        robot.fitness = 1000 - int(distance) + int(robot.total_back_move)
        robot.fitness -= robot.total_stop * 10
        robot.fitness -= robot.collision_count * 5
        robot.fitness -= robot.spin_count * 10
        robot.fitness += (200 - robot.time_to_eat) * 5

    # descending sort and rank: the best 10 will be on the list at index 0 to 9
    simbot.robots.sort(key=lambda robot: robot.fitness, reverse=True)

    # empty the list
    next_gen_robots.clear()
    ELITINSM = 0.1
    num_elites = int(ELITINSM * len(simbot.robots))

    # adding the best to the next generation.

    next_gen_robots.extend(simbot.robots[:num_elites])
    num_robots = len(simbot.robots)

    def select():
        # Generate a random number between 0 and the sum of all ranks
        rand = random.uniform(0, num_robots * (num_robots + 1) / 2)
        accum = 0

        # Go through the individuals, starting from the best
        for i in range(num_robots):
            # Add the rank of the current individual
            accum += num_robots - i

            # If the accumulated rank is greater than the random number, select this individual
            if accum > rand:
                return simbot.robots[i]

    for _ in range(num_robots - num_elites):
        select1 = select()  # design the way for selection by yourself
        select2 = select()  # design the way for selection by yourself

        while select1 == select2:
            select2 = select()

        # Doing crossover
        crossover_point = random.randint(0, simbot.robots[0].RULE_LENGTH - 1)

        # Create the first offspring
        offspring1 = StupidRobot()
        for i in range(simbot.robots[0].NUM_RULES):
            offspring1.RULES[i] = (
                select1.RULES[i][:crossover_point] + select2.RULES[i][crossover_point:]
            )

        # print(offspring1.RULES)
        # using next_gen_robots for temporary keep the offsprings, later they will be copy
        # to the robots
        # Define the mutation rate
        mutation_rate = 0.01
        sweak_rate = 0.05
        for i in range(simbot.robots[0].NUM_RULES):
            for j in range(simbot.robots[0].RULE_LENGTH):
                if random.random() < mutation_rate:
                    offspring1.RULES[i][j] = random.randrange(256)
                if random.random() < sweak_rate:
                    offspring1.RULES[i][j] = offspring1.RULES[i][j] + random.choice(
                        range(-5, 6)
                    )

        next_gen_robots.append(offspring1)

    # # write the best rule to file
    # write_rule(simbot.robots[0], "best_gen{0}.csv".format(simbot.simulation_count))

    avg_fitness = sum(robot.fitness for robot in simbot.robots) / len(simbot.robots)
    max_fitness = max(robot.fitness for robot in simbot.robots)
    avg_fitness_value_list.append(avg_fitness)
    max_fitness_value_list.append(max_fitness)

    print(f"Average fitness: {avg_fitness}")
    print(f"Maximum fitness: {max_fitness}")

    # If we have enough fitness values, check if the fitness has changed significantly
    if len(max_fitness_value_list) > FITNESS_CHECK_GENERATIONS:
        # Calculate the change in fitness
        if (
            abs(max_fitness_value_list[-1] - max_fitness_value_list[-2])
            < FITNESS_CHANGE_THRESHOLD
        ):
            not_change += 1
        else:
            not_change = 0
        # If the fitness hasn't changed much, stop the simulation
        if not_change > 20:
            print("Fitness hasn't changed significantly. Stopping the simulation.")


class StupidRobot(Robot):
    RULE_LENGTH = 11
    NUM_RULES = 10

    def __init__(self, **kwarg):
        super(StupidRobot, self).__init__(**kwarg)
        self.RULES = [[0] * self.RULE_LENGTH for _ in range(self.NUM_RULES)]
        # initial list of rules
        self.rules = [0.0] * self.NUM_RULES
        self.turns = [0.0] * self.NUM_RULES
        self.moves = [0.0] * self.NUM_RULES
        self.total_back_move = 0
        self.fitness = 0
        self.prev_pos = [self.pos, self.pos]
        self.prev_direction = [self._direction, self._direction]
        self.time = 0
        self.total_stop = 0
        self.time_to_eat = 200

    def update(self):
        """Update method which will be called each frame"""
        if self.just_eat and self.time_to_eat == 200:
            self.time_to_eat = self.time

        self.time += 1
        self.ir_values = self.distance()
        (
            self.S0,
            self.S1,
            self.S2,
            self.S3,
            self.S4,
            self.S5,
            self.S6,
            self.S7,
        ) = self.ir_values
        self.target = self.smell()
        for i, RULE in enumerate(self.RULES):
            self.rules[i] = 1.0
            for k, RULE_VALUE in enumerate(RULE):
                if k < 8:
                    if RULE_VALUE % 5 == 1:
                        if k == 0:
                            self.rules[i] *= self.near(self.S0)
                        elif k == 1:
                            self.rules[i] *= self.near(self.S1)
                        elif k == 2:
                            self.rules[i] *= self.near(self.S2)
                        elif k == 3:
                            self.rules[i] *= self.near(self.S3)
                        elif k == 4:
                            self.rules[i] *= self.near(self.S4)
                        elif k == 5:
                            self.rules[i] *= self.near(self.S5)
                        elif k == 6:
                            self.rules[i] *= self.near(self.S6)
                        elif k == 7:
                            self.rules[i] *= self.near(self.S7)
                    elif RULE_VALUE % 5 == 2:
                        if k == 0:
                            self.rules[i] *= self.far(self.S0)
                        elif k == 1:
                            self.rules[i] *= self.far(self.S1)
                        elif k == 2:
                            self.rules[i] *= self.far(self.S2)
                        elif k == 3:
                            self.rules[i] *= self.far(self.S3)
                        elif k == 4:
                            self.rules[i] *= self.far(self.S4)
                        elif k == 5:
                            self.rules[i] *= self.far(self.S5)
                        elif k == 6:
                            self.rules[i] *= self.far(self.S5)
                        elif k == 7:
                            self.rules[i] *= self.far(self.S7)
                elif k == 8:
                    temp_val = RULE_VALUE % 6
                    if temp_val == 1:
                        self.rules[i] *= self.smell_left()
                    elif temp_val == 2:
                        self.rules[i] *= self.smell_center()
                    elif temp_val == 3:
                        self.rules[i] *= self.smell_right()
                elif k == 9:
                    self.turns[i] = (RULE_VALUE % 181) - 90
                elif k == 10:
                    self.moves[i] = (RULE_VALUE % 21) - 10

        answerTurn = 0.0
        answerMove = 0.0

        for turn, move, rule in zip(self.turns, self.moves, self.rules):
            answerTurn += turn * rule
            answerMove += move * rule

        if answerMove < 0:
            self.total_back_move += answerMove
        if answerMove == 0:
            self.total_stop += 1

        if (
            Util.distance(self.prev_pos[1], self.pos) < 10
            and abs(self._direction - self.prev_direction[1]) > 100
        ):
            self.spin_count += 1
        else:
            self.spin_count = 0

        # Update the previous direction and position

        self.prev_direction[1] = self.prev_direction[0]
        self.prev_direction[0] = self._direction
        self.prev_pos[1] = self.prev_pos[0]
        self.prev_pos[0] = self.pos

        self.turn(answerTurn)
        self.move(answerMove)

    def near(self, dist):
        if dist <= 0:
            return 1.0
        elif dist >= 100:
            return 0.0
        else:
            return 1 - (dist / 100.0)

    def far(self, dist):
        if dist <= 0:
            return 0.0
        elif dist >= 100:
            return 1.0
        else:
            return dist / 100.0

    def smell_right(self):
        if self.target >= 45:
            return 1.0
        elif self.target <= 0:
            return 0.0
        else:
            return self.target / 45.0

    def smell_left(self):
        if self.target <= -45:
            return 1.0
        elif self.target >= 0:
            return 0.0
        else:
            return 1 - (-1 * self.target) / 45.0

    def smell_center(self):
        if self.target <= 45 and self.target >= 0:
            return self.target / 45.0
        if self.target <= -45 and self.target <= 0:
            return 1 - (-1 * self.target) / 45.0
        else:
            return 0.0


if __name__ == "__main__":
    app = PySimbotApp(
        robot_cls=StupidRobot,
        num_robots=50,
        max_tick=200,
        interval=1 / 1000.0,
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
