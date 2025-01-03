from typing import Tuple, List
from venv import logger
from pysimbotlib.core import Robot, PySimbotApp, Simbot
from base_robot import BaseRobot
from enum import Enum
from sensors import DirectionalDistances
from dataclasses import dataclass, fields
from itertools import product
import random
import json
from config import REFRESH_INTERVAL
from kivy.logger import Logger
import matplotlib.pyplot as plt
import numpy as np


class Action(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    EX_LEFT = 4
    EX_RIGHT = 5
    MOVE_LEFT = 6
    MOVE_RIGHT = 7


class Distance(Enum):
    NEAR = "near"
    MEDIUM = "medium"
    FAR = "far"


class FoodDistance(Enum):
    NEAR = "near"
    MIDDLE = "middle"
    FAR = "far"


class Angle(Enum):
    LEFT = "left"
    FRONT = "front"
    RIGHT = "right"


LEARNING_RATE: float = 0.5
DISCOUNT_FACTOR: float = 0.9
EXPLORATION_RATE: float = 0.1
# MIN_EXPLORATION_RATE: float = 0.1
# DECAY_RATE: float = 0.95

eat_counts = []
collision_counts = []
time_steps = []


@dataclass
class State:
    left_sensor: Distance
    front_left_sensor: Distance
    front_sensor: Distance
    front_right_sensor: Distance
    right_sensor: Distance
    food_distance: FoodDistance
    food_angle: Angle

    def __str__(self) -> str:
        return f"[{self.left_sensor.value}, {self.front_left_sensor.value}{self.front_sensor.value}, {self.front_right_sensor.value}, {self.right_sensor.value}, {self.food_distance.value}, {self.food_angle.value}]"

    def __hash__(self):
        return hash(
            (
                self.left_sensor,
                self.front_left_sensor,
                self.front_sensor,
                self.front_right_sensor,
                self.right_sensor,
                self.food_distance,
                self.food_angle,
            )
        )

    def __eq__(self, other):
        if isinstance(other, State):
            return (
                self.front_sensor == other.front_sensor
                and self.front_left_sensor == other.front_left_sensor
                and self.front_right_sensor == other.front_right_sensor
                and self.left_sensor == other.left_sensor
                and self.right_sensor == other.right_sensor
                and self.food_distance == other.food_distance
                and self.food_angle == other.food_angle
            )
        return False

    def to_dict(self):
        # Convert the object into a dictionary that can be saved to JSON
        return {
            "left_sensor": self.left_sensor.value,
            "front_left_sensor": self.front_left_sensor.value,
            "front_sensor": self.front_sensor.value,
            "front_right_sensor": self.front_right_sensor.value,
            "right_sensor": self.right_sensor.value,
            "food_distance": self.food_distance.value,
            "food_angle": self.food_angle.value,
        }


def after_simulation(simbot: Simbot):
    Logger.info("GA: Start GA Process ...")
    robot = simbot.robots[0]
    robot.export_qtable()


def graph():
    # Constants
    WINDOW_SIZE = 2000  # จำนวน ticks ต่อช่วง
    num_windows = len(time_steps) // WINDOW_SIZE  # จำนวนช่วงทั้งหมด

    # Calculate average rates per 1000 ticks
    avg_time_steps = []
    avg_eat_rates = []
    avg_collision_rates = []

    for i in range(num_windows):
        # Slice the data for the current window
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE

        # Calculate the averages
        avg_time_steps.append(sum(time_steps[start_idx:end_idx]) / WINDOW_SIZE)
        avg_eat_rates.append(
            sum(eat_counts[start_idx:end_idx]) / (WINDOW_SIZE * REFRESH_INTERVAL)
        )
        avg_collision_rates.append(
            sum(collision_counts[start_idx:end_idx]) / (WINDOW_SIZE * REFRESH_INTERVAL)
        )

    # Create the figure
    plt.figure(figsize=(14, 6))

    # Plot the Eat Rate
    plt.subplot(1, 2, 1)  # กราฟแรก
    plt.plot(
        avg_time_steps,
        avg_eat_rates,
        label="Average Eat Rate",
        color="green",
        marker="o",
    )
    plt.xlabel("Time Steps (Average)")
    plt.ylabel("Eat Rate (per unit time)")
    plt.title("Average Eat Rate Over Time (1000 Ticks Window)")
    plt.legend()
    plt.grid(True)

    # Plot the Collision Rate
    plt.subplot(1, 2, 2)  # กราฟที่สอง
    plt.plot(
        avg_time_steps,
        avg_collision_rates,
        label="Average Collision Rate",
        color="red",
        marker="x",
    )
    plt.xlabel("Time Steps (Average)")
    plt.ylabel("Collision Rate (per unit time)")
    plt.title("Average Collision Rate Over Time (1000 Ticks Window)")
    plt.legend()
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


class QLearnRobot(BaseRobot):
    def __init__(self) -> None:
        super().__init__()
        self.cur_state = State(
            front_sensor=Distance.FAR,
            front_left_sensor=Distance.FAR,
            front_right_sensor=Distance.FAR,
            left_sensor=Distance.FAR,
            right_sensor=Distance.FAR,
            food_distance=FoodDistance.FAR,
            food_angle=Angle.FRONT,
        )
        self.cur_action = Action.FORWARD
        self.qtable = self.create_qtable()
        self.exploration_rate = EXPLORATION_RATE

    def dist_threshold(
        self, distance: float, threshold_1: float, threshold_2: float
    ) -> Distance:
        if distance <= threshold_1:
            return Distance.NEAR
        elif distance <= threshold_2:
            return Distance.MEDIUM
        else:
            return Distance.FAR
        # return Distance.NEAR if distance <= threshold else Distance.FAR

    def get_enum_for_threshold(
        self, enum_class: Enum, thresholds: List[float], threshold_value: float
    ) -> Enum:
        if len(thresholds) != len(enum_class) - 1:
            raise ValueError(
                f"Thresholds list must have {len(enum_class) - 1} elements."
            )

        if threshold_value < thresholds[0]:
            return list(enum_class)[0]
        elif threshold_value >= thresholds[-1]:
            return list(enum_class)[-1]

        for idx, threshold in enumerate(thresholds):
            if threshold_value < threshold:
                return list(enum_class)[idx]

    def angle_threshold(self, angle: float, threshold: float) -> Angle:
        if angle < -1 * threshold:
            return Angle.LEFT
        elif angle > threshold:
            return Angle.RIGHT
        else:
            return Angle.FRONT

    def update_state(self) -> State:
        dist: DirectionalDistances = self.sensor_data.distances
        self.cur_state = State(
            left_sensor=self.dist_threshold(dist.left, 15, 30),
            front_left_sensor=self.dist_threshold(dist.front_left, 15, 30),
            front_sensor=self.dist_threshold(dist.front, 15, 30),
            front_right_sensor=self.dist_threshold(dist.front_right, 15, 30),
            right_sensor=self.dist_threshold(dist.right, 15, 30),
            food_distance=self.get_enum_for_threshold(
                FoodDistance, [100, 200], self.food_dist
            ),
            food_angle=self.angle_threshold(self.sensor_data.smell_nearest(), 30),
        )
        return self.cur_state

    def generate_all_possible_states(self):
        # Manually create all possible combinations of enum values
        all_states = []

        # Define the possible values for each field in State
        left_sensors = [Distance.NEAR, Distance.MEDIUM, Distance.FAR]
        front_left_sensors = [Distance.NEAR, Distance.MEDIUM, Distance.FAR]
        front_sensors = [Distance.NEAR, Distance.MEDIUM, Distance.FAR]
        front_right_sensors = [Distance.NEAR, Distance.MEDIUM, Distance.FAR]
        right_sensors = [Distance.NEAR, Distance.MEDIUM, Distance.FAR]
        food_distances = [FoodDistance.NEAR, FoodDistance.MIDDLE, FoodDistance.FAR]
        food_angles = [Angle.LEFT, Angle.FRONT, Angle.RIGHT]

        # Create combinations of the above values
        for (
            left_sensor,
            front_left_sensor,
            front_sensor,
            front_right_sensor,
            right_sensor,
            food_distance,
            food_angle,
        ) in product(
            left_sensors,
            front_left_sensors,
            front_sensors,
            front_right_sensors,
            right_sensors,
            food_distances,
            food_angles,
        ):
            state = State(
                left_sensor=left_sensor,
                front_left_sensor=front_left_sensor,
                front_sensor=front_sensor,
                front_right_sensor=front_right_sensor,
                right_sensor=right_sensor,
                food_distance=food_distance,
                food_angle=food_angle,
            )
            all_states.append(state)

        return all_states

    def create_qtable(self):
        all_possible_states = self.generate_all_possible_states()

        qtable = {}

        for state in all_possible_states:
            for action in Action:
                qtable[(state, action)] = 0

        return qtable

    def update_qtable(
        self, reward: int, learning_rate: float = 0.5, discount_fac: float = 0.9
    ):
        """
        Update the Q-value for the state-action pair based on the Q-learning rule.
        """
        current_q_value = self.qtable.get((self.cur_state, self.cur_action), 0)

        max_next_q_value = max(
            self.qtable.get((self.cur_state, next_action), 0) for next_action in Action
        )

        # Q-learning update rule
        new_q_value = current_q_value + learning_rate * (
            reward + discount_fac * max_next_q_value - current_q_value
        )

        self.qtable[(self.cur_state, self.cur_action)] = new_q_value

    def print_qtable(self):
        qtable = self.qtable
        # Define the headers
        headers = ["State", "Action", "Q-value"]

        # Find the maximum width for each column for pretty printing
        state_width = (
            max(len(str(state)) for state, _ in qtable.keys()) + 2
        )  # Corrected unpacking of keys
        action_width = (
            max(len(str(action)) for _, action in qtable.keys()) + 2
        )  # Corrected unpacking of keys
        qvalue_width = max(len(f"{qvalue:.2f}") for _, qvalue in qtable.items()) + 2

        # Print headers with appropriate spacing
        print(
            f"{headers[0]:<{state_width}}{headers[1]:<{action_width}}{headers[2]:<{qvalue_width}}"
        )
        print(
            "-" * (state_width + action_width + qvalue_width)
        )  # Print a separator line

        # Print each state, action, and Q-value
        for (state, action), qvalue in qtable.items():
            print(
                f"{str(state):<{state_width}}{str(action):<{action_width}}{qvalue:<{qvalue_width}.2f}"
            )

    def load_qtable(self, filename="qtable.json"):
        with open(filename, "r") as file:
            serializable_qtable = json.load(file)

        # Convert back to the original format with State objects
        for state_dict_str, action in serializable_qtable.keys():
            state_dict = json.loads(state_dict_str)
            state = State(
                left_sensor=Sensor(state_dict["left_sensor"]),
                front_left_sensor=Sensor(state_dict["front_left_sensor"]),
                front_sensor=Sensor(state_dict["front_sensor"]),
                front_right_sensor=Sensor(state_dict["front_right_sensor"]),
                right_sensor=Sensor(state_dict["right_sensor"]),
                food_distance=Sensor(state_dict["food_distance"]),
                food_angle=Sensor(state_dict["food_angle"]),
            )
            self.qtable[(state, action)] = serializable_qtable[(state_dict_str, action)]
        print(f"Q-table loaded from {filename}")

    def export_qtable(self, filename="qtable.json"):
        # Convert the Q-table to a JSON-serializable format
        serializable_qtable = {}
        for (state, action), qvalue in self.qtable.items():
            # Convert the state to a dictionary using the to_dict method
            state_dict = state.to_dict()
            serializable_qtable[(str(state_dict), action)] = qvalue

        # Save to JSON file
        with open(filename, "w") as file:
            json.dump(serializable_qtable, file, indent=4)
        print(f"Q-table exported to {filename}")

    def reward(self) -> int:
        reward = 0

        if self.cur_state.food_distance == FoodDistance.FAR:
            reward += 1
        elif self.cur_state.food_distance == FoodDistance.MIDDLE:
            reward += 3
        elif self.cur_state.food_distance == FoodDistance.NEAR:
            reward += 5

        if self.cur_action == Action.FORWARD:
            reward += 2

        # if self.cur_action == Action.LEFT or self.cur_action == Action.RIGHT:
        #     print("A")
        #     reward -= 2

        # if self.cur_action == Action.EX_LEFT or self.cur_action == Action.EX_RIGHT:
        #     print("B")
        #     reward -= 10

        if (
            self.cur_action == Action.FORWARD
            and self.cur_state.food_angle == Angle.FRONT
        ):
            reward += 5

        if (
            self.cur_action != Action.FORWARD
            or self.cur_action != Action.MOVE_LEFT
            or self.cur_action != Action.MOVE_RIGHT
        ) and self.cur_state.food_angle == Angle.FRONT:
            reward -= 5

        if self.collision:
            reward -= 20

        if self.just_eat:
            reward = 100

        return reward

    def choose_action(self):
        if random.uniform(0, 1) < self.exploration_rate:
            self.cur_action = random.choice(list(Action))
        else:
            q_values = [
                self.qtable.get((self.cur_state, action), 0) for action in Action
            ]
            # print(q_values)
            max_q_value = max(q_values)
            max_actions = [
                action
                for action, value in zip(Action, q_values)
                if value == max_q_value
            ]
            # print(max_actions)
            self.cur_action = random.choice(max_actions)

        # if self._sm.iteration % 100 == 0:
        #     self.exploration_rate = max(
        #         MIN_EXPLORATION_RATE, self.exploration_rate * DECAY_RATE
        #     )
        #     print("%.4f" % self.exploration_rate)

        if self.cur_action == Action.FORWARD:
            self.move(10)
        elif self.cur_action == Action.LEFT:
            self.turn(-20)
        elif self.cur_action == Action.RIGHT:
            self.turn(20)
        elif self.cur_action == Action.MOVE_LEFT:
            self.move(5)
            self.turn(-20)
        elif self.cur_action == Action.MOVE_RIGHT:
            self.move(5)
            self.turn(20)
        elif self.cur_action == Action.EX_LEFT:
            self.turn(-90)
        elif self.cur_action == Action.EX_RIGHT:
            self.turn(90)

        return self.cur_action

    def create_move_strategy(self):
        pass

    def create_turn_strategy(self):
        pass

    def update(self):

        # Track eat and collision counts per iteration
        eat_counts.append(1 if self.just_eat else 0)
        collision_counts.append(1 if self.collision else 0)
        time_steps.append(self._sm.iteration)
        self.choose_action()
        self.update_state()
        reward = self.reward()
        # print("reward:", reward)
        self.update_qtable(
            reward=reward, learning_rate=LEARNING_RATE, discount_fac=DISCOUNT_FACTOR
        )
        # self.print_qtable()


def main():
    """Main application entry point"""
    Logger.info("Starting PySimbotApp with RL robot.")
    try:
        app = PySimbotApp(
            robot_cls=QLearnRobot,
            num_robots=1,
            max_tick=100000,
            interval=REFRESH_INTERVAL,
            map="default_map2",
            simulation_forever=False,
            food_move_after_eat=True,
            # customfn_before_simulation=before_simulation,
            customfn_after_simulation=after_simulation,
        )
        app.run()
    except Exception as e:
        Logger.critical("Critical failure in PySimbotApp:", exc_info=True)


if __name__ == "__main__":
    main()

graph()
