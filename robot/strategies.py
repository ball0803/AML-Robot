from typing import Callable, Dict
from abc import ABC, abstractmethod
from kivy.logger import Logger
from sensors import SensorData, DirectionalDistances
from fuzzy_logic import (
    CombinedMembershipFunctions,
    MembershipFunction,
    FuzzyInterface,
    FuzzyVariable,
)
import random


class Turn(ABC):
    @abstractmethod
    def calculate(self) -> float:
        pass


class Move(ABC):
    @abstractmethod
    def calculate(self) -> float:
        pass


class ReactiveTurn(Turn):
    def __init__(
        self,
        sensor: SensorData,
        calculate_smooth_turn: Callable[[float], float],
        calculate_smooth_sharp_turn: Callable[[float], float],
    ) -> None:
        self.sensor = sensor
        self.calculate_smooth_turn = calculate_smooth_turn
        self.calculate_smooth_sharp_turn = calculate_smooth_sharp_turn

    def calculate(self) -> float:
        if self.sensor.is_front_safe():
            Logger.info("Safe Distance detected. Calculating smooth turn.")
            turn_value = self.calculate_smooth_turn(abs(self.sensor.smell()))
            return -turn_value if self.sensor.smell_food_on_left() else turn_value
        elif self.sensor.is_front_safe_either_close_by():
            Logger.warning("Object close by. Adjusting turn.")
            turn_value = self.calculate_smooth_turn(
                abs(
                    self.sensor.distances.front_left - self.sensor.distances.front_right
                )
            )
            return (
                -turn_value
                if self.sensor.distances.front_left > self.sensor.distances.front_right
                else turn_value
            )
        elif self.sensor.is_about_to_hit():
            Logger.error("Collision imminent! Executing sharp turn.")
            turn_value = self.calculate_smooth_sharp_turn(
                abs(
                    self.sensor.distances.front_left - self.sensor.distances.front_right
                )
            )
            return (
                -turn_value
                if self.sensor.distances.front_left > self.sensor.distances.front_right
                else turn_value
            )
        return 0


class ReactiveMove(Move):
    def __init__(
        self,
        sensor: SensorData,
        calculate_smooth_move: Callable[[float], float],
    ) -> None:
        self.sensor = sensor
        self.calculate_smooth_move = calculate_smooth_move

    def calculate(self) -> float:
        if self.sensor.is_front_safe_both_close_by():
            Logger.info("Front Sensor is clear.")
            return self.calculate_smooth_move(self.sensor.distances.front)
        return 0


class FuzzyTurn(Turn):
    def __init__(
        self,
        sensor: SensorData,
        interface_left: FuzzyInterface,
        interface_right: FuzzyInterface,
    ) -> None:
        self.sensor = sensor
        self.fuzzy_interface_left = interface_left
        self.fuzzy_interface_right = interface_right

    def calculate(self) -> float:
        sensor: Dict[str, float] = self.sensor.distances_as_dict()
        sensor["smell"] = self.sensor.smell_nearest_degree(offset=45)
        right = self.fuzzy_interface_right.evaluate_rules(sensor)
        left = self.fuzzy_interface_left.evaluate_rules(sensor)
        print(f"left: {left} right: {right}")
        turn_value = right - left
        return turn_value


class FuzzyMove(Move):
    def __init__(
        self,
        sensor: SensorData,
        interface_front: FuzzyInterface,
        interface_back: FuzzyInterface,
    ) -> None:
        self.sensor = sensor
        self.fuzzy_interface_front = interface_front
        self.fuzzy_interface_back = interface_back

    def calculate(self) -> float:
        sensor: Dict[str, float] = self.sensor.distances_as_dict()
        front = self.fuzzy_interface_front.evaluate_rules(sensor)
        back = self.fuzzy_interface_back.evaluate_rules(sensor)
        move_value = front - back
        return move_value


def distance() -> list:
    return [random.uniform(0, 100) for _ in range(8)]


def smell() -> list:
    return random.uniform(0, 100)


def smell_nearest() -> list:
    return random.uniform(0, 100)


def main():
    dummy_sensor = SensorData(
        distances=distance, smell=smell, smell_nearest=smell_nearest
    )
    turn = FuzzyTurn(sensor=dummy_sensor)
    turn_value = turn.calculate()
    print(turn_value)


if __name__ == "__main__":
    main()
