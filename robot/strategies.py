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
from GeneticAlgorithm import Genotype
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
        interface: FuzzyInterface,
    ) -> None:
        self.sensor = sensor
        self.fuzzy_interface = interface

    def calculate(self) -> float:
        sensor: Dict[str, float] = self.sensor.distances_as_dict()
        sensor["smell"] = self.sensor.smell_nearest_degree(offset=45)
        turn_value = self.fuzzy_interface.evaluate_rules(sensor)
        return turn_value


class FuzzyMove(Move):
    def __init__(
        self,
        sensor: SensorData,
        interface: FuzzyInterface,
    ) -> None:
        self.sensor = sensor
        self.fuzzy_interface = interface

    def calculate(self) -> float:
        sensor: Dict[str, float] = self.sensor.distances_as_dict()
        move_value = self.fuzzy_interface.evaluate_rules(sensor)

        return move_value


class GeneticTurn(Turn):
    def __init__(self, sensor: SensorData, genotype: Genotype):
        self.sensor = sensor
        self.genotype = genotype

    def calculate(self) -> float:
        self.sensor.distances_as_dict()
        turn, _ = self.genotype.evaluate(
            args={
                **self.sensor.distances_as_input(),
                "smell_direction": {"x": self.sensor.smell_nearest()},
            }
        )
        return turn


class GeneticMove(Move):
    def __init__(self, sensor: SensorData, genotype: Genotype):
        self.sensor = sensor
        self.genotype = genotype

    def calculate(self) -> float:
        self.sensor.distances_as_dict()
        _, move = self.genotype.evaluate(
            args={
                **self.sensor.distances_as_input(),
                "smell_direction": {"x": self.sensor.smell_nearest()},
            }
        )
        return move

class NNTurn(Turn):
    def __init__(self, sensor: SensorData, model, scaler, config):
        self.sensor = sensor
        self.model = model
        self.scaler = scaler
        self.config = config

    def calculate(self) -> float:
        """Predict robot movement using neural network"""
        try:
            output = self.model.predict(
                self.sensor.sensor_input(self.scaler, self.config),
                verbose=0
            )
            
            turn = self.scaler.scale(
                output[0][0],
                self.config.NORMALIZED_RANGE,
                self.config.TURN_RANGE
            ).item()

            return turn
            
        except Exception as e:
            Logger.error(f'Prediction: Failed to predict movement: {e}')
            return 0

class NNMove(Move):
    def __init__(self, sensor: SensorData, model, scaler, config):
        self.sensor = sensor
        self.model = model
        self.scaler = scaler
        self.config = config

    def calculate(self) -> float:
        """Predict robot movement using neural network"""
        try:
            output = self.model.predict(
                self.sensor.sensor_input(self.scaler, self.config),
                verbose=0
            )
            
            move = self.scaler.scale(
                output[0][1],
                self.config.NORMALIZED_RANGE,
                self.config.MOVE_RANGE
            ).item()
            
            return move
            
        except Exception as e:
            Logger.error(f'Prediction: Failed to predict movement: {e}')
            return 0


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
