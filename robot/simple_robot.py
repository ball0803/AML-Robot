from typing import Callable
from kivy.logger import Logger
from pysimbotlib.core import Robot
from sensors import SensorData
from strategies import ReactiveMove, ReactiveTurn
from base_robot import BaseRobot


class SimpleRobot(BaseRobot):
    def __init__(self) -> None:
        super().__init__()

    def create_move_strategy(self) -> ReactiveMove:
        return ReactiveMove(
            sensor=self.sensor_data,
            calculate_smooth_move=self.create_speed_calculator(
                min_distance=self.SAFE_DIST,
                max_distance=100,
                min_speed=self.MOVE_SPEED,
                max_speed=self.MOVE_SPEED + 10,
            ),
        )

    def create_turn_strategy(self) -> ReactiveTurn:
        return ReactiveTurn(
            sensor=self.sensor_data,
            calculate_smooth_turn=self.create_speed_calculator(
                min_distance=0,
                max_distance=100,
                min_speed=self.TURN_SPEED,
                max_speed=self.TURN_SPEED + 20,
            ),
            calculate_smooth_sharp_turn=self.create_speed_calculator(
                min_distance=0,
                max_distance=100,
                min_speed=self.TURN_SHARP_SPEED,
                max_speed=self.TURN_SHARP_SPEED + 10,
            ),
        )

    @staticmethod
    def create_speed_calculator(
        min_distance: float = 0,
        max_distance: float = 100,
        min_speed: float = 0,
        max_speed: float = 20,
    ) -> Callable[[float], float]:
        def calculate_speed(distance: float) -> float:
            distance = max(min(distance, max_distance), min_distance)
            normalized_distance = (distance - min_distance) / (
                max_distance - min_distance
            )
            speed = min_speed + (max_speed - min_speed) * (1 - normalized_distance)
            Logger.debug(f"Calculated speed: {speed} for distance: {distance}")
            return speed

        return calculate_speed
