from typing import Callable
from kivy.logger import Logger
from pysimbotlib.core import Robot
from sensors import SensorData
from strategies import ReactiveMove, ReactiveTurn
import random


class SimpleRobot(Robot):
    SAFE_DIST: float = 30.0
    CLOSE_DIST: float = 5.0
    HIT_DIST: float = 0.0
    MOVE_SPEED: float = 10.0
    TURN_SPEED: float = 20.0
    TURN_SHARP_SPEED: float = 50.0

    def __init__(self) -> None:
        super().__init__()
        sensor_data: SensorData = self.sensor()
        self.move_strategy = ReactiveMove(
            sensor=sensor_data,
            calculate_smooth_move=self.create_speed_calculator(
                min_distance=self.SAFE_DIST,
                max_distance=100,
                min_speed=self.MOVE_SPEED,
                max_speed=self.MOVE_SPEED + 10,
            ),
        )
        self.turn_strategy = ReactiveTurn(
            sensor=sensor_data,
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

    def update(self):
        Logger.debug("Updating robot state.")
        try:
            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            move_value: float = self.move_strategy.calculate()
            self.move(move_value)

            Logger.debug(f"Computed move value: {move_value}, turn value: {turn_value}")

            Logger.info("Robot state updated successfully.")
        except Exception as e:
            Logger.error("Error during robot update:", exc_info=True)

    def sensor(self) -> SensorData:
        return SensorData(
            distances=super().distance,
            smell=super().smell,
            smell_nearest=super().smell_nearest,
            safe_dist=self.SAFE_DIST,
            close_dist=self.CLOSE_DIST,
            hit_dist=self.HIT_DIST,
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

    def alter_movement(
        self,
        current_speed: float,
        current_turn: float,
        speed_variation: float = 5.0,
        turn_variation: float = 10.0,
    ) -> tuple[float, float]:
        # Randomly alter the speed
        speed_change: float = random.uniform(-speed_variation, speed_variation)
        new_speed: float = max(
            current_speed + speed_change, 0
        )  # Ensure speed is non-negative

        # Randomly alter the turn angle
        turn_change: float = random.uniform(-turn_variation, turn_variation)
        new_turn: float = current_turn + turn_change

        Logger.debug(
            f"Altered movement: new speed = {new_speed}, new turn = {new_turn}"
        )

        return new_speed, new_turn
