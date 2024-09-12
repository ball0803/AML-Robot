from typing import Callable, Tuple
from kivy.logger import Logger
from pysimbotlib.core import Robot
from sensors import SensorData
from strategies import Move, Turn
import random


class BaseRobot(Robot):
    SAFE_DIST: float = 30.0
    CLOSE_DIST: float = 5.0
    HIT_DIST: float = 0.0
    MOVE_SPEED: float = 10.0
    TURN_SPEED: float = 20.0
    TURN_SHARP_SPEED: float = 50.0

    def __init__(self) -> None:
        super().__init__()
        self.sensor_data = self.sensor()
        self.move_strategy: Move = self.create_move_strategy()
        self.turn_strategy: Turn = self.create_turn_strategy()

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
            stuck=super().stuck,
            safe_dist=self.SAFE_DIST,
            close_dist=self.CLOSE_DIST,
            hit_dist=self.HIT_DIST,
        )

    # Abstract methods to be implemented by subclasses
    def create_move_strategy(self) -> Move:
        raise NotImplementedError

    def create_turn_strategy(self) -> Turn:
        raise NotImplementedError

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

    def alter_movement(
        self,
        current_speed: float,
        current_turn: float,
        speed_variation: float = 5.0,
        turn_variation: float = 10.0,
    ) -> Tuple[float, float]:
        speed_change: float = random.uniform(-speed_variation, speed_variation)
        new_speed: float = max(current_speed + speed_change, 0)

        turn_change: float = random.uniform(-turn_variation, turn_variation)
        new_turn: float = current_turn + turn_change

        Logger.debug(
            f"Altered movement: new speed = {new_speed}, new turn = {new_turn}"
        )

        return new_speed, new_turn
