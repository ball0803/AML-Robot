import os
import platform
from kivy.config import Config
from kivy.logger import Logger, LOG_LEVELS
from pysimbotlib.core import Robot
from pysimbotlib.core import PySimbotApp
from collections import namedtuple
from typing import Callable
from abc import ABC, abstractmethod

# Set environment variable for video playback based on the platform
if platform.system() in ["Linux", "Darwin"]:
    os.environ["KIVY_VIDEO"] = "ffpyplayer"

# Configure Kivy Logger
Config.set("kivy", "log_level", "debug")
Config.set("kivy", "log_enable", 1)
Config.set("kivy", "log_dir", "logs")
Config.set("kivy", "log_name", "kivy_%y-%m-%d_%_.txt")
Config.set("kivy", "log_maxfiles", 10)

Logger.setLevel(LOG_LEVELS["debug"])

FRAME_RATE = 30
REFRESH_INTERVAL = 1 / FRAME_RATE


# Define the namedtuple for distances once
DirectionalDistances = namedtuple(
    "DirectionalDistances",
    [
        "front",
        "front_right",
        "right",
        "back_right",
        "back",
        "back_left",
        "left",
        "front_left",
    ],
)


DirectionalDistances = namedtuple(
    "DirectionalDistances",
    [
        "front",
        "front_right",
        "right",
        "back_right",
        "back",
        "back_left",
        "left",
        "front_left",
    ],
)


class SensorData:
    def __init__(
        self,
        distances: Callable[[], list[float]],
        smell: Callable[[], float],
        smell_nearest: Callable[[], float],
        safe_dist: float = 30,
        close_dist: float = 5,
        hit_dist: float = 0,
    ) -> None:
        self._distances = distances
        self.smell = smell
        self.smell_nearest = smell_nearest
        self.SAFE_DIST = safe_dist
        self.CLOSE_DIST = close_dist
        self.HIT_DIST = hit_dist

    @property
    def distances(self):
        return DirectionalDistances(*self._distances())

    def is_front_safe(self) -> bool:
        return (
            self.distances.front >= self.SAFE_DIST
            and self.distances.front_left >= self.SAFE_DIST
            and self.distances.front_right >= self.SAFE_DIST
        )

    def is_front_safe_either_close_by(self) -> bool:
        return self.distances.front >= self.SAFE_DIST and (
            self.distances.front_left >= self.CLOSE_DIST
            or self.distances.front_right >= self.CLOSE_DIST
        )

    def is_front_safe_both_close_by(self) -> bool:
        return (
            self.distances.front >= self.SAFE_DIST
            and self.distances.front_left >= self.CLOSE_DIST
            and self.distances.front_right >= self.CLOSE_DIST
        )

    def is_about_to_hit(self) -> bool:
        return self.distances.front > self.HIT_DIST

    def smell_food_on_left(self) -> bool:
        return self.smell_nearest() < 0


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


if __name__ == "__main__":
    Logger.info("Starting PySimbotApp with MyRobot.")
    try:
        app = PySimbotApp(
            robot_cls=SimpleRobot,
            num_robots=1,
            interval=REFRESH_INTERVAL,
            enable_wasd_control=True,
        )
        app.run()
    except Exception as e:
        Logger.critical("Critical failure in PySimbotApp:", exc_info=True)
