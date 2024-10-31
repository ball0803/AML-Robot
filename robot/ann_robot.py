#!/usr/bin/python3

import os
import random
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from pysimbotlib.core import PySimbotApp
from base_robot import BaseRobot
from kivy.config import Config
from kivy.logger import Logger
from sensors import SensorData
from config import REFRESH_INTERVAL
from strategies import Move, Turn, NNTurn, NNMove

# Configuration
tf.get_logger().setLevel('ERROR')
Config.set('graphics', 'maxfps', 10)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class SensorConfig:
    """Configuration for sensor scaling"""
    DISTANCE_RANGE: Tuple[float, float] = (0, 100)
    ANGLE_RANGE: Tuple[float, float] = (-180, 180)
    TURN_RANGE: Tuple[float, float] = (-90, 90)
    MOVE_RANGE: Tuple[float, float] = (-10, 10)
    NORMALIZED_RANGE: Tuple[float, float] = (0, 1)
    INPUT_FEATURES: int = 9

class DataScaler:
    """Handle data scaling operations"""
    @staticmethod
    def scale(data: np.ndarray, 
              from_interval: Tuple[float, float], 
              to_interval: Tuple[float, float]=(0, 1)) -> np.ndarray:
        """Scale data from one interval to another"""
        from_min, from_max = from_interval
        to_min, to_max = to_interval
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        denominator = (from_max - from_min + epsilon)
        
        return to_min + (data - from_min) * (to_max - to_min) / denominator

class NNRobot(BaseRobot):
    """Neural Network controlled robot"""
    def __init__(self, **kwargs):
        """Initialize robot with neural network model"""
        self.config = SensorConfig()
        self.scaler = DataScaler()
        
        try:
            self.model = load_model('ann_model.keras')  # Updated to .keras extension
            Logger.info('Model: Successfully loaded neural network model')
        except Exception as e:
            Logger.error(f'Model: Failed to load model: {e}')
            raise

        super(NNRobot, self).__init__(**kwargs)

    def create_turn_strategy(self) -> NNTurn:
        return NNTurn(
            sensor=self.sensor_data,
            model=self.model,
            scaler=self.scaler,
            config=self.config,
        )

    def create_move_strategy(self) -> NNMove:
        return NNMove(
            sensor=self.sensor_data,
            model=self.model,
            scaler=self.scaler,
            config=self.config,
        )

    def update(self):
        Logger.debug("Updating robot state.")
        try:
            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            move_value: float = self.move_strategy.calculate()
            self.move(move_value)

            Logger.debug(f"Computed move value: {move_value}, turn value: {turn_value}")

            # Handle stuck condition if necessary
            if self.stuck:
                self.handle_stuck_condition()

            Logger.info("Robot state updated successfully.")
        except Exception as e:
            Logger.error("Error during robot update:", exc_info=True)

    def handle_stuck_condition(self):
        """Handle robot stuck condition"""
        turn_angle = random.randint(-10, 10)
        self.turn(turn_angle)
        self.move(-5)
        Logger.debug(f'Robot: Stuck condition handled - Turn: {turn_angle}')

def main():
    """Main application entry point"""
    Logger.info("Starting PySimbotApp with MyRobot.")
    try:
        app = PySimbotApp(
            robot_cls=NNRobot,
            num_robots=1,
            interval=REFRESH_INTERVAL,
            enable_wasd_control=True,
        )
        app.run()
    except Exception as e:
        Logger.critical("Critical failure in PySimbotApp:", exc_info=True)

if __name__ == '__main__':
    main()
