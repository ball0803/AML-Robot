from kivy.logger import Logger
from pysimbotlib.core import Robot, PySimbotApp, Simbot
from config import REFRESH_INTERVAL
import pandas as pd
import os

file_path = 'move_history.csv'

def load_or_create_history():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        Logger.info("Existing history loaded")
    else:
        df = pd.DataFrame(columns=[
            'ir0',
            'ir1',
            'ir2',
            'ir3',
            'ir4',
            'ir5',
            'ir6',
            'ir7',
            'angle',
            'turn',
            'move',
        ])
        df.to_csv(file_path, index=False)
        Logger.info("New history file created")
    
    return df

def before_simulation(simbot: Simbot):
    simbot.move_history = load_or_create_history()

def after_simulation(simbot: Simbot):
    simbot.move_history.to_csv(file_path, index=False)

class ManualRobot(Robot):
    def __init__(self) -> None:
        super().__init__()
        self.prev_direction = self._direction
        self.prev_position = list(self.pos)


    def is_change(self) -> bool:
        return self.prev_position != self.pos or self.prev_direction != self._direction

    def update_pos_dir(self) -> None:
        self.prev_direction = self._direction
        self.prev_position = list(self.pos)

    def update(self):
        try:
            if self.is_change() and len(self._sm.history) != 0:
                self.update_pos_dir()
                new_row = pd.Series(list(self.distance()) + [self.smell_nearest()] + self._sm.history[-1][-2:],
                                index=self._sm.move_history.columns)
                self._sm.move_history = self._sm.move_history._append(new_row, ignore_index=True)
        except Exception as e:
            Logger.error("Error during robot update:", exc_info=True)


if __name__ == "__main__":
    Logger.info("Starting PySimbotApp with Manual Robot.")
    try:
        app = PySimbotApp(
            robot_cls=ManualRobot,
            num_robots=1,
            interval=REFRESH_INTERVAL,
            enable_wasd_control=True,
            save_wasd_history=True,
            customfn_before_simulation=before_simulation,
            customfn_after_simulation=after_simulation,
        )
        app.run()
    except Exception as e:
        Logger.critical("Critical failure in PySimbotApp:", exc_info=True)

