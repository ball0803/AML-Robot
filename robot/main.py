from kivy.logger import Logger
from pysimbotlib.core import PySimbotApp
from simple_robot import SimpleRobot
from fuzzy_robot import FuzzyRobot
from config import REFRESH_INTERVAL

if __name__ == "__main__":
    Logger.info("Starting PySimbotApp with MyRobot.")
    try:
        app = PySimbotApp(
            robot_cls=FuzzyRobot,
            num_robots=1,
            interval=REFRESH_INTERVAL,
            enable_wasd_control=True,
        )
        app.run()
    except Exception as e:
        Logger.critical("Critical failure in PySimbotApp:", exc_info=True)
