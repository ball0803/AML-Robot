import os
import platform
from kivy.config import Config
from kivy.logger import Logger, LOG_LEVELS

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

FRAME_RATE = 120
REFRESH_INTERVAL = 1 / FRAME_RATE
