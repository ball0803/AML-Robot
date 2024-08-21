from typing import Callable
from collections import namedtuple
from kivy.logger import Logger

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
    def distances(self) -> DirectionalDistances:
        return DirectionalDistances(*self._distances())

    def distances_as_dict(self) -> dict:
        return self.distances._asdict()

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

    def smell_degree(self, offset: float = 0) -> float:
        smell_deg = self.smell()
        if smell_deg > 0:
            return 360 + smell_deg + offset
        else:
            return smell_deg + offset

    def smell_nearest_degree(self, offset: float = 0) -> float:
        smell_deg: float = self.smell_nearest()
        if smell_deg > 0:
            return 360 + smell_deg + offset
        else:
            return smell_deg + offset
