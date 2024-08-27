from typing import Callable
from kivy.logger import Logger
from pysimbotlib.core import Robot
from strategies import FuzzyTurn, FuzzyMove
from sensors import SensorData, DirectionalDistances
from fuzzy_logic import (
    CombinedMembershipFunctions,
    MembershipFunction,
    FuzzyInterface,
    FuzzyVariable,
)


class FuzzyRobot(Robot):
    SAFE_DIST: float = 30.0
    CLOSE_DIST: float = 5.0
    HIT_DIST: float = 0.0
    MOVE_SPEED: float = 10.0
    TURN_SPEED: float = 20.0
    TURN_SHARP_SPEED: float = 50.0

    def __init__(self) -> None:
        super().__init__()
        sensor_data: SensorData = self.sensor()
        distance_msf = CombinedMembershipFunctions()
        distance_msf.add_membership(
            "close",
            MembershipFunction.create(function="gaussian", c=0.0, sigma=15.0),
        )
        distance_msf.add_membership(
            "medium",
            MembershipFunction.create(function="gaussian", c=20.0, sigma=20.0),
        )
        distance_msf.add_membership(
            "far",
            MembershipFunction.create(function="gaussian", c=100.0, sigma=50.0),
        )
        smell_direction_msf = CombinedMembershipFunctions()
        smell_direction_msf.add_membership(
            "front", MembershipFunction.create(function="gaussian", c=45, sigma=135)
        )
        smell_direction_msf.add_membership(
            "right", MembershipFunction.create(function="gaussian", c=115, sigma=135)
        )
        smell_direction_msf.add_membership(
            "back", MembershipFunction.create(function="gaussian", c=205, sigma=135)
        )
        smell_direction_msf.add_membership(
            "left", MembershipFunction.create(function="gaussian", c=295, sigma=135)
        )

        fuzzy_interface_turn = FuzzyInterface(
            input_mfs={
                "front": distance_msf,
                "front_right": distance_msf,
                "right": distance_msf,
                "back_right": distance_msf,
                "back": distance_msf,
                "back_left": distance_msf,
                "left": distance_msf,
                "front_left": distance_msf,
                "smell": smell_direction_msf,
            },
        )

        fuzzy_interface_move = FuzzyInterface(
            input_mfs={
                "front": distance_msf,
                "front_right": distance_msf,
                "right": distance_msf,
                "back_right": distance_msf,
                "back": distance_msf,
                "back_left": distance_msf,
                "left": distance_msf,
                "front_left": distance_msf,
            },
        )

        # fuzzy_interface_turn.add_rule(
        #     lambda values: self.smell() * values[sensor_data.smell_side()]["far"],
        #     1,
        # )

        fuzzy_interface_turn.add_rule(
            lambda values: self.smell()
            * values[sensor_data.smell_side()]["far"]
            * max(
                values[sensor_data.side_with_offset(sensor_data.smell_side(), -1)][
                    "medium"
                ],
                values[sensor_data.side_with_offset(sensor_data.smell_side(), -1)][
                    "far"
                ],
            )
            * max(
                values[sensor_data.side_with_offset(sensor_data.smell_side(), 1)][
                    "medium"
                ],
                values[sensor_data.side_with_offset(sensor_data.smell_side(), 1)][
                    "far"
                ],
            ),
            1,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: sensor_data.smell_food_on_left_sign()
            * values["front"]["close"]
            * values["front_left"]["far"]
            * values["front_right"]["far"],
            45,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: sensor_data.smell_food_on_left_sign()
            * values["front"]["close"]
            * values["front_left"]["close"]
            * values["front_right"]["close"],
            90,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: values["front"]["close"]
            * max(values["front_left"]["close"], values["front_left"]["medium"])
            * max(values["left"]["close"], values["left"]["medium"]),
            135,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: values["front"]["far"]
            * values["front_left"]["close"]
            * values["front_right"]["close"],
            180,
        )

        # --------------------------------------------------------------------------------------------

        fuzzy_interface_turn.add_rule(
            lambda values: values["front"]["close"]
            * max(values["front_right"]["close"], values["front_right"]["medium"]),
            -90,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: max(
                values["front_left"]["far"], values["front_left"]["medium"]
            )
            * max(values["front"]["far"], values["front"]["medium"])
            * values["front_right"]["close"]
            * max(values["right"]["far"], values["right"]["medium"]),
            -45,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: max(
                values["front_left"]["far"], values["front_left"]["medium"]
            )
            * values["front"]["far"]
            * values["front_right"]["close"]
            * values["right"]["close"],
            -45,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: values["front"]["close"]
            * max(values["right"]["close"], values["right"]["medium"])
            * max(values["front_right"]["close"], values["front_right"]["medium"]),
            -135,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: values["front"]["close"]
            * max(values["front_left"]["close"], values["front_left"]["medium"]),
            90,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: max(
                values["front_right"]["far"], values["front_right"]["medium"]
            )
            * max(values["front"]["far"], values["front"]["medium"])
            * values["front_left"]["close"]
            * max(values["left"]["far"], values["left"]["medium"]),
            45,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: max(
                values["front_right"]["far"], values["front_right"]["medium"]
            )
            * values["front"]["far"]
            * values["front_left"]["close"]
            * values["left"]["close"],
            45,
        )

        # -------------------------------- robot movement -----------------------------------------

        fuzzy_interface_move.add_rule(
            lambda values: values["front"]["far"],
            30,
        )

        fuzzy_interface_move.add_rule(
            lambda values: values["front_left"]["medium"],
            10,
        )

        fuzzy_interface_move.add_rule(
            lambda values: values["front_right"]["medium"],
            10,
        )

        fuzzy_interface_move.add_rule(
            lambda values: sum(
                [
                    values["front"]["close"],
                    values["front_left"]["close"],
                    values["front_right"]["close"],
                ]
            )
            / 3,
            -20,
        )

        self.turn_strategy = FuzzyTurn(
            sensor=sensor_data,
            interface=fuzzy_interface_turn,
        )

        self.move_strategy = FuzzyMove(
            sensor=sensor_data,
            interface=fuzzy_interface_move,
        )

    def update(self):
        Logger.debug("Updating robot state.")
        try:
            move_value: float = self.move_strategy.calculate()
            self.move(move_value)

            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            # Logger.debug(f"Computed move value: {move_value}, turn value: {turn_value}")

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
