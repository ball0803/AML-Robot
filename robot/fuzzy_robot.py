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
            MembershipFunction.create(function="gaussian", c=0.0, sigma=10.0),
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
        turn_msf = CombinedMembershipFunctions()
        turn_msf.add_membership(
            "little",
            MembershipFunction.create(
                function="trapezoidal", a=0.0, b=0.0, c=10.0, d=15.0
            ),
        )
        turn_msf.add_membership(
            "medium",
            MembershipFunction.create(
                function="trapezoidal", a=15.0, b=30.0, c=60.0, d=75.0
            ),
        )
        turn_msf.add_membership(
            "full",
            MembershipFunction.create(
                function="trapezoidal", a=70.0, b=80.0, c=130.0, d=130.0
            ),
        )

        move_msf = CombinedMembershipFunctions()
        move_msf.add_membership(
            "little",
            MembershipFunction.create(function="gaussian", c=0.0, sigma=10.0),
        )
        move_msf.add_membership(
            "medium",
            MembershipFunction.create(function="gaussian", c=20.0, sigma=20.0),
        )
        move_msf.add_membership(
            "full",
            MembershipFunction.create(function="gaussian", c=40.0, sigma=20.0),
        )

        fuzzy_interface_left = FuzzyInterface(
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
            output_mfs=turn_msf,
        )

        fuzzy_interface_right = FuzzyInterface(
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
            output_mfs=turn_msf,
        )

        fuzzy_interface_move_front = FuzzyInterface(
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
            output_mfs=move_msf,
        )

        fuzzy_interface_move_back = FuzzyInterface(
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
            output_mfs=move_msf,
        )

        # Adding fuzzy rules considering smell direction
        fuzzy_interface_left.add_rule(
            lambda values: values["front"]["close"] * values["left"]["far"],
            "full",
        )

        fuzzy_interface_left.add_rule(
            lambda values: 0,
            "medium",
        )

        fuzzy_interface_left.add_rule(
            lambda values: 0,
            "little",
        )

        fuzzy_interface_right.add_rule(
            lambda values: values["front"]["close"] * values["right"]["far"],
            "full",
        )

        # fuzzy_interface_left.add_rule(
        #     lambda values: (
        #         values["front"]["medium"]
        #         * values["right"]["medium"]
        #         * values["smell"]["front"]
        #     ),
        #     "medium",
        # )

        # fuzzy_interface_right.add_rule(
        #     lambda values: (
        #         values["front"]["medium"]
        #         * values["left"]["medium"]
        #         * values["smell"]["front"]
        #     ),
        #     "medium",
        # )

        # fuzzy_interface_left.add_rule(
        #     lambda values: (
        #         values["front"]["medium"]
        #         * values["left"]["medium"]
        #         * values["smell"]["left"]
        #     ),
        #     "little",
        # )

        # fuzzy_interface_right.add_rule(
        #     lambda values: (
        #         values["front"]["medium"]
        #         * values["right"]["medium"]
        #         * values["smell"]["right"]
        #     ),
        #     "little",
        # )

        fuzzy_interface_move_front.add_rule(
            lambda values: values["front"]["far"], "full"
        )

        fuzzy_interface_move_front.add_rule(
            lambda values: values["front"]["medium"], "little"
        )

        fuzzy_interface_move_back.add_rule(
            lambda values: 0.5 * values["front"]["close"],
            "little",
        )

        self.turn_strategy = FuzzyTurn(
            sensor=sensor_data,
            interface_left=fuzzy_interface_left,
            interface_right=fuzzy_interface_right,
        )

        self.move_strategy = FuzzyMove(
            sensor=sensor_data,
            interface_front=fuzzy_interface_move_front,
            interface_back=fuzzy_interface_move_back,
        )

    def update(self):
        Logger.debug("Updating robot state.")
        try:
            turn_value: float = self.turn_strategy.calculate()
            self.turn(turn_value)

            # move_value: float = self.move_strategy.calculate()
            # self.move(move_value)

            # Logger.debug(f"Computed move value: {move_value}, turn value: {turn_value}")

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
