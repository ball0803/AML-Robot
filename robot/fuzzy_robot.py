from typing import Tuple
from strategies import FuzzyTurn, FuzzyMove
from fuzzy_logic import (
    CombinedMembershipFunctions,
    MembershipFunction,
    FuzzyInterface,
)
from base_robot import BaseRobot


class FuzzyRobot(BaseRobot):
    def __init__(self) -> None:
        self.fuzzy_interface_move, self.fuzzy_interface_turn = self.setup_interface()
        self.setup_move_rule(fuzzy_interface_move=self.fuzzy_interface_move)
        self.setup_turn_rule(fuzzy_interface_turn=self.fuzzy_interface_turn)
        super().__init__()

    def create_turn_strategy(self) -> FuzzyTurn:
        return FuzzyTurn(
            sensor=self.sensor_data,
            interface=self.fuzzy_interface_turn,
        )

    def create_move_strategy(self) -> FuzzyMove:
        return FuzzyMove(
            sensor=self.sensor_data,
            interface=self.fuzzy_interface_move,
        )

    def setup_turn_rule(self, fuzzy_interface_turn: FuzzyInterface) -> None:
        fuzzy_interface_turn.add_rule(
            lambda values: self.smell()
            * values[self.sensor_data.smell_side()]["far"]
            * max(
                values[
                    self.sensor_data.side_with_offset(self.sensor_data.smell_side(), -1)
                ]["medium"],
                values[
                    self.sensor_data.side_with_offset(self.sensor_data.smell_side(), -1)
                ]["far"],
            )
            * max(
                values[
                    self.sensor_data.side_with_offset(self.sensor_data.smell_side(), 1)
                ]["medium"],
                values[
                    self.sensor_data.side_with_offset(self.sensor_data.smell_side(), 1)
                ]["far"],
            ),
            1,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: self.sensor_data.smell_food_on_left_sign()
            * values["front"]["close"]
            * values["front_left"]["far"]
            * values["front_right"]["far"],
            45,
        )

        fuzzy_interface_turn.add_rule(
            lambda values: self.sensor_data.smell_food_on_left_sign()
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

    def setup_move_rule(self, fuzzy_interface_move: FuzzyInterface) -> None:
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

    def setup_interface(self) -> Tuple[FuzzyInterface, FuzzyInterface]:
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

        return fuzzy_interface_move, fuzzy_interface_turn
