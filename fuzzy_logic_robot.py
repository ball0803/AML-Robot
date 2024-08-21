#!/usr/bin/python3
import numpy as np
import os, platform
if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"

from pysimbotlib.core import PySimbotApp, Robot
from kivy.logger import Logger
from kivy.config import Config
import random

# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')

# update robot every 0.5 seconds (2 frames per sec)
# REFRESH_INTERVAL = 1/2
REFRESH_INTERVAL = 0.05

class MyRobot(Robot):
    
    def update(self):
        # ROTATE_SPEED = 10
        # self.turn(10)
        # self.move(10)
        # if abs(self.smell()) > 0:
        #     self.turn(self.smell())
        # Logger.info("Smell Angle: {0}".format(self.smell()))
        sensor = self.distance()
        # print(self.smell())
        self.just_pass = 0
        # output = self.fuzzy_inference(front_sensor)
        print("--------------------------------")
    #    print(self.fuzzy_inference(left_sensor))
        turn_value = self.fuzzy_inference_2(sensor)
        move_value = self.defuzzify_move(self.distance_membership(sensor[0]))
        self.turn(int(turn_value))
        self.move(move_value)
        if self.stuck:
            print("step bro i'm stuck")
        # output_values = np.linspace(0, 90, len(output))
        # centroid = np.sum(output_values *output) / np.sum(output)

    #    print(self.fuzzy_inference(right_sensor))
    #    print(self.fuzzy_inference(back_sensor))
        # Logger.info("Distance: {0}".format(self.distance()))

    def gaussian(self, x, mean, std_dev):
        exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
        return np.exp(exponent)

    def trapezoidal_membership(self, x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        elif a <= x <= b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        elif c <= x <= d:
            return (d - x) / (d - c)


    def distance_membership(self, distance):
        close_distance = self.gaussian(distance, 0, 20)
        medium_distance = self.gaussian(distance, 50, 20)
        far_distance = self.gaussian(distance, 100, 20)

        return [round(close_distance, 3), round(medium_distance, 3), round(far_distance, 3)]

    def defuzzify_turn(self, angle):
        little = self.defuzzify_centroid(self.trapezoidal_membership, 0, 0, 10, 15)
        medium = self.defuzzify_centroid(self.trapezoidal_membership, 15, 30, 60, 75)
        full = self.defuzzify_centroid(self.trapezoidal_membership, 70, 80, 90, 90)
        combined_output = self.calculate_combined_centroid([full, medium, little], angle)

        return combined_output
    
    def defuzzify_move(self, distance):
        little = self.defuzzify_centroid(self.trapezoidal_membership, 0, 0, 10, 15)
        medium = self.defuzzify_centroid(self.trapezoidal_membership, 10, 20, 20, 25)
        full = self.defuzzify_centroid(self.trapezoidal_membership, 20, 30, 60, 60)
        combined_output = self.calculate_combined_centroid([little, medium, full], distance)

        return combined_output

    def defuzzify_centroid(self, trapezoidal_function, a, b, c, d):
        numerator = 0
        denominator = 0
        step = 0.01  # Step size for integration (adjust as needed for precision)

        for x in range(int(a), int(d) + 1):
            membership = trapezoidal_function(x, a, b, c, d)
            numerator += x * membership
            denominator += membership

        if denominator == 0:
            return 0.0  # Avoid division by zero

        centroid = numerator / denominator
        return centroid

    def calculate_combined_centroid(self, centroids, percentages):
        centroids = np.asarray(centroids)
        percentages = np.asarray(percentages)
        if centroids.shape != percentages.shape or centroids.size == 0 or percentages.size == 0:
                raise ValueError("Input arrays must have the same shape and not be empty.")

        # Initialize variables for the weighted sum and total percentage
        weighted_sum = 0
        total_percentage = 0

        # Calculate the weighted sum
        for centroid, percentage in zip(centroids, percentages):
            weighted_sum += centroid * percentage
            total_percentage += percentage

        # Calculate the combined centroid
        if total_percentage == 0:
            return 0.0  # Avoid division by zero

        combined_centroid = weighted_sum / total_percentage
        return combined_centroid

    def fuzzy_inference(self, sensor):
        left = self.distance_membership(sensor[0])
        center = self.distance_membership(sensor[1])
        right = self.distance_membership(sensor[2])

        left_result = [max(left), left.index(max(left))]
        center_result = [max(center), center.index(max(center)), center.index(sorted(center)[-2])]
        right_result = [max(right), right.index(max(right))]
        # sensors = [left, center, right]

        fuzzy_variable = center
        count = 1
        if left_result[1] == center_result[1] or left_result[1] == center_result[2]:
            fuzzy_variable[center_result[1]] += left[center_result[1]]
            fuzzy_variable[int(center_result[2])] += left[int(center_result[2])]
            count += 1
        if right_result[1] == center_result[1] or right_result[1] == center_result[2]:
            fuzzy_variable[center_result[1]] += right[center_result[1]]
            fuzzy_variable[int(center_result[2])] += right[int(center_result[2])]
            count += 1
                
        
        fuzzy_variable = np.array(fuzzy_variable)    
        fuzzy_variable = fuzzy_variable/count

        return fuzzy_variable.tolist()

    def food_direction_simple(self, food_angle):
        smell_side = 0
        smell_half_side = 0
        if (food_angle > -45 and food_angle <= 0):
            smell_side = 0
            smell_half_side = 0
        elif(food_angle > 0 and food_angle <= 45):
            smell_side = 0
            smell_half_side = 1
        elif (food_angle > 45 and food_angle <= 90):
            smell_side = 1
            smell_half_side = 0
        elif (food_angle > 90 and food_angle <= 135):
            smell_side = 1
            smell_half_side = 1
        elif (food_angle > 135 and food_angle <= 180):
            smell_side = 2
            smell_half_side = 0
        elif (food_angle > -135 and food_angle <= -180):
            smell_side = 2
            smell_half_side = 1
        elif (food_angle > -90 and food_angle <= -45):
            smell_side = 3
            smell_half_side = 0
        elif (food_angle > -135 and food_angle <= -90):
            smell_side = 3
            smell_half_side = 1
        
        return smell_side, smell_half_side
    

    def fuzzy_inference_2(self, sensors):
        front = self.distance_membership(sensors[0])
        front_right = self.distance_membership(sensors[1])
        right = self.distance_membership(sensors[2])
        back_right = self.distance_membership(sensors[3])
        back = self.distance_membership(sensors[4])
        back_left = self.distance_membership(sensors[5])
        left = self.distance_membership(sensors[6])
        front_left = self.distance_membership(sensors[7])

        front_sensor = (sensors[7], sensors[0], sensors[1])
        right_sensor = sensors[1:4]
        back_sensor = sensors[3:6]
        left_sensor = sensors[5:]

        front_simplify = self.fuzzy_inference(front_sensor)
        right_simplify = self.fuzzy_inference(right_sensor)
        back_simplify = self.fuzzy_inference(back_sensor)
        left_simplify = self.fuzzy_inference(left_sensor)

        front_result_simplify = [max(front_simplify), front_simplify.index(max(front_simplify))] 
        right_result_simplify = [max(right_simplify), right_simplify.index(max(right_simplify))] 
        back_result_simplify = [max(back_simplify), back_simplify.index(max(back_simplify))]
        left_result_simplify = [max(left_simplify), left_simplify.index(max(left_simplify))]

        front_result = [max(front), front.index(max(front))] 
        front_right_result = [max(front_right), front_right.index(max(front_right))] 
        right_result = [max(right), right.index(max(right))] 
        back_right_result = [max(back_right), back_right.index(max(back_right))] 
        back_result = [max(back), back.index(max(back))]
        back_left_result = [max(back_left), back_left.index(max(back_left))]
        left_result = [max(left), left.index(max(left))]
        front_left_result = [max(front_left), front_left.index(max(front_left))]

        sensors_result = [front_result, right_result, back_result, left_result]
        all_sensors_result = [front_result, front_right_result, right_result, back_right_result, back_result, back_left_result, left_result, front_left_result]
        sensors_result_simplify = [front_result_simplify, right_result_simplify, back_result_simplify, left_result_simplify]
        print(all_sensors_result)
        print(sensors_result)
        side_count = 0
        far_count = 0
        side_count_all = 0
        side_appear = []
        side_appear_all = []

        food_angle = self.smell()
        smell_side, smell_half_side = self.food_direction_simple(food_angle)
        # print(food_angle)
        # print(smell_side)
        turn = 0

        for idx, sensor in enumerate(sensors_result):
            if sensor[1] == 0:
                side_count += 1
                side_appear.append(idx)
            if sensor[1] == 2:
                far_count += 1
                

        for idx, sensor in enumerate(all_sensors_result):
            if sensor[1] == 0:
                side_count_all += 1
                side_appear_all.append(idx)
        
        print(side_appear)
        print("food angle: ", food_angle)
        print("smell side: ", smell_side)

        if side_count == 4 and self.stuck:
            self.turn(5)

        elif side_count == 3:
            if 0 not in side_appear:
                if (smell_side == 0 and smell_half_side == 1) or (smell_side == 1) or (smell_side == 2 and smell_half_side == 0):
                    turn = 45-self.defuzzify_turn(front_right)
                elif (smell_side == 0 and smell_half_side == 0) or (smell_side == 3) or (smell_side == 2 and smell_half_side == 1):
                    turn = -45+self.defuzzify_turn(front_left)
            elif 1 not in side_appear:
                if (smell_side == 1 and smell_half_side == 1) or (smell_side == 2) or (smell_side == 3 and smell_half_side == 0):
                    turn = 90+45-self.defuzzify_turn(back_right)
                elif (smell_side == 1 and smell_half_side == 0) or (smell_side == 0) or (smell_side == 3 and smell_half_side == 1):
                    turn = 90+-45+self.defuzzify_turn(front_right)
            elif 3 not in side_appear:
                if (smell_side == 3 and smell_half_side == 1) or (smell_side == 0) or (smell_side == 1 and smell_half_side == 0):
                    turn = -90+45-self.defuzzify_turn(front_left)
                elif (smell_side == 3 and smell_half_side == 0) or (smell_side == 2) or (smell_side == 1 and smell_half_side == 1):
                    turn = -90+-45+self.defuzzify_turn(back_left)
            elif 2 not in side_appear:
                if (smell_side == 2 and smell_half_side == 1) or (smell_side == 3) or (smell_side == 0 and smell_half_side == 0):
                    turn = -180+45-self.defuzzify_turn(back_left)
                elif (smell_side == 2 and smell_half_side == 0) or (smell_side == 1) or (smell_side == 0 and smell_half_side == 1):
                    turn = 180+-45+self.defuzzify_turn(back_right)
        elif side_count == 2:
            if side_appear[0] == 0 and side_appear[1] == 1:
                    if front_left_result[1] == 0:
                        turn = -45-self.defuzzify_turn(front_right)
                    elif front_right_result[1] == 0:
                        turn = 45-self.defuzzify_turn(front_right)
                    else:
                        turn = -self.defuzzify_turn(front)


            elif side_appear[0] == 1 and side_appear[1] == 2 and self.stuck:
                    if front_right_result[1] == 0:
                        turn = 45 - self.defuzzify_turn(front_right)
                    elif back_right_result[1] == 0:
                        turn = 135 - self.defuzzify_turn(back_right)
                    else:
                        turn = 90 + self.defuzzify_turn(right)

                    
            elif side_appear[0] == 2 and side_appear[1] == 3 and self.stuck:
                    if back_right_result[1] == 0:
                        turn = 135 - self.defuzzify_turn(back_right)
                    elif back_left_result[1] == 0:
                        turn = -135 - self.defuzzify_turn(back_left)
                    else:
                        turn = 180 - self.defuzzify_turn(back)


            elif side_appear[0] == 0 and side_appear[1] == 3:
                    if  front_right_result[1] == 0:
                        turn = 45+self.defuzzify_turn(front_right)
                    elif front_left_result[1] == 0:
                        turn = -45+self.defuzzify_turn(front_left)
                    else:
                        turn = self.defuzzify_turn(front)


            elif self.stuck and side_appear[0] == 0 and side_appear[1] == 2:
                if smell_side == 0:
                    turn = self.defuzzify_turn(front) if smell_half_side else -self.defuzzify_turn(front)
                elif smell_side == 1:
                    turn = 90 - self.defuzzify_turn(back) if smell_half_side else 90-self.defuzzify_turn(front)
                elif smell_side == 2:
                    turn = 180 - self.defuzzify_turn(back) if smell_half_side else -180+self.defuzzify_turn(back)
                elif smell_side == 3:
                    turn = -90 - self.defuzzify_turn(back) if smell_half_side else -90+self.defuzzify_turn(front)
                else:
                    turn = self.smell()
            elif self.stuck and side_appear[0] == 1 and side_appear[1] == 3:
                if smell_side == 1:
                    turn = 90-self.defuzzify_turn(right) if smell_half_side else 90+self.defuzzify_turn(right)
                elif smell_side == 3:
                    turn = -90-self.defuzzify_turn(left) if smell_half_side else -90+self.defuzzify_turn(left)
                elif smell_side == 2:
                    if smell_half_side == 1:
                        turn = 180-self.defuzzify_turn(right)
                    elif smell_half_side == 0:
                        turn = -180+self.defuzzify_turn(left)
                elif smell_side == 0:
                    if smell_half_side == 1:
                        turn = 90-self.defuzzify_turn(right)
                    elif smell_half_side == 0:
                        turn = -90+self.defuzzify_turn(left)
                else:
                    turn = self.smell()
                self.just_pass = 1
        elif side_count == 1 or self.stuck:

            if front_right_result[1] == 0:
                turn = 45-self.defuzzify_turn(front_right)
            elif front_left_result[1] == 0:
                turn = -45+self.defuzzify_turn(front_left)
            elif right_result[1] == 0 and  self.stuck:
                turn = -self.defuzzify_turn(right)
            elif left_result[1] == 0 and self.stuck:
                turn = self.defuzzify_turn(left)
            elif front_result[1] == 0:
                if smell_half_side == 0:
                    if right_result_simplify[1] == 0:
                        turn = -self.defuzzify_turn(front)
                    else:
                        turn = self.defuzzify_turn(front)
                elif smell_half_side == 1:
                    if left_result_simplify[1] == 0:
                        turn = self.defuzzify_turn(front)
                    else:
                        turn = -self.defuzzify_turn(front)
                        

        elif side_count == 0 and self.just_pass == 0:
            if not sensors_result_simplify[smell_side][1] == 1:
                turn = self.smell()        
        self.just_pass = 0


            
        print(turn)
        return turn
            

if __name__ == '__main__':
    app = PySimbotApp(robot_cls=MyRobot, num_robots=1, interval=REFRESH_INTERVAL, enable_wasd_control=True)
    app.run()