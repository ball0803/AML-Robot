#!/usr/bin/python3
import math
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
# REFRESH_INTERVAL = 0.1 
# REFRESH_INTERVAL = 0.05
REFRESH_INTERVAL = 0.01

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

    def distance_membership_2(self, distance):
        close_distance = self.trapezoidal_membership(distance, 0, 0, 20, 40)
        far_distance = self.trapezoidal_membership(distance, 20, 50, 101, 101)

        return [round(close_distance, 3), round(far_distance, 3)]

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
        if food_angle < 0:
            food_angle += 360

        if food_angle < 22.5 or food_angle > 337.5:
            return 0
        else:
            return (food_angle-22.5)//45+1


        # return smell_side
    
    def turn_normalize(self, angle):
        if angle > 180:
            return angle-360
        else:
            return angle

    def angle_direction(self, angle1, angle2):
        # Calculate the difference between the angles
        diff = (angle2 - angle1) % 360

        # Check if the difference is less than or greater than 180 degrees
        if diff < 180 and diff > 0:
            return 0
        else:
            return 1

    def print_sensor(self, sensor):
        print(f"\t{sensor[7]} {sensor[0]} {sensor[1]}")
        print(f"\t{sensor[6]}   ^   {sensor[2]}")
        print(f"\t{sensor[5]} {sensor[4]} {sensor[3]}")
        
    def fuzzy_inference_2(self, sensors):
        front = self.distance_membership_2(sensors[0])
        front_right = self.distance_membership_2(sensors[1])
        right = self.distance_membership_2(sensors[2])
        back_right = self.distance_membership_2(sensors[3])
        back = self.distance_membership_2(sensors[4])
        back_left = self.distance_membership_2(sensors[5])
        left = self.distance_membership_2(sensors[6])
        front_left = self.distance_membership_2(sensors[7])

        fuzzy_sensor = [front, front_right, right, back_right, back, back_left, left, front_left]
        fuzzy_sensor_close = [front[0], front_right[0], right[0], back_right[0], back[0], back_left[0], left[0], front_left[0]]
        fuzzy_sensor_far = [front[1], front_right[1], right[1], back_right[1], back[1], back_left[1], left[1], front_left[1]]

        food_angle = int(self.food_direction_simple(self.smell()))
        # print(food_angle)
        # print(sensors)
        # print(fuzzy_sensor)
        # self.print_sensor(fuzzy_sensor)
        self.print_sensor(fuzzy_sensor_close)
        print("------------------")
        self.print_sensor(fuzzy_sensor_far)
        
        turn = 0
        turn += fuzzy_sensor[food_angle][1]*self.smell()
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[7][1]*-45)
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[1][1]*45)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[1][0]*fuzzy_sensor[0][1]*-22.5)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[7][0]*fuzzy_sensor[0][1]*22.5)
        # turn += self.turn_normalize(self.stuck*fuzzy_sensor[0][0]*fuzzy_sensor[1][1]*22.5)
        # turn += self.turn_normalize(self.stuck*fuzzy_sensor[0][0]*fuzzy_sensor[7][1]*-22.5)

        turn += self.turn_normalize(fuzzy_sensor[7][0]*fuzzy_sensor[0][0]*fuzzy_sensor[1][0]*fuzzy_sensor[2][1]*90)
        turn += self.turn_normalize(fuzzy_sensor[7][0]*fuzzy_sensor[0][0]*fuzzy_sensor[1][0]*fuzzy_sensor[6][1]*-90)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[7][0]*fuzzy_sensor[0][0]*fuzzy_sensor[1][0]*fuzzy_sensor[3][1]*135)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[7][0]*fuzzy_sensor[0][0]*fuzzy_sensor[1][0]*fuzzy_sensor[5][1]*-135)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[7][0]*fuzzy_sensor[0][1]*fuzzy_sensor[1][0]*fuzzy_sensor[3][1]*135)
        turn += self.turn_normalize(self.stuck*fuzzy_sensor[7][0]*fuzzy_sensor[0][1]*fuzzy_sensor[1][0]*fuzzy_sensor[5][1]*-135)

        turn += self.turn_normalize(fuzzy_sensor[1][0]*fuzzy_sensor[2][0]*fuzzy_sensor[6][0]*fuzzy_sensor[5][0]*fuzzy_sensor[7][1]*fuzzy_sensor[0][1]*-22.5)
        turn += self.turn_normalize(fuzzy_sensor[7][0]*fuzzy_sensor[6][0]*fuzzy_sensor[2][0]*fuzzy_sensor[3][0]*fuzzy_sensor[0][1]*fuzzy_sensor[1][1]*22.5)
        turn += self.turn_normalize(fuzzy_sensor[1][0]*fuzzy_sensor[5][0]*fuzzy_sensor[7][1]*-45)
        turn += self.turn_normalize(fuzzy_sensor[7][0]*fuzzy_sensor[3][0]*fuzzy_sensor[1][1]*45)
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[4][0]*fuzzy_sensor[6][1]*-90)
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[4][0]*fuzzy_sensor[2][1]*90)

        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[2][0]*fuzzy_sensor[4][0]*fuzzy_sensor[6][1]*-90)
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[2][0]*fuzzy_sensor[4][1]*fuzzy_sensor[6][0]*180)
        turn += self.turn_normalize(fuzzy_sensor[0][0]*fuzzy_sensor[2][1]*fuzzy_sensor[4][0]*fuzzy_sensor[6][0]*90)

        # turn += self.turn_normalize(random.randrange(-1,1)*15)
        turn += self.turn_normalize(self.stuck*random.randrange(-1,1)*15)
        print(turn)
        # for i in range(8):
        #     direction = self.angle_direction(i*45, self.smell())
        #     if direction:
        #         turn += self.turn_normalize(i*45-22.5*fuzzy_sensor[i][0])
        #     else:
        #         turn += self.turn_normalize(i*45+22.5*fuzzy_sensor[i][0])

        # for i in range(8):
        #     if i == food_angle:
        #         turn += self.turn_normalize(i*45-fuzzy_sensor[i][0]*fuzzy_sensor[(i+1)%8][0]*22.5)
        #     elif (i+1)%8 == food_angle:
        #         turn += self.turn_normalize(((i+1)%8)*45+fuzzy_sensor[i][0]*fuzzy_sensor[(i+1)%8][0]*22.5)

        # for i in range(4):
        #     direction = self.angle_direction(i*45, self.smell())
        #     if direction:
        #         turn += self.turn_normalize(i*45-90*fuzzy_sensor[i-2][1]*fuzzy_sensor[i][0]*fuzzy_sensor[i+4][0])
        #     else:
        #         turn += self.turn_normalize(i*45+90*fuzzy_sensor[i+2][1]*fuzzy_sensor[i][0]*fuzzy_sensor[i+4][0])
        # # print(self.smell(), food_angle)
        # for i in range(8):
        #     direction = self.angle_direction(i*45, self.smell())
        #     if direction:
        #         turn += self.turn_normalize(i*45-22.5*fuzzy_sensor[i][1]*fuzzy_sensor[(i+2)%8][0]*fuzzy_sensor[(i+4)%8][0]*fuzzy_sensor[(i+6)%8][0])
        #     else :
        #         turn += self.turn_normalize(i*45+22.5*fuzzy_sensor[i][1]*fuzzy_sensor[(i+2)%8][0]*fuzzy_sensor[(i+4)%8][0]*fuzzy_sensor[(i+6)%8][0])

            
        return turn
            

if __name__ == '__main__':
    app = PySimbotApp(robot_cls=MyRobot, num_robots=1, interval=REFRESH_INTERVAL, enable_wasd_control=True)
    app.run()