#!/usr/bin/python3

import os, platform
import random
if platform.system() == "Linux" or platform.system() == "Darwin":
    os.environ["KIVY_VIDEO"] = "ffpyplayer"
    
from pysimbotlib.core import PySimbotApp, Robot
from kivy.logger import Logger
from kivy.config import Config
# Force the program to show user's log only for "info" level or more. The info log will be disabled.
Config.set('kivy', 'log_level', 'info')
from collections import deque

REFRESH_INTERVAL = 1

class MyRobot(Robot):

    def __init__(self, **kwargs):
        super(MyRobot, self).__init__(**kwargs)
        self.initial_map = (
            ((0, 0), (0, 0, 1, 1)),
            ((1, 0), (0, 1, 1, 0)),
            ((2, 0), (0, 0, 1, 1)),
            ((3, 0), (1, 0, 1, 0)),
            ((4, 0), (0, 1, 1, 0)),
            ((0, 1), (0, 0, 0, 1)),
            ((1, 1), (1, 0, 0, 0)),
            ((2, 1), (0, 1, 0, 0)),
            ((3, 1), (0, 1, 1, 1)),
            ((4, 1), (0, 1, 0, 1)),
            ((0, 2), (1, 0, 0, 1)),
            ((1, 2), (1, 0, 1, 0)),
            ((2, 2), (1, 0, 0, 0)),
            ((3, 2), (1, 0, 0, 0)),
            ((4, 2), (1, 1, 0, 0))
        )
        self.initial_state = self.initial_state_set(self.initial_map)
        self.end = (3, 1)
        self.visited = set()
        self.state = self.initial_state
        self.directions = {'u': (0, 1), 'r': (1, 0), 'd': (0, -1), 'l': (-1, 0)}
        self.solved_maze = None
    
    def update(self):
        if not len(self.state) == 1 or not list(self.state)[0][0] == self.end:
            self.state = self.sees(self.state)
            action = self.getAction(self.state)
            self.action(action)
            self.state = self.result(action, self.state)
            print(self.state)
        else:
            print("Done")

    def whatDoIsee(self):
        nodeSize = 50
        distances = self.distance()
        walls = []
        for i in [0, 2, 4, 6]:
            if distances[i] < nodeSize / 2:
                walls.append(True)
            else:
                walls.append(False)
        return walls
    
    def GTNN(self, num_node):
        nodeSize = 50
        nodes_traveled = 0
        while nodes_traveled < num_node:
            walls = self.whatDoIsee()
            if walls[0]:
                break
            self.move(nodeSize)
            nodes_traveled += 1

        return nodes_traveled
    
    def initial_state_set(self, intial_map):
        state_set = []
        orientation_set = ['u', 'r', 'd', 'l']
        for node in intial_map:
            for orientation in orientation_set:
                state_set.append((node[0], orientation))
        
        return state_set
    
    def simulate_perception(self, state, initial_map):
        orientation_set = {'u': 0, 'r': -1, 'd': -2, 'l': -3}
        for node in initial_map:
            if node[0] == state[0]:
                de = deque(node[1], maxlen=len(node[1]))
                de.rotate(orientation_set[state[1]])
                return list(de)
        
    def sees(self, current_state_set):
        actual_percept = self.whatDoIsee()

        # Filter the state-set based on the current percept
        filtered_states = []

        for state in current_state_set:
            expected_percept = self.simulate_perception(state, self.initial_map)
            
            # Check if the simulated percept matches the actual percept
            if expected_percept == actual_percept:
                filtered_states.append(state)

        return filtered_states

    def getAction(self, state_set):
        perception = self.whatDoIsee()
        direction = ['u', 'r', 'd', 'l']
        if len(state_set) == 1:
            current_state = list(state_set)[0]
            if not self.solved_maze:
                self.solved_maze = self.solve_maze(current_state)
                return self.solved_maze.pop(0)
            else:
                return self.solved_maze.pop(0)
        else:
            if not perception[0]:
                return 'move'
            if not perception[1]:
                return 'turn right'
            if not perception[3]:
                return 'turn left'

    def find_node(self, nodes, target):
        for node in nodes:
            if node[0] == target:
                return node[1]
        return None

    def is_valid_move(self, position, direction):
        x, y = position
        dx, dy = self.directions[direction]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x <= len(self.initial_map) and 0 <= new_y <= len(self.initial_map[0]):
            cell = self.find_node(self.initial_map, position)
            if cell[['u', 'r', 'd', 'l'].index(direction)] == 0 and ((new_x, new_y), direction) not in self.visited:
                return True

        return False

    def bfs(self, start):
        queue = deque([(start[0], start[1], [])])  # Initialize the queue with the start position and direction

        while queue:
            position, direction, path = queue.popleft()  # Dequeue a node from the queue

            if position == self.end:
                return path

            if (position, direction) in self.visited:
                continue

            self.visited.add((position, direction))

            # Try moving forward
            new_position = (position[0], position[1]+1) if direction == 'u' else \
                            (position[0]+1, position[1]) if direction == 'r' else \
                            (position[0], position[1]-1) if direction == 'd' else \
                            (position[0]-1, position[1])  # direction == 'l'
            
            if self.is_valid_move(position, direction):
                queue.append((new_position, direction, path + ['move']))

            # Try turning left
            new_direction = {
                'u': 'l',
                'r': 'u',
                'd': 'r',
                'l': 'd'
            }[direction]
            queue.append((position, new_direction, path + ['turn left']))

            # Try turning right
            new_direction = {
                'u': 'r',
                'r': 'd',
                'd': 'l',
                'l': 'u'
            }[direction]
            queue.append((position, new_direction, path + ['turn right']))

        return None

    def solve_maze(self, start):
        path = self.bfs(start)
        if path:
            return path
        else:
            return "No path found."

    def action(self, action):
        if action == 'move':
            self.move(50)
        elif action == 'turn right':
            self.turn(90)
        elif action == 'turn left':
            self.turn(-90)
        
    def result(self, action, state_set):
        direction = ['u', 'r', 'd', 'l']
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        if action == 'move':
            return [((state[0][0] + directions[direction.index(state[1])][0], state[0][1] + directions[direction.index(state[1])][1]), state[1]) for state in state_set]
        if action == 'turn right':
            return [((state[0][0], state[0][1]), direction[(direction.index(state[1]) + 1) % 4]) for state in state_set]
        if action == 'turn left':
            return [((state[0][0], state[0][1]), direction[(direction.index(state[1]) - 1) % 4]) for state in state_set]

if __name__ == '__main__':
    # possible map value: ["default", "no_wall"]
    app = PySimbotApp(robot_cls=MyRobot, map="test", interval=REFRESH_INTERVAL, enable_wasd_control=True)
    app.run()