from collections import deque

class MazeSolver:
    def __init__(self, initial_map, start, end):
        self.initial_map = initial_map
        self.start = start
        self.end = end
        self.visited = set()
        self.directions = {'u': (0, 1), 'r': (1, 0), 'd': (0, -1), 'l': (-1, 0)}

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
                queue.append((new_position, direction, path + ['M']))

            # Try turning left
            new_direction = {
                'u': 'l',
                'r': 'u',
                'd': 'r',
                'l': 'd'
            }[direction]
            queue.append((position, new_direction, path + ['L']))

            # Try turning right
            new_direction = {
                'u': 'r',
                'r': 'd',
                'd': 'l',
                'l': 'u'
            }[direction]
            queue.append((position, new_direction, path + ['R']))

        return None

    def dfs(self, position, direction, path):
        # print(self.visited)
        print("position :", position, "end :", self.end, "path :", path)
        if position == self.end:
            return path

        if (position, direction) in self.visited:
            return None

        self.visited.add((position, direction))

        # Try moving forward
        new_position = (position[0], position[1]+1) if direction == 'u' else \
                        (position[0]+1, position[1]) if direction == 'r' else \
                        (position[0], position[1]-1) if direction == 'd' else \
                        (position[0]-1, position[1])  # direction == 'l'
        
        if self.is_valid_move(position, direction):
            move_forward = self.dfs(new_position, direction, path + ['M'])
            if move_forward:
                return move_forward

        # Try turning left
        new_direction = {
            'u': 'l',
            'r': 'u',
            'd': 'r',
            'l': 'd'
        }[direction]
        turn_left = self.dfs(position, new_direction, path + ['L'])
        if turn_left:
            return turn_left

        # Try turning right
        new_direction = {
            'u': 'r',
            'r': 'd',
            'd': 'l',
            'l': 'u'
        }[direction]
        turn_right = self.dfs(position, new_direction, path + ['R'])
        if turn_right:
            return turn_right

        return None

    def solve_maze(self):
        path = self.bfs(self.start)
        # path = self.dfs(self.start[0], self.start[1], [])

        if path:
            return path
        else:
            return "No path found."

# Example usage
maze_solver = MazeSolver((
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
            ((4, 2), (1, 1, 0, 0))),
            ((2, 2), 'r'),
            (3, 1)
        )

result = maze_solver.solve_maze()
print(result)