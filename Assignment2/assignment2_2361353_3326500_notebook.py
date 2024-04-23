############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 1 ################

class FloodFillSolver():
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the coordinates that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    
    def __call__(self, road_grid, source, destination):
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both coordinates should be included in the path.
        To find the shortest route a version of a flood fill algorithm is used, see the explanation above.
        A route consists of a list of coordinates.

        Hint: The history is already given as a dictionary with as keys the coordinates in the state-space graph and
        as values the previous coordinate from which this coordinate was visited.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :param source: The coordinate where the path starts.
        :type source: tuple[int]
        :param destination: The coordinate where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of coordinates and the length of the route.
        :rtype: list[tuple[int]], float
        """
        self.queue = deque([source])
        self.history = {source: None}
        self.road_grid = road_grid
        self.destination = destination
        self.main_loop()

        return self.find_path()     

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one coordinate to the next has a length of 1.
        For example: The distance between coordinates (0,0) and (0,1) is 1 and 
                     The distance between coordinates (3,0) and (3,3) is 3. 

        The distance is the Manhattan distance of the path.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        # if self.destination not in self.history:
        #     return [], 0

        # path = []
        # current_node = self.destination
        # while current_node is not None:
        #     path.append(current_node)
        #     if current_node in self.history:  # Check if current_node is in self.history
        #         current_node = self.history[current_node]
        #     else:
        #         break  # Break the loop if current_node is not in self.history
        # path.reverse()
        # return path, len(path) - 1

        if self.destination not in self.history:
            return [], 0

        path = []
        current_node = self.destination
        while current_node is not None:
            path.append(current_node)
            current_node = self.history.get(current_node)  # Safe access to potentially non-existent keys
        path.reverse()
        return path, len(path) - 1
            
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        # while self.queue:
        #     current_node = self.queue.popleft()
        #     if current_node == self.destination:
        #         return
        #     for neighbor in self.next_step(current_node):
        #         if neighbor not in self.history:
        #             self.queue.append(neighbor)
        #             self.history[neighbor] = current_node

        # while self.queue:
        #     current_node = self.queue.popleft()
        #     if self.base_case(current_node):
        #         return
        #     for new_node in self.next_step(current_node):
        #         self.step(current_node, new_node)

        while self.queue:
            current_node = self.queue.popleft()
            if current_node == self.destination:
                return
            for new_node in self.next_step(current_node):
                self.step(current_node, new_node)
        
        

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the base case is found or not
        :rtype: bool
        """
        return node == self.destination
        
    def step(self, node, new_node):
        """
        One flood-fill step.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :param new_node: The next node/coordinate that can be visited from the current node/coordinate
        :type new_node: tuple[int]       
        """
        
        if new_node not in self.history and self.road_grid[new_node] != 0:  # Ensure new_node is a road and not already visited
            self.queue.append(new_node)
            self.history[new_node] = node

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        # directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        # possible_steps = []
        # x, y = node
        # for dx, dy in directions:
        #     nx, ny = x + dx, y + dy
        #     if 0 <= nx < self.road_grid.shape[0] and 0 <= ny < self.road_grid.shape[1]:
        #         if self.road_grid[nx, ny] == 1:  # assuming 1 is road and 0 is house
        #             possible_steps.append((nx, ny))
        # return possible_steps

        # return [(node[0] + 1, node[1]), (node[0] - 1, node[1]), (node[0], node[1] + 1), (node[0], node[1] - 1)]

        # possible_steps = [(node[0] + 1, node[1]), (node[0] - 1, node[1]), (node[0], node[1] + 1), (node[0], node[1] - 1)]
        # valid_steps = [(x, y) for x, y in possible_steps if 0 <= x < self.road_grid.shape[0] and 0 <= y < self.road_grid.shape[1]and self.road_grid[x, y] == 1]
        # return valid_steps

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = node
        valid_steps = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Ensure next step is within grid bounds and is a road (assuming 1 represents roads)
            if 0 <= nx < self.road_grid.shape[0] and 0 <= ny < self.road_grid.shape[1]:
                if self.road_grid[nx, ny] != 0:  # Only consider valid road parts
                    valid_steps.append((nx, ny))
        return valid_steps


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
