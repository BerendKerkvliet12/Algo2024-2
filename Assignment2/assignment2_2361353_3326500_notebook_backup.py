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

############ CODE BLOCK 10 ################

class GraphBluePrint():
    """
    You can ignore this class, it is just needed due to technicalities.
    """
    def find_nodes(self): pass
    def find_edges(self): pass
    
class Graph(GraphBluePrint):   
    """
    Attributes:
        :param adjacency_list: The adjacency list with the road distances and speed limit.
        :type adjacency_list: dict[tuple[int]: set[edge]], where an edge is a fictional datatype 
                              which is a tuple containing the datatypes tuple[int], int, float
        :param map: The map of the graph.
        :type map: Map
    """
    def __init__(self, map_, start=(0, 0)):
        """
        This function transforms any (city or lower) map into a graph representation.

        :param map_: The map that needs to be transformed.
        :type map_: Map
        :param start: The start node from which we will find all other nodes.
        :type start: tuple[int]
        """
        self.map = map_
        self.start = start
        self.adjacency_list = {}
        self.find_nodes()
        self.find_edges()
        
        
    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        So far, we called this method `step`. However, this class is more than just the search algorithm,
        therefore, we gave it a bit more descriptive name.

        Note, that we only want to find the nodes, so history does not need to contain a partial path (previous node).
        In `find_edges` (the next cell), we will add edges for each node.
        """
        queue = deque([self.start])
        visited = set()
        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)
            actions = self.neighbour_coordinates(current_node)
            self.adjacency_list_add_node(current_node, actions)
            for action in actions:
                queue.append(action)
        
        
                    
    def adjacency_list_add_node(self, coordinate, actions):
        """
        This is a helper function for the breadth-first search algorithm to add a coordinate to the `adjacency_list` and
        to determine if a coordinate needs to be added to the `adjacency_list`.

        Reminder: A coordinate should only be added to the adjacency list if it is a corner, a crossing, or a dead end.
                  Adding the coordinate to the adjacency_list is equivalent to saying that it is a node in the graph.

        :param coordinate: The coordinate that might need to be added to the adjacency_list.
        :type coordinate: tuple[int]
        :param actions: The actions possible from this coordinate, an action is defined as an action in the coordinate state-space.
        :type actions: list[tuple[int]]
        """
        if len(actions) > 2 or len(actions) == 1:
            self.adjacency_list[coordinate] = set()
                           
    def neighbour_coordinates(self, coordinate):
        """
        This method returns the next possible actions and is part of the breadth-first search algorithm.
        Similar to `find_nodes`, we often call this method `next_step`.
        
        :param coordinate: The current coordinate
        :type coordinate: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = coordinate
        valid_steps = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1]:
                if self.map[nx, ny] != 0:
                    valid_steps.append((nx, ny))
        return valid_steps
    
    def __repr__(self):
        """
        This returns a representation of a graph.

        :return: A string representing the graph object.
        :rtype: str
        """
        return f"Graph with {len(self.adjacency_list)} nodes."
    def __getitem__(self, key):
        """
        A magic method that makes using keys possible.
        This makes it possible to use self[node] instead of self.adjacency_list[node]

        :return: The nodes that can be reached from the node `key`.
        :rtype: set[tuple[int]]
        """
        return self.adjacency_list[key]

    def __contains__(self, key):
        """
        This magic method makes it possible to check if a coordinate is in the graph.

        :return: This returns if the coordinate is in the graph.
        :rtype: bool
        """
        return key in self.adjacency_list

    def get_random_node(self):
        """
        This returns a random node from the graph.
        
        :return: A random node
        :rtype: tuple[int]
        """
        return RNG.choice(list(self.adjacency_list.keys()))
        
    def show_coordinates(self, size=5, color='k'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.

        :param size: The size of the dots, default to 5
        :type size: int
        :param color: The Matplotlib color of the dots, defaults to black
        :type color: string
        """
        for node in self.adjacency_list:
            plt.scatter(node[1], node[0], s=size, color=color)
         

    def show_edges(self, width=0.05, color='r'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.
        
        :param width: The width of the arrows, default to 0.05
        :type width: float
        :param color: The Matplotlib color of the arrows, defaults to red
        :type color: string
        """
        for node, edges in self.adjacency_list.items():
            for edge in edges:
                plt.arrow(node[1], node[0], edge[0][1] - node[1], edge[0][0] - node[0], width=width, color=color)

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        Dont use find_edges_for_node, it does not exist!!!!
        """
        for node in self.adjacency_list:
            self.find_next_node_in_adjacency_list(node, self.direction)
        
            

    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        #use adjency list to find the next node in the direction
        """
        dx, dy = direction
        x, y = node
        nx, ny = x + dx, y + dy
        while (nx, ny) in self.adjacency_list:
            self.adjacency_list[node].add(((nx, ny), 1, 1))
            self.adjacency_list[(nx, ny)].add((node, 1, 1))
            nx, ny = nx + dx, ny + dy


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################