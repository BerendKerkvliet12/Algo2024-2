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
        self.adjacency_list = {}
        self.map = map_
        self.start = start
        
        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell
        
    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        So far, we called this method `step`. However, this class is more than just the search algorithm,
        therefore, we gave it a bit more descriptive name.

        Note, that we only want to find the nodes, so history does not need to contain a partial path (previous node).
        In `find_edges` (the next cell), we will add edges for each node.
        """
        queue = deque([self.start])
        history = {self.start}
        self.adjacency_list[self.start] = set()
        while queue:
            current_node = queue.popleft()
            neighbours = self.neighbour_coordinates(current_node)
            for neighbour in neighbours:
                if neighbour not in history:
                    history.add(neighbour)
                    queue.append(neighbour)
                    self.adjacency_list_add_node(neighbour, neighbours)
        
        
                    
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
        if len(actions) > 2: # If the coordinate is a corner or crossing
            self.adjacency_list[coordinate] = set()
        if len(actions) == 0: # If the coordinate is a dead end
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
        max_height, max_width = self.map.shape
        possible_moves = [
            (coordinate[0] + 1, coordinate[1]),  # Move down
            (coordinate[0] - 1, coordinate[1]),  # Move up
            (coordinate[0], coordinate[1] + 1),  # Move right
            (coordinate[0], coordinate[1] - 1)   # Move left
        ]
        # Check if the moves are within the map's boundaries
        return [move for move in possible_moves if 0 <= move[0] < max_height and 0 <= move[1] < max_width and self.map.grid[move[0], move[1]] != 0]
    
    def __repr__(self):
        """
        This returns a representation of a graph.

        :return: A string representing the graph object.
        :rtype: str
        """
        # You can change this to anything you like, such that you can easily print a Graph object. An example is already given.
        return repr(dict(sorted(self.adjacency_list.items()))).replace("},", "},\n")

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
        return tuple(RNG.choice(list(self.adjacency_list)))
        
    def show_coordinates(self, size=5, color='k'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.

        :param size: The size of the dots, default to 5
        :type size: int
        :param color: The Matplotlib color of the dots, defaults to black
        :type color: string
        """
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)        

    def show_edges(self, width=0.05, color='r'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.
        
        :param width: The width of the arrows, default to 0.05
        :type width: float
        :param color: The Matplotlib color of the arrows, defaults to red
        :type color: string
        """
        for node, edge_list in self.adjacency_list.items():
            for next_node,_,_ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1])*0.975, (next_node[0] - node[0])*0.975, color=color, length_includes_head=True, width=width, head_width=4*width)

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        for node in self.adjacency_list.keys():
            neighbors = self.neighbour_coordinates(node)
            for neighbor in neighbors:
                if neighbor in self.adjacency_list:  # Check if the neighbor is also a node
                    distance = self.map.grid[neighbor]  # Assume distance or weight could be directly from map values
                    edge = (neighbor, distance)  # Simplified edge definition
                    self.adjacency_list[node].add(edge)

    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """
        x, y = node
        dx, dy = direction
        while 0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]:
            if self.map.grid[x, y] != 0:
                return (x, y), self.map.grid[x, y]
            x += dx
            y += dy
        return None, 0

############ CODE BLOCK 120 ################

class FloodFillSolverGraph(FloodFillSolver):
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the nodes that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __call__(self, graph, source, destination):      
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        Hint: The history is already given as a dictionary with as keys the node in the state-space graph and
        as values the previous node from which this node was visited.

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts.
        :type source: tuple[int]
        :param destination: The node where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of nodes and the length of the route.
        :rtype: list[tuple[int]], float
        """       
        self.queue = deque([source])
        self.history = {source: None}
        
        while self.queue:
            current = self.queue.popleft()
            
            if current == destination:
                break
            
            for neighbor in self.next_step(current):
                if neighbor not in self.history:
                    self.queue.append(neighbor)
                    self.history[neighbor] = current
        
        path = []
        step = destination
        while step is not None:
            path.append(step)
            step = self.history.get(step)
        
        path.reverse()
        return path, len(path)


    # def find_path(self):
    #     """
    #     This method finds the shortest paths between the source node and the destination node.
    #     It also returns the length of the path. 
        
    #     Note, that going from one node to the next has a length of 1.

    #     :return: A path that is the optimal route from source to destination and its length.
    #     :rtype: list[tuple[int]], float
    #     """
    #     raise NotImplementedError("Please complete this method")       

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        y, x = node
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < graph.map.shape[0] and 0 <= nx < graph.map.shape[1] and graph.map[ny, nx] != 0:
                neighbors.append((ny, nx))
        return neighbors

############ CODE BLOCK 130 ################

class BFSSolverShortestPath():
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the distances it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple(int), float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the distance it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], int]]
    """   
    def __call__(self, graph, source, destination):      
        """
        This method gives the shortest route through the graph from the source to the destination node.
        You start at the source node and the algorithm ends if you reach the destination node, 
        both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts
        :type source: tuple[int] 
        :param destination: The node where the path ends
        :type destination: tuple[int]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: The shortest route and the time it takes. The route consists of a list of nodes.
        :rtype: list[tuple[int]], float
        """       
        self.priorityqueue = [(0, source)]
        self.history = {source: (None, 0)}
        self.destination = destination
        self.graph = graph
        self.vehicle_speed = 1  # For now, we assume the vehicle speed is 1
        
        self.main_loop()
        return self.find_path()    

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        path = []
        step = self.destination
        while step is not None:
            path.append(step)
            step = self.history[step][0]
        
        path.reverse()
        return path, self.history[self.destination][1]      

    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.priorityqueue:
            self.priorityqueue.sort()  # Ensure the list is sorted to always pop the smallest element
            current_distance, current_node = self.priorityqueue.pop(0)
            
            if self.base_case(current_node):
                break
            
            for neighbor in self.next_step(current_node):
                distance = current_distance + self.new_cost(current_node, neighbor, speed_limit=1)
                
                if neighbor not in self.history or distance < self.history[neighbor][1]:
                    self.history[neighbor] = (current_node, distance)
                    self.priorityqueue.append((distance, neighbor))

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        return node == self.destination

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        For now, speed_limit can be ignored.

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        y1, x1 = previous_node
        y2, x2 = new_node
        return abs(y1 - y2) + abs(x1 - x2)

    def step(self, node, new_node, distance, speed_limit):
        """
        One step in the BFS algorithm. For now, speed_limit can be ignored.

        :param node: The current node
        :type node: tuple[int]
        :param new_node: The next node that can be visited from the current node
        :type new_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        """
        new_cost = self.history[node][1] + distance
        if new_node not in self.history or new_cost < self.history[new_node][1]:
            self.history[new_node] = (node, new_cost)
            self.priorityqueue.append((new_cost, new_node))
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        y, x = node
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.graph.map.shape[0] and 0 <= nx < self.graph.map.shape[1] and self.graph.map[ny, nx] != 0:
                neighbors.append((ny, nx))
        return neighbors


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
