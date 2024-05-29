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

        if self.destination not in self.history: # If the destination is not in the history, it is not reachable
            return [], 0

        path = []
        current_node = self.destination
        while current_node is not None: # While we have not reached the source
            path.append(current_node)
            current_node = self.history.get(current_node)  # Get the previous node
        path.reverse()
        return path, len(path) - 1 # The length of the path is the number of steps taken - 1
            
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue: # While there are still nodes to visit
            current_node = self.queue.popleft() 
            if self.base_case(current_node): # If the base case is reached,
                return
            for new_node in self.next_step(current_node): # Get the next possible steps
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
        if new_node not in self.history: # If the new node has not been visited yet
            self.history[new_node] = node
            self.queue.append(new_node) # Add the new node to the queue

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = node
        valid_steps = []
        for dx, dy in directions: # Check all possible directions
            nx, ny = x + dx, y + dy
            # Ensure next step is within grid bounds and is a road (assuming 1 represents roads)
            if 0 <= nx < self.road_grid.shape[0] and 0 <= ny < self.road_grid.shape[1]:
                if self.road_grid[nx, ny] != 0:  # Only consider valid road parts
                    valid_steps.append((nx, ny))
        return valid_steps # Return the valid steps

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
        while queue: # While there are still nodes to visit
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node) # Add the current node to the visited set
            actions = self.neighbour_coordinates(current_node) # Get the possible actions from the current node
            self.adjacency_list_add_node(current_node, actions) # Add the current node to the adjacency list
            for action in actions:
                queue.append(action) # Add the possible actions to the queue
        
        
                    
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
        if len(actions) in [1, 3, 4]: # If the coordinate is a dead end or crossing
            self.adjacency_list[coordinate] = set()
        #add corners in the graph
        elif len(actions) == 2: 
        #check if the two actions form a corner
            (x1 , y1) , (x2, y2) = actions
            if (x1 != x2 and y1 != y2):
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
            if 0 <= nx < self.map.shape[0] and 0 <= ny < self.map.shape[1]: # Check if the next step is within the grid
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
        for node, edges in self.adjacency_list.items(): # For all nodes in the adjacency list
            for edge in edges:
                plt.arrow(node[1], node[0], edge[0][1] - node[1], edge[0][0] - node[0], width=width, color=color)

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for node in self.adjacency_list:
            for direction in directions:
                neighbor, distance = self.find_next_node_in_adjacency_list(node, direction) 
                if neighbor:
                    speed_limit = self.map[neighbor[0], neighbor[1]] # The speed limit is the value of the map at the neighbor
                    self.adjacency_list[node].add((neighbor, distance, speed_limit)) # Add the neighbor to the adjacency list


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
        distance = 0

        while True:
            x += dx
            y += dy
            distance += 1

            if not (0 <= x < self.map.shape[0] and 0 <= y < self.map.shape[1]):
                return None, 0  # Out of bounds

            if self.map[x, y] == 0:
                return None, 0  # Encountered an obstacle

            if (x, y) in self.adjacency_list:
                return (x, y), distance  # Found the next node

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
        self.graph = graph
        self.destination = destination

        self.main_loop()

        return self.find_path()

        
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return [edge[0] for edge in self.graph[node]] # Return the nodes that can be reached from the current node

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
        self.destination = tuple(destination)
        self.graph = graph

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
        
        path.reverse() # Reverse the path to get the correct order
        return path, self.history[self.destination][1]  # Return the path and the time it takes to get to the destination

    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.priorityqueue: 
            current_distance, current_node = self.priorityqueue.pop(0)
            
            if self.base_case(current_node): # If the base case is reached
                break
            
            for neighbor, distance, speed_limit in self.next_step(current_node): # Get the next possible steps
                self.step(current_node, neighbor, distance, speed_limit) # Take a step

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        return np.array_equal(node, self.destination)


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
        return self.history[previous_node][1] + distance # The cost is the distance from the previous node to the new node
        

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
        new_cost = self.new_cost(node, distance, speed_limit)
        if new_node not in self.history or new_cost < self.history[new_node][1]: # If the new node has not been visited yet or the new cost is lower
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
        return list(self.graph[node])

############ CODE BLOCK 200 ################

class BFSSolverFastestPath(BFSSolverShortestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited 
                              including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the time it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):      
        """
        This method gives a fastest route through the grid from source to destination.

        This is the same as the `__call__` method from `BFSSolverShortestPath` except that 
        we need to store the vehicle speed. 
        
        Here, you can see how we can overwrite the `__call__` method but 
        still use the `__call__` method of BFSSolverShortestPath using `super`.
        """
        self.vehicle_speed = vehicle_speed
        return super(BFSSolverFastestPath, self).__call__(graph, source, destination)

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        Use the `speed_limit` and `vehicle_speed` to determine the time/cost it takes to go to
        the new node from the previous_node and add the time it took to reach the previous_node to it..

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        effective_speed = min(self.vehicle_speed, speed_limit) # The effective speed is the minimum of the vehicle speed and the speed limit
        travel_time = distance / effective_speed
        return self.history[previous_node][1] + travel_time # The cost is the time it takes to get from the previous node to the new node

############ CODE BLOCK 210 ################

def coordinate_to_node(map_, graph, coordinate):
    """
    This function finds a path from a coordinate to its closest nodes.
    A closest node is defined as the first node you encounter if you go a certain direction.
    This means that unless the coordinate is a node, you will need to find two closest nodes.
    If the coordinate is a node then return a list with only the coordinate itself.

    :param map_: The map of the graph
    :type map_: Map
    :param graph: A Graph of the map
    :type graph: Graph
    :param coordinate: The coordinate from which we want to find the closest node in the graph
    :type coordinate: tuple[int]
    :return: This returns a list of closest nodes which contains either 1 or 2 nodes.
    :rtype: list[tuple[int]]
    """
    if coordinate in graph.adjacency_list: # If the coordinate is a node
        return [coordinate]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    closest_nodes = []
    visited = set()
    queue = deque([(coordinate, 0)])

    while queue and len(closest_nodes) < 2: # While there are still nodes to visit
        current, dist = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        
        for direction in directions:
            next_coord = list(current) # Convert to list to be able to change the values
            while 0 <= next_coord[0] < map_.shape[0] and 0 <= next_coord[1] < map_.shape[1]: # While the next coordinate is within the grid
                next_coord[0] += direction[0]
                next_coord[1] += direction[1]
                next_tuple = tuple(next_coord)
                if map_[next_tuple[0], next_tuple[1]] == 0: # If the next coordinate is an obstacle
                    break
                if next_tuple in graph.adjacency_list: # If the next coordinate is a node
                    closest_nodes.append(next_tuple)
                    break
                if next_tuple not in visited: # If the next coordinate has not been visited yet
                    queue.append((next_tuple, dist + 1))
            if len(closest_nodes) == 2: # If we have found two closest nodes
                break

    return closest_nodes

############ CODE BLOCK 220 ################

def create_country_graphs(map_):
    """
    This function returns a list of all graphs of a country map, where the first graph is the highways and the rest are the cities.

    :param map_: The country map
    :type map_: Map
    :return: A list of graphs
    :rtype: list[Graph]
    """
    main_graph = Graph(map_)
    highway_graph = Graph(map_)
    highway_graph.adjacency_list = {}
    city_graphs = []

    visited = set()
    
    for node, value in np.ndenumerate(map_):
        if value == 2:  # Assuming 2 represents highways
            highway_graph.adjacency_list[node] = set()
        elif value == 1:  # Assuming 1 represents city roads
            if node not in visited:
                city_queue = deque([node])
                city_nodes = set()
                
                while city_queue:
                    current_node = city_queue.popleft()
                    if current_node in visited:
                        continue
                    visited.add(current_node)
                    city_nodes.add(current_node)
                    
                    for neighbor in main_graph.neighbour_coordinates(current_node):
                        if map_[neighbor] == 1 and neighbor not in visited:
                            city_queue.append(neighbor)
                
                city_graph = Graph(map_)
                city_graph.adjacency_list = {n: set() for n in city_nodes}
                city_graph.find_edges()
                city_graphs.append(city_graph)

    highway_graph.find_edges()
    return [highway_graph] + city_graphs

############ CODE BLOCK 230 ################

class BFSSolverMultipleFastestPaths(BFSSolverFastestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that are visited and as values the node that leads to this node including the time it takes from the start node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
        :param found_destinations: The destinations already found with Dijkstra.
        :type found_destinations: list[tuple[int]]
    """
    def __init__(self, find_at_most=3):
        """
        This init makes it possible to make a different Dijkstra algorithm 
        that find more or less destination nodes before it stops searching.

        :param find_at_most: The number of found destination nodes before the algorithm stops
        :type find_at_most: int
        """
        self.find_at_most = find_at_most
    
    def __call__(self, graph, sources, destinations, vehicle_speed):      
        """
        This method gives the top three fastest routes through the grid from any of the sources to any of the destinations.
        You start at the sources and the algorithm ends if you reach enough destinations, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param sources: The nodes where the path starts and the time it took to get here.
        :type sources: list[tuple[tuple[int], float]]
        :param destinations: The nodes where the path ends and the time it took to get here.
        :type destinations: list[tuple[tuple[int], float]]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """       
        self.priorityqueue = sorted(sources, key=lambda x:x[1])
        self.history = {s: (None, t) for s, t in sources}
        
        self.destinations = destinations
        self.destination_nodes = [dest[0] for dest in destinations]
        self.found_destinations = []

        raise NotImplementedError("Please complete this method")       

    def find_n_paths(self):
        """
        This method needs to find the top `n` fastest paths between any source node and any destination node.
        This does not mean that each source node has to be in a path nor that each destination node needs to be in a path.

        Hint1: The fastest path is stored in each node by linking to the previous node. 
               Therefore, if you start searching from a destination node,
               you always find the optimal path from that destination node.
               This is similar if you only had one destination node.         

        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """
        raise NotImplementedError("Please complete this method")       
        
    def base_case(self, node):
        """
        This method checks if the base case is reached and
        updates self.found_destinations

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        raise NotImplementedError("Please complete this method")


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
