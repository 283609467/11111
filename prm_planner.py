import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from path_planner.utils import ObstaclesGrid
import heapq

class Node:
    def __init__(self, x, y):
        """
        Represents a node in the PRM roadmap.

        Args:
            x (float): X-coordinate of the node.
            y (float): Y-coordinate of the node.
        """
        self.x = x
        self.y = y

class PRMPlanner:
    def __init__(self, start, goal, map_size, obstacles, num_samples=2500, k_neighbors=30, step_size=2):
        """
        Initializes the PRM planner.

        Args:
            start (tuple): (x, y) coordinates of the start position.
            goal (tuple): (x, y) coordinates of the goal position.
            map_size (tuple): (width, height) of the environment.
            obstacles (ObstaclesGrid): Object that stores obstacle information.
            num_samples (int): Number of random samples for roadmap construction.
            k_neighbors (int): Number of nearest neighbors to connect in the roadmap.
            step_size (float): Step size used for collision checking.
        """
        self.start = Node(start[1] * 10, start[0] * 10)
        self.goal = Node(goal[1] * 10, goal[0] * 10)
        
        self.map_size = map_size
        self.obstacles = obstacles
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.step_size = step_size
        self.roadmap = []  
        self.edges = {} 

    def construct_roadmap(self):
        """
        Constructs the probabilistic roadmap by sampling nodes and connecting them.

        Returns:
        None
        """
        self.roadmap = [self.start, self.goal]
        

        for _ in range(self.num_samples):
            new_node = self.sample_free_point()
            if new_node:
                self.roadmap.append(new_node)
                

        for node in self.roadmap:
            self.edges[node] = []
            

        for node in self.roadmap:
            neighbors = self.find_k_nearest(node, self.k_neighbors)
            for neighbor in neighbors:

                if node != neighbor and not self.is_colliding(node, neighbor):
                    self.edges[node].append(neighbor)

    def sample_free_point(self):
        """
        Samples a random collision-free point in the environment.

        Returns:
        Node: A randomly sampled node.
        """
        max_attempts = 100
        margin = 3  
        for _ in range(max_attempts):
            rand_x = np.random.uniform(0, self.map_size[0])
            rand_y = np.random.uniform(0, self.map_size[1])
            
            row = int(rand_y)
            col = int(rand_x)
            
            if 0 <= row < self.map_size[1] and 0 <= col < self.map_size[0]:
                r_min = max(0, row - margin)
                r_max = min(self.map_size[1], row + margin + 1)
                c_min = max(0, col - margin)
                c_max = min(self.map_size[0], col + margin + 1)
                
                if not np.any(self.obstacles.map[r_min:r_max, c_min:c_max]):
                    return Node(rand_x, rand_y)
        return None

    def find_k_nearest(self, node, k):
        """
        Finds the k-nearest neighbors of a node in the roadmap.

        Args:
            node (Node): The node for which neighbors are searched.
            k (int): The number of nearest neighbors to find.

        Returns:
            list: A list of k-nearest neighbor nodes.
        """
        distances = []
        for other_node in self.roadmap:
            if other_node != node:
                dist = np.sqrt((node.x - other_node.x)**2 + (node.y - other_node.y)**2)
                distances.append((dist, other_node))
                

        distances.sort(key=lambda x: x[0])
        return [item[1] for item in distances[:k]]

    def is_colliding(self, node1, node2):
        """
        Checks if the path between two nodes collides with an obstacle.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        dist = np.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
        num_checks = max(2, int(dist * 3)) 
        margin = 3  

        for i in range(num_checks + 1):
            t = i / num_checks
            check_x = node1.x + t * (node2.x - node1.x)
            check_y = node1.y + t * (node2.y - node1.y)

            row = int(check_y)
            col = int(check_x)

            if row < 0 or row >= self.map_size[1] or col < 0 or col >= self.map_size[0]:
                return True
                
            r_min = max(0, row - margin)
            r_max = min(self.map_size[1], row + margin + 1)
            c_min = max(0, col - margin)
            c_max = min(self.map_size[0], col + margin + 1)
            
            if np.any(self.obstacles.map[r_min:r_max, c_min:c_max]):
                return True

        return False

    def plan(self):
        """
        Plans a path from start to goal using the constructed roadmap.

        Returns:
        list: A list of (x, y) tuples representing the path.
        """
        queue = [(0, id(self.start), self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}

        while queue:
            current_cost, _, current = heapq.heappop(queue)


            if current == self.goal:
                break

            for next_node in self.edges[current]:

                step_cost = np.sqrt((current.x - next_node.x)**2 + (current.y - next_node.y)**2)
                new_cost = cost_so_far[current] + step_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # f = g + h 
                    priority = new_cost + np.sqrt((self.goal.x - next_node.x)**2 + (self.goal.y - next_node.y)**2)
                    heapq.heappush(queue, (priority, id(next_node), next_node))
                    came_from[next_node] = current
                    

        if self.goal not in came_from:
            print("Path not found by PRM.")
            return None
            
        path = []
        current = self.goal
        while current is not None:
            path.append((current.x, current.y))
            current = came_from[current]
            
        path.reverse()
        return path
