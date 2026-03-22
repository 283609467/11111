import numpy as np
import matplotlib.pyplot as plt
from path_planner.utils import ObstaclesGrid

class Node:
    def __init__(self, x, y, parent=None):
        """
        Represents a node in the RRT tree.
        
        Args:
            x (float): X-coordinate of the node.
            y (float): Y-coordinate of the node.
            parent (Node, optional): Parent node in the tree.
        """
        self.x = x
        self.y = y
        self.parent = parent  

class RRTPlanner:
    def __init__(self, start, goal, map_size, obstacles, max_iter=2000, step_size=5):
        """
        Initializes the RRT planner.
        """

        self.start = Node(start[1] * 10, start[0] * 10)
        self.goal = Node(goal[1] * 10, goal[0] * 10)
        
        self.map_size = map_size
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = [self.start]

    def plan(self):
        """
        Implements the RRT algorithm to find a path from start to goal.

        Returns:
            list: A list of (x, y) tuples representing the path from start to goal.
        """
        for i in range(self.max_iter):
            rand_node = self.sample_random_point()  
            nearest_node = self.find_nearest_node(rand_node)  
            new_node = self.steer(nearest_node, rand_node) 

            if new_node and not self.is_colliding(new_node, nearest_node): 
                self.tree.append(new_node)

                if self.reached_goal(new_node):  
                    return self.construct_path(new_node) 
        
        print("Path not found.")
        return None

    def sample_random_point(self):
        """
        Samples a random point in the map.
        
        Returns:
            Node: A randomly sampled node.
        """
        rand_x = np.random.uniform(0, self.map_size[0])
        rand_y = np.random.uniform(0, self.map_size[1])
        return Node(rand_x, rand_y)

    def find_nearest_node(self, rand_node):
        """
        Finds the nearest node in the tree to a given random node.

        Args:
            rand_node (Node): The randomly sampled node.

        Returns:
            Node: The nearest node in the tree.
        """
        nearest_node = self.tree[0]
        min_dist = np.inf
        for node in self.tree:
            dist = np.sqrt((node.x - rand_node.x)**2 + (node.y - rand_node.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def steer(self, nearest_node, rand_node):
        """
        Generates a new node by moving from the nearest node toward the random node.

        Args:
            nearest_node (Node): The nearest node in the tree.
            rand_node (Node): The randomly sampled node.

        Returns:
            Node: A new node in the direction of rand_node.
        """
        dist = np.sqrt((nearest_node.x - rand_node.x)**2 + (nearest_node.y - rand_node.y)**2)
        if dist == 0:
            return None


        if dist > self.step_size:
            theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
            new_x = nearest_node.x + self.step_size * np.cos(theta)
            new_y = nearest_node.y + self.step_size * np.sin(theta)
        else:
            new_x = rand_node.x
            new_y = rand_node.y


        return Node(new_x, new_y, parent=nearest_node)

    def is_colliding(self, new_node, nearest_node):
        """
        Checks if the path between nearest_node and new_node collides with an obstacle.

        Args:
            new_node (Node): The new node to check.
            nearest_node (Node): The nearest node in the tree.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        dist = np.sqrt((new_node.x - nearest_node.x)**2 + (new_node.y - nearest_node.y)**2)
        num_checks = max(2, int(dist)) 
        margin = 3  

        for i in range(num_checks + 1):
            t = i / num_checks
            check_x = nearest_node.x + t * (new_node.x - nearest_node.x)
            check_y = nearest_node.y + t * (new_node.y - nearest_node.y)

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

    def reached_goal(self, new_node):
        """
        Checks if the goal has been reached.

        Args:
            new_node (Node): The most recently added node.

        Returns:
            bool: True if goal is reached, False otherwise.
        """
        dist = np.sqrt((new_node.x - self.goal.x)**2 + (new_node.y - self.goal.y)**2)
        
        if dist <= self.step_size:
            return True
        return False

    def construct_path(self, end_node):
        """
        Constructs the final path by backtracking from the goal node to the start node.

        Args:
            end_node (Node): The node at the goal position.

        Returns:
            list: A list of (x, y) tuples representing the path from start to goal.
        """

        path = [(self.goal.x, self.goal.y)]
        

        current_node = end_node
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
            

        path.reverse()
        return path
