"""
Based on implementation of Atsushi Sakai (https://github.com/AtsushiSakai/PythonRobotics)
Most significant changes made:
    - Change the unit of the parameters i.e. sx, sy, gx, gy.
    - A binary gridmap as input directly in stead of calculating it by obstacle positions.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

class PathPlanner:

    def planning(self, sx, sy, gx, gy, obstacle_map):
        """
        path search
        :param sx: start x position [pix]
        :param sy: start y position [pix]
        :param gx: goal x position [pix]
        :param gy: goal y position [pix]
        :param obstacle_map: a binary 2d ndarray. The value of a iterm is Ture (obstacle) or False (free).
        :return:
                shortest_path: a list of grid positions (x, y) in pixels
        """
        raise NotImplementedError()

class AStarPlanner(PathPlanner):

    def __init__(self):
        self.motion = self.get_motion_model()
        self.obstacle_map = None

    def planning(self, sx, sy, gx, gy, obstacle_map, weight = 1.0, type = 'manhattan'):
        """
        A* path search
        :param sx: start x position [pix]
        :param sy: start y position [pix]
        :param gx: goal x position [pix]
        :param gy: goal y position [pix]
        :param obstacle_map: a binary 2d numpy array. The value of a iterm is Ture (obstacle) or False (free).
                the value of point(x, y) in the map is obstacle_map[y, x].
        :param weight: weight of heuristic
        :param type: name of the method to calculate heuristic term
        :return:
                shortest_path: a list of grid positions (x, y)
        """
        self.obstacle_map = obstacle_map
        start_node = self.Node(sx, sy, 0, -1, 0)
        goal_node = self.Node(gx, gy, 0, -1, 0)

        if self.__vertify_node(start_node) == False or self.__vertify_node(goal_node) == False:
            # raise ValueError() # start or goal  is not valid
            return list()

        open_set = dict() # open set, i.e. a set of nodes to be evaluated
        closed_set = dict() # closed set, i.e. a set of nodes already evaluated
        open_set[self.__calc_grid_index(start_node)] = start_node
        while True:
            if len(open_set) == 0:
                break

            c_id = min(open_set,   # find the index of a node which has the maximum score
                    key=lambda o: open_set[o].score)

            current = open_set[c_id] # the current node, i.e. the node which has the highest heuristic score in the open set.
            if current == goal_node: # find the goal
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id] # remove the current node from the open set
            closed_set[c_id] = current # add it to the closed set

            # expand search grid based on motion model
            for i, m in enumerate(self.motion):
                dx, dy, cost = m # increments of position and cost
                node = self.Node(current.x + dx, current.y + dy,  # new search node
                                 current.cost + cost, c_id, 0)       # cost and parent's id
                n_id = self.__calc_grid_index(node)

                if not self.__vertify_node(node): # not a valid node
                    continue

                if n_id in closed_set: # this node has been evaluated
                    continue

                node.score = node.cost + self.__calc_heristic(node, goal_node, weight = weight, type = type)
                if n_id not in open_set:
                    open_set[n_id] = node # add the new node in the open set to be evaluated
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node # update the node in open set with smaller cost


        return self.__calc_final_path(goal_node, closed_set)

    class Node:

        def __init__(self, x, y, cost, parent_index, score):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost # cost of moving
            self.score = score # score = cost + heuristic
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

        def __eq__(self, other):
            if self.x == other.x and self.y == other.y:
                return True
            else:
                return False

    def __calc_final_path(self, goal_node, closed_set):
        """
        Generate final course
        :param goal_node:
        :param closed_set:  a set of node already evaluated
        :return:
                shortest_path: a list of grid positions (x, y)
        """
        shortest_path = list()
        shortest_path.append((goal_node.x, goal_node.y))
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            shortest_path.append((n.x, n.y))
            parent_index = n.parent_index
        return shortest_path

    def __vertify_node(self, node):
        height, width = self.obstacle_map.shape
        # valid index check
        if node.x < 0 or node.x >= width or node.y < 0 or node.y >= height:
            return False

        # collision check
        if self.obstacle_map[node.y, node.x]:
            return False

        return True

    def __calc_grid_index(self, node):
        height, width = self.obstacle_map.shape
        n_id = width * node.y + node.x
        return n_id

    def get_motion_model(self):
        """
        Calculate next steps and costs in all directions
        :return: dx, dy, cost
        """
        sq2 = math.sqrt(2)
        motion = list()
        motion.append((1, 0, 1))
        motion.append((0, 1, 1))
        motion.append((-1, 0, 1))
        motion.append((0, -1, 1))
        motion.append((-1, -1, sq2))
        motion.append((-1, 1, sq2))
        motion.append((1, -1, sq2))
        motion.append((1, 1, sq2))
        return motion

    def __calc_heristic(self, n1, n2, weight = 1.0, type = 'manhattan'):
        """
        Calculate heuristic term
        :param n1: A Node object
        :param n2: A node object
        :param weight: weight of heuristic
        :param type: name of the method to calculate heuristic term
        :return: A heuristic value
        """
        if type == 'manhattan':
            d = n1.x - n2.x + n1.y - n2.y # manhattan distance
        elif type == 'euclidean':
            d = math.hypot(n1.x - n2.x, n1.y - n2.y)  # euclidean distance as heuristic term
        else:
            d = 0

        d = weight*d
        return d


if __name__ == '__main__':
    import time
    # Generate a 2d grid map
    map = np.full((100, 100), False, dtype=np.bool)
    for i in range(80):
        map[i, 20] = True
    for i in range(0, 100):
        map[i, 40] = True
    for i in range(90):
        map[i, 60] = True
    for i in range(10, 100):
        map[i, 80] = True

    start_time = time.time()
    # set start and goal
    start = [5, 2]  # x, y
    goal = [90,98]  # x, y
    # calculate shortest path
    astar = AStarPlanner()
    path = astar.planning(start[0], start[1], goal[0], goal[1], map, weight=5, type='euclidean')
    print ("time cost:", time.time()-start_time)
    plt.imshow(map, origin='lower')
    map[start[1], start[0]] = True
    map[goal[1], goal[0]] = True
    x, y = zip(*path)
    plt.plot(x, y, 'bx')

    plt.plot(start[0], start[1],'r*')
    plt.plot(goal[0], goal[1], 'r*')

    plt.show()
