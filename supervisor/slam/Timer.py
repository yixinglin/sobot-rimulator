import time

T_NOT_REACH_GOAL = 60 # time that robot not reaches a goal in seconds
T_NO_BEST_DIST_CHANGE = 30 # time that robot not updates the best distance to goal

class Timer:
    """
    Timer for state transition. Avoid robot only moving around a circle endlessly
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Timing how long does the robot not reach the goal?
        self.previous_goal = [0, 0]
        self.start_time_to_goal = time.time()
        # Timing how long does best_distance_to_goal not changed?
        self.previous_best_distance = float("inf")
        self.start_time_best_distance_to_goal = time.time()

    def not_reach_goal_in_time(self, goal):
        """
        :param goal: goal position
        :return: True or False
        """
        flg = False
        if goal != self.previous_goal:  # update previous goal, if goal is changed
            self.previous_goal = goal[:]
            self.start_time_to_goal = time.time()
            flg = False
        elif time.time() - self.start_time_to_goal > T_NOT_REACH_GOAL:  # goal dose not change over 30s
            self.start_time_to_goal = time.time()
            flg = True
        return flg

    def previous_best_distance_not_changed_in_time(self, best_distance_to_goal):
        if best_distance_to_goal != self.previous_best_distance:  # update previous distance if it was changed
            self.start_time_best_distance_to_goal = time.time()
            self.previous_best_distance = best_distance_to_goal
            return False
        elif time.time() - self.start_time_best_distance_to_goal > T_NO_BEST_DIST_CHANGE:  # no changes in time
            self.start_time_to_goal = time.time()
            print("Best distance dose not changed in time.")
            return True
        return False