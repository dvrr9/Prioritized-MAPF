from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

class PTNode:
    def __init__(self, times_visited=0, parent=None, priority=None, plan=None) -> None:
        self.times_visited = 0
        self.parent = parent
        # TODO: add class for priority edges
        self.priority = priority # (lower, higher) priority agents
        self.plan = plan
        self.cost = sum(len(path) - 1 for path in self.plan)
    
    def has_conflict(self, a1, a2) -> bool:
        """
        Check for conflict between agents a1 and a2
        """
        min_path_length = min(len(self.plan[a1]), len(self.plan[a2]))
        for time in range(min_path_length):
            pos1 = self.plan[a1][time]
            pos2 = self.plan[a2][time]
            if (
                pos1 == pos2 # vertex conflict
                or ( # edge conflict
                    time < min_path_length - 1
                    and self.plan[a2][time + 1] == pos1
                    and self.plan[a1][time + 1] == pos2
                )
            ):
                return True
        if len(self.plan[a1]) != len(self.plan[a2]):
            stopped = a1 if len(self.plan[a1])==min_path_length else a2
            other = a2 if len(self.plan[a1])==min_path_length else a1
            stop_pos = self.plan[stopped][-1]

            for time in range(min_path_length, len(self.plan[other])):
                if self.plan[other][time] == stop_pos:
                    return True
        return False

    def find_collision(self): #TODO: check for those without constraint
        """
        Finds arbitrary collision if exists and returns pair of conflicting agents
        """
        for a1 in range(len(self.plan)):
            for a2 in range(a1 + 1, len(self.plan)):
                if self.has_conflict(a1, a2):
                    return (a1, a2)
        return None

    def update_cost(self):
        """
        Updates cost according to existing plan
        """
        self.cost = sum(len(path) - 1 for path in self.plan)