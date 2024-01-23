from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from conflicttype import ConflictType
import random

random.seed(239)


class PTNode:
    def __init__(self, times_visited=0, parent=None, priority=None, plan=None) -> None:
        self.times_visited = 0
        self.parent = parent
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
                pos1 == pos2
                or (
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

    def find_collision(self):
        """
        Finds arbitrary collision if exists and returns pair of conflicting agents
        """
        list1 = list(range(len(self.plan)))
        random.shuffle(list1)
        for a1 in list1:
            list2 = list(range(a1 + 1, len(self.plan)))
            random.shuffle(list2)
            for a2 in list2:
                if self.has_conflict(a1, a2):
                    return (a1, a2)
        return None

    def update_cost(self):
        """
        Updates cost according to existing plan
        """
        self.cost = sum(len(path) - 1 for path in self.plan)


class PTNodeGPBS:
    def __init__(self, n_agents, times_visited=0, parent=None, priority=None, plan=None) -> None:
        self.times_visited = 0
        self.parent = parent
        self.priority = priority # (lower, higher) priority agents
        self.n_agents = n_agents
        self.agent_conflicts = [set() for i in range(n_agents)]
        self.conflicts = []
        self.plan = plan
        self.cost = sum(len(path) - 1 for path in self.plan)
    
    def has_conflict(self, a1, a2) -> ConflictType:
        """
        Check for conflict between agents a1 and a2
        """
        min_path_length = min(len(self.plan[a1]), len(self.plan[a2]))
        for time in range(min_path_length):
            pos1 = self.plan[a1][time]
            pos2 = self.plan[a2][time]
            if pos1 == pos2:
                return ConflictType.VERTEX
            if (
                time < min_path_length - 1
                and self.plan[a2][time + 1] == pos1
                and self.plan[a1][time + 1] == pos2
            ):
                return ConflictType.EDGE
        if len(self.plan[a1]) != len(self.plan[a2]):
            stopped = a1 if len(self.plan[a1])==min_path_length else a2
            other = a2 if len(self.plan[a1])==min_path_length else a1
            stop_pos = self.plan[stopped][-1]

            for time in range(min_path_length, len(self.plan[other])):
                if self.plan[other][time] == stop_pos:
                    return ConflictType.GOAL_VERTEX
        return ConflictType.NO_CONFLICT

    def find_collision(self, priorities):
        """
        Finds arbitrary collision if exists and returns pair of conflicting agents
        """
        if self.conflicts:
            conflict_type = self.conflicts[-1][1]
            if conflict_type == ConflictType.GOAL_VERTEX: # TR Heuristic
                conflict = self.conflicts[-1][0]
                if len(self.plan[conflict[0]]) < len(self.plan[conflict[1]]):
                    return conflict
                else:
                    return tuple(reversed(conflict))
            else: # IC Heuristic
                lp_agents = [priorities.get_lower_priority_agents(i).union({i}) for i in range(self.n_agents)]
                hp_agents = [priorities.get_higher_priority_agents(i).union({i}) for i in range(self.n_agents)]
                all_conflicts = {conflict for conflict, _ in self.conflicts}
                max_new_conflicts = 0
                arg_max = self.conflicts[0][0]
                for conflict, _ in self.conflicts:
                    new_conflicts = 0
                    for a1 in lp_agents[conflict[0]]:
                        for a2 in hp_agents[conflict[1]]:
                            if (min(a1, a2), max(a1, a2)) not in all_conflicts:
                                new_conflicts += 1
                    if new_conflicts > max_new_conflicts:
                        max_new_conflicts = new_conflicts
                        arg_max = conflict

                    new_conflicts = 0
                    for a1 in lp_agents[conflict[1]]:
                        for a2 in hp_agents[conflict[0]]:
                            if (min(a1, a2), max(a1, a2)) not in all_conflicts:
                                new_conflicts += 1
                    if new_conflicts > max_new_conflicts:
                        max_new_conflicts = new_conflicts
                        arg_max = tuple(reversed(conflict))
                return arg_max
        return None

    def update_cost(self):
        """
        Updates cost according to existing plan
        """
        self.cost = sum(len(path) - 1 for path in self.plan)

    def update_conflicts(self):
        self.conflicts = []
        for a1 in range(self.n_agents):
            for a2, conflict_type in self.agent_conflicts[a1]:
                if a1 < a2:
                    self.conflicts.append(((a1, a2), conflict_type))
