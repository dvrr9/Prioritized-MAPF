from enum import Enum


class ConflictType(Enum):
    NO_CONFLICT = 0
    VERTEX = 1
    EDGE = 2
    GOAL_VERTEX = 3
