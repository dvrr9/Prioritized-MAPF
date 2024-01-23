from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import numpy.typing as npt
from node import Node
import numpy as np


def convert_string_to_cells(cell_str: str) -> npt.NDArray:
    """
    Converts a string representation of a grid map, with '#' for obstacles and '.' for free cells, into a binary matrix.

    Parameters
    ----------
    cell_str : str
        String containing grid map information ('#' for obstacles and '.' for free cells).

    Returns
    ----------
    cells : np.ndarray
        Binary matrix representing the grid map.
    """
    lines = cell_str.replace(" ", "").split("\n")

    cells = np.array(
        [[1 if char != "." else 0 for char in line] for line in lines if line],
        dtype=np.int8,
    )
    return cells


def make_path(goal: Node) -> Tuple[List[Tuple[int, int]], Union[int, float]]:
    """
    Creates a path by tracing parent pointers from the goal node back to the start node.
    It also calculates the solution's duration.

    Parameters
    ----------
    goal : Node
        The goal node in the search tree from which to trace back.

    Returns
    -------
    Tuple[List[Tuple[int, int]], Union[int, float]]
        A tuple containing the path as a list of (i, j) coordinates, and the duration of the solution.
    """

    duration = goal.g
    current = goal
    path = [(goal.i, goal.j)]
    while current.parent:
        for i in range(current.g - current.parent.g):
            path.append((current.parent.i, current.parent.j))
        current = current.parent
    return path[::-1], duration