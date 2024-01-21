from typing import Tuple, Optional, Callable, Iterable, Type, Union, List
from node import NodeAStar
from map import Map
from searchTreePQD import SearchTreePQD


def compute_cost(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """
    Computes the cost of simple moves between cells (i1, j1) and (i2, j2).

    Parameters
    ----------
    i1 : int
        Row number of the first cell in the grid.
    j1 : int
        Column number of the first cell in the grid.
    i2 : int
        Row number of the second cell in the grid.
    j2 : int
        Column number of the second cell in the grid.

    Returns
    ----------
    int | float
        Cost of the move between cells.

    Raises
    ----------
    ValueError
        If trying to compute the cost of a non-supported move (only cardinal moves are supported).
    """
    if abs(i1 - i2) + abs(j1 - j2) == 1:  # Cardinal move
        return 1
    else:
        raise ValueError("Trying to compute the cost of a non-supported move! ONLY cardinal moves are supported.")


def astar(
    task_map: Map,
    start_i: int,
    start_j: int,
    goal_i: int,
    goal_j: int,
    steps_limit: int,
    heuristic_func: Callable,
    search_tree: Type[SearchTreePQD],
) -> Tuple[bool, Optional[NodeAStar], int, int, Optional[Iterable[NodeAStar]], Optional[Iterable[NodeAStar]]]:
    """
    Implements the A* search algorithm.

    Parameters
    ----------
    task_map : Map
        The grid or map being searched.
    start_i, start_j : int, int
        Starting coordinates.
    goal_i, goal_j : int, int
        Goal coordinates.
    heuristic_func : Callable
        Heuristic function for estimating the distance from a node to the goal.
    search_tree : Type[SearchTreePQD]
        The search tree to use.

    Returns
    -------
    Tuple[bool, Optional[Node], int, int, Optional[Iterable[Node]], Optional[Iterable[Node]]]
        Tuple containing:
        - A boolean indicating if a path was found.
        - The last node in the found path or None.
        - Number of algorithm iterations.
        - Size of the resultant search tree.
        - OPEN set nodes for visualization or None.
        - CLOSED set nodes.
    """
    ast = search_tree()
    steps = 0

    start_node = NodeAStar(start_i, start_j, g=0, h=heuristic_func(start_i, start_j, goal_i, goal_j))
    ast.add_to_open(start_node)

    goal_node = NodeAStar(goal_i, goal_j)

    while not ast.open_is_empty():
        steps += 1
        node = ast.get_best_node_from_open()
        if node is None:
            break
        ast.add_to_closed(node)
        if node == goal_node:
            break
        i1 = node.i
        j1 = node.j
        neighbors = task_map.get_neighbors(i1, j1)
        for i2, j2 in neighbors:
            new_node = NodeAStar(
                i=i2,
                j=j2,
                g=node.g + compute_cost(i1, j1, i2, j2),
                h=heuristic_func(i2, j2, goal_i, goal_j),
                parent=node
            )
            if not ast.was_expanded(new_node):
                ast.add_to_open(new_node)

    path_found = ast.was_expanded(goal_node)
    last_node = None
    if path_found:
        for node in ast.expanded:
            if node == goal_node:
                last_node = node
                break

    return path_found, last_node, steps, len(ast), ast.opened, ast.expanded
