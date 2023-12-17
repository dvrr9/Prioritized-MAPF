from catable import CATable
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from node import Node
from map import Map
from searchTreePQD import SearchTreePQD

def compute_cost_timesteps(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """
    Computes the cost of simple moves between cells (i1, j1) and (i2, j2) - `wait` action is allowed.

    Parameters
    ----------
        i1 : int
            Number of the first cell row in grid.
        j1 : int
            Number of the first cell column in grid.
        i2 : int
            Number of the second cell row in grid.
        j2 : int
            Number of the second cell column in grid.

    Returns
    ----------
    int | float
        Cost of the move between cells.

    Raises
    ----------
    ValueError
        If trying to compute the cost of a non-supported move (only cardinal moves are supported).
    """

    d = abs(i1 - i2) + abs(j1 - j2)
    if d == 0:  # wait
        return 1
    elif d == 1:  # cardinal move
        return 1
    else:
        raise ValueError(
            "Trying to compute the cost of a non-supported move! ONLY cardinal moves are supported."
        )
    
def get_neighbors_timestep(
    i: int, j: int, t: int, grid_map: Map, ca_table: CATable
) -> List[Tuple[int, int]]:
    """
    Returns a list of neighboring cells as (i, j) tuples. The function returns neighbors
    that allow only cardinal moves, as well as the current cell, to account for the possibility
    of a wait action.

    Parameters
    ----------
    i, j : int
        Coordinates of the cell on the grid map.
    grid_map : Map
        Static grid map information.
    ca_table : CATable
        Collision avoidance table.

    Returns
    -------
    neighbours : List[Tuple[int, int]]
        List of neighboring cell coordinates (i, j).
    """
    neighbors = grid_map.get_neighbors(i, j)
    neighbors.append((i, j))
    results = []
    for neighbor in neighbors:
        i_n, j_n = neighbor
        if ca_table.check_move(i, j, i_n, j_n, t):
            results.append(neighbor)
    

    # YOUR CODE HERE
    return results

def manhattan_distance(i1: int, j1: int, i2: int, j2: int) -> Union[int, float]:
    """
    Calculates the Manhattan distance, which is the sum of the absolute differences
    of the coordinates, between two cells on a grid.

    Parameters
    ----------
    i1, j1 : int
        Coordinates (i, j) of the first cell on the grid.
    i2, j2 : int
        Coordinates (i, j) of the second cell on the grid.

    Returns
    -------
    int | float
        The Manhattan distance between the two specified cells.
    """
    # YOUR CODE HERE
    
    return abs(i1 - i2) + abs(j1 - j2)

def astar_timesteps(
    task_map: Map,
    ca_table: CATable,
    start_i: int,
    start_j: int,
    goal_i: int,
    goal_j: int,
    steps_max: int,
    heuristic_func: Callable,
    search_tree: Type[SearchTreePQD],
) -> Tuple[
    bool, Optional[Node], int, int, Optional[Iterable[Node]], Optional[Iterable[Node]]
]:
    """
    Implementation of A* algorithm without re-expansion on dynamic obstacles domain.
    """

    ast = search_tree()
    steps = 0
    start_node = Node(
        start_i, start_j, 0, g=0, h=heuristic_func(start_i, start_j, goal_i, goal_j)
    )


    ast.add_to_open(start_node)

    while not ast.open_is_empty():
        cur = ast.get_best_node_from_open()
        if cur is None:
            break
        if cur.i == goal_i and cur.j == goal_j and  cur.t > ca_table.last_visited(goal_i, goal_j):
            return True, cur, steps, len(ast), ast.opened, ast.expanded
        ast.add_to_closed(cur)
        for i, j in get_neighbors_timestep(cur.i, cur.j, cur.t, task_map, ca_table):
            new_node = Node(i, j, cur.t + 1)
            if not ast.was_expanded(new_node): 
                new_node.g = cur.g + compute_cost_timesteps(cur.i, cur.j, i, j)
                new_node.h = heuristic_func(new_node.i, new_node.j, goal_i, goal_j)
                new_node.f = new_node.g  + new_node.h
                new_node.parent = cur
                ast.add_to_open(new_node)
        steps += 1
        if steps > steps_max:
            return False, None, steps, len(ast), None, ast.expanded
        
    return False, None, steps, len(ast), None, ast.expanded
