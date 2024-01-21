import math
import os
import traceback
import time
import random
import warnings
from pathlib import Path
from heapq import heappop, heappush
from random import randint, shuffle
from sys import float_info
from textwrap import dedent
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from IPython.display import Image as Img, display
from PIL import Image, ImageDraw, ImageOps
from ipywidgets import IntProgress

# if you don't have joblib installed, please run `pip install` first; we will need this library only for testing purposes
# !pip install joblib
from joblib import Parallel, delayed

EPS = float_info.epsilon


class Map:
    """
    Represents a square grid environment for our moving agent.

    Attributes
    ----------
    _width : int
        The number of columns in the grid.

    _height : int
        The number of rows in the grid.

    _cells : np.ndarray
        A binary matrix representing the grid where 0 represents a traversable cell, and 1 represents a blocked cell.
    """

    def __init__(self, cells: npt.NDArray):
        """
        Initializes the map using a 2D array of cells.

        Parameters
        ----------
        cells : np.ndarray
            A binary matrix representing the grid. 0 indicates a traversable cell, and 1 indicates a blocked cell.
        """
        self._width = cells.shape[1]
        self._height = cells.shape[0]
        self._cells = cells

    def in_bounds(self, i: int, j: int) -> bool:
        """
        Checks if the cell (i, j) is within the grid boundaries.

        Parameters
        ----------
        i : int
            Row number of the cell in the grid.
        j : int
            Column number of the cell in the grid.

        Returns
        ----------
        bool
            True if the cell is inside the grid, False otherwise.
        """
        return 0 <= j < self._width and 0 <= i < self._height

    def traversable(self, i: int, j: int) -> bool:
        """
        Checks if the cell (i, j) is not an obstacle.

        Parameters
        ----------
        i : int
            Row number of the cell in the grid.
        j : int
            Column number of the cell in the grid.

        Returns
        ----------
        bool
            True if the cell is traversable, False if it's blocked.
        """
        return not self._cells[i, j]

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Gets a list of neighboring cells as (i, j) tuples.
        Assumes that the grid is 4-connected, allowing moves only in cardinal directions.

        Parameters
        ----------
        i : int
            Row number of the cell in the grid.
        j : int
            Column number of the cell in the grid.

        Returns
        ----------
        neighbors : List[Tuple[int, int]]
            List of neighboring cells.
        """
        neighbors = []
        delta = ((0, 1), (1, 0), (0, -1), (-1, 0))
        for dx, dy in delta:
            ni, nj = i + dx, j + dy
            if self.in_bounds(ni, nj) and self.traversable(ni, nj):
                neighbors.append((ni, nj))
        return neighbors

    def get_size(self) -> Tuple[int, int]:
        """
        Returns the size of the grid in cells.

        Returns
        ----------
        (height, width) : Tuple[int, int]
            Number of rows and columns in the grid.
        """
        return self._height, self._width
    
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

class Node:
    """
    Represents a search node.

    Attributes
    ----------
    i : int
        Row coordinate of the corresponding grid element.
    j : int
        Column coordinate of the corresponding grid element.
    g : float | int
        g-value of the node (also equals time moment when the agent reaches the cell).
    h : float | int
        h-value of the node (always 0 for Dijkstra).
    f : float | int
        f-value of the node (always equal to g-value for Dijkstra).
    parent : Node
        Pointer to the parent node.
    """

    def __init__(
        self,
        i: int,
        j: int,
        t: int,
        g: Union[float, int] = 0,
        h: Union[float, int] = 0,
        f: Optional[Union[float, int]] = None,
        parent: "Node" = None,
    ):
        """
        Initializes a search node.

        Parameters
        ----------
        i : int
            Row coordinate of the corresponding grid element.
        j : int
            Column coordinate of the corresponding grid element.
        g : float | int
            g-value of the node (also equals time moment when the agent reaches the cell).
        h : float | int
            h-value of the node (always 0 for Dijkstra).
        f : float | int
            f-value of the node (always equal to g-value for Dijkstra).
        parent : Node
            Pointer to the parent node.
        """
        self.i = i
        self.j = j
        self.g = g
        self.h = h
        self.t = t
        if f is None:
            self.f = self.g + h
        else:
            self.f = f
        self.parent = parent

    def __eq__(self, other):
        """
        Checks if two search nodes are the same, which is needed to detect duplicates in the search tree.
        """
        return self.i == other.i and self.j == other.j

    def __hash__(self):
        """
        Makes the Node object hashable, allowing it to be used in sets/dictionaries.
        """
        
        return hash((self.i, self.j, self.t))

    def __lt__(self, other):
        """
        Compares the keys (i.e., the f-values) of two nodes, needed for sorting/extracting the best element from OPEN.
        """
        
        return self.f < other.f


class SearchTreePQD:
    """
    SearchTree using a priority queue for OPEN and a dictionary for CLOSED.
    """

    def __init__(self):
        self._open = []  # Priority queue for nodes in OPEN
        self._closed = {}  # Dictionary for nodes in CLOSED (expanded nodes)
        self._enc_open_dublicates = 0  # Number of dublicates encountered in OPEN

    def __len__(self) -> int:
        """
        Returns the size of the search tree. Useful for assessing the memory
        footprint of the algorithm, especially at the final iteration.
        """
        # YOUR CODE HERE
        
        return len(self._open) + len(self._closed)

    def open_is_empty(self) -> bool:
        """
        Checks if OPEN is empty.
        If true, the main search loop should be interrupted.
        """
        # YOUR CODE HERE
        
        return len(self._open) == 0

    def add_to_open(self, item: Node):
        """
        Adds a node to the search tree, specifically to OPEN. This node is either
        entirely new or a duplicate of an existing node in OPEN.
        This implementation detects duplicates lazily; thus, nodes are added to
        OPEN without initial duplicate checks.
        """
        # YOUR CODE HERE
        
        heappush(self._open, item)

    def get_best_node_from_open(self) -> Optional[Node]:
        """
        Retrieves the best node from OPEN, defined by the minimum key.
        This node will then be expanded in the main search loop.

        Duplicates are managed here. If a node has been expanded previously
        (and is in CLOSED), it's skipped and the next best node is considered.

        Returns None if OPEN is empty.
        """
        # YOUR CODE HERE
        
        while not self.open_is_empty():
            item = heappop(self._open)
            if not item in self._closed:
                return item
            self._enc_open_dublicates += 1
        return None

    def add_to_closed(self, item: Node):
        """
        Adds a node to the CLOSED dictionary.
        """
        # YOUR CODE HERE
        
        self._closed[item] = item

    def was_expanded(self, item: Node) -> bool:
        """
        Checks if a node has been previously expanded.
        """
        # YOUR CODE HERE
        return item in self._closed

    @property
    def opened(self):
        return self._open

    @property
    def expanded(self):
        return self._closed.values()

    @property
    def number_of_open_dublicates(self):
        return self._enc_open_dublicates
    
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

class CATable:
    """
    Implements a collision avoidance table for effectively checking
    collisions with dynamic obstacles.

    Attributes
    ----------
    _pos_time_table : Dict[Tuple[int, int, int], int]
        A table that checks if a cell (i, j) is occupied at time t.
        If the cell is occupied, the key (i, j, t) corresponds to a value
        equal to the ID of the trajectory passing through cell (i, j) at time t.

    _max_time_table : Dict[Tuple[int, int], int]
        This table stores information about the time t from which cell (i, j)
        will be permanently occupied. This is crucial to avoid collisions if
        the cell (i, j) is the final position of any trajectory. In such cases,
        starting from time t, it becomes impossible to pass through the cell.
        The key (i, j) corresponds to the time t equal to the duration of the
        trajectory that ends in this cell.

    _last_visit_table : Dict[Tuple[int, int], int | float]
        A table that stores information about the last moment when the cell (i, j)
        was occupied. This information is necessary to ensure that an agent remaining
        in its goal position (i, j) at time t1 does not collide with a dynamic obstacle
        at time t2 > t1.
    """

    def __init__(self):
        self._pos_time_table = dict()
        self._max_time_table = dict()
        self._last_visit_table = dict()

    def add_trajectory(self, traj_id: int, trajectory: List[Tuple[int, int]]):
        """
        Adds a trajectory to the collision avoidance table. Each element of the trajectory
        represents the position at a specific time moment. For example, trajectory[0] is the
        position at time t = 0, trajectory[1] is at time t = 1, and so on.

        Parameters
        ----------
        traj_id : int
            Unique identifier for the trajectory.
        trajectory : List[Tuple[int, int]]
            The trajectory represented as a sequence of grid positions (i, j).
        """
        
        for t, (i, j) in enumerate(trajectory):
            self._pos_time_table[(i, j, t)] = traj_id
            self._last_visit_table[(i, j)] = max(t, self._last_visit_table.get((i, j), 0))

        i_last, j_last = trajectory[-1]

        t_last = len(trajectory) - 1
        
        self._max_time_table[(i_last, j_last)] = t_last



    def check_move(self, i1: int, j1: int, i2: int, j2: int, t_start: int) -> bool:
        """
        Checks if the move from cell (i1, j1) to cell (i2, j2)
        between time moments t_start and t_start+1 will not result in
        a collision with other trajectories.

        Parameters
        ----------
        i1, j1 : int
            Coordinates of the starting cell on the grid map.
        i2, j2 : int
            Coordinates of the target cell on the grid map.
        t_start : int
            The time moment when the move begins.

        Returns
        -------
        bool
            True if the move is permissible without causing a collision, False otherwise.
        """
   

        #vertex

      

        if (i2, j2, t_start+1) in self._pos_time_table or self._max_time_table.get((i2, j2), t_start + 2) < t_start + 1:
            return False
        

        #edge

        if  ((i1, j1, t_start+1) in self._pos_time_table and 
            (i2, j2, t_start) in self._pos_time_table and 
            self._pos_time_table[(i1, j1, t_start+1)] == self._pos_time_table[(i2, j2, t_start)]
            ):
            return False
        
        return True


    def last_visited(self, i: int, j: int):
        """
        Returns the last time moment 't' when the cell (i, j) was occupied.
        If the cell has never been occupied, -1 is returned. If the cell becomes
        permanently occupied starting from a certain time moment, math.inf is returned.

        Parameters
        ----------
        i, j : int
            Coordinates of the cell on the grid map.

        Returns
        -------
        int | float
            The time 't' when the cell (i, j) was last occupied, -1 if never occupied,
            or math.inf if permanently occupied from a certain time moment.
        """
        if (i, j) in self._max_time_table:
            return math.inf
        
        return self._last_visit_table.get((i, j), -1)
        

    def __is_cell_available(self, i: int, j: int, t: int) -> bool:
        """
        Checks if a cell (i, j) is available at the specified time moment t.

        Parameters
        ----------
        i, j : int
            Coordinates of the cell on the grid map.
        t : int
            The time moment for which availability is checked.

        Returns
        -------
        bool
            True if the cell is not occupied at time t, False otherwise.
        """
        # YOUR CODE HERE
        
        
        if (i, j, t) in self._pos_time_table:
            return False
        
        if self._max_time_table.get((i, j), t+1) < t:
            return False
        
        return True

    def __is_reverse_move_valid(
        self, i1: int, j1: int, i2: int, j2: int, t_start: int, t_end: int
    ) -> bool:
        """
        Checks whether there is no concurrent move in the reverse direction along the same
        edge between the specified time moments t_start and t_end.

        Parameters
        ----------
        i1, j1 : int
            Coordinates of the starting cell for the move on the grid map.
        i2, j2 : int
            Coordinates of the target cell for the move on the grid map.
        t_start : int
            Time moment when the move starts.
        t_end : int
            Time moment when the move ends.

        Returns
        -------
        bool
            True if there is no reverse move along the same edge within the given time frame,
            False if there is a reverse move.
        """
        # YOUR CODE HERE
        if  ((i1, j1, t_end) in self._pos_time_table and 
            (i2, j2, t_start) in self._pos_time_table and 
            self._pos_time_table[(i1, j1, t_end)] == self._pos_time_table[(i2, j2, t_start)]
            ):
            return False
        
        return True
        
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

def draw_grid(draw_obj: ImageDraw, grid_map: Map, scale: Union[float, int]):
    """
    Draws static obstacles on the grid using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    grid_map : Map
        The grid map containing obstacle information.
    scale : float or int
        The scaling factor for drawing.
    """
    height, width = grid_map.get_size()
    for row in range(height):
        for col in range(width):
            if not grid_map.traversable(row, col):
                top_left = (col * scale, row * scale)
                bottom_right = ((col + 1) * scale - 1, (row + 1) * scale - 1)
                draw_obj.rectangle(
                    [top_left, bottom_right], fill=(70, 80, 80), width=0.0
                )


def draw_start_goal(
    draw_obj: ImageDraw,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    scale: Union[float, int],
):
    """
    Draws start and goal cells on the grid using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    start : Tuple[int, int]
        The start cell coordinates.
    goal : Tuple[int, int]
        The goal cell coordinates.
    scale : float or int
        The scaling factor for drawing.
    """

    def draw_cell(cell, fill_color):
        top_left = ((cell[1] + 0.1) * scale, (cell[0] + 0.1) * scale)
        bottom_right = ((cell[1] + 0.9) * scale - 1, (cell[0] + 0.9) * scale - 1)
        draw_obj.rounded_rectangle(
            [top_left, bottom_right], fill=fill_color, width=0.0, radius=scale * 0.22
        )

    draw_cell(start, (40, 180, 99))  # Start cell color
    draw_cell(goal, (231, 76, 60))  # Goal cell color


def draw_dyn_object(
    draw_obj: ImageDraw,
    path: List[Tuple[int, int]],
    step: int,
    frame_num: int,
    frames_per_step: int,
    scale: Union[float, int],
    color: Tuple[int, int, int],
    circle: bool,
):
    """
    Draws the position of a dynamic object at a specific time using draw_obj.

    Parameters
    ----------
    draw_obj : ImageDraw
        The drawing object to use.
    path : List[Tuple[int, int]]
        The path of the dynamic object.
    step : int
        The current step in the path.
    frame_num : int
        The current frame number.
    frames_per_step : int
        The number of frames per step.
    scale : float or int
        The scaling factor for drawing.
    color : Tuple[int, int, int]
        The fill color for the object.
    circle : bool
        Whether to draw the object as a circle.
    """
    path_len = len(path)
    curr_i, curr_j = path[min(path_len - 1, step)]
    next_i, next_j = path[min(path_len - 1, step + min(frame_num, 1))]

    di = frame_num * (next_i - curr_i) / frames_per_step
    dj = frame_num * (next_j - curr_j) / frames_per_step

    top_left = (float(curr_j + dj + 0.25) * scale, float(curr_i + di + 0.25) * scale)
    bottom_right = (
        float(curr_j + dj + 0.75) * scale - 1,
        float(curr_i + di + 0.75) * scale - 1,
    )
    if circle:
        draw_obj.ellipse([top_left, bottom_right], fill=color)
    else:
        draw_obj.rectangle([top_left, bottom_right], fill=color)

def create_frame(
    grid_map,
    scale,
    width,
    height,
    step,
    quality,
    starts,
    goals,
    paths,
    agent_colors,
    dyn_obst_traj,
    dyn_obst_color,
):
    frames = []
    for n in range(quality):
        im = Image.new("RGB", (width * scale, height * scale), color=(234, 237, 237))
        draw = ImageDraw.Draw(im)
        draw_grid(draw, grid_map, scale)

        if starts and goals:
            for start, goal in zip(starts, goals):
                draw_start_goal(draw, start, goal, scale)

        if paths:
            for path, agent_color in zip(paths, agent_colors):
                draw_dyn_object(
                    draw, path, step, n, quality, scale, agent_color, True
                )

        if dyn_obst_traj:
            for dyn_obst in dyn_obst_traj:
                draw_dyn_object(
                    draw,
                    dyn_obst,
                    step,
                    n,
                    quality,
                    scale,
                    dyn_obst_color,
                    False,
                )

        im = ImageOps.expand(im, border=2, fill="black")
        frames.append(im)
    return frames


def save_animation(images, output_filename, quality):
    images[0].save(
        f"{output_filename}.png",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=500 / quality,
        loop=0,
    )
    display(Img(filename=f"{output_filename}.png"))


def draw(
    grid_map: Map,
    starts: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    paths: Optional[List[List[Tuple[int, int]]]] = None,
    dyn_obst_traj: Optional[List[List[Tuple[int, int]]]] = None,
    output_filename: str = "animated_trajectories",
):
    """
    Visualizes the environment, agent paths, and dynamic obstacles trajectories.

    Parameters
    ----------
    grid_map : Map
        Environment representation as a grid.
    starts : List[Tuple[int, int]] | None
        Starting positions of agents.
    goals : List[Tuple[int, int]] | None
        Goal positions of agents.
    paths : List[List[Tuple[int, int]]] | None
        Paths of agents between start and goal positions.
    dyn_obst_traj : List[List[Tuple[int, int]]] | None
        Trajectories of dynamic obstacles.
    output_filename : str
        Name of the file for the resulting animated visualization.
    """
    scale = 30
    quality = 6
    height, width = grid_map.get_size()

    dyn_obst_color = (70, 80, 80)

    agent_colors = [
        (randint(30, 230), randint(30, 230), randint(30, 230)) for _ in starts or []
    ]
    max_time = max((len(path) for path in paths), default=1) if paths else 1
    images = []

    for step in range(max_time):
        images.extend(
            create_frame(
                grid_map,
                scale,
                width,
                height,
                step,
                quality,
                starts,
                goals,
                paths,
                agent_colors,
                dyn_obst_traj,
                dyn_obst_color
            )
        )

    save_animation(images, output_filename, quality)

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
    path = []
    while current.parent:
        path.append((current.i, current.j))
        current = current.parent
    path.append((current.i, current.j))
    return path[::-1], duration

def read_lists_from_file(path: str) -> List[List[Tuple[int, int]]]:
    """
    Reads data from a file and returns it in the form of a list of lists of tuples.

    Each line in the file should contain pairs of integers. Single integer lines
    signify the end of a current list of tuples.

    Parameters
    ----------
    path : str
        Path to the file to be read.

    Returns
    -------
    List[List[Tuple[int, int]]]
        A list of lists, where each sublist contains tuples of integers.
    """
    with open(path) as file:
        main_list = []
        curr_list = []
        for line in file:
            if not line.strip():
                continue
            nums = tuple(map(int, line.split()))

            if len(nums) == 1:
                if curr_list:
                    main_list.append(curr_list)
                    curr_list = []
            else:
                curr_list.append(nums)

        if curr_list:
            main_list.append(curr_list)

        return main_list
    
def check_start_goal(
    start: Tuple[int, int], goal: Tuple[int, int], trajectory: List[Tuple[int, int]]
) -> bool:
    """
    Checks whether the given trajectory begins at the start cell and ends at the goal cell.

    Parameters
    ----------
    start : Tuple[int, int]
        The starting cell coordinates.
    goal : Tuple[int, int]
        The goal cell coordinates.
    trajectory : List[Tuple[int, int]]
        The trajectory as a list of cell coordinates.

    Returns
    -------
    bool
        True if the trajectory starts at the start cell and ends at the goal cell, False otherwise.
    """
    return bool(trajectory) and trajectory[0] == start and trajectory[-1] == goal


def process_trajectory(
    traj_id: int,
    trajectory: List[Tuple[int, int]],
    pos_time: Dict,
    max_times: Dict,
    last_times: Dict,
):
    """
    Processes a single trajectory for collision checking by updating the given dictionaries
    with trajectory information.

    Parameters
    ----------
    traj_id : int
        The trajectory identifier.
    trajectory : List[Tuple[int, int]]
        The trajectory as a list of cell coordinates.
    pos_time : Dict
        Dictionary for storing positional time information.
    max_times : Dict
        Dictionary for storing maximum time information.
    last_times : Dict
        Dictionary for storing last time information.
    """
    for t, (i, j) in enumerate(trajectory):
        pos_time[(i, j, t)] = traj_id
        last_times[(i, j)] = max(last_times.get((i, j), -1), t)

    last_times[trajectory[-1]] = math.inf
    max_times[trajectory[-1]] = len(trajectory) - 1


def process_dyn_obstacles(
    dyn_obst_traj: List[List[Tuple[int, int]]]
) -> Tuple[Dict, Dict, Dict]:
    """
    Processes dynamic obstacles' trajectories for collision checking.

    Parameters
    ----------
    dyn_obst_traj : List[List[Tuple[int, int]]]
        List of dynamic obstacles' trajectories.

    Returns
    -------
    Tuple[Dict, Dict, Dict]
        Three dictionaries containing positional time, maximum time, and last time information.
    """
    pos_time, max_times, last_times = dict(), dict(), dict()

    for traj_id, trajectory in enumerate(dyn_obst_traj):
        process_trajectory(traj_id, trajectory, pos_time, max_times, last_times)

    return pos_time, max_times, last_times


def check_collisions(
    trajectory: List[Tuple[int, int]], pos_time: Dict, max_times: Dict, last_times: Dict
) -> bool:
    """
    Checks for collisions in the given trajectory with other trajectories.

    Parameters
    ----------
    trajectory : List[Tuple[int, int]]
        The trajectory to check for collisions.
    pos_time : Dict
        Dictionary containing positional time information of other trajectories.
    max_times : Dict
        Dictionary containing maximum time information.
    last_times : Dict
        Dictionary containing last time information.

    Returns
    -------
    bool
        True if no collisions are found, False otherwise.
    """
    for t1 in range(len(trajectory) - 1):
        i1, j1 = trajectory[t1]
        t2 = t1 + 1
        i2, j2 = trajectory[t2]

        if (i2, j2, t2) in pos_time or (
            (i2, j2) in max_times and max_times[(i2, j2)] <= t2
        ):
            return False
        if (
            (i1, j1, t2) in pos_time
            and (i2, j2, t1) in pos_time
            and pos_time[(i1, j1, t2)] == pos_time[(i2, j2, t1)]
        ):
            return False
    if len(trajectory) - 1 <= last_times.get(trajectory[-1], -1):
        return False
    return True


def check_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[Tuple[int, int]],
    dyn_obst_traj: List[List[Tuple[int, int]]],
) -> bool:
    """
    Verifies whether the provided solution is correct and does not collide with the
    trajectories of dynamic obstacles.

    Parameters
    ----------
    start, goal : Tuple[int, int]
        The start and goal cells for the agent.
    path : List[Tuple[int, int]]
        The sequence of cells representing the path between the start and goal positions.
    dyn_obst_traj : List[List[Tuple[int, int]]]
        A list of sequences of cells, representing the trajectories of dynamic obstacles.

    Returns
    -------
    bool
        True if the path is correct and does not collide with dynamic obstacles, False otherwise.
    """

    return check_start_goal(start, goal, path) and check_collisions(
        path, *process_dyn_obstacles(dyn_obst_traj)
    )

def read_map_from_file(path: str) -> npt.NDArray:
    with open(path) as map_file:
        type = next(map_file).split(' ')[-1]
        height = int(next(map_file).split(' ')[-1])
        width = int(next(map_file).split(' ')[-1])
        next(map_file)
        # Read the map section
        map_lines = [next(map_file) for _ in range(height)]
        map_str = "".join(map_lines)
        cells = convert_string_to_cells(map_str)
    return cells


def simple_test(search_function: Callable, task: Union[int, None], *args):
    """
    Function `simple_test` runs `search_function` on one task. Use a number from 0 to 24 to choose a specific debug task
    on a simple map, or use None to select a random task from this pool. The function displays:
     - 'Path found!' and some statistics if a path was found.
     - 'Path not found!' if a path was not discovered.
     - 'Execution error' if an error occurred during the execution of the search_function.
    In the first case, the function also provides a visualization of the task.

    Parameters
    ----------
    search_function : Callable
        Implementation of the search method.
    task : int | None
        A number from 0 to 24 to choose a specific debug task on a simple map,
        or None to select a random task from this pool.
    *args
        Additional arguments passed to the search function.
    """

    def get_map_data():
        map_str = dedent(
            """
            . . . # # . . . . . . . . # # . . . # . . # # . . . . . . .  
            . . . # # # # # . . # . . # # . . . . . . # # . . . . . . . 
            . . . . . . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
            . . . . . . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . . . . # . . . . . . . # . . . . . . . . . . . 
            . . . # # # # # # # # # # # # # . # # . # # # # # # # . # # 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
        """
        )
        cells = convert_string_to_cells(map_str)
        return Map(cells)

    task_map = get_map_data()

    starts = [(1, 28), (2, 29), (3, 20), (3, 20), (0, 0)]
    goals = [(0, 1), (6, 2), (5, 6), (13, 0), (4, 23)]
    dyn_obst_traj = read_lists_from_file(os.path.join("data", "dyn_obst_traj.txt"))

    ca_table = CATable()
    for traj_id, traj in enumerate(dyn_obst_traj):
        ca_table.add_trajectory(traj_id, traj)

    durations = [54, 47, 48, 71, 56]

    if (task is None) or not (0 <= task < 5):
        task = randint(0, 4)

    start = starts[task]
    goal = goals[task]
    true_duration = durations[task]
    try:
        (
            found,
            end_node,
            number_of_steps,
            nodes_created,
            *other_results,
        ) = search_function(
            task_map, ca_table, start[0], start[1], goal[0], goal[1], *args
        )
        if found:
            path, duration = make_path(end_node)
            correct = check_path(start, goal, path, dyn_obst_traj) and int(
                duration
            ) == int(true_duration)
            print(
                f"Path found! Duartion: {duration}. Nodes created {nodes_created}. Number of steps: {number_of_steps}. Correct: {correct}"
            )
            draw(task_map, [start], [goal], [path], dyn_obst_traj)

        else:
            print("Path not found!")
    except Exception as e:
        print(f"Execution error: {e}")
        traceback.print_exc()


def update_plan_for_agent(start_i, start_j, goal_i, goal_j, trajectories, task_map, search_function, *args):

    ca_table = CATable()
    for traj_id, traj in enumerate(trajectories):
        ca_table.add_trajectory(traj_id, traj)
  
    (
        found,
        end_node,
        number_of_steps,
        nodes_created,
        *other_results,
    ) = search_function(
        task_map, ca_table, start_i, start_j, goal_i, goal_j, *args
    )

    if found:
        return make_path(end_node)[0]
    else:
        return None


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

    def find_collision(self) -> Optional[Node]: #TODO: check for those without constraint
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
    
def PP(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    is_find = True

    for _ in range(10):

        priorities = [i for i in range(n_agents)]
        random.shuffle(priorities)
        paths = []
        for agent in priorities:
            start_i, start_j = starts[agent]
            goal_i, goal_j = goals[agent]

            new_path = update_plan_for_agent(start_i, start_j, goal_i, goal_j, paths, task_map, search_function, *args)
            if new_path:
                paths.append(new_path)
            else:
                is_find = False
                break
        if is_find:
            paths_w_priorities = list(zip(priorities, paths))
            paths_w_priorities.sort()
            return [path_w_priority[1] for path_w_priority in paths_w_priorities]
    return None
    
class Priorities:
    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self.priorities = []
        self.lh_edges = [set() for i in range(n_agents)] # edges from lower to higher priority agents
        self.hl_edges = [set() for i in range(n_agents)] # edges from higher to lower priority agents

    def add_priority(self, lower: int, higher: int):
        self.lh_edges[lower].add(higher)
        self.hl_edges[higher].add(lower)
        self.priorities.append((lower, higher))

    def remove_priority(self, lower: int, higher: int):
        self.lh_edges[lower].remove(higher)
        self.hl_edges[higher].remove(lower)
        self.priorities.pop()

    def get_last_conflict(self):
        if len(self.priorities) == 0:
            raise ValueError("No conflicts found")
        return self.priorities[-1]

    def has_edge(self, lower: int, higher: int) -> bool:
        return higher in self.lh_edges[lower]

    def get_lower_priority_agents(self, agent: int) -> list[int]:
        lower_priority_agents = set()
        for new_agent in self.hl_edges[agent]:
            lower_priority_agents.add(new_agent)
            lower_priority_agents.update(self.get_lower_priority_agents(new_agent))
        return lower_priority_agents

    def get_higher_priority_agents(self, agent: int) -> list[int]:
        higher_priority_agents = set()
        for new_agent in self.lh_edges[agent]:
            higher_priority_agents.add(new_agent)
            higher_priority_agents.update(self.get_higher_priority_agents(new_agent))
        return higher_priority_agents

def PBS(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    root = PTNode(plan=[])
    for start, goal in zip(starts, goals):
        root.plan.append(update_plan_for_agent(*start, *goal, [], task_map, search_function, *args))
    priorities = []
    priority_matrix = [[-1 for i in range(n_agents)] for j in range(n_agents)]
    stack = [root]

    def get_lower_priority_agents(agent: int) -> set[int]:
        lower_priority_agents = set()
        for new_agent in range(n_agents):
            if priority_matrix[agent][new_agent] == 1:
                lower_priority_agents.add(new_agent)
                lower_priority_agents.update(get_lower_priority_agents(new_agent))
        return lower_priority_agents
    
    def get_higher_priority_agents(agent: int) -> set[int]:
        higher_priority_agents = set()
        for new_agent in range(n_agents):
            if priority_matrix[agent][new_agent] == 0:
                higher_priority_agents.add(new_agent)
                higher_priority_agents.update(get_higher_priority_agents(new_agent))
        return higher_priority_agents

    def topsort(agents: list[int]) -> list[int]:
        visited = [False for i in range(len(agents))]
        topsort_order = []
        def dfs(u):
            visited[u] = True
            for v in range(len(agents)):
                a1 = agents[u]
                a2 = agents[v]
                if priority_matrix[a1][a2] == 1 and not visited[v]:
                    dfs(v)
            topsort_order.append(agents[u])
        
        dfs(len(agents) - 1)
        return reversed(topsort_order)

    def update_plan(node: PTNode, agent: int) -> bool:
        agents = list(get_lower_priority_agents(agent))
        agents.append(agent)
        agents_to_update = topsort(agents)
        for curr_agent in agents_to_update:
            hp_agents = get_higher_priority_agents(curr_agent)
            trajectories = [node.plan[hp_agent] for hp_agent in hp_agents]
            new_path = update_plan_for_agent(
                root.plan[curr_agent][0][0],
                root.plan[curr_agent][0][1],
                root.plan[curr_agent][-1][0],
                root.plan[curr_agent][-1][1],
                trajectories,
                task_map,
                search_function,
                *args
            )
            if new_path is None:
                return False
            node.plan[curr_agent] = new_path
        node.update_cost()
        return True

    while len(stack):
        node = stack[-1]
        if node.times_visited:
            stack.pop()
            lp_agent = priorities[-1][0]
            hp_agent = priorities[-1][1]
            priority_matrix[hp_agent][lp_agent] = -1
            priority_matrix[lp_agent][hp_agent] = -1
            priorities.pop()
            continue

        if node.priority is not None:
            priorities.append(node.priority)
            lp_agent = priorities[-1][0]
            hp_agent = priorities[-1][1]
            priority_matrix[hp_agent][lp_agent] = 1
            priority_matrix[lp_agent][hp_agent] = 0

        collision = node.find_collision()
        if collision is None:
            return node.plan
        
        new_nodes = []

        node1 = PTNode(
            parent=node,
            priority=collision,
            plan=node.plan
        )
        node2 = PTNode(
            parent=node,
            priority=tuple(reversed(collision)),
            plan=node.plan
        )

        if update_plan(node1, node1.priority[0]):
            new_nodes.append(node1)        
        if update_plan(node2, node2.priority[0]):
            new_nodes.append(node2)

        new_nodes.sort(key=lambda x: -x.cost)
        stack.extend(new_nodes)
        node.times_visited += 1
    
    return None


def GPBS(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    root = PTNode(plan=[])
    for start, goal in zip(starts, goals):
        root.plan.append(update_plan_for_agent(*start, *goal, [], task_map, search_function, *args))
    priorities = Priorities(n_agents)
    stack = [root]
    last_removed_conflict = (-1, -1) # (lower, higher) last remove conflict agents

    def topsort(agents: list[int]) -> list[int]:
        visited = [False for i in range(len(agents))]
        topsort_order = []
        def dfs(u):
            visited[u] = True
            for v in range(len(agents)):
                a1 = agents[u]
                a2 = agents[v]
                if priorities.has_edge(a2, a1) and not visited[v]:
                    dfs(v)
            topsort_order.append(agents[u])
        
        dfs(len(agents) - 1)
        return reversed(topsort_order)

    def update_plan(node: PTNode, agent: int) -> bool:
        agents = list(priorities.get_lower_priority_agents(agent))
        agents.append(agent)
        agents_to_update = topsort(agents)
        for curr_agent in agents_to_update:
            hp_agents = priorities.get_higher_priority_agents(curr_agent)
            trajectories = [node.plan[hp_agent] for hp_agent in hp_agents]
            new_path = update_plan_for_agent(
                root.plan[curr_agent][0][0],
                root.plan[curr_agent][0][1],
                root.plan[curr_agent][-1][0],
                root.plan[curr_agent][-1][1],
                trajectories,
                task_map,
                search_function,
                *args
            )
            if new_path is None:
                return False
            node.plan[curr_agent] = new_path
        node.update_cost()
        return True

    while len(stack):
        node = stack[-1]
        if node.times_visited == 2:
            stack.pop()
            lower, higher = priorities.get_last_conflict()
            last_removed_conflict = (lower, higher)
            priorities.remove_priority(lower, higher)
            priorities.pop()
        elif node.times_visited == 1:
            collision(tuple(reversed(last_removed_conflict)))
            new_node = PTNode(
                parent=node,
                priority=collision,
                plan=node.plan
            )
            if update_plan(new_node, new_node.priority[0]):
                stack.append(new_node)
        else:
            if node.priority is not None:
                lower, higher = node.priority
                priorities.add_priority(lower, higher)

            collision = node.find_collision()
            if collision is None:
                return node.plan
            new_node = PTNode(
                parent=node,
                priority=collision,
                plan=node.plan
            )
            if update_plan(new_node, new_node.priority[0]):
                stack.append(new_node)
        node.times_visited += 1
    
    return None


def get_map_data():
        map_str = dedent(
            """
            . . . # # . . . . . . . . # # . . . # . . # # . . . . . . .  
            . . . # # # # # . . # . . # # . . . . . . # # . . . . . . . 
            . . . . . . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
            . . . # # . . # . . # . . # # . . . # . . # # # # # . . . . 
            . . . . . . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . # . . # . . # # . . . # . . # . . . . . . . . 
            . . . # # . . . . . # . . . . . . . # . . . . . . . . . . . 
            . . . # # # # # # # # # # # # # . # # . # # # # # # # . # # 
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . . 
            . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            . . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
        """
        )
        cells = convert_string_to_cells(map_str)
        return Map(cells)
def read_tasks_from_file(path: str) -> npt.NDArray:
    tasks = []
    with open(path) as tasks_file:
        next(tasks_file)
        for line in tasks_file:
            values = line.split('\t')[4:]
            tasks.append([int(x) for x in values[:-1]])
            # Swap coordinates because of file format
            tasks[-1][0], tasks[-1][1] = tasks[-1][1], tasks[-1][0]
            tasks[-1][2], tasks[-1][3] = tasks[-1][3], tasks[-1][2]
            tasks[-1].append(float(values[-1]))
    return np.array(sorted(tasks, key=lambda x: x[-1]))



def test(search_function, scen_path, map_path, *args) -> Dict:
    """
    The `massive_test` function runs the `search_function` on a set of different tasks
    (for example, from the directory `data/`) using *args as optional arguments.
    For every task, it displays a short report:
     - 'Path found!' along with some statistics if a path was found.
     - 'Path not found!' if a path wasn't found.
     - 'Execution error' if an error occurred during the execution of the search_function.

    The function returns a dictionary containing statistics with the following keys:
     - "corr" — correctness of each path length (True/False).
     - "len" — the length of each path (0.0 if a path wasn't found).
     - "st_size" — the size of the resultant search tree for each task.
     - "steps" — the number of algorithm steps for each task.

    Parameters
    ----------
    search_function : Callable
        The implemented search method.
    data_path : str
        Path to the directory containing tasks.
    num_of_tasks : int
        The number of tasks to be used for evaluation.

    Returns
    -------
    stat : Dict
        A dictionary containing statistics.

    """
    stat = {
        "corr": [],
        "len": [],
        "time": []
    }
    max_agent_cnt = 30
    tasks = read_tasks_from_file(scen_path)
    cells = read_map_from_file(map_path)
    task_map = Map(cells)
    for agent_num in range(1, max_agent_cnt):
        starts = tasks[:agent_num, [0, 1]].astype(int).tolist()
        goals = tasks[:agent_num, [2, 3]].astype(int).tolist()

        starts = [tuple(start) for start in starts]
        goals = [tuple(goal) for goal in goals]
        

        try:
            start = time.time()
            plan = search_function(starts, goals, task_map, *args)
            end = time.time()

            if plan:
                path_length = [len(path) for path in plan]
           
                stat["len"].append(sum(path_length))
                stat["corr"].append(True)
                stat["time"].append(end - start)
            else:
                print(f"Task: #{agent_num}. Path not found!")
                stat["len"].append(-1)
                stat["corr"].append(False)
                stat["time"].append(end - start)

        except Exception as e:
            print(f"Execution error: {e}")
            traceback.print_exc()

    return stat




