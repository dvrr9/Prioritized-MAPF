from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
import math
import traceback
from draw import draw
from textwrap import dedent
from other import make_path, convert_string_to_cells
from random import randint
from catable import CATable
import os
from map import Map



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





    

    






