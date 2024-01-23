from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from my_read import read_map_from_file, read_tasks_from_file
import time
from map import Map
import traceback
from tqdm import tqdm
from test_paths_corr import check_paths


def test(search_function, scen_path, map_path, min_agent_cnt, max_agent_cnt, agent_step, scen_num, time_threshold, *args) -> Dict:
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
        "time": [],
        "corr_w_time_threshold": []
    }

    cells = read_map_from_file(map_path)
    task_map = Map(cells)
    for agent_num in tqdm(range(min_agent_cnt, max_agent_cnt, agent_step)):
        tmp_len = []
        tmp_corr = []
        tmp_time = []
        tmp_corr_w_time_threshold = []
        for scen_num in range(1, scen_num + 1):
            tasks = read_tasks_from_file(scen_path + f"-{scen_num}.scen")
            starts = tasks[:agent_num, [0, 1]].astype(int).tolist()
            goals = tasks[:agent_num, [2, 3]].astype(int).tolist()

            starts = [tuple(start) for start in starts]
            goals = [tuple(goal) for goal in goals]

            try:
                start = time.time()
                plan = search_function(starts, goals, task_map, *args)
                end = time.time()
                is_corr_path = (not plan is None) and check_paths(starts, goals, plan)

                if is_corr_path:
                    path_length = [len(path) for path in plan]
                    tmp_len.append(sum(path_length))
                    tmp_corr.append(is_corr_path)
                    tmp_time.append(end - start)
                    tmp_corr_w_time_threshold.append((end - start) < time_threshold)
                else:
                    tmp_len.append(0)
                    tmp_corr.append(is_corr_path)
                    tmp_corr_w_time_threshold.append(is_corr_path)
                    tmp_time.append(end - start)

            except Exception as e:
                print(f"Execution error: {e}")
                traceback.print_exc()
        stat["len"].append(sum(tmp_len) / max(1, sum(tmp_corr)))
        stat["corr"].append(sum(tmp_corr) / scen_num)
        stat["corr_w_time_threshold"].append(sum(tmp_corr_w_time_threshold) / scen_num)
        stat["time"].append(sum(tmp_time) / scen_num)

    return stat
