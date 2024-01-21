from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from my_read import read_map_from_file, read_tasks_from_file
import time
from map import Map
import traceback
from tqdm import tqdm

def test(search_function, scen_path, map_path, min_agent_cnt, max_agent_cnt, agent_step, *args) -> Dict:
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
    
    
    cells = read_map_from_file(map_path)
    task_map = Map(cells)
    for agent_num in tqdm(range(min_agent_cnt, max_agent_cnt, agent_step)):
        tmp_len = []
        tmp_corr = []
        tmp_time = []
        for scen_num in range(1, 26):
            tasks = read_tasks_from_file(scen_path + f"-{scen_num}.scen")
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
                    tmp_len.append(sum(path_length))
                    tmp_corr.append(True)
                    tmp_time.append(end - start)
                else:
                    #print(f"Task: #{agent_num}. Path not found!")
                    tmp_len.append(0)
                    tmp_corr.append(False)
                    tmp_time.append(end - start)

            except Exception as e:
                print(f"Execution error: {e}")
                traceback.print_exc()
        stat["len"].append(sum(tmp_len) / sum(tmp_corr))
        stat["corr"].append(sum(tmp_corr) / 25)
        stat["time"].append(sum(tmp_time) / 25)

    return stat