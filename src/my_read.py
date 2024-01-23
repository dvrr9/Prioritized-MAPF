import numpy.typing as npt
from other import convert_string_to_cells
import numpy as np


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
    return np.array(tasks)