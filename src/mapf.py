from highLevelSearch import PP, GPBS, PBS
from lowLevelSearch import sipp, manhattan_distance
from my_read import read_map_from_file, read_tasks_from_file
from searchTreePQD import SearchTreePQD
from map import Map

import argparse
import numpy as np

 
parser = argparse.ArgumentParser()

parser.add_argument("map_file_path")
parser.add_argument("scens_file_path")
parser.add_argument("algo")
parser.add_argument("-o", "--output_file_name")
 
args = parser.parse_args()
 
algos = {
    "GPBS": GPBS,
    "PBS": PBS,
    "PP": PP
}

algo = algos[args.algo]

tasks = read_tasks_from_file(args.scens_file_path)

starts = tasks[:, [0, 1]].astype(int).tolist()
goals = tasks[:, [2, 3]].astype(int).tolist()

starts = [tuple(start) for start in starts]
goals = [tuple(goal) for goal in goals]

cells = read_map_from_file(path=args.map_file_path)

task_map = Map(cells)

paths = algo(
    starts,
    goals,
    task_map,
    sipp,
    np.inf,
    manhattan_distance,
    SearchTreePQD
)
if args.output_file_name:
    with open(args.output_file_name, 'w') as file:
        for path in paths:
            file.write("%s\n" % path)


