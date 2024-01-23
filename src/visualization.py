from textwrap import dedent 
from other import convert_string_to_cells
from map import Map
from highLevelSearch import GPBS
from lowLevelSearch import astar_timesteps, manhattan_distance, SearchTreePQD
from draw import draw


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

starts = [(1, 28), (2, 29), (3, 20), (0, 0)]
goals = [(0, 1), (6, 2), (5, 6), (4, 23)]
paths = GPBS(
    starts,
    goals,
    get_map_data(),
    astar_timesteps,
    10000,
    manhattan_distance,
    SearchTreePQD
)


draw(
    grid_map=get_map_data(),
    starts=starts,
    goals=goals,
    paths=paths,
    output_filename= "animated_trajectories",
)