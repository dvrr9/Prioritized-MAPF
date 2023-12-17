from PIL import Image, ImageDraw, ImageOps
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from map import Map
from IPython.display import Image as Img, display
from random import randint


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
