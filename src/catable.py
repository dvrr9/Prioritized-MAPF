import math

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

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
        self._trajectories = []

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
        self._trajectories.append(trajectory)
        for t, (i, j) in enumerate(trajectory):
            if (i, j, t) in self._pos_time_table:
                self._pos_time_table[(i, j, t)].add(traj_id)
            else:
                self._pos_time_table[(i, j, t)] = {traj_id}

            self._last_visit_table[(i, j)] = max(t, self._last_visit_table.get((i, j), 0))

        i_last, j_last = trajectory[-1]

        t_last = len(trajectory) - 1
        if (i_last, j_last) in self._max_time_table:
            raise Exception('two agent end in the same spot')
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
        if (i2, j2, t_start+1) in self._pos_time_table or self._max_time_table.get((i2, j2), t_start + 2) <= t_start + 1:
            return False
        
        #edge
        if  self._pos_time_table.get((i1, j1, t_start+1), set()).intersection(
            self._pos_time_table.get((i2, j2, t_start), set())
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
        