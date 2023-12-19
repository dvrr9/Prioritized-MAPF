from test import test 
from highLevelSearch import PBS
from lowLevelSearch import astar_timesteps, manhattan_distance
from searchTreePQD import SearchTreePQD
print(
test(
    PBS,
    "../scens/scen-even-32/random-32-32-20-even",
    "../maps/random-32-32-20.map",
    astar_timesteps,
    1000000,
    manhattan_distance,
    SearchTreePQD
)
)