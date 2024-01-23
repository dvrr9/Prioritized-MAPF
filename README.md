# Prioritized Multi-Agent Path Finding

## Description
Multi-Agent Path Finding (MAPF) problem is a common problem in different areas (e.g. automated storages, unmanned vehicles etc).
One way to solve it is Prioritezed MAPF algorithms. The main principle of those algorithms is to add agents one by one according to their priority.
These algorithms usually consists of 2-level.
- Low level reprsents search of path in environment with dynamic obstacles
- High level reprsents DFS-like algorithm on special Nodes with some additional heuristics

In this work we implemented **PBS** and **Greedy PBS (GPBS)** algorithms and compared them with baseline solution (**Prioritized Planning** algorithm)

For **GPBS** algorithm we implemented following heuristics:
- Partial Expansion
- Target Reasoning
- Implicit Constraints

On the low-level we use **SIPP** algorithm

## Usage
First, you need to clone the repository:
```
git clone https://github.com/dvrr9/Prioritized-MAPF
```
After that you can run python script from ```/src``` directory:
```
python mapf.py <map_file_path> <scens_file_path> <algo_name> [output_file_name]
```
About arguments:
> <map_file_path> - path to file with map in format from https://movingai.com/
> 
> <scens_file_path> - scenarios for agent in format from https://movingai.com/
> 
> <algo_name> - one of algorithms from: ["PP", "PBS", "GPBS"]
> 
> --output_file_name - (optional) filename to save paths constructed for agents

## References
[1] Čáp, M., Novák, P., Kleiner, A. and Selecký, M., 2015. Prioritized planning algorithms for trajectory coordination of multiple mobile robots. IEEE transactions on automation science and engineering, 12(3), pp.835-849.
https://arxiv.org/pdf/1409.2399

[2] Ma, H., Harabor, D., Stuckey, P.J., Li, J. and Koenig, S., 2019, July. Searching with consistent prioritization for multi-agent path finding. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence (pp. 7643-7650).
https://ojs.aaai.org/index.php/AAAI/article/download/4758/4636

[3] Chan, S.H., Stern, R., Felner, A. and Koenig, S., 2023, July. Greedy Priority-Based Search for Suboptimal Multi-Agent Path Finding. In Proceedings of the 16th International Symposium on Combinatorial Search, SoCS 2023 (pp. 11-19).
https://ojs.aaai.org/index.php/SOCS/article/view/27278/27051
