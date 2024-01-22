import random
import copy
from collections import defaultdict

from catable import CATable
from PTnode import PTNode, PTNodeGPBS
from priorities import Priorities
from conflicttype import ConflictType
from other import make_path

from astar import astar


def update_plan_for_agent(start_i, start_j, goal_i, goal_j, trajectories, task_map, search_function, *args):
    ca_table = CATable()
    for traj_id, traj in enumerate(trajectories):
        ca_table.add_trajectory(traj_id, traj)
  
    (
        found,
        end_node,
        number_of_steps,
        nodes_created,
        *other_results,
    ) = search_function(
        task_map, ca_table, start_i, start_j, goal_i, goal_j, *args
    )

    if found:
        return make_path(end_node)[0]
    else:
        return None
    
def PP(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    is_find = True

    for _ in range(10):

        priorities = [i for i in range(n_agents)]
        random.shuffle(priorities)
        paths = []
        for agent in priorities:
            start_i, start_j = starts[agent]
            goal_i, goal_j = goals[agent]

            new_path = update_plan_for_agent(start_i, start_j, goal_i, goal_j, paths, task_map, search_function, *args)
            if new_path:
                paths.append(new_path)
            else:
                is_find = False
                break
        if is_find:
            paths_w_priorities = list(zip(priorities, paths))
            paths_w_priorities.sort()
            return [path_w_priority[1] for path_w_priority in paths_w_priorities]
    return None

def PBS(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    root = PTNode(plan=[])
    for start, goal in zip(starts, goals):
        last_node = astar(
            task_map,
            *start,
            *goal,
            *args
        )[1]
        assert last_node is not None, f"Didn't find path for agent with start={start} and goal={goal}"
        root.plan.append(make_path(last_node)[0])
        # root.plan.append(update_plan_for_agent(*start, *goal, [], task_map, search_function, *args))
    priorities = Priorities(n_agents)
    stack = [root]
    # print("Initial paths for agents are constructed!")

    def topsort(agents: list[int]) -> list[int]:
        visited = [False for i in range(len(agents))]
        topsort_order = []
        def dfs(u):
            visited[u] = True
            for v in range(len(agents)):
                a1 = agents[u]
                a2 = agents[v]
                if priorities.has_edge(lower=a2, higher=a1) and not visited[v]:
                    dfs(v)
            topsort_order.append(agents[u])
        
        dfs(len(agents) - 1)
        return topsort_order[::-1]
    
    def update_plan(node: PTNode, agent: int) -> bool:
        agents = list(priorities.get_lower_priority_agents(agent))
        agents.append(agent)
        agents_to_update = topsort(agents)
        need_update = defaultdict(bool)
        for curr_agent in agents_to_update:
            hp_agents = priorities.get_higher_priority_agents(curr_agent)
            for hp_agent in hp_agents:
                # # print(hp_agent, curr_agent)
                # # print(node.has_conflict(curr_agent, hp_agent))
                if node.has_conflict(curr_agent, hp_agent):
                    need_update[curr_agent] = True
                    break
        were_updated = []
        for curr_agent in agents_to_update:
            if not need_update[curr_agent]:
                continue
            were_updated.append(curr_agent)
            hp_agents = priorities.get_higher_priority_agents(curr_agent)
            # print(f'Update path for agent: {curr_agent} | Higher priority agents: {hp_agents}')
            trajectories = [node.plan[hp_agent] for hp_agent in hp_agents]
            new_path = update_plan_for_agent(
                root.plan[curr_agent][0][0],
                root.plan[curr_agent][0][1],
                root.plan[curr_agent][-1][0],
                root.plan[curr_agent][-1][1],
                trajectories,
                task_map,
                search_function,
                *args
            )
            if new_path is None:
                # print("Path wasn't found")
                return False
            node.plan[curr_agent] = new_path
            for lp_agent in priorities.get_lower_priority_agents(curr_agent):
                if node.has_conflict(curr_agent, lp_agent):
                    need_update[lp_agent] = True
        node.update_cost()
        # print(f"Successful paths update, were updated agents: {were_updated}")
        return True

    while len(stack):
        node = stack[-1]
        if node.times_visited:
            # print("Second visit to PTNode")
            stack.pop()
            if len(stack) == 0: # can't find paths for root (i.e. no solution found)
                continue
            # priorities.remove_last_conflict()
            last_conflict = priorities.remove_last_conflict()
            # print(f"Removing last conflict: {last_conflict}")
            continue
        # print("First visit to PTNode")
        # print(f"CurrNode conflict after update: {node.has_conflict(11, 84)}")
        if node.priority is not None:
            lower, higher = node.priority
            # print(f"Node conflict: {node.priority[0], node.priority[1]}")
            priorities.add_priority(lower=lower, higher=higher)
       
        collision = node.find_collision()
        if collision is None:
            return node.plan
        
        new_nodes = []
        node1 = PTNode(
            parent=node,
            priority=collision,
            # plan=node.plan.copy()
            plan=copy.deepcopy(node.plan)
        )
        node2 = PTNode(
            parent=node,
            priority=tuple(reversed(collision)),
            # plan=node.plan.copy()
            plan=copy.deepcopy(node.plan)
        )

        # print(f"Resolving conflict for first PTNode: {node1.priority[0], node1.priority[1]}")
        priorities.add_priority(lower=node1.priority[0], higher=node1.priority[1])
        if update_plan(node1, node1.priority[0]):
            new_nodes.append(node1)
            # print(f"Node1 conflict after update: {node1.has_conflict(11, 84)}")
        priorities.remove_last_conflict()
        # print("Resolving conflicts completed!")

        # print(f"Resolving conflict for second PTNode: {node2.priority[0], node2.priority[1]}")
        priorities.add_priority(lower=node2.priority[0], higher=node2.priority[1])
        if update_plan(node2, node2.priority[0]):
            new_nodes.append(node2)
            # print(f"Node1 conflict after update: {node2.has_conflict(11, 84)}")
        priorities.remove_last_conflict()
        # print("Resolving conflicts completed!")

        new_nodes.sort(key=lambda x: x.cost, reverse=True)
        stack.extend(new_nodes)
        node.times_visited += 1
    
    return None


def GPBS(starts, goals, task_map, search_function, *args):
    n_agents = len(starts)
    root = PTNodeGPBS(plan=[], n_agents=n_agents)
    for start, goal in zip(starts, goals):
        last_node = astar(
            task_map,
            *start,
            *goal,
            *args
        )[1]
        assert last_node is not None, f"Didn't find path for agent with start={start} and goal={goal}"
        root.plan.append(make_path(last_node)[0])
        # root.plan.append(update_plan_for_agent(*start, *goal, [], task_map, search_function, *args))
    priorities = Priorities(n_agents)
    for a1 in range(n_agents):
        for a2 in range(a1 + 1, n_agents):
            conflict = root.has_conflict(a1, a2)
            if conflict != ConflictType.NO_CONFLICT:
                root.agent_conflicts[a1].add((a2, conflict))
                root.agent_conflicts[a2].add((a1, conflict))
    root.update_conflicts()
    stack = [root]
    # print("Initial paths for agents are constructed!")
    last_conflict = None

    def topsort(agents: list[int]) -> list[int]:
        visited = [False for i in range(len(agents))]
        topsort_order = []
        def dfs(u):
            visited[u] = True
            for v in range(len(agents)):
                a1 = agents[u]
                a2 = agents[v]
                if priorities.has_edge(lower=a2, higher=a1) and not visited[v]:
                    dfs(v)
            topsort_order.append(agents[u])
        
        dfs(len(agents) - 1)
        return topsort_order[::-1]
    
    def update_plan(node: PTNodeGPBS, agent: int) -> bool:
        agents = list(priorities.get_lower_priority_agents(agent))
        agents.append(agent)
        agents_to_update = topsort(agents)
        need_update = defaultdict(bool)
        for curr_agent in agents_to_update:
            hp_agents = priorities.get_higher_priority_agents(curr_agent)
            for hp_agent in hp_agents:
                # # print(hp_agent, curr_agent)
                # # print(node.has_conflict(curr_agent, hp_agent))
                if node.has_conflict(curr_agent, hp_agent) != ConflictType.NO_CONFLICT:
                    need_update[curr_agent] = True
                    break
        # were_updated = []
        for curr_agent in agents_to_update:
            if not need_update[curr_agent]:
                continue
            # were_updated.append(curr_agent)
            hp_agents = priorities.get_higher_priority_agents(curr_agent)
            # print(f'Update path for agent: {curr_agent} | Higher priority agents: {hp_agents}')
            trajectories = [node.plan[hp_agent] for hp_agent in hp_agents]
            new_path = update_plan_for_agent(
                root.plan[curr_agent][0][0],
                root.plan[curr_agent][0][1],
                root.plan[curr_agent][-1][0],
                root.plan[curr_agent][-1][1],
                trajectories,
                task_map,
                search_function,
                *args
            )
            if new_path is None:
                # print("Path wasn't found")
                return False
            node.plan[curr_agent] = new_path
            lp_agents = priorities.get_lower_priority_agents(curr_agent)
            for lp_agent in lp_agents:
                if node.has_conflict(curr_agent, lp_agent) != ConflictType.NO_CONFLICT:
                    need_update[lp_agent] = True
            # Update conflicts
            for conflict_agent, conflict_type in node.agent_conflicts[curr_agent]:
                node.agent_conflicts[conflict_agent].remove((curr_agent, conflict_type))
            node.agent_conflicts[curr_agent] = set()
            for new_agent in range(node.n_agents):
                if new_agent not in hp_agents and new_agent not in lp_agents and new_agent != curr_agent:
                    conflict = node.has_conflict(curr_agent, new_agent)
                    if conflict != ConflictType.NO_CONFLICT:
                        node.agent_conflicts[curr_agent].add((new_agent, conflict))
                        node.agent_conflicts[new_agent].add((curr_agent, conflict))
            
        node.update_cost()
        node.update_conflicts()
        # print(f"Successful paths update, were updated agents: {were_updated}")
        return True

    while len(stack):
        node = stack[-1]
        if node.times_visited == 2:
            # print("Second visit to PTNode")
            stack.pop()
            if len(stack) == 0: # can't find paths for root (i.e. no solution found)
                continue
            # priorities.remove_last_conflict()
            last_conflict = priorities.remove_last_conflict()
            # print(f"Removing last conflict: {last_conflict}")
            continue
        # if node.times_visited == 1:
        #     new_conflict = tuple(reversed(last_conflict))

        # print("First visit to PTNode")
        # print(f"CurrNode conflict after update: {node.has_conflict(11, 84)}")
        if node.priority is not None and node.times_visited == 0:
            lower, higher = node.priority
            # print(f"Node conflict: {node.priority[0], node.priority[1]}")
            priorities.add_priority(lower=lower, higher=higher)
        
        if node.times_visited == 0:
            collision = node.find_collision(priorities)
            if collision is None:
                return node.plan
        else:
            collision = tuple(reversed(last_conflict))
        
        # new_nodes = []
        new_node = PTNodeGPBS(
            parent=node,
            priority=collision,
            plan=node.plan.copy(),
            n_agents=n_agents
            # plan=copy.deepcopy(node.plan)
        )

        # node2 = PTNodeGPBS(
        #     parent=node,
        #     priority=tuple(reversed(collision)),
        #     plan=node.plan.copy(),
        #     n_agents=n_agents
        #     # plan=copy.deepcopy(node.plan)
        # )

        # Add agent_conflict to child nodes
        for i in range(n_agents):
            new_node.agent_conflicts[i] = node.agent_conflicts[i].copy()
            # node2.agent_conflicts[i] = node.agent_conflicts[i].copy()

        # print(f"Resolving conflict for first PTNode: {node1.priority[0], node1.priority[1]}")
        priorities.add_priority(lower=new_node.priority[0], higher=new_node.priority[1])
        if update_plan(new_node, new_node.priority[0]):
            stack.append(new_node)
        else:
            last_conflict = new_node.priority
            # new_nodes.append(node1)
            # print(f"Node1 conflict after update: {node1.has_conflict(11, 84)}")
        priorities.remove_last_conflict()
        # print("Resolving conflicts completed!")

        # print(f"Resolving conflict for second PTNode: {node2.priority[0], node2.priority[1]}")
        # priorities.add_priority(lower=node2.priority[0], higher=node2.priority[1])
        # if update_plan(node2, node2.priority[0]):
            # new_nodes.append(node2)
            # print(f"Node1 conflict after update: {node2.has_conflict(11, 84)}")
        # priorities.remove_last_conflict()
        # print("Resolving conflicts completed!")

        # new_nodes.sort(key=lambda x: x.cost, reverse=True)
        # stack.extend(new_nodes)
        node.times_visited += 1
    
    return None

# def GPBS(starts, goals, task_map, search_function, *args):
#     n_agents = len(starts)
#     root = PTNode(plan=[])
#     for start, goal in zip(starts, goals):
#         root.plan.append(update_plan_for_agent(*start, *goal, [], task_map, search_function, *args))
#     priorities = Priorities(n_agents)
#     stack = [root]
#     last_removed_conflict = (-1, -1) # (lower, higher) last remove conflict agents

#     def topsort(agents: list[int]) -> list[int]:
#         visited = [False for i in range(len(agents))]
#         topsort_order = []
#         def dfs(u):
#             visited[u] = True
#             for v in range(len(agents)):
#                 a1 = agents[u]
#                 a2 = agents[v]
#                 if priorities.has_edge(a2, a1) and not visited[v]:
#                     dfs(v)
#             topsort_order.append(agents[u])
        
#         dfs(len(agents) - 1)
#         return reversed(topsort_order)

#     def update_plan(node: PTNode, agent: int) -> bool:
#         agents = list(priorities.get_lower_priority_agents(agent))
#         agents.append(agent)
#         agents_to_update = topsort(agents)
#         for curr_agent in agents_to_update:
#             hp_agents = priorities.get_higher_priority_agents(curr_agent)
#             trajectories = [node.plan[hp_agent] for hp_agent in hp_agents]
#             new_path = update_plan_for_agent(
#                 root.plan[curr_agent][0][0],
#                 root.plan[curr_agent][0][1],
#                 root.plan[curr_agent][-1][0],
#                 root.plan[curr_agent][-1][1],
#                 trajectories,
#                 task_map,
#                 search_function,
#                 *args
#             )
#             if new_path is None:
#                 return False
#             node.plan[curr_agent] = new_path
#         node.update_cost()
#         return True

#     while len(stack):
#         node = stack[-1]
#         if node.times_visited == 2:
#             stack.pop()
#             lower, higher = priorities.get_last_conflict()
#             last_removed_conflict = (lower, higher)
#             priorities.remove_priority(lower, higher)
#             priorities.pop()
#         elif node.times_visited == 1:
#             collision(tuple(reversed(last_removed_conflict)))
#             new_node = PTNode(
#                 parent=node,
#                 priority=collision,
#                 plan=node.plan
#             )
#             if update_plan(new_node, new_node.priority[0]):
#                 stack.append(new_node)
#         else:
#             if node.priority is not None:
#                 lower, higher = node.priority
#                 priorities.add_priority(lower, higher)

#             collision = node.find_collision()
#             if collision is None:
#                 return node.plan
#             new_node = PTNode(
#                 parent=node,
#                 priority=collision,
#                 plan=node.plan
#             )
#             if update_plan(new_node, new_node.priority[0]):
#                 stack.append(new_node)
#         node.times_visited += 1
    
#     return None
