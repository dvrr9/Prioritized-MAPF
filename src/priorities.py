class Priorities:
    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self.priorities = []
        self.lh_edges = [set() for i in range(n_agents)] # edges from lower to higher priority agents
        self.hl_edges = [set() for i in range(n_agents)] # edges from higher to lower priority agents

    def add_priority(self, lower: int, higher: int):
        assert higher not in self.hl_edges[lower], \
            f"Agent with higher priority {higher} is already in lower priority agents for {lower}"
        assert lower not in self.lh_edges[higher], \
            f"Agent with lower priority {lower} is already in higher priority agents for {higher}"
        self.lh_edges[lower].add(higher)
        self.hl_edges[higher].add(lower)
        self.priorities.append((lower, higher))
    
    def remove_last_conflict(self):
        lower, higher = self.priorities.pop()
        self.lh_edges[lower].remove(higher)
        self.hl_edges[higher].remove(lower)
        return (lower, higher)

    def get_last_conflict(self):
        if len(self.priorities) == 0:
            raise ValueError("No conflicts found")
        return self.priorities[-1]

    def has_edge(self, lower: int, higher: int) -> bool:
        return higher in self.lh_edges[lower]

    def get_lower_priority_agents(self, agent: int) -> set[int]:
        lower_priority_agents = set()
        for new_agent in self.hl_edges[agent]:
            lower_priority_agents.add(new_agent)
            lower_priority_agents.update(self.get_lower_priority_agents(new_agent))
        return lower_priority_agents

    def get_higher_priority_agents(self, agent: int) -> set[int]:
        higher_priority_agents = set()
        for new_agent in self.lh_edges[agent]:
            higher_priority_agents.add(new_agent)
            higher_priority_agents.update(self.get_higher_priority_agents(new_agent))
        return higher_priority_agents
