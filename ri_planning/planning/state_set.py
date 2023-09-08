from typing import Collection, FrozenSet
from ..map.map import Graph, GraphEdge, reverse_route


class StateSet:
    """TODO"""

    states: FrozenSet[str]
    graph: Graph
    internal_edges: FrozenSet[GraphEdge]
    external_edges: FrozenSet[GraphEdge]
    out_edges: FrozenSet[GraphEdge]
    in_edges: FrozenSet[GraphEdge]
    # whether we the target/source concerns just the border,
    # e.g. if we are happy to arrive to a certain set without
    # entering it# whether we the target/source concerns
    # just the border, e.g. if we are happy to arrive
    # to a certain set without entering it
    on_border: bool

    def __init__(self,
                 G: Graph,
                 states: Collection[str],
                 on_border: bool = True):
        self.states = frozenset(states)
        self.graph = G.subgraph(states)
        edges = list(self.graph.edges(keys=True))
        all_edges = list(G.edges(states, keys=True))
        self.internal_edges = frozenset(edges + reverse_route(edges))
        self.external_edges = (
            frozenset(all_edges + reverse_route(all_edges)) -
            self.internal_edges)
        self.out_edges = frozenset(
            [e for e in self.external_edges if e[0] in states])
        self.in_edges = self.external_edges - self.out_edges
        self.on_border = on_border

    def __repr__(self) -> str:
        return f'<{",".join(self.states)}>'
