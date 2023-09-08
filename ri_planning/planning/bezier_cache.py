import collections
import itertools
from typing import (TYPE_CHECKING, Any, Collection, Dict, Iterable, Iterator,
                    List, Optional, Tuple, DefaultDict, Set)

import networkx as nx
import yaml

from ..map.map import Graph, GraphEdge, reverse, reverse_route
from .dual_graph import DualGraph

if TYPE_CHECKING:
    from .bezier_planner import Plan, Planner

BezierGraph = Any  # nx.MultiDiGraph
BezierSimpleGraph = Any  # nx.DiGraph
CacheKey = Tuple[GraphEdge, ...]


def rotations() -> Iterator[Tuple[int, int]]:
    return itertools.product(*[[0, 1]] * 2)  # type: ignore


def rotate(node: GraphEdge, i: int) -> GraphEdge:
    return node[i], node[1 - i], node[2]


def edges2key(edges: Iterable[GraphEdge]) -> str:
    return "->".join(["&".join(edge) for edge in edges])


def key2edges(key: str) -> Tuple[CacheKey, ...]:
    return tuple([tuple(edge.split("&")) for edge in key.split("->")])  # type: ignore


def simple_graph(multigraph: nx.MultiDiGraph,
                 default_data: Dict = {}) -> nx.DiGraph:
    edges: List[Tuple[Any, Any, Dict]] = sum(
        [[(s, k, data), (k, e, default_data)]
         for s, e, k, data in multigraph.edges(data=True, keys=True)], [])
    return nx.DiGraph(edges)


def add_path(edge: Tuple[GraphEdge, GraphEdge], planner: 'Planner', graph: Graph,
             bezier_graph: BezierGraph, sparse: bool) -> None:
    A, B = edge
    RA, RB = reverse(B), reverse(A)
    plans = planner.all_paths(A, B, number=4, graph=graph, use_cache=False, sparse=sparse)
    for plan in plans:
        bezier_graph.add_edge(A,
                              B,
                              tuple(plan.transitions),
                              plan=plan,
                              curvature=plan.costs[1],
                              length=plan.costs[0])
        rplan = plan.reversed()
        bezier_graph.add_edge(RA,
                              RB,
                              tuple(rplan.transitions),
                              plan=rplan,
                              curvature=rplan.costs[1],
                              length=rplan.costs[0])


class BezierCache:

    paths: Dict[CacheKey, 'Plan']
    graph: BezierGraph
    planner: 'Planner'
    fixed_predecessors: DefaultDict[str, Set[GraphEdge]]
    fixed_successors: DefaultDict[str, Set[GraphEdge]]

    def __init__(self, planner: 'Planner', path: str):
        self.planner = planner
        edges = self.init_graph(graph=planner.G,
                                dual_graph=planner.D,
                                nodes=planner.fixed)
        if path:
            self.load(path)
        else:
            for edge in edges:
                add_path(edge,
                         planner,
                         graph=self.G_cache,
                         bezier_graph=self.graph,
                         sparse=planner.sparse)
            self.plans = {
                key: data['plan']
                for s1, s2, key, data in self.graph.edges(data=True, keys=True)
            }
        self.simple_graph = simple_graph(self.graph, {
            'length': 0,
            'curvature': 0
        })

    def init_graph(
            self, graph: Graph, dual_graph: DualGraph,
            nodes: Collection[GraphEdge]) -> List[Tuple[GraphEdge, GraphEdge]]:
        D = nx.line_graph(graph)
        # [node[0] for node, data in H.nodes(data = True) if 'fixed' in data]
        self.G_cache = nx.MultiGraph(graph)
        self.G_cache.remove_edges_from(nodes)

        E1 = D.subgraph(nodes)
        E2 = nx.line_graph(graph)
        E2.remove_nodes_from(nodes)
        self.graph = nx.MultiDiGraph()
        edges = []

        # self.plotGraph(D)

        self.fixed_predecessors = collections.defaultdict(set)
        self.fixed_successors = collections.defaultdict(set)

        for s1, s2, t in nodes:
            self.fixed_predecessors[s1].add((s2, s1, t))
            self.fixed_successors[s1].add((s1, s2, t))
            self.fixed_predecessors[s2].add((s1, s2, t))
            self.fixed_successors[s2].add((s2, s1, t))

        for u, v in E1.edges():
            # print(u, v)
            connected = False
            for i, j in rotations():
                if u[i] == v[j]:
                    edges.append((rotate(u, 1 - i), rotate(v, j)))
                    connected = True
                    break
            assert connected

        for internal_nodes in nx.connected_components(E2):
            states = set()
            for node in internal_nodes:
                states.update(set(node[:2]))
            c = internal_nodes
            for e in internal_nodes:
                c = c | set(D.neighbors(e))
            nodes_to_be_connected = set(c) & set(nodes)

            in_directed_fixed_transition = set()
            out_directed_fixed_transition = set()
            for a, b, t in nodes_to_be_connected:
                # either a or b are in the component
                # #TODO:240 reverse the costruction:
                # take the connected comp. of E1,
                # add the neighbors,  ...
                # TODO Why should I disable it?
                # assert ((a in states) != (b in states)), \
                #     ("%s %s %s in %s (%s)\n%s\n%s" %
                #      (a, b, t, states, internal_nodes, c,
                #       nodes_to_be_connected))
                if a in states:
                    e_out = (a, b, t)
                    e_in = (b, a, t)
                else:
                    e_out = (b, a, t)
                    e_in = (a, b, t)
                in_directed_fixed_transition.add(e_in)
                out_directed_fixed_transition.add(e_out)

            for s in states:
                self.fixed_predecessors[s].update(in_directed_fixed_transition)
                self.fixed_successors[s].update(out_directed_fixed_transition)

            L = dual_graph.subgraph(list(c) + reverse_route(list(c)))
            for u, v in itertools.product(internal_nodes,
                                          out_directed_fixed_transition):
                for i in [0, 1]:
                    w = rotate(u, i)
                    if nx.has_path(L, w, v):
                        self.fixed_successors[w].add(v)
                        self.fixed_predecessors[reverse(w)].add(reverse(v))

            for u, v in itertools.product(in_directed_fixed_transition,
                                          out_directed_fixed_transition):
                if not u[1] is v[0]:
                    edges.append((u, v))
        return edges

    def get_plan(self, key: CacheKey) -> Optional['Plan']:
        return self.plans.get(key, None)

    def load(self, path: str) -> None:

        with open(path, 'r') as f:
            value = yaml.safe_load(f)
            paths = value.get('paths', {})
            self.planner.min_width = value.get('min_width',
                                               self.planner.min_width)
            self.planner.tol = value.get('tol', self.planner.tol)

        self.plans = {
            key2edges(k): Plan.from_dict(self.planner, v)
            for k, v in paths.items()
        }

        for transitions, plan in self.plans.items():
            A = transitions[0]
            B = transitions[-1]
            RA, RB = reverse(B), reverse(A)
            self.graph.add_edge(A,
                                B,
                                transitions,
                                plan=plan,
                                curvature=plan.costs[1],
                                length=plan.costs[0])
            rplan = plan.reversed()
            self.graph.add_edge(RA,
                                RB,
                                tuple(reverse_route(transitions)),
                                plan=rplan,
                                curvature=rplan.costs[1],
                                length=rplan.costs[0])

    def save_cache(self, path: str) -> None:
        paths = {edges2key(k): v.to_dict for k, v in self.plans.items()}
        value = {
            'tol': self.planner.tol,
            'min_width': self.planner.min_width,
            'paths': paths,
            'sparse': self.planner.sparse
        }
        with open(path, 'w') as f:
            yaml.dump(value, f)
