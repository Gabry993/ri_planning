from dataclasses import dataclass
from typing import (Any, Callable, Dict, Generator,
                    List, Optional, Tuple, Union, cast, Type, TypeVar)

import networkx as nx
import numpy as np
import shapely as s
from scipy import interpolate

from ..map.map import Graph, GraphEdge, GraphNode, Layer, State, Transition, reverse, reverse_route
from ..utilities import Pose, n_grams
from .dual_graph import DualGraph, init_dual_graph
from .state_set import StateSet

StateFilter = Callable[[State], bool]

PlanVertexUnion = Union[StateSet, GraphNode, GraphEdge, Pose]
SelfPlan = TypeVar("SelfPlan", bound="Plan")


@dataclass
class PlanVertex:
    """begin and end of a target"""

    state: Optional[Union[GraphNode, StateSet]] = None
    transition: Optional[GraphEdge] = None
    pose: Optional[Pose] = None

    @property
    def primary(self) -> Optional[PlanVertexUnion]:
        return self.pose or self.transition or self.state

    @property
    def reversed(self) -> 'PlanVertex':
        return PlanVertex(
            self.state,
            reverse(self.transition) if self.transition else None,
            self.pose.reversed if self.pose else None)


def endpoint_for_state(state: str, layer: Optional[Layer],
                       graph: Optional[Graph]) -> PlanVertexUnion:
    if (layer and state in layer.map.states and state not in layer.states):
        # start is a state BUT not in our layer,
        # replace it with a StateSet
        s = layer.map.states[state]
        # the set of all navigation states contained into the state s
        states = s.contains[layer.id]
        if states:
            # some states are contained,  we set the destination to their union
            return StateSet(graph, states, on_border=False)
        elif s.geometry:
            # no states is fully contained,  therefore we set the destination
            # to the "center" of the state.
            # vertex(...) will take care of finding the correspondig state
            # in the navigation layer
            return Pose(s.geometry.coords[0])
    return state


def vertex_for_endpoint(ep: Optional[PlanVertexUnion], i: int,
                        layer: Optional[Layer],
                        dual_graph: Optional[DualGraph]) -> PlanVertex:
    vertex = PlanVertex()
    if isinstance(ep, str):
        if layer and ep in layer.states:
            vertex.state = ep
    elif isinstance(ep, StateSet):
        vertex.state = ep
    elif isinstance(ep, tuple):
        if dual_graph and ep in dual_graph:
            vertex.state = ep[1 - i]
            vertex.transition = ep
    else:
        vertex.pose = cast(Pose, ep)
        if layer:
            _, state, position = next(
                layer.nearest_states(vertex.pose.position))
            if state:
                vertex.state = state.id
                vertex.pose.position = position
    return vertex


# TODO(Jerome 2023): set states (old version only set states for dual and visibility graphs)


class Plan:
    """TODO"""

    planner: Optional['Planner']
    layer: Optional[Layer]
    start: PlanVertexUnion
    end: PlanVertexUnion
    vertices: Tuple[PlanVertex, PlanVertex]
    agent: Any

    _transitions: List[GraphEdge] = []
    states: List[GraphNode] = []

    def __init__(self,
                 planner: Optional['Planner'] = None,
                 start: Optional[PlanVertexUnion] = None,
                 end: Optional[PlanVertexUnion] = None,
                 vertices: Optional[Tuple[PlanVertex, PlanVertex]] = None,
                 agent: Any = None):
        super().__init__()
        self.agent = agent
        self.planner = planner
        if planner:
            self.layer = planner.layer
            graph: Optional[Graph] = planner.G
            dual_graph: Optional[DualGraph] = planner.D
            if isinstance(start, str):
                start = endpoint_for_state(start, self.layer, graph)
            if isinstance(end, str):
                end = endpoint_for_state(end, self.layer, graph)
        else:
            self.layer = None
            graph = None
            dual_graph = None

        if start:
            self.start = start
        elif vertices and vertices[0].primary:
            self.start = vertices[0].primary
        else:
            raise ValueError("No start")

        if end:
            self.end = end
        elif vertices and vertices[1].primary:
            self.end = vertices[1].primary
        else:
            raise ValueError("No end")

        if vertices:
            self.vertices = vertices
        elif self.layer and start and end:
            self.vertices = (vertex_for_endpoint(start, 0, self.layer,
                                                 dual_graph),
                             vertex_for_endpoint(end, 1, self.layer,
                                                 dual_graph))
        else:
            raise ValueError("No vertices")
        self.states = []
        self._transitions = []
        self.geometry = None
        self.costs = np.array([])

    def clone(self: SelfPlan) -> SelfPlan:
        p = self.__class__(self.planner)
        p.layer = self.layer
        p.start = self.start
        p.end = self.end
        p.agent = self.agent
        p.vertices = self.vertices
        return p

    @classmethod
    def from_dict(cls, planner: 'Planner', value: Dict) -> 'Plan':
        p = cls(planner=planner)
        p.costs = np.array(value['costs'])
        return p

    @property
    def to_dict(self) -> Dict[str, Any]:
        value = {'costs': self.costs.tolist()}
        return value

    def geometry_with_resolution(
            self,
            resolution: int,
            with_poses: bool = False) -> Tuple[s.LineString, np.ndarray]:
        if not self.geometry:
            return None, np.array([])
        L = self.costs[0]
        if resolution:
            resolution = max(min(resolution, L / 5), 1)
        curve = np.asarray(self.geometry.coords)
        if resolution:
            delta = [0] + [
                s.Point(p1).distance(s.Point(p2))
                for p1, p2 in n_grams(curve, 2)
            ]
            s = np.cumsum(np.array(delta))
            le = np.linspace(0, s[-1], int(L / resolution))
            curve = interpolate.interp1d(s, curve.T)(le).T
        x, y = curve.T
        dx = np.gradient(x)
        dy = np.gradient(y)
        angle = np.arctan2(dy, dx)
        if len(angle) > 1:
            angle[0] = angle[1]
            angle[-1] = angle[-2]
        curve = np.around(curve)
        poses = []
        if with_poses:
            angle = np.around(angle, 2)
            poses = np.vstack([curve.T, angle]).T.tolist()
        return s.LineString(curve), poses

    def route_in_layer(
            self, layer: Layer) -> Tuple[List[GraphNode], List[GraphEdge]]:
        # print 'route_in_layer', layer, self.states, self.transitions
        if layer is self.layer:
            return self.states, self.transitions
        # ptransitions = [player.transitions[t]
        #                 for s1, s2, t in self.transitions]
        # print self.states
        # print self.transitions
        if not self.layer:
            return [], []
        pstates = [self.layer.states[s] for s in self.states]
        # print pstates, ptransitions

        s1 = None
        states: List[GraphNode] = []
        transitions: List[GraphEdge] = []
        for state in pstates:
            state2 = state.in_layer(layer)
            if state2:
                s2 = state2.id
            else:
                # print '?', state, state2, layer
                return [], []
            if s1 is not s2:
                states.append(s2)
                if s1:
                    # print 'state2', state2
                    # print 's1, s2', s1, s2
                    # print layer.graph[s1]
                    # TODO,  find the right transition
                    t = layer.graph[s1][s2].keys()[0]
                    transitions.append((s1, s2, t))
                s1 = s2
        return states, transitions

    @property
    def transitions(self) -> List[GraphEdge]:
        return self._transitions

    @transitions.setter
    def transitions(self, route: List[GraphEdge]) -> None:
        self._transitions = route
        if self.planner:
            G = self.planner.G
            self.costs = np.array(
                [sum([G[s1][s2][t]['weight'] for s1, s2, t in route])])

    def reverse(self) -> None:
        if self.vertices:
            vs = []
            for vertex in self.vertices[::-1]:
                vs.append(vertex.reversed)
            self.vertices = cast(Tuple[PlanVertex, PlanVertex], tuple(vs))
        if self.states:
            self.states = self.states[::-1]
        if self.transitions:
            self.transitions = reverse_route(self.transitions)

    # TODO(Jerome 2023): Should be equivalent to reverse . close
    def reversed(self: SelfPlan) -> SelfPlan:
        vertices = None
        if self.vertices:
            vs = []
            for vertex in self.vertices[::-1]:
                vs.append(vertex.reversed)
            vertices = cast(Tuple[PlanVertex, PlanVertex], tuple(vs))
        rplan = self.__class__(self.planner, vertices=vertices)
        if self.states:
            rplan.states = self.states[::-1]
        if self.transitions:
            rplan.transitions = reverse_route(self.transitions)
        rplan.costs = self.costs
        return rplan

    @property
    def wkt(self) -> str:
        if self.geometry:
            return self.geometry.wkt
        return ''


def transitions_along_path(path: List[str], graph: Graph) -> List[GraphEdge]:
    if graph.is_multigraph():
        return [(s1, s2,
                 sorted([(data['weight'], t)
                         for t, data in graph[s1][s2].items()])[0][1])
                for s1, s2 in n_grams(path, 2)]
    else:
        return [(s1, s2, graph[s1][s2]['id']) for s1, s2 in n_grams(path, 2)]


def subgraph(G: Graph, s1: Union[GraphNode, StateSet],
             s2: Union[GraphNode, StateSet]) -> Graph:
    # The nodes in s1 and s2 are NOT contracted!
    # Therefore there could be paths like
    # A -> s2[0] -> s2[1]
    # If we don't want such paths (in all paths),
    # we would need to contract the nodes,  see
    # http://networkx.readthedocs.org/en/stable/reference/generated/networkx.algorithms.minors.contracted_nodes.html

    if G.is_multigraph:
        S = nx.MultiGraph(G)
    else:
        S = nx.Graph(G)
    if isinstance(s1, StateSet):
        S.remove_edges_from(s1.graph.edges())
        S.add_edges_from([(s1, s, {'weight': 0}) for s in s1.states])
    if isinstance(s2, StateSet):
        S.remove_edges_from(s2.graph.edges())
        S.add_edges_from([(s, s2, {'weight': 0}) for s in s2.states])
    return S


class Planner:
    """Planner

    :ivar layer: navigation layer
    :ivar G: copy of the layer graph with state.id (str) as nodes
             and transition.id (str) to identify edges
    :ivar D: line graph a G with (start.id: str, end.id: str, transition.id: str) as nodes
             and edges that traverse states (s1, e1, i1), (s2=e1, e2, i2)
    :ivar transition_neighbors: a map of transitions (ids) connected by a state

    """

    def __init__(self,
                 layer: Layer,
                 filter_states: Optional[StateFilter] = None):
        super().__init__()
        self.layer = layer
        self.G = nx.MultiGraph(layer.graph)
        if filter_states:
            self.G.remove_nodes_from(
                [n for n in self.G if filter_states(layer.states[n])])
        self.D = init_dual_graph(self.G)
        self.init_neightbors()

    def init_neightbors(self) -> None:
        self.transition_neighbors: Dict[str, List[Transition]] = {}
        for node in self.D:
            self.transition_neighbors[node[2]] = [
                self.layer.transitions[t]
                for s1, s2, t in nx.all_neighbors(self.D, node)
            ]

    def shortest_path(self,
                      start: PlanVertexUnion,
                      end: PlanVertexUnion,
                      graph: Optional[Graph] = None,
                      path_class: Type[SelfPlan] = Plan) -> SelfPlan:

        plan = path_class(self, start, end)
        if not plan.vertices:
            raise ValueError("No vertices")
        t1 = plan.vertices[0].transition
        t2 = plan.vertices[1].transition
        s1 = plan.vertices[0].state
        s2 = plan.vertices[1].state
        if s1 is None or s2 is None:
            raise ValueError("No valid source/target states")
        SG = graph or self.G
        if isinstance(s1, StateSet) or isinstance(s2, StateSet):
            G = subgraph(SG, s1, s2)
        elif t1 or t2:
            G = nx.MultiGraph(SG)
            if t1:
                G.remove_edge(*t1)
            if t2:
                G.remove_edge(*t2)
        else:
            G = SG
        path = nx.shortest_path(G, s1, s2, 'weight')
        if isinstance(s1, StateSet):
            path.pop(0)
        if isinstance(s2, StateSet):
            path.pop()
        # TODO(Jerome 2023): Here we are
        plan.states = path
        transitions = transitions_along_path(path, graph=self.G)
        if t1:
            transitions.insert(0, t1)
        if t2:
            transitions.append(t2)
        plan.transitions = transitions
        return plan

    def all_paths(self,
                  start: PlanVertexUnion,
                  end: PlanVertexUnion,
                  number: int = -1,
                  graph: Optional[nx.Graph] = None,
                  path_class: Type[SelfPlan] = Plan) -> Generator[SelfPlan, None, None]:
        # start_time=time()
        plan = path_class(self, start, end)
        vertices = plan.vertices
        if not plan.vertices:
            return
        t1 = plan.vertices[0].transition
        t2 = plan.vertices[1].transition
        s1 = plan.vertices[0].state
        s2 = plan.vertices[1].state
        # layer = self.layer

        if not s1 or not s2:
            raise ValueError("No states")

        if not graph:
            graph = self.G
        if isinstance(s1, StateSet) or isinstance(s2, StateSet):
            graph = subgraph(graph, s1, s2)
            graph = nx.Graph(graph)
        # elif t1 or t2:
        #     G=nx.Graph(G)
        #     if t1:
        #         G.remove_edge(*t1)
        #     if t2:
        #         G.remove_edge(*t2)
        else:
            graph = nx.Graph(graph)

        # if not G:
        #     G=nx.Graph(layer.graph)
        # else:
        #     G=nx.Graph(G)
        k = 0
        if s1 not in graph or s2 not in graph or not nx.has_path(graph, s1, s2):
            return
        for path in nx.shortest_simple_paths(graph, s1, s2, 'weight'):
            _path = path[:]
            if isinstance(s1, StateSet):
                _path.pop(0)
            if isinstance(s2, StateSet):
                _path.pop()
            transitions = transitions_along_path(_path, graph=graph)
            # print '%s) %s' % (k, time()-start_time)
            if t1:
                transitions.insert(0, t1)
            if t2:
                transitions.append(t2)
            plan = path_class(self, vertices=vertices)
            plan.states = _path
            plan.transitions = transitions
            yield plan
            k = k + 1
            if k == number:
                return
