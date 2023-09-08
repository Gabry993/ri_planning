from typing import (Any, Collection, Dict, Generator, List, Mapping, Optional,
                    Set, Tuple, Type, TypeVar)

import networkx as nx
import numpy as np
import shapely as s

from ..map.map import Graph, GraphEdge, GraphNode, Layer, reverse
from ..utilities import Pose, angle, Vector
from .dual_graph import DualGraph
from .dual_planner import Planner as DualPlanner
from .dual_planner import compute_states_and_transitions
from .planner import Plan as BasePlan
from .planner import PlanVertex, PlanVertexUnion, StateFilter, StateSet
from .visibility_graph import (VisibilityGraph, VisibilityNode,
                               VisibilityRoute, add_vertex_for_state,
                               init_visibility_graph, visibility_subgraph)

SelfPlan = TypeVar("SelfPlan", bound="Plan")


class Plan(BasePlan):

    hgraph: Optional[VisibilityGraph] = None
    hpath: Optional[VisibilityRoute] = None

    def __init__(self,
                 planner: Optional['Planner'] = None,
                 start: Optional[PlanVertexUnion] = None,
                 end: Optional[PlanVertexUnion] = None,
                 vertices: Optional[Tuple[PlanVertex, PlanVertex]] = None,
                 agent: Any = None):
        super().__init__(planner, start, end, vertices, agent)
        self.hgraph = self.hpath = None

    @property
    def has_visibility_plan(self) -> bool:
        return self.hpath is not None

    @property
    def visibility_plan(
            self
    ) -> Tuple[Optional[VisibilityGraph], Optional[VisibilityRoute]]:
        return (self.hgraph, self.hpath)

    @visibility_plan.setter
    def visibility_plan(
            self, hplan: Tuple[VisibilityGraph, VisibilityRoute]) -> None:
        self.hgraph, self.hpath = hplan
        if len(self.hpath) > 1:
            self.geometry = s.LineString(
                [np.array(pose.position) for node, pose in self.hpath])

        else:
            node, pose = self.hpath[0]
            self.geometry = s.Point(pose.position)
        if self.geometry:
            self.costs = np.array([self.geometry.length])
        if not self.transitions:
            self._transitions = [
                node_ for node_, point_ in self.hpath
                if node_[-1] in self.layer.transitions
            ]
        if not self.states:
            self.states = [node_[1] for node_, point_ in self.hpath[:-1]]

    def reversed(self: SelfPlan) -> SelfPlan:
        rplan = super().reversed()
        if self.hgraph and self.hpath:
            rplan.hgraph = self.hgraph.reverse()
            rplan.hpath = [(reverse(node), pose.reversed)
                           for node, pose in self.hpath[::-1]]
            if self.geometry:
                rplan.geometry = s.LineString(self.geometry.coords[::-1])
        return rplan

    def project(self, position: Vector, delta: float = 0.0) -> Pose:
        path_s = self.geometry.project(s.Point(position)) + delta
        p1 = self.geometry.interpolate(path_s)
        p2 = self.geometry.interpolate(path_s + 0.1)
        delta = np.asarray(p2.coords[0]) - np.asarray(p1.coords[0])
        return Pose(tuple(p1.coords[0]), angle(delta))


def node_for_start_state_set(
    layer: Layer,
    graph: Graph,
    visibility_graph: VisibilityGraph,
    nodes_for_transition: Mapping[GraphEdge, List[VisibilityNode]],
    state_set: StateSet,
) -> VisibilityNode:
    if state_set.on_border:
        for e in state_set.out_edges:
            s1, s2, t = e
            for n in nodes_for_transition[e]:
                visibility_graph.add_edge(state_set, n, weight=0)
    else:
        for state in state_set.states:
            fstate = layer.states[state]
            if not fstate.geometry:
                continue
            p = Pose(fstate.geometry.coords[0])
            node = add_vertex_for_state(layer, graph, visibility_graph, state,
                                        p)
            visibility_graph.add_edge(state_set, node, weight=0)
    return state_set


def node_for_end_state_set(layer: Layer, graph: Graph,
                           visibility_graph: VisibilityGraph,
                           nodes_for_transition: Mapping[GraphEdge,
                                                         List[VisibilityNode]],
                           state_set: StateSet) -> VisibilityNode:
    if state_set.on_border:
        for e in state_set.in_edges:
            s1, s2, t = e
            for n in nodes_for_transition[e]:
                visibility_graph.add_edge(n, state_set, weight=0)
    else:
        for state in state_set.states:
            fstate = layer.states[state]
            if not fstate.geometry:
                continue
            p = Pose(fstate.geometry.coords[0])
            node = add_vertex_for_state(layer,
                                        graph,
                                        visibility_graph,
                                        s,
                                        p,
                                        reverse=True)
            visibility_graph.add_edge(node, state_set, weight=0)
    return state_set


def prepare_planning(
    layer: Layer,
    graph: Graph,
    dual_graph: DualGraph,
    visibility_graph: VisibilityGraph,
    start: PlanVertexUnion,
    end: PlanVertexUnion,
    states: Collection[GraphNode] = [],
    transitions: Collection[GraphEdge] = [],
) -> Tuple[VisibilityGraph, VisibilityNode, VisibilityNode]:

    states, transitions = compute_states_and_transitions(
        dual_graph, start, end, states, transitions)
    S = visibility_subgraph(visibility_graph, states, transitions)

    node_1: Optional[VisibilityNode] = None
    node_2: Optional[VisibilityNode] = None
    state_1: Optional[GraphNode] = None
    state_2: Optional[GraphNode] = None
    p_1: Optional[Pose] = None
    p_2: Optional[Pose] = None
    if isinstance(start, StateSet):
        node_1 = node_for_start_state_set(
            layer, graph, S, visibility_graph.nodes_for_transition, start)
    elif isinstance(start, str):
        state_1 = start
        fstate = layer.states[start]
        if not fstate.geometry:
            raise ValueError("No position")
        p_1 = Pose(fstate.geometry.coords[0])
        node_1 = add_vertex_for_state(layer, graph, S, state_1, p_1)
    elif isinstance(start, tuple):
        if start in S.dual:
            node_1 = visibility_graph.nodes_for_transition[start][0]
    else:
        state = layer.state_with_position(start.position)
        if state:
            state_1 = state.id
            node_1 = add_vertex_for_state(layer, graph, S, state_1, start)
            p_1 = start

    if isinstance(end, StateSet):
        node_2 = node_for_end_state_set(layer, graph, S,
                                        visibility_graph.nodes_for_transition,
                                        end)
    elif isinstance(end, str):
        state_2 = end
        fstate = layer.states[end]
        if not fstate.geometry:
            raise ValueError("No position")
        p_2 = Pose(fstate.geometry.coords[0])
        node_2 = add_vertex_for_state(layer,
                                      graph,
                                      S,
                                      state_2,
                                      p_2,
                                      reverse=True)
    elif isinstance(end, tuple):
        if end in S.dual:
            # we assume that v2 is a transition specified by (s1, s2, t_id)
            node_2 = visibility_graph.nodes_for_transition[end][0]
    else:
        state = layer.state_with_position(end.position)
        if state:
            state_2 = state.id
            node_2 = add_vertex_for_state(layer,
                                          graph,
                                          S,
                                          state_2,
                                          end,
                                          reverse=True)
            p_2 = end

    if state_1 and state_1 is state_2:
        # return straight line
        if p_1 == p_2:
            node_2 = node_1
        elif p_1 and p_2:
            a = angle(np.asarray(p_2.position) - np.asarray(p_1.position))
            cell = layer.states[state_1].duality
            if cell and cell.geometry:
                S.add_edge(node_1,
                           node_2,
                           cell=cell.geometry,
                           id=state_1,
                           weight=p_1.distance(p_2),
                           angle=a)

    if not (node_1 and node_1 in S and node_2 and node_2 in S):
        raise ValueError("Failed to prepare the graph")

    return S, node_1, node_2


class Planner(DualPlanner):
    """docstring for VisibilityPlanner"""

    fixed: Set[GraphEdge]

    def __init__(self,
                 layer: Layer,
                 filter_states: Optional[StateFilter] = None,
                 min_width: float = 0):
        super().__init__(layer, filter_states=filter_states)
        self.min_width = min_width
        self.H, self.fixed = init_visibility_graph(
            layer,
            dual_graph=self.D,
            transition_neighbors=self.transition_neighbors,
            min_width=min_width)

    def visibility_plan(
        self,
        start: PlanVertexUnion,
        end: PlanVertexUnion,
        states: Collection[GraphNode] = [],
        transitions: Collection[GraphEdge] = [],
        graph: Optional[Graph] = None,
    ) -> Tuple[VisibilityGraph, VisibilityRoute]:
        S, node_1, node_2 = prepare_planning(layer=self.layer,
                                             graph=graph or self.G,
                                             dual_graph=self.D,
                                             visibility_graph=self.H,
                                             start=start,
                                             end=end,
                                             states=states,
                                             transitions=transitions)
        path = nx.shortest_path(S, node_1, node_2, 'weight')
        if isinstance(start, StateSet):
            path.pop(0)
        if isinstance(end, StateSet):
            path.pop()
        return S, path

    def add_visibility_plan(self,
                            plan: Plan,
                            graph: Optional[Graph] = None) -> None:
        start: Optional[PlanVertexUnion] = None
        end: Optional[PlanVertexUnion] = None
        if plan.vertices:
            start = plan.vertices[0].primary
            end = plan.vertices[1].primary
        if not (start and end):
            raise ValueError("No vertices")
        plan.visibility_plan = self.visibility_plan(
            start,
            end,
            states=graph.nodes() if graph else [],
            transitions=plan.transitions,
            graph=graph)

    def shortest_path(self,
                      start: PlanVertexUnion,
                      end: PlanVertexUnion,
                      graph: Optional[Graph] = None,
                      weights: Dict[str, float] = {},
                      path_class: Type[SelfPlan] = Plan) -> SelfPlan:
        if weights:
            plan = DualPlanner.shortest_path(self,
                                             start,
                                             end,
                                             graph=graph,
                                             weights=weights,
                                             path_class=path_class)
        else:
            plan = path_class(self, start, end)
        self.add_visibility_plan(plan, graph=graph)
        return plan

    def all_paths(
            self,
            start: PlanVertexUnion,
            end: PlanVertexUnion,
            number: int = -1,
            graph: Optional[nx.Graph] = None,
            path_class: Type[SelfPlan] = Plan
    ) -> Generator[SelfPlan, None, None]:
        for plan in super().all_paths(start, end, number, graph, path_class):
            self.add_visibility_plan(plan)
            yield plan


"""
    def multiobjective_path(self, start, end, G=None, weights=None,
                            mtol=1, itol=0.1):
        ps = DualPlanner.multiobjective_path(self, start, end, G=G,
                                             weights=weights, mtol=mtol,
                                             itol=itol)
        for p in ps:
            self.add_visibility_plan(p)
        return ps
"""
