# import collections
# import itertools
from typing import (Any, Collection, Dict, List, Optional, Set, Tuple, Union,
                    cast, Type, TypeVar)

import networkx as nx
import numpy as np
import shapely as s

from ..map.map import Graph, GraphEdge, GraphNode, Layer
from ..utilities import Pose, Vector, angle, normalize
from .dual_graph import (DualGraph, DualNode, add_properties, add_vertex,
                         choices_for_state, dual_subgraph)
from .planner import Plan as BasePlan
from .planner import Planner as BasePlanner
from .planner import PlanVertex, PlanVertexUnion, StateFilter, StateSet

SelfPlan = TypeVar("SelfPlan", bound="Plan")

"""MO
def total_desc(ds):
    d = ds[0]
    for d1 in ds[1:]:
        d = add_desc(d, d1)
    return d


def add_desc(d1, d2):
    if type(d1) == ListType:
        return [add_desc(f1, f2) for f1, f2 in zip(d1, d2)]
    else:
        pass

    t1, l1, c1, s1 = d1
    t2, l2, c2, s2 = d2

    assert l1 == l2

    if type(s1) == ListType:
        return (t1 + t2, l1, c1 + c2,
                [add_desc(su1, su2) for su1, su2 in zip(d1[3], d2[3])])
    else:
        return (t1 + t2, l1, max(c1, c2), s1 + s2)


# HACK
def compute_costs(path, D, weights):
    # if weights:
    #     es = [D[s1][s2] for s1, s2 in zip(path[:-1], path[1:])]
    #     return {w: sum([e[w] for e in es if w in e]) for w in weights}
    # else:
    #     return {}
    # print 'compute_costs'
    ps = zip(path[:-1], path[1:])
    # print ps
    es = [D[p1][p2] for p1, p2 in ps]
    # print es
    descs = [d['desc'] for d in es]
    return total_desc(descs)
"""


def delta_angle_between_poses(p1: Pose, p2: Pose) -> float:
    alpha_1 = p1.orientation
    alpha_2 = p2.orientation
    v1 = np.asarray(p1.position)
    v2 = np.asarray(p2.position)
    alpha = angle(v2 - v1)
    r = 0.0
    if alpha_1 is not None:
        r = r + abs(normalize(alpha - alpha_1))
    if alpha_2 is not None:
        r = r + abs(normalize(alpha - alpha_2))
    return r


def add_edge_in_state(layer: Layer, dual_graph: DualGraph, state: str,
                      p1: Pose, p2: Pose) -> None:
    node_1 = (state, p1)
    node_2 = (state, p2)
    data = {
        'length': p1.distance(p2),
        'diversity': 0,
        'choices': choices_for_state(layer.states[state]),
        'angle': delta_angle_between_poses(p1, p2)
    }
    e = (node_1, node_2, data)
    dual_graph.add_edges_from([e])
    dual_graph.edges_for_state[s].append((e[0], e[1], dual_graph[e[0]][e[1]]))


# OLD COMMENT
#
# QUESTION: don't know why I added these edges
# for the visibility graph
# ANSWER: on_border is a flag to search
# for a path upto the border or
# to the center of one of the members.
def node_for_start_state_set(
        layer: Layer,
        graph: Graph,
        dual_graph: DualGraph,
        state_set: StateSet,
        weights: Union[List[str], Dict[str, float]] = []) -> DualNode:

    if state_set.on_border:
        for e in state_set.out_edges:
            dual_graph.add_edge(state_set, e, **{k: 0 for k in weights})

    else:
        for state in state_set.states:
            g = layer.states[s].geometry
            if not g:
                continue
            node = add_vertex(layer, graph, dual_graph, state,
                              Pose(g.coords[0]))
            dual_graph.add_edge(state_set, node, **{k: 0 for k in weights})
    return state_set


def node_for_end_state_set(
        layer: Layer,
        graph: Graph,
        dual_graph: DualGraph,
        state_set: StateSet,
        weights: Union[List[str], Dict[str, float]] = []) -> DualNode:
    if state_set.on_border:
        for e in state_set.in_edges:
            dual_graph.add_edge(e, state_set, **{k: 0 for k in weights})
    else:
        for state in state_set.states:
            g = layer.states[s].geometry
            if not g:
                continue
            node = add_vertex(layer,
                              graph,
                              dual_graph,
                              state,
                              Pose(g.coords[0]),
                              reverse=True)
            dual_graph.add_edge(node, state_set, **{k: 0 for k in weights})
    return state_set


def compute_states_and_transitions(
    dual_graph: DualGraph,
    start: PlanVertexUnion,
    end: PlanVertexUnion,
    states: Collection[GraphNode] = [],
    transitions: Collection[GraphEdge] = []
) -> Tuple[Collection[GraphNode], Collection[GraphEdge]]:
    internal_transitions: Set[GraphEdge] = set()
    if isinstance(start, StateSet):
        internal_transitions |= start.internal_edges
    if isinstance(end, StateSet):
        internal_transitions |= end.internal_edges
    if internal_transitions:
        if not transitions:
            transitions = set(dual_graph.nodes()) - internal_transitions
        else:
            transitions = set(transitions) - internal_transitions
    return states, transitions


def prepare_planning(
    layer: Layer,
    graph: Graph,
    dual_graph: DualGraph,
    start: PlanVertexUnion,
    end: PlanVertexUnion,
    states: Collection[GraphNode] = [],
    transitions: Collection[GraphEdge] = [],
    weights: Union[List[str], Dict[str, float]] = [],
) -> Tuple[DualGraph, DualNode, DualNode]:
    """
        1. if start/end are sets of states: we remove the internal edges from the dual graph nodes
        2. we limit to the relevant subgraph
        3. find the start/end nodes:
            a. if state set -> connect them to the graph [TODO check] => StateSet
            b. if transition -> already connected => GraphEdge
            c. if state -> connect them to the graph => (State, Pose)
            d. if pose -> connect them to the graph => (State, Pose)
        4. if start and end are in the same state -> connect them by a straight line
        5. update the weights

    :param      layer:        The layer
    :param      graph:        The graph
    :param      dual_graph:   The dual graph
    :param      v1:           The v 1
    :param      v2:           The v 2
    :param      states:       The states
    :param      transitions:  The transitions
    :param      weights:      The weights
    :param      a1:           A 1
    :param      a2:           A 2

    :returns:   { description_of_the_return_value }
    """

    # TODO do not add to the whole graph (only to traversable vertices)

    # 1
    states, transitions = compute_states_and_transitions(dual_graph, start, end, states, transitions)
    # 2
    S = dual_subgraph(dual_graph, states, transitions)
    # 3
    node_1: Optional[DualNode] = None
    node_2: Optional[DualNode] = None
    state_1: Optional[GraphNode] = None
    state_2: Optional[GraphNode] = None
    p_1: Optional[Pose] = None
    p_2: Optional[Pose] = None
    if isinstance(start, StateSet):
        node_1 = node_for_start_state_set(layer, graph, S, start, weights)
    elif isinstance(start, str):
        state_1 = start
        g = layer.states[state_1].geometry
        if not g:
            raise ValueError("No geometry")
        node_1 = add_vertex(layer, graph, S, start, Pose(g.coords[0]))
    elif isinstance(start, tuple):
        node_1 = start
    else:
        p_1 = start
        state = layer.state_with_position(start.position)
        if state:
            state_1 = state.id
            node_1 = add_vertex(layer, graph, S, state_1, start)

    if isinstance(end, StateSet):
        node_2 = node_for_end_state_set(layer, graph, S, end, weights)
    elif isinstance(end, str):
        state_2 = end
        g = layer.states[end].geometry
        if not g:
            raise ValueError("No geometry")
        node_2 = add_vertex(layer,
                            graph,
                            S,
                            end,
                            Pose(g.coords[0]),
                            reverse=True)
    elif isinstance(end, tuple):
        node_2 = end
    else:
        p_2 = end
        state = layer.state_with_position(end.position)
        if state:
            state_2 = state.id
            node_2 = add_vertex(layer, graph, S, state_2, end, reverse=True)

    # 4
    if state_1 and state_1 is state_2:
        # return straight line
        if p_1 == p_2:
            node_2 = node_1
        elif p_1 and p_2:
            add_edge_in_state(layer, S, state_1, p_1, p_2)

    if not (node_1 and node_1 in S and node_2 and node_2 in S):
        raise ValueError("Failed to prepare the graph")
    # 5
    """
    if weights:
        update_weights(S, graph, weights)
    """
    return S, node_1, node_2


def position_for_node(layer: Layer, node: DualNode) -> Vector:
    if isinstance(node, StateSet):
        coords = np.asarray([
            geo.coords[0]
            for geo in [layer.states[s].geometry for s in node.states] if geo
        ])
        return cast(np.ndarray, np.mean(coords, 0))
    elif len(node) == 2:
        _, pose = cast(Tuple[GraphNode, Pose], node)
        return pose.position
    else:
        _, _, t_id = cast(GraphEdge, node)
        if t_id in layer.transitions:
            transition = layer.transitions[t_id]
            if transition.duality:
                return np.asarray(
                    transition.duality.geometry.centroid.coords[0])
        raise ValueError(f"No position for transition {t_id}")


class Plan(BasePlan):

    dgraph: Optional[DualGraph]
    dplan: Optional[List[DualNode]]

    def __init__(self,
                 planner: Optional['Planner'] = None,
                 start: Optional[PlanVertexUnion] = None,
                 end: Optional[PlanVertexUnion] = None,
                 vertices: Optional[Tuple[PlanVertex, PlanVertex]] = None,
                 agent: Any = None):
        super().__init__(planner, start, end, vertices, agent)
        self.dgraph = self.dplan = None

    @property
    def has_dual_plan(self) -> bool:
        return self.dplan is not None

    @property
    def dual_plan(
            self) -> Tuple[Optional[DualGraph], Optional[List[DualNode]]]:
        return (self.dgraph, self.dplan)

    @dual_plan.setter
    def dual_plan(self, value: Tuple[DualGraph, List[DualNode]]) -> None:
        self.dgraph, self.dplan = value
        if self.layer:
            if len(self.dplan) > 1:
                # print(self.dplan)
                self.geometry = s.LineString([
                    position_for_node(self.layer, node) for node in self.dplan
                ])
                if self.geometry:
                    self.costs = np.array([self.geometry.length])
        if not self.transitions:
            self._transitions = [
                node_ for node_ in self.dplan
                if (len(node_) == 3 and node_[2] in self.layer.transitions)
            ]
        if not self.states:
            # TODO check
            self.states = [
                node_[0][1] if len(node_) == 2 else node_[1]
                for node_ in self.dplan[:-1]
            ]


class Planner(BasePlanner):
    """
    """

    def __init__(self,
                 layer: Layer,
                 filter_states: Optional[StateFilter] = None):
        super().__init__(layer, filter_states=filter_states)
        add_properties(self.layer, self.D)

    def dual_plan(
        self,
        start: PlanVertexUnion,
        end: PlanVertexUnion,
        states: Collection[GraphNode] = [],
        transitions: Collection[GraphEdge] = [],
        graph: Optional[Graph] = None,
        weights: Union[List[str], Dict[str, float]] = []
    ) -> Tuple[DualGraph, List[DualNode]]:
        S, node_1, node_2 = prepare_planning(layer=self.layer,
                                             graph=graph or self.G,
                                             dual_graph=self.D,
                                             start=start,
                                             end=end,
                                             states=states,
                                             transitions=transitions,
                                             weights=weights)
        weight = 'weight' if weights else 'length'
        path = nx.shortest_path(S, node_1, node_2, weight)
        if isinstance(start, StateSet):
            path.pop(0)
        if isinstance(end, StateSet):
            path.pop()
        return S, path

    def add_dual_plan(
            self,
            plan: Plan,
            graph: Optional[Graph] = None,
            weights: Union[List[str], Dict[str, float]] = []) -> None:
        start: Optional[PlanVertexUnion] = None
        end: Optional[PlanVertexUnion] = None
        if plan.vertices:
            start = plan.vertices[0].primary
            end = plan.vertices[1].primary
        if not (start and end):
            raise ValueError("No vertices")
        plan.dual_plan = self.dual_plan(start=start,
                                        end=end,
                                        states=graph.nodes() if graph else [],
                                        transitions=plan.transitions,
                                        graph=graph,
                                        weights=weights)
        """ MO
        S, p = plan.dual_plan
        if G:
            plan._mo_costs = compute_costs(p, S, weights)
        """

    def shortest_path(self,
                      start: PlanVertexUnion,
                      end: PlanVertexUnion,
                      graph: Optional[Graph] = None,
                      weights: Dict[str, float] = {},
                      path_class: Type[SelfPlan] = Plan) -> SelfPlan:
        # print('shortest_path', start, end, G.nodes())
        plan = path_class(self, start, end)
        self.add_dual_plan(plan, graph=graph, weights=weights)
        return plan


"""MO
    def multiobjective_path(self,
                            start: PlanVertexArg,
                            end: PlanVertexArg,
                            G: Optional[Graph] = None,
                            weights: List[str] = [],
                            mtol: float = 1,
                            itol: float = 0.1,
                            max_i: int = -1):
        plan = Plan(self, start, end)
        A, B = [(v.position or v.transition or v.state) for v in plan.vertices]
        if G:
            states = G.nodes()
        else:
            states = None
        S, node_1, node_2 = self.prepare_planning(A,
                                                B,
                                                states=states,
                                                transitions=plan.transitions,
                                                weights=weights,
                                                G=G)

        ps = {
            w: nx.shortest_simple_paths(S, node_1, node_2, weight=w)
            for w in weights
        }

        pareto_optimal_set = pareto_set(plan,
                                        ps,
                                        S,
                                        mtol=mtol,
                                        itol=itol,
                                        max_i=max_i)
        return pareto_optimal_set
"""
