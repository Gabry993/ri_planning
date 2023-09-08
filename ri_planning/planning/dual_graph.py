import collections
import itertools
from typing import Any, Collection, Dict, List, Tuple, Union, Callable

import networkx as nx
import numpy as np

from ..map.map import Graph, GraphNode, GraphEdge, Layer, Map, State
from ..utilities import Vector, Pose, angle, normalize, orient
from .state_set import StateSet

DualGraph = Any
DualNode = Union[GraphEdge, Tuple[GraphNode, Pose], StateSet]
DualEdgeWithData = Tuple[DualNode, DualNode, Dict[str, Any]]


def init_dual_graph(graph: Graph) -> DualGraph:
    dual_graph = nx.line_graph(nx.MultiDiGraph(graph))
    # remove transitions self loops: (a, b, c) <-> (b, a, c)
    # add_edge_length_to_dual_graph(self.layer, self.D)
    self_loops = [(n1, n2, k) for n1, n2, k in dual_graph.edges(keys=True)
                  if n1[2] is n2[2]]
    dual_graph.remove_edges_from(self_loops)
    dual_graph.edges_for_state = collections.defaultdict(list)
    for edge in dual_graph.edges(data=True):
        (_, state, _), _, _ = edge
        dual_graph.edges_for_state[state].append(edge)
    return dual_graph


def dual_subgraph(dual_graph: DualGraph, states: Collection[GraphNode],
                  transitions: Collection[GraphEdge]) -> DualGraph:
    if states:
        edges = [dual_graph.edges_for_state[s] for s in states]
        edges = list(itertools.chain(*edges))
        S = nx.DiGraph(edges)
    else:
        S = nx.DiGraph(dual_graph)
    if transitions:
        nodes = transitions
        # nodes = list(itertools.chain(*nodes))
        S = S.subgraph(nodes)
    if states:
        for n in S.nodes():
            S._node[n] = dual_graph._node[n]
    S.edges_for_state = collections.defaultdict(list)
    for e in S.edges(data=True):
        s = e[0][1]
        S.edges_for_state[s].append(e)
    # S.nodes_for_transition = collections.defaultdict(list)
    return S


def add_properties(layer: Layer, dual_graph: DualGraph) -> None:
    add_edge_length(layer, dual_graph)
    add_node_angle(layer, dual_graph)
    add_edge_angle(layer, dual_graph)
    add_edge_diversity(layer, dual_graph)
    add_edge_choice(layer, dual_graph)


# LENGTH


def add_edge_length(layer: Layer, dual_graph: DualGraph) -> None:
    for edge in dual_graph.edges(data=True):
        (_, _, ti1), (_, _, ti2), data = edge
        t1_geo = layer.transitions[ti1].geometry
        t2_geo = layer.transitions[ti2].geometry
        if t1_geo and t2_geo:
            p1 = t1_geo.centroid.coords
            p2 = t2_geo.centroid.coords
            le = np.linalg.norm(np.asarray(p1) - np.asarray(p2))
            data['length'] = le


# ORIENTATION


def get_boundary(layer: Layer,
                 edge: GraphEdge) -> Tuple[Tuple[Vector, Vector], float]:
    """
    :param      layer:  The layer
    :param      edge:   The edge
    :type       edge:   DualNode

    :returns:   The two extrema and the normal angle of the boundary corresponding to the an edge
    """
    s_id, _, t_id = edge
    state = layer.states[s_id]
    boundary = layer.transitions[t_id].duality
    if not state.geometry or not boundary:
        raise ValueError(f"No valid orientation for edge {edge}")
    a = np.asarray(state.geometry.coords[0])
    b, c = np.asarray(boundary.geometry.coords[:2])
    if orient(np.asarray([a, b, c])) > 0:
        b, c = c, b
    e = c - b
    return (b, c), angle(e) + np.pi * 0.5


def add_node_angle(layer: Layer, dual_graph: DualGraph) -> None:
    for e, d in dual_graph.nodes(data=True):
        vs, angle = get_boundary(layer, e)
        d['angle'] = angle
        d['vs'] = vs


def best_beta(beta_m: float, beta_M: float, gamma: float) -> float:
    if gamma < 0:
        if beta_M < gamma:
            return gamma - 2 * beta_M
        elif beta_m > 0:
            return -gamma + 2 * beta_m
        else:
            return -gamma
    else:
        if beta_M < 0:
            return gamma - 2 * beta_M
        elif beta_m > gamma:
            return -gamma + 2 * beta_m
        else:
            return gamma


def get_minimal_angle(layer: Layer, dual_edge: DualEdgeWithData,
                      dual_graph: DualGraph) -> float:
    # TODO distinguish between constrained and not constrained
    edge1, edge2 = dual_edge[:2]
    d1 = dual_graph._node[edge1]
    d2 = dual_graph._node[edge2]
    alpha_1 = d1['angle']
    alpha_2 = d2['angle']
    v1s = d1['vs']
    v2s = d2['vs']
    # alpha_1 = angle(v1s[1] - v1s[0]) + math.pi * 0.5
    # alpha_2 = angle(v2s[1] - v2s[0]) + math.pi * 0.5
    gamma = normalize(alpha_2 - alpha_1)
    betas = sorted(
        [normalize(angle(v2 - v1) - alpha_1) for v1 in v1s for v2 in v2s])
    return best_beta(betas[0], betas[-1], gamma)


def add_edge_angle(layer: Layer, dual_graph: DualGraph) -> None:
    for e in dual_graph.edges(data=True):
        e[2]['angle'] = get_minimal_angle(layer, e, dual_graph)


def delta_angle_from_vertex(layer: Layer,
                            pose: Pose,
                            edge: GraphEdge,
                            dual_graph: DualGraph,
                            direction: int = 1) -> float:
    # TODO distinguish between constrained and not constrained
    d = dual_graph._node[edge]
    alpha_e = d['angle']
    alpha = pose.orientation
    v = pose.position
    vs = d['vs']
    # alpha_e = angle(vs[1] - vs[0]) + math.pi * 0.5
    if alpha is not None:
        gamma = normalize(direction * (alpha_e - alpha))
        betas = sorted(
            [normalize(direction * (angle(ve - v) - alpha)) for ve in vs])
        beta_m = betas[0]
        beta_M = betas[-1]
        return best_beta(beta_m, beta_M, gamma)
    else:
        betas = [normalize(angle(ve - v) - alpha_e) for ve in vs]
        if betas[0] * betas[1] <= 0:
            return 0
        else:
            return min(abs(b) for b in betas)


# DIVERSITY


def connection_spaces(layer: Layer) -> List[State]:
    return [
        state for state in layer.states.values()
        if (state.duality and state.duality.type == 'ConnectionSpace')
    ]


def connects_different_types(layer: Layer, state_id: str) -> bool:
    if state_id not in layer.states:
        return False
    cells = [layer.states[state].duality for state in layer.graph[state_id]]
    types = [cell.type for cell in cells if cell]
    if len(types) == 2 and types[0] != types[1]:
        return True
    else:
        return False


def doors_between_different_types(layer: Layer) -> List[str]:
    return [
        state.id for state in connection_spaces(layer)
        if connects_different_types(layer, state.id)
    ]


def diversity(m: Map, state_1: State, state_2: State,
              gates_id: List[str]) -> int:
    geo_layer = m.geometric_layer
    if not geo_layer:
        return 0
    geo_state_1 = state_1.in_layer(geo_layer)
    geo_state_2 = state_2.in_layer(geo_layer)
    if not geo_state_1 or not geo_state_2:
        return 0

    if geo_state_1 is geo_state_2:
        return 0

    if geo_state_1.id in gates_id:
        return 1

    cell_1 = geo_state_1.duality
    cell_2 = geo_state_2.duality
    if not (cell_1 and cell_2):
        return 0
    if cell_1.type == cell_2.type:
        return 0
    if 'ConnectionSpace' in [cell_1.type, cell_2.type]:
        return 0
    return 1


def add_edge_diversity(layer: Layer, dual_graph: DualGraph) -> None:
    gates = doors_between_different_types(layer)
    for (s1, s2, _), _, data in dual_graph.edges(data=True):
        data['diversity'] = diversity(layer.map, layer.states[s1],
                                      layer.states[s2], gates)


# CHOICES

# Do not choices for edges added to S.


def choices_for_state(state: State) -> int:
    return max(len(state.connects) - 2, 0)


def add_edge_choice(layer: Layer, dual_graph: DualGraph) -> None:
    for (s1, s2, _), _, data in dual_graph.edges(data=True):
        # same as nx.degree(planner.G)[s2]
        data['choices'] = choices_for_state(layer.states[s2])


# WEIGHTS

"""

## (value, _ -> (value, name, value), name, )
AttributeDescription = Tuple[float, str, float]
AttributeCostDescription = Tuple[float, Callable[[Any], AttributeDescription], str, float]
StateCostDescription = List[Tuple[str, List[AttributeCostDescription]]]
StateCostFn = Callable[..., float]
StateCost = Tuple[StateCostFn, StateCostDescription]
StateCosts = Dict[str, StateCost]
SubCostDescription = Tuple[float, str, AttributeDescription, float]
CostDescription = Tuple[float, str, float, List[SubCostDescription]]


def update_weighted_cost(state_costs: StateCosts, edge_data: Dict[str, Any], weights: Dict[str, float]
                         ) -> Tuple[float, CostDescription]:
    data: Dict[str, CostDescription] = {}
    for k, (v, desc) in state_costs.items():
        sv = v(**edge_data)
        ds: List[SubCostDescription] = sum(
            [[(edge_data.get(q, 0) * l[0], l[2], l[1](edge_data),
               edge_data.get(q, 0) * l[3]) for l in ls]
             for (q, ls) in desc], [])
        data[k] = (sv * weights.get(k, 0), k, sv, ds)
    cost = sum(data[k][0] for k in state_costs)
    description: CostDescription = (cost, 'cost', cost, [data[k] for k in state_costs])
    return cost, description


def update_costs(state_costs: StateCosts, edge_data: Dict[str, Any]
                 ) -> Dict[str, Tuple[float, CostDescription]]:
    data: Dict[str, Tuple[float, CostDescription]] = {}
    for k, (cost_fn, desc) in state_costs.items():
        cost = cost_fn(**edge_data)
        ds = sum([[(edge_data.get(q, 0) * l[0], l[2], l[1](edge_data),
                   edge_data.get(q, 0) * l[3]) for l in ls]
                 for (q, ls) in desc], [])
        data[k] = (cost, (cost, k, cost, ds))
    return data


def update_weights(dual_graph: DualGraph, graph: Graph,
                   weights: Union[List[str], Dict[str, float]]) -> None:

    # DONE mapping state, edge -> weight (state) * length(edge)
    # TODO add mapping state, edge -> weight (edge) [e.g. angles]
    # TODO add mapping state, edge -> weight (state) * 1 [e.g. degree]
    # i.e. in general
    # weight(edge, type) = weight(state, type) * weight(edge, type)
    # where sometimes
    # - weight(state, type) = 1
    # - weight(edge, type) = 1
    # - weight(edge, type) = len(edge)
    # weight(edge) = sum_k weights[k] * weight(edge, k)
    # G.node[s] could be functions that maps edge to weight
    for state, edges in list(dual_graph.edges_for_state.items()):
        # state = self.layer.states[s]
        # state_weights = {w: state.__getattribute__(w) for w in weights}
        if state not in graph:
            # DONE: should never ever happen
            raise ValueError(f"State {state} is not in graph")
            # CHANGED, it means that the state is not traversable
            # -> remove it
            # for e in es:
            #     dual_graph.remove_edges_from(es)
            # continue
        # CHANGED: costs are now function to be applied on the edge data
        state_weights = {w: graph._node[state].get(w, 0) for w in weights}
        # print 'state_weights', state_weights
        for _, _, data in edges:
            # l = data['length']
            if isinstance(weights, dict):
                # a dictionary -> single objective
                _data = {}
                for k, (v, desc) in state_weights.items():
                    sv = v(**data)
                    # [(data[l] * w, w, l) for [for l,es in desc]]
                    d: List[Tuple[float, float, float, float]] = sum(
                        [[(data.get(q, 0) * l[0], l[2], l[1](data),
                           data.get(q, 0) * l[3]) for l in ls]
                         for (q, ls) in desc], [])
                    _data[k] = (sv * weights.get(k, 0), k, sv, d)
                w = sum(_data[k][0] for k in state_weights)
                data['weight'] = w
                data['desc'] = (w, 'cost', w,
                                [_data[k] for k in state_weights])
                # print 'data[desc]', data['desc']
                # data['weight'] = sum([v(**data) * weights[k]
                #                       for k, v in state_weights.items()])
            else:
                # a list -> multiobjective
                data['desc'] = []
                for k, (v, desc) in state_weights.items():
                    sv = v(**data)
                    d = sum([[(data.get(q, 0) * l[0], l[2], l[1](data),
                               data.get(q, 0) * l[3]) for l in ls]
                             for (q, ls) in desc], [])
                    # data[k] = v(**data)
                    data[k] = sv
                    data['desc'].append((sv, k, sv, d))
"""

def add_vertex(
        layer: Layer,
        graph: Graph,
        dual_graph: DualGraph,
        state: GraphNode,
        pose: Pose,
        reverse: bool = False) -> DualNode:
    """
    { item_description }
    """
    node: DualNode = (state, pose)
    dual_graph.add_node(node)
    for neighbor_state, transitions in graph[state].items():
        for transition_id in transitions:
            boundary = layer.transitions[transition_id].duality
            if not boundary:
                continue
            p = boundary.geometry.centroid.coords
            length = np.linalg.norm(np.asarray(p) - np.asarray(pose.position))
            choices = choices_for_state(layer.states[state])
            data = {'length': length, 'choices': choices, 'diversity': 0}
            if reverse:
                neighbor_node = (neighbor_state, state, transition_id)
                edge: DualEdgeWithData = (neighbor_node, node, data)
            else:
                neighbor_node = (state, neighbor_state, transition_id)
                edge = (node, neighbor_node, data)
            data['angle'] = delta_angle_from_vertex(
                layer, pose, neighbor_node, dual_graph, -1 if reverse else 1)
            dual_graph.add_edges_from([edge])
            dual_graph.edges_for_state[state].append(
                (edge[0], edge[1], dual_graph[edge[0]][edge[1]]))
    return node
