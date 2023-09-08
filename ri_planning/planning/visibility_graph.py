import collections
import itertools
from typing import Any, Collection, Dict, List, Set, Tuple, Union, cast

import networkx as nx
import numpy as np
import shapely as s
from shapely import ops

from ..map.map import Boundary, Cell, Graph, GraphEdge, Layer, Transition
from ..utilities import Vector, Pose, angle, are_parallel, n_grams, normalize
from .dual_graph import DualGraph, DualNode
from .state_set import StateSet

VisibilityGraph = Any
VisibilityNode = Union[Tuple[DualNode, Pose], StateSet]
VisibilityEdge = Tuple[VisibilityNode, VisibilityNode]
VisibilityRoute = List[Tuple[DualNode, Pose]]


def cross_corridor(transition: Transition) -> bool:
    border = transition.duality
    if not border:
        return False
    for cell in border.cells:
        if cross_corridor_in_cell(border, cell):
            return True
    return False


# TODO(Jerome 2023): dcel
def cross_corridor_in_cell(border: Boundary, cell: Cell) -> bool:
    chain = border.chainInCell(cell)
    if not chain:
        print(border, cell)
        raise NameError("No chain?")
    start, end = chain
    e1 = start.next
    e2 = end.previous
    beta = angle(np.asarray(start.coords[1]) - np.asarray(start.coords[0]))
    for e in [e1, e2]:
        b = e.face_boundary
        if b.duality:
            return False
        alpha = angle(np.asarray(e.coords[1]) - np.asarray(e.coords[0]))
        delta = abs(normalize(alpha - beta)) - np.pi / 2
        if abs(delta) > 0.2:
            return False
    return True


def is_visible(border: Boundary, polygon: s.Polygon, vertex: Vector) -> bool:
    try:
        pp = s.Polygon([vertex] + list(border.geometry.coords))
        pp = pp.buffer(0)
        if pp.is_valid:
            po = pp.intersection(polygon)
            return cast(bool, po.geom_type == 'Polygon')
    except Exception as e:
        print(e)
    return True


def transition_is_anti_door(
        transition: Transition,
        transition_neighbors: Dict[str, List[Transition]]) -> bool:
    if not transition.duality:
        return False
    border = transition.duality.geometry
    doors = [
        d for d in transition_neighbors[transition.id]
        if transition_is_connection_space_boundary(d)
    ]
    for door in doors:
        if not door.duality or not door.duality.geometry:
            continue
        prop = door.duality.geometry.length / border.length
        if (prop > 0.9 and prop < 1.1
                and are_parallel(door.duality.geometry, border)):
            return True
    return False


def transition_should_have_fixed_position(transition: Transition,
                                          transition_neighbors: Dict[
                                              str, List[Transition]],
                                          min_width: float) -> bool:
    return (transition_is_connection_space_boundary(transition)
            or transition_is_anti_door(transition, transition_neighbors)
            or (transition.duality is not None and transition.duality.geometry
                and transition.duality.geometry.length < min_width
                and transition_is_across_corridor(transition)))


# HACK: else cannot fix borders on df map
def transition_is_across_corridor(transition: Transition) -> bool:
    return True or cross_corridor(transition)


def transition_is_connection_space_boundary(transition: Transition) -> bool:
    return any(state.duality.type == 'ConnectionSpace'
               for state in transition.connects if state.duality)


def init_visibility_graph(
        layer: Layer, dual_graph: DualGraph,
        transition_neighbors: Dict[str, List[Transition]],
        min_width: float) -> Tuple[VisibilityGraph, Set[GraphEdge]]:
    H = nx.DiGraph()
    fixed: Set[DualNode] = set()
    H.dual = dual_graph
    H.vertices = {}
    H.nodes_for_transition = collections.defaultdict(list)
    H.edges_for_state = collections.defaultdict(list)
    for node in dual_graph.nodes():
        s1, s2, t_id = node
        transition = layer.transitions[t_id]
        if not transition.duality:
            continue
        line = transition.duality.geometry
        if transition_should_have_fixed_position(transition,
                                                 transition_neighbors,
                                                 min_width):
            fixed.add(node)
            gamma = 0.5
            p0 = line.interpolate(gamma, normalized=True)
            p0 = Pose(p0.coords[0])
            H.vertices[node] = [p0]
            H.add_node((node, p0), **{
                'id': t_id,
                'gamma': gamma,
                'fixed': True
            })
            H.nodes_for_transition[node].append((node, p0))
        else:
            H.vertices[node] = [Pose(c) for c in line.coords]
            for v in H.vertices[node]:
                gamma = line.project(s.Point(v.position), normalized=True)
                H.add_node((node, v), **{'id': t_id, 'gamma': gamma})
                H.nodes_for_transition[node].append((node, v))

    for node in dual_graph.nodes():
        for v in H.vertices[node]:
            add_vertex_for_node(layer, H, node, v)
    return H, fixed


def visibility_subgraph(
        visibility_graph: VisibilityGraph,
        states: Collection[str] = [],
        transitions: Collection[GraphEdge] = []) -> VisibilityGraph:
    H = visibility_graph
    if states:
        edges = [H.edges_for_state[s] for s in states]
        edges = list(itertools.chain(*edges))
        S = nx.DiGraph(edges)
        if not transitions:
            transitions = [t for t in H.dual.nodes() if t[1] in states]
    else:
        S = nx.DiGraph(H)
    if transitions:
        nodes = [H.nodes_for_transition[t] for t in transitions]
        nodes = list(itertools.chain(*nodes))
        S = nx.DiGraph(S.subgraph(nodes))
        S.dual = nx.DiGraph(H.dual.subgraph(transitions))
    else:
        S.dual = nx.DiGraph(H.dual)
    if states:
        for n in S.nodes():
            S._node[n] = H._node[n]
    # #TODO:200 maybe slow
    S.vertices = {n: H.vertices[n] for n in S.dual}
    S.edges_for_state = collections.defaultdict(list)
    S.nodes_for_transition = collections.defaultdict(list)
    return S


def add_vertex_for_node(layer: Layer,
                        visibility_graph: VisibilityGraph,
                        node: DualNode,
                        v: Pose,
                        reverse: bool = False) -> None:
    checked: Dict[DualNode, bool] = {}
    if isinstance(node, tuple):
        if len(node) == 3:
            s2 = cast(str, node[1])
        else:
            s2 = cast(str, node[0])
    else:
        return
    L = visibility_graph.dual
    # G = self.G
    # the directed dual graph (built with line_graph(...) contains
    # also all edges ((A, B), (B, A)) which are not usefull for planning.
    # Let us discharge them.
    if reverse:
        neighbors = L.predecessors
        # self_loop = lambda m : m[0]==node[1]
        j = 0
    else:
        neighbors = L.successors
        # self_loop=lambda m : m[1]==node[0]
        j = 1
    # #CHANGED:10 removed self_loops from L directly
    cell = layer.states[s2].duality
    if cell and cell.geometry:
        lines = [([node, next_node], True, cell.geometry,
                  [node[0], node[1], next_node[1]])
                 for next_node in neighbors(node)]
    else:
        lines = []
    # if not self_loop(next_node)]
    while lines:
        nodes, first, inside, states = lines.pop(0)
        next_node = nodes[-1]
        if checked.get(next_node, False):
            continue
        checked[next_node] = True
        for w in visibility_graph.vertices[next_node]:
            line = s.LineString([v.position, w.position])
            r = first
            if not r:
                diff = line.difference(inside)
                r = diff.is_empty or diff.length < 0.1
            if r:
                nodes_in_line = [(node, v)]
                for node1, node2 in n_grams(nodes[:-1], 2):
                    s1, s2, t_id = node2
                    boundary = layer.transitions[t_id].duality
                    if not boundary:
                        continue
                    pz = boundary.geometry.intersection(line)
                    if pz.geom_type == 'GeometryCollection':
                        if len(pz.geoms):
                            z = pz.geoms[0]
                        else:
                            continue
                    if len(pz.coords) != 1:
                        # its a line,  like when when one of the faces
                        # is a triangles:
                        p0 = z.interpolate(0, normalized=True)
                        p1 = z.interpolate(1, normalized=True)
                        if not (p0 != s.Point(w.position) or p1 != s.Point(w.position)):
                            raise NameError("?")
                        z = w
                    else:
                        z = Pose(pz.coords[0])
                    gamma = boundary.geometry.project(s.Point(z.position),
                                                      normalized=True)
                    new_node = (node2, z)
                    if new_node not in visibility_graph:
                        visibility_graph.add_node(
                            new_node, **{
                                'id': t_id,
                                'gamma': gamma,
                                'intermediate': True
                            })
                        nodes0 = visibility_graph.nodes_for_transition[
                            new_node[0]]
                        nodes0.append(new_node)
                    nodes_in_line.append(new_node)
                nodes_in_line.append((next_node, w))
                if reverse:
                    nodes_in_line.reverse()
                for (node1, v1), (node2, v2) in n_grams(nodes_in_line, 2):
                    a = angle(np.asarray(v2.position) - np.asarray(v1.position))
                    if len(node1) == 2:
                        s1 = node1[0]
                    else:
                        s1 = node1[1]
                    cell = layer.states[s1].duality
                    if not cell or not cell.geometry:
                        continue
                    edge = ((node1, v1), (node2, v2), {
                        'id': s1,
                        'weight': v1.distance(v2),
                        'angle': a,
                        'cell': cell.geometry
                    })
                    assert (edge[0] in visibility_graph
                            and edge[1] in visibility_graph)
                    visibility_graph.add_edges_from([edge])
                    visibility_graph.edges_for_state[s1].append(edge)

            if first:
                visible = True
            else:
                try:
                    border = layer.transitions[next_node[2]].duality
                    visible = first or (border is not None
                                        and is_visible(border, inside, v.position))
                except:
                    # logging.error('EEEEE %s' % next_node)
                    visible = False

            if visible and len(visibility_graph.vertices[next_node]) > 1:
                for n in neighbors(next_node):

                    if (checked.get(n, False)
                            or (n[j] in states and not n[1] == n[0])):
                        continue
                    s_id = n[1 - j]
                    cell = layer.states[s_id].duality
                    if cell and cell.geometry:
                        inside = ops.unary_union([inside, cell.geometry])
                        lines.append(
                            (nodes + [n], False, inside, states + [n[j]]))


def add_vertex_for_state(layer: Layer,
                         graph: Graph,
                         visibility_graph: VisibilityGraph,
                         s1: str,
                         v: Pose,
                         reverse: bool = False) -> VisibilityNode:
    L_node = (s1, v)
    L = visibility_graph.dual
    H_node = (L_node, v)
    visibility_graph.add_node(H_node, **{'id': s1})
    transitions = []
    for s2, w in graph[s1].items():
        for t in w.keys():
            if reverse:
                transitions.append((s2, s1, t))
            else:
                transitions.append((s1, s2, t))

    L.add_node(L_node)
    visibility_graph.vertices[L_node] = [v]
    for t in transitions:
        if t not in L:
            continue
        if reverse:
            L.add_edge(t, L_node)
            for e in L.neighbors(t):
                if len(e) == 2:
                    L.add_edge(L_node, e)
                    L.add_edge(e, L_node)
        else:
            L.add_edge(L_node, t)

    add_vertex_for_node(layer, visibility_graph, L_node, v, reverse=reverse)
    return H_node
