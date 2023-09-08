from typing import List, Optional, Tuple, Set, Dict, Any, cast

import numpy as np
import shapely as s

from ..map.map import Layer, Transition, GraphEdge
from ..utilities import angle, avg_angle, n_grams, normalize, orient, Vector
from .bezier import bezier_node as bn
from .bezier import bezier_path as bp
from .visibility_graph import VisibilityRoute, VisibilityGraph


def is_convex(c: s.Polygon) -> bool:
    ch = c.convex_hull
    area_ch: float = ch.area
    area: float = c.area
    return area_ch < 1.005 * area or (area_ch - area) < 2000


def convex_union(c1: Optional[s.Polygon],
                 c2: Optional[s.Polygon]) -> Optional[s.Polygon]:
    if not c1 or not c2:
        return None
    u = c1.union(c2)
    if is_convex(u):
        return u
    else:
        return None


def perp_angle(border: s.LineString,
               centers: Tuple[Optional[s.Point], Optional[s.Point]]) -> float:
    A = border.coords[0]
    B = border.coords[1]
    ve = np.array(B) - np.array(A)
    # get the orientation of the border
    orientation = 0
    if centers[0]:
        C = np.array(centers[0].coords[0])
        orientation = orient(np.array([C, A, B]))
    elif centers[1]:
        C = np.array(centers[1].coords[0])
        orientation = -orient(np.array([C, A, B]))
    if orientation == 0:
        raise NameError("Not enough information to orient")
    return angle(ve) - orientation * np.pi * 0.5


def hplan_all(
        graph: VisibilityGraph,
        path: VisibilityRoute) -> Tuple[VisibilityGraph, VisibilityRoute, Any]:
    edges = [graph[po][npo] for po, npo in n_grams(path, 2)]
    return graph, path, edges


def hplan_sparse(
        graph: VisibilityGraph, path: VisibilityRoute,
        fixed: Set[GraphEdge]) -> Tuple[VisibilityGraph, VisibilityRoute, Any]:
    edges = [graph[po][npo] for po, npo in n_grams(path, 2)]
    pedge = edges[0]
    # pnode = hpath[0][0]
    for pnode, edge in list(zip(path, edges))[1:]:
        node, _ = pnode
        cell = edge['cell']
        if isinstance(node, tuple) and len(node) == 3:
            node = cast(GraphEdge, node)
            if not node[2] in fixed:
                u = convex_union(cell, pedge['cell'])
                if u:
                    pedge['cell'] = u
                    edges.remove(edge)
                    path.remove(pnode)
                    continue
            pedge = edge
    return graph, path, edges


def make_B(edges: List[Optional[Dict]], position: Vector) -> bn.B:
    angle = [e['angle'] for e in edges if e][0]
    cells = {i: e['cell'] for i, e in enumerate(edges) if e}
    n = bn.B(cells=cells, position=position, angle=angle)
    n.free = (False, True, True)
    return n


def make_T(layer: Layer, t: Transition, edges: List[Optional[Dict]], gamma: float,
           fixed: bool) -> bn.T:
    states = [layer.states[e['id']] if e else None for e in edges]
    cells = {i: e['cell'] for i, e in enumerate(edges) if e}
    angles = [e['angle'] if e else None for e in edges]
    # TODO: what if no valid angle? For example when the same vertex
    # belongs to 3 different polygons
    angles = [a for a in angles if a is not None and not np.isnan(a)]
    free: Optional[Tuple[bool, bool, bool]] = None
    delta_angle = []
    if fixed:
        free = (False, False, True)
        if not t.duality:
            raise ValueError("No geometry")
        geos = cast(Tuple[Optional[s.LineString], Optional[s.LineString]],
                    tuple((s.geometry if s else None) for s in states))
        a = perp_angle(t.duality.geometry, geos)
        delta_angle = [
            abs(normalize(a - a_)) if a_ is not None else 0 for a_ in angles
        ]
    else:
        a = avg_angle(angles)
    gamma = max(min(gamma, 0.95), 0.05)
    if not t.duality:
        raise ValueError("Undefined boundary")
    n = bn.T(cells=cells, boundary=t.duality.geometry, gamma=gamma, angle=a)
    if free is not None:
        n.free = free
    n.delta_angle = delta_angle
    return n


def guess_length(node: bn.Node,
                 angle: float,
                 s: int,
                 next_node: Optional[bn.Node] = None) -> float:
    lm = node.max_length_for_angle(s, angle)
    # if next_node and next_node.position is not None and node.position is not None:
    if next_node:
        lm = min(
            lm,
            np.linalg.norm(
                np.asarray(node.position.coords[0]) -
                np.asarray(next_node.position.coords[0])) * 0.33)
    return lm


def make_bezier(
        layer: Layer,
        visibility_graph: VisibilityGraph,
        visibility_path: VisibilityRoute,
        fixed: Set[GraphEdge],
        sparse: bool = False,
        fix_extrema_transitions: bool = True) -> Optional[bp.PathWithNodes]:
    if sparse:
        H, vv, edges = hplan_sparse(visibility_graph,
                                    visibility_path,
                                    fixed=fixed)
    else:
        H, vv, edges = hplan_all(visibility_graph, visibility_path)

    if not edges:
        return None

    # H, vv = plan.visibility_plan

    # edges = [H[po][npo] for po, npo in n_grams(vv, 2)]

    # print edges
    nodes: List[bp.Node] = []
    first = 0
    last = len(vv) - 1
    for i, v in enumerate(vv):
        attr = H._node[v]
        L_node, position = v
        # print '++ node', i, ' == ', v, ' == ', attr
        if i == 0:
            p_edges = [None, edges[0]]
        elif i == len(vv) - 1:
            p_edges = [edges[-1], None]
        else:
            p_edges = edges[i - 1:i + 1]

        if attr['id'] in layer.states:
            # s = layer.states[attr['id']]
            # print("B", s, p_edges, position)
            n: bp.Node = make_B(p_edges, position.position)
        elif attr['id'] in layer.transitions:
            t = layer.transitions[attr['id']]
            if fix_extrema_transitions and (i == first or i == last):
                is_fixed = True
                gamma = 0.5

            else:
                is_fixed = attr.get('fixed', False)
                gamma = attr['gamma']
            # print("T", p_edges, position, gamma, fixed)
            n = make_T(layer, t, p_edges, gamma, is_fixed)
        else:
            raise ValueError("Wrong path, node does not correspond "
                             "to either a state or a transition")
        nodes.append(n)
    for n1, n2 in list(n_grams(nodes, 2))[1:-1]:
        if n1.free[0] is False and n2.free[0] is False:
            if n1.delta_angle[1] < 0.01 and 1 in n1.cells:
                del n1.cells[1]
            if n2.delta_angle[0] < 0.01 and 0 in n2.cells:
                del n2.cells[0]
    for n1, n2 in list(n_grams(nodes, 2)):
        n1.set_length(guess_length(n1, n1.angle, 1, n2), 1)
        n2.set_length(guess_length(n2, n2.angle, 0, n1), 0)
    for n1 in nodes:
        if len(n1.cells) == 0:
            n1.free = (False, False, False)
        n1.setup_cp_fast()
    return bp.PathWithNodes(nodes)


def setup_path(path: bp.PathWithNodes, angle: Optional[float], s: int) -> None:
    if angle is not None:
        if s == 0:
            j = 1
            next_node = path.nodes[1]
        else:
            j = 0
            next_node = path.nodes[-2]
        n = path.nodes[s]
        n.angle = angle
        n.free = (False, False, True)
        n.set_length(guess_length(n, angle, j, next_node), j)
        n.setup_cp_fast()
