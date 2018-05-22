from collections import defaultdict
from itertools import combinations, product

from indoorgml.map import Boundary, Cell, Layer, State, Transition, Map
from indoorgml.convex_partition import partition
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, linemerge, split, snap
from typing import List, Optional, Dict
from tqdm import tqdm_notebook as tqdm


def intersection(g1: BaseGeometry, g2: BaseGeometry, tolerance: float = 0.1) -> BaseGeometry:
    return snap(g2, g1, tolerance=0.1).intersection(g1)


def obstacle(layer: Layer, clearance: float, tolerance: float = 0.1) -> BaseGeometry:
    obstacles: List[BaseGeometry] = []
    for cell_id, cell in layer.cells.items():
        if not cell.type == 'CellSpace':
            continue
        obstacles.append(cell.geometry.buffer(clearance, resolution=16, cap_style=3,
                                              join_style=2, mitre_limit=2.0))
    return cascaded_union(obstacles).simplify(tolerance)


def all_lines(layers: List[Layer]) -> BaseGeometry:
    return cascaded_union([b.geometry for layer in layers for b in layer.boundaries.values()])


def g(ps: BaseGeometry) -> List[Polygon]:
    if ps.type == 'Polygon':
        return [ps]
    if ps.type == 'MultiPolygon':
        return list(ps)
    return []


def split_polygon(polygon: Polygon, line: LineString) -> List[Polygon]:
    line = snap(line, polygon, 0.1)
    return sum([g(x) for x in split(polygon, line)], [])


def split_geometry(source: Polygon, lines: BaseGeometry,
                   obstacle: BaseGeometry) -> List[Polygon]:
    p = source.difference(obstacle)
    if p.is_empty:
        return []
    geo = g(p)
    if lines.type == 'LineString':
        lines = [lines]
    for l in lines:
        geo = sum((split_polygon(p, l) for p in geo), [])
        # geo = sum((list(split(p, l)) for p in geo), [])
    return geo


def f(g1: LineString, g2: LineString) -> bool:
    return g1.relate_pattern(g2, 'T********')


def add_copy_of_state(layer: Layer, state: State, uid: str) -> State:
    nstate = State(uid)
    nstate._meta = state._meta
    layer.states[uid] = nstate
    return nstate


def add_copy_of_cell(layer: Layer, cell: Cell, geometry: List[Polygon]) -> List[Cell]:
    cells = []
    for polygon in geometry:
        i = len(layer.cells) + 1
        c = Cell(f'{layer.id}C{i}')
        c.geometry = polygon
        c.type = cell.type
        c.function = cell.function
        c.usage = cell.usage
        c.cls_name = cell.cls_name
        cells.append(c)
        layer.cells[c.id] = c
        state = add_copy_of_state(layer, cell.duality, uid=f'{layer.id}S{i}')
        c.duality = state
        state.duality = c
        state.geometry = polygon.centroid
        layer.rtree_index.add(i, polygon.bounds, obj=state.id)
        layer.graph.add_node(state.id)

        add_boundary(layer, c, None, polygon.exterior)
        for ring in polygon.interiors:
            add_boundary(layer, cell, None, ring)

    for cell1, cell2 in combinations(cells, 2):
        if f(cell1.geometry.boundary, cell2.geometry.boundary):
            # line = cell1.geometry.intersection(cell2.geometry)
            line = intersection(cell1.geometry.boundary, cell2.geometry.boundary)
            ls: List[LineString] = []
            if line.type == 'MultiLineString':
                line = linemerge(line)
            if line.type == 'MultiLineString':
                ls = line
            elif line.type == 'LineString':
                ls = [line]
            for l in ls:
                if l.length > 0.1:
                    add_boundary(layer, cell1, cell2, l)
            # else:
            #     print('A', line.type)
            #     print(cell1.geometry.wkt)
            #     print(cell2.geometry.wkt)
                # print(line.wkt)
    return cells


def add_boundary(layer: Layer, cell: Cell, cell2: Optional[Cell], line: LineString) -> None:
    if line.length < 0.01 or line.is_empty:
        return
    i = len(layer.boundaries) + 1
    b = Boundary(f'{layer.id}B{i}')
    if cell2:
        b.type = 'NavigableBoundary'
    b.geometry = line
    old_boundary = False
    rest_boundaries = []
    for c in [cell, cell2]:
        if not c:
            continue
        for pb in list(c.boundary):
            if f(pb.geometry, line):
                if b.geometry.equals(pb.geometry):
                    b = pb
                    old_boundary = True
                    break

                diff = pb.geometry.difference(line)
                try:
                    pline = linemerge(diff)
                except ValueError:
                    pline = diff
                if pline.length == 0 or pline.is_empty:
                    raise NotImplemented
                if pline.type == 'MultiLineString':
                    pb.geometry = pline[0]
                    if len(pline) > 1:
                        c1, c2 = pb.cells
                        rest_boundaries.append((c1, c2, pline[1]))
                    pline = pline[0]
                else:
                    pb.geometry = pline
                if pb.duality:
                    pb.duality.reset_geometry()
    layer.boundaries[b.id] = b
    if old_boundary:
        for c in b.cells:
            if c:
                c.boundary.remove(b)
    if cell2:
        b.cells = [cell, cell2]
        if old_boundary and b.duality:
            t = b.duality
        else:
            t = Transition(f'{layer.id}T{i}')
            layer.transitions[t.id] = t
        t.duality = b
        b.duality = t
        t.set_states(cell.duality, cell2.duality, layer)
        layer.graph.add_edge(t.start.id, t.end.id, t.id, id=t.id, weight=t.weight)
    else:
        b.cells = [cell, None]
    for c in b.cells:
        if c:
            c.boundary.add(b)

    for c1, c2, line in rest_boundaries:
        add_boundary(layer, c1, c2, line)


def from_layer(source: Layer, uid: str, clearance: float, tolerance: float = 0.1,
               intersect: List[Layer] = [], convex: bool = True) -> Layer:
    layer = Layer(uid)
    layer.description = (f'Layer obtained from {source} adding a clearance of {clearance} cm')
    layer.cls_name = source.cls_name
    layer.usage = 'Navigation'
    layer._meta = source._meta
    cells: Dict[Cell, List[Cell]] = defaultdict(list)
    # lines = all_lines(intersect)
    obs = obstacle(source, clearance, tolerance)
    layers = {l.id: l for l in intersect}
    for cell_id, cell in tqdm(source.cells.items()):
        if cell.type == 'CellSpace':
            continue
        state = cell.duality
        boundaries = set(
            sum([list(layers[k].states[x].duality.boundary) for k, xs in state.contains.items()
                 for x in xs if k in layers], []))
        lines = cascaded_union([b.geometry for b in boundaries])

        geometry = split_geometry(cell.geometry, lines, obs)
        if convex:
            geometry = sum([partition(p, dist_tol=1, abs_tol=10, rel_tol=0.001)
                            for p in geometry], [])
        geometry = [x for x in geometry if x.area > 1]
        if geometry:
            n_cell = add_copy_of_cell(layer, cell, geometry)
            cells[cell] = n_cell

    for boundary in tqdm(source.boundaries.values()):
        if len(boundary.cells) == 2:
            source_1, source_2 = boundary.cells
            if source_1 not in cells or source_2 not in cells:
                continue
            # cells1 = [c for c in cells[source_1] if f(boundary.geometry, c.geometry.boundary)]
            # cells2 = [c for c in cells[source_2] if f(boundary.geometry, c.geometry.boundary)]
            b = boundary.geometry
            cells1 = [c for c in cells[source_1] if has_boundary(c, b)]
            cells2 = [c for c in cells[source_2] if has_boundary(c, b)]

            for cell1, cell2 in product(cells1, cells2):
                b1 = cell1.geometry.boundary.buffer(0.001)
                b2 = cell2.geometry.boundary.buffer(0.001)
                b = b1.intersection(b2).intersection(boundary.geometry)
                # b = cell1.geometry.boundary.intersection(cell2.geometry.boundary)
                # b = b.intersection(boundary.geometry)
                if b.type == 'MultiLineString':
                    b = linemerge(b)
                add_boundary(layer, cell1, cell2, b)
    return layer


def has_boundary(cell: Cell, line: LineString) -> bool:
    b = cell.geometry.boundary.buffer(0.001)
    # if line.disjoint(b):
    #     return False
    # b = snap(b, line, 0.1)
    return f(b, line)


def navigation_layer(map_: Map, uid: str, clearance: float, tolerance: float = 0.1,
                     convex: bool = True) -> Layer:

    geometric_layer = next(l for l in map_.layers.values()
                           if l.cls_name == 'TOPOGRAPHIC' and l.usage == 'Geometry')
    layers = [l for l in map_.layers.values() if l.cls_name != 'TOPOGRAPHIC']
    return from_layer(geometric_layer, uid=uid, clearance=clearance, tolerance=tolerance,
                      intersect=layers, convex=convex)
