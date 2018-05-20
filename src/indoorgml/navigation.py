from collections import defaultdict
from itertools import combinations, product

from indoorgml.map import Boundary, Cell, Layer, State, Transition, Map
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union, linemerge, split
from typing import List, Optional


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


def split_geometry(source: Polygon, lines: List[LineString],
                   obstacle: BaseGeometry) -> List[Polygon]:
    p = source.difference(obstacle)
    if p.is_empty:
        return []
    geo = [p]
    if lines.type == 'LineString':
        lines = [lines]
    for l in lines:
        geo = sum((list(split(p, l)) for p in geo), [])
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
        state.duality = cell
        state.geometry = polygon.representative_point()
        layer.rtree_index.add(i, polygon.bounds, obj=state.id)
        layer.graph.add_node(state.id)

        add_boundary(layer, c, None, polygon.exterior)
        for ring in polygon.interiors:
            add_boundary(layer, cell, None, ring)

    for cell1, cell2 in combinations(cells, 2):
        if f(cell1.geometry.boundary, cell2.geometry.boundary):
            line = cell1.geometry.intersection(cell2.geometry)
            add_boundary(layer, cell1, cell2, line)
    return cells


def add_boundary(layer: Layer, cell: Cell, cell2: Optional[Cell], line: LineString) -> None:
    i = len(layer.boundaries) + 1
    b = Boundary(f'{layer.id}B{i}')
    if cell2:
        b.type = 'NavigableSpaceBoundary'
    b.geometry = line
    layer.boundaries[b.id] = b
    for c in [cell, cell2]:
        if not c:
            continue
        for pb in c.boundary:
            if f(pb.geometry, line):
                diff = pb.geometry.difference(line)
                try:
                    pline = linemerge(diff)
                except ValueError:
                    pline = diff
                if pline.type == 'MultiLineString':
                    pb.geometry = pline[0]
                    if len(pline) > 1:
                        c1, c2 = pb.cells
                        add_boundary(layer, c1, c2, pline[1])
                    pline = pline[0]
                else:
                    pb.geometry = pline
                t = pb.duality
                if t:
                    t.reset_geometry()
                    layer.graph[t.start][t.end][t.id]['weigth'] = t.geometry.length
    if cell2:
        b.cells = [cell, cell2]
        t = Transition(f'{layer.id}T{i}')
        t.duality = b
        b.duality = t
        t.set_states(cell.duality, cell2.duality, layer)
        layer.transitions[t.id] = t
        layer.graph.add_edge(t.start.id, t.end.id, t.id, id=t.id, weight=t.weight)
    else:
        b.cells = [cell, None]
    for c in b.cells:
        if c:
            c.boundary.append(b)


def from_layer(source: Layer, uid: str, clearance: float, tolerance: float = 0.1,
               intersect: List[Layer] = []) -> Layer:
    layer = Layer(uid)
    layer.description = (f'Layer obtained from {source} adding a clearance of {clearance} cm')
    layer.cls_name = source.cls_name
    layer.usage = 'Navigation'
    layer._meta = source._meta
    cells = defaultdict(list)
    lines = all_lines(intersect)
    obs = obstacle(source, clearance, tolerance)

    for cell_id, cell in source.cells.items():
        if cell.type == 'CellSpace':
            continue
        geometry = split_geometry(cell.geometry, lines, obs)
        if geometry:
            n_cell = add_copy_of_cell(layer, cell, geometry)
            cells[cell] = n_cell

    for boundary in source.boundaries.values():
        if len(boundary.cells) == 2:
            source_1, source_2 = boundary.cells
            cells1 = [c for c in cells[source_1] if f(boundary.geometry, c.geometry.boundary)]
            cells2 = [c for c in cells[source_2] if f(boundary.geometry, c.geometry.boundary)]
            for cell1, cell2 in product(cells1, cells2):
                b = cell1.geometry.boundary.intersection(cell2.geometry.boundary)
                b = b.intersection(boundary.geometry)
                add_boundary(layer, cell1, cell2, b)
    return layer


def navigation_layer(map_: Map, uid: str, clearance: float, tolerance: float = 0.1):

    geometric_layer = next(l for l in map_.layers.values()
                           if l.cls_name == 'TOPOGRAPHIC' and l.usage == 'Geometry')
    layers = [l for l in map_.layers.values() if l.cls_name != 'TOPOGRAPHIC']
    return from_layer(geometric_layer, uid=uid, clearance=clearance, tolerance=tolerance,
                      intersect=layers)
