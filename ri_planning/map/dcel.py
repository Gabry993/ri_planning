import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
from shapely import geometry
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize
from shapely.ops import split as split_geo

from .map import Boundary, BoundingBox, Cell, Iterator, Layer
from ..utilities import Vector

once = lru_cache(1, False)


def chains(polygon: Polygon, sign: int = 1) -> Tuple['Chain', List['Chain']]:
    polygon = orient(polygon, sign)
    e = Edge.from_ring(polygon.exterior)
    es = [Edge.from_ring(ring) for ring in polygon.interiors]
    return (Chain.loop(e), [Chain.loop(e) for e in es])


class Chain(NamedTuple):
    start: 'Edge'
    end: 'Edge'

    @classmethod
    def loop(cls, e: 'Edge') -> 'Chain':
        return cls(e, e.previous)

    @property
    def line(self) -> LineString:
        return LineString([e.origin._point for e in self.edges] +
                          [self.end.next.origin._point])

    @property
    def bounds(self) -> BoundingBox:
        return cast(BoundingBox, self.line.bounds)

    @property
    def edges(self) -> Iterator['Edge']:
        return self.start.to(self.end)

    @property
    def face_boundary(self) -> Optional[Boundary]:
        return self.start.face_boundary

    @face_boundary.setter
    def face_boundary(self, boundary: Boundary) -> None:
        for e in self.edges:
            e.face_boundary = boundary

    @property
    def face(self) -> Optional[Cell]:
        return self.start.face

    @face.setter
    def face(self, cell: Cell) -> None:
        for e in self.edges:
            e.face = cell

    @property
    def twin(self) -> Optional['Chain']:
        if self.start.twin and self.end.twin:
            return Chain(self.end.twin, self.start.twin)
        return None

    @classmethod
    def from_ring(cls, line: LinearRing,
                  face: Cell) -> Tuple['Chain', 'Chain']:
        e1 = Edge.from_ring(line, face)
        e2 = Edge.from_ring(LinearRing[line.coords[::-1]], face)
        chain1 = Chain(e1, e1.previous)
        chain2 = Chain(e2, e2.previous)
        glue_chain(chain1, chain2)
        return (chain1, chain2)

    def split(self, line: LineString) -> List['Chain']:
        for e in self.edges:
            es = split_geo(e, list)
            if len(es):
                p = es[1].coords[0]
                e.split(p)
                return [Chain(self.start, e)] + Chain(e.next,
                                                      self.end).split(line)
        return [self]


# def orient(p1, p2, p3):
#     vol = np.linalg.det(
#         np.array([[p1.x, p1.y, 1], [p2.x, p2.y, 1], [p3.x, p3.y, 1]]))
#     if(abs(vol) < 1e-6):
#         return 0
#     return np.sign(vol)


def glue_chain(chain1: Chain, chain2: Chain) -> None:
    edges1 = list(chain1.edges)
    edges2 = list(chain2.edges)
    if len(edges1) != len(edges1):
        raise NameError('Twin chains have not the same length')
    for a, b in zip(edges1, reversed(edges2)):
        a.twin = b


class DCEL:

    boundary_chains: Dict[Boundary, Dict[Cell, Chain]]
    inner_edges: Dict[Cell, List['Edge']]
    outer_edge: Dict[Cell, 'Edge']
    outer_edges: List['Edge']
    layer: Layer

    def __init__(self, layer: Layer) -> None:
        self.layer = layer
        self.inner_edges = {}
        self.outer_edge = {}
        self.outer_edges = []
        self.boundary_chains = defaultdict(dict)
        self.init_layer()
        for c in layer.cells.values():
            self.init_cell(c)
        for b in layer.boundaries.values():
            self.init_boundary(b)

    @property
    def bounds(self) -> BoundingBox:
        return cast(BoundingBox, self.layer.bounds)

    def init_layer(self) -> None:
        polygons = polygonize([
            b.geometry for b in self.layer.boundaries.values()
            if len(b.cells) == 1
        ])
        for polygon in polygons:
            c, cs = chains(polygon, sign=-1)
            self.outer_edges.extend([c.start for c in [c] + cs])

    def init_cell(self, cell: Cell) -> None:
        gs = polygonize([b.geometry for b in cell.boundary])
        if not gs:
            print("ERROR DCEL.init_cell", cell,
                  [b.geometry for b in cell.boundary])
            return
        polygon = gs[0]
        c, cs = chains(polygon, sign=1)
        self.set_loop(c, cs, cell)

    def set_loop(self, chain: Chain, internal_chains: List[Chain],
                 cell: Cell) -> None:
        chain.face = cell
        for c in internal_chains:
            c.face = cell
        self.outer_edge[cell] = chain.start
        self.inner_edges[cell] = [c.start for c in internal_chains]

    def set_chain(self, boundary: Boundary, chain: Chain,
                  cell: Optional[Cell]) -> None:
        if cell:
            self.boundary_chains[boundary][cell] = chain
        chain.face_boundary = boundary

    def init_boundary(self, boundary: Boundary) -> None:
        for cell in boundary.cells:
            chain = self.find_chain_in_cell(boundary, cell)
            self.set_chain(boundary, chain, cell)
        if len(boundary.cells) == 1:
            chain = self.find_chain_in_layer(boundary)
            self.set_chain(boundary, chain, None)
        bcs = self.boundary_chains[boundary].values()
        if len(bcs) == 2:
            glue_chain(*bcs)

    def find_chain_in_edges(self, b: Boundary, edges: List['Edge']) -> Chain:
        start = None
        end = None
        for c in edges:
            for e in c.follow():
                if e.belongs_to_line(b.geometry):
                    start = e
                    end = e
                    while start.previous.belongs_to_line(
                            b.geometry) and start.previous is not e:
                        start = start.previous
                    if start.previous is not e:
                        while end.next.belongs_to_line(b.geometry):
                            end = end.next
                    else:
                        start = e
                        end = e.previous
                    break
        assert (start and end), 'Could not find chain for boundary'
        return Chain(start, end)

    def find_chain_in_cell(self, b: Boundary, cell: Cell) -> Chain:
        if cell in self.outer_edge:
            edges = [self.outer_edge[cell]]
        else:
            print(cell, "?", self.outer_edge.keys())
            edges = []
        if cell in self.inner_edges:
            edges += self.inner_edges[cell]
        return self.find_chain_in_edges(b, edges)

    def find_chain_in_layer(self, b: Boundary) -> Chain:
        return self.find_chain_in_edges(b, self.outer_edges)

    def set_chain_in_cell(self, b: Boundary, cell: Cell, chain: Chain) -> None:
        self.boundary_chains[b][cell] = chain
        for e in chain.edges:
            e.face_boundary = b

    def chain_in_cell(self, b: Boundary, cell: Cell) -> Chain:
        return self.boundary_chains[b][cell]


class Vertex:

    edge: Optional['Edge']

    def __init__(self, p: Point):
        self._point = p
        self.edge = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._point, name)


class Edge:
    """docstring for Edge"""

    face: Optional[Cell] = None
    face_boundary: Optional[Boundary] = None

    def __init__(self, p1: Union[Point, Vertex], p2: Union[Point,
                                                           Vertex]) -> None:
        super().__init__()
        if isinstance(p1, Vertex):
            p1 = p1._point
            v1 = p1
        else:
            v1 = Vertex(p1)
        if isinstance(p2, Vertex):
            p2 = p2._point
        self._line = LineString([p1, p2])
        self._origin: Vertex
        self.origin = cast(Vertex, v1)
        self._next: 'Edge'
        self._previous: 'Edge'
        self._twin: Optional['Edge'] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._line, name)

    def __repr__(self) -> str:
        return 'Edge ' + cast(str, self.wkt)

    @property  # type: ignore
    @once
    def angle(self) -> float:
        a = np.array(self)
        vector = a[1] - a[0]
        return math.atan2(vector[1], vector[0])

    @property
    def origin(self) -> 'Vertex':
        return self._origin

    @origin.setter
    def origin(self, value: 'Vertex') -> None:
        self._origin = value
        value.edge = self

    @property
    def twin(self) -> Optional['Edge']:
        return self._twin

    @twin.setter
    def twin(self, value: Optional['Edge']) -> None:
        if value:
            match = not (
                self.next and not (self.next.origin.wkt == value.origin.wkt) or
                value.next and not (self.origin.wkt == value.next.origin.wkt))
            assert (match), 'Twin edges do not match'
            value._twin = self
            if self.next:
                value.origin = self.next.origin
        self._twin = value

    @property
    def next(self) -> 'Edge':
        return self._next

    @next.setter
    def next(self, value: 'Edge') -> None:
        self._next = value
        value._previous = self

    @property
    def previous(self) -> 'Edge':
        return self._previous

    @previous.setter
    def previous(self, value: 'Edge') -> None:
        self._previous = value
        value._next = self

    @property
    def line(self) -> LineString:
        return LineString([e.origin._point for e in self.follow()])

    def belongs_to_line(self, line: LineString) -> bool:
        return self.intersects(line) and not line.touches(self._line)

    def follow(self, order: int = 1) -> Iterator['Edge']:
        e = self
        while True:
            yield e
            if order > 0:
                e = e.next
            else:
                e = e.previous
            if e is self:
                return

    def to(self, end: 'Edge') -> Iterator['Edge']:
        e = self
        while True:
            yield e
            if e is end:
                return
            e = e.next
            if e is self:
                return

    @classmethod
    def from_ring(self,
                  line: LinearRing,
                  face: Optional[Cell] = None) -> 'Edge':
        vertices = [geometry.Point(p) for p in line.coords[:-1]]
        nv = len(vertices)
        edges = [
            Edge(v, vertices[(i + 1) % nv]) for i, v in enumerate(vertices)
        ]
        for i, e in enumerate(edges):
            e.next = edges[(i + 1 + nv) % nv]
            e.face = face
        return edges[0]

    def split(self, p: Vector) -> None:
        ne = Edge(p, self.next.origin)
        ne.next = self.next
        ne.face = self.face
        ne.face_boundary = self.face_boundary
        self.next = ne
        self.coords[1] = ne.coords[0]
        if self.twin and self.twin.origin != p:
            self.twin.split(p)
            self.twin, ne.twin = self.twin.next, self.twin
