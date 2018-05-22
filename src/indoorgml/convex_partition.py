import operator

from shapely.affinity import scale
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import snap, split
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
Point2D = Tuple[float, float]


def partition(polygon: Polygon, rel_tol: float = 0, abs_tol: float = 0,
              dist_tol: float = 0, delta: int = 2) -> List[Polygon]:
    return [p.source for p in APolygon(polygon, rel_tol=rel_tol, abs_tol=abs_tol,
            dist_tol=dist_tol, delta=delta).partition]


def h(a, b, c, d) -> bool:
    d1 = np.array(b) - np.array(a)
    d2 = np.array(d) - np.array(c)
    return np.dot(d1, d2) / np.linalg.norm(d1) / np.linalg.norm(d2) < -0.5


def h1(a, b, c, d) -> bool:
    return np.dot(np.array(b) - np.array(a), np.array(d) - np.array(c)) < 0


def hs(a, b, c) -> bool:
    d1 = np.array(b) - np.array(a)
    d2 = np.array(c) - np.array(b)
    return np.cross(d1, d2) / np.linalg.norm(d1) / np.linalg.norm(d2) < -0.01


FLAT = 0
REGULAR = 1
NON_REGULAR = 2


def cut(line: LineString, distance: float) -> bool:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return False
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return False
        if pd > distance:
            cp = line.interpolate(distance)
            line.coords = coords[:i] + [(cp.x, cp.y)] + coords[i:]
            return True
    return False


def v_type(p: Point2D, vertex: Point2D, n: Point2D) -> int:
    r = LinearRing([p, vertex, n])
    if not r.is_valid:
        return FLAT
    if r.is_ccw:
        return REGULAR
    return NON_REGULAR


def simplify_ring(ring: LinearRing) -> LinearRing:
    ring = ring.simplify(0)
    cs = ring.coords
    if v_type(cs[-2], cs[0], cs[1]) == FLAT:
        cs = cs[1:-1]
        ring = LinearRing(ring)
    return ring


def simplify(polygon: Polygon) -> Polygon:
    return polygon
    return Polygon(simplify_ring(polygon.exterior), list(map(simplify_ring, polygon.interiors)))


class APolygon:

    source: Polygon
    rel_tol: float
    abs_tol: float
    dist_tol: float
    delta: int
    cache: Dict[Tuple[Point2D, Point2D, Point2D], LineString]
    nr_vertices: Set[Point2D]
    new_diagonals: Set[Tuple[Point2D, Point2D]]

    def visible(self, vertex: Point2D, point: Point2D) -> bool:
        line = LineString([vertex, point])
        return self.source.covers(line) and self.source.boundary.touches(line)

    def add_to_cache(self, p: Point2D, a: Point2D, b: Point2D) -> None:
        if self.visible(p, a) and self.visible(p, b):
            edge = LineString([a, b])
            p1 = edge.interpolate(edge.project(Point(p)))
            # e = self.source.exterior
            # d1 = e.project(p1)
            # if cut(e, d1):
            #     self.source = Polygon(e, self.source.interiors)
            diagonal = snap(LineString((p, p1)), self.source, 0.01)
        else:
            diagonal = None
        if diagonal:
            self.nr_vertices.add(p)
        self.cache[(p, a, b)] = diagonal

    @property
    def best_diagonal(self) -> Optional[LineString]:
        if self.has_holes:
            e = self.source.exterior
            i = min(self.source.interiors, key=lambda i: i.distance(e))
            v1, v2 = sorted(i.coords[:-1], key=lambda x: e.distance(Point(x)))[:2]
            d1 = e.project(Point(v1))
            d2 = e.project(Point(v2))
            e1 = e.interpolate(d1)
            e2 = e.interpolate(d2)
            c1 = cut(e, d1)
            c2 = cut(e, d2)
            if c1 or c2:
                self.source = Polygon(e, self.source.interiors)
            return snap(LineString([e2, v2, v1, e1]), self.source.boundary, 0.01)
        while self.cache:
            try:
                _, d, k = min(((d.length, d, k) for k, d in self.cache.items() if d),
                              key=operator.itemgetter(0))
            except ValueError as e:
                print("!!!!EEEEE!!!!!")
                return None
            if self.source.boundary.covers(d):
                # HACK: should never happend but it does
                del self.cache[k]
            else:
                return d
        return None

    @property
    def is_convex(self) -> bool:
        if self.has_holes:
            return False
        polygon = self.source
        ch = polygon.convex_hull
        if polygon.equals(ch):
            return True
        if polygon.hausdorff_distance(ch) <= self.dist_tol:
            return True
        area_ch = ch.area
        area = polygon.area
        return (area_ch < (1 + self.rel_tol) * area) or (area_ch - area < self.abs_tol)

    def __init__(self, polygon: Polygon, rel_tol: float = 0, abs_tol: float = 0,
                 dist_tol: float = 0, delta: int = 2, parent: Optional['APolygon'] = None) -> None:

        self.source = simplify(orient(polygon))
        self.cache = {}
        self.nr_vertices = set()
        has_source = False
        if parent:
            if not parent.has_holes:
                source_cache = parent.cache
                source_nr_vertices = parent.nr_vertices
                has_source = True
            self.rel_tol = parent.rel_tol
            self.abs_tol = parent.abs_tol
            self.dist_tol = parent.dist_tol
            self.delta = parent.delta
            self.new_diagonals = set(parent.new_diagonals)
        else:
            self.rel_tol = rel_tol
            self.abs_tol = abs_tol
            self.dist_tol = dist_tol
            self.delta = delta
            self.new_diagonals = set()

        if self.has_holes:
            return

        coords = self.source.exterior.coords
        n = len(coords) - 1
        vs = coords[:-1] * 2
        delta = self.delta

        for i, ts in enumerate(list(zip(vs, vs[1:], vs[2:]))[:n]):
            o, p, q = ts

            if has_source and p not in source_nr_vertices:
                continue

            if not hs(o, p, q):
                continue

            # if v_type(o, p, q) != NON_REGULAR:
            #     continue
            # ws = vs[i + 1 + delta:i + 1 - delta + n + 1]
            # es = zip(ws, ws[1:])

            ws = vs[i + 2:i + 1 + n]
            es = list(zip(ws, ws[1:]))
            m = len(es)
            for (a, b) in es[::-1]:
                if h(a, b, o, p):
                    break
                m = m - 1
            es = es[:m]
            valid = False
            for (a, b) in es:
                if not valid:
                    if h(a, b, p, q):
                        valid = True
                    else:
                        self.cache[(p, a, b)] = None
                        continue
                if has_source and (p, a, b) in source_cache:
                    d = source_cache[(p, a, b)]
                    self.cache[(p, a, b)] = d
                    if d:
                        self.nr_vertices.add(p)
                else:
                    if (a, b) in self.new_diagonals:
                        self.cache[(p, a, b)] = None
                    else:
                        self.add_to_cache(p, a, b)

    def split(self, diagonal: LineString) -> Tuple['APolygon', 'APolygon']:
        try:
            left, right = split(self.source, diagonal)
        except ValueError:
            diagonal1 = snap(scale(diagonal, 1.01, 1.01, 1.01,
                                   origin=diagonal.coords[0]), self.source, tolerance=0.01)
            try:
                left, right = split(self.source, diagonal1)
            except ValueError as e:
                print(self.source.wkt)
                print(diagonal1.wkt)
                raise e
        return (APolygon(left, parent=self), APolygon(right, parent=self))

    @property
    def has_holes(self) -> bool:
        return len(self.source.interiors) > 0

    @property
    def partition(self) -> List['APolygon']:
        if self.is_convex:
            return [self]
        diagonal = self.best_diagonal
        if not diagonal or not diagonal.is_valid or not diagonal.type == 'LineString':
            return [self]
        d = (diagonal.coords[0], diagonal.coords[-1])
        self.new_diagonals.add(d)
        self.new_diagonals.add(d[::-1])
        left, right = self.split(diagonal)
        return left.partition + right.partition
