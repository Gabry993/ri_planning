import operator

from shapely.affinity import scale
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import snap, split
from typing import List, Tuple, Dict, Set, Optional

Point2D = Tuple[float, float]

FLAT = 0
REGULAR = 1
NON_REGULAR = 2


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
    return Polygon(simplify_ring(polygon.exterior), list(map(simplify_ring, polygon.interiors)))


class APolygon:

    source: Polygon
    rel_tol: float
    abs_tol: float
    dist_tol: float
    delta: int
    cache: Dict[Tuple[Point2D, Point2D, Point2D], LineString]
    nr_vertices: Set[Point2D]

    def visible(self, vertex: Point2D, point: Point2D) -> bool:
        return self.source.covers(LineString([vertex, point]))

    def add_to_cache(self, p: Point2D, a: Point2D, b: Point2D):
        if self.visible(p, a) and self.visible(p, b):
            edge = LineString([a, b])
            p1 = edge.interpolate(edge.project(Point(p)))
            diagonal = LineString((p, p1))
        else:
            diagonal = None
        if diagonal:
            self.nr_vertices.add(p)
        self.cache[(p, a, b)] = diagonal

    @property
    def best_diagonal(self) -> LineString:
        if self.has_holes:
            e = self.source.exterior
            i = min(self.source.interiors, key=lambda i: i.distance(e))
            v1, v2 = sorted(i.coords, key=lambda i: e.distance(Point(i)))[:2]
            return LineString([e.interpolate(e.project(Point(v2))), v2, v1,
                               e.interpolate(e.project(Point(v1)))])
        return min(((d.length, d) for d in self.cache.values() if d), key=operator.itemgetter(0))[1]

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
        else:
            self.rel_tol = rel_tol
            self.abs_tol = abs_tol
            self.dist_tol = dist_tol
            self.delta = delta

        if self.has_holes:
            return

        coords = self.source.exterior.coords
        n = len(coords) - 1
        vs = coords[:-1] * 2
        delta = self.delta

        for i, ts in enumerate(list(zip(vs, vs[1:], vs[2:]))[:n]):
            p = ts[1]
            if has_source and p not in source_nr_vertices:
                continue
            if v_type(*ts) != NON_REGULAR:
                continue
            ws = vs[i + 1 + delta:i + 1 - delta + n + 1]
            es = zip(ws, ws[1:])
            for (a, b) in es:
                if has_source and (p, a, b) in source_cache:
                    self.cache[(p, a, b)] = source_cache[(p, a, b)]
                    self.nr_vertices.add(p)
                else:
                    self.add_to_cache(p, a, b)

    def split(self, diagonal: LineString) -> Tuple['APolygon', 'APolygon']:

        try:
            left, right = split(self.source, diagonal)
        except ValueError:
            diagonal = snap(scale(diagonal, 1.01, 1.01, 1.01,
                                  origin=diagonal.coords[0]), self.source, tolerance=0.01)
            return self.split(diagonal)
        return (APolygon(left, parent=self), APolygon(right, parent=self))

    @property
    def has_holes(self) -> bool:
        return len(self.source.interiors) > 0

    @property
    def partition(self) -> List['APolygon']:
        if self.is_convex:
            return [self]
        left, right = self.split(self.best_diagonal)
        return left.partition + right.partition
