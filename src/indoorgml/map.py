import itertools
import math
from abc import ABC
from collections import ChainMap, defaultdict
from functools import lru_cache, partial
from heapq import heappop, heappush
import lxml.etree as ET

import networkx as nx
from rtree import index
from shapely.geometry import LinearRing, LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Set, Tuple, Type, TypeVar, Union, cast, Iterable)

from collections.abc import Mapping as MappingABC

# import xml.etree.ElementTree as ET

once = lru_cache(1, False)

Frame = Tuple[float, float, float]
Point2D = Tuple[float, float]
BoundinxBox = Tuple[float, float, float, float]

T = TypeVar('T', bound='GMLFeature')

nsmap: Dict[str, str] = {}
nsmap_i: Dict[str, str] = {}


def add_ns(ET, name, url):
    nsmap[name] = url
    nsmap_i[url] = name


def name_ns(name: str, ns: str) -> str:
    return f'{{{nsmap[ns]}}}{name}'


add_ns(ET, 'xlink', 'http://www.w3.org/1999/xlink')
add_ns(ET, 'gml', 'http://www.opengis.net/gml/3.2')
add_ns(ET, 'indoorCore', 'http://www.opengis.net/indoorgml/1.0/core')
add_ns(ET, 'indoorNavi', 'http://www.opengis.net/indoorgml/1.0/navigation')
add_ns(ET, 'xsi', 'http://www.w3.org/2001/XMLSchema-instance')
# WGS84 localization
add_ns(ET, 'alma', 'http://www.idsia.ch/alma')

gml = partial(name_ns, ns='gml')
core = partial(name_ns, ns='indoorCore')
nav = partial(name_ns, ns='indoorNavi')
xlink = name_ns('href', 'xlink')


def meta2dict(meta: ET.Element) -> Dict[Tuple[Optional[str], str], Any]:
    return dict(ChainMap(*(meta2dict_partial(m) for m in meta)))


def meta2dict_partial(meta: ET.Element) -> Dict[Tuple[Optional[str], str], Any]:
    h, s, t = meta.tag.partition('}')
    if not t:
        ns = None
        t = meta.tag
    else:
        ns = h[1:]
    v: Union[dict, str, bool]
    if len(meta) > 0:
        v = dict(ChainMap(*(meta2dict_partial(m) for m in meta)))
    else:
        if meta.text:
            v = meta.text
        else:
            v = True
    name = None
    if ns:
        name = nsmap_i.get(ns, None)
    return {(name, t): v}


def dict2meta_partial(meta: Dict) -> List[ET.Element]:
    es: List[ET.Element] = []
    for k, v in meta.items():
        e = ET.Element(name_ns(k[1], ns=k[0]))
        print(e)
        es.append(e)
        if isinstance(v, dict):
            e.extend(dict2meta_partial(v))
        elif isinstance(v, str):
            e.text = v
        elif v is True:
            pass
        else:
            raise NameError(f'Could not encode value {v} as xml')
    return es


def dict2meta(meta: Dict) -> ET.Element:
    root = ET.Element(gml('metaDataProperty'))
    metadata = ET.SubElement(root, gml('GenericMetaData'))
    metadata.extend(dict2meta_partial(meta))
    return root


# ALMOST generic metadata:

def add_bool_metadata(cls: Type[T], name: str) -> None:
    def set_meta(self: T, value: bool) -> None:
        if value:
            self._meta[('alma', name)] = True
        else:
            del self._meta[('alma', name)]

    def get_meta(self: T) -> bool:
        return self.meta(name, ns='alma') or False

    setattr(cls, name, property(fget=get_meta, fset=set_meta))


def set_crs(self: 'Map', value: Tuple[float, float, float]) -> None:
    x, y, a = value
    self._meta[('alma', 'crs')] = {
        ('alma', 'translation'): {('gml', 'pos'): f'{x} {y}'},
        ('alma', 'rotation'): f'{a}'}


def get_crs(self: 'Map') -> Optional[Tuple[float, float, float]]:
    crs = self.meta('crs', ns='alma')
    if not crs:
        return None
    x, y = map(float, crs[('alma', 'translation')][('gml', 'pos')].split())
    a = float(crs[('alma', 'rotation')])
    return (x, y, a)


def coord_to_gml(shape: BaseGeometry) -> str:
    return " ".join(["%.1f %.1f" % p for p in shape.coords])


def point_to_gml(point: Point, uid: Optional[str] = None) -> ET.Element:
    attribs = {'srsDimension': '2'}
    if uid is not None:
        attribs[gml('id')] = uid
    node = ET.Element(gml('Point'), attribs)
    pos = ET.SubElement(node, gml('pos'))
    pos.text = coord_to_gml(point)
    return node


def line_to_gml(line: LineString, uid: Optional[str] = None) -> ET.Element:
    attribs = {}
    if uid is not None:
        attribs[gml('id')] = uid
    node = ET.Element(gml('LineString'), attribs)
    pos = ET.SubElement(node, gml('posList'))
    pos.text = coord_to_gml(line)
    return node


def ring_to_gml(ring: LinearRing, uid: Optional[str] = None) -> ET.Element:
    attribs = {}
    if uid is not None:
        attribs[gml('id')] = uid
    node = ET.Element(gml('LinearRing'), attribs)
    pos = ET.SubElement(node, gml('posList'))
    pos.text = coord_to_gml(ring)
    return node


def polygon_to_gml(polygon: Polygon, uid: Optional[str]) -> ET.Element:
    attribs = {}
    if uid is not None:
        attribs[gml('id')] = uid
    node = ET.Element(gml('Polygon'), attribs)
    exterior = ET.SubElement(node, gml('exterior'))
    exterior.append(ring_to_gml(polygon.exterior))
    for ring in polygon.interiors:
        ET.SubElement(node, gml('interior')).append(ring_to_gml(ring))
    return node


_ns2types = {'indoorCore': ['CellSpace', 'CellSpaceBoundary', 'SpaceLayer', 'MultiLayeredGraph',
                            'State', 'Transition'],
             'indoorNavi': ['NavigableSpace', 'NavigableBoundary', 'GeneralSpace', 'TransferSpace',
                            'TransitionSpace', 'AnchorSpace', 'ConnectionSpace']}

_gml_ns = {t: ns for ns, ts in _ns2types.items() for t in ts}


def gml_id(node: ET.Element) -> str:
    return node.get('{%s}id' % nsmap['gml'])


def gml_pos(gml_pos: Optional[ET.Element]) -> Optional[Point2D]:
    if gml_pos is not None:
        if gml_pos.text is not None:
            coords: List[float] = [float(n) for n in gml_pos.text.split()]
            return tuple(coords[:2])  # type: ignore
    return None


def gml_pos_list(gml_pos_list: Optional[ET.Element]) -> Optional[List[Point2D]]:
    if gml_pos_list is not None:
        if gml_pos_list.text:
            coord = [float(n) for n in gml_pos_list.text.split()]
            return list(zip(coord[::2], coord[1::2]))
    return None


def polygon_for_bb(bb: BoundinxBox) -> Polygon:
    minx, miny, maxx, maxy = bb
    return Polygon(((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)))


class GMLFeature(ABC):
    """A minimal GML feature"""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    type: str = 'GMLFeature'
    geometry: Optional[BaseGeometry]
    _meta: Dict[Tuple[Optional[str], str], Any]

    def __hash__(self):
        return hash(self.id)

    def _repr_svg_(self) -> str:
        from indoorgml.display.svg import svg_for
        return svg_for(self)

    def __init__(self, uid: str) -> None:
        """ Initialize a GMLFeature with id = uid

            example::
            gml = GMLFeature('feature')
        """
        self.id = uid

    def save_xml(self, file_path: str) -> None:
        tree = ET.ElementTree()
        tree._setroot(self.xml_root)  # type: ignore
        tree.write(file_path, encoding='utf-8',
                   xml_declaration=True, pretty_print=True)

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        ns = _gml_ns[self.type]
        root = ET.Element(name_ns(self.type, ns), {gml('id'): self.id},
                          nsmap=nsmap)
        if self.name:
            name = ET.SubElement(root, gml('name'))
            name.text = self.name
        if self.description:
            description = ET.SubElement(root, gml('description'))
            description.text = self.description

        if self._meta:
            root.append(dict2meta(self._meta))

        return root

    @property  # type: ignore
    @once
    def xml_root(self) -> ET.Element:
        ref: Set[str] = set()
        root = self.xml(ref=ref)
        schemas = {
            nsmap['indoorCore']: 'http://schemas.opengis.net/indoorgml/1.0/indoorgmlcore.xsd',
            nsmap['indoorNavi']: 'http://schemas.opengis.net/indoorgml/1.0/indoorgmlnavi.xsd'}
        root.set(name_ns('schemaLocation', 'xsi'), ' '.join(
            ["%s %s" % i for i in schemas.items()]))
        return root

    def __repr__(self) -> str:
        name = f' ({self.name})' if self.name else ''
        description = f' - {self.description}' if self.description else ''
        return f"{self.type}: {self.id}{name}{description}"

    @classmethod
    def from_xml(cls: Type[T], node: ET.Element) -> T:
        instance = cls(gml_id(node))
        t = node.tag.split('}')
        instance.type = t.pop()
        nameNode = node.find('gml:name', nsmap)
        if nameNode is not None:
            instance.name = nameNode.text
        descNode = node.find('gml:description', nsmap)
        if descNode is not None:
            instance.description = descNode.text

        meta = node.find('gml:metaDataProperty/gml:GenericMetaData', nsmap)
        if meta is not None:
            instance._meta = meta2dict(meta)
        else:
            instance._meta = {}
        return instance

    @property
    def bounds(self) -> BoundinxBox:
        return cast(BaseGeometry, self.geometry).bounds

    def meta(self, name: str, ns: Optional[str] = None) -> Optional[Any]:
        return self._meta.get((ns, name), None)


class State(GMLFeature):
    """State is modeled after an indoorGML State"""

    connects: List['Transition']
    geometry: Optional[Point]
    duality: Optional['Cell']
    type: str = 'State'
    layer: 'Layer'

    equals: Mapping[str, Set[str]]
    contains: Mapping[str, Set[str]]
    overlaps: Mapping[str, Set[str]]
    intersects: Mapping[str, Set[str]]
    touches: Mapping[str, Set[str]]
    inside: Mapping[str, Set[str]]
    up: List[str]
    down: List[str]

    def __init__(self, uid: str) -> None:
        super(State, self).__init__(uid)
        self.connects = []
        self.geometry = None
        self.duality = None

        # according to indoorGML interlayer edges TYPES
        self.equals = defaultdict(set)
        self.inside = defaultdict(set)
        self.contains = defaultdict(set)
        self.overlaps = defaultdict(set)
        # OR of equals within contains overlaps
        self.intersects = defaultdict(set)
        # NOT in indoorGML
        self.touches = defaultdict(set)

    @property
    def is_navigable(self) -> bool:
        return self.duality is not None and self.duality.type != 'CellSpace'

    @property
    def neighbors(self) -> List['State']:
        if not self.layer.graph.has_node(self.id):
            return []
        return [self.layer.states[s] for s in self.layer.graph[self.id]]

    def transitionsTo(self, state) -> List['Transition']:
        if not self.layer.graph.has_node(self.id):
            return []
        ls = self.layer.graph[self.id].get(state.id, {}).values()
        return [self.layer.transitions[t['id']] for t in ls]

    def has_sensor(self) -> bool:
        return (self.layer.cls_name == 'SENSOR' and self.name is not None and
                self.name != 'NO' and self.duality is not None)

    def init_from_xml(self, node: ET.Element, layer: 'Layer') -> None:
        geometry_node = node.find('indoorCore:geometry//gml:pos', nsmap)
        if geometry_node is not None:
            pos = node.find('indoorCore:geometry//gml:pos', nsmap)
            point = gml_pos(pos)
            if point:
                self.geometry = Point(tuple(point))
        duality_node = node.find('indoorCore:duality/*', nsmap)
        if duality_node is not None:
            cell = layer.cells[gml_id(duality_node)]
            cell.duality = self
            self.duality = cell
        self.layer = layer

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(State, self).xml(ref=ref)
        if self.duality:

            if ref and self.duality.id in ref:
                ET.SubElement(root, core('duality'),
                              {xlink: f"#{self.duality.id}"})
            else:
                ET.SubElement(root, core('duality')).append(
                    self.duality.xml(ref=ref))

        for transition in self.connects:
            ET.SubElement(root, core('connects'), {
                          xlink: f"#{transition.id}"})
        if self.geometry:
            geometry = ET.SubElement(root, core('geometry'))
            geometry.append(point_to_gml(self.geometry, f'G_{self.id}'))
        return root

    def distance(self, position: Point2D) -> float:
        """ returns:: (distance,  nearest position in state) """
        p = Point(position)
        if not self.duality:
            return math.inf
        polygon = self.duality.geometry
        if p.intersects(polygon):
            return 0
        return p.distance(polygon.exterior)

    def nearest_point(self, position: Point2D) -> Optional[Point2D]:
        """ returns:: (distance,  nearest position in state) """
        p = Point(position)
        if not self.duality:
            return None
        polygon = self.duality.geometry
        if p.intersects(polygon):
            return position
        ring = polygon.exterior
        n_point = ring.interpolate(ring.project(p))
        return n_point.coords[0]


add_bool_metadata(State, name='open')
add_bool_metadata(State, name='nontraversable')
add_bool_metadata(State, name='invisible')


class Transition(GMLFeature):
    """Transition is modeled after an indoorGML Transition"""

    type: str = 'Transition'
    geometry: Optional[LineString] = None
    duality: Optional['Boundary'] = None
    end: State

    @property
    def connects(self) -> List[State]:
        return [self.start, self.end]

    def set_states(self, a: State, b: State, layer: 'Layer') -> None:
        self.start = a
        self.end = b
        a.connects.append(self)
        b.connects.append(self)
        if not self.geometry:
            if a.geometry and b.geometry and self.duality:
                self.geometry = LineString(
                    [a.geometry, self.duality.geometry.centroid, b.geometry])
        if self.geometry:
            self.weight = self.geometry.length
        else:
            self.weight = 0

    def init_from_xml(self, node: ET.Element, layer: 'Layer') -> None:
        a, b = [s.get('{%s}href' % nsmap['xlink'])[1:]
                for s in node.findall('indoorCore:connects', nsmap)]
        b_node = node.find('indoorCore:duality', nsmap)
        if b_node is not None:
            b_id = b_node.get('{%s}href' % nsmap['xlink'])[1:]
            self.duality = layer.boundaries[b_id]
        line: Optional[List[Point2D]] = None
        for pos_list in node.findall('.//gml:posList', nsmap):
            line = gml_pos_list(pos_list)
        if line and len(line) > 1:
            self.geometry = LineString(line)
        self.set_states(layer.states[a], layer.states[b], layer)

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(Transition, self).xml(ref=ref)
        for state in self.connects:
            ET.SubElement(root, core('connects'), {
                          xlink: f"#{state.id}"})
        if self.duality:
            ET.SubElement(root, core('duality'), {
                          xlink: f"#{self.duality.id}"})
        if self.geometry:
            geometry = ET.SubElement(root, core('geometry'))
            geometry.append(line_to_gml(self.geometry, f'G_{self.id}'))
        return root


class Cell(GMLFeature):
    """Cell is modeled after an indoorGML CellSpace"""

    boundary: List['Boundary']
    duality: State
    geometry: Polygon
    type: str = 'CellSpace'
    usage: Optional[str] = None
    cls_name: Optional[str] = None
    function: Optional[str] = None

    # outer_edge: dcel.Edge
    # inner_edges: List[dcel.Edge]

    def __init__(self, uid: str) -> None:
        super(Cell, self).__init__(uid)
        self.boundary = []
        # self.inner_edges = []

    # @property
    # def edges(self) -> List[dcel.Edge]:
    #     return [self.outer_edge] + self.inner_edges

    # def add_boundary(self, boundary: 'Boundary') -> None:
    #     self.boundary.append(boundary)
    #     boundary.add_cell(self)
    #
    # def remove_boundary(self, boundary: 'Boundary') -> None:
    #     if boundary in self.boundary:
    #         self.boundary.remove(boundary)
    #         boundary.remove_cell(self)



    # def reset_geometry(self) -> None:
    #     outer = geometry.LineString([e.origin for e in self.outer_edge.follow()])
    #     inner = [geometry.LineString([e.origin for e in edge.follow()])
    #              for edge in self.inner_edges]
    #     self.geometry = geometry.Polygon(outer, inner)
    #     if self.duality:
    #         self.duality.reset_geometry()

    def init_from_xml(self, node: ET.Element, layer: 'Layer') -> None:
        external_ref = node.find('indoorCore:externalReference', nsmap)
        if external_ref is not None:
            raise NameError("Not implemented yet")

        if self.type != 'CellSpace':
            for n in node.findall('indoorNavi:class', nsmap):
                self.cls_name = n.text
            for n in node.findall('indoorNavi:usage', nsmap):
                self.usage = n.text
            for n in node.findall('indoorNavi:function', nsmap):
                self.function = n.text
        self.boundary = []
        for boundary_node in node.findall('indoorCore:partialboundedBy', nsmap):
            ref = boundary_node.get('{%s}href' % nsmap['xlink'])
            if ref is None:
                b_id = gml_id(boundary_node[0])
            else:
                b_id = ref[1:]
            boundary = layer.boundaries[b_id]
            self.boundary.append(boundary)
            boundary.cells.append(self)

        polygonXML = node.find('indoorCore:Geometry2D/gml:Polygon', nsmap)
        if polygonXML is None:
            self.geometry = None
        else:
            interior = []
            exterior = None

            for pos_list in polygonXML.findall('gml:exterior//gml:posList', nsmap):
                exterior = gml_pos_list(pos_list)

            for loop in polygonXML.findall('gml:interior//gml:LinearRing', nsmap):
                ls = None
                for pos_list in loop.findall('.//gml:posList', nsmap):
                    ls = gml_pos_list(pos_list)
                if ls:
                    interior.append(LinearRing(ls))

            if exterior:
                self.geometry = Polygon(exterior, interior)

            if not self.geometry.is_valid:
                raise Exception("Invalid Cell %s: %s" %
                                (self.id, self.geometry.wkt))

    # @classmethod
    # def from_polygon(cls, uid, polygon, prefix=''):
    #     cell = Cell(f"{prefix}C{uid}")
    #     cell.geometry = polygon
    #     return cell

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(Cell, self).xml(ref=ref)
        if self.geometry:
            geometry_node = ET.SubElement(root, core('Geometry2D'))
            geometry_node.append(polygon_to_gml(self.geometry, f'G_{self.id}'))
        if self.duality:
            ET.SubElement(root, core('duality'), {
                          xlink: "#" + self.duality.id})
        for b in self.boundary:
            if ref is not None and b.id in ref:
                ET.SubElement(root, core('partialboundedBy'),
                              {xlink: f'#{b.id}'})
            else:
                p = ET.SubElement(root, core('partialboundedBy'))
                p.append(b.xml(ref=ref))
                if ref:
                    ref.add(b.id)

        if self.type != "CellSpace":
            c = ET.SubElement(root, nav('class'))
            if self.cls_name:
                c.text = self.cls_name
            c = ET.SubElement(root, nav('function'))
            if self.function:
                c.text = self.function
            c = ET.SubElement(root, nav('usage'))
            if self.usage:
                c.text = self.usage
        return root


class Boundary(GMLFeature):
    """Boundary is modeled after an indoorGML CellSpaceBoundary"""

    geometry: LineString
    type: str = 'CellSpaceBoundary'
    duality: Optional[Transition] = None
    cells: List[Cell]

    # outer_edge: dcel.Edge
    # inner_edges: List[dcel.Edge]

    def __init__(self, uid: str) -> None:
        super(Boundary, self).__init__(uid)
        # self._chains: Dict[str: dcel.Edge] = {}
        self.cells = []

    # def follow(self, cell: Cell) -> Generator[dcel.Edge]:
    #     start, end = self.chainInCell(cell)
    #     if start and end:
    #         yield start.to(end)
    #

    #

    #


    # def add_cell(self, cell: Cell) -> None:
    #     self.cells.append(cell)

    # def remove_cell(self, cell: Cell) -> None:
    #     if cell in self.cells:
    #         self.cells.remove(cell)



    # def reset_geometry(self) -> None:
    #     if len(self.cells) == 0:
    #         return
    #     chain = self.chain_in_cell(self.cells[0])
    #     start, end = chain
    #     vertices = [e.origin for e in start.to(end)] + [end.next.origin]
    #
    #     self.geometry = LineString(vertices)
    #
    #     if self.geometry.length < 0.01:
    #         raise NameError('Boundary is too short')
    #
    #     if self.duality:
    #         self.duality.reset_geometry()

    def init_from_xml(self, node: ET.Element, layer: 'Layer') -> None:
        line: Optional[List[Point2D]] = None
        for pos_list in node.findall('indoorCore:geometry2D/gml:LineString/gml:posList', nsmap):
            line = gml_pos_list(pos_list)
        if line:
            self.geometry = LineString(line)
        if not self.geometry.is_valid:
            raise Exception("Invalid Boundary %s %s" %
                            (self.id, self.geometry.wkt))

    # @classmethod
    # def from_line(uid: str, line: LineString, prefix: str = ''):
    #     b = Boundary(f"{prefix}B{uid}")
    #     b.geometry = line
    #     return b

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(Boundary, self).xml(ref=ref)
        if self.duality:
            ET.SubElement(root, 'indoorCore:duality', {
                          xlink: f"#{self.duality.id}"})
        geometry = ET.SubElement(root, core('geometry2D'))
        geometry.append(line_to_gml(self.geometry, f'G_{self.id}'))
        return root


class Layer(GMLFeature):
    """Layer is modeled after an indoorGML SpaceLayer"""

    cls_name: Optional[str] = None
    usage: Optional[str] = None
    function: Optional[str] = None
    states: Dict[str, State]
    external_states: List[State]
    transitions: Dict[str, Transition]
    cells: Dict[str, Cell]
    boundaries: Dict[str, Boundary]
    graph: nx.MultiGraph
    type: str = 'SpaceLayer'
    map: 'Map'
    rtree_index: index.Index

    def state_with_name(self, name: str) -> Optional[State]:
        try:
            return next(s for s in self.states.values() if s.name == name)
        except StopIteration:
            return None

    @classmethod
    def from_xml(cls, node: ET.Element) -> 'Layer':  # type: ignore
        layer = super(cls, cls).from_xml(node)
        layer.graph = nx.MultiGraph()
        layer.rtree_index = index.Index()
        layer.states = {}
        layer.transitions = {}
        layer.cells = {}
        layer.boundaries = {}

        state_node = {}
        transition_node = {}
        boundary_node = {}
        cell_node = {}

        for n in node.findall('./indoorCore:class', nsmap):
            layer.cls_name = n.text

        for n in node.findall('./indoorCore:usage', nsmap):
            layer.usage = n.text

        for n in node.findall('./indoorCore:function', nsmap):
            layer.function = n.text

        for n in node.findall("indoorCore:nodes//indoorCore:State", nsmap):
            state = State.from_xml(n)
            layer.states[state.id] = state
            state_node[state.id] = n

        for n in node.findall("indoorCore:nodes//indoorCore:duality/*", nsmap):
            cell = Cell.from_xml(n)
            layer.cells[cell.id] = cell
            cell_node[cell.id] = n

        for n in node.findall("indoorCore:nodes//indoorCore:partialboundedBy", nsmap):
            if not n.get('{%s}href' % nsmap['xlink']):
                boundary = Boundary.from_xml(n[0])
                layer.boundaries[boundary.id] = boundary
                boundary_node[boundary.id] = n[0]

        for n in node.findall("indoorCore:edges//indoorCore:Transition", nsmap):
            transition = Transition.from_xml(n)
            layer.transitions[transition.id] = transition
            transition_node[transition.id] = n

        for i, cell in layer.cells.items():
            cell.init_from_xml(cell_node[i], layer)

        for i, boundary in layer.boundaries.items():
            boundary.init_from_xml(boundary_node[i], layer)

        for j, (i, state) in enumerate(layer.states.items()):
            state.init_from_xml(state_node[i], layer)
            if state.duality:
                bounding_box = state.duality.geometry.bounds
                layer.rtree_index.add(j, bounding_box, obj=state.id)

        for i, transition in layer.transitions.items():
            transition.init_from_xml(transition_node[i], layer)
            layer.graph.add_edge(transition.start.id, transition.end.id, i, id=i,
                                 weight=transition.weight)

        layer.external_states = [
            s for s in layer.states.values() if not s.geometry]

        # layer.init_DCEL();

        return layer

    # only for geometrical layers (transition  =  adjacency)

    def nearest_states(self, position: Point2D) -> Iterator[Tuple[float, State]]:
        heap: List[Tuple[float, State]] = []
        p = Point(position)
        for i in self.rtree_index.nearest(position, num_results=len(self.states), objects=True):
            s = self.states[i.object]
            d = s.distance(position)
            cell = cast(Cell, s.duality)
            max_d = polygon_for_bb(cell.geometry.bounds).distance(p)
            heappush(heap, (d, s))
            while True:
                d, s = heappop(heap)
                if d < max_d:
                    yield (d, s)
                else:
                    heappush(heap, (d, s))
                    break

    def nearest_state_among(self, position: Point2D,
                            states_with: Callable[[State], bool]=lambda s: True
                            ) -> Optional[State]:

        for distance, state in self.nearest_states(position):
            if states_with(state):
                return state
        return None

    def state_at(self, position: Point2D) -> Optional[State]:
        p = Point(position)
        try:
            i = next(i.object for i
                     in self.rtree_index.intersection(position, objects=True)
                     if not cast(Cell, self.states[i.object].duality).geometry.disjoint(p))
            return self.states[i]
        except StopIteration:
            return None

    def states_in_box(self, bounding_box: BoundinxBox) -> List[State]:
        polygon = polygon_for_bb(bounding_box)
        return [self.states[i.object] for i
                in self.rtree_index.intersection(bounding_box, objects=True)
                if polygon.intersects(cast(Cell, self.states[i.object].duality).geometry)]

    def navigable_subgraph(self) -> nx.MultiGraph:
        return nx.MultiGraph(
            [(t.start.id, t.end.id, {'id': i, 'weight': t.weight})
             for i, t in self.transitions.items()
             if t.duality and t.duality.type == 'NavigableBoundary'])

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(Layer, self).xml()

        if self.usage:
            u = ET.SubElement(root, core('usage'))
            u.text = self.usage
        if self.cls_name:
            u = ET.SubElement(root, core('class'))
            u.text = self.cls_name
        if self.function:
            u = ET.SubElement(root, core('function'))
            u.text = self.function
        nodes = ET.SubElement(root, core('nodes'), {
                              gml('id'): f'Nodes_{self.id}'})
        edges = ET.SubElement(root, core('edges'), {
                              gml('id'): f'Edges_{self.id}'})
        for state in self.states.values():
            m = ET.SubElement(nodes, core('stateMember'))
            m.append(state.xml(ref=ref))
        for transition in self.transitions.values():
            if not hasattr(transition, 'start'):
                continue
            m = ET.SubElement(edges, core('transitionMember'))
            m.append(transition.xml(ref=ref))
        return root

    @property
    def geometry(self) -> BaseGeometry:  # type: ignore
        return cascaded_union([cell.geometry for cell in self.cells.values()])


class Map(GMLFeature, MappingABC):
    """Map is modeled after an indoorGML Multi Layered Graph"""

    layers: Dict[str, Layer]
    source: Optional[str] = None
    frame: Optional[Frame] = None

    # def semantic_layer(self):
    #     semantic_layers = [l for l in self.layers.values()
    #                        if l.usage == "Semantic"]
    #     if len(semantic_layers):
    #         return semantic_layers[0]
    #     else:
    #         return None
    #
    # def navigation_layer(self):
    #     navigation_layers = [l for l in self.layers.values()
    #                          if l.usage == "Navigation"]
    #     if len(navigation_layers):
    #         return navigation_layers[0]
    #     elif self.geometricLayer:
    #         return self.geometricLayer
    #     else:
    #         return None

    def add_layer(self, layer: Layer) -> None:
        self.layers[layer.id] = layer
        layer.map = self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, uid: str) -> Layer:
        return self.layers[uid]

    def __iter__(self) -> Iterator:
        return iter(self.layers)

    # def setCrsFromNode(self, node):
    #     n = node.find("WGS84:translation", nsmap)
    #     if n is not None:
    #         self.origin = coordinatesFromGML(n)
    #
    #     n = node.find("WGS84:rotation", nsmap)
    #     if n is not None:
    #         self.angle = float(n.text)

    @classmethod
    def from_file(cls, file_name: str) -> 'Map':
        """Load a multi layered graph from an indoorGML document"""
        tree = ET.parse(file_name)
        node = tree.getroot()
        if not node.tag == "{%s}MultiLayeredGraph" % nsmap['indoorCore']:
            node = node.find("indoorCore:MultiLayeredGraph", nsmap)
        if node is not None:
            m = cls.from_xml(node)
            m.source = file_name
            return m
        else:
            raise Exception('Malformed xml file: no MultiLayeredGraph tag')

    @property
    def geometric_layer(self) -> Optional[Layer]:
        try:
            return next(layer for layer in self.layers.values()
                        if layer.cls_name == 'TOPOGRAPHIC' and layer.usage == 'Geometry')
        except StopIteration:
            return None

    @property
    def states(self) -> Mapping[str, State]:
        return ChainMap(*(layer.states for layer in self.layers.values()))

    @property
    def cells(self) -> Mapping[str, Cell]:
        return ChainMap(*(layer.cells for layer in self.layers.values()))

    @property
    def boundaries(self) -> Mapping[str, Boundary]:
        return ChainMap(*(layer.boundaries for layer in self.layers.values()))

    @property
    def transitions(self) -> Mapping[str, Transition]:
        return ChainMap(*(layer.transitions for layer in self.layers.values()))

    @property
    def geometry(self) -> BaseGeometry:  # type: ignore
        return cascaded_union([layer.geometry for layer in self.layers.values()])

# TODO nav graph method of layer (nav sub graph for geo   graph for nav nOne esle)

    @classmethod
    def from_xml(cls, node: ET.Element) -> 'Map':    # type: ignore
        map_ = super(cls, cls).from_xml(node)
        map_.layers = {}
        for n in node.findall(".//indoorCore:SpaceLayer", nsmap):
            layer = Layer.from_xml(n)
            map_.add_layer(layer)
        map_.init_interlayer_edges()
        return map_

    # def push_layer(self, layer):
    #     self.addLayer(layer)
    #     self.states.update(layer.states)
    #     self.cells.update(layer.cells)
    #     self.boundaries.update(layer.boundaries)
    #     self.transitions.update(layer.transitionsmap)
    #     self.find_interlayer_edges()
    #     self.find_bounds()

    def init_interlayer_edges(self) -> None:

        from de9im import patterns
        relations = (
            (patterns.equal, 'equals', 'equals'),
            (patterns.within, 'inside', 'contains'),
            (patterns.contains, 'contains', 'inside'),
            (patterns.overlaps_regions, 'overlaps', 'overlaps'),
            (patterns.intersects, 'intersects', 'intersects'),
            (patterns.touches, 'touches', 'touches'))

        for s in self.states.values():
            # according to indoorGML interlayer edges TYPES
            s.equals = defaultdict(set)
            s.inside = defaultdict(set)
            s.contains = defaultdict(set)
            s.overlaps = defaultdict(set)
            # OR of equals within contains overlaps
            s.intersects = defaultdict(set)
            # NOT in indoorGML
            s.touches = defaultdict(set)
            # does not make sense between cells
            # (only between object with different dimentionality,
            # like a line crosses a polygon when the interior of
            # the line intersects the polygon but is not contained into it)
            # s.crosses = defaultdict(set)

        c = itertools.combinations(self.layers.items(), 2)

        for (i1, l1), (i2, l2) in c:
            for s1 in l1.states.values():
                if not s1.duality:
                    continue
                g1 = s1.duality.geometry
                for s2 in l2.states_in_box(g1.bounds):
                    if not s2.duality:
                        continue
                    g2 = s2.duality.geometry
                    DE9IM = g1.relate(g2)
                    for p, rel, inv in relations:
                        if p.matches(DE9IM):
                            getattr(s1, rel)[i2].add(s2.id)
                            getattr(s2, inv)[i1].add(s1.id)

        # remove overlapping if it is small enough -> within or contains
        MIN_AREA_TO_OVERLAP = 10
        for li1, layer in self.layers.items():
            for i1, s1 in layer.states.items():
                if not s1.duality:
                    continue
                g1 = s1.duality.geometry
                for li2, states in s1.overlaps.items():
                    layer = self.layers[li2]
                    for i2 in list(states)[:]:
                        s2 = layer.states[i2]
                        if not s2.duality:
                            continue
                        g2 = s2.duality.geometry
                        g12 = g1.difference(g2).area
                        g21 = g2.difference(g1).area
                        g1i2 = g1.intersection(g2).area

                        if g1i2 < MIN_AREA_TO_OVERLAP:
                            s1.overlaps[li2].remove(i2)
                            s1.touches[li2].add(i2)
                            s2.overlaps[li1].remove(i1)
                            s2.touches[li1].add(i1)
                            continue

                        if g12 < MIN_AREA_TO_OVERLAP and g12 < MIN_AREA_TO_OVERLAP:
                            continue
                        if g12 < MIN_AREA_TO_OVERLAP:
                            s1.overlaps[li2].remove(i2)
                            s1.inside[li2].add(i2)
                            s2.overlaps[li1].remove(i1)
                            s2.contains[li1].add(i1)
                            continue
                        if g21 < MIN_AREA_TO_OVERLAP:
                            s1.overlaps[li2].remove(i2)
                            s1.contains[li2].add(i2)
                            s2.overlaps[li1].remove(i1)
                            s2.inside[li1].add(i1)
                            continue

        def area(state_id):
            return self.states[state_id].duality.geometry.area

        for i, s in self.states.items():
            ws = s.inside.values() or [set()]
            s.up = sorted(set.union(*ws), key=area, reverse=False)
            cs = s.contains.values() or [set()]
            s.down = sorted(set.union(*cs), key=area, reverse=True)

    def xml(self, ref: Optional[Set[str]] = None) -> ET.Element:
        root = super(Map, self).xml()
        layers_node = ET.SubElement(
            root, core('spaceLayers'), {gml('id'): "layers"})
        for i in sorted(self.layers):
            layer = self.layers[i]
            m = ET.SubElement(layers_node, core('spaceLayerMember'))
            m.append(layer.xml(ref=ref))
        return root
