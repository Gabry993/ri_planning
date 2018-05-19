import xml.etree.ElementTree as ET

from indoorgml.map import (Boundary, Cell, GMLFeature, Layer, Map, State,
                           Transition)
from indoorgml.dcel import Chain, Edge, DCEL
from shapely.geometry import LineString, Polygon
from typing import Callable, Dict, Optional, List, cast, Union

from functools import lru_cache

once = lru_cache(1)


ON_CLICK = Callable[[GMLFeature], str]


def click(state):
    return f"(function(event){{alert('{state.id}'); event.stopPropagation();}})(event)"


SVG_NS = 'http://www.w3.org/2000/svg'

map_css = """

.TransitionSpace {
    fill: lightgreen;
    stroke: none;
}

.GeneralSpace {
    fill: olive;
    stroke: none;
}

.ConnectionSpace {
    fill: orange;
    stroke: none;
}


.TOPOGRAPHIC .CellSpace {
    fill: black;
    stroke: none;
}

.CellSpace {
    fill: BurlyWood ;
    stroke: white;
    stroke-width: 10;
}

.CellSpaceBoundary {
    display: none;
}

.NavigableBoundary {
    display: none;
    stroke: green;
    stroke-dasharray: 40 40;
    stroke-width: 20;
    fill: none;
}

"""


def pathForLine(line: LineString) -> str:
    if not line:
        return ""
    pathList = ["%s %s" % (x, y) for (x, y) in line.coords]
    pathList.insert(0, 'M')
    pathList.insert(2, 'L')
    return " ".join(pathList)


def pathForPolygon(polygon: Polygon) -> str:
    borders = [polygon.exterior.coords]
    for i in polygon.interiors:
        borders.append(i.coords)
    d: List[str] = []
    for b in borders:
        if(not len(b)):
            continue
        pathList = ["%s %s" % (x, y) for (x, y) in b]
        pathList.insert(0, 'M')
        pathList.insert(3, 'L')
        d += pathList
    d.append('Z')
    return " ".join(d)


@once
def svg_for(gml: GMLFeature, width: int = 1000, height: int = 500) -> str:

    min_x, min_y, max_x, max_y = gml.geometry.bounds
    min_y, max_y = -max_y, -min_y
    vb_height = max_y - min_y + 200
    vb_width = max_x - min_x + 200
    if isinstance(gml, Map):
        vb_width = (max_x - min_x + 500) * len(gml.layers) + 100

    svg = ET.Element('svg', {'xmlns': SVG_NS, 'width': f'{width}', 'height': f'{height}',
                             'viewBox': f"{min_x - 100} {min_y - 100} {vb_width} {vb_height}"})
    # ET.SubElement(svg, 'rect', width="100%", height="100%", fill="white")
    style = ET.SubElement(
        svg, 'style', {'type': "text/css", 'media': "screen"})
    style.text = map_css

    g = ET.SubElement(svg, 'g', {'transform': 'scale(1, -1)'})

    e = None
    if isinstance(gml, Cell):
        e = from_cell(gml)

    if isinstance(gml, Boundary):
        e = from_boundary(gml)
    if isinstance(gml, State):
        e = from_state(gml)
    if isinstance(gml, Transition):
        e = from_transition(gml)
    if isinstance(gml, Layer):
        e = from_layer(gml)
    if isinstance(gml, Map):
        e = from_map(gml)
    if e:
        g.append(e)
    return ET.tostring(svg, encoding='unicode')


def svg_for_dcel(x: Union[Edge, DCEL, Chain], width: int = 1000, height: int = 500) -> str:
    min_x, min_y, max_x, max_y = x.bounds
    min_y, max_y = -max_y, -min_y
    vb_height = max_y - min_y + 200
    vb_width = max_x - min_x + 200

    svg = ET.Element('svg', {'xmlns': SVG_NS, 'width': f'{width}', 'height': f'{height}',
                             'viewBox': f"{min_x - 100} {min_y - 100} {vb_width} {vb_height}"})

    defs = ET.SubElement(svg, 'defs')
    marker = ET.SubElement(defs, 'marker', {'id': 'arrow', 'markerWidth': "10",
                                            'markerHeight': "10", 'refX': "10", 'refY': "3",
                                            'orient': "auto", 'markerUnits': "strokeWidth",
                                            'viewBox': "0 0 20 20"})
    ET.SubElement(marker, 'path', {'d': "M0,0 L0,6 L10,3 z", 'fill': "black",
                                   'fill-opacity': "0.3"})
    g = ET.SubElement(svg, 'g', {'transform': 'scale(1, -1)'})
    e = None
    if isinstance(x, Edge):
        e = from_edge(x)
    if isinstance(x, Chain):
        e = from_chain(x)
    if isinstance(x, DCEL):
        e = from_dcel(x)
    if e is not None:
        g.append(e)
    return ET.tostring(svg, encoding='unicode')


def from_edge(edge: Edge) -> ET.Element:
    x1, y1 = edge.origin.coords[0]
    x2, y2 = edge.next.origin.coords[0]
    return ET.Element('line', {'x1': f'{x1}', 'y1': f'{y1}', 'x2': f'{x2}', 'y2': f'{y2}',
                               'stroke': "black", 'stroke-width': "5", 'stroke-opacity': "0.3",
                               'marker-end': "url(#arrow)"})


def from_chain(chain: Chain) -> ET.Element:
    g = ET.Element('g')
    for e in chain.edges:
        g.append(from_edge(e))
    return g


def from_dcel(dcel: DCEL) -> ET.Element:
    g = ET.Element('g')
    for b, vs in dcel.boundary_chains.items():
        for _, chain in vs.items():
            g.append(from_chain(chain))
    return g


def from_gml(gml: GMLFeature,
             attribs: Optional[Dict[str, str]] = None,
             on_click: Optional[ON_CLICK] = click) -> Optional[ET.Element]:
    if not gml.geometry:
        return None
    if attribs is None:
        attribs = {}
    attribs['id'] = gml.id
    if 'class' not in attribs:
        attribs['class'] = f'{gml.type} {gml.name}'
    if on_click:
        o = on_click(gml)
        if o:
            attribs['onclick'] = o
    g = ET.Element('g', attribs)

    return g


def add_geometry(g: ET.Element, gml: GMLFeature, d: str,
                 attribs: Dict[str, str] = {}) -> ET.Element:
    attribs.update({'d': d, 'id': f'p_{gml.id}'})
    return ET.SubElement(g, 'path', attribs)


def from_transition(transition: Transition) -> Optional[ET.Element]:
    g = from_gml(transition)
    if g:
        attribs = {'class': transition.duality.type if transition.duality else ''}
        add_geometry(g, transition, d=pathForLine(
            transition.geometry), attribs=attribs)
    return g


def from_boundary(boundary: Boundary) -> Optional[ET.Element]:
    g = from_gml(boundary)
    if g is not None:
        add_geometry(g, boundary, d=pathForLine(boundary.geometry))
    return g


def from_cell(cell: Cell) -> Optional[ET.Element]:
    g = from_gml(cell)
    if g is not None:
        attribs = {'fill-rule': 'evenodd'}
        add_geometry(g, cell, d=pathForPolygon(cell.geometry), attribs=attribs)
        for b in cell.boundary:
            e = from_boundary(b)
            if e:
                g.append(e)
    return g


def from_state(state: State) -> Optional[ET.Element]:
    g = from_gml(state)
    if g:
        (x, y) = state.geometry.coords[0]
        attribs = {'class': state.duality.type if state.duality else '',
                   'r': '20', 'cx': str(x), 'cy': str(-y)}
        ET.SubElement(g, 'circle', attribs)
        if state.name:
            label = ET.SubElement(
                g, 'text', {'x': str(x - 50), 'y': str(-(y - 50))})
            label.text = state.name
    return g


def from_layer_graph(layer: Layer) -> ET.Element:
    g = ET.Element('g', {'class': 'graph'})
    for i in layer.graph.nodes():
        n = from_state(layer.states[i])
        if n:
            g.append(n)
    return g


def from_layer(layer: Layer, with_graph: bool = False) -> ET.Element:
    attribs = {'class': f'{layer.type} {layer.cls_name} {layer.usage}'}
    g = cast(ET.Element, from_gml(layer, attribs))
    g_cells = ET.SubElement(g, 'g', {'class': 'cells'})
    for cell in layer.cells.values():
        e = from_cell(cell)
        if e:
            g_cells.append(e)
    if with_graph:
        g.append(from_layer_graph(layer))
    return g


def from_map(map_: Map, with_graph: bool = False) -> ET.Element:
    g = cast(ET.Element, from_gml(map_))
    min_x, _, max_x, _ = map_.geometry.bounds
    dx = max_x - min_x + 500
    for i, (_, layer) in enumerate(sorted(map_.layers.items())):
        e = from_layer(layer, with_graph=with_graph)
        if e:
            gl = ET.SubElement(
                g, 'g', {'transform': f'translate({i * dx}, 0)'})
            gl.append(e)
    return g
