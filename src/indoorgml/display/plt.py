from functools import partial
from typing import Any, Callable, Tuple, Optional, Iterable

from matplotlib import pyplot as plt

from descartes import PolygonPatch
from indoorgml.map import Boundary, Cell, GMLFeature, Layer, State, Transition

Color = Tuple[str, float]


def plot_name(gml: GMLFeature, ax: Any = plt) -> None:
    x, y = gml.geometry.xy
    ax.text(x[0], y[0], gml.name, fontsize=8, horizontalalignment='center',
            verticalalignment='center')


def plot_gml(gml: GMLFeature, ax: Any = plt,
             color: Optional[Callable[[GMLFeature], Color]] = None,
             style: str = '.', with_name: bool = False) -> None:
    if gml.geometry:
        (x, y) = gml.geometry.xy
        if color:
            c, a = color(gml)
        else:
            c, a = 'b', 1
        ax.plot(x, y, style, color=c, alpha=a, ms=10, linewidth=5)
        if with_name:
            plot_name(gml, ax=ax)


def boundary_color(boundary: Boundary) -> Color:
    if boundary.type == 'NavigableBoundary':
        return ('g', 1)
    return ('k', 1)


def fill_geo(cell: Cell) -> Color:
    t = cell.type
    cs = {
        'CellSpace': ('k', 1),
        'ConnectionSpace': ('orange', 1),
        'TransitionSpace': ('bisque', 1),
        'GeneralSpace': ('y', 1)}
    return cs.get(t, ('bisque', 1))


def fill_semantic(cell: Cell) -> Color:
    if cell.duality.name:
        return ('khaki', 1)
    else:
        return ('gray', 1)


def fill_sensor(cell: Cell) -> Color:
    if cell.duality.name != 'NO':
        return ('c', 1)
    else:
        return ('gray', 1)


plot_state = partial(plot_gml, style='o')
plot_transition = partial(plot_gml, style='-')
plot_boundary = partial(plot_gml, style='-', color=boundary_color)


def plot_cell(cell: Cell, ax: Any = plt,
              color: Optional[Callable[[GMLFeature], Color]] = None,
              with_name: bool = False
              ) -> None:
    if color:
        c, a = color(cell)
    else:
        c, a = 'k', 0.5
    ax.add_patch(PolygonPatch(cell.geometry, facecolor=c, edgecolor='none', alpha=a))
    if with_name:
        plot_name(cell, ax=ax)


def layer_color(layer: Layer) -> Callable[[GMLFeature], Color]:
    if layer.cls_name == 'TAGS':
        return fill_semantic
    if layer.cls_name == 'SENSORS':
        return fill_sensor
    if layer.cls_name == 'TOPOGRAPHIC':
        return fill_geo
    return lambda gml: ('orange', 0.1)


def default_color(gml: GMLFeature) -> Color:
    return ('blue', 0.5)


def default_graph_color(gml: GMLFeature) -> Color:
    if isinstance(gml, Transition):
        return ('orange', 0.5)
    else:
        return ('red', 0.5)


def plot_cells(cells: Iterable[Cell], ax: Any = plt, layer: Optional[Layer] = None,
               color: Optional[Callable[[GMLFeature], Color]] = None,
               ) -> None:
    if color is None:
        if layer:
            color = layer_color(layer)
        else:
            color = default_color
    for cell in cells:
        plot_cell(cell, ax=ax, color=color)


def plot_graph(layer: Layer, ax: Any = plt,
               color: Optional[Callable[[GMLFeature], Color]] = None,
               with_name: bool = False) -> None:
    if color is None:
        color = default_graph_color
    for t in layer.transitions.values():
        plot_transition(t, color=color, ax=ax)
    for s in layer.states.values():
        plot_state(s, color=color, ax=ax)
    plt.axis('equal')


def plot_layer(layer: Layer, ax: Any = plt,
               color: Optional[Callable[[GMLFeature], Color]] = None,
               with_name: bool = False) -> None:
    plot_cells(layer.cells.values(), ax=ax, layer=layer, color=color)


def plot(gml: GMLFeature, ax: Any = plt,
         color: Optional[Callable[[GMLFeature], Color]] = None,
         with_name: bool = False) -> None:
    f = {Layer: plot_layer, Transition: plot_transition, Boundary: plot_boundary, Cell: plot_cell,
         State: plot_state}.get(gml.__class__, plot_gml)
    f(gml, ax=ax, color=color, with_name=with_name)
    plt.axis('equal')
