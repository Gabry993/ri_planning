from typing import Optional

from map import Boundary, Cell, Layer, LineString, Polygon, State, Transition


def add_boundary_to_cell(cell: Cell, boundary: Boundary) -> None:
    cell.boundary.append(boundary)
    boundary.cells.append(cell)


def add_boundary_to_cells(layer: Layer, boundary: Boundary, cell1: Cell,
                          cell2: Cell) -> None:
    cell1.boundary.append(boundary)
    cell2.boundary.append(boundary)
    boundary.cells = [cell1, cell2]
    boundary.duality.set_states(cell1.duality, cell1.duality, layer=layer)


def add_boundary(layer: Layer,
                 ngeometry: LineString,
                 boundary: Optional[Boundary] = None,
                 type: str = 'CellSpaceBoundary') -> Boundary:
    i = len(layer.boundaries) + 1
    b = Boundary(f'{layer.id}B{i}')
    b.geometry = ngeometry
    layer.boundaries[b.id] = b
    if boundary:
        b.type = boundary.type
    else:
        b.type = type
    t = Transition(f'{layer.id}T{i}')
    t.duality = b
    b.duality = t
    layer.transitions[t.id] = t
    return b


def add_copy_of_cell(layer: Layer, cell: Cell, ngeometry: Polygon) -> Cell:
    i = len(layer.cells) + 1
    state = cell.duality
    ncell = Cell(f'{layer.id}C{i}')
    ncell.type = cell.type
    ncell.function = cell.function
    ncell.usage = cell.usage
    ncell.cls_name = cell.cls_name
    ncell.geometry = ngeometry
    layer.cells[ncell.id] = ncell
    nstate = add_copy_of_state(layer, state, uid=f'{layer.id}S{i}')
    ncell.duality = nstate
    nstate.duality = ncell
    nstate.geometry = ngeometry.representative_point()
    layer.rtree_index.add(i, ngeometry.bounds, obj=nstate.id)
    return ncell


def add_copy_of_state(layer: Layer, state: State, uid: str) -> State:
    nstate = State(uid)
    nstate._meta = state._meta
    layer.states[uid] = nstate
    return nstate
