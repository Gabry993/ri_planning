import math
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple,
                    TypeVar, Union, cast)

import numpy as np
import shapely as s

from ...utilities import Vector, unit

State = Sequence[float]
Constraint = Callable[[State], float]
ConstraintList = List[Dict[str, Any]]

MIN_L = 0.01
GAMMA = 0
ANGLE = 1
LENGTH = 2
MIN_GAMMA = 0.01
MAX_GAMMA = 0.99

SelfNode = TypeVar("SelfNode", bound="Node")

sides = (0, 1)

TV = TypeVar("TV")


def reverse_side(value: Dict[int, TV]) -> Dict[int, TV]:
    return {1 - k: v for k, v in value.items()}


class Node:

    cells: Dict[int, s.Polygon]
    boundary: Optional[s.LineString]
    _free: Tuple[bool, bool, bool]
    default_free: Tuple[bool, bool, bool] = (False, False, False)
    _position: Optional[s.Point]
    _position_np: Vector
    _gamma: Optional[float]
    _angle: Optional[float]
    _length: Dict[int, float]
    symmetric: bool
    max_l: Dict[int, float]
    index_in_x: List[int]
    _cp3: List[Vector]
    _cp5: Dict[int, List[Vector]]
    cp_fast: Callable[[Optional[State], int], np.ndarray]

    def __init__(self,
                 cells: Dict[int, s.Polygon],
                 boundary: Optional[s.LineString] = None,
                 free: Optional[Tuple[bool, bool, bool]] = None,
                 position: Optional[Union[s.Point, Vector]] = None,
                 gamma: Optional[float] = None,
                 angle: Optional[float] = None,
                 length: Dict[int, float] = {},
                 x: Optional[State] = None):
        self.cells = cells
        self.boundary = boundary
        self._gamma = None
        self._position = None
        # self._cached_x = None
        # self._cached_control_for_x = [None for s in self.cells]
        # self.control = [None for s in self.cells]
        self.position = position
        self._length = {**length}
        self.gamma = gamma
        self.angle = angle
        self.free = free if free is not None else self.default_free
        # symmetric True means same length on both sides of the node
        self.symmetric = False
        self.max_l = {}
        self.set_x(x)
        for side in sides:
            if side in cells:
                minx, miny, maxx, maxy = cells[side].bounds
                self.max_l[side] = max(maxx - minx, maxy - miny)
            else:
                self.max_l[side] = float('inf')

    @property
    def position(self) -> Optional[s.Point]:
        return self._position

    @position.setter
    def position(self, value: Union[None, s.Point, Vector]) -> None:
        if value:
            if isinstance(value, s.Point):
                self._position = value
                self._position_np = np.array(value.coords[0])
            else:
                self._position = s.Point(value)
                self._position_np = np.array(value)

    @property
    def gamma(self) -> Optional[float]:
        return self._gamma

    @gamma.setter
    def gamma(self, value: Optional[float]) -> None:
        if value is not None and value != self._gamma:
            self._gamma = min(MAX_GAMMA, max(MIN_GAMMA, value))
            self.position = self.position_for(self._gamma)

    @property
    def free(self) -> Tuple[bool, bool, bool]:
        return self._free

    @free.setter
    def free(self, value: Tuple[bool, bool, bool]) -> None:
        if not len(value) == 3:
            raise ValueError("Wrong number of degrees of freedom")
        self._free = value
        j = 0
        self.index_in_x = []
        for i, v in enumerate(value):
            self.index_in_x.append(j)
            if v:
                j = j + 1
        self.setup_cp_fast()

    @staticmethod
    def reverse(node: SelfNode) -> SelfNode:
        angle = (node.angle + math.pi) if node.angle is not None else None
        c = node.__class__(cells=reverse_side(node.cells),
                           boundary=node.boundary,
                           free=node.free,
                           position=node.position,
                           gamma=node.gamma,
                           angle=angle,
                           length=reverse_side(node._length),
                           x=None)
        c.max_l = reverse_side(node.max_l)
        c._cp5 = {(1 - k): v[::-1] for k, v in node._cp5.items()}
        c.setup_cp_fast()
        return c

    def max_length_for_angle(self, side: int, angle: float) -> float:
        if side in self.cells:
            if side not in self.max_l:
                return float('inf')
            l = self.max_l[side]
            if s == 0:
                v = unit(angle + math.pi)
            else:
                v = unit(angle)
            p = self.position
            if p is None:
                return float('inf')
            border = self.cells[side].exterior
            coords = np.asarray(p.coords[0])
            line = s.LineString([coords + v * 1, coords + v * 2 * l])
            pi = line.intersection(border)
            if not pi.is_empty:
                l = pi.distance(p) * 0.99
        else:
            l = float('inf')
        return l

    def set_length(self, value: float, side: Optional[int]) -> None:
        if side is None:
            for i in [0, 1]:
                if i in self.cells:
                    self._length[i] = value
        elif side in self.cells:
            self._length[side] = value
            if self.symmetric and 1 - side in self.cells:
                self._length[1 - side] = value

    def length(self, side: Optional[int]) -> float:
        if side is None or self.symmetric:
            return list(self._length.values())[0]
        else:
            return self._length[side]

    def number_of_free_parameters(self, side: Optional[int] = None) -> int:
        num = self.free.count(True)
        if side is not None and side in self.cells and self.free[LENGTH]:
            num -= 1
        if (side is None and self.free[LENGTH] and not self.symmetric
                and len(self.cells) == 2):
            num += 1
        return num

    def max_length(self, cell: s.Polygon, angle: float) -> float:
        # assert state is not None
        p = self.position
        (minx, miny, maxx, maxy) = cell.bounds
        max_l: float = max(maxx - minx, maxy - miny)
        border = cell.exterior
        v = unit(angle)
        if p is None:
            raise ValueError("No position")
        coords = np.asarray(p.coords[0])
        line = s.LineString([coords + v * 1, coords + v * 2 * max_l])
        pi = line.intersection(border)
        if not pi.is_empty:
            return cast(float, pi.distance(p) * 0.99)
        else:
            # print 'EMPTY'
            # print angle
            return max_l

    def distance_of_x(self, side: int, x: State, y: State) -> float:
        d1: float = self.node_position(x).distance(self.node_position(y))
        d2: float = self.control_point_position(s, x).distance(
            self.control_point_position(s, y))
        return d1 + d2

    def number_of_free_lengths(self) -> int:
        num = len(self.cells)
        if self.symmetric and num == 2:
            num -= 1
        return num

    # overwrite
    def constraints(self, j: int) -> ConstraintList:
        constraints = []

        def min_l(k: int) -> Constraint:
            return lambda x: x[k] - MIN_L

        def max_l(k: int, m: float) -> Constraint:
            return lambda x: m - x[k]

        def inside(k: int) -> Constraint:
            return lambda x: self.margin(k, x[j:])

        if self.free[LENGTH]:
            k = self.index_in_x[LENGTH] + j
            for side in self.cells:
                constraints.append({'type': 'ineq', 'fun': min_l(k)})
                if self.symmetric:
                    break
                # constraints.append({'type': 'ineq',
                #                     'fun': lambda x: self.max_l-x[li]})
                k += 1

        # control point must be inside the corresponing state:
        if self.free[ANGLE] or self.free[GAMMA]:
            for side in self.cells:
                constraints.append({'type': 'ineq', 'fun': inside(side)})
        elif self.free[LENGTH]:
            # we can precompute the bounds on length
            ml = []
            k = self.index_in_x[LENGTH] + j
            for side, cell in self.cells.items():
                if self.angle is None:
                    raise ValueError("Undefined angle")
                if side == 0:
                    a = self.angle + math.pi
                else:
                    a = self.angle
                _max_length = self.max_length(cell, a) / 100
                if not self.symmetric:
                    constraints.append({
                        'type': 'ineq',
                        'fun': max_l(k, _max_length)
                    })
                    k += 1
                else:
                    ml.append(_max_length)

            if ml:
                constraints.append({'type': 'ineq', 'fun': max_l(k, min(ml))})

        return constraints

    def opt_constraints(self, i: int, j: int) -> ConstraintList:
        constraints = []

        def inside(k: int) -> Constraint:
            return lambda x: self.margin(k, x[j:])

        for side in self.cells:
            if s:
                constraints.append({
                    'name': f'{i}_inside_{side}',
                    'fun': inside(side)
                })
        return constraints

    def node_position(self, x: Optional[State] = None) -> s.Point:
        if x is not None:
            (gamma, _, _) = self.state_for(x)
            return self.position_for(gamma)
        else:
            return self.position

    # overwrite
    def position_for(self, gamma: float) -> s.Point:
        return self.position

    def lenght_to_x(self) -> List[float]:
        if self.symmetric:
            return [self._length[0] * 0.01]
        else:
            return [
                length * 0.01 for side, length in self._length.items()
                if side in self.cells
            ]

    def x_to_length(self, x: State) -> Dict[int, float]:
        if self.symmetric and self.cells:
            return {side: x[0] * 100.0 for side in sides}
        else:
            vs = {}
            i = 0
            if 0 in self.cells:
                vs[0] = x[0] * 100.0
                i += 1
            if 1 in self.cells:
                vs[1] = x[i] * 100.0
            return vs

    def x(self) -> State:
        x: List[float] = []
        if self.free[GAMMA]:
            x.append(self.gamma)  # type: ignore
        if self.free[ANGLE]:
            x.append(self.angle)  # type: ignore
        if self.free[LENGTH]:
            x.extend(self.lenght_to_x())
        return x

    def opt_variables(self, i: int) -> List[Dict[str, Any]]:
        x = []
        if self.free[GAMMA]:
            x.append({
                'name': f'{i}_gamma',
                'value': self.gamma,
                'lower': MIN_GAMMA,
                'upper': MAX_GAMMA
            })
        if self.free[ANGLE]:
            x.append({
                'name': f'{i}_angle',
                'value': self.angle,
                'lower': -math.pi,
                'upper': math.pi
            })
        if self.free[LENGTH]:
            for j, l in enumerate(self.lenght_to_x()):
                x.append({
                    'name': f'{i}_length_{j}',
                    'value': l,
                    'lower': MIN_L,
                    'upper': math.inf
                })
        return x

    def set_x(self, x: Optional[State]) -> None:
        if x is None or not len(x) == self.number_of_free_parameters():
            return
        i = 0
        if self.free[GAMMA]:
            self.gamma = x[i]
            i += 1
        if self.free[ANGLE]:
            self.angle = x[i]
            i += 1
        if self.free[LENGTH]:
            self._length = self.x_to_length(x[i:])
        self.setup_cp_fast()

    def state_for(self, x: State, side: Optional[int] = None) -> State:
        state = [self.gamma, self.angle, self.length(s)]
        i = 0
        if self.free[GAMMA]:
            state[GAMMA] = x[i]
            i += 1
        if side is None or side in self.cells:
            if self.free[ANGLE]:
                state[ANGLE] = x[i]
                i += 1
            if self.free[LENGTH]:
                l = self.x_to_length(x[i:])
                if side is not None:
                    state[LENGTH] = l[side]
                else:
                    # TODO(Jerome 2023): this is a list and seems very wrong
                    # state[LENGTH] = l
                    state[LENGTH:] = l
        return state  # type: ignore

    def control_point_position(self,
                               side: int,
                               x: Optional[State] = None) -> s.Point:
        if x is not None:
            # if x != self._cached_x:
            #     self._cached_x=[None for s in self.cells]
            #     self._cached_x=x
            # elif self._cached_control_for_x[i]:
            #     # cached
            #     return self._cached_control_for_x[i]
            a: Optional[float]
            (gamma, a, l) = self.state_for(x, side=side)

            p = self.position_for(gamma)
        else:
            # if self.control[i]:
            #     return self.control[i]
            (p, a, l) = (self.position, self.angle, self.length(side))
        if p is None or l is None or a is None:
            return None
        if side not in self.cells:
            return p
        d = 1
        if len(self.cells) == 2 and side == 0:
            d = -1
        p = s.Point(np.asarray(p.coords[0]) + d * unit(a) * l)
        # if not x:
        #     self.control[i]=p
        # else:
        #     self._cached_control_for_x[i]=p
        return p

    def position_fast(self) -> Callable[[Optional[State]], Vector]:
        return lambda x: self._position_np

    def setup_cp_fast(self) -> None:
        i = 0
        p = self.position_fast()
        if self.free[GAMMA]:
            i = 1

        if self.angle is None:
            raise ValueError("Undefined angle")

        _v = unit(self.angle)

        if self.free[ANGLE]:
            ai = i

            def e(x: Optional[State], s: int) -> np.ndarray:
                if x is not None:
                    a = x[ai]
                    v = np.array([np.cos(a), np.sin(a)])
                else:
                    v = _v
                if s == 0:
                    return -v
                else:
                    return v

            i = i + 1
        else:

            def e(x: Optional[State], s: int) -> np.ndarray:
                if s == 0:
                    return -_v
                else:
                    return _v

        _l = self.length
        if self.free[LENGTH]:
            li = i

            def l(x: Optional[State], side: int) -> float:
                if x is not None:
                    return self.x_to_length(x[li:])[side]  # x[li]*100
                else:
                    return _l(side)

            i = i + 1
        else:

            def l(x: Optional[State], side: int) -> float:
                return _l(side)

        def cp_fast(x: Optional[State], side: int) -> np.ndarray:
            k = p(x)
            if side not in self.cells:
                return np.array([k])
            if isinstance(k, s.Point):
                return np.array(
                    [k.coords[0], k.coords[0] + l(x, side) * e(x, side)])
            else:
                return np.array([k, k + l(x, side) * e(x, side)])

        self.cp_fast = cp_fast

    def margin(self, i: int, x: Optional[State] = None) -> float:
        cp = self.cp_fast(x, i)[1]
        cp = s.Point(cp)
        # cp = self.control_point_position(i, x)
        cell = self.cells[i]
        if cell.contains(cp):
            return cast(float, cell.exterior.distance(cp))
        else:
            return cast(float, -cell.exterior.distance(cp))

    def plot(self) -> None:
        from matplotlib.pyplot import plot
        for cell in self.cells.values():
            x, y = cell.exterior.xy
            plot(x, y, color='grey')
        if self.free[GAMMA]:
            color = 'red'
        else:
            color = 'black'
        if self.free[LENGTH]:
            if self.symmetric and len(self.cells) == 0:
                color_c = 'yellow'
            else:
                color_c = 'red'
        else:
            color_c = 'black'
        if self.free[ANGLE]:
            shape_c = 'o'
        else:
            shape_c = 'D'
        if self.position:
            x, y = self.position.xy
            plot(x, y, '*', color=color, markersize=10)
            for side in self.cells:
                x, y = self.control_point_position(side).xy
                plot(x, y, shape_c, color=color_c, markersize=6)


# ignorare le transizioni che sono a distanza
# (cioe' maxmin dist tra i start/end poits) breve,
# tanto non dovrebbero avere grande rilevanza,
# o ev. transizioni a distanza breve e delta angolo basso o zero
# (cioe' parallele)


class T(Node):

    default_free = (True, True, True)
    boundary: s.LineString
    delta_angle: List[float]

    # overwritten
    def constraints(self, j: int) -> ConstraintList:
        constraints = super().constraints(j)
        if (self.free[GAMMA]):
            gi = self.index_in_x[GAMMA] + j
            constraints += [{
                'type': 'ineq',
                'fun': lambda x: x[gi] - 0.001
            }, {
                'type': 'ineq',
                'fun': lambda x: 0.999 - x[gi]
            }]

        return constraints

    # overwritten
    def position_for(self, gamma: float) -> s.Point:
        return self.boundary.interpolate(gamma, normalized=True)

    def plot(self) -> None:
        from matplotlib.pyplot import plot
        x, y = self.boundary.xy
        plot(x, y, color='grey', linewidth=4)
        super().plot()

    def position_fast(self) -> Callable[[Optional[State]], Vector]:
        if self.free[GAMMA]:
            # assume that the border a made by a single segment
            p1 = cast(np.ndarray, np.asarray(self.boundary.coords[0]))
            p2 = cast(np.ndarray, np.asarray(self.boundary.coords[-1]))
            d = p2 - p1

            def p(x: Optional[State]) -> Vector:
                if x is not None:
                    return p1 + d * cast(float, x[GAMMA])  # type: ignore
                else:
                    return self._position_np

            return p
        else:
            return lambda x: self._position_np


class B(Node):
    # overwritten
    default_free = (False, False, True)
