import itertools
import math
from typing import Any, Tuple, TypeVar, Iterable, Iterator, Union, Optional
import shapely as s
import numpy as np


from dataclasses import dataclass

import numpy as np  # type: ignore

T = TypeVar("T")
Vector = Union[Tuple[float, float], np.ndarray]


@dataclass
class Pose:
    position: Vector
    orientation: Optional[float] = None

    @property
    def reversed(self) -> 'Pose':
        return Pose(self.position, (self.orientation + np.pi) if self.orientation else None)

    def distance(self, other: 'Pose') -> float:
        return float(np.linalg.norm(np.asarray(self.position) - np.asarray(other.position)))

    def __hash__(self) -> int:
        return hash((*self.position, self.orientation))


# TODO(Jerome):  more precise array types
# see https://numpy.org/doc/stable/release/1.21.0-notes.html#added-a-mypy-plugin-for-handling-platform-specific-numpy-number-precisions


def n_grams(a: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    z = (itertools.islice(a, i, None) for i in range(n))
    return zip(*z)


def argmin(iterable: Iterable[Tuple[T, float]]) -> T:
    return min(enumerate(iterable), key=lambda x: x[1])[0]


def orient(points: np.ndarray) -> int:
    m = np.vstack([points.T, np.array([1, 1, 1])])
    vol = np.linalg.det(m)
    if (abs(vol) < 1e-6):
        return 0
    return np.sign(vol)


def unit(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)])


def to_geoms_list(geoms: Any) -> Any:
    try:
        return geoms.geoms
    except:
        return geoms


def normalize(a: float) -> float:
    return np.unwrap([0, a])[1]


def angle(vector: Vector) -> float:
    try:
        return normalize(math.atan2(vector[1], vector[0]))
    except Exception:
        return 0.0


# def angle(vector: Vector) -> float:
#     try:
#         if np.linalg.norm(vector):
#             return normalize(math.atan2(vector[1], vector[0]))
#         else:
#             return 0
#     except Exception:
#         raise ValueError('zero norm vector -> no angle')


def avg_angle(angles: Iterable[Optional[float]]) -> float:
    vangle = [a for a in angles if a is not None]
    if not vangle:
        return 0
    v = np.sum([[np.cos(a), np.sin(a)] for a in vangle], axis=0)
    return angle(v)


def are_parallel(line1: s.LineString, line2: s.LineString) -> bool:
    n1 = np.asarray(line1.coords)
    n2 = np.asarray(line2.coords)
    v1 = n1[1] - n1[0]
    v2 = n2[1] - n2[0]
    m: float = np.linalg.norm(v1) * np.linalg.norm(v2)
    return abs(v1.dot(v2)) > 0.99 * m


# def position(point: s.BaseGeometry) -> Vector:
#     return np.asarray(point.coords[0])
