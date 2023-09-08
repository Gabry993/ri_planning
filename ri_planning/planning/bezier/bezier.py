from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.special import binom

from ...utilities import orient

ControlPoint = np.ndarray  # TODO: 2D, float


def line(e: ControlPoint, p: ControlPoint,
         t0: float) -> Callable[[float], ControlPoint]:

    def f(t: float) -> ControlPoint:
        return p + e * (t - t0)

    return f


"""
def distance_of_bezier_control_points(n1, n2):
    s = sum([a.distance(b) for a, b in zip(n1, n2)])
    # print s
    return s
"""


def Bernstein(n: int, k: int) -> Callable[[np.ndarray], np.ndarray]:
    """Bernstein polynomial."""
    coeff = binom(n, k)

    def _bpoly(x: np.ndarray) -> np.ndarray:
        return coeff * x**k * (1 - x)**(n - k)

    return _bpoly


_num = 0
_B: List[List[np.ndarray]] = [[]]


def set_num(num: int) -> None:
    global _num
    global _B
    _B = [[]]
    _num = num
    t = np.linspace(0, 1, num=num)
    # c = [None]
    for i in range(1, 7):
        B: List[np.ndarray] = []
        for j in range(i):
            B.append(Bernstein(i - 1, j)(t))
        _B.append(B)


set_num(100)

# A curve with 0-th, 1-th and 2-th order derivatives that could be empty


@dataclass
class Curve:
    c0: np.ndarray = np.zeros(0)
    c1: np.ndarray = np.zeros(0)
    c2: np.ndarray = np.zeros(0)


def Bezier(points: np.ndarray,
           curve: bool = True,
           derivatives: bool = True) -> Curve:
    N = len(points) - 1
    c = Curve()
    if curve:
        c.c0 = np.zeros((_num, 2))
        for ii, p in enumerate(points):
            c.c0 += np.outer(_B[N + 1][ii], p)
    if derivatives:
        c.c1 = np.zeros((_num, 2))
        c.c2 = np.zeros((_num, 2))
        points1 = N * np.diff(points, axis=0)
        points2 = (N - 1) * np.diff(points1, axis=0)
        for ii, p in enumerate(points1):
            c.c1 += np.outer(_B[N][ii], p)
        for ii, p in enumerate(points2):
            c.c2 += np.outer(_B[N - 1][ii], p)
    return c


def cost_bending(points: np.ndarray) -> float:
    k, ds = curvature(points)
    dk = np.gradient(k)
    n = len(k)
    c = np.sum(k * k * ds / n + dk * dk / ds * n)
    return c


# def cost_bending(points):
#     k, ds = curvature(points)
#     dk = np.gradient(k)
#     c = np.sum(k * k * ds + dk * dk / ds) / len(k)
#     return c


def cost_jerk(points: np.ndarray) -> float:
    k, ds = curvature(points)
    dk = np.gradient(k)
    n = len(k)
    c = np.sum((k**4) * ds / n + dk * dk / ds * n)
    return c


cost = cost_jerk


def curvature(points: np.ndarray) -> np.ndarray:
    c = Bezier(points, curve=False, derivatives=True)
    return curvature_fast(c.c1, c.c2)


# def curvature_fast(curve1, curve2):
#     dx, dy = curve1.T
#     ddx, ddy = curve2.T
#     ds = np.linalg.norm(curve1, axis=1)
#     k = (dx * ddy - dy * ddx) / (ds ** 3)
#     return k, ds


def curvature_fast(curve1: np.ndarray,
                   curve2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx, dy = curve1.T
    ddx, ddy = curve2.T
    ds2 = dx * dx + dy * dy
    ds = np.sqrt(ds2)
    k = (dx * ddy - dy * ddx) / (ds * ds2)
    return k, ds


#  def curvature_very_fast(points):
#      n=np.zeros((_num,  1))
#      d=np.zeros((_num,  1))
#      points1=3*np.diff(points, axis=0)
#      points2=2*np.diff(points1, axis=0)
#
#      for i, j in itertools.product(range(3), range(2)):
#          n+=np.outer(_B3xB2[3*i+j], np.cross(points1[i], points2[j]))
#      for i, j in itertools.product(range(3), range(3)):
#          d+=np.outer(_B3xB3[3*i+j], np.inner(points1[i], points1[j]))
#
#      d=np.sqrt(d)
#      return n/(d**3), d


def elevate_degree(points: np.ndarray) -> np.ndarray:
    """Elevate the Bezier Curve with control points 'points' of 1 degree"""
    points0 = np.concatenate([points[:1], points], 0)
    points1 = np.concatenate([points, points[-1:]], 0)
    # print points0, points1
    N = float(len(points))
    return np.array([(p1 * k + p2 * (N - k)) / N
                     for k, (p1, p2) in enumerate(zip(points0, points1))])


def k_at(points: np.ndarray, side: int) -> float:
    if side == 0:
        return k(points)
    else:
        return -k(points[::-1])


def k(points: np.ndarray) -> float:
    N = len(points) - 1
    if N < 2:
        return 0
    D = N * (points[1] - points[0])
    D2 = N * (N - 1) * (points[2] - points[1])
    S = np.linalg.norm(D)
    return np.cross(D, D2) / (S**3)


def adjust_k(cp5: np.ndarray, target_k: float) -> None:
    if orient(cp5[:3]) == orient(cp5[1:4]):
        e = cp5[3] - cp5[2]
        ne = np.linalg.norm(e)
        e = e / ne
        l = cp5[1] - cp5[0]
        nl = np.linalg.norm(l)
        l = l / nl
        a = abs(np.cross(e, l))
        h = abs(np.cross(l, cp5[2] - cp5[1]))
        if a > 1e-10:
            # print abs(k(cp5)), 'should be equal to', 0.8*h/(nl**2)
            target_h = min((5.0 / 4.0) * (nl**2) * abs(target_k), ne * a + h)
            p = line(e / a, cp5[2], h)(target_h)
            np.put(cp5, [4, 5], np.array(p))
