# import itertools
import math
from time import time
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple, Union)

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize

from ...utilities import n_grams, orient
from .bezier import (Bezier, _num, adjust_k, cost, curvature, elevate_degree,
                     k_at, line)
from .bezier_node import Node, State

# #TODO:250 se i punti di controllo sono collineare la disnza a meno importanza


class Path:

    cp3: np.ndarray
    cp5: np.ndarray

    def __init__(self,
                 cp3: Union[Sequence[float], np.ndarray] = [],
                 cp5: Union[Sequence[float], np.ndarray] = []) -> None:
        if len(cp3):
            self.cp3 = cp3  # np.array(cp3)
        if len(cp5):
            self.cp5 = cp5  # np.array(cp5)

    @classmethod
    def from_dict(cls, value: Dict) -> 'Path':
        return cls(cp3=value.get('cp3', []), cp5=value.get('cp5', []))

    def reverse(self) -> 'Path':
        try:
            cp3 = [np.array(v)[::-1] for v in self.cp3][::-1]
        except AttributeError:
            cp3 = []
        try:
            cp5 = [np.array(v)[::-1] for v in self.cp5][::-1]
        except AttributeError:
            cp5 = []
        return Path(cp3=cp3, cp5=cp5)

    @property
    def to_dict(self) -> Dict[str, List[List[float]]]:
        return {
            k: [i.tolist() for i in v]
            for k, v in (("cp3", self.cp3), ("cp5", self.cp5)) if len(v)
        }

    def all_control_points(self, degree: int = 3) -> Iterator[np.ndarray]:
        if degree == 3:
            return iter(self.cp3)
        elif degree == 5:
            return iter(self.cp5)
        else:
            return iter([])

    def curve(self,
              degree: int = 3,
              step: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Parametrization of the curve by length: step is in cm!!

        If step is None,  then return the Bezier parametrization
        of the curve (which is not uniform)
        """
        _xy = []
        vxvy = []
        for cp in self.all_control_points(degree):
            c = Bezier(cp, derivatives=True)
            _xy.append(c.c0)
            vxvy.append(c.c1)
        xy = np.concatenate(_xy)
        vxvy = np.concatenate(vxvy).T
        angle = np.arctan2(vxvy[1], vxvy[0])
        angle = np.unwrap(angle)
        if step is None:
            return xy, angle
        else:
            s = self.s(degree=degree)
            L = s[-1]
            l = np.arange(0, stop=L, step=step)
            return (interpolate.interp1d(s, xy.T)(l).T,
                    interpolate.interp1d(s, angle)(l))

    def curve_fn(
        self,
        degree: int = 3,
        step: float = 1
    ) -> Tuple[Callable[[float], np.ndarray], Callable[[float], float]]:
        _xy = []
        vxvy = []
        for cp in self.all_control_points(degree):
            c = Bezier(cp, derivatives=True)
            _xy.append(c.c0)
            vxvy.append(c.c1)
        xy = np.concatenate(_xy)
        vxvy = np.concatenate(vxvy).T
        angle = np.arctan2(vxvy[1], vxvy[0])
        angle = np.unwrap(angle)
        # cannot use self.s (this uses a continous curve)
        ss = np.diff(xy)
        # ss = self.s(degree=degree)
        ss = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        ss = np.insert(ss, 0, 0)
        return (interpolate.interp1d(ss, xy.T, fill_value="extrapolate"),
                interpolate.interp1d(ss, angle, fill_value="extrapolate"))

    def plot(self,
             with_controls: bool = True,
             color: str = 'k',
             degree: int = 3,
             **kwargs: Any) -> None:
        from matplotlib.pyplot import plot
        for cp in self.all_control_points(degree):
            # c = next(color)
            x, y = Bezier(cp, derivatives=False).c0.T
            plot(x, y, color=color, **kwargs)
            if with_controls:
                x, y = cp.T
                plot(x, y, 'o', color=color)

    def curvature(self, degree: int = 3) -> np.ndarray:
        return np.concatenate(
            [curvature(cps) for cps in self.all_control_points(degree)], 1)

    def k(self, degree: int = 3, step: Optional[float] = None) -> np.ndarray:
        """If step is not None returns the curvature parametrized by length"""
        k, ds = self.curvature(degree=degree)
        if step:
            s = np.cumsum(ds) / _num
            np.roll(s, 1)
            s[0] = 0
            l = np.arange(0, stop=s[-1], step=step)
            return interpolate.interp1d(s, k)(l)
        return k

    def ds(self, degree: int = 3) -> np.ndarray:
        _, ds = self.curvature(degree=degree)
        return ds

    def s(self, degree: int = 3) -> np.ndarray:
        ds = self.ds(degree=degree)
        s = np.cumsum(ds) / _num
        np.roll(s, 1)
        s[0] = 0
        return s

    def length(self, degree: int = 3) -> float:
        return self.s(degree=degree)[-1]

    def cost(self, degree: int = 3) -> float:
        return sum([cost(cps) for cps in self.all_control_points(degree)])

    def join(self, path2: 'Path') -> 'Path':
        path1 = self
        cp3_0 = path1.cp3
        cp3_1 = path2.cp3
        cp5_0 = path1.cp5
        cp5_1 = path2.cp5
        cp5_j_0 = cp5_0[-1]
        cp5_j_1 = cp5_1[0]
        k_0 = k_at(cp5_j_0, 1)
        k_1 = k_at(cp5_j_1, 0)
        kk = k_0 * k_1
        if kk > 1e-10:
            k = math.sqrt(kk)
            if k_0 < 0:
                k = -k
        else:
            k = 0

        if len(cp5_j_1) > 2:
            adjust_k(cp5_j_1, k)
            pass

        if len(cp5_j_0) > 2:
            cp5_j_0 = np.flipud(cp5_j_0)
            adjust_k(cp5_j_0, -k)
            cp5_j_0 = np.flipud(cp5_j_0)

        cp3 = cp3_0 + cp3_1
        cp5 = cp5_0[:-1] + [cp5_j_0, cp5_j_1] + cp5_1[1:]
        return Path(cp3=cp3, cp5=cp5)


class PathWithNodes(Path):

    tol: float
    nodes: Sequence[Node]

    @property
    def cp3(self) -> np.ndarray:
        return list(self.all_control_points(3))
        # return np.asarray(list(self.all_control_points(3)))

    @property
    def cp5(self) -> np.ndarray:
        return list(self.all_control_points(5))
        # return np.asarray(list(self.all_control_points(5)))

    def cache(self) -> 'Path':
        return Path(cp3=self.cp3, cp5=self.cp5)

    def __init__(self, nodes: Sequence[Node]):
        super().__init__()
        self.nodes = np.array(nodes)
        self.tol = -1

    @property
    def segments(self) -> Iterator[Tuple[Node, Node]]:
        return n_grams(self.nodes, 2)

    @staticmethod
    def control_points(node1: Node,
                       node2: Node,
                       x1: Optional[State] = None,
                       x2: Optional[State] = None,
                       degree: int = 3) -> np.ndarray:
        if degree == 5:
            return np.concatenate([node1._cp5[1], node2._cp5[0]], 0)
        else:
            return np.concatenate(
                [node1.cp_fast(x1, 1),
                 node2.cp_fast(x2, 0)[::-1]])

    @staticmethod
    def curvature_of_segment(node1: Node,
                             node2: Node,
                             x1: Optional[State] = None,
                             x2: Optional[State] = None,
                             degree: int = 3) -> np.ndarray:
        cp = PathWithNodes.control_points(node1, node2, x1, x2, degree=degree)
        return curvature(cp)

    @staticmethod
    def cost_of_segment(node1: Node,
                        node2: Node,
                        x1: Optional[State] = None,
                        x2: Optional[State] = None,
                        degree: int = 3) -> float:
        if 1 not in node1.cells and 0 not in node2.cells:
            return 0
        cp = PathWithNodes.control_points(node1, node2, x1, x2, degree=degree)
        return cost(cp)

    def all_control_points(self, degree: int = 3) -> Iterator[np.ndarray]:
        return (self.control_points(node1, node2, degree=degree)
                for node1, node2 in self.segments)

    def optimize_full(self,
                      tol: float = 0.001,
                      G2: bool = True,
                      **kwargs: Any) -> None:
        for node in self.nodes:
            node.setup_cp_fast()
        if tol > 0:
            for chain, is_line in self.chains():
                if not is_line:
                    # #TODO:170 make this faster
                    self.optimize_nodes(chain, tol=tol, **kwargs)
        self.update_control_points_for_quintic_bezier()
        if G2:
            self.make_G2()
        self.tol = tol

    def optimize_nodes(self,
                       node_indices: Iterable[int],
                       tol: float = 0.01,
                       maxiter: int = 3000,
                       debug: bool = False,
                       **kwargs: Any) -> Tuple[Any, Dict]:
        if debug:
            start_time = time()
        optimized = {}
        j = 0
        x_slices = {}
        segments = set()
        initial_state: List[float] = []
        constraints = []
        for i in node_indices:
            if i < 0 or i >= len(self.nodes):
                continue
            node = self.nodes[i]
            n = node.number_of_free_parameters()
            if n > 0:
                if 0 in node.cells and i > 0:
                    segments.add((i - 1, i))
                if 1 in node.cells and i < len(self.nodes) - 1:
                    segments.add((i, i + 1))
                x_slices[i] = range(j, j + n)
                x = node.x()
                optimized[i] = {'old_x': x}
                initial_state.extend(x)
                constraints.extend(node.constraints(j))
                j += n

        if not segments:
            if debug:
                print(f"{i} No optimization")
            return (None, optimized)

        # print x_slices
        # print segments
        # print initial_state
        # print constraints

        # print [(i, j) for i, j in segments]
        # print [((self.nodes[i], x_slices.get(i, None)),
        #        (self.nodes[j], x_slices.get(j))) for i, j in segments]

        def cost(x: List[float]) -> float:
            node_x = {i: x[r] for i, r in x_slices.items()}
            return 100 * sum([
                self.cost_of_segment(self.nodes[i], self.nodes[j],
                                     node_x.get(i, None), node_x.get(j, None))
                for i, j in segments
            ])

        def update(x: List[float]) -> None:
            for i, r in x_slices.items():
                self.nodes[i].set_x(x[r])
                optimized[i]['new_x'] = x[r]

        if debug:
            end_time = time()
        r = minimize(cost,
                     initial_state,
                     method='COBYLA',
                     constraints=tuple(constraints),
                     options={
                         'rhobeg': 0.1,
                         'maxiter': maxiter,
                         'tol': tol
                     })

        if r.success or r.status == 2 or r.status == 4:
            update(r.x)
        if debug:
            st = 1000 * (time() - start_time)
            et = 1000 * (end_time - start_time)
            print(f"{node_indices} Optimization took {st} ms ({et} ms) "
                  f"-> {r.success} with {r.nfev} evaluations. "
                  f"New cost: {r.fun}")
        if not r.success and debug:
            print(r.message)
        return (r, optimized)

    def chains(self) -> List[Tuple[range, bool]]:
        # we assume that segments are well paired
        # (i.e. that no segment has nodes1.state[1] different
        # from nodes2.state[0])
        is_line = [1 not in n.cells for n in self.nodes[:-1]]
        borders = [0] + [
            k + 1 for k, (i, j) in enumerate(n_grams(is_line, 2)) if i is not j
        ] + [len(self.nodes) - 1]
        # print borders
        all_indices = range(0, len(self.nodes))
        return [(all_indices[i:j + 1], is_line[i])
                for i, j in n_grams(borders, 2)]

    def plot(self,
             with_controls: bool = True,
             color: str = 'k',
             degree: int = 3,
             with_nodes: bool = False,
             **kwargs: Any) -> None:
        # if color:
        #     col_gen = itertools.cycle([color])
        # else:
        #     col_gen = itertools.cycle('bgrcmyk')
        super(PathWithNodes, self).plot(with_controls=with_controls,
                                        color=color,
                                        degree=degree,
                                        **kwargs)
        if not with_controls and with_nodes:
            for node in self.nodes:
                # node.plot(color=col_gen)
                node.plot()

    def plot_with_cubic_bezier(self,
                               with_controls: bool = True,
                               with_nodes: bool = False,
                               color: str = 'k',
                               **kwargs: Any) -> None:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
        from matplotlib.pyplot import figure, plot

        fig = figure()
        ax = fig.add_subplot(111)
        for node1, node2 in self.segments:
            cp = self.control_points(node1, node2, degree=3)
            path = MPath([c.coords[0] for c in cp])
            patch = PathPatch(path, facecolor='none', edgecolor='r', lw=4)
            ax.add_patch(patch)
            if with_controls:
                x, y = cp.T
                plot(x, y, 'o', color=color)
        if not with_controls and with_nodes:
            for node in self.nodes:
                node.plot()

    def plot_k(self) -> None:
        from matplotlib.pyplot import plot
        l = [n._k3 for n in self.nodes][1:-1]
        n = _num
        r = np.concatenate([[[n / 2 + i * n - 1, p[0]], [n + i * n - 1, p[0]],
                             [n + i * n - 1, p[1]],
                             [3 * n / 2 + i * n - 1, p[1]]]
                            for i, p in enumerate(l)])
        plot(r.T[0], r.T[1], 'r')
        plot(self.k(degree=3), 'b')

    #  def _update(self):
    #      self.segments_cache=[]
    #      for node1, node2 in self.segments:
    #          cp=control_points(node1, node2)
    #          c0=Bezier(cp, derivatives=False)
    #          self.segments_cache.append(node1, node2, control_points, c0)

    def reverse(self) -> 'PathWithNodes':
        nodes = [Node.reverse(n) for n in self.nodes[::-1]]
        return PathWithNodes(nodes)

    def update_control_points_for_quintic_bezier(self) -> None:
        for n in self.nodes:
            n._cp5 = {}
            n._k3 = {}
            n._line5 = {}
            n._l5 = {}
            n._h5 = {}
            n._h5_max = {}

        for node1, node2 in self.segments:
            cp3 = self.control_points(node1, node2, degree=3)
            if 1 not in node1.cells:
                # it's a line:
                node1._cp5[1] = cp3[:1]
                node2._cp5[0] = cp3[-1:]
                node1._k3[1] = 0
                node2._k3[0] = 0
                continue

            node1._k3[1] = k_at(cp3, 0)
            node2._k3[0] = k_at(cp3, 1)

            cp5 = elevate_degree(elevate_degree(cp3))
            node1._cp5[1] = cp5[:3]
            node2._cp5[0] = cp5[3:]

            e = cp5[3] - cp5[2]
            ne = np.linalg.norm(e)
            e = e / ne

            l1 = cp5[1] - cp5[0]
            nl1 = np.linalg.norm(l1)
            l1 = l1 / nl1
            h1 = abs(np.cross(l1, cp5[2] - cp5[1]))
            a = abs(np.cross(e, l1))

            # #TODO:100 find another rule for the non convex case
            # (which are rare)

            if a > 1e-10 and orient(cp5[:3]) == orient(cp5[1:4]):
                e1 = e / a
                node1._line5[1] = line(e1, cp5[2], h1)
                node1._l5[1] = nl1
                node1._h5[1] = h1
                node1._h5_max[1] = ne * a + h1

            l2 = cp5[-2] - cp5[-1]
            nl2 = np.linalg.norm(l2)
            l2 = l2 / nl2
            h2 = abs(np.cross(l2, cp5[-3] - cp5[-2]))
            a = abs(np.cross(e, l2))

            if a > 1e-10 and orient(cp5[-4:-1]) == orient(cp5[-3:]):
                e2 = -e / a
                node2._line5[0] = line(e2, cp5[3], h2)
                node2._l5[0] = nl2
                node2._h5[0] = h2
                node2._h5_max[0] = ne * a + h2


# #TODO:230 non funziona se convesso

    def make_G2(self) -> None:
        for n in self.nodes:
            if len(n._k3) > 1:
                if n._k3[0] * n._k3[1] > 1e-10:
                    k = math.sqrt(n._k3[0] * n._k3[1])
                    if n._k3[0] < 0:
                        k = -k
                else:
                    k = 0
                n._k5 = k
                # print 'k', k
                # print n._k3, '~~>', n._k5
                for side in n.cells:
                    if side not in n._line5:
                        continue
                    h = (5.0 / 4.0) * ((n._l5[side])**2) * abs(n._k5)
                    # h = n._h5[s]
                    # print 'h', h, 'should be less than', n._h5_max[s]
                    n._h5[side] = h
                    if h > n._h5_max[side]:
                        h = n._h5_max[side]
                        # print h, n._h5_max[s]
                        # raise NameError("AHI")
                    if side == 0:
                        i = 0
                    else:
                        i = 2
                    # print n._cp5[s][i]
                    # print 'adjustM', h, n._cp5[s][i]
                    n._cp5[side][i] = n._line5[side](h)
                    # print '->', n._cp5[s][i]

                    # print "-->", n._cp5[s][i]
            else:
                n._k5 = list(n._k3.values())[0]

                # CHANGED

                n._k5 = 0
                for side in n.cells:
                    if side not in n._line5:
                        continue
                    n._cp5[side][2 * side] = n._line5[side](0)
"""
    def Test_degree(self):
        degrees = [3, 5]
        for node1, node2 in self.segments:
            cp = [self.control_points(node1, node2, degree=d) for d in degrees]
            for s in [0, 1]:
                assert np.isclose(k_at(cp[0], s),
                                  k_at(cp[1], s),
                                  rtol=1e-05,
                                  atol=1e-08,
                                  equal_nan=False)
        print('Curvatures of degree 3 and degree 5 curves correspond')
"""
