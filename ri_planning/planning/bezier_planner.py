import itertools
from typing import (Any, Callable, Collection, Dict, Generator, List, Optional,
                    Set, Tuple, Type, TypeVar)

import networkx as nx
import numpy as np
import shapely as s

from ..map.map import Graph, GraphEdge, Layer, State
from ..utilities import Vector, n_grams, Pose
from .bezier import bezier_path as bp
from .bezier_cache import BezierCache, BezierSimpleGraph, CacheKey
from .bezier_utils import make_bezier, setup_path
from .planner import PlanVertex, PlanVertexUnion, StateFilter
from .visibility_planner import Plan as VisibilityPlan
from .visibility_planner import Planner as VisibilityPlanner

SelfPlan = TypeVar("SelfPlan", bound="Plan")


class Plan(VisibilityPlan):

    _bezier_path: Optional[bp.Path]
    _position_fn: Callable[[float], np.ndarray]
    _orientation_fn: Callable[[float], float]

    def __init__(self,
                 planner: Optional['Planner'] = None,
                 start: Optional[PlanVertexUnion] = None,
                 end: Optional[PlanVertexUnion] = None,
                 vertices: Optional[Tuple[PlanVertex, PlanVertex]] = None,
                 agent: Any = None):
        super().__init__(planner, start, end, vertices, agent)
        self._bezier_path = None

    @classmethod
    def from_dict(cls, planner: 'Planner', value: Dict) -> 'Plan':
        p = super().from_dict(planner, value)
        if 'bezier' in value:
            p._bezier_path = bp.Path.from_dict(value['bezier'])
        return p

    @property
    def to_dict(self) -> Dict[str, Any]:
        value = super().to_dict()
        if self.bezier_plan:
            value['bezier'] = self.bezier_plan.to_dict
        return value

    @property
    def bezier_plan(self) -> Optional[bp.Path]:
        return self._bezier_path

    @bezier_plan.setter
    def bezier_plan(self, path: bp.Path) -> None:
        self._bezier_path = path
        if path:
            self.costs = np.array([path.length(), path.cost()])
        # s = path.s()
        # TODO,  compute geometry,  ...
        curve, _ = path.curve(degree=3)
        self.geometry = s.LineString(curve)
        self._position_fn, self._orientation_fn = path.curve_fn(degree=3, step=10.0)

    def geometry_with_resolution(
            self,
            resolution: int,
            with_poses: bool = False) -> Tuple[s.LineString, np.ndarray]:
        if not self.bezier_plan:
            return super().geometry_with_resolution(resolution, with_poses)
        L = self.costs[0]
        if resolution:
            resolution = max(min(resolution, L / 5), 1)
        curve, angle = self.bezier_plan.curve(degree=5, step=resolution)

        # TODO(Jerome 2023): this is a repetition
        curve = np.around(curve)
        poses = []
        if with_poses:
            angle = np.around(angle, 2)
            poses = np.vstack([curve.T, angle]).T.tolist()
        return s.LineString(curve), poses

    def reversed(self: SelfPlan) -> SelfPlan:
        rplan = super().reversed()
        if self.bezier_plan:
            rplan.bezier_plan = self.bezier_plan.reverse()
        return rplan

    def project(self, position: Vector, delta: float = 0.0) -> Pose:
        path_s = self.geometry.project(s.Point(position)) + delta
        return Pose(self._position_fn(path_s), self._orientation_fn(path_s))


def join_bezier(plans: List[Plan]) -> bp.Path:
    if not plans[0].bezier_plan:
        raise ValueError("No Bezier path")
    bezier_plan = plans[0].bezier_plan
    for p in plans[1:]:
        bezier_plan = bezier_plan.join(p.bezier_plan)
    return bezier_plan


def join_plans_into(plan: Plan, plans: List[Plan],
                    transitions: List[GraphEdge]) -> None:
    plan.bezier_plan = join_bezier(plans)
    if len(plans) > 1:
        plan._transitions = sum([t[1:] for t in transitions],
                                [transitions[0][0]])
        plan.states = [e[1] for e in plan._transitions[:-1]]
    else:
        plan._transitions = []
        plan.states = plans[0].states
    if not plan.states[0] is plan.vertices[0].state:
        plan.states.insert(0, plan.vertices[0].state)
    if not plan.states[-1] is plan.vertices[1].state:
        plan.states.append(plan.vertices[1].state)


def subdivide_plan(
    plan: Plan, cache: BezierCache, nodes: Collection[GraphEdge]
) -> Tuple[Optional[Plan], List[Plan], Optional[Plan]]:
    H, path = plan.visibility_plan
    planner = plan.planner
    is_fixed = [t in nodes for t in plan.transitions]
    if is_fixed.count(True) < 2:
        return (None, [], None)
    indices = [i for i, fixed in enumerate(is_fixed) if fixed]
    cached_plans: List[Plan] = []
    transitions = tuple(plan.transitions)
    for i1, i2 in n_grams(indices, 2):
        key = transitions[i1:i2 + 1]
        cached_plan = cache.get_plan(key)
        if not cached_plan:
            raise ValueError(f'No cached path for key {key}')
            # G = self.cache.subgraph([s2 for s1, s2, t in key[:-1]])
            # plan = self.shortest_path(A, B, number=4, tol=tol, G=G,
            #                           cache=False, sparse=true)
        cached_plans.append(cached_plan)
    i1 = indices[0]
    i2 = indices[-1]
    if i1 == 0 and plan.states[0] is plan.transitions[0][1]:
        plan1 = None
    else:
        vertices = (plan.vertices[0],
                    PlanVertex(state=plan.states[i1],
                               transition=plan.transitions[i1]))
        plan1 = Plan(planner, vertices=vertices)
        plan1.states = plan.states[:(i1 + 1)]
        plan1.transitions = plan.transitions[:(i1 + 1)]
    if (i2 == (len(indices) - 1)
            and plan.states[-1] is plan.transitions[-1][0]):
        plan2 = None
    else:
        vertices = (PlanVertex(state=plan.states[i2],
                               transition=plan.transitions[i2]),
                    plan.vertices[1])
        plan2 = Plan(planner, vertices=vertices)
        plan2.states = plan.states[i2:]
        plan2.transitions = plan.transitions[i2:]

    if path:
        if plan1:
            if plan1.states[0] is plan1.transitions[0][0]:
                i1 += 1
                i2 += 1
                plan1.visibility_plan = (H, path[:i1 + 1])
            plan2.visibility_plan = (H, path[i2:])

    return (plan1, cached_plans, plan2)


def bezier_from_visibility_plan(layer: Layer,
                                plan: Plan,
                                fixed: Set[GraphEdge],
                                tol: float = 0.01,
                                sparse: bool = False,
                                fix_extrema_transitions: bool = True,
                                **kwargs: Any) -> bp.Path:
    visibility_graph, visibility_path = plan.visibility_plan
    path = make_bezier(layer,
                       visibility_graph,
                       visibility_path,
                       fixed=fixed,
                       sparse=sparse,
                       fix_extrema_transitions=fix_extrema_transitions)
    if path:
        pose = plan.vertices[0].pose
        start_angle = pose.orientation if pose else None
        pose = plan.vertices[1].pose
        end_angle = pose.orientation if pose else None
        setup_path(path, start_angle, 0)
        setup_path(path, end_angle, -1)
        if tol > 0:
            path.optimize_full(tol=tol, **kwargs)
    return path


def route_from_path_on_simple_cache_graph(path, graph: Graph):
    edges = list(n_grams(path, 2))
    pairs = list(n_grams(edges, 2))[::2]
    return [(x[0], y[1], list(y[0]), graph[x[0]][x[1]]) for x, y in pairs]


class Planner(VisibilityPlanner):

    cache: Optional[BezierCache]

    def __init__(self,
                 layer: Layer,
                 filter_states: Optional[StateFilter] = None,
                 min_width: float = 100,
                 use_cache: bool = False,
                 tol: float = 0.01,
                 sparse: bool = False,
                 cache_path: str = ''):
        if not filter_states:

            def _filter_states(state: State) -> bool:
                if not state.duality:
                    return False
                cell = state.duality
                if cell.usage and 'Stair' in cell.usage:
                    return True
                return False

            filter_states = _filter_states
        super().__init__(layer, filter_states, min_width)
        self.tol = tol
        self.sparse = sparse
        self.use_cache = use_cache
        if use_cache:
            self.cache = BezierCache(self, cache_path)
        else:
            self.cache = None

    def add_bezier_plan(self,
                        plan: Plan,
                        graph: Optional[Graph] = None,
                        use_cache: Optional[bool] = None,
                        sparse: Optional[bool] = None,
                        tol: Optional[float] = None,
                        local_cache: Dict[CacheKey, bp.Path] = {},
                        **kwargs: Any) -> None:
        tol = self.tol if tol is None else tol
        sparse = self.sparse if sparse is None else sparse
        use_cache = self.use_cache if use_cache is None else use_cache

        if use_cache and self.cache:
            plan1, cached_plans, plan2 = subdivide_plan(
                plan, self.cache, self.fixed)
            plans = cached_plans
            if cached_plans:
                if plan1:
                    key = tuple(plan1.transitions)
                    if key not in local_cache:
                        if plan1.visibility_plan[0] is None:
                            self.add_visibility_plan(plan1, graph=graph)
                        local_cache[key] = bezier_from_visibility_plan(
                            self.layer, plan1, self.fixed, tol, sparse,
                            **kwargs)
                    plan1.bezier_plan = local_cache[key]
                if plan2:
                    key = tuple(plan2.transitions)
                    if key not in local_cache:
                        if plan2.visibility_plan[0] is None:
                            self.add_visibility_plan(plan2, graph=graph)
                        local_cache[key] = bezier_from_visibility_plan(
                            self.layer, plan2, self.fixed, tol, sparse,
                            **kwargs)
                    plan2.bezier_plan = local_cache[key]
                bezier_plan = plans[0].bezier_plan
                if plan1:
                    bezier_plan = plan1.bezier_plan.join(bezier_plan)
                for p in plans[1:]:
                    bezier_plan = bezier_plan.join(p.bezier_plan)
                if plan2:
                    bezier_plan = bezier_plan.join(plan2.bezier_plan)
                plan.bezier_plan = bezier_plan
                return
        key = tuple(plan.transitions)
        if key not in local_cache:
            if plan.visibility_plan[0] is None:
                self.add_visibility_plan(plan, graph=graph)
            local_cache[key] = bezier_from_visibility_plan(
                self.layer, plan, self.fixed, tol, sparse, **kwargs)
        plan.bezier_plan = local_cache[key]

    def all_paths(
            self,
            start: PlanVertexUnion,
            end: PlanVertexUnion,
            number: int = -1,
            graph: Optional[nx.Graph] = None,
            path_class: Type = Plan,
            tol: Optional[float] = None,
            use_cache: Optional[bool] = None,
            sparse: Optional[bool] = None) -> Generator[Plan, None, None]:
        # to precompute the whole visibility plans
        # all_plans=VisibilityPlanner.all_path(start, end, number=number, G=G)
        # to avoid computing the whole visibility plans and start
        # instead from the topological plan
        all_plans = super().all_paths(start, end, number, graph, path_class)
        local_cache: Dict[CacheKey, bp.Path] = {}
        for plan in all_plans:
            self.add_bezier_plan(plan,
                                 tol=tol,
                                 use_cache=use_cache,
                                 sparse=sparse,
                                 local_cache=local_cache)
            yield plan

    def shortest_path(self,
                      start: PlanVertexUnion,
                      end: PlanVertexUnion,
                      graph: Optional[Graph] = None,
                      weights: Dict[str, float] = {},
                      tol: Optional[float] = None,
                      use_cache: Optional[bool] = None,
                      sparse: Optional[bool] = None,
                      path_class: Type[SelfPlan] = Plan,
                      **kwargs: Any) -> SelfPlan:
        plan = VisibilityPlanner.shortest_path(self,
                                               start=start,
                                               end=end,
                                               weights=weights,
                                               graph=graph,
                                               path_class=path_class)
        # plan=Planner.shortest_path(self, start, end)
        self.add_bezier_plan(plan,
                             graph=graph,
                             tol=tol,
                             use_cache=use_cache,
                             sparse=sparse,
                             local_cache={},
                             **kwargs)
        return plan

    def shortest_path_by_cost(self,
                              start: PlanVertexUnion,
                              end: PlanVertexUnion,
                              graph: Optional[Graph] = None,
                              weight: str = 'curvature',
                              tol: Optional[float] = None,
                              lazy_tol: Optional[float] = None) -> Plan:
        plan = Plan(self, start, end)
        tol = self.tol if tol is None else tol
        if lazy_tol and (lazy_tol > tol or lazy_tol < 0):
            graph_tol = lazy_tol
        else:
            graph_tol = tol
        graph, A, B = self.paths_graph_for_plan(plan,
                                                start,
                                                end,
                                                tol=graph_tol,
                                                graph=graph)
        path = nx.shortest_path(graph, A, B, weight=weight)
        path = route_from_path_on_simple_cache_graph(path, graph)
        plans = [d['plan'] for _, _, _, d in path]
        transitions = [ts for _, _, ts, _ in path]
        if lazy_tol:
            # need (maybe) to optimize the first and last plan
            first = plans[0]
            last = plans[-1]
            if first.bezier_plan.tol == -1 or first.bezier_plan.tol > tol:
                first.bezier_plan.optimize_full(tol=tol)
            if last.bezier_plan.tol == -1 or last.bezier_plan.tol > tol:
                last.bezier_plan.optimize_full(tol=tol)
        join_plans_into(plan, plans, transitions)
        # print 'Done'
        plan.G = graph
        return plan

    def paths_graph_for_plan(
            self,
            plan: Plan,
            start: PlanVertexUnion,
            end: PlanVertexUnion,
            tol: float = 0.01,
            graph: Optional[Graph] = None) -> BezierSimpleGraph:
        # TODO(Jerome 2023) Why are this different than start or end?
        A, B = [v.transition or v.state for v in plan.vertices]
        if not self.cache:
            self.cache = BezierCache(self, '')

        # PG = nx.MultiDiGraph(self.path_cache_graph)
        PG = nx.DiGraph(self.cache.simple_graph)
        # remove edges non in G:

        if graph is not None:
            Gc = nx.Graph(self.cache.G_cache)
            # print "Remove ghost edges"
            for a, e in self.cache.simple_graph.edges():
                if type(a[0]) != tuple:
                    # e is a curve, check if in G
                    if not all([y in graph for _, y, _ in e]):
                        PG.remove_edge(s, e)
            Gc.remove_nodes_from(n for n in self.cache.G_cache
                                 if n not in graph)
            # print "Remove ghost edges Done"
        else:
            Gc = self.cache.G_cache

        # add the missing edges:
        # print 'Add missing edges'
        data = {'length': 0, 'curvature': 0}
        if not (A in self.fixed and B in self.fixed):
            pA = set()
            pB = set()
            if A in self.fixed:
                cA = [A]
            else:
                cA = self.cache.fixed_successors[A]
            if B in self.fixed:
                cB = [B]
            else:
                cB = self.cache.fixed_predecessors[B]

            for a, b in itertools.product(cA, cB):
                if nx.has_path(PG, a, b):
                    pA.add(a)
                    pB.add(b)

            for a in pA:
                # print start, a
                for _plan in self.all_paths(start, a, tol=tol, graph=Gc):
                    PG.add_edge(start,
                                tuple(_plan.transitions),
                                plan=_plan,
                                curvature=_plan.costs[1],
                                length=_plan.costs[0])
                    PG.add_edge(tuple(_plan.transitions), a, **data)

            for b in pB:
                for _plan in self.all_paths(b, end, tol=tol, graph=Gc):
                    PG.add_edge(b,
                                tuple(_plan.transitions),
                                plan=_plan,
                                curvature=_plan.costs[1],
                                length=_plan.costs[0])
                    PG.add_edge(tuple(_plan.transitions), end, **data)

        if nx.has_path(Gc, plan.vertices[0].state, plan.vertices[1].state):
            # print 'Same comp', plan.state1, plan.state2
            # they are in the same componentent,  i.e. we can connect them
            # without traversing fixed borders.
            # This costs a lot in open graphs like h1
            if A not in PG or B not in PG[A]:
                for _plan in self.all_paths(start, end, tol=tol, graph=Gc):
                    PG.add_edge(start,
                                tuple(_plan.transitions),
                                plan=_plan,
                                curvature=_plan.costs[1],
                                length=_plan.costs[0])
                    PG.add_edge(tuple(_plan.transitions), end, **data)
        # print 'Add missing edges Done'
        return (PG, start, end)


"""
    @staticmethod
    def edges_for_path_on_multigraph(G, path, weight='weight'):
        return [(s1, s2,
                 sorted([(data.get(weight, 0), t)
                         for t, data in G[s1][s2].items()])[0][1])
                for s1, s2 in n_grams(path, 2)]
"""
"""
    def all_paths_by_cost(self,
                          start,
                          end,
                          number=-1,
                          G=None,
                          tol=0.01,
                          weight='curvature'):
        plan = Plan(self, start, end)
        G, A, B = self.paths_graph_for_plan(plan, start, end, tol=tol)
        # shortest_simple_paths bot implemented for Multi(Di)Graphs
        U = multi_to_unigraph(G, weight=weight)
        # print U.nodes()
        # print U.edges(data=True)
        for path in nx.shortest_simple_paths(U, A, B, weight=weight):
            # edges = self.edges_for_path_on_multigraph(G, path, weight=weight)
            # plans = [G[u][v][e].get('plan') for u, v, e in edges]
            edges = [(u[0], v[0], u[1]) for u, v in n_grams(path, 2) if (
                isinstance(u, tuple) and isinstance(v, tuple) and u[1] is v[1])
                     ]
            plans = [G[u][v][e].get('plan') for u, v, e in edges]
            transitions = [list(e) for u, v, e in edges]
            self.join_plans_into(plan, plans, transitions)
            yield plan

    def multiobjective_path(self,
                            start,
                            end,
                            G=None,
                            weights=None,
                            mtol=1,
                            itol=0.1,
                            tol=0.01,
                            cache=False,
                            sparse=False):
        ps = VisibilityPlanner.multiobjective_path(self,
                                                   start,
                                                   end,
                                                   G=G,
                                                   weights=weights,
                                                   mtol=mtol,
                                                   itol=itol)
        local_cache = {}
        for p in ps:
            self.add_bezier_plan(p, tol, local_cache, cache, sparse)
        return ps

    def multiobjective_path_by_geometry(self,
                                        start,
                                        end,
                                        tol=0.01,
                                        mtol=1,
                                        lazy_tol=None,
                                        G=None):
        p = Plan(self, start, end)
        if lazy_tol and (lazy_tol > tol or lazy_tol < 0):
            graph_tol = lazy_tol
        else:
            graph_tol = tol
        # print 'Start shortest_path_by_cost'
        G, A, B = self.paths_graph_for_plan(p, start, end, tol=graph_tol, G=G)
        # print 'Built graph'

        ps = {
            w: nx.shortest_simple_paths(G, A, B, weight=w)
            for w in ['length', 'curvature']
        }

        pareto_optimal_set = pareto_set_geo(p, ps, G, mtol=mtol)
        solutions = []
        for path, costs, rel_costs in pareto_optimal_set:
            plan = p.clone()
            plans = [d['plan'] for _, _, _, d in path]
            transitions = [ts for _, _, ts, _ in path]
            # plans = [G[u][v][e].get('plan') for u, v, e in edges]
            plan.rel_costs = rel_costs
            # transitions = [list(e) for u, v, e in edges]
            if lazy_tol:
                # need (maybe) to optimize the first and last plan
                first = plans[0]
                last = plans[-1]
                if first.bezier_plan.tol == -1 or first.bezier_plan.tol > tol:
                    first.bezier_plan.optimize_full(tol=tol)
                if last.bezier_plan.tol == -1 or last.bezier_plan.tol > tol:
                    last.bezier_plan.optimize_full(tol=tol)
            self.join_plans_into(plan, plans, transitions)
            solutions.append(plan)
        return solutions
"""
