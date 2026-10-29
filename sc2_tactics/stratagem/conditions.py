"""Three-valued evaluation of StratagemDSL expressions (the JSON AST of the spec).

The spec carries the strategy's intent (``success`` / ``failure``) and derived
facts as the AST StratagemDSL itself exports (``registry.schema``)::

    {"kind": "call", "name": "destroyed", "args": [{"kind": "name", "name": "enemy_home"}]}
    {"kind": "binary", "op": "and", "left": ..., "right": ...}
    {"kind": "unary", "op": "not", "operand": ...}
    {"kind": "number" | "bool" | "string", "value": ...}
    {"kind": "name", "name": ...}           # an entity, a derived fact or a plan place

The semantics follow the Galaxy lowering (``backend/galaxy/expressions.py``) and
``play/worldguards.py`` of StratagemDSL:

    alive(X)            living members > 0          destroyed(X) / defeated(X)  == 0
    count(X)            living members              count_in(F, R)   members inside circle R
    in_region(F, R)     >= 1 member in R            distance(A, B)   between group centres
    nearest_distance    closest member pair         killed_in(F, R)  members of F that died in R
    targeting(A, B)     most of A's orders aim at B life_ratio(X)    sum hp / sum max hp
    mission_time()      seconds since the EPISODE started (see README: Galaxy counts since game start)
    after(t)            mission_time() >= t         elapsed(P), tokens(P), active(P): plan tracker
    construction_complete / under_construction / airborne / player_unit_count

Anything else -- or a fact the world cannot know -- evaluates to ``None``
(unknown); an unknown condition never ends an episode.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Protocol

UNKNOWN = None

SUPPORTED = {"alive", "destroyed", "defeated", "count", "count_in", "in_region", "distance", "nearest_distance",
             "killed_in", "targeting", "mission_time", "after", "elapsed", "tokens", "active", "life_ratio",
             "construction_complete", "under_construction", "airborne", "player_unit_count", "apparent_count",
             "visible_to", "path_passable", "center_passable", "path_connected"}


class World(Protocol):
    """What the evaluator asks the game state. Return None for "cannot know"."""

    def count(self, name: str) -> Optional[int]: ...
    def centroid(self, name: str) -> Optional[tuple[float, float]]: ...
    def count_in(self, name: str, region: str) -> Optional[int]: ...
    def nearest_distance(self, a: str, b: str) -> Optional[float]: ...
    def killed_in(self, force: str, region: str) -> Optional[int]: ...
    def targeting(self, a: str, b: str) -> Optional[bool]: ...
    def life_ratio(self, name: str) -> Optional[float]: ...
    def construction_complete(self, name: str) -> Optional[bool]: ...
    def airborne(self, name: str) -> Optional[bool]: ...
    def player_unit_count(self, name: str) -> Optional[int]: ...
    def seconds(self) -> float: ...
    def elapsed(self, phase: str) -> Optional[float]: ...
    def tokens(self, place: str) -> Optional[int]: ...


def _name(arg: Any) -> Optional[str]:
    return arg.get("name") if isinstance(arg, dict) and arg.get("kind") == "name" else None


def evaluate(expr: Optional[dict], world: World, derived: Mapping[str, dict] = (), _seen: frozenset = frozenset()):
    """bool / float / None (unknown)."""
    derived = dict(derived or {})
    if expr is None:
        return True
    kind = expr.get("kind")
    if kind == "bool":
        return bool(expr["value"])
    if kind == "number":
        return float(expr["value"])
    if kind == "string":
        return UNKNOWN
    if kind == "name":
        n = expr["name"]
        if n in derived and n not in _seen:
            return evaluate(derived[n], world, derived, _seen | {n})
        return UNKNOWN
    if kind == "call":
        return _call(expr, world, derived, _seen)
    if kind == "unary":
        v = evaluate(expr["operand"], world, derived, _seen)
        if expr["op"] == "not":
            return UNKNOWN if not isinstance(v, bool) else (not v)
        if expr["op"] == "-":
            return -v if isinstance(v, float) else UNKNOWN
        return UNKNOWN
    if kind == "binary":
        op = expr["op"]
        left = evaluate(expr["left"], world, derived, _seen)
        right = evaluate(expr["right"], world, derived, _seen)
        if op in ("and", "or", "implies"):
            lb = left if isinstance(left, bool) else UNKNOWN
            rb = right if isinstance(right, bool) else UNKNOWN
            if op == "implies":
                lb = UNKNOWN if lb is UNKNOWN else (not lb)
                op = "or"
            if op == "and":
                if lb is False or rb is False:
                    return False
                return True if (lb is True and rb is True) else UNKNOWN
            if lb is True or rb is True:
                return True
            return False if (lb is False and rb is False) else UNKNOWN
        if isinstance(left, bool) or isinstance(right, bool):
            if op in ("==", "!=") and isinstance(left, bool) and isinstance(right, bool):
                return (left == right) if op == "==" else (left != right)
            return UNKNOWN
        if not (isinstance(left, float) and isinstance(right, float)):
            return UNKNOWN
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/": return UNKNOWN if right == 0 else left / right
        if op == "<": return left < right
        if op == "<=": return left <= right
        if op == ">": return left > right
        if op == ">=": return left >= right
        if op == "==": return left == right
        if op == "!=": return left != right
    return UNKNOWN


def _num(v) -> Optional[float]:
    return float(v) if v is not None else UNKNOWN


def _call(e: dict, world: World, derived, _seen):
    n = e["name"]
    args = e.get("args", [])
    names = [_name(a) for a in args]
    if n in ("tokens", "active") and names and names[0]:
        k = world.tokens(names[0])
        if k is None:
            return UNKNOWN
        return float(k) if n == "tokens" else k > 0
    if n in ("path_passable", "center_passable", "path_connected"):
        return True
    if n == "visible_to":
        return True            # generated maps disable fog unless setup(fog=true)
    if n == "mission_time" and not args:
        return float(world.seconds())
    if n == "after" and len(args) == 1:
        t = evaluate(args[0], world, derived, _seen)
        return world.seconds() >= t if isinstance(t, float) else UNKNOWN
    if n == "elapsed" and names and names[0]:
        return _num(world.elapsed(names[0]))
    if n in ("alive", "destroyed", "defeated", "count") and len(names) == 1 and names[0]:
        c = world.count(names[0])
        if c is None:
            return UNKNOWN
        if n == "alive":
            return c > 0
        if n == "count":
            return float(c)
        return c == 0
    if n in ("count_in", "in_region") and len(names) == 2 and all(names):
        c = world.count_in(names[0], names[1])
        if c is None:
            return UNKNOWN
        return float(c) if n == "count_in" else c > 0
    if n == "distance" and len(names) == 2 and all(names):
        a, b = world.centroid(names[0]), world.centroid(names[1])
        if a is None or b is None:
            return UNKNOWN
        return math.hypot(a[0] - b[0], a[1] - b[1])
    if n == "nearest_distance" and len(names) == 2 and all(names):
        return _num(world.nearest_distance(names[0], names[1]))
    if n == "killed_in" and len(names) == 2 and all(names):
        return _num(world.killed_in(names[0], names[1]))
    if n == "targeting" and len(names) == 2 and all(names):
        return world.targeting(names[0], names[1])
    if n == "life_ratio" and len(names) == 1 and names[0]:
        return _num(world.life_ratio(names[0]))
    if n in ("construction_complete", "under_construction") and len(names) == 1 and names[0]:
        v = world.construction_complete(names[0])
        if v is None:
            return UNKNOWN
        return v if n == "construction_complete" else (not v and (world.count(names[0]) or 0) > 0)
    if n == "airborne" and len(names) == 1 and names[0]:
        return world.airborne(names[0])
    if n in ("player_unit_count", "apparent_count") and len(names) == 1 and names[0]:
        return _num(world.player_unit_count(names[0]))
    return UNKNOWN


def truth(expr: Optional[dict], world: World, derived: Mapping[str, dict] = ()) -> Optional[bool]:
    v = evaluate(expr, world, derived)
    return v if isinstance(v, bool) else UNKNOWN


def calls_in(expr: Any, derived: Mapping[str, dict] = (), _seen: frozenset = frozenset()) -> set[str]:
    """Every function name an expression (and the derived facts it uses) calls."""
    derived = dict(derived or {})
    out: set[str] = set()
    if isinstance(expr, dict):
        if expr.get("kind") == "call":
            out.add(expr["name"])
        if expr.get("kind") == "name" and expr.get("name") in derived and expr["name"] not in _seen:
            out |= calls_in(derived[expr["name"]], derived, _seen | {expr["name"]})
        for v in expr.values():
            if isinstance(v, (dict, list)):
                out |= calls_in(v, derived, _seen)
    elif isinstance(expr, list):
        for v in expr:
            out |= calls_in(v, derived, _seen)
    return out


def unsupported_calls(expr: Any, derived: Mapping[str, dict] = ()) -> set[str]:
    return {c for c in calls_in(expr, derived) if c not in SUPPORTED}
