"""Optional: follow the strategy's plan (its Petri net) from the observations.

Needs StratagemDSL importable (``pip install -e StratagemDSL-RL``; Python >= 3.10).
It reuses StratagemDSL's own ``play.plantracker.PlanTracker`` -- the same
token game the map script plays -- fed with the raw units of each step. The map
script keeps the real marking inside the game where the API cannot read it, so
this is an estimate; with fog disabled (the default of generated maps) it
follows the script closely.

What the env does with it (all optional):
    reward_phase      bonus the first time an episode enters each non-terminal phase
    obs/state_plan_phase   one-hot of the phases believed active
    termination="plan"     end the episode when the tracked plan reaches a terminal
    elapsed(P) / tokens(P) / active(P) in intent conditions
    info["plan_phases"], info["plan_fired"]
"""
from __future__ import annotations

from typing import Optional

from .spec import ScenarioSpec


def available() -> bool:
    try:
        import stratagemdsl.play.plantracker  # noqa: F401
        return True
    except Exception:
        return False


class PlanProgress:
    def __init__(self, spec: ScenarioSpec):
        from stratagemdsl.frontend import parse_source
        from stratagemdsl.play.scenario import Force, Scenario
        from stratagemdsl.views.routes import Location

        if not spec.strategy_source:
            raise ValueError("the spec has no strategy_source; re-export it with tools/strat2sc2te.py")
        self.spec = spec
        self.strategy = parse_source(spec.strategy_source)
        forces = {f.name: Force(f.name, f.kind, f.side, f.owner, list(f.unit_types), f.count, f.spawn, f.role,
                                list(f.composition)) for f in spec.forces}
        regions = {r.name: Location(r.name, r.center[0], r.center[1], r.radius, kind="region")
                   for r in spec.regions}
        self.scenario = Scenario(self.strategy, None, forces, regions, {})
        self.name_of = {tid: name for name, tid in spec.type_ids.items()}
        try:
            from pysc2.lib import units as pysc2_units
            self._pysc2_units = pysc2_units
        except Exception:
            self._pysc2_units = None
        self.phase_ids = spec.phase_ids
        self.reset(0)

    def reset(self, loop0: int) -> None:
        from stratagemdsl.play.observe import ForceTracker
        from stratagemdsl.play.plantracker import PlanTracker

        self.loop0 = loop0
        self.tracker = ForceTracker(self.scenario, player_id=self.spec.ally_player)
        self.pt = PlanTracker(self.strategy, self.scenario, self.tracker)
        self._inside: dict[int, frozenset] = {}
        self._timers: set = set()
        self.ever_entered: set[str] = set(self.active_phases())
        self._history_seen = len(self.pt.history)

    def _type_name(self, tid: int) -> str:
        n = self.name_of.get(int(tid))
        if n is None and self._pysc2_units is not None:
            try:
                n = self._pysc2_units.get_unit_type(int(tid)).name
            except Exception:
                n = None
        return n or str(int(tid))

    def update(self, raw_units, game_loop: int) -> dict:
        """Feed one observation. Returns {"entered": [...new phases this episode...], "fired": [...]}."""
        from stratagemdsl.play.game import Frame, Unit

        players = (self.spec.ally_player, self.spec.enemy_player)
        units = []
        for u in raw_units:
            if u.owner not in players or getattr(u, "is_hallucination", False):
                continue
            name = self._type_name(u.unit_type)
            st = self.spec.unit_stats.get(name, {})
            units.append(Unit(int(u.tag), name, int(u.owner), float(u.pos.x), float(u.pos.y), float(u.health),
                              float(u.shield), float(u.health) / u.health_max if u.health_max else 1.0,
                              structure=bool(st.get("structure", False))))
        self._update(Frame(int(game_loop - self.loop0), units, self.spec.ally_player))
        fired = []
        for h in self.pt.history[self._history_seen:]:
            fired.extend(h.get("fired", []))
        self._history_seen = len(self.pt.history)
        active = self.active_phases()
        entered = [p for p in active if p not in self.ever_entered]
        self.ever_entered.update(active)
        return {"entered": entered, "fired": fired}

    def _update(self, frame) -> None:
        """PlanTracker.update, but with one enter/leave event per *unit* crossing a region.

        The map's Galaxy triggers (TriggerAddEventUnitRegion) fire once per unit, so
        when 13 zerglings run into a lure zone the event is dispatched 13 times and a
        chain like review_release -> lure_via_mainway advances in the same tick.
        StratagemDSL's PlanTracker reports one event per force; this follows the map.
        """
        from stratagemdsl.net import StepLog
        from stratagemdsl.play.game import LOOPS_PER_SECOND

        pt = self.pt
        pt.tracker.update(frame)
        pt.loop = frame.game_loop
        pt.cursor = len(pt.tracker.events)
        occurrences = []
        inside_now: dict[int, frozenset] = {}
        for u in frame.units:
            force = pt.tracker.force_of.get(u.tag)
            if force not in self.scenario.forces or self.scenario.forces[force].kind == "asset":
                continue
            now = frozenset(r for r, loc in self.scenario.regions.items()
                            if (u.x - loc.x) ** 2 + (u.y - loc.y) ** 2 <= loc.radius ** 2)
            before = self._inside.get(u.tag, frozenset())
            occurrences += [("enter", (force, r)) for r in sorted(now - before)]
            occurrences += [("leave", (force, r)) for r in sorted(before - now)]
            inside_now[u.tag] = now
        self._inside = inside_now
        for name, st in pt.tracker.states.items():
            n = len(st.alive)
            before = pt.counts.get(name)
            if before is not None and n < before:
                occurrences += [("unit_died", (name,))] * (before - n)
            pt.counts[name] = n
        if pt.state.outcome:
            return
        oracle = pt._oracle_event if pt.optimistic_events else pt._oracle
        for name, args in occurrences:
            for key in pt.keys:
                evname, kargs = key[1]
                named = [v for kind, v in kargs if kind == "name"]
                if evname == name and len(named) <= len(args) and list(args[:len(named)]) == named:
                    log = StepLog()
                    pt.state = pt.game.step(pt.state, key, oracle, log)
                    if log.fired:
                        pt._record(log, f"{name}({', '.join(args)})")
        for key in pt.keys:
            evname, kargs = key[1]
            if evname == "timer" and len(kargs) >= 2:
                place, secs = kargs[0][1], float(kargs[1][1])
                at = pt.entered_at.get(place)
                if at is not None and (pt.loop - at) / LOOPS_PER_SECOND >= secs and (key, at) not in self._timers:
                    self._timers.add((key, at))
                    log = StepLog()
                    pt.state = pt.game.step(pt.state, key, oracle, log)
                    if log.fired:
                        pt._record(log, f"timer({place}, {secs:g})")
        if ("poll",) in pt.game.groups:
            log = StepLog()
            pt.state = pt.game.step(pt.state, ("poll",), pt._oracle, log)
            pt._record(log, "poll")

    # ------------------------------------------------------------ queries
    def active_phases(self) -> list[str]:
        return self.pt.active_phases()

    @property
    def outcome(self) -> Optional[int]:
        """1 success, 2 failure, None running."""
        return self.pt.state.outcome or None

    def tokens(self, place: str) -> Optional[int]:
        return self.pt.game.marking_dict(self.pt.state).get(place, 0)

    def elapsed(self, phase: str) -> Optional[float]:
        at = self.pt.entered_at.get(phase)
        return None if at is None else (self.pt.loop - at) / 22.4

    def is_terminal(self, phase: str) -> bool:
        return any(p["id"] == phase and p.get("terminal") for p in self.spec.plan.get("phases", []))

    def one_hot(self) -> list[float]:
        active = set(self.active_phases())
        return [1.0 if p in active else 0.0 for p in self.phase_ids]
