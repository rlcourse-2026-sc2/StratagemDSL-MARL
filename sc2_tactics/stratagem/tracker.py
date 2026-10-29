"""Which SC2 unit fills which slot, and the world facts the intent is evaluated on.

Slots are fixed per scenario (that is what MARL needs: a fixed number of agents
and enemies). They come from the spec's forces, in force order:

    agents       units of the allied *controlled* forces (kind=force, side=ally)
    ally_assets  allied assets (structures etc.) -- in the state, not agents
    enemies      every unit of every enemy force/asset, including forces the plan
                 spawns later (``initial = false``): their slots stay empty until then

A unit is matched once, when it first appears: same owner, a unit type of the
force's family (SiegeTank also matches SiegeTankSieged), the nearest spawn
point (or live centre of the force), and only into a slot nobody filled yet.
After that it keeps its slot by tag, through morphs and burrowing. Units no
force can take (hallucinations, broodlings, locusts ...) are "extras": they
appear in ``info`` but have no slot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .spec import ScenarioSpec

LOOPS_PER_SECOND = 22.4
ATTACK_ABILITIES = {23, 3674}
MOVE_ABILITIES = {16, 3794}


@dataclass
class Slot:
    index: int                 # index inside its group
    group: str                 # agent | ally_asset | enemy
    force: str
    family: str                # family of the composition entry this slot was made for
    unit: Any = None           # latest raw unit while alive (s2clientprotocol raw_pb2.Unit)
    tag: Optional[int] = None
    filled: bool = False
    alive: bool = False
    died_at: Optional[int] = None

    @property
    def present(self) -> bool:
        return self.alive and self.unit is not None


class UnitTracker:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.groups: dict[str, list[Slot]] = {"agent": [], "ally_asset": [], "enemy": []}
        for group, forces in (("agent", spec.controlled_forces), ("ally_asset", spec.ally_asset_forces),
                              ("enemy", spec.enemy_forces)):
            for f in forces:
                for unit_name, n in f.composition:
                    for _ in range(n):
                        self.groups[group].append(Slot(len(self.groups[group]), group, f.name, spec.family(unit_name)))
        self.by_force: dict[str, list[Slot]] = {}
        for slots in self.groups.values():
            for s in slots:
                self.by_force.setdefault(s.force, []).append(s)
        self._family_ids = {f.name: set().union(*[spec.family_ids(t) for t, _ in f.composition] or [set()])
                            for f in spec.forces}
        self.players = (spec.ally_player, spec.enemy_player)
        self.plan = None           # optional PlanProgress (elapsed / tokens)
        self.reset(0)

    # ------------------------------------------------------------ episode
    def reset(self, loop0: int) -> None:
        for slots in self.groups.values():
            for s in slots:
                s.unit, s.tag, s.filled, s.alive, s.died_at = None, None, False, False, None
        self.tag_slot: dict[int, Slot] = {}
        self.extras: dict[int, Any] = {}
        self.kills_in: dict[tuple[str, str], int] = {}
        self.events: list[dict] = []
        self.regions_of_force: dict[str, set[str]] = {}
        self.owner_counts: dict[int, int] = {}
        self.loop0 = loop0
        self.loop = loop0

    # ------------------------------------------------------------- update
    def observe(self, raw_units, game_loop: int) -> list[dict]:
        """Update from one observation; returns the events of this step."""
        self.loop = game_loop
        start = len(self.events)
        present = {}
        counts: dict[int, int] = {}
        for u in raw_units:
            if u.owner not in self.players:
                continue
            counts[u.owner] = counts.get(u.owner, 0) + 1
            if getattr(u, "is_hallucination", False):
                self.extras[u.tag] = u
                continue
            present[u.tag] = u
        self.owner_counts = counts
        for slots in self.groups.values():
            for s in slots:
                if not s.alive:
                    continue
                u = present.get(s.tag)
                if u is not None:
                    s.unit = u
                else:
                    self._died(s)
        for tag in [t for t in self.extras if t not in present]:
            self.extras.pop(tag, None)
        new = [u for t, u in present.items() if t not in self.tag_slot and t not in self.extras]
        if new:
            self._assign(new)
        self._region_events()
        return self.events[start:]

    def _died(self, s: Slot) -> None:
        last = s.unit
        s.alive, s.died_at = False, self.loop
        if last is not None:
            for r in self.spec.regions:
                if r.contains(last.pos.x, last.pos.y):
                    self.kills_in[(s.force, r.name)] = self.kills_in.get((s.force, r.name), 0) + 1
        self.events.append({"loop": self.loop - self.loop0, "event": "died", "force": s.force, "group": s.group,
                            "slot": s.index})

    def _ref(self, force: str) -> Optional[tuple[float, float]]:
        return self.centroid(force) or self.spec.force_by_name[force].spawn

    def _assign(self, units: list) -> None:
        pairs = []
        for u in units:
            for f in self.spec.forces:
                if f.owner is not None and f.owner != u.owner:
                    continue
                if f.owner is None and u.owner != (self.spec.ally_player if f.side == "ally" else self.spec.enemy_player):
                    continue
                if u.unit_type not in self._family_ids.get(f.name, ()):
                    continue
                ref = self._ref(f.name)
                d = math.hypot(u.pos.x - ref[0], u.pos.y - ref[1]) if ref else 1e9
                pairs.append((d, u.tag, f.name))
        pairs.sort()
        free = {name: [s for s in slots if not s.filled] for name, slots in self.by_force.items()}
        chosen: dict[str, list] = {}
        taken: set[int] = set()
        by_tag = {u.tag: u for u in units}
        for _d, tag, name in pairs:
            if tag in taken or len(chosen.get(name, [])) >= len(free.get(name, [])):
                continue
            chosen.setdefault(name, []).append(by_tag[tag])
            taken.add(tag)
        for name, us in chosen.items():
            us.sort(key=lambda u: (self.spec.type_index(u.unit_type), u.unit_type, round(u.pos.x, 1), round(u.pos.y, 1)))
            open_slots = free[name]
            for u in us:
                fam = self.spec.family_of_id.get(u.unit_type)
                slot = next((s for s in open_slots if not s.filled and s.family == fam), None) \
                    or next((s for s in open_slots if not s.filled), None)
                if slot is None:
                    self.extras[u.tag] = u
                    continue
                slot.unit, slot.tag, slot.filled, slot.alive = u, u.tag, True, True
                self.tag_slot[u.tag] = slot
                self.events.append({"loop": self.loop - self.loop0, "event": "spawned", "force": name,
                                    "group": slot.group, "slot": slot.index})
        for u in units:
            if u.tag not in taken:
                self.extras[u.tag] = u

    def _region_events(self) -> None:
        for name in self.by_force:
            f = self.spec.force_by_name[name]
            if f.kind == "asset":
                continue
            members = self.members(name)
            inside = {r.name for r in self.spec.regions if any(r.contains(u.pos.x, u.pos.y) for u in members)}
            before = self.regions_of_force.get(name, set())
            for r in sorted(inside - before):
                self.events.append({"loop": self.loop - self.loop0, "event": "enter", "force": name, "region": r})
            for r in sorted(before - inside):
                self.events.append({"loop": self.loop - self.loop0, "event": "leave", "force": name, "region": r})
            self.regions_of_force[name] = inside

    # ------------------------------------------------------------ queries
    def slots(self, group: str) -> list[Slot]:
        return self.groups[group]

    def members(self, name: str) -> list:
        return [s.unit for s in self.by_force.get(name, []) if s.present]

    def matches_force(self, u) -> bool:
        """Could this raw unit fill a slot of some force (owner and unit family)?"""
        if getattr(u, "is_hallucination", False):
            return False
        for f in self.spec.forces:
            owner = f.owner if f.owner is not None else (
                self.spec.ally_player if f.side == "ally" else self.spec.enemy_player)
            if owner == u.owner and u.unit_type in self._family_ids.get(f.name, ()):
                return True
        return False

    def expected_initial(self) -> dict[int, int]:
        """owner -> number of units the round spawns at once (forces with initial = true)."""
        out: dict[int, int] = {}
        for f in self.spec.forces:
            if f.initial and f.owner is not None:
                out[f.owner] = out.get(f.owner, 0) + f.count
        return out

    def summary(self) -> dict:
        out = {}
        for name, slots in self.by_force.items():
            alive = [s for s in slots if s.present]
            c = self.centroid(name)
            out[name] = {"alive": len(alive), "slots": len(slots), "spawned": sum(s.filled for s in slots),
                         "at": [round(c[0], 1), round(c[1], 1)] if c else None,
                         "regions": sorted(self.regions_of_force.get(name, set()))}
        return out

    # ----------------------------------------------- World (conditions.py)
    def count(self, name: str) -> Optional[int]:
        if name not in self.by_force:
            return None
        return len(self.members(name))

    def centroid(self, name: str) -> Optional[tuple[float, float]]:
        m = self.members(name)
        if not m:
            return None
        return (sum(u.pos.x for u in m) / len(m), sum(u.pos.y for u in m) / len(m))

    def count_in(self, name: str, region: str) -> Optional[int]:
        r = self.spec.region_by_name.get(region)
        if name not in self.by_force or r is None:
            return None
        return sum(1 for u in self.members(name) if r.contains(u.pos.x, u.pos.y))

    def nearest_distance(self, a: str, b: str) -> Optional[float]:
        ma, mb = self.members(a), self.members(b)
        if not ma or not mb:
            return None
        return min(math.hypot(u.pos.x - v.pos.x, u.pos.y - v.pos.y) for u in ma for v in mb)

    def killed_in(self, force: str, region: str) -> Optional[int]:
        if force not in self.by_force or region not in self.spec.region_by_name:
            return None
        return self.kills_in.get((force, region), 0)

    def targeting(self, a: str, b: str) -> Optional[bool]:
        ma = self.members(a)
        if a not in self.by_force:
            return None
        if not ma:
            return False
        tags = {u.tag for u in self.members(b)} if b in self.by_force else set()
        if b in self.spec.region_by_name:
            loc = self.spec.region_by_name[b].center
        else:
            loc = self.centroid(b)
        hits = 0
        for u in ma:
            if not u.orders:
                continue
            o = u.orders[0]
            if o.ability_id not in ATTACK_ABILITIES | MOVE_ABILITIES:
                continue
            if o.HasField("target_unit_tag") and o.target_unit_tag in tags:
                hits += 1
            elif o.HasField("target_world_space_pos") and loc is not None and \
                    math.hypot(o.target_world_space_pos.x - loc[0], o.target_world_space_pos.y - loc[1]) < 4:
                hits += 1
        return hits * 2 > len(ma)

    def life_ratio(self, name: str) -> Optional[float]:
        if name not in self.by_force:
            return None
        m = self.members(name)
        tot = sum(u.health_max for u in m)
        return (sum(u.health for u in m) / tot) if tot else 0.0

    def construction_complete(self, name: str) -> Optional[bool]:
        if name not in self.by_force:
            return None
        m = self.members(name)
        return bool(m) and all(u.build_progress >= 1.0 for u in m)

    def airborne(self, name: str) -> Optional[bool]:
        if name not in self.by_force:
            return None
        return any(u.is_flying for u in self.members(name))

    def player_unit_count(self, name: str) -> Optional[int]:
        f = self.spec.force_by_name.get(name)
        if f is None or f.owner is None:
            return None
        return self.owner_counts.get(f.owner, 0)

    def seconds(self) -> float:
        return (self.loop - self.loop0) / LOOPS_PER_SECOND

    def elapsed(self, phase: str) -> Optional[float]:
        return self.plan.elapsed(phase) if self.plan is not None else None

    def tokens(self, place: str) -> Optional[int]:
        return self.plan.tokens(place) if self.plan is not None else None
