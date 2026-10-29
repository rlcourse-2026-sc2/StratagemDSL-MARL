"""The scenario spec written by ``tools/strat2sc2te.py`` (format ``stratagemdsl.sc2te-scenario.v1``).

Only the standard library is needed here: the spec is plain JSON, unit type
names are resolved to SC2 unit ids with sc2te_env's own ``constants/<ver>/Units.json``
(falling back to ``pysc2.lib.units``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

FORMAT = "stratagemdsl.sc2te-scenario.v1"
_CONSTANTS = Path(__file__).resolve().parent.parent / "constants"

# DSL / editor names that differ from the unit names the game reports
UNIT_NAME_ALIASES = {
    "HellionTank": "Hellbat",          # editor id vs. PySC2 name of unit 484
    "Hellbat": "HellionTank",
    "Lurker": "LurkerMP",
    "LurkerBurrowed": "LurkerMPBurrowed",
    "SwarmHost": "SwarmHostMP",
    "SwarmHostBurrowed": "SwarmHostBurrowedMP",
    "Viking": "VikingFighter",
}

_TABLE_CACHE: dict[str, dict[str, int]] = {}


def unit_type_table(version: Optional[str] = None) -> dict[str, int]:
    """Unit name -> unit type id, from sc2te_env's constants (4.10.0 by default)."""
    key = version or "4.10.0"
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    table: dict[str, int] = {}
    candidates = [_CONSTANTS / key / "Units.json", _CONSTANTS / "4.10.0" / "Units.json",
                  _CONSTANTS / "5.0.13" / "Units.json"]
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for u in data.get("Units", []):
                table.setdefault(u["name"], int(u["id"]))
            break
    try:  # complete it with PySC2's enums (same ids for every unit used in these maps)
        from pysc2.lib import units as pysc2_units
        for race in (pysc2_units.Protoss, pysc2_units.Terran, pysc2_units.Zerg, pysc2_units.Neutral):
            for member in race:
                table.setdefault(member.name, int(member))
    except Exception:
        pass
    _TABLE_CACHE[key] = table
    return table


def resolve_unit_type(name: str, version: Optional[str] = None) -> Optional[int]:
    table = unit_type_table(version)
    if name in table:
        return table[name]
    alias = UNIT_NAME_ALIASES.get(name)
    return table.get(alias) if alias else None


@dataclass
class ForceSpec:
    name: str
    kind: str                       # force | asset
    side: str                       # ally | enemy
    owner: Optional[int]
    role: str
    composition: list[tuple[str, int]]
    count: int
    unit_types: list[str]
    spawn: Optional[tuple[float, float]]
    initial: bool
    controlled: bool
    air: bool


@dataclass
class RegionSpec:
    name: str
    role: str
    center: tuple[float, float]
    radius: float

    def contains(self, x: float, y: float) -> bool:
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2


@dataclass
class ScenarioSpec:
    raw: dict
    path: Optional[Path]
    name: str
    stem: str
    map_path: Path
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    gameover: bool
    players: dict
    races: dict
    forces: list[ForceSpec]
    regions: list[RegionSpec]
    intent: dict
    derived: dict
    unit_stats: dict
    plan: dict
    env: dict
    strategy_source: str = ""
    game_version: Optional[str] = None
    # filled by _index()
    force_by_name: dict = field(default_factory=dict)
    region_by_name: dict = field(default_factory=dict)
    type_ids: dict = field(default_factory=dict)        # unit name -> id
    family_of_id: dict = field(default_factory=dict)    # unit id -> family name
    families: list = field(default_factory=list)        # one-hot vocabulary (family names)
    unknown_types: list = field(default_factory=list)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, source: Union[str, os.PathLike, dict], game_version: Optional[str] = None) -> "ScenarioSpec":
        path = None
        if isinstance(source, dict):
            raw = source
        else:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"scenario not found: {path}\n  (relative paths are taken from the current directory "
                    f"{Path.cwd()}; pass the folder written by tools/strat2sc2te.py -o ..., e.g. "
                    f"StratagemDSL-rl/build/sc2te/<name>, or its *.sc2te.json)")
            if path.is_dir():                       # a folder written by strat2sc2te: take its *.sc2te.json
                found = sorted(path.glob("*.sc2te.json"))
                if not found:
                    raise FileNotFoundError(f"no *.sc2te.json in {path}")
                path = found[0]
            raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} file (format={raw.get('format')!r}); "
                             "write it with tools/strat2sc2te.py")
        m = raw["map"]
        map_path = Path(m["path"])
        if not map_path.is_file() and path is not None:   # moved to another machine: look next to the spec
            for cand in (path.parent / m.get("file", map_path.name), path.parent / map_path.name):
                if cand.is_file():
                    map_path = cand
                    break
        forces = [ForceSpec(f["name"], f["kind"], f.get("side", ""), f.get("owner"), f.get("role", ""),
                            [(t, int(n)) for t, n in f.get("composition", [])], int(f.get("count", 0)),
                            list(f.get("unit_types", [])), tuple(f["spawn"]) if f.get("spawn") else None,
                            bool(f.get("initial", True)), bool(f.get("controlled", False)), bool(f.get("air", False)))
                  for f in raw["forces"]]
        regions = [RegionSpec(r["name"], r.get("role", ""), tuple(r["center"]), float(r["radius"]))
                   for r in raw.get("regions", [])]
        spec = cls(raw=raw, path=path, name=raw.get("name") or raw.get("stem"), stem=raw.get("stem") or raw["name"],
                   map_path=map_path, width=int(m["width"]), height=int(m["height"]),
                   bounds=tuple(m.get("bounds") or (0, 0, m["width"], m["height"])),
                   gameover=bool(m.get("gameover", False)), players=dict(raw.get("players") or {"ally": 1, "enemy": 2}),
                   races=dict(raw.get("races") or {}), forces=forces, regions=regions,
                   intent=dict(raw.get("intent") or {}), derived=dict(raw.get("derived") or {}),
                   unit_stats=dict(raw.get("unit_stats") or {}), plan=dict(raw.get("plan") or {}),
                   env=dict(raw.get("env") or {}), strategy_source=raw.get("strategy_source", ""),
                   game_version=game_version)
        spec._index()
        return spec

    def _index(self) -> None:
        self.force_by_name = {f.name: f for f in self.forces}
        self.region_by_name = {r.name: r for r in self.regions}
        names = set(self.unit_stats)
        for f in self.forces:
            names.update(f.unit_types)
            names.update(t for t, _ in f.composition)
        self.type_ids, self.unknown_types = {}, []
        for n in sorted(names):
            tid = resolve_unit_type(n, self.game_version)
            if tid is None:
                self.unknown_types.append(n)
            else:
                self.type_ids[n] = tid
        # one-hot vocabulary: the families of the composed types, allies first, in force order
        fams: list[str] = []
        for side in ("ally", "enemy"):
            for f in self.forces:
                if f.side != side:
                    continue
                for t, _ in f.composition:
                    fam = self.family(t)
                    if fam not in fams:
                        fams.append(fam)
        self.families = fams
        self.family_of_id = {}
        for n, tid in self.type_ids.items():
            self.family_of_id.setdefault(tid, self.family(n))

    # ------------------------------------------------------------ queries
    def family(self, unit_name: str) -> str:
        st = self.unit_stats.get(unit_name)
        return st.get("family", unit_name) if st else unit_name

    def type_index(self, unit_type_id: int) -> int:
        """Index in the one-hot vocabulary, -1 for a type the strategy does not use."""
        fam = self.family_of_id.get(int(unit_type_id))
        try:
            return self.families.index(fam) if fam is not None else -1
        except ValueError:
            return -1

    def stats(self, unit_type_id: int) -> dict:
        for n, tid in self.type_ids.items():
            if tid == int(unit_type_id) and n in self.unit_stats:
                return self.unit_stats[n]
        fam = self.family_of_id.get(int(unit_type_id))
        return self.unit_stats.get(fam, {}) if fam else {}

    def family_ids(self, unit_name: str) -> set[int]:
        """Every unit id belonging to the family of ``unit_name`` (SiegeTank -> {33, 32})."""
        fam = self.family(unit_name)
        return {tid for tid, f in self.family_of_id.items() if f == fam} | (
            {self.type_ids[unit_name]} if unit_name in self.type_ids else set())

    @property
    def controlled_forces(self) -> list[ForceSpec]:
        return [f for f in self.forces if f.controlled]

    @property
    def ally_asset_forces(self) -> list[ForceSpec]:
        return [f for f in self.forces if f.side == "ally" and not f.controlled]

    @property
    def enemy_forces(self) -> list[ForceSpec]:
        return [f for f in self.forces if f.side == "enemy"]

    @property
    def n_agents(self) -> int:
        return sum(f.count for f in self.controlled_forces)

    @property
    def n_ally_assets(self) -> int:
        return sum(f.count for f in self.ally_asset_forces)

    @property
    def n_enemies(self) -> int:
        return sum(f.count for f in self.enemy_forces)

    @property
    def ally_player(self) -> int:
        return int(self.players.get("ally", 1))

    @property
    def enemy_player(self) -> int:
        return int(self.players.get("enemy", 2))

    @property
    def phase_ids(self) -> list[str]:
        return [p["id"] for p in self.plan.get("phases", [])]

    def describe(self) -> str:
        lines = [f"{self.name}: {self.races.get('ally_name', self.races.get('ally'))} vs "
                 f"{self.races.get('enemy_name', self.races.get('enemy'))} on {self.map_path.name} "
                 f"({self.width}x{self.height})",
                 f"  agents {self.n_agents}, ally assets {self.n_ally_assets}, enemies {self.n_enemies}, "
                 f"regions {len(self.regions)}, episode limit {self.env.get('limit')}"]
        for f in self.forces:
            comp = ", ".join(f"{n} {t}" for t, n in f.composition)
            tag = "agents" if f.controlled else f.kind
            lines.append(f"  - {f.side:5} {f.name} [{tag}] {comp}" + ("" if f.initial else " (spawned later)"))
        txt = self.intent.get("text") or {}
        if txt:
            lines.append(f"  success: {txt.get('success')}")
            lines.append(f"  failure: {txt.get('failure')}")
        if self.unknown_types:
            lines.append(f"  unresolved unit types: {self.unknown_types}")
        return "\n".join(lines)


def load_spec(source, game_version: Optional[str] = None) -> ScenarioSpec:
    return ScenarioSpec.load(source, game_version)
