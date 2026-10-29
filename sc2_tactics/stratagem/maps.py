"""Register a generated map with PySC2 and with sc2te_env's ``map_param_registry``.

sc2te_env normally needs two hand-written things per map: an entry in
``sc2_tactics/maps/sc2_tactics_maps.py`` and the .SC2Map under
``$SC2PATH/Maps/Tactics_Maps``. For StratagemDSL maps both are derived from
the spec at run time:

* a ``pysc2.maps.lib.Map`` subclass named after the map, pointing at the packed
  .SC2Map (copied to ``$SC2PATH/Maps/Stratagem_Maps`` when ``install=True`` and
  SC2PATH is set, else used from its absolute path -- PySC2 joins an absolute
  path unchanged);
* a registry entry with n_agents / n_enemies / races / limit / unit_type_bits /
  map_type, so ``get_map_params(map_name)`` works like for the built-in maps.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from .spec import ScenarioSpec

MAPS_SUBDIR = "Stratagem_Maps"


def default_map_name(spec: ScenarioSpec) -> str:
    return "strat_" + re.sub(r"\W", "_", spec.stem)


def registry_entry(spec: ScenarioSpec, *, limit: Optional[int] = None) -> dict:
    return {
        "n_agents": spec.n_agents,
        "n_enemies": spec.n_enemies,
        "n_ally_assets": spec.n_ally_assets,
        "limit": int(limit or spec.env.get("limit") or 300),
        "a_race": spec.races.get("ally", "R"),
        "b_race": spec.races.get("enemy", "R"),
        "unit_type_bits": len(spec.families),
        "map_type": "stratagem",
        "stratagem_spec": str(spec.path) if spec.path else None,
        "support_info": {"unit_families": list(spec.families), "regions": [r.name for r in spec.regions]},
    }


def install_map_file(spec: ScenarioSpec) -> tuple[str, str]:
    """(directory, filename) for the PySC2 Map class; copies into SC2PATH/Maps when possible."""
    src = spec.map_path
    sc2path = os.environ.get("SC2PATH")
    if sc2path:
        dst_dir = Path(sc2path) / "Maps" / MAPS_SUBDIR
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                shutil.copyfile(src, dst)
            return MAPS_SUBDIR, src.stem
        except OSError:
            pass
    return str(src.parent.resolve()), src.stem


def register(spec: ScenarioSpec, map_name: Optional[str] = None, *, limit: Optional[int] = None,
             install: bool = True, pysc2_map: bool = True) -> str:
    """Make ``map_name`` known to sc2te_env (and to PySC2). Returns the name."""
    name = map_name or default_map_name(spec)
    from smac.env.sc2_tactics.maps import sc2_tactics_maps
    registry = sc2_tactics_maps.get_tactics_map_registry()
    entry = registry_entry(spec, limit=limit)
    old = registry.get(name)
    if old is not None and old.get("map_type") != "stratagem":
        raise ValueError(f"map name {name!r} is already a built-in sc2te map; pass another map_name")
    registry[name] = entry
    if pysc2_map:
        _register_pysc2_map(spec, name, install)
    return name


def _register_pysc2_map(spec: ScenarioSpec, name: str, install: bool) -> None:
    from pysc2.maps import lib

    directory, filename = install_map_file(spec) if install else (str(spec.map_path.parent.resolve()),
                                                                  spec.map_path.stem)
    existing = _find_map_class(lib.Map, name)
    if existing is not None:              # re-registering (another env in the same process): just repoint it
        existing.directory, existing.filename = directory, filename
        return
    cls = type(name, (lib.Map,), dict(directory=directory, filename=filename, players=2, step_mul=8,
                                      game_steps_per_episode=0))
    # keep a module-level reference so the subclass is not garbage collected
    globals().setdefault("_MAP_CLASSES", {})[name] = cls


def _find_map_class(base, name):
    stack = list(base.__subclasses__())
    while stack:
        c = stack.pop()
        if c.__name__ == name:
            return c
        stack.extend(c.__subclasses__())
    return None
