#!/usr/bin/env python
"""StratagemDSL .strat  ->  packed .SC2Map + sc2te scenario spec (``<stem>.sc2te.json``).

Run this where StratagemDSL is installed (``pip install -e StratagemDSL-RL``).
It needs neither StarCraft II nor PySC2. The JSON it writes is everything the
sc2te_env side (``smac.env.sc2_tactics.stratagem``) needs to train on the map:

    map        packed .SC2Map (absolute path), the base map, size, playable bounds,
               whether it was built with --gameover / --oracle
    races      allied / enemy race (SMAC letters P/T/Z and names)
    forces     every force/asset: side, owner, kind, composition, spawn point, initial
               (spawned at round start or later by the plan), controlled (= RL agents)
    regions    named regions of the strategy: centre and radius in world cells
    intent     success / failure / primary / secondary as expression ASTs
               (the env evaluates them to decide win / loss)
    derived    derived facts referenced by the intent
    unit_stats ranges, sight, hp, shield, air/structure for every unit type used
               (and the morphs a unit can take: SiegeTankSieged, RoachBurrowed, ...)
    plan       phases and transitions (ids, purposes, terminals) for logging / shaping
    strategy_source  the .strat text, so plan tracking works without the file
    env        suggested episode limit and step_mul

Examples
--------
    # one strategy, normal MARL map (respawns every round, no GameOver)
    python tools/strat2sc2te.py examples/stratagems/dhls_18_enter_signal.strat \
        --base-map assets/base_maps/dhls_te.SC2Map -o build/sc2te

    # several files at once (base map guessed from the file name like `strat pack`)
    python tools/strat2sc2te.py work/*.strat -o build/sc2te

    # a map you already packed yourself
    python tools/strat2sc2te.py work/my_lure.strat --map build/my_lure/my_lure.SC2Map \
        --base-map "assets/base_maps/Agna Valley-1.SC2Map" -o build/sc2te
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Optional

FORMAT = "stratagemdsl.sc2te-scenario.v1"
RACE_LETTER = {"protoss": "P", "terran": "T", "zerg": "Z", "random": "R"}

# calls the sc2te-side evaluator understands (see sc2_tactics/stratagem/conditions.py)
SUPPORTED_CALLS = {"alive", "destroyed", "defeated", "count", "count_in", "in_region", "distance", "nearest_distance",
                   "killed_in", "targeting", "mission_time", "after", "elapsed", "tokens", "active", "life_ratio",
                   "construction_complete", "under_construction", "airborne", "player_unit_count", "apparent_count",
                   "visible_to", "path_passable", "center_passable", "path_connected"}


def _require_stratagemdsl():
    try:
        import stratagemdsl  # noqa: F401
    except ImportError:
        sys.exit("StratagemDSL is not installed in this Python environment.\n"
                 "    cd StratagemDSL-RL && pip install -r requirements.txt && pip install -e .")


def _calls(ast, out: set) -> set:
    if isinstance(ast, dict):
        if ast.get("kind") == "call":
            out.add(ast.get("name"))
        for v in ast.values():
            _calls(v, out)
    elif isinstance(ast, list):
        for v in ast:
            _calls(v, out)
    return out


def _unit_stats(unit_types: set[str]) -> dict:
    from stratagemdsl.play.unitdata import BURROWED_NAME, MORPHS, STATS, stats_of
    from stratagemdsl.views.routes import AIR_UNITS, FOOTPRINT

    names = set(unit_types)
    changed = True
    while changed:                       # close over morphs: SiegeTank -> SiegeTankSieged -> SiegeTank ...
        changed = False
        for (frm, _word), to in MORPHS.items():
            if frm in names and to not in names:
                names.add(to)
                changed = True
        for frm, to in BURROWED_NAME.items():
            if frm in names and to not in names:
                names.add(to)
                changed = True
    family = {}
    for (frm, _word), to in MORPHS.items():
        family.setdefault(to, frm)
    for frm, to in BURROWED_NAME.items():
        family.setdefault(to, frm)
    out = {}
    for name in sorted(names):
        base = name
        seen = set()
        while base in family and base not in unit_types and base not in seen:   # e.g. RoachBurrowed -> Roach
            seen.add(base)
            base = family[base]
        if name in STATS or base == name:
            st = stats_of(name, structure_hint=name in FOOTPRINT)
        else:                              # a morph without its own numbers: the family's, and
            st = stats_of(base, structure_hint=base in FOOTPRINT)
            if name.endswith("Burrowed") and name not in ("LurkerBurrowed", "WidowMineBurrowed"):
                from dataclasses import replace
                st = replace(st, range_ground=0.0, range_air=0.0, speed=0.0)   # burrowed: cannot attack
        out[name] = {"hp": st.hp, "shield": st.shield, "range_ground": st.range_ground, "range_air": st.range_air,
                     "sight": st.sight, "speed": st.speed, "radius": st.radius,
                     "air": bool(st.air or name in AIR_UNITS), "structure": bool(st.structure or name in FOOTPRINT),
                     "family": base}
    return out


def _auto_limit(width: int, height: int, step_mul: int) -> int:
    # about 1.5 map spans at a slow ground speed (2.25 cells/s), in agent steps
    seconds = 1.5 * (width + height) / 2.25
    return int(min(600, max(150, math.ceil(seconds * 22.4 / step_mul))))


def build_spec(source: Path, map_path: Path, base_map: Path, *, gameover: bool, oracle: bool,
               limit: Optional[int], step_mul: int) -> dict:
    from stratagemdsl.core.model import Phase
    from stratagemdsl.frontend import parse_file
    from stratagemdsl.frontend.unparse import _print_expr
    from stratagemdsl.packaging.terrain import load_profile
    from stratagemdsl.play.scenario import build_scenario
    from stratagemdsl.registry.schema import strategy_to_schema_dict

    strategy = parse_file(source)
    profile = load_profile(base_map)
    sc = build_scenario(strategy, profile)
    schema = strategy_to_schema_dict(strategy)
    decl = {e.name: e for e in strategy.world}

    ally_owner = sc.ally_owner
    enemy_owner = next((f.owner for f in sc.forces.values() if f.side == "enemy" and f.owner), 2)
    forces = []
    unit_types: set[str] = set()
    for f in sc.forces.values():
        comp = [[t, int(n)] for t, n in f.composition] or [[t, 1] for t in f.unit_types]
        unit_types.update(t for t, _ in comp)
        initial = decl[f.name].properties.get("initial", True) if f.name in decl else True
        initial = not (str(initial).strip('"').lower() == "false" or initial is False)
        forces.append({
            "name": f.name, "kind": f.kind, "side": f.side, "owner": f.owner, "role": f.role,
            "composition": comp, "count": int(sum(n for _, n in comp)), "unit_types": sorted({t for t, _ in comp}),
            "spawn": [round(f.spawn[0], 3), round(f.spawn[1], 3)] if f.spawn else None,
            "initial": initial,
            "controlled": f.kind == "force" and f.side == "ally" and (f.owner in (None, ally_owner)),
            "air": f.air,
        })
    roles = {e.name: str(e.properties.get("role", "")).strip('"') for e in strategy.world}
    regions = [{"name": n, "role": roles.get(n, ""), "center": [round(r.x, 3), round(r.y, 3)],
                "radius": float(r.radius)} for n, r in sorted(sc.regions.items())]

    intent = schema.get("intent") or {}
    derived = {d["name"]: d["expr"] for d in schema.get("derived", [])}
    used = _calls([intent.get("success"), intent.get("failure")], set())
    for d in derived.values():
        _calls(d, used)
    warnings = []
    unsupported = sorted(c for c in used if c not in SUPPORTED_CALLS)
    if unsupported:
        warnings.append(f"intent uses {unsupported}: the env treats them as unknown (never terminal)")
    if not any(f["controlled"] for f in forces):
        warnings.append("no allied force to control: n_agents would be 0")

    side = {f["name"]: f["side"] for f in forces}
    phases = []
    for n in strategy.plan.nodes:
        if isinstance(n, Phase):
            orders = [_print_expr(a) for a in (n.actions.actions if n.actions else [])]
            phases.append({"id": n.id, "purpose": n.purpose, "terminal": n.terminal, "actors": list(n.actors),
                           "ally_orders": [o for o, a in zip(orders, n.actions.actions if n.actions else [])
                                           if a.args and side.get(getattr(a.args[0], "name", None)) == "ally"],
                           "orders": orders})
    transitions = [{"id": t.id, "inputs": [a.place for a in t.inputs if a.kind == "normal"],
                    "outputs": [a.place for a in t.outputs],
                    "event": _print_expr(t.event) if t.event is not None else None,
                    "guard": _print_expr(t.guard) if t.guard is not None else None}
                   for t in strategy.plan.transitions]

    controlled = [f for f in forces if f["controlled"]]
    ally_assets = [f for f in forces if f["side"] == "ally" and not f["controlled"]]
    enemies = [f for f in forces if f["side"] == "enemy"]
    data = map_path.read_bytes()
    return {
        "format": FORMAT,
        "name": strategy.name,
        "stem": source.stem,
        "source": str(source.resolve()),
        "strategy_source": source.read_text(encoding="utf-8"),
        "map": {"path": str(map_path.resolve()), "file": map_path.name, "base_map": str(base_map.resolve()),
                "width": profile.width, "height": profile.height, "bounds": list(profile.bounds),
                "gameover": gameover, "oracle": oracle, "sha256": hashlib.sha256(data).hexdigest()},
        "players": {"ally": ally_owner, "enemy": enemy_owner},
        "races": {"ally": RACE_LETTER.get(sc.race("ally"), "R"), "enemy": RACE_LETTER.get(sc.race("enemy"), "R"),
                  "ally_name": sc.race("ally"), "enemy_name": sc.race("enemy")},
        "forces": forces,
        "regions": regions,
        "intent": {"pattern": intent.get("pattern"), "success": intent.get("success"),
                   "failure": intent.get("failure"), "primary": intent.get("primary"),
                   "secondary": intent.get("secondary"),
                   "text": {k: _print_expr(getattr(strategy.intent, k)) for k in ("success", "failure", "primary")
                            if strategy.intent is not None and getattr(strategy.intent, k, None) is not None}},
        "derived": derived,
        "unit_stats": _unit_stats(unit_types),
        "plan": {"start": list(strategy.plan.start), "phases": phases, "transitions": transitions},
        "sizes": {"n_agents": sum(f["count"] for f in controlled),
                  "n_ally_assets": sum(f["count"] for f in ally_assets),
                  "n_enemies": sum(f["count"] for f in enemies)},
        "env": {"limit": int(limit) if limit else _auto_limit(profile.width, profile.height, step_mul),
                "step_mul": step_mul},
        "warnings": warnings,
    }


def disable_svg() -> None:
    """Skip Graphviz: the export never needs the PlanGraph/PetriNet SVGs (.dot/.mmd are still written).

    A broken Graphviz on Windows (e.g. conda's dot.exe failing to load fontconfig-1.dll:
    "libintl_dgettext not found") pops up a modal error box for every SVG and stalls the run.
    """
    import stratagemdsl.views.plangraph as plangraph
    plangraph.render_svg = lambda dot_text: None


def export_one(source: Path, out_dir: Path, *, base_map: Optional[Path], map_path: Optional[Path], gameover: bool,
               oracle: bool, loose: bool, limit: Optional[int], step_mul: int) -> Optional[Path]:
    from stratagemdsl import paths, pipeline

    stem = source.stem
    target = out_dir / stem
    if map_path is None:
        r = pipeline.run(source, until="pack", strict=not loose, out_dir=target, base_map=base_map,
                         oracle=oracle, gameover=gameover)
        for d in r.diagnostics:
            print("   ", d)
        if not r.ok:
            print(f"[FAILED] {source}: {r.failed_stage}: {r.error}")
            return None
        map_path, base_map = target / f"{stem}.SC2Map", r.base_map
    elif base_map is None:
        from stratagemdsl.views.routes import guess_base_map
        base_map = guess_base_map(source, paths.base_maps_dir())
    spec = build_spec(source, map_path, base_map, gameover=gameover, oracle=oracle, limit=limit, step_mul=step_mul)
    target.mkdir(parents=True, exist_ok=True)
    out = target / f"{stem}.sc2te.json"
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = spec["sizes"]
    print(f"[OK] {stem}: {spec['races']['ally_name']} vs {spec['races']['enemy_name']}, "
          f"n_agents={s['n_agents']} n_ally_assets={s['n_ally_assets']} n_enemies={s['n_enemies']} "
          f"regions={len(spec['regions'])} limit={spec['env']['limit']}")
    print(f"     map  {map_path}")
    print(f"     spec {out}")
    for w in spec["warnings"]:
        print(f"     warning: {w}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sources", nargs="+", help=".strat files")
    p.add_argument("-o", "--out-dir", default="build/sc2te", help="one sub-directory per strategy (default build/sc2te)")
    p.add_argument("--base-map", help="base map (default: guessed from the file name, like `strat pack`)")
    p.add_argument("--map", help="use this already packed .SC2Map instead of packing (single source only)")
    p.add_argument("--gameover", action="store_true",
                   help="pack with GameOver at the plan's terminal (one game = one episode; env restarts the game)")
    p.add_argument("--oracle", action="store_true", help="pack an oracle map (the map also commands allied forces)")
    p.add_argument("--loose", action="store_true", help="loose checking (map not guaranteed to run)")
    p.add_argument("--limit", type=int, help="episode limit in agent steps (default: from the map size)")
    p.add_argument("--step-mul", type=int, default=8)
    p.add_argument("--svg", action="store_true",
                   help="also render PlanGraph/PetriNet SVGs with Graphviz `dot` (off: avoids broken-Graphviz popups)")
    args = p.parse_args(argv)
    _require_stratagemdsl()
    if not args.svg:
        disable_svg()
    if args.map and len(args.sources) != 1:
        p.error("--map works with a single source")
    out_dir = Path(args.out_dir)
    ok = 0
    for s in args.sources:
        r = export_one(Path(s), out_dir, base_map=Path(args.base_map) if args.base_map else None,
                       map_path=Path(args.map) if args.map else None, gameover=args.gameover, oracle=args.oracle,
                       loose=args.loose, limit=args.limit, step_mul=args.step_mul)
        ok += r is not None
    print(f"{ok}/{len(args.sources)} exported to {out_dir}")
    return 0 if ok == len(args.sources) else 1


if __name__ == "__main__":
    sys.exit(main())
