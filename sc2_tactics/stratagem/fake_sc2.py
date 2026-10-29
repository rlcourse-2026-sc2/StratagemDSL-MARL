"""A StarCraft II stand-in at the protocol level, for running the env without the game.

``StratagemTacticsEnv(..., engine="fake")`` talks to this object exactly like to
PySC2's ``RemoteController`` (create/join, game_info, observe, actions, step,
debug kill, restart, status) and gets real ``s2clientprotocol`` messages back.
Behind it runs StratagemDSL's ``play.plansim.PlanSim``: the plan's Petri net
scripts the enemy like the generated map does, ground units walk A* paths over
the map's cliff levels, combat is flat DPS.

Use it to debug the pipeline (spec, slots, actions, termination, reset,
PyMARL wiring) on a laptop. It is NOT a substitute for SC2: numbers, pathing,
collisions and abilities are approximations, and ``spawn(...)`` reinforcements
are present from the start.

Needs StratagemDSL importable (``pip install -e StratagemDSL-RL``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from pysc2.lib import protocol
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from .spec import ScenarioSpec, resolve_unit_type

TAG_STRIDE = 1 << 36
STOP_IDS = {4, 3665, 18, 3793}
MOVE_IDS = {16, 3794}
ATTACK_IDS = {23, 3674}


def _ability_name(ability_id: int) -> str:
    try:
        from pysc2.lib import actions
        for f in actions.RAW_FUNCTIONS:
            if f.ability_id == ability_id:
                return f.name
    except Exception:
        pass
    return str(ability_id)


class _FakeRunConfig:
    def save_replay(self, replay_data, replay_dir, prefix=None):
        path = Path(replay_dir or ".") / f"{prefix or 'fake'}.fake_replay.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("engine=fake: no replay\n")
        return str(path)


class FakeController:
    def __init__(self, spec: ScenarioSpec, seed: Optional[int] = None, max_loops: int = 10 ** 9):
        from stratagemdsl.frontend import parse_source
        from stratagemdsl.packaging.terrain import load_profile
        from stratagemdsl.play.plansim import PlanSim

        self.spec = spec
        self.strategy = parse_source(spec.strategy_source)
        try:
            self.profile = load_profile(spec.map_path)          # the packed map keeps the base map's terrain
        except Exception:
            self.profile = load_profile(Path(spec.raw["map"]["base_map"]))
        gameover = spec.gameover

        class _Sim(PlanSim):
            @property
            def done(self):                 # a MARL map keeps running after the plan's terminal
                if gameover:
                    return self.outcome is not None or self.loop >= self.max_loops
                return self.loop >= self.max_loops

        self.sim = _Sim(self.strategy, self.profile, max_loops=max_loops, fog=False, seed=seed,
                        oracle=bool(spec.raw["map"].get("oracle")))
        self.serial = 0
        self.status = protocol.Status.launched
        self._ended_result: Optional[int] = None
        self._type_id: dict[str, int] = {}

    # ------------------------------------------------------------ lifecycle
    def create_game(self, req):
        self.status = protocol.Status.init_game

    def join_game(self, req):
        self._new_round()
        self.status = protocol.Status.in_game

    def restart(self):
        self._new_round()
        self.status = protocol.Status.in_game

    def _new_round(self):
        self.serial += 1
        self.sim.reset()
        self._ended_result = None
        self._loop_offset = getattr(self, "_loop_offset", 0) + (self.sim.loop if self.serial > 1 else 0)

    def quit(self):
        self.status = protocol.Status.quit

    def close(self):
        self.quit()

    def save_replay(self):
        return b""

    # --------------------------------------------------------------- tags
    def _ext(self, sim_tag: int) -> int:
        return self.serial * TAG_STRIDE + sim_tag

    def _sim_unit(self, ext_tag: int):
        if ext_tag // TAG_STRIDE != self.serial:
            return None
        return self.sim.by_tag.get(ext_tag % TAG_STRIDE)

    def _tid(self, name: str) -> int:
        if name not in self._type_id:
            self._type_id[name] = resolve_unit_type(name, self.spec.game_version) or 0
        return self._type_id[name]

    # ---------------------------------------------------------- game info
    def game_info(self):
        p = self.profile
        w, h = p.width, p.height
        info = sc_pb.ResponseGameInfo(map_name=self.spec.map_path.name)
        sr = info.start_raw
        sr.map_size.x, sr.map_size.y = w, h
        left, bottom, right, top = p.bounds
        sr.playable_area.p0.x, sr.playable_area.p0.y = int(left), int(bottom)
        sr.playable_area.p1.x, sr.playable_area.p1.y = int(right), int(top)
        bits = bytearray()
        heights = bytearray()
        for y in range(h):
            for x0 in range(0, w, 8):
                b = 0
                for i in range(8):
                    x = x0 + i
                    if x < w and p.walkable(x + 0.5, y + 0.5):
                        b |= 1 << (7 - i)
                bits.append(b)
            for x in range(w):
                lv = p.level[y][x]
                heights.append(0 if lv == 0 else max(0, min(255, 64 + 32 * (lv if lv > 0 else 1))))
        sr.pathing_grid.bits_per_pixel = 1
        sr.pathing_grid.size.x, sr.pathing_grid.size.y = w, h
        sr.pathing_grid.data = bytes(bits)
        sr.terrain_height.bits_per_pixel = 8
        sr.terrain_height.size.x, sr.terrain_height.size.y = w, h
        sr.terrain_height.data = bytes(heights)
        return info

    # ----------------------------------------------------------- stepping
    def step(self, count: int = 1, game_loop=None):
        if self.status != protocol.Status.in_game:
            raise protocol.ProtocolError(f"step while {self.status}")
        self.sim.advance(int(count))
        self._check_end()

    def _check_end(self):
        if self.spec.gameover and self.sim.outcome is not None and self._ended_result is None:
            self._ended_result = sc_pb.Victory if self.sim.outcome == "victory" else sc_pb.Defeat
            self.status = protocol.Status.ended

    def actions(self, req):
        from stratagemdsl.play.game import Order

        for act in req.actions:
            if not act.HasField("action_raw") or not act.action_raw.HasField("unit_command"):
                continue
            c = act.action_raw.unit_command
            tags = [self._sim_unit(t).tag for t in c.unit_tags if self._sim_unit(t) is not None]
            if not tags:
                continue
            target = self._sim_unit(c.target_unit_tag) if c.HasField("target_unit_tag") else None
            point = (c.target_world_space_pos.x, c.target_world_space_pos.y) \
                if c.HasField("target_world_space_pos") else None
            meta = {"queued": bool(c.queue_command)}
            aid = c.ability_id
            if aid in STOP_IDS:
                order = Order("hold" if aid in (18, 3793) else "stop", tags, meta=meta)
            elif aid in MOVE_IDS:
                p = point or ((target.x, target.y) if target is not None else None)
                if p is None:
                    continue
                order = Order("move", tags, point=p, meta=meta)
            elif aid in ATTACK_IDS:
                if target is not None:
                    order = Order("attack_unit", tags, target_tag=target.tag, meta=meta)
                elif point is not None:
                    order = Order("attack_pt", tags, point=point, meta=meta)
                else:
                    continue
            else:
                meta["ability"] = _ability_name(aid)
                order = Order("ability", tags, point=point, target_tag=target.tag if target else None, meta=meta)
            self.sim.issue([order], owner=self.spec.ally_player)
        return sc_pb.ResponseAction()

    def debug(self, debug_commands):
        cmds = debug_commands if isinstance(debug_commands, (list, tuple)) else [debug_commands]
        killed = False
        for d in cmds:
            if d.HasField("kill_unit"):
                for t in d.kill_unit.tag:
                    u = self._sim_unit(t)
                    if u is not None:
                        u.hp = 0
                        killed = True
        if killed:
            self.sim.advance(0)
            players = (self.spec.ally_player, self.spec.enemy_player)
            if not any(u.alive and u.owner in players for u in self.sim.units):
                self._new_round()          # the map's gt_Reset: respawn when both players have no unit

    # -------------------------------------------------------- observation
    def observe(self, disable_fog=False, target_game_loop=0):
        obs = sc_pb.ResponseObservation()
        o = obs.observation
        o.game_loop = self._loop_offset + self.sim.loop
        raw = o.raw_data
        for u in self.sim.units:
            if not u.alive:
                continue
            pu = raw.units.add()
            pu.display_type = r_pb.Visible
            pu.alliance = r_pb.Self if u.owner == self.spec.ally_player else r_pb.Enemy
            pu.tag = self._ext(u.tag)
            pu.unit_type = self._tid(u.unit_type)
            pu.owner = u.owner
            pu.pos.x, pu.pos.y, pu.pos.z = float(u.x), float(u.y), 10.0
            pu.facing = float(u.facing)
            pu.radius = float(u.radius)
            pu.build_progress = 1.0
            pu.is_flying = bool(u.air)
            pu.is_burrowed = bool(u.burrowed)
            pu.health, pu.health_max = max(0.0, float(u.hp)), float(u.stats.hp)
            pu.shield, pu.shield_max = max(0.0, float(u.shield)), float(u.stats.shield)
            pu.energy, pu.energy_max = float(u.energy), float(u.stats.energy)
            pu.weapon_cooldown = float(u.weapon_cooldown * 22.4)
            od = u.order
            if od is not None and od.kind in ("move", "attack_pt", "attack_unit", "attack_group"):
                uo = pu.orders.add()
                uo.ability_id = 16 if od.kind == "move" else 23
                if od.kind == "attack_unit" and od.target_tag is not None:
                    uo.target_unit_tag = self._ext(od.target_tag)
                elif od.point is not None:
                    uo.target_world_space_pos.x, uo.target_world_space_pos.y = od.point
        if self._ended_result is not None:
            pr = obs.player_result.add()
            pr.player_id, pr.result = self.spec.ally_player, self._ended_result
            pr = obs.player_result.add()
            pr.player_id = self.spec.enemy_player
            pr.result = sc_pb.Defeat if self._ended_result == sc_pb.Victory else sc_pb.Victory
        return obs


class FakeSC2:
    """What ``run_config.start()`` returns: an object with ``.controller`` and ``.close()``."""

    def __init__(self, spec: ScenarioSpec, seed: Optional[int] = None):
        self.controller = FakeController(spec, seed=seed)
        self.run_config = _FakeRunConfig()
        self.controller.create_game(None)
        self.controller.join_game(None)

    def close(self):
        self.controller.close()
