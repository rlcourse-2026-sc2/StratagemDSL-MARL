"""Offline checks of the bridge: no StarCraft II needed (engine="fake").

    STRAT_SPECS=build/sc2te python -m pytest -q tests/test_offline.py

STRAT_SPECS is a folder with one sub-folder per strategy, as written by
tools/strat2sc2te.py. Needs SMAC with sc2te_env overlaid (and this package in
smac/env/sc2_tactics/stratagem) plus StratagemDSL for the fake engine.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv, conditions, load_spec

SPECS = Path(os.environ.get("STRAT_SPECS", "build/sc2te"))
ALL = sorted(p for p in SPECS.glob("*/*.sc2te.json")) if SPECS.is_dir() else []
needs_specs = pytest.mark.skipif(not ALL, reason=f"no *.sc2te.json under {SPECS} (set STRAT_SPECS)")


def _run(env, policy, max_steps=None):
    obs, state = env.reset()
    assert len(obs) == env.n_agents
    assert all(o.shape == (env.get_obs_size(),) for o in obs)
    assert state.shape == (env.get_state_size(),)
    done, steps, info = False, 0, {}
    while not done:
        avail = env.get_avail_actions()
        assert all(len(a) == env.n_actions and sum(a) >= 1 for a in avail)
        r, done, info = env.step(policy(env, avail))
        steps += 1
        if max_steps and steps >= max_steps:
            break
    obs = env.get_obs()
    assert all(o.shape == (env.get_obs_size(),) for o in obs)
    assert env.get_state().shape == (env.get_state_size(),)
    return steps, info


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda env, avail: [int(rng.choice(np.nonzero(a)[0])) for a in avail]


@needs_specs
@pytest.mark.parametrize("spec_path", ALL, ids=[p.parent.name for p in ALL])
def test_random_episodes(spec_path):
    env = StratagemTacticsEnv(spec_path, engine="fake", seed=1, episode_limit=60, info_detail=True)
    try:
        for _ in range(2):                       # second episode exercises the respawn reset
            steps, info = _run(env, random_policy())
            assert steps <= 60
            assert info.get("outcome") in ("success", "failure", "draw", "timeout")
        assert env.get_stats()["battles_game"] == 2
    finally:
        env.close()


@needs_specs
def test_spec_round_trip():
    for p in ALL:
        spec = load_spec(p)
        assert spec.n_agents > 0
        assert not spec.unknown_types, (p, spec.unknown_types)
        bad = conditions.unsupported_calls([spec.intent["success"], spec.intent["failure"]], spec.derived)
        assert not bad, (p, bad)


def test_conditions_three_valued():
    class W:
        def count(self, n): return {"a": 0, "b": 3}.get(n)
        def centroid(self, n): return {"b": (0.0, 0.0)}.get(n)
        def count_in(self, n, r): return 1
        def nearest_distance(self, a, b): return None
        def killed_in(self, f, r): return 2
        def targeting(self, a, b): return None
        def life_ratio(self, n): return 0.5
        def construction_complete(self, n): return True
        def airborne(self, n): return False
        def player_unit_count(self, n): return 4
        def seconds(self): return 12.0
        def elapsed(self, p): return None
        def tokens(self, p): return None

    name = lambda n: {"kind": "name", "name": n}
    call = lambda f, *a: {"kind": "call", "name": f, "args": list(a)}
    w = W()
    assert conditions.truth(call("destroyed", name("a")), w) is True
    assert conditions.truth(call("alive", name("b")), w) is True
    assert conditions.truth(call("alive", name("zzz")), w) is None
    unknown_or_true = {"kind": "binary", "op": "or", "left": call("alive", name("zzz")), "right": call("alive", name("b"))}
    assert conditions.truth(unknown_or_true, w) is True
    unknown_and = {"kind": "binary", "op": "and", "left": call("alive", name("zzz")), "right": call("alive", name("b"))}
    assert conditions.truth(unknown_and, w) is None
    after = call("after", {"kind": "number", "value": 10})
    assert conditions.truth(after, w) is True
    ge = {"kind": "binary", "op": ">=", "left": call("killed_in", name("a"), name("r")), "right": {"kind": "number", "value": 2}}
    assert conditions.truth(ge, w) is True


@needs_specs
def test_reset_kills_leftovers_like_broodlings():
    """Dying Zerg structures release broodlings: reset must kill them before the round can respawn."""
    spec_path = next((p for p in ALL if "dhls" in p.name), None)
    if spec_path is None:
        pytest.skip("needs the dhls example")
    from stratagemdsl.play.plansim import SimUnit
    from stratagemdsl.play.unitdata import stats_of

    env = StratagemTacticsEnv(spec_path, engine="fake", seed=0, episode_limit=5)
    try:
        env.reset()
        ctrl = env._controller
        original = ctrl.debug
        released = []

        def debug_with_broodlings(cmds):
            hatcheries = [u for u in ctrl.sim.units if u.alive and u.unit_type == "Hatchery"]
            original(cmds)
            for h in hatcheries:
                if not h.alive and not released:     # one batch of 2 broodlings, owner 1, no force
                    for k in range(2):
                        b = SimUnit(0x7000000 + k, "Broodling", 1, h.x + k, h.y, stats_of("Broodling"), 30, 0)
                        ctrl.sim.units.append(b)
                        ctrl.sim.by_tag[b.tag] = b
                        released.append(b)
        ctrl.debug = debug_with_broodlings
        for _ in range(2):
            _run(env, random_policy())
        assert released, "test hook did not fire"
        assert all(not b.alive for b in released)
        assert sum(1 for s in env.tracker.slots("agent") if s.present) == env.n_agents
    finally:
        env.close()
