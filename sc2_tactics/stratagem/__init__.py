"""StratagemDSL maps in sc2te_env.

    tools/strat2sc2te.py  (StratagemDSL side)  .strat -> .SC2Map + <stem>.sc2te.json
    StratagemTacticsEnv   (this package)        the spec -> a SMAC-style multi-agent env

    from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv, load_spec
    env = StratagemTacticsEnv("build/sc2te/dhls_18_enter_signal")      # folder or *.sc2te.json
"""
from .spec import FORMAT, ScenarioSpec, load_spec, resolve_unit_type
from .env import ABILITY_PRESETS, StratagemTacticsEnv
from . import conditions, maps

__all__ = ["FORMAT", "ScenarioSpec", "load_spec", "resolve_unit_type", "StratagemTacticsEnv", "ABILITY_PRESETS",
           "conditions", "maps", "make_env"]


def make_env(scenario, **kwargs) -> StratagemTacticsEnv:
    """Factory for PyMARL-style registries: ``partial(make_env, ...)`` or ``env_args: {scenario: ...}``."""
    return StratagemTacticsEnv(scenario, **kwargs)
