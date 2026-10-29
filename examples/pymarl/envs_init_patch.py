# Add to PyMARL's src/envs/__init__.py (pymarl, pymarl2 and most forks use this registry):

from functools import partial

from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv


def env_fn(env, **kwargs):          # already defined in PyMARL's envs/__init__.py; keep one copy
    return env(**kwargs)


REGISTRY = globals().get("REGISTRY", {})
REGISTRY["sc2te_strat"] = partial(env_fn, env=StratagemTacticsEnv)
