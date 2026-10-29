"""A scripted baseline that follows the strategy's plan through the env's macro actions.

Not a learner: it reads the tracked plan (needs StratagemDSL) and, per agent,
* executes the allied orders of the active phases -- attack(F, target) -> attack_force:target
  (or move_to:region), move/retreat(F, region) -> move_to:region;
* otherwise produces the event the plan waits for: enter(F, region) -> move_to:region;
* otherwise attacks the first enemy in range, else stops.
Useful as a sanity check (can the map be won at all through this action space?)
and as a reference curve next to MARL.

    python examples/plan_follower.py build/sc2te/dhls_18_enter_signal --engine fake --episodes 3
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv

CALL = re.compile(r"(\w+)\(\s*(\w+)\s*(?:,\s*(\w+))?")


class PlanFollower:
    def __init__(self, env: StratagemTacticsEnv):
        if env.plan is None:
            raise RuntimeError("PlanFollower needs plan tracking (StratagemDSL importable)")
        self.env = env
        self.names = env.get_action_names()
        self.index = {n: i for i, n in enumerate(self.names)}
        self.force_of_agent = [s.force for s in env.tracker.slots("agent")]

    def _wanted(self, force: str) -> list[str]:
        pt = self.env.plan.pt
        wanted = []
        for text in pt.orders_for(force):
            m = CALL.match(text)
            if not m:
                continue
            verb, _actor, target = m.groups()
            if verb in ("attack", "retarget") and target:
                wanted += [f"attack_force:{target}", f"attack_to:{target}", f"move_to:{target}"]
            elif verb in ("move", "retreat") and target:
                wanted += [f"move_to:{target}", f"attack_to:{target}"]
        for text in pt.expected_events(force):
            m = CALL.match(text)
            if m and m.group(1) == "enter" and m.group(3):
                wanted += [f"move_to:{m.group(3)}", f"attack_to:{m.group(3)}"]
        return wanted

    def act(self) -> list[int]:
        out = []
        cache: dict[str, list[str]] = {}
        for a_id, avail in enumerate(self.env.get_avail_actions()):
            if avail[0]:
                out.append(0)
                continue
            force = self.force_of_agent[a_id]
            wanted = cache.setdefault(force, self._wanted(force))
            choice = next((self.index[n] for n in wanted if n in self.index and avail[self.index[n]]), None)
            if choice is None:
                attacks = [i for i, n in enumerate(self.names) if n.startswith("attack:") and avail[i]]
                choice = attacks[0] if attacks else 1
            out.append(choice)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("--engine", default="sc2", choices=["sc2", "fake"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    env = StratagemTacticsEnv(args.scenario, engine=args.engine, seed=args.seed, plan_tracking="on",
                              info_detail=True)
    print(env.describe())
    pol = PlanFollower(env)
    for ep in range(args.episodes):
        env.reset()
        done, ret, steps, info = False, 0.0, 0, {}
        while not done:
            r, done, info = env.step(pol.act())
            ret += r[0] if isinstance(r, tuple) else r
            steps += 1
        print(f"episode {ep}: {info.get('outcome')} after {steps} steps, return {ret:.2f}, "
              f"phases {info.get('plan_phases')}")
    print(env.get_stats())
    env.close()


if __name__ == "__main__":
    main()
