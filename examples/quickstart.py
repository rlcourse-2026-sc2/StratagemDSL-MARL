"""Load a StratagemDSL map in StarCraft II and play a few episodes with random agents.

    python examples/quickstart.py build/sc2te/dhls_18_enter_signal                 # real StarCraft II
    python examples/quickstart.py build/sc2te/dhls_18_enter_signal --delay 0.1     # slow enough to watch
    python examples/quickstart.py build/sc2te/dhls_18_enter_signal --replay-dir replays   # save a replay per episode
    python examples/quickstart.py build/sc2te/dhls_18_enter_signal --fake          # no SC2 (needs StratagemDSL)

On Windows / macOS PySC2 starts the normal game client, so a game window opens
and shows the map; on Linux the headless build shows nothing (watch the replay).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", help="folder or *.sc2te.json written by tools/strat2sc2te.py")
    p.add_argument("--fake", action="store_true", help="no SC2: StratagemDSL's PlanSim stand-in")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.0, help="seconds to sleep per step (to watch the window)")
    p.add_argument("--replay-dir", help="save a replay after every episode into this folder")
    p.add_argument("--idle", action="store_true", help="agents only 'stop' (watch what the map script does)")
    p.add_argument("--print-every", type=int, default=20, help="progress line every N steps (0: off)")
    args = p.parse_args()

    env = StratagemTacticsEnv(args.scenario, engine="fake" if args.fake else "sc2", reward_tuple=False,
                              replay_dir=str(Path(args.replay_dir).resolve()) if args.replay_dir else "")
    print(env.describe())
    info = env.get_env_info()
    print("env_info:", {k: info[k] for k in ("n_agents", "n_actions", "obs_shape", "state_shape", "episode_limit")})
    print("actions:", info["action_names"])
    print("starting StarCraft II ..." if not args.fake else "fake engine")

    rng = np.random.default_rng(0)
    for episode in range(args.episodes):
        env.reset()
        print(f"episode {episode}: map loaded, units spawned: "
              f"{ {k: v['alive'] for k, v in env.tracker.summary().items()} }")
        terminated, episode_reward, step_info, t = False, 0.0, {}, 0
        while not terminated:
            avail = env.get_avail_actions()
            if args.idle:
                actions = [1 if a[1] else 0 for a in avail]
            else:
                actions = [int(rng.choice(np.nonzero(a)[0])) for a in avail]
            reward, terminated, step_info = env.step(actions)
            episode_reward += reward
            t += 1
            for e in env.last_detail.get("events", []):
                if e["event"] in ("enter", "leave"):
                    print(f"    step {t}: {e['force']} {e['event']}s {e['region']}")
            if args.print_every and t % args.print_every == 0:
                alive = {k: v["alive"] for k, v in env.tracker.summary().items()}
                print(f"    step {t}/{env.episode_limit}  game time {env.tracker.seconds():.0f}s  alive {alive}")
            if args.delay:
                time.sleep(args.delay)
        # PyMARL logs the numeric info of the last step; env.last_detail has the rest
        print(f"episode {episode}: return {episode_reward:.2f}, outcome {env.last_detail['outcome']}, "
              f"last info {step_info}")
        if args.replay_dir and not args.fake:
            env.save_replay()
    print(env.get_stats())
    env.close()


if __name__ == "__main__":
    main()
