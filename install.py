#!/usr/bin/env python
"""Install the bridge into the SMAC package that sc2te_env lives in (Linux / macOS / Windows).

    python install.py                          # smac found on the import path
    python install.py --sc2te ../sc2te_env     # also overlay sc2te_env onto smac/env first
    python install.py --target /path/to/smac/env --link   # symlink instead of copy (for development)

sc2te_env is written as an overlay of SMAC's ``smac/env`` folder (its modules
import ``smac.env.sc2_tactics...``), so the order is: pip install SMAC ->
copy sc2te_env into smac/env -> copy this package into smac/env/sc2_tactics/stratagem.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def smac_env_dir() -> Path:
    try:
        import smac
    except ImportError:
        sys.exit("SMAC is not installed: pip install git+https://github.com/oxwhirl/smac.git")
    return Path(smac.__file__).resolve().parent / "env"


def overlay_sc2te(src: Path, target: Path) -> None:
    for name in ("sc2_tactics", "pettingzoo"):
        if (src / name).is_dir():
            shutil.copytree(src / name, target / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("__init__.py", "multiagentenv.py"):
        if (src / name).is_file():
            shutil.copyfile(src / name, target / name)
    print(f"sc2te_env  -> {target}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", help="smac/env directory (default: from the installed smac package)")
    p.add_argument("--sc2te", help="path of a sc2te_env checkout to overlay first")
    p.add_argument("--link", action="store_true", help="symlink the package instead of copying it")
    args = p.parse_args()
    target = Path(args.target) if args.target else smac_env_dir()
    if args.sc2te:
        overlay_sc2te(Path(args.sc2te).resolve(), target)
    tactics = target / "sc2_tactics"
    if not (tactics / "sc2_tactics_env.py").is_file():
        sys.exit(f"sc2te_env is not in {target}: run again with --sc2te <path to sc2te_env>")
    dst = tactics / "stratagem"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    src = HERE / "sc2_tactics" / "stratagem"
    if args.link:
        os.symlink(src, dst, target_is_directory=True)
    else:
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"bridge     -> {dst}")
    try:
        sys.path_importer_cache.clear()
        from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv  # noqa: F401
        print("import smac.env.sc2_tactics.stratagem: ok")
    except Exception as e:  # pragma: no cover - depends on the installation
        print(f"installed, but the import failed: {e!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
