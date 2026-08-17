"""Freeze the cockpit into a standalone Windows executable.

    python build_cockpit.py            -> dist/Cockpit.exe
    python build_cockpit.py --debug    -> dist/CockpitDebug.exe (keeps a console)

The result needs no Python installed and no dependencies beyond a running
MetaTrader 5 terminal. It is a real Win32 application — Tk widgets, not a
browser — so there is no Electron runtime and no HTML anywhere in it.

## Why the debug build exists

`--windowed` detaches the process from a console, which also discards stderr.
A frozen app that dies on a missing module then exits silently with nothing to
read. The debug target is identical except it keeps a console, so when the
windowed build misbehaves there is somewhere for the traceback to go.

## Excluded modules

pandas, scipy and matplotlib are pulled in by `research/` but never touched by
the cockpit's path through the code. Excluding them takes the bundle from well
over 100 MB to about 20.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

COMMON = [
    "--noconfirm", "--clean", "--onefile",
    "--paths", ".",
    "--hidden-import", "MetaTrader5",
    "--hidden-import", "numpy",
    "--collect-submodules", "core",
    "--collect-submodules", "live",
    # present in requirements for research/, never on the cockpit's path
    "--exclude-module", "matplotlib",
    "--exclude-module", "pandas",
    "--exclude-module", "scipy",
    "--exclude-module", "pytest",
    "--exclude-module", "PIL",
    "--exclude-module", "IPython",
]


def build(debug: bool) -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed.  pip install pyinstaller")
        return 1

    name = "CockpitDebug" if debug else "Cockpit"
    cmd = [sys.executable, "-m", "PyInstaller", *COMMON, "--name", name]
    if not debug:
        cmd.append("--windowed")
    cmd.append("cockpit_app.py")

    print(" ".join(cmd), "\n")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        return r.returncode

    exe = os.path.join(ROOT, "dist", f"{name}.exe")
    if os.path.exists(exe):
        mb = os.path.getsize(exe) / 1e6
        print(f"\nbuilt {exe}  ({mb:.1f} MB)")
        print("Requires a running, logged-in MetaTrader 5 terminal.")
        print("Read-only: it cannot place, modify or close an order.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--debug", action="store_true",
                    help="keep a console so tracebacks are visible")
    ap.add_argument("--clean-artifacts", action="store_true",
                    help="remove build/, dist/ and the generated .spec files")
    a = ap.parse_args(argv)

    if a.clean_artifacts:
        for d in ("build", "dist"):
            p = os.path.join(ROOT, d)
            if os.path.isdir(p):
                shutil.rmtree(p)
                print("removed", p)
        for f in os.listdir(ROOT):
            if f.endswith(".spec"):
                os.remove(os.path.join(ROOT, f))
                print("removed", f)
        return 0

    return build(a.debug)


if __name__ == "__main__":
    raise SystemExit(main())
