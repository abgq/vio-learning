"""Run every solution and check script and store its output in scripts/solutions/output/<script>.txt.

  .venv/Scripts/python.exe scripts/solutions/run_all.py           # everything (about 20 minutes)
  .venv/Scripts/python.exe scripts/solutions/run_all.py --quick   # skip the long Monte Carlo studies
Exit code 1 if any script fails. The C++ experiments of Chapter 30 have their own runner, cpp/build_and_run.ps1.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
LONG = {"ch27_window_study.py", "ch29_ideal_augmentation.py", "ch29_nullspace_leak.py", "ch27_ch29_planar_numbers.py"}


def main(quick):
    OUT.mkdir(exist_ok=True)
    scripts = sorted(p for p in HERE.glob("ch*.py")) + sorted((HERE.parent / "checks").glob("*.py"))
    failed = []
    for script in scripts:
        if quick and script.name in LONG:
            print(f"skip  {script.relative_to(HERE.parent)}")
            continue
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script)], cwd=HERE.parents[1], capture_output=True, text=True)
        (OUT / (script.stem + ".txt")).write_text(res.stdout + (("\n[stderr]\n" + res.stderr) if res.returncode else ""),
                                                  encoding="utf-8", newline="\n")
        status = "ok  " if res.returncode == 0 else "FAIL"
        if res.returncode:
            failed.append(script.name)
        print(f"{status}  {script.relative_to(HERE.parent)}  ({time.time() - t0:.0f} s)", flush=True)
    print("all scripts ran" if not failed else f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main("--quick" in sys.argv))
