"""Run one unchanged lab in an isolated folder with all expected input names."""
import argparse
from pathlib import Path
import shutil
import subprocess
import os

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("lab", choices=[f"lab{i}" for i in range(1, 7)])
p.add_argument("--binary", type=Path, help="Executable path, relative to current directory")
a = p.parse_args()
binary = a.binary.resolve() if a.binary else root/"build"/(a.lab + (".exe" if os.name == "nt" else ""))
if not binary.is_file():
    p.error(f"Build the project first; executable not found: {binary}")
run = root/"runs"/a.lab
run.mkdir(parents=True, exist_ok=True)
for src in (root/"examples").iterdir():
    if src.suffix in {".png", ".jpg"}:
        shutil.copy2(src, run/src.name)
subprocess.run([str(binary)], cwd=run, check=True)
print(f"Results: {run}")
