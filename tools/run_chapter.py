"""Execute the code cells of one or more MyST Markdown chapters outside the book build.

Usage: ``python tools/run_chapter.py book/02-labeling/05-kinetic-model.md [more.md ...] [--figures DIR]``

Each file's ```{code-cell} python``` blocks run in order in one fresh namespace with the
non-interactive matplotlib backend, the working directory set to the repository root (as
mystmd does), and every figure closed after the cell. Printed output is shown under the cell
number, exceptions stop that chapter and are reported with the cell's source, and the wall
time per cell is printed so slow cells (the book's budget is about 60 s per cell) are visible.
With ``--figures DIR`` every figure is also saved as PNG for inspection.

Exit status is the number of chapters that failed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```", re.S | re.M)


def cells(text: str) -> list[str]:
    out = []
    for m in CELL.finditer(text):
        body = m.group(1)
        # drop the leading option lines (:tags: [...]) of the cell
        lines = body.split("\n")
        while lines and lines[0].startswith(":"):
            lines.pop(0)
        out.append("\n".join(lines))
    return out


def run(path: Path, figures: Path | None) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = path.read_text(encoding="utf-8")
    ns: dict = {"__name__": "__main__"}
    print(f"== {path.relative_to(ROOT)}")
    t_total = time.time()
    for i, code in enumerate(cells(src), 1):
        t0 = time.time()
        try:
            exec(compile(code, f"{path.name}#cell{i}", "exec"), ns)
        except Exception:
            print(f"-- cell {i} FAILED after {time.time() - t0:.1f} s:\n{code}\n")
            traceback.print_exc()
            plt.close("all")
            return False
        dt = time.time() - t0
        if figures is not None:
            for k, num in enumerate(plt.get_fignums()):
                plt.figure(num).savefig(figures / f"{path.stem}_cell{i:02d}_{k}.png", dpi=80)
        plt.close("all")
        flag = "  (slow)" if dt > 60 else ""
        print(f"-- cell {i}: {dt:.1f} s{flag}")
    print(f"== ok: {len(cells(src))} cells in {time.time() - t_total:.0f} s\n")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="+")
    ap.add_argument("--figures", help="directory to save every figure as PNG")
    a = ap.parse_args()
    os.chdir(ROOT)
    figures = None
    if a.figures:
        figures = Path(a.figures).resolve()
        figures.mkdir(parents=True, exist_ok=True)
    failed = 0
    for ch in a.chapters:
        if not run(Path(ch).resolve(), figures):
            failed += 1
    sys.exit(failed)


if __name__ == "__main__":
    main()
