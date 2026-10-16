"""Step 4: success at every frontier point of configs/frontier.txt, from results/results.jsonl.

    python summarize.py                 # every goal distance with results
    python summarize.py --d 75

Success is any-time and latching, so the success rate at a budget T' <= T is read off a run at T
from each episode's first-success frame. A frontier point is listed only when its own (h, H) cell
has been run at a budget of at least its T. Reports mean +- std over run seeds.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def frontier(path):
    """-> [(d, wm, bench, h, H, T)] in file order."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            d, wm, bench, h, H, T = line.split()
            rows.append((int(d), wm, bench, int(h), int(H), int(T)))
    return rows


def success_at(row, T):
    return 100.0 * sum(1 for x in row["first_success_steps"] if 0 <= x <= T - 1) / row["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "results.jsonl"))
    ap.add_argument("--frontier", default=str(REPO / "configs" / "frontier.txt"))
    ap.add_argument("--d", type=int, nargs="*", help="goal distances to list (default: all)")
    a = ap.parse_args()
    runs = defaultdict(dict)
    for line in Path(a.results).read_text().splitlines():
        r = json.loads(line)
        runs[(r["d"], r["wm"], r["bench"], r["h"], r["H"])][r["run_seed"]] = r   # last run per seed

    print(f"{'d':>2}  {'model':6} {'bench':8} {'h':>2} {'H':>3} {'T':>4}  {'success (%)':>15}  seeds")
    listed = 0
    for d, wm, bench, h, H, T in sorted(frontier(a.frontier), key=lambda p: (p[0], p[1])):
        if a.d and d not in a.d:
            continue
        rows = sorted((r for r in runs.get((d, wm, bench, h, H), {}).values() if r["T"] >= T),
                      key=lambda r: r["run_seed"])
        if not rows:
            continue
        s = [success_at(r, T) for r in rows]
        sd = np.std(s, ddof=1) if len(s) > 1 else 0.0
        counts = "/".join(str(sum(0 <= x <= T - 1 for x in r["first_success_steps"])) for r in rows)
        print(f"{d:>2}  {wm:6} {bench:8} {h:>2} {H:>3} {T:>4}  {np.mean(s):6.2f} +- {sd:5.2f}  "
              f"{len(s)} ({counts} of {rows[0]['n']})")
        listed += 1
    if not listed:
        print("(no frontier point has results yet)")


if __name__ == "__main__":
    main()
