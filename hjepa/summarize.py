"""Success rates of hjepa/reproduce.sh: mean ± standard deviation over seeds 42/43/44.

    python hjepa/summarize.py
"""
import os
import re
from pathlib import Path

import numpy as np

CELLS = {"cube": (1, 12), "pusht": (3, 34), "ant": (3, 9), "fourroom": (1, 8)}
SEEDS = (42, 43, 44)


def success(bench, h, H, seed):
    f = (Path(os.environ["HJEPA_HOME"]) / "ckpts" / bench / f"{bench}_lewm" / f"seed{seed}"
         / f"metro_h{h}_H{H}" / "metrics.yaml")
    if not f.exists():
        return None
    return float(re.search(r"^success_rate:\s*(\S+)", f.read_text(), re.M).group(1))


def main():
    print("| benchmark | h | H | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean ± SD |")
    print("|---|---|---|" + "---|" * len(SEEDS) + "---|")
    for bench, (h, H) in CELLS.items():
        r = [success(bench, h, H, s) for s in SEEDS]
        done = [x for x in r if x is not None]
        sd = np.std(done, ddof=1) if len(done) > 1 else float("nan")
        mean = f"{np.mean(done):.1f} ± {sd:.1f}" + ("" if len(done) == len(SEEDS) else f" (n={len(done)})") if done else "—"
        print(f"| {bench} | {h} | {H} | " + " | ".join("—" if x is None else f"{x:.0f}" for x in r) + f" | {mean} |")


if __name__ == "__main__":
    main()
