"""Step 2b: the latent-gap -> frames curve that prices every bridge.

    python scripts/calibrate_edge_cost.py --wm lewm --bench pusht

For D = 1..H_max, the p-th percentile of the latent distance between two frames of the same
episode D frames apart (running maximum, so the curve is non-decreasing). A bridge of latent gap g
costs the D at which the curve reaches g, floored at 1 and clamped at H. Metro at waypoint budget
H uses the first H knots; `cap` (the widest admissible bridge) is the curve's top.

Writes <artifacts>/<wm>/<bench>/edge_cost.npz.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="lewm", choices=["lewm", "pldm"])
    ap.add_argument("--bench", required=True, choices=["pusht", "cube", "tworoom", "reacher"])
    ap.add_argument("--artifacts", default=str(REPO / "artifacts"))
    ap.add_argument("--horizon", type=int, default=40, help="H_max in frames")
    ap.add_argument("--pctl", type=float, default=25.0)
    ap.add_argument("--n_probe", type=int, default=20000)
    ap.add_argument("--n_per_D", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.artifacts) / a.wm / a.bench
    bz = torch.from_numpy(np.asarray(np.load(d / "bank_z.npy", mmap_mode="r"), np.float32)).cuda()
    bep = np.load(d / "bank_ep.npy").astype(np.int64)
    N = len(bep)

    base = np.random.default_rng(a.seed).choice(N - a.horizon - 1, a.n_probe, replace=False)
    lat = np.empty(a.horizon, np.float64)
    for D in range(1, a.horizon + 1):
        i = base[bep[base] == bep[base + D]][: a.n_per_D]           # same-episode pairs only
        gap = torch.linalg.vector_norm(bz[i + D] - bz[i], dim=1).cpu().numpy()
        lat[D - 1] = np.percentile(gap, a.pctl)
    raw = lat.copy()
    lat = np.maximum.accumulate(lat)
    np.savez(d / "edge_cost.npz", lat=lat, lat_raw=raw, frm=np.arange(1, a.horizon + 1, dtype=np.float64),
             pctl=a.pctl, horizon=a.horizon, n_probe=a.n_probe, seed=a.seed)
    print(f"wrote {d}/edge_cost.npz: 1 frame at {lat[0]:.3f} latent, {a.horizon} frames at {lat[-1]:.3f}")


if __name__ == "__main__":
    main()
