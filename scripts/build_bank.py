"""Step 1: encode every frame of a dataset with the frozen world model -> the graph's nodes.

    python scripts/build_bank.py --wm lewm --bench pusht

Writes <artifacts>/<wm>/<bench>/bank_{z,ep,st}.npy: latent (float16), episode and step of each frame.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stable_worldmodel as swm

from metro.episodes import episode_column

REPO = Path(__file__).resolve().parent.parent


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="lewm", help="checkpoint $STABLEWM_HOME/<bench>/<wm>_object.ckpt")
    ap.add_argument("--bench", required=True, choices=["pusht", "cube", "tworoom", "reacher"])
    ap.add_argument("--artifacts", default=str(REPO / "artifacts"))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--max_episode", type=int, default=None,
                    help="keep only episodes with a smaller id (9800: H-JEPA's Cube training split)")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / "configs" / f"{a.bench}.yaml").read_text())
    out = Path(a.artifacts) / a.wm / a.bench
    out.mkdir(parents=True, exist_ok=True)

    model = swm.policy.AutoCostModel(f"{a.bench}/{a.wm}").to("cuda").eval()
    model.requires_grad_(False)
    dataset = swm.data.HDF5Dataset(cfg["dataset"], frameskip=5, num_steps=1)
    col = episode_column(dataset)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=True, prefetch_factor=4)

    zs, eps, sts, t0 = [], [], [], time.time()
    for i, batch in enumerate(loader):
        emb = model.encode({"pixels": batch["pixels"].to("cuda", non_blocking=True)})["emb"]
        zs.append(emb.reshape(-1, emb.size(-1)).half().cpu().numpy())
        eps.append(batch[col].reshape(-1).to(torch.int32).numpy())
        sts.append(batch["step_idx"].reshape(-1).to(torch.int32).numpy())
        if i % 400 == 0:
            print(f"  {i * a.batch_size}/{len(dataset)} frames  {time.time() - t0:.0f}s", flush=True)
    z, ep, st = np.concatenate(zs), np.concatenate(eps), np.concatenate(sts)
    key = ep.astype(np.int64) * 100_000 + st.astype(np.int64)
    _, keep = np.unique(key, return_index=True)
    keep.sort()
    if a.max_episode is not None:
        keep = keep[ep[keep] < a.max_episode]
    np.save(out / "bank_z.npy", z[keep])
    np.save(out / "bank_ep.npy", ep[keep])
    np.save(out / "bank_st.npy", st[keep])
    print(f"wrote {out}: {len(keep)} frames from {len(np.unique(ep[keep]))} episodes "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
