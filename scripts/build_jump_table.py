"""Step 2a: exact k-nearest neighbours of every frame among frames of OTHER episodes (bridge candidates).

    python scripts/build_jump_table.py --wm lewm --bench pusht

Writes <artifacts>/<wm>/<bench>/jump_{idx,dist}.npy, columns sorted by distance: the first k_edges
(= 4, what Metro uses) of an exact k_search = 32 nearest-neighbour search. Frames with bit-identical
latents make exact distance ties, and a top-4 search breaks them differently from the first 4 of a
top-32 search, so the search width is kept at 32 to reproduce the reported graphs.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="lewm", help="checkpoint $STABLEWM_HOME/<bench>/<wm>_object.ckpt")
    ap.add_argument("--bench", required=True, choices=["pusht", "cube", "tworoom", "reacher", "ant", "fourroom"])
    ap.add_argument("--artifacts", default=str(REPO / "artifacts"))
    ap.add_argument("--k_search", type=int, default=32, help="neighbours searched per frame")
    ap.add_argument("--k_edges", type=int, default=4, help="nearest of them written, <= k_search")
    ap.add_argument("--block", type=int, default=1024,
                    help="query rows per cdist; lowered automatically to fit the free GPU memory")
    a = ap.parse_args()
    assert a.k_edges <= a.k_search, f"--k_edges {a.k_edges} > --k_search {a.k_search}"
    d = Path(a.artifacts) / a.wm / a.bench
    bz = torch.from_numpy(np.asarray(np.load(d / "bank_z.npy", mmap_mode="r"), np.float32)).cuda()
    bep = torch.from_numpy(np.load(d / "bank_ep.npy").astype(np.int64)).cuda()
    N = bz.shape[0]
    # each query row costs about 4 float32 rows of length N at peak (cdist + its intermediates);
    # rows are independent, so the block size does not change the result
    fit = int(0.8 * torch.cuda.mem_get_info()[0]) // (4 * 4 * N)
    a.block = max(1, min(a.block, fit))
    print(f"  {N} frames, block {a.block}", flush=True)
    idx = np.empty((N, a.k_edges), np.int32)
    dst = np.empty((N, a.k_edges), np.float16)
    t0 = time.time()
    for s in range(0, N, a.block):
        e = min(s + a.block, N)
        dd = torch.cdist(bz[s:e], bz)
        dd.masked_fill_(bep[None, :] == bep[s:e][:, None], float("inf"))
        v, j = torch.topk(dd, a.k_search, dim=1, largest=False)
        idx[s:e] = j[:, :a.k_edges].cpu().numpy().astype(np.int32)
        dst[s:e] = v[:, :a.k_edges].cpu().numpy().astype(np.float16)
        if (s // a.block) % 200 == 0:
            print(f"  {e}/{N}  {time.time() - t0:.0f}s", flush=True)
    np.save(d / "jump_idx.npy", idx)
    np.save(d / "jump_dist.npy", dst)
    print(f"wrote {d}/jump_{{idx,dist}}.npy (nearest {a.k_edges} of {a.k_search}) "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
