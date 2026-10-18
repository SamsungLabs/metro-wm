# Adapted from H-JEPA; MIT License, see LICENSES/H-JEPA-LICENSE.
"""Graph step 1 for H-JEPA's Visual AntMaze and FourRoom Distractors: encode every training frame.

AntMaze's LeWM fuses pixels with proprioception, which Metro's LeWM class lacks, so these banks are
built in H-JEPA's environment with H-JEPA's own model and the preprocessing its planner applies to
observations (`_prepare_info`: training normaliser on proprio, image transform on pixels). The latents
are the ones hjepa/metro_in_hjepa.py computes for the observation and goal. Steps 2-3 (jump table,
edge cost) only read these arrays and use Metro's scripts unchanged.

    python hjepa/build_bank.py --bench ant --seed 42 --out artifacts/hj_lewm_s42/ant

Writes <out>/bank_{z,ep,st}.npy: latent (float16), episode and step of every frame the LeWM was trained
on: the whole AntMaze training file, and for FourRoom the 1M-transition episode subset its training
config draws (H-JEPA's `_select_episodes_for_transition_budget`, subset seed 0).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import h_jepa
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, h_jepa.__path__[0])                          # H-JEPA's modules, as its eval.py sees them
import planning_eval
import stable_worldmodel as swm
from data import _select_episodes_for_transition_budget

TRAIN = {"ant": "visual_antmaze_medium_explore_stitch_train", "fourroom": "fourroom_tp35_d1"}
SUBSET = {"fourroom": (1_000_000, 0)}       # H-JEPA config/train/base/<bench>.yaml: (total_transitions, subset_seed)
SPAN = 20                                   # level-1 frameskip 5 x (history 3 + rollout 1): shortest usable episode


def training_rows(f, bench):
    """Row blocks (start, end) of the episodes the LeWM was trained on, in file order."""
    ep_len, ep_off = f["ep_len"][:].astype(np.int64), f["ep_offset"][:].astype(np.int64)
    eps = np.arange(len(ep_len))
    assert (f["ep_idx"][:][ep_off] == eps).all(), "ep_idx is not the episode's position in ep_len / ep_offset"
    if bench in SUBSET:
        stub = type("D", (), dict(lengths=ep_len, last_level={"span": SPAN}, h5_path=f.filename))
        eps = _select_episodes_for_transition_budget(stub, SUBSET[bench][0], np.random.default_rng(SUBSET[bench][1]))
        assert (ep_len >= SPAN).all(), "an episode is shorter than a training clip: the subset would differ"
    return [(int(ep_off[e]), int(ep_off[e] + ep_len[e])) for e in eps]


class Frames(Dataset):
    """Blocks of consecutive rows of the training file, preprocessed as the planner does."""

    def __init__(self, path, bench, process, transform, blocks):
        self.path, self.bench, self.blocks = path, bench, blocks
        self.policy = swm.policy.BasePolicy(process=process, transform=transform)
        self.f = None

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        if self.f is None:
            self.f = h5py.File(self.path, "r")
        s, e = self.blocks[i]
        info = {"pixels": self.f["pixels"][s:e][:, None]}                     # (B, 1, H, W, C) uint8
        if self.bench == "ant":                                              # H-JEPA's keys_to_merge: qpos[2:] + qvel
            info["proprio"] = np.concatenate([self.f["qpos"][s:e, 2:], self.f["qvel"][s:e]], -1)[:, None]
        prep = self.policy._prepare_info(info)
        out = {"pixels": prep["pixels"], "ep": torch.from_numpy(self.f["ep_idx"][s:e].astype(np.int64)),
               "st": torch.from_numpy(self.f["step_idx"][s:e].astype(np.int64))}
        if "proprio" in prep:
            out["proprio"] = prep["proprio"]
        return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(TRAIN))
    ap.add_argument("--seed", type=int, required=True, help="H-JEPA checkpoint seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=12)
    a = ap.parse_args()
    home = Path(os.environ["HJEPA_HOME"])
    cfg = planning_eval.load_eval_config(f"{a.bench}_flat")
    cfg.cache_dir = str(home)
    cfg.policy = str(home / "ckpts" / a.bench / f"{a.bench}_lewm" / f"seed{a.seed}" / f"{a.bench}_lewm_object.ckpt")
    process = planning_eval._build_eval_process_from_policy_normalizer(cfg)
    process = {k: v for k, v in process.items() if k == "proprio"}           # the only processed input the encoder reads
    transform = {"pixels": planning_eval.img_transform(cfg)}
    model = planning_eval.load_model(cfg)
    model.requires_grad_(False)

    path = home / f"{TRAIN[a.bench]}.h5"
    with h5py.File(path, "r") as f:
        spans = training_rows(f, a.bench)
        print(f"{path.name}: {len(f['ep_idx'])} rows; encoding {len(spans)} episodes, "
              f"{sum(e - s for s, e in spans)} rows", flush=True)
    blocks = [(b, min(b + a.block, e)) for s, e in spans for b in range(s, e, a.block)]   # never across episodes
    n = sum(e - s for s, e in blocks)
    loader = DataLoader(Frames(path, a.bench, process, transform, blocks), batch_size=None, shuffle=False,
                        num_workers=a.num_workers, prefetch_factor=4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zs, eps, sts, t0 = [], [], [], time.time()
    for i, b in enumerate(loader):
        info = {"pixels": b["pixels"].cuda(non_blocking=True)}
        if "proprio" in b:
            info["proprio"] = b["proprio"].cuda(non_blocking=True)
        with sdpa_kernel(SDPBackend.MATH):                                   # the attention kernel of H-JEPA's evaluation
            z = model.encode(info)["embed_0"][:, -1].float()
        zs.append(z.half().cpu().numpy())
        eps.append(b["ep"].numpy())
        sts.append(b["st"].numpy())
        if i % 500 == 0:
            print(f"  {sum(len(x) for x in eps)}/{n} rows  {time.time() - t0:.0f}s", flush=True)
    z, ep, st = np.concatenate(zs), np.concatenate(eps), np.concatenate(sts)
    assert (np.diff(ep) >= 0).all(), "rows are not grouped by episode"
    assert (np.diff(st)[ep[1:] == ep[:-1]] == 1).all(), "steps are not consecutive within episodes"
    np.save(out / "bank_z.npy", z)
    np.save(out / "bank_ep.npy", ep.astype(np.int32))
    np.save(out / "bank_st.npy", st.astype(np.int32))
    print(f"wrote {out}: {len(z)} frames from {len(np.unique(ep))} episodes, dim {z.shape[1]}, "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
