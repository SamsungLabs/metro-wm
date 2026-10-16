# Adapted from LeWorldModel (LeWM); MIT License, see LICENSES/LeWM-LICENSE.
"""Step 3: evaluate Metro in the simulator.

    python eval.py --wm lewm --bench pusht --h 5 --H 28 --T 300 --run_seed 7

h: micro plan length in action blocks of 5 frames. H: waypoint budget in frames. T: episode budget
in frames. The graph is re-entered every 2 micro plans (10h frames), so T must be a multiple of 10h.
Appends one row to <out>/results.jsonl.
"""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
import yaml
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from metro import swm_patches
from metro.episodes import sample_eval_episodes
from metro.graph import MetroGraph
from metro.policy import MetroPolicy

swm_patches.apply()

REPO = Path(__file__).resolve().parent
ACTION_BLOCK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="lewm", choices=["lewm", "pldm"])
    ap.add_argument("--bench", required=True, choices=["pusht", "cube", "tworoom", "reacher"])
    ap.add_argument("--h", type=int, required=True, help="micro plan length, in action blocks")
    ap.add_argument("--H", type=int, required=True, help="waypoint budget, in frames")
    ap.add_argument("--T", type=int, required=True, help="episode budget, in frames")
    ap.add_argument("--run_seed", type=int, default=7, help="planner (CEM) seed")
    ap.add_argument("--seed", type=int, default=42, help="selects the evaluation episodes")
    ap.add_argument("--d", type=int, default=75, help="goal offset, in frames")
    ap.add_argument("--num_eval", type=int, default=50)
    ap.add_argument("--k_edges", type=int, default=4)
    ap.add_argument("--artifacts", default=str(REPO / "artifacts"))
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--save_video", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / "configs" / f"{a.bench}.yaml").read_text())
    replan = 2 * a.h * ACTION_BLOCK
    assert a.T % replan == 0, f"T={a.T} must be a multiple of the macro replan interval 10h={replan}"
    tag = f"{a.wm}_{a.bench}_h{a.h}_H{a.H}_T{a.T}_d{a.d}_rs{a.run_seed}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    world = swm.World(**cfg["world"], num_envs=a.num_eval, max_episode_steps=2 * a.T, history_size=1,
                      frame_skip=1, image_shape=(224, 224))
    transform = transforms.Compose([transforms.ToImage(), transforms.Resize(size=224)])

    dataset = swm.data.HDF5Dataset(cfg["dataset"], keys_to_cache=cfg["keys_to_cache"],
                                   cache_dir=swm.data.utils.get_cache_dir())
    process = {}
    for c in cfg["keys_to_cache"]:
        v = dataset.get_col_data(c)
        process[c] = preprocessing.StandardScaler().fit(v[~np.isnan(v).any(axis=1)])
        if c != "action":
            process[f"goal_{c}"] = process[c]

    model = swm.policy.AutoCostModel(f"{a.bench}/{a.wm}").to("cuda").eval()
    model.requires_grad_(False)
    solver = swm.solver.CEMSolver(model=model, batch_size=1, num_samples=300, var_scale=1.0,
                                  n_steps=cfg["n_steps"], topk=30, device="cuda", seed=a.run_seed)

    _, episodes, starts = sample_eval_episodes(dataset, a.num_eval, a.seed, a.d)
    graph = MetroGraph(Path(a.artifacts) / a.wm / a.bench, horizon=a.H, k_edges=a.k_edges)
    policy = MetroPolicy(micro_solver=solver, encoder=model, graph=graph,
                         micro_config=swm.PlanConfig(horizon=a.h, receding_horizon=a.h,
                                                     action_block=ACTION_BLOCK),
                         episodes=episodes, replan_interval=replan, process=process,
                         transform={"pixels": transform, "goal": transform})
    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset, start_steps=starts.tolist(), goal_offset_steps=a.d, eval_budget=a.T,
        episodes_idx=episodes.tolist(), callables=cfg["callables"], save_video=a.save_video,
        video_path=out / f"{tag}_videos")
    seconds = time.time() - t0 - float(getattr(world, "last_video_seconds", 0.0))

    succ = np.asarray(metrics["episode_successes"]).astype(bool)
    row = dict(tag=tag, wm=a.wm, bench=a.bench, h=a.h, H=a.H, T=a.T, d=a.d, seed=a.seed,
               run_seed=a.run_seed, k_edges=a.k_edges, successes=int(succ.sum()), n=int(len(succ)),
               success_rate=float(100 * succ.mean()), seconds=seconds,
               episodes=episodes.tolist(), start_steps=starts.tolist(),
               episode_successes=succ.tolist(),
               first_success_steps=np.asarray(metrics["first_success_step"]).astype(int).tolist())
    with (out / "results.jsonl").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[metro] {tag}: {succ.sum()}/{len(succ)} = {100 * succ.mean():.2f}%  "
          f"({seconds / len(succ):.1f} s/episode)")


if __name__ == "__main__":
    main()
