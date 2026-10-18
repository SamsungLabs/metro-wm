# Adapted from H-JEPA; MIT License, see LICENSES/H-JEPA-LICENSE.
"""Metro-WM sub-goals for H-JEPA's flat planner, in H-JEPA's own evaluation harness (arXiv:2610.06805).

H-JEPA's flat LeWM baseline (their gradient-descent planner, tasks, seeds and evaluation code, all
unchanged) plans toward the goal latent. Here, before every solve, the Metro graph is re-entered at the
current observation and the planner aims at the returned sub-goal latent instead. Every solve plans
over plan_config.horizon action blocks (H-JEPA's flat planner otherwise shrinks the horizon toward the
end of the episode) and, unless plan_config.receding_horizon is given, replans after every block.

Run from H-JEPA/h_jepa in H-JEPA's environment, with eval.py's arguments plus the graph:

    python <metro-wm>/hjepa/metro_in_hjepa.py --config-name pusht_flat seed=42 \
        policy=$HJEPA_HOME/ckpts/pusht/pusht_lewm/seed42/pusht_lewm_object.ckpt output.dir=metro_h3_H34 \
        plan_config.horizon=3 +metro.artifacts=<metro-wm>/artifacts/hj_lewm_s42/pusht +metro.H=34
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"

import h_jepa
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import DictConfig
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, h_jepa.__path__[0])                          # H-JEPA's modules, as its eval.py sees them
sys.path.append(str(Path(__file__).resolve().parent.parent))   # after H-JEPA's modules: only `metro` is needed
import planning_eval
import stable_worldmodel as swm
from metro.graph import MetroGraph


class HeldOutGraph(MetroGraph):
    """MetroGraph without leave-one-out, for tasks that are not episodes of the graph.

    H-JEPA's test tasks are held out from the graph altogether, so tasks are keyed by ids that are never
    graph episodes and nothing is removed. `prepare` then appends the goal to a transpose of the graph
    built once, instead of slicing and copying the whole graph for every task. Only the goal's row matters
    to a backward Dijkstra from the goal, so distances and routes are those of MetroGraph.prepare."""

    CHUNK = 1 << 20

    def _dist(self, z, ep):
        """Distance from z to every node, over 1M-node slices of the bank (AntMaze's has 10M nodes)."""
        return torch.cat([torch.cdist(z[None], self.bz[s:s + self.CHUNK])[0] for s in range(0, self.N, self.CHUNK)])

    def prepare(self, ep, z_goal):
        if ep in self.tasks:
            return
        assert not (self.bep == ep).any(), f"task key {ep} is a graph episode: use MetroGraph for leave-one-out"
        if not hasattr(self, "_GT"):
            self._GT = self.G.T.tocsr()
            self._GT.sort_indices()
            self._ident = np.arange(self.N)
        GT, M = self._GT, self.N
        near = torch.topk(self._dist(z_goal, ep), self.k_edges, largest=False)
        gidx = near.indices.cpu().numpy()
        glat = near.values.cpu().numpy().astype(float)
        gcost = self.frames(glat).astype(np.float32)
        legal = glat <= self.cap
        if not legal.any():
            legal = np.zeros_like(legal)
            legal[0] = True
        gi, gc = gidx[legal], gcost[legal]
        order = np.argsort(gi)
        gi, gc = gi[order].astype(GT.indices.dtype), gc[order]
        A = sp.csr_matrix((np.concatenate([GT.data, gc]), np.concatenate([GT.indices, gi]),
                           np.concatenate([GT.indptr, [GT.indptr[-1] + len(gi)]])), shape=(M + 1, M + 1))
        dv, pred = dijkstra(A, indices=M, directed=True, return_predecessors=True)
        self.tasks[ep] = dict(M=M, dv=dv[:M].astype(np.float32), pred=pred[:M].astype(np.int32),
                              remap=self._ident, inv=self._ident, z_goal=z_goal)


class MetroGoalPolicy(swm.policy.WorldModelPolicy):
    """H-JEPA's WorldModelPolicy whose solver aims at a Metro sub-goal, re-chosen at every solve."""

    def __init__(self, graph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.chunk = 0
        self.reasons = Counter()
        self._z_goal = None

    def reset(self):
        super().reset()
        self.chunk += 1                    # H-JEPA's evaluation resets the policy once per task chunk
        self.graph.tasks.clear()
        self._z_goal = None

    @torch.no_grad()
    def _encode(self, prep, key):
        """Latent of the observation (key 'pixels') or of the goal ('goal'), as H-JEPA's JEPA cost computes
        it. Models that fuse pixels with proprioception (AntMaze) also get the matching proprio stream."""
        model = self.solver.model
        dev = next(model.parameters()).device
        info = {"pixels": prep[key].to(dev)}
        if hasattr(model.encoder, "proprio_encoder"):
            proprio = prep["proprio" if key == "pixels" else "goal_proprio"]
            info["proprio"] = torch.as_tensor(proprio).to(dev).float()
        return model.encode(info)["embed_0"][:, -1].float()

    def get_action(self, info_dict, **kwargs):
        if len(self._action_buffer) == 0:                          # the solver is about to plan
            prep = self._prepare_info(dict(info_dict))             # _prepare_info modifies its argument
            z_obs = self._encode(prep, "pixels")
            if self._z_goal is None:
                self._z_goal = self._encode(prep, "goal")
            subs = []
            for i in range(z_obs.shape[0]):
                key = -(self.chunk * 1000 + i + 1)                  # one route per env, never a graph episode id
                self.graph.prepare(key, self._z_goal[i].to(self.graph.dev))
                z, reason = self.graph.subgoal(key, z_obs[i].to(self.graph.dev))
                subs.append(z.float())
                self.reasons[reason] += 1
            info_dict = {**info_dict, "goal_embed_0": torch.stack(subs)[:, None, :].cpu()}
        return super().get_action(info_dict, **kwargs)


def fix_horizon(solver):
    """Plan every solve over the configured horizon, instead of H-JEPA's schedule that shrinks it toward
    the end of the episode budget."""
    solve = solver.solve

    def solve_fixed(info_dict, init_action=None, planning_horizon=None, steps_taken=None, eval_budget=None):
        return solve(info_dict, init_action=init_action, planning_horizon=int(solver.horizon),
                     steps_taken=steps_taken, eval_budget=eval_budget)

    solver.solve = solve_fixed


_build_policy = planning_eval._build_policy


def build_metro_policy(cfg, process, transform, model=None):
    base = _build_policy(cfg, process, transform, model=model)
    fix_horizon(base.solver)
    graph = HeldOutGraph(cfg.metro.artifacts, horizon=int(cfg.metro.H), k_edges=int(cfg.metro.get("k_edges", 4)))
    policy = MetroGoalPolicy(graph=graph, solver=base.solver, config=base.cfg, process=base.process,
                             transform=base.transform, eval_budget=base.eval_budget)
    build_metro_policy.last = policy
    return policy


planning_eval._build_policy = build_metro_policy


@hydra.main(version_base=None, config_path=str(Path(planning_eval.__file__).parent / "config" / "eval"),
            config_name="pusht_flat")
def run(cfg: DictConfig):
    assert "metro" in cfg, "pass +metro.artifacts=<graph dir> +metro.H=<waypoint budget>"
    assert "GradientSolver" in cfg.solver._target_, "Metro replaces the goal of H-JEPA's flat planner: use a *_flat config"
    overridden = {o.split("=")[0].lstrip("+~") for o in HydraConfig.get().overrides.task}
    if "plan_config.receding_horizon" not in overridden:
        cfg.plan_config.receding_horizon = 1                    # replan after every action block
    planning_eval.run_planning_eval(cfg)
    print("metro sub-goal reasons:", json.dumps(dict(build_metro_policy.last.reasons)))


if __name__ == "__main__":
    run()
