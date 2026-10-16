"""The Metro graph: sub-goals routed through a graph of recorded frames.

Every frame of the dataset is a node. Temporal edges run frame -> next frame inside an episode
(1 frame, directed). Bridge edges join a frame to its k nearest frames in other episodes
(symmetric), priced in frames by the latent-gap -> frames curve and dropped above `cap`.
`prepare` inserts z_goal as a node and runs one backward Dijkstra from it; `subgoal` re-enters the
graph at z_obs with a one-step Bellman backup and walks the route H frames to the next waypoint.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra


def load_edge_cost(path, horizon):
    """The pricing curve at waypoint budget H: the first H knots of the curve built at H_max."""
    ec = np.load(path)
    assert int(ec["horizon"]) >= horizon, f"{path} is built up to H={int(ec['horizon'])} < {horizon}"
    lat = np.asarray(ec["lat"], float)[:horizon]
    frm = np.arange(1, horizon + 1, dtype=float)
    return lat, frm


class MetroGraph:
    def __init__(self, artifact_dir, horizon, k_edges=4, device="cuda"):
        a = Path(artifact_dir)
        self.dev = torch.device(device)
        self.bz = torch.from_numpy(np.asarray(np.load(a / "bank_z.npy", mmap_mode="r"), np.float32)).to(self.dev)
        self.bep = np.load(a / "bank_ep.npy").astype(np.int64)
        self.bep_t = torch.from_numpy(self.bep).to(self.dev)
        self.N = self.bz.shape[0]
        self.H = int(horizon)
        self.k_edges = int(k_edges)
        self.LAT, self.FRM = load_edge_cost(a / "edge_cost.npz", self.H)
        self.cap = round(float(self.LAT.max()), 4)    # a bridge is an edge iff the curve prices it

        ji = np.load(a / "jump_idx.npy").astype(np.int64)
        assert ji.shape[1] >= self.k_edges, (
            f"{a}/jump_idx.npy has {ji.shape[1]} neighbours per frame, fewer than k_edges={self.k_edges}; "
            f"rebuild it with scripts/build_jump_table.py --k_edges {self.k_edges}")
        ji = ji[:, :self.k_edges]
        jd = np.load(a / "jump_dist.npy").astype(np.float32)[:, :self.k_edges]
        same = np.zeros(self.N, bool)
        same[:-1] = self.bep[1:] == self.bep[:-1]
        ts = np.nonzero(same)[0]
        keep = (jd <= self.cap).ravel()
        jr = np.repeat(np.arange(self.N), ji.shape[1])[keep]
        jc = ji.ravel()[keep]
        jv = self.frames(jd).ravel()[keep].astype(np.float32)
        assert not (self.bep[jr] == self.bep[jc]).any(), "a jump-table entry is within-episode"
        B = sp.csr_matrix((jv, (jr, jc)), shape=(self.N, self.N))
        T = sp.csr_matrix((np.ones(len(ts), np.float32), (ts, ts + 1)), shape=(self.N, self.N))
        self.G = (T + B.maximum(B.T)).tocsr()           # bridges are symmetric and priced once
        print(f"[graph] {self.N} nodes, {len(ts)} temporal + {len(jr)} bridge edges "
              f"(k_edges {self.k_edges}, H {self.H}, cap {self.cap})", flush=True)
        self.tasks = {}

    def frames(self, latent_gap):
        """Latent gap -> frames. `np.interp` clamps at the top, so no bridge costs more than H."""
        return np.maximum(np.interp(latent_gap, self.LAT, self.FRM), 1.0)

    def _dist(self, z, ep):
        """Distance from z to every node, with the query's own episode removed (leave-one-out)."""
        return torch.cdist(z[None], self.bz).masked_fill(self.bep_t[None, :] == ep, float("inf"))[0]

    def prepare(self, ep, z_goal):
        """Insert z_goal and run ONE backward Dijkstra for this episode, with the episode removed."""
        if ep in self.tasks:
            return
        keepn = self.bep != ep
        remap = np.cumsum(keepn) - 1
        inv = np.nonzero(keepn)[0]
        sub = self.G[keepn][:, keepn]
        M = sub.shape[0]
        near = torch.topk(self._dist(z_goal, ep), self.k_edges, largest=False)
        gidx = near.indices.cpu().numpy()
        glat = near.values.cpu().numpy().astype(float)
        gcost = self.frames(glat).astype(np.float32)
        legal = glat <= self.cap
        if not legal.any():
            legal = np.zeros_like(legal)
            legal[0] = True
        gi = remap[gidx][legal]
        rows = np.concatenate([gi, np.full(len(gi), M)])
        cols = np.concatenate([np.full(len(gi), M), gi])
        vals = np.concatenate([gcost[legal], gcost[legal]]).astype(np.float32)
        sub = sp.vstack([sp.hstack([sub, sp.csr_matrix((M, 1))]), sp.csr_matrix((1, M + 1))]).tocsr()
        sub = sub + sp.csr_matrix((vals, (rows, cols)), shape=(M + 1, M + 1))
        dv, pred = dijkstra(sub.T.tocsr(), indices=M, directed=True, return_predecessors=True)
        self.tasks[ep] = dict(M=M, dv=dv[:M].astype(np.float32), pred=pred[:M].astype(np.int32),
                              remap=remap, inv=inv, z_goal=z_goal)

    def _walk(self, c, s, used0):
        """Follow the route from subgraph node `s` until H frames are consumed."""
        cur, used = s, float(used0)
        while True:
            nxt = int(c["pred"][cur])
            if nxt < 0 or nxt >= c["M"]:                    # no successor, or the next hop is z_goal
                break
            u, v = int(c["inv"][cur]), int(c["inv"][nxt])
            if self.bep[u] != self.bep[v]:
                step = float(self.frames(float(torch.linalg.vector_norm(self.bz[u] - self.bz[v]))))
            else:
                step = 1.0
            if used + step > self.H and cur != s:           # the first hop is always taken
                break
            used += step
            cur = nxt
        return int(c["inv"][cur])

    def subgoal(self, ep, z_obs):
        """Re-enter the graph at z_obs. Returns (sub-goal latent, reason)."""
        c = self.tasks[ep]
        near = torch.topk(self._dist(z_obs, ep), self.k_edges, largest=False)
        ci, cd = near.indices.cpu().numpy(), near.values.cpu().numpy()
        legal = cd <= self.cap
        if not legal.any():
            legal = np.zeros_like(legal)
            legal[0] = True
        cs = c["remap"][ci]
        tot = np.where(legal, self.frames(cd) + c["dv"][cs], np.inf)
        b = int(np.argmin(tot))
        s, total = int(cs[b]), float(tot[b])

        if not np.isfinite(total):
            # no route from any entry candidate: step to the candidate closest to the goal if that
            # is progress, otherwise aim at the goal directly
            gz = c["z_goal"]
            dg = torch.linalg.vector_norm(
                self.bz[torch.as_tensor(ci, device=self.dev)] - gz[None], dim=1).cpu().numpy()
            j = int(np.argmin(dg))
            if float(dg[j]) < float(torch.linalg.vector_norm(z_obs - gz)):
                return self.bz[int(ci[j])], "unreachable_steppingstone"
            return gz, "unreachable"
        if total <= self.H:                                 # goal within one waypoint budget
            return c["z_goal"], "within_H"
        node = self._walk(c, s, used0=float(self.frames(cd[b])))
        return self.bz[node], "route"
