# Adapted from stable-worldmodel 0.0.5; MIT License, see LICENSES/stable-worldmodel-LICENSE.
"""Metro policy: a frozen micro planner (CEM over the world model) chasing graph sub-goals.

Every `replan_interval` frames the current observation and the goal are encoded, the graph is
re-entered and a new sub-goal latent is chosen. The micro planner solves for `h` action blocks
toward that latent, executes them all, and solves again until the next macro replan.
"""
from collections import deque

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy


class MetroPolicy(BasePolicy):
    def __init__(self, micro_solver, encoder, graph, micro_config, episodes, replan_interval,
                 process=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "metro"
        self.micro_solver = micro_solver
        self.encoder = encoder
        self.graph = graph
        self.micro_cfg = micro_config
        self.episodes = np.asarray(episodes).ravel()          # env i plays episodes[i]
        self.replan_interval = int(replan_interval)
        self.process = process or {}
        self.transform = transform or {}
        self.step_counter = 0
        self.current_subgoal = None
        self._z_goal = None
        self._action_buffer = None
        self._next_micro_init = None
        self.reasons = []

    def set_env(self, env):
        self.env = env
        n_envs = getattr(env, "num_envs", 1)
        assert n_envs == len(self.episodes), f"{n_envs} envs but {len(self.episodes)} episodes"
        self.micro_solver.configure(action_space=env.action_space, n_envs=n_envs, config=self.micro_cfg)
        self._action_buffer = deque(maxlen=self.micro_cfg.receding_horizon * self.micro_cfg.action_block)

    @torch.no_grad()
    def _encode(self, pixels):
        dev = next(self.encoder.parameters()).device
        return self.encoder.encode({"pixels": pixels.to(dev)})["emb"]

    @torch.no_grad()
    def _replan_subgoal(self, info_dict):
        if self._z_goal is None:
            self._z_goal = self._encode(info_dict["goal"])[:, -1].float()
        z_obs = self._encode(info_dict["pixels"])[:, -1].float()
        subs = []
        for i, ep in enumerate(self.episodes):
            ep = int(ep)
            self.graph.prepare(ep, self._z_goal[i].to(self.graph.dev))
            z, reason = self.graph.subgoal(ep, z_obs[i].to(self.graph.dev))
            subs.append(z)
            self.reasons.append(reason)
        dev = next(self.encoder.parameters()).device
        self.current_subgoal = torch.stack(subs).unsqueeze(1).to(dev)

    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        if self.step_counter % self.replan_interval == 0 or self.current_subgoal is None:
            self._replan_subgoal(info_dict)
            self._action_buffer.clear()
        info_dict["goal_emb"] = self.current_subgoal

        if len(self._action_buffer) == 0:
            outputs = self.micro_solver(info_dict, init_action=self._next_micro_init)
            best = outputs["actions"]
            keep = self.micro_cfg.receding_horizon
            self._next_micro_init = best[:, keep:] if self.micro_cfg.warm_start else None
            plan = best[:, :keep].reshape(self.env.num_envs, keep * self.micro_cfg.action_block, -1)
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft().reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        self.step_counter += 1
        return action
