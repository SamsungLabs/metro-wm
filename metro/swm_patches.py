# Adapted from stable-worldmodel 0.0.5; MIT License, see LICENSES/stable-worldmodel-LICENSE.
"""Runtime overrides of stable-worldmodel 0.0.5, applied by `apply()` (replaces the source patch).

    import stable_worldmodel as swm
    from metro import swm_patches
    swm_patches.apply()

1. Reacher gets a `qpos_match` task: success (and `terminated`) once every joint is within
   0.05 rad of the goal qpos, which the evaluation sets through `set_target_qpos`.
2. `DMControlWrapper.step` ends an episode when the subclass's `_is_terminated` says so.
3. `World.evaluate_from_dataset` also returns `first_success_step` (the loop index at which each
   episode first succeeds, -1 if never), times video encoding separately as
   `world.last_video_seconds`, and writes 4-panel videos
   (initial | rollout | goal | ground truth).

The overrides lean on 0.0.5 internals, so `apply()` refuses any other version.
"""
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

SWM_VERSION = "0.0.5"
QPOS_THRESHOLD = 0.05
_applied = False


def apply():
    """Install the overrides once. A stable-worldmodel that already carries them is left alone."""
    global _applied
    if _applied:
        return
    got = version("stable-worldmodel")
    assert got == SWM_VERSION, f"metro.swm_patches targets stable-worldmodel {SWM_VERSION}, found {got}"
    from stable_worldmodel.envs.dmcontrol import dmcontrol
    if not hasattr(dmcontrol.DMControlWrapper, "_is_terminated"):     # not the source-patched tree
        _patch_dmcontrol_step()
        _patch_reacher()
        _patch_world()
    _applied = True


# ------------------------------------------------------------------------------ DMControl step

def _patch_dmcontrol_step():
    from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper

    def _is_terminated(self, step):
        """True if `step` (a dm_control TimeStep) ends the episode as a success. Base: never."""
        return False

    def step(self, action):
        reward = 0
        action = action.astype(self.action_spec_dtype)
        terminated = False
        for _ in range(self.action_repeat):
            step = self.env.step(action)
            reward += step.reward or 0
            if self._is_terminated(step):
                terminated = True
                break
        self._cumulative_reward += reward
        return self._obs_to_array(step.observation), reward, terminated, False, self.info

    DMControlWrapper._is_terminated = _is_terminated
    DMControlWrapper.step = step


# ------------------------------------------------------------------------------ Reacher qpos_match

def _qpos_match_task_class():
    from dm_control.suite import reacher

    class ReacherQPosMatchTask(reacher.Reacher):
        """Reacher that succeeds when every joint is within `qpos_threshold` of `target_qpos`.

        The dm_control episode itself only ends on a match if `terminate_episode=True`.
        """

        def __init__(self, target_size, random=None, qpos_threshold=QPOS_THRESHOLD,
                     terminate_episode=False):
            super().__init__(target_size=target_size, random=random)
            self.target_qpos = None
            self.qpos_threshold = qpos_threshold
            self.terminate_episode = terminate_episode

        def is_qpos_matched(self, physics):
            if self.target_qpos is None:
                return False
            return bool(np.all(np.abs(physics.data.qpos - self.target_qpos) < self.qpos_threshold))

        def get_termination(self, physics):
            if not self.terminate_episode:
                return None
            return 0.0 if self.is_qpos_matched(physics) else None

    return ReacherQPosMatchTask


def _patch_reacher():
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import reacher
    from dm_control.suite.wrappers import action_scale
    from stable_worldmodel.envs.dmcontrol import reacher as swm_reacher
    from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper

    W = swm_reacher.ReacherDMControlWrapper
    task_cls = _qpos_match_task_class()
    # __init__ validates `task` against these module globals
    swm_reacher._TASKS = (*swm_reacher._TASKS, "qpos_match")
    swm_reacher._TASK_TARGET_SIZES["qpos_match"] = swm_reacher._SMALL_TARGET

    orig_init = W.__init__

    def __init__(self, task="hard", seed=None, environment_kwargs=None):
        self._task_name = task                    # compile_model, called from __init__, reads it
        orig_init(self, task=task, seed=seed, environment_kwargs=environment_kwargs)

    def compile_model(self, seed=None, environment_kwargs=None):
        """Upstream `compile_model`, except that `qpos_match` builds a ReacherQPosMatchTask."""
        assert self._mjcf_model is not None, "No MJCF model to compile!"
        self._mjcf_tempdir = tempfile.TemporaryDirectory()
        mjcf.export_with_assets(self._mjcf_model, self._mjcf_tempdir.name, out_file_name="reacher.xml")
        physics = reacher.Physics.from_xml_path(os.path.join(self._mjcf_tempdir.name, "reacher.xml"))
        if self._task_name == "qpos_match":
            task = task_cls(target_size=self._target_size, random=seed)
        else:
            task = reacher.Reacher(target_size=self._target_size, random=seed)
        env = control.Environment(physics, task, time_limit=swm_reacher._DEFAULT_TIME_LIMIT,
                                  **(environment_kwargs or {}))
        self.env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        self._dirty = False

    def set_target_qpos(self, target_qpos):
        assert self._task_name == "qpos_match", "set_target_qpos() is only valid for the 'qpos_match' task."
        self.env.task.target_qpos = np.asarray(target_qpos, dtype=np.float64)

    def reset(self, seed=None, options=None):
        obs, info = DMControlWrapper.reset(self, seed=seed, options=options)
        if self._task_name == "qpos_match" and options is not None:
            target_qpos = options.get("target_qpos")
            if target_qpos is not None:
                self.env.task.target_qpos = np.asarray(target_qpos, dtype=np.float64)
        return obs, info

    def _is_terminated(self, step):
        return self._task_name == "qpos_match" and self.env.task.is_qpos_matched(self.env.physics)

    W.__init__ = __init__
    W.compile_model = compile_model
    W.set_target_qpos = set_target_qpos
    W.reset = reset
    W._is_terminated = _is_terminated


# ------------------------------------------------------------------------------ World evaluation

def _patch_world():
    from stable_worldmodel.world import World

    orig = World.evaluate_from_dataset

    def evaluate_from_dataset(self, dataset, episodes_idx, start_steps, goal_offset_steps,
                              eval_budget, callables=None, save_video=True, video_path="./"):
        """Upstream evaluation, plus `first_success_step`, `last_video_seconds`, 4-panel videos.

        Upstream calls `self.step()` once per frame of the budget; a per-instance hook on it
        records the first success of each episode (and the frames, for the video) without
        copying the loop. The video is then written here, outside the timed evaluation.
        """
        first = np.full(len(episodes_idx), -1, dtype=np.int64)
        frames = [] if save_video else None
        n_calls = 0
        world_step = self.step

        def step():
            nonlocal n_calls
            if frames is not None:
                frames.append(np.array(self.infos["pixels"][:, -1]))
            world_step()
            first[np.logical_and(self.terminateds, first < 0)] = n_calls
            n_calls += 1

        self.step = step
        try:
            results = orig(self, dataset, episodes_idx, start_steps, goal_offset_steps, eval_budget,
                           callables=callables, save_video=False, video_path=video_path)
        finally:
            del self.step
        assert n_calls == eval_budget, f"expected {eval_budget} World.step calls, saw {n_calls}"
        results["first_success_step"] = first

        t0 = time.time()
        if save_video:
            frames[-1] = np.array(self.infos["pixels"][:, -1])     # upstream: last frame is final
            _write_videos(self, dataset, episodes_idx, start_steps, goal_offset_steps,
                          np.stack(frames, axis=1), video_path)
        self.last_video_seconds = time.time() - t0
        return results

    World.evaluate_from_dataset = evaluate_from_dataset


def _write_videos(world, dataset, episodes_idx, start_steps, goal_offset_steps, video_frames,
                  video_path):
    """One mp4 per env, panels left to right: initial | rollout | goal | ground-truth episode."""
    import imageio

    starts = np.asarray(start_steps)
    data = dataset.load_chunk(np.asarray(episodes_idx), starts, starts + goal_offset_steps)
    target_frames = torch.stack([ep["pixels"].permute(0, 2, 3, 1) for ep in data]).numpy()
    target_len = target_frames.shape[1]
    out_dir = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(world.num_envs):
        # faststart puts the moov atom first so the file streams / previews everywhere
        out = imageio.get_writer(out_dir / f"rollout_{i}.mp4", fps=15, codec="libx264",
                                 output_params=["-movflags", "+faststart"])
        for t in range(video_frames.shape[1]):
            out.append_data(np.hstack([video_frames[i, 0], video_frames[i, t],
                                       target_frames[i, -1], target_frames[i, t % target_len]]))
        out.close()
    print(f"Video saved to {out_dir}")
