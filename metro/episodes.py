# Adapted from LeWorldModel (LeWM); MIT License, see LICENSES/LeWM-LICENSE.
"""The evaluation episode set: `num_eval` (episode, start step) pairs drawn with `seed`."""
import numpy as np


def episode_column(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def sample_eval_episodes(dataset, num_eval, seed, goal_offset):
    """-> (rows, episodes, start_steps), sorted by dataset row."""
    col = episode_column(dataset)
    ep_all = np.asarray(dataset.get_col_data(col)).ravel()
    st_all = np.asarray(dataset.get_col_data("step_idx")).ravel()
    eps_u = np.unique(ep_all)
    lengths = np.array([st_all[ep_all == e].max() + 1 for e in eps_u])
    max_start = {e: lengths[i] - goal_offset - 1 for i, e in enumerate(eps_u)}
    valid = np.nonzero(st_all <= np.array([max_start[e] for e in ep_all]))[0]
    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    r = dataset.get_row_data(rows)
    return rows, np.asarray(r[col]).ravel(), np.asarray(r["step_idx"]).ravel()
