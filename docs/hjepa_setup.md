# Metro-WM: H-JEPA setup (gradient planner)

This setup runs Metro-WM inside the evaluation of H-JEPA and compares Metro-WM with H-JEPA and HWM on OGBench Cube, PushT, Visual AntMaze and FourRoom Distractors using a gradient planner.

## 1. Installation

Metro-WM runs inside H-JEPA's evaluation, so this setup needs two environments.
Metro-WM's environment, tested with Python 3.10 and CUDA 12.1:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y pygame-ce pygame && pip install --no-deps pygame==2.6.1
```

H-JEPA's environment, following §1 of the [H-JEPA README](https://github.com/kevinghst/H-JEPA#1-installation) (tested at commit `97054ec`).
Then point the scripts at both:

```bash
export HJEPA_PY=/path/to/envs/hjepa/bin/python      # H-JEPA's environment
export METRO_PY=/path/to/metro-wm/.venv/bin/python  # Metro-WM's environment
```

## 2. Data and models

```bash
export HJEPA_ROOT=/path/to/H-JEPA HJEPA_HOME=/path/to/hjepa/data
export STABLEWM_HOME=/path/to/storage
export MUJOCO_GL=egl
```

All tasks and models are downloaded directly from [H-JEPA](https://github.com/kevinghst/H-JEPA), following its README.

### Datasets

- AntMaze and FourRoom: the training sets `visual_antmaze_medium_explore_stitch_train.h5` and `fourroom_tp35_d1.h5` (H-JEPA README §3.1).
- The evaluation tasks (H-JEPA README §3.2).
- PushT and Cube: H-JEPA's LeWMs are trained on LeWM's datasets, downloaded from Hugging Face ([quentinll/lewm](https://huggingface.co/collections/quentinll/lewm)):

```bash
cd $STABLEWM_HOME
HF=https://huggingface.co/datasets/quentinll
curl -L -o pusht_expert_train.h5.zst $HF/lewm-pusht/resolve/main/pusht_expert_train.h5.zst
zstd -d --rm pusht_expert_train.h5.zst
mkdir -p ogbench && curl -L $HF/lewm-cube/resolve/main/cube_single_expert.tar.zst | tar --zstd -xf - -C ogbench
```

### H-JEPA's LeWM

Note that these weights differ from the original LeWM weights. Download H-JEPA's flat LeWM weights:

```bash
hf download jepa-world-models/h-jepa --local-dir $HJEPA_HOME/ckpts --include "*_lewm/*"
```

Resulting layout:

```
$HJEPA_HOME/
├── visual_antmaze_medium_explore_stitch_train.h5
├── fourroom_tp35_d1.h5
├── pusht_expert_val.h5
├── cube_single_expert_val.h5
└── ckpts/<bench>/<bench>_lewm/seed<seed>/<bench>_lewm_object.ckpt
$STABLEWM_HOME/
├── pusht_expert_train.h5
└── ogbench/cube_single_expert.h5
```

## 3. Building the graph

The graph is built offline, once per benchmark and model seed, from the world model's training data only.

```bash
# all 4 benchmarks x seeds 42/43/44, one job per GPU
./hjepa/build_graphs.sh
# script also works for a specific benchmark and seed
./hjepa/build_graphs.sh pusht 42
```

For each pair this runs three steps and writes to `artifacts/hj_lewm_s<seed>/<bench>/`:

1. The frozen encoder encodes every frame of the training data.
   - PushT and Cube: `scripts/convert_hjepa.py` ports H-JEPA's checkpoint to Metro-WM's LeWM class and checks that its outputs match, then `scripts/build_bank.py` encodes the training data. For Cube, `--max_episode 9800` keeps H-JEPA's training split.
   - AntMaze and FourRoom: `hjepa/build_bank.py` encodes the training data with H-JEPA's own model, since AntMaze's LeWM also reads proprioception.
   - `bank_z.npy`: the latent of each frame (the graph's vertices).
   - `bank_ep.npy`, `bank_st.npy`: the episode and step each frame comes from.
2. `scripts/build_jump_table.py` finds, for every frame, its 4 nearest frames in other episodes.
   - `jump_idx.npy`, `jump_dist.npy`: their indices and latent distances, nearest first.
3. `scripts/calibrate_edge_cost.py` calibrates how a latent distance converts into frames.
   - `edge_cost.npz`: the latent distance between two frames used to price the bridge edges.

## 4. Running Metro-WM

`h` is the micro planner horizon in action blocks of 5 frames and `H` the waypoint budget in frames; `(h, H)` was chosen on 50 tuning tasks disjoint from the test tasks.

```bash
# Metro-LeWM on PushT, seed 42, run from H-JEPA's clone
cd $HJEPA_ROOT/h_jepa
$HJEPA_PY /path/to/metro-wm/hjepa/metro_in_hjepa.py --config-name pusht_flat seed=42 \
  policy=$HJEPA_HOME/ckpts/pusht/pusht_lewm/seed42/pusht_lewm_object.ckpt output.dir=metro_h3_H34 \
  plan_config.horizon=3 +metro.artifacts=/path/to/metro-wm/artifacts/hj_lewm_s42/pusht +metro.H=34
```

Each run is H-JEPA's own evaluation, so its results land next to the checkpoint, in `$HJEPA_HOME/ckpts/<bench>/<bench>_lewm/seed<seed>/metro_h<h>_H<H>/metrics.yaml`.

For convenience, we provide a script `hjepa/reproduce.sh` to reproduce all Metro-WM results in this setup:

```bash
# all 4 benchmarks x seeds 42/43/44, one run per GPU
./hjepa/reproduce.sh
# only these benchmarks
./hjepa/reproduce.sh cube pusht
```

`hjepa/reproduce.sh` prints the success rates when it finishes; to print them again, run `python hjepa/summarize.py`.
