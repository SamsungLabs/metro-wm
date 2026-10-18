# Metro-WM: LeWM setup (CEM planner)

This setup compares Metro-WM with Hi-LeWM, Hi-LeWM-C, and Hi-LeWM-C (staged) on PushT, OGBench Cube, TwoRoom and Reacher using a CEM planner.

## 1. Installation

Tested with Python 3.10 and CUDA 12.1.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y pygame-ce pygame && pip install --no-deps pygame==2.6.1
```

## 2. Data and models

```bash
export STABLEWM_HOME=/path/to/storage
export MUJOCO_GL=egl
```

All datasets and models are downloaded directly from [LeWM](https://github.com/lucas-maes/le-wm).

### Datasets

From Hugging Face ([quentinll/lewm](https://huggingface.co/collections/quentinll/lewm)):

```bash
cd $STABLEWM_HOME
HF=https://huggingface.co/datasets/quentinll
curl -L -o pusht_expert_train.h5.zst $HF/lewm-pusht/resolve/main/pusht_expert_train.h5.zst
zstd -d --rm pusht_expert_train.h5.zst
curl -L $HF/lewm-tworooms/resolve/main/tworoom.tar.zst | tar --zstd -xf -
curl -L $HF/lewm-reacher/resolve/main/reacher.tar.zst | tar --zstd -xf -
mkdir -p ogbench && curl -L $HF/lewm-cube/resolve/main/cube_single_expert.tar.zst | tar --zstd -xf - -C ogbench
```

### LeWM

Download the released weights from Hugging Face and convert them to the object checkpoint format
used by stable-worldmodel.

```bash
for b in pusht cube tworoom reacher; do
  repo=lewm-$b; [ $b = tworoom ] && repo=lewm-tworooms
  hf download quentinll/$repo --local-dir checkpoints/$repo
  python scripts/convert_lewm.py --bench $b --hf_dir checkpoints/$repo
done
```

### PLDM (weights currently unavailable)

The PLDM checkpoints were released by the LeWM authors in a
[Google Drive folder](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e)
that is no longer available. Until they are re-hosted, please
contact the LeWM authors through the
[LeWM issue tracker](https://github.com/lucas-maes/le-wm/issues).

Our PLDM results use these four archives from that folder; check any copy against them:

| benchmark | file in the Drive folder | SHA-256 |
|---|---|---|
| PushT | `pusht/ckpt/pldm.tar.zst` | `05cb7863e8cdef2c80abd0790366a1a0b191345a3f24ad364d719016918ccec5` |
| Cube | `ogbench/cube/ckpt/pldm.tar.zst` | `24f66071f7370ffa63dd064f87021d3a57177298742b8fe3bd5f11bf4260e1ef` |
| TwoRoom | `two-room/ckpt/pldm.tar.zst` | `ed6d44b3957fb4964c1cffc3db783d744d3a35c6bb02b018b0cdc62b81f3ef3a` |
| Reacher | `reacher/ckpt/pldm.tar.zst` | `0daf036bcb287b64e8887651dc447cbe91e7fd448ac4f399739bc7d0401aa8d4` |


Resulting layout (the `pldm_object.ckpt` files only if you have the PLDM archives):

```
$STABLEWM_HOME/
├── pusht_expert_train.h5
├── tworoom.h5
├── reacher.h5
├── ogbench/cube_single_expert.h5
├── pusht/    lewm_object.ckpt  pldm_object.ckpt
├── cube/     lewm_object.ckpt  pldm_object.ckpt
├── tworoom/  lewm_object.ckpt  pldm_object.ckpt
└── reacher/  lewm_object.ckpt  pldm_object.ckpt
```

## 3. Building the graph

The graph is built offline, once per world model and benchmark.

```bash
# LeWM on all 4 benchmarks, one per GPU
./scripts/build_artifacts.sh
# if you have PLDM checkpoints, you can build both LeWM and PLDM on all benchmarks
WMS="lewm pldm" ./scripts/build_artifacts.sh
# script also works for a specific model and benchmark pair
./scripts/build_artifacts.sh lewm pusht
```

For each pair this runs three steps and writes to `artifacts/<wm>/<bench>/`:

1. `scripts/build_bank.py` encodes every frame of the dataset with the frozen encoder.
   - `bank_z.npy`: the latent of each frame (the graph's vertices).
   - `bank_ep.npy`, `bank_st.npy`: the episode and step each frame comes from.
2. `scripts/build_jump_table.py` finds, for every frame, its 4 nearest frames in other episodes.
   - `jump_idx.npy`, `jump_dist.npy`: their indices and latent distances, nearest first.
3. `scripts/calibrate_edge_cost.py` calibrates how a latent distance converts into frames.
   - `edge_cost.npz`: the latent distance between two frames used to price the bridge edges.

## 4. Running Metro-WM

`h` is the micro planner horizon, `H` the waypoint budget, `T` the episode budget and `d` the goal offset distance (see the paper).

```bash
# Metro-LeWM on PushT, d=75 (best result in the paper)
d=75
for rs in 7 8 9; do
  python eval.py --wm lewm --bench pusht --d $d --h 5 --H 28 --T $((4 * d)) --run_seed $rs
done
```

Each run appends a row to `results/results.jsonl` with the per-episode outcomes, the frame of first success and the wall clock.
To visualise the results, add `--save_video`.

For convenience, we provide a script `reproduce.sh` to reproduce all Metro-WM results:

```bash
# Metro-LeWM only for d=75, the highest-budget cell of each benchmark's frontier
./reproduce.sh
# Metro-LeWM and Metro-PLDM for d=75, the highest-budget cell of each benchmark's frontier
WMS="lewm pldm" ./reproduce.sh
# every (h,H) cell of Metro-LeWM's pareto frontier for all d, as shown in Figure 3 of the paper
# it reads off `configs/frontier.txt` from Appendix G.4 of the paper
FRONTIER=1 D="25 50 75" ./reproduce.sh
```

Note that each cell runs once at the maximum budget `T = 4d`, and the success rate at every smaller budget on the frontier is read off that run.
`reproduce.sh` prints these success rates when it finishes; to print them again from `results/results.jsonl`, run `python summarize.py` (or `python summarize.py --d 75` for d=75).

