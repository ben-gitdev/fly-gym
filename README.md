# fly-gym: FLYNN — Robust Neural Network for Robot Navigation using Fly Brain Topology

Code accompanying **"FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology"**
([arXiv:2607.00025](https://arxiv.org/abs/2607.00025); see [Citation](#citation)).

**Play it in your browser:** [race FLYNN](https://ben-gitdev.github.io/fly-gym/), driving the same
robot through the same arena with only what the network sees (see [Race the agent](#race-the-agent)).

FLYNN is a recurrent neural network whose connectivity is derived directly from the FlyWire FAFB v783
*Drosophila* connectome shipped with this repo. It's trained with DAgger
imitation learning to drive a two-wheeled, two-camera robot to a goal around randomly placed obstacles
in a MuJoCo arena, and is benchmarked against a synthetic Watts-Strogatz control network
("SmallWorldNet", matched on the connectome's degree/path-length statistics) and two conventional CNN
baselines (EfficientNet-B0, MobileNetV3-Large) to ask how much of its navigation ability is
attributable to the real fly wiring diagram, as opposed to network size, topology, or a conventional
vision pipeline.

## Overview

- **Body / world:** a two-wheeled differential-drive robot with two forward-facing cameras, simulated
  in [MuJoCo](https://mujoco.org/), navigating toward a goal in an arena scattered with cylindrical
  obstacles. Checkerboard (training/in-distribution) and photo-realistic (out-of-distribution eval)
  scene variants are selected via `MuJoCoTwoCamEnv(texture_mode=...)`.
- **Brain:** every unit in the RNN corresponds to one neuron in the fly connectome, and every recurrent
  weight corresponds to a synapse-count-weighted connection from the connectome edge list. Camera
  pixels are sampled at the real photoreceptor-column positions and passed through a virtual-retina
  model (log transform + high-pass L1/L2 / low-pass L3 filtering) before entering the RNN,
  approximating the fly's early visual processing (photoreceptors -> lamina L1-L3); a fixed set of
  descending neurons is read out through a small MLP into `[velocity, heading]`. A
  custom `MemoryEfficientSparseMM` autograd function (CSR forward pass, cached-COO O(NNZ) backward
  pass) makes training a ~140k-neuron recurrent cell tractable on a single GPU.
- **Training:** DAgger imitation learning against an analytic **VFH\*** (Vector Field Histogram + A*)
  planner + PID teacher.
- **Comparisons:** the same task/training pipeline trains the Watts-Strogatz "SmallWorldNet" control
  (isolating topology from the real wiring diagram) and the EfficientNet-B0/MobileNetV3-Large CNN
  baselines, so navigation performance, robustness, and internal dynamics (via PCA of hidden-state
  trajectories) can be compared across architectures.

## Repository structure

```
core/                    Shared utilities: connectome/edge-list loading + cell construction
                         (utils.py), A* global path planning (astar.py), the VFH*+PID teacher
                         algorithm (vfhplus.py)
environment/             MuJoCo environment: two-wheeled robot, two cameras, randomized
                         obstacles/goal, checkerboard (training) vs. photo-realistic (OOD-eval)
                         scene variants selected via MuJoCoTwoCamEnv(texture_mode=...)
agents/                  Per-architecture agent wrappers: FLYNN/SmallWorldNet sensory front-end
                         (connectome_rnn_agent.py), EfficientNet/MobileNet CNN baselines, and the
                         VFH*+PID DAgger teacher (teacher_analytic_agent.py)
models/                  FLYNN's core RNN cell with a custom memory-efficient sparse-autograd
                         backward pass (connectome_rnn_model.py), and the teacher's path-follow
                         controller
connectomes/             Connectome data. Small per-modality neuron-ID CSVs and the SmallWorldNet
                         generator script are tracked; the large raw edge lists (hundreds of MB)
                         are not -- see Data below.
tests/                   Dev/diagnostic tools: gradient-correctness sanity check for the custom
                         sparse autograd (debug_rnn_grad.py), turn-vs-straight threshold tuning
                         (tune_direction_threshold.py), VFH*+PID teacher sanity check
                         (test_vfhplus.py)
tools/                   Post-hoc analysis/aggregation scripts that consume eval_data/ rollout
                         output: collision_statistics.py, count_collisions.py,
                         compare_trajectories.py, visualize_episodes.py,
                         analysis_pca_statistics.py, analysis_pca.py
results/                 Result figures/stats produced by tools/collision_statistics.py and
                         tests/tune_direction_threshold.py: Table I bar charts, SPL distribution
                         plots, and their aggregate CSVs

shared_config.py         Shared paths/hyperparameters for the FLYNN/SmallWorldNet training and
                         eval scripts. Defaults to FLYNN; see the comment at the top to switch to
                         SmallWorldNet.
train_connectome_rnn_dagger.py   DAgger training for FLYNN / SmallWorldNet
train_visionnet_dagger.py        DAgger + camera-dropout training for the EfficientNet/MobileNet
                                  baselines
run_connectome_rnn_checkpoint.py Evaluation rollouts for a trained FLYNN/SmallWorldNet checkpoint
run_vision_agent_checkpoint.py   Evaluation rollouts for a trained EfficientNet/MobileNet checkpoint
play_vs_connectome_rnn.py        Demo game: race a trained FLYNN/SmallWorldNet checkpoint, driving
                                 the same robot with the arrow keys from the agent's own inputs
export_web_game.py               Records the agent's runs and the MuJoCo calibration for the
                                 browser version of the game in docs/ (static site, GitHub Pages)
```

## Models

### FLYNN (`agents/connectome_rnn_agent.py`, `models/connectome_rnn_model.py`)

Each unit is one neuron from the connectome edge list; its state follows a leaky recurrent update
`h_new = (1 - alpha_type) * h + alpha_type * phi(W_sparse @ h + b)`, where `alpha_type` is a learnable
per-cell-type leak rate and `phi` is `tanh`/`relu`. Sensory input: camera pixels through the virtual
retina into visual-column neurons, tactile head-bristle neurons driven by collision/contact angle, and
wind-direction input routed through a small MLP into the Johnston's organ neurons. Cell types
parameterize per-type leak rates and let training scripts selectively freeze/train weights, biases, or
per-type alphas.

### SmallWorldNet control

The same `LeakyConnectomeRNNCell` architecture, but wired with a Watts-Strogatz small-world graph
(`connections_ws_small_world.csv`, matched on the real connectome's node/edge/degree statistics)
instead of the real connectome — isolating whether small-world topology alone (a property the fly
connectome is known to have) explains task performance, independent of the real wiring diagram. Select
it via `BASE_PATH`/`EDGE_PATH` in `shared_config.py`.

### Vision-CNN baselines (non-connectome)

- **`MobileNetAgent`** (`agents/mobilenet_agent.py`) — MobileNetV3-Large (first
  conv adapted to 1-channel grayscale, classifier removed) feeding a `GRUCell`, plus wind-direction and
  collision scalars, into a small MLP policy head.
- **`EfficientNetAgent`** (`agents/efficientnet_agent.py`) — identical scheme with an EfficientNet-B0
  backbone.

### Analytic teacher (`agents/teacher_analytic_agent.py`, `core/vfhplus.py`)

`PlannerAnalyticTeacher` is not a trained network but a classical controller used to generate expert
demonstrations for DAgger: a VFH\* planner (polar obstacle histogram + short-horizon A\* search with
heading-consistency costs) produces a local waypoint, a PID line-follower steers toward it, and a
scripted back-up/turn/forward recovery sequence handles collisions.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

`torch`/`torchvision` were tested with a CUDA 12.8 build; if you need GPU support, install the wheel
matching your own CUDA toolkit from https://pytorch.org/get-started/locally/ rather than relying on
the plain PyPI wheel.

## Data and checkpoints

Connectome data and trained checkpoints are hosted
on Hugging Face:
**[benquan1/fly-gym-trained-policies](https://huggingface.co/datasets/benquan1/fly-gym-trained-policies)**
(399MB total). **Licensing there is mixed.** `drosophila adult connectome.7z` is a
CSV export of the real FlyWire connectome, which FlyWire licenses CC BY-NC 4.0
(Attribution-NonCommercial) — see `connectomes/drosophila adult connectome/data source.txt` for the
full attribution notice; that license is not superseded by this repo's own MIT `LICENSE` file. The same
CC BY-NC 4.0 terms also apply to `connectome_rnn_dagger_princeton_full_vision.pt` (the FLYNN
checkpoint): FLYNN's recurrent weight matrix's
sparsity structure *is* the connectome edge list, scaled — so the checkpoint directly incorporates
FlyWire data. The other 3 checkpoints and the synthetic
SmallWorldNet connectome involve no FlyWire data at all and are this project's own work, MIT-licensed
same as the rest of this repo.

```bash
git clone https://huggingface.co/datasets/benquan1/fly-gym-trained-policies
```

What's in it and where each file goes:

- `drosophila adult connectome.7z` (45.1MB) — extract into
  `connectomes/drosophila adult connectome/` to get `connections_princeton.csv` (~261MB), the real
  FAFB v783 edge list (`pre_root_id`, `post_root_id`, `syn_count` or equivalent aliases), sourced from
  FlyWire.ai / the `philshiu/Drosophila_brain_model` project (see
  `connectomes/drosophila adult connectome/data source.txt` for provenance).
- `ws_small_world_connectome.7z` (53.5MB) — extract into `connectomes/ws_small_world/` to get
  `connections_ws_small_world.csv` (~169MB), the synthetic SmallWorldNet control edge list.
  Regeneratable from a fixed seed instead via `connectomes/ws_small_world/generate_ws_network_new.py`.
- `connectome_rnn_dagger_princeton_full_vision.pt` (108MB) — FLYNN, trained with full vision.
  **CC BY-NC 4.0** (see above), not MIT.
- `connectome_rnn_dagger_small_world_full_vision.pt` (155MB) — SmallWorldNet control, trained with
  full vision. MIT.
- `efficientnet_dagger_final_robust.pt` (21.2MB) / `mobilenet_dagger_final_robust.pt` (16MB) — the
  EfficientNet-B0 / MobileNetV3-Large baselines, trained with camera dropout. MIT.

Each checkpoint above is trained once, on full vision; `run_connectome_rnn_checkpoint.py` and
`run_vision_agent_checkpoint.py` reproduce all 4 vision-ablation conditions (full / right-eye-only /
left-eye-only / blind) from that single checkpoint by masking the input at eval time. 
Drop checkpoints into `checkpoints/` (create the folder if it doesn't already exist)
and pass the path straight to the eval scripts, e.g.:

```bash
python run_connectome_rnn_checkpoint.py checkpoints/connectome_rnn_dagger_princeton_full_vision.pt
```


## Usage

Train FLYNN or SmallWorldNet (edit the `BASE_PATH` toggle at the top of `shared_config.py` to choose
which):

```bash
python train_connectome_rnn_dagger.py
```

Train an EfficientNet/MobileNet baseline (edit `AGENT` at the top of the file):

```bash
python train_visionnet_dagger.py
```

Evaluate a trained checkpoint (checkpoint path is a required argument for both):

```bash
python run_connectome_rnn_checkpoint.py checkpoints/<your_checkpoint>.pt
python run_vision_agent_checkpoint.py checkpoints/<your_checkpoint>.pt
```

By default each script runs all 4 vision conditions in turn. Use `--vision` to pick one or more of
them. Each code is two digits, `<left eye><right eye>`, where `1` = eye enabled and `0` = eye blind:

| `--vision` | Condition |
|---|---|
| `11` | Full vision |
| `10` | Left eye only |
| `01` | Right eye only |
| `00` | Blind |

```bash
python run_connectome_rnn_checkpoint.py checkpoints/<your_checkpoint>.pt --vision 10
python run_vision_agent_checkpoint.py checkpoints/<your_checkpoint>.pt --vision 11 00
```

Each condition writes its results to its own folder under `eval_data/`. Run either script with
`--help` to see all options.

Both scripts default to the checkerboard (in-distribution) scene. Pass `--texture-mode realistic` to
evaluate against the photo-realistic (out-of-distribution) textures instead:

```bash
python run_connectome_rnn_checkpoint.py checkpoints/<your_checkpoint>.pt --texture-mode realistic
python run_vision_agent_checkpoint.py checkpoints/<your_checkpoint>.pt --texture-mode realistic
```

### Race the agent

`play_vs_connectome_rnn.py` is a small demo game. You drive the robot with the arrow keys in a copy
of the eval arena that has the same layout, start pose and goal as the agent's. You see only what
the agent sees: its photoreceptor-sampled eye input, an arrow for the goal direction (its "wind"
input) and left/right bump lamps (its head-bristle input). Your robot moves at the agent's cruise
speed (velocity command 0.7, about 1.25 m/s). The checkpoint races alongside, rolled out as in
the eval script.

```bash
python play_vs_connectome_rnn.py checkpoints/connectome_rnn_dagger_princeton_full_vision.pt \
    --connectome "connectomes/drosophila adult connectome/connections_princeton.csv"
```

Up/Down drive forward/backward, Left/Right turn, SPACE starts a round, P pauses, R retries the
layout, N skips to a new one, and ESC quits. The results screen shows both paths on a top-down
map. `--vision` (same codes as above) blinds you and the agent equally. Times are in simulated
seconds, so the race stays fair if your machine can't run it at full real-time speed; pass
`--fps 30` for an even pace in that case.

#### Web version

**Play it at https://ben-gitdev.github.io/fly-gym/.**

To try the web version locally, serve the folder 

```bash
python -m http.server 8000 --directory docs
```

and open
http://localhost:8000:

The live site is served by GitHub Pages from `main`, folder `/docs`.

## License

[MIT](LICENSE).

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{flynn2026,
  title         = {FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology},
  author        = {Wang, Benquan and Chen, Jingdao},
  eprint        = {2607.00025},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.00025},
  year          = {2026}
}
```
