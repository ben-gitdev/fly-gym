# fly-gym: FLYNN — Robust Neural Network for Robot Navigation using Fly Brain Topology

Code accompanying **"FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology"**
([arXiv:2607.00025](https://arxiv.org/abs/2607.00025); see [Citation](#citation)).

FLYNN is a recurrent neural network whose connectivity is derived directly from the FlyWire FAFB v783
*Drosophila* connectome (139,255 neurons, 5,342,445 synaptic connections). It's trained with DAgger
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
  descending neurons is read out through a small MLP into `[velocity, heading]`. The connectome weight
  matrix is rescaled to a target spectral radius at load time for stable recurrent dynamics, and a
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
                         (tune_direction_threshold.py)

shared_config.py         Shared paths/hyperparameters for the FLYNN/SmallWorldNet training and
                         eval scripts. Defaults to FLYNN; see the comment at the top to switch to
                         SmallWorldNet.
train_connectome_rnn_dagger.py   DAgger training for FLYNN / SmallWorldNet
train_visionnet_dagger.py        DAgger + camera-dropout training for the EfficientNet/MobileNet
                                  baselines
run_connectome_rnn_checkpoint.py Evaluation rollouts for a trained FLYNN/SmallWorldNet checkpoint
run_vision_agent_checkpoint.py   Evaluation rollouts for a trained EfficientNet/MobileNet checkpoint
test_vfhplus.py                  Sanity-check tool for the VFH*+PID teacher

collision_statistics.py  Aggregates raw per-episode eval data into the collision/success-rate/SPL/
                         speed statistics and bar charts reported in the paper
count_collisions.py      Per-episode collision/SPL/speed augmentation of raw eval rollout data
compare_trajectories.py  Trajectory-overlay comparison figures
visualize_episodes.py    Per-episode top-down trajectory plots (obstacles, path, start/end, target)
analysis_pca_statistics.py  KDE + vector-arithmetic (full-vision ≈ left-eye + right-eye) analysis
                             of FLYNN's internal hidden-state trajectories
analysis_pca.py          PCA visualization of hidden-state trajectories from a single eval run
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

- **`MobileNetAgent`** (`agents/mobilenet_agent.py`) — ImageNet-pretrained MobileNetV3-Large (first
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

Connectome data and trained checkpoints are excluded from version control due to size. The
connectome's small per-modality CSVs and the SmallWorldNet generator script are tracked under
`connectomes/`. Not tracked:

- `connectomes/drosophila adult connectome/connections_princeton.csv` (~261MB) — the real FAFB v783
  edge list (`pre_root_id`, `post_root_id`, `syn_count` or equivalent aliases), sourced from
  FlyWire.ai / the `philshiu/Drosophila_brain_model` project (see
  `connectomes/drosophila adult connectome/data source.txt` for provenance).
- `connectomes/ws_small_world/connections_ws_small_world.csv` (~169MB) — the synthetic SmallWorldNet
  control edge list. Regeneratable from a fixed seed via
  `connectomes/ws_small_world/generate_ws_network_new.py`.
- `connectomes/drosophila adult connectome/visual_column_L1_L2_L3_rear_view_{left,right}.csv`,
  `head_bristles_{left,right}.csv`, `descending_neurons.csv`, `consolidated_cell_types.csv`,
  `JO-C_and_JO-E.csv` — small per-modality neuron-ID CSVs (photoreceptor/lamina positions, tactile
  head bristles, descending/output neurons, cell types, Johnston's organ wind-sensing neurons) *are*
  tracked and load automatically once the large edge list above is in place.
- Trained checkpoints (`checkpoints/*.pt`).

<!-- TODO: host the files above (Zenodo/HuggingFace/institutional storage) and link them here, along
     with a small download script, so a fresh clone can reproduce Table I without a full retrain. -->

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

Both eval scripts write per-episode rollout data to `eval_data/<run_name>/`. Aggregate that into the
reported statistics and figures with:

```bash
python collision_statistics.py
python count_collisions.py
python compare_trajectories.py
python visualize_episodes.py
python analysis_pca_statistics.py
```

`analysis_pca.py` and `compare_trajectories.py` still have local data-path constants near the top of
each file that need to point at your own eval-data location before running them.

## License

[MIT](LICENSE).

## Citation

<!-- TODO: confirm the full author list before publishing -- neither this file nor arXiv:2607.00025's
     metadata gave a reliable full author list at the time this was written. -->

If you use this code, please cite the accompanying paper:

```bibtex
@misc{flynn2026,
  title         = {FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology},
  eprint        = {2607.00025},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.00025},
  year          = {2026}
}
```
