# FLYNN

Code accompanying **"FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology"**
(see [Citation](#citation)).

FLYNN is a recurrent neural network whose connectivity is derived directly from the FlyWire FAFB v783
*Drosophila* connectome (139,255 neurons, 5,342,445 synaptic connections). It's trained with DAgger
imitation learning to drive a two-wheeled, two-camera robot to a goal around randomly placed obstacles
in MuJoCo, and is compared against a synthetic Watts-Strogatz control network (SmallWorldNet) matched
on connectome degree/path-length statistics, plus two conventional CNN baselines (EfficientNet-B0,
MobileNetV3-Large).

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

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

`torch`/`torchvision` were tested with a CUDA 12.8 build; if you need GPU support, install the wheel
matching your own CUDA toolkit from https://pytorch.org/get-started/locally/ rather than relying on
the plain PyPI wheel.

## Data and checkpoints

The connectome's small per-modality CSVs (neuron IDs for photoreceptors, wind-sensing, descending
neurons, cell types, etc.) and the SmallWorldNet generator script are tracked in this repo under
`connectomes/`. The large raw edge lists are not:

- `connectomes/drosophila adult connectome/connections_princeton.csv` (~261MB) — the real FAFB v783
  connectome edge list, sourced from FlyWire.ai / the `philshiu/Drosophila_brain_model` project (see
  `connectomes/drosophila adult connectome/data source.txt` for provenance).
- `connectomes/ws_small_world/connections_ws_small_world.csv` (~169MB) — the synthetic SmallWorldNet
  control edge list. Regeneratable from a fixed seed via
  `connectomes/ws_small_world/generate_ws_network_new.py`.
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

## License

[MIT](LICENSE).

## Citation

<!-- TODO: confirm the full author list and venue/year below before publishing. -->

```bibtex
@inproceedings{flynn2026,
  title     = {FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology},
  author    = {Wang, Benquan},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
