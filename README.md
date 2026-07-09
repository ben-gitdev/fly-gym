# fly-gym: A Connectome-Constrained RNN for Visual Navigation

This repository trains an artificial recurrent neural network whose connectivity is directly constrained by the **_Drosophila_ FAFB/FlyWire brain connectome** (Princeton FlyWire, `flywire_fafb_v783`) to perform vision-guided goal-navigation and obstacle avoidance in a simulated 3D arena. The connectome-derived network is trained with imitation learning and reinforcement learning and is benchmarked against several non-biological baselines to ask how much of the network's navigation ability is attributable to the real fly wiring diagram, as opposed to network size, topology, or a conventional CNN vision pipeline.

This code accompanies the paper: **[arXiv:2607.00025](https://arxiv.org/abs/2607.00025)**.

## Overview

- **Body / world:** a two-wheeled differential-drive robot with two forward-facing cameras, simulated in [MuJoCo](https://mujoco.org/), navigating toward a goal in an arena scattered with cylindrical obstacles.
- **Brain:** every unit in the RNN corresponds to one neuron in the fly connectome, and every recurrent weight corresponds to a synapse-count-weighted connection from the connectome edge list. Camera images are converted into photoreceptor-like input via a virtual retina model (log transform + high-pass L1/L2 and low-pass L3 filtering) and injected into the correct visual-column neurons; the network's descending neurons are read out into a velocity/steering command.
- **Training:** the connectome RNN (and the baseline networks) are first trained with **DAgger** imitation learning against an analytic **VFH\*** (Vector Field Histogram + A\*) planner + PID teacher, and can optionally be fine-tuned further with **on-policy PPO** reinforcement learning.
- **Comparisons:** the same task and training pipeline are used to train several non-connectome baselines (see [Models](#models)) so that navigation performance, robustness, and internal dynamics (via PCA of hidden-state trajectories) can be compared across architectures.

## Repository Structure

```
fly-gym/
├── agents/                          # Policy wrappers (observation -> action)
│   ├── connectome_rnn_agent.py      # ConnectomeAgent: virtual retina + connectome RNN + readout/value heads
│   ├── mobilenet_agent.py           # MobileNetV3-Large encoder + GRU baseline
│   ├── efficientnet_agent.py        # EfficientNet-B0 encoder + GRU baseline
│   ├── dual_backbone_agent.py       # Two independent CNN backbones (one per eye) + GRU
│   └── teacher_analytic_agent.py    # PlannerAnalyticTeacher: VFH* planner + PID + collision recovery
├── models/
│   ├── connectome_rnn_model.py      # LeakyConnectomeRNNCell + memory-efficient sparse-matmul autograd
│   ├── connectome_rnn_model_no custom autograd.py  # Reference cell using plain torch.sparse ops (for gradient debugging)
│   └── teacher_analytic_model.py    # PID line-follower controller used by the teacher
├── core/
│   ├── utils.py                     # Connectome/edge-list loading, sparse matrix + spectral-radius utilities
│   ├── vfhplus.py                   # VFH+ polar-histogram local planner with A* lookahead
│   └── astar.py                     # Grid A* planner (used for distance-to-goal and collision counting)
├── environment/
│   ├── mujoco_two_cam_env_random_obstacles.py  # Gymnasium env: two-camera robot, obstacles, reward shaping
│   └── mujoco_model_random_obstacles.xml       # MuJoCo scene (robot, walls, cylinder obstacles, goal)
├── scripts/
│   └── create_random_connectome.py  # Builds a degree-preserving randomized-edge control connectome
├── shared_config.py                 # Central paths/hyperparameters shared by the training scripts
├── train_connectome_rnn_dagger.py   # DAgger imitation learning for the connectome RNN
├── train_connectome_rnn_rl.py       # PPO fine-tuning for the connectome RNN
├── train_visionnet_dagger.py        # DAgger imitation learning for MobileNet/EfficientNet/DualBackbone
├── run_connectome_rnn_checkpoint.py # Roll out / evaluate a trained connectome RNN checkpoint
├── run_vision_agent_checkpoint.py   # Roll out / evaluate a trained vision-baseline checkpoint
├── run_critic_warmup.py             # Warm-start the PPO value head before RL fine-tuning
├── analysis_pca.py                  # PCA of recorded hidden-state trajectories (per sensory condition)
├── analysis_pca_statistics.py       # Paired statistics on PCA trajectory distances across conditions
├── collision_statistics.py          # Aggregate collision counts / success rate across eval runs
├── count_collisions.py              # Count discrete collision events from a trajectory file
├── compare_trajectories.py          # Overlay trajectories from multiple models per episode
├── visualize_episodes.py            # Plot a single episode's trajectory and obstacles
├── tune_direction_threshold.py      # Sweep the straight-vs-turn classification threshold used by DAgger balancing
├── debug_rnn_grad.py / test_vfhplus.py  # Unit-style sanity checks for the sparse autograd op and the VFH* teacher
└── test_verification_data/          # Small fixture data used by the test scripts
```

## The Navigation Task & Environment

`environment/mujoco_two_cam_env_random_obstacles.py` implements a `gymnasium.Env`:

- **Robot:** differential-drive base with a left/right camera pair (`cam_left`, `cam_right`), simulated at 128x128 by default.
- **Arena:** a bounded square arena (configurable half-extent) with up to 20 randomly placed cylindrical obstacles and a randomly placed goal, re-sampled every episode.
- **Observation:** stereo camera images, a goal-direction vector, a binary collision flag + contact angle, a wind/airflow direction vector (for the connectome agent's Johnston's organ input), and a privileged state vector (pose, velocity, k-nearest obstacles) used only by the RL value head.
- **Action:** `[velocity, heading_angle]`, converted internally to left/right wheel commands.
- **Reward:** progress toward the goal, exploration bonus, control/time penalties, a collision penalty, and a "danger" penalty that discourages driving toward nearby walls/obstacles.
- **Episode data:** trajectories, obstacle layouts, collision flags, per-neuron recorded activity, and full hidden-state tensors can all be logged to `eval_data/` for offline analysis.

Path-planning utilities (`core/astar.py`, `core/vfhplus.py`) are used both by the analytic teacher and for computing obstacle-aware distance-to-goal.

## Models

### Connectome RNN (`agents/connectome_rnn_agent.py`, `models/connectome_rnn_model.py`)

The primary model under study. Each unit is one neuron from the connectome edge list (`connections_princeton.csv`); its state is updated by a leaky recurrent update

```
h_new = (1 - alpha_type) * h + alpha_type * phi(W_sparse @ h + b)
```

where `alpha_type` is a learnable per-cell-type leak rate and `phi` is `tanh`/`relu`. Key implementation details:

- **`MemoryEfficientSparseMM`** — a custom `torch.autograd.Function` that uses CSR format for a fast forward pass and cached COO indices for an O(NNZ) memory-efficient backward pass, with gradient checkpointing and batched chunking to fit large connectomes (~10-100k neurons) on a single GPU.
- **Virtual retina** — camera pixels are sampled at the real photoreceptor-column positions (`visual_column_L1_L2_L3_*.csv`) and passed through a learnable log-transform + high-pass (L1/L2) / low-pass (L3) temporal filter before entering the RNN, approximating the fly's early visual processing (R-cells -> lamina L1-L3).
- **Other sensory input** — tactile head bristles (`head_bristles_*.csv`) driven by collision/contact angle, and wind-direction input routed through a small MLP into the Johnston's organ neurons (`JO-C_and_JO-E.csv`).
- **Output** — a fixed set of descending neurons (`descending_neurons.csv`) is read out through a small MLP into `[velocity, heading]`.
- **Spectral scaling** — the connectome weight matrix is rescaled to a target spectral radius (`TARGET_RHO`) at load time for stable recurrent dynamics.
- Cell types (`consolidated_cell_types.csv`) parameterize per-type leak rates and let training scripts selectively freeze/train weights, biases, or per-type alphas via `shared_config.configure_optimizer`.

### Topological control connectomes

Two "null model" wirings use the exact same `LeakyConnectomeRNNCell` architecture but replace the real connectome edge list, isolating the contribution of the fly's actual wiring diagram from network size/sparsity alone:

- **Randomized-edge control** (`scripts/create_random_connectome.py`) — shuffles pre-/post-synaptic endpoints among the same set of neuron IDs (removing duplicate edges) and redraws synapse-count weights, destroying the real connectivity structure while preserving the number of nodes and edges.
- **Small-world control ("smallworldnet")** — the same RNN wired with a Watts-Strogatz small-world graph (`connections_ws_small_world.csv`) instead of the real connectome, used to test whether small-world topology alone (a property the fly connectome is known to have) explains task performance. Select it via `BASE_PATH`/`EDGE_PATH` in `shared_config.py`.

### Vision-CNN baselines (non-connectome)

- **`MobileNetAgent`** (`agents/mobilenet_agent.py`) — ImageNet-pretrained MobileNetV3-Large (first conv adapted to 1-channel grayscale, classifier removed) feeding a `GRUCell`, plus wind-direction and collision scalars, into a small MLP policy head.
- **`EfficientNetAgent`** (`agents/efficientnet_agent.py`) — identical scheme with an EfficientNet-B0 backbone.
- **`DualBackboneAgent`** (`agents/dual_backbone_agent.py`) — two independent backbones (EfficientNet-B0 or MobileNetV3-Large, selectable), one per eye, whose features are concatenated before the GRU — a closer analogue of the connectome agent's separate left/right visual streams.

### Analytic teacher (`agents/teacher_analytic_agent.py`, `core/vfhplus.py`)

`PlannerAnalyticTeacher` is not a trained network but a classical controller used to generate expert demonstrations for DAgger: a **VFH\*** planner (polar obstacle histogram + short-horizon A\* search with heading-consistency costs) produces a local waypoint, a PID line-follower steers toward it, and a scripted back-up/turn/forward recovery sequence handles collisions.

## Training Pipelines

All scripts are configured by editing constants near the top of the file (there is no CLI) and share connectome/environment settings from `shared_config.py`.

| Script | Purpose |
|---|---|
| `train_connectome_rnn_dagger.py` | DAgger imitation learning for the connectome RNN. Rolls out the current policy (vectorized across `N_ENVS` MuJoCo envs), mixes in teacher actions per a decaying `beta` schedule, buckets training chunks into `start/straight/turn/collision/pre-collision` categories for balanced sampling, and trains with truncated BPTT (`T_BURN` warmup + `T_UNROLL` steps) with gradient accumulation. |
| `train_connectome_rnn_rl.py` | PPO-style on-policy RL fine-tuning of the connectome RNN (GAE advantages, truncated-BPTT policy/value updates, clipped surrogate objective), typically initialized from a DAgger checkpoint. |
| `run_critic_warmup.py` | Warms up the PPO value head against a frozen/near-frozen policy before full RL fine-tuning begins. |
| `train_visionnet_dagger.py` | Same DAgger recipe applied to `MobileNetAgent` / `EfficientNetAgent` / `DualBackboneAgent` (`AGENT` constant selects which), with camera-dropout augmentation (blackout left/right/both eyes) for robustness. |

`shared_config.configure_optimizer` centralizes which parameter groups (RNN weights, biases, per-type alpha, readout head, value head, input-scale gains, wind MLP, virtual-retina parameters) are trainable, with a reduced learning rate for the per-cell-type leak parameters.

## Evaluation & Analysis

| Script | Purpose |
|---|---|
| `run_connectome_rnn_checkpoint.py` | Rolls out a trained connectome-RNN checkpoint over many episodes; can selectively blind the left/right eye (`vision=[bool, bool]`), record or clamp ("overwrite") the activity of specific neurons by `root_id`, and optionally dump full hidden-state trajectories to `.npy` for PCA. |
| `run_vision_agent_checkpoint.py` | Same evaluation harness for the CNN baselines. |
| `compare_trajectories.py` / `visualize_episodes.py` | Plot per-episode trajectories over the obstacle layout, optionally overlaying multiple models for direct comparison. |
| `count_collisions.py` / `collision_statistics.py` | Count discrete collision events per episode and aggregate collision-rate / success statistics across evaluation folders. |
| `analysis_pca.py` / `analysis_pca_statistics.py` | Project recorded hidden-state trajectories into a shared low-dimensional PCA space to visualize/quantify how sensory ablations (full vision, left-eye-only, right-eye-only, blind) shift the network's internal dynamics, with paired statistical tests across episodes. |
| `tune_direction_threshold.py` | Sweeps the heading-change threshold used to classify "straight" vs. "turn" segments for DAgger's balanced training buffer. |
| `debug_rnn_grad.py`, `test_vfhplus.py` | Sanity checks for the custom sparse-matmul autograd rule and for the VFH* teacher's obstacle-avoidance behavior in an interactive MuJoCo window. |

> Several analysis scripts (`analysis_pca.py`, `analysis_pca_statistics.py`, `compare_trajectories.py`) contain hard-coded local data paths from the original authoring machine — update the path constants near the top of each file before running them.

## Data Requirements

Connectome data, checkpoints, and generated datasets are intentionally excluded from version control (see `.gitignore`: `*.csv`, `*.parquet`, `*.pt`, `connectomes/`, `checkpoints/`, `eval_data/`, `loss/`). To run training you must supply your own connectome export (e.g., from FlyWire/FAFB codex) under `connectomes/drosophila adult connectome/` with the files referenced in `shared_config.py`:

- `connections_princeton.csv` — edge list (`pre_root_id`, `post_root_id`, `syn_count` or equivalent aliases)
- `visual_column_L1_L2_L3_rear_view_{left,right}.csv` — photoreceptor/lamina neuron positions and L1/L2/L3 type labels
- `head_bristles_{left,right}.csv` — tactile mechanosensory neurons
- `descending_neurons.csv` — output/descending neuron IDs
- `consolidated_cell_types.csv` — per-neuron cell-type labels
- `JO-C_and_JO-E.csv` — Johnston's organ (wind-sensing) neurons

Run `scripts/create_random_connectome.py` to derive the randomized-edge control from `connections_princeton.csv`, and supply a `connections_ws_small_world.csv` (Watts-Strogatz small-world graph over the same neuron count) to enable the small-world control condition.

## Installation

The project targets Python 3.10+ with a CUDA-capable GPU (the sparse-matmul cell also runs on CPU/MPS via `core.utils.get_device`, but training a full-scale connectome RNN is impractical without a GPU). There is no `requirements.txt`/`pyproject.toml` in the repo; based on the imports used throughout the codebase you will need:

```
torch
torchvision
numpy
pandas
scipy
scikit-learn
matplotlib
gymnasium
mujoco
glfw
opencv-python
```

## Quick Start

```bash
# 1. Place connectome CSVs under connectomes/drosophila adult connectome/ (see Data Requirements)

# 2. Train the connectome RNN with DAgger imitation learning
python train_connectome_rnn_dagger.py

# 3. (Optional) Fine-tune with on-policy PPO
python run_critic_warmup.py
python train_connectome_rnn_rl.py

# 4. Evaluate a checkpoint (edit CHECKPOINT at the top of the script first)
python run_connectome_rnn_checkpoint.py

# 5. Train a CNN baseline for comparison (set AGENT = "efficientnet" | "mobilenet" in the script)
python train_visionnet_dagger.py
```

## Citation

If you use this code, please cite the accompanying paper:

```
@misc{fly-gym-2026,
  title  = {[see arXiv:2607.00025 for the full title/author list]},
  eprint = {2607.00025},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2607.00025},
  year   = {2026}
}
```
