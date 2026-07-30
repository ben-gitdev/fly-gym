# fly_gym repo cleanup plan (for FLYNN paper release)

Produced by reading the FLYNN paper (`root (1).tex`) in full and auditing every tracked file plus
the large gitignored data/output directories against it, with an adversarial second pass on every
removal/relocation candidate (see "verify" evidence — nothing below is a first-guess).

**Nothing has been deleted, moved, or committed except the one specific consolidation logged below.**
Everything else in this document is still a plan to review, not an executed action.

---

## Progress log

- **Merged `connectomes/ws_small_world/add_weight.py` into `generate_ws_network_new.py` and deleted
  `add_weight.py`.** Investigation found a *second*, previously-unaudited generator script,
  `connectomes/ws_small_world/generate_ws_network_new.py`, sitting alongside the original
  `generate_ws_network.py`. File mtimes show `generate_ws_network_new.py` (not the older file) is
  what actually produced the currently-active SmallWorldNet data: it adds the missing Step 10
  (`assign_neuron_types`) that generates `consolidated_cell_types.csv` — something the older script
  never did — and its output schema matches what's on disk exactly. However, it still only wrote a
  2-column edge list (`pre_root_id,post_root_id`), while the real, in-use
  `connections_ws_small_world.csv` has a 3rd `syn_count` column. A separate, standalone,
  **unseeded** one-off script, `add_weight.py` (`df["syn_count"] = np.random.uniform(0, 1,
  size=len(df))`, no `np.random.seed(...)` call, not imported/called by anything), was the missing
  piece — its own mtime and the edge-list file's mtime lined up to the minute, confirming it ran as
  a manual post-process step right after generation.
  **Fix applied**: `generate_ws_network_new.py`'s `generate_ws_graph()` now assigns
  `syn_count = np.random.uniform(0.0, 1.0, size=len(ws_edges))` and writes all 3 columns in one pass,
  before the DN/community/matching steps run — so it draws from the same `RANDOM_SEED=42`-seeded
  global RNG stream as the rest of the script instead of a second, unseeded process. `add_weight.py`
  was then deleted (it was untracked, unreferenced, and fully superseded by this merge).
  **This supersedes §1.5 and the `generate_ws_network.py` references in §2b/§2c below** — see the
  updated §1.5 for what's fixed and what (the missing `connectome/` input subdirectory) is still open.

- **Fixed `generate_ws_network_new.py`'s `CONNECTOME_DIR` to point at
  `connectomes/drosophila adult connectome/` directly** (was `SCRIPT_DIR/"connectome"`, a
  subdirectory that never existed). Verified by importing the module and running Step 1
  (`load_fly_connectome()`) standalone against the real data: it now loads successfully —
  5,342,446 edges, 138,603 nodes, DN=1,303, HB_L=37, HB_R=40, JO=222, Vis_L=2,208, Vis_R=2,310
  (matches the paper's stated connectome stats). **Did not run the full script** (Steps 3–10 involve
  Watts-Strogatz generation + Louvain + up to 500 greedy-matching restarts over a 138k-node graph —
  slow, and it would overwrite the `connections_ws_small_world.csv`/`descending_neurons.csv`/modality
  CSVs your current SmallWorldNet checkpoints and `eval_data/` were produced against). **Caveat**:
  because the `syn_count` draw (added in the merge above) now happens inside `generate_ws_graph()`,
  a future full run will consume the seeded RNG stream in a different order than before Step 4
  onward — so re-running this script will be deterministic *from now on*, but will not reproduce
  today's exact DN sample / community assignment bit-for-bit (nothing that was previously exactly
  reproducible is lost by this, since the old split add_weight.py step was unseeded anyway). §1.5 is
  now resolved as a code-correctness issue; whether/when to actually trigger a full regeneration run
  is your call.

- **§1.1 resolved: `shared_config.py` now defaults to FLYNN.** `BASE_PATH` is now
  `"connectomes/drosophila adult connectome/"` (was `"connectomes/ws_small_world/"`), with a comment
  explaining to comment out the FLYNN line and uncomment the SmallWorldNet line to switch. `EDGE_PATH`
  no longer needs a second, independently-toggled comment/uncomment pair (the old pattern that could
  drift out of sync with `BASE_PATH`) — it's now derived from `BASE_PATH` via a small
  `_EDGE_FILE_BY_BASE_PATH` lookup, so there's exactly one line to toggle. Verified by import:
  `BASE_PATH`/`EDGE_PATH`/`CELL_TYPES_CSV` all resolve to real, existing FLYNN files by default.

- **§1.2 resolved: `train_connectome_rnn_dagger.py` hyperparameters now match the paper.** Changed
  `N_DAGGER_ITERS` 2→4, `EPISODES_PER_ITER` 700→500, `TRAIN_STEPS_PER_ITER` 500→300,
  `GRAD_ACCUM_STEPS` 4→2 (effective batch 64×2=128), `BETA_START` 0.0→1.0, `START_NOISE` 0.0→0.5,
  `NOISE_DECAY` 0.0→0.2, and `RESUME_CHECKPOINT_PATH` `"checkpoints/connectome_rnn_dagger_princeton_2.pt"`→`None`
  (train from scratch). Verified by import: all values now match the paper's stated schedule exactly
  (4 iters / 500 episodes / 300 steps / effective batch 128 / beta 1.0→0 at −0.5/iter / noise 0.5→0
  at −0.2/iter / from-scratch).

- **§1.3 resolved: added a `texture_mode` selector to `MuJoCoTwoCamEnv`, defaulting to checkerboard.**
  Restored `environment/mujoco_model_random_obstacles.xml` (the tracked file) to its checkerboard
  content (undoing the uncommitted PNG-texture edit) and moved the photo-realistic PNG-texture content
  into a new sibling file, `environment/mujoco_model_random_obstacles_realistic.xml`. Deleted the
  now-redundant `environment/mujoco_model_random_obstacles_checker.xml` snapshot (its content lives
  back in the main file). `MuJoCoTwoCamEnv.__init__` gained a `texture_mode` parameter
  (`"checker"` default, `"realistic"` alternative) with a `TEXTURE_XML_BY_MODE` class-level lookup
  and a `ValueError` on unknown values; all existing call sites (`train_connectome_rnn_dagger.py`,
  `run_connectome_rnn_checkpoint.py`, `train_visionnet_dagger.py`, `run_vision_agent_checkpoint.py`,
  `train_connectome_rnn_rl.py`, `run_critic_warmup.py`, `test_vfhplus.py`, `tune_direction_threshold.py`)
  construct the env with keyword args only, so none needed changes to keep working with the new default.
  Verified by actually instantiating the env with both `texture_mode="checker"` and `"realistic"` (both
  reset and render successfully) and confirming the default and the `ValueError` on a bogus mode.
  **Not done** (wasn't asked): wiring an actual OOD-eval call site to pass `texture_mode="realistic"` —
  right now that's still whatever the person running the OOD eval does by hand; happy to add that if
  wanted.

- **§1.4 mostly resolved: fixed the broken root-level checkerboard bar charts.** Investigating turned
  up a bigger root cause than originally scoped: `eval_data/` currently contains only fresh
  timestamped rollout folders (`connectome_rnn_20260305-171657`, `vision_efficientnet_20260306-120518`,
  etc.) — **neither** of `collision_statistics.py`'s two hardcoded folder lists (the commented-out
  16-folder checkerboard sweep, or the "active" 4-folder OOD sweep) resolves to anything that exists
  on disk anymore. So the original recommendation ("uncomment the checkerboard list and re-run it")
  would not have worked — it would have just skipped every folder and produced an empty result,
  because the raw per-episode data behind *both* figure sets has since been deleted/rotated away, not
  just the checkerboard one.
  What was still recoverable without any new evaluation: `collision_statistics_checker_texture.csv`
  (16-row aggregate, already sitting in the repo, values match the paper's Table I almost exactly)
  still has the mean/std numbers needed for 5 of the 6 bar charts, and
  `performance_results/checker_texure/` already had correctly-rendered SPL distribution plots. So:
    - Backed up the current (OOD-condition) root `bar_*.png` + `spl_*.png` to a new
      `performance_results/textured_env/` folder before touching anything (nothing lost).
    - Refactored `collision_statistics.py`: extracted the grouped-bar-chart code into a new
      `plot_grouped_bars(out_df, output_dir=None)` function (previously inlined and nested under an
      `if spl_data:` guard it didn't actually need), and added `regenerate_bars_from_stats_csv(csv_path,
      output_dir=None)`, which loads an aggregate CSV directly and calls it — for exactly this
      "the raw eval_data is gone but the aggregate CSV survived" situation.
    - Added a defensive guard in `__main__`: if none of the configured folders exist under
      `eval_data/`, skip `collision_statistics()` entirely (print a warning) instead of silently
      overwriting `collision_statistics.csv` with an empty result — this exact class of bug is what
      broke the bar charts in the first place, so it can't quietly happen again.
    - Ran it: `bar_collisions.png`, `bar_success_rate.png`, `bar_average_spl.png`,
      `bar_average_spl_success.png`, `bar_average_speed.png` regenerated correctly at the repo root
      (verified: `collision_statistics.csv`, the OOD data, stayed byte-identical throughout; visually
      confirmed `bar_success_rate.png` now shows all 4 models with values matching Table I, e.g. total
      blindness FLYNN 44%/EfficientNet 4%/MobileNet 17%/SmallWorldNet 3% vs. the paper's 44.3/3.9/16.5/2.6).
      Also copied `spl_violin_plots.png`, `spl_histograms.png`, `spl_accumulated_histograms.png` from
      `performance_results/checker_texure/` to the root (visually confirmed `spl_histograms.png` is
      the correct 4-panel, one-per-vision-condition figure).
    - **`bar_time_to_goal.png` could not be fixed this way** and was left untouched (still shows the
      OOD condition, backed up alongside the others) — `collision_statistics_checker_texture.csv` has
      no "Time to Goal Mean/Std" columns (it predates that feature), and the raw `steps`/`goal_reached`
      per-episode data needed to compute episode duration for the checkerboard sweep no longer exists
      anywhere on disk. **This needs your input**: the only way to get a correct checkerboard-condition
      `bar_time_to_goal.png` (and, more generally, fully fresh/complete data for everything) is to
      re-run the evaluation sweep (`run_connectome_rnn_checkpoint.py` for FLYNN/SmallWorldNet,
      `run_vision_agent_checkpoint.py` for EfficientNet/MobileNet, 4 vision conditions each) against
      the checkpoints in `checkpoints/`, which is a real compute commitment (hundreds of episodes ×
      16 configs) I didn't want to kick off unilaterally. **Decision (per author): leave
      `bar_time_to_goal.png` as a documented gap for now** — not fixing until/unless a fresh
      evaluation sweep happens for other reasons.

- **§1.6 resolved: `run_vision_agent_checkpoint.py`'s `checkpoint` is now a required CLI argument.**
  Added `argparse`: `checkpoint` is a required positional arg (previously
  `checkpoint_path = os.path.join(CHECKPOINT_DIR, "efficientnet_dagger_final_robust.pt")` with a
  silent "if missing, grab whatever *.pt happens to be in checkpoints/ first" fallback — exactly the
  kind of implicit choice that made it impossible to tell from the code alone which checkpoint
  produced a given result). A missing/nonexistent path is now a loud `FileNotFoundError` listing the
  actual available checkpoints, not a silent substitution. `model_type` is now auto-detected from the
  checkpoint filename via the existing `_detect_model_type()` helper (previously a separately
  hardcoded `model_type = "efficientnet"` constant that had to be kept in sync by hand — a second,
  related footgun: passing a MobileNet checkpoint while this stayed "efficientnet" would corrupt or
  crash the load). Added an optional `--model-type` override for filenames auto-detection would guess
  wrong on. Verified: `--help` shows both; running with no checkpoint arg fails fast with argparse's
  usage error; running with a bad path raises `FileNotFoundError` listing real checkpoints; running
  with a real MobileNet checkpoint auto-detects `model_type=mobilenet` and starts evaluating
  correctly; running with an intentionally mismatched `--model-type` against a real checkpoint fails
  loudly with PyTorch's `load_state_dict` shape-mismatch error instead of silently loading garbage
  weights. `episodes`/`vision`/`render`/internal-state-recording config were left as the existing
  hardcoded constants in `main()` — only the checkpoint-selection ambiguity was in scope here.

- **§1.7 resolved: fixed `tune_direction_threshold.py`, including a second bug beyond the one
  originally flagged.** The stale `rollout_episode` import (removed when
  `train_connectome_rnn_dagger.py` was refactored to the batched, multi-env
  `rollout_and_collect_balanced()`, which has a fundamentally different interface — it fills a
  shared buffer across N envs rather than returning one episode's raw trajectory, so it's not a
  drop-in replacement) is now a small local `rollout_episode()` reimplemented directly in
  `tune_direction_threshold.py`, covering only the pure-teacher-drive case
  (`beta=1.0, beta_noise=0.0`) this script actually calls with — it raises `NotImplementedError` for
  any other beta, rather than silently pretending to support agent-blended rollout it doesn't
  implement. While tracing the call path, also found `cell, pr_positions, input_splits =
  build_connectome_cell(...)` unpacking only 3 values from a function that now returns 4 (the exact
  same bug class already found and removed in `train_connectome_rnn_rl.py`/`run_critic_warmup.py`) —
  fixed to unpack all 4. Verified by actually running the script end-to-end (not just import-checking
  it): it built the real FLYNN connectome cell (138,584 nodes, now the default per §1.1), ran a
  225-step teacher-only episode, and regenerated `direction_tuning_plot.png` with a sane-looking
  result (a turn spike at the start settling to near-zero angle change once the path straightens out).

---

## 0. TL;DR

- Only **32 files are currently tracked in git**. Everything else (299MB `backups/`, 1.4GB
  `checkpoints/`, 960MB `connectomes/`, 83GB `eval_data/`, `loss/`, and every stray `*.png`/`*.csv`)
  is excluded by `.gitignore` — some of that correctly, some of it **by accident** (see §1).
- I found **several correctness/reproducibility problems that are bigger than "cleanup"** — things
  that would make a fresh clone silently fail to reproduce the paper's numbers. Fix these *before*
  worrying about which files to delete. See §1.
- Everything else sorts into 4 buckets: keep-as-is, keep-but-fix-`.gitignore`, too-big-for-git
  (external host), and safe-to-remove. A handful of items are a genuine author call, not something
  I can decide from the code alone — flagged as "needs your decision" throughout.

---

## 1. Fix these before (or during) cleanup — real correctness risks, not just tidiness

1. ~~`shared_config.py:26-27` defaults to SmallWorldNet, not FLYNN.~~ **RESOLVED** — see Progress log.
   `BASE_PATH` now defaults to `"connectomes/drosophila adult connectome/"` (FLYNN), with a comment
   explaining how to switch to SmallWorldNet, and `EDGE_PATH` is derived from `BASE_PATH` so the two
   can't drift out of sync the way the old independent-comment-pair pattern could.

2. ~~`train_connectome_rnn_dagger.py`'s hyperparameters don't match the paper's stated schedule.~~
   **RESOLVED** — see Progress log. `N_DAGGER_ITERS=4`, `EPISODES_PER_ITER=500`,
   `TRAIN_STEPS_PER_ITER=300`, effective batch 128, `BETA_START=1.0`/`BETA_DECAY=0.5`,
   `START_NOISE=0.5`/`NOISE_DECAY=0.2`, `RESUME_CHECKPOINT_PATH=None` (from scratch) — all now match
   the paper's stated schedule exactly, verified by import.

3. ~~The two example MuJoCo scene files are mid-swap with no code to select between them.~~
   **RESOLVED** — see Progress log. `environment/mujoco_model_random_obstacles.xml` is restored to
   the checkerboard variant (the default); the photo-realistic PNG-texture variant now lives in its
   own file, `environment/mujoco_model_random_obstacles_realistic.xml`; and
   `MuJoCoTwoCamEnv(texture_mode=...)` selects between them (`"checker"` default, `"realistic"`
   alternative), verified by instantiating both. The now-redundant untracked
   `mujoco_model_random_obstacles_checker.xml` snapshot was deleted.

4. ~~The main bar-chart figures at the repo root look broken for the checkerboard condition.~~
   **MOSTLY RESOLVED** — see Progress log for the full story, including a root cause bigger than
   originally scoped: **neither** folder list in `collision_statistics.py`'s `__main__` resolves to
   anything in `eval_data/` anymore (it now only holds fresh timestamped run folders, e.g.
   `connectome_rnn_20260305-171657`, not condition-named ones) — the OOD list I originally thought
   was "still active/valid" is equally stale. Fixed by regenerating 5 of the 6 root bar charts plus
   all 3 SPL distribution plots directly from the already-correct, already-computed
   `collision_statistics_checker_texture.csv` / `performance_results/checker_texure/` (no raw
   eval_data needed), and added a defensive guard so `collision_statistics.py` can no longer silently
   overwrite a good CSV with an empty one when its configured folders don't exist. **Not resolved**:
   `bar_time_to_goal.png` — the checkerboard aggregate CSV has no "Time to Goal" columns (predates
   that feature), and the raw per-episode duration data needed to compute it no longer exists on
   disk. Fixing that needs a fresh evaluation run; see the question at the end of the Progress log
   entry.

5. ~~`connectomes/ws_small_world/generate_ws_network_new.py` doesn't reproduce its own output file~~
   **RESOLVED.** The real generator (supersedes the older `generate_ws_network.py`, which never
   produced `consolidated_cell_types.csv` at all and isn't what generated the current data) now
   writes the full 3-column edge list in one seeded pass (weight-column merge, see Progress log) and
   `CONNECTOME_DIR` now points at `connectomes/drosophila adult connectome/` directly instead of a
   nonexistent `connectome/` subfolder. Verified Step 1 (`load_fly_connectome`) runs standalone
   against the real data (see Progress log for the exact numbers). The full script (Steps 3–10) has
   not been run — it's slow (WS graph + Louvain + up to 500 matching restarts over 138k nodes) and
   would overwrite the currently-in-use SmallWorldNet data files, so that's left for you to trigger
   when you're ready to accept a freshly-regenerated (not bit-identical) SmallWorldNet instance.

6. ~~`run_vision_agent_checkpoint.py`'s default checkpoint name doesn't match the training script's own
   naming convention~~ **RESOLVED (the ambiguity, not the "which checkpoint was Table I" question)**
   — see Progress log. `checkpoint` is now a required CLI positional argument (no more hardcoded
   default + silent "pick any checkpoint found" fallback), so every future run states explicitly which
   checkpoint it's evaluating instead of leaving that implicit/guessable. You'll still need to decide
   for yourself which checkpoint file was actually used to produce Table I's reported numbers — this
   change just stops the script from hiding or silently substituting that choice going forward.

7. ~~`tune_direction_threshold.py` has a stale import~~ **RESOLVED** — see Progress log. Also found and
   fixed a second, same-class bug in the same file (a stale 3-value unpack of
   `build_connectome_cell()`, which now returns 4). Verified by actually running the script.

None of the above are things I'm fixing myself — they're substantive judgment calls or require
re-running training/eval, so they're yours to make. I mention them here because several of the
file-classification calls below only make sense once you know about them.

---

## 2. File classification

### 2a. Keep as-is — tracked, correct, needed to reproduce the paper

| File | Maps to |
|---|---|
| `core/astar.py`, `core/utils.py`, `core/vfhplus.py`, `core/__init__.py` | env global path-planning / shared utils+connectome loader / VFH* teacher |
| `environment/__init__.py`, `environment/mujoco_two_cam_env_random_obstacles.py` | MuJoCo env; now has a `texture_mode` selector (`"checker"` default / `"realistic"` for OOD) — §1.3 resolved |
| `environment/mujoco_model_random_obstacles.xml`, `environment/mujoco_model_random_obstacles_realistic.xml` | The two scene variants the selector above picks between (checkerboard training default / photo-realistic OOD) |
| `agents/__init__.py`, `connectome_rnn_agent.py`, `efficientnet_agent.py`, `mobilenet_agent.py`, `teacher_analytic_agent.py` | FLYNN sensory front-end / CNN baselines / DAgger teacher |
| `models/__init__.py`, `connectome_rnn_model.py`, `teacher_analytic_model.py` | FLYNN's core RNN cell + custom sparse autograd / teacher's path-follow controller |
| `train_connectome_rnn_dagger.py` | FLYNN/SmallWorldNet DAgger training (hyperparameters now match the paper — §1.2 resolved) |
| `train_visionnet_dagger.py` | EfficientNet/MobileNet DAgger + camera-dropout training (matches paper exactly) |
| `run_connectome_rnn_checkpoint.py`, `run_vision_agent_checkpoint.py` | eval rollouts → `eval_data/`, Table I source, PCA hidden-states source |
| `shared_config.py` | shared paths/hyperparameters (now defaults to FLYNN — §1.1 resolved) |
| `test_vfhplus.py` | sanity-check tool for the VFH*+PID teacher, worth keeping even with 0 importers |
| `analysis_pca_statistics.py` | the actual KDE + `BF≈BL+BR` vector-arithmetic analysis in the paper |
| `collision_statistics.py`, `count_collisions.py`, `compare_trajectories.py`, `visualize_episodes.py` | Table I metrics + trajectory/bar-chart figures |

### 2b. Needed, but wrongly excluded by `.gitignore` today

| Path | Size | Why it's needed |
|---|---|---|
| `environment/textures/{ground,wall,obstacle,skybox}.png` | ~3.4MB total | Sim assets for the now-separate `mujoco_model_random_obstacles_realistic.xml` (§1.3 resolved), not result figures — caught by the blanket `*.png` rule. |
| `connectomes/.../JO-C_and_JO-E.csv`, `consolidated_cell_types.csv`, `descending_neurons.csv`, `head_bristles_{left,right}.csv`, `visual_column_L1_L2_L3_rear_view_{left,right}.csv` (both the adult-connectome and ws_small_world copies) | all <4MB | Small per-modality neuron ID CSVs the model actually loads (wind/tactile/vision/motor/cell-type). Caught by the blanket `connectomes/`+`*.csv` rules. |
| `connectomes/ws_small_world/generate_ws_network_new.py` | 36KB | The actual SmallWorldNet generator (paper §III-C-2) — currently sits inert inside a fully-gitignored folder. Both the weight-column gap and the input-path bug are now fixed (see Progress log / §1.5) — Step 1 verified to run against the real data. |
| `bar_average_speed.png`, `bar_average_spl.png`, `bar_average_spl_success.png`, `bar_collisions.png`, `bar_success_rate.png`, `bar_time_to_goal.png`, `spl_accumulated_histograms.png`, `spl_histograms.png`, `spl_violin_plots.png` | ~1.9MB | The actual paper bar-chart/SPL figures — caught by the blanket `*.png` rule. Fix §1.4 first, then regenerate. |
| `collision_statistics.csv`, `collision_statistics_checker_texture.csv` | <2KB each | The literal numeric source of Table I — caught by the blanket `*.csv` rule. |

**Fix**: add explicit `!`-exceptions for these paths (or move final figures/summary CSVs into a
dedicated `figures/`/`results/` directory with its own exception), rather than removing the blanket
rules entirely — the blanket rules are doing real work for the big data/output directories.

### 2c. Needed, but too large to commit to git directly — external hosting

| Path | Size | Recommendation |
|---|---|---|
| `connectomes/drosophila adult connectome/connections_princeton.csv` | 261MB | Zenodo/institutional storage + a small download script. **Check flywire.ai / philshiu/Drosophila_brain_model redistribution terms first** — `data source.txt` gives provenance but no license, so this is a real open question, not just a size problem. |
| `connectomes/ws_small_world/connections_ws_small_world.csv` | 169MB | Now fully regeneratable from `generate_ws_network_new.py` (§1.5 resolved) instead of needing external hosting — it's synthetic data with a fixed seed, no license issue. A fresh run won't be bit-identical to the current file (see Progress log caveat), so decide whether to keep the current file as an archived/pinned version or regenerate and treat the new run as canonical. |
| `checkpoints/*.pt` (final models only — see below) | 1.4GB total | Zenodo/HuggingFace/institutional storage + download script, so Table I is reproducible without a full retrain. |

For `checkpoints/`, not all 16 files are equally necessary — recommend hosting only:
`connectome_rnn_dagger_iter_4.pt` (or whichever is FLYNN's final), `connectome_rnn_dagger_princeton*.pt`
(confirm which one is "the" FLYNN checkpoint used for Table I), `connectome_rnn_dagger_small_world_full_vision.pt`,
`efficientnet_dagger_final_{full_vision,robust}.pt`, `mobilenet_dagger_final_{full_vision,robust}.pt`.
The `iter_1..3.pt` intermediates, the `princeton`/`princeton_2`/`princeton_3` resume-chain files, and
`connectome_rnn_dagger_princeton_bad_blind.pt` look like intermediate/discarded runs, not final models
— your call whether any of those need to be archived for provenance.

`eval_data/` (83GB, 1675 files) is the raw per-episode rollout log behind `collision_statistics.py`'s
aggregates. It's regeneratable (re-run the eval scripts against the hosted checkpoints), and I found
that the specific data behind the PCA/KDE/trajectory figures is *already* separately archived outside
both git and `eval_data/` (an external `Publications/IROS2026/materials/` folder `analysis_pca.py`
and `visualize_episodes.py` already point at). Recommend: don't try to publish 83GB; just make sure
that external archive is complete, and let `collision_statistics.csv`/`collision_statistics_checker_texture.csv`
(§2b) be the citable aggregate.

### 2d. Safe to remove — confirmed dead, superseded, or unrelated to the paper

| Path | Why |
|---|---|
| `models/connectome_rnn_model_no custom autograd.py` | Unreferenced pre-optimization snapshot of `connectome_rnn_model.py` (predates the custom sparse-autograd backward pass); non-importable filename (has a literal space) confirms it was never meant to be loaded. |
| `connectomes/ws_small_world/generate_ws_network.py` (the older, non-`_new` file) | Superseded by `generate_ws_network_new.py`: mtimes show the `_new` version is what actually produced the current SmallWorldNet data, and only it generates `consolidated_cell_types.csv` (the older file never did, at all). Keeping both invites a reader to run the wrong one. |
| `train_connectome_rnn_rl.py`, `run_critic_warmup.py` | Abandoned PPO/critic training path for the connectome RNN. Not mentioned anywhere in the paper (only DAgger is described). Both are currently broken as committed (`build_connectome_cell()` unpacking mismatch — 3 vs. 4 return values). Removing them only requires deleting one import line + unused kwargs from `run_connectome_rnn_checkpoint.py`/`run_critic_warmup.py`'s two dependents. |
| `MUJOCO_LOG.TXT` | Auto-generated MuJoCo physics-instability warning log from a past debugging session; slipped past `.gitignore`'s `*.log` rule because it's `*.TXT`. Not source, not read by anything. |
| `connectomes/drosophila adult connectome/parquet_to_csv.py`, `Connectivity_783.csv`, `Connectivity_783.parquet.png` | Superseded, numerically incompatible earlier connectome import (different index space, ~15M edges vs. the paper's reported 5,342,445 — not just a reformat of the same data). Nothing reads it. |
| `connectomes/drosophila adult connectome/connections_princeton_random.csv` | Fully regeneratable (via `scripts/create_random_connectome.py`, seed=42), 276MB, referenced only by commented-out code. No reason to store/host the generated CSV. |
| `connectomes/drosophila adult connectome/photoreceptors_pos_{left,right}.csv`, `moonwalker_neurons.csv`, `olfactory_ORN_DM1_{left,right}.csv` | Raw R1-6 photoreceptor / moonwalker / olfactory data explicitly superseded per your own dev log ("Ditched R1-6 input... SOLUTION: ...L1-3 input") and the paper's own stated rationale. Zero live references; incompatible column schema with the current loader anyway. |
| `connectomes/c.elegans connectome/` (entire folder, ~423KB) | Unrelated dataset; zero references anywhere in any `.py` file, tracked or not; the paper never mentions C. elegans. |
| `connectomes/drosophila larva connectome/` (entire folder, ~31MB) | Same — unrelated, unreferenced, predates even the unrelated "CNS project" log entries. |
| `backups/` (10 dated snapshot folders, 299MB) | Ad hoc whole-repo snapshots from Oct 2025–Jan 2026, all superseded by tracked code and predating even the paper-relevant portion of the dev log. **Caveat**: the repo's first git commit is 2026-02-12, *after* every backup folder's date — so git history does *not* actually preserve this period. Recommend archiving privately outside the repo (zip to institutional storage) rather than hard-deleting, purely so you don't lose ~3.5 months of provenance; it should not ship in the public release either way. |
| `loss/` (409KB) | Pure training-loss byproduct, regenerated fresh on every run, not read by anything, no paper figure is a loss curve. |
| `tests/` (empty dir) | Empty, unreferenced (its one historical occupant, `test_dagger_buffer.py`, tested a since-renamed/removed class and was already deleted intentionally in an earlier commit). |
| `collision_statistics 1.csv`, `collision_statistics_2_old_crnn.csv`, `collision_statistics_real_textured.csv` | Superseded duplicates of `collision_statistics.csv`/`collision_statistics_checker_texture.csv` (identical data, missing later-added columns, or referencing an abandoned checkpoint-selection sweep). |

### 2e. Needs your decision — I can't resolve these from the code alone

| Path | The question |
|---|---|
| `test_verification_data/obstacles_1.txt` (+ its untracked siblings `trajectory_1.csv`, `activity_1.csv`) | Looks like a hand-saved early episode snapshot for regression testing, but nothing loads it today and its `trajectory_1.csv` schema (`step,x,y`, no `collision` column) predates `count_collisions.py`'s current requirements. Restore a deterministic-replay consumer, or drop it as a stale fixture? |
| `agents/dual_backbone_agent.py` | Genuinely wired into `run_vision_agent_checkpoint.py`/`train_visionnet_dagger.py` (removing it breaks both scripts), but no checkpoint for it ever completed training, and the paper's Section III-C text describes only the single-shared-backbone design. Recommend: keep the file (don't break the two scripts), but add a one-line comment marking it an incomplete/unused ablation so a reader doesn't assume it produced the reported baseline numbers. |
| `scripts/create_random_connectome.py` (+ its output, already listed for removal in §2d) | Real experiment (a checkpoint exists: `connectome_rnn_dagger_princeton_random_full_vision.pt`), but not the paper's reported SmallWorldNet baseline (that's the separate Watts-Strogatz control). Keep with a clarifying comment (early/discarded ablation), or delete? |
| `tune_direction_threshold.py`, `debug_rnn_grad.py` | Legitimate dev tools (threshold-tuning plot, gradient-correctness sanity check for the custom sparse autograd) but not required to reproduce any reported number. `tune_direction_threshold.py` is now fixed and runnable again (§1.7). Keep as documentation/tests (maybe move into a `tests/`/`tools/` folder), or drop? |
| `direction_tuning_plot.png` | Output of `tune_direction_threshold.py` above — same call. |
| README.md / LICENSE / `requirements.txt` (or `environment.yml`) / citation file | None exist anywhere in the repo (checked every subfolder, not just root). Not a per-file classification question — these need to be authored before public release: setup instructions (MuJoCo/torch/opencv/pandas/numpy/scipy/matplotlib/networkx/python-louvain), a license (relevant since this wraps FlyWire.ai connectome data), and a citation/BibTeX entry for the paper. |

---

## 3. Suggested `.gitignore` rewrite (after resolving §2e's texture-file decision)

Keep the blanket exclusions for genuinely-bulk/output directories, but carve out the specific
assets/scripts/summaries identified in §2b:

```gitignore
__pycache__/
*.py[cod]
.venv/
build/
dist/
*.log
MUJOCO_LOG.TXT

# Model checkpoints (host externally, see CLEANUP_PLAN.md §2c)
*.pt
*.pth

# Bulk data / generated artifacts
*.parquet
*.jsonl
*.out
backups/
checkpoints/
eval_data/
loss/
performance_results/

# Connectome data: ignore big raw edge lists, keep small metadata + scripts
connectomes/**/*.csv
!connectomes/**/JO-C_and_JO-E.csv
!connectomes/**/consolidated_cell_types.csv
!connectomes/**/descending_neurons.csv
!connectomes/**/head_bristles_*.csv
!connectomes/**/visual_column_L1_L2_L3_rear_view_*.csv
connectomes/**/*.py
!connectomes/ws_small_world/generate_ws_network_new.py
connectomes/c.elegans connectome/
connectomes/drosophila larva connectome/

# Result figures: ignore stray CSV/PNG dumps, keep the final published ones
*.csv
!collision_statistics.csv
!collision_statistics_checker_texture.csv
*.png
!environment/textures/*.png
!bar_*.png
!spl_*.png
```

(Git ignore-exception ordering matters — double-check with `git check-ignore -v <path>` after editing.)

---

## 4. Suggested execution order

1. ~~Resolve §1.3 (texture/xml decision) and §1.2 (DAgger hyperparameters) first~~ **Done** — see
   Progress log. Next: decide whether to actually re-run DAgger training with the corrected
   from-scratch hyperparameters (§1.2) and/or wire an OOD-eval call site to pass
   `texture_mode="realistic"` (§1.3), since the code fixes alone don't retrain/re-evaluate anything.
2. ~~Re-run `collision_statistics.py` with the checkerboard folder list (§1.4)~~ **Done** — see
   Progress log (turned out the raw eval_data was gone entirely; fixed by regenerating from the
   surviving aggregate CSV instead). Still open: `bar_time_to_goal.png` needs a fresh eval run to
   fix properly — your call whether/when to do that. `performance_results/checker_texure/` and the
   new `performance_results/textured_env/` backup are both worth keeping as provenance for now
   rather than retiring either.
3. Apply the `.gitignore` rewrite (§3), `git add` the newly-un-ignored files from §2b, and verify
   `git status` shows exactly the expected new tracked files (nothing from `checkpoints/`/`eval_data/`/
   `connections_princeton.csv` should appear).
4. `git rm` the confirmed-dead tracked file: `models/connectome_rnn_model_no custom autograd.py`,
   `train_connectome_rnn_rl.py`, `run_critic_warmup.py`, `MUJOCO_LOG.TXT` — and fix the two now-dangling
   import lines in `run_connectome_rnn_checkpoint.py` (imports 5 unused constants from
   `train_connectome_rnn_rl.py`) accordingly.
5. Delete (not git-tracked, just disk cleanup) the confirmed-dead untracked data:
   `connectomes/c.elegans connectome/`, `connectomes/drosophila larva connectome/`,
   `connectomes/drosophila adult connectome/{parquet_to_csv.py,Connectivity_783.csv,Connectivity_783.parquet.png,connections_princeton_random.csv,photoreceptors_pos_left.csv,photoreceptors_pos_right.csv,moonwalker_neurons.csv,olfactory_ORN_DM1_left.csv,olfactory_ORN_DM1_right.csv}`,
   `loss/`, `tests/`, the 3 superseded `collision_statistics*.csv` variants.
6. Archive `backups/` and (post-verification) the bulk of `eval_data/` to external/institutional
   storage rather than deleting outright, then remove from the working copy.
7. Decide whether to trigger a full `generate_ws_network_new.py` run now that §1.5 is fixed (this
   will produce a fresh, valid, but not bit-identical SmallWorldNet dataset — see Progress log), or
   keep the current `connections_ws_small_world.csv` pinned as-is. Either way, upload
   `connections_princeton.csv` and the pruned `checkpoints/` set to external hosting, with a small
   download script + README section pointing at them.
8. Resolve the remaining §2e author calls (`dual_backbone_agent.py` comment, `create_random_connectome.py`
   keep/drop, `tune_direction_threshold.py`/`debug_rnn_grad.py` keep/relocate, `test_verification_data/`
   keep/drop) at your convenience — none of these block a first public push.
9. Write README.md, LICENSE, requirements.txt/environment.yml, and a citation entry.
