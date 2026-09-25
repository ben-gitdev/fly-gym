"""
Export the data for the browser version of the race-the-agent game (docs/index.html).

The web game re-creates play_vs_connectome_rnn.py without Python.  The player's robot is
simulated in the browser; the agent's runs are recorded here, rolled out exactly as the
desktop game does it.  That split works because the two robots never interact -- each drives
its own copy of the arena -- so a recorded run is equivalent to a live one.

Writes docs/data/game.json and one docs/data/runs/<seed>.bin per layout:

  game.json   * scene: what the browser needs to simulate and render the player's robot --
                the photoreceptor rays of both eyes, the lighting and materials of the arena,
                and the robot's measured response to each arrow-key command
              * layouts: obstacles, goal, start pose, and the agent's result per layout
  <seed>.bin  the agent's run, gzip-compressed:
                float32 [T+1, 3]   pose (x, y, yaw) at each step, starting from the reset
                uint8   [T+1]      bump sensors: bit 0 left, bit 1 right
                uint8   [T+1, D]   eye input at each of the D photoreceptor positions
                                   (left eye, then right), 0-255; delta-coded over time
                                   (frame 0 raw, then frame t minus frame t-1, mod 256)

    python export_web_game.py checkpoints/connectome_rnn_dagger_princeton_full_vision.pt \
        --connectome "connectomes/drosophila adult connectome/connections_princeton.csv"
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os

import mujoco
import numpy as np
import torch

from core.utils import get_device
from play_vs_connectome_rnn import DEFAULT_MAX_STEPS, PLAYER_SPEED, PLAYER_TURN, AgentRacer, path_length
from run_connectome_rnn_checkpoint import (
    VISION_CONDITIONS,
    PhotoreceptorView,
    _load_agent,
    _make_env,
    resolve_connectome_path,
)
from shared_config import DTYPE, N_OBSTACLES

OUT_DIR = os.path.join("docs", "data")
EYES = ("left", "right")
GRAY_WEIGHTS = [0.2989, 0.5870, 0.1140]  # ConnectomeAgent.gray_weights

# Checker colours of the obstacle texture as they render (even, odd squares), fitted to MuJoCo
# camera images.  With the raw texture colours the browser's lighting model renders obstacles
# ~12% too bright; with these its eye input is within ~2/255 of MuJoCo's on average.
OBSTACLE_TEXTURE = ([0.76, 0.35, 0.33], [0.62, 0.20, 0.19])
# Which band of each wall face shows the texture's first (TL/BR) colour: measured from renders.
WALL_EVEN_IS_FIRST = {"wall_n": True, "wall_e": True, "wall_s": False, "wall_w": False}
# Contact response of the browser's kinematic robot, fitted to MuJoCo's high-friction contacts,
# where a robot pushing into a surface barely slides and pivots until it faces it: while pushing,
# its speed is capped at CONTACT_SLIDE x the cruise speed, and it turns towards the surface at
# CONTACT_PIVOT * sin(angle between its motion and the surface normal) rad/s.
CONTACT_SLIDE = 0.15
CONTACT_PIVOT = 1.7


def _texture_colors(m: mujoco.MjModel, name: str):
    """(first, second) checker colour of a builtin checker texture: TL/BR and TR/BL quadrants."""
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    w, h, c = m.tex_width[t], m.tex_height[t], m.tex_nchannel[t]
    img = m.tex_data[m.tex_adr[t]:m.tex_adr[t] + w * h * c].reshape(h, w, c) / 255.0
    return img[: h // 2, : w // 2].reshape(-1, c).mean(0), img[: h // 2, w // 2:].reshape(-1, c).mean(0)


def _geom(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def _round(a, nd=5):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _park_obstacles(env) -> None:
    """Move every obstacle into the far corner so the robot can be measured in free space."""
    m, d = env.model, env.data
    for i in range(env.n_obstacles):
        d.qpos[env._obstacle_x_qposadr[i]] = d.qpos[env._obstacle_y_qposadr[i]] = -6.3
        m.jnt_range[env.obstacle_x_joint_ids[i]] = m.jnt_range[env.obstacle_y_joint_ids[i]] = [-6.3 - 1e-4, -6.3 + 1e-4]
        env._obstacle_xy[i] = [-6.3, -6.3]


def _place(env, x: float, y: float, yaw: float) -> None:
    env.place_robot(x, y, yaw, env.data.qpos.copy(), env.data.qvel.copy())
    mujoco.mj_forward(env.model, env.data)


def _step_response(env, drive: int, turn: int) -> tuple:
    """Hold one key combination for 2 s from rest, then release it for 1 s, in free space;
    per-step forward speed and turn rate."""
    dt = env.model.opt.timestep * env.frame_skip
    mujoco.mj_resetData(env.model, env.data)
    env.reset(seed=0)
    _park_obstacles(env)
    _place(env, 2.0, 0.0, 0.0)
    for _ in range(25):  # settle on the wheels
        env.step(np.zeros(2, np.float32))
    poses = [(*env._base_xy(), env._base_yaw())]
    for k in range(150):
        held = k < 100
        env.step(np.array([PLAYER_SPEED * drive * held, PLAYER_TURN * turn * held], np.float32))
        poses.append((*env._base_xy(), env._base_yaw()))
    p = np.array(poses)
    yaw = np.unwrap(p[:, 2])
    v = (np.diff(p[:, 0]) * np.cos(yaw[1:]) + np.diff(p[:, 1]) * np.sin(yaw[1:])) / dt
    return v, np.diff(yaw) / dt


def _time_constants(x: np.ndarray, dt: float) -> dict:
    """First-order time constants that best fit a step response (held for steps 0-99, released
    from step 100): how fast it rises to its steady value, and how fast it decays after."""
    steady = x[50:100].mean()
    rise, fall = x[:25] / steady, x[100:125] / steady
    t, taus = dt * np.arange(1, 26), np.linspace(0.02, 0.5, 241)
    return {"tau_rise": round(float(taus[np.argmin([np.sum((1 - np.exp(-t / k) - rise) ** 2) for k in taus])]), 3),
            "tau_fall": round(float(taus[np.argmin([np.sum((np.exp(-t / k) - fall) ** 2) for k in taus])]), 3)}


def measure_motion(env) -> dict:
    """How the robot responds to the arrow keys, measured in MuJoCo in free space: the steady
    (speed, turn rate) for each key combination, and how fast speed and turning respond."""
    dt = env.model.opt.timestep * env.frame_skip
    table = {}
    for drive in (1, 0, -1):
        for turn in (1, 0, -1):
            v, w = _step_response(env, drive, turn)
            table[f"{drive},{turn}"] = [round(float(v[50:100].mean()), 4), round(float(w[50:100].mean()), 4)]
    return {"table": table,
            "speed_lag": _time_constants(_step_response(env, 1, 0)[0], dt),  # driving forward
            "turn_lag": _time_constants(_step_response(env, 0, 1)[1], dt),   # turning on the spot
            "contact_slide": CONTACT_SLIDE, "contact_pivot": CONTACT_PIVOT}


def eye_geometry(agent, env) -> tuple:
    """Photoreceptor positions per eye, the input index holding each one's value, and the camera
    frames in the robot's body frame."""
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    env.reset(seed=0)
    _park_obstacles(env)
    _place(env, 0.0, 0.0, 0.0)
    for _ in range(25):  # settle, so the cameras sit at their driving height
        env.step(np.zeros(2, np.float32))
    base, yaw = d.xpos[env.base_body_id].copy(), env._base_yaw()
    c, s = math.cos(-yaw), math.sin(-yaw)
    to_body = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # undo the tiny residual yaw

    eyes, reps = [], []
    for eye in EYES:
        # L1/L2/L3 neurons of a column sample the image at the same position, so each unique
        # position has one value; any neuron at that position can stand for it.
        pts, idx = [], []
        for layer in ("L1", "L2", "L3"):
            grid = getattr(agent, f"grid_{layer}_{eye}").view(-1, 2).cpu().numpy()
            start = agent.input_splits[f"pr_{layer}_{eye}"][0]
            pts.append(grid)
            idx.append(start + np.arange(len(grid)))
        pts, idx = np.concatenate(pts), np.concatenate(idx)
        uniq, first = np.unique(np.round(pts, 6), axis=0, return_index=True)
        reps.append(idx[first])

        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{eye}")
        R = to_body @ d.cam_xmat[cid].reshape(3, 3)
        pos = to_body @ (d.cam_xpos[cid] - base)
        eyes.append({
            "name": eye,
            # Offset from the base in the body frame, except z, which is the camera's height.
            "cam_pos": _round([pos[0], pos[1], d.cam_xpos[cid][2]]),
            "right": _round(R[:, 0]), "up": _round(R[:, 1]), "fwd": _round(-R[:, 2]),
            "tan_half_fov": round(math.tan(math.radians(m.cam_fovy[cid]) / 2), 6),
            "image_size": int(env.W),
            "dots": _round(uniq, 6),  # grid_sample coordinates: x right, y down, in [-1, 1]
        })
    return eyes, np.concatenate(reps)


def scene_description(env, eyes: list, dot_radius: float, max_steps: int, vision) -> dict:
    m = env.model
    lights = [{"dir": _round(m.light_dir[i]), "diffuse": round(float(m.light_diffuse[i][0]), 5)} for i in range(m.nlight)]
    ambient = float(m.vis.headlight.ambient[0] + sum(m.light_ambient[i][0] for i in range(m.nlight)))

    floor_mat = m.geom_matid[_geom(m, "floor")]
    first, second = _texture_colors(m, "texplane")
    floor = {"square": 1.0 / float(m.mat_texrepeat[floor_mat][0]),
             # (even, odd) squares; the floor's even squares show the texture's second colour
             "albedo": [_round(m.mat_rgba[floor_mat][:3] * second), _round(m.mat_rgba[floor_mat][:3] * first)]}

    walls = []
    first, second = _texture_colors(m, "texwall")
    for name, even_first in WALL_EVEN_IS_FIRST.items():
        g = _geom(m, name)
        pos, size, mat = m.geom_pos[g], m.geom_size[g], m.geom_matid[g]
        axis = 0 if size[0] < size[1] else 1  # the thin axis is the face normal's
        sign = 1.0 if pos[axis] > 0 else -1.0
        band_axis = 1 - axis
        even, odd = (first, second) if even_first else (second, first)
        walls.append({
            "axis": axis, "pos": round(float(pos[axis] - sign * size[axis]), 5),  # inner face
            "normal": [-sign if k == axis else 0.0 for k in range(3)],
            "band_axis": band_axis, "band": 1.0 / float(m.mat_texrepeat[mat][band_axis]),
            "zmax": round(float(pos[2] + size[2]), 5),
            "albedo": [_round(m.mat_rgba[mat][:3] * even), _round(m.mat_rgba[mat][:3] * odd)],
        })

    obstacle_mat = m.geom_matid[env.obstacle_ids[0]]
    return {
        "dt": m.opt.timestep * env.frame_skip,
        "max_steps": max_steps,
        "vision": list(vision),
        "arena_half_extent": env.arena,
        "robot_radius": float(m.geom_size[_geom(m, "base_geom")][0]),
        "goal_radius": env.goal_radius,
        "ball": {"radius": float(m.site_size[env.goal_site_id][0]), "z": float(m.site_pos[env.goal_site_id][2])},
        "motion": measure_motion(env),
        "light": {"ambient": round(ambient, 5), "lights": lights,
                  "headlight_diffuse": round(float(m.vis.headlight.diffuse[0]), 5), "gray": GRAY_WEIGHTS},
        "floor": floor,
        "walls": walls,
        "obstacle_texture": {"square": 1.0 / float(m.mat_texrepeat[obstacle_mat][0]),
                             "tex": [list(OBSTACLE_TEXTURE[0]), list(OBSTACLE_TEXTURE[1])]},
        "eyes": eyes,
        "dot_radius": round(dot_radius, 5),  # as a fraction of the eye panel's size
    }


def layout_description(env) -> dict:
    m, d = env.model, env.data
    obstacles = []
    for gid, bid, (x, y) in zip(env.obstacle_ids, env.obstacle_body_ids, env._obstacle_xy[:env.n_obstacles]):
        r, half_h = m.geom_size[gid][:2]
        zc = d.xpos[bid][2]
        obstacles.append(_round([x, y, r, max(0.0, zc - half_h), zc + half_h, *m.geom_rgba[gid][:3]], 4))
    return {"obstacles": obstacles, "goal": _round(env._goal_xy, 4),
            "start": _round([*env._base_xy(), env._base_yaw()], 5)}


@torch.no_grad()
def record_run(bot: AgentRacer, seed: int, reps: np.ndarray) -> tuple:
    """Roll the agent out on one layout, as the desktop game does, and record what it did."""
    bot.reset(seed)
    layout = layout_description(bot.env)
    poses, bumps, frames = [], [], []

    def observe():
        poses.append([*bot.env._base_xy(), bot.env._base_yaw()])
        bumps.append(int((bot.segment("tactile_left") > 0).any()) | int((bot.segment("tactile_right") > 0).any()) << 1)
        # Same quantisation as PhotoreceptorView.render().
        frames.append((np.clip(bot.x_np[reps], 0.0, 1.0) * 255.0).astype(np.uint8))

    observe()
    while not bot.finished:
        bot.step(bot.act())
        observe()

    frames = np.stack(frames)
    deltas = np.concatenate([frames[:1], np.diff(frames, axis=0)])  # uint8 arithmetic wraps mod 256
    payload = (np.asarray(poses, np.float32).tobytes() + np.asarray(bumps, np.uint8).tobytes()
               + deltas.astype(np.uint8).tobytes())
    layout["agent"] = {"steps": bot.steps, "reached": bool(bot.reached), "bumps": int(bot.bumps),
                       "path": round(path_length(bot.path), 3)}
    return layout, gzip.compress(payload, compresslevel=9)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a connectome RNN checkpoint's runs for the browser version of the "
                    "race-the-agent game, and export the scene the browser simulates."
    )
    parser.add_argument("checkpoint", help="Path to the .pt checkpoint to record. Required")
    parser.add_argument("--connectome", default=None, metavar="EDGE_CSV",
                        help="Connectome edge-list CSV the checkpoint was trained on, as in "
                             "run_connectome_rnn_checkpoint.py.")
    parser.add_argument("--vision", choices=list(VISION_CONDITIONS), default="11",
                        help="Vision condition for both players, as in play_vs_connectome_rnn.py "
                             "(default: 11, full vision).")
    parser.add_argument("--agent-name", default="FLYNN", help="Name shown for the agent (default: FLYNN).")
    parser.add_argument("--layouts", type=int, default=50,
                        help="Number of layouts to record, seeds --first-seed onwards (default: 50).")
    parser.add_argument("--first-seed", type=int, default=1, help="Seed of the first layout (default: 1).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Time limit in 0.02 s control steps (default: {DEFAULT_MAX_STEPS} = 30 s).")
    parser.add_argument("--out", default=OUT_DIR, help=f"Output directory (default: {OUT_DIR}).")
    return parser.parse_args()


def main():
    args = parse_args()
    edge_path = resolve_connectome_path(args.connectome)
    device = get_device()
    agent, _ = _load_agent(args.checkpoint, device=device, dtype=DTYPE, edge_path=edge_path)
    vision = VISION_CONDITIONS[args.vision]

    env = _make_env(render_mode=None, n_obstacles=N_OBSTACLES)
    env.max_steps = args.max_steps
    env.stall_limit = args.max_steps + 1  # as in play_vs_connectome_rnn.py
    try:
        eyes, reps = eye_geometry(agent, env)
        pr_view = PhotoreceptorView(agent)
        scene = scene_description(env, eyes, pr_view.radius / pr_view.eye_size, args.max_steps, vision)
        print(f"[export] {sum(len(e['dots']) for e in eyes)} photoreceptor positions; motion: {scene['motion']}")

        os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
        bot = AgentRacer(args.agent_name, None, env, agent, vision, device, DTYPE)
        layouts, total = [], 0
        for seed in range(args.first_seed, args.first_seed + args.layouts):
            layout, blob = record_run(bot, seed, reps)
            layout["seed"] = seed
            layout["run"] = f"runs/{seed}.bin"
            with open(os.path.join(args.out, layout["run"]), "wb") as f:
                f.write(blob)
            layouts.append(layout)
            total += len(blob)
            a = layout["agent"]
            print(f"[export] layout {seed}: {'goal' if a['reached'] else 'time up'} in {a['steps']} steps, "
                  f"{a['bumps']} bumps, {len(blob) / 1024:.0f} KB")

        game = {"agent_name": args.agent_name, "checkpoint": os.path.basename(args.checkpoint),
                "scene": scene, "layouts": layouts}
        with open(os.path.join(args.out, "game.json"), "w") as f:
            json.dump(game, f, separators=(",", ":"))
        reached = sum(l["agent"]["reached"] for l in layouts)
        print(f"[export] {len(layouts)} layouts ({reached} reached by the agent), runs {total / 2**20:.1f} MB "
              f"-> {args.out}")
    finally:
        env.renderer.close()
        env.close()


if __name__ == "__main__":
    main()
