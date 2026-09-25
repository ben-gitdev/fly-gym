"""
Race a trained connectome RNN (FLYNN / SmallWorldNet) checkpoint to the goal.

You and the agent each drive the eval robot in your own copy of the eval environment.  Both
copies are reset with the same seed, so the two robots start from the same pose, in the same
obstacle layout, with the same goal (the white ball).  The agent is rolled out exactly as in
run_connectome_rnn_checkpoint.py.

You get exactly the information the agent gets.  Every HUD element is drawn from the agent's
input vector x (ConnectomeAgent.obs_to_x), computed from your own robot's observation:

  * eye view    -- the photoreceptor-sampled eye input (PhotoreceptorView), not the raw cameras
  * goal arrow  -- the "wind" direction fed to the Johnston's-organ neurons: the goal's
                   direction relative to your heading, with no distance
  * bump lamps  -- the left/right head-bristle (tactile) input

Controls: Up/Down drive forward/backward at the agent's cruise speed; Left/Right turn (in
place when neither Up nor Down is held).  SPACE starts a round / moves on to the next layout,
P pauses, R restarts the layout, N skips to a new layout, ESC quits.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional

import mujoco
import numpy as np
import pygame
import torch

from core.utils import get_device, obs_to_torch
from run_connectome_rnn_checkpoint import (
    VISION_CONDITIONS,
    PhotoreceptorView,
    _load_agent,
    _make_env,
    resolve_connectome_path,
)
from shared_config import DTYPE, N_OBSTACLES

# The VFH*+PID teacher always commands v_fwd = 0.7 (models/teacher_analytic_model.py) and the
# trained agent imitates it: its median velocity output is ~0.70, i.e. ~1.25 m/s on the ground.
# The player drives at exactly this speed, forward or backward.
PLAYER_SPEED = 0.7
# Heading-angle command while Left/Right is held (~100 deg/s).  vel_angle_to_action() turns it
# into a wheel differential of +-0.5 * PLAYER_TURN, which keeps both wheels inside [-1, 1] at
# PLAYER_SPEED, so turning never changes the forward speed.
PLAYER_TURN = 0.4
# 30 s at the env's 0.02 s control step.  Longer than the eval's 600-step (12 s) limit: the
# agent typically needs 5-8 s, a human driving from its eye input needs more.
DEFAULT_MAX_STEPS = 1500

VISION_LABELS = {"11": "full vision", "10": "left eye only", "01": "right eye only", "00": "blind"}

BG = (22, 24, 28)
PANEL = (38, 41, 48)
TEXT = (232, 234, 238)
MUTED = (140, 146, 158)
PLAYER_COLOR = (242, 142, 43)
AGENT_COLOR = (86, 165, 235)
GOOD = (96, 204, 116)
BAD = (232, 84, 72)
BUMP_ON = (240, 72, 60)
BUMP_OFF = (64, 58, 60)
GRAY_PALETTE = [(i, i, i) for i in range(256)]

MARGIN = 20
HEADER_H = 48
TITLE_H = 36
HUD_H = 124
FOOTER_H = 36


class Racer:
    """One robot in its own copy of the eval env, plus what the HUD and results screen need."""

    def __init__(self, name, color, env, agent, vision, device, dtype):
        self.name, self.color = name, color
        self.env, self.agent, self.vision = env, agent, vision
        self.device, self.dtype = device, dtype

    def reset(self, seed: int) -> None:
        # Start every round from MuJoCo's initial state, as a freshly built env would.
        # env.reset() only re-poses the base and wheels, so the caster's swivel angle would
        # otherwise carry over from the previous round -- differently for the two robots.
        mujoco.mj_resetData(self.env.model, self.env.data)
        self.obs, _ = self.env.reset(seed=seed)
        self.steps = 0
        self.finished = self.reached = False
        self.bumps = 0
        self.path = [self.env._base_xy()]
        self._observe()

    def _observe(self) -> None:
        # The agent's input vector for this robot's current observation.  AgentRacer feeds it
        # to the network; for both racers it is the only thing the HUD is drawn from.
        # obs_to_x is a pure function of the observation (the stateful retina filtering runs
        # later, inside agent.step), so calling it for the player leaves the agent untouched.
        self.obs_t = obs_to_torch(self.obs, device=self.device, dtype=self.dtype, vision=self.vision)
        self.x = self.agent.obs_to_x(self.obs_t)
        self.x_np = self.x[0].float().cpu().numpy()

    def segment(self, name: str) -> np.ndarray:
        start, end = self.agent.input_splits.get(name, (0, 0))
        return self.x_np[start:end]

    def step(self, action: np.ndarray) -> None:
        was_colliding = bool(self.obs["sensors"]["collision"])
        self.obs, _, done, trunc, _ = self.env.step(action)
        self.steps += 1
        self.path.append(self.env._base_xy())
        self.bumps += bool(self.obs["sensors"]["collision"]) and not was_colliding
        if done or trunc:
            self.finished = True
            # Same success test as rollout_episode().
            final_dist = float(np.linalg.norm(self.env._goal_xy - self.env._base_xy()))
            self.reached = final_dist < self.env.goal_radius
        self._observe()


class AgentRacer(Racer):
    """The checkpoint driving, stepped exactly as in rollout_episode()."""

    def reset(self, seed: int) -> None:
        self.agent.reset_vision_state()
        self.h = torch.zeros(1, self.agent.cell.N, device=self.device, dtype=self.dtype)
        super().reset(seed)

    def act(self) -> np.ndarray:
        self.h, action = self.agent.step(self.h, self.obs_t, x=self.x)
        return action.squeeze(0).cpu().numpy()


def keys_to_action(keys) -> np.ndarray:
    """Arrow keys -> the same [velocity, heading angle] action the agent outputs."""
    drive = keys[pygame.K_UP] - keys[pygame.K_DOWN]
    turn = keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]  # heading angle > 0 turns left
    return np.array([PLAYER_SPEED * drive, PLAYER_TURN * turn], dtype=np.float32)


def path_length(path) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum()) if len(path) > 1 else 0.0


class Game:
    def __init__(self, player: Racer, bot: AgentRacer, pr_view: PhotoreceptorView,
                 seed: int, vision_label: str, fps: Optional[int] = None):
        self.player, self.bot = player, bot
        self.racers = (player, bot)
        self.pr_view = pr_view
        self.vision_label = vision_label
        self.dt = player.env.model.opt.timestep * player.env.frame_skip
        self.fps = fps or round(1.0 / self.dt)  # default: one control step per dt, real time
        self.score = {player: 0, bot: 0}

        self.eye = pr_view.eye_size
        self.col_w = 2 * self.eye + 4  # PhotoreceptorView.render(): two eyes + separator
        self.col_x = (MARGIN, 2 * MARGIN + self.col_w)
        self.view_y = HEADER_H + TITLE_H
        self.hud_y = self.view_y + self.eye

        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption(f"{bot.name} vs {player.name}")
        self.screen = pygame.display.set_mode(
            (3 * MARGIN + 2 * self.col_w, self.hud_y + HUD_H + FOOTER_H)
        )
        self.font = pygame.font.SysFont("segoeui,helvetica,arial", 18)
        self.font_bold = pygame.font.SysFont("segoeui,helvetica,arial", 20, bold=True)
        self.font_big = pygame.font.SysFont("segoeui,helvetica,arial", 34, bold=True)
        self.new_round(seed)

    # ---------------- Game flow ----------------
    def new_round(self, seed: int) -> None:
        self.seed = seed
        for racer in self.racers:
            racer.reset(seed)
        self.state = "ready"  # -> "playing" -> "results"
        self.paused = False

    def on_key(self, key: int) -> None:
        if key == pygame.K_SPACE and self.state == "ready":
            self.state = "playing"
        elif key == pygame.K_SPACE and self.state == "results":
            self.new_round(self.seed + 1)
        elif key == pygame.K_p and self.state == "playing":
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.new_round(self.seed)
        elif key == pygame.K_n:
            self.new_round(self.seed + 1)

    def tick(self, player_action: np.ndarray) -> None:
        """Advance both robots one control step (a robot that has finished stays put)."""
        if not self.player.finished:
            self.player.step(player_action)
        if not self.bot.finished:
            self.bot.step(self.bot.act())
        if self.player.finished and self.bot.finished:
            self.finish_round()

    def finish_round(self) -> None:
        arrived = [r for r in self.racers if r.reached]
        best = min((r.steps for r in arrived), default=None)
        winners = [r for r in arrived if r.steps == best]
        if not winners:
            self.result_text, self.result_color = "Nobody reached the goal", MUTED
        elif len(winners) > 1:
            self.result_text, self.result_color = "Dead heat!", TEXT
        else:
            w = winners[0]
            self.score[w] += 1
            self.result_text = f"{w.name} {'win' if w is self.player else 'wins'}!"
            self.result_color = w.color
        self.state = "results"

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    self.on_key(event.key)
            if self.state == "playing" and not self.paused:
                self.tick(keys_to_action(pygame.key.get_pressed()))
            self.draw()
            # tick() sleeps via SDL_Delay, whose ~15 ms granularity on Windows turns a 20 ms
            # frame budget into ~35 ms; the busy loop holds real time.
            clock.tick_busy_loop(self.fps)

    # ---------------- Drawing ----------------
    def text(self, s: str, font, color, pos, anchor: str = "topleft") -> pygame.Rect:
        img = font.render(s, True, color)
        rect = img.get_rect(**{anchor: pos})
        self.screen.blit(img, rect)
        return rect

    def draw(self) -> None:
        self.screen.fill(BG)
        self.draw_header()
        for racer, x0 in zip(self.racers, self.col_x):
            self.draw_racer(racer, x0)
        self.draw_footer()
        if self.state == "ready":
            self.banner("Press SPACE to start",
                        f"Beat {self.bot.name} to the white ball. You see what it sees: "
                        "its eye input, goal arrow and bump sensors.")
        elif self.state == "playing" and self.paused:
            self.banner("Paused", "Press P to resume")
        elif self.state == "results":
            self.draw_results()
        pygame.display.flip()

    def draw_header(self) -> None:
        mid = HEADER_H // 2
        self.text(f"Layout {self.seed}   |   {self.vision_label}", self.font, MUTED, (MARGIN, mid), "midleft")
        x = self.screen.get_width() - MARGIN
        for s, color in reversed([
            (f"{self.player.name} {self.score[self.player]}", self.player.color),
            ("  :  ", MUTED),
            (f"{self.score[self.bot]} {self.bot.name}", self.bot.color),
        ]):
            x = self.text(s, self.font_bold, color, (x, mid), "midright").left

    def draw_racer(self, racer: Racer, x0: int) -> None:
        s = self.screen
        mid = HEADER_H + TITLE_H // 2 + 1
        pygame.draw.rect(s, racer.color, (x0, HEADER_H, self.col_w, 3))
        self.text(racer.name, self.font_bold, racer.color, (x0, mid), "midleft")
        t = racer.steps * self.dt
        if racer.reached:
            status, color = f"GOAL  {t:.2f} s", GOOD
        elif racer.finished:
            status, color = "TIME UP", BAD
        else:
            status, color = f"{t:.2f} s    bumps {racer.bumps}", TEXT
        self.text(status, self.font_bold, color, (x0 + self.col_w, mid), "midright")

        img = self.pr_view.render(racer.x)
        eyes = pygame.image.frombuffer(img, (img.shape[1], img.shape[0]), "P")  # 8-bit gray
        eyes.set_palette(GRAY_PALETTE)
        s.blit(eyes, (x0, self.view_y))

        # The goal compass sits under the seam between the eyes (straight ahead), with the
        # left/right bump lamps on their own side.
        cx, cy = x0 + self.col_w // 2, self.hud_y + HUD_H // 2
        r = HUD_H // 2 - 16
        self.draw_compass(racer, cx, cy, r)
        for side, sign in (("L", -1), ("R", +1)):
            on = bool((racer.segment("tactile_left" if side == "L" else "tactile_right") > 0).any())
            lamp = pygame.Rect(0, 0, 100, 36)
            lamp.center = (cx + sign * (r + 95), cy)
            pygame.draw.rect(s, BUMP_ON if on else BUMP_OFF, lamp, border_radius=8)
            self.text(f"{side} BUMP", self.font_bold, TEXT if on else MUTED, lamp.center, "center")

    def draw_compass(self, racer: Racer, cx: int, cy: int, r: int) -> None:
        s = self.screen
        pygame.draw.circle(s, PANEL, (cx, cy), r)
        pygame.draw.circle(s, MUTED, (cx, cy), r, 2)
        pygame.draw.line(s, MUTED, (cx, cy - r), (cx, cy - r + 8), 2)  # "ahead" tick
        wind = racer.segment("wind")
        if len(wind) < 2:
            return
        # wind[:2] = (cos, sin) of the goal bearing relative to the heading, > 0 to the left.
        # On screen ahead is up and left is left (y grows downward).
        a = math.atan2(wind[1], wind[0])
        c, sn = math.cos(a), math.sin(a)
        arrow = [(0.0, -0.82), (0.34, -0.25), (0.12, -0.25), (0.12, 0.62),
                 (-0.12, 0.62), (-0.12, -0.25), (-0.34, -0.25)]
        pygame.draw.polygon(s, racer.color, [(cx + r * (px * c + py * sn), cy + r * (py * c - px * sn))
                                             for px, py in arrow])

    def draw_footer(self) -> None:
        if self.state == "results":
            return  # the results panel shows its own key hints
        hint = {
            "ready": "SPACE start    Arrow keys drive    N new layout    ESC quit",
            "playing": "Up/Down forward/back    Left/Right turn    P pause    R restart    N new layout    ESC quit",
        }[self.state]
        w, h = self.screen.get_size()
        self.text(hint, self.font, MUTED, (w // 2, h - FOOTER_H // 2), "center")

    def banner(self, title: str, subtitle: str) -> None:
        w = self.screen.get_width()
        box = pygame.Surface((w - 2 * MARGIN, 120), pygame.SRCALPHA)
        pygame.draw.rect(box, (10, 11, 14, 220), box.get_rect(), border_radius=12)
        rect = self.screen.blit(box, box.get_rect(center=(w // 2, self.view_y + self.eye // 2)))
        self.text(title, self.font_big, TEXT, (rect.centerx, rect.centery - 18), "center")
        self.text(subtitle, self.font, MUTED, (rect.centerx, rect.centery + 26), "center")

    def draw_results(self) -> None:
        s = self.screen
        w, h = s.get_size()
        shade = pygame.Surface((w, h), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        s.blit(shade, (0, 0))
        box = pygame.Rect(0, 0, min(w - 2 * MARGIN, 780), min(h - 2 * MARGIN, 420))
        box.center = (w // 2, h // 2)
        pygame.draw.rect(s, PANEL, box, border_radius=12)
        self.text(self.result_text, self.font_big, self.result_color, (box.centerx, box.top + 34), "center")

        map_rect = pygame.Rect(box.left + 24, box.top + 70, box.height - 100, box.height - 100)
        self.draw_map(map_rect)

        x_label = map_rect.right + 30
        rows = ["Result", "Time", "Path", "Bumps"]
        for i, label in enumerate(rows, start=1):
            self.text(label, self.font, MUTED, (x_label, map_rect.top + 40 * i), "topleft")
        for racer, x in zip(self.racers, (x_label + 150, x_label + 280)):
            cells = [
                ("goal", GOOD) if racer.reached else ("time up", BAD),
                (f"{racer.steps * self.dt:.2f} s" if racer.reached else "--", TEXT),
                (f"{path_length(racer.path):.1f} m", TEXT),
                (str(racer.bumps), TEXT),
            ]
            self.text(racer.name, self.font_bold, racer.color, (x, map_rect.top), "midtop")
            for i, (cell, color) in enumerate(cells, start=1):
                self.text(cell, self.font, color, (x, map_rect.top + 40 * i), "midtop")
        self.text("White ring: start    Green ring: goal zone", self.font, MUTED,
                  (x_label, map_rect.bottom - 40), "bottomleft")
        self.text("SPACE next layout    R retry    ESC quit", self.font_bold, TEXT,
                  (x_label, map_rect.bottom), "bottomleft")

    def draw_map(self, rect: pygame.Rect) -> None:
        """Top-down view of the layout (identical in both envs) and both driven paths."""
        s = self.screen
        env = self.player.env
        a = env.arena
        k = rect.width / (2 * a)

        def px(xy):  # world metres -> screen pixels; pygame rejects numpy scalars
            return (float(rect.left + (xy[0] + a) * k), float(rect.top + (a - xy[1]) * k))

        pygame.draw.rect(s, (58, 62, 70), rect)
        pygame.draw.rect(s, MUTED, rect, 3)
        for gid, xy in zip(env.obstacle_ids, env._obstacle_xy[:env.n_obstacles]):
            color = [int(255 * c) for c in env.model.geom_rgba[gid, :3]]
            pygame.draw.circle(s, color, px(xy), float(env.model.geom_size[gid, 0] * k))
        goal = px(env._goal_xy)
        pygame.draw.circle(s, GOOD, goal, env.goal_radius * k, 2)
        pygame.draw.circle(s, (255, 255, 255), goal, float(env.model.site_size[env.goal_site_id, 0] * k))
        for racer in self.racers:
            pts = [px(p) for p in racer.path]
            if len(pts) > 1:
                pygame.draw.lines(s, racer.color, False, pts, 3)
            pygame.draw.circle(s, racer.color, pts[-1], 5)
        pygame.draw.circle(s, (255, 255, 255), px(self.player.path[0]), 6, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Race a trained connectome RNN (FLYNN/SmallWorldNet) checkpoint: you drive "
                    "the same robot with the arrow keys, seeing only what the agent sees."
    )
    parser.add_argument("checkpoint", help="Path to the .pt checkpoint to race against. Required")
    parser.add_argument(
        "--connectome",
        default=None,
        metavar="EDGE_CSV",
        help="Connectome edge-list CSV the checkpoint was trained on, as in "
             "run_connectome_rnn_checkpoint.py. Defaults to shared_config's EDGE_PATH, with a "
             "warning that it may not match the checkpoint.",
    )
    parser.add_argument(
        "--vision",
        choices=list(VISION_CONDITIONS),
        default="11",
        help="Vision condition for both you and the agent, as <left><right> with 1 = eye "
             "enabled: 11 = full vision (default), 10 = left eye only, 01 = right eye only, "
             "00 = blind.",
    )
    parser.add_argument("--agent-name", default="FLYNN", help="Name shown for the agent (default: FLYNN).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Layout seed of the first round; each new layout adds 1 (default: 1).")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Time limit in 0.02 s control steps (default: {DEFAULT_MAX_STEPS} = 30 s).")
    parser.add_argument("--eye-size", type=int, default=256,
                        help="Size in pixels of each eye panel (default: 256).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Control steps per second (default: 50, i.e. real time). Lower "
                             "slows the whole game down, for both players.")
    return parser.parse_args()


def main():
    args = parse_args()
    edge_path = resolve_connectome_path(args.connectome)
    device = get_device()
    print("[game] Building the connectome and loading the checkpoint (this takes a while)...")
    agent, _ = _load_agent(args.checkpoint, device=device, dtype=DTYPE, edge_path=edge_path)
    vision = VISION_CONDITIONS[args.vision]

    racers = []
    for cls, name, color in ((Racer, "YOU", PLAYER_COLOR), (AgentRacer, args.agent_name, AGENT_COLOR)):
        env = _make_env(render_mode=None, n_obstacles=N_OBSTACLES)
        env.max_steps = args.max_steps
        # The eval ends an episode after 50 steps without progress.  In a race a stuck robot
        # just loses time, so only the goal and the time limit end a run.
        env.stall_limit = args.max_steps + 1
        racers.append(cls(name, color, env, agent, vision, device, DTYPE))

    try:
        game = Game(*racers, PhotoreceptorView(agent, eye_size=args.eye_size), seed=args.seed,
                    vision_label=VISION_LABELS[args.vision], fps=args.fps)
        with torch.no_grad():
            game.run()
    finally:
        pygame.quit()
        for racer in racers:
            racer.env.renderer.close()
            racer.env.close()


if __name__ == "__main__":
    main()
