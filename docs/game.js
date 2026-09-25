// Browser version of play_vs_connectome_rnn.py: the same screen, rules and controls.  Your robot
// is simulated here (sim.js); the agent's side is a run recorded by export_web_game.py.

import { Eyes, Recording, Robot } from "./sim.js";

const EYE = 256; // eye panel size, px
const MARGIN = 20, HEADER_H = 48, TITLE_H = 36, HUD_H = 124, FOOTER_H = 36, GAP = 14;
const COL_W = 2 * EYE + 4; // two eyes + separator, as PhotoreceptorView.render()
const BLOCK_H = TITLE_H + EYE + HUD_H; // one racer: title bar, eye panel, compass row
const BACKGROUND = 40; // PhotoreceptorView's background gray

const C = {
  bg: "#16181c", panel: "#262930", text: "#e8eaee", muted: "#8c929e",
  player: "#f28e2b", agent: "#56a5eb", good: "#60cc74", bad: "#e85448",
  bumpOn: "#f0483c", bumpOff: "#403a3c", floor: "#3a3e46",
};
const FONT = "system-ui, 'Segoe UI', Helvetica, Arial, sans-serif";
const VISION_LABELS = { "true,true": "full vision", "true,false": "left eye only", "false,true": "right eye only", "false,false": "blind" };
const GAME_KEYS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "KeyP", "KeyR", "KeyN"]);

// The two racers side by side on wide screens, stacked on narrow portrait ones (phones).  With
// touch controls there is no key-hint footer.
function computeLayout(tall, touch) {
  const footer = touch ? 0 : FOOTER_H;
  return tall
    ? { tall, touch, footer, width: 2 * MARGIN + COL_W, height: HEADER_H + 2 * BLOCK_H + GAP + footer,
        blocks: [[MARGIN, HEADER_H], [MARGIN, HEADER_H + BLOCK_H + GAP]] }
    : { tall, touch, footer, width: 3 * MARGIN + 2 * COL_W, height: HEADER_H + BLOCK_H + footer,
        blocks: [[MARGIN, HEADER_H], [2 * MARGIN + COL_W, HEADER_H]] };
}

// Both eyes' photoreceptor values drawn as dots at their grid positions, like PhotoreceptorView:
// a fixed pixel -> photoreceptor label map, filled from the values each frame.
class EyePanel {
  constructor(scene) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = COL_W;
    this.canvas.height = EYE;
    this.ctx = this.canvas.getContext("2d");
    this.image = this.ctx.createImageData(COL_W, EYE);
    this.labels = new Int32Array(COL_W * EYE).fill(-1);
    for (let y = 0; y < EYE; y++) for (let x = EYE; x < EYE + 4; x++) this.labels[y * COL_W + x] = -2; // separator
    const r = Math.max(1, Math.round(scene.dot_radius * EYE));
    let base = 0;
    scene.eyes.forEach((eye, e) => {
      const x0 = e * (EYE + 4);
      eye.dots.forEach(([gx, gy], i) => {
        const cx = Math.round(((gx + 1) / 2) * (EYE - 1)), cy = Math.round(((gy + 1) / 2) * (EYE - 1));
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            const px = cx + dx, py = cy + dy;
            if (dx * dx + dy * dy <= r * r && px >= 0 && px < EYE && py >= 0 && py < EYE) this.labels[py * COL_W + x0 + px] = base + i;
          }
        }
      });
      base += eye.dots.length;
    });
  }

  draw(ctx, values, x, y) {
    const d = this.image.data, L = this.labels;
    for (let p = 0, q = 0; p < L.length; p++, q += 4) {
      const l = L[p];
      d[q] = d[q + 1] = d[q + 2] = l >= 0 ? values[l] : l === -2 ? 255 : BACKGROUND;
      d[q + 3] = 255;
    }
    this.ctx.putImageData(this.image, 0, 0);
    ctx.drawImage(this.canvas, x, y);
  }
}

// On-screen buttons for touch screens (#touch-controls in index.html).  Drive/turn buttons count as
// held keys while a finger is on them, so several can be held at once; the others are taps.
class TouchControls {
  constructor(game, root) {
    this.game = game;
    this.primary = root.querySelector('[data-tap="primary"]');
    for (const b of root.querySelectorAll("[data-hold]")) {
      const code = b.dataset.hold;
      const release = () => {
        b.classList.remove("held");
        game.keys.delete(code);
      };
      b.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        b.setPointerCapture(e.pointerId); // keep holding even if the finger slides off
        b.classList.add("held");
        game.keys.add(code);
      });
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(type, release);
    }
    for (const b of root.querySelectorAll("[data-tap]")) {
      b.addEventListener("click", () => (b.dataset.tap === "primary" ? game.primaryAction() : game.onKey(b.dataset.tap)));
    }
    // Never keep focus: a focused button would also "click" on a keyboard's Space or Enter.
    for (const b of root.querySelectorAll("button")) b.addEventListener("click", () => b.blur());
    root.addEventListener("contextmenu", (e) => e.preventDefault()); // no long-press menu
  }

  sync() {
    const g = this.game, label = { loading: "Loading", error: "Retry", ready: "Start", results: "Next" }[g.state]
      ?? (g.paused ? "Resume" : "Pause");
    if (this.primary.textContent !== label) this.primary.textContent = label;
    this.primary.disabled = g.state === "loading";
  }
}

class Game {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.data = data;
    this.scene = data.scene;
    this.dt = this.scene.dt;
    this.agentName = data.agent_name;
    this.visionLabel = VISION_LABELS[this.scene.vision.join(",")];
    this.eyes = new Eyes(this.scene);
    this.panel = new EyePanel(this.scene);
    this.playerEyes = new Uint8Array(this.eyes.nDots);
    this.score = { player: 0, agent: 0 };
    this.recordings = new Map();
    this.keys = new Set();
    this.state = "loading";
    this.message = "Loading...";
    this.ctx = canvas.getContext("2d");
    this.touch = false;
    this.setTouch(matchMedia("(any-pointer: coarse)").matches);
    const controls = document.getElementById("touch-controls");
    this.controls = controls ? new TouchControls(this, controls) : null;

    window.addEventListener("keydown", (e) => {
      if (!GAME_KEYS.has(e.code)) return;
      e.preventDefault();
      if (!e.repeat) this.onKey(e.code);
      this.keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
    // A touch anywhere switches to touch controls (for touch screens the media query misses).
    window.addEventListener("pointerdown", (e) => e.pointerType === "touch" && this.setTouch(true), { capture: true });
    window.addEventListener("resize", () => this.applyLayout());
    // Tapping/clicking the game starts a round, or moves on from the results.
    canvas.addEventListener("click", () => (this.state === "ready" || this.state === "results") && this.primaryAction());
  }

  setTouch(on) {
    if (this.touch === on && this.L) return;
    this.touch = on;
    document.body.classList.toggle("touch", on);
    this.applyLayout();
  }

  applyLayout() {
    const tall = window.innerWidth < 800 && window.innerHeight > window.innerWidth;
    const dpr = window.devicePixelRatio || 1;
    if (this.L && this.L.tall === tall && this.L.touch === this.touch && this.dpr === dpr) return;
    const L = computeLayout(tall, this.touch);
    [this.L, this.dpr] = [L, dpr];
    this.canvas.width = Math.round(L.width * dpr); // resizing also resets the context state
    this.canvas.height = Math.round(L.height * dpr);
    this.canvas.style.aspectRatio = `${L.width} / ${L.height}`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.ctx.imageSmoothingEnabled = false;
  }

  // ---------------- game flow ----------------
  recording(index) {
    if (!this.recordings.has(index)) {
      const layout = this.data.layouts[index];
      this.recordings.set(index, Recording.load(`data/${layout.run}`, layout, this.eyes.nDots));
    }
    return this.recordings.get(index);
  }

  async newRound(index) {
    const n = this.data.layouts.length;
    index = ((index % n) + n) % n;
    const layout = this.data.layouts[index];
    this.loading = index;
    this.state = "loading";
    this.message = `Loading layout ${layout.seed}...`;
    this.paused = false;
    let rec;
    try {
      rec = await this.recording(index);
    } catch (err) {
      if (this.loading !== index) return;
      this.message = `Could not load layout ${layout.seed} (${err.message}). ${this.touch ? "Retry, or tap New layout." : "R retries, N skips."}`;
      this.recordings.delete(index);
      this.state = "error";
      return;
    }
    if (this.loading !== index) return; // another round was started meanwhile
    // Switch everything at once, so the loading screen keeps showing the previous round.
    [this.index, this.layout, this.rec] = [index, layout, rec];
    this.player = new Robot(this.scene, layout);
    this.eyeStep = -1;
    this.k = 0; // the agent's step count
    this.state = "ready";
    // Keep this round's run and prefetch the next one.
    for (const key of this.recordings.keys()) if (key !== index && key !== (index + 1) % n) this.recordings.delete(key);
    this.recording((index + 1) % n).catch(() => this.recordings.delete((index + 1) % n));
  }

  onKey(code) {
    const current = this.state === "error" ? this.loading : this.index;
    if (code === "Space" && this.state === "ready") this.state = "playing";
    else if (code === "Space" && this.state === "results") this.newRound(current + 1);
    else if (code === "KeyP" && this.state === "playing") this.paused = !this.paused;
    else if (code === "KeyR" && this.state !== "loading") this.newRound(current);
    else if (code === "KeyN" && this.state !== "loading") this.newRound(current + 1);
  }

  // Start / Pause / Resume / Next / Retry, whichever the state calls for.
  primaryAction() {
    const code = { ready: "Space", results: "Space", playing: "KeyP", error: "KeyR" }[this.state];
    if (!code) return;
    if (this.touch && this.state === "ready") {
      // Bring the game and its buttons fully on screen.
      const box = this.canvas.parentElement.getBoundingClientRect();
      if (box.top < 0 || box.bottom > window.innerHeight) this.canvas.parentElement.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    this.onKey(code);
  }

  get agentFinished() {
    return this.k >= this.rec.steps;
  }

  tick() {
    const k = this.keys;
    const drive = (k.has("ArrowUp") ? 1 : 0) - (k.has("ArrowDown") ? 1 : 0);
    const turn = (k.has("ArrowLeft") ? 1 : 0) - (k.has("ArrowRight") ? 1 : 0);
    if (!this.player.finished) this.player.step(drive, turn);
    if (!this.agentFinished) this.k++;
    if (this.player.finished && this.agentFinished) this.finishRound();
  }

  finishRound() {
    const p = this.player, results = [
      { who: "player", name: "YOU", reached: p.reached, steps: p.steps, color: C.player },
      { who: "agent", name: this.agentName, reached: this.layout.agent.reached, steps: this.rec.steps, color: C.agent },
    ];
    const arrived = results.filter((r) => r.reached);
    const best = Math.min(...arrived.map((r) => r.steps));
    const winners = arrived.filter((r) => r.steps === best);
    if (winners.length === 0) [this.resultText, this.resultColor] = ["Nobody reached the goal", C.muted];
    else if (winners.length > 1) [this.resultText, this.resultColor] = ["Dead heat!", C.text];
    else {
      const w = winners[0];
      this.score[w.who]++;
      [this.resultText, this.resultColor] = [`${w.name} ${w.who === "player" ? "win" : "wins"}!`, w.color];
    }
    this.state = "results";
  }

  run() {
    let last = performance.now(), acc = 0;
    const frame = (now) => {
      const elapsed = Math.min(0.25, (now - last) / 1000);
      last = now;
      if (this.state === "playing" && !this.paused) {
        acc += elapsed;
        while (acc >= this.dt && this.state === "playing") {
          this.tick();
          acc -= this.dt;
        }
      } else acc = 0;
      this.draw();
      this.controls?.sync();
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // ---------------- drawing ----------------
  text(s, font, color, x, y, align = "left", baseline = "middle") {
    const ctx = this.ctx;
    ctx.font = `${font} ${FONT}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(s, x, y);
    return ctx.measureText(s).width;
  }

  // Split s into lines no wider than maxWidth.
  wrap(s, font, maxWidth) {
    this.ctx.font = `${font} ${FONT}`;
    const lines = [];
    let line = "";
    for (const word of s.split(" ")) {
      const next = line ? `${line} ${word}` : word;
      if (line && this.ctx.measureText(next).width > maxWidth) {
        lines.push(line);
        line = word;
      } else line = next;
    }
    return line ? [...lines, line] : lines;
  }

  draw() {
    const ctx = this.ctx, L = this.L;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, L.width, L.height);
    if (!this.player) {
      this.wrap(this.message, "600 22px", L.width - 2 * MARGIN).forEach((line, i, all) =>
        this.text(line, "600 22px", C.muted, L.width / 2, L.height / 2 + 30 * (i - (all.length - 1) / 2), "center"));
      return;
    }
    this.drawHeader();
    if (this.player.steps !== this.eyeStep) {
      this.eyes.render(this.layout, this.player.x, this.player.y, this.player.yaw, this.scene.vision, this.playerEyes);
      this.eyeStep = this.player.steps;
    }
    const p = this.player, r = this.rec, k = this.k;
    this.drawRacer(L.blocks[0], "YOU", C.player, {
      steps: p.steps, finished: p.finished, reached: p.reached, bumps: p.bumps,
      eyes: this.playerEyes, wind: p.windAngle, tactile: p.tactile,
    });
    this.drawRacer(L.blocks[1], this.agentName, C.agent, {
      steps: r.frame(k), finished: this.agentFinished, reached: this.agentFinished && this.layout.agent.reached,
      bumps: r.bumpsAt[r.frame(k)], eyes: r.eyeValues(k), wind: r.windAngle(k), tactile: r.tactile(k),
    });
    this.drawFooter();
    if (this.state === "loading" || this.state === "error") this.banner(this.message, "");
    else if (this.state === "ready") {
      this.banner(this.touch ? "Tap to start" : "Press SPACE to start",
        `Beat ${this.agentName} to the white ball. You see what it sees: its eye input, goal arrow and bump sensors.`);
    } else if (this.state === "playing" && this.paused) this.banner("Paused", this.touch ? "Tap Resume to carry on" : "Press P to resume");
    else if (this.state === "results") this.drawResults();
  }

  drawHeader() {
    const mid = HEADER_H / 2;
    this.text(`Layout ${this.layout.seed}   |   ${this.visionLabel}`, "18px", C.muted, MARGIN, mid);
    let x = this.L.width - MARGIN;
    for (const [s, color] of [[`${this.score.agent} ${this.agentName}`, C.agent], ["  :  ", C.muted], [`YOU ${this.score.player}`, C.player]]) {
      x -= this.text(s, "bold 20px", color, x, mid, "right");
    }
  }

  drawRacer([x0, y0], name, color, s) {
    const ctx = this.ctx, mid = y0 + TITLE_H / 2 + 1;
    ctx.fillStyle = color;
    ctx.fillRect(x0, y0, COL_W, 3);
    this.text(name, "bold 20px", color, x0, mid);
    const t = (s.steps * this.dt).toFixed(2);
    const [status, sc] = s.reached ? [`GOAL  ${t} s`, C.good] : s.finished ? ["TIME UP", C.bad] : [`${t} s    bumps ${s.bumps}`, C.text];
    this.text(status, "bold 20px", sc, x0 + COL_W, mid, "right");

    this.panel.draw(ctx, s.eyes, x0, y0 + TITLE_H);

    // Goal compass under the seam between the eyes (straight ahead), bump lamps on their side.
    const cx = x0 + COL_W / 2, cy = y0 + TITLE_H + EYE + HUD_H / 2, r = HUD_H / 2 - 16;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = C.panel;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = C.muted;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy - r + 8);
    ctx.stroke();
    // Wind: the goal's bearing relative to the heading, > 0 to the left; ahead is up on screen.
    const a = s.wind, c = Math.cos(a), sn = Math.sin(a);
    const arrow = [[0, -0.82], [0.34, -0.25], [0.12, -0.25], [0.12, 0.62], [-0.12, 0.62], [-0.12, -0.25], [-0.34, -0.25]];
    ctx.beginPath();
    arrow.forEach(([px, py], i) => ctx[i ? "lineTo" : "moveTo"](cx + r * (px * c + py * sn), cy + r * (py * c - px * sn)));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ["L", "R"].forEach((side, i) => {
      const on = s.tactile[i], lx = cx + (i ? 1 : -1) * (r + 95);
      ctx.beginPath();
      ctx.roundRect(lx - 50, cy - 18, 100, 36, 8);
      ctx.fillStyle = on ? C.bumpOn : C.bumpOff;
      ctx.fill();
      this.text(`${side} BUMP`, "bold 20px", on ? C.text : C.muted, lx, cy + 1, "center");
    });
  }

  drawFooter() {
    const hint = {
      ready: "SPACE start    Arrow keys drive    N new layout",
      playing: "Up/Down forward/back    Left/Right turn    P pause    R restart    N new layout",
    }[this.state];
    if (this.L.footer && hint) this.text(hint, "18px", C.muted, this.L.width / 2, this.L.height - FOOTER_H / 2, "center");
  }

  banner(title, subtitle) {
    const L = this.L, ctx = this.ctx, w = L.width - 2 * MARGIN;
    const lines = subtitle ? this.wrap(subtitle, "18px", w - 40) : [];
    const h = 80 + 26 * lines.length;
    const cy = L.tall ? L.height / 2 : HEADER_H + TITLE_H + EYE / 2; // over the eye views
    ctx.beginPath();
    ctx.roundRect(MARGIN, cy - h / 2, w, h, 12);
    ctx.fillStyle = "rgba(10, 11, 14, 0.86)";
    ctx.fill();
    const top = cy - h / 2 + 40;
    this.text(title, "bold 34px", C.text, L.width / 2, top, "center");
    lines.forEach((line, i) => this.text(line, "18px", C.muted, L.width / 2, top + 40 + 26 * i, "center"));
  }

  drawResults() {
    const L = this.L, ctx = this.ctx;
    ctx.fillStyle = "rgba(0, 0, 0, 0.59)";
    ctx.fillRect(0, 0, L.width, L.height);
    // Wide: map left, table right.  Tall: map on top, table below.
    const bw = Math.min(L.width - 2 * MARGIN, 780);
    const ms = L.tall ? Math.min(bw - 48, 420) : 320; // map size
    const bh = L.tall ? 70 + ms + 24 + 5 * 34 + 84 : 420;
    const bx = (L.width - bw) / 2, by = (L.height - bh) / 2;
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 12);
    ctx.fillStyle = C.panel;
    ctx.fill();
    this.text(this.resultText, "bold 34px", this.resultColor, L.width / 2, by + 34, "center");

    const mx = L.tall ? (L.width - ms) / 2 : bx + 24, my = by + 70;
    const agentPath = this.rec.path(this.k);
    this.drawMap(mx, my, ms, agentPath);

    // Table: a label column and one column per racer.
    const xl = L.tall ? bx + 24 : mx + ms + 30, ty = L.tall ? my + ms + 24 : my, row = L.tall ? 34 : 40;
    const colX = L.tall ? [bx + bw * 0.52, bx + bw * 0.8] : [xl + 150, xl + 280];
    ["Result", "Time", "Path", "Bumps"].forEach((label, i) => this.text(label, "18px", C.muted, xl, ty + row * (i + 1), "left", "top"));
    const p = this.player, pathLen = (pts) => pts.reduce((acc, q, i) => (i ? acc + Math.hypot(q[0] - pts[i - 1][0], q[1] - pts[i - 1][1]) : 0), 0);
    const rows = [
      ["YOU", C.player, p.reached, p.steps, pathLen(p.path), p.bumps],
      [this.agentName, C.agent, this.layout.agent.reached, this.rec.steps, pathLen(agentPath), this.rec.bumpsAt[this.rec.steps]],
    ];
    rows.forEach(([name, color, reached, steps, len, bumps], j) => {
      const x = colX[j];
      this.text(name, "bold 20px", color, x, ty, "center", "top");
      const cells = [
        [reached ? "goal" : "time up", reached ? C.good : C.bad],
        [reached ? `${(steps * this.dt).toFixed(2)} s` : "--", C.text],
        [`${len.toFixed(1)} m`, C.text],
        [String(bumps), C.text],
      ];
      cells.forEach(([s, c], i) => this.text(s, "18px", c, x, ty + row * (i + 1), "center", "top"));
    });
    const hint = this.touch ? "Next: new layout    Retry: same layout" : "SPACE next layout    R retry";
    const bottom = L.tall ? by + bh - 20 : my + ms;
    this.text("White ring: start    Green ring: goal zone", "18px", C.muted, xl, bottom - 40, "left", "bottom");
    this.text(hint, "bold 20px", C.text, xl, bottom, "left", "bottom");
  }

  drawMap(x, y, size, agentPath) {
    const ctx = this.ctx, a = this.scene.arena_half_extent, k = size / (2 * a);
    const px = ([wx, wy]) => [x + (wx + a) * k, y + (a - wy) * k];
    const circle = (cx, cy, r) => {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    };
    ctx.fillStyle = C.floor;
    ctx.fillRect(x, y, size, size);
    ctx.lineWidth = 3;
    ctx.strokeStyle = C.muted;
    ctx.strokeRect(x, y, size, size);
    for (const [ox, oy, r, , , cr, cg, cb] of this.layout.obstacles) {
      circle(...px([ox, oy]), r * k);
      ctx.fillStyle = `rgb(${255 * cr | 0}, ${255 * cg | 0}, ${255 * cb | 0})`;
      ctx.fill();
    }
    const [gx, gy] = px(this.layout.goal);
    circle(gx, gy, this.scene.goal_radius * k);
    ctx.lineWidth = 2;
    ctx.strokeStyle = C.good;
    ctx.stroke();
    circle(gx, gy, this.scene.ball.radius * k);
    ctx.fillStyle = "#fff";
    ctx.fill();
    for (const [pts, color] of [[this.player.path, C.player], [agentPath, C.agent]]) {
      ctx.beginPath();
      pts.forEach((q, i) => ctx[i ? "lineTo" : "moveTo"](...px(q)));
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.lineJoin = "round";
      ctx.stroke();
      circle(...px(pts[pts.length - 1]), 5);
      ctx.fillStyle = color;
      ctx.fill();
    }
    circle(...px(this.layout.start), 6);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }
}

async function main() {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  try {
    const res = await fetch("data/game.json");
    if (!res.ok) throw new Error(`data/game.json: HTTP ${res.status}`);
    const game = new Game(canvas, await res.json());
    window.game = game; // for debugging from the console
    const seed = Number(new URLSearchParams(location.search).get("layout"));
    const start = game.data.layouts.findIndex((l) => l.seed === seed);
    game.run();
    await game.newRound(Math.max(0, start));
  } catch (err) {
    canvas.width = 1092;
    canvas.height = 120;
    ctx.fillStyle = C.bad;
    ctx.font = `18px ${FONT}`;
    ctx.fillText(`Could not start the game: ${err.message}`, 20, 60);
    throw err;
  }
}

main();
