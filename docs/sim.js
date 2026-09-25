// The player's side of play_vs_connectome_rnn.py, without MuJoCo: the robot's motion and its
// photoreceptor input, both calibrated against MuJoCo by export_web_game.py, which writes every
// constant used here into data/game.json.  Also loads the agent's recorded runs.

// 4x4 sub-pixel samples per photoreceptor, spanning +-2.5 px: stands in for the 5x5 average pool
// the agent applies to its 128x128 camera images before sampling them.
const SUB = [-1.875, -0.625, 0.625, 1.875];
const CONTACT_EPS = 0.005; // m: a surface this close still presses on the bump sensors
const TACTILE_FRONT = 0.1745; // rad: contacts this close to straight ahead press both sides
// A contact counts as a new bump only after this many contact-free steps.  MuJoCo's contact flag
// flickers while the robot presses against something, which would otherwise count as many bumps.
const BUMP_GAP = 10;

// Bumps counted up to each step of a contact-flag sequence (truthy = touching).
export function countBumps(touching) {
  const counts = new Uint16Array(touching.length);
  let free = BUMP_GAP;
  for (let k = 0; k < touching.length; k++) {
    counts[k] = (k ? counts[k - 1] : 0) + (touching[k] && free >= BUMP_GAP ? 1 : 0);
    free = touching[k] ? 0 : free + 1;
  }
  return counts;
}

export function wrapPi(a) {
  return a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
}

function mod2(k) {
  return ((k % 2) + 2) % 2;
}

// ---------------------------------------------------------------- vision

// Renders the value each photoreceptor samples, for a robot pose in a layout.  The scene is
// what the MuJoCo cameras see: a checkered floor, banded walls, checkered cylinders and the white
// goal ball under fixed lights, against a black sky.  The cameras are level, so every ray in an
// image column shares one horizontal direction; walls and cylinders are vertical, so they are
// intersected once per column and each ray only checks the height of those hits.
export class Eyes {
  constructor(scene) {
    this.scene = scene;
    this.nDots = scene.eyes.reduce((n, e) => n + e.dots.length, 0);
    this.eyes = scene.eyes.map((e) => this._prepare(e));
  }

  _prepare(eye) {
    const S = eye.image_size, T = eye.tan_half_fov;
    const colIndex = new Map(), cols = [];
    const nSamp = eye.dots.length * SUB.length * SUB.length;
    const sampCol = new Int32Array(nSamp), sampSlope = new Float32Array(nSamp);
    let k = 0;
    for (const [gx, gy] of eye.dots) {
      // grid_sample(align_corners=True) position -> pixel coordinates
      const pu = ((gx + 1) / 2) * (S - 1), pv = ((gy + 1) / 2) * (S - 1);
      for (const du of SUB) {
        const u = Math.min(S - 1, Math.max(0, pu + du));
        const key = u.toFixed(4);
        if (!colIndex.has(key)) {
          const nx = ((2 * (u + 0.5)) / S - 1) * T;
          const hx = nx * eye.right[0] + eye.fwd[0], hy = nx * eye.right[1] + eye.fwd[1];
          const hz = nx * eye.right[2] + eye.fwd[2], len = Math.hypot(hx, hy);
          colIndex.set(key, cols.length);
          cols.push({ hx: hx / len, hy: hy / len, hz, len });
        }
        const c = colIndex.get(key), col = cols[c];
        for (const dv of SUB) {
          const v = Math.min(S - 1, Math.max(0, pv + dv));
          const ny = (1 - (2 * (v + 0.5)) / S) * T;
          sampCol[k] = c;
          sampSlope[k] = (ny * eye.up[2] + col.hz) / col.len; // rise per metre travelled horizontally
          k++;
        }
      }
    }
    const fwdLen = Math.hypot(eye.fwd[0], eye.fwd[1]);
    return { eye, cols, sampCol, sampSlope, fwd: [eye.fwd[0] / fwdLen, eye.fwd[1] / fwdLen] };
  }

  // Gray level of a lit surface: clip(albedo * lighting) per channel, then the agent's
  // grayscale weights.  n is the (unit) surface normal, fwd the camera's headlight direction.
  _shade(albedo, nx, ny, nz, fwdx, fwdy) {
    const L = this.scene.light;
    let lum = L.ambient;
    for (const l of L.lights) lum += l.diffuse * Math.max(0, -(nx * l.dir[0] + ny * l.dir[1] + nz * l.dir[2]));
    lum += L.headlight_diffuse * Math.max(0, -(nx * fwdx + ny * fwdy));
    const g = L.gray;
    return g[0] * Math.min(1, albedo[0] * lum) + g[1] * Math.min(1, albedo[1] * lum) + g[2] * Math.min(1, albedo[2] * lum);
  }

  // Fills out (Uint8Array, one value per photoreceptor, left eye first) with 0-255 values,
  // quantised as PhotoreceptorView does.  vision = [left, right]; a blind eye sees 0.
  render(layout, x, y, yaw, vision, out) {
    const sc = this.scene, c = Math.cos(yaw), s = Math.sin(yaw);
    const obstacles = layout.obstacles, [gx, gy] = layout.goal;
    const ballR = sc.ball.radius, ballZ = sc.ball.z, sq = sc.floor.square;
    const sqO = sc.obstacle_texture.square, tex = sc.obstacle_texture.tex;
    const nCand = sc.walls.length + obstacles.length; // hits per column, at most
    let base = 0;
    for (let e = 0; e < this.eyes.length; e++) {
      const { eye, cols, sampCol, sampSlope, fwd } = this.eyes[e];
      const nd = eye.dots.length;
      if (!vision[e]) {
        out.fill(0, base, base + nd);
        base += nd;
        continue;
      }
      const ox = x + c * eye.cam_pos[0] - s * eye.cam_pos[1];
      const oy = y + s * eye.cam_pos[0] + c * eye.cam_pos[1];
      const oz = eye.cam_pos[2];
      const fwx = c * fwd[0] - s * fwd[1], fwy = s * fwd[0] + c * fwd[1];
      const floorLum = [0, 1].map((k) => this._shade(sc.floor.albedo[k], 0, 0, 1, fwx, fwy));

      // Per column: walls and cylinders hit, sorted by horizontal distance.
      const nc = cols.length;
      const eb = this.eyes[e];
      if (!eb.buf || eb.buf.nCand !== nCand) {
        eb.buf = { nCand, dist: new Float32Array(nc * nCand), top: new Float32Array(nc * nCand), gray: new Float32Array(nc * nCand),
                   count: new Int32Array(nc), ball: new Uint8Array(nc), dx: new Float32Array(nc), dy: new Float32Array(nc) };
      }
      const { dist: cDist, top: cTop, gray: cGray, count: cCount, ball: cBall, dx: cDx, dy: cDy } = eb.buf;
      for (let ci = 0; ci < nc; ci++) {
        const col = cols[ci];
        const dx = c * col.hx - s * col.hy, dy = s * col.hx + c * col.hy;
        cDx[ci] = dx;
        cDy[ci] = dy;
        const o = ci * nCand;
        let n = 0;
        const insert = (dist, top, gray) => {
          let j = n++;
          while (j > 0 && cDist[o + j - 1] > dist) {
            cDist[o + j] = cDist[o + j - 1];
            cTop[o + j] = cTop[o + j - 1];
            cGray[o + j] = cGray[o + j - 1];
            j--;
          }
          cDist[o + j] = dist;
          cTop[o + j] = top;
          cGray[o + j] = gray;
        };
        for (const w of sc.walls) {
          const d = w.axis === 0 ? dx : dy, p = w.axis === 0 ? ox : oy;
          const t = (w.pos - p) / d;
          if (!(t > 0) || !isFinite(t)) continue;
          const along = (w.band_axis === 0 ? ox + dx * t : oy + dy * t);
          const albedo = w.albedo[mod2(Math.floor(along / w.band))];
          insert(t, w.zmax, this._shade(albedo, w.normal[0], w.normal[1], 0, fwx, fwy));
        }
        for (const ob of obstacles) {
          const [cx, cy, r, , top, cr, cg, cb] = ob;
          const px = ox - cx, py = oy - cy;
          const b = px * dx + py * dy, q = px * px + py * py - r * r;
          const disc = b * b - q;
          if (disc < 0) continue;
          const t = -b - Math.sqrt(disc);
          if (t <= 0) continue;
          const hx = px + dx * t, hy = py + dy * t;
          const tx = tex[mod2(Math.floor(hx / sqO) + Math.floor(hy / sqO))];
          insert(t, top, this._shade([cr * tx[0], cg * tx[1], cb * tx[2]], hx / r, hy / r, 0, fwx, fwy));
        }
        cCount[ci] = n;
        // Can this column's rays reach the goal ball at all?
        const bx = gx - ox, by = gy - oy, along = bx * dx + by * dy;
        cBall[ci] = along > 0 && Math.abs(bx * dy - by * dx) < ballR ? 1 : 0;
      }

      // Per ray: the first wall/cylinder hit below its top, else the floor, else the sky;
      // then the ball if it is closer.  Average the 16 samples of each photoreceptor.
      const perDot = SUB.length * SUB.length;
      for (let di = 0; di < nd; di++) {
        let sum = 0;
        for (let k = di * perDot, end = k + perDot; k < end; k++) {
          const ci = sampCol[k], slope = sampSlope[k];
          const floorDist = slope < 0 ? oz / -slope : Infinity;
          let dist = floorDist, gray = 0;
          if (floorDist < Infinity) {
            const fx = ox + cDx[ci] * floorDist, fy = oy + cDy[ci] * floorDist;
            gray = floorLum[mod2(Math.floor(fx / sq) + Math.floor(fy / sq))];
          }
          const o = ci * nCand;
          for (let j = 0, n = cCount[ci]; j < n; j++) {
            const t = cDist[o + j];
            if (t >= floorDist) break;
            if (oz + slope * t <= cTop[o + j]) {
              dist = t;
              gray = cGray[o + j];
              break;
            }
          }
          if (cBall[ci]) {
            // |(o + d*t, oz + slope*t) - ball|^2 = R^2, t = horizontal distance
            const bx = ox - gx, by = oy - gy, bz = oz - ballZ;
            const A = 1 + slope * slope, B = 2 * (bx * cDx[ci] + by * cDy[ci] + bz * slope);
            const C = bx * bx + by * by + bz * bz - ballR * ballR, disc = B * B - 4 * A * C;
            if (disc >= 0) {
              const t = (-B - Math.sqrt(disc)) / (2 * A);
              if (t > 0 && t < dist) gray = this._shade([1, 1, 1], 0, 0, 1, fwx, fwy);
            }
          }
          sum += gray;
        }
        out[base + di] = Math.floor((sum / perDot) * 255);
      }
      base += nd;
    }
    return out;
  }
}

// ---------------------------------------------------------------- motion

// The robot as a kinematic body: speed and turn rate approach the value MuJoCo measured for the
// held arrow keys with first-order lags, and contacts reproduce MuJoCo's high-friction ones --
// pushing into a surface all but stops the robot and pivots it to face the surface.
export class Robot {
  constructor(scene, layout) {
    this.scene = scene;
    this.layout = layout;
    this.reset();
  }

  reset() {
    [this.x, this.y, this.yaw] = this.layout.start;
    this.v = 0;
    this.w = 0;
    this.steps = 0;
    this.finished = false;
    this.reached = false;
    this.bumps = 0;
    this.touch = null; // contact angle relative to the heading, or null
    this.free = BUMP_GAP; // contact-free steps so far
    this.path = [[this.x, this.y]];
  }

  get tactile() {
    const a = this.touch;
    if (a === null) return [false, false];
    const front = Math.abs(a) < TACTILE_FRONT;
    return [front || a > 0, front || a < 0];
  }

  // The goal's direction relative to the heading ("wind"), > 0 to the left.
  get windAngle() {
    const [gx, gy] = this.layout.goal;
    return wrapPi(Math.atan2(gy - this.y, gx - this.x) - this.yaw);
  }

  // One control step with drive, turn in {-1, 0, 1} (Up - Down, Left - Right).
  step(drive, turn) {
    const sc = this.scene, m = sc.motion, dt = sc.dt;
    // Speed and turn rate each approach the measured value for the held keys, with MuJoCo's
    // time constants for building up (key held) and dying down (key released).
    const [vt, wt] = m.table[`${drive},${turn}`];
    const lag = (cur, target, p) => cur + (target - cur) * (1 - Math.exp(-dt / (target === 0 ? p.tau_fall : p.tau_rise)));
    this.v = lag(this.v, vt, m.speed_lag);
    this.w = lag(this.w, wt, m.turn_lag);
    this.yaw = wrapPi(this.yaw + this.w * dt);
    this.x += this.v * Math.cos(this.yaw) * dt;
    this.y += this.v * Math.sin(this.yaw) * dt;

    // Contacts: push the robot out of what it overlaps; remember the closest surface.
    const R = sc.robot_radius;
    let touchN = null, touchGap = CONTACT_EPS;
    const contact = (nx, ny, gap) => {
      // (nx, ny): unit vector from the robot towards the surface; gap: distance to it
      if (gap < 0) {
        this.x += nx * gap;
        this.y += ny * gap;
      }
      if (gap < touchGap) {
        touchGap = gap;
        touchN = [nx, ny];
      }
      // Pushing into it: MuJoCo's friction all but stops the robot and pivots it to face it.
      const heading = this.v >= 0 ? this.yaw : this.yaw + Math.PI;
      if (gap < CONTACT_EPS && this.v * (Math.cos(this.yaw) * nx + Math.sin(this.yaw) * ny) > 0) {
        const cap = m.contact_slide * m.table["1,0"][0];
        this.v = Math.sign(this.v) * Math.min(Math.abs(this.v), cap);
        this.yaw = wrapPi(this.yaw + m.contact_pivot * Math.sin(wrapPi(Math.atan2(ny, nx) - heading)) * dt);
      }
    };
    for (const [cx, cy, r] of this.layout.obstacles) {
      const dx = cx - this.x, dy = cy - this.y, d = Math.hypot(dx, dy);
      if (d - R - r < CONTACT_EPS) contact(dx / d, dy / d, d - R - r);
    }
    for (const wl of sc.walls) {
      // Inner face at coordinate wl.pos along wl.axis; wl.normal points into the arena.
      const p = wl.axis === 0 ? this.x : this.y, sign = -wl.normal[wl.axis];
      const gap = sign * (wl.pos - p) - R;
      if (gap < CONTACT_EPS) contact(wl.axis === 0 ? sign : 0, wl.axis === 1 ? sign : 0, gap);
    }

    this.touch = touchN ? wrapPi(Math.atan2(touchN[1], touchN[0]) - this.yaw) : null;
    if (this.touch !== null && this.free >= BUMP_GAP) this.bumps++;
    this.free = this.touch !== null ? 0 : this.free + 1;
    this.steps++;
    this.path.push([this.x, this.y]);
    const [gx, gy] = this.layout.goal;
    this.reached = Math.hypot(gx - this.x, gy - this.y) < sc.goal_radius;
    this.finished = this.reached || this.steps >= sc.max_steps;
  }
}

// ---------------------------------------------------------------- the agent's recorded run

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf[0] !== 0x1f || buf[1] !== 0x8b) return buf; // the server already decoded the gzip
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// One of the agent's runs, as written by export_web_game.py: per step, its pose, bump sensors
// and eye input.  frame k is the state after k control steps (clamped to the end of the run).
export class Recording {
  static async load(url, layout, nDots) {
    const bytes = await fetchBytes(url);
    const n = layout.agent.steps + 1;
    const rec = new Recording();
    rec.layout = layout;
    rec.steps = layout.agent.steps;
    rec.poses = new Float32Array(bytes.buffer, bytes.byteOffset, n * 3);
    rec.touch = bytes.subarray(n * 12, n * 13);
    rec.eyes = bytes.subarray(n * 13, n * 13 + n * nDots);
    if (rec.eyes.length !== n * nDots) throw new Error(`${url}: truncated recording`);
    for (let i = nDots; i < rec.eyes.length; i++) rec.eyes[i] += rec.eyes[i - nDots]; // undo delta coding (wraps mod 256)
    rec.nDots = nDots;
    rec.bumpsAt = countBumps(rec.touch); // bumps counted up to each step
    return rec;
  }

  frame(k) {
    return Math.min(k, this.steps);
  }
  pose(k) {
    const i = 3 * this.frame(k);
    return [this.poses[i], this.poses[i + 1], this.poses[i + 2]];
  }
  eyeValues(k) {
    const i = this.frame(k) * this.nDots;
    return this.eyes.subarray(i, i + this.nDots);
  }
  tactile(k) {
    const t = this.touch[this.frame(k)];
    return [(t & 1) !== 0, (t & 2) !== 0];
  }
  windAngle(k) {
    const [x, y, yaw] = this.pose(k), [gx, gy] = this.layout.goal;
    return wrapPi(Math.atan2(gy - y, gx - x) - yaw);
  }
  path(k) {
    const pts = [];
    for (let i = 0; i <= this.frame(k); i++) pts.push([this.poses[3 * i], this.poses[3 * i + 1]]);
    return pts;
  }
}
