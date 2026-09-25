import { Agent } from "../dist/index.js";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

// ---- CLI (no deps) ----
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i === -1 || i + 1 >= args.length ? def : args[i + 1];
};
const MODEL_DIR = opt("--model", null);
const HERE_MODEL = ["./model-ml", "../../model-ml"]
  .map((d) => fileURLToPath(new URL(d, import.meta.url)))
  .find((d) => existsSync(d));
if (!MODEL_DIR && !HERE_MODEL) {
  console.error("no local weights found; downloading convaiinnovations/laya (english) — or export your own: python laya-ts/scripts/export_onnx.py --model-dir <ckpt> --out-dir ./model-ml");
}
const TICK_MS = Math.max(20, Number(opt("--tick", "120")) || 120);
const W = Math.max(10, Number(opt("--width", "20")) || 20);
const H = Math.max(6, Number(opt("--height", "12")) || 12);
const MAX_TICKS = Math.max(0, Number(opt("--ticks", "0")) || 0);
const HEADLESS = MAX_TICKS > 0;
if (!process.stdout.isTTY && !HEADLESS) {
  console.error("snake: not a TTY — rerun with --ticks N for headless mode");
  process.exit(2);
}

const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
const VEC = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
const DANGER_W = 2; // danger weight vs food pull in fusion score
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- Game state ----
let snake, food, dir, score, alive;
let run = 0, best = 0;
function reset() {
  snake = [{ x: W >> 1, y: H >> 1 }, { x: (W >> 1) - 1, y: H >> 1 }];
  dir = "RIGHT";
  score = 0;
  alive = true;
  spawnFood();
}
function spawnFood() {
  do {
    food = { x: (Math.random() * W) | 0, y: (Math.random() * H) | 0 };
  } while (snake.some((s) => s.x === food.x && s.y === food.y));
}
function cellAfter(d) {
  const [dx, dy] = VEC[d];
  return { x: snake[0].x + dx, y: snake[0].y + dy };
}
// Exact lethal check. Tail cell is safe unless we are growing into food.
function lethal(d) {
  const n = cellAfter(d);
  if (n.x < 0 || n.y < 0 || n.x >= W || n.y >= H) return true;
  const growing = n.x === food.x && n.y === food.y;
  const body = growing ? snake : snake.slice(0, -1);
  return body.some((s) => s.x === n.x && s.y === n.y);
}
function step(d) {
  dir = d;
  if (lethal(d)) { alive = false; return "dead"; }
  const n = cellAfter(d);
  snake.unshift(n);
  if (n.x === food.x && n.y === food.y) { score++; spawnFood(); return "ate"; }
  snake.pop();
  return "moved";
}
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

// ---- Renderer (zero-dep ANSI) ----
const bar = (v, n = 10) => {
  const f = Math.max(0, Math.min(n, Math.round(v * n)));
  return "█".repeat(f) + "░".repeat(n - f);
};
function render(dec) {
  const grid = Array.from({ length: H }, () => Array(W).fill(" "));
  grid[food.y][food.x] = "*";
  snake.forEach((s, i) => { grid[s.y][s.x] = i === 0 ? "O" : "o"; });
  let s = "\x1b[2J\x1b[H\x1b[?25l";
  s += "#" .repeat(W + 2) + "\n";
  for (const row of grid) s += "#" + row.join("") + "#\n";
  s += "#".repeat(W + 2) + "\n";
  s += `score ${score}  len ${snake.length}  run ${run}  best ${best}  brain=${dec.src}\n`;
  for (const d of DIRS) {
    s += `${d.padEnd(5)} danger ${dec.danger[d].toFixed(2)} ${bar(dec.danger[d])}  pull ${dec.pull[d] >= 0 ? "+" : ""}${dec.pull[d].toFixed(2)}\n`;
  }
  s += `-> ${dec.dir}  conf ${dec.conf.toFixed(2)}${dec.veto ? "  VETO" : ""}${dec.errors ? `  errx${dec.errors}` : ""}\n`;
  process.stdout.write(s);
}
process.on("exit", () => process.stdout.write("\x1b[?25h"));
process.on("SIGINT", () => process.exit(0));

// ---- Brain ----
let agent = null;
let src = "heuristic";
let errStreak = 0;
function boardText() {
  const h = snake[0];
  return `Snake on ${W}x${H}, head (${h.x},${h.y}) facing ${dir}, food (${food.x},${food.y}), body ${snake.map((s) => `(${s.x},${s.y})`).join(" ")}.`;
}
function foodPull() {
  const d0 = manhattan(snake[0], food);
  const out = {};
  for (const d of DIRS) out[d] = d0 - manhattan(cellAfter(d), food); // +1 closer, -1 farther
  return out;
}
async function decide() {
  const pull = foodPull();
  const exact = Object.fromEntries(DIRS.map((d) => [d, lethal(d) ? 1 : 0]));
  let danger = { ...exact };
  let conf = 0;
  if (agent && errStreak < 5) {
    try {
      const q = {};
      for (const d of DIRS) {
        q[`danger_${d.toLowerCase()}`] = {
          type: "noul",
          instructions: `Will the snake die (wall or own body) if it moves ${d} next? ${boardText()}`,
        };
      }
      const out = await agent.systemOne(boardText(), q);
      for (const d of DIRS) danger[d] = out.answers[`danger_${d.toLowerCase()}`]?.noul ?? exact[d];
      conf = DIRS.reduce((a, d) => a + (out.answers[`danger_${d.toLowerCase()}`]?.confidence ?? 0), 0) / 4;
      errStreak = 0;
    } catch { errStreak++; }
  }
  if (errStreak >= 5) src = "heuristic";
  let pick = DIRS[0], bestScore = -Infinity;
  for (const d of DIRS) {
    const s = pull[d] - DANGER_W * danger[d];
    if (s > bestScore) { bestScore = s; pick = d; }
  }
  let veto = false;
  if (exact[pick] === 1 && DIRS.some((d) => exact[d] === 0)) {
    veto = true;
    pick = DIRS.filter((d) => exact[d] === 0)
      .map((d) => [d, pull[d] - DANGER_W * danger[d]])
      .sort((a, b) => b[1] - a[1])[0][0];
  }
  return { dir: pick, danger, pull, conf, veto, src, errors: errStreak };
}

// ---- Main ----
reset();
try {
  agent = MODEL_DIR ? await Agent.load(MODEL_DIR)
    : HERE_MODEL ? await Agent.load(HERE_MODEL)
    : await Agent.load("convaiinnovations/laya");
  src = "model";
} catch (e) {
  console.error(`snake: model load failed (${e.message}) — heuristic mode`);
}
let ticks = 0;
for (;;) {
  const dec = await decide();
  step(dec.dir);
  ticks++;
  if (!HEADLESS) render(dec);
  if (!alive) {
    run++;
    best = Math.max(best, score);
    if (!HEADLESS) { render({ ...dec, dir: dec.dir }); await sleep(800); }
    reset();
  }
  if (HEADLESS && ticks >= MAX_TICKS) {
    console.log(`smoke ok ticks=${ticks} score=${score} run=${run} best=${best} brain=${src}`);
    process.exit(0);
  }
  if (!HEADLESS) await sleep(TICK_MS);
}
