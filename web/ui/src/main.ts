import "./style.css";

type WasmOptions = {
  seed: number;
  niterations: number;
  populations: number;
  population_size: number;
  ncycles_per_iteration: number;
  maxsize: number;
  topn: number;
  has_headers: boolean;
};

type EquationSummary = {
  complexity: number;
  loss: number;
  cost: number;
  equation: string;
};

type SearchSnapshot = {
  total_cycles: number;
  cycles_completed: number;
  total_evals: number;
  best: EquationSummary;
  pareto_front: EquationSummary[];
};

type WorkerMsg =
  | { type: "ready" }
  | { type: "snapshot"; snap: SearchSnapshot }
  | { type: "done" }
  | { type: "stopped" }
  | { type: "error"; error: string };

const DEFAULT_CSV = `x1,x2,y
0.0,0.0,0.0
0.0,1.0,1.0
1.0,0.0,1.0
1.0,1.0,2.0
2.0,0.0,4.0
2.0,1.0,5.0
`;

const DEFAULT_OPS = ["+", "-", "*", "/", "sin", "cos", "exp", "log"];

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {}
): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function formatSci(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  return x.toExponential(3);
}

function copyToClipboard(text: string): void {
  void navigator.clipboard.writeText(text);
}

const app = document.querySelector<HTMLDivElement>("#app")!;

const title = el("h2");
title.textContent = "Symbolic Regression (WASM, single-thread worker)";

const grid = el("div", { class: "grid" });

const left = el("div", { class: "card" });
const right = el("div", { class: "card" });

const csvLabel = el("label");
csvLabel.textContent = "CSV (last column is y; headers optional):";
const csvTextarea = el("textarea") as HTMLTextAreaElement;
csvTextarea.value = DEFAULT_CSV;

const opsLabel = el("label");
opsLabel.textContent = "Operators (comma-separated tokens):";
const opsInput = el("input", { type: "text" }) as HTMLInputElement;
opsInput.value = DEFAULT_OPS.join(",");

const controls = el("div", { class: "row" });

function labeledNumber(labelText: string, value: number, min = 0): [HTMLLabelElement, HTMLInputElement] {
  const l = el("label");
  l.textContent = labelText;
  const i = el("input", { type: "number", min: String(min) }) as HTMLInputElement;
  i.value = String(value);
  return [l, i];
}

const [seedL, seedI] = labeledNumber("seed", 0);
const [niterL, niterI] = labeledNumber("niterations", 200, 1);
const [popsL, popsI] = labeledNumber("populations", 8, 1);
const [popsizeL, popsizeI] = labeledNumber("population_size", 64, 1);
const [cyclesL, cyclesI] = labeledNumber("ncycles_per_iteration", 200, 1);
const [maxsizeL, maxsizeI] = labeledNumber("maxsize", 30, 1);
const [topnL, topnI] = labeledNumber("topn", 12, 1);

const headersWrap = el("label");
const headersCb = el("input", { type: "checkbox" }) as HTMLInputElement;
headersCb.checked = true;
headersWrap.append(headersCb, " has_headers");

const runBtn = el("button") as HTMLButtonElement;
runBtn.textContent = "Run";
const stopBtn = el("button") as HTMLButtonElement;
stopBtn.textContent = "Stop";
stopBtn.disabled = true;

const status = el("div", { class: "small" });
status.textContent = "Idle.";

controls.append(
  seedL,
  seedI,
  niterL,
  niterI,
  popsL,
  popsI,
  popsizeL,
  popsizeI,
  cyclesL,
  cyclesI,
  maxsizeL,
  maxsizeI,
  topnL,
  topnI,
  headersWrap,
  runBtn,
  stopBtn
);

left.append(csvLabel, csvTextarea, opsLabel, opsInput, controls, status);

const bestTitle = el("h3");
bestTitle.textContent = "Best";
const bestBox = el("div", { class: "mono" });
bestBox.textContent = "(none)";

const hofTitle = el("h3");
hofTitle.textContent = "Pareto front (latest)";
const list = el("div", { class: "list" });

right.append(bestTitle, bestBox, hofTitle, list);

grid.append(left, right);
app.append(title, grid);

let worker: Worker | null = null;
let latestSnap: SearchSnapshot | null = null;

function setRunning(isRunning: boolean) {
  runBtn.disabled = isRunning;
  stopBtn.disabled = !isRunning;
}

function renderSnapshot(snap: SearchSnapshot) {
  latestSnap = snap;

  const pct =
    snap.total_cycles > 0 ? (100 * snap.cycles_completed) / snap.total_cycles : 0;
  status.textContent = `cycles ${snap.cycles_completed}/${snap.total_cycles} (${pct.toFixed(
    1
  )}%), evals=${snap.total_evals}`;

  bestBox.textContent = `${snap.best.complexity}\t${formatSci(snap.best.loss)}\t${snap.best.equation}`;

  list.replaceChildren();
  for (const eq of snap.pareto_front.slice().reverse()) {
    const row = el("div", { class: "eq" });
    const left = el("div", { class: "mono" });
    left.textContent = `C=${eq.complexity}`;
    const mid = el("div", { class: "mono" });
    mid.textContent = `loss=${formatSci(eq.loss)}`;
    const right = el("div");
    const eqn = el("div", { class: "mono" });
    eqn.textContent = eq.equation;
    const copy = el("button") as HTMLButtonElement;
    copy.textContent = "Copy";
    copy.onclick = () => copyToClipboard(eq.equation);
    right.append(eqn, copy);
    row.append(left, mid, right);
    list.append(row);
  }
}

function parseOptions(): WasmOptions {
  return {
    seed: Number(seedI.value) | 0,
    niterations: Math.max(1, Number(niterI.value) | 0),
    populations: Math.max(1, Number(popsI.value) | 0),
    population_size: Math.max(1, Number(popsizeI.value) | 0),
    ncycles_per_iteration: Math.max(1, Number(cyclesI.value) | 0),
    maxsize: Math.max(1, Number(maxsizeI.value) | 0),
    topn: Math.max(1, Number(topnI.value) | 0),
    has_headers: headersCb.checked
  };
}

function parseOperators(): string[] {
  const toks = opsInput.value
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return toks.length > 0 ? toks : DEFAULT_OPS;
}

function spawnWorker(): Worker {
  if (worker) worker.terminate();
  const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  w.onmessage = (e: MessageEvent<WorkerMsg>) => {
    const msg = e.data;
    if (msg.type === "ready") {
      setRunning(true);
      status.textContent = "Running...";
      w.postMessage({ type: "run", stepCycles: 1, snapshotEverySteps: 1 });
      return;
    }
    if (msg.type === "snapshot") {
      renderSnapshot(msg.snap);
      return;
    }
    if (msg.type === "done") {
      setRunning(false);
      status.textContent = "Done.";
      return;
    }
    if (msg.type === "stopped") {
      setRunning(false);
      status.textContent = "Stopped.";
      return;
    }
    if (msg.type === "error") {
      setRunning(false);
      status.textContent = `Error: ${msg.error}`;
      return;
    }
  };
  worker = w;
  return w;
}

runBtn.onclick = () => {
  const w = spawnWorker();
  setRunning(true);
  status.textContent = "Initializing WASM...";

  w.postMessage({
    type: "init",
    csvText: csvTextarea.value,
    options: parseOptions(),
    operators: parseOperators()
  });
};

stopBtn.onclick = () => {
  if (!worker) return;
  worker.postMessage({ type: "stop" });
};

