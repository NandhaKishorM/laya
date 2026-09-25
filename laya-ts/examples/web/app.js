// Browser triage demo: Agent.load fetches the split ONNX model over HTTP,
// createWebProvider tries WebGPU then falls back to WASM automatically.
import { Agent, checkQuestion, triageQuestions } from "../../dist/index.js";

const $ = (id) => document.getElementById(id);
const pill = $("pill");
const bar = $("bar");
const loadMsg = $("loadmsg");
const cardLoad = $("card-load");
const cardRun = $("card-run");
const cardQ = $("card-q");
const runBtn = $("run");
const outEl = $("out");
const runMeta = $("runmeta");

// ---- Custom questions ----
let questions = triageQuestions();
let qseq = 0;
const CRIT_HINT = {
  choice: "Criteria — one per line, <code>label: description</code>",
  score: "Levels — one per line, best first",
  noul: "Optional — <code>true: ...</code> / <code>false: ...</code> lines",
};
const CRIT_PLACEHOLDER = {
  choice: "billing: invoices, payments, refunds\nother:",
  score: "calm and neutral\nclearly annoyed",
  noul: "true: yes, furious",
};
const CRIT_SAMPLE = {
  choice: "billing: invoices, payments, refunds\nother:",
  score: "calm and neutral\nclearly annoyed\nvery angry",
  noul: "",
};

function renderQuestions() {
  const list = $("qlist");
  list.replaceChildren();
  const ids = Object.keys(questions);
  if (ids.length === 0) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "No questions — add one below.";
    list.append(p);
  }
  for (const qid of ids) {
    const q = questions[qid];
    const item = document.createElement("div");
    item.className = "qitem";
    const code = document.createElement("code");
    code.textContent = qid;
    const type = document.createElement("span");
    type.textContent = q.type;
    item.append(code, type);
    const del = document.createElement("button");
    del.textContent = "remove";
    del.onclick = () => {
      delete questions[qid];
      renderQuestions();
      syncRunEnabled();
    };
    item.appendChild(del);
    list.appendChild(item);
  }
}

function syncRunEnabled() {
  runBtn.disabled = !agent || Object.keys(questions).length === 0;
}

$("qtype").onchange = () => {
  const t = $("qtype").value;
  $("qcritlabel").innerHTML = CRIT_HINT[t];
  $("qcrit").placeholder = CRIT_PLACEHOLDER[t];
  if (!$("qcrit").value.trim()) $("qcrit").value = CRIT_SAMPLE[t];
};

function parseCriteria(type, raw) {
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  if (type === "score") return lines;
  if (type === "noul") {
    const crit = {};
    for (const l of lines) {
      const i = l.indexOf(":");
      const k = (i === -1 ? l : l.slice(0, i)).trim().toLowerCase();
      if (k !== "true" && k !== "false") throw new Error(`noul lines must start with true: or false:, got ${JSON.stringify(l)}`);
      crit[k] = i === -1 ? "" : l.slice(i + 1).trim();
    }
    return crit;
  }
  const crit = {};
  for (const l of lines) {
    const i = l.indexOf(":");
    const k = (i === -1 ? l : l.slice(0, i)).trim();
    if (!k) throw new Error(`empty label in ${JSON.stringify(l)}`);
    if (k in crit) throw new Error(`duplicate label ${JSON.stringify(k)}`);
    crit[k] = i === -1 ? "" : l.slice(i + 1).trim();
  }
  return crit;
}

$("qadd").onclick = () => {
  const errEl = $("qerr");
  errEl.textContent = "";
  try {
    const type = $("qtype").value;
    const ins = $("qins").value.trim();
    if (!ins) throw new Error("instructions are required");
    const qid = `custom${++qseq}`;
    const qdef = { type, instructions: ins };
    if (type !== "noul" || $("qcrit").value.trim()) qdef.criteria = parseCriteria(type, $("qcrit").value);
    checkQuestion(qid, qdef);
    questions[qid] = qdef;
    $("qins").value = "";
    $("qcrit").value = "";
    renderQuestions();
    syncRunEnabled();
  } catch (e) {
    errEl.textContent = e.message;
  }
};

$("qpreset").onclick = () => {
  questions = triageQuestions();
  renderQuestions();
  syncRunEnabled();
  $("qerr").textContent = "";
};

let agent = null;
let ticker = null;

// Anything with a scheme goes as-is; dotted/absolute paths and bare
// names resolve against the page (local-first: HF repos are owner/name).
// Without this, `../model-ml` is misread as a HuggingFace repo ID.
function resolveModelUrl(raw) {
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(raw)) return raw;
  if (raw.startsWith(".") || raw.startsWith("/") || !raw.includes("/")) {
    return new URL(raw, document.baseURI).href;
  }
  return raw;
}

function setPill(state, text) {
  pill.className = `pill${state === "idle" ? "" : ` ${state}`}`;
  pill.textContent = text;
}

function startLoadingClock(t0) {
  stopLoadingClock();
  ticker = setInterval(() => {
    loadMsg.textContent = `Fetching config → tokenizer → weights → sessions… ${((performance.now() - t0) / 1000).toFixed(0)}s elapsed`;
  }, 500);
}

function stopLoadingClock() {
  if (ticker !== null) {
    clearInterval(ticker);
    ticker = null;
  }
}

function markReady(seconds, gpu) {
  stopLoadingClock();
  bar.hidden = true;
  setPill("ready", "ready");
  cardLoad.classList.remove("active");
  cardLoad.classList.add("done");
  cardRun.classList.remove("locked");
  cardRun.classList.add("active");
  loadMsg.textContent = `Ready in ${seconds}s. WebGPU ${gpu ? "available" : "unavailable — WASM fallback"}.`;
  $("load").disabled = true;
  $("model").disabled = true;
  cardQ.classList.remove("locked");
  cardQ.classList.add("active");
  renderQuestions();
  syncRunEnabled();
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Type a message and hit Triage.";
  outEl.replaceChildren(hint);
  $("text").focus();
}

function markError(msg) {
  stopLoadingClock();
  bar.hidden = true;
  setPill("error", "error");
  loadMsg.textContent = msg;
  $("load").disabled = false;
}

$("load").onclick = async () => {
  const raw = $("model").value.trim();
  if (!raw) {
    markError("Enter a model URL first.");
    return;
  }
  const url = resolveModelUrl(raw);
  setPill("loading", "loading");
  bar.hidden = false;
  $("load").disabled = true;
  const t0 = performance.now();
  startLoadingClock(t0);
  try {
    agent = await Agent.load(url);
    markReady(((performance.now() - t0) / 1000).toFixed(1), !!navigator.gpu);
  } catch (e) {
    markError(`Load failed: ${e.message}`);
  }
};

async function triage() {
  if (!agent || runBtn.disabled) return;
  const text = $("text").value.trim();
  if (!text) {
    runMeta.textContent = "Type a message first.";
    return;
  }
  runBtn.disabled = true;
  runMeta.textContent = "Running…";
  outEl.replaceChildren();
  const t0 = performance.now();
  try {
    const r = await agent.predict({ body: text }, questions);
    const ms = (performance.now() - t0).toFixed(0);
    runMeta.textContent = `${ms}ms · ${r.usage.input_tokens} input tokens`;
    for (const [qid, a] of Object.entries(r.answers)) {
      // Built with DOM APIs throughout: qids, labels, and scores all derive
      // from user input and must never pass through innerHTML.
      const row = document.createElement("div");
      row.className = "answer";
      const value = a.type === "choice" ? String(a.choice)
        : a.type === "score" ? a.score.toFixed(2)
        : a.noul.toFixed(2);
      const headline = document.createElement("div");
      headline.append(document.createTextNode(`${qid}: `));
      const bold = document.createElement("b");
      bold.textContent = value;
      headline.append(bold);
      const conf = a.answer_confidence ?? a.confidence ?? 0;
      const confSpan = document.createElement("span");
      confSpan.className = "probs";
      confSpan.textContent = ` (conf ${conf.toFixed(2)})`;
      headline.append(confSpan);
      row.append(headline);
      const barOuter = document.createElement("div");
      barOuter.className = "conf";
      const barInner = document.createElement("div");
      barInner.style.width = `${Math.round(Math.min(1, Math.max(0, conf)) * 100)}%`;
      barOuter.append(barInner);
      row.append(barOuter);
      if (a.probabilities) {
        const probs = document.createElement("div");
        probs.className = "probs";
        probs.textContent = Object.entries(a.probabilities)
          .sort((x, y) => y[1] - x[1])
          .slice(0, 4)
          .map(([k, v]) => `${k} ${Number(v).toFixed(2)}`)
          .join(" · ");
        row.append(probs);
      }
      outEl.append(row);
    }
  } catch (e) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = `Failed: ${e.message}`;
    outEl.replaceChildren(p);
    runMeta.textContent = "";
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.onclick = triage;
$("text").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") triage();
});
