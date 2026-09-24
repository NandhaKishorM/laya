import { Agent, Router } from "../dist/index.js";
import { existsSync } from "node:fs";

// run as: node laya-ts/examples/try-batch.mjs (repo root) or node examples/try-batch.mjs (laya-ts dir)
// --model overrides; otherwise first existing dir wins (repo ships weights at laya-ts/examples/model-ml)
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i === -1 || i + 1 >= args.length ? null : args[i + 1];
};
const MODEL_DIR = flag("--model") ??
  ["./model-ml", "laya-ts/examples/model-ml", "./examples/model-ml"].find((d) => existsSync(d)) ?? "./model-ml";
const agent = await Agent.load(MODEL_DIR);

const states = [
  { body: "charged twice, refund please" },
  { body: "मुझसे दो बार शुल्क लिया गया" },
  { body: "Der Kunde wurde zweimal belastet" },
];
const questions = {
  department: {
    type: "choice",
    instructions: "Which department should handle this request?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages", other: "rest" },
  },
};

// One forward pass for all states (vs N sequential predicts).
const batched = await agent.predictMany(states, questions);
for (const [i, r] of batched.entries()) {
  const a = r.answers.department;
  console.log(`batched[${i}]:`, a?.choice, `conf=${a?.confidence}`, JSON.stringify(a?.probabilities));
}

// Router groups by language, one load per checkpoint, order preserved.
// Same local weights answer both groups so the demo runs offline.
const router = new Router();
router.attach("english", agent);
router.attach("multilingual", agent);
const routed = await router.predictMany(states, questions);
for (const [i, r] of routed.entries()) {
  console.log(`routed[${i}]:`, r.routing.model, "->", r.answers.department?.choice);
}
