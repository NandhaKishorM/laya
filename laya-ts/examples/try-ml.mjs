import { Agent, Router } from "../dist/index.js";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

// run from anywhere: --model wins, else the export next to this script or
// at the repo root; otherwise the published multilingual checkpoint is
// downloaded (export your own with python laya-ts/scripts/export_onnx.py).
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i === -1 || i + 1 >= args.length ? null : args[i + 1];
};
const localDir = flag("--model") ?? ["./model-ml", "../../model-ml"]
  .map((d) => fileURLToPath(new URL(d, import.meta.url)))
  .find((d) => existsSync(d));
let agent;
if (localDir) {
  agent = await Agent.load(localDir);
} else {
  console.error("no local weights found; downloading convaiinnovations/laya (multilingual) — or export your own: python laya-ts/scripts/export_onnx.py --model-dir <ckpt> --out-dir ./model-ml");
  agent = await Agent.load("convaiinnovations/laya", { subfolder: "multilingual" });
}
console.log("loaded:", agent.cfg.encoder ?? "multilingual");

// Direct predict (Hindi -> multilingual weights)
const direct = await agent.predict(
  { body: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।" },
  {
    department: {
      type: "choice",
      instructions: "Which department should handle this request?",
      criteria: {
        billing: "invoices, payments, refunds",
        technical: "bugs, outages, system errors",
        sales: "pricing, new contracts",
        other: "everything else",
      },
    },
    churn_risk: { type: "noul", instructions: "Does the user threaten to cancel or leave?" },
  }
);
console.log(JSON.stringify(direct, null, 2));

// Router path (auto language detect + routing metadata)
const router = new Router();
router.attach("multilingual", agent);
const routed = await router.predict({ body: "Der Kunde wurde zweimal belastet" }, {
  department: {
    type: "choice",
    instructions: "Which department should handle this request?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs, outages", other: "rest" },
  },
});
console.log("routing:", JSON.stringify(routed.routing));
console.log("answer:", JSON.stringify(routed.answers.department));
