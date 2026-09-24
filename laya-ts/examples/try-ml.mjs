import { Agent, Router } from "../dist/index.js";
import { existsSync } from "node:fs";

// run as: node laya-ts/examples/try-ml.mjs (repo root) or node examples/try-ml.mjs (laya-ts dir)
// --model overrides; otherwise first existing dir wins (repo ships weights at laya-ts/examples/model-ml)
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i === -1 || i + 1 >= args.length ? null : args[i + 1];
};
const MODEL_DIR = flag("--model") ??
  ["./model-ml", "laya-ts/examples/model-ml", "./examples/model-ml"].find((d) => existsSync(d)) ?? "./model-ml";
const agent = await Agent.load(MODEL_DIR);
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
