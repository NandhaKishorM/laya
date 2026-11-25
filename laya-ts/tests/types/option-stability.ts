import {
  Agent, OptionStabilityHook, calibrateReliability, decideReliability, makeVariants,
  type OptionStabilityOptions, type OptionStabilityReport, type ReliabilityDecision,
} from "../../src/index.js";

declare const agent: Agent;
const opts: OptionStabilityOptions = { accept: null, escalate: null, seed: "shared" };
agent.addHook(new OptionStabilityHook(opts));
const cuts: [number, number] = calibrateReliability([0.8, 0.9], [false, true], { targetAccuracy: 1 });
const decision: ReliabilityDecision | null = decideReliability(0.9, ...cuts);
const variants = makeVariants(["approve", "reject"], "seed", { nShuffles: 2 });
const labels: string[] = variants[0].order;

async function inspect() {
  const result = await agent.predict("state", { q: { type: "choice", criteria: ["approve", "reject"] } });
  if (result.answers.q.type === "choice") {
    const report: OptionStabilityReport | undefined = result.answers.q.reliability;
    const score: number | null | undefined = report?.soft_stability;
    return score;
  }
}

// @ts-expect-error explicit thresholds are required
new OptionStabilityHook({ seed: "missing thresholds" });
// @ts-expect-error thresholds must be numeric or null
new OptionStabilityHook({ accept: "0.9", escalate: null });
// @ts-expect-error invalid probe name
new OptionStabilityHook({ accept: null, escalate: null, probes: "confidence" });
void decision;
void labels;
void inspect;
