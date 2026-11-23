import { Agent, Router, type PredictContext } from "../../src/index.js";

// `onPredictStart` / `onPredictEnd` take one callable or a list of them on every public entry point,
// as `PredictHookArg` and Python's `normalise_hooks` do.
const start = (_ctx: PredictContext): void => {};
const end = (_ctx: PredictContext): void => {};
const state = "Please refund the duplicate charge";
const questions = { intent: { type: "noul", instructions: "Refund?" } };

const router = new Router({ onPredictStart: [start, start], onPredictEnd: end });
void router.predict(state, questions, { onPredictStart: [start], onPredictEnd: [end] });
void router.predictBatch([{ state, questions }], null, { onPredictStart: [start] });
void router.predictLong(state, questions, { onPredictEnd: [end] });

declare const agent: Agent;
void agent.predict(state, questions, { onPredictStart: [start, start] });
void agent.predictBatch([state], questions, { onPredictEnd: [end] });
void agent.predictLong(state, questions, { onPredictStart: [start] });

// @ts-expect-error a list holds callables only
new Router({ onPredictStart: [start, 42] });
