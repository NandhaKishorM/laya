import {
  Agent, Router, type TokenizerLike, type LongPredictOptions, type LongResult,
  type RoutedLongResult, type SystemOneResult,
} from "../../src/index.js";

declare const agent: Agent;
declare const router: Router;
declare const encodeOnly: Omit<TokenizerLike, "decode">;
const oldTokenizer: TokenizerLike = encodeOnly;
const options: LongPredictOptions = { window: 80, stride: 40, batchSize: 2, aggregate: "auto", hooksTimeout: 1 };
const questions = { q: { type: "noul", instructions: "Is this urgent?" } };
const long: Promise<LongResult> = agent.predictLong("document", questions, options);
const routed: Promise<RoutedLongResult> = router.predictLong("document", questions, { ...options, lang: "en" });
const ordinary: Promise<SystemOneResult> = agent.predict("document", questions);
async function inspect() {
  const truncated: boolean | number | undefined = (await long).usage.truncated;
  const originalTruncated: boolean | undefined = (await ordinary).usage.truncated;
  const windowStart: number | undefined = (await long).answers.q.window?.token_start;
  return [truncated, originalTruncated, windowStart];
}
// @ts-expect-error long prediction uses the checkpoint budget, not ordinary per-call truncation
agent.predictLong("document", questions, { maxLen: 128 });
// @ts-expect-error headMaxLen does not belong to the long options
router.predictLong("document", questions, { headMaxLen: 64 });
// @ts-expect-error a window winner's confidence is not a calibrated document confidence gate
agent.predictLong("document", questions, { minConfidence: 0.9 });
// @ts-expect-error only the reference auto aggregator is supported
agent.predictLong("document", questions, { aggregate: "mean" });
void oldTokenizer;
void routed;
void inspect;
