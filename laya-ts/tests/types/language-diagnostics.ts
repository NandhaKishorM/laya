import {
  analyse, Router, type AnalyseOptions, type AnalyseResult, type LanguageDiagnostic,
  type LanguageDiagnosticReason, type RouteOptions, type RouteBatchOptions, type PredictBatchOptions,
} from "../../src/index.js";

const router = new Router();
const state = { customer: "Ich kann mich nicht anmelden und brauche Hilfe" };
const questions = { intent: { type: "noul", instructions: "Refund?" } };
const analysisOpts: AnalyseOptions = { explain: true };
const routeOpts: RouteOptions & AnalyseOptions = { explain: true, langGuess: () => null };
const batchOpts: RouteBatchOptions = { explain: true, hooksRaise: true };
const oldRouteOpts: RouteOptions = { lang: "en" };
const oldBatchOpts: PredictBatchOptions = { hooksRaise: true };
const result: AnalyseResult = analyse(state, analysisOpts);
const diagnostic: LanguageDiagnostic | undefined = result.diagnostic;
const reason: LanguageDiagnosticReason | undefined = diagnostic?.reason;
const path: Array<string | number> | null | undefined = diagnostic?.source?.path;
const line: number | null | undefined = diagnostic?.source?.line;
router.route(state, questions, routeOpts);
router.routeBatch([{ state, questions }], batchOpts);
router.route(state, questions, oldRouteOpts);
router.routeBatch([{ state, questions }], oldBatchOpts);
// @ts-expect-error explain is boolean
analyse(state, { explain: "true" });
// @ts-expect-error shared prediction options do not acquire routing diagnostics
router.predict(state, questions, { explain: true });
// @ts-expect-error prediction batches keep their previous API
router.predictBatch([{ state, questions }], { explain: true });
// @ts-expect-error long prediction does not opt into diagnostic metadata
router.predictLong(state, questions, { explain: true });
void reason;
void path;
void line;
