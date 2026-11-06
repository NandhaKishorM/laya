import { Router } from "../../src/index.js";
import type { BatchRequest, PredictBatchOptions } from "../../src/index.js";

declare const router: Router;
declare const requests: BatchRequest[];

// `predictBatch` takes the abstention threshold `predict` takes, under both spellings and under
// both shapes Python's `check_min_confidence` accepts: a scalar and a per-bucket map.
export const scalar: PredictBatchOptions = { minConfidence: 0.8 };
export const snake: PredictBatchOptions = { min_confidence: 0.8 };
export const perBucket: PredictBatchOptions = { minConfidence: { "choice:2": 0.99 } };
export const unset: PredictBatchOptions = { minConfidence: null };

void router.predictBatch(requests, null, { minConfidence: 0.8 });
void router.predictMany(requests, null, { min_confidence: { default: 0.5 } });

// @ts-expect-error routeBatch runs no inference, so it refuses a threshold it cannot gate at
void router.routeBatch(requests, { minConfidence: 0.8 });
