import type { BatchRequest } from "../../src/index.js";

export const requests: BatchRequest[] = [
  { state: "ticket", questions: {}, maxLen: 64, headMaxLen: 16 },
  { state: "defaults", questions: {}, maxLen: null, headMaxLen: null },
  { state: "omitted", questions: {} },
];

// @ts-expect-error token budgets must be numbers or null
export const invalid: BatchRequest = { state: "ticket", questions: {}, maxLen: "64" };
