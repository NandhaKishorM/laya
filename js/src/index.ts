/**
 * Laya SDK main entry point.
 * @module index
 */

// Client
export { LayaClient } from "./client";

// Compat alias — TypeSafeClient -> LayaClient (for search-replace migration)
export { LayaClient as TypeSafeClient } from "./client";

// Builders
export { choice, score, noul } from "./primitives";

// Types
export type {
  LayaClientOptions,
  DecideRequest,
  DecideResponse,
  BatchDecideRequest,
  BatchDecideResponse,
  ChoiceQuestion,
  ScoreQuestion,
  NoulQuestion,
  AnyQuestion,
  ChoiceAnswer,
  ScoreAnswer,
  NoulAnswer,
  AnyAnswer,
  State,
  ActionResult,
} from "./types";

// Errors
export { LayaError, LayaAPIError, LayaTimeoutError, LayaNetworkError } from "./errors";
