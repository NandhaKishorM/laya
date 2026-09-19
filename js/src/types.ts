/**
 * Type definitions for Laya SDK.
 * @module types
 */

/** A question with discrete choices. */
export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string | null>;
}

/** A question with ordered string choices that get mapped to ordinal scores. */
export interface ScoreQuestion {
  type: "score";
  instructions: string;
  criteria: string[];
}

/** A boolean yes/no question. */
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: { true?: string; false?: string };
}

/** Any of the supported question types. */
export type AnyQuestion = ChoiceQuestion | ScoreQuestion | NoulQuestion;

/** Action meta-signal result. */
export interface ActionResult {
  act_probability: number;
}

/** The answer to a choice question. */
export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionResult;
}

/** The answer to a score question. */
export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
  action: ActionResult;
}

/** The answer to a noul question. */
export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  action: ActionResult;
}

/** Any of the supported answer types. */
export type AnyAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

/** The state of the system being analyzed. */
export type State = string | Record<string, unknown> | unknown[];

/** Options for configuring the Laya Client. */
export interface LayaClientOptions {
  baseURL?: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  fetch?: typeof fetch;
}

/** Request to decide on questions given a state. */
export interface DecideRequest {
  state: State;
  questions: Record<string, AnyQuestion>;
  model?: string;
}

/** The decision response. */
export interface DecideResponse {
  model: string;
  answers: Record<string, AnyAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}

/** Request to decide on questions for multiple states in a batch. */
export interface BatchDecideRequest {
  states: State[];
  questions: Record<string, AnyQuestion>;
  model?: string;
}

/** The batch decision response. */
export interface BatchDecideResponse {
  model: string;
  results: DecideResponse[];
  total_usage: { input_tokens: number; output_tokens: number };
}

/** System health response. */
export interface HealthResponse {
  status: string;
  [key: string]: unknown;
}

/** Available models response. */
export interface ModelsResponse {
  data: Array<{ id: string; [key: string]: unknown }>;
  [key: string]: unknown;
}
