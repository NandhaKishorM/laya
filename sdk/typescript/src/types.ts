/** Values that survive the JSON transport without being changed or discarded. */
export type JsonValue = string | number | boolean | null | readonly JsonValue[] | { readonly [key: string]: JsonValue };
export type State = string | readonly JsonValue[] | { readonly [key: string]: JsonValue } | null;

export interface ChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: JsonValue;
  readonly criteria: readonly string[] | Readonly<Record<string, JsonValue>>;
}
export interface ScoreQuestion {
  readonly type: 'score';
  readonly instructions: JsonValue;
  readonly criteria: readonly JsonValue[];
}
export interface NoulQuestion {
  readonly type: 'noul';
  readonly instructions: JsonValue;
  readonly criteria?: Readonly<{ true?: JsonValue; false?: JsonValue }> | null;
}
export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Readonly<Record<string, Question>>;
export type ModelName = 'english' | 'multilingual' | 'typed-decisions';
export type ModelAlias = ModelName | 'en' | 'laya' | 'default' | 'multi' | 'ml' | 'laya-multilingual'
  | 'typed' | 'typed_decisions' | 'laya-typed-decisions' | 'decisions';

export interface RequestOptions {
  /** Abort waiting for the response. Running inference may still finish on the server. */
  signal?: AbortSignal;
  /** Override the client timeout. Zero disables the timeout. */
  timeoutMs?: number;
}
export interface PredictOptions extends RequestOptions {
  /** Local Laya checkpoint or alias. Overrides the client's configured model. */
  model?: string;
}

interface AnswerBase {
  /** Optional action metadata returned by Laya. */
  action?: { act_probability: number };
}
export interface ChoiceAnswer<Label extends string = string> extends AnswerBase {
  type: 'choice';
  confidence: number;
  choice: Label;
  probabilities: Record<Label, number>;
}
export interface ScoreAnswer extends AnswerBase {
  type: 'score';
  confidence: number;
  /** Expected zero-based rubric index, not a normalized probability. */
  score: number;
  legend: Record<string, JsonValue>;
  probabilities: Record<string, number>;
}
export interface NoulAnswer extends AnswerBase {
  type: 'noul';
  /** Optional confidence metadata returned by Laya. */
  confidence?: number;
  /** P(true), between 0 and 1. */
  noul: number;
}
type Labels<C> = C extends readonly string[] ? C[number] : Extract<keyof C, string>;
export type ChoiceLabels<Q extends ChoiceQuestion> = Labels<Q['criteria']>;
export type Answer<Q extends Question = Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer<ChoiceLabels<Q>> : Q extends ScoreQuestion ? ScoreAnswer : NoulAnswer;
export type Answers<Q extends Questions> = { -readonly [K in keyof Q]: Answer<Q[K]> };

export interface LanguageDetection {
  script: string;
  script_profile: Record<string, number>;
  language: string | null;
  is_english: boolean;
  language_undecided: boolean;
  diacritic_rate: number;
  non_latin_fraction: number;
}
export interface RouteDecision {
  model: ModelName;
  repo: string;
  reason: string;
  detection: LanguageDetection | null;
  workflow: string | null;
}
export interface Prediction<Q extends Questions = Questions> {
  model: string;
  answers: Answers<Q>;
  usage: { input_tokens: number; output_tokens: number };
  /** Optional routing metadata returned by Laya. */
  routing?: RouteDecision;
}
/** Laya's self-hosted health probe. */
export interface Health {
  status: 'ok';
  loaded: ModelName[];
  device: string;
}

/** Preserve question IDs, primitive types, and literal choice labels when declaring a schema. */
export function defineQuestions<const Q extends Questions>(questions: Q): Q {
  return questions;
}
