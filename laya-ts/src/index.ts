export const VERSION = "0.1.0";
export { Agent, checkQuestion, toInternal, defaultTokenizer } from "./agent.js";
export type {
  QuestionDef,
  SystemOneResult,
  ChoiceAnswer,
  ScoreAnswer,
  NoulAnswer,
  SystemAnswer,
  SystemUsage,
  ActionInfo,
  AgentCfg,
  AgentOptions,
  PredictOptions,
} from "./agent.js";
export { createNodeProvider, createWebProvider, feed, feedHead, loadNodeBundle, loadWebBundle, PINNED_REVISIONS, resolveRevision } from "./providers.js";
export type { Batch, SessionProvider, ProviderOptions, NodeBundle, WebBundle } from "./providers.js";
export { Router, normaliseName, DEFAULT_MODELS } from "./router.js";
export type { RoutedResult, RouteDecision, ModelName, ModelSpec } from "./router.js";
export { shortlistChoice, predictShortlist, embedFnFromAgent, DEFAULT_SHORTLIST_K } from "./shortlist.js";
export type { EmbedFn, ShortlistMeta } from "./shortlist.js";
export { analyse, isEnglish, guessLatinLanguage, detectScript } from "./lang.js";
export type { AnalyseResult, LatinProfile } from "./lang.js";
export { cleanEmailBody, emailState } from "./email.js";
export {
  PredictContext,
  HookRegistry,
  BaseHook,
  normaliseHooks,
  dispatch,
  aggregateUsage,
  composeHooks,
  defaultHooks,
  setDefaultHooks,
  addDefaultHook,
  clearDefaultHooks,
  HOOK_EVENTS,
} from "./hooks.js";
export type { Hook, HookEvent, HookArg, PredictHook, PredictHookArg } from "./hooks.js";
export { triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions } from "./presets.js";
export {
  decide,
  planFromJsonSchema,
  questionsFromJsonSchema,
  answersToJson,
  SchemaError,
  MAX_PROPERTIES,
  MAX_OPTIONS,
  MAX_SCORE_LEVELS,
} from "./structured.js";
export type { DecisionResult, DecideOptions, DecideRunner, PlannedField } from "./structured.js";
export {
  renderOptions,
  serializeState,
  buildSequence,
  buildQuestionPrefix,
  sequenceWithState,
  softmax,
  confidenceFromProbs,
  answerConfidence,
  clampTemperature,
  tempBucket,
  collateItems,
  TEMP_MIN,
  TEMP_MAX,
} from "./common.js";
export type { QType, InternalQ, CollateItem, CollatedBatch, QuestionPrefix } from "./common.js";
export { q, best, topK, isConfident } from "./dx.js";
export { bpeEncode, metaspaceEncode, encodeWithData, parseTokenizerJson, loadTokenizerJson, CHECKPOINT_IDS, SPECIAL_ALIASES, METASPACE_REPLACEMENT } from "./tokenizer.js";
export type { TokenizerLike, TokenizerData, TokenizerIds, PreTokenizerKind } from "./tokenizer.js";
