import { Laya, defineQuestions, emailQuestions, triageQuestions, type Questions, type Answer,
  type Usage, type LanguageDetection, type ChoiceAnswer } from 'laya-client';

const client = new Laya();
const questions = defineQuestions({
  team: { type: 'choice', instructions: 'Team?', criteria: { billing: null, support: { description: 'help' } } },
  priority: { type: 'score', instructions: 'Urgency?', criteria: ['low', 'high'] },
  refund: { type: 'noul', instructions: 'Refund?' },
  arrayChoice: { type: 'choice', instructions: 'Pick', criteria: ['a', 'b'] },
});
const result = await client.predict('Hello', questions);
const label: 'billing' | 'support' = result.answers.team.choice;
const arrayLabel: 'a' | 'b' = result.answers.arrayChoice.choice;
const probability: number = result.answers.team.probabilities.billing;
const score: number = result.answers.priority.score;
const noul: number = result.answers.refund.noul;
const routingModel: string | undefined = result.routing?.model;
const actionProbability: number | undefined = result.answers.team.action?.act_probability;
const noulConfidence: number | undefined = result.answers.refund.confidence;
const stateTokens: number = result.usage.state_tokens;
const stateTokensDropped: number = result.usage.state_tokens_dropped;
const truncated: boolean = result.usage.truncated;
const truncatedQuestions: string[] = result.usage.truncated_questions;
const collapsedSpans: number | null | undefined = result.usage.options?.refund?.tokens_per_option;
const mixedSegment: string | null | undefined = result.routing?.detection?.mixed_segment;
const answerConfidence: number = result.answers.team.answer_confidence;
const noulAnswerConfidence: number = result.answers.refund.answer_confidence;
const gateState: 'passed' | 'abstained' | 'unevaluated' | undefined = result.answers.priority.abstention;
const gateThreshold: number | undefined = result.answers.priority.abstention_threshold;
const flagged: true | undefined = result.answers.refund.low_confidence;
// @ts-expect-error `flag_low_confidence` writes `True` or writes nothing, so no answer reads false.
const flagAsFalse: false | undefined = result.answers.refund.low_confidence;
// @ts-expect-error A gate state is one of the three core reports, not any string.
const inventedGate: Answer['abstention'] = 'skipped';
const gatedAnswer: ChoiceAnswer = { type: 'choice', confidence: 0.5, answer_confidence: 0.42,
  choice: 'a', probabilities: { a: 0.42 }, low_confidence: true, abstention: 'abstained',
  abstention_threshold: 0.8 };
// @ts-expect-error The gate writes `true` or writes nothing, so a cleared answer has no flag at all.
const clearedAnswer: ChoiceAnswer = { ...gatedAnswer, low_confidence: false };
// An ungated call is the ordinary answer shape: no flag, no gate state, no threshold echo.
const ungatedReport: ChoiceAnswer = { type: 'choice', confidence: 0.5, answer_confidence: 0.42,
  choice: 'a', probabilities: { a: 0.42 } };
// @ts-expect-error An answer must carry the calibrated confidence it was gated on.
const uncalibratedAnswer: ChoiceAnswer = { type: 'choice', confidence: 0.5, choice: 'a',
  probabilities: { a: 0.5 } };
// @ts-expect-error Routing metadata is optional.
result.routing.model;
// @ts-expect-error Action metadata is optional.
result.answers.team.action.act_probability;
// @ts-expect-error The truncation report is a boolean flag, not a count or a string.
const truncatedText: string = result.usage.truncated;
const answeredUsage: Usage = { input_tokens: 43, output_tokens: 0, state_tokens: 12,
  state_tokens_dropped: 0, truncated: false, truncated_questions: [] };
// @ts-expect-error `windows` is Agent.predict_long's own key; /v1/systemone never answers with it.
const windowedUsage: Usage = { ...answeredUsage, windows: 1 };
// @ts-expect-error The mixed segment names the text that was read, not a score.
const mixedAsNumber: number = result.routing?.detection?.mixed_segment;
// @ts-expect-error Every branch of analyse() reports the key, so the type does not make it optional.
const undetermined: LanguageDetection = { script: 'latin', script_profile: { latin: 1.0 }, language: 'de',
  is_english: false, language_undecided: false, diacritic_rate: 0.0083, non_latin_fraction: 0.0 };
// @ts-expect-error /v1/systemone does not expose standalone routing.
client.route('hello');
// @ts-expect-error task must be a string.
client.predict('hello', questions, { task: 12 });
// @ts-expect-error maxLen must be a number.
client.predict('hello', questions, { maxLen: '1024' });
// @ts-expect-error minConfidence must be a threshold or a threshold map.
client.predict('hello', questions, { minConfidence: '0.9' });
client.predict('hello', questions, { model: 'multilingual' });
client.predict('hello', questions, { task: 'typed', lang: 'de', langGuess: 'fr',
  maxLen: 2048, headMaxLen: 256, minConfidence: 0.8 });
// A per-bucket map gates each option count at the level its own calibration earns (#394).
client.predict('hello', questions, { minConfidence: { 'choice:2': 0.9, default: 0.3 } });
// @ts-expect-error A threshold map with a non-threshold value is not a gate.
client.predict('hello', questions, { minConfidence: { 'choice:2': 'high' } });
new Laya({ model: 'english' });
// @ts-expect-error A score answer has no choice field.
result.answers.priority.choice;
// @ts-expect-error Unknown question IDs are not valid.
result.answers.unknown;
// @ts-expect-error Unknown choice labels are not valid.
result.answers.team.probabilities.sales;
// @ts-expect-error Noul criteria must describe true or false.
defineQuestions({ q: { type: 'noul', instructions: 'test', criteria: { yes: 'yes' } } });
// @ts-expect-error A choice must have criteria.
defineQuestions({ q: { type: 'choice', instructions: 'test' } });
// @ts-expect-error Dates are not JSON state.
client.predict(new Date(), questions);

const inline = await client.predict('Hi', { q: { type: 'choice', instructions: 'Pick', criteria: ['x', 'y'] } });
const inlineLabel: 'x' | 'y' = inline.answers.q.choice;
const preset = await client.predict('Hi', triageQuestions());
const intent: 'refund' | 'technical_help' | 'billing_question' | 'information' | 'cancellation' | 'other' = preset.answers.intent.choice;
const custom = await client.predict('Hi', emailQuestions({ finance: 'money', engineering: 'bugs' }));
const category: 'finance' | 'engineering' = custom.answers.category.choice;
const dynamic: Questions = questions;
const answer: Answer | undefined = (await client.predict('Hi', dynamic)).answers.anything;
if (answer?.type === 'choice') {
  const dynamicLabel: string = answer.choice;
  const dynamicProbability: number | undefined = answer.probabilities.anyLabel;
  void [dynamicLabel, dynamicProbability];
}
void [label, arrayLabel, probability, score, noul, inlineLabel, intent, category, routingModel,
  actionProbability, noulConfidence, stateTokens, stateTokensDropped, truncated, truncatedQuestions,
  collapsedSpans, answeredUsage, mixedSegment, undetermined, answerConfidence,
  noulAnswerConfidence, gateState, gateThreshold, flagged, gatedAnswer];
