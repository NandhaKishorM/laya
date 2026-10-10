import { LayaResponseError } from './errors.js';
import type { Questions } from './types.js';
import { isRecord } from './validation.js';

function expect(condition: unknown, field: string): asserts condition {
  if (!condition) throw new LayaResponseError(`Invalid Laya response: ${field}`);
}
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const count = (value: unknown): value is number => Number.isInteger(value);
const probability = (value: unknown) => number(value) && value >= 0 && value <= 1;
const model = (value: unknown) => typeof value === 'string' && ['english', 'multilingual', 'typed-decisions'].includes(value);

export function validateRoute(value: unknown): void {
  expect(isRecord(value), 'routing');
  expect(model(value.model) && typeof value.repo === 'string' && typeof value.reason === 'string', 'routing model/repo/reason');
  expect(value.workflow === null || typeof value.workflow === 'string', 'routing workflow');
  const detection = value.detection;
  if (detection === null) return;
  expect(isRecord(detection), 'routing detection');
  expect(typeof detection.script === 'string' && isRecord(detection.script_profile), 'script detection');
  expect(Object.values(detection.script_profile).every(number), 'script profile');
  expect(detection.language === null || typeof detection.language === 'string', 'detected language');
  expect(typeof detection.is_english === 'boolean' && typeof detection.language_undecided === 'boolean', 'language flags');
  expect(probability(detection.diacritic_rate) && probability(detection.non_latin_fraction), 'language fractions');
  // Checked when the server sends it, like the optional fields of `health` and of `usage`: one
  // field the route report added (#384) must not fail every prediction on a deployment that
  // predates it. When it is there it is `null` or the segment text, as every branch of
  // `laya.lang.analyse()` reports it.
  if (detection.mixed_segment !== undefined) {
    expect(detection.mixed_segment === null || typeof detection.mixed_segment === 'string',
      'detection mixed segment');
  }
}

export function validateHealth(value: unknown): void {
  // Liveness is the only field every caller gets. The rest is withheld from an unauthenticated
  // probe on a server with LAYA_API_KEY set, so requiring it here would reject a healthy
  // response; each field is still checked when the server does send it.
  expect(isRecord(value) && value.status === 'ok', 'health');
  if (value.device !== undefined) expect(typeof value.device === 'string', 'device');
  if (value.loaded !== undefined) {
    expect(Array.isArray(value.loaded) && value.loaded.every(model), 'loaded');
  }
}

export function validateUsage(value: unknown): void {
  // The two token counts are what Jev decodes, so they are the only required fields: a
  // self-hosted deployment predating the truncation report (#174) answers with those two and is
  // a valid prediction. `Usage` types the fields the current server always sends. Each of the
  // rest is checked whenever the server does send it, and a half-report is refused rather than
  // passed through, because these are the only fields that make a truncated answer visible.
  expect(isRecord(value), 'usage');
  expect(Number.isInteger(value.input_tokens) && Number.isInteger(value.output_tokens), 'usage');
  if (value.state_tokens !== undefined) expect(Number.isInteger(value.state_tokens), 'usage.state_tokens');
  if (value.state_tokens_dropped !== undefined) {
    expect(Number.isInteger(value.state_tokens_dropped), 'usage.state_tokens_dropped');
  }
  if (value.truncated !== undefined) expect(typeof value.truncated === 'boolean', 'usage.truncated');
  if (value.truncated_questions !== undefined) {
    expect(Array.isArray(value.truncated_questions) &&
      value.truncated_questions.every((qid: unknown) => typeof qid === 'string'), 'usage.truncated_questions');
  }
  if (value.options === undefined) return;
  expect(isRecord(value.options), 'usage.options');
  for (const collapse of Object.values(value.options)) {
    expect(isRecord(collapse), 'usage.options');
    expect(count(collapse.total) && count(collapse.distinct) && collapse.distinct <= collapse.total,
      'usage.options');
    expect(collapse.tokens_per_option === null || count(collapse.tokens_per_option),
      'usage.options.tokens_per_option');
  }
}

export function validatePrediction(value: unknown, questions: Questions): void {
  expect(isRecord(value) && typeof value.model === 'string' && isRecord(value.answers), 'prediction');
  validateUsage(value.usage);
  if (value.routing !== undefined) validateRoute(value.routing);
  for (const [id, question] of Object.entries(questions)) {
    const answer = value.answers[id];
    expect(isRecord(answer) && answer.type === question.type, `answers.${id}.type`);
    if (question.type !== 'noul' || answer.confidence !== undefined) {
      expect(probability(answer.confidence), `answers.${id}.confidence`);
    }
    if (answer.action !== undefined) {
      expect(isRecord(answer.action) && probability(answer.action.act_probability), `answers.${id}.action`);
    }
    // Checked when the server sends them, like the optional fields of `health` and of `usage`:
    // `answer_confidence` (#126) and the gate fields (#361) are recent additions to an answer that
    // predates them, so requiring a key here would fail every prediction against a deployment that
    // has not shipped it. A caller that asked for a gate (`minConfidence`) and got no `abstention`
    // back reads it as `undefined` on the typed answer, which is the truth: the gate did not run.
    if (answer.answer_confidence !== undefined) {
      expect(probability(answer.answer_confidence), `answers.${id}.answer_confidence`);
    }
    // HTTP-only metadata: older servers omit it, and Jev defines none for noul.
    if (answer.x_jev_confidence !== undefined) {
      expect(question.type !== 'noul' && probability(answer.x_jev_confidence),
        `answers.${id}.x_jev_confidence`);
    }
    if (answer.low_confidence !== undefined) {
      expect(answer.low_confidence === true, `answers.${id}.low_confidence`);
    }
    if (answer.abstention !== undefined || answer.abstention_threshold !== undefined) {
      // The gate writes both onto the same answer or onto neither, so one without the other is a
      // half-report: an `abstention` with no echoed threshold cannot be re-split, and a threshold
      // with no state says nothing about the answer it was measured against.
      expect(answer.abstention === 'passed' || answer.abstention === 'abstained' ||
        answer.abstention === 'unevaluated', `answers.${id}.abstention`);
      expect(probability(answer.abstention_threshold), `answers.${id}.abstention_threshold`);
    }
    if (question.type === 'noul') {
      expect(probability(answer.noul), `answers.${id}.noul`);
    } else {
      expect(isRecord(answer.probabilities) && Object.values(answer.probabilities).every(probability), `answers.${id}.probabilities`);
      if (question.type === 'choice') {
        const labels = Array.isArray(question.criteria) ? question.criteria : Object.keys(question.criteria);
        expect(typeof answer.choice === 'string' && labels.includes(answer.choice), `answers.${id}.choice`);
        expect(labels.every(label => Object.hasOwn(answer.probabilities as object, label)), `answers.${id}.probabilities labels`);
      } else {
        expect(number(answer.score) && answer.score >= 0 && answer.score <= question.criteria.length - 1, `answers.${id}.score`);
        expect(isRecord(answer.legend), `answers.${id}.legend`);
        expect(question.criteria.every((_, i) => Object.hasOwn(answer.legend as object, String(i)) &&
          Object.hasOwn(answer.probabilities as object, String(i))), `answers.${id}.rubric levels`);
      }
    }
  }
}
