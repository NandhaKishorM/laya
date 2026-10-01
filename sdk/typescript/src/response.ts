import { LayaResponseError } from './errors.js';
import type { Questions } from './types.js';
import { isRecord } from './validation.js';

function expect(condition: unknown, field: string): asserts condition {
  if (!condition) throw new LayaResponseError(`Invalid Laya response: ${field}`);
}
const number = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
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
}

export function validateHealth(value: unknown): void {
  expect(isRecord(value) && value.status === 'ok' && typeof value.device === 'string', 'health');
  expect(Array.isArray(value.loaded) && value.loaded.every(model), 'loaded');
}

export function validatePrediction(value: unknown, questions: Questions): void {
  expect(isRecord(value) && typeof value.model === 'string' && isRecord(value.answers), 'prediction');
  expect(isRecord(value.usage) && Number.isInteger(value.usage.input_tokens) && Number.isInteger(value.usage.output_tokens), 'usage');
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
