import { LayaValidationError } from './errors.js';

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Reject JS values JSON.stringify would silently drop or rewrite. */
export function validateJson(value: unknown, path = 'request', ancestors = new Set<object>()): void {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (typeof value !== 'object' || value === null) {
    throw new LayaValidationError(`${path} must contain only JSON values`);
  }
  if (ancestors.has(value)) throw new LayaValidationError(`${path} contains a circular reference`);
  if (ancestors.size > 100) throw new LayaValidationError(`${path} exceeds the JSON nesting limit (100)`);
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new LayaValidationError(`${path} must be a plain JSON object`);
  }
  ancestors.add(value);
  if (Array.isArray(value)) {
    for (const item of value) validateJson(item, `${path}[]`, ancestors);
  } else {
    for (const [key, item] of Object.entries(value)) validateJson(item, `${path}.${key}`, ancestors);
  }
  ancestors.delete(value);
}

export function validateQuestions(questions: unknown, allowEmpty: boolean): void {
  if (!isRecord(questions) || (!allowEmpty && Object.keys(questions).length === 0)) {
    throw new LayaValidationError('questions must be an object containing at least one question');
  }
  for (const [id, q] of Object.entries(questions)) {
    const invalid = (message: string): never => { throw new LayaValidationError(`questions.${id}: ${message}`); };
    if (!isRecord(q)) invalid('expected a question object');
    const question = q as Record<string, unknown>;
    if (!Object.hasOwn(question, 'instructions')) invalid('instructions is required');
    if (Object.keys(question).some(key => !['type', 'instructions', 'criteria'].includes(key))) {
      invalid('unknown question field');
    }
    const criteria = question.criteria;
    switch (question.type) {
      case 'choice':
        if (Array.isArray(criteria)) {
          if (!criteria.length || criteria.some(c => typeof c !== 'string') || new Set(criteria).size !== criteria.length) {
            invalid('choice criteria must contain unique string labels');
          }
        } else if (!isRecord(criteria) || !Object.keys(criteria).length) {
          invalid('choice requires a nonempty criteria object or array');
        }
        break;
      case 'score':
        if (!Array.isArray(criteria) || !criteria.length) invalid('score requires a nonempty criteria array');
        break;
      case 'noul':
        if (criteria !== undefined && criteria !== null &&
            (!isRecord(criteria) || Object.keys(criteria).some(k => k !== 'true' && k !== 'false'))) {
          invalid('noul criteria may only describe true and false');
        }
        break;
      default: invalid('type must be choice, score, or noul');
    }
  }
}

export function validateTimeout(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 2_147_483_647) {
    throw new LayaValidationError('timeoutMs must be between 0 and 2147483647');
  }
}
