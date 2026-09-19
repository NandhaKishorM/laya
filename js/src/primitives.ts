/**
 * Primitive builder functions for constructing questions.
 * @module primitives
 */

import { ChoiceQuestion, ScoreQuestion, NoulQuestion } from './types';

/**
 * Creates a ChoiceQuestion.
 * @param instructions - The question instructions.
 * @param criteria - Mapping of choices to descriptions.
 * @returns A constructed ChoiceQuestion.
 */
export function choice(instructions: string, criteria: Record<string, string>): ChoiceQuestion;
/**
 * Creates a ChoiceQuestion.
 * @param options - An object containing instructions and criteria.
 * @returns A constructed ChoiceQuestion.
 */
export function choice(options: { instructions: string; criteria: Record<string, string> }): ChoiceQuestion;
export function choice(
  arg1: string | { instructions: string; criteria: Record<string, string> },
  arg2?: Record<string, string>
): ChoiceQuestion {
  if (typeof arg1 === 'string') {
    return {
      type: 'choice',
      instructions: arg1,
      criteria: arg2 as Record<string, string>,
    };
  }
  return {
    type: 'choice',
    instructions: arg1.instructions,
    criteria: arg1.criteria,
  };
}

/**
 * Creates a ScoreQuestion.
 * @param instructions - The question instructions.
 * @param criteria - An ordered array of choices.
 * @returns A constructed ScoreQuestion.
 */
export function score(instructions: string, criteria: string[]): ScoreQuestion;
/**
 * Creates a ScoreQuestion.
 * @param options - An object containing instructions and criteria.
 * @returns A constructed ScoreQuestion.
 */
export function score(options: { instructions: string; criteria: string[] }): ScoreQuestion;
export function score(
  arg1: string | { instructions: string; criteria: string[] },
  arg2?: string[]
): ScoreQuestion {
  if (typeof arg1 === 'string') {
    return {
      type: 'score',
      instructions: arg1,
      criteria: arg2 as string[],
    };
  }
  return {
    type: 'score',
    instructions: arg1.instructions,
    criteria: arg1.criteria,
  };
}

/**
 * Creates a NoulQuestion (yes/no).
 * @param instructions - The question instructions.
 * @returns A constructed NoulQuestion.
 */
export function noul(instructions: string): NoulQuestion;
/**
 * Creates a NoulQuestion (yes/no).
 * @param options - An object containing instructions and optional true/false mapping.
 * @returns A constructed NoulQuestion.
 */
export function noul(options: { instructions: string; criteria?: { true?: string; false?: string } }): NoulQuestion;
export function noul(
  arg1: string | { instructions: string; criteria?: { true?: string; false?: string } }
): NoulQuestion {
  if (typeof arg1 === 'string') {
    return {
      type: 'noul',
      instructions: arg1,
    };
  }
  return {
    type: 'noul',
    instructions: arg1.instructions,
    ...(arg1.criteria ? { criteria: arg1.criteria } : {}),
  };
}
