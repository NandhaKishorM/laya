import { describe, it, expect } from 'vitest';
import { choice, score, noul } from '../src/primitives';

describe('primitives', () => {
  describe('choice', () => {
    it('supports positional arguments', () => {
      const q = choice('Which team?', { billing: 'invoices', technical: 'bugs' });
      expect(q).toEqual({
        type: 'choice',
        instructions: 'Which team?',
        criteria: { billing: 'invoices', technical: 'bugs' }
      });
    });

    it('supports object arguments', () => {
      const q = choice({ instructions: 'Which team?', criteria: { billing: 'invoices' } });
      expect(q).toEqual({
        type: 'choice',
        instructions: 'Which team?',
        criteria: { billing: 'invoices' }
      });
    });
  });

  describe('score', () => {
    it('supports positional arguments', () => {
      const q = score('How urgent?', ['low', 'medium', 'high']);
      expect(q).toEqual({
        type: 'score',
        instructions: 'How urgent?',
        criteria: ['low', 'medium', 'high']
      });
    });

    it('supports object arguments', () => {
      const q = score({ instructions: 'How urgent?', criteria: ['low', 'high'] });
      expect(q).toEqual({
        type: 'score',
        instructions: 'How urgent?',
        criteria: ['low', 'high']
      });
    });
  });

  describe('noul', () => {
    it('supports positional arguments', () => {
      const q = noul('Is this urgent?');
      expect(q).toEqual({
        type: 'noul',
        instructions: 'Is this urgent?'
      });
    });

    it('supports object arguments with criteria', () => {
      const q = noul({ instructions: 'Is this urgent?', criteria: { true: 'yes', false: 'no' } });
      expect(q).toEqual({
        type: 'noul',
        instructions: 'Is this urgent?',
        criteria: { true: 'yes', false: 'no' }
      });
    });

    it('supports object arguments without criteria', () => {
      const q = noul({ instructions: 'Is this urgent?' });
      expect(q).toEqual({
        type: 'noul',
        instructions: 'Is this urgent?'
      });
    });
  });
});
