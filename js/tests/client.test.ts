import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LayaClient } from '../src/client';
import { LayaAPIError, LayaTimeoutError, LayaNetworkError } from '../src/errors';
import { choice, score, noul } from '../src/primitives';

describe('LayaClient', () => {
  let mockFetch: any;

  beforeEach(() => {
    mockFetch = vi.fn();
  });

  it('correctly serializes request and parses answers', async () => {
    const mockResponse = {
      model: 'test-model',
      answers: {
        dept: {
          type: 'choice',
          choice: 'billing',
          probabilities: { billing: 0.9, tech: 0.1 },
          confidence: 0.8,
          action: { act_probability: 0.9 }
        },
        urgency: {
          type: 'noul',
          noul: 0.95,
          confidence: 0.9,
          action: { act_probability: 0.95 }
        },
        frustration: {
          type: 'score',
          score: 1.5,
          legend: { '0': 'low', '1': 'high' },
          probabilities: { '0': 0.1, '1': 0.9 },
          confidence: 0.85,
          action: { act_probability: 0.85 }
        }
      },
      usage: { input_tokens: 10, output_tokens: 20 }
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse
    });

    const client = new LayaClient({ baseURL: 'http://test', fetch: mockFetch });
    const response = await client.systemOne({
      state: 'test state',
      questions: {
        dept: choice('team?', { billing: 'bills', tech: 'bugs' }),
        urgency: noul('urgent?'),
        frustration: score('frustrated?', ['low', 'high'])
      }
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toBe('http://test/v1/decide');
    const reqBody = JSON.parse(callArgs[1].body);
    expect(reqBody.state).toBe('test state');
    expect(reqBody.questions.dept.type).toBe('choice');
    expect(reqBody.questions.urgency.type).toBe('noul');
    expect(reqBody.questions.frustration.type).toBe('score');

    expect(response.answers.dept.type).toBe('choice');
    if (response.answers.dept.type === 'choice') {
      expect(response.answers.dept.choice).toBe('billing');
    }
  });

  it('retries on 503 error', async () => {
    mockFetch
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ answers: {} }) });

    const client = new LayaClient({ fetch: mockFetch, retryDelay: 10, retries: 2 });
    await client.systemOne({ state: '', questions: {} });
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it('throws LayaAPIError on 422', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({ code: 'invalid_request', message: 'Bad param' })
    });

    const client = new LayaClient({ fetch: mockFetch, retries: 0 });
    await expect(client.systemOne({ state: '', questions: {} })).rejects.toThrow(LayaAPIError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('throws LayaTimeoutError on timeout', async () => {
    mockFetch.mockImplementation(async (url: string, opts: any) => {
      return new Promise((_, reject) => {
        setTimeout(() => {
          const err = new Error('AbortError');
          err.name = 'AbortError';
          reject(err);
        }, 100);
      });
    });

    const client = new LayaClient({ fetch: mockFetch, timeout: 10, retries: 0 });
    await expect(client.systemOne({ state: '', questions: {} })).rejects.toThrow(LayaTimeoutError);
  });
});
