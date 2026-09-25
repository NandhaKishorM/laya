import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import {
  Laya, LayaAPIError, LayaAbortError, LayaConnectionError, LayaTimeoutError,
  LayaValidationError, LayaResponseError, emailQuestions, triageQuestions,
} from 'laya-client';

const questions = { refund: { type: 'noul', instructions: 'Refund?' } };
const json = (value, status = 200) => new Response(JSON.stringify(value), { status });
const health = { status: 'ok', loaded: [], device: 'auto' };
const route = { model: 'english', repo: 'convaiinnovations/laya', reason: 'test', detection: null, workflow: null };
const prediction = { model: 'laya-rl-agent', routing: route, usage: { input_tokens: 10, output_tokens: 0 },
  answers: { refund: { type: 'noul', noul: 0.9, confidence: 0.9, action: { act_probability: 0.8 } } } };

test('ESM and CommonJS exports can be consumed by ordinary JavaScript', () => {
  const cjs = createRequire(import.meta.url)('laya-client');
  assert.equal(typeof cjs.Laya, 'function');
  assert.deepEqual(cjs.triageQuestions(), triageQuestions());
});

test('systemone request preserves Unicode/JSON, auth, model override and proxy prefix', async () => {
  let calls = 0;
  const client = new Laya({ baseURL: 'https://example.com/laya///', apiKey: 'key', headers: { 'X-App': 'test' },
    fetch: async (url, init) => {
      calls++;
      assert.equal(url, 'https://example.com/laya/v1/systemone');
      assert.equal(init.method, 'POST');
      assert.equal(init.headers.get('Authorization'), 'Bearer key');
      assert.equal(init.headers.get('Content-Type'), 'application/json');
      assert.equal(init.headers.get('X-App'), 'test');
      assert.deepEqual(JSON.parse(init.body), { state: { text: 'नमस्ते', count: 0 }, questions, model: 'en' });
      return json(prediction);
    } });
  await client.predict({ text: 'नमस्ते', count: 0 }, questions, { model: 'en' });
  assert.equal(calls, 1);
});

test('Laya health uses GET and accepts the existing server response', async () => {
  const calls = [];
  const client = new Laya({ fetch: async (url, init) => {
    calls.push([new URL(url).pathname, init.method, init.body]);
    return json(health);
  } });
  assert.deepEqual(await client.health(), health);
  assert.deepEqual(calls, [['/health', 'GET', undefined]]);
  assert.equal(client.route, undefined);
});

const allAnswerQuestions = {
  team: { type: 'choice', instructions: 'Team?', criteria: ['billing', 'support'] },
  urgency: { type: 'score', instructions: 'Urgency?', criteria: ['low', 'high'] },
  refund: questions.refund,
};
const allAnswerPrediction = {
  model: 'english', usage: { input_tokens: 20, output_tokens: 10 },
  answers: {
    team: { type: 'choice', choice: 'billing', probabilities: { billing: 0.8, support: 0.2 }, confidence: 0.6 },
    urgency: { type: 'score', score: 0.7, probabilities: { '0': 0.3, '1': 0.7 }, legend: { '0': 'low', '1': 'high' }, confidence: 0.4 },
    refund: { type: 'noul', noul: 0.9 },
  },
};

test('default request omits model and supports every Laya answer type', async () => {
  const client = new Laya({ baseURL: 'http://127.0.0.1:8000', apiKey: 'test-key', fetch: async (url, init) => {
    assert.equal(url, 'http://127.0.0.1:8000/v1/systemone');
    assert.equal(init.headers.get('Authorization'), 'Bearer test-key');
    assert.deepEqual(JSON.parse(init.body), {
      state: 'Refund please',
      questions: { ...allAnswerQuestions, team: { ...allAnswerQuestions.team, criteria: { billing: null, support: null } } },
    });
    return json(allAnswerPrediction);
  } });
  assert.deepEqual(await client.predict('Refund please', allAnswerQuestions), allAnswerPrediction);
  assert.deepEqual(allAnswerQuestions.team.criteria, ['billing', 'support'], 'normalization must not mutate the schema');
});

test('client model default and prediction overrides select local checkpoints', async () => {
  const models = [];
  const client = new Laya({ model: 'english', fetch: async (_url, init) => {
    models.push(JSON.parse(init.body).model);
    return json(prediction);
  } });
  await client.predict('hello', questions);
  await client.predict('hello', questions, { model: 'multilingual' });
  assert.deepEqual(models, ['english', 'multilingual']);
});

test('unsupported routing options and invalid models fail before sending a request', async () => {
  const client = new Laya({ fetch: async () => { assert.fail('must not send'); } });
  for (const options of [{ task: 'typed' }, { lang: 'hi' }, { model: '' }, { model: 1 }]) {
    await assert.rejects(client.predict('hello', questions, options), LayaValidationError);
  }
  for (const model of ['', '  ', 1]) assert.throws(() => new Laya({ model }), LayaValidationError);
});

test('bad JavaScript inputs fail before fetch without lossy serialization', async () => {
  let calls = 0;
  const client = new Laya({ fetch: async () => { calls++; return json({}); } });
  const cyclic = {}; cyclic.self = cyclic;
  for (const state of [undefined, 1, true, { n: NaN }, { n: Infinity }, { n: 1n }, { n: undefined }, new Date(), cyclic]) {
    await assert.rejects(client.predict(state, questions), LayaValidationError);
  }
  for (const q of [{}, { q: { type: 'wat', instructions: 'test' } },
    { q: { type: 'choice', instructions: 'Pick', criteria: [] } },
    { q: { type: 'choice', instructions: 'Pick', criteria: ['a', 'a'] } },
    { q: { type: 'score', instructions: 'Score', criteria: [] } },
    { q: { type: 'noul', instructions: 'True?', criteria: { yes: 'yes' } } },
    { q: { type: 'noul', instructions: 'True?', typo: 1 } }]) {
    await assert.rejects(client.predict('hello', q), LayaValidationError);
  }
  assert.equal(calls, 0);
});

test('invalid configuration is rejected', () => {
  for (const baseURL of ['oops', 'ftp://example.com', 'https://x/?key=secret', 'https://user:password@x/', 'https://x/#fragment']) {
    assert.throws(() => new Laya({ baseURL }), LayaValidationError);
  }
  for (const timeoutMs of [-1, Infinity, NaN, 2 ** 31]) {
    assert.throws(() => new Laya({ timeoutMs }), LayaValidationError);
  }
});

test('structured API errors keep status, code and validation details; no retries', async () => {
  let calls = 0;
  const details = [{ location: ['questions'], message: 'invalid' }];
  const client = new Laya({ fetch: async () => { calls++; return json({ error: { message: 'Invalid request', code: 'validation_error', details } }, 422); } });
  await assert.rejects(client.predict('Hello', questions), error => {
    assert.ok(error instanceof LayaAPIError);
    assert.equal(error.status, 422);
    assert.equal(error.code, 'validation_error');
    assert.deepEqual(error.details, details);
    return true;
  });
  assert.equal(calls, 1);
});

test('FastAPI errors preserve detail strings and validation arrays', async () => {
  for (const [status, detail] of [[401, 'invalid or missing bearer token'],
    [422, [{ loc: ['body', 'questions'], msg: 'Field required', type: 'missing' }]],
    [413, 'request body too large'], [500, 'inference failed']]) {
    await assert.rejects(new Laya({ fetch: async () => json({ detail }, status) }).predict('hello', questions), error => {
      assert.ok(error instanceof LayaAPIError);
      assert.equal(error.status, status);
      assert.equal(error.code, 'http_error');
      assert.equal(error.message, typeof detail === 'string' ? detail : `Laya returned HTTP ${status}`);
      assert.deepEqual(error.details, detail);
      return true;
    });
  }
});

test('non-JSON proxy errors remain API errors, malformed successes are response errors', async () => {
  const client = new Laya({ fetch: async () => new Response('<html>bad gateway</html>', { status: 502 }) });
  await assert.rejects(client.health(), error => error instanceof LayaAPIError && error.status === 502);
  for (const raw of ['<html>success?</html>', 'null', '[]', '{}']) {
    await assert.rejects(new Laya({ fetch: async () => new Response(raw) }).health(), LayaResponseError);
  }
});

test('malformed answers cannot masquerade as typed predictions', async () => {
  for (const mutate of [p => { p.answers = {}; }, p => { p.answers.refund.noul = 2; },
    p => { p.answers.refund.type = 'choice'; }, p => { p.routing.repo = ['repo', 'sub']; },
    p => { p.routing.model = ['english']; },
    p => { p.usage.input_tokens = '10'; }]) {
    const payload = structuredClone(prediction);
    mutate(payload);
    await assert.rejects(new Laya({ fetch: async () => json(payload) }).predict('Hello', questions), LayaResponseError);
  }
});

test('prediction validation rejects malformed required fields and optional extensions', async () => {
  for (const mutate of [p => { delete p.answers.team.confidence; }, p => { delete p.answers.urgency.legend; },
    p => { p.answers.team.choice = 'unknown'; }, p => { delete p.answers.team.probabilities.billing; },
    p => { p.answers.urgency.score = 2; }, p => { p.answers.refund.confidence = 'high'; },
    p => { p.answers.refund.action = { act_probability: 2 }; }, p => { p.routing = null; }]) {
    const payload = structuredClone(allAnswerPrediction);
    mutate(payload);
    await assert.rejects(new Laya({ fetch: async () => json(payload) }).predict('hello', allAnswerQuestions), LayaResponseError);
  }
});

test('network errors preserve the cause', async () => {
  const cause = new TypeError('fetch failed');
  await assert.rejects(new Laya({ fetch: async () => { throw cause; } }).health(), error =>
    error instanceof LayaConnectionError && error.cause === cause);
});

const hangingFetch = async (_url, init) => new Promise((_resolve, reject) => {
  init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
});

test('timeout aborts fetch and per-request override wins', async () => {
  const client = new Laya({ timeoutMs: 1000, fetch: hangingFetch });
  await assert.rejects(client.health({ timeoutMs: 10 }), LayaTimeoutError);
});

test('timeout includes response body consumption', async () => {
  const client = new Laya({ timeoutMs: 10, fetch: async (_url, init) => ({
    ok: true, text: () => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
    }),
  }) });
  await assert.rejects(client.health(), LayaTimeoutError);
});

test('caller cancellation works during a request and before fetch', async () => {
  const controller = new AbortController();
  const client = new Laya({ timeoutMs: 0, fetch: hangingFetch });
  const promise = client.health({ signal: controller.signal });
  controller.abort('user cancelled');
  await assert.rejects(promise, LayaAbortError);
  let called = false;
  await assert.rejects(new Laya({ fetch: async () => { called = true; return json(health); } })
    .health({ signal: controller.signal }), LayaAbortError);
  assert.equal(called, false);
});

test('presets are fresh and custom email categories are copied', () => {
  const first = triageQuestions(); first.frustration.criteria[0] = 'changed';
  assert.equal(triageQuestions().frustration.criteria[0], 'calm and neutral');
  const categories = { finance: { description: 'money' } };
  const email = emailQuestions(categories);
  email.category.criteria.finance.description = 'changed';
  assert.equal(categories.finance.description, 'money');
  assert.ok(emailQuestions().category.criteria.billing);
});
