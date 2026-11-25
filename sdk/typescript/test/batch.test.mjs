import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { Laya, LayaAPIError, LayaAbortError, LayaConnectionError, LayaResponseError,
  LayaTimeoutError, LayaValidationError } from 'laya-client';

const questions = { refund: { type: 'noul', instructions: 'Refund?' } };
const prediction = value => ({ model: 'fixture', answers: { refund: { type: 'noul', noul: value } },
  usage: { input_tokens: 3, output_tokens: 0 } });
const batch = (...values) => ({ results: values.map(prediction),
  total_usage: { input_tokens: values.length * 3, output_tokens: 0 } });
const json = (value, status = 200) => new Response(JSON.stringify(value), { status });

test('batch sends one authenticated request, preserving order, controls and extensions', async () => {
  const schema = { ...questions, team: { type: 'choice', instructions: 'Team?',
    criteria: ['billing', 'support'], option_order: [1, 0] } };
  const states = Object.freeze(['first', { text: 'नमस्ते', flag: false }, ['last', 0]]);
  const payload = batch(0.1, 0.8, 0.3);
  payload.extra = { deployment: 'local' };
  payload.total_usage.extension = 'kept';
  for (const result of payload.results) {
    result.answers.team = { type: 'choice', choice: 'support', confidence: 0.5,
      probabilities: { billing: 0.2, support: 0.8 }, answer_confidence: 0.8,
      abstention: 'passed', abstention_threshold: 0.4 };
    result.extension = ['kept'];
  }
  let calls = 0;
  const client = new Laya({ baseURL: 'https://example.com/laya/', apiKey: 'test-key',
    model: 'multilingual', headers: { 'X-App': 'test' }, fetch: async (url, init) => {
      calls++;
      assert.equal(url, 'https://example.com/laya/v1/systemone/batch');
      assert.equal(init.method, 'POST');
      assert.equal(init.headers.get('Authorization'), 'Bearer test-key');
      assert.equal(init.headers.get('X-App'), 'test');
      assert.equal(init.headers.get('Content-Type'), 'application/json');
      assert.deepEqual(JSON.parse(init.body), { states, questions: { ...schema,
        team: { ...schema.team, criteria: { billing: null, support: null } } },
        model: 'english', task: 'typed', lang: 'en', lang_guess: 'fr', max_len: 128,
        head_max_len: 64, min_confidence: { default: 0.4 }, batch_size: 2, sort_by_length: true });
      return json(payload);
    } });
  const result = await client.predictBatch(states, schema, { model: 'english', task: 'typed', lang: 'en',
    langGuess: 'fr', maxLen: 128, headMaxLen: 64, minConfidence: { default: 0.4 },
    batchSize: 2, sortByLength: true });
  assert.deepEqual(result, payload);
  assert.deepEqual(schema.team.criteria, ['billing', 'support']);
  assert.equal(calls, 1);
});

test('batch defaults stay absent and client model is inherited', async () => {
  for (const model of [undefined, 'english']) {
    const bodies = [];
    const client = new Laya({ model, fetch: async (_url, init) => {
      bodies.push(JSON.parse(init.body)); return json(batch(0.2));
    } });
    await client.predictBatch(['hello'], questions);
    await client.predictBatch(['hello'], questions, { batchSize: 1, sortByLength: false, minConfidence: 0 });
    const base = { states: ['hello'], questions, ...(model ? { model } : {}) };
    assert.deepEqual(bodies, [base, { ...base, batch_size: 1, sort_by_length: false, min_confidence: 0 }]);
  }
});

test('a deployment accepting more than 64 states gets one complete batch', async () => {
  const states = Array.from({ length: 65 }, (_, i) => `state ${i}`);
  let calls = 0;
  const client = new Laya({ fetch: async (_url, init) => {
    calls++;
    assert.deepEqual(JSON.parse(init.body).states, states);
    return json(batch(...states.map((_, i) => i / 100)));
  } });
  assert.equal((await client.predictBatch(states, questions)).results.length, 65);
  assert.equal(calls, 1);
});

test('batch validates the request that was sent if caller inputs change while waiting', async () => {
  const states = ['one', 'two'];
  const schema = structuredClone(questions);
  let finish;
  const client = new Laya({ fetch: async () => new Promise(resolve => { finish = resolve; }) });
  const pending = client.predictBatch(states, schema);
  states.pop();
  schema.refund.type = 'score';
  schema.other = { type: 'noul', instructions: 'New question' };
  finish(json(batch(0.2, 0.7)));
  assert.deepEqual(await pending, batch(0.2, 0.7));
});

test('invalid states, questions and controls are refused before any request', async () => {
  const cyclic = {}; cyclic.self = cyclic;
  let calls = 0;
  const client = new Laya({ fetch: async () => { calls++; return json(batch(0.2)); } });
  for (const states of [[], null, 'hello', {}, [undefined], new Array(1), [NaN], [Infinity],
    [() => {}], [new Date()], [cyclic], [true]]) {
    await assert.rejects(client.predictBatch(states, questions), LayaValidationError);
  }
  await assert.rejects(client.predictBatch(['hello'], {}), LayaValidationError);
  for (const options of [{ batchSize: 0 }, { batchSize: -1 }, { batchSize: 1.5 }, { batchSize: NaN },
    { batchSize: Infinity }, { batchSize: '2' }, { batchSize: null }, { batchSize: true },
    { sortByLength: 0 }, { sortByLength: 'true' }, { sortByLength: null },
    { task: '' }, { lang: '' }, { langGuess: false }, { model: '' }, { maxLen: 0 },
    { headMaxLen: '64' }, { minConfidence: -1 }, { minConfidence: {} }, { timeoutMs: -1 }]) {
    await assert.rejects(client.predictBatch(['hello'], questions, options), LayaValidationError);
  }
  assert.equal(calls, 0);
});

test('malformed batch envelopes and aggregate counts never return partial results', async () => {
  const good = batch(0.2, 0.7);
  const bad = [null, [], {}, { ...good, results: {} }, { ...good, results: [] },
    batch(0.2), batch(0.2, 0.7, 0.9), { ...good, total_usage: null }, { ...good, total_usage: [] },
    { ...good, total_usage: {} }];
  for (const key of ['input_tokens', 'output_tokens']) {
    for (const value of [undefined, null, -1, 0.5, '3', true, NaN, Infinity]) {
      bad.push({ ...good, total_usage: { ...good.total_usage, [key]: value } });
    }
  }
  for (const payload of bad) {
    let calls = 0;
    const client = new Laya({ fetch: async () => { calls++; return json(payload); } });
    await assert.rejects(client.predictBatch(['one', 'two'], questions), LayaResponseError);
    assert.equal(calls, 1);
  }
});

test('every batch item uses prediction validation and errors identify the failed index', async () => {
  for (const broken of [null, {}, { ...prediction(0.2), usage: null }, prediction(1.5),
    { ...prediction(0.2), answers: {} }]) {
    const payload = { ...batch(0.2, 0.7), results: [prediction(0.2), broken] };
    const client = new Laya({ fetch: async () => json(payload) });
    await assert.rejects(client.predictBatch(['one', 'two'], questions), error =>
      error instanceof LayaResponseError && /results\[1\]/.test(error.message)
      && error.cause instanceof LayaResponseError);
  }
});

test('strict and older prediction shapes are accepted without inventing metadata', async () => {
  const payload = batch(0.2);
  // Aggregate accounting is server-owned: the client must not replace or infer it.
  payload.total_usage.input_tokens = 11;
  const out = await new Laya({ fetch: async () => json(payload) }).predictBatch(['hello'], questions);
  assert.deepEqual(out, payload);
  assert.equal(out.results[0].routing, undefined);
  assert.equal(out.results[0].answers.refund.abstention, undefined);
});

test('batch HTTP errors preserve status/details and never split, retry or fall back', async () => {
  for (const status of [401, 404, 405, 413, 422, 500, 503]) {
    const calls = [];
    const detail = [{ loc: ['body', 'states'], msg: 'fixture refusal' }];
    const client = new Laya({ fetch: async (url, init) => {
      calls.push([url, init.method]); return json({ detail }, status);
    } });
    await assert.rejects(client.predictBatch(['one', 'two'], questions), error => {
      assert.ok(error instanceof LayaAPIError);
      assert.equal(error.status, status);
      assert.deepEqual(error.details, detail);
      return true;
    });
    assert.deepEqual(calls, [['http://127.0.0.1:8000/v1/systemone/batch', 'POST']]);
  }
});

test('batch JSON and connection failures are exposed without retry', async () => {
  for (const [fetcher, ErrorType] of [
    [async () => new Response('{broken'), LayaResponseError],
    [async () => { throw new TypeError('socket failed'); }, LayaConnectionError],
    [async () => ({ ok: true, text: async () => { throw new TypeError('body interrupted'); } }), LayaConnectionError],
  ]) {
    let calls = 0;
    const client = new Laya({ fetch: async (...args) => { calls++; return fetcher(...args); } });
    await assert.rejects(client.predictBatch(['hello'], questions), ErrorType);
    assert.equal(calls, 1);
  }
});

test('batch cancellation before sending and while reading a body shares the transport contract', async () => {
  const controller = new AbortController();
  let calls = 0;
  const client = new Laya({ timeoutMs: 0, fetch: async (_url, init) => {
    calls++;
    return { ok: true, text: () => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
    }) };
  } });
  const pending = client.predictBatch(['hello'], questions, { signal: controller.signal });
  await Promise.resolve();
  controller.abort('caller stopped waiting');
  await assert.rejects(pending, LayaAbortError);
  assert.equal(calls, 1);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
  await assert.rejects(client.predictBatch(['hello'], questions, { signal: controller.signal }), LayaAbortError);
  assert.equal(calls, 1);
});

test('batch timeout includes body consumption and per-call timeout wins', async () => {
  let calls = 0;
  const client = new Laya({ timeoutMs: 0, fetch: async (_url, init) => {
    calls++;
    return { ok: true, text: () => new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true });
    }) };
  } });
  await assert.rejects(client.predictBatch(['hello'], questions, { timeoutMs: 10 }), LayaTimeoutError);
  assert.equal(calls, 1);
});

test('CommonJS batch consumption and successful request cleanup', async () => {
  const { Laya: CommonLaya } = createRequire(import.meta.url)('laya-client');
  const controller = new AbortController();
  const result = await new CommonLaya({ fetch: async () => json(batch(0.4)) })
    .predictBatch(['hello'], questions, { signal: controller.signal });
  assert.equal(result.results[0].answers.refund.noul, 0.4);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});
