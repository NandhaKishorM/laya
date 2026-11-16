import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Laya, LayaAPIError, LayaAbortError, LayaConnectionError, LayaResponseError,
  LayaTimeoutError, LayaValidationError } from 'laya-client';

const SNAPSHOT = { model: 'support', repo: 'acme/support', reason: 'explicit model override' };

function fixture(payload = SNAPSHOT, options = {}) {
  const seen = [];
  const client = new Laya({ baseURL: 'https://example.com/proxy/', apiKey: 'route-test', ...options,
    fetch: async (url, init) => {
      seen.push({ url, ...init, body: JSON.parse(init.body) });
      return new Response(JSON.stringify(payload));
    },
  });
  return { client, seen };
}

test('route sends one authenticated snapshot request with an empty default question set', async () => {
  const { client, seen } = fixture();
  assert.deepEqual(await client.route('hello'), SNAPSHOT);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].url, 'https://example.com/proxy/v1/route');
  assert.equal(seen[0].method, 'POST');
  assert.equal(seen[0].headers.get('authorization'), 'Bearer route-test');
  assert.equal(seen[0].headers.get('content-type'), 'application/json');
  assert.deepEqual(seen[0].body, { state: 'hello', questions: {} });
});

test('route forwards only supplied routing controls and honors the configured model', async () => {
  const { client, seen } = fixture({ ...SNAPSHOT, diagnostic: null }, { model: 'english' });
  await client.route('hello');
  assert.deepEqual(seen[0].body, { state: 'hello', questions: {}, model: 'english' });
  await client.route({ text: 'hello' }, {}, { model: 'support', task: 'custom-workflow',
    lang: 'de', langGuess: 'fr', explain: true, timeoutMs: 0 });
  assert.deepEqual(seen[1].body, { state: { text: 'hello' }, questions: {}, model: 'support',
    task: 'custom-workflow', lang: 'de', lang_guess: 'fr', explain: true });
  await client.route('hello', {}, { explain: false });
  assert.equal(seen[2].body.explain, false);
});

test('route normalizes choice shorthand without mutating state or question extensions', async () => {
  const { client, seen } = fixture();
  const state = Object.freeze({ text: 'hello' });
  const questions = Object.freeze({ q: Object.freeze({ type: 'choice', instructions: 'Pick',
    criteria: Object.freeze(['__proto__', 'constructor']), option_order: Object.freeze([1, 0]) }) });
  await client.route(state, questions);
  assert.deepEqual(seen[0].body.questions.q, { type: 'choice', instructions: 'Pick',
    criteria: JSON.parse('{"__proto__":null,"constructor":null}'), option_order: [1, 0] });
  assert.deepEqual(questions.q.criteria, ['__proto__', 'constructor']);
});

test('route leaves prediction validation to prediction and keeps JSON question values intact', async () => {
  const { client, seen } = fixture();
  const questions = { intent: { instructions: 'route by this id', extension: { version: 1 } }, unused: null };
  await client.route(['hello'], questions);
  assert.deepEqual(seen[0].body.questions, questions);
  await assert.rejects(client.predict('hello', questions), LayaValidationError);
  assert.equal(seen.length, 1);
});

test('route preserves custom model names, source-less repos and arbitrary JSON diagnostics', async () => {
  for (const diagnostic of [null, false, 0, 'operator-defined', [], { source: null, custom: [true, { key: 1 }] }]) {
    const expected = { ...SNAPSHOT, model: 'attached', repo: null, diagnostic, extension: 'retained' };
    const { client } = fixture(expected);
    assert.deepEqual(await client.route('hello', {}, { explain: true }), expected);
  }
  const { client } = fixture();
  assert.equal(Object.hasOwn(await client.route('hello'), 'diagnostic'), false);
  assert.equal(Object.hasOwn(await client.route('hello', {}, { explain: false }), 'diagnostic'), false);
});

test('route refuses invalid input before fetching', async () => {
  const circular = {}; circular.self = circular;
  const { client, seen } = fixture();
  for (const state of [undefined, 42, true, new Date(), { value: NaN }, circular]) {
    await assert.rejects(async () => client.route(state), LayaValidationError);
  }
  for (const questions of [null, [], 'q', new Date(), { q: undefined }, circular]) {
    await assert.rejects(async () => client.route('hello', questions), LayaValidationError);
  }
  for (const criteria of [[], ['a', 'a'], ['a', 1]]) {
    await assert.rejects(async () => client.route('hello', { q: { type: 'choice', criteria } }), LayaValidationError);
  }
  for (const options of [null, [], false, { explain: 'false' }, { explain: null }, { explain: 1 },
    { model: '' }, { task: ' ' }, { lang: 42 }, { langGuess: '' }, { timeoutMs: -1 }]) {
    await assert.rejects(async () => client.route('hello', {}, options), LayaValidationError);
  }
  assert.equal(seen.length, 0);
});

test('route refuses prediction, hook and unknown options instead of silently dropping them', async () => {
  const { client, seen } = fixture();
  for (const key of ['maxLen', 'headMaxLen', 'minConfidence', 'batchSize', 'sortByLength',
    'hooks', 'onRoute', 'onPredictStart', 'onPredictEnd', 'hooksRaise', 'hooksTimeout',
    'max_len', 'lang_guess', 'media', 'allowVega', 'typo']) {
    await assert.rejects(async () => client.route('hello', {}, { [key]: null }), error =>
      error instanceof LayaValidationError && error.message.includes(key));
  }
  assert.equal(seen.length, 0);
});

test('route rejects malformed snapshots and missing requested diagnostics', async () => {
  for (const payload of [null, [], {}, { ...SNAPSHOT, model: null }, { ...SNAPSHOT, repo: [] },
    { model: 'english', reason: 'missing repo' }, { ...SNAPSHOT, reason: 1 }]) {
    const { client } = fixture(payload);
    await assert.rejects(async () => client.route('hello'), LayaResponseError);
  }
  const { client } = fixture();
  await assert.rejects(async () => client.route('hello', {}, { explain: true }), LayaResponseError);
  for (const text of ['not JSON', '{"model":"support","repo":null,"reason":"hook","diagnostic":1e309}']) {
    const invalid = new Laya({ fetch: async () => new Response(text) });
    await assert.rejects(async () => invalid.route('hello', {}, { explain: true }), LayaResponseError);
  }
});

test('route retains valid operator diagnostics beyond the outgoing request nesting limit', async () => {
  let diagnostic = { source: null };
  for (let i = 0; i < 110; i++) diagnostic = { nested: diagnostic };
  const expected = { ...SNAPSHOT, diagnostic };
  const { client } = fixture(expected);
  assert.deepEqual(await client.route('hello', {}, { explain: true }), expected);
});

test('route preserves server errors without retrying or falling back to inference', async () => {
  for (const status of [401, 404, 405, 413, 422, 500, 503]) {
    const seen = [];
    const client = new Laya({ fetch: async url => {
      seen.push(new URL(url).pathname);
      return new Response(JSON.stringify({ detail: 'specific server refusal' }), { status });
    } });
    await assert.rejects(client.route('hello'), error => error instanceof LayaAPIError &&
      error.status === status && error.message === 'specific server refusal');
    assert.deepEqual(seen, ['/v1/route']);
  }
  const client = new Laya({ fetch: async () => { throw new Error('network unavailable'); } });
  await assert.rejects(client.route('hello'), LayaConnectionError);
});

test('route handles already-aborted and in-flight cancellation through the shared transport', async () => {
  const controller = new AbortController();
  controller.abort('caller stopped');
  const { client, seen } = fixture();
  await assert.rejects(client.route('hello', {}, { signal: controller.signal }), error =>
    error instanceof LayaAbortError && error.cause === 'caller stopped');
  assert.equal(seen.length, 0);
  const active = new AbortController();
  const waiting = new Laya({ fetch: async (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
    active.abort('cancel in fetch');
  }) });
  await assert.rejects(waiting.route('hello', {}, { signal: active.signal }), LayaAbortError);
});

test('route applies per-call timeouts', async () => {
  const client = new Laya({ timeoutMs: 0, fetch: async (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }) });
  await assert.rejects(client.route('hello', {}, { timeoutMs: 10 }), LayaTimeoutError);
});

test('route snapshots request controls before callers mutate them and isolates concurrent calls', async () => {
  const pending = [];
  const client = new Laya({ fetch: async (_url, init) => new Promise(resolve => {
    pending.push({ body: JSON.parse(init.body), resolve });
  }) });
  const options = { explain: true };
  const state = { text: 'first' };
  const first = client.route(state, {}, options);
  options.explain = false;
  state.text = 'changed';
  const second = client.route('second');
  assert.deepEqual(pending[0].body, { state: { text: 'first' }, questions: {}, explain: true });
  pending[1].resolve(new Response(JSON.stringify(SNAPSHOT)));
  pending[0].resolve(new Response(JSON.stringify(SNAPSHOT)));
  await assert.rejects(first, LayaResponseError);
  assert.deepEqual(await second, SNAPSHOT);
});
