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

test('health accepts the liveness-only answer a locked-down server gives an anonymous probe', async () => {
  // A server with LAYA_API_KEY set answers an unauthenticated /health with {status: 'ok'} and
  // withholds loaded/device, because those name resident checkpoints, their revision SHAs and
  // the host device state. That is a healthy response and must not be a LayaResponseError.
  const client = new Laya({ fetch: async () => json({ status: 'ok' }) });
  assert.deepEqual(await client.health(), { status: 'ok' });

  // the detail is still validated whenever the server does send it
  const bad = new Laya({ fetch: async () => json({ status: 'ok', loaded: ['nope'], device: 'cpu' }) });
  await assert.rejects(bad.health(), LayaResponseError);
  const badDevice = new Laya({ fetch: async () => json({ status: 'ok', device: 7 }) });
  await assert.rejects(badDevice.health(), LayaResponseError);
  // and a missing or wrong status is still a failure
  for (const payload of [{}, { status: 'degraded' }]) {
    await assert.rejects(new Laya({ fetch: async () => json(payload) }).health(), LayaResponseError);
  }
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

test('invalid model and control values fail before sending a request', async () => {
  const client = new Laya({ fetch: async () => { assert.fail('must not send'); } });
  for (const options of [
    { model: '' }, { model: 1 },
    { task: '' }, { task: 12 },
    { lang: 12 }, { lang: '   ' }, { langGuess: 42 },
    { maxLen: 0 }, { maxLen: 2.5 }, { headMaxLen: -1 },
    { minConfidence: -0.1 }, { minConfidence: 1.5 }, { minConfidence: NaN },
    { minConfidence: true }, { minConfidence: [] }, { minConfidence: {} },
    { minConfidence: { 'choice:2': 1.5 } }, { minConfidence: { 'choice:2': 'high' } },
    { minConfidence: { 'choice:2': NaN } },
  ]) {
    await assert.rejects(client.predict('hello', questions, options), LayaValidationError, JSON.stringify(options));
  }
  for (const model of ['', '  ', 1]) assert.throws(() => new Laya({ model }), LayaValidationError);
});

test('per-request controls reach the wire under their server names', async () => {
  const client = new Laya({ fetch: async (_url, init) => {
    assert.deepEqual(JSON.parse(init.body), {
      state: 'hello',
      questions,
      task: 'typed',
      lang: 'de',
      lang_guess: 'fr',
      max_len: 2048,
      head_max_len: 256,
      min_confidence: 0.8,
    });
    return json(prediction);
  } });
  await client.predict('hello', questions, {
    task: 'typed', lang: 'de', langGuess: 'fr',
    maxLen: 2048, headMaxLen: 256, minConfidence: 0.8,
  });
});

test('a per-bucket threshold map reaches the wire as min_confidence', async () => {
  // The server's abstention gate resolves each answer's threshold from its own option-count
  // bucket (#394), so the map must arrive verbatim: re-keying or dropping `default` here
  // would silently change which answers abstain.
  const map = { 'choice:2': 0.9, default: 0.3 };
  const client = new Laya({ fetch: async (_url, init) => {
    assert.deepEqual(JSON.parse(init.body).min_confidence, map);
    return json(prediction);
  } });
  await client.predict('hello', questions, { minConfidence: map });
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

// The key set is what a live `/v1/systemone` answer carries: both agents build these six keys for
// any non-empty question set, and `options` joins them when the head budget leaves some question's
// options sharing a token span (#538). The counts are chosen to exercise a cut state, which a
// checkpoint that small cannot produce, and both reported `tokens_per_option` shapes.
const truncationReport = { input_tokens: 43, output_tokens: 0, state_tokens: 120,
  state_tokens_dropped: 76, truncated: true, truncated_questions: ['refund'] };

test('the truncation report reaches the caller verbatim and is validated when sent', async () => {
  const reported = structuredClone(prediction);
  reported.usage = truncationReport;
  const client = new Laya({ fetch: async () => json(reported) });
  assert.deepEqual((await client.predict('Hello', questions)).usage, truncationReport);

  const collapsed = structuredClone(reported);
  collapsed.usage.options = { refund: { total: 4, distinct: 1, tokens_per_option: null },
    team: { total: 2, distinct: 1, tokens_per_option: 4 } };
  const collapseClient = new Laya({ fetch: async () => json(collapsed) });
  assert.deepEqual((await collapseClient.predict('Hello', questions)).usage.options,
    collapsed.usage.options);

  // a server answering only the two keys Jev decodes is still a valid prediction
  const legacy = structuredClone(prediction);
  const legacyClient = new Laya({ fetch: async () => json(legacy) });
  assert.deepEqual((await legacyClient.predict('Hello', questions)).usage,
    { input_tokens: 10, output_tokens: 0 });

  // but a half-report is refused rather than passed through as a budget fact
  for (const mutate of [
    u => { delete u.input_tokens; },
    u => { u.state_tokens = '120'; },
    u => { u.state_tokens_dropped = 76.5; },
    u => { u.truncated = 'true'; },
    u => { u.truncated_questions = 'refund'; },
    u => { u.truncated_questions = [1]; },
    u => { u.options = []; },
    u => { u.options = { refund: { total: 4, distinct: 1 } }; },
    u => { u.options = { refund: { total: 4, distinct: 5, tokens_per_option: 4 } }; },
    u => { u.options = { refund: { total: 4, distinct: 1, tokens_per_option: '4' } }; },
  ]) {
    const bad = structuredClone(reported);
    mutate(bad.usage);
    await assert.rejects(new Laya({ fetch: async () => json(bad) }).predict('Hello', questions),
      LayaResponseError, JSON.stringify(bad.usage));
  }
});

// A live `/v1/systemone` answer's `routing.detection`, verbatim, for a state whose subject and
// notes are English but whose customer message is German. Every branch of `laya.lang.analyse()`
// reports the same eight keys, which is what tests/test_router.py pins on the Python side;
// `mixed_segment` is the one that names the segment that pulled the state off English (#384).
const mixedDetection = {
  script: 'latin', script_profile: { latin: 1.0 }, language: 'de', is_english: false,
  language_undecided: false, diacritic_rate: 0.0083, non_latin_fraction: 0.0,
  mixed_segment: 'Ich möchte meine Bestellung stornieren, danke',
};

test('routing detection reports the segment that pulled a mostly-English state off English', async () => {
  const routed = structuredClone(prediction);
  routed.routing = { ...route, detection: mixedDetection,
    reason: 'Latin script, mostly English, but a line or field reads as \'de\'' };
  const client = new Laya({ fetch: async () => json(routed) });
  const result = await client.predict('Hello', questions);
  assert.equal(result.routing.detection.mixed_segment, 'Ich möchte meine Bestellung stornieren, danke');

  // null is the ordinary answer: the route read the state as one language
  const single = structuredClone(routed);
  single.routing.detection.mixed_segment = null;
  const singleClient = new Laya({ fetch: async () => json(single) });
  assert.equal((await singleClient.predict('Hello', questions)).routing.detection.mixed_segment, null);

  // a deployment predating #384 sends no such key, and one display field must not fail the call
  const older = structuredClone(routed);
  delete older.routing.detection.mixed_segment;
  const olderClient = new Laya({ fetch: async () => json(older) });
  assert.equal((await olderClient.predict('Hello', questions)).routing.detection.language, 'de');

  for (const segment of [7, false, ['Ich'], {}]) {
    const bad = structuredClone(routed);
    bad.routing.detection.mixed_segment = segment;
    await assert.rejects(new Laya({ fetch: async () => json(bad) }).predict('Hello', questions),
      LayaResponseError, JSON.stringify(segment));
  }
});

// Captured from `tests/sdk_server_fixture.py` over HTTP with `min_confidence: 0.35`, so the three
// answer types each carry their real gate report. The counts are what that fixture's random weights
// give; `abstention: 'unevaluated'` is not reachable there -- every answer it builds carries a
// usable confidence -- so only the two states the gate really reported here are asserted verbatim.
const gateQuestions = {
  team: { type: 'choice', instructions: 'Team?', criteria: ['billing', 'technical'] },
  urgency: { type: 'score', instructions: 'Urgency?', criteria: ['low', 'medium', 'high'] },
  refund: questions.refund,
};
const gateReport = {
  model: 'laya-rl-agent', usage: { input_tokens: 10, output_tokens: 0 },
  answers: {
    team: { type: 'choice', choice: 'technical', probabilities: { billing: 0.4965, technical: 0.5035 },
      confidence: 0.0, answer_confidence: 0.5035, action: { act_probability: 0.5017 },
      abstention: 'passed', abstention_threshold: 0.35 },
    urgency: { type: 'score', score: 1.0105, probabilities: { '0': 0.3338, '1': 0.3218, '2': 0.3444 },
      legend: { '0': 'low', '1': 'medium', '2': 'high' }, confidence: 0.0003, answer_confidence: 0.3444,
      action: { act_probability: 0.5238 }, low_confidence: true, abstention: 'abstained',
      abstention_threshold: 0.35 },
    refund: { type: 'noul', noul: 0.499, confidence: 0.501, answer_confidence: 0.501,
      action: { act_probability: 0.3968 }, abstention: 'passed', abstention_threshold: 0.35 },
  },
};

test('the confidence an answer was gated on and the gate report reach the caller', async () => {
  const client = new Laya({ fetch: async () => json(gateReport) });
  const result = await client.predict('hello', gateQuestions);
  assert.deepEqual(result.answers, gateReport.answers);
  assert.equal(result.answers.urgency.low_confidence, true);
  assert.equal(result.answers.urgency.abstention, 'abstained');
  assert.equal(result.answers.team.abstention, 'passed');
  assert.equal(result.answers.team.abstention_threshold, 0.35);

  // 'unevaluated' is the third state core reports (laya/confidence.py); accepted like the others.
  const unevaluated = structuredClone(gateReport);
  unevaluated.answers.refund.abstention = 'unevaluated';
  assert.equal((await new Laya({ fetch: async () => json(unevaluated) })
    .predict('hello', gateQuestions)).answers.refund.abstention, 'unevaluated');

  // An ungated call answers exactly the payload it answered before the gate existed: no abstention,
  // no threshold, no flag. Absence is the report, so it must not read as a failure.
  const ungated = structuredClone(gateReport);
  for (const answer of Object.values(ungated.answers)) {
    delete answer.low_confidence;
    delete answer.abstention;
    delete answer.abstention_threshold;
  }
  const bare = await new Laya({ fetch: async () => json(ungated) }).predict('hello', gateQuestions);
  assert.equal(bare.answers.urgency.abstention, undefined);
  assert.equal(bare.answers.urgency.low_confidence, undefined);

  // A deployment predating `answer_confidence` (#126) still answers, exactly as it did before.
  const older = structuredClone(ungated);
  for (const answer of Object.values(older.answers)) delete answer.answer_confidence;
  assert.equal((await new Laya({ fetch: async () => json(older) })
    .predict('hello', gateQuestions)).answers.team.choice, 'technical');

  for (const mutate of [
    answer => { answer.abstention = 'skipped'; },
    answer => { delete answer.abstention_threshold; },
    answer => { answer.abstention_threshold = 1.5; },
    answer => { answer.abstention_threshold = '0.35'; },
    answer => { answer.low_confidence = false; },
    answer => { delete answer.abstention; },
    answer => { answer.answer_confidence = 'high'; },
    answer => { answer.answer_confidence = 2; },
  ]) {
    const payload = structuredClone(gateReport);
    mutate(payload.answers.urgency);
    await assert.rejects(new Laya({ fetch: async () => json(payload) })
      .predict('hello', gateQuestions), LayaResponseError);
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
