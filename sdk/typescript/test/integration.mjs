import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { Laya, LayaAPIError, triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions } from 'laya-client';

function assertPredictionParity(result, expected) {
  const { routing, ...prediction } = structuredClone(result);
  for (const [id, answer] of Object.entries(expected.answers)) {
    const actual = prediction.answers[id];
    if (answer.type === 'noul') {
      assert.equal(Object.hasOwn(actual, 'x_jev_confidence'), false, `${id}: noul has no Jev confidence`);
      continue;
    }
    // Derive the HTTP extension independently from the direct model's probabilities,
    // not from the server's enrichment helper or from the received HTTP probabilities.
    const keys = Object.keys(answer.probabilities);
    if (answer.type === 'score') keys.sort((a, b) => Number(a) - Number(b));
    const values = keys.map(key => answer.probabilities[key]);
    const n = values.length;
    const total = values.reduce((a, b) => a + b, 0);
    const p = values.map(value => total ? value / total : 1 / n);
    let confidence = 1;
    if (n > 1 && answer.type === 'choice') {
      confidence = (Math.max(...p) - 1 / n) / (1 - 1 / n);
    } else if (n > 1) {
      const mode = p.indexOf(Math.max(...p));
      const spread = p.reduce((sum, value, i) => sum + value * Math.abs(i - mode), 0);
      const uniformSpread = p.reduce((sum, _, i) => sum + Math.abs(i - (n - 1) / 2) / n, 0);
      confidence = Math.max(0, 1 - spread / uniformSpread);
    }
    assert.equal(typeof actual.x_jev_confidence, 'number', `${id}: HTTP Jev confidence missing`);
    assert.ok(Math.abs(actual.x_jev_confidence - confidence) <= 0.0000500001,
      `${id}: HTTP Jev confidence must match the formula rounded to four decimals`);
    delete actual.x_jev_confidence;
  }
  assert.deepEqual(prediction, expected, 'SDK core answers must exactly match direct Python inference');
}

test('parity checks the HTTP extension without weakening core prediction equality', () => {
  const expected = { model: 'test', usage: { input_tokens: 1, output_tokens: 0 }, answers: {
    choice: { type: 'choice', probabilities: { a: 0.8, b: 0.2 }, choice: 'a' },
    score: { type: 'score', probabilities: { '0': 0.1, '1': 0.8, '2': 0.1 }, score: 1 },
    single: { type: 'choice', probabilities: { only: 1 }, choice: 'only' },
    singleScore: { type: 'score', probabilities: { '0': 1 }, score: 0 },
    noul: { type: 'noul', noul: 0.9 },
  } };
  const actual = structuredClone(expected);
  for (const [id, value] of Object.entries({ choice: 0.6, score: 0.7, single: 1, singleScore: 1 })) {
    actual.answers[id].x_jev_confidence = value;
  }
  const before = structuredClone(actual);
  assertPredictionParity(actual, expected);
  assert.deepEqual(actual, before, 'parity check must not mutate the response');
  for (const mutate of [
    r => { delete r.answers.choice.x_jev_confidence; },
    r => { r.answers.choice.x_jev_confidence = 0.61; },
    r => { r.answers.score.x_jev_confidence = 0.8; },
    r => { r.answers.noul.x_jev_confidence = 0.9; },
    r => { r.answers.choice.probabilities.a = 0.7; },
    r => { r.answers.choice.choice = 'b'; },
    r => { r.usage.input_tokens = 2; },
    r => { r.answers.score.unexpected = true; },
  ]) {
    const broken = structuredClone(actual);
    mutate(broken);
    assert.throws(() => assertPredictionParity(broken, expected), assert.AssertionError);
  }
});

test('JavaScript → HTTP → Python Router → real offline Agent inference', { timeout: 90_000 }, async t => {
  const script = fileURLToPath(new URL('../../../tests/sdk_server_fixture.py', import.meta.url));
  const server = spawn(process.env.PYTHON ?? 'python3', [script], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', chunk => { stderr += chunk; });
  t.after(async () => {
    if (server.exitCode === null && server.signalCode === null) {
      const exited = once(server, 'exit');
      server.kill('SIGTERM');
      const timer = setTimeout(() => server.kill('SIGKILL'), 5000);
      try { await exited; } finally { clearTimeout(timer); }
    }
  });
  const fixture = await new Promise((resolve, reject) => {
    const lines = createInterface({ input: server.stdout });
    const timer = setTimeout(() => reject(new Error(`Server startup timed out: ${stderr}`)), 60_000);
    const failed = code => { clearTimeout(timer); reject(new Error(`Server exited (${code}): ${stderr}`)); };
    server.once('exit', failed);
    server.once('error', error => { clearTimeout(timer); reject(error); });
    lines.on('line', line => {
      if (line.startsWith('LAYA_TEST_SERVER=')) {
        clearTimeout(timer);
        server.off('exit', failed);
        resolve(JSON.parse(line.slice('LAYA_TEST_SERVER='.length)));
      }
    });
  });
  const client = new Laya({ baseURL: fixture.baseURL, apiKey: 'integration-test', timeoutMs: 10_000 });
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try { await client.health(); ready = true; break; } catch { await delay(50); }
  }
  assert.ok(ready, `Server never became ready: ${stderr}`);

  const result = await client.predict(fixture.state, fixture.questions, { model: 'english' });
  assertPredictionParity(result, fixture.expected);
  assert.equal(result.routing.model, 'english');
  assert.equal(result.answers.single.choice, 'only');
  assert.equal(result.answers.single.probabilities.only, 1);
  assert.ok(result.usage.input_tokens > 0);
  assert.equal(result.usage.output_tokens, 0);
  assert.equal((await client.predict({ text: 'मुझे पैसे वापस चाहिए' }, fixture.questions)).routing.model, 'multilingual');
  assert.equal((await client.predict('hello', fixture.questions, { model: 'typed_decisions' })).routing.model, 'typed-decisions');

  for (const preset of [triageQuestions, emailQuestions, guardQuestions, moderationQuestions, routerQuestions]) {
    const questions = preset();
    const response = await client.predict({ message: 'hello', body: 'hello', prompt: 'hello', post: 'hello', request: 'hello' }, questions);
    assert.deepEqual(Object.keys(response.answers), Object.keys(questions));
  }
  const unauthenticated = new Laya({ baseURL: fixture.baseURL });
  assert.equal((await unauthenticated.health()).status, 'ok');
  await assert.rejects(unauthenticated.predict('hello', fixture.questions), error =>
    error instanceof LayaAPIError && error.status === 401 && error.message === 'invalid or missing bearer token');
  await assert.rejects(client.predict('x'.repeat(50_001), fixture.questions), error =>
    error instanceof LayaAPIError && error.status === 413 && error.message.includes('state too large'));
  assert.equal((await fetch(`${fixture.baseURL}/v1/predict`, { method: 'POST' })).status, 404);
  assert.equal((await fetch(`${fixture.baseURL}/v1/route`, { method: 'POST' })).status, 404);
});
