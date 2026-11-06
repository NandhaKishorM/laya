/**
 * The design page's request contract must be the contract the client implements.
 *
 * `docs/typescript-sdk.md` is the wire reference for callers who read the design doc instead of the
 * guide, and its control paragraph had drifted into a positive lie: it said `/v1/systemone` does not
 * expose `task` and `lang` and that the SDK rejects those options, while `body()` had long been
 * forwarding them (and `lang_guess`, `max_len`, `head_max_len`, `min_confidence` besides). A caller who
 * believed the page would keep paying server-side defaults it could have overridden.
 *
 * So the page is read through the same seam a server sees: the options are set, the request body the
 * client hands to `fetch` is captured, and the table is compared against the keys that were really put
 * on the wire. Nothing here transcribes the client's option list -- if `body()` gains or drops a control
 * the table has to move with it, in both directions.
 *
 * The same seam holds the page's statement about the paths it calls. It promised `/v1/systemone` was
 * "the only endpoint the client calls" in the sentence that introduces `health`, while `health()` had
 * been issuing `GET /health` the whole time, so an exclusivity claim is now read back off the requests
 * the client actually makes rather than taken on faith from the prose.
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { Laya } from 'laya-client';

const PAGE = fileURLToPath(new URL('../../../docs/typescript-sdk.md', import.meta.url));
const DECLARATION = new URL('../dist/esm/client.d.ts', import.meta.url);
const QUESTIONS = { single: { type: 'choice', instructions: 'Pick one', criteria: ['a', 'b'] } };

/** One valid prediction, so `predict()` gets far enough to build its request. */
const RESULT = {
  model: 'english',
  answers: { single: { type: 'choice', confidence: 1, answer_confidence: 1, choice: 'a',
    probabilities: { a: 1, b: 0 } } },
  usage: { input_tokens: 4, output_tokens: 0 },
};

/** The liveness answer, which is the only `/health` payload every caller is guaranteed to get. */
const HEALTH = { status: 'ok' };

/** The request body the client would send for `options`, minus `state` and `questions`. */
async function controls(options) {
  let body;
  const laya = new Laya({ fetch: async (url, init) => {
    body = JSON.parse(init.body);
    return new Response(JSON.stringify(RESULT), { headers: { 'content-type': 'application/json' } });
  } });
  await laya.predict('the state', QUESTIONS, options);
  const { state, questions, ...rest } = body;
  assert.deepEqual([state, Object.keys(questions)], ['the state', ['single']],
    'a request must always carry the state and the questions it was given');
  return Object.keys(rest).sort();
}

/** The `[method, path]` pairs one call really put on the wire, and the error it threw, if any. */
async function attempt(run) {
  const seen = [];
  const laya = new Laya({ fetch: async (url, init) => {
    const path = new URL(url).pathname;
    seen.push([init.method, path]);
    const payload = path === '/health' ? HEALTH : path === '/v1/systemone/batch'
      ? { results: [RESULT], total_usage: { input_tokens: 4, output_tokens: 0 } } : RESULT;
    return new Response(JSON.stringify(payload), { headers: { 'content-type': 'application/json' } });
  } });
  let error = null;
  try {
    await run(laya);
  } catch (cause) {
    error = cause;
  }
  return { seen, error };
}

/** The page as prose, so a claim is checked against the sentence that makes it rather than the file. */
function sentences(page) {
  return page.split(/(?<=[.!?])\s+/);
}

/** A value for every option the table names, chosen to survive the client's own validation. */
const OPTION_VALUES = { model: 'english', task: 'typed', lang: 'de', langGuess: 'en',
  maxLen: 512, headMaxLen: 24, minConfidence: 0.8 };

function controlTable(page) {
  const lines = page.split('\n');
  const start = lines.indexOf('| option | request field |');
  assert.notEqual(start, -1, 'the design page has no "| option | request field |" table; retarget this');
  const rows = new Map();
  for (const line of lines.slice(start + 2)
      .filter(line => line.startsWith('| `'))) {
    const cells = line.split('|').slice(1, -1).map(cell => cell.trim().replace(/^`|`$/g, ''));
    assert.equal(cells.length, 2, `table row ${line} does not name exactly one option and one field`);
    rows.set(cells[0], cells[1]);
  }
  assert.ok(rows.size, 'the control table has no rows');
  return rows;
}

test('the design page names every control the client puts on the wire, and only those', async () => {
  const page = readFileSync(PAGE, 'utf8');
  const table = controlTable(page);

  const sent = await controls(OPTION_VALUES);
  // The fixture is what exercises each row, so it must stay the same size as the table: a row nobody
  // sets would be an untested mapping, and an option the table omits would be an undocumented one.
  assert.deepEqual([...table.keys()].sort(), Object.keys(OPTION_VALUES).sort(),
    'the table names options ' + JSON.stringify([...table.keys()].sort())
    + ', this fixture exercises ' + JSON.stringify(Object.keys(OPTION_VALUES).sort()));
  assert.deepEqual(sent, [...table.values()].sort(),
    'the page lists ' + JSON.stringify([...table.values()].sort()) + ' as the request fields, the '
    + 'client sends ' + JSON.stringify(sent) + ' for the same options');

  // An option that is dropped on the floor would still be listed and still be accepted, so the mapping
  // is checked one option at a time: setting only `langGuess` must arrive as `lang_guess` and nothing else.
  for (const [option, field] of table) {
    assert.deepEqual(await controls({ [option]: OPTION_VALUES[option] }), [field],
      `predict({ ${option} }) must reach the server as exactly \`${field}\``);
  }

  // The other half of the contract the page states: an option the caller left out stays off the wire, so
  // the deployment's own Router settings answer rather than a client-side default.
  assert.deepEqual(await controls({}), [],
    'a bare predict() sends state and questions only');
});

test('the client calls exactly the endpoints the page says it calls', async () => {
  const page = readFileSync(PAGE, 'utf8');
  const paths = new Map();
  for (const [method, run] of [['predict', l => l.predict('the state', QUESTIONS, {})],
                               ['predictBatch', l => l.predictBatch(['the state'], QUESTIONS)],
                               ['health', l => l.health()]]) {
    const { seen, error } = await attempt(run);
    assert.equal(error, null, `${method}() must reach the fixture, got ${error && error.name}: ${error && error.message}`);
    assert.equal(seen.length, 1, `${method}() made ${JSON.stringify(seen)}, and the page describes one request per call`);
    paths.set(method, seen[0]);
  }
  assert.deepEqual([...paths], [['predict', ['POST', '/v1/systemone']],
    ['predictBatch', ['POST', '/v1/systemone/batch']], ['health', ['GET', '/health']]],
    "the client's network surface moved; update the page and its executable contract");

  // Each call has to be stated by a sentence that names the method and the path it reaches. The page
  // used to promise `/v1/systemone` was "the only endpoint the client calls" in the same sentence that
  // introduced `health`, while `health()` had been issuing `GET /health` the whole time.
  for (const [method, [, path]] of paths) {
    const stated = sentences(page).find(s => s.includes('`' + method + '`') && s.includes('`' + path + '`'));
    assert.ok(stated, `the client sends ${paths.get(method)[0]} ${path} from ${method}(), which no sentence of the page pairs with that method`);
  }

  // An exclusivity claim is decidable at the same seam: "the only endpoint the client calls" holds only
  // if the client calls one path, and "the only <kind> endpoint" holds only if the sentence names the
  // path of every method of that kind. `health` probes; `predict` and `predictBatch` carry decisions.
  const KINDS = { inference: ['predict', 'predictBatch'] };
  for (const sentence of sentences(page)) {
    const claim = /\bthe only\b[^.]{0,40}?\bendpoints?\b/i.exec(sentence);
    if (!claim) continue;
    const kind = (/\bthe only\s+([\w-]+)\s+endpoints?\b/i.exec(sentence) || [])[1];
    const covered = kind === undefined ? [...paths.keys()] : KINDS[kind.toLowerCase()];
    assert.ok(covered, `the page claims ${JSON.stringify(claim[0])} and this test cannot say which client `
      + `methods "${kind}" covers; add it to KINDS`);
    if (kind === undefined) {
      assert.equal(covered.length, 1, `the page calls one endpoint the only one the client calls, but the `
        + `client calls ${covered.length}: ${JSON.stringify([...paths.values()])}`);
    }
    for (const method of covered) {
      assert.ok(sentence.includes('`' + paths.get(method)[1] + '`'),
        `the page claims ${JSON.stringify(claim[0])} without naming ${paths.get(method)[1]}, which ${method}() calls`);
    }
  }
});

test('no sentence on the page claims a control it documents is unsupported', async () => {
  const page = readFileSync(PAGE, 'utf8');
  const fields = [...controlTable(page).values()];
  // Splitting on sentence boundaries rather than the whole page: "a blank `task` is refused locally" is
  // true and must keep passing, while "does not expose `task`" is the lie this guards against.
  for (const sentence of page.split(/(?<=[.!?])\s+/)) {
    if (!/\bdoes not (?:expose|support|accept)\b|\bnot exposed\b|\bnot supported\b|\blegacy\b/i
        .test(sentence)) continue;
    const named = fields.filter(field => sentence.includes('`' + field + '`'));
    assert.deepEqual(named, [],
      `the page denies ${named.join(', ')} while its own table documents it as a request field: `
      + `"${sentence.replace(/\s+/g, ' ').trim()}"`);
  }
});

// The declaration is built, not committed -- `npm test` runs the build before the tests, so a bare
// `node --test test/docs.test.mjs` on a fresh checkout has nothing to read yet.
test('the endpoint surface the page describes is the one the client declares',
  { skip: !existsSync(DECLARATION) && 'run `npm run build` first: dist/esm/client.d.ts is the surface' },
  () => {
    const page = readFileSync(PAGE, 'utf8');
    // The published declaration is what a caller can actually reach, and it is the one place that keeps
    // TypeScript's `private` (a prototype probe would report `body` and `request` as public too).
    const declaration = readFileSync(DECLARATION, 'utf8');
    // A member is public if it is spelled out as a signature; `private body;` collapses to the keyword.
    const declared = [...declaration.matchAll(/^ {4}(?!constructor)(\w+)[<(]/gm)]
      .map(([, name]) => name).sort();
    assert.deepEqual(declared, ['health', 'predict', 'predictBatch'],
      'the client gained or lost a public method; the page describes exactly these three');
    for (const name of declared) {
      assert.ok(page.includes('`' + name + '`'),
        `the page never names the ${name}() method the client declares`);
    }
    // The page's negative claim, on the code side: there is no standalone routing call to make.
    for (const absent of ['route', 'decide', 'classify']) {
      assert.equal(Laya.prototype[absent], undefined, `Laya unexpectedly exposes ${absent}()`);
    }
  });
