// Exercise the installed tarball, outside the package's self-referencing imports.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const sdk = fileURLToPath(new URL('..', import.meta.url));
const npm = process.env.npm_execpath;
assert.ok(npm, 'Run this check with npm run test:package');
const temporary = resolve(tmpdir());
const work = mkdtempSync(join(temporary, 'laya-client-consumer-'));
const consumer = join(work, 'consumer');
const run = (args, cwd) => execFileSync(process.execPath, args,
  { cwd, encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });

try {
  mkdirSync(consumer);
  run([npm, 'run', 'build'], sdk);
  const [packed] = JSON.parse(run([npm, 'pack', '--json', '--ignore-scripts', '--pack-destination', work], sdk));
  writeFileSync(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  run([npm, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', join(work, packed.filename)], consumer);
  const installed = JSON.parse(readFileSync(join(consumer, 'node_modules/laya-client/package.json'), 'utf8'));
  assert.equal(installed.name, 'laya-client');
  assert.deepEqual(installed.dependencies ?? {}, {}, 'the HTTP client must not pull in inference dependencies');

  const exercise = `
    const assert = require('node:assert/strict');
    const questions = { q: { type: 'noul', instructions: 'Ready?' } };
    const result = { model: 'fixture', answers: { q: { type: 'noul', noul: 0.8 } },
      usage: { input_tokens: 2, output_tokens: 0 } };
    let calls = 0;
    const client = new sdk.Laya({ baseURL: 'https://example.com', fetch: async (url, init) => {
      calls++;
      assert.equal(url, 'https://example.com/v1/systemone/batch');
      assert.deepEqual(JSON.parse(init.body), { states: ['hello'], questions });
      return new Response(JSON.stringify({ results: calls === 1 ? [result] : [], total_usage: result.usage }));
    } });
    (async () => {
      assert.deepEqual((await client.predictBatch(['hello'], questions)).results, [result]);
      await assert.rejects(client.predictBatch(['hello'], questions), sdk.LayaResponseError);
      assert.equal(calls, 2);
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `;
  writeFileSync(join(consumer, 'esm.mjs'),
    "import * as sdk from 'laya-client';\nimport { createRequire } from 'node:module';\nconst require = createRequire(import.meta.url);\n" + exercise);
  writeFileSync(join(consumer, 'commonjs.cjs'), "const sdk = require('laya-client');\n" + exercise);
  for (const file of ['esm.mjs', 'commonjs.cjs']) run([file], consumer);

  // Reuse the public consumer contracts against installed declarations, with no source-path alias.
  for (const file of ['types.ts', 'commonjs.cts']) copyFileSync(join(sdk, 'test', file), join(consumer, file));
  const { compilerOptions } = JSON.parse(readFileSync(join(sdk, 'tsconfig.json'), 'utf8'));
  writeFileSync(join(consumer, 'tsconfig.json'), JSON.stringify({
    compilerOptions: { ...compilerOptions, rootDir: '.', outDir: './unused', noEmit: true, types: [] },
    include: ['*.ts', '*.cts'],
  }));
  run([join(sdk, 'node_modules/typescript/bin/tsc'), '-p', join(consumer, 'tsconfig.json')], consumer);
  console.log('Installed tarball: ESM, CommonJS, strict consumer types, and batch validation passed.');
} finally {
  assert.equal(dirname(resolve(work)), temporary);
  assert.ok(basename(work).startsWith('laya-client-consumer-'));
  rmSync(work, { recursive: true, force: true });
}
