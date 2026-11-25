import { readFile, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, join } from 'node:path';
import { Agent } from '../dist/index.js';
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('usage: node scripts/predict_long_onnx_parity.mjs python.json comparison.json');
const report = JSON.parse(await readFile(input, 'utf8'));
for (const [name, expected] of Object.entries(report.sha256)) {
  if (basename(name) !== name) throw new Error(`invalid artifact name: ${name}`);
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(join(report.bundle, name))) hash.update(chunk);
  if (hash.digest('hex') !== expected) throw new Error(`artifact changed since Python comparison: ${name}`);
}
const agent = await Agent.load(report.bundle, { device: 'cpu', numThreads: 2, expectedSha256: report.sha256 });
const provider = agent.provider;
const inputs = [];
agent.provider = { ...provider, async runEncoder(batch) {
  inputs.push(structuredClone({ input_ids: batch.inputIds, attention_mask: batch.attentionMask,
    marker_pos: batch.markerPos, marker_mask: batch.markerMask, qtype: batch.qtype }));
  return provider.runEncoder(batch);
} };
let maxAnswerDelta = 0;
let compared = 0;
for (const c of report.cases) {
  inputs.length = 0;
  const began = performance.now();
  c.typescript = await agent.predictLong(c.state, c.questions, c.options);
  c.typescript_inputs = structuredClone(inputs);
  c.typescript_seconds = (performance.now() - began) / 1000;
  c.differences = [];
  function compare(a,b,path) {
    if (typeof a === 'number' && typeof b === 'number') {
      const delta = Math.abs(a-b);
      if (path.startsWith('answers.') && !path.includes('.window.')) maxAnswerDelta = Math.max(maxAnswerDelta, delta);
      if (!Number.isFinite(a) || !Number.isFinite(b) || delta > 1e-4) c.differences.push({path,python:a,typescript:b,delta});
      compared++;
    } else if (a !== null && b !== null && typeof a === 'object' && typeof b === 'object') {
      for (const key of new Set([...Object.keys(a),...Object.keys(b)])) compare(a[key],b[key],path?`${path}.${key}`:key);
    } else if (a !== b && path !== 'model') c.differences.push({path,python:a,typescript:b});
  }
  compare(c.python_inputs,c.typescript_inputs,'inputs');
  compare(c.python,c.typescript,'');
  console.log(c.name, JSON.stringify(c.typescript.usage), 'differences', c.differences.length);
}
report.environment.node = process.version;
report.environment.node_onnxruntime = (await import('onnxruntime-node')).env.versions.node;
report.comparison = { tolerance: 1e-4, max_answer_absolute_delta: maxAnswerDelta, numeric_fields_compared: compared,
  model_name_difference: 'Existing TS laya-rl-agent vs Python laya-rl-agent-onnx',
  differences: report.cases.reduce((n,c)=>n+c.differences.length,0) };
await writeFile(output, JSON.stringify(report,null,2));
console.log(JSON.stringify(report.comparison));
if(report.comparison.differences) process.exitCode = 1;
