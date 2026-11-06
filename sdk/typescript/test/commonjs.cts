import sdk = require('laya-client');
const client = new sdk.Laya();
const questions = sdk.defineQuestions({ q: { type: 'noul', instructions: 'Does it work?' } });
async function run() {
  const result = await client.predict('Yes', questions);
  const probability: number = result.answers.q.noul;
  // @ts-expect-error Noul answers have no score.
  result.answers.q.score;
  const batch: sdk.BatchPrediction<typeof questions> = await client.predictBatch(['Yes'], questions,
    { batchSize: 1, sortByLength: false } satisfies sdk.BatchPredictOptions);
  const usage: sdk.BatchUsage = batch.total_usage;
  const batchProbability: number = batch.results[0]!.answers.q.noul;
  // @ts-expect-error Batch answer types retain the question's type in CommonJS too.
  batch.results[0]!.answers.q.choice;
  void [usage, batchProbability];
  return probability;
}
void run;
