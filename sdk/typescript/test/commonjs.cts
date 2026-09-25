import sdk = require('laya-client');
const client = new sdk.Laya();
const questions = sdk.defineQuestions({ q: { type: 'noul', instructions: 'Does it work?' } });
async function run() {
  const result = await client.predict('Yes', questions);
  const probability: number = result.answers.q.noul;
  // @ts-expect-error Noul answers have no score.
  result.answers.q.score;
  return probability;
}
void run;
