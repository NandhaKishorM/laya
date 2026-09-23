import { Laya, defineQuestions, emailQuestions, triageQuestions, type Questions, type Answer } from '@laya/typescript-sdk';

const client = new Laya();
const questions = defineQuestions({
  team: { type: 'choice', instructions: 'Team?', criteria: { billing: null, support: { description: 'help' } } },
  priority: { type: 'score', instructions: 'Urgency?', criteria: ['low', 'high'] },
  refund: { type: 'noul', instructions: 'Refund?' },
  arrayChoice: { type: 'choice', instructions: 'Pick', criteria: ['a', 'b'] },
});
const result = await client.predict('Hello', questions);
const label: 'billing' | 'support' = result.answers.team.choice;
const arrayLabel: 'a' | 'b' = result.answers.arrayChoice.choice;
const probability: number = result.answers.team.probabilities.billing;
const score: number = result.answers.priority.score;
const noul: number = result.answers.refund.noul;
// @ts-expect-error A score answer has no choice field.
result.answers.priority.choice;
// @ts-expect-error Unknown question IDs are not valid.
result.answers.unknown;
// @ts-expect-error Unknown choice labels are not valid.
result.answers.team.probabilities.sales;
// @ts-expect-error Noul criteria must describe true or false.
defineQuestions({ q: { type: 'noul', instructions: 'test', criteria: { yes: 'yes' } } });
// @ts-expect-error A choice must have criteria.
defineQuestions({ q: { type: 'choice', instructions: 'test' } });
// @ts-expect-error Dates are not JSON state.
client.predict(new Date(), questions);

const inline = await client.predict('Hi', { q: { type: 'choice', instructions: 'Pick', criteria: ['x', 'y'] } });
const inlineLabel: 'x' | 'y' = inline.answers.q.choice;
const preset = await client.predict('Hi', triageQuestions());
const intent: 'refund' | 'technical_help' | 'billing_question' | 'information' | 'cancellation' | 'other' = preset.answers.intent.choice;
const custom = await client.predict('Hi', emailQuestions({ finance: 'money', engineering: 'bugs' }));
const category: 'finance' | 'engineering' = custom.answers.category.choice;
const dynamic: Questions = questions;
const answer: Answer | undefined = (await client.predict('Hi', dynamic)).answers.anything;
if (answer?.type === 'choice') {
  const dynamicLabel: string = answer.choice;
  const dynamicProbability: number | undefined = answer.probabilities.anyLabel;
  void [dynamicLabel, dynamicProbability];
}
void [label, arrayLabel, probability, score, noul, inlineLabel, intent, category];
