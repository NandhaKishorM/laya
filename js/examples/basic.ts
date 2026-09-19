import { LayaClient, choice, score, noul } from '../src';

async function main() {
  const client = new LayaClient({ baseURL: 'http://localhost:8000' });

  try {
    const { answers } = await client.systemOne({
      state: { body: 'I was charged twice!' },
      questions: {
        department: choice('Which team?', { billing: 'invoices', technical: 'bugs' }),
        isUrgent: noul('Is this urgent?'),
        frustration: score('How frustrated?', ['calm', 'concerned', 'very angry']),
      },
    });

    console.log('Department:', answers.department.type === 'choice' ? answers.department.choice : '');
    console.log('Urgent (prob):', answers.isUrgent.type === 'noul' ? answers.isUrgent.noul : '');
    console.log('Frustration score:', answers.frustration.type === 'score' ? answers.frustration.score : '');
  } catch (error) {
    console.error('Error:', error);
  }
}

main();
