import { Laya, triageQuestions } from '@laya/typescript-sdk';

const laya = new Laya({
  baseURL: process.env.LAYA_BASE_URL ?? 'http://127.0.0.1:8000',
  apiKey: process.env.LAYA_API_KEY,
  timeoutMs: 600_000,
});

try {
  const result = await laya.predict(
    { message: 'I was billed twice for March. Please refund the duplicate today.' },
    triageQuestions(),
    { model: 'english'}
  );
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(error);
  process.exitCode = 1;
}
