import {
  Laya,
  defineQuestions,
} from "@laya/typescript-sdk";
const laya = new Laya({
  baseURL: "http://127.0.0.1:8000",
  // Allow time for the first model download.
  timeoutMs: 600_000,
});

const questions = defineQuestions({
  refund: {
    type: "noul",
    instructions: "Does the customer request a refund?",
  },
});

const state = {
  message: "I was charged twice. Please refund it.",
};



async function main() {
  const health = await laya.health();
  console.log("Server:", health);

  for (let i = 1; i <= 10; i++) {
  const start = performance.now();

  const result = await laya.predict(state, questions, {
    model: "english", // prefer the "english" model for this test
  });

  const elapsedMs = performance.now() - start;

  console.log(
    `Request ${i}: ${elapsedMs.toFixed(2)} ms, ` +
    `tokens=${result.usage.input_tokens}, ` +
    `model=${result.routing?.model ?? result.model}`,
  );
}
}

main().catch((error: unknown) => {
  console.error(error);
});
