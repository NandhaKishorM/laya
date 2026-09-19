// --- BEFORE (@typesafe-ai/sdk) ---
// import { TypeSafeClient, choice, score, noul } from "@typesafe-ai/sdk";
// const client = new TypeSafeClient();

// --- AFTER (laya-sdk) ---
import { TypeSafeClient, choice, score, noul } from "../src";

const client = new TypeSafeClient({
  baseURL: "http://localhost:8000" // Point to your self-hosted Laya server
});

async function main() {
  // Everything below here remains EXACTLY the same!
  const { answers } = await client.systemOne({
    state: { body: "I was charged twice!" },
    questions: {
      department: choice("Which team should handle this?", {
        billing: "Payment issues",
        technical: "Bugs",
      }),
      isUrgent: noul("Does the message convey urgency?"),
      frustration: score("How frustrated is the customer?", [
        "Calm", "Frustrated", "Very angry",
      ]),
    },
  });

  console.log("Department:", answers.department.type === 'choice' ? answers.department.choice : '');
  console.log("Urgent:", answers.isUrgent.type === 'noul' ? answers.isUrgent.noul : '');
  console.log("Frustration:", answers.frustration.type === 'score' ? answers.frustration.score : '');
}

main();
