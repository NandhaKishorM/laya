import { describe, expect, it } from "vitest";
import { cleanEmailBody } from "../src/email.js";
import { triageQuestions, guardQuestions } from "../src/presets.js";
describe("email+presets", () => {
  it("cuts quoted history", () => {
    const out = cleanEmailBody("Refund please\n\nOn Mon, Bob wrote:\nold text");
    expect(out).toContain("Refund please"); expect(out).not.toContain("old text");
  });
  it("triage preset has 5 questions", () => {
    expect(Object.keys(triageQuestions()).sort()).toEqual(
      ["churn_risk", "frustration", "intent", "is_urgent", "refund_requested"]);
  });
  it("cuts device footer", () => {
    const out = cleanEmailBody("Please refund my order\n\nSent from my iPhone");
    expect(out).toContain("Please refund my order");
    expect(out).not.toContain("iPhone");
  });
  it("guard preset has jailbreak and harm_severity", () => {
    const g = guardQuestions() as Record<string, any>;
    expect(g.jailbreak.type).toBe("noul");
    expect(g.harm_severity.type).toBe("score");
    expect(g.harm_severity.criteria.length).toBe(4);
  });
});
