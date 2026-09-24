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
  it("keeps a closing sentence that is not a sign-off (Python parity)", () => {
    const body = "Please review the draft when you can.\nIt is two pages.\nThanks for the quick reply.";
    expect(cleanEmailBody(body)).toBe(body);
  });
  it("does not cut words that merely start like a closing", () => {
    const body = "Please review the draft when you can.\nIt is two pages.\nThanksgiving is next week.";
    expect(cleanEmailBody(body)).toBe(body);
  });
  it("still cuts a real sign-off with a capitalised name", () => {
    const out = cleanEmailBody("Please review the draft when you can.\nIt is two pages.\nThanks,\nMaria");
    expect(out).toBe("Please review the draft when you can.\nIt is two pages.");
  });
  it("cuts warmest regards plus a diacritic name", () => {
    const out = cleanEmailBody("Please review the draft when you can.\nIt is two pages.\nWarmest regards,\nŁukasz");
    expect(out).toBe("Please review the draft when you can.\nIt is two pages.");
  });
  it("cuts thanks in advance plus a two-word name", () => {
    const out = cleanEmailBody("Please review the draft when you can.\nIt is two pages.\nThanks in advance,\nPriya Nair");
    expect(out).toBe("Please review the draft when you can.\nIt is two pages.");
  });
  it("bounds input to 4x maxChars before regex work (Python parity)", () => {
    const out = cleanEmailBody("word ".repeat(3000) + "\nOn Mon, Bob wrote:\nold text");
    expect(out.length).toBe(3000);
    expect(out).not.toContain("old text");
  });
  it("guard preset has jailbreak and harm_severity", () => {
    const g = guardQuestions() as Record<string, any>;
    expect(g.jailbreak.type).toBe("noul");
    expect(g.harm_severity.type).toBe("score");
    expect(g.harm_severity.criteria.length).toBe(4);
  });
});
