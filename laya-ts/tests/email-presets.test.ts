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
  // `From:` opens ordinary prose as well as a reply header, and a reply header always carries
  // the sender, so the marker only cuts when an address follows -- Python's `From:\s.*[@<]`.
  // The looser `From:\s.+$` this port shipped with deleted the rest of every request whose body
  // happened to contain a line starting "From: ", which is issue #338 on the Python side.
  it("keeps a request whose prose line starts with From: (Python parity)", () => {
    const body = "Hi team, the export failed again this morning.\n"
      + "From: my side the integration works, but the downstream job still times out.\n"
      + "Could you take a look before Friday?";
    expect(cleanEmailBody(body)).toBe(body);
  });
  it("still cuts a From: line that carries an address", () => {
    const out = cleanEmailBody("Please refund my order\n\nFrom: Bob <bob@example.com>\nold text");
    expect(out).toContain("Please refund my order");
    expect(out).not.toContain("old text");
  });
  // A bare `From: Name` header has no address, so it is told apart from a sentence by its
  // neighbours: the English client lines are the translations of the `De:` pair below.
  it("cuts a bare From: name header followed by Sent:", () => {
    const out = cleanEmailBody("Please refund order 123.\nFrom: Maria Souza\nSent: Monday\nold");
    expect(out).toBe("Please refund order 123.");
  });
  it("does not cut a bare From: name with no header after it", () => {
    const body = "Please refund order 123.\nFrom: Maria Souza\nPlease help with my refund.";
    expect(cleanEmailBody(body)).toBe(body);
  });
  it("cuts a De: name header followed by a dated Date: (Python parity)", () => {
    const out = cleanEmailBody("Please refund order 123.\nDe: Maria Souza\nDate: 12/09/2026\nold");
    expect(out).toBe("Please refund order 123.");
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
  // Python's `_DISCLAIMER` ties the English branch to a disclaimer noun and a disclaimer tail;
  // laya-ts matched the bare word, so a one-sentence body that merely used "confidential" was
  // cleaned to "" and the model was scored on an empty state. Expectations are Python's.
  it("keeps a request that only mentions the word confidential (Python parity)", () => {
    for (const body of [
      "Is this confidential?",
      "What is your confidentiality policy?",
      "Please keep this confidential but process my refund.",
      "Please treat this as confidential.",
      "This is confidential - can you help?",
      "Is the attached document confidential?",
      "Confidential: I need a refund.",
      "Please unlock my account, the contents are not confidential to anyone.",
      "This message is intended solely for the named addressee.",
      "Please forward this to billing. It is intended for the use of the recipient only.",
    ]) {
      expect(cleanEmailBody(body), body).toBe(body);
    }
  });
  it("never cleans a body down to nothing (Python parity)", () => {
    expect(cleanEmailBody("Is this confidential?").trim()).not.toBe("");
  });
  it("still drops the real footers those branches exist for (Python parity)", () => {
    for (const body of [
      "This email is confidential and intended solely for the named addressee.",
      "This message is confidential and intended solely for the use of the individual to whom it is addressed.",
      "The information in this email is confidential and may be privileged.",
      "This email and any files transmitted with it are\nconfidential and intended solely for the named addressee.",
    ]) {
      expect(cleanEmailBody(body).trim(), body).toBe("");
    }
  });
  it("keeps the request around a footer (Python parity)", () => {
    expect(
      cleanEmailBody(
        "My account is locked.\nThis email is confidential and intended solely for the named addressee.\nPlease unlock it.",
      ),
    ).toBe("My account is locked. Please unlock it.");
    expect(
      cleanEmailBody("Please unlock it. This email is confidential and intended solely for the named addressee."),
    ).toBe("Please unlock it.");
  });
  it("guard preset has jailbreak and harm_severity", () => {
    const g = guardQuestions() as Record<string, any>;
    expect(g.jailbreak.type).toBe("noul");
    expect(g.harm_severity.type).toBe("score");
    expect(g.harm_severity.criteria.length).toBe(4);
  });
});
