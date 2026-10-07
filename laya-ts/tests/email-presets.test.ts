import { describe, expect, it } from "vitest";
import { cleanEmailBody, emailState } from "../src/email.js";
import { triageQuestions, guardQuestions } from "../src/presets.js";
describe("email+presets", () => {
  it("cuts quoted history", () => {
    const out = cleanEmailBody("Refund please\n\nOn Mon, Bob wrote:\nold text");
    expect(out).toContain("Refund please"); expect(out).not.toContain("old text");
  });
  it("emailState passes maxChars to the body budget, as Python does (#589)", () => {
    const body = "word ".repeat(1000) + "please wire the money today";
    expect(emailState("s", body).body).toHaveLength(3000);
    const long = emailState("s", body, null, true, {}, 8000);
    expect(long.body).toContain("wire the money");
    expect(long).not.toHaveProperty("max_chars");
    expect(emailState("s", body, null, false, {}, 10).body).toBe(body);
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
  it("cuts extended device footers (Python parity)", () => {
    const request = "Please refund my order";
    for (const footer of [
      "Sent from my iPhone 15 Pro",
      "Sent from my Android phone",
      "Sent from my iPad Pro",
      "Sent from my iPhone using Tapatalk",
      "Sent from my iPhone device",
      "Sent from my iPhone Max",
      "Sent from my iPhone mini",
      "Sent from my iPhone Plus",
    ]) {
      expect(cleanEmailBody(`${request}\n\n${footer}`), footer).toBe(request);
    }
  });
  it("keeps device mentions inside the request (Python parity)", () => {
    for (const sentence of [
      "Sent from my iPhone by mistake.",
      "Sent from my iPad yesterday.",
      "Sent from my Android by mistake.",
      "Sent from my mobile yesterday.",
      "Sent from my iPhone using Tapatalk to report a problem.",
    ]) {
      const body = `Hi support,\n${sentence}\nPlease cancel the duplicate order.`;
      expect(cleanEmailBody(body), sentence).toBe(body);
    }
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
  // French mail, the cases tests/test_email.py checks on the Python side (#736, #913)
  it("cleans a French reply the way Python does (Python parity)", () => {
    const body = [
      "Bonjour,",
      "",
      "J'ai été facturé deux fois sur la facture de mars. Merci de rembourser le double paiement aujourd'hui.",
      "",
      "Cordialement,",
      "Jean Dupont",
      "",
      "Envoyé depuis mon iPhone",
      "",
      "Ce message peut contenir des informations confidentielles. Si vous avez reçu ce message par erreur, merci de le supprimer.",
      "",
      "Le lun. 22 sept. 2026 à 10:14, Support <support@x.com> a écrit :",
      "> Bonjour Jean, nous avons reçu votre demande d'annulation du contrat Enterprise.",
      "",
    ].join("\n");
    expect(cleanEmailBody(body)).toBe(
      "Bonjour,\n\nJ'ai été facturé deux fois sur la facture de mars. " +
        "Merci de rembourser le double paiement aujourd'hui.",
    );
  });
  it("cuts French device footers and keeps French device sentences (Python parity)", () => {
    const request = "Merci de rembourser la facture.";
    for (const footer of [
      "Envoyé depuis mon iPhone",
      "Envoyé de mon iPad.",
      "Envoyé depuis iPhone",
      "Envoyé de iPad",
      "ENVOYÉ DEPUIS MON IPHONE",
    ]) {
      expect(cleanEmailBody(`${request}\n\n${footer}`), footer).toBe(request);
    }
    for (const sentence of ["Envoyé depuis mon iPhone par erreur.", "Envoyé de mon iPad hier."]) {
      const body = `Bonjour,\n${sentence}\n${request}`;
      expect(cleanEmailBody(body), sentence).toBe(body);
    }
  });
  it("cuts French quoted history and sign-offs (Python parity)", () => {
    const cases: Array<[string, string]> = [
      [
        "Voici le justificatif de paiement.\n\n-----Message d'origine-----\n" +
          "De : Marie <marie@acme.com>\nObjet : résilier le contrat\nNous voulons résilier le contrat.",
        "Voici le justificatif de paiement.",
      ],
      [
        "Voici le justificatif.\n\nDe : Marie Dupont\nEnvoyé : lundi 22 septembre 2026\n" +
          "Objet : résilier le contrat\nNous voulons résilier le contrat.",
        "Voici le justificatif.",
      ],
      [
        "L'accès est rétabli, merci.\n\nLe lun. 22 sept. 2026 à 10:14, Support Technique <\n" +
          "support@acme.com> a écrit :\n> ancien texte",
        "L'accès est rétabli, merci.",
      ],
      ["Bonjour,\nLa facture de mars n'est pas arrivée.\nMerci,\nJean", "Bonjour,\nLa facture de mars n'est pas arrivée."],
      [
        "Bonjour,\nLa facture de mars n'est pas arrivée.\nBien à vous,\nMarie",
        "Bonjour,\nLa facture de mars n'est pas arrivée.",
      ],
      [
        "J'ai besoin de la facture de mars.\n\nSi vous avez reçu ce message par erreur, supprimez-le.",
        "J'ai besoin de la facture de mars.",
      ],
      ["Voici le devis demandé.\n\nCe document est à l'usage exclusif du destinataire.", "Voici le devis demandé."],
    ];
    for (const [body, expected] of cases) expect(cleanEmailBody(body), body).toBe(expected);
  });
  it("keeps French requests that only look like markers (Python parity)", () => {
    for (const body of [
      "Le contrat confidentiel doit être signé avant vendredi.",
      "Bonjour,\nMerci pour votre aide.\nRappelez-moi.",
      "Bonjour,\nLe rapport que vous avez écrit :\nla commande 4411 n'est pas arrivée.",
      "Le montant est destiné exclusivement au paiement de la facture. Pouvez-vous confirmer ?",
      "J'ai besoin de congés.\nDe : 10/09 à 15/09\nC'est possible ?",
    ]) {
      expect(cleanEmailBody(body), body).toBe(body);
    }
  });
  it("guard preset has jailbreak and harm_severity", () => {
    const g = guardQuestions() as Record<string, any>;
    expect(g.jailbreak.type).toBe("noul");
    expect(g.harm_severity.type).toBe("score");
    expect(g.harm_severity.criteria.length).toBe(4);
  });
});
