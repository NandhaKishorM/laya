import { describe, expect, it } from "vitest";
import { Router, guessLatinLanguage } from "../src/index.js";

describe("Swedish support routing", () => {
  it.each([
    "Kan inte logga in",
    "Vi kan inte logga in",
    "Ingen åtkomst",
    "Ingen atkomst",
    "Glömt lösenord",
    "Glomt losenord",
    "Kan ni hjälpa mig?",
    "min faktura ar fel",
    "Var är mitt paket? Spårningen har inte uppdaterats",
    "Om ni inte kan fa tillbaka de raderade filerna i dag avslutar jag mitt abonnemang.",
  ])("names and routes the support request: %s", (text) => {
    const router = new Router();
    expect(guessLatinLanguage(text)).toBe("sv");
    expect(router.route(text, {}).model).toBe("multilingual");
    expect(router.loaded).toEqual([]);
  });

  it.each([
    "Ingen adgang", "Fakturaen feil", "Glemt passord", "Pakken forsinket",
    "Jeg kan ikke logge ind på min konto", "No account access", "Password forgotten",
  ])("does not name a neighbouring-language or English control Swedish: %s", (text) => {
    expect(guessLatinLanguage(text)).not.toBe("sv");
  });

  it("keeps an English login request on the English checkpoint", () => {
    expect(new Router().route("I cannot login to my account", {}).model).toBe("english");
  });
});
