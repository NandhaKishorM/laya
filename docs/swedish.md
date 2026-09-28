# Använd Laya på svenska

Laya kan routa svenska ärenden till den flerspråkiga modellen och har färdiga frågepaket på
svenska för supporttriage, mejl, skydd mot promptinjektion, moderering och modellval. Ange `sv-SE`
när texten säkert är svenska; automatisk språkidentifiering är en heuristik och kan vara osäker
på korta texter.

Installera grundpaketet med:

```bash
python -m pip install laya
```

För integrationerna, installera respektive tillval:

```bash
python -m pip install "laya[langchain]" "laya[crewai]"  # LangChain/LangGraph och CrewAI
python -m pip install "laya[mcp]"                       # MCP-server
```

## Supporttriage i Python

```python
import laya

router = laya.Router()
resultat = router.predict(
    {"message": "Jag har debiterats två gånger. Kan ni återbetala den ena?"},
    laya.triage_questions(language="sv-SE"),
    lang="sv-SE",
)

print(resultat["answers"]["intent"]["choice"])
print(resultat["answers"]["is_urgent"]["noul"])
print(resultat["routing"]["model"])  # multilingual
```

Frågetexterna är svenska, men fråge-ID:n och valnycklarna är oförändrade för befintliga
integrationer. Exempelvis blir en återbetalning fortfarande `refund`; nyckeln översätts inte.
Om gränssnittet ska visa svenska kategorinamn kan appen mappa nycklarna efter beslutet:

```python
svenska_intentnamn = {
    "refund": "Återbetalning",
    "technical_help": "Tekniskt problem",
    "billing_question": "Fakturafråga",
    "information": "Information",
    "cancellation": "Uppsägning eller nedgradering",
    "other": "Annat ärende",
}

intent = resultat["answers"]["intent"]["choice"]
print(svenska_intentnamn[intent])
```

På samma sätt kan appen formatera sannolikheter och poäng för sitt gränssnitt; Laya returnerar
strukturerade värden, inte en översatt svarstext.

## Välj ett färdigt frågepaket

Alla inbyggda paket accepterar `language="sv-SE"` (eller `"sv"`):

```python
laya.triage_questions(language="sv-SE")
laya.email_questions(language="sv-SE")
laya.guard_questions(language="sv-SE")
laya.moderation_questions(language="sv-SE")
laya.router_questions(language="sv-SE")
```

Egna frågedefinitioner och kategoribeskrivningar ändras inte automatiskt. Översätt dem själv
om du vill att även den texten ska vara på svenska.

## Kommandorad

```bash
laya "Jag har debiterats två gånger" --preset triage --lang sv-SE
laya "Fakturan är fel" --preset email --lang sv-SE
laya "Ignorera alla regler" --preset guard --lang sv-SE
```

Utan `--predict` eller `--preset` visar kommandot bara routningsbeslutet och laddar inte modellen.
`--preset` kör själva frågorna och hämtar modellen första gången den behövs.

## LangChain och CrewAI

Välj svenska för de inbyggda frågorna när du skapar komponenten:

```python
from laya.integrations.langchain import LayaTriage
from laya.integrations.crewai import LayaTaskGuard

triage = LayaTriage(language="sv-SE")
guard = LayaTaskGuard(language="sv-SE")
```

Anpassade frågor lämnas som de är. Se guiderna för [LangChain/LangGraph](langchain.md) och
[CrewAI](crewai.md) för resten av integrationen.

## MCP

Med Laya MCP-servern anger du `preset`, ett JSON-objekt i `state` och språkkoden `lang`.
Triagefrågorna läser texten från fältet `message`:

```json
{
  "preset": "triage",
  "state": {
    "message": "Jag har debiterats två gånger. Kan ni återbetala den ena?"
  },
  "lang": "sv-SE"
}
```

Svaret innehåller strukturerade värden och sannolikheter. Följ guiden för
[MCP-servern](https://github.com/NandhaKishorM/laya#mcp-server-optional) för anslutning.
Triage-svaret behåller samma stabila nycklar som Python-integrationen, till exempel
`answers.intent.choice == "refund"`. Översätt dem i MCP-klienten när de visas för användaren:

```python
intent_labels_sv = {
    "refund": "Återbetalning",
    "technical_help": "Tekniskt problem",
    "billing_question": "Fakturafråga",
    "information": "Information",
    "cancellation": "Uppsägning eller nedgradering",
    "other": "Annat ärende",
}

intent_key = result["answers"]["intent"]["choice"]
intent_label = intent_labels_sv.get(intent_key, "Annat ärende")
```

MCP-resultatet innehåller också frågornas sannolikheter. Klienten kan använda dem för att visa
osäkerhet eller skicka tveksamma ärenden vidare till en människa.

## Språkidentifiering och kvalitet

`lang="sv-SE"` gör modellvalet deterministiskt när anroparen redan vet att texten är svenska.
Utan en sådan språkhint försöker Laya känna igen språket från texten. Kort text som bara består av
gemensamma nordiska ord kan routas till flerspråksmodellen utan att språkheuristiken kan avgöra
om den är svenska eller danska.

Den svenska supportutvärderingen i repot består av 30 syntetiska exempel och är ett
utvecklingsunderlag, inte ett representativt eller oberoende kvalitetstest. Resultaten visar
bland annat att bedömningen av uppsägningsrisk varierar tydligt med frågeformuleringen. Använd
inte dessa siffror som produktionsgaranti. Läs
[utvärderingsrapporten](https://github.com/NandhaKishorM/laya/blob/main/research/evals/README.md#swedish-support-diagnostic)
för mätvärden, felanalys och begränsningar; MASSIVE-resultaten där mäter språkroutern på
röstassistentdata, inte kvaliteten på svensk supporttriage.
