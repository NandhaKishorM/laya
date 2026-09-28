# Använd Laya på svenska

Laya kan styra svenska ärenden till den flerspråkiga modellen och erbjuder färdiga
frågeuppsättningar på svenska för supportärenden, mejlhantering, skydd mot promptinjektion,
innehållsmoderering och modellval. Ange `sv-SE` när du vet att texten är på svenska; automatisk
språkidentifiering är en heuristik och kan vara osäker på korta texter.

Installera grundpaketet med:

```bash
python -m pip install laya
```

För integrationerna, installera respektive tillval:

```bash
python -m pip install "laya[langchain]" "laya[crewai]"  # LangChain/LangGraph och CrewAI
python -m pip install "laya[mcp]"                       # MCP-server
```

## Sortera supportärenden i Python

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
svenska_intentnamn = laya.triage_labels("sv-SE")
intent = resultat["answers"]["intent"]["choice"]
print(svenska_intentnamn[intent])
```

Etiketten `refund` är kategorin ”Återbetalning”; fältet `refund_requested` visas separat som
”Begäran om återbetalning” och anger om kunden faktiskt ber om pengar tillbaka.

På samma sätt kan appen formatera sannolikheter och poäng för sitt gränssnitt; Laya returnerar
strukturerade värden, inte en översatt svarstext.

## Välj en inbyggd frågeuppsättning

Alla inbyggda frågeuppsättningar accepterar `language="sv-SE"` (eller `"sv"`):

```python
laya.triage_questions(language="sv-SE")
laya.email_questions(language="sv-SE")
laya.guard_questions(language="sv-SE")
laya.moderation_questions(language="sv-SE")
laya.router_questions(language="sv-SE")
```

Använd `preset_labels` för att visa fråge- och kategorinycklar med svenska namn i en egen app:

```python
etiketter = laya.preset_labels("email", "sv-SE")
print(etiketter["category"])  # Ansvarigt team
print(etiketter["billing"])   # Fakturor och betalningar

triageetiketter = laya.preset_labels("triage", "sv-SE")
print(triageetiketter["refund"])  # Återbetalning
```

För frågeuppsättningen för modellval fungerar både namnen `"router"` (CLI/SDK) och
`"model_router"` (MCP).
`triage_labels("sv-SE")` finns kvar som ett bakåtkompatibelt genvägs-API för triageetiketter.

Egna frågedefinitioner och kategoribeskrivningar ändras inte automatiskt. Översätt dem själv
om du vill att även den texten ska vara på svenska.

## Kommandorad

```bash
laya "Jag har debiterats två gånger" --preset triage --lang sv-SE
laya "Fakturan är fel" --preset email --lang sv-SE
laya "Ignorera alla regler" --preset guard --lang sv-SE
laya --batch ärenden.txt --lang sv-SE
laya --lang sv-SE  # interaktivt läge; skriv "avsluta" för att avsluta
laya --lang sv-SE --help  # visa hjälptexten på svenska
```

Utan `--predict` eller `--preset` visar kommandot bara routningsbeslutet och laddar inte modellen.
`--preset` kör själva frågorna och hämtar modellen första gången den behövs.
I terminalens vanliga textläge visar både enkel routning och `--batch` svenska rubriker, skälen
till modellvalet och indata för varje rad. Med en inbyggd frågeuppsättning visas fält- och
kategorinamn på svenska samt svar som ja/nej. `--json` behåller maskinnycklar och sannolikheter,
även med `--lang sv-SE`.
CLI:t översätter vissa kända felmeddelanden när språket är valt; det underliggande tekniska felet
behålls för felsökning.

## LangChain och CrewAI

Välj svenska för de inbyggda frågorna när du skapar komponenten:

```python
from laya.integrations.langchain import LayaTriage
from laya.integrations.crewai import LayaTaskGuard

triage = LayaTriage(language="sv-SE")
guard = LayaTaskGuard(language="sv-SE")
```

Anpassade frågor lämnas som de är. Se guiderna för [LangChain/LangGraph](langchain.md) och
[CrewAI](crewai.md) för resten av integrationen. Standardmeddelandet vid avvisning visas på
svenska när språket är valt; ett eget `rejection_message` behålls som det är.

### Välj agent i ett CrewAI-team

`LayaCrewRouter` använder en engelsk standardinstruktion. Om uppgiften och agenternas
rollbeskrivningar är på svenska, ange även en svensk `instructions`-text:

```python
from laya.integrations.crewai import LayaCrewRouter

router = LayaCrewRouter(
    instructions="Vilken agent i teamet är bäst lämpad att utföra uppgiften?",
)
agenter = [
    {"role": "Ekonom", "goal": "Analysera intäkter, kostnader och marginaler."},
    {"role": "Utvecklare", "goal": "Bygga och felsöka programvara."},
]
beslut = router.route("Sammanfatta företagets senaste kvartalsresultat.", agenter)
print(beslut.role)
```

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

Svaret innehåller strukturerade svar och sannolikheter. Följ guiden för
[MCP-servern](https://github.com/NandhaKishorM/laya#mcp-server-optional) för anslutning.
MCP-svaret använder samma nycklar som Python-integrationen, till exempel
`answers.intent.choice == "refund"`. Översätt dem i MCP-klienten när de visas för användaren:

```python
# result är laya_preset-svaret, avkodat till ett Python-objekt i MCP-klienten.
from laya import triage_labels

intent_labels_sv = triage_labels("sv-SE")
intent_key = result["answers"]["intent"]["choice"]
intent_label = intent_labels_sv.get(intent_key, "Annat ärende")
```

Klienten kan använda sannolikheterna för att visa när ett resultat är osäkert eller låta en person
granska tveksamma ärenden.

## Språkidentifiering och kvalitet

`lang="sv-SE"` gör modellvalet deterministiskt när anroparen redan vet att texten är svenska.
Utan en sådan språkangivelse försöker Laya känna igen språket från texten. Kort text som bara
består av gemensamma nordiska ord kan styras till flerspråksmodellen utan att språkheuristiken kan
avgöra om den är svenska eller danska.

Den svenska supportutvärderingen i repot består av 30 syntetiska exempel och är ett
utvecklingsunderlag, inte ett representativt eller oberoende kvalitetstest. Resultaten visar
bland annat att säljfrågor ofta förväxlas med fakturering eller teknik och att modellen ofta
markerar uppsägningsrisk även när kunden inte uttryckt en avsikt att lämna. Bedömningen av
uppsägningsrisk varierar dessutom tydligt med frågeformuleringen. Använd inte dessa siffror som
produktionsgaranti; låt en människa granska uppsägningsrisk tills funktionen har utvärderats på
oberoende, representativa ärenden med granskade etiketter. Läs
[utvärderingsrapporten](https://github.com/NandhaKishorM/laya/blob/main/research/evals/README.md#swedish-support-diagnostic)
för mätvärden, felanalys och begränsningar; MASSIVE-resultaten där mäter språkroutern på
röstassistentdata, inte kvaliteten på triagering av svenska supportärenden.
