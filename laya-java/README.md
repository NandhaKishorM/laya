# laya-java

JVM inference for Laya: `Agent.predict` / `Agent.predictBatch` over the BPE tokenizer, the sequence
builder, ONNX Runtime and typed answers. JDK 17+. One dependency — `com.microsoft.onnxruntime`.

Laya is a non-autoregressive "System 1" decision model: a bidirectional encoder plus a typed head
that answers `choice` / `score` / `noul` questions about a state in **one forward pass**, with
calibrated confidences. Every question about the same state becomes one row of a single batch, so
four questions about a document cost one batched encode rather than four round trips.

## Status

Implemented: tokenizer, sequence builder, config, ONNX inference (fused or split graph), answer
decoding, `predict`, `predictBatch`, usage and truncation reporting, script and language detection
(`lang.LanguageDetection`), the question presets (`Presets`), the checkpoint `Router` with its
load-and-evict lifecycle, the embedding `Shortlist` with its LRU cache, and the email cleaner
and state builder (`LayaEmail`).

Not implemented yet: hooks, `predictLong`, structured `decide`, the
`laya-java-client` HTTP module, Android. **Not published to Maven Central** — see
[Installing](#installing).

## Where to go next

| | |
|---|---|
| [MODELS.md](MODELS.md) | how to get the checkpoint and the graph `Agent.open` takes |
| [samples/](samples/) | three runnable programs: quickstart, routing and shortlisting, benchmark |
| [CHANGELOG.md](CHANGELOG.md) | what is in 0.1.0, and where Java is not Python |

## Export a graph (once per checkpoint)

```bash
python scripts/export_onnx.py --model <ckpt-dir-or-hub-id> --output ./model/laya.onnx
```

That writes the fused graph. The split form (`encoder.onnx` + `head.onnx`, from
`laya-ts/scripts/export_onnx.py`) is also supported; `LayaSession` opens whichever a directory
holds, preferring the fused one.

A checkpoint directory must contain `rl_agent_config.json` and `tokenizer/` — that is where the
budgets, the fitted temperatures and the special tokens live, and none of them may be defaulted.
The two shipped checkpoints disagree on every one: english is `max_len 512 / head_max_len 192` with
fitted per-bucket temperatures, multilingual is `1024 / 256` with none, and multilingual reuses
`<bos>` as its classification token and `<eos>` as its separator.

## Ask questions

```java
import com.convaiinnovations.laya.*;
import java.nio.file.Path;
import java.util.*;

// Options are POSITIONAL, so use a LinkedHashMap: two orders are two different questions.
Map<String, Object> criteria = new LinkedHashMap<>();
criteria.put("refund", "money back for a duplicate charge");
criteria.put("escalate", "pass it to a human");
criteria.put("ignore", "no action needed");

Map<String, Question> questions = new LinkedHashMap<>();
questions.put("intent", Question.choice("What does the customer want?", criteria));
questions.put("urgent", Question.noul("This needs a human today."));
questions.put("severity", Question.score("Rate the severity.",
        List.of("none", "minor", "major", "critical")));

try (Agent agent = Agent.open(Path.of("./checkpoint"), Path.of("./model"))) {
    Prediction p = agent.predict("We were billed twice for March and want a refund today.",
            questions, "en");   // the language tag may select a temperature override

    Answer.Choice intent = (Answer.Choice) p.answer("intent");
    System.out.println(intent.choice());                 // "refund"
    System.out.println(intent.probabilities());          // {refund=..., escalate=..., ignore=...}
    System.out.println(intent.answerConfidence());       // max(p), the calibrated one

    Answer.Noul urgent = (Answer.Noul) p.answer("urgent");
    System.out.println(urgent.noul());                   // P(the statement holds)

    Answer.Score severity = (Answer.Score) p.answer("severity");
    System.out.println(severity.score());                // expected value over level indices
    System.out.println(severity.legend());               // {0=none, 1=minor, 2=major, 3=critical}
}
```

`Answer` is sealed: the three shapes are a closed set, so a reader knows there is no fourth, and an
`instanceof` pattern needs no `else` that throws.

```java
Answer a = p.answer("intent");
if (a instanceof Answer.Choice c) {
    handle(c.choice(), c.probabilities());
} else if (a instanceof Answer.Score sc) {
    handle(sc.score(), sc.legend());
} else if (a instanceof Answer.Noul n) {
    handle(n.noul());
}
```

On **JDK 21 or newer** a pattern `switch` over it is exhaustive and a missing case is a compile
error. That is a language feature of 21, not of this library — on JDK 17, which this module
targets, pattern switches are a preview feature and the `instanceof` form above is the portable
one.

### Two confidences, deliberately

`answerConfidence()` is the probability mass on the reported answer — `max(p)` — and it is the
quantity temperature scaling fits and every calibration figure in laya is computed on.
`confidence()` means something different per type: normalised entropy for a choice or a score, and
`max(p, 1-p)` for a noul. **They are on different scales and must not be compared against the same
threshold.**

The calibration guarantee on `answerConfidence` is conditional, and the condition is not met by
default: it holds only after the temperatures have been fitted and validated on held-out data for
that checkpoint and option count. The shipped checkpoints are over-confident.

## Usage, and what was silently dropped

A state longer than the room left after the question's head is truncated **without an error**, and a
caller cannot reconstruct that from outside: the budget is in tokens, not characters, and the room
left moves per checkpoint and per question.

```java
Usage u = p.usage();
u.inputTokens();            // non-padding tokens fed to the encoder, across every row
u.stateTokens();            // what the state serialised to, before any budget was applied
u.stateTokensDropped();     // the worst case across this state's questions
u.truncated();              // whether any question dropped state tokens
u.truncatedQuestions();     // which ones
u.collapsedOptions();       // questions whose options no longer have a token span each
```

`collapsedOptions()` is the one a reviewer of your prompts will care about: two options whose
rendered text is identical for the first 48 tokens collapse to the same span, the model cannot tell
them apart, and **nothing else downstream can see it** because the marker count still matches. It
reports the count the question *defines*, not the number of markers that reached the sequence.

A question whose option markers do not all fit `max_len` is **refused**, not answered partially —
answering would return a distribution over whichever options survived, with the rest absent from the
answer and unchoosable.

## Many states, one call

```java
List<Object> states = List.of(emailA, emailB, emailC);
List<Prediction> out = agent.predictBatch(states, questions);        // all in one graph call
List<Prediction> out2 = agent.predictBatch(states, questions, "en",
        /* batchSize */ 16, /* sortByLength */ true);                // bounded memory, less padding
```

Results come back in the caller's order whatever the grouping was. `batchSize` bounds peak memory
and `sortByLength` cuts padding; neither changes an answer.

## Routing between checkpoints

The English checkpoint does not degrade gently off English, it collapses — 0.100 on 20-option
MASSIVE Hindi intent against 0.050 for random guessing, and it reports high confidence while doing
so. So a `Router` picks the checkpoint before the question is asked.

```java
import com.convaiinnovations.laya.Router;
import java.nio.file.Path;

try (Router router = Router.builder()
        .checkpointsRoot(Path.of("./checkpoints"))   // ./checkpoints/english, /multilingual, ...
        .maxLoaded(2)                                // a memory ceiling, not a cache hint
        .build()) {

    Router.RouteDecision decided = router.route(state);
    System.out.println(decided.model() + ": " + decided.reason());
    // multilingual: non-Latin script (devanagari, 100% of letters); the English checkpoint
    // cannot read it

    Prediction answer = router.predict(state, questions);   // routes, loads, answers
}
```

`route` runs no model and reads no disk, so it is safe on every request. Precedence, highest
first: an explicit model, an explicit task, a detected typed-decisions workflow (opt-in only), an
explicit language, a caller's hint, the built-in detection, then the configured default.

`maxLoaded` evicts the least recently used checkpoint, and `Router.Lease` keeps one open for as
long as you hold it — `predict` leases internally, so the ordinary path needs no thought. An agent
handed in with `attach` is never closed: the caller keeps ownership.

A deployment whose traffic is mostly not English should set
`defaultCheckpoint(Checkpoint.MULTILINGUAL)`: an unidentified Latin-script state is no evidence of
English, and that is the only knob which says so.

## Detection on its own

`lang.LanguageDetection` is the router's evidence, usable without one:

```java
import com.convaiinnovations.laya.lang.LanguageDetection;

LanguageDetection.Analysis seen = LanguageDetection.analyse(state);
seen.script();              // "latin", "han", "devanagari", ... or "unknown"
seen.language();            // best effort, null when undecided
seen.english();             // whether the English checkpoint can read it
seen.mixedSegment();        // the line or field that made a mostly-English state non-English
```

Undecided is not English: when nothing identifies the language, non-English letters or a shared
Swedish–Danish marker still prefer the multilingual checkpoint.

## Presets

Five ready-made question sets, word for word the reference's:

```java
import com.convaiinnovations.laya.Presets;

Map<String, Question> questions = Presets.triage();        // or email(), guard(),
                                                           // moderation(), router()
Presets.stateField(questions);                             // "message" — the key it reads
```

Each call returns a fresh mutable map, so dropping a question you do not want is safe. A preset
names the state key it reads in backticks, and `stateField` reads it back out — so a caller can
place its text under the right key instead of guessing.

## Shortlisting a large label set

Choice options share one `head_max_len`, so a seventeen-label set leaves few tokens per label.
`Shortlist` embeds the state and each option, keeps the top `k`, and runs one prediction over the
reduced set — no second decision pass.

```java
import com.convaiinnovations.laya.Shortlist;

Shortlist.Embedder embedder = texts -> myBiEncoder.embed(texts);   // your vectors
Shortlist.Shortlisted out = Shortlist.predict(agent, state, questions,
        Shortlist.cached(embedder), 20);

out.prediction();                    // the answer, over the kept labels only
out.shortlist().get("intent");       // which labels survived, and their scores
```

`Shortlist.cached(...)` embeds each text once under an LRU bound, so a fixed label list is
embedded on the first call and only the new query thereafter. `k` at or above the label count is a
passthrough: the labels come back in order and **the embedder is never called**.

Both `Agent` and `Router` implement `Predictor`, so shortlisting works identically against a fixed
checkpoint or a routed one.

## Cleaning an email

`laya.email`'s cleaner and state builder. The markers cover English, Portuguese, Spanish and
French mail clients, because the router already sends the last three to the multilingual
checkpoint and an English-only cleaner left their quoted history — often a *different* request —
weighing on the answer as much as the new message.

```java
import com.convaiinnovations.laya.LayaEmail;
import java.util.Map;

String body = """
        I was charged twice for order 8812. Please refund one of them.

        Atenciosamente,
        Ana Souza
        Enviado do meu iPhone

        Esta mensagem e confidencial e de uso exclusivo do destinatario.

        Em ter., 3 de set. de 2025, Suporte <suporte@x.com> escreveu:
        > Podemos ajudar?
        """;

// quoted history, sign-off, device footer and disclaimer all go
String clean = LayaEmail.cleanEmailBody(body);

Map<String, Object> state = LayaEmail.emailState("Cobranca duplicada", body, "ana@x.com");
// {subject=Cobranca duplicada, body=I was charged twice..., from=ana@x.com}

Prediction p = agent.predict(state, LayaEmail.emailQuestions());
```

`emailState` takes the same `maxChars` budget and passes it through, and it is worth raising for
a long message: at the default the body stops after 3,000 characters, so a request arriving in
the last paragraphs never reaches the model.

What it deliberately does **not** cut is the interesting half. `From: my side the integration
works, but please refund...` is prose, not a header, so a reply header is recognised only when an
address follows it or its own `Sent:`/`Enviado:` line does. `Thanks for the quick reply.` is not
a sign-off, `Obrigado pelo retorno, mas ...` is a request, and `Is this confidential?` is a
question — a cleaner that is too eager deletes what the sender actually wrote, which is worse
than leaving one boilerplate line behind.

## Lower-level pieces

```java
import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.nio.file.Path;

Tokenizer tok = Tokenizer.fromModelDirectory(Path.of("./checkpoint"));
int[] ids = tok.encode("charged twice");
int[] capped = tok.encode(longText, 48);          // stops early; same prefix as the full encoding
String text = tok.decode(ids);                    // ids back to text; special tokens dropped
String withSpecials = tok.decode(ids, false);     // ...kept
AgentConfig cfg = AgentConfig.fromModelDirectory(Path.of("./checkpoint"));
```

`decode` follows the reference rather than tidying after it, and on the multilingual checkpoint
that means it is **lossy**: `Metaspace` prepends its marker, so `decode(encode("Hello world"))` is
`" Hello world"`. The English checkpoint happens to round-trip. Correcting the space would make
every window of a long document tokenize differently from the reference, so the loss is kept.

`Agent.using(tokenizer, config, session)` assembles an agent from parts — for a caller that already
holds them, or to drive the batching and usage accounting through a stub
`infer.InferenceSession` instead of a 1.2 GB graph.

## Threads

`Agent.open(model, graph)` leaves the thread count to ONNX Runtime, which is what the Python
runtime does. Pass `Agent.open(model, graph, 1)` for a request-per-thread server, where the
parallelism is already in the requests. One `Agent` is **not** safe for concurrent `predict` calls
unless ONNX Runtime is configured for it; hold one per worker, or serialise access.

## Build and test

```bash
cd laya-java
./gradlew build                 # -Xlint:all -Werror
./gradlew test                  # the suite that needs no checkpoint
```

The parity tests need a checkpoint and a graph, and **abort with an actionable message** without
them rather than passing vacuously:

```bash
LAYA_CHECKPOINTS=/path/to/checkpoints \
LAYA_ONNX_GRAPH=/path/to/model/laya.onnx \
  ./gradlew test
```

A blank value counts as absent, so a CI cell that owns no graph can set `LAYA_ONNX_GRAPH=''`
without turning an abort into a failure.

### Testing on another JDK

The artifact is compiled for 17 and the toolchain pins the **compiler** to 17, so installing a
different JDK does not change what the tests run on. `-PtestJavaVersion` moves the test JVM only —
the bytecode stays at release 17:

```bash
./gradlew test -PtestJavaVersion=24    # compiled for 17, executed on 24
```

CI runs the model-free suite on **17, 21 and 24** — three different Unicode versions (13.0, 15.0
and 16.0). The artifact is compiled for 17, so it runs on 17 and anything newer; those three are
the versions the suite is actually asserted against.

This matters more here than in most ports. `\p{L}` and `\p{N}` in `java.util.regex` follow the
JDK's own Unicode version, and `Character.isLetter` disagrees with itself across JDK 17 (Unicode
13.0) and JDK 24 (Unicode 16.0) on 751 of the code points this port has to classify — 0 of 751 on
one, 751 of 751 on the other. Everything Unicode-shaped is therefore compiled in from the
reference, and this flag is how that is checked. `TestJvmVersionTest` asserts the tests really are
running on the JDK that was asked for, because Gradle writes `<properties/>` empty into the JUnit
XML and nothing downstream can tell 17 from 24.

## Parity: generated, not asserted

`laya-java/fixtures/*.json` are generated **from** the Python package and committed. The drift check
is what keeps them honest:

```bash
python laya-java/scripts/gen_fixtures.py            # regenerate
python laya-java/scripts/gen_fixtures.py --check     # fail if a committed fixture would change
```

A family that needs a checkpoint or a graph this run cannot reach is **left alone** and reported as
unverified, so running the generator without the checkpoints cannot replace good expectations with
skip markers. `--strict` turns "could not verify" into an error, for a lane that is supposed to have
them.

Where it does *not* match Python bit for bit, and why: the ONNX logits are float32 and Python
computes the softmax in float32, while this port — like `laya-ts` and `laya-dotnet` — computes it in
`double`. Over 200,000 random logit rows the four-decimal probabilities differ on 88 of them.
Matching float32 exactly would mean reproducing NumPy's own float32 `exp` (which differs from
rounding a double `exp` on 0.65% of values) and its pairwise summation order; neither is a contract
NumPy publishes. Rounding, by contrast, *is* exact: `BigDecimal` at scale 4 with `HALF_EVEN` is
Python's `round(v, 4)` on Python's operand.

## Installing

**Not published yet.** Build from source into your local repository:

```bash
cd laya-java && ./gradlew publishToMavenLocal
```

```kotlin
dependencies { implementation("com.convaiinnovations:laya-java:0.1.0-SNAPSHOT") }
```

Publishing to Maven Central needs the `com.convaiinnovations` namespace verified by the project
owner, a Central portal token and a PGP signing key. `.github/workflows/release-java.yml` is wired
for it and has never been run; signing activates only when a key is present, so `build` and
`publishToMavenLocal` work without any of it.
