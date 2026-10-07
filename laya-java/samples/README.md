# laya-java samples

Three runnable programs. They are **single-file source programs**, not a Gradle module — run them
with `java Foo.java` and no build step of their own.

| | needs | shows |
|---|---|---|
| [`Quickstart.java`](Quickstart.java) | a checkpoint **and** a graph | four typed questions in one forward pass, in English and Hindi |
| [`Routing.java`](Routing.java) | both checkpoints; a graph for the one it routes to | the routing decision and its reason, the load-and-evict lifecycle, and shortlisting 400 labels |
| [`Benchmark.java`](Benchmark.java) | nothing, or a checkpoint and a graph | what the SDK costs, warm and best-of-N |

`Benchmark.java` runs with no arguments and still measures something: detection and the email
cleaner need no model at all.

## Why single files and not a module

A `samples` subproject would inherit the root build's `maven-publish` and `signing`
configuration, which creates a `MavenPublication` for every subproject — and
`publishAllPublicationsToCentralRepository` would push it. That is not hypothetical: the empty
`laya-java-client` module was removed from `settings.gradle.kts` for exactly this reason, because
a version burned on Maven Central cannot be replaced. Samples are demonstration code and have no
business being publishable, so they stay out of the build graph.

They are still compiled. The `jvm build + model-free tests` CI lane compiles all three against the
built classes, so a sample that stops matching the API fails the build rather than rotting.

## Running them

You need two things on the classpath: the SDK and ONNX Runtime.

```bash
# 1. the artifacts the SDK reads -- see ../MODELS.md
python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual

# 2. the SDK jar, and ONNX Runtime from the Gradle cache
cd laya-java
./gradlew --no-daemon :laya-java:jar

JAR=$(ls laya-java/build/libs/laya-java-*.jar | head -1)
ORT=$(find ~/.gradle/caches/modules-2 -name 'onnxruntime-1.20.0.jar' | head -1)

# 3. run
java -cp "$JAR:$ORT" samples/Quickstart.java \
     .work/checkpoints/multilingual .work/onnx/multilingual
```

`java Foo.java` compiles in memory and runs `main`; nothing is written to disk. It needs JDK 11 or
newer, and the SDK targets 17.

If `find` turns up nothing for ONNX Runtime, resolve it once:

```bash
./gradlew --no-daemon :laya-java:dependencies --configuration runtimeClasspath | grep onnxruntime
```

## What each one is for

### `Quickstart.java`

The smallest program that covers the whole path: open a checkpoint, build a state, ask the preset
email questions, read typed answers back. It asks the **same question objects** about an English
state and a Hindi one, which is the point of a question set being plain data.

It reads answers by pattern-matching the sealed `Answer` interface rather than by looking up a
map, because a choice is not a score and the compiler can check that every shape is handled.

It also prints `usage`. That is not decoration: it says whether the model actually saw the state,
and a truncated request names the question that was cut.

### `Routing.java`

Two halves that fail in opposite directions.

`route` needs no model and loads nothing, so the first half just prints the decision and its
reason for four states in four scripts. The reason string is what you will read in a log when a
request went somewhere you did not expect.

The second half gives the router an `AgentFactory` and `maxLoaded(1)`, so the load-and-evict
lifecycle is visible: it loads on demand, keeps one checkpoint resident, and closes the least
recently used — but never one a caller still holds a lease on.

The third part shortlists 400 labels down to 5 with a deliberately terrible embedder: hashed
character trigrams, which have no notion of meaning and match on shared spelling. It is there to
make the shape clear without pulling in a model. **In a real program that is your bi-encoder, and
the quality of the shortlist is the quality of that encoder** — a shortlist is a filter in front
of the model, so a bad one removes the right answer before the model can be asked about it.

### `Benchmark.java`

Warm, best-of-N, with the cold first call reported separately instead of averaged in.

A first measurement on a cold JVM is dominated by class loading, JIT, and — for this SDK — the
one-time walk of Unicode that builds the email cleaner's character classes. Quoting that as
throughput is how a port comes to look slower than it is. Quoting a single warm run is how noise
comes to look like a regression. So the samples' own numbers are the best of 5 after 20 warmup
calls, and the two cold figures are printed as cold.

It also reports the batch cost **per state**, which is the number to use when sizing a queue.
