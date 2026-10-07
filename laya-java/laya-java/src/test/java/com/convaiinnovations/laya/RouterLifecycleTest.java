package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Router.Checkpoint;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The {@link Router}'s load-and-evict lifecycle, driven through a stub agent factory.
 *
 * <p>A seam rather than a real checkpoint, for the reason {@code InferenceSession} is one: the
 * parts most likely to be wrong here -- which agent the eviction picks, whether an attached one is
 * ever closed, whether two threads asking for the same cold checkpoint pay for two copies -- have
 * nothing to do with ONNX, and a real checkpoint is hundreds of megabytes. These run in
 * milliseconds and need no download, so they run in every CI job rather than only the parity lane.
 *
 * <p>Closing is what these tests watch most closely, because it is the one place this port cannot
 * copy the reference. There, eviction drops a reference and CPython's refcounting keeps an agent
 * alive for whoever still holds it. A JVM will not free a native ONNX session that way, so the
 * router has to close it -- and must not close one that a caller is still using, or that the
 * caller owns.
 */
final class RouterLifecycleTest {

    private static Path checkpointRoot;

    @BeforeAll
    static void writeCheckpoint(@TempDir Path root) throws IOException {
        // One synthetic checkpoint, reused for every Checkpoint value: the lifecycle does not care
        // what is inside, only which instance it hands out and when it closes one.
        checkpointRoot = root;
        TinyCheckpoint.write(root, 64, 32);
    }

    /** A factory that counts builds and keeps every session it handed out, so closes are visible. */
    private static final class StubAgents implements Router.AgentFactory {

        private final Map<Checkpoint, Integer> builds = new EnumMap<>(Checkpoint.class);
        private final Map<Checkpoint, List<TinyCheckpoint.RecordingSession>> sessions =
                new EnumMap<>(Checkpoint.class);
        private final AtomicInteger total = new AtomicInteger();
        /** Every entry into the factory, successful or not: a failed load still costs a download. */
        private final Map<Checkpoint, Integer> attempts = new EnumMap<>(Checkpoint.class);
        private volatile Checkpoint failOn;
        private volatile CountDownLatch blockOn;

        @Override
        public Agent create(Checkpoint checkpoint) throws IOException {
            synchronized (this) {
                attempts.merge(checkpoint, 1, Integer::sum);
            }
            // The gate is waited on BEFORE the failure check on purpose. A build that fails
            // instantly is finished before a second caller even arrives, so nothing is ever
            // shared and a test counting attempts measures the harness rather than the router.
            // A real load that fails -- a download that times out -- takes just as long as one
            // that succeeds.
            CountDownLatch gate = blockOn;
            if (gate != null) {
                try {
                    assertTrue(gate.await(10, TimeUnit.SECONDS), "the build gate never opened");
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IOException(interrupted);
                }
            }
            if (checkpoint == failOn) {
                throw new IOException("no checkpoint for " + checkpoint.wireName());
            }
            TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
            synchronized (this) {
                builds.merge(checkpoint, 1, Integer::sum);
                sessions.computeIfAbsent(checkpoint, key -> new ArrayList<>()).add(session);
            }
            total.incrementAndGet();
            return TinyCheckpoint.agent(checkpointRoot, session);
        }

        synchronized int buildsOf(Checkpoint checkpoint) {
            return builds.getOrDefault(checkpoint, 0);
        }

        synchronized int attemptsOf(Checkpoint checkpoint) {
            return attempts.getOrDefault(checkpoint, 0);
        }

        synchronized List<TinyCheckpoint.RecordingSession> sessionsOf(Checkpoint checkpoint) {
            return List.copyOf(sessions.getOrDefault(checkpoint, List.of()));
        }

        synchronized TinyCheckpoint.RecordingSession lastSession(Checkpoint checkpoint) {
            List<TinyCheckpoint.RecordingSession> all = sessionsOf(checkpoint);
            assertFalse(all.isEmpty(), "nothing was ever built for " + checkpoint.wireName());
            return all.get(all.size() - 1);
        }
    }

    private static Router.Builder routerWith(StubAgents agents) {
        return Router.builder().agents(agents);
    }

    private static Map<String, Question> oneQuestion() {
        Map<String, Question> questions = new LinkedHashMap<>();
        questions.put("urgent", Question.noul("Is this urgent?"));
        return questions;
    }

    @Test
    @DisplayName("a second call reuses the loaded agent rather than building another")
    void loadIsCached() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).build()) {
            Agent first = router.load("english");
            Agent second = router.load("en");
            assertSame(first, second, "the alias names the same checkpoint, so the same agent");
            assertEquals(1, agents.buildsOf(Checkpoint.ENGLISH));
            assertEquals(List.of(Checkpoint.ENGLISH), router.loaded());
        }
    }

    @Test
    @DisplayName("loaded() is least-recently-used first, and a use moves a checkpoint to the end")
    void loadedOrderIsLru() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            router.load("english");
            router.load("multilingual");
            router.load("typed-decisions");
            assertEquals(List.of(Checkpoint.ENGLISH, Checkpoint.MULTILINGUAL,
                    Checkpoint.TYPED_DECISIONS), router.loaded());
            router.load("english");
            assertEquals(List.of(Checkpoint.MULTILINGUAL, Checkpoint.TYPED_DECISIONS,
                    Checkpoint.ENGLISH), router.loaded(),
                    "using english must move it to the most-recently-used end");
        }
    }

    @Test
    @DisplayName("past maxLoaded the least recently used is evicted, and closed")
    void evictionClosesTheVictim() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(2).build()) {
            router.load("english");
            router.load("multilingual");
            TinyCheckpoint.RecordingSession englishSession = agents.lastSession(Checkpoint.ENGLISH);
            assertFalse(englishSession.closed, "nothing has been evicted yet");

            router.load("typed-decisions");
            assertEquals(List.of(Checkpoint.MULTILINGUAL, Checkpoint.TYPED_DECISIONS),
                    router.loaded(), "english was least recently used");
            // The part the reference gets for free and this does not: a native session that is
            // merely dropped is leaked, so the router has to close it.
            assertTrue(englishSession.closed, "the evicted agent must be closed, not just dropped");
            assertFalse(agents.lastSession(Checkpoint.MULTILINGUAL).closed);
        }
    }

    @Test
    @DisplayName("an attached agent is never closed by the router, even when evicted")
    void attachedAgentsAreNotClosed() throws IOException {
        StubAgents agents = new StubAgents();
        TinyCheckpoint.RecordingSession mine = new TinyCheckpoint.RecordingSession();
        Agent owned = TinyCheckpoint.agent(checkpointRoot, mine);
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            assertSame(owned, router.attach("english", owned));
            assertSame(owned, router.load("english"), "an attached agent is already built");
            assertEquals(0, agents.buildsOf(Checkpoint.ENGLISH), "nothing should have been built");

            // Force it out, with maxLoaded back at a level that evicts.
            router.load("multilingual");
            router.load("typed-decisions");
            assertFalse(router.loaded().contains(Checkpoint.ENGLISH), "it should have been evicted");
            assertFalse(mine.closed, "the caller owns an attached agent; the router must not close it");
        }
        assertFalse(mine.closed, "closing the router must not close an attached agent either");
        owned.close();
        assertTrue(mine.closed, "and the owner can still close it afterwards");
    }

    @Test
    @DisplayName("attach raises maxLoaded, so it never immediately evicts what it attached")
    void attachRaisesMaxLoaded() throws IOException {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            assertEquals(1, router.maxLoaded());
            router.load("english");
            router.attach("multilingual", TinyCheckpoint.agent(checkpointRoot,
                    new TinyCheckpoint.RecordingSession()));
            assertEquals(2, router.maxLoaded(), "it had to grow to fit both");
            assertTrue(router.loaded().contains(Checkpoint.MULTILINGUAL),
                    "the agent just attached must still be resident");
        }
    }

    @Test
    @DisplayName("preload builds everything and raises maxLoaded to fit it")
    void preloadFitsEverything() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            router.preload();
            assertEquals(3, router.maxLoaded());
            assertEquals(3, router.loaded().size(), "all three must be resident, none evicted");
            for (Checkpoint checkpoint : Checkpoint.values()) {
                assertEquals(1, agents.buildsOf(checkpoint), checkpoint + " built exactly once");
            }
            // A cold load costs seconds and routing costs microseconds, so a preloaded router
            // answers without ever paying a load. A second preload must not rebuild anything.
            router.preload();
            assertEquals(3, agents.total.get(), "preloading twice must not rebuild");
        }
    }

    @Test
    @DisplayName("preloading in stages does not evict what an earlier stage loaded")
    void preloadInStages() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            router.preload(List.of("english"));
            router.preload(List.of("multilingual"));
            assertEquals(List.of(Checkpoint.ENGLISH, Checkpoint.MULTILINGUAL), router.loaded(),
                    "the second stage must not have evicted the first");
            assertFalse(agents.lastSession(Checkpoint.ENGLISH).closed);
        }
    }

    @Test
    @DisplayName("unload frees one checkpoint and closes it; unloadAll frees the rest")
    void unloadFrees() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            router.preload();
            assertEquals(List.of(Checkpoint.ENGLISH),
                    router.unload("en"), "it reports what it freed");
            assertTrue(agents.lastSession(Checkpoint.ENGLISH).closed);
            assertEquals(2, router.loaded().size());
            assertEquals(List.of(), router.unload("en"), "unloading it twice frees nothing");

            List<Checkpoint> rest = router.unloadAll();
            assertEquals(2, rest.size(), "both remaining checkpoints were freed");
            assertEquals(List.of(), router.loaded());
            for (Checkpoint checkpoint : Checkpoint.values()) {
                assertTrue(agents.lastSession(checkpoint).closed, checkpoint + " must be closed");
            }
        }
    }

    @Test
    @DisplayName("a lease keeps an evicted agent open until it is released")
    void leaseOutlivesEviction() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            Router.Lease lease = router.lease("english");
            TinyCheckpoint.RecordingSession session = agents.lastSession(Checkpoint.ENGLISH);
            Agent leased = lease.agent();

            // Evict it while the lease is open. Closing it here would break a prediction already
            // running on it, which is the whole reason a lease exists.
            router.load("multilingual");
            assertFalse(router.loaded().contains(Checkpoint.ENGLISH), "it was evicted");
            assertFalse(session.closed, "an agent under lease must stay open");
            assertSame(leased, lease.agent(), "and must still be reachable through the lease");

            lease.close();
            assertTrue(session.closed, "released, retired, and now closed");
            lease.close();          // idempotent, so try-with-resources round an early return is safe
            assertThrows(IllegalStateException.class, lease::agent,
                    "a released lease must not hand out an agent it no longer holds");
        }
    }

    @Test
    @DisplayName("two threads asking for the same cold checkpoint share one build")
    void concurrentLoadsShareOneBuild() throws Exception {
        // This is the case the in-flight handle exists for. An earlier draft registered one handle
        // and then waited on a different one, which left every waiter blocked on a latch nobody
        // counted down -- a deadlock, not a duplicate build, so it would not have shown up as a
        // wasted load.
        StubAgents agents = new StubAgents();
        CountDownLatch gate = new CountDownLatch(1);
        agents.blockOn = gate;
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            int threads = 6;
            CountDownLatch ready = new CountDownLatch(threads);
            CountDownLatch go = new CountDownLatch(1);
            List<Agent> seen = Collections.synchronizedList(new ArrayList<>());
            List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
            List<Thread> workers = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                Thread worker = new Thread(() -> {
                    ready.countDown();
                    try {
                        assertTrue(go.await(10, TimeUnit.SECONDS));
                        seen.add(router.load("english"));
                    } catch (Throwable failure) {
                        failures.add(failure);
                    }
                }, "load-" + i);
                worker.start();
                workers.add(worker);
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS), "the workers never started");
            go.countDown();
            // Let them pile up on the one build, then let it finish.
            Thread.sleep(50);
            gate.countDown();
            for (Thread worker : workers) {
                worker.join(TimeUnit.SECONDS.toMillis(20));
                assertFalse(worker.isAlive(), worker.getName() + " never finished: a waiter is"
                        + " blocked on a latch that was never counted down");
            }
            assertEquals(List.of(), failures, "a worker failed");
            assertEquals(threads, seen.size());
            assertEquals(1, agents.buildsOf(Checkpoint.ENGLISH),
                    "six callers must share one build, not pay for six copies");
            for (Agent agent : seen) {
                assertSame(seen.get(0), agent, "every caller must get the same agent");
            }
        }
    }

    @Test
    @DisplayName("a resident checkpoint answers while a different one is still loading")
    void residentLoadDoesNotWaitForAColdBuild() throws Exception {
        // The property the cache fast path in `acquire` exists for, and the only observable
        // difference between having it and not: a resident checkpoint is returned without
        // touching the build lock, so a request it can already serve is not stalled behind a cold
        // load of another model. In a server that difference is one checkpoint loading versus all
        // traffic stopping. A mutant that removed the fast path passed every other test here.
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            router.load("english");
            assertEquals(1, agents.buildsOf(Checkpoint.ENGLISH));

            CountDownLatch gate = new CountDownLatch(1);
            agents.blockOn = gate;                  // the next build, whichever it is, waits here
            CountDownLatch coldStarted = new CountDownLatch(1);
            List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
            Thread cold = new Thread(() -> {
                coldStarted.countDown();
                try {
                    router.load("multilingual");
                } catch (Throwable failure) {
                    failures.add(failure);
                }
            }, "cold-load");
            cold.start();
            assertTrue(coldStarted.await(10, TimeUnit.SECONDS), "the cold load never started");
            Thread.sleep(150);                      // let it reach the blocked factory

            CountDownLatch warmDone = new CountDownLatch(1);
            Thread warm = new Thread(() -> {
                try {
                    router.load("en");
                    warmDone.countDown();
                } catch (Throwable failure) {
                    failures.add(failure);
                }
            }, "warm-load");
            warm.start();
            assertTrue(warmDone.await(5, TimeUnit.SECONDS),
                    "a request for an already-loaded checkpoint blocked behind a cold load of a"
                    + " different one");

            gate.countDown();
            cold.join(TimeUnit.SECONDS.toMillis(20));
            warm.join(TimeUnit.SECONDS.toMillis(20));
            assertFalse(cold.isAlive(), "the cold load never finished");
            assertEquals(List.of(), failures, "a worker failed");
            assertEquals(1, agents.buildsOf(Checkpoint.ENGLISH), "english was never rebuilt");
            assertEquals(1, agents.buildsOf(Checkpoint.MULTILINGUAL));
        }
    }

    @Test
    @DisplayName("a failed build propagates and leaves no in-flight entry behind")
    void failedBuildDoesNotWedgeTheRouter() {
        StubAgents agents = new StubAgents();
        agents.failOn = Checkpoint.MULTILINGUAL;
        try (Router router = routerWith(agents).build()) {
            UncheckedIOException failure = assertThrows(UncheckedIOException.class,
                    () -> router.load("multilingual"));
            assertTrue(failure.getMessage().contains("multilingual"), failure.getMessage());
            assertEquals(List.of(), router.loaded());

            // The point: a second attempt must get the same error rather than hang on a latch the
            // failed attempt never counted down.
            assertThrows(UncheckedIOException.class, () -> router.load("multilingual"));
            // and an unrelated checkpoint still loads
            agents.failOn = null;
            router.load("english");
            assertEquals(List.of(Checkpoint.ENGLISH), router.loaded());
        }
    }

    @Test
    @DisplayName("concurrent callers share a FAILING build too, rather than each retrying it")
    void concurrentCallersShareAFailure() throws Exception {
        // This is what the in-flight latch earns that the build lock does not. Correctness-wise
        // the two are the same -- a mutant that removed the latch still produced one build per
        // checkpoint and passed every other test here -- but on the failure path the latch hands
        // the builder's error to the waiters, where without it each waiter re-enters the factory
        // and pays for the attempt again. For a load that fails by timing out a download, that is
        // the difference between one slow failure and six.
        StubAgents agents = new StubAgents();
        agents.failOn = Checkpoint.MULTILINGUAL;
        CountDownLatch gate = new CountDownLatch(1);
        agents.blockOn = gate;
        try (Router router = routerWith(agents).build()) {
            int threads = 6;
            CountDownLatch ready = new CountDownLatch(threads);
            CountDownLatch go = new CountDownLatch(1);
            List<Throwable> thrown = Collections.synchronizedList(new ArrayList<>());
            List<Thread> workers = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                Thread worker = new Thread(() -> {
                    ready.countDown();
                    try {
                        assertTrue(go.await(10, TimeUnit.SECONDS));
                        router.load("multilingual");
                        thrown.add(new AssertionError("the load should have failed"));
                    } catch (Throwable failure) {
                        thrown.add(failure);
                    }
                }, "failing-load-" + i);
                worker.start();
                workers.add(worker);
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS));
            go.countDown();
            Thread.sleep(100);          // let them pile up on the one attempt
            gate.countDown();
            for (Thread worker : workers) {
                worker.join(TimeUnit.SECONDS.toMillis(20));
                assertFalse(worker.isAlive(), worker.getName() + " never finished");
            }
            assertEquals(threads, thrown.size());
            // One caller built and sees the IOException wrapper directly; the rest waited and are
            // handed a per-thread wrapper carrying it as the cause. The builder's own exception
            // instance is deliberately NOT rethrown on the waiters: that gave every waiter a
            // stack trace of frames it never ran, and let concurrent handlers mutate one
            // Throwable's suppressed list, which Throwable does not support.
            int builders = 0;
            int waiters = 0;
            java.util.Set<Throwable> distinct = java.util.Collections.newSetFromMap(
                    new java.util.IdentityHashMap<>());
            for (Throwable failure : thrown) {
                distinct.add(failure);
                if (failure instanceof UncheckedIOException) {
                    builders++;
                } else {
                    assertTrue(failure instanceof IllegalStateException,
                            "a waiter must see a wrapper, got " + failure);
                    assertTrue(failure.getCause() instanceof UncheckedIOException,
                            "the wrapper must carry the builder's failure as its cause, got "
                            + failure.getCause());
                    waiters++;
                }
            }
            assertEquals(1, builders, "exactly one caller does the building");
            assertEquals(threads - 1, waiters, "and the rest wait on it");
            assertEquals(threads, distinct.size(),
                    "each caller must get its OWN throwable, not one shared instance");
            assertEquals(1, agents.attemptsOf(Checkpoint.MULTILINGUAL),
                    "six callers must share one failed attempt, not pay for six");
            assertEquals(List.of(), router.loaded());
        }
    }

    @Test
    @DisplayName("a router with no factory says so instead of failing obscurely")
    void routingWithoutAFactory() {
        try (Router router = Router.withDefaults()) {
            // Routing still works: it reads no disk. That is what makes this a usable mode and
            // not a misconfiguration.
            assertEquals(Checkpoint.ENGLISH,
                    router.route("I cannot log in to my account today").model());
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> router.load("english"));
            assertTrue(failure.getMessage().contains("route but not load"), failure.getMessage());
            assertTrue(failure.getMessage().contains("checkpointsRoot"),
                    "the message must say how to fix it: " + failure.getMessage());
        }
    }

    @Test
    @DisplayName("predict routes, then answers on the checkpoint that won")
    void predictRoutesThenAnswers() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            Prediction english = router.predict("I cannot log in to my account at all today",
                    oneQuestion());
            assertEquals(1, agents.buildsOf(Checkpoint.ENGLISH));
            assertEquals(0, agents.buildsOf(Checkpoint.MULTILINGUAL));
            assertEquals(1, english.answers().size());

            router.predict("계정에 로그인할 수 없어요",
                    oneQuestion());
            assertEquals(1, agents.buildsOf(Checkpoint.MULTILINGUAL),
                    "Korean must not be answered on the English checkpoint");
        }
    }

    @Test
    @DisplayName("predict leaves no lease behind, so the agent is closable afterwards")
    void predictReleasesItsLease() {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(1).build()) {
            router.predict("I cannot log in to my account at all today", oneQuestion());
            TinyCheckpoint.RecordingSession session = agents.lastSession(Checkpoint.ENGLISH);
            // If predict leaked its lease, eviction would retire the slot but never close it --
            // a leak that no other test here would notice.
            router.load("multilingual");
            assertTrue(session.closed, "predict must release its lease when it returns");
        }
    }

    @Test
    @DisplayName("predict tells the checkpoint the language routing detected")
    void predictPassesTheDetectedLanguage(@TempDir Path root) throws IOException {
        // The multilingual checkpoint takes a language and applies that language's fitted
        // temperatures, and the detected language is the best answer available. A port that
        // passed null here would route correctly and then ask the question WITHOUT the language
        // it had just worked out, which changes the confidence it reports -- silently.
        //
        // Made observable rather than asserted: this checkpoint overrides the noul temperature
        // for `pt`, so the same state and question answer differently with and without the
        // language. An earlier version of this test compared "multilingual" to "multilingual",
        // which is true whatever the router does.
        TinyCheckpoint.write(root, 64, 32);
        Files.writeString(root.resolve("rl_agent_config.json"),
                "{\"max_len\": 64, \"head_max_len\": 32,"
                + " \"temperature\": [1.0, 1.0, 1.0], \"temperature_by_options\": {},"
                + " \"lang_temperatures\": {\"pt\": {\"temperature\": [1.0, 1.0, 2.0]}}}",
                StandardCharsets.UTF_8);

        String portuguese = "Você pode me mandar a nota fiscal do pedido que eu fiz ontem?";
        double withoutLanguage;
        double withPortuguese;
        try (Agent direct = TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession())) {
            withoutLanguage = noulOf(direct.predict(portuguese, oneQuestion(), null));
            withPortuguese = noulOf(direct.predict(portuguese, oneQuestion(), "pt"));
        }
        assertNotEquals(withoutLanguage, withPortuguese,
                "this checkpoint must answer differently with the language, or the assertion"
                + " below cannot tell whether the router passed it");

        Router.AgentFactory factory = checkpoint ->
                TinyCheckpoint.agent(root, new TinyCheckpoint.RecordingSession());
        try (Router router = Router.builder().agents(factory).maxLoaded(3).build()) {
            Router.RouteDecision decided = router.route(portuguese);
            assertEquals(Checkpoint.MULTILINGUAL, decided.model());
            assertEquals("pt", decided.detection().language(), "the detector named Portuguese");

            double routed = noulOf(router.predict(decided, portuguese, oneQuestion()));
            assertEquals(withPortuguese, routed,
                    "the router must pass the language it detected, not null");

            // An explicit language wins over the detected one, which is the reference's rule --
            // and it decides without running detection at all.
            Router.RouteDecision pinned = router.route("plain english words here now", null,
                    Router.RouteOptions.none().lang("fr"));
            assertEquals(Checkpoint.MULTILINGUAL, pinned.model());
            assertNull(pinned.detection(),
                    "an explicit language decides without running detection");
        }
    }

    /** The probability a noul answer reports, which the fitted temperature moves. */
    private static double noulOf(Prediction prediction) {
        Answer answer = prediction.answers().get("urgent");
        assertTrue(answer instanceof Answer.Noul,
                "expected a noul answer, got " + answer.getClass().getSimpleName());
        return ((Answer.Noul) answer).noul();
    }

    @Test
    @DisplayName("a throwing close during eviction does not strand the new checkpoint")
    void aThrowingCloseDoesNotStrandTheNewAgent() throws IOException {
        // Eviction closes the victim. If that close throws, the publish of the NEW slot has
        // already happened -- so skipping the bookkeeping that follows left its lease stuck at
        // one forever, which meant the agent could never be retired or closed: an unreachable,
        // unclosable native session, and every waiter told the load failed when it had not.
        TinyCheckpoint.RecordingSession angry = new TinyCheckpoint.RecordingSession() {
            @Override
            public void close() {
                super.close();
                throw new IllegalStateException("a native close can fail");
            }
        };
        List<TinyCheckpoint.RecordingSession> built = new ArrayList<>();
        Router.AgentFactory factory = checkpoint -> {
            if (checkpoint == Checkpoint.ENGLISH) {
                return TinyCheckpoint.agent(checkpointRoot, angry);
            }
            TinyCheckpoint.RecordingSession session = new TinyCheckpoint.RecordingSession();
            built.add(session);
            return TinyCheckpoint.agent(checkpointRoot, session);
        };
        try (Router router = Router.builder().agents(factory).maxLoaded(1).build()) {
            router.load("english");
            assertThrows(IllegalStateException.class, () -> router.load("multilingual"),
                    "the failing close is surfaced, not swallowed");

            // The point: multilingual really is resident, and it is still CLOSABLE.
            assertEquals(List.of(Checkpoint.MULTILINGUAL), router.loaded());
            assertEquals(1, built.size());
            assertFalse(built.get(0).closed, "nothing has released it yet");
            assertEquals(List.of(Checkpoint.MULTILINGUAL), router.unload("multilingual"));
            assertTrue(built.get(0).closed,
                    "the new agent must be closable; a stranded lease would keep it open forever");
        }
    }

    @Test
    @DisplayName("closing one lease twice from two threads cannot close an agent another holds")
    void closingALeaseTwiceIsSafeUnderARace() throws Exception {
        // A plain check-then-set on the released flag loses this race: both threads see it unset,
        // both decrement, the count reaches zero while lease B is open, and the agent is closed
        // under a caller still using it -- the single failure the lease mechanism exists to stop.
        for (int attempt = 0; attempt < 200; attempt++) {
            StubAgents agents = new StubAgents();
            try (Router router = routerWith(agents).maxLoaded(2).build()) {
                Router.Lease a = router.lease("english");
                Router.Lease b = router.lease("english");
                TinyCheckpoint.RecordingSession session =
                        agents.lastSession(Checkpoint.ENGLISH);
                router.unload("english");           // retired, not closed: two leases are open
                assertFalse(session.closed);

                int racers = 4;
                CountDownLatch go = new CountDownLatch(1);
                List<Thread> threads = new ArrayList<>();
                for (int i = 0; i < racers; i++) {
                    Thread thread = new Thread(() -> {
                        try {
                            assertTrue(go.await(10, TimeUnit.SECONDS));
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                        a.close();
                    }, "closer-" + i);
                    thread.start();
                    threads.add(thread);
                }
                go.countDown();
                for (Thread thread : threads) {
                    thread.join(TimeUnit.SECONDS.toMillis(10));
                }
                assertFalse(session.closed,
                        "lease b is still open, so the agent must not be closed (attempt "
                        + attempt + ")");
                assertEquals(session, agents.lastSession(Checkpoint.ENGLISH));
                b.close();
                assertTrue(session.closed, "and closing the last lease does close it");
            }
        }
    }

    @Test
    @DisplayName("attaching the agent the router already holds does not close it")
    void attachingTheSameInstanceIsSafe() {
        // The javadoc invites exactly this: "register an already-built agent instead of loading a
        // second copy". Retiring the previous slot closed the very agent being attached, leaving
        // a live slot holding a closed session that nothing could ever replace.
        StubAgents agents = new StubAgents();
        // Managed by hand rather than with try-with-resources, because what is being asserted is
        // what close() does -- and an explicit close() inside a resource block is a -Xlint:try
        // warning, which is an error here.
        Router router = routerWith(agents).maxLoaded(2).build();
        Agent english = router.load("english");
        TinyCheckpoint.RecordingSession session = agents.lastSession(Checkpoint.ENGLISH);
        assertSame(english, router.attach("english", english));
        assertFalse(session.closed, "attaching the same instance must not close it");
        assertSame(english, router.load("english"), "and it is still the resident agent");
        assertEquals(1, router.loaded().size());
        // it is caller-owned now, so the router must leave it alone even on close()
        router.close();
        assertFalse(session.closed, "ownership moved to the caller, so the router leaves it");
        english.close();
        assertTrue(session.closed, "and the owner can still close it");
    }

    @Test
    @DisplayName("a slow close does not stall a lease on an already-resident checkpoint")
    void aSlowCloseDoesNotStallOtherCallers() throws Exception {
        // Closing under the registry lock stalled every other caller, including ones asking for a
        // checkpoint already in memory -- which is precisely what the fast path exists to keep
        // fast. Measured at 1.3 s for an unrelated lease behind one slow close.
        CountDownLatch closing = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        TinyCheckpoint.RecordingSession slow = new TinyCheckpoint.RecordingSession() {
            @Override
            public void close() {
                closing.countDown();
                try {
                    assertTrue(release.await(10, TimeUnit.SECONDS));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                }
                super.close();
            }
        };
        Router.AgentFactory factory = checkpoint -> checkpoint == Checkpoint.ENGLISH
                ? TinyCheckpoint.agent(checkpointRoot, slow)
                : TinyCheckpoint.agent(checkpointRoot, new TinyCheckpoint.RecordingSession());
        try (Router router = Router.builder().agents(factory).maxLoaded(2).build()) {
            router.load("english");
            router.load("typed-decisions");
            // On its own thread: the eviction's close now happens outside every lock, which is
            // the fix being tested -- so this call itself blocks until the close finishes.
            Thread evicting = new Thread(() -> router.load("multilingual"), "evicting-load");
            evicting.start();

            assertTrue(closing.await(10, TimeUnit.SECONDS), "the slow close never started");
            CountDownLatch leased = new CountDownLatch(1);
            Thread other = new Thread(() -> {
                try (Router.Lease lease = router.lease("typed-decisions")) {
                    assertNotNull(lease.agent());
                    leased.countDown();
                }
            }, "resident-lease");
            other.start();
            assertTrue(leased.await(5, TimeUnit.SECONDS),
                    "a lease on a resident checkpoint blocked behind an unrelated slow close");
            release.countDown();
            other.join(TimeUnit.SECONDS.toMillis(10));
            evicting.join(TimeUnit.SECONDS.toMillis(10));
            assertFalse(evicting.isAlive(), "the evicting load never finished");
            assertTrue(slow.closed, "and the slow close did complete");
        }
    }

    @Test
    @DisplayName("close frees what the router built and refuses further loads")
    void closeFreesAndSeals() {
        StubAgents agents = new StubAgents();
        Router router = routerWith(agents).maxLoaded(3).build();
        router.preload();
        router.close();
        for (Checkpoint checkpoint : Checkpoint.values()) {
            assertTrue(agents.lastSession(checkpoint).closed, checkpoint + " must be closed");
        }
        assertEquals(List.of(), router.loaded());
        assertThrows(IllegalStateException.class, () -> router.load("english"));
        assertThrows(IllegalStateException.class, () -> router.lease("english"));
        router.close();     // idempotent
        // and routing a closed router is still harmless, because it touches nothing
        assertEquals(Checkpoint.ENGLISH, router.route("plain english text here now").model());
    }

    @Test
    @DisplayName("maxLoaded below one is refused, because zero resident checkpoints answers nothing")
    void maxLoadedMustBePositive() {
        assertThrows(IllegalArgumentException.class, () -> Router.builder().maxLoaded(0));
        assertThrows(IllegalArgumentException.class, () -> Router.builder().maxLoaded(-1));
    }

    @Test
    @DisplayName("attaching over an existing agent retires the one it replaced")
    void attachReplaces() throws IOException {
        StubAgents agents = new StubAgents();
        try (Router router = routerWith(agents).maxLoaded(3).build()) {
            router.load("english");
            TinyCheckpoint.RecordingSession built = agents.lastSession(Checkpoint.ENGLISH);
            Agent replacement = TinyCheckpoint.agent(checkpointRoot,
                    new TinyCheckpoint.RecordingSession());
            router.attach("english", replacement);
            assertSame(replacement, router.load("english"));
            assertTrue(built.closed,
                    "the agent the router built and then replaced must be closed, not leaked");
            assertEquals(1, router.loaded().size(), "and english is resident exactly once");
        }
    }
}
