// What this SDK costs, measured the only way a JVM measurement means anything: warm, best of N,
// and with the thing you are timing separated from the thing you are not.
//
// A first measurement on a cold JVM is dominated by class loading, JIT and -- for this SDK -- the
// one-time walk of Unicode that builds the character classes. Quoting that number as throughput
// is how a port comes to look slower than it is, and quoting a single warm run is how noise comes
// to look like a regression. So every figure below is the best of N after a warmup, and the cold
// first call is reported separately rather than averaged into the rest.
//
//   python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
//   java -cp "<jar>:<onnxruntime>" samples/Benchmark.java .work/checkpoints/multilingual .work/onnx/multilingual
//
// See samples/README.md for the classpath.

import com.convaiinnovations.laya.Agent;
import com.convaiinnovations.laya.LayaEmail;
import com.convaiinnovations.laya.Presets;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.lang.LanguageDetection;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public final class Benchmark {

    // Enough warmup to get past the interpreter, and no more. A model-free call is
    // microseconds, so 20 of them cost nothing; a forward pass is tens of milliseconds and a
    // batch of 16 is more, so the same count there would run for minutes and read as a hang. A
    // sample that looks hung is a sample nobody finishes running.
    private static final int WARMUP = 20;
    private static final int ROUNDS = 5;
    private static final int MODEL_WARMUP = 1;
    private static final int MODEL_ROUNDS = 2;
    /** Small, because on CPU this is seconds per round and a sample has to finish. */
    private static final int BATCH = 8;

    public static void main(String[] args) throws Exception {
        // ------------------------------------------------- the model-free half, which needs nothing
        String shortState = "I cannot log in and need a password reset.";
        String longState = shortState.repeat(90);              // ~3.8 KB
        String email = "I was charged twice for order 8812.\n\nAtenciosamente,\nAna Souza\n"
                + "Enviado do meu iPhone\n\nEsta mensagem e confidencial.\n\n"
                + "Em ter., 3 de set. de 2025, Suporte <s@x.com> escreveu:\n> Podemos ajudar?";

        System.out.println("model-free (no checkpoint, no graph)");
        // The cold call is reported, not hidden: it is what a caller's first request pays.
        long coldDetect = time(() -> LanguageDetection.analyse(shortState));
        System.out.printf("  %-34s %8.3f ms   <- cold, includes class init%n",
                "detect, first ever call", coldDetect / 1e6);
        bench("detect, 42-char state", () -> LanguageDetection.analyse(shortState));
        bench("detect, 3.8 KB state", () -> LanguageDetection.analyse(longState));

        long coldClean = time(() -> LayaEmail.cleanEmailBody(email));
        System.out.printf("  %-34s %8.3f ms   <- cold, builds the character classes%n",
                "email clean, first ever call", coldClean / 1e6);
        bench("email clean", () -> LayaEmail.cleanEmailBody(email));

        if (args.length < 2) {
            System.out.println();
            System.out.println("pass <checkpoint-dir> <graph-dir> to also measure the forward pass");
            return;
        }

        // ------------------------------------------------- the forward pass
        Map<String, Question> questions = Presets.email();
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("subject", "Charged twice for order 8812");
        state.put("body", "I was billed two times this morning. Please refund one charge.");

        List<Object> batch = new ArrayList<>();
        for (int i = 0; i < BATCH; i++) {
            batch.add(state);
        }

        System.out.println();
        System.out.println("forward pass (" + questions.size() + " questions per state)");
        try (Agent agent = Agent.open(Path.of(args[0]), Path.of(args[1]))) {
            long coldPredict = time(() -> agent.predict(state, questions));
            System.out.printf("  %-34s %8.3f ms   <- cold, includes the ORT session%n",
                    "predict, first ever call", coldPredict / 1e6);
            bench("predict, 1 state", () -> agent.predict(state, questions),
                    MODEL_WARMUP, MODEL_ROUNDS);
            long batched = best(() -> agent.predictBatch(batch, questions),
                    MODEL_WARMUP, MODEL_ROUNDS);
            System.out.printf("  %-34s %8.3f ms%n",
                    "predictBatch, " + BATCH + " states", batched / 1e6);
            // Per state, which is the number that matters when you are sizing a queue.
            System.out.printf("  %-34s %8.3f ms%n",
                    "  ... per state", batched / 1e6 / batch.size());
        }

        System.out.println();
        System.out.printf("Model-free figures are the best of %d after %d warmup calls; the "
                + "model-backed%nones are the best of %d after %d, because a forward pass is "
                + "milliseconds and not%nmicroseconds. The two cold figures are single calls.%n",
                ROUNDS, WARMUP, MODEL_ROUNDS, MODEL_WARMUP);
        System.out.println("Best, not mean: the mean of a JVM measurement is a mean over how much");
        System.out.println("other work the machine happened to be doing.");
        System.out.println();
        System.out.println("If the per-state batch figure is no better than the single-state one,");
        System.out.println("that is the expected shape on CPU rather than a problem: batching");
        System.out.println("amortises tokenization, collation and the session call, and none of");
        System.out.println("those is the cost here -- the matmuls are, and they do not amortise.");
        System.out.println("Batching pays on a GPU, and on CPU it pays for small models.");
    }

    /** One untimed-warmup-free call, for the cold number. */
    private static long time(Supplier<?> work) {
        long started = System.nanoTime();
        Object result = work.get();
        long elapsed = System.nanoTime() - started;
        if (result == null) {
            throw new IllegalStateException("the work returned nothing");
        }
        return elapsed;
    }

    private static void bench(String label, Supplier<?> work) {
        bench(label, work, WARMUP, ROUNDS);
    }

    private static void bench(String label, Supplier<?> work, int warmup, int rounds) {
        System.out.printf("  %-34s %8.3f ms%n", label, best(work, warmup, rounds) / 1e6);
    }

    /** Best of {@code rounds}, after {@code warmup} calls, with every result consumed. */
    private static long best(Supplier<?> work, int warmup, int rounds) {
        for (int i = 0; i < warmup; i++) {
            consume(work.get());
        }
        long best = Long.MAX_VALUE;
        for (int round = 0; round < rounds; round++) {
            long started = System.nanoTime();
            Object result = work.get();
            best = Math.min(best, System.nanoTime() - started);
            consume(result);
        }
        return best;
    }

    /** Keep the JIT from deciding the work is dead and removing it. */
    private static void consume(Object result) {
        if (result != null && result.hashCode() == Integer.MIN_VALUE) {
            System.out.print("");
        }
    }
}
