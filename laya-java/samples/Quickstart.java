// The SDK quickstart as a runnable program: four typed questions about one support email, in one
// forward pass, read out as typed answers.
//
// Run it with the single-file launcher -- no build, no module, no Gradle:
//
//   python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
//   cd laya-java && ./gradlew --no-daemon jar
//   java -cp "laya-java/build/libs/laya-java-0.1.0-SNAPSHOT.jar:$(cat samples/.onnx-classpath)" \
//        samples/Quickstart.java .work/checkpoints/multilingual .work/onnx/multilingual
//
// See samples/README.md for the classpath, which is the only fiddly part.

import com.convaiinnovations.laya.Agent;
import com.convaiinnovations.laya.Answer;
import com.convaiinnovations.laya.Prediction;
import com.convaiinnovations.laya.Presets;
import com.convaiinnovations.laya.Question;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class Quickstart {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: Quickstart <checkpoint-dir> <graph-dir>");
            System.exit(2);
        }
        Path checkpoint = Path.of(args[0]);
        Path graph = Path.of(args[1]);

        // The questions are plain data. The same objects answer an English state and a Hindi one;
        // nothing about them is language-specific, which is the point of the preset.
        Map<String, Question> questions = Presets.email();

        // A state is a map, and every field is shown to the model. `subject` and `body` here
        // because that is what the email preset is written against.
        Map<String, Object> english = new LinkedHashMap<>();
        english.put("subject", "Charged twice for order 8812");
        english.put("body", "I was billed two times for the same order this morning. "
                + "Please refund one of the charges. I have attached both receipts.");

        Map<String, Object> hindi = new LinkedHashMap<>();
        hindi.put("subject", "ऑर्डर 8812 का "
                + "दो बार बिल");
        hindi.put("body", "मुझे उसी ऑर्डर "
                + "के लिए दो बार "
                + "बिल किया गया। "
                + "कृपया रिफंड करें।");

        // try-with-resources: the agent owns a native ONNX session, and closing it is not optional.
        try (Agent agent = Agent.open(checkpoint, graph)) {
            report("English", agent.predict(english, questions));
            report("Hindi", agent.predict(hindi, questions));
        }
    }

    /** Print every answer by its actual type, which is what the sealed interface is for. */
    private static void report(String label, Prediction prediction) {
        System.out.println();
        System.out.println("== " + label + " ==");
        for (String id : prediction.answers().keySet()) {
            Answer answer = prediction.answer(id);
            // A choice is not a score. Pattern matching over the sealed interface means the
            // compiler checks that every shape is handled, rather than a map lookup that cannot.
            if (answer instanceof Answer.Choice choice) {
                System.out.printf("  %-12s %-22s confidence %.3f%n",
                        id, choice.choice(), choice.confidence());
            } else if (answer instanceof Answer.Score score) {
                System.out.printf("  %-12s %-22.3f confidence %.3f%n",
                        id, score.score(), score.confidence());
            } else if (answer instanceof Answer.Noul noul) {
                System.out.printf("  %-12s %-22.3f confidence %.3f%n",
                        id, noul.noul(), noul.confidence());
            }
        }
        // Usage is not decoration: it says whether the model actually saw the state, and a
        // truncated request names the question that was cut.
        System.out.printf("  usage        %d input tokens, %d state tokens dropped, truncated=%s%n",
                prediction.usage().inputTokens(),
                prediction.usage().stateTokensDropped(),
                prediction.usage().truncated());
    }
}
