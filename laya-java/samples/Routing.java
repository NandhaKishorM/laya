// Routing and shortlisting: pick a checkpoint from the state's own script, then cut a large label
// set down before the model sees it.
//
// The two go together because they fail in opposite directions. The English checkpoint does not
// gently degrade off English -- on 20-option intent it scores 0.100 on Hindi against 0.050 for
// random guessing, and reports high confidence while doing it -- so the router's job is to never
// send it text it cannot read. The shortlist's job is the other end: a question with 400 options
// does not fit a context window, so something has to choose which 20 the model is asked about,
// and if that choice is wrong the right answer was never on the ballot.
//
//   python laya-java/scripts/prepare_checkpoint.py --checkpoint multilingual
//   java -cp "<jar>:<onnxruntime>" samples/Routing.java .work/checkpoints .work/onnx
//
// See samples/README.md for the classpath.

import com.convaiinnovations.laya.Agent;
import com.convaiinnovations.laya.Question;
import com.convaiinnovations.laya.Router;
import com.convaiinnovations.laya.Shortlist;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Routing {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: Routing <checkpoints-root> <onnx-root>");
            System.exit(2);
        }
        Path checkpoints = Path.of(args[0]);
        Path graphs = Path.of(args[1]);

        // ---------------------------------------------------------------- the decision alone
        //
        // `route` needs no model and loads nothing. It is worth looking at on its own, because
        // the reason string is the part you will read in a log when a request went to a
        // checkpoint you did not expect.
        try (Router router = Router.withDefaults()) {
            for (String state : List.of(
                    "I cannot log in and need a password reset.",
                    "मुझे अपना "
                            + "पासवर्ड बदलना "
                            + "है।",
                    "Je n'arrive pas à me connecter.",
                    "注文をキャンセルしたいです。")) {
                Router.RouteDecision decision = router.route(state);
                System.out.printf("%-10s  %s%n", decision.model().wireName(), decision.reason());
            }
        }

        // ---------------------------------------------------------------- routing a real request
        //
        // `agents` is how the router opens a checkpoint. It is a factory rather than a path so
        // that the lifecycle stays the router's: it loads on demand, keeps at most `maxLoaded`
        // resident, and closes the least recently used one -- but never one a caller still holds.
        System.out.println();
        try (Router router = Router.builder()
                .maxLoaded(1)                       // deliberately 1, so the eviction is visible
                .agents(checkpoint -> Agent.open(
                        checkpoints.resolve(checkpoint.wireName()),
                        graphs.resolve(checkpoint.wireName())))
                .build()) {

            // A choice is a MAP of label to criterion, rendered in the map's iteration order.
            // A criterion of "" means "no description", and the option renders as its label.
            Map<String, String> intents = new LinkedHashMap<>();
            intents.put("reset a password", "the sender cannot sign in");
            intents.put("cancel an order", "the sender wants an order stopped");
            intents.put("request a refund", "the sender wants money back");
            intents.put("report a bug", "something is broken");
            intents.put("something else", "");
            Map<String, Question> questions = new LinkedHashMap<>();
            questions.put("intent", Question.choice("What does the sender want?", intents));

            // A Portuguese state, deliberately: it routes to multilingual, so this sample needs
            // only ONE graph. An English state would route to the english checkpoint and ask you
            // to export a second 1.3 GB artifact to see the same thing.
            Map<String, Object> state = new LinkedHashMap<>();
            state.put("subject", "Nao consigo entrar na conta");
            state.put("body", "Minha senha parou de funcionar depois da atualizacao de hoje.");

            // One call: route, load if needed, predict. The decision is reported alongside.
            Router.RouteDecision decision = router.route(state, questions);
            System.out.println("routed to " + decision.model().wireName()
                    + " because: " + decision.reason());
            System.out.println("answer:    "
                    + router.predict(decision, state, questions).answer("intent"));
        }

        // ---------------------------------------------------------------- shortlisting
        //
        // A deliberately terrible embedder: hashed character trigrams. It is here to make the
        // shape clear without pulling in a model -- in a real program this is your bi-encoder,
        // and the quality of the shortlist is the quality of that encoder, not of laya.
        System.out.println();
        Map<String, String> departments = new LinkedHashMap<>();
        for (int i = 1; i <= 400; i++) {
            departments.put("department-" + i, "");
        }
        departments.put("billing disputes and duplicate charges", "refunds and double charges");

        Shortlist.Embedder embedder = Routing::hashedTrigrams;
        Question question = Question.choice("Which team should own this?", departments);
        Shortlist.Ranking ranking = Shortlist.rank(
                "I was charged twice for the same order and want one charge refunded.",
                question, embedder, 5);

        System.out.printf("shortlisted %d of %d labels (passthrough=%s)%n",
                ranking.labels().size(), ranking.total(), ranking.passthrough());
        for (int i = 0; i < ranking.labels().size(); i++) {
            System.out.printf("  %.4f  %s%n", ranking.scores()[i], ranking.labels().get(i));
        }
        System.out.println();
        System.out.println("The real answer is on the shortlist only because the embedder put it");
        System.out.println("there. A shortlist is a filter in front of the model, so a bad one");
        System.out.println("removes the right answer before the model can be asked about it.");
    }

    /**
     * Hashed character trigrams, L2-normalised. A demo, not a recommendation: it has no notion of
     * meaning and matches on shared spelling.
     */
    private static double[][] hashedTrigrams(List<String> texts) {
        int width = 256;
        double[][] rows = new double[texts.size()][width];
        for (int r = 0; r < texts.size(); r++) {
            String text = texts.get(r).toLowerCase(java.util.Locale.ROOT);
            for (int i = 0; i + 3 <= text.length(); i++) {
                int bucket = Math.abs(text.substring(i, i + 3).hashCode()) % width;
                rows[r][bucket] += 1.0;
            }
            double norm = 0.0;
            for (double v : rows[r]) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);
            if (norm > 0.0) {
                for (int c = 0; c < width; c++) {
                    rows[r][c] /= norm;
                }
            }
        }
        return rows;
    }
}
