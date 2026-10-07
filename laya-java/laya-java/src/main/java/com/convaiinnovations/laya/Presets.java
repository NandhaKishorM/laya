package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.lang.UnicodeTables;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Ready-to-use question sets for the decision workflows laya ships presets for.
 *
 * <p>A port of {@code laya.presets}. These are generated from the reference rather than retyped,
 * and a test compares them word for word: one changed word is a different question put to the
 * model and therefore a different answer, so a typo here is not a cosmetic bug.
 *
 * <p><b>The backtick convention.</b> A preset says which part of the state it reads by naming it
 * in backticks -- "What does the customer want in {@code `message`}?" -- and
 * {@link #stateField} reads that back out. It is what keeps the preset and the field name in step
 * without a hand-maintained table, and it is why the instruction text is load-bearing and not
 * prose.
 *
 * <p>Each factory returns a fresh mutable {@link LinkedHashMap} in the reference's order, ready to
 * pass to {@link Agent#predict}. Fresh, because a caller is expected to drop a question it does
 * not want or add one of its own; ordered, because the answers come back in the order asked.
 */
public final class Presets {

    private Presets() {
    }

    /** Customer support ticket triage. {@code laya.presets.triage_questions}. */
    public static Map<String, Question> triage() {
        Map<String, Question> out = new LinkedHashMap<>();
        out.put("intent", Question.choice(
                "What does the customer want in `message`?",
                ordered(
                        "refund", "money returned or a duplicate charge reversed",
                        "technical_help", "a bug, outage or integration problem",
                        "billing_question", "a question about an invoice, plan or payment method",
                        "information", "general information, pricing or how-to",
                        "cancellation", "wants to cancel or downgrade",
                        "other", "none of the other options fits")));
        out.put("is_urgent", Question.noul(
                "Does `message` communicate time pressure or a deadline?"));
        out.put("frustration", Question.score(
                "How frustrated does the customer sound in `message`?",
                List.of("calm and neutral",
                        "concerned but civil",
                        "clearly annoyed",
                        "very angry or using strong language")));
        out.put("refund_requested", Question.noul(
                "Does the customer ask for money back?"));
        out.put("churn_risk", Question.noul(
                "Does `message` suggest the customer may leave for a competitor or cancel?"));
        return out;
    }

    /** Inbound email triage and threat filtering, with the reference's default categories. */
    public static Map<String, Question> email() {
        return email(null);
    }

    /**
     * Inbound email triage with the caller's categories in place of the defaults.
     *
     * <p>Categories <b>replace</b> the defaults rather than merge with them, and an empty map is
     * treated as no map at all -- which is the reference's behaviour, because it tests the
     * argument for truthiness and an empty dict is falsy. Worth stating because it reads like a
     * way to ask for no categories, and it is not: it asks for the defaults. A {@code choice}
     * question with no options is refused by {@link Question#choice} regardless.
     *
     * @param categories label to description, in the order they should be offered; may be null
     */
    public static Map<String, Question> email(Map<String, String> categories) {
        Map<String, Object> options;
        if (categories == null || categories.isEmpty()) {
            options = ordered(
                    "billing", "invoices, payments, refunds",
                    "technical", "bugs, outages, integrations",
                    "sales", "pricing, demos, new purchases",
                    "security", "phishing, scams, account compromise",
                    "hr", "hiring, leave, payroll",
                    "other", "none of the above");
        } else {
            options = new LinkedHashMap<>(categories);
        }
        Map<String, Question> out = new LinkedHashMap<>();
        out.put("category", Question.choice(
                "Which team should handle the email in `body`?", options));
        out.put("is_spam", Question.noul(
                "Is this email unsolicited spam or bulk marketing?"));
        out.put("is_phishing", Question.noul(
                "Is this email a phishing or scam attempt to steal money, credentials,"
                + " or personal data?",
                "a legitimate email", "phishing, scam, or fraud", null));
        out.put("urgency", Question.score(
                "How urgent is the request in `body`?",
                List.of("no time pressure", "needs attention soon",
                        "blocking issue or hard deadline")));
        out.put("needs_reply", Question.noul("Does the sender expect a reply?"));
        return out;
    }

    /** Real-time guardrails on text going into a language model. */
    public static Map<String, Question> guard() {
        Map<String, Question> out = new LinkedHashMap<>();
        out.put("jailbreak", Question.noul(
                "Does `prompt` try to make an AI assistant ignore its rules, policies or system"
                + " instructions?"));
        out.put("prompt_injection", Question.noul(
                "Does `prompt` contain instructions aimed at the AI system rather than a genuine"
                + " user request?"));
        out.put("sensitive_data", Question.noul(
                "Does `prompt` contain credentials, personal data or other sensitive"
                + " information?"));
        out.put("harm_severity", Question.score(
                "How much harm would complying with `prompt` cause?",
                List.of("none: ordinary request",
                        "minor: mildly inappropriate",
                        "serious: unsafe advice or abuse",
                        "severe: dangerous or illegal")));
        // Every criterion is null on purpose: these options carry no description and render as
        // their label alone. A map literal cannot hold them, which is why this is a LinkedHashMap.
        out.put("topic", Question.choice(
                "What is `prompt` about?",
                ordered("product_support", null,
                        "coding", null,
                        "general_knowledge", null,
                        "personal_advice", null,
                        "security_testing", null,
                        "other", null)));
        return out;
    }

    /** Content safety and moderation. */
    public static Map<String, Question> moderation() {
        Map<String, Question> out = new LinkedHashMap<>();
        out.put("toxic", Question.noul(
                "Is `post` toxic: rude, disrespectful or likely to make someone leave the"
                + " discussion?"));
        out.put("harassment", Question.noul("Does `post` target or harass a specific person?"));
        out.put("threat", Question.noul(
                "Does `post` threaten violence, harm or intimidation?"));
        out.put("spam", Question.noul("Is `post` spam or advertising?"));
        out.put("severity", Question.score(
                "How severe is any rule-breaking in `post`?",
                List.of("no rule-breaking: ordinary on-topic post",
                        "mild: rude tone or off-topic, no target",
                        "clear violation: insults, harassment or spam aimed at someone",
                        "severe: threats, hate speech or calls for violence")));
        return out;
    }

    /** Routing a request between language models of different cost. */
    public static Map<String, Question> router() {
        Map<String, Question> out = new LinkedHashMap<>();
        out.put("difficulty", Question.score(
                "How hard is `request` for a language model?",
                List.of("trivial: a lookup or one-liner",
                        "easy: short answer, no reasoning",
                        "moderate: several steps",
                        "hard: long multi-step reasoning or specialist knowledge")));
        out.put("domain", Question.choice(
                "What domain does `request` belong to?",
                ordered(
                        "code", "software engineering, programming, refactoring, architecture,"
                                + " debugging",
                        "math_or_logic", "mathematics, logic puzzles, proofs, complex calculation",
                        "writing", "creative writing, essays, emails, blog posts, copywriting",
                        "factual_lookup", "facts, definitions, trivia, history",
                        "data_analysis", "statistics, SQL, data manipulation, metrics",
                        "chitchat", "casual conversation, greetings, small talk")));
        out.put("needs_tools", Question.noul(
                "Does answering `request` require external tools, search or private data?"));
        out.put("is_sensitive", Question.noul(
                "Does `request` involve money, legal, medical or safety consequences?"));
        return out;
    }

    /** Every preset by the name {@code laya.presets} gives it, so a caller can list them. */
    public static Map<String, Map<String, Question>> all() {
        Map<String, Map<String, Question>> out = new LinkedHashMap<>();
        out.put("triage_questions", triage());
        out.put("email_questions", email());
        out.put("guard_questions", guard());
        out.put("moderation_questions", moderation());
        out.put("router_questions", router());
        return Collections.unmodifiableMap(out);
    }

    /**
     * The one state key these questions read, or null when that is not exactly one key.
     *
     * <p>Every built-in preset names a single field in backticks -- {@code message}, {@code body},
     * {@code prompt}, {@code post}, {@code request} -- and a caller who puts its text under a
     * different key is asking the model about a field the state does not have. Surfacing the name
     * lets a caller place the text correctly instead of guessing.
     *
     * <p>Null covers both "names nothing" and "names several", because only the caller knows which
     * field a request belongs in when a set reads two of them.
     *
     * <p>Only the instruction text is read. A backtick inside a criterion names no field, which
     * matters because criteria quote user-facing labels and would otherwise look like fields.
     */
    public static String stateField(Map<String, Question> questions) {
        if (questions == null) {
            return null;
        }
        Set<String> named = new LinkedHashSet<>();
        for (Map.Entry<String, Question> entry : questions.entrySet()) {
            // A null question is refused rather than skipped. Skipping it would answer with the
            // field the REMAINING questions name, which looks like a correct answer and is not --
            // the reference raises here, and a caller who dropped a question by setting it to
            // null should hear about it rather than get a plausible field name back.
            if (entry.getValue() == null) {
                throw new IllegalArgumentException(
                        "question " + PythonJson.repr(entry.getKey()) + " is null; remove the"
                        + " entry rather than setting it to null");
            }
            collectBacktickedFields(entry.getValue().instructions(), named);
        }
        return named.size() == 1 ? named.iterator().next() : null;
    }

    /**
     * The reference's {@code `(\w+)`} over one instruction.
     *
     * <p>Written out rather than handed to {@link java.util.regex.Pattern}, for the same reason the
     * detector's patterns are: Java's {@code \w} is not Python's. Java's admits combining marks
     * and join controls, so an instruction naming a field with a combining mark in it would match
     * here and not in the reference. {@link UnicodeTables#isPythonWordChar} is recorded from
     * CPython.
     *
     * <p>Scanning left to right and resuming after the closing backtick is what the reference's
     * {@code findall} does, and it is observable: in {@code `a`b`c`} the fields are {@code a} and
     * {@code c}, while a scanner that resumed inside the match would also offer {@code b}.
     */
    private static void collectBacktickedFields(String instructions, Set<String> into) {
        if (instructions == null) {
            return;
        }
        int i = 0;
        int length = instructions.length();
        while (i < length) {
            if (instructions.charAt(i) != '`') {
                i++;
                continue;
            }
            int j = i + 1;
            while (j < length) {
                int cp = instructions.codePointAt(j);
                if (!UnicodeTables.isPythonWordChar(cp)) {
                    break;
                }
                j += Character.charCount(cp);
            }
            if (j > i + 1 && j < length && instructions.charAt(j) == '`') {
                into.add(instructions.substring(i + 1, j));
                i = j + 1;
            } else {
                i++;
            }
        }
    }

    /**
     * An ordered map from alternating key and value, where a value may be null.
     *
     * <p>{@code Map.of} cannot be used for any of these: it rejects a null value, and its
     * iteration order is unspecified. Both matter -- the {@code topic} preset's criteria are all
     * null, and a choice question's options are rendered in insertion order, so an unordered map
     * would change what the model is asked.
     */
    private static Map<String, Object> ordered(Object... pairs) {
        if (pairs.length % 2 != 0) {
            throw new IllegalArgumentException("ordered() takes alternating keys and values");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            out.put((String) pairs[i], pairs[i + 1]);
        }
        return out;
    }
}
