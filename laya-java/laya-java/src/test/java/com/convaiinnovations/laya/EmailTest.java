package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.lang.UnicodeTables;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@code laya.email}, against expectations recorded by running the reference.
 *
 * <p>The corpus is not a sample. Every case is a rule the module's comments argue for: a marker
 * that must fire, and -- more of them -- a near-miss that must NOT, because a cleaner that is too
 * eager deletes the sender's request, which is worse than leaving boilerplate behind. Eight of the
 * cases exist because a mutant survived without them, and they are named so that a later edit
 * which "simplifies" one of those rules has something to fail.
 */
final class EmailTest {

    /**
     * Cases the corpus must hold by NAME, not by count.
     *
     * <p>A count is not coverage. Trimmed to its old floors -- which were 70 against 86 and 5
     * against 9 -- this corpus still passed while losing every astral case and eleven
     * near-misses, and three mutants that the full corpus kills then survived: the code-point
     * width gate, the surrogate-safe cut and the UNICODE_CHARACTER_CLASS flag on `\b`. Each
     * name below is a rule nothing else in the corpus pins.
     */
    private static final List<String> REQUIRED_CLEANED = List.of(
            // near-misses: the cleaner must NOT fire. More of these than hits, because a cleaner
            // that is too eager deletes what the sender actually wrote.
            "not-signoff-pt-with-name", "not-signoff-sentence",
            "not-signoff-lowercase-other-script", "not-signoff-symbol-name", "not-signoff-zwnj",
            "not-signoff-obrigado-mas", "not-signoff-merci-mais", "not-footer-with-request",
            "not-disclaimer-question", "not-disclaimer-colon", "not-disclaimer-print",
            "not-disclaimer-contract", "not-disclaimer-fr-contract",
            "not-signoff-baseless-mark", "not-signoff-too-wide", "not-signoff-four-tokens",
            // outside the BMP, where the reference counts code points and Java reaches for chars
            "astral-word-char-breaks-no-boundary", "astral-elsewhere-still-matches",
            "astral-signoff-at-the-width-limit", "astral-signoff-just-over-the-limit",
            "astral-mark-in-a-disclaimer",
            // the signature window's five parameters, the footer width in both directions, and
            // the body-side strip -- each the only case that tells its rule from a plausible
            // wrong one
            "window-floor-of-one", "window-offset-dominates", "window-offset-excludes",
            "window-ratio-excludes", "window-ratio-includes",
            "footer-at-the-width-limit", "footer-just-over-the-limit",
            "body-strip-is-pythons", "body-strip-both-ends");

    private static final List<String> REQUIRED_BUDGETS = List.of(
            "budget-four-x-bound", "budget-splits-a-surrogate-pair",
            "budget-ends-on-a-surrogate-pair", "budget-all-astral",
            "budget-four-x-bound-astral",
            // a negative budget is not a zero budget
            "budget-negative-small", "budget-negative-one", "budget-negative-astral",
            "budget-negative-empties");

    @SuppressWarnings("unchecked")
    private static List<Object> rows(String key) {
        return (List<Object>) Fixtures.load("email.json").get(key);
    }

    private static List<String> names(String key) {
        List<String> out = new ArrayList<>();
        for (Object entry : rows(key)) {
            out.add((String) asMap(entry).get("name"));
        }
        return out;
    }

    @Test
    @DisplayName("the corpus still holds every case that pins a rule nothing else pins")
    void corpusHoldsItsLoadBearingCases() {
        List<String> cleaned = names("cleaned");
        for (String required : REQUIRED_CLEANED) {
            assertTrue(cleaned.contains(required),
                    () -> "the corpus lost " + required + ", which is the only case pinning its "
                          + "rule; regenerate from a gen_fixtures.py that still has it");
        }
        List<String> budgets = names("budgets");
        for (String required : REQUIRED_BUDGETS) {
            assertTrue(budgets.contains(required),
                    () -> "the corpus lost the budget case " + required);
        }
        assertEquals(95, cleaned.size(), "cases were added or removed; update the count knowingly");
        assertEquals(13, budgets.size(), "budget cases were added or removed");
        assertEquals(7, names("states").size(), "state cases were added or removed");
    }

    @TestFactory
    @DisplayName("every cleaned body is byte-identical to the reference")
    List<DynamicTest> cleanedMatches() {
        List<Object> cases = rows("cleaned");
        assertEquals(95, cases.size(),
                "the email corpus changed size; corpusHoldsItsLoadBearingCases says which "
                + "cases may not go");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            tests.add(dynamicTest(name, () -> assertEquals(
                    row.get("result"), LayaEmail.cleanEmailBody((String) row.get("body")),
                    () -> "clean_email_body diverged on " + name)));
        }
        return tests;
    }

    @TestFactory
    @DisplayName("every budget cuts where the reference cuts")
    List<DynamicTest> budgetsMatch() {
        List<Object> cases = rows("budgets");
        assertEquals(13, cases.size(), "the budget cases changed size");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            int budget = ((Number) row.get("max_chars")).intValue();
            tests.add(dynamicTest(name + " (max " + budget + ")", () -> assertEquals(
                    row.get("result"), LayaEmail.cleanEmailBody((String) row.get("body"), budget),
                    () -> "clean_email_body diverged on " + name)));
        }
        return tests;
    }

    @TestFactory
    @DisplayName("every state matches the reference, field for field and in order")
    @SuppressWarnings("unchecked")
    List<DynamicTest> statesMatch() {
        List<Object> cases = rows("states");
        assertEquals(7, cases.size(), "the state cases changed size");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            tests.add(dynamicTest(name, () -> {
                Map<String, Object> want = (Map<String, Object>) row.get("result");
                Map<String, Object> got = LayaEmail.emailState(
                        (String) row.get("subject"), (String) row.get("body"),
                        (String) row.get("sender"), (Boolean) row.get("clean"),
                        ((Number) row.get("max_chars")).intValue(),
                        new LinkedHashMap<>(asMap(row.get("extra"))));
                assertEquals(want, got, () -> "email_state diverged on " + name);
                // Key order is not cosmetic: it is the order the model is shown the fields in.
                assertEquals(new ArrayList<>(want.keySet()), new ArrayList<>(got.keySet()),
                        () -> "email_state field order diverged on " + name);
            }));
        }
        return tests;
    }

    @Test
    @DisplayName("the re-export is the preset, and hands out a fresh map each time")
    void questionsAreThePreset() {
        // The DisplayName used to say "the same object graph", which is the opposite of true and
        // of what is wanted: `Presets.email()` builds a fresh map, so a caller cannot corrupt a
        // later caller's questions. Comparing two key sets could not tell the difference.
        Map<String, Question> first = LayaEmail.emailQuestions();
        Map<String, Question> second = LayaEmail.emailQuestions();
        assertEquals(Presets.email().keySet(), first.keySet());
        assertNotSame(first, second, "each call must hand out its own map");
        first.remove(first.keySet().iterator().next());
        assertEquals(Presets.email().keySet(), LayaEmail.emailQuestions().keySet(),
                "mutating a returned map changed what a later caller gets");
        assertEquals(Boolean.TRUE, Fixtures.load("email.json").get("questions_match_presets"),
                "the reference's re-export no longer equals laya.presets.email_questions");
    }

    @Test
    @DisplayName("the categories overload passes the caller's categories through")
    void questionsWithCategories() {
        // Untested until a review pointed it out: `Presets.email(categories)` could have been
        // `Presets.email(null)` -- silently ignoring the argument -- with nothing failing.
        Map<String, String> categories = new LinkedHashMap<>();
        categories.put("refund", "the sender wants money back");
        categories.put("access", "the sender cannot log in");
        Map<String, Question> questions = LayaEmail.emailQuestions(categories);
        assertEquals(Presets.email(categories).keySet(), questions.keySet());
        // The labels, not toString(): `Question` has no toString, so comparing those compared
        // identity hashes and was true however the method behaved. The first version of this
        // test did exactly that and the mutant that drops the argument survived it.
        assertEquals(List.of("refund", "access"), questions.get("category").labels(),
                "the overload returned the DEFAULT questions, so the categories were dropped");
        assertEquals(Presets.email(categories).get("category").renderOptions(),
                questions.get("category").renderOptions());
        assertNotEquals(Presets.email().get("category").labels(),
                questions.get("category").labels(),
                "the caller's categories are indistinguishable from the defaults");
    }

    // ------------------------------------------------------------------ the sign-off rule

    @Test
    @DisplayName("a name may begin with any Lu, Lt or Lo letter, and with nothing else")
    void signoffInitials() {
        // Each of these is a rule UPPER would get wrong; see UnicodeTablesTest for the counts.
        assertTrue(LayaEmail.isEnglishSignoff("Regards, Ana"));
        assertTrue(LayaEmail.isEnglishSignoff("Regards, \u0141ukasz"), "Lu outside Latin-1");
        assertTrue(LayaEmail.isEnglishSignoff("Regards, \u5C71\u7530"), "Lo: a caseless script");
        assertTrue(LayaEmail.isEnglishSignoff("Regards, \u01C5arko"), "Lt: titlecase");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks, \u24B6"),
                "U+24B6 is Uppercase but category So, so it is not a letter");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks, \u017Caneta"), "a lowercase name is prose");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks for the quick reply."));
        assertFalse(LayaEmail.isEnglishSignoff("Regards, Ana Maria Souza Lima"),
                "the tail allows at most three tokens");
    }

    @Test
    @DisplayName("a mark rides on its base, and a mark without one still separates tokens")
    void marksAreDroppedOnlyWithABase() {
        assertEquals("Jose", LayaEmail.dropMarks("Jose\u0301"), "a mark with a base goes");
        assertEquals("Jose", LayaEmail.dropMarks("Jo\u034Fse"),
                "including U+034F, whose canonical combining class is zero");
        assertEquals("\u0301Ana", LayaEmail.dropMarks("\u0301Ana"),
                "a mark opening the string has no base");
        assertEquals("a \u0301b", LayaEmail.dropMarks("a \u0301b"),
                "nor does one following a space");
        // and the consequence, which is the reason the rule exists
        assertTrue(LayaEmail.isEnglishSignoff("Regards, Jose\u0301"));
        assertFalse(LayaEmail.isEnglishSignoff("Regards, \u0301Ana"),
                "a kept mark is not a token opener, so this is not a sign-off");
    }

    // --------------------------------------------------- the two performance-critical equalities

    @Test
    @DisplayName("javaWord is exactly the JDK's own regex word class, over every code point")
    void javaWordEqualsThePattern() throws Exception {
        // It replaced a matcher call per character, which cost more than the rest of the cleaner.
        // The replacement is only safe while it answers identically, so that is asserted rather
        // than argued: this side of boundaryView has to be the JDK's answer, not ours.
        Method javaWord = LayaEmail.class.getDeclaredMethod("javaWord", int.class);
        javaWord.setAccessible(true);
        Pattern word = Pattern.compile("\\w", Pattern.UNICODE_CHARACTER_CLASS);
        int examined = 0;
        for (int cp = 0; cp < 0x110000; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                continue;
            }
            examined++;
            boolean spelled = (Boolean) javaWord.invoke(null, cp);
            boolean pattern = word.matcher(new String(Character.toChars(cp))).matches();
            if (spelled != pattern) {
                assertEquals(pattern, spelled, String.format("U+%04X", cp));
            }
        }
        // Not asserted against a constant the loop itself produces -- that only restates the
        // loop. What matters is that the comparison ran on a lot of code points and on the ones
        // where the two classes are known to disagree.
        final int compared = examined;
        assertTrue(compared > 1_000_000, () -> "only " + compared + " code points examined");
        for (int codePoint : new int[] {0x0301, 0x200C, 0x200D, 0x00BD, 0x2160, 0x11001, 0x10107}) {
            final int cp = codePoint;
            boolean spelled = (Boolean) javaWord.invoke(null, cp);
            assertEquals(word.matcher(new String(Character.toChars(cp))).matches(), spelled,
                    () -> String.format("U+%04X, a code point the two word classes argue about", cp));
        }
    }

    @Test
    @DisplayName("boundaryView rewrites exactly where the two word classes disagree")
    void boundaryViewSubstitutes() {
        assertSame("plain ascii", LayaEmail.boundaryView("plain ascii"),
                "an ASCII-only message is returned without copying");
        assertSame("caf\u00E9 r\u00E9sum\u00E9", LayaEmail.boundaryView("caf\u00E9 r\u00E9sum\u00E9"),
                "ordinary accented letters are word characters in both, so nothing is rewritten");
        // Java says word, Python says not -> U+0000, a non-word character to Java too.
        assertEquals("a\u0000b", LayaEmail.boundaryView("a\u0301b"), "a combining mark");
        assertEquals("a\u0000b", LayaEmail.boundaryView("a\u200Cb"), "ZWNJ, a join control");
        assertEquals("a\u0000b", LayaEmail.boundaryView("a\u200Db"), "ZWJ");
        // Python says word, Java says not -> '0', which no literal in any pattern contains.
        assertEquals("a0b", LayaEmail.boundaryView("a\u00BDb"), "U+00BD is No: Python's word, not Java's");
        // CODE POINT count is preserved, which is the unit a regex quantifier counts in -- and
        // not the char count, which an astral disagreement does change: U+1D16D is two chars and
        // one code point, and its substitute is one of each. Asserting chars passed only because
        // every case in the list was in the BMP.
        for (String text : List.of("a\u0301b", "a\u200Cb", "a\u00BDb", "x\u0301\u00BD\u200Cy",
                                   "a\uD834\uDD6Db", "a\uD800\uDD07b")) {
            String view = LayaEmail.boundaryView(text);
            assertEquals(text.codePointCount(0, text.length()),
                    view.codePointCount(0, view.length()),
                    () -> "boundaryView changed the code point count of " + text);
        }
    }

    @Test
    @DisplayName("neither substitute can appear in a pattern literal")
    void substitutesAreNotLiterals() throws Exception {
        // The whole argument for substituting is that the replacement cannot complete or break a
        // literal. Checked BEHAVIOURALLY: an earlier version scanned the pattern source for a
        // raw NUL, which no pattern can hold -- `appendEscaped` writes every class member as a
        // backslash-u escape -- while the two forms a NUL literal actually takes in a Java
        // regex, backslash-x-zero-zero and backslash-u-four-zeros, both read as ordinary text to
        // `indexOf`. So it could not fail from any plausible edit. (Spelled out rather than
        // written, because the compiler translates a backslash-u escape before it lexes, even
        // inside a comment, and an invalid one there is a compile error.)
        String nul = "\u0000";
        for (Pattern pattern : allPatterns()) {
            assertFalse(pattern.matcher(nul).find(),
                    () -> "this pattern matches a lone NUL, so U+0000 is no longer a safe "
                          + "substitute: " + head(pattern.pattern()));
        }
        // And the digit side, for DISCLAIMER, which is the only pattern the view ever reaches.
        // Quantifier bounds are stripped first: `[^.]{0,60}` is not a digit literal, and a check
        // that did not strip them could not see `covid19` either, because every digit there is
        // next to another one.
        String disclaimer = ((Pattern) patternField("DISCLAIMER")).pattern()
                .replaceAll("\\{\\d+(,\\d+)?\\}", "");
        assertFalse(disclaimer.matches(".*\\d.*"),
                "a digit became a literal in DISCLAIMER; '0' is no longer a safe substitute");
        // the guard above is only meaningful if it would notice one
        assertTrue(("x" + disclaimer + "7").matches(".*\\d.*"),
                "the digit check cannot see a digit, so it proves nothing");
    }

    /** A pattern's opening, for a failure message, without risking its length. */
    private static String head(String source) {
        return source.substring(0, Math.min(40, source.length()));
    }

    @Test
    @DisplayName("every literal in the disclaimer filter is load-bearing")
    void everyHintLiteralIsNecessary() throws Exception {
        Pattern hint = (Pattern) patternField("DISCLAIMER_HINT");
        Pattern disclaimer = (Pattern) patternField("DISCLAIMER");

        // One disclaimer per literal, chosen so that literal is the ONLY one the filter can
        // match it by. That is what makes each row a test of necessity rather than of
        // membership: the previous version of this test checked thirteen examples against the
        // whole filter, every example was satisfied by SOME literal, and five literals could be
        // deleted with the suite green -- leaking "Esta mensagem e sigilosa.", "Destinada
        // exclusivamente ao destinatario." and "Reserve uniquement au destinataire." into the
        // model's input.
        String[][] perLiteral = {
            {"confidenc", "Esta mensagem e confidencial."},
            {"confidenti", "This email is confidential and intended solely for the addressee."},
            {"sigilos", "Esta mensagem e sigilosa."},
            {"privil[e\u00e9]gi", "Ce message est privilegie."},
            {"antes de imprimir", "Antes de imprimir pense no meio ambiente."},
            {"received this", "If you have received this email in error please delete it."},
            {"receb", "Se recebeu esta mensagem por engano, apague."},
            {"recib", "Usted ha recibido este mensaje por error."},
            {"avez re[\u00e7c]u", "Vous avez recu ce message par erreur."},
            {"exclusiv", "Uso exclusivo do destinatario desta mensagem."},
            {"[\u00fau]nicamente", "Unicamente ao destinatario desta mensagem."},
            {"uniquement", "Reserve uniquement au destinataire."},
        };
        List<String> alternatives = List.of(hint.pattern().split("\\|"));
        assertEquals(perLiteral.length, alternatives.size(),
                () -> "the filter has " + alternatives.size() + " literals and this test names "
                      + perLiteral.length + "; a literal with no example is a literal that can "
                      + "be deleted without failing anything");

        for (String[] row : perLiteral) {
            String literal = row[0];
            String example = row[1];
            assertTrue(alternatives.contains(literal),
                    () -> "the filter no longer holds the literal " + literal);
            // the example really is a disclaimer, so leaking it would matter
            assertTrue(disclaimer.matcher(LayaEmail.boundaryView(example)).find(),
                    () -> "this is no longer a disclaimer at all: " + example);
            assertTrue(hint.matcher(example).find(),
                    () -> "the filter rejects a disclaimer the pattern accepts: " + example);
            // and WITHOUT this literal the filter misses it, which is what "necessary" means
            StringBuilder reduced = new StringBuilder();
            for (String alternative : alternatives) {
                if (alternative.equals(literal)) {
                    continue;
                }
                if (reduced.length() > 0) {
                    reduced.append('|');
                }
                reduced.append(alternative);
            }
            Pattern without = Pattern.compile(reduced.toString(),
                    Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE | Pattern.UNIX_LINES);
            assertFalse(without.matcher(example).find(),
                    () -> "deleting " + literal + " from the filter would still catch "
                          + example + ", so this row does not test that literal");
        }
    }

    @Test
    @DisplayName("the filter rejects nothing the pattern accepts, over the whole corpus")
    void hintIsNecessaryOverTheCorpus() throws Exception {
        Pattern hint = (Pattern) patternField("DISCLAIMER_HINT");
        Pattern disclaimer = (Pattern) patternField("DISCLAIMER");
        int fired = 0;
        for (Object entry : rows("cleaned")) {
            String body = (String) asMap(entry).get("body");
            for (String raw : body.split("\n", -1)) {
                final String paragraph = raw;
                if (disclaimer.matcher(LayaEmail.boundaryView(paragraph)).find()) {
                    fired++;
                    assertTrue(hint.matcher(paragraph).find(),
                            () -> "the filter rejected a paragraph the pattern accepts: "
                                  + paragraph);
                }
            }
        }
        // A secondary check. `everyHintLiteralIsNecessary` is the coverage gate -- it proves each
        // literal is load-bearing -- and this one proves the implication holds on real bodies as
        // well as on the twelve constructed ones.
        final int hits = fired;
        assertEquals(11, hits,
                "the corpus's disclaimer coverage changed; this is a count, so change it "
                + "knowingly rather than lowering it");
    }

    // ------------------------------------------------------------------------- the state

    @Test
    @DisplayName("a null or empty sender adds no field, and a null extra is dropped")
    void stateOmissions() {
        assertFalse(LayaEmail.emailState("s", "b", null).containsKey("from"));
        assertFalse(LayaEmail.emailState("s", "b", "").containsKey("from"),
                "Python's `if sender:` is false for an empty string");
        assertTrue(LayaEmail.emailState("s", "b", "a@b.c").containsKey("from"));
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("kept", 1);
        extra.put("dropped", null);
        Map<String, Object> state = LayaEmail.emailState("s", "b", null, true, 3000, extra);
        assertTrue(state.containsKey("kept"));
        assertFalse(state.containsKey("dropped"), "a null extra is dropped, as the reference drops it");
        assertEquals(List.of("subject", "body", "kept"), new ArrayList<>(state.keySet()));
    }

    @Test
    @DisplayName("a null subject or body is empty, not an exception")
    void stateNulls() {
        Map<String, Object> state = LayaEmail.emailState(null, null);
        assertEquals("", state.get("subject"));
        assertEquals("", state.get("body"));
        assertEquals("", LayaEmail.cleanEmailBody(null));
    }

    @Test
    @DisplayName("a negative budget drops the tail, as Python's slice does")
    void negativeBudget() {
        // This test used to assert the empty string for -5, which is what the port returned and
        // is NOT what the reference returns. Python slices twice -- at `max_chars * 4` and at
        // `max_chars` -- and a negative index drops the LAST |n| code points.
        //
        // Both slices apply, which is why a short body still empties: "a message" is 9 code
        // points, `[: -12]` already takes everything. A body only survives a budget of -n if it
        // is longer than 5n code points. Every expectation below was read off the reference, not
        // reasoned about -- the first version of this test guessed "a mess" and was wrong.
        assertEquals("", LayaEmail.cleanEmailBody("a message", 0), "zero really is empty");
        assertEquals("a", LayaEmail.cleanEmailBody("a message", 1));
        assertEquals("", LayaEmail.cleanEmailBody("a message", -3),
                "9 code points less 12 is nothing, before the second slice is reached");
        assertEquals("", LayaEmail.cleanEmailBody("a message", -99));
        assertEquals("I cannot log in a", LayaEmail.cleanEmailBody(
                "I cannot log in and need a password reset.", -5),
                "41 code points: the first slice keeps 21, the second drops 5 of those");
        // code points, not chars, on the negative side too
        assertEquals("\uD835\uDC00".repeat(10), LayaEmail.cleanEmailBody(
                "\uD835\uDC00".repeat(20), -2),
                "20 astral characters: the first slice keeps 12, the second drops 2");
        assertEquals("", LayaEmail.cleanEmailBody("\uD835\uDC00".repeat(3), -1),
                "and a short astral body empties the same way a short ASCII one does");
    }

    @Test
    @DisplayName("an extra may not name a parameter of emailState")
    void extraMayNotShadowAParameter() {
        // `body` is the one that bites: the state is model input, so a silently accepted `body`
        // extra ships an UNCLEANED body -- no quoted history removed, no signature cut, no
        // budget. The reference cannot do this at all; `email_state(**{"body": x})` raises.
        for (String reserved : List.of("subject", "body", "sender", "clean", "max_chars")) {
            Map<String, Object> extra = new LinkedHashMap<>();
            extra.put(reserved, "shadowed");
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                    () -> LayaEmail.emailState("S", "b", null, true, 3000, extra),
                    () -> "an extra named " + reserved + " was accepted into the state");
            assertTrue(refused.getMessage().contains(reserved),
                    () -> "the refusal does not name the key: " + refused.getMessage());
        }
        // `from` is NOT reserved -- the reference lets it through, and it replaces the sender in
        // place rather than being appended.
        Map<String, Object> from = new LinkedHashMap<>();
        from.put("from", "override@x.com");
        Map<String, Object> state = LayaEmail.emailState("S", "b", "ana@x.com", true, 3000, from);
        assertEquals(List.of("subject", "body", "from"), new ArrayList<>(state.keySet()));
        assertEquals("override@x.com", state.get("from"));
    }

    @Test
    @DisplayName("the strip used on a subject is Python's, not Java's trim")
    void subjectStripIsPythons() {
        // U+00A0 and U+2007 are whitespace to str.strip() and not to String.trim(). The same gap
        // sat in the router's checkpoint names and routed a caller to the wrong model.
        assertEquals("Refund", LayaEmail.emailState("\u00A0Refund\u2007", "b").get("subject"));
        assertTrue(UnicodeTables.isSpace(0x00a0) && UnicodeTables.isSpace(0x2007),
                "the premise of the assertion above");
    }

    // ------------------------------------------------------------------------- helpers

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }

    private static Object patternField(String name) throws Exception {
        Class<?> patterns = Class.forName("com.convaiinnovations.laya.LayaEmail$Patterns");
        Field field = patterns.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(null);
    }

    private static List<Pattern> allPatterns() throws Exception {
        Class<?> patterns = Class.forName("com.convaiinnovations.laya.LayaEmail$Patterns");
        List<Pattern> out = new ArrayList<>();
        for (Field field : patterns.getDeclaredFields()) {
            field.setAccessible(true);
            Object value;
            try {
                value = field.get(null);
            } catch (IllegalAccessException problem) {
                continue;
            }
            if (value instanceof Pattern) {
                out.add((Pattern) value);
            } else if (value instanceof List<?>) {
                for (Object element : (List<?>) value) {
                    if (element instanceof Pattern) {
                        out.add((Pattern) element);
                    }
                }
            }
        }
        assertTrue(out.size() >= 15, () -> "only " + out.size() + " patterns found by reflection");
        return out;
    }
}
