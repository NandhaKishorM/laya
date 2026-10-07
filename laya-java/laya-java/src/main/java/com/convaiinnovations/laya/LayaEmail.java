package com.convaiinnovations.laya;

import com.convaiinnovations.laya.lang.UnicodeTables;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * {@code laya.email}: strip quoted history, signatures and disclaimers from a mail body, and build
 * a state from a message.
 *
 * <p>The markers cover English, Portuguese, Spanish and French mail clients, which is the set the
 * reference covers and for the reason it gives: the router already sends Portuguese, Spanish and
 * French states to the multilingual checkpoint, so an English-only cleaner left Gmail's
 * {@code Em ... escreveu:}, Outlook's {@code -----Mensagem original-----}, the
 * {@code Atenciosamente} sign-off and the confidentiality footer in the input, and the quoted
 * history -- often a different request -- weighed on the answer as much as the new message.
 *
 * <h2>Why this is scanners and explicit classes rather than the reference's patterns</h2>
 *
 * <p>Every pattern here was translated, not transcribed, because four of Python's regex constructs
 * mean something else in {@code java.util.regex}:
 *
 * <ul>
 *   <li>{@code \s}. Python's is 29 code points -- exactly {@code str.isspace()}, verified over all
 *       of Unicode. Java's is 6 by default, and 25 under {@code UNICODE_CHARACTER_CLASS}, which is
 *       the White_Space property and so excludes U+001C to U+001F. Those four are Cc, not
 *       White_Space, and Python treats them as space. The class is built below from the same
 *       table the digest gates, so it cannot drift from the reference.
 *   <li>{@code \d}. Python's is the Nd category, 680 code points at Unicode 15. Java's is ASCII
 *       by default and, under {@code UNICODE_CHARACTER_CLASS}, follows THE JDK'S Unicode version
 *       -- which is the defect that shipped in this port's pre-tokenizer, where the same jar gave
 *       different token ids on different JDKs. Built from the table for the same reason.
 *   <li>{@code \b}. A word boundary is defined in terms of {@code \w}, and Java's {@code \w}
 *       admits combining marks and join controls where Python's does not, so the two disagree
 *       about whether a boundary exists after {@code confidential} followed by U+0301. Each
 *       {@code \b} is therefore a lookaround over the Python word class.
 *   <li>{@code .} and {@code $}. Python's {@code .} excludes only {@code \n}; Java's excludes
 *       every line terminator, including U+0085 and U+2028, and its {@code $} looks for any of
 *       them too. {@code UNIX_LINES} makes both mean what Python means, so every pattern carries
 *       it.
 * </ul>
 *
 * <p>Case folding is left to {@code CASE_INSENSITIVE | UNICODE_CASE}. The accented literals in the
 * markers are Latin-1 Supplement and Latin Extended-A, whose case mappings have not moved in any
 * Unicode version, and the fixture is run on two JDKs with different Unicode versions to keep that
 * an observation rather than an assumption.
 */
public final class LayaEmail {

    private LayaEmail() {
    }

    // ------------------------------------------------------- Python's character classes

    /**
     * {@code \s}, {@code \S} and {@code \d} from a single pass over every code point.
     *
     * <p>Built from {@link UnicodeTables}, so these classes and the rest of the port read the
     * same Unicode -- the one recorded from CPython -- and a table change moves both together.
     * Writing the ranges out by hand would be a second copy that can disagree with the first,
     * which is how this port's pre-tokenizer came to classify 9,917 code points by the JDK's
     * Unicode instead of the reference's.
     *
     * <p>One pass rather than one per class, because this walk is the whole cost of the first
     * call into {@link Patterns}. The two space classes come from the same predicate, so the
     * negated one costs nothing extra: a range of spaces is a gap in the complement and the other
     * way round.
     */
    private static String[] classes() {
        StringBuilder space = new StringBuilder("[");
        StringBuilder digit = new StringBuilder("[");
        int cp = 0;
        while (cp <= 0x10FFFF) {
            boolean isSpace = UnicodeTables.isSpace(cp);
            boolean isDigit = UnicodeTables.isDigit(cp);
            if (!isSpace && !isDigit) {
                cp++;
                continue;
            }
            int start = cp;
            if (isSpace) {
                while (cp + 1 <= 0x10FFFF && UnicodeTables.isSpace(cp + 1)) {
                    cp++;
                }
                appendRange(space, start, cp);
            } else {
                while (cp + 1 <= 0x10FFFF && UnicodeTables.isDigit(cp + 1)) {
                    cp++;
                }
                appendRange(digit, start, cp);
            }
            cp++;
        }
        // An empty class is `[]` or `[^]`, which Java rejects -- and it would arrive as
        // ExceptionInInitializerError on the first call and NoClassDefFoundError on every one
        // after, naming neither the table nor this method. A table that lost its contents is a
        // generator regression, so it says so.
        if (space.length() == 1 || digit.length() == 1) {
            throw new IllegalStateException(
                    "UnicodeTables gave an empty class: isSpace matched "
                    + (space.length() == 1 ? "no" : "some") + " code points and isDigit matched "
                    + (digit.length() == 1 ? "no" : "some")
                    + "; regenerate with laya-java/scripts/gen_unicode_tables.py");
        }
        String spaces = space.append(']').toString();
        return new String[] {spaces, "[^" + spaces.substring(1), digit.append(']').toString()};
    }

    /** One range of a character class, or one code point when the range holds one. */
    private static void appendRange(StringBuilder out, int from, int to) {
        appendEscaped(out, from);
        if (to > from) {
            out.append('-');
            appendEscaped(out, to);
        }
    }

    /** A code point as a regex escape, so no member of a class is ever a metacharacter. */
    private static void appendEscaped(StringBuilder out, int codePoint) {
        if (codePoint <= 0xFFFF) {
            out.append(String.format("\\u%04X", codePoint));
        } else {
            out.append(String.format("\\x{%X}", codePoint));
        }
    }

    /**
     * {@code re.I} plus the two flags that make Java's syntax mean Python's.
     *
     * <p>{@code UNIX_LINES} is not optional: without it Java's {@code .} excludes every line
     * terminator -- U+0085 and U+2028 among them -- where Python's excludes only {@code \n}, and
     * Java's {@code $} looks for any of them too.
     */
    private static Pattern compile(String regex) {
        return Pattern.compile(regex, Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE
                | Pattern.UNIX_LINES);
    }

    /** For the one pattern the reference does not pass {@code re.I}, scoping it with inline flags. */
    private static Pattern compileCased(String regex) {
        return Pattern.compile(regex, Pattern.UNIX_LINES);
    }

    /**
     * {@link #compile} plus {@code UNICODE_CHARACTER_CLASS}, for the one pattern that uses
     * {@code \b}.
     *
     * <p>The flag is what makes {@code \b} read Java's Unicode word class; without it {@code \b}
     * reads {@code [a-zA-Z0-9_]}, and then {@link #boundaryView} -- which reconciles Python's
     * class with the Unicode one -- is reconciling against a class nothing consults. The symptom
     * was a false positive: "Esta mensagem" followed by an astral letter has no word boundary to
     * Python, an ASCII {@code \b} found one anyway, and a paragraph the reference keeps was
     * deleted whole.
     *
     * <p>Only {@code \b} changes meaning here. This pattern has no {@code \s}, {@code \d} or
     * {@code \w} of its own -- those are the classes built from the tables, spelled out -- so the
     * flag cannot reach anything else.
     */
    private static Pattern compileWithBoundaries(String regex) {
        return Pattern.compile(regex, Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE
                | Pattern.UNIX_LINES | Pattern.UNICODE_CHARACTER_CLASS);
    }

    /** The default body budget, which {@link #emailState} passes through. */
    public static final int DEFAULT_MAX_CHARS = 3000;

    /**
     * The patterns, built on first use.
     *
     * <p>A holder rather than fields of {@code LayaEmail}, because this costs about 55 ms --
     * one walk of all 1,114,112 code points to build the character classes, plus compiling the
     * patterns. That is a fair price for classes that cannot disagree with CPython, but only for
     * a caller who cleans an email: paid in {@code LayaEmail}'s own initialiser it would land on
     * every caller of the package, including one that only ever opens an {@link Agent}.
     */
    private static final class Patterns {

        /**
         * {@code \s}, {@code \S} and {@code \d}, built in ONE walk of Unicode.
         *
         * <p>One walk of 1,114,112 code points rather than one per class. The walk itself is
         * 15.8 ms warm; the first {@code cleanEmailBody} is 52 to 57 ms, most of it compiling
         * the patterns. It was about 160 ms when a fourth, unused class was built here too.
         */
        private static final String[] CLASSES = classes();

        /** Python's {@code \s}: {@code str.isspace()}, 29 code points. */
        static final String S = CLASSES[0];
        /** Python's {@code \S}. */
        static final String NS = CLASSES[1];
        /** Python's {@code \d}: the Nd category. */
        static final String D = CLASSES[2];
        /**
         * {@code \b}, Java's own -- applied to a string {@link #boundaryView} has brought into
         * agreement with Python's word class, so the boundary it finds is Python's boundary.
         *
         * <p>The exact alternative, a lookaround over Python's own word class, was written
         * first and measured:
         * it inlines a 10,622-character class at 26 boundary sites, which is a 277,309-character
         * pattern and 1,742 us per paragraph against 7.8 us here -- 220 times the cost, and the
         * whole cost of cleaning a message. The two agreed on all 20,000 paragraphs of the sweep
         * corpus, which is exactly why the lookaround is not kept as the "safe" option:
         * agreement on a corpus is not equivalence, and Java's {@code \w} follows THE JDK'S
         * Unicode version. Deciding the boundary from the recorded table instead is what makes
         * the answer the same on every JDK, which is the property this port lost once already.
         */
        static final String BB = "\\b";
        static final String BA = "\\b";


        static final List<Pattern> QUOTE_HEADERS = List.of(
                compile("^" + S + "*On .{0,300}wrote:" + S + "*$"),
                // "Em resposta ao que você escreveu:" is body text; a client's attribution always
                // carries a date.
                compile("^" + S + "*Em (?=.*" + D + ").{0,300}escreveu:" + S + "*$"),
                compile("^" + S + "*El (?=.*" + D + ").{0,300}escribi[óo]:" + S + "*$"),
                // Gmail's French attribution carries a date too; a bare "Le rapport que vous avez
                // écrit :" is body text. French typography puts a space before the colon.
                compile("^" + S + "*Le (?=.*" + D + ").{0,300}a [eé]crit" + S + "*:" + S + "*$"),
                compile("^" + S + "*-{2,}" + S + "*(Original|Forwarded) Message" + S + "*-{2,}"),
                compile("^" + S + "*-{2,}" + S + "*(Mensagem (original|encaminhada)"
                        + "|Mensaje (original|reenviado)|Message d'origine)" + S + "*-{2,}"),
                compile("^" + S + "*_{8,}" + S + "*$"),
                // `From:` opens ordinary prose too ("From: my side the integration works, but
                // please refund..."), and a reply header always carries the sender, so the header
                // is only recognised when an address follows. A bare `From: Name` is caught by
                // HEADER_FROM_NAME/HEADER_NEXT instead, which need the header's own `Sent:`.
                compile("^" + S + "*From:" + S + ".*[@<]"),
                // `De:` also opens ordinary Portuguese and Spanish lines ("De: 10/09 a 15/09").
                // French Outlook writes `De :` with a space.
                compile("^" + S + "*De" + S + "*:" + S + ".*[@<]"));

        // Gmail wraps a long attribution line, leaving `fulano@x.com> escreveu:` alone on the
        // next line. That tail cuts too, and takes the `On/Em/El/Le ...` head it belongs to with
        // it. The French tail keeps its spaced colon.
        static final Pattern ATTRIBUTION_TAIL = compile(
                // `\S++` POSSESSIVE, not greedy. `.{0,120}` offers 121 starting points, and at
                // each one a greedy `\S+` runs to the end of the line and then gives every
                // character back, because the `\s+` after it can never match where a `\S` was
                // just surrendered -- the two classes are complements. The result was 121 full
                // scans of the line: measured 55 ms on a 4,000,000-character line at the default
                // budget and 4.8 s at `maxChars` of 300,000, against 0.018 ms for an ordinary
                // message. Possessive forbids the backtracking that could never have succeeded.
                "^.{0,120}" + NS + "@" + NS + "++" + S + "+(wrote|escreveu|escribi[óo]|a [eé]crit)"
                + S + "*:" + S + "*$");
        static final Pattern ATTRIBUTION_HEAD = compile("^" + S + "*(On|Em|El|Le) (?=.*" + D + ")");

        // Exchange often leaves the address out of Outlook's reply header ("De: Maria Souza"), so
        // a bare `De:` only cuts when the header's own `Enviado:` line, or a dated `Data:`/
        // `Fecha:` line, follows it. `Para:` is not enough: "De: 10/09 / Para: 15/09" is how a
        // leave request reads. The same is true of a bare English `From: Maria Souza`.
        static final Pattern HEADER_FROM_NAME = compile("^" + S + "*(De|From)" + S + "*:" + S + "+" + NS);
        static final Pattern HEADER_NEXT = compile(
                "^" + S + "*(Enviad[oa]( em| el)?:" + S + "|Envoy[ée]( le)?" + S + "*:" + S
                + "|Sent:" + S + "|(Data|Fecha|Date):" + S + ".*" + D + "{4})");

        // A closing is the closing word plus punctuation and at most a name. The closing words are
        // matched case-insensitively and the name is not, so the flag is scoped in the reference
        // with `(?i:...)`; `(?iu:...)` is the same scoping with Unicode case folding, which is
        // what Python's `re.I` on a str pattern does.
        static final Pattern SIGNOFF_HEAD = compileCased(
                "^" + S + "*(?iu:best|kind|warmest|warm|many thanks|thanks|thank you|regards"
                + "|cheers|sincerely)"
                + "(?iu:" + S + "+(?:and|&)" + S + "+regards|" + S
                + "+(?:regards|wishes|again|in advance|a lot|so much|very much))?");

        static final Pattern SIGNOFF_DASHES = compile("^" + S + "*--" + S + "*$");

        // Portuguese, Spanish and French sign-offs match only on their own: "Obrigado pelo
        // retorno, mas ..." is a request, not a signature, so unlike the English marker no
        // trailing words are allowed.
        static final Pattern SIGNOFF_ROMANCE = compile(
                "^" + S + "*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]"
                + "|(muito )?obrigad[oa]s?( desde já| pela atenção)?"
                + "|(com os melhores )?cumprimentos|saudações|"
                + "(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?|"
                + "(bien )?cordialement|salutations( distinguées)?|bien à vous|merci( d'avance)?|"
                + "bonne journée)[" + inner(S) + ",!.]*$");

        // Mobile and mail-app footers. Only a line that is nothing BUT the footer matches --
        // "Enviado do meu celular o comprovante ontem." is a request -- and such a line may run
        // to 60 characters, since Samsung's default is longer than a sign-off's 40.
        private static final String DEVICE =
                "iphone|ipad|android|ios|mobile|celular|telemóvel|móvil|galaxy|smartphone|samsung"
                + "|tablet|outlook|yahoo|mail|e-?mail|gmail|windows";
        static final Pattern DEVICE_FOOTER = compile(
                "^" + S + "*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?"
                + "|sent from( my)?|envoy[ée] (depuis|de)( mon| ma| mes)?)"
                + " (" + DEVICE + ")( (" + DEVICE + "|para|for|no|na|" + D + "+|phone|device|pro"
                + "|max|mini|plus|using [a-z][a-z0-9_.+-]*))*"
                + "|(obter o|get) outlook (para|for) (ios|android))[" + inner(S) + ".!]*$");

        static final Pattern DISCLAIMER = compileWithBoundaries(
                // English: tied to a disclaimer noun AND a disclaimer tail, the way the
                // Portuguese branches are. The bare word matched any sentence that merely
                // mentioned it, so "Is this confidential?" was deleted whole. `[^.]` rather than
                // `[^.\n]`: a footer wraps, so "are\nconfidential" must still match.
                "(" + BB + "(e-?mail|message|information|communication|transmission|contents?)"
                + BA + "[^.]{0,60}" + BB + "confidential" + BA + "[^.]{0,60}" + BB
                + "(intended|solely|addressee|recipient|privileged|disclos|unauthori[sz]ed)|"
                + BB + "confidential" + BA + "[^.]{0,60}" + BB + "(and (may|is) (also )?privileged)|"
                + "if you (have )?received this (e-?mail|message) in error|"
                // Portuguese and Spanish: tied to "this message/e-mail" rather than the bare word
                // `confidencial`, which a sender's own request uses just as often.
                + BB + "(esta|este) (mensagem|e-?mail|mensaje|correo)" + BA + "[^.]{0,80}"
                + "(confidencia|sigilos|privilegiad)|"
                + BB + "(uso exclusivo|exclusivamente|únicamente|unicamente)" + BA + "[^.]{0,30}"
                + "(destinatári|destinatari|pessoa|persona|entidade|entidad)|"
                + BB + "(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)" + BA
                + "[^.]{0,20} por (engano|erro)|"
                + BB + "(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)" + BA
                + "[^.]{0,20} por error|"
                // the "think before printing" footer, tied to its environmental ending rather
                // than to `antes de imprimir`, which a request uses too.
                + BB + "antes de imprimir" + BA + "[^.]{0,100}"
                + "(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|"
                + BB + "(meio|medio) ambiente" + BA + "[^.]{0,30}antes de imprimir|"
                // French: tied to "ce message/cet e-mail" rather than the bare word
                // `confidentiel`, which a sender's own request uses just as often.
                + BB + "(ce|cet|cette) (message|e-?mail|mail|courriel)" + BA + "[^.]{0,80}"
                + "(confidentiel|privil[eé]gi)|"
                + BB + "avez re[çc]u (ce|cet|cette) (message|e-?mail|mail)" + BA
                + "[^.]{0,20} par erreur|"
                + BB + "(usage exclusif|exclusivement|uniquement)" + BA + "[^.]{0,30}destinataire)");

        /**
         * A necessary condition for DISCLAIMER, checked first.
         *
         * <p>Each of DISCLAIMER's twelve top-level alternations requires at least one of these
         * literals, so a paragraph holding none of them cannot match and does not have to be
         * scanned by the full pattern. Eleven of the twelve carry word boundaries -- 26 of them
         * between them; the "received this ... in error" branch carries none.
         *
         * <p>It is worth far less than it was. When `\b` was a lookaround over a
         * 10,622-character class this filter took DISCLAIMER from 4.9 ms per case to 0.018 ms and
         * was the difference between usable and not. Once `\b` became Java's own, DISCLAIMER
         * stopped being expensive, and this is now an ordinary literal alternation of the same
         * shape as the pattern it guards -- measured at parity with it on a paragraph holding no
         * literal, and worth 1.16x on the committed corpus and 1.4x on ordinary paragraphs. It
         * is kept for that, and because a cheap necessary condition in front of a complicated
         * sufficient one is the right shape, not because it is 40 times anything.
         *
         * <p>This is a filter, never a decision: a hit still runs DISCLAIMER, so a false positive
         * costs only time. What would be a defect is a false NEGATIVE, so the literals are
         * matched case-insensitively with Unicode folding, exactly as `re.I` on a str pattern
         * folds -- it folds more than lowercasing does, U+017F LATIN SMALL LETTER LONG S against
         * `s` among it, and a filter that lowered the text instead would reject a paragraph the
         * pattern accepts. DISCLAIMER additionally carries UNICODE_CHARACTER_CLASS, which it
         * needs for `\b` and this has no use for, having no class of its own.
         */
        static final Pattern DISCLAIMER_HINT = compile(
                // Each literal is the SHORTEST form that covers its branch, and every one of
                // them is load-bearing: `EmailTest` gives each a disclaimer whose only hint
                // literal is that one, and asserts that removing it stops the filter matching.
                // An earlier version listed the branches' own words -- `confidential`,
                // `recebeu|recebido|receber`, `uso exclusivo|exclusivamente|usage exclusif` --
                // and five of those were deletable with the suite still green, because every
                // example that reached the filter satisfied it through some OTHER literal.
                //
                // Shortening is always safe in this direction. The filter has to be a NECESSARY
                // condition for DISCLAIMER and nothing more, so a literal that matches more than
                // its branch costs time on a paragraph that then fails the real pattern; a
                // literal that matches less lets a disclaimer through to the model.
                "confidenc|confidenti|sigilos|privil[eé]gi"
                + "|antes de imprimir|received this"
                + "|receb|recib|avez re[çc]u"
                + "|exclusiv|[úu]nicamente|uniquement");

        static final Pattern SENTENCE = compile("(?<=[.!?])" + S + "+");
        static final Pattern BLANK_LINE = compile("\n" + S + "*\n");
        static final Pattern SPACES_AND_TABS = compile("[ \t]+");

        private Patterns() {
        }
    }

    /**
     * The same text, with every code point on which Java's {@code \w} and Python's disagree
     * replaced by one they agree about.
     *
     * <p>{@code \b} is defined in terms of {@code \w}, so a boundary is Python's boundary
     * exactly when the two classes agree on the characters either side of it. They do not agree
     * everywhere: Java's admits the combining marks and join controls Python's excludes, and
     * Python's admits the letter- and other-numbers (U+2160 ROMAN NUMERAL ONE, U+00BD VULGAR
     * FRACTION ONE HALF) that Java's, being alphabetic plus Nd plus Pc, excludes. Which code
     * points fall in each set is a property of the RUNNING JDK, so it is probed here rather than
     * tabulated -- a table would be right on one JDK and wrong on the next, which is the defect
     * this port shipped in its pre-tokenizer.
     *
     * <p>Two substitutes, chosen so a replacement cannot change which literal matches:
     * <ul>
     *   <li>Java says word, Python says not: U+0000, which Java also calls a non-word character.
     *   <li>Python says word, Java says not: {@code '0'}, which Java calls a word character.
     *       DISCLAIMER is the only pattern matched against the view, and no literal in it
     *       contains a digit, so a digit can neither complete nor break one. (Other patterns do
     *       hold digits -- DEVICE_FOOTER's {@code [a-z0-9_.+-]}, HEADER_NEXT's four-digit year --
     *       and none of them ever sees this string.)
     * </ul>
     * Both substitutes are matched by the {@code [^.]} windows the patterns measure, so a window
     * counts the same number of characters as it does in the original.
     *
     * <p>ASCII cannot disagree -- both classes are the same there -- so an ASCII-only message,
     * which is most of them, returns the original string without copying it.
     */
    static String boundaryView(String text) {
        int firstDisagreement = -1;
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            if (cp > 0x7F && javaWord(cp) != UnicodeTables.isPythonWordChar(cp)) {
                firstDisagreement = i;
                break;
            }
            i += Character.charCount(cp);
        }
        if (firstDisagreement < 0) {
            return text;
        }
        StringBuilder out = new StringBuilder(text.length());
        out.append(text, 0, firstDisagreement);
        i = firstDisagreement;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            if (cp > 0x7F && javaWord(cp) != UnicodeTables.isPythonWordChar(cp)) {
                out.append(UnicodeTables.isPythonWordChar(cp) ? '0' : NUL_PLACEHOLDER);
            } else {
                out.appendCodePoint(cp);
            }
            i += Character.charCount(cp);
        }
        return out.toString();
    }

    /** Java calls this a non-word character, and no pattern here holds it as a literal. */
    private static final char NUL_PLACEHOLDER = '\u0000';

    /**
     * Whether the RUNNING JDK calls this code point a word character.
     *
     * <p>Spelled out from the definition {@code java.util.regex.Pattern} documents for
     * {@code \w} under {@code UNICODE_CHARACTER_CLASS} -- alphabetic, the three mark categories,
     * decimal digits, connector punctuation and the join controls -- rather than asked of a
     * a matcher on Java's own {@code \\w}, which allocates a string and a matcher per character
     * and cost more than the rest of the cleaner put together. Every term reads the JDK's own
     * Unicode, so this stays whatever that JDK's {@code \w} is, which is the point: the view is
     * built by comparing the JDK against the recorded table, so this side has to be the JDK's
     * answer and not ours. {@code EmailTest} compares the two forms over all 1,114,112 code
     * points, so the spelled-out version cannot drift from the pattern it replaces.
     */
    private static boolean javaWord(int codePoint) {
        if (Character.isAlphabetic(codePoint) || Character.isDigit(codePoint)) {
            return true;
        }
        switch (Character.getType(codePoint)) {
            case Character.NON_SPACING_MARK:
            case Character.ENCLOSING_MARK:
            case Character.COMBINING_SPACING_MARK:
            case Character.DECIMAL_DIGIT_NUMBER:
            case Character.CONNECTOR_PUNCTUATION:
                return true;
            default:
                return codePoint == 0x200C || codePoint == 0x200D;   // the join controls
        }
    }

    /** A character class's contents, for nesting one inside a larger class. */
    private static String inner(String characterClass) {
        return characterClass.substring(1, characterClass.length() - 1);
    }

    // ------------------------------------------------------------------- the sign-off rule

    /**
     * Python's {@code _drop_marks}: remove combining marks that ride on a preceding character.
     *
     * <p>Only a mark with a base goes. One that opens the string, or follows a space, has nothing
     * to ride on and stays, so it still separates tokens -- which is what makes the port's leading
     * {@code \p{Lu}\p{Lt}\p{Lo}} reject it.
     */
    static String dropMarks(String text) {
        StringBuilder kept = new StringBuilder(text.length());
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            int width = Character.charCount(cp);
            boolean hasBase = kept.length() > 0;
            boolean baseIsSpace = hasBase
                    && UnicodeTables.isSpace(kept.codePointBefore(kept.length()));
            if (!(UnicodeTables.isMark(cp) && hasBase && !baseIsSpace)) {
                kept.appendCodePoint(cp);
            }
            i += width;
        }
        return kept.toString();
    }

    /**
     * Python's {@code _SIGNOFF_TAIL} and {@code _SIGNOFF_TOKEN}, as one scan.
     *
     * <p>{@code ^[\s,;:!.]*(?:[^\W\d_][\w'-]*[\s,.]*){0,3}$}, then every token's first letter must
     * be Lu, Lt or Lo. A scanner rather than a pattern because the two classes are Python's: the
     * token opener is {@code [^\W\d_]} and the rest is {@code \w}, and Java's {@code \w} is a
     * different set -- it admits the combining marks this has just dropped and the join controls
     * that must keep separating tokens.
     *
     * <p>The repetition is greedy and each turn consumes at least one character, so taking as many
     * as possible and then requiring the end of input is what the pattern's backtracking amounts
     * to: with a fourth token present no smaller count reaches the end either.
     */
    private static boolean tailIsNameOnly(String tail) {
        int i = 0;
        int length = tail.length();
        i = skip(tail, i, ",;:!.");                       // [\s,;:!.]*
        for (int token = 0; token < 3 && i < length; token++) {
            int cp = tail.codePointAt(i);
            if (!UnicodeTables.isWordChar(cp)) {          // [^\W\d_]
                break;
            }
            // Every token's first letter carries the rule the whole scan exists for.
            if (!UnicodeTables.isInitial(cp)) {
                return false;
            }
            i += Character.charCount(cp);
            while (i < length) {                          // [\w'-]*
                int next = tail.codePointAt(i);
                if (!UnicodeTables.isPythonWordChar(next) && next != '\'' && next != '-') {
                    break;
                }
                i += Character.charCount(next);
            }
            i = skip(tail, i, ",.");                      // [\s,.]*
        }
        return i == length;
    }

    /** Advance over Python whitespace and any of {@code extra}. */
    private static int skip(String text, int from, String extra) {
        int i = from;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            if (!UnicodeTables.isSpace(cp) && extra.indexOf(cp) < 0) {
                return i;
            }
            i += Character.charCount(cp);
        }
        return i;
    }

    /** Python's {@code _is_english_signoff}. */
    static boolean isEnglishSignoff(String line) {
        Matcher head = Patterns.SIGNOFF_HEAD.matcher(line);
        if (!head.lookingAt()) {
            return false;
        }
        return tailIsNameOnly(dropMarks(line.substring(head.end())));
    }

    /** Whether any signature marker claims this line. */
    private static boolean isSignature(String line) {
        return Patterns.SIGNOFF_DASHES.matcher(line).lookingAt()
                || isEnglishSignoff(line)
                || Patterns.SIGNOFF_ROMANCE.matcher(line).lookingAt();
    }

    // -------------------------------------------------------------------- disclaimers

    /**
     * Python's {@code _starts_new_sentence}: the first letter is uppercase.
     *
     * <p>A line in an uncased script never starts a new piece, so wrapped boilerplate in those
     * scripts still drops whole.
     */
    private static boolean startsNewSentence(String line) {
        int i = 0;
        while (i < line.length()) {
            int cp = line.codePointAt(i);
            if (UnicodeTables.isAlpha(cp)) {
                return UnicodeTables.isUpper(cp);
            }
            i += Character.charCount(cp);
        }
        return false;
    }

    /** Python's {@code _split_fused_lines}. */
    private static List<String> splitFusedLines(String sentence) {
        if (sentence.indexOf('\n') < 0) {
            return List.of(sentence);
        }
        List<String> pieces = new ArrayList<>();
        StringBuilder buffer = new StringBuilder();
        for (String raw : sentence.split("\n", -1)) {
            String line = UnicodeTables.strip(raw);
            if (line.isEmpty()) {
                continue;
            }
            if (buffer.length() > 0 && startsNewSentence(line)) {
                pieces.add(buffer.toString());
                buffer.setLength(0);
                buffer.append(line);
            } else {
                if (buffer.length() > 0) {
                    buffer.append(' ');
                }
                buffer.append(line);
            }
        }
        if (buffer.length() > 0) {
            pieces.add(buffer.toString());
        }
        return pieces;
    }

    /**
     * Python's {@code _strip_disclaimer}.
     *
     * <p>A paragraph goes whole only when EVERY sentence in it is boilerplate; otherwise only the
     * boilerplate sentences go. A footer that runs on without a blank line used to take the
     * sender's actual request with it, which is worse than leaving one boilerplate line behind.
     */
    /** DISCLAIMER, behind its filter. */
    private static boolean isDisclaimer(String text) {
        return Patterns.DISCLAIMER_HINT.matcher(text).find()
                && Patterns.DISCLAIMER.matcher(boundaryView(text)).find();
    }

    private static String stripDisclaimer(String paragraph) {
        if (!isDisclaimer(paragraph)) {
            return paragraph;               // nothing to do: keep the original line structure
        }
        List<String> pieces = new ArrayList<>();
        for (String part : Patterns.SENTENCE.split(paragraph, -1)) {
            String trimmed = UnicodeTables.strip(part);
            if (trimmed.isEmpty()) {
                continue;
            }
            if (isDisclaimer(trimmed)) {
                pieces.addAll(splitFusedLines(trimmed));
            } else {
                pieces.add(trimmed);
            }
        }
        StringBuilder out = new StringBuilder();
        for (String piece : pieces) {
            if (isDisclaimer(piece)) {
                continue;
            }
            if (out.length() > 0) {
                out.append(' ');
            }
            out.append(piece);
        }
        return out.toString();
    }

    // -------------------------------------------------------------------- the cleaner

    /** {@link #cleanEmailBody(String, int)} with the default budget. */
    public static String cleanEmailBody(String body) {
        return cleanEmailBody(body, DEFAULT_MAX_CHARS);
    }

    /**
     * Remove quoted history, signatures and disclaimers, and cut the result to {@code maxChars}.
     *
     * @param body     the raw message body; {@code null} is treated as empty, as Python's
     *                 {@code (body or "")} is
     * @param maxChars the length the result is cut to
     */
    public static String cleanEmailBody(String body, int maxChars) {
        String text = (body == null ? "" : body)
                .replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n");
        // Bound regex work before the expensive patterns below: DISCLAIMER uses [^.]{0,60/80/100}
        // alternations whose cost grows with input length, and only maxChars are ever returned.
        //
        // In CODE POINTS, because Python slices a str by code point. Cutting by char index splits
        // a surrogate pair, and the half character that comes out is not what the reference
        // returns -- 4,641 of 20,000 astral bodies in a one-off differential sweep.
        // Unconditional, because the helper is a no-op when the bound exceeds the length and
        // because a NEGATIVE bound has a meaning of its own: Python drops the last |n| code
        // points. The `len(text) > max_chars * 4` guard the reference writes is redundant once
        // the slice is faithful -- it is always true for a negative bound anyway.
        text = cutToCodePoints(text, (long) maxChars * 4);
        String[] source = text.split("\n", -1);
        List<String> lines = new ArrayList<>(source.length);
        for (int i = 0; i < source.length; i++) {
            String line = source[i];
            if (!lines.isEmpty() && matchesAny(Patterns.QUOTE_HEADERS, line)) {
                break;
            }
            if (!lines.isEmpty() && Patterns.HEADER_FROM_NAME.matcher(line).lookingAt()
                    && i + 1 < source.length
                    && Patterns.HEADER_NEXT.matcher(source[i + 1]).lookingAt()) {
                break;
            }
            // `!lines.isEmpty()` and the trailing colon first, because both are necessary and
            // both are O(1) against a pattern whose cost grows with the line. The colon really is
            // necessary: the pattern ends `\s*:\s*$`, a line split on "\n" holds no newline,
            // and UNIX_LINES makes `$` the end of input -- so a tail that matches is a line whose
            // rstrip ends with a colon. Together with the possessive quantifier above this is
            // 55 ms to 11 ms at the default budget, and 4.8 s to 0.1 s at 300,000.
            if (!lines.isEmpty() && stripTrailing(line).endsWith(":")
                    && Patterns.ATTRIBUTION_TAIL.matcher(line).lookingAt()) {
                if (Patterns.ATTRIBUTION_HEAD.matcher(lines.get(lines.size() - 1)).lookingAt()) {
                    lines.remove(lines.size() - 1);
                }
                break;
            }
            if (stripLeading(line).startsWith(">")) {
                continue;
            }
            lines.add(stripTrailing(line));
        }
        // The signature search starts partway in, so a closing that opens the message is left
        // alone: `max(1, min(int(len*0.6), len-8))`.
        int cut = lines.size();
        int start = Math.max(1, Math.min((int) (lines.size() * 0.6), lines.size() - 8));
        for (int i = start; i < lines.size(); i++) {
            // Code points, as Python's len() counts them. An astral character is two chars to
            // Java, so counting chars made a 40-code-point closing 41 wide and left a signature
            // in the model's input.
            String candidate = UnicodeTables.strip(lines.get(i));
            int width = candidate.codePointCount(0, candidate.length());
            if ((width <= 40 && isSignature(lines.get(i)))
                    || (width <= 60 && Patterns.DEVICE_FOOTER.matcher(lines.get(i)).lookingAt())) {
                cut = i;
                break;
            }
        }
        lines = lines.subList(0, cut);

        StringBuilder joined = new StringBuilder();
        for (int i = 0; i < lines.size(); i++) {
            if (i > 0) {
                joined.append('\n');
            }
            joined.append(lines.get(i));
        }
        StringBuilder out = new StringBuilder();
        for (String paragraph : Patterns.BLANK_LINE.split(joined.toString(), -1)) {
            String cleaned = UnicodeTables.strip(stripDisclaimer(paragraph));
            if (cleaned.isEmpty()) {
                continue;
            }
            if (out.length() > 0) {
                out.append("\n\n");
            }
            out.append(cleaned);
        }
        String result = Patterns.SPACES_AND_TABS.matcher(out.toString()).replaceAll(" ");
        return cutToCodePoints(result, maxChars);
    }

    /**
     * Python's {@code text[:n]} in code points, for any {@code n} -- including a negative one,
     * which drops the LAST {@code |n|} code points rather than returning nothing.
     *
     * <p>That sign is not a curiosity. {@code clean_email_body} slices twice, once at
     * {@code max_chars * 4} and once at {@code max_chars}, and a negative budget runs through
     * both: the reference answers {@code "I cannot log in a"} for a budget of -5 where this
     * returned the empty string. 17 of 171 body-and-budget combinations differed, every one of
     * them negative, and the test that was here asserted the empty string as correct.
     *
     * <p>The sibling port has the same semantics for free -- JavaScript's {@code slice} reads a
     * negative end the way Python does -- which is why only this port needed the helper.
     *
     * <p>Never cuts between a high and a low surrogate: a half pair is not a character the
     * reference can return. Cutting by char index returned one, and a one-off differential sweep
     * of 20,000 astral bodies against the reference disagreed on 4,641 of them for this reason
     * alone. That sweep is not in the tree -- generated corpora are not committed here -- so the
     * cases that hold the line are the four named {@code budget-*} entries of
     * {@code fixtures/email.json}, each of which fails if this reverts to a char index.
     */
    private static String cutToCodePoints(String text, long limit) {
        if (limit >= 0 && text.length() <= limit) {
            return text;                        // cannot hold more code points than chars
        }
        int codePoints = text.codePointCount(0, text.length());
        long end = limit < 0 ? codePoints + limit : Math.min(limit, codePoints);
        if (end <= 0) {
            return "";
        }
        if (end >= codePoints) {
            return text;
        }
        return text.substring(0, text.offsetByCodePoints(0, (int) end));
    }

    private static boolean matchesAny(List<Pattern> patterns, String line) {
        for (Pattern pattern : patterns) {
            if (pattern.matcher(line).lookingAt()) {
                return true;
            }
        }
        return false;
    }

    /** Python's {@code str.lstrip()} with no argument. */
    private static String stripLeading(String text) {
        int i = 0;
        while (i < text.length()) {
            int cp = text.codePointAt(i);
            if (!UnicodeTables.isSpace(cp)) {
                break;
            }
            i += Character.charCount(cp);
        }
        return text.substring(i);
    }

    /** Python's {@code str.rstrip()} with no argument. */
    private static String stripTrailing(String text) {
        int end = text.length();
        while (end > 0) {
            int cp = text.codePointBefore(end);
            if (!UnicodeTables.isSpace(cp)) {
                break;
            }
            end -= Character.charCount(cp);
        }
        return text.substring(0, end);
    }

    // -------------------------------------------------------------------- the state

    /** {@link #emailState(String, String, String, boolean, int, Map)} with the defaults. */
    public static Map<String, Object> emailState(String subject, String body) {
        return emailState(subject, body, null, true, DEFAULT_MAX_CHARS, Map.of());
    }

    /** {@link #emailState(String, String, String, boolean, int, Map)} with a sender. */
    public static Map<String, Object> emailState(String subject, String body, String sender) {
        return emailState(subject, body, sender, true, DEFAULT_MAX_CHARS, Map.of());
    }

    /**
     * A clean state for email classification.
     *
     * <p>{@code maxChars} is the budget {@link #cleanEmailBody} cuts the body to, and it is worth
     * raising for a long message: at the default the body stops after 3,000 characters, so a
     * request arriving in the last paragraphs never reaches the model. Ignored when
     * {@code clean} is false, which passes the body through whole.
     *
     * <p>Any entry of {@code extra} becomes a field of the state and is therefore read by the
     * model; a wrong key here is an input mutation, not an error, which is the reference's
     * behaviour and is why it is not validated. A {@code null} value is dropped, as the reference
     * drops it.
     *
     * @return a mutable, insertion-ordered map: {@code subject}, {@code body}, then
     *         {@code from} if a sender was given, then the extras. An extra named {@code from}
     *         replaces that value and keeps its position, as Python's {@code dict.update} does,
     *         and a null key is permitted, again as the reference permits it -- both are input
     *         mutations rather than errors, which is the reference's choice. An extra naming a
     *         PARAMETER of this method is refused; see {@link #refuseReservedKeys}.
     * @throws IllegalArgumentException if {@code extra} names {@code subject}, {@code body},
     *         {@code sender}, {@code clean} or {@code max_chars}
     */
    public static Map<String, Object> emailState(String subject, String body, String sender,
                                                 boolean clean, int maxChars,
                                                 Map<String, Object> extra) {
        refuseReservedKeys(extra);
        Map<String, Object> state = new LinkedHashMap<>();
        state.put("subject", UnicodeTables.strip(subject == null ? "" : subject));
        state.put("body", clean ? cleanEmailBody(body, maxChars) : (body == null ? "" : body));
        // Python's `if sender:` -- an empty string is falsy there, so an empty sender adds no
        // field rather than an empty one.
        if (sender != null && !sender.isEmpty()) {
            state.put("from", sender);
        }
        if (extra != null) {
            for (Map.Entry<String, Object> entry : extra.entrySet()) {
                if (entry.getValue() != null) {
                    state.put(entry.getKey(), entry.getValue());
                }
            }
        }
        return state;
    }

    /**
     * Refuse an {@code extra} that names one of this method's own parameters.
     *
     * <p>In the reference these cannot be state fields at all: {@code subject}, {@code body} and
     * {@code max_chars} raise a {@code TypeError}, and {@code sender} and {@code clean} bind the
     * parameter instead. A Java {@code Map} has no such protection, so every one of them was
     * being {@code put} into the state -- and {@code body} is the one that bites, because the
     * state is model input and the value lands there UNCLEANED: no quoted history removed, no
     * signature cut, no budget applied. A caller translating {@code email_state(**payload)} got
     * silence where the reference gives an error.
     *
     * <p>Refusing all five rather than emulating the two that bind: a Java caller has explicit
     * parameters for those, so an entry naming one is ambiguous rather than meaningful, and
     * erroring is the direction that cannot ship the wrong thing to a model.
     */
    private static void refuseReservedKeys(Map<String, Object> extra) {
        if (extra == null) {
            return;
        }
        for (String reserved : List.of("subject", "body", "sender", "clean", "max_chars")) {
            if (extra.containsKey(reserved)) {
                throw new IllegalArgumentException(
                        "extra may not contain \"" + reserved + "\": it names a parameter of "
                        + "emailState, and the reference cannot put it in a state either. Pass it "
                        + "as the argument instead.");
            }
        }
    }

    /**
     * The email question set, re-exported because {@code from laya.email import email_questions}
     * is a path callers already have. One definition, in {@link Presets}; this is the alias.
     */
    public static Map<String, Question> emailQuestions() {
        return Presets.email();
    }

    /** {@link #emailQuestions()} with the caller's own categories, as {@link Presets#email} takes. */
    public static Map<String, Question> emailQuestions(Map<String, String> categories) {
        return Presets.email(categories);
    }
}
