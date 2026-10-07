package com.convaiinnovations.laya.tokenizer;

import com.convaiinnovations.laya.json.Json;
import java.io.ByteArrayOutputStream;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The inverse of tokenization: a list of token strings back into text.
 *
 * <p>Needed because {@code predictLong} scans a long state in overlapping token windows and hands
 * each window back to {@code predictBatch} as TEXT, so every window has to be decoded. It is also
 * worth having on its own -- {@code laya-ts} and {@code laya-dotnet} both expose a decode -- but
 * the windowing is what made it necessary.
 *
 * <p>Unlike the pre-tokenizer, this has no Python reference to port: decoding lives in the Rust
 * {@code tokenizers} crate, so the rules below were probed out of it against the two shipped
 * checkpoints, and the recorded round-trips in {@code fixtures/decode_text.json} are the contract.
 *
 * <h2>Decoding is not lossless, and that is the contract</h2>
 *
 * <p>On the english checkpoint it happens to round-trip. On the multilingual one it does not, and
 * the port must reproduce the loss exactly rather than improve on it:
 *
 * <pre>
 *   english      "Hello world"  -> "Hello world"
 *   multilingual "Hello world"  -> " Hello world"     &lt;- a leading space appears
 *   multilingual "a\tb\nc"      -> " a\tb\n c"        &lt;- and one after the newline
 * </pre>
 *
 * <p>That is Metaspace: its pre-tokenizer prepends U+2581 to the text ({@code prepend_scheme:
 * always}), and the decoder turns every U+2581 back into a space, including the prepended one. The
 * reference lives with it -- {@code predict_long}'s own comment says the 50% window overlap
 * absorbs the boundary drift -- so a port that trimmed the space would re-tokenize every window
 * differently from the reference and diverge on every answer downstream.
 */
abstract class TokenDecoder {

    /** The decoded text for these tokens, in order. */
    abstract String decode(List<String> tokens);

    /**
     * The decoder a {@code tokenizer.json} declares, or a plain join when it declares none.
     *
     * <p>A checkpoint with no {@code decoder} section is not an error: the crate joins the tokens
     * with a space, and refusing to load would reject a valid tokenizer over a decode path a
     * caller may never use. See {@link Join} for the separator, which this got wrong.
     */
    static TokenDecoder from(Map<String, Object> node) {
        if (node == null) {
            return new Join();
        }
        String type = Json.string(node, "type", null);
        if (type == null) {
            throw new Json.JsonException("a decoder node must declare a type");
        }
        switch (type) {
            case "ByteLevel":
                return new ByteLevel();
            case "Replace":
                return new Replace(node);
            case "ByteFallback":
                return new ByteFallback();
            case "Fuse":
                return new Fuse();
            case "Strip":
                return new Strip(node);
            case "Sequence": {
                Object steps = node.get("decoders");
                if (!(steps instanceof List)) {
                    throw new Json.JsonException("a Sequence decoder needs a decoders array");
                }
                List<TokenDecoder> parts = new ArrayList<>();
                for (Object step : (List<?>) steps) {
                    if (!(step instanceof Map)) {
                        throw new Json.JsonException("every Sequence decoder step must be an object");
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> stepNode = (Map<String, Object>) step;
                    parts.add(from(stepNode));
                }
                return new Sequence(parts);
            }
            default:
                // Refusing beats guessing. A WordPiece or Metaspace decoder would produce
                // plausible-looking text with the wrong spaces, and the windows predictLong builds
                // from it would re-tokenize to different ids with nothing to show it had happened.
                throw new Json.JsonException(
                        "laya-java implements the ByteLevel, Replace, ByteFallback, Fuse, Strip and"
                        + " Sequence decoders; this checkpoint declares \"" + type + "\". Refusing"
                        + " to load rather than decode it wrongly.");
        }
    }

    // ------------------------------------------------------------------ the steps

    /**
     * No decoder declared: the tokens joined with a SPACE.
     *
     * <p>A space, not nothing. The crate's fallback for an absent decoder is
     * {@code tokens.join(" ")}, and this joined with nothing -- wrong for every multi-token
     * decode, not for an edge case. Measured on the multilingual vocabulary with its
     * {@code decoder} key removed: ids {@code [25957, 2134]} give {@code "\u2581Hello
     * \u2581world"}, with a space between the pieces.
     *
     * <p>Unreachable from either shipped checkpoint, since both declare a decoder. That is the
     * reason it was wrong and stayed wrong: this is the one branch of {@code from} that is
     * permissive rather than refusing, so it is where a third-party checkpoint lands, and the
     * recorded fixtures can never reach it. It has its own test for that reason.
     */
    private static final class Join extends TokenDecoder {
        @Override
        String decode(List<String> tokens) {
            return String.join(" ", tokens);
        }
    }

    /**
     * The inverse of the ByteLevel pre-tokenizer: every stand-in character is one byte.
     *
     * <p>The bytes are then UTF-8, and a token boundary can fall INSIDE a multi-byte character --
     * CJK is three bytes and routinely splits across two tokens -- so the whole token list is
     * concatenated to bytes first and decoded once at the end. Decoding token by token would turn
     * every split character into replacement characters.
     */
    private static final class ByteLevel extends TokenDecoder {
        @Override
        String decode(List<String> tokens) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            for (String token : tokens) {
                // Whole-token fallback, not per character. The crate folds over a token's
                // characters and, if ANY of them is outside the stand-in alphabet, uses the
                // token's own UTF-8 bytes for the WHOLE token. Falling back per character
                // coincides on a token that is entirely literal -- which is the only kind this
                // vocabulary has -- and would differ on a mixed one, so the rule is written as
                // the crate writes it rather than as the corpus happens to need.
                byte[] mapped = mapAll(token);
                if (mapped == null) {
                    byte[] raw = token.getBytes(StandardCharsets.UTF_8);
                    bytes.write(raw, 0, raw.length);
                    continue;
                }
                bytes.write(mapped, 0, mapped.length);
            }
            return lossyUtf8(bytes.toByteArray());
        }

        /** Every character as its byte, or null when one of them is not a stand-in. */
        private static byte[] mapAll(String token) {
            byte[] out = new byte[token.length()];
            for (int i = 0; i < token.length(); i++) {
                int b = PreTokenizer.byteForChar(token.charAt(i));
                if (b < 0) {
                    return null;
                }
                out[i] = (byte) b;
            }
            return out;
        }


    }

    /** {@code Replace}: a literal substring for another, applied to every token. */
    private static final class Replace extends TokenDecoder {

        private final String pattern;
        private final String content;

        Replace(Map<String, Object> node) {
            Object patternNode = node.get("pattern");
            String literal = null;
            if (patternNode instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> p = (Map<String, Object>) patternNode;
                literal = Json.string(p, "String", null);
                if (literal == null && p.containsKey("Regex")) {
                    // A regex Replace would need the crate's regex dialect, and getting it subtly
                    // wrong moves spaces in decoded text. None of the shipped checkpoints uses
                    // one, so this refuses rather than approximating.
                    throw new Json.JsonException(
                            "laya-java implements only a literal String pattern in a Replace"
                            + " decoder; this checkpoint declares a Regex one");
                }
            } else if (patternNode instanceof String) {
                literal = (String) patternNode;
            }
            if (literal == null) {
                throw new Json.JsonException("a Replace decoder needs a String pattern");
            }
            this.pattern = literal;
            this.content = Json.string(node, "content", "");
        }

        @Override
        String decode(List<String> tokens) {
            StringBuilder out = new StringBuilder();
            for (String token : tokens) {
                out.append(token.replace(pattern, content));
            }
            return out.toString();
        }

        @Override
        List<String> transform(List<String> tokens) {
            // One piece per token, NOT one joined string: ByteFallback runs after this in the
            // multilingual checkpoint's Sequence and needs the boundaries to find a byte run.
            List<String> out = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                out.add(token.replace(pattern, content));
            }
            return out;
        }
    }

    /**
     * {@code ByteFallback}: a run of {@code <0xNN>} tokens is one UTF-8 sequence.
     *
     * <p>Reachable on the shipped multilingual checkpoint, which declares
     * {@code byte_fallback: true} and carries all 255 {@code <0xNN>} tokens. U+F0000 encodes as
     * four of them -- {@code <0xF3> <0xB0> <0x80> <0x80>} -- so a run has to be gathered and
     * decoded together; one at a time gives four replacement characters.
     */
    private static final class ByteFallback extends TokenDecoder {
        @Override
        String decode(List<String> tokens) {
            StringBuilder out = new StringBuilder();
            ByteArrayOutputStream run = new ByteArrayOutputStream();
            for (String token : tokens) {
                int b = byteToken(token);
                if (b >= 0) {
                    run.write(b);
                    continue;
                }
                flush(run, out);
                out.append(token);
            }
            flush(run, out);
            return out.toString();
        }

        @Override
        List<String> transform(List<String> tokens) {
            // A run that decodes is one piece; a run that does not becomes one piece PER BYTE.
            // The crate pushes U+FFFD into its token list once per byte it could not use, and a
            // later step in the Sequence sees those boundaries. Measured with
            // `Sequence[ByteFallback, Replace("\ufffd\ufffd" -> "X")]` over the two bytes of a
            // cut U+F0000: the crate leaves the pair alone because the pieces are separate, and
            // emitting them as one piece of two characters matched the pattern and gave "X".
            //
            // Invisible on the shipped multilingual checkpoint because Fuse is the next step and
            // re-joins everything, so the strings agree. The override exists to preserve
            // boundaries, and this was the one shape where it did not.
            List<String> out = new ArrayList<>();
            ByteArrayOutputStream run = new ByteArrayOutputStream();
            for (String token : tokens) {
                int b = byteToken(token);
                if (b >= 0) {
                    run.write(b);
                    continue;
                }
                flushPieces(run, out);
                out.add(token);
            }
            flushPieces(run, out);
            return out;
        }

        private static void flushPieces(ByteArrayOutputStream run, List<String> out) {
            if (run.size() == 0) {
                return;
            }
            byte[] bytes = run.toByteArray();
            run.reset();
            String text = strictUtf8OrNull(bytes);
            if (text != null) {
                out.add(text);
                return;
            }
            for (int i = 0; i < bytes.length; i++) {
                out.add("\ufffd");
            }
        }

        private static void flush(ByteArrayOutputStream run, StringBuilder out) {
            if (run.size() > 0) {
                out.append(strictOrReplacementPerByte(run.toByteArray()));
                run.reset();
            }
        }
    }

    /** {@code Fuse}: concatenate the pieces into one string. */
    private static final class Fuse extends TokenDecoder {
        @Override
        String decode(List<String> tokens) {
            StringBuilder out = new StringBuilder();
            for (String token : tokens) {
                out.append(token);
            }
            return out.toString();
        }
    }

    /** {@code Strip}: remove up to n copies of a character from each end of every token. */
    private static final class Strip extends TokenDecoder {

        private final char content;
        private final int start;
        private final int stop;

        Strip(Map<String, Object> node) {
            String c = Json.string(node, "content", " ");
            if (c.length() != 1) {
                throw new Json.JsonException(
                        "a Strip decoder's content must be one character, got \"" + c + "\"");
            }
            this.content = c.charAt(0);
            this.start = Json.integer(node, "start", 0);
            this.stop = Json.integer(node, "stop", 0);
        }

        @Override
        String decode(List<String> tokens) {
            StringBuilder out = new StringBuilder();
            for (String token : tokens) {
                out.append(apply(token));
            }
            return out.toString();
        }

        @Override
        List<String> transform(List<String> tokens) {
            List<String> out = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                out.add(apply(token));
            }
            return out;
        }

        String apply(String token) {
            int from = 0;
            int to = token.length();
            for (int i = 0; i < start && from < to && token.charAt(from) == content; i++) {
                from++;
            }
            for (int i = 0; i < stop && to > from && token.charAt(to - 1) == content; i++) {
                to--;
            }
            return token.substring(from, to);
        }
    }

    /**
     * {@code Sequence}: each step transforms the token LIST, not the joined string.
     *
     * <p>That distinction is the whole behaviour of the multilingual checkpoint's
     * {@code Sequence[Replace, ByteFallback, Fuse]}. Replace rewrites each token, ByteFallback
     * needs to see token boundaries to find a {@code <0xNN>} run, and only Fuse joins. Running the
     * steps over an already-joined string would make ByteFallback unable to tell a literal
     * {@code "<0x41>"} in the text from a byte token.
     */
    private static final class Sequence extends TokenDecoder {

        private final List<TokenDecoder> steps;

        Sequence(List<TokenDecoder> steps) {
            this.steps = List.copyOf(steps);
        }

        @Override
        String decode(List<String> tokens) {
            return String.join("", transform(tokens));
        }

        @Override
        List<String> transform(List<String> tokens) {
            // The chained list, NOT one joined piece. Without this override a Sequence inherits
            // the base default, which re-joins -- so a Sequence NESTED in a Sequence handed one
            // string to the outer sequence's next step where the crate hands it N pieces. The
            // crate's `decode_chain` chains `decode_chain` through its steps and returns a list.
            //
            // Measured with `Sequence[Sequence[Replace, ByteFallback], Strip(" ", start=1)]`: the
            // crate gives "Helloworld", because Strip sees two pieces and takes the leading space
            // off each; re-joining gave "Hello world". Neither shipped checkpoint nests deeper
            // than one level, which is why nothing caught it and why it has its own test.
            List<String> current = new ArrayList<>(tokens);
            for (TokenDecoder step : steps) {
                current = step.transform(current);
            }
            return current;
        }
    }

    /**
     * This step's effect on a token list, which is what a {@code Sequence} composes.
     *
     * <p>The default is "whatever this decoder produces, as one piece". The steps that have to
     * preserve boundaries for a later step override it.
     */
    List<String> transform(List<String> tokens) {
        return List.of(decode(tokens));
    }

    // ------------------------------------------------------------------ helpers

    /** The byte a {@code <0xNN>} token stands for, or -1 when the token is not one. */
    private static int byteToken(String token) {
        if (token == null || token.length() != 6
                || token.charAt(0) != '<' || token.charAt(1) != '0' || token.charAt(2) != 'x'
                || token.charAt(5) != '>') {
            return -1;
        }
        return parseByte(token.charAt(3), token.charAt(4));
    }

    /**
     * The two characters between {@code <0x} and {@code >} as a byte, following Rust's
     * {@code u8::from_str_radix(.., 16)} -- which is the function the crate calls.
     *
     * <p>Case-INSENSITIVE, and a leading {@code +} is accepted. Both were measured, after this
     * rejected lowercase on the stated grounds that "the crate writes these uppercase, and
     * accepting both would make a literal {@code <0xff>} in ordinary text decode as a byte". That
     * second clause describes what the crate actually does, rather than a hazard it avoids: asked
     * to decode a vocabulary whose only token is {@code <0xff>}, the crate returns U+FFFD -- it
     * read the byte -- and the same for {@code <0xfF>}. {@code <0x+5>} is byte 0x05.
     *
     * <p>{@code <0x 5>} and {@code <0XFF>} stay literal, because a space is not a digit and the
     * {@code <0x} prefix is matched case-sensitively. Both measured too, so the boundary is
     * recorded rather than assumed in the other direction this time.
     *
     * <p>No shipped vocabulary contains a lowercase byte token -- all 255 in the multilingual
     * checkpoint are uppercase -- so this was not a live bug. It was a wrong rule with a
     * confident comment, which is worse to leave in place than a wrong rule without one.
     */
    private static int parseByte(char high, char low) {
        if (high == '+') {
            return hex(low);
        }
        int h = hex(high);
        int l = hex(low);
        return h < 0 || l < 0 ? -1 : (h << 4) | l;
    }

    private static int hex(char c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        return -1;
    }

    /**
     * UTF-8 with replacement, as the crate's lossy decode does.
     *
     * <p>Not {@code new String(bytes, UTF_8)}, which is also lossy but has differed from Rust's
     * {@code from_utf8_lossy} on how many replacement characters a truncated sequence produces.
     * A window boundary cuts multi-byte characters routinely, so this is a path that runs, not an
     * edge case: the decoder is configured explicitly to REPLACE rather than report, and one
     * replacement per malformed input sequence is what both produce.
     */
    private static String lossyUtf8(byte[] bytes) {
        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPLACE)
                .onUnmappableCharacter(CodingErrorAction.REPLACE);
        try {
            return decoder.decode(ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException impossible) {
            throw new IllegalStateException("a REPLACE decoder cannot report an error", impossible);
        }
    }

    /**
     * {@code ByteFallback}'s rule, which is NOT lossy decoding: strict UTF-8, and on failure one
     * U+FFFD per BYTE of the run.
     *
     * <p>Measured, after assuming otherwise cost 629 of 201,279 window slices. A window boundary
     * cuts U+F0000 after two of its four bytes, leaving {@code F3 B0}; lossy decoding calls that
     * one maximal subpart and emits ONE replacement character, and the crate emits TWO -- one for
     * each byte it could not use. The two rules agree only when the run is valid, which is every
     * case that does not straddle a boundary, which is why this looked right until the slices were
     * compared.
     *
     * <p>The sibling {@code ByteLevel} decoder really does use lossy decoding, so the two paths in
     * this file deliberately differ. That is the crate's shape, not an oversight.
     */
    private static String strictOrReplacementPerByte(byte[] bytes) {
        String text = strictUtf8OrNull(bytes);
        return text != null ? text : "\ufffd".repeat(bytes.length);
    }

    /** The run as strict UTF-8, or null when it is not valid UTF-8. */
    private static String strictUtf8OrNull(byte[] bytes) {
        CharsetDecoder strict = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT);
        try {
            return strict.decode(ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException invalid) {
            return null;
        }
    }
}
