package com.convaiinnovations.laya.json;

import java.io.IOException;
import java.io.Reader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal, dependency-free JSON reader for the two documents a checkpoint ships:
 * {@code tokenizer.json} and {@code rl_agent_config.json}.
 *
 * <p>The runtime takes exactly one dependency, ONNX Runtime. A JSON library in the core would force
 * its version on every consumer of this library, and two known schemas do not justify that. This
 * reader is therefore deliberately small, and deliberately strict: it rejects what it does not
 * understand instead of guessing, because a tokenizer parsed loosely produces token ids that are
 * wrong in ways no answer-level test can localise.
 *
 * <p>Two properties matter for the schemas it is built for:
 * <ul>
 *   <li><b>Object key order is preserved</b> ({@link LinkedHashMap}). A {@code choice} question's
 *       criteria order is positional in laya -- two orders are two different questions -- so a
 *       reader that folded keys into a hash order would silently change what the model is asked.
 *   <li><b>Numbers are exact.</b> Integers arrive as {@link Long} and only widen to
 *       {@link java.math.BigInteger} when they must; anything with a fraction or exponent becomes a
 *       {@link Double}. A 256,000-entry vocabulary read through a lossy number path is a
 *       mis-tokenisation waiting to happen.
 * </ul>
 *
 * <p>Values map to {@code Map<String, Object>}, {@code List<Object>}, {@link String},
 * {@link Long}/{@link Double}, {@link Boolean} and {@code null}.
 */
public final class Json {

    private final Reader reader;
    private int pending = -2;          // -2 = nothing buffered, -1 = end of input
    private long index;

    private Json(Reader reader) {
        this.reader = reader;
    }

    /** Reads one JSON document from {@code reader} and fails if anything but whitespace follows. */
    public static Object parse(Reader reader) throws IOException {
        Json json = new Json(reader);
        json.skipWhitespace();
        Object value = json.readValue();
        json.skipWhitespace();
        if (json.peek() != -1) {
            throw json.fail("trailing content after the document");
        }
        return value;
    }

    /** Reads one JSON document from a string. Convenience for configs and tests. */
    public static Object parse(String text) {
        try {
            return parse(new java.io.StringReader(text));
        } catch (IOException impossible) {
            throw new IllegalStateException("StringReader cannot fail", impossible);
        }
    }

    // ------------------------------------------------------------------ typed accessors
    //
    // A checkpoint is data, so every lookup says which key it wanted when it is disappointed. A
    // ClassCastException naming neither the file nor the key is not a diagnosis.

    /** The object at {@code key}, or a failure naming {@code key} and what was there instead. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(Map<String, Object> source, String key) {
        Object value = require(source, key);
        if (!(value instanceof Map)) {
            throw typeError(key, "an object", value);
        }
        return (Map<String, Object>) value;
    }

    /** The array at {@code key}, or a failure naming {@code key}. */
    @SuppressWarnings("unchecked")
    public static List<Object> array(Map<String, Object> source, String key) {
        Object value = require(source, key);
        if (!(value instanceof List)) {
            throw typeError(key, "an array", value);
        }
        return (List<Object>) value;
    }

    /** The string at {@code key}, or {@code fallback} when the key is absent or null. */
    public static String string(Map<String, Object> source, String key, String fallback) {
        Object value = source.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof String)) {
            throw typeError(key, "a string", value);
        }
        return (String) value;
    }

    /** The integer at {@code key}, or {@code fallback} when the key is absent or null. */
    public static int integer(Map<String, Object> source, String key, int fallback) {
        Object value = source.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof Long)) {
            throw typeError(key, "an integer", value);
        }
        long wide = (Long) value;
        if (wide < Integer.MIN_VALUE || wide > Integer.MAX_VALUE) {
            throw new JsonException(key + " does not fit in an int: " + wide);
        }
        return (int) wide;
    }

    /** The boolean at {@code key}, or {@code fallback} when the key is absent or null. */
    public static boolean bool(Map<String, Object> source, String key, boolean fallback) {
        Object value = source.get(key);
        if (value == null) {
            return fallback;
        }
        if (!(value instanceof Boolean)) {
            throw typeError(key, "a boolean", value);
        }
        return (Boolean) value;
    }

    private static Object require(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value == null) {
            throw new JsonException("missing required key " + key);
        }
        return value;
    }

    private static JsonException typeError(String key, String wanted, Object found) {
        return new JsonException(key + " must be " + wanted + ", found "
                + (found == null ? "null" : found.getClass().getSimpleName()));
    }

    /** Thrown for a malformed document or a key that is absent or of the wrong shape. */
    public static final class JsonException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        /** Public so that the stages that read a checkpoint can report a malformed one. */
        public JsonException(String message) {
            super(message);
        }
    }

    // ------------------------------------------------------------------ the reader itself

    /**
     * How deep a document may nest.
     *
     * <p>Without a limit, deeply nested input threw {@link StackOverflowError} -- an
     * {@link Error}, which a server's {@code catch (Exception)} does not contain and which can
     * leave a thread's invariants broken. CPython raises a catchable {@code RecursionError}.
     * Both {@code tokenizer.json} and {@code rl_agent_config.json} arrive from a remote download
     * and are parsed by this reader, so the input is not trusted; the limit sits far above
     * anything a real checkpoint nests and far below the stack.
     */
    private static final int MAX_DEPTH = 512;

    /** CPython's integer-string limit, which also bounds a quadratic BigDecimal conversion. */
    private static final int MAX_INTEGER_DIGITS = 4300;

    private int depth;

    /**
     * A hex digit, ASCII only.
     *
     * <p>{@code Character.digit} is Unicode-aware, so fullwidth and Arabic-Indic digits were
     * accepted as hex: two byte-different documents parsed to the same value, and input CPython
     * rejects was accepted.
     */
    private static int asciiHex(int c) {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        }
        if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return -1;
    }

    private Object readValue() throws IOException {
        int c = peek();
        switch (c) {
            case '{':
                return readObject();
            case '[':
                return readArray();
            case '"':
                return readString();
            case 't':
                expect("true");
                return Boolean.TRUE;
            case 'f':
                expect("false");
                return Boolean.FALSE;
            case 'n':
                expect("null");
                return null;
            case -1:
                throw fail("unexpected end of input");
            default:
                return readNumber();
        }
    }

    private Map<String, Object> readObject() throws IOException {
        if (++depth > MAX_DEPTH) {
            throw new JsonException("JSON nested deeper than " + MAX_DEPTH + " levels");
        }
        try {
            return readObjectBody();
        } finally {
            depth--;
        }
    }

    private Map<String, Object> readObjectBody() throws IOException {
        read();                                     // '{'
        Map<String, Object> out = new LinkedHashMap<>();
        skipWhitespace();
        if (peek() == '}') {
            read();
            return out;
        }
        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                throw fail("object keys must be strings");
            }
            String key = readString();
            skipWhitespace();
            if (read() != ':') {
                throw fail("expected ':' after an object key");
            }
            skipWhitespace();
            // Last one wins, matching every mainstream parser; `put` on a LinkedHashMap keeps the
            // position of the first insertion, which is also what Python's json does.
            out.put(key, readValue());
            skipWhitespace();
            int next = read();
            if (next == '}') {
                return out;
            }
            if (next != ',') {
                throw fail("expected ',' or '}' in an object");
            }
        }
    }

    private List<Object> readArray() throws IOException {
        if (++depth > MAX_DEPTH) {
            throw new JsonException("JSON nested deeper than " + MAX_DEPTH + " levels");
        }
        try {
            return readArrayBody();
        } finally {
            depth--;
        }
    }

    private List<Object> readArrayBody() throws IOException {
        read();                                     // '['
        List<Object> out = new ArrayList<>();
        skipWhitespace();
        if (peek() == ']') {
            read();
            return out;
        }
        while (true) {
            skipWhitespace();
            out.add(readValue());
            skipWhitespace();
            int next = read();
            if (next == ']') {
                return out;
            }
            if (next != ',') {
                throw fail("expected ',' or ']' in an array");
            }
        }
    }

    private String readString() throws IOException {
        read();                                     // opening quote
        StringBuilder out = new StringBuilder();
        while (true) {
            int c = read();
            if (c == -1) {
                throw fail("unterminated string");
            }
            if (c == '"') {
                return out.toString();
            }
            if (c != '\\') {
                if (c < 0x20) {
                    throw fail("a raw control character in a string");
                }
                out.append((char) c);
                continue;
            }
            int escape = read();
            switch (escape) {
                case '"': out.append('"'); break;
                case '\\': out.append('\\'); break;
                case '/': out.append('/'); break;
                case 'b': out.append('\b'); break;
                case 'f': out.append('\f'); break;
                case 'n': out.append('\n'); break;
                case 'r': out.append('\r'); break;
                case 't': out.append('\t'); break;
                case 'u':
                    // Appended as a UTF-16 code unit without pairing: a surrogate pair arrives as
                    // two backslash-u escapes, and appending each unit reassembles it. Validating
                    // here would reject nothing a tokenizer ships and would risk rejecting a lone
                    // surrogate that the source file legitimately contains.
                    out.append((char) readHex4());
                    break;
                default:
                    throw fail("unknown escape \\" + (escape == -1 ? "<eof>" : (char) escape));
            }
        }
    }

    private int readHex4() throws IOException {
        int value = 0;
        for (int i = 0; i < 4; i++) {
            int c = read();
            int digit = asciiHex(c);
            if (c == -1 || digit < 0) {
                throw fail("a \\u escape needs four hex digits");
            }
            value = (value << 4) | digit;
        }
        return value;
    }

    private Object readNumber() throws IOException {
        StringBuilder text = new StringBuilder();
        boolean integral = true;
        while (true) {
            int c = peek();
            if (c == '-' || c == '+' || (c >= '0' && c <= '9')) {
                text.append((char) read());
            } else if (c == '.' || c == 'e' || c == 'E') {
                integral = false;
                text.append((char) read());
            } else {
                break;
            }
        }
        if (text.length() == 0) {
            throw fail("expected a value");
        }
        String literal = text.toString();
        try {
            if (integral) {
                try {
                    return Long.valueOf(literal);
                } catch (NumberFormatException tooWide) {
                    if (literal.length() > MAX_INTEGER_DIGITS) {
                        // CPython refuses past 4,300 digits (the CVE-2020-10735 mitigation), and
                        // this conversion is QUADRATIC: a 1 MB literal cost 21 seconds of a
                        // request thread, from one field of an untrusted tokenizer.json.
                        throw new JsonException("integer literal of " + literal.length()
                                + " digits exceeds the " + MAX_INTEGER_DIGITS
                                + " this reader allows");
                    }
                    return new BigDecimal(literal).toBigIntegerExact();
                }
            }
            return Double.valueOf(literal);
        } catch (NumberFormatException | ArithmeticException malformed) {
            throw fail("not a number: " + literal);
        }
    }

    private void expect(String word) throws IOException {
        for (int i = 0; i < word.length(); i++) {
            if (read() != word.charAt(i)) {
                throw fail("expected '" + word + "'");
            }
        }
    }

    private void skipWhitespace() throws IOException {
        while (true) {
            int c = peek();
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                read();
            } else {
                return;
            }
        }
    }

    private int peek() throws IOException {
        if (pending == -2) {
            pending = reader.read();
        }
        return pending;
    }

    private int read() throws IOException {
        int c = peek();
        pending = -2;
        if (c != -1) {
            index++;
        }
        return c;
    }

    private JsonException fail(String message) {
        return new JsonException(message + " at offset " + index);
    }
}
