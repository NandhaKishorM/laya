package com.convaiinnovations.laya.tokenizer;

import com.convaiinnovations.laya.json.Json;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;

/**
 * A HuggingFace {@code tokenizer.json} BPE tokenizer, loaded and run without a Python or native
 * dependency.
 *
 * <p>The pipeline is the reference implementation's, in its order: added tokens are matched on the
 * raw input, the remaining text is normalized, added tokens that ask to be matched on normalized
 * text are matched next, and what is left is pre-tokenized and merged.
 *
 * <p>Both laya checkpoints are supported, and they differ in every stage -- NFC against a space
 * replacement, byte-level against metaspace, no unknown token against an unknown token with a
 * byte fallback. A port built against one of them and assumed to cover the other would be wrong on
 * every input.
 *
 * <p>Instances are immutable after construction and safe to share between threads.
 */
public final class Tokenizer {

    private final Normalizer normalizer;
    private final PreTokenizer preTokenizer;
    private final AddedVocabulary rawAdded;
    private final AddedVocabulary normalizedAdded;
    private final Bpe bpe;
    private final Map<String, Integer> ids;
    private final String[] tokens;
    private final Map<String, Integer> specials;
    private final Map<String, Integer> roles;
    private final String maskToken;
    private final TokenDecoder decoder;
    /** Ids the reference's decode drops unless asked not to; see {@link #decode(int[])}. */
    private final java.util.Set<Integer> specialIds;

    /** The roles a checkpoint names in {@code tokenizer_config.json}, in sequence-building order. */
    private static final String[] ROLE_KEYS =
            {"cls_token", "sep_token", "mask_token", "pad_token", "unk_token"};

    /**
     * Loads the tokenizer a laya checkpoint directory ships, the way laya loads it.
     *
     * <p>{@code <modelDir>/tokenizer}, which is what {@code onnx_agent.py:156} and
     * {@code agent.py:235} resolve. Taking a different path to the same {@code tokenizer.json} is
     * not equivalent: the multilingual checkpoint carries a byte-identical copy at its root
     * <i>without</i> {@code tokenizer_config.json}, and loading that one reports no special tokens
     * at all rather than {@code cls=<bos>}, {@code sep=<eos>}, {@code mask=<mask>}.
     */
    public static Tokenizer fromModelDirectory(Path modelDirectory) throws IOException {
        Path inner = modelDirectory.resolve("tokenizer");
        return fromDirectory(Files.isRegularFile(inner.resolve("tokenizer.json"))
                ? inner : modelDirectory);
    }

    /**
     * Loads {@code tokenizer.json} from {@code directory}, with its {@code tokenizer_config.json}
     * when one is present.
     *
     * <p>The special tokens live in the config, not in {@code tokenizer.json}: the multilingual
     * checkpoint reuses {@code <bos>} as its classification token and {@code <eos>} as its
     * separator, which nothing in the tokenizer itself reveals.
     */
    public static Tokenizer fromDirectory(Path directory) throws IOException {
        Map<String, Object> config = null;
        Path configPath = directory.resolve("tokenizer_config.json");
        if (Files.isRegularFile(configPath)) {
            try (Reader reader = Files.newBufferedReader(configPath, StandardCharsets.UTF_8)) {
                Object parsed = Json.parse(reader);
                if (!(parsed instanceof Map)) {
                    throw new Json.JsonException("a tokenizer_config.json must be a JSON object");
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> asMap = (Map<String, Object>) parsed;
                config = asMap;
            }
        }
        try (Reader reader = new BufferedReader(
                Files.newBufferedReader(directory.resolve("tokenizer.json"), StandardCharsets.UTF_8),
                1 << 16)) {
            Object parsed = Json.parse(reader);
            if (!(parsed instanceof Map)) {
                throw new Json.JsonException("a tokenizer.json must be a JSON object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> document = (Map<String, Object>) parsed;
            return new Tokenizer(document, config);
        }
    }

    /** Loads the {@code tokenizer.json} at {@code path}, with no special-token config. */
    public static Tokenizer fromFile(Path path) throws IOException {
        try (Reader reader = new BufferedReader(
                Files.newBufferedReader(path, StandardCharsets.UTF_8), 1 << 16)) {
            return fromReader(reader);
        }
    }

    /** Loads a {@code tokenizer.json} document from a reader. */
    public static Tokenizer fromReader(Reader reader) throws IOException {
        Object parsed = Json.parse(reader);
        if (!(parsed instanceof Map)) {
            throw new Json.JsonException("a tokenizer.json must be a JSON object");
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> document = (Map<String, Object>) parsed;
        return new Tokenizer(document, null);
    }

    /** Loads a {@code tokenizer.json} document from a string. For tests and embedded fixtures. */
    public static Tokenizer fromJson(String json) {
        try {
            return fromReader(new StringReader(json));
        } catch (IOException impossible) {
            throw new IllegalStateException("StringReader cannot fail", impossible);
        }
    }

    private Tokenizer(Map<String, Object> document, Map<String, Object> tokenizerConfig) {
        Map<String, Object> model = Json.object(document, "model");
        String modelType = Json.string(model, "type", null);
        if (!"BPE".equals(modelType)) {
            throw new Json.JsonException(
                    "laya-java implements the BPE tokenizer model; this checkpoint declares \""
                    + modelType + "\". Refusing to load rather than mis-tokenize silently.");
        }

        this.normalizer = Normalizer.from(objectOrNull(document, "normalizer"));
        this.preTokenizer = PreTokenizer.from(objectOrNull(document, "pre_tokenizer"));
        this.decoder = TokenDecoder.from(objectOrNull(document, "decoder"));

        // --- the id map: model.vocab UNION added_tokens, with added_tokens authoritative.
        //
        // Not a belt-and-braces union. In the english checkpoint 88 of the 116 added tokens are
        // absent from model.vocab entirely, carrying ids 50280-50367 above the vocabulary's own
        // maximum of 50279 -- `[MASK]` (50284) among them. Building the id map from model.vocab
        // alone would leave masked prediction with no mask id at all.
        Map<String, Integer> vocab = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : Json.object(model, "vocab").entrySet()) {
            if (!(entry.getValue() instanceof Long)) {
                throw new Json.JsonException(
                        "vocabulary entry " + entry.getKey() + " has a non-integer id");
            }
            vocab.put(entry.getKey(), ((Long) entry.getValue()).intValue());
        }

        List<AddedToken> added = readAddedTokens(document);
        // The ids a decode drops by default. Taken from each added token's own `special` flag
        // rather than from the laya config's four roles: the english checkpoint declares 116
        // added tokens and the config names only cls/sep/mask/pad, so the roles would leave
        // every other control token in the decoded text.
        java.util.Set<Integer> special = new java.util.LinkedHashSet<>();
        for (AddedToken token : added) {
            if (token.special()) {
                special.add(token.id());
            }
        }
        this.specialIds = java.util.Set.copyOf(special);
        for (AddedToken token : added) {
            Integer existing = vocab.get(token.content());
            if (existing != null && existing != token.id()) {
                // Measured as absent-or-equal across both checkpoints, never conflicting. If a
                // checkpoint ever did conflict, the added-token table is the one the reference
                // implementation consults first, so it wins -- loudly, not silently.
                System.getLogger(Tokenizer.class.getName()).log(
                        System.Logger.Level.WARNING,
                        "added token {0} has id {1} but the vocabulary says {2}; using the "
                        + "added-token id, as the reference implementation does",
                        token.content(), token.id(), existing);
            }
            vocab.put(token.content(), token.id());
        }
        this.ids = Map.copyOf(vocab);

        // --- merges, and the invariant that lets ids be resolved after merging rather than before
        List<String[]> merges = new ArrayList<>();
        for (Object entry : Json.array(model, "merges")) {
            merges.add(readMerge(entry));
        }
        checkMergeClosure(merges, this.ids);

        this.bpe = new Bpe(
                this.ids,
                merges,
                Json.string(model, "unk_token", null),
                Json.bool(model, "fuse_unk", false),
                Json.bool(model, "ignore_merges", false),
                Json.bool(model, "byte_fallback", false));

        checkEveryCharacterIsRepresentable();

        // --- added tokens are matched at two different stages, by their `normalized` flag
        List<AddedToken> onRaw = new ArrayList<>();
        List<AddedToken> onNormalized = new ArrayList<>();
        for (AddedToken token : added) {
            (token.normalized() ? onNormalized : onRaw).add(token);
        }
        this.rawAdded = new AddedVocabulary(onRaw);
        this.normalizedAdded = new AddedVocabulary(onNormalized);

        // --- reverse map, for decoding
        int highest = 0;
        for (int id : this.ids.values()) {
            highest = Math.max(highest, id);
        }
        this.tokens = new String[highest + 1];
        for (Map.Entry<String, Integer> entry : this.ids.entrySet()) {
            this.tokens[entry.getValue()] = entry.getKey();
        }

        Map<String, Integer> found = new LinkedHashMap<>();
        for (AddedToken token : added) {
            if (token.special()) {
                found.put(token.content(), token.id());
            }
        }
        this.specials = Map.copyOf(found);

        Map<String, Integer> byRole = new LinkedHashMap<>();
        String mask = null;
        if (tokenizerConfig != null) {
            for (String key : ROLE_KEYS) {
                String content = specialTokenContent(tokenizerConfig, key);
                if (content == null) {
                    continue;
                }
                Integer id = this.ids.get(content);
                if (id == null) {
                    throw new Json.JsonException(
                            "tokenizer_config.json names " + content + " as " + key
                            + ", but neither the vocabulary nor the added-token table has it");
                }
                byRole.put(key, id);
                if ("mask_token".equals(key)) {
                    mask = content;
                }
            }
        }
        this.roles = Map.copyOf(byRole);
        this.maskToken = mask;
    }

    /**
     * One special token's text from {@code tokenizer_config.json}.
     *
     * <p>The field is a bare string in both laya checkpoints, but the format also allows the
     * expanded {@code AddedToken} object form ({@code {"content": "[MASK]", "lstrip": true, ...}}),
     * which newer {@code transformers} versions write. Both are read, because a checkpoint saved by
     * a different version is still the same checkpoint.
     */
    private static String specialTokenContent(Map<String, Object> config, String key) {
        Object value = config.get(key);
        if (value == null) {
            return null;
        }
        if (value instanceof String) {
            return (String) value;
        }
        if (value instanceof Map) {
            Object content = ((Map<?, ?>) value).get("content");
            if (content instanceof String) {
                return (String) content;
            }
        }
        throw new Json.JsonException(
                key + " in tokenizer_config.json must be a string or an object with a content "
                + "field, found " + value.getClass().getSimpleName());
    }

    // ------------------------------------------------------------------ encoding

    /**
     * Token ids for {@code text}, with no special tokens added.
     *
     * <p>Special tokens are the sequence builder's business, not the tokenizer's: laya composes a
     * question and its options into one sequence and decides where a mask goes, and a tokenizer
     * that silently prepended a classification token would corrupt that composition.
     */
    public int[] encode(String text) {
        // Delegating rather than duplicating: an unlimited encode and a truncating one that each
        // walked the pipeline themselves would be two implementations of the same contract, free to
        // drift apart under a later edit with nothing to catch it.
        return encode(text, -1);
    }

    /**
     * Token ids for {@code text}, stopping once {@code limit} ids have been produced.
     *
     * <p>Equivalent to encoding in full and keeping the first {@code limit} ids -- pre-tokenized
     * pieces are merged independently, so a prefix of the pieces yields a prefix of the ids -- but
     * it stops early instead of merging a criterion's whole tail. This is the
     * {@code truncation=True, max_length=48} cap {@code build_head} applies per option, whose own
     * comment records the same reasoning.
     *
     * @param limit the maximum number of ids, or a negative value for no limit
     */
    public int[] encode(String text, int limit) {
        if (limit == 0) {
            return new int[0];
        }
        List<Integer> out = new ArrayList<>(limit < 0 ? Math.max(16, text.length() / 3) : limit);
        for (AddedVocabulary.Span span : rawAdded.split(text)) {
            if (limit >= 0 && out.size() >= limit) {
                break;
            }
            if (span.isAddedToken()) {
                out.add(span.tokenId());
                continue;
            }
            String normalized = normalizer.normalize(span.text());
            if (normalizedAdded.isEmpty()) {
                encodePlain(normalized, out, limit);
                continue;
            }
            for (AddedVocabulary.Span inner : normalizedAdded.split(normalized)) {
                if (limit >= 0 && out.size() >= limit) {
                    break;
                }
                if (inner.isAddedToken()) {
                    out.add(inner.tokenId());
                } else {
                    encodePlain(inner.text(), out, limit);
                }
            }
        }
        int size = limit < 0 ? out.size() : Math.min(out.size(), limit);
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            result[i] = out.get(i);
        }
        return result;
    }

    private void encodePlain(String text, List<Integer> out, int limit) {
        for (String piece : preTokenizer.preTokenize(text)) {
            if (limit >= 0 && out.size() >= limit) {
                return;
            }
            bpe.encodePiece(piece, out);
        }
    }

    // ------------------------------------------------------------------ lookups

    /** The number of distinct ids this tokenizer can emit, counting added tokens. */
    /**
     * These ids back into text, dropping special tokens -- the reference's default.
     *
     * <p>Used by {@code predictLong}, which splits a long state into overlapping token windows and
     * hands each one back as text so it is re-tokenized as an ordinary state.
     *
     * <p><b>This does not round-trip, and is not meant to.</b> On the english checkpoint it
     * happens to; on the multilingual one {@code "Hello world"} decodes to {@code " Hello world"},
     * because Metaspace prepends U+2581 when encoding and the decoder turns every U+2581 back
     * into a space. That is the reference's behaviour and the port reproduces it exactly -- a
     * decode that trimmed the space would re-tokenize each window differently and move every
     * answer downstream.
     */
    public String decode(int[] ids) {
        return decode(ids, true);
    }

    /**
     * These ids back into text.
     *
     * @param ids                the token ids, in order
     * @param skipSpecialTokens  drop ids flagged {@code special} in {@code added_tokens}, which is
     *                           what the reference does unless told otherwise
     */
    public String decode(int[] ids, boolean skipSpecialTokens) {
        if (ids == null) {
            throw new IllegalArgumentException("ids must not be null");
        }
        List<String> pieces = new ArrayList<>(ids.length);
        for (int id : ids) {
            if (skipSpecialTokens && specialIds.contains(id)) {
                continue;
            }
            String token = idToToken(id);
            if (token == null) {
                // An id outside the vocabulary. The crate skips it; failing here would turn a
                // caller's off-by-one into an exception from a method whose job is to be lossy.
                continue;
            }
            pieces.add(token);
        }
        return decoder.decode(pieces);
    }

    public int vocabSize() {
        return ids.size();
    }

    /** The highest id this tokenizer can emit, which may exceed {@link #vocabSize()} minus one. */
    public int highestId() {
        return tokens.length - 1;
    }

    /** The id for {@code token}, or empty when the tokenizer has no such entry. */
    public OptionalInt tokenToId(String token) {
        Integer id = ids.get(token);
        return id == null ? OptionalInt.empty() : OptionalInt.of(id);
    }

    /** The token for {@code id}, or null when the id is unused or out of range. */
    public String idToToken(int id) {
        return id < 0 || id >= tokens.length ? null : tokens[id];
    }

    /**
     * The id of a special token by its content, or empty when this checkpoint ships none.
     *
     * <p>Empty is a real case, not a defensive one: the multilingual checkpoint's laya config
     * declares no classification, separator, mask or padding token at all, where the english one
     * declares all four. Callers must handle the absence rather than assume a checkpoint has them.
     */
    public OptionalInt specialId(String content) {
        Integer id = specials.get(content);
        return id == null ? OptionalInt.empty() : OptionalInt.of(id);
    }

    /** The special tokens this checkpoint declares, in file order. */
    public Map<String, Integer> specialTokens() {
        return specials;
    }

    /** The classification token's id, as {@code tokenizer_config.json} names it. */
    public OptionalInt clsId() {
        return role("cls_token");
    }

    /** The separator token's id. */
    public OptionalInt sepId() {
        return role("sep_token");
    }

    /** The mask token's id, which marks each option's position in a sequence. */
    public OptionalInt maskId() {
        return role("mask_token");
    }

    /** The padding token's id. */
    public OptionalInt padId() {
        return role("pad_token");
    }

    /**
     * The mask token's text, which must be scrubbed out of caller-supplied text.
     *
     * <p>Null when the checkpoint names none. A caller who can leave this string in an instruction,
     * an option or a state can inject option markers into the sequence, which is why
     * {@code build_head} replaces it with a space in all three.
     */
    public String maskToken() {
        return maskToken;
    }

    private OptionalInt role(String key) {
        Integer id = roles.get(key);
        return id == null ? OptionalInt.empty() : OptionalInt.of(id);
    }

    // ------------------------------------------------------------------ load-time checks

    /**
     * Every merge's two components and its result must be vocabulary entries.
     *
     * <p>This is what makes {@link Bpe} able to resolve ids <i>after</i> merging instead of
     * classifying each character first: a symbol with no vocabulary entry cannot appear in any
     * merge pair, so it cannot have merged with anything. Measured across both laya checkpoints --
     * 630,613 merges, zero absent components and zero absent results -- and checked here so the
     * equivalence is enforced rather than assumed of some future checkpoint.
     */
    private static void checkMergeClosure(List<String[]> merges, Map<String, Integer> vocab) {
        for (int rank = 0; rank < merges.size(); rank++) {
            String[] pair = merges.get(rank);
            for (String component : pair) {
                if (!vocab.containsKey(component)) {
                    throw new Json.JsonException(
                            "merge at rank " + rank + " uses the symbol \"" + component
                            + "\", which the vocabulary does not contain. laya-java resolves ids "
                            + "after merging, which this would make unsound.");
                }
            }
            if (!vocab.containsKey(pair[0] + pair[1])) {
                throw new Json.JsonException(
                        "merge at rank " + rank + " produces \"" + pair[0] + pair[1]
                        + "\", which the vocabulary does not contain.");
            }
        }
    }

    /**
     * Refuses a checkpoint that could silently drop text.
     *
     * <p>A model with no unknown token and no byte fallback can only be safe if every symbol its
     * pre-tokenizer can produce is a vocabulary entry. That holds for byte-level pre-tokenization,
     * whose 256-character alphabet is finite and checkable -- and the english checkpoint is exactly
     * that case, declaring no {@code unk_token} at all. For any other pre-tokenizer the input
     * alphabet is unbounded, so the combination is rejected at load time rather than discovered as
     * missing text in a request.
     */
    private void checkEveryCharacterIsRepresentable() {
        if (bpe.unknownToken() != null || bpe.hasByteFallback()) {
            return;
        }
        if (!(preTokenizer instanceof PreTokenizer.ByteLevel)) {
            throw new Json.JsonException(
                    "this checkpoint declares no unk_token and no byte_fallback, and its "
                    + "pre-tokenizer is not byte-level, so a character outside the vocabulary "
                    + "would be dropped without trace. Refusing to load.");
        }
        List<String> reachable = PreTokenizer.ByteLevel.alphabetReachableFromText();
        List<String> missing = new ArrayList<>();
        for (String symbol : reachable) {
            if (!ids.containsKey(symbol)) {
                missing.add(symbol);
            }
        }
        if (!missing.isEmpty()) {
            throw new Json.JsonException(
                    "this checkpoint declares no unk_token and no byte_fallback, and " + missing.size()
                    + " of the " + reachable.size() + " byte-level alphabet symbols that text can "
                    + "reach are absent from its vocabulary (first: \"" + missing.get(0)
                    + "\"), so some inputs would lose text silently. Refusing to load.");
        }
    }

    // ------------------------------------------------------------------ reading the document

    private static List<AddedToken> readAddedTokens(Map<String, Object> document) {
        Object node = document.get("added_tokens");
        if (node == null) {
            return List.of();
        }
        if (!(node instanceof List)) {
            throw new Json.JsonException("added_tokens must be an array");
        }
        List<AddedToken> out = new ArrayList<>();
        Map<Integer, String> seen = new HashMap<>();
        for (Object entry : (List<?>) node) {
            if (!(entry instanceof Map)) {
                throw new Json.JsonException("an added_tokens entry must be an object");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> token = (Map<String, Object>) entry;
            String content = Json.string(token, "content", null);
            if (content == null) {
                throw new Json.JsonException("an added_tokens entry must declare content");
            }
            int id = Json.integer(token, "id", -1);
            String clash = seen.put(id, content);
            if (clash != null && !clash.equals(content)) {
                throw new Json.JsonException(
                        "added token id " + id + " is claimed by both \"" + clash + "\" and \""
                        + content + "\"");
            }
            out.add(new AddedToken(
                    content,
                    id,
                    Json.bool(token, "special", false),
                    Json.bool(token, "lstrip", false),
                    Json.bool(token, "rstrip", false),
                    Json.bool(token, "single_word", false),
                    Json.bool(token, "normalized", false)));
        }
        return out;
    }

    /**
     * One merge entry, which the two laya checkpoints both spell as a two-element array.
     *
     * <p>The older space-joined string form is accepted too, because the format carries both and a
     * checkpoint in the wild may use either -- but it is split on the <i>first</i> space only, and
     * rejected if that does not yield two non-empty halves, rather than guessed at.
     */
    private static String[] readMerge(Object entry) {
        if (entry instanceof List) {
            List<?> pair = (List<?>) entry;
            if (pair.size() != 2 || !(pair.get(0) instanceof String) || !(pair.get(1) instanceof String)) {
                throw new Json.JsonException("a merge must be two strings, got " + pair);
            }
            return new String[] {(String) pair.get(0), (String) pair.get(1)};
        }
        if (entry instanceof String) {
            String text = (String) entry;
            int space = text.indexOf(' ');
            if (space <= 0 || space == text.length() - 1) {
                throw new Json.JsonException(
                        "a space-joined merge must have two non-empty halves, got \"" + text + "\"");
            }
            return new String[] {text.substring(0, space), text.substring(space + 1)};
        }
        throw new Json.JsonException("a merge must be an array of two strings or a joined string");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> objectOrNull(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value == null) {
            return null;
        }
        if (!(value instanceof Map)) {
            throw new Json.JsonException(key + " must be an object or null");
        }
        return (Map<String, Object>) value;
    }
}
