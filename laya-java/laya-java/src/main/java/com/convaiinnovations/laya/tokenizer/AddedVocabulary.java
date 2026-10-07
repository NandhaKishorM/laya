package com.convaiinnovations.laya.tokenizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Splits raw input on a checkpoint's added tokens before the normalizer and BPE see it.
 *
 * <p>Matching is <b>leftmost-longest</b>, which is not interchangeable with first-match: the added
 * tokens overlap heavily -- 253 of the english checkpoint's and 1,335 of the multilingual one's are
 * substrings of another added token. The english checkpoint carries a separate added token for a run
 * of 2 spaces, 3 spaces, ... up to 24, and the multilingual one for runs of newlines. First-match
 * would cut a run of 24 spaces into twelve 2-space tokens; leftmost-longest emits the one token the
 * reference implementation emits.
 *
 * <p>Verified against the reference implementation: {@code "a   b"} yields the 3-space token, and a
 * 24-space run yields the single 24-space token.
 */
final class AddedVocabulary {

    /** One output span: either a resolved added token, or raw text still to be tokenized. */
    record Span(String text, Integer tokenId) {
        boolean isAddedToken() {
            return tokenId != null;
        }
    }

    private final Map<Character, List<AddedToken>> byFirstChar;
    private final boolean empty;

    /**
     * @param tokens the added tokens to match; pass only those whose {@code normalized} flag suits
     *               the stage this instance runs at
     */
    AddedVocabulary(List<AddedToken> tokens) {
        this.byFirstChar = new HashMap<>();
        for (AddedToken token : tokens) {
            byFirstChar.computeIfAbsent(token.content().charAt(0), key -> new ArrayList<>())
                    .add(token);
        }
        // Longest first within a bucket, so the first hit at a position IS the longest hit. The
        // comparator breaks length ties on content so the order is total and the split is
        // reproducible across runs regardless of the table's file order.
        Comparator<AddedToken> longestFirst =
                Comparator.comparingInt((AddedToken t) -> t.content().length()).reversed()
                        .thenComparing(AddedToken::content);
        for (List<AddedToken> bucket : byFirstChar.values()) {
            bucket.sort(longestFirst);
        }
        this.empty = tokens.isEmpty();
    }

    /** Whether this stage has anything to match, so callers can skip the scan entirely. */
    boolean isEmpty() {
        return empty;
    }

    /**
     * Cuts {@code text} into added-token spans and the raw text between them.
     *
     * <p>Spans are returned in input order and concatenating their text reproduces the input,
     * <i>except</i> for whitespace absorbed by an {@code lstrip} or {@code rstrip} token, which is
     * deliberately dropped. Both laya checkpoints set {@code lstrip} on their mask token, and the
     * reference implementation discards the whitespace it absorbs: {@code "hi   [MASK]"} encodes to
     * two tokens, with the three spaces gone.
     */
    List<Span> split(String text) {
        if (empty || text.isEmpty()) {
            return text.isEmpty() ? List.of() : List.of(new Span(text, null));
        }
        List<Span> out = new ArrayList<>();
        int plainFrom = 0;          // start of the pending run of unmatched text
        int at = 0;
        while (at < text.length()) {
            AddedToken hit = longestAt(text, at);
            if (hit == null) {
                at++;
                continue;
            }
            int start = at;
            int stop = at + hit.content().length();
            if (hit.lstrip()) {
                // Absorb whitespace to the left, but never back past text already emitted: a
                // previous added token's span is settled and must not be swallowed.
                while (start > plainFrom && Unicode.isWhiteSpace(text.charAt(start - 1))) {
                    start--;
                }
            }
            if (hit.rstrip()) {
                while (stop < text.length() && Unicode.isWhiteSpace(text.charAt(stop))) {
                    stop++;
                }
            }
            if (start > plainFrom) {
                out.add(new Span(text.substring(plainFrom, start), null));
            }
            out.add(new Span(hit.content(), hit.id()));
            plainFrom = stop;
            at = stop;
        }
        if (plainFrom < text.length()) {
            out.add(new Span(text.substring(plainFrom), null));
        }
        return out;
    }

    /** The longest added token matching at {@code at}, or null. */
    private AddedToken longestAt(String text, int at) {
        List<AddedToken> bucket = byFirstChar.get(text.charAt(at));
        if (bucket == null) {
            return null;
        }
        for (AddedToken token : bucket) {           // longest first
            String content = token.content();
            if (!text.startsWith(content, at)) {
                continue;
            }
            if (token.singleWord() && !standsAlone(text, at, at + content.length())) {
                continue;
            }
            return token;
        }
        return null;
    }

    /**
     * Whether an occurrence is flanked by non-word characters, for {@code single_word} tokens.
     *
     * <p>Neither laya checkpoint sets {@code single_word} on any of its 365 added tokens, so this
     * branch is unexercised by them. It is implemented rather than rejected because a checkpoint
     * that did set it would otherwise be tokenized wrongly with no signal at all.
     */
    private static boolean standsAlone(String text, int start, int stop) {
        boolean leftClear = start == 0 || !Unicode.isAlphanumeric(text.charAt(start - 1));
        boolean rightClear = stop >= text.length() || !Unicode.isAlphanumeric(text.charAt(stop));
        return leftClear && rightClear;
    }
}
