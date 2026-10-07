package com.convaiinnovations.laya.tokenizer;

/**
 * One entry of a checkpoint's {@code added_tokens} table.
 *
 * <p>These are matched against the input <b>before</b> BPE sees it and are never merged, which is
 * why they are not simply vocabulary entries. Two properties of the laya checkpoints make this table
 * load-bearing rather than decorative:
 *
 * <ul>
 *   <li>In the english checkpoint <b>88 of the 116 added tokens are absent from {@code model.vocab}
 *       altogether</b>, carrying ids 50280-50367 above the vocabulary's own maximum of 50279 --
 *       {@code [MASK]} (50284) among them. A loader that built its id map from {@code model.vocab}
 *       alone would have no id for the token masked prediction depends on.
 *   <li>The contents <b>overlap</b>: 253 added tokens in the english checkpoint and 1,335 in the
 *       multilingual one are substrings of another added token (a run of three spaces inside a run
 *       of four; {@code "\n"} inside {@code "\n\n"}). Matching must therefore be leftmost-<i>longest</i>.
 * </ul>
 *
 * @param content    the literal text matched
 * @param id         the token id, authoritative even when {@code model.vocab} disagrees or is silent
 * @param special    whether the token is a control token rather than ordinary text
 * @param lstrip     absorb whitespace immediately to the left of a match into it, and discard it
 * @param rstrip     the same to the right
 * @param singleWord match only when the occurrence is not flanked by word characters
 * @param normalized match against normalized text rather than the raw input
 */
record AddedToken(
        String content,
        int id,
        boolean special,
        boolean lstrip,
        boolean rstrip,
        boolean singleWord,
        boolean normalized) {

    AddedToken {
        if (content == null || content.isEmpty()) {
            throw new IllegalArgumentException("an added token cannot have empty content");
        }
        if (id < 0) {
            throw new IllegalArgumentException("added token " + content + " has a negative id: " + id);
        }
    }
}
