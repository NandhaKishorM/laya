"""Regression: Agent.system_one preserves the newest conversation turn.

The public Agent API advertises `state` as accepting a chronological
conversation-turn list. build_sequence defaults to truncate_left=False
(st[:room]), which preserves the head and drops the tail — for a chronological
list that means the newest turn is silently lost. For list-shaped state the agent
now truncates from the left so the most recent intent survives. Strings and
dicts are unaffected (backward compatible).
"""
import re
import zlib

import torch
from transformers import AutoConfig, AutoModel

from laya import common
from laya.agent import Agent
from laya.common import DecisionModel, build_sequence, serialize_state


class WhitespaceSplit:
    """Stands in for the real `tokenizers` pre-tokenizer of the same name.

    `_pre_tokenizer_permits_cut` keys on the class name, exactly as it does for the shipped
    ByteLevel and Metaspace, so the name here is load-bearing and not decoration.
    """


class _KeepsTheSpace:
    """Rewrites a space to U+2581 one-for-one, as the multilingual checkpoint's normalizer does."""

    def normalize_str(self, text):
        return text.replace(" ", "\u2581")


class _DropsWhitespace:
    """Deletes whitespace, as a CJK-oriented finetune's normalizer plausibly does."""

    def normalize_str(self, text):
        return "".join(text.split())


class _NotPrefixPreserving:
    """Keeps a space where the probe expects one, but what precedes the cut is not a prefix.

    Contrived -- reversal -- but it is the one shape the space check alone cannot see: the cut
    character survives in the right position while the text before it normalizes to something else
    entirely, so slicing there does not slice the normalized form.
    """

    def normalize_str(self, text):
        return text[::-1]


class _Backend:
    """The `backend_tokenizer` a fast tokenizer exposes; `pre_tokenizer` and `normalizer` are read."""

    pre_tokenizer = WhitespaceSplit()
    normalizer = None


class _MetaspaceLikeBackend(_Backend):
    """Normalizes the way multilingual does -- the cut survives as U+2581, so cutting is safe."""

    normalizer = _KeepsTheSpace()


class _StrippingBackend(_Backend):
    """Deletes the cut character, so no boundary is left for the pre-tokenizer to split on."""

    normalizer = _DropsWhitespace()


class _ReversingBackend(_Backend):
    """Keeps the cut character but does not preserve what comes before it."""

    normalizer = _NotPrefixPreserving()


class _NullBackend:
    """A T5/Llama-shaped tokenizer: Metaspace lives in the normalizer and nothing splits."""

    pre_tokenizer = None
    normalizer = None


class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"
    backend_tokenizer = _Backend()

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _tiny_agent():
    agent = object.__new__(Agent)
    cfg = AutoConfig.for_model("bert", hidden_size=16, num_hidden_layers=1, num_attention_heads=1,
                               intermediate_size=32, vocab_size=64)
    agent.cfg = {"max_len": 30, "head_max_len": 12, "encoder": "tiny"}
    agent.tok = _FakeTok()
    agent.model = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2).eval()
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    return agent


def _state_prefix_len(tok, q, max_len, head_max_len):
    ref, _ = build_sequence(tok, "", q, max_len, head_max_len)
    return len(ref) - 1


Q = {"t": "choice", "ins": "What action?", "crit": {"refund": "money back", "escalate": "manager", "hold": "wait"}}


def test_agent_list_vs_string_truncation_direction():
    """Agent must truncate left for lists, right for strings (via mock inspection)."""
    from unittest.mock import patch

    agent = _tiny_agent()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    list_capture = {}
    string_capture = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        if isinstance(state, list):
            list_capture["truncate_left"] = truncate_left
        else:
            string_capture["truncate_left"] = truncate_left
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one([{"role": "user", "content": "hi"}, {"role": "user", "content": "newest"}], questions)
        agent.system_one("string state", questions)

    assert list_capture["truncate_left"] is True
    assert string_capture["truncate_left"] is False


def test_agent_system_one_passes_truncate_left_for_list():
    """Agent.system_one must pass truncate_left=True for list state."""
    from unittest.mock import patch

    agent = _tiny_agent()
    conversation = [{"role": "user", "content": "hello"}, {"role": "user", "content": "NEWEST"}]
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        captured["truncate_left"] = truncate_left
        captured["state"] = state
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one(conversation, questions)

    assert captured["truncate_left"] is True, "list state must trigger truncate_left=True"
    assert isinstance(captured["state"], list)


def test_agent_system_one_string_state_default_truncation():
    """Agent.system_one with string state keeps default (truncate_left=False)."""
    from unittest.mock import patch

    agent = _tiny_agent()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}

    captured = {}

    def fake_build(tok, state, q, max_len, head_max_len, truncate_left=False, **kwargs):
        captured["truncate_left"] = truncate_left
        return build_sequence(tok, state, q, max_len, head_max_len,
                              truncate_left=truncate_left, **kwargs)

    with patch("laya.agent.build_sequence", side_effect=fake_build):
        agent.system_one("just a string state", questions)

    assert captured["truncate_left"] is False, "string state must keep default truncation"


def test_string_state_preserves_head():
    """A string state must still preserve the head (backward compatible)."""
    tok = _FakeTok()
    state = "HEADMARKERWORD " + "filler " * 20 + " TAILMARKER"
    state_ids = tok(state.replace("[MASK]", " "), add_special_tokens=False)["input_ids"]
    head_id = state_ids[0]
    tail_id = state_ids[-1]
    assert head_id != tail_id, "head and tail must have different ids"

    seq, _ = build_sequence(tok, state, Q, 30, 12)
    prefix = _state_prefix_len(tok, Q, 30, 12)
    kept = seq[prefix:-1]

    assert head_id in kept, "string state must preserve the head (default mode)"
    assert tail_id not in kept, "string state must drop the tail (default mode)"


def test_build_sequence_default_unchanged():
    """build_sequence's default behavior is unchanged for non-list callers."""
    tok = _FakeTok()
    state = "OLDFRONT " + "filler " * 20 + " NEWBACK"
    seq_default, _ = build_sequence(tok, state, Q, 30, 12)
    seq_explicit, _ = build_sequence(tok, state, Q, 30, 12, truncate_left=False)
    assert seq_default == seq_explicit, "default must remain truncate_left=False"


def test_predict_batch_usage_reports_truncation():
    """Agent.predict_batch reports each state's truncation from the budget build_sequence applied (#174).

    Two heads of different length over one budget: a 25-word state is cut only for the longer
    head, a 30-word state for both but by different amounts, so the per-question accounting and
    the max across questions are both visible.
    """
    agent = _tiny_agent()
    questions = {
        "refund": {"type": "noul", "instructions": "Refund?"},
        "route": {"type": "choice", "instructions": "Which team should own this request given the contract?",
                  "criteria": {"billing": "invoices and refunds", "tech": "bugs and outages"}},
    }
    states = [" ".join("w%d" % i for i in range(n)) for n in (10, 25, 30)]
    results = agent.predict_batch(states, questions, max_len=48, head_max_len=32)

    for state, result in zip(states, results):
        stats = {qid: build_sequence(agent.tok, state, Agent._to_internal(q), 48, 32,
                                     return_truncation_stats=True)[2]
                 for qid, q in questions.items()}
        usage = result["usage"]
        assert usage["state_tokens"] == len(state.split())
        assert usage["state_tokens_dropped"] == max(s["state_tokens_dropped"] for s in stats.values())
        assert usage["truncated"] is (usage["state_tokens_dropped"] > 0)
        assert usage["truncated_questions"] == [qid for qid in questions if stats[qid]["truncated"]]

    fits, one, both = (r["usage"] for r in results)
    assert fits["truncated"] is False and fits["truncated_questions"] == []
    assert one["truncated"] is True and one["truncated_questions"] == ["route"]
    assert both["truncated_questions"] == ["refund", "route"]
    dropped = [build_sequence(agent.tok, states[2], Agent._to_internal(q), 48, 32,
                              return_truncation_stats=True)[2]["state_tokens_dropped"]
               for q in questions.values()]
    assert dropped[0] < dropped[1] == both["state_tokens_dropped"], dropped

# --- the state head: tokenize only as far as a prefix-truncated sequence can hold -------------
#
# `Agent._encode_state` used to tokenize the whole serialized state and let `build_sequence`
# keep `state_ids[:room]`. For every state shape but a list that slice is a prefix, so the rest
# of the document was tokenized only to be thrown away. `encode_state_head` stops early. These
# checks are the parity contract: the ids that reach the model must not move, and a list must
# still be tokenized in full so left-truncation keeps seeing the newest turn.

_PRE_TOKEN = re.compile(r"\s+|\S+")


class _PreTokTok(_FakeTok):
    """A tokenizer whose ids depend on a whole piece, not on single characters.

    `_FakeTok` above maps each whitespace-separated word to a length, which hides the failure
    mode that matters: real ids are a function of a whole piece, so a slice that cuts one in half
    changes them. Both shipped checkpoints also carry added tokens that are runs of one
    whitespace character (2-24 spaces on english, 1-31 newlines or tabs on multilingual) and
    match them greedily, which makes a whitespace run's own ids depend on how long the run is --
    modelled here by giving every maximal run of whitespace an id of its own. A cut point that is
    wrong for the real tokenizers is wrong here too.
    """

    def __init__(self):
        self.texts = []

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        self.texts.append(text)
        ids = [5 + (zlib.crc32(t.encode()) % 59) for t in _PRE_TOKEN.findall(text)]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def get_added_vocab(self):
        """Whitespace runs only, as both shipped checkpoints have -- so cutting is safe."""
        return {" " * n: 200 + n for n in range(2, 25)}


class _MetaspacePhraseTok(_PreTokTok):
    """A sentencepiece-derived checkpoint: its added phrase spells the space as U+2581.

    HF matches a `normalized=True` added token against the *normalized* text, and a Metaspace-style
    normalizer has already rewritten " " to U+2581 by then -- so the phrase arrives as
    `\u2581New\u2581York` and contains no U+0020 at all. A gate that only looks for U+0020 lets it
    straight through, which is measurably wrong: the head stops being a prefix of the full ids.
    """

    def get_added_vocab(self):
        return dict(_PreTokTok.get_added_vocab(self),
                    **{"\u2581New\u2581York": 310, "\u2581order\u2581id": 311})


class _AddedPhraseTok(_PreTokTok):
    """A checkpoint that added a domain phrase, so an added token holds an *internal* space.

    `laya` loads any directory a training run wrote, so this is a reachable configuration rather
    than a hypothetical, and it breaks `_state_cut`'s rule: the trie matches "New York" greedily
    across a space whose predecessor is a letter, which is a cut the rule accepts.
    """

    def get_added_vocab(self):
        return dict(_PreTokTok.get_added_vocab(self), **{"New York": 300, "order id": 301})


class _StripNormalizerTok(_PreTokTok):
    """Ordinary whitespace-splitting pre-tokenizer, but the normalizer deletes the cut."""

    backend_tokenizer = _StrippingBackend()


class _ReversingNormalizerTok(_PreTokTok):
    """Normalizer keeps the cut character but is not prefix-preserving."""

    backend_tokenizer = _ReversingBackend()


class _MetaspaceLikeTok(_PreTokTok):
    """Normalizer rewrites the cut to U+2581 rather than dropping it, so the cut is still safe."""

    backend_tokenizer = _MetaspaceLikeBackend()


class _NoSplitTok(_PreTokTok):
    """A checkpoint whose pre-tokenizer does not split on whitespace at all.

    T5 and Llama are shaped this way: `Metaspace` sits in the *normalizer* as `Prepend` +
    `Replace`, `pre_tokenizer` is null, so BPE merges run over the whole string and the vocabulary
    holds pieces spanning what used to be a space. Cutting inside one changes the ids before the
    cut, which no length check can detect.
    """

    backend_tokenizer = _NullBackend()


def _states():
    """(label, state) over every shape `_encode_state` accepts, on both sides of the budget."""
    big = common.STATE_HEAD_CHARS_PER_TOKEN * 30  # comfortably past the budget at max_len=30
    return [
        ("empty string", ""),
        ("one word", "refund"),
        ("no space at all", "x" * big),
        # Cuttable, but far denser in characters per token than any real text: every cut in
        # reach of the budget leaves a prefix short of it, which is the one case
        # `encode_state_head`'s `len(head) >= need` guard exists for. Real states cannot be
        # this dense on the multilingual tokenizer (31 chars is its longest token surface),
        # but the english one carries whitespace- and dash-run tokens of up to 512 characters.
        ("dense but cuttable", ("-" * 256 + " ") * 320),
        ("short prose", "the customer wants a refund on order 41"),
        # Carries the mask token, so the `.replace(mask_token, " ")` that `_encode_state` applies
        # is actually exercised rather than being a no-op on every shape.
        ("mask token in the state", "refund [MASK] escalate [MASK] hold " * 40),
        ("prose past the budget", "refund escalate hold " * big),
        ("leading whitespace", "   \n\t  refund the order " * 40),
        ("whitespace runs", "refund" + " " * 200 + "escalate" + "\n" * 200 + "hold " * 400),
        ("space then newline runs", "refund \n" * 900),
        ("tabs only", "refund\tescalate\thold\t" * 600),
        ("unicode", "客户要求退款 \U0001f621 ответственность " * 400),
        ("unicode combining", "état café " * 700),
        ("dict", {"customer": "angry", "order": 41}),
        ("dict past the budget", {"f%d" % i: "value %d" % i for i in range(big)}),
        ("nested dict", {"a": {"b": {"c": ["d", {"e": "f" * 200}]}}, "g": list(range(big))}),
        ("dict of unicode", {"ключ %d" % i: "значение %d" % i for i in range(big)}),
        ("list of turns", [{"role": "user", "content": "hello"}, {"role": "user", "content": "NEWEST"}]),
        ("list past the budget", [{"role": "user", "content": "turn %d" % i} for i in range(big)]),
        ("list of strings", ["oldest turn"] + ["filler turn %d" % i for i in range(big)] + ["newest turn"]),
    ]


def test_state_head_ids_match_a_full_tokenization():
    """encode_state_head returns the full ids, or a prefix of them that covers the budget."""
    compared = 0
    for tok in (_FakeTok(), _PreTokTok()):
        for need in (1, 8, 30, 512):
            for label, state in _states():
                # Exactly what `_encode_state` builds -- including the mask replacement, without
                # which this asserts the contract on input production never passes.
                text = serialize_state(state).replace(tok.mask_token, " ")
                full = tok(text, add_special_tokens=False)["input_ids"]
                head = common.encode_state_head(tok, text, need)
                assert head == full[:len(head)], "%s (need=%d): head ids diverge from the full ids" % (label, need)
                assert len(head) >= min(need, len(full)), \
                    "%s (need=%d): head has %d ids, short of the budget and not the whole state" \
                    % (label, need, len(head))
                compared += 1
    assert compared == 2 * 4 * len(_states()) == 160, compared


def test_state_head_stops_before_the_end_of_a_large_state():
    """The point of the change: a state far past the budget is not tokenized in full."""
    tok = _PreTokTok()
    text = "refund escalate hold " * 5000
    head = common.encode_state_head(tok, text, 512)
    assert len(head) >= 512
    assert text not in tok.texts, \
        "the whole state was handed to the tokenizer; the head budget did nothing"
    # A small density probe, then one prefix sized from what it measured. Two tokenizations is the
    # design, not a retry -- and the probe must stay small, or its own cost stops being noise.
    assert len(tok.texts) == 2, "expected a probe and one prefix, got %d" % len(tok.texts)
    assert len(tok.texts[0]) <= common._STATE_HEAD_PROBE_CHARS, \
        "the probe is meant to be a fraction of the budget, not a full attempt"
    assert sum(len(t) for t in tok.texts) < len(text), \
        "the head path must tokenize less than the state, in total"


def test_state_head_falls_back_to_the_full_state_when_the_prefix_is_short():
    """A state with no cut point in reach is tokenized in full -- correct, just not faster."""
    tok = _PreTokTok()
    text = "x" * 40000
    head = common.encode_state_head(tok, text, 512)
    assert head == tok(text)["input_ids"]
    assert text in tok.texts, "the fallback must tokenize the whole state"


def test_state_head_does_not_pay_for_a_prefix_it_will_not_use():
    """A state too dense to fill the budget from a prefix must cost the probe and nothing more.

    This is the regression the probe exists to prevent. Sizing the first attempt at
    `need * STATE_HEAD_CHARS_PER_TOKEN` characters and only then discovering the state is denser
    meant tokenizing a large prefix, throwing it away, and tokenizing the whole state as well --
    1.64x the characters of the state, measured, for a 0.57x slowdown against simply tokenizing it.
    So the density the probe measures decides whether a prefix is worth taking at all.
    """
    tok = _PreTokTok()
    # 64 dash characters per token -- far denser than any real text -- and cuttable often enough
    # that a cut point falls inside the probe window, so the probe really does run.
    text = ("-" * 64 + " ") * 160
    assert len(text) > 512 * common.STATE_HEAD_CHARS_PER_TOKEN * common._STATE_HEAD_MIN_RATIO, \
        "the fixture must be past the gate, or this asserts nothing"

    expected = tok(text)["input_ids"]     # taken first: calling the tokenizer records a text
    tok.texts.clear()

    head = common.encode_state_head(tok, text, 512)
    assert head == expected, "the result must still be the full ids"
    assert len(tok.texts) == 2, \
        "expected a probe and then the full state, got %d tokenizations" % len(tok.texts)
    assert len(tok.texts[0]) <= common._STATE_HEAD_PROBE_CHARS, "the first must be the small probe"
    assert tok.texts[1] == text, "the second must be the full state"
    # The whole point: total work stays close to the state itself.
    assert sum(len(t) for t in tok.texts) < len(text) * 1.1, \
        "the head path cost %.2fx the state" % (sum(len(t) for t in tok.texts) / len(text))

    # And the decision is taken before the soundness gates, which is the other half of the fix.
    # A dense state tokenizes few ids from many characters, so its own tokenization is cheap and
    # the gates' fixed 53-91 us showed up as a 0.82x regression when they were consulted first.
    calls = []
    original = common._added_tokens_permit_cut
    common._added_tokens_permit_cut = lambda t: (calls.append(1), original(t))[1]
    try:
        tok2 = _PreTokTok()
        common.encode_state_head(tok2, text, 512)
    finally:
        common._added_tokens_permit_cut = original
    assert not calls, \
        "the gates were consulted for a state that was never going to be cut (%d calls)" % len(calls)


def test_added_token_gate_fails_closed_and_covers_both_spellings():
    """The gate must refuse what it cannot inspect, and treat U+2581 as a space.

    Two holes found by review. A tokenizer with no `get_added_vocab` was answered `True` -- "cannot
    establish the property" answered as "property holds" -- while the pre-tokenizer and normalizer
    gates both answer `False` for the same situation. And a sentencepiece-derived added token spells
    the cut U+2581, so a U+0020-only scan passed it.
    """
    class NoAddedVocabAPI(_PreTokTok):
        get_added_vocab = None

    tok = _PreTokTok()
    del tok  # only the classes matter here
    bare = NoAddedVocabAPI()
    assert common._added_tokens_permit_cut(bare) is False, \
        "a tokenizer whose added tokens cannot be enumerated must not be cut"
    assert common._added_tokens_permit_cut(_PreTokTok()) is True
    assert common._added_tokens_permit_cut(_AddedPhraseTok()) is False        # U+0020 spelling
    assert common._added_tokens_permit_cut(_MetaspacePhraseTok()) is False    # U+2581 spelling

    # And it still falls back correctly end to end for the U+2581 variant.
    text = "shipped to New York and billed to New York " * 400
    unsafe = _MetaspacePhraseTok()
    assert common.encode_state_head(unsafe, text, 512) == unsafe(text)["input_ids"]


def test_state_head_retry_does_not_repeat_an_identical_prefix():
    """A retry that cannot advance must stop, not pay for the same prefix again.

    `_state_cut` clamps to the last space at or before the budget, so a larger budget can select the
    same cut. With no further space in the text, every attempt would tokenize an identical prefix:
    measured as probe + three identical 100-character attempts + the full state before this.
    """
    tok = _PreTokTok()
    text = "x" * 100 + " " + "y" * 200000
    expected = tok(text)["input_ids"]
    tok.texts.clear()

    head = common.encode_state_head(tok, text, 512)
    assert head == expected, "the result must still be the full ids"
    sizes = [len(t) for t in tok.texts]
    assert sizes.count(sizes[0]) == 1, \
        "the same prefix was tokenized more than once: %s" % sizes
    assert len(tok.texts) <= 3, "expected probe + at most one attempt + the full state, got %s" % sizes


def test_state_head_retries_from_the_density_it_measured():
    """A state denser than the first estimate must still be optimized, on a later attempt.

    `STATE_HEAD_CHARS_PER_TOKEN` is a starting point, not a bound: a state at ~20 characters per
    token leaves the first prefix well short of the budget. The next attempt is sized from the
    density the first one measured, so it succeeds instead of falling back to tokenizing the whole
    state. Without the retry -- or with a budget that does not grow -- this state silently costs a
    full tokenization.
    """
    tok = _PreTokTok()
    # 41 chars per 2 tokens in this tokenizer = 20.5 chars/token, well past the estimate of 8.
    text = ("-" * 40 + " ") * 2000
    full = tok(text)["input_ids"]
    tok.texts.clear()

    head = common.encode_state_head(tok, text, 512)
    assert head == full[:len(head)], "the head must still be a prefix of the full ids"
    assert len(head) >= 512, "the head must cover the budget"
    assert len(head) < len(full), \
        "the whole state was tokenized; the retry did not recover from a short first attempt"
    assert 1 < len(tok.texts) <= common._STATE_HEAD_MAX_ATTEMPTS, \
        "expected a second attempt, got %d tokenization(s)" % len(tok.texts)
    assert text not in tok.texts, "no attempt should have handed over the whole state"


def test_state_head_guard_rejects_a_short_head_when_density_rises_past_the_probe():
    """The `len(head) >= need` guard, on the only shape that actually reaches it.

    The probe makes an *under*-estimate safe, so for a document of uniform density the first
    attempt sized from the measured density always covers the budget and the guard never has to
    reject anything. It is reachable only when the text past the probe is denser than the text the
    probe read -- by more than the 1.25x margin -- because then `want` is extrapolated from a
    density that no longer holds and the prefix it selects comes up short.

    This state is 512 sparse characters (4 chars/token) followed by 82 000 dense ones
    (20.5 chars/token), so the probe measures 4 and the retry has to climb to ~17. Two attempts
    come up short and are rejected before the third covers the budget. Without the guard the first
    short attempt is returned: 229 ids where 512 were asked for, silently truncating the state.

    `..._ids_match_a_full_tokenization` asserts the same contract over its 20 shapes, but every one
    of them is of uniform density, so deleting the guard leaves the whole suite green -- which it
    did, until this check existed.
    """
    tok = _PreTokTok()
    text = ("x" * 7 + " ") * 64 + ("-" * 40 + " ") * 2000
    full = tok(text)["input_ids"]
    tok.texts.clear()

    head = common.encode_state_head(tok, text, 512)
    assert head == full[:len(head)], "the head must be a prefix of the full ids"
    assert len(head) >= 512, \
        "the head has %d ids, short of the 512 asked for -- a short attempt was returned" % len(head)
    assert len(head) < len(full), "the whole state was tokenized; the retry did not recover"

    # Pin that the guard was actually exercised, so this cannot quietly stop testing it. Counted
    # over the retries only -- `tok.texts[0]` is the probe, which its own check rejects, not this
    # guard. A range rather than an equality because raising `_STATE_HEAD_MAX_ATTEMPTS` must not
    # turn this red: the retry succeeds on its third pass and stops there either way.
    assert 2 <= len(tok.texts) <= 1 + common._STATE_HEAD_MAX_ATTEMPTS, \
        "expected the probe plus at least one retry, got %d tokenization(s)" % len(tok.texts)
    short_retries = sum(1 for t in tok.texts[1:] if len(_PreTokTok()(t)["input_ids"]) < 512)
    assert short_retries >= 2, \
        "expected at least two short retries for the guard to reject, got %d" % short_retries


def test_state_head_stops_at_the_probe_when_the_probe_already_covers_the_budget():
    """The probe short-circuit: one tokenization, not two.

    A small `need` against a sparse state makes the probe itself cover the budget. Returning it
    immediately is the whole point -- falling through to the retry loop re-tokenizes a longer prefix
    and produces a correct but needlessly large head off a second pass. That is invisible to an
    ids-only assertion, so it is asserted here by counting tokenizations: deleting the short-circuit
    leaves every other check in this file green.
    """
    tok = _PreTokTok()
    text = ("x" * 7 + " ") * 4000                  # uniform 4 chars/token, 32 000 chars
    full = tok(text)["input_ids"]
    tok.texts.clear()

    head = common.encode_state_head(tok, text, 8)
    assert head == full[:len(head)], "the head must be a prefix of the full ids"
    assert len(head) >= 8, "the head must cover the budget"
    assert len(tok.texts) == 1, \
        "the probe already covered the budget, so it should be the only tokenization; got %d" \
        % len(tok.texts)


def test_state_head_refuses_to_cut_when_an_added_token_holds_an_internal_space():
    """A checkpoint that added a domain phrase must be tokenized in full, not cut.

    `_state_cut` cuts at a space whose predecessor is not whitespace. An added token like
    "New York" straddles exactly that: the trie matches it greedily, so a cut inside it makes the
    head stop being a prefix of the full ids. The `len(head) >= need` guard cannot catch this --
    the head is long enough, it is simply wrong -- so the cut has to be refused up front.
    """
    safe, unsafe = _PreTokTok(), _AddedPhraseTok()
    assert common._added_tokens_permit_cut(safe) is True
    assert common._added_tokens_permit_cut(unsafe) is False
    # The added-token property is one of two gates; the pre-tokenizer one is checked below.
    assert common._pre_tokenizer_permits_cut(safe) is True

    text = "shipped to New York and billed to New York " * 400
    # The safe tokenizer keeps the optimization ...
    assert len(common.encode_state_head(safe, text, 512)) < len(safe(text)["input_ids"])
    # ... and the unsafe one is tokenized in full, so its ids cannot diverge.
    head = common.encode_state_head(unsafe, text, 512)
    assert head == unsafe(text)["input_ids"]
    assert text in unsafe.texts, "the fallback must hand the whole state to the tokenizer"


def test_splits_on_whitespace_reads_the_flags_that_actually_decide():
    """The gate must key on `ByteLevel.use_regex` and `Metaspace.split`, not just the class name.

    Both flags can be off in a third-party checkpoint, and with either off the pre-tokenizer stops
    putting a space first in every piece -- which is the whole property `_state_cut` relies on.
    Named classes stand in for the real `tokenizers` ones; the gate keys on `type(pt).__name__`.
    """
    class ByteLevel:
        def __init__(self, use_regex): self.use_regex = use_regex

    class Metaspace:
        def __init__(self, split): self.split = split

    class Whitespace:
        pass

    class Sequence(list):
        pass

    class Lowercase:
        pass

    ok = common._splits_on_whitespace
    assert ok(ByteLevel(use_regex=True)) is True
    assert ok(ByteLevel(use_regex=False)) is False, "ByteLevel without its regex does not split"
    assert ok(Metaspace(split=True)) is True
    assert ok(Metaspace(split=False)) is False, "Metaspace(split=False) leaves one piece"
    assert ok(Whitespace()) is True
    # A Sequence is safe when any member splits, and not otherwise.
    assert ok(Sequence([Lowercase(), Metaspace(split=True)])) is True
    assert ok(Sequence([Lowercase(), Metaspace(split=False)])) is False
    assert ok(Sequence([])) is False

    class NonIterableSequence:
        """Named `Sequence` but not walkable, as an older `tokenizers` build may be."""
        pass
    NonIterableSequence.__name__ = "Sequence"
    # Unverifiable, so refused -- the optimization is lost, which is correct and not silent-wrong.
    assert ok(NonIterableSequence()) is False
    # Unknown and absent both mean "cannot establish it", which is not the same as safe.
    assert ok(Lowercase()) is False
    assert ok(None) is False


def test_state_head_refuses_to_cut_when_nothing_splits_on_whitespace():
    """A tokenizer with no whitespace-splitting pre-tokenizer must be tokenized in full.

    `_state_cut` needs a space to open a pre-token, so BPE merges cannot cross the cut. A T5- or
    Llama-shaped tokenizer keeps Metaspace in the normalizer and splits nowhere, so its merges span
    what used to be a space and a cut inside one rewrites the ids *before* it -- measured against a
    real BPE built that way, the head diverges at token 0. The length check is blind to it.
    """
    safe, unsafe = _PreTokTok(), _NoSplitTok()
    assert common._pre_tokenizer_permits_cut(safe) is True
    assert common._pre_tokenizer_permits_cut(unsafe) is False
    # A tokenizer we cannot inspect at all is treated as unverifiable, not as safe.
    assert common._pre_tokenizer_permits_cut(object()) is False

    text = "the customer wants a refund on order 41 " * 2000
    assert len(common.encode_state_head(safe, text, 512)) < len(safe(text)["input_ids"])
    head = common.encode_state_head(unsafe, text, 512)
    assert head == unsafe(text)["input_ids"]
    assert text in unsafe.texts, "the fallback must hand the whole state to the tokenizer"


def test_state_head_refuses_to_cut_when_the_normalizer_drops_the_cut():
    r"""A normalizer that deletes whitespace leaves no boundary at the cut, so refuse to cut.

    The third of the three properties `_state_cut` needs. It cannot be decided from the normalizer's
    class: `Replace` is the multilingual checkpoint's own normalizer and is safe there, because it
    rewrites a space to U+2581 one-for-one and Metaspace then splits on it. The same class with
    `Regex(r"\s+") -> ""` deletes the cut, a BPE piece spans it, and the head stops being a prefix
    while staying long enough that `len(head) >= need` waves it through.
    """
    assert common._normalizer_permits_cut(_PreTokTok()) is True           # no normalizer at all
    assert common._normalizer_permits_cut(_MetaspaceLikeTok()) is True    # space -> U+2581
    assert common._normalizer_permits_cut(_StripNormalizerTok()) is False
    # A sentencepiece checkpoint spells the same boundary U+2581, so checking only U+0020 misses it.
    assert common._added_tokens_permit_cut(_MetaspacePhraseTok()) is False, \
        "an added phrase spelled with U+2581 must be refused just as a U+0020 one is"
    # Keeping the cut character is not enough on its own: what precedes it must still normalize to
    # a prefix of the whole, or slicing there does not slice the normalized form.
    assert common._normalizer_permits_cut(_ReversingNormalizerTok()) is False
    assert common._normalizer_permits_cut(object()) is False              # cannot be inspected

    text = "the customer wants a refund on order 41 " * 2000
    unsafe = _StripNormalizerTok()
    head = common.encode_state_head(unsafe, text, 512)
    assert head == unsafe(text)["input_ids"], "a dropped cut must fall back to the full ids"
    assert text in unsafe.texts, "the fallback must hand the whole state to the tokenizer"
    # And the safe one still gets the optimization, so the gate is not simply off.
    safe = _MetaspaceLikeTok()
    assert len(common.encode_state_head(safe, text, 512)) < len(safe(text)["input_ids"])


def test_state_head_leaves_a_state_barely_over_the_budget_alone():
    """Below `_STATE_HEAD_MIN_RATIO` the prefix is nearly the whole state, so it is not worth taking.

    The head would tokenize almost the same text and the cut scan plus three gates would be pure
    overhead -- measured as a real regression against simply tokenizing the state (0.94x on english,
    0.88x on multilingual just above the budget). So the head path is skipped until the state is
    enough bigger than the budget to pay for itself.
    """
    need = 512
    budget = need * common.STATE_HEAD_CHARS_PER_TOKEN
    assert common._STATE_HEAD_MIN_RATIO > 1, "a ratio of 1 is what the regression measured"

    unit = "refund escalate hold "

    def sized(chars):
        return (unit * (chars // len(unit) + 4))[:chars]

    barely = sized(int(budget * 1.1))
    well_over = sized(int(budget * 4))
    assert len(barely) == int(budget * 1.1) and len(well_over) == int(budget * 4), \
        "the fixture must actually reach those sizes"

    tok = _PreTokTok()
    common.encode_state_head(tok, barely, need)
    assert tok.texts == [barely], \
        "a state only 1.1x the budget must be tokenized once, whole -- no prefix attempt"

    tok = _PreTokTok()
    common.encode_state_head(tok, well_over, need)
    assert tok.texts and len(tok.texts[0]) < len(well_over), \
        "a state 4x the budget must still take the prefix path"
    assert well_over not in tok.texts, "and must not tokenize the whole state"


def test_agent_sequences_are_unchanged_by_the_head_budget():
    """Every state shape must produce byte-identical question rows, head budget or not."""
    agent = _tiny_agent()
    agent.tok = _PreTokTok()
    questions = {
        "act": {"type": "choice", "instructions": "What action?", "criteria": {"refund": "", "hold": ""}},
        "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "high"]},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }
    ids = list(questions)
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
    max_len, head_max_len = agent.cfg["max_len"], agent.cfg["head_max_len"]

    compared = 0
    for label, state in _states():
        items = agent._encode_state(state, ids, internal)
        # The reference: what build_sequence produced when it was always handed the whole state.
        full = agent.tok(serialize_state(state).replace(agent.tok.mask_token, " "),
                         add_special_tokens=False)["input_ids"]
        for qid, item in zip(ids, items):
            seq, markers = build_sequence(agent.tok, state, internal[qid], max_len, head_max_len,
                                          truncate_left=isinstance(state, list), state_ids=full)
            assert item["ids"] == seq, "%s/%s: the sequence changed" % (label, qid)
            assert item["markers"] == markers, "%s/%s: the markers moved" % (label, qid)
            compared += 1
    assert compared == 3 * len(_states()) == 60, compared


def test_onnx_agent_takes_the_same_two_paths_as_the_torch_agent():
    """`ONNXAgent._encode_state` gets the identical change, so pin that it behaves identically.

    The two call sites are the same six lines, which is exactly the kind of duplication that drifts:
    a later edit to one is easy to miss in the other. This compares the ids and markers they
    produce for every state shape rather than trusting that the code still looks the same. No
    onnxruntime and no checkpoint -- `_encode_state` only reads `self.tok`.
    """
    from laya.onnx_agent import ONNXAgent

    agent = _tiny_agent()
    agent.tok = _PreTokTok()
    onnx = ONNXAgent.__new__(ONNXAgent)      # skip __init__: no onnxruntime, no checkpoint
    onnx.tok = agent.tok

    questions = {
        "act": {"type": "choice", "instructions": "What action?", "criteria": {"refund": "", "hold": ""}},
        "risk": {"type": "score", "instructions": "How risky?", "criteria": ["low", "high"]},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }
    ids = list(questions)
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
    max_len, head_max_len = agent.cfg["max_len"], agent.cfg["head_max_len"]

    compared = 0
    for label, state in _states():
        mine = agent._encode_state(state, ids, internal)
        theirs = onnx._encode_state(state, ids, internal, max_len, head_max_len)
        assert len(mine) == len(theirs) == len(ids), label
        for a, b in zip(mine, theirs):
            assert a["ids"] == b["ids"], "%s: the ONNX path built different ids" % label
            assert a["markers"] == b["markers"], "%s: the ONNX path moved the markers" % label
            compared += 1
    assert compared == 3 * len(_states()) == 60, compared

    # Identical ids is necessary but not sufficient: reverting the ONNX site to a full
    # tokenization also produces identical ids, which is the whole point of the change. So pin that
    # the ONNX side still *takes* the prefix path, or it could silently lose it.
    big = "refund escalate hold " * 5000
    onnx.tok = _PreTokTok()
    onnx._encode_state(big, ids, internal, max_len, head_max_len)
    assert big not in onnx.tok.texts, \
        "the ONNX path handed the whole state to the tokenizer; it lost the head budget"
    # ... and that a conversation list is still tokenized in full there, for the same reason.
    turns = [{"role": "user", "content": "turn %d" % i} for i in range(400)]
    onnx.tok = _PreTokTok()
    onnx._encode_state(turns, ids, internal, max_len, head_max_len)
    assert serialize_state(turns) in onnx.tok.texts, \
        "a conversation list must still be tokenized in full on the ONNX path"


def test_agent_tokenizes_a_conversation_list_in_full():
    """A list is truncated from the left, so its tokens are at the end: no head budget for it."""
    agent = _tiny_agent()
    agent.tok = _PreTokTok()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}
    conversation = [{"role": "user", "content": "turn %d" % i} for i in range(4000)] + \
                   [{"role": "user", "content": "NEWEST"}]
    text = serialize_state(conversation)
    agent._encode_state(conversation, ["q"], {"q": agent._to_internal(questions["q"])})
    assert text in agent.tok.texts, "a list state must still be tokenized in full"
    full = agent.tok(text, add_special_tokens=False)["input_ids"]
    seq, _ = build_sequence(agent.tok, conversation, agent._to_internal(questions["q"]),
                            agent.cfg["max_len"], agent.cfg["head_max_len"], truncate_left=True)
    assert seq[-2] == full[-1], "left-truncation must still end on the newest turn"


def test_agent_string_state_is_not_tokenized_in_full():
    """The counterpart: a string state past the budget never reaches the tokenizer whole."""
    agent = _tiny_agent()
    agent.tok = _PreTokTok()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}
    state = "the customer wants a refund on this order " * 3000
    agent._encode_state(state, ["q"], {"q": agent._to_internal(questions["q"])})
    assert state not in agent.tok.texts, "a string state must not be tokenized in full"
    assert max(len(t) for t in agent.tok.texts) < len(state)


def test_usage_input_tokens_is_the_sequence_length_not_the_state_total():
    """Nothing reports the state's full token count, which is why the head budget is invisible.

    `usage["input_tokens"]` is the attention mask's sum -- the padded sequence length, capped by
    `max_len`. A state of 30000 characters and one of 60 report the same number once both fill
    the sequence, so truncating the tokenization cannot change any reported figure.
    """
    agent = _tiny_agent()
    agent.tok = _PreTokTok()
    questions = {"q": {"type": "choice", "instructions": "Act?", "criteria": {"a": "", "b": ""}}}
    short = agent.predict_batch(["refund escalate hold " * 40], questions)[0]
    long = agent.predict_batch(["refund escalate hold " * 3000], questions)[0]
    assert short["usage"]["input_tokens"] == long["usage"]["input_tokens"] == agent.cfg["max_len"], \
        (short["usage"], long["usage"])


# --- mutants that survived an adversarial review; each check below kills one ------------------

class Backend_for_no_api:
    """A backend with an inspectable, splitting pre-tokenizer, so the added-token gate is the only
    thing that can refuse."""
    normalizer = None

    class pre_tokenizer:  # noqa: N801 - matches the `type(pt).__name__` the gate reads
        pass


Backend_for_no_api.pre_tokenizer = type("WhitespaceSplit", (), {})()


class _MetaspaceRunTok(_PreTokTok):
    """A sentencepiece checkpoint whose whitespace-run added tokens are spelled with U+2581.

    This is what the shipped MULTILINGUAL tokenizer actually carries: 30 added tokens that are runs
    of 2-31 U+2581. `"\u2581".isspace()` is False, so `_added_tokens_permit_cut`'s
    `or piece[k - 1] in _CUT_CHARS` clause is the ONLY thing that lets them through. Deleting that
    clause flipped the real multilingual tokenizer from permitted to refused -- every non-string
    state falling back to a full tokenization, 40 001 ids instead of 662 on an 80 000-character
    state -- with the entire CI lane still green, because every fixture here spells its runs with
    U+0020.
    """

    def get_added_vocab(self):
        return {"\u2581" * n: 400 + n for n in range(2, 32)}


def test_added_token_gate_permits_a_metaspace_checkpoints_whitespace_runs():
    """The POSITIVE direction for U+2581, which no other test asserts."""
    tok = _MetaspaceRunTok()
    assert common._added_tokens_permit_cut(tok) is True, \
        "a run of U+2581 is a single-character whitespace run and must not block the cut"
    # and the head path really is taken for it, not just the gate
    text = ("-" * 40 + " ") * 2000
    head = common.encode_state_head(tok, text, 512)
    assert len(head) >= 512 and len(head) < len(tok(text)["input_ids"])


def test_added_token_gate_refuses_a_tokenizer_with_no_added_vocab_api():
    """The missing-attribute guard, which the `get_added_vocab = None` fixture does not reach.

    With that fixture the attribute EXISTS and is None, so `get()` raises into the `except` and the
    `getattr` default is never used -- replacing the default with `lambda: {}` left the suite green
    while treating an uninspectable tokenizer as having no added tokens at all.
    """
    class NoApi:
        mask_token = "[MASK]"
        backend_tokenizer = Backend_for_no_api()

    assert common._added_tokens_permit_cut(NoApi()) is False


def test_added_token_gate_refuses_when_get_added_vocab_raises():
    """Fail-closed has two paths and the existing fixture satisfies both, so each masked the other.

    `get_added_vocab = None` returns False at the missing-attribute guard AND, if that guard is
    removed, raises TypeError into the `except`. A tokenizer whose `get_added_vocab()` *raises* had
    no check at all: mutating either guard left the suite green.
    """
    class Raises(_PreTokTok):
        def get_added_vocab(self):
            raise RuntimeError("vocab unavailable")

    assert common._added_tokens_permit_cut(Raises()) is False


def test_normalizer_gate_refuses_when_the_probe_raises():
    """Same masking in `_normalizer_permits_cut`: the existing fixture takes the `bt is None` path,
    so the `except Exception` around the probe was untested.
    """
    class Boom:
        def normalize_str(self, s):
            raise RuntimeError("normalizer exploded")

    class Backend:
        normalizer = Boom()
        pre_tokenizer = None

    class Tok(_PreTokTok):
        backend_tokenizer = Backend()

    assert common._normalizer_permits_cut(Tok()) is False


def test_added_token_gate_sees_a_cut_char_at_index_one():
    """`range(1, len(piece))` must start at 1. An added token whose space sits at index 1 -- "a b",
    or a sentencepiece user symbol "x\u2581y" -- is the only thing that start index protects, and
    `range(2, ...)` left the suite green while permitting an unsafe cut.
    """
    class IndexOne(_PreTokTok):
        def get_added_vocab(self):
            return dict(_PreTokTok.get_added_vocab(self), **{"a b": 500, "x\u2581y": 501})

    assert common._added_tokens_permit_cut(IndexOne()) is False


def test_state_cut_walks_past_a_whitespace_run_longer_than_one_probe():
    """`_STATE_CUT_PROBES` is load-bearing for exactly the shape its comment describes.

    A whitespace run straddling the budget needs more than one probe to step over. With the constant
    at 1 the cut is lost and the whole state is tokenized -- 80 700 characters instead of 2 159, a
    37x regression -- and no existing fixture put a run there, so the suite stayed green.
    """
    tok = _PreTokTok()
    text = "w " * 250 + " " * 200 + "q " * 40000
    full = tok(text)["input_ids"]
    tok.texts.clear()
    head = common.encode_state_head(tok, text, 512)
    assert head == full[:len(head)], "the head must still be a prefix"
    assert len(head) >= 512, "the head must cover the budget"
    assert len(head) < len(full), "the whole state was tokenized; the cut walk gave up in the run"
    assert text not in tok.texts, "no attempt should have handed over the whole state"


def test_the_retry_does_not_re_tokenize_a_prefix_it_already_measured():
    """The in-loop `last_cut = cut` assignment, which the existing no-advance test never reaches.

    Its fixture breaks on the FIRST iteration, where `last_cut` still holds its pre-loop value.
    Deleting the in-loop assignment made this state tokenize the same 2 599-character prefix three
    times, with the suite green.
    """
    tok = _PreTokTok()
    text = ("-" * 7 + " ") * 64 + "-" * 2087 + " " + "z" * 300000
    tok.texts.clear()
    common.encode_state_head(tok, text, 512)
    sizes = [len(t) for t in tok.texts]
    assert len(sizes) == len(set(sizes)), \
        "the same prefix was tokenized more than once: %s" % sizes


def test_the_retry_stops_when_the_next_attempt_would_cover_the_whole_state():
    """The loop-top budget guard. Deleting it tokenized 26 231 characters where 10 818 suffice --
    2.4x, and 3.3x the state itself, i.e. worse than not optimizing at all -- with the suite green.
    """
    tok = _PreTokTok()
    text = ("x" * 7 + " ") * 64 + ("-" * 40 + " ") * 183
    tok.texts.clear()
    common.encode_state_head(tok, text, 512)
    total = sum(len(t) for t in tok.texts)
    assert total <= 2 * len(text), \
        "tokenized %d characters for a %d-character state" % (total, len(text))


def test_the_normalizer_check_runs_on_the_text_being_cut():
    """A normalizer that is prefix-breaking only on shapes the three fixed probes do not contain.

    Measured on the shipped english tokenizer carrying `Replace(Regex(r" (?=[A-Z])"), "")`: all three
    gates returned True and 2 of 200 capitalised-prose states came back non-prefix. The check has to
    look at the real cut, not at three hard-coded strings.
    """
    class DropsSpaceBeforeCapital:
        def normalize_str(self, s):
            out = []
            for i, ch in enumerate(s):
                if ch == " " and i + 1 < len(s) and s[i + 1].isupper():
                    continue
                out.append(ch)
            return "".join(out)

    class Backend:
        normalizer = DropsSpaceBeforeCapital()
        pre_tokenizer = None

    class Tok(_PreTokTok):
        backend_tokenizer = Backend()

        def __call__(self, text, **kw):
            # The normalizer runs before the pre-tokenizer, exactly as a real tokenizer does. Without
            # this the fake ignores its own normalizer, the ids match either way, and the end-to-end
            # assertion below cannot fail -- which is how the first version of this test passed while
            # the call site it exists to pin could be deleted.
            return _PreTokTok.__call__(self, Backend.normalizer.normalize_str(text), **kw)

    tok = Tok()
    # the three fixed probes are all lowercase, so the general gate is fooled ...
    assert common._normalizer_permits_cut(tok) is True
    # ... and the real-text check is what refuses
    text = ("Refund Order Escalate Ticket Zebra " * 2000)
    cut = common._state_cut(text, 4096)
    assert cut and text[cut - 1] != " "
    assert common._normalizer_keeps_cut(tok, text, cut) is False

    full = tok(text)["input_ids"]
    head = common.encode_state_head(tok, text, 512)
    assert head == full, "a cut this normalizer destroys must fall back to the full ids"


def test_every_cut_is_checked_against_the_text_being_cut(monkeypatch):
    """`_normalizer_keeps_cut` must be CONSULTED, on the real text, before any cut is tokenized.

    Pinned by observation rather than by output, deliberately. A normalizer bad enough to corrupt a
    head usually collapses it far below `need`, so the length guard rejects it and the retry returns
    the same full ids anyway -- deleting the call site changes no ids at all on such a fixture, which
    is how the first version of this test passed while the call it exists to pin could be removed.
    What is not reproducible by accident is whether the check ran.
    """
    seen = []
    real = common._normalizer_keeps_cut

    def recording(tok, text, cut):
        seen.append((len(text), cut))
        return real(tok, text, cut)

    monkeypatch.setattr(common, "_normalizer_keeps_cut", recording)
    tok = _PreTokTok()
    text = ("-" * 40 + " ") * 2000            # dense and cuttable: reaches the gates and the retry
    tok.texts.clear()
    head = common.encode_state_head(tok, text, 512)

    assert len(head) >= 512, "this fixture must take the head path for the check to be reachable"
    prefixes = [t for t in tok.texts if len(t) < len(text)]
    assert len(prefixes) > 1, "this fixture must probe and then retry, or it pins only one call site"

    # EVERY cut whose prefix gets tokenized must have been checked first -- the probe's cut at the
    # gate, and each retry's cut in the loop. Counting them is what distinguishes the two call sites:
    # asserting only that the check "ran" passes while either one of them is deleted.
    assert len(seen) == len(prefixes), \
        "%d prefixes tokenized but %d cuts checked -- a cut was taken unchecked" \
        % (len(prefixes), len(seen))
    for seen_len, cut in seen:
        assert seen_len == len(text), "the check was handed something other than the state text"
        assert 0 < cut <= len(text), cut
        assert text[cut] == " " and not text[cut - 1].isspace(), \
            "the checked position is not the cut that `_state_cut` chose"
