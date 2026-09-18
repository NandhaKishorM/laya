"""P0 #2: truncation direction must be selectable and must be reported, not silent."""
import pytest

from laya.common import build_sequence


class StubTokenizer:
    """Whitespace tokenizer with a stable id space -- no model download needed."""

    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        # ids >= 10 so they never collide with the special ids above
        return {"input_ids": [10 + (abs(hash(w)) % 1000) for w in text.split()]}


@pytest.fixture
def tok():
    return StubTokenizer()


QUESTION = {"t": "noul", "ins": "does it hold", "crit": None}


def test_no_truncation_callback_when_state_fits(tok):
    events = []
    build_sequence(tok, "short state", QUESTION, max_len=512, head_max_len=192,
                   on_truncate=lambda k, t: events.append((k, t)))
    assert events == []


def test_truncation_reports_kept_and_total(tok):
    state = " ".join("word%d" % i for i in range(500))
    events = []
    build_sequence(tok, state, QUESTION, max_len=128, head_max_len=64,
                   on_truncate=lambda k, t: events.append((k, t)))
    assert len(events) == 1
    kept, total = events[0]
    assert total == 500
    assert kept < total
    assert kept > 0


def test_truncate_right_keeps_the_head(tok):
    state = " ".join("w%d" % i for i in range(400))
    head_ids = tok(state)["input_ids"]
    ids, _ = build_sequence(tok, state, QUESTION, max_len=100, head_max_len=48, truncate_left=False)
    assert head_ids[0] in ids, "right-truncation keeps the beginning of the state"
    assert head_ids[-1] not in ids or head_ids[-1] in head_ids[:50]


def test_truncate_left_keeps_the_tail(tok):
    state = " ".join("w%d" % i for i in range(400))
    all_ids = tok(state)["input_ids"]
    ids, _ = build_sequence(tok, state, QUESTION, max_len=100, head_max_len=48, truncate_left=True)
    assert all_ids[-1] in ids, "left-truncation keeps the end of the state"


def test_left_and_right_truncation_differ(tok):
    state = " ".join("w%d" % i for i in range(400))
    right, _ = build_sequence(tok, state, QUESTION, max_len=100, head_max_len=48, truncate_left=False)
    left, _ = build_sequence(tok, state, QUESTION, max_len=100, head_max_len=48, truncate_left=True)
    assert right != left


def test_sequence_never_exceeds_max_len(tok):
    state = " ".join("w%d" % i for i in range(2000))
    for max_len in (64, 128, 512, 1024):
        ids, markers = build_sequence(tok, state, QUESTION, max_len=max_len, head_max_len=48)
        assert len(ids) <= max_len
        assert all(m < max_len for m in markers)


def test_larger_max_len_keeps_more_state(tok):
    """The context-window knob actually widens the window."""
    state = " ".join("w%d" % i for i in range(3000))
    small, _ = build_sequence(tok, state, QUESTION, max_len=512, head_max_len=192)
    large, _ = build_sequence(tok, state, QUESTION, max_len=4096, head_max_len=192)
    assert len(large) > len(small)


def test_markers_align_with_option_count(tok):
    q = {"t": "choice", "ins": "pick one", "crit": {"a": "x", "b": "y", "c": "z"}}
    _, markers = build_sequence(tok, "state", q, max_len=512, head_max_len=192)
    assert len(markers) == 3
