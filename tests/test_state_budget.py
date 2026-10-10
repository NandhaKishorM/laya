"""Regression: the state budget is sized from the head actually built, not from `head_max_len`.

`head_max_len` is a cap. `build_sequence` fills the option prompt budget only as far as the
question and its option descriptions need, then sizes the state as `max_len - head_len - 1`
from the head it actually built.

`max_len - head_max_len` is wrong in *both* directions, which is what the README now says:

* a question that does not fill the cap -> the cap-based figure **understates** the room
  (conservative, but it discards state the model would have read);
* a high-cardinality question -> it **overstates** it. `per = max(4, (head_max_len - 16) // k)`
  floors at 4 tokens per option and `head_ids` keeps a floor of 8, so past roughly
  `head_max_len / 4` options the head grows beyond the cap, and a state sized to the cap-based
  figure is **truncated** rather than merely conservative.

This suite pins the relationship, and — when the checkpoints are present — reads the figures
back out of `README.md` and checks them against a live measurement, so the renderer, the
documentation and this file cannot drift apart silently.

`examples/33_high_cardinality_choice.py` is the page that teaches this budget, and it taught three
things the code does not do: that the cut waits for the options to *overflow* `head_max_len` (it
fires on the spare budget, while they still fit), that 16 is a minimum *reserved for the instruction*
(the instruction is held to 8), and that the option block *lives in* `head_max_len` (past ~4 tokens
per option the head grows past the cap and takes the state's room with it). The last sections of
this file pin the page to `build_head` — behaviourally, and by reading the budget's literals out of
the function instead of typing them in here.

A third surface teaches the same budget from the other side: the *default scan window* of
`predict_long`, which `window_budget` guards with a floor. README.md and
`docs/hooks/patterns.md` both stated the guarded quantity on its own, which is wrong exactly where
the floor is the answer -- at `max_len=512, head_max_len=448` the stated figure is 56 while the scan
runs 64-token windows -- and neither page said that the state's room still binds below the floor, or
that a head which fills the sequence is refused instead of scanning at the 0 the subtraction leaves.
The final sections pin those two pages to `window_budget`, and `window_budget` to what they now
promise.

A fourth surface teaches the budget by naming where it can be set: `docs/routing.md`'s
`predict_batch()` section listed the four per-request *routing* overrides and stopped, so the two
token budgets a request can carry -- the per-request form of the very thing this suite measures --
were missing from the guide that sells heterogeneous batches. The last sections read the keys out of
`Router.route_batch` and `Router.predict_batch` themselves (indexed for required, `.get()` for
optional, with the routing delegation verified) and pin the page to them, plus `structured.py`'s
own declared list of those keys, which is what `decide_batch` builds per-request dicts from.
"""
import ast
import inspect
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import build_head, build_sequence, render_options, state_room, window_budget  # noqa: E402

PASS, FAIL, SKIPPED = [], [], []

# The same 4-option question the README's token-budget section walks through.
Q = {"t": "choice", "ins": "Which department should handle this request?",
     "crit": {"billing": "invoices, payments, refunds",
              "technical": "bugs, outages, system errors",
              "sales": "pricing, new contracts, plan upgrades",
              "other": "everything else"}}


class _Skip(Exception):
    pass


def skip(reason):
    """A skip that is visible in both runners.

    Under pytest this raises `pytest.skip`, so `-rs` reports it. As a plain script it raises
    `_Skip`, which the runner prints and counts -- a bare `return` passes vacuously and no
    runner flag reveals it.
    """
    print("  SKIP: " + reason)
    SKIPPED.append(reason)
    if __name__ != "__main__":
        import pytest
        pytest.skip(reason)
    raise _Skip(reason)


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


class _FakeTok:
    """Deterministic stand-in: one id per word, so head length is predictable."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 4, 2
    mask_token = "[MASK]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (len(w) % 90) for w in text.split() if w]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def id_of(self, word):
        return self(" " + word)["input_ids"][0]


def _head_len(tok, q, head_max_len):
    """Length of the built prompt the state is appended to, excluding the trailing [SEP]."""
    seq, _ = build_sequence(tok, "", q, max_len=10 ** 6, head_max_len=head_max_len)
    return len(seq) - 1


def _kept_state(tok, state, q, max_len, head_max_len):
    head = _head_len(tok, q, head_max_len)
    seq, _ = build_sequence(tok, state, q, max_len=max_len, head_max_len=head_max_len)
    return seq[head:-1]


def _many_options(k, desc="a short description of this option"):
    return {"t": "choice", "ins": "Which intent is this?",
            "crit": {"intent_%d" % i: desc for i in range(k)}}


def _models_dir():
    return os.environ.get("LAYA_LAB_MODELS", os.path.expanduser("~/laya_models"))


def _real_tok(name):
    """The checkpoint's tokenizer, or None. Callers decide whether absence is a skip."""
    path = os.path.join(_models_dir(), name, "tokenizer")
    if not os.path.isdir(path):
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


# --------------------------------------------------------------------------- invariant checks

def test_state_budget_is_sized_from_the_actual_head():
    """A state longer than the room is cut to exactly `max_len - head_len - 1`."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, Q, head_max_len)
    assert head < head_max_len, "fixture assumes a question that does not fill the cap"

    room = max_len - head - 1
    state = " ".join("w%d" % i for i in range(room + 200))
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    check("kept state length is max_len - head_len - 1", len(kept), room)


def test_dropping_the_minus_one_is_caught():
    """The `-1` is load-bearing: one token more than the room must lose its tail."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, Q, head_max_len)
    marker = "TAILMARKER"
    exact = max_len - head - 1                    # the true room

    state = " ".join("w%d" % i for i in range(exact - 1)) + " " + marker
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    check("a state of exactly the room is kept whole", len(kept), exact)
    assert tok.id_of(marker) in kept, "the tail marker must survive an exactly-sized state"

    over = " ".join("w%d" % i for i in range(exact)) + " " + marker
    kept_over = _kept_state(tok, over, Q, max_len, head_max_len)
    assert tok.id_of(marker) not in kept_over, (
        "a state one token over the room must lose its tail, so the room is max_len-head_len-1 "
        "and not max_len-head_len")


def test_cap_based_figure_understates_a_short_question():
    """A state of exactly `max_len - head_max_len` tokens survives for a short question."""
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    documented = max_len - head_max_len
    head = _head_len(tok, Q, head_max_len)
    assert max_len - head - 1 > documented, "fixture assumes the cap-based figure understates"

    marker = "UNIQUETAILMARKER"
    state = " ".join("w%d" % i for i in range(documented - 3)) + " " + marker
    kept = _kept_state(tok, state, Q, max_len, head_max_len)
    assert tok.id_of(marker) in kept, (
        "a %d-token state must survive a %d budget for this question" % (documented, max_len))


def test_cap_based_figure_overstates_a_high_cardinality_question():
    """Past ~`head_max_len / 4` options the head exceeds the cap, so the cap-based figure lies.

    This is the direction that matters: a caller sizing a state to `max_len - head_max_len`
    is truncated, not merely conservative.
    """
    tok = _FakeTok()
    max_len, head_max_len = 512, 192
    head = _head_len(tok, _many_options(100), head_max_len)
    assert head > head_max_len, (
        "a 100-option question must build a head beyond the %d cap, got %d" % (head_max_len, head))
    check("cap-based figure overstates the room at k=100",
          max_len - head_max_len > max_len - head - 1, True)


# --------------------------------------------------------------------- documentation agreement

def _readme_quotes():
    """The figures README.md states, parsed out of the token-budget bullet."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    heads = {int(m) for m in re.findall(r"\*\*(\d+)-token\*\* head", text)}
    rooms = {int(m) for m in re.findall(r"\*\*(\d+) tokens for state\*\*", text)}
    return heads, rooms


def test_readme_quotes_the_measured_figures():
    """README's head/room figures equal a live measurement, for each shipped checkpoint.

    This is what makes the suite stop the README and the renderer desynchronising; without it
    the file only guards the renderer.
    """
    heads, rooms = _readme_quotes()
    assert heads and rooms, "could not parse the head/room figures out of README.md"

    checked = 0
    for name, max_len, head_max_len in (("english", 512, 192), ("multilingual", 1024, 256)):
        try:
            tok = _real_tok(name)
        except Exception as e:                      # transformers absent or unloadable
            skip("%s: tokenizer unavailable (%s)" % (name, e))
            continue
        if tok is None:
            skip("%s: no tokenizer under %s" % (name, _models_dir()))
            continue
        head = _head_len(tok, Q, head_max_len)
        room = max_len - head - 1
        assert head in heads, (
            "README does not quote a %d-token head for %s (quotes %s)" % (head, name, sorted(heads)))
        assert room in rooms, (
            "README does not quote a %d-token state room for %s (quotes %s)"
            % (room, name, sorted(rooms)))
        checked += 1
    if not checked:
        skip("no checkpoints under %s" % _models_dir())


# ---------------------------------------------------- example 33: the page against the budget it teaches

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "examples", "33_high_cardinality_choice.py")

# The two formulas, in the symbolic form the page has to teach them in.
PER_FORMULA = re.compile(r"per\s*=\s*max\(\s*(\d+)\s*,\s*\(\s*head_max_len\s*-\s*(\d+)\s*\)"
                         r"\s*//\s*N\s*\)")
INSTRUCTION_FORMULA = re.compile(r"head_ids\[:\s*max\(\s*(\d+)\s*,\s*opt_budget\s*\)\]")
# `max(4, (192-16)//77) = 4` -- a worked instance, whose arithmetic has to come out right.
WORKED_INSTANCE = re.compile(r"max\(\s*(\d+)\s*,\s*\((\d+)\s*-\s*(\d+)\)\s*//\s*(\d+)\s*\)"
                             r"\s*=\s*(\d+)")
CONDITIONAL = re.compile(r"\b(?:when|once|if|unless)\b[^,;.]{0,80}", re.I)
OVERFLOW = re.compile(r"overflow", re.I)
# 16 taught as the instruction's reserve: the number that is really the instruction's floor is 8.
RESERVE_16_FOR_INSTRUCTION = re.compile(
    r"\b16\b[^.]{0,60}\b(?:reserv\w*|minimum)\b[^.]{0,60}instruction"
    r"|instruction[^.]{0,60}\b(?:reserv\w*|minimum)\b[^.]{0,60}\b16\b", re.I)
BLOCK_INSIDE_CAP = re.compile(
    r"\b(?:lives|live|fits|fit|stays|stay|kept)\b[^.]{0,30}\b(?:in|within|under|inside)\b[^.]{0,30}"
    r"head_max_len", re.I)
CAP_IS_NOT_HARD = re.compile(r"not a hard cap|no hard cap|past the cap|past head_max_len"
                             r"|longer than head_max_len|head longer than|grows past", re.I)
INSTRUCTION_FLOOR_8 = re.compile(r"max\(\s*8\s*,|8-token|\b8\b[^.]{0,40}instruction"
                                 r"|instruction[^.]{0,40}\b8\b", re.I)


def _page_prose():
    """The example's string constants, whitespace-normalised, as (joined, sentences).

    Sentences are split inside each constant, never over a join of them: two adjacent constants
    can read as one sentence, which would let a true sentence vouch for a false neighbour.
    """
    if not os.path.exists(EXAMPLE):
        raise AssertionError("examples/33_high_cardinality_choice.py is not where this suite looks "
                             "for it (%s)" % EXAMPLE)
    with open(EXAMPLE, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    constants = [re.sub(r"\s+", " ", node.value) for node in ast.walk(tree)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    sentences = [sent.strip() for text in constants
                 for sent in re.split(r"(?<=[.!?])\s+", text) if sent.strip()]
    return " ".join(constants), sentences


def _option_question(lengths, ins="Which one?"):
    """A choice question whose options cost exactly `lengths` tokens under `_FakeTok`.

    One id per word, and an option renders as `label: <criterion>`, so the criterion carries
    `length - 2` words: the label and its colon are the other two.
    """
    return {"t": "choice", "ins": ins,
            "crit": {"o%d" % i: " ".join(["w"] * (length - 2)) for i, length in enumerate(lengths)}}


def _full_option_len(tok, q):
    """Tokens each option costs before the budget cuts it, `[MASK]` included."""
    return [1 + len(tok(" " + opt)["input_ids"]) for opt in render_options(q)]


def _budget_literals():
    """(threshold, reserve, per-floor, instruction-floor) exactly as `build_head` writes them.

    Read out of the function rather than typed in here, so a page holding a stale number fails
    instead of agreeing with itself.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(build_head)))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt):
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Name) and left.id == "opt_budget" and isinstance(right, ast.Constant):
                found["threshold"] = right.value
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "max" and len(node.args) == 2:
            first, second = node.args
            if not isinstance(first, ast.Constant):
                continue
            if isinstance(second, ast.BinOp) and isinstance(second.op, ast.FloorDiv) \
                    and isinstance(second.left, ast.BinOp) and isinstance(second.left.op, ast.Sub) \
                    and getattr(second.left.left, "id", "") == "head_max_len" \
                    and isinstance(second.left.right, ast.Constant):
                found["per_floor"] = first.value
                found["reserve"] = second.left.right.value
            if isinstance(second, ast.Name) and second.id == "opt_budget":
                found["instruction_floor"] = first.value
    missing = {"threshold", "reserve", "per_floor", "instruction_floor"} - set(found)
    assert not missing, (
        "build_head no longer writes the budget the way this suite reads it: %s missing"
        % sorted(missing))
    return found


def test_the_option_cut_waits_on_spare_tokens_not_on_overflow():
    """The cut fires on the leftover head budget, and fires while the options still fit.

    This is the ground truth the page has to match: `head_max_len - sum(options) < 16`, not the
    options overflowing `head_max_len`. A question whose options fill the head exactly -- 192 of
    192, nothing dropped -- is already cut.
    """
    tok = _FakeTok()
    head_max_len = 192

    fills_it = _option_question([48, 48, 48, 48])
    full = _full_option_len(tok, fills_it)
    check("fixture: four options of 48 fill the head without overflowing it",
          (sum(full), sum(full) > head_max_len, head_max_len - sum(full)), (192, False, 0))
    _, _, stats = build_head(tok, fills_it, head_max_len)
    assert stats["tokens_per_option"] is not None, (
        "a question that leaves 0 spare tokens must already be cut to "
        "max(4, (head_max_len - 16) // N); it kept its full lengths, so the cut is not where "
        "this suite says it is")
    check("cut size for 4 options at head_max_len=192", stats["tokens_per_option"], 44)

    spare_16 = _option_question([44, 44, 44, 44])
    check("fixture: 16 spare tokens", head_max_len - sum(_full_option_len(tok, spare_16)), 16)
    _, _, stats_16 = build_head(tok, spare_16, head_max_len)
    check("16 spare tokens is not yet the cut", stats_16["tokens_per_option"], None)

    spare_15 = _option_question([45, 44, 44, 44])
    check("fixture: 15 spare tokens", head_max_len - sum(_full_option_len(tok, spare_15)), 15)
    _, _, stats_15 = build_head(tok, spare_15, head_max_len)
    assert stats_15["tokens_per_option"] is not None, (
        "15 spare tokens is under the 16 the cut waits for, so this must be cut")


def test_the_instruction_floor_is_eight_not_the_reserved_sixteen():
    """Past the floor the instruction keeps 8 tokens -- 16 is not reserved for it.

    77 options at the 4-token floor leave the head short of itself, and the instruction is clipped
    to `max(8, opt_budget)` with a negative `opt_budget`: the retained prefix is 8, from a question
    whose instruction is far longer.
    """
    tok = _FakeTok()
    ins = ("Which of these many clearly distinct banking intents best matches the customer "
           "message right now, please answer carefully")
    q = _many_options(77)
    q["ins"] = ins
    full_instruction = len(tok("%s question: %s" % (q["t"], ins))["input_ids"])
    assert full_instruction > 16, (
        "fixture needs an instruction long enough that a 16-token reserve would show up")

    ids, markers, stats = build_head(tok, q, 192)
    check("77 options are cut to the per-option floor", stats["tokens_per_option"], 4)
    check("instruction tokens retained", markers[0] - 2, 8)
    check("the retained prefix is below the 16 the page must not call its reserve",
          markers[0] - 2 < 16, True)
    check("the head built runs past head_max_len", len(ids) > 192, True)


def test_page_teaches_the_spare_as_the_trigger_not_overflow():
    """The page must state the cut's condition as the spare budget, and never as an overflow."""
    _, sentences = _page_prose()

    clauses = [m.group(0) for sent in sentences for m in CONDITIONAL.finditer(sent)]
    assert clauses, (
        "the page states no conditional clause at all, so the rules below would pass by reading "
        "nothing: it has to describe when the budget cuts")
    bad = [c for c in clauses if OVERFLOW.search(c)]
    assert not bad, (
        "the page makes the options overflowing the head the trigger (%s). `build_head` cuts on "
        "`opt_budget < 16`, which happens while the options still fit the head -- see "
        "test_the_option_cut_waits_on_spare_tokens_not_on_overflow" % bad)
    stated = [c for c in clauses if "16" in c]
    assert stated, (
        "no condition on the page names the 16 tokens the cut waits for, so the trigger it teaches "
        "is not the one `build_head` implements")


def test_page_does_not_reserve_sixteen_for_the_instruction():
    """`16` is the spare threshold and the division's reserve; the instruction's floor is 8."""
    joined, sentences = _page_prose()

    formula = INSTRUCTION_FORMULA.search(joined)
    assert formula, (
        "the page no longer shows the instruction clip (`head_ids[: max(8, opt_budget)]`), so its "
        "reader cannot see what the instruction is actually held to")
    floor = [sent for sent in sentences if "instruction" in sent.lower() and INSTRUCTION_FLOOR_8.search(sent)]
    assert floor, "no sentence gives the instruction its floor as a number"
    wrong = [sent for sent in sentences if RESERVE_16_FOR_INSTRUCTION.search(sent)]
    assert not wrong, (
        "the page calls 16 the instruction's reserve or minimum (%s). It is not: the instruction "
        "keeps `max(8, opt_budget)` tokens, and a 77-option question leaves it with 8 -- see "
        "test_the_instruction_floor_is_eight_not_the_reserved_sixteen" % wrong)
    role = [sent for sent in sentences
            if re.search(r"\b16\b", sent) and re.search(r"threshold|spare|keeps back|reserves? the "
                                                        r"division|division", sent, re.I)]
    assert role, "the page never says what the 16 in its own formula is for"


def test_page_does_not_put_the_option_block_inside_the_cap():
    """`head_max_len` is a target the head can pass, and the page has to say so."""
    joined, sentences = _page_prose()

    inside = [sent for sent in sentences if BLOCK_INSIDE_CAP.search(sent)]
    assert not inside, (
        "the page puts the option block inside `head_max_len` (%s). Past roughly head_max_len / 4 "
        "options every one sits at the 4-token floor and the head grows past the cap -- 319 tokens "
        "against a 192 cap for the question this page runs -- so the state is sized from the head "
        "built, not from the cap" % inside)
    honest = [sent for sent in sentences if CAP_IS_NOT_HARD.search(sent)]
    assert honest, (
        "nothing on the page says the head can run past `head_max_len`, which is the consequence "
        "a high-cardinality reader needs")


def test_page_arithmetic_is_build_head_arithmetic():
    """Every number the page's formulas carry is the number `build_head` carries."""
    joined, _ = _page_prose()
    code = _budget_literals()

    per = PER_FORMULA.search(joined)
    assert per, (
        "the page no longer states the cut as `per = max(4, (head_max_len - 16) // N)`, so this "
        "suite cannot check it against build_head")
    instruction = INSTRUCTION_FORMULA.search(joined)
    assert instruction, (
        "the page no longer states the instruction clip as `head_ids[: max(8, opt_budget)]`, so "
        "this suite cannot check that number either")
    check("per-option floor on the page", int(per.group(1)), code["per_floor"])
    check("reserve on the page", int(per.group(2)), code["reserve"])
    check("instruction floor on the page", int(instruction.group(1)), code["instruction_floor"])
    assert code["threshold"] == code["reserve"], (
        "build_head now triggers on a spare of %s but reserves %s in the division: the page states "
        "one number where the code has two, so its prose has to say which is which"
        % (code["threshold"], code["reserve"]))

    worked = WORKED_INSTANCE.findall(joined)
    assert worked, "the page states no worked instance of its own formula"
    for floor, head, reserve, options, result in worked:
        want = max(int(floor), (int(head) - int(reserve)) // max(1, int(options)))
        check("worked instance max(%s, (%s-%s)//%s)" % (floor, head, reserve, options),
              int(result), want)


# ------------------------------------------- the default scan window, against the floor it teaches

# `window_budget` has guarded the default window with a floor since the feature landed, and the two
# markdown pages that teach the default taught the guarded quantity alone. Scoped to the pages that
# teach the *window* on purpose, like the docstring gate in tests/test_predict_long.py: the bare
# subtraction is a true statement about the STATE room elsewhere in the README (`state_room`
# measures the head that was actually built), and a sweep that demanded the floor on every mention
# of it would be wrong.

DOC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW_PAGES = ("README.md", os.path.join("docs", "hooks", "patterns.md"))
SUBTRACTION = re.compile(r"max_len\s*-\s*head_max_len")


def _window_default_expr():
    """The default window exactly as `window_budget` computes it, as `(expression, floor)`.

    Read out of the function rather than typed in here, so a page holding a stale number fails
    instead of agreeing with itself -- and so does the other direction, a `window_budget` that
    stopped guarding the default, because then the guard this function looks for is not there.
    """
    src = textwrap.dedent(inspect.getsource(window_budget))
    tree = ast.parse(src)
    assigns = [node for node in ast.walk(tree)
               if isinstance(node, ast.Assign)
               and any(getattr(t, "id", "") == "requested" for t in node.targets)]
    assert assigns, (
        "`window_budget` no longer works the default out into `requested`, so this suite cannot read "
        "the expression the pages have to teach")
    calls = [node for node in ast.walk(assigns[0])
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "max"
             and len(node.args) == 2 and isinstance(node.args[0], ast.Constant)
             and isinstance(node.args[0].value, int)]
    assert calls, (
        "`window_budget` no longer guards the default window with max(<constant>, the budget), so "
        "the floor README.md and docs/hooks/patterns.md teach is not the code's: the prose has to "
        "change, not these assertions")
    return ast.get_source_segment(src, calls[0]), calls[0].args[0].value


def _markdown_paragraphs():
    """Paragraphs of the repo's markdown, as `(repo-relative path, first line, normalised text)`.

    A paragraph is a run of contiguous non-blank lines -- the unit a reader sees, and the unit both
    pages carry this teaching in. Code fences come along as paragraphs too, so a fence that pairs
    the budget with the word `window` has to teach the floor like prose does.
    """
    paths = [os.path.join(DOC_ROOT, "README.md")]
    for dirpath, _dirnames, filenames in sorted(os.walk(os.path.join(DOC_ROOT, "docs"))):
        for filename in sorted(filenames):
            if filename.endswith(".md"):
                paths.append(os.path.join(dirpath, filename))

    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        start, block = None, []
        for i, line in enumerate(lines, 1):
            if line.strip():
                if start is None:
                    start = i
                block.append(line)
            elif block:
                out.append((os.path.relpath(path, DOC_ROOT), start,
                            re.sub(r"\s+", " ", " ".join(block))))
                start, block = None, []
        if block:
            out.append((os.path.relpath(path, DOC_ROOT), start, re.sub(r"\s+", " ", " ".join(block))))
    return out


def test_the_window_default_is_floored():
    """Widening the head stops shaving the default once the floor is reached.

    The ground truth the two pages have to match: at `max_len=512` a `head_max_len=448` leaves a
    budget under the floor and the scan still runs floor-sized windows, and at 504 the budget is 0
    while the window is still the floor. The 4-option question leaves room far above the floor here,
    so it is the floor that decides and not the room clamp.
    """
    expr, floor = _window_default_expr()
    tok = _FakeTok()
    for head_max_len in (448, 504):
        budget = 512 - head_max_len - 8
        assert budget < floor, (
            "fixture: head_max_len=%d has to leave a budget under the floor, or the floor is not "
            "what this case tests (budget %d, floor %d)" % (head_max_len, budget, floor))
        room = state_room(tok, Q, 512, head_max_len)
        assert room > floor, (
            "fixture: this question has to leave room above the floor (%d), or the room clamp and "
            "not the floor decides the window" % room)
        size, step, got_room = window_budget(tok, [Q], 512, head_max_len)
        assert size == floor, (
            "the default window at max_len=512, head_max_len=%d is %d, not the budget's %d: `%s` "
            "alone is what the pages must not teach" % (head_max_len, size, budget, expr))
        assert got_room == room and step == floor // 2, (
            "the floor is meant to size the window, not to move the room or the stride: %d/%d"
            % (got_room, step))


def test_the_room_still_binds_below_the_floor():
    """The floor does not rescue a state from its own options, and a full head is refused.

    A question whose head leaves less room than the floor scans at that room, and a head that fills
    the sequence raises instead of scanning at the 0 the subtraction would leave -- both of them are
    in docs/hooks/patterns.md now, so they need a witness in the code.
    """
    expr, floor = _window_default_expr()
    tok = _FakeTok()
    tight = _many_options(110)
    room = state_room(tok, tight, 512, 192)
    assert 0 < room < floor, (
        "fixture: this question has to leave room under the floor, got %d against floor %d -- the "
        "fake tokenizer's head arithmetic moved, and the page's claim with it" % (room, floor))
    size, step, got_room = window_budget(tok, [tight], 512, 192)
    assert (size, step, got_room) == (room, room // 2, room), (
        "a question under the floor has to scan at its own room, not at the floor: %r"
        % ((size, step, got_room),))

    full = _many_options(130)
    assert state_room(tok, full, 512, 192) == 0, (
        "fixture: this question has to fill the sequence, or the refusal below tests nothing")
    try:
        window_budget(tok, [full], 512, 192)
    except ValueError as e:
        assert "no room for the state" in str(e), (
            "the refusal has to say why the scan cannot proceed: %s" % str(e)[:120])
    else:
        raise AssertionError(
            "a head that fills the sequence must be refused, not scan at the 0 `%s` would leave "
            "unguarded" % expr)


def test_no_page_states_the_window_budget_without_the_floor():
    """Every markdown paragraph pairing the budget with `window` teaches the floored expression."""
    expr, _floor = _window_default_expr()
    paragraphs = _markdown_paragraphs()
    for page in WINDOW_PAGES:
        assert any(path == page for path, _line, _text in paragraphs), (
            "the sweep read no paragraph from %s, so it cannot check what the page teaches" % page)

    bare = []
    for path, line, text in paragraphs:
        restated = text.replace(expr, "")
        found = SUBTRACTION.search(restated)
        if found and re.search(r"\bwindows?\b", restated, re.I):
            bare.append("%s:%d -- ...%s..." % (path, line,
                                               restated[max(0, found.start() - 60):found.end() + 60]))
    assert not bare, (
        "these paragraphs state the window budget as the bare subtraction, which is not what "
        "`window_budget` computes -- the default is `%s`: %s" % (expr, bare))


def test_both_pages_teach_the_floored_default():
    """The floor has to be taught as the default on both pages that carry the word `window`."""
    expr, floor = _window_default_expr()
    paragraphs = _markdown_paragraphs()
    for page in WINDOW_PAGES:
        stated = [para for path, _line, para in paragraphs
                  if path == page and expr in para and "window" in para.lower()]
        assert stated, (
            "%s no longer states the default window as `%s` (floor %d), so the page teaches a "
            "number the code does not compute" % (page, expr, floor))
        taught = [para for para in stated
                  if re.search(r"default\w*", para[max(0, para.find(expr) - 220):
                                                   para.find(expr)].lower())]
        assert taught, (
            "%s carries `%s`, but nowhere inside the sentence that names the default -- a page can "
            "state the floor beside a different claim and still teach the budget to a reader"
            % (page, expr))


# ------------------------------------------- the keys a batch request may set, against the page

# The sections above pin the budget's *arithmetic*. This one pins the page that teaches where the
# budget can be set at all, which is the other half of why a budget is usable: `docs/routing.md`'s
# `predict_batch()` section listed four per-request keys where `Router` reads six, omitting the two
# token budgets. So the per-request form of `max_len` -- the reason one wide question does not shrink
# the rest of the batch, which is this file's whole subject -- was undiscoverable on the page that
# sells heterogeneous batches, while `Router.predict_batch`'s docstring and
# `laya/structured.py`'s `_ROUTER_REQUEST_KEYS` both named all six. Nothing compared the guide to the
# methods, so the omission cost nothing to keep.
#
# Scoped to the two sentences that list a request's keys on purpose, the way the window sweep is
# scoped to the pages that teach the window: `docs/routing.md` is the guide a reader of
# `predict_batch()` lands on, and the reference page is generated from the docstring, which is
# already right. A third page, README.md, names all six and passes unchanged.

ROUTING_PAGE = os.path.join("docs", "routing.md")

# Per method: the paragraph's anchor, and the span inside it that is the list of optional keys.
# Anchored by wording, not by line number, so a rewrite of either sentence fails here with a message
# instead of checking nothing.
REQUEST_SECTIONS = {
    "route_batch": ("Every `route_batch()` item needs",
                    re.compile(r"optional routing overrides \((?P<list>[^)]*)\)")),
    "predict_batch": ("Each request requires",
                      re.compile(r"it may also include (?P<list>.*?)[.;]")),
}
BACKTICKED = re.compile(r"`([a-z_]+)`")


def _batch_request_keys():
    """`{method: (required, optional)}` as `Router` actually reads them off a request dict.

    Required are the keys indexed out of a request (`request["state"]`), optional the keys looked up
    with `.get()`, which is how `Router` distinguishes an unset override from an empty one. Read from
    the source rather than transcribed here, so a key added to the code and not to the page fails,
    and a page promising a key the code stopped reading fails too.

    `predict_batch` gets its routing overrides from `route_batch`, which it calls with the requests
    it was handed; that delegation is checked, not assumed, and the optional set is the union of the
    two methods only once it is found.
    """
    path = os.path.join(DOC_ROOT, "laya", "router.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    router = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "Router")
    methods = {node.name: node for node in router.body if isinstance(node, ast.FunctionDef)}
    for name in REQUEST_SECTIONS:
        assert name in methods, (
            "`Router.%s` is no longer a method of `laya/router.py`, so nothing here can say what a "
            "batch request may name" % name)

    def receiver(node):
        while isinstance(node, ast.Subscript):
            node = node.value
        return node.id if isinstance(node, ast.Name) else ""

    def reads(name):
        required, optional = set(), set()
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Subscript):
                if receiver(node.value).startswith("request") and isinstance(node.slice, ast.Constant):
                    required.add(node.slice.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "get"
                  and receiver(node.func.value).startswith("request")
                  and node.args and isinstance(node.args[0], ast.Constant)):
                optional.add(node.args[0].value)
        return required, optional

    route = reads("route_batch")
    predict = reads("predict_batch")
    delegates = any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "route_batch" and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self" and node.args
        and receiver(node.args[0]).startswith("request")
        for node in ast.walk(methods["predict_batch"]))
    assert delegates, (
        "`Router.predict_batch` no longer routes its requests through `self.route_batch(requests, ...)`, "
        "so the routing overrides a batch item may name have to be re-derived before the guide is "
        "called complete")
    return {"route_batch": route,
            "predict_batch": (predict[0], predict[1] | route[1])}


def _page_request_sentences():
    """`{method: paragraph}` for the two sentences in `docs/routing.md` that list a request's keys."""
    paragraphs = {method: [text for path, _line, text in _markdown_paragraphs()
                           if path == ROUTING_PAGE and anchor in text]
                  for method, (anchor, _pattern) in REQUEST_SECTIONS.items()}
    for method, listed in paragraphs.items():
        assert len(listed) == 1, (
            "%s carries %d paragraphs containing %r, and this sweep reads exactly one: the page has "
            "to keep one sentence per method that lists what a request may name"
            % (ROUTING_PAGE, len(listed), REQUEST_SECTIONS[method][0]))
    return {method: listed[0] for method, listed in paragraphs.items()}


def test_the_routing_page_lists_every_key_a_batch_request_can_set():
    """What the guide names as optional is what `Router` reads, key for key."""
    derived = _batch_request_keys()
    for method, paragraph in _page_request_sentences().items():
        match = REQUEST_SECTIONS[method][1].search(paragraph)
        assert match, (
            "%s still has the %s paragraph, but no longer in the form that lists the optional keys "
            "(looked for: %s), so the page has to be re-read against this message"
            % (ROUTING_PAGE, method, REQUEST_SECTIONS[method][1].pattern))
        named = set(BACKTICKED.findall(match.group("list")))
        required, optional = derived[method]
        assert named == optional, (
            "`Router.%s` reads %s off a request, but the guide's %s sentence names %s"
            % (method, sorted(optional), method, sorted(named) or "nothing"))
        for key in sorted(required):
            assert "`%s`" % key in paragraph, (
                "`Router.%s` indexes `request[\"%s\"]`, which raises on a request that omits it, so "
                "the page has to say the key is required" % (method, key))


def test_the_declared_request_key_list_is_the_one_router_reads():
    """`structured._ROUTER_REQUEST_KEYS` cannot drift from the methods it documents itself."""
    from laya.structured import _ROUTER_REQUEST_KEYS
    derived = _batch_request_keys()
    read = derived["route_batch"][1] | derived["predict_batch"][1]
    assert set(_ROUTER_REQUEST_KEYS) == read, (
        "`laya/structured.py` declares the keys `Router.predict_batch` reads off each request as %s, "
        "while the two methods read %s -- one of the two is stale, and `decide_batch` builds its "
        "per-request dicts from the declaration"
        % (sorted(_ROUTER_REQUEST_KEYS), sorted(read)))


if __name__ == "__main__":
    for fn in (test_state_budget_is_sized_from_the_actual_head,
               test_dropping_the_minus_one_is_caught,
               test_cap_based_figure_understates_a_short_question,
               test_cap_based_figure_overstates_a_high_cardinality_question,
               test_the_option_cut_waits_on_spare_tokens_not_on_overflow,
               test_the_instruction_floor_is_eight_not_the_reserved_sixteen,
               test_page_teaches_the_spare_as_the_trigger_not_overflow,
               test_page_does_not_reserve_sixteen_for_the_instruction,
               test_page_does_not_put_the_option_block_inside_the_cap,
               test_page_arithmetic_is_build_head_arithmetic,
               test_the_window_default_is_floored,
               test_the_room_still_binds_below_the_floor,
               test_no_page_states_the_window_budget_without_the_floor,
               test_both_pages_teach_the_floored_default,
               test_the_routing_page_lists_every_key_a_batch_request_can_set,
               test_the_declared_request_key_list_is_the_one_router_reads,
               test_readme_quotes_the_measured_figures):

        try:
            fn()
        except AssertionError as e:
            FAIL.append("%s: %s" % (fn.__name__, e))
        except _Skip:
            pass
        else:
            PASS.append(fn.__name__)

    print("\n%d passed, %d failed, %d skipped" % (len(PASS), len(FAIL), len(SKIPPED)))
    for f in FAIL:
        print("  FAIL " + f)
    for s in SKIPPED:
        print("  SKIP " + s)
    sys.exit(1 if FAIL else 0)
