"""Regression: stripping identifiers must stay linear in the length of a word-character run.

`_IDENTIFIER` removes dot- and @-joined tokens (`github.com`, `user@acme.com`, `v1.2.3`)
before `_WORD` counts function words. Without the leading lookbehind, the greedy `[\\w-]*`
is retried at every offset inside a run of word characters and each attempt rescans the
run before failing on the absent `[.@]` -- quadratic in the run's length:

    4 000 characters of one token      205 ms   (0.45 ms for prose of the same size)
   50 000 characters of one token   30 678 ms

`analyse()` runs on the default routing path -- no `model`, no `lang` -- once per request,
and `state_text` hands it up to 4000 characters, so an unauthenticated caller of
`POST /v1/systemone` could buy ~115x the CPU of a normal request. `latin_profile` and
`guess_latin_language` take unbounded text, where the same input costs ~31 s.

The lookbehind removes no match: `[\\w-]` and `[.@]` are disjoint, so an attempt from
inside a run consumes to exactly the same separator as an attempt from the run's start,
and the two always succeed or fail together. The parity block below is the evidence --
it is checked against the previous pattern, not against hand-written expectations.

Run: python tests/test_identifier_complexity.py
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.lang import _IDENTIFIER, analyse, latin_profile  # noqa: E402

# The pattern as it was before the lookbehind, kept here so parity is asserted against
# the real previous behaviour rather than against expectations someone typed out.
_PREVIOUS = re.compile(r"[\w-]*(?:[.@][\w-]+)+", re.UNICODE)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def ok(name, cond, extra=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ("  -> " + str(extra)) if extra else ""))


# ---------------------------------------------------------------- parity ----
# Every identifier the pattern exists to strip, plus the shapes that tempt a
# bounded-quantifier fix into leaving a residue token behind.
PARITY = [
    "github.com", "user@acme.com", "v1.2.3", "U.S.A.", "arrivato.", "a.b", "a@b",
    "sub.domain.co.uk", "first.last@sub.example.org", "192.168.0.1", "-a.b-", "_x.y_",
    "foo..bar", ".com", "a.", "@a", "a@", "e.g.", "Ü.Ö", "naïve.café", "a-b.c-d",
    "x-a.b", "9.9", "a..b", "a.-b", "a-.b-.c", "", ".", "@",
    "Contact support@acme.com or github.com/acme for v2.10.1 details.",
    "No identifiers here at all just words",
    # A local part at RFC 5321's 64-character limit, and a run past any plausible bound:
    # a length-bounded pattern strips only part of these and leaves the rest as a word.
    "a" * 64 + "@example.com",
    "a" * 200 + ".example.com",
    "grazie mille per " + "wzqxk" * 15 + ".example.com",
]
for token in PARITY:
    check("parity/%r" % (token[:34],), _IDENTIFIER.sub(" ", token), _PREVIOUS.sub(" ", token))

# Randomised differential test: the two patterns must agree on everything.
import random  # noqa: E402

random.seed(11)
_ALPHABETS = ["aA1._@- ", "._@-", "áéÜß._@-", "abcXYZ019._@- -_"]
mismatch = None
for i in range(20_000):
    alphabet = _ALPHABETS[i % len(_ALPHABETS)]
    s = "".join(random.choice(alphabet) for _ in range(random.randint(0, 60)))
    if _IDENTIFIER.sub(" ", s) != _PREVIOUS.sub(" ", s):
        mismatch = s
        break
ok("parity/20 000 random strings agree with the previous pattern",
   mismatch is None, repr(mismatch))

# Detection behaviour, not just the regex.
check("a state of two domains is still not prose",
      analyse({"body": "github.com acme.com"})["language"], None)
check("the routing answer for ordinary prose is unchanged",
      analyse({"body": "The customer was billed twice and wants a refund"})["is_english"], True)


# ------------------------------------------------------------ complexity ----
def elapsed(fn, *a):
    started = time.perf_counter()
    fn(*a)
    return time.perf_counter() - started


# Ceilings are set to fail the previous pattern rather than merely to be generous.
# Previous cost / ceiling / current cost, measured on a 2023 laptop:
#   4 000 chars     205 ms  /  50 ms  /  0.14 ms      (~350x headroom)
#  20 000 chars   4 860 ms  / 1.00 s  /  0.73 ms     (~1370x headroom)
for n, ceiling in ((4_000, 0.05), (20_000, 1.00)):
    took = elapsed(_IDENTIFIER.sub, " ", "a" * n)
    ok("%d word characters strip in under %.0f ms (took %.1f ms)" % (n, ceiling * 1000, took * 1000),
       took < ceiling, "%.1f ms" % (took * 1000))

# Hyphens and underscores are word characters here too, so the run can be built from them,
# and a single trailing separator must not reopen the quadratic path.
for label, text in (("hyphens", "a-" * 10_000),
                    ("underscores", "a_" * 10_000),
                    ("one trailing dot", "a" * 20_000 + ".")):
    took = elapsed(_IDENTIFIER.sub, " ", text)
    ok("%s: 20k strips in under 1 s (took %.1f ms)" % (label, took * 1000), took < 1.0,
       "%.1f ms" % (took * 1000))

# Scaling, which is machine-independent in a way a wall-clock ceiling is not: linear work
# quadruples with the input, quadratic work grows sixteenfold. 8 sits between the two.
small = min(elapsed(_IDENTIFIER.sub, " ", "a" * 5_000) for _ in range(3))
large = min(elapsed(_IDENTIFIER.sub, " ", "a" * 20_000) for _ in range(3))
ratio = large / small if small > 0 else 0
ok("4x the input costs under 8x the time, ratio %.1f (linear ~4, quadratic ~16)" % ratio,
   ratio < 8, "ratio %.1f  (%.2f ms -> %.2f ms)" % (ratio, small * 1000, large * 1000))

# The public entry point a request actually reaches. Compared against prose of the same
# size rather than a wall-clock ceiling: at 4000 characters the gap between quadratic and
# linear is only a few-fold, which CI jitter can cover, but the ratio against prose is
# ~1x when the work is linear and ~100x when it is not.
_PROSE_4K = ("The customer was billed twice and wants a refund. " * 80)[:4_000]


def _best(state, reps=3):
    return min(elapsed(analyse, state) for _ in range(reps))


token_cost = _best({"body": "a" * 4_000})
prose_cost = _best({"body": _PROSE_4K})
ratio_4k = token_cost / max(prose_cost, 1e-5)
ok("one long token costs about what prose of the same size costs (ratio %.1f)" % ratio_4k,
   ratio_4k < 10, "%.2f ms vs %.2f ms" % (token_cost * 1000, prose_cost * 1000))
took = elapsed(latin_profile, "a" * 50_000)
ok("latin_profile() on 50k of one token is under 1 s (took %.1f ms)" % (took * 1000),
   took < 1.0, "%.1f ms" % (took * 1000))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
