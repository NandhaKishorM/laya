"""Residency cap: what `Router` holds, and what the residency page tells readers about it.

`examples/24_preload_and_memory.py` is the page a reader copies a `max_loaded` setting from. Its
banner used to open with "At the default `max_loaded=1`", a number `laya/router.py` retired in #180
and README.md documents as 2. So the page taught the retired cap while demonstrating eviction, and
a reader who believed it thought they had to raise the cap to stop the churn the page was showing.
The checks below read the page's own text and syntax tree -- no checkpoint, no device. `Router()`
loads nothing, so none of this needs weights.
"""
import ast
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# The page under test lives in this checkout, so the library it is compared against has to come
# from here too: an installed `laya` would let the page and a stale default agree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya.router import Router


class MockAgent:
    def __init__(self, name):
        self.name = name


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "24_preload_and_memory.py"
TREE = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
CODE_DEFAULT = Router().max_loaded

# Every string constant the page carries: the banner blurb, the module docstring and each print's
# template. Sentences are split inside a constant -- two prints glued together are not one
# sentence -- but a folded one, so a sentence wrapped across two source lines stays one sentence.
CONSTANTS = [n.value for n in ast.walk(TREE) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
PROSE = re.sub(r"\s+", " ", " ".join(CONSTANTS))
SENTENCES = [s.strip() for text in CONSTANTS
             for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if s.strip()]
# A cap the page states, whether typed in or interpolated, and the typed-in subset of those.
CAP_MENTION = re.compile(r"max_loaded\s*=\s*(?:%[-\d.]*[dfi]|\d+)")
CAP = re.compile(r"max_loaded\s*=\s*(\d+)")
MAGNITUDE = re.compile(r"\b(?:thousands?|millions?|billions?|orders of magnitude)\b", re.I)


def callee(call):
    """`laya.Router` / `r.route` / `timed` -- a dotted name, so a rename cannot hide a call."""
    parts, node = [], call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def assigned_names(node):
    """The names an assignment binds, including tuple unpacking (`_, ms = timed(...)`)."""
    out = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Tuple):
            out.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    return out


def measured(tree):
    """(wall-clock deltas, `timed()` medians) -- the two ways this page gets a real number."""
    perf, timed_ms = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = set(assigned_names(node))
        value = node.value
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub) \
                and "perf_counter" in ast.dump(value):
            perf |= names
        if isinstance(value, ast.Call) and callee(value) == "timed":
            timed_ms |= names
    return perf, timed_ms


def printed_timed_route(tree):
    """The names a `timed(...)` call hands back when what it wrapped was a `route()` call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and callee(node) == "timed" and node.args:
            if any(isinstance(c, ast.Call) and callee(c).split(".")[-1] == "route"
                   for c in ast.walk(node.args[0])):
                return True
    return False


def ratio_of_two_measurements(tree, perf, timed_ms):
    """A division whose operands are one wall-clock delta and one `timed()` median."""
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            ids = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if (ids & perf) and (ids & timed_ms):
                return sorted(ids & (perf | timed_ms))
    return []


def derived_default(tree):
    """Names bound to a `Router(...)` construction's `.max_loaded` -- the default, read not typed."""
    out = set()
    for node in ast.walk(tree):
        value = node.value if isinstance(node, ast.Assign) else None
        if isinstance(value, ast.Attribute) and value.attr == "max_loaded" \
                and isinstance(value.value, ast.Call) and callee(value.value).split(".")[-1] == "Router":
            out |= set(assigned_names(node))
    return out


class RouterMemoryTests(unittest.TestCase):
    def test_mps_cache_is_released_after_unload_and_eviction(self):
        for operation in ("one", "all", "evict"):
            for available in (False, True):
                with self.subTest(operation=operation, available=available):
                    empty_cache = Mock()
                    torch = SimpleNamespace(
                        cuda=SimpleNamespace(is_available=lambda: False),
                        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: available)),
                        mps=SimpleNamespace(empty_cache=empty_cache),
                    )
                    r = Router(max_loaded=1)
                    r._agents["english"] = MockAgent("en")
                    r._order.append("english")
                    with patch.dict(sys.modules, {"torch": torch}):
                        if operation == "one":
                            r.unload("english")
                        elif operation == "all":
                            r.unload()
                        else:
                            r._agents["multilingual"] = MockAgent("multi")
                            r._order.append("multilingual")
                            r._evict()
                    self.assertNotIn("english", r.loaded)
                    self.assertEqual(empty_cache.call_count, int(available))

    def test_evict_and_unload_memory_release(self):
        r = Router(max_loaded=1)
        r.attach("english", MockAgent("en"))
        r.attach("multilingual", MockAgent("multi"))
        self.assertEqual(len(r.loaded), 2)

        # Trigger unload of specific model
        r.unload("english")
        self.assertNotIn("english", r.loaded)
        self.assertIn("multilingual", r.loaded)

        # Trigger unload all
        r.unload()
        self.assertEqual(len(r.loaded), 0)

    def test_the_default_cap_holds_more_languages_than_a_cap_of_one(self):
        """Why the retired number matters: the shipped default keeps more than one language hot.

        Agents are poked in through the internals `load()` would use, because `load()` needs a
        checkpoint and this suite runs with none. `_evict()` is the residency rule itself, so the
        expected count is derived from the cap rather than typed: if the library's default moves,
        this test follows it instead of going red for the wrong reason.
        """
        for cap in (CODE_DEFAULT, 1):
            r = Router(max_loaded=cap)
            for i in range(cap + 2):                 # two more languages than it can hold
                name = "model_%d" % i
                r._agents[name] = MockAgent(name)
                r._order.append(name)
                r._evict()
            self.assertEqual(len(r.loaded), cap, "max_loaded=%d held %r" % (cap, r.loaded))
            self.assertEqual(r.loaded[-1], "model_%d" % (cap + 1),
                             "max_loaded=%d evicted the wrong end: %r" % (cap, r.loaded))


class ExampleResidencyPageTests(unittest.TestCase):
    """What the residency page may claim about `max_loaded`, held to `laya/router.py`."""

    def test_default_is_more_than_one_checkpoint(self):
        # The witness that the page-claims below are about a library that keeps two resident: with
        # a default of 1 every claim would be true of the retired behaviour instead, and the
        # sentences checked here would have nothing to disagree with.
        self.assertGreater(CODE_DEFAULT, 1, repr(CODE_DEFAULT))

    def test_page_states_the_default_cap_that_router_has(self):
        # A sentence may only pair the word "default" with a cap the constructor actually takes.
        # Restoring "At the default `max_loaded=1`" fails here: 1 != 2.
        paired = [s for s in SENTENCES if re.search(r"\bdefault\b", s, re.I) and CAP_MENTION.search(s)]
        self.assertTrue(paired, "the page states no default cap at all, so this rule sees nothing")
        for sentence in paired:
            for stated in CAP.findall(sentence):
                self.assertEqual(int(stated), CODE_DEFAULT,
                                 "page says %r; laya/router.py says max_loaded=%d"
                                 % (sentence[:90], CODE_DEFAULT))

    def test_page_reads_its_default_from_the_library(self):
        # The banner must interpolate a value it obtained from a Router, not a literal it typed.
        names = derived_default(TREE)
        self.assertTrue(names, "nothing in the page assigns from `Router(...).max_loaded`")
        banners = [n for n in ast.walk(TREE)
                   if isinstance(n, ast.Call) and callee(n) == "banner" and len(n.args) > 2]
        self.assertTrue(banners, "the page calls banner() with no blurb to hold the default")
        for call in banners:
            referenced = {n.id for n in ast.walk(call.args[2]) if isinstance(n, ast.Name)}
            self.assertTrue(referenced & names,
                            "the banner states a default it did not read: %r vs %r"
                            % (sorted(referenced), sorted(names)))

    def test_page_states_no_cap_it_never_configures(self):
        configured = {kw.value.value for node in ast.walk(TREE) if isinstance(node, ast.Call)
                      for kw in node.keywords
                      if kw.arg == "max_loaded" and isinstance(kw.value, ast.Constant)}
        stated = {int(n) for n in CAP.findall(PROSE)}
        allowed = set(configured) | {CODE_DEFAULT}
        self.assertTrue(configured, "the page configures no cap, so this rule has no baseline")
        self.assertEqual(sorted(stated - allowed), [],
                         "page states %r but configures only %r" % (sorted(stated - allowed),
                                                                    sorted(allowed)))

    def test_reload_versus_route_ratio_is_measured(self):
        # The page used to close with "thousands of times more expensive than the routing itself" --
        # a magnitude with nothing measured behind it. Both sides must now be timed, and the number
        # printed must divide one measurement by the other.
        perf, timed_ms = measured(TREE)
        self.assertTrue(perf and timed_ms, "the page takes no wall-clock readings")
        self.assertTrue(printed_timed_route(TREE), "nothing times a route() call")
        self.assertTrue(ratio_of_two_measurements(TREE, perf, timed_ms),
                        "no ratio divides a cold-build delta by a timed route median")

    def test_no_magnitude_word_without_a_measurement(self):
        # A caught-in-the-act class of lie: prose that asserts a scale this page never measured.
        self.assertFalse(MAGNITUDE.findall(PROSE),
                         "page claims a scale: %r" % MAGNITUDE.findall(PROSE))
        # The rule above only has teeth if the page really does compare the two readings; a bare
        # `%`-placeholder or a digit in front of "times" is what that looks like in a template.
        self.assertRegex(PROSE, r"(?:%[-\d.]*[dfi]|\d+(?:\.\d+)?)\s+times",
                         "the page makes no ratio claim, so the scale rule sees nothing")


# ------------------------------------------------------------------ examples/21: route() cost page
# `examples/21_routing_without_running.py` measures `median_ms` and `total_ms` from real
# `perf_counter`/`timed()` calls, prints both, and then used to close with:
#     "Ten routing decisions cost well under a millisecond in total, and the router still
#      holds zero checkpoints."
# Every clause in that sentence is a quantity the same page already measured but did not read
# back: "Ten" was a hardcoded count while `len(CASES)` sat right there; "well under a
# millisecond" was a threshold assertion that goes stale on slower hardware; "still holds zero
# checkpoints" asserted residency rather than reading `r.loaded`. The page now phrases its
# verdict through two pure helpers, `cost_verdict(total_ms, threshold_ms)` and
# `resident_state(r)`, and interpolates the raw measurement into the sentence, so a slower
# machine or a longer `CASES` list rephrases the claim instead of silently contradicting it.
EXAMPLE_21 = Path(__file__).resolve().parents[1] / "examples" / "21_routing_without_running.py"
TREE_21 = ast.parse(EXAMPLE_21.read_text(encoding="utf-8"))
HELPERS_21 = ("cost_verdict", "resident_state")

# The page as it ships on main: three literal assertions the closing paragraph makes. Every
# ban below must fire on this wording, and every positive rule must NOT match this text.
OLD_PAGE_21 = '''
   Ten routing decisions cost well under a millisecond in total, and the router still
   holds zero checkpoints. `route()` is therefore safe to call on every inbound request,
   even one you end up answering with a conventional LLM.
'''

HARD_COUNT = re.compile(r"\bTen routing decisions\b")
THRESHOLD_CLAIM = re.compile(r"well under a millisecond", re.I)
RESIDENCY_CLAIM = re.compile(r"still\s+holds\s+zero\s+checkpoints", re.I)


def helpers_21():
    """Exec the example's top-level pure helpers -- no weights, no `router()` construction."""
    nodes = [n for n in TREE_21.body
             if isinstance(n, ast.FunctionDef) and n.name in HELPERS_21]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EXAMPLE_21), "exec"), ns)  # noqa: S102
    return ns, {n.name for n in nodes}


class ExampleRouteCostPageTests(unittest.TestCase):
    """What the routing-cost page may claim about the numbers it just measured."""

    def test_page21_helpers_are_defined_and_pure(self):
        _, defined = helpers_21()
        self.assertEqual(defined, set(HELPERS_21), "example 21 lost or renamed a helper")
        for name in HELPERS_21:
            node = next(n for n in TREE_21.body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            prints = [c for c in ast.walk(node)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                      and c.func.id == "print"]
            self.assertFalse(prints, "%s prints instead of returning" % name)

    def test_cost_verdict_flips_on_the_measured_total(self):
        ns, _ = helpers_21()
        # under-threshold: the phrase names the bar and says "under"
        self.assertIn("under 1.0 ms", ns["cost_verdict"](0.437))
        # over-threshold: the phrase stops claiming "under"
        over = ns["cost_verdict"](3.7)
        self.assertIn("or over", over)
        self.assertNotIn("under", over)
        # a caller-set threshold is respected
        self.assertIn("under 5.0 ms", ns["cost_verdict"](3.7, threshold_ms=5.0))

    def test_resident_state_reads_the_router_state(self):
        ns, _ = helpers_21()
        empty = SimpleNamespace(loaded=[])
        one = SimpleNamespace(loaded=["english"])
        two = SimpleNamespace(loaded=["english", "multilingual"])
        self.assertEqual(ns["resident_state"](empty), "the router holds zero checkpoints")
        # non-empty must report the actual count and names, not silently say zero
        self.assertIn("1", ns["resident_state"](one))
        self.assertIn("english", ns["resident_state"](one))
        self.assertIn("2", ns["resident_state"](two))
        self.assertIn("multilingual", ns["resident_state"](two))
        self.assertNotIn("zero", ns["resident_state"](one) + ns["resident_state"](two))

    def test_page21_drops_the_three_hardcoded_closing_claims(self):
        src = EXAMPLE_21.read_text(encoding="utf-8")
        for name, rule in (("hard-coded count", HARD_COUNT),
                           ("unmeasured threshold", THRESHOLD_CLAIM),
                           ("asserted residency", RESIDENCY_CLAIM)):
            self.assertTrue(rule.search(OLD_PAGE_21),
                            "ban does not fire on main's wording: %s" % name)
            self.assertFalse(rule.search(src), "%s is still in example 21" % name)

    def test_page21_closing_interpolates_helpers_count_and_ms(self):
        src = EXAMPLE_21.read_text(encoding="utf-8")
        for frag in ("cost_verdict(total_ms)", "resident_state(r)", "len(CASES)"):
            self.assertIn(frag, src, "the closing sentence no longer interpolates %r" % frag)
            self.assertNotIn(frag, OLD_PAGE_21)
        # the sentence still carries a raw measured number, not only a qualitative verdict
        self.assertRegex(src, r"%d routing decisions cost %s in total \(%\.3f ms")


if __name__ == "__main__":
    unittest.main()
