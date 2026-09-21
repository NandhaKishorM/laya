"""Quilt decision-ledger tests. Stdlib-only: no torch, no model weights.

The ledger's whole point is that its rows verify anywhere — including here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.quilt import GENESIS, QuiltLedger, QuiltLayaBridge, canonical, fnv1a32, sha256_hex  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, got):
    check(name, bool(got), True)


# --------------------------------------------------------------------- hash primitives
check("fnv1a32 offset basis (empty input)", "%08x" % fnv1a32(b""), "811c9dc5")
check("fnv1a32 is deterministic", fnv1a32(b"laya4quilt"), fnv1a32(b"laya4quilt"))
check("fnv1a32 avalanches on one byte", fnv1a32(b"laya4quilu") != fnv1a32(b"laya4quilt"), True)
check("fnv1a32 accepts str", fnv1a32("quilt"), fnv1a32(b"quilt"))
check("canonical sorts keys", canonical({"b": 1, "a": 2}), canonical({"a": 2, "b": 1}))
check("canonical is tight", canonical({"a": 1, "b": [2, 3]}), '{"a":1,"b":[2,3]}')
check("sha256_hex stable", sha256_hex({"x": 1}), sha256_hex({"x": 1}))
check_true("sha256_hex is 64 hex chars", len(sha256_hex("laya")) == 64)

# --------------------------------------------------------------------- booking
CLOCK = [1000.0]


def clock():
    CLOCK[0] += 1.0
    return CLOCK[0]


led = QuiltLedger(actor="crab-test", engine="fake-v1", clock=clock)
check("genesis chain on empty ledger", QuiltLedger("a", "e").rows, [])

FAKE_ANSWERS = {
    "model": "fake-v1",
    "answers": {
        "triage": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.8, "escalate": 0.2}, "confidence": 0.64},
        "urgency": {"type": "score", "score": 2.5, "legend": {"0": "low", "1": "mid", "2": "high"}, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.6},
        "abuse": {"type": "noul", "noul": 0.12, "confidence": 0.88, "action": {"act_probability": 0.97}},
    },
    "usage": {"input_tokens": 42, "output_tokens": 0},
}


def fake_engine(state, questions):
    return FAKE_ANSWERS


r1 = led.decide("customer email body", {"triage": {"type": "choice", "instructions": "route", "criteria": {"refund": None, "escalate": None}}}, fake_engine)
check("BIND row booked", r1["op"], "BIND")
check("BIND carries actor (witness-stake)", r1["actor"], "crab-test")
check("BIND carries engine identity", r1["engine"], "fake-v1")
check("BIND answers stored verbatim", r1["payload"]["answers"]["triage"]["choice"], "refund")
check("BIND engine_model recorded", r1["payload"]["engine_model"], "fake-v1")
check("tick starts at 1", r1["tick"], 1)
check("genesis prev on first row", r1["chain_prev"], GENESIS)
check_true("row_hash is 8 hex", len(r1["row_hash"]) == 8)

r2 = led.decide("second state", {"q": {"type": "noul", "instructions": "is it abuse?"}}, fake_engine)
check("second row links first", r2["chain_prev"], r1["row_hash"])
check("ticks strictly increase (partial order)", r2["tick"], 2)
check("distinct content -> distinct hashes", r1["row_hash"] == r2["row_hash"], False)

# --------------------------------------------------------------------- verification
ok, bad = led.verify()
check("fresh ledger verifies", (ok, bad), (True, None))

tampered = QuiltLedger(actor="crab-test", engine="fake-v1", clock=clock)
tampered.decide("state", {"q": {"type": "noul", "instructions": "x"}}, fake_engine)
tampered.decide("state2", {"q": {"type": "noul", "instructions": "y"}}, fake_engine)
victim = tampered.rows[0]
victim["payload"]["answers"]["triage"]["choice"] = "hacked"  # attacker rewrites history
ok, bad = tampered.verify()
check("tamper detected", ok, False)
check("tamper pinned at the mutated row", bad, victim["row_hash"])

chain_break = QuiltLedger(actor="a", engine="e", clock=clock)
chain_break.decide("s", {"q": {"type": "noul", "instructions": "i"}}, fake_engine)
chain_break.decide("s2", {"q": {"type": "noul", "instructions": "i"}}, fake_engine)
chain_break.rows[1]["chain_prev"] = "deadbeef"  # splice attempt
ok, bad = chain_break.verify()
check("chain splice detected", ok, False)

# --------------------------------------------------------------------- refusal visibility
refused_led = QuiltLedger(actor="a", engine="e", clock=clock)


def broken_engine(state, questions):
    return {"no_answers_key": True}


rr = refused_led.decide("s", {"q": {"type": "noul", "instructions": "i"}}, broken_engine)
check("malformed engine -> REFUSED row", rr["op"], "REFUSED")
check("refusal names the reason", rr["payload"]["reason"], "ENGINE_RESULT_INVALID")
check("refusal keeps state binding", rr["payload"]["state_hash"], sha256_hex("s"))
ok, _ = refused_led.verify()
check("refusals verify like any row", ok, True)


def exploding_engine(state, questions):
    raise RuntimeError("checkpoint corrupted")


rr2 = refused_led.decide("s2", {"q": {"type": "noul", "instructions": "i"}}, exploding_engine)
check("exception -> REFUSED row", rr2["op"], "REFUSED")
check_true("refusal detail names the exception", "checkpoint corrupted" in rr2["payload"]["detail"])

# --------------------------------------------------------------------- ensembles & resolution
ens = QuiltLedger(actor="crab-test", engine="base", clock=clock)


def engine_a(state, q):
    out = dict(FAKE_ANSWERS)
    out["model"] = "fake-a"
    return out


def engine_b(state, q):
    out = dict(FAKE_ANSWERS)
    out["answers"] = dict(FAKE_ANSWERS["answers"])
    out["answers"]["triage"] = dict(FAKE_ANSWERS["answers"]["triage"], choice="escalate", probabilities={"refund": 0.3, "escalate": 0.7})
    out["model"] = "fake-b"
    return out


QUESTIONS = {"triage": {"type": "choice", "instructions": "route", "criteria": {"refund": None, "escalate": None}}}
group = ens.ensemble("g1", "customer email body", QUESTIONS, {"fake-a": engine_a, "fake-b": engine_b})
check("ensemble books 2 BIND rows", len(group["rows"]), 2)
check("ensemble books exactly one LINK", sum(1 for r in ens.rows if r["op"] == "LINK"), 1)

row_a = next(r for r in ens.rows if r["row_hash"] == group["rows"][0])
row_b = next(r for r in ens.rows if r["row_hash"] == group["rows"][1])
check("rivals disagree (preserved)", row_a["payload"]["answers"]["triage"]["choice"] != row_b["payload"]["answers"]["triage"]["choice"], True)
check("each rival carries its engine identity", (row_a["engine"], row_b["engine"]), ("fake-a", "fake-b"))
check("rivals bound to same state hash", row_a["payload"]["state_hash"] == row_b["payload"]["state_hash"], True)

res = ens.resolve("g1", group["rows"][1], "escalate has higher calibrated confidence under dispute")
check("resolution is an EFFECT row", res["op"], "EFFECT")
check("resolution cites the chosen row", res["payload"]["chosen"], group["rows"][1])
check("resolution cites its rivals", group["rows"][0] in res["payload"]["rivals"], True)
check("resolution does not delete rivals", sum(1 for r in ens.rows if r["op"] == "BIND"), 2)
ok, _ = ens.verify()
check("ensemble ledger verifies", ok, True)

bad_res = ens.resolve("g1", "f" * 8, "nothing")
check("resolving a missing target -> REFUSED", bad_res["op"], "REFUSED")
check("missing-target reason named", bad_res["payload"]["reason"], "RESOLVE_TARGET_NOT_FOUND")

solo = QuiltLedger(actor="a", engine="e", clock=clock)
one = solo.ensemble("g0", "s", QUESTIONS, {"only": engine_a})
check("solo ensemble -> REFUSED", one["op"], "REFUSED")
check("solo ensemble reason named", one["payload"]["reason"], "ENSEMBLE_REQUIRES_2_ENGINES")

# --------------------------------------------------------------------- views
ok, _ = led.verify()
check_true("ledger still verifies before views", ok)
v = led.view("json")
check("json view books a VIEW row", led.rows[-1]["op"], "VIEW")
check("json view returns chain head", v["chain_head"], led.rows[-2]["row_hash"])
head_before_canon = led.rows[-1]["row_hash"]
c = led.view("canon")
check("canon view provenance carries chain head", c["provenance"]["ledger_chain_head"], head_before_canon)
check("canon view lists BIND entries", len(c["entries"]), 2)
check_true("canon entries hashed", all(len(e["hash"]) == 8 for e in c["entries"]))
try:
    led.view("xml")
    check("unknown view refuses loudly", "no-exception", "exception")
except ValueError:
    check("unknown view refuses loudly", "exception", "exception")

# --------------------------------------------------------------------- bridge (lazy-import contract)
bridge = QuiltLayaBridge(route="english")
check("bridge defers agent construction", bridge.agent, None)
check("bridge route recorded", bridge.served_by, "english")
fn = bridge.decide_fn()
check("decide_fn is callable", callable(fn), True)

# --------------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all quilt ledger tests passed")
sys.exit(1 if FAIL else 0)
