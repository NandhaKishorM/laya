"""Edge-case sweep: the inference paths the repo's other suites do not cover.

`tests/test_local_e2e.py` and `laya_smoke_test.py` check that Laya answers the right things.
This checks the shapes and API paths around that: one forward pass answering many questions,
high-cardinality choice sets (the `head_max_len` option budget), odd input shapes and encodings,
long-context truncation, the router lifecycle (preload / LRU / attach / unload) and explicit
device selection, including CPU-vs-MPS agreement.

Needs the local checkpoints (~2.3 GB, see setup_laya.sh):

    .venv/bin/python verify/edge_sweep.py [--models ./models] [--device auto|cpu|mps]

Exits non-zero if a check fails.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # repository root; <root>/laya/email.py must not shadow stdlib email

import torch  # noqa: E402

import laya  # noqa: E402
from laya.common import build_sequence  # noqa: E402
from laya.router import Router  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print("   %s %-50s %s" % ("PASS" if ok else "FAIL", name, detail if not ok else ""))
    if not ok:
        FAILS.append("%s (%s)" % (name, detail))


def head(title):
    print("\n" + "=" * 74 + "\n  " + title + "\n" + "=" * 74)


def shapes_ok(answers, expected):
    keys = set(answers) == set(expected)
    probs = all(abs(sum(v["probabilities"].values()) - 1.0) < 0.01
                for v in answers.values() if "probabilities" in v)
    serialisable = isinstance(json.dumps(answers), str)
    return keys and probs and serialisable, "keys=%s probs=%s json=%s" % (keys, probs, serialisable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    models = {"english": os.path.join(args.models, "laya"),
              "multilingual": os.path.join(args.models, "laya-multilingual"),
              "typed-decisions": os.path.join(args.models, "laya-typed-decisions")}
    for name, path in models.items():
        if not os.path.exists(os.path.join(path, "model.safetensors")):
            sys.exit("missing checkpoint %r at %s -- run ./setup_laya.sh" % (name, path))
    device = None if args.device == "auto" else args.device

    mixed = {"a": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
             "b": {"type": "score", "instructions": "Severity?", "criteria": ["low", "mid", "high", "critical"]},
             "c": {"type": "noul", "instructions": "Is it urgent?"},
             "d": {"type": "noul", "instructions": "Is there a threat to cancel?"},
             "e": {"type": "score", "instructions": "Frustration?", "criteria": ["calm", "annoyed", "angry"]},
             "f": {"type": "choice", "instructions": "Language?", "criteria": ["en", "de", "other"]}}
    # a rubric rich enough that a duplicate-charge mail has a fair chance at "billing"
    cats = {"billing": "invoices, payments, refunds", "technical": "bugs, outages, integrations",
            "sales": "pricing, demos, new purchases", "hr": "hiring, leave, payroll"}
    triage = {"dept": {"type": "choice", "instructions": "Which team should handle `message`?", "criteria": cats},
              "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}

    # ------------------------------------------------------------ 1. many questions, one pass
    head("1. One forward pass answering many questions")
    en = laya.load(models["english"], device=device)
    many = {"q%02d" % i: {"type": "noul", "instructions": "Does the message mention item %d?" % i}
            for i in range(12)}
    t0 = time.perf_counter()
    answers = en.predict({"message": "Item 3 and item 7 were both double charged on invoice 4411."},
                         many)["answers"]
    print("      12-question pass: %.0f ms" % ((time.perf_counter() - t0) * 1000))
    report("12 questions in one pass", shapes_ok(answers, many)[0], shapes_ok(answers, many)[1])
    report("mixed primitives in one pass",
           shapes_ok(en.predict({"body": "Rückerstattung bitte!"}, mixed)["answers"], mixed)[0])

    # ------------------------------------------------------------ 2. high-cardinality choice
    head("2. High-cardinality choice (the head_max_len option budget)")
    for n in (20, 77, 120):
        q = {"dept": {"type": "choice", "instructions": "Which intent?",
                      "criteria": {"intent_%03d" % i: "label number %d" % i for i in range(n)}}}
        try:
            a = en.predict({"message": "I was charged twice for invoice 4411"}, q)["answers"]["dept"]
            report("%d-option choice answers" % n, len(a["probabilities"]) == n,
                   "got %d labels, top=%s p=%.3f" % (len(a["probabilities"]), a["choice"],
                                                     max(a["probabilities"].values())))
        except Exception as e:
            report("%d-option choice answers" % n, False, "%s: %s" % (type(e).__name__, str(e)[:80]))
    q = {"s": {"type": "score", "instructions": "Rate 0-19?", "criteria": ["level %d" % i for i in range(20)]}}
    try:
        a = en.predict({"message": "bad"}, q)["answers"]["s"]
        report("20-level score question", len(a["probabilities"]) == 20, "score=%.2f" % a["score"])
    except Exception as e:
        report("20-level score question", False, "%s: %s" % (type(e).__name__, str(e)[:80]))

    # ------------------------------------------------------------ 3. input shapes
    head("3. Input shapes and encodings")
    for name, state in [("empty string", ""),
                        ("whitespace only", "   \n\t  "),
                        ("emoji + mixed scripts", "退款! Rückerstattung! मुझे पैसे चाहिए 🙂🧾"),
                        ("JSON document", {"a": 1, "nested": {"b": [1, 2, 3]}, "note": "üñíçødé"}),
                        ("conversation turns", [{"role": "user", "content": "charged twice"},
                                                {"role": "agent", "content": "checking"},
                                                {"role": "user", "content": "refund now"}])]:
        try:
            report("state: %s" % name, shapes_ok(en.predict(state, mixed)["answers"], mixed)[0])
        except Exception as e:
            report("state: %s" % name, False, "%s: %s" % (type(e).__name__, str(e)[:80]))

    # ------------------------------------------------------------ 4. long-context truncation
    # Note: usage["input_tokens"] counts every question in the call, so it is a batch total.
    # Truncation is therefore checked per sequence, through build_sequence directly.
    head("4. Long-context truncation (per sequence)")
    long_text = "We were billed twice for invoice 4411 and nobody has replied. " * 300
    for name in ("english", "multilingual"):
        agent = en if name == "english" else laya.load(models["multilingual"], device=device)
        max_len, head_len = agent.cfg["max_len"], agent.cfg["head_max_len"]
        lengths = [len(build_sequence(agent.tok, {"body": long_text}, agent._to_internal(mixed[qid]),
                                      max_len, head_len)[0]) for qid in mixed]
        single = agent.predict({"body": long_text}, {"q": mixed["c"]})["usage"]["input_tokens"]
        batch = agent.predict({"body": long_text}, mixed)["usage"]["input_tokens"]
        report("%s truncates to max_len=%d" % (name, max_len),
               max(lengths) <= max_len and single <= max_len and batch == sum(lengths),
               "per-seq max=%d single=%d batch=%d sum=%d" % (max(lengths), single, batch, sum(lengths)))
        if name != "english":
            del agent
    del en

    # ------------------------------------------------------------ 5. multilingual breadth
    head("5. Multilingual: scripts and long context")
    ml = laya.load(models["multilingual"], device=device)
    for label, text in [("german", "Ich wurde zweimal fuer Rechnung 4411 belastet, bitte erstatten Sie den Betrag."),
                        ("chinese", "发票4411被重复扣款，请今天退款。"),
                        ("arabic", "تم خصم المبلغ مرتين للفاتورة 4411، يرجى رد المبلغ اليوم."),
                        ("khmer", "វិក្កយបត្រ 4411 ត្រូវបានគិតថ្លៃពីរដង សូមសងប្រាក់វិញ។")]:
        a = ml.predict({"message": text}, triage)["answers"]
        report("%s duplicate-charge -> billing" % label, a["dept"]["choice"] == "billing",
               "got %s (p=%.2f)" % (a["dept"]["choice"], max(a["dept"]["probabilities"].values())))
    del ml

    # ------------------------------------------------------------ 6. router lifecycle
    head("6. Router lifecycle: preload, LRU, attach, unload")
    r = Router(models=models, device=device, max_loaded=2, preload=True)
    report("preload=True loads all three", sorted(r.loaded) == ["english", "multilingual", "typed-decisions"],
           "loaded=%s" % sorted(r.loaded))
    r.unload()
    report("unload() frees everything", r.loaded == [], "loaded=%s" % r.loaded)

    r = Router(models=models, device=device, max_loaded=2)
    for n in ("english", "multilingual", "typed-decisions"):
        r.load(n)
    report("max_loaded=2 evicts least-recently-used", r.loaded == ["multilingual", "typed-decisions"],
           "loaded=%s" % r.loaded)
    spare = laya.load(models["english"], device=device)
    r.attach("english", spare)
    report("attach() registers an existing agent", "english" in r.loaded, "loaded=%s" % r.loaded)
    del spare
    r.unload()
    report("unload() after attach", r.loaded == [], "loaded=%s" % r.loaded)

    # ------------------------------------------------------------ 7. device selection
    head("7. Explicit device selection")
    for dev in ("cpu", "mps"):
        if dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("   SKIP mps not available on this machine")
            continue
        agent = laya.load(models["multilingual"], device=dev)
        try:
            a = agent.predict({"message": "Ich wurde zweimal belastet, bitte erstatten."}, triage)["answers"]
            report("device=%r runs (resolved to %s)" % (dev, agent.device.type),
                   agent.device.type == dev and shapes_ok(a, triage)[0])
            if dev == "mps":
                cpu = laya.load(models["multilingual"], device="cpu")
                same = (json.dumps(cpu.predict({"message": "Ich wurde zweimal belastet, bitte erstatten."},
                                               triage)["answers"], sort_keys=True)
                        == json.dumps(a, sort_keys=True))
                report("MPS answer is identical to CPU", same)
                del cpu
        except Exception as e:
            report("device=%r runs" % dev, False, "%s: %s" % (type(e).__name__, str(e)[:80]))
        del agent

    head("RESULT")
    if FAILS:
        print("   %d check(s) failed:" % len(FAILS))
        for f in FAILS:
            print("     - " + f)
        return 1
    print("   all edge-case checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
