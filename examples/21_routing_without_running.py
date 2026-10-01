"""Example 21 -- routing decisions without running a model.

Calls `router.route(...)` only: pick a checkpoint from the state and its detected script,
paying microseconds and loading no weights at all.
"""
import time

from _common import banner, heading, router, timed

banner("21", "Route without running", """
    `route()` is the whole routing decision and nothing else: no checkpoint is built, no
    forward pass happens. It is pure-Python script and language detection, so it costs
    microseconds per call -- cheap enough to run on every request, before you decide
    whether a model is needed at all.

    The signal that matters most is script: the English checkpoint does not degrade
    gracefully off English, it collapses while staying confident, so non-Latin text must
    never reach it. Latin-script languages get a best-effort language guess on top.
    """)

r = router()          # builds no checkpoints; just wires the three names to local paths

heading("route() alone: no weights resident before or after")
print("   router.loaded before: %r" % r.loaded)

CASES = [
    ("English", "Hi, we were billed twice for March. Please refund the duplicate charge."),
    ("German", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung."),
    ("Hindi", "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"),
    ("Japanese", "同じ請求が二回発生しました。返金をお願いします。"),
    ("Arabic", "تم خصم المبلغ مرتين من بطاقتي، أرجو إعادة المبلغ."),
    ("Thai", "ถูกเรียกเก็บเงินซ้ำสองครั้ง กรุณาคืนเงินด้วย"),
    ("French", "Le client a été facturé deux fois, il demande un remboursement immédiat."),
    ("Korean", "요금이 두 번 청구되었습니다. 환불해 주세요."),
    ("Russian", "С меня дважды списали деньги, пожалуйста, верните средства."),
    ("no letters", "!!! 12345 ???"),
]

print("   %-11s %-13s %-9s %-6s %-7s %s" %
      ("input", "model", "script", "lang", "english", "reason"))
for name, text in CASES:
    d = r.route({"body": text})
    det = d["detection"]
    print("   %-11s %-13s %-9s %-6s %-7s %s" % (
        name, d["model"], det["script"], det["language"] or "-",
        det["is_english"], d["reason"]))

heading("the detection payload behind a decision")
d = r.route({"body": CASES[2][1]})
for key, value in d["detection"].items():
    print("   %-18s %s" % (key, value))
print("   %-18s %r" % ("workflow", d["workflow"]))

heading("explicit overrides win over detection")
routed = r.route({"body": CASES[2][1]})
forced_model = r.route({"body": CASES[2][1]}, model="english")
forced_lang = r.route({"body": CASES[0][1]}, lang="de")
for label, d in [("detected Hindi", routed),
                 ("model='english'", forced_model),
                 ("lang='de'", forced_lang)]:
    print("   %-18s -> %-13s %s" % (label, d["model"], d["reason"]))

heading("what that costs")
_, median_ms = timed(lambda: r.route({"body": CASES[1][1]}), repeat=200)
t0 = time.perf_counter()
for _, text in CASES:
    r.route({"body": text})
total_ms = (time.perf_counter() - t0) * 1000
print("   one route() call       %.3f ms (median of 200)" % median_ms)
print("   all %d cases           %.3f ms total" % (len(CASES), total_ms))
print("   router.loaded after:   %r" % r.loaded)
print("""
   Ten routing decisions cost well under a millisecond in total, and the router still
   holds zero checkpoints. `route()` is therefore safe to call on every inbound request,
   even one you end up answering with a conventional LLM.
   """)
