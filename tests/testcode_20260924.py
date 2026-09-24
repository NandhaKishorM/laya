from laya import Router

router = Router()
questions = {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "refunds, invoices", "technical": "bugs, outages"}},
    "urgency":    {"type": "score",  "instructions": "How urgent?",
                   "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul",   "instructions": "Does the user threaten to leave?"},
}
r = router.predict("We were billed twice, refund today or we cancel.", questions)
print(r["answers"]["department"]["choice"], r["answers"]["churn_risk"]["noul"], r["routing"]["model"])