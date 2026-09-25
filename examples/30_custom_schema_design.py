"""Example 30 -- writing your own question schema.

The presets are fixed schemas; this builds a four-question bug-triage schema from scratch
as ordinary Python, runs it on two contrasting reports, then takes the design apart.
"""

from _common import banner, describe, device_line, heading, load, top

banner("30", "Designing a question schema", """
    A preset is just a Python dict: a question id, a `type`, an `instructions` string and
    (for choice and score) `criteria`. Writing your own is therefore API design, not a
    model change -- the same checkpoint answers whatever you ask, in the same one forward
    pass.

    The schema below triages inbound bug reports on four axes:

      category        choice   what kind of bug is it?
      severity        score    how bad is it, on an ordered scale?
      reproducible    noul     can an engineer reproduce it?
      data_loss_risk  noul     might data be lost or corrupted?

    It runs on a cosmetic UI report and on a crash that corrupts saved data; the design
    notes afterwards use the values that actually came back.
    """)

# --- the schema, exactly as the checkpoint sees it -----------------------------------------
BUG_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What kind of bug report is `message`?",
        "criteria": {
            "crash": "a crash, exception or freeze",
            "performance": "slow or high resource use",
            "ui": "layout, styling or usability",
            "data_loss": "lost, corrupted or missing data",
            "other": "none of the other options fits",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the bug in `message`?",
        "criteria": ["cosmetic or trivial", "annoying but workable", "blocks a workflow",
                     "data loss or security impact"],
    },
    "reproducible": {
        "type": "noul",
        "instructions": "Does `message` give steps or conditions that make the bug reproducible?",
    },
    "data_loss_risk": {
        "type": "noul",
        "instructions": "Does `message` suggest that data may be lost or corrupted?",
    },
}

# --- two contrasting records, plus a probe for the criteria discussion ---------------------
RECORDS = [
    ("cosmetic UI report", {
        "message": "On mobile the settings page buttons overlap and the Save button is cut off, "
                   "so I cannot tap it. Desktop looks fine.",
    }),
    ("data-corrupting crash", {
        "message": "Saving an invoice with more than 200 lines crashes the app and the saved "
                   "data is corrupted; the invoice comes back blank. Steps: open the invoice, "
                   "add a line, press Save. Reproduced three times.",
    }),
]

PROBE = ("label typo", {
    "message": "The export button has a typo in its label -- it says 'Exprot'. Everything "
               "else works fine.",
})

agent = load("english")
device_line(agent)

runs = []
for label, state in RECORDS:
    heading(label)
    result = agent.predict(state, BUG_QUESTIONS)
    runs.append((label, result["answers"], result))
    describe(result["answers"])
    print("   %d input tokens, %d output tokens (one forward pass for %d questions)"
          % (result["usage"]["input_tokens"], result["usage"]["output_tokens"],
             len(BUG_QUESTIONS)))

ui, crash = runs[0][1], runs[1][1]
heading("the two records as rows")
print("   %-24s %-11s %-8s %-13s %s"
      % ("record", "category", "severity", "reproducible", "data_loss_risk"))
for label, answers, _ in runs:
    print("   %-24s %-11s %-8.2f %-13.3f %.3f"
          % (label, answers["category"]["choice"], answers["severity"]["score"],
             answers["reproducible"]["noul"], answers["data_loss_risk"]["noul"]))

heading("1. how many options?")
print("   `category` has %d options, and both records are still decided clearly:"
      % len(BUG_QUESTIONS["category"]["criteria"]))
print("   UI %.3f confidence, crash %.3f confidence."
      % (ui["category"]["confidence"], crash["category"]["confidence"]))
print("   Add an option only if your policy acts on it differently. More labels split the")
print("   same evidence, cost headroom, and (past ten options) run into the sharpening")
print("   temperature from examples 33 and 34 -- a label nobody routes on is pure noise.")

heading("2. criteria must be mutually exclusive")
print("   The crash record is both a `crash` and a `data_loss` event, so those two options")
print("   overlap by construction. The model still answered %s at p=%.3f -- but do not"
      % (crash["category"]["choice"], max(crash["category"]["probabilities"].values())))
print("   rely on that. Write a precedence rule into the instructions when two labels can")
print("   both apply, for example: \"if data is corrupted, choose data_loss even when the")
print("   trigger was a crash\". Otherwise two teams can each believe the ticket is theirs.")
probe = agent.predict(PROBE[1], BUG_QUESTIONS)["answers"]["category"]
print("   A probe shows the other edge of the same idea -- the taxonomy only covers what you")
print("   name: a label typo lands in %s. Top labels: %s"
      % (probe["choice"], top(probe["probabilities"], 3)))
print("   If copy defects matter to the policy, add them to a criterion.")

heading("3. when score beats choice")
print("   `severity` is ordered, and the two records land at different points: crash %.2f / 3"
      % crash["severity"]["score"])
print("   versus UI %.2f / 3. A choice question (\"severe: yes/no\") would throw that ordering"
      % ui["severity"]["score"])
print("   away. Use a score when the levels are ordered and you will threshold or compare")
print("   them, and a choice when the labels are unordered. Read the score itself: `confidence`")
print("   on a score is normalised entropy, so a wide-but-ordered distribution looks")
print("   unconfident even when the expected level is informative.")

heading("4. every question shares one forward pass")
tokens = runs[0][2]["usage"]["input_tokens"]
print("   All %d questions were answered in one `predict()` call: %d input tokens, %d output"
      % (len(BUG_QUESTIONS), tokens, runs[0][2]["usage"]["output_tokens"]))
print("   tokens. The state is repeated once per question row and the questions are rows of a")
print("   single batch -- splitting this into four calls would return the same answers with")
print("   four sets of kernel launches (example 36 measures that difference). `output_tokens`")
print("   is zero because Laya generates no text: the answers are heads on that one pass.")
print("""
   The schema is the interface: the checkpoint, the device and the number of passes are
   unchanged by how many questions you stack into it. What changes is how many decisions
   you get back for the cost of one.
   """)
