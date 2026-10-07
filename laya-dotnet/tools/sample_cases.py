"""The two states and four questions the .NET sample app asks, as golden cases.

Kept beside the other cases rather than inline in `golden_cases.py`'s literal list because
these mirror a *different* file: `laya-dotnet/samples/Laya.Sample/Program.cs`. If that sample
changes, these must change with it, and a separate module makes that coupling visible.

The Hindi body is written as escapes deliberately. It was extracted from the sample source
programmatically, not retyped: an earlier hand transcription put U+0915 U+094B (ko) where
the sample has U+091A (ca), which silently compared two different sentences and looked like
a parity bug in the C# port.
"""

SAMPLE_ENGLISH_STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan.",
}

SAMPLE_HINDI_STATE = {
    "body": "मार्च का बिल "
            "दो बार लिया गया, "
            "कृपया रिफंड "
            "करें।",
}

# Option order is load-bearing: choice labels are positional, so this dict doubles as the
# order fixture for the sample's QuestionSet.
SAMPLE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

SAMPLE_CASES = [
    # The sample's English run. Dict state, so this also pins serialize_state's dialect on
    # the exact payload a user sees printed on their first run of the SDK.
    {
        "name": "sample_app_english",
        "state": SAMPLE_ENGLISH_STATE,
        "questions": SAMPLE_QUESTIONS,
    },
    # The sample's Hindi run: the same questions against Devanagari, which is the claim that
    # the multilingual checkpoint needs no router. Byte-fallback territory for the tokenizer.
    {
        "name": "sample_app_hindi",
        "state": SAMPLE_HINDI_STATE,
        "questions": SAMPLE_QUESTIONS,
    },
]
