"""Example 07 -- a structured dict as the state.

`state` need not be a string. A dict is serialised to JSON for you, so a record with
nested fields can be passed straight in and questioned about its contents.
"""
import json

from _common import banner, describe, device_line, heading, load

banner("07", "A JSON record as the state", """
    `predict()` accepts a str, a dict, or a list of conversation turns. A dict is
    serialised with `json.dumps(state, ensure_ascii=False)` and the resulting JSON text
    becomes the state the model reads.

    That is the whole trick -- there is no schema engine and no field lookup. Laya reads
    the serialised record as text, so the keys, values and wording all shape the answer,
    and deeply nested or ambiguous fields are harder for it than a flat status string.
    This record keeps the customer nested and the order statuses explicit, and the
    questions below land on the right side of a 0.5 threshold.
    """)

agent = load("english")
device_line(agent)

ORDER = {
    "order_id": "ORD-7781",
    "customer": {
        "name": "Dana Reyes",
        "email": "dana@example.com",
        "tier": "gold",
    },
    "placed_on": "2024-03-11",
    "items": "1 mechanical keyboard, 2 USB-C cables",
    "item_count": 2,
    "order_total": 154.00,
    "payment_status": "captured",
    "delivery_status": "delivered",
    "delivery_carrier": "DHL",
    "refund_status": "requested",
    "refund_amount": 129.00,
    "customer_note": "The mechanical keyboard arrived with a cracked case; I would like a "
                     "refund for it.",
}

heading("the dict, serialised exactly as the model sees it")
serialised = json.dumps(ORDER, ensure_ascii=False)
print("   %s" % serialised)
print("\n   %d characters of JSON -- that string is the state for every question below."
      % len(serialised))

QUESTIONS_FOR_ORDER = {
    "is_gold": {"type": "noul", "instructions": "Is this customer on the gold tier?"},
    "was_delivered": {"type": "noul", "instructions": "Was this order delivered?"},
    "payment_captured": {"type": "noul",
                         "instructions": "Has the payment for this order been captured?"},
    "refund_requested": {"type": "noul",
                         "instructions": "Has the customer asked for a refund on this order?"},
    "item_damaged": {"type": "noul", "instructions": "Was an item damaged on arrival?"},
    "carrier": {"type": "choice", "instructions": "Which carrier is used for this order?",
                "criteria": {"dhl": "DHL", "ups": "UPS", "fedex": "FedEx", "other": "other"}},
    "refund_state": {"type": "choice",
                     "instructions": "What is the state of the refund on this order?",
                     "criteria": {"none": "no refund is involved",
                                  "requested": "the customer asked for a refund",
                                  "approved": "the refund was approved and paid"}},
}

heading("questions about its fields")
result = agent.predict(ORDER, QUESTIONS_FOR_ORDER)
describe(result["answers"], indent="   ")
print("\n   usage: %d input tokens, %d output tokens"
      % (result["usage"]["input_tokens"], result["usage"]["output_tokens"]))
