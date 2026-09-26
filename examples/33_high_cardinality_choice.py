"""Example 33 -- high-cardinality choice questions and the option token budget.

Laya splits every sequence into an option head (`head_max_len`) and a state body
(`max_len - head_max_len`). A choice question with many options shares one fixed head budget, so
at some point each option gets only a few tokens. This example measures that squeeze on a
20-option and a 77-option (Banking77) question, then raises the budgets to show the answer and
the token budget recover.
"""

from _common import laya, banner, device_line, load

from laya.common import build_sequence

banner("33", "High-cardinality choice", """
    Laya formats a call as:

      [CLS] <typed question + option descriptions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state

    The option block lives in `head_max_len`; the state gets what is left of `max_len`. When the
    full options overflow the head, the code grants each one

      per = max(4, (head_max_len - 16) // N)

    tokens (`16` is the reserved minimum for the instruction prefix). At the English default
    head_max_len = 192 a 77-option Banking77 question gets `max(4, (192-16)//77) = 4` tokens per
    option -- the `[MASK]` plus three subwords -- so labels stop being distinguishable. The
    instruction is then clipped to its 8-token floor as well.

    This is the documented weak spot: raise `head_max_len`/`max_len`, or split the label space
    coarse-to-fine. Notice how the answer changes here, and that confidence does not warn you.
    """)

# --- a real Banking77 message and its 77 real intent labels ---------------------------------
MESSAGE = {"text": "My card was swallowed by the ATM this morning. How do I get it back?"}
EXPECTED = "card_swallowed"

BANKING77 = [
    "card_arrival", "card_linking", "exchange_rate", "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement", "pending_cash_withdrawal", "fiat_currency_support",
    "card_delivery_estimate", "automatic_top_up", "card_not_working", "exchange_via_app",
    "lost_or_stolen_card", "age_limit", "pin_blocked", "contactless_not_working",
    "top_up_by_bank_transfer_charge", "pending_top_up", "cancel_transfer", "top_up_limits",
    "wrong_amount_of_cash_received", "card_payment_fee_charged", "transfer_not_received_by_recipient",
    "supported_cards_and_currencies", "getting_virtual_card", "card_acceptance",
    "top_up_reverted", "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised", "edit_personal_details", "why_verify_identity",
    "unable_to_verify_identity", "get_physical_card", "visa_or_mastercard", "topping_up_by_card",
    "disposable_card_limits", "compromised_card", "atm_support",
    "direct_debit_payment_not_recognised", "passcode_forgotten", "declined_cash_withdrawal",
    "pending_card_payment", "lost_or_stolen_phone", "request_refund", "declined_transfer",
    "refund_not_showing_up", "declined_card_payment", "pending_transfer", "terminate_account",
    "card_swallowed", "transaction_charged_twice", "verify_source_of_funds", "transfer_timing",
    "reverted_card_payment", "change_pin", "beneficiary_not_allowed", "transfer_fee_charged",
    "receiving_money", "failed_transfer", "transfer_into_account", "verify_top_up",
    "getting_spare_card", "top_up_by_cash_or_cheque", "order_physical_card",
    "virtual_card_not_working", "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card", "top_up_failed", "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised", "exchange_charge", "top_up_by_card_charge",
    "activate_my_card", "cash_withdrawal_charge", "card_about_to_expire",
    "apple_pay_or_google_pay", "verify_my_identity", "country_support",
]
TWENTY = [lbl for lbl in BANKING77 if lbl in (
    "card_swallowed", "atm_support", "card_arrival", "card_not_working", "lost_or_stolen_card",
    "card_linking", "activate_my_card", "card_delivery_estimate", "order_physical_card",
    "get_physical_card", "getting_virtual_card", "virtual_card_not_working", "card_acceptance",
    "contactless_not_working", "pin_blocked", "change_pin", "passcode_forgotten",
    "card_about_to_expire", "apple_pay_or_google_pay", "supported_cards_and_currencies")]

def intent_question(labels):
    return {
        "type": "choice",
        "instructions": "Which banking intent best matches the customer message?",
        "criteria": {lbl: "customer asks about " + lbl.replace("_", " ") for lbl in labels},
    }

def internal(question):
    return {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}

def option_lengths(tok, question):
    """Tokens per option before the squeeze, capped at 48 exactly as `build_sequence` does."""
    lengths = []
    for opt in laya.render_options(internal(question)):
        body = tok(" " + opt.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        lengths.append(1 + len(body[:48]))          # +1 for the option's own [MASK]
    return lengths

def budget_report(agent, question, max_len, head_max_len):
    """Print the budget arithmetic and return the measured token span of each option."""
    n = len(question["criteria"])
    full = option_lengths(agent.tok, question)
    if head_max_len - sum(full) < 16:
        per = max(4, (head_max_len - 16) // n)
        print("   sum(options)=%d > head_max_len-16=%d, so per = max(4, (%d-16)//%d) = %d tokens"
              % (sum(full), head_max_len - 16, head_max_len, n, per))
    else:
        print("   sum(options)=%d <= head_max_len-16=%d, so every option keeps its full length"
              % (sum(full), head_max_len - 16))

    ids, markers = build_sequence(agent.tok, MESSAGE, internal(question), max_len, head_max_len)
    sep_id = agent.tok.sep_token_id

    def block_end(i):
        if i + 1 < len(markers):
            return markers[i + 1]
        j = markers[i]                              # last option: stop at the [SEP] after it
        while j < len(ids) and ids[j] != sep_id:
            j += 1
        return j

    spans = [block_end(i) - markers[i] for i in range(len(markers))]
    print("   measured: %d/%d options present, %d..%d tokens each (avg %.1f), "
          "instruction prefix %d tokens, sequence %d of max_len=%d"
          % (len(markers), n, min(spans), max(spans), sum(spans) / max(1, len(spans)),
             markers[0] - 2, len(ids), max_len))     # markers[0] = [CLS] + instructions + [SEP]
    return spans

def ask(agent, question, label):
    answer = agent.predict(MESSAGE, {"intent": question})["answers"]["intent"]
    ranked = sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
    print("   %-24s -> %-22s p=%.3f conf=%.3f  %s   next: %s"
          % (label, answer["choice"], ranked[0][1], answer["confidence"],
             "HIT" if answer["choice"] == EXPECTED else "MISS", "  ".join(
                 "%s=%.3f" % kv for kv in ranked[1:3] if kv[1] > 0.0001) or "all others 0.000"))

agent = load("english")
device_line(agent)
print("   defaults: max_len=%d  head_max_len=%d" % (agent.cfg["max_len"], agent.cfg["head_max_len"]))
print("   state: %r   expected intent: %s" % (MESSAGE["text"][:66], EXPECTED))

twenty = intent_question(TWENTY)
seventy_seven = intent_question(BANKING77)

print("\n   == 20 options (a reduced label space), head_max_len=%d ==" % agent.cfg["head_max_len"])
budget_report(agent, twenty, agent.cfg["max_len"], agent.cfg["head_max_len"])
ask(agent, twenty, "20 options, default")

print("\n   == 77 options, head_max_len=%d ==" % agent.cfg["head_max_len"])
budget_report(agent, seventy_seven, agent.cfg["max_len"], agent.cfg["head_max_len"])
ask(agent, seventy_seven, "77 options, default")

# --- cfg is mutable at runtime: give the option block room ----------------------------------
for head, ctx in ((512, 1024), (1024, 2048)):
    agent.cfg["head_max_len"], agent.cfg["max_len"] = head, ctx
    print("\n   == 77 options, head_max_len=%d, max_len=%d ==" % (head, ctx))
    budget_report(agent, seventy_seven, ctx, head)
    ask(agent, seventy_seven, "77 options, raised")

print("\n   confidence is 1.000 either way: the `choice:11+` temperature (0.1006) sharpens the")
print("   logits, so it cannot warn you that a 4-token option was misread. Raise the budget or")
print("   split the label space; do not trust confidence on large option sets.")
