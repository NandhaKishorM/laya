"""Label-free, lossless comparison inputs and exact finite-quantity arithmetic."""
import copy
import json
import operator
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

VERDICTS = ("supported", "contradicted", "insufficient_context")
OPERATORS = ("lt", "le", "gt", "ge", "eq", "ne")
SEMANTICS = "Use the first two named quantities in their listed order. Missing fields or values, unsupported units, or incompatible dimensions make the assertion undetermined."
RELATIONS = (
    "The first quantity is strictly less than the second quantity.",
    "The first quantity is less than or equal to the second quantity.",
    "The first quantity is strictly greater than the second quantity.",
    "The first quantity is greater than or equal to the second quantity.",
    "The first quantity is equal to the second quantity.",
    "The first quantity is not equal to the second quantity.",
)
INSTRUCTION = ("choice question: Select the relationship expressed by the assertion between its first "
               "and second named quantities. Interpret negation. Identify the meaning, not whether "
               "the assertion is true. No quantity values are supplied.")
QUESTIONS = {"verdict": {"type": "choice", "instructions": "Judge the assertion using only the supplied field values and units.",
                        "criteria": dict(zip(VERDICTS, ("The quantities establish the assertion.",
                                                       "The quantities establish that the assertion is false.",
                                                       "Required quantities or compatible units are unavailable.")))}}
UNITS = {
    "mg": ("mass", Fraction(1, 1000)), "g": ("mass", Fraction(1)), "kg": ("mass", Fraction(1000)),
    "mm": ("length", Fraction(1, 1000)), "cm": ("length", Fraction(1, 100)), "m": ("length", Fraction(1)),
    "ms": ("duration", Fraction(1, 1000)), "s": ("duration", Fraction(1)), "min": ("duration", Fraction(60)),
    "B": ("storage", Fraction(1)), "kB": ("storage", Fraction(1000)), "KiB": ("storage", Fraction(1024)),
    "MB": ("storage", Fraction(1000000)), "MiB": ("storage", Fraction(1048576)),
}


def validate_state(state):
    if not isinstance(state, dict) or state.get("task") != "numeric_comparison":
        raise ValueError("Expected task=numeric_comparison")
    if set(state) - {"task", "assertion", "semantics", "fields"}:
        raise ValueError("Unsupported comparison state keys")
    text = state.get("assertion")
    if not isinstance(text, str) or not text.strip() or len(text) > 16384:
        raise ValueError("Assertion must be nonempty text of at most 16384 characters")
    if state.get("semantics", SEMANTICS) != SEMANTICS:
        raise ValueError("Unsupported comparison semantics")
    fields = state.get("fields")
    if not isinstance(fields, list) or len(fields) > 7:
        raise ValueError("Expected at most seven named fields")
    names = []
    for field in fields:
        if not isinstance(field, dict) or set(field) - {"name", "value", "unit"}:
            raise ValueError("Fields accept only name, value and unit")
        name = field.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 256:
            raise ValueError("Field names must be nonempty strings of at most 256 characters")
        if field.get("unit") is not None and (not isinstance(field["unit"], str) or len(field["unit"]) > 64):
            raise ValueError("Unit must be a short string or null")
        value = field.get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int, float))):
            raise ValueError("Value must be a number, decimal string or null")
        if len(str(value)) > 80:
            raise ValueError("Numeric values may contain at most 80 characters")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate field names are ambiguous")
    result = copy.deepcopy(state)
    result["semantics"] = SEMANTICS
    return result


def quantity(field):
    unit, value = field.get("unit"), field.get("value")
    if not isinstance(unit, str) or unit not in UNITS or value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, int, float)) or len(str(value)) > 80:
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number.adjusted()) > 40:
            return None
        dimension, scale = UNITS[unit]
        return dimension, Fraction(number) * scale
    except (InvalidOperation, ValueError, TypeError):
        return None


def comparison_table(fields):
    import torch
    result = torch.zeros(8, 8, 6, 3)
    result[..., 2] = 1
    values = [quantity(field) for field in fields]
    for i, left in enumerate(values):
        for j, right in enumerate(values):
            if left is None or right is None or left[0] != right[0]:
                continue
            for op, function in enumerate((operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne)):
                label = 0 if function(left[1], right[1]) else 1
                result[i, j, op, 2] = 0
                result[i, j, op, label] = 1
    return result


def encode(state, tok):
    """Build both network inputs without annotations, labels or truncation."""
    from .common import render_options, serialize_state
    text = serialize_state(state)
    if any(token in text for token in tok.all_special_tokens):
        raise ValueError("Comparison inputs cannot contain reserved tokenizer tokens")
    tokenize = lambda value: tok(value, add_special_tokens=False)["input_ids"]
    claim = tokenize(state["assertion"])
    if not claim:
        raise ValueError("Empty tokenized assertion")
    op_ids = [tok.cls_token_id] + tokenize(INSTRUCTION) + [tok.sep_token_id]
    op_positions = []
    for relation in RELATIONS:
        op_positions.append(len(op_ids))
        op_ids += [tok.mask_token_id] + tokenize(relation)
    op_start = len(op_ids) + 1
    op_ids += [tok.sep_token_id] + claim + [tok.sep_token_id]
    q = QUESTIONS["verdict"]
    head = tokenize("choice question: " + q["instructions"])
    ids = [tok.cls_token_id] + head + [tok.sep_token_id]
    positions = []
    for option in render_options({"t": "choice", "crit": q["criteria"]}):
        positions.append(len(ids))
        ids += [tok.mask_token_id] + tokenize(" " + option)
    state_start = len(ids) + 1
    tokenized = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids += [tok.sep_token_id] + tokenized["input_ids"] + [tok.sep_token_id]
    if len(op_ids) > 512 or len(ids) > 1024:
        raise ValueError("Comparison input exceeds token budget; truncation is forbidden")
    quoted = json.dumps(state["assertion"], ensure_ascii=False)
    start = text.index('"assertion": ' + quoted) + len('"assertion": ')
    offsets = tokenized["offset_mapping"]
    assertion = [state_start + j for j, (a, b) in enumerate(offsets) if a < start + len(quoted) and b > start]
    if not assertion:
        raise ValueError("Cannot locate assertion tokens")
    links, owners = [], {}
    for field_index, field in enumerate(state["fields"]):
        linked = set()
        for match in re.finditer(r"(?<!\w)" + re.escape(field["name"]) + r"(?!\w)", state["assertion"]):
            a = start + len(json.dumps(state["assertion"][:match.start()], ensure_ascii=False)) - 1
            b = start + len(json.dumps(state["assertion"][:match.end()], ensure_ascii=False)) - 1
            linked.update(state_start + j for j, (x, y) in enumerate(offsets) if x < b and y > a)
        for index in linked:
            if index in owners and owners[index] != field_index:
                raise ValueError("A token overlaps two field names")
            owners[index] = field_index
        links.append(sorted(linked))
    return {"op_ids": op_ids, "op_positions": op_positions, "op_start": op_start, "op_length": len(claim),
            "ids": ids, "positions": positions, "assertion": assertion, "links": links,
            "table": comparison_table(state["fields"])}


def collate(rows, pad_id):
    import torch

    def padded(key):
        ids = torch.full((len(rows), max(len(r[key]) for r in rows)), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, row in enumerate(rows):
            ids[i, :len(row[key])] = torch.tensor(row[key])
            mask[i, :len(row[key])] = True
        return ids, mask

    op_ids, op_mask = padded("op_ids")
    claim = torch.zeros_like(op_mask)
    ids, mask = padded("ids")
    assertion = torch.zeros_like(mask)
    links = torch.zeros(len(rows), 8, ids.shape[1])
    fields = torch.zeros(len(rows), 8, dtype=torch.bool)
    fields[:, 7] = True
    for i, row in enumerate(rows):
        claim[i, row["op_start"]:row["op_start"] + row["op_length"]] = True
        assertion[i, row["assertion"]] = True
        fields[i, :len(row["links"])] = True
        for field, indexes in enumerate(row["links"]):
            links[i, field, indexes] = 1
        links[i, 7] = 1 - links[i, :7].sum(0)
    return ({"ids": op_ids, "mask": op_mask, "claim_mask": claim,
             "positions": torch.tensor([r["op_positions"] for r in rows])},
            {"ids": ids, "mask": mask, "positions": torch.tensor([r["positions"] for r in rows]),
             "assertion_mask": assertion, "identity_links": links, "field_mask": fields,
             "comparison_table": torch.stack([r["table"] for r in rows])})
