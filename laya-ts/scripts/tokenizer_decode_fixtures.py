"""Record/check checkpoint decode -> re-encode vectors with Hugging Face tokenizers.

    python laya-ts/scripts/tokenizer_decode_fixtures.py --checkpoint-root SNAPSHOT
    python laya-ts/scripts/tokenizer_decode_fixtures.py --check

The checked-in tokenizer subsets retain the original token IDs and merge order.
Recording checks each subset against the complete checkpoint tokenizer; --check
executes the same HF decoder/encoder without downloading model assets. Supply
--checkpoint-root with --check to additionally verify the original asset hashes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/tokenizer-decode.json"
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
FILES = {"english": "tokenizer/tokenizer.json", "multilingual": "multilingual/tokenizer/tokenizer.json"}
SOURCE_HASHES = {
    "tokenizer/tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "multilingual/tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
}


def vectors(tok, kind):
    texts = [
        ("empty", ""), ("whitespace", "  Hello\n\tworld  "),
        ("unicode", "客户退款 café e\u0301 😀🫠 𠀀𠀁"),
        ("bom-nul", "\ufeffa\0b\ufeff"),
        ("special", "[CLS] hello [MASK] [SEP]" if kind == "english" else "<bos> hello <mask><eos>"),
    ]
    cases = []

    def add(name, ids):
        decoded = tok.decode(ids, skip_special_tokens=False)
        cases.append({"name": name, "ids": ids, "decoded": decoded,
                      "reencoded": tok.encode(decoded, add_special_tokens=False).ids})

    for name, text in texts:
        add(name, tok.encode(text, add_special_tokens=False).ids)
    boundary = tok.encode("𠀀𠀁𠀂 😀🫠 café 客户退款", add_special_tokens=False).ids
    for start in range(min(12, len(boundary))):
        add(f"boundary-{start}", boundary[start:start + 3])
    if kind == "multilingual":
        for name, values in [
            ("fallback-valid", [0xF0, 0x9F, 0x98, 0x80]),
            ("fallback-invalid-prefix", [0x61, 0xF0]),
            ("fallback-invalid-suffix", [0xF0, 0x61]),
            ("fallback-bom", [0xEF, 0xBB, 0xBF, 0]),
        ]:
            add(name, [tok.token_to_id(f"<0x{v:02X}>") for v in values])
        paragraph = ("客户表示，最近的账单中同一项订阅被收取了两次费用。"
                     "我们已经检查账户记录，确认用户只申请了一次续订。"
                     "请核实支付记录，说明重复扣费原因，并在必要时安排退款。"
                     "客户可以提供发票号码和付款凭证。\n\n")
        ids = tok.encode(paragraph * 4, add_special_tokens=False).ids
        add("reencoding-growth", ids[16:49])
    add("unknown-id", [2**31 - 1])
    return cases


def subset(raw, cases):
    """Keep every merge decomposition of the used tokens, in original rank order."""
    vocab = raw["model"]["vocab"]
    by_id = {v: k for k, v in vocab.items()}
    used_ids = {i for case in cases for key in ("ids", "reencoded") for i in case[key]}
    added = [t for t in raw["added_tokens"] if t["id"] in used_ids
             or any(t["content"] in case["decoded"] for case in cases)]
    wanted = {by_id[i] for case in cases for key in ("ids", "reencoded")
              for i in case[key] if i in by_id}
    wanted.update(t["content"] for t in added if t["content"] in vocab)
    if raw["model"].get("unk_token"):
        wanted.add(raw["model"]["unk_token"])
    merges = [m.split(" ") if isinstance(m, str) else m for m in raw["model"]["merges"]]
    decompositions = {}
    for pair in merges:
        decompositions.setdefault("".join(pair), []).append(pair)
    pending = list(wanted)
    while pending:
        for pair in decompositions.get(pending.pop(), []):
            for token in pair:
                if token not in wanted:
                    wanted.add(token)
                    pending.append(token)
    result = deepcopy(raw)
    result["added_tokens"] = added
    result["model"]["vocab"] = {k: v for k, v in vocab.items() if k in wanted}
    # HF otherwise allocates newly added token IDs from the smaller vocabulary size.
    result["model"]["vocab"].update({t["content"]: t["id"] for t in added})
    result["model"]["merges"] = [pair for pair in merges if "".join(pair) in wanted]
    # Encoding requests no special tokens. Avoid unused processor references in the subset.
    result["post_processor"] = None
    return result


def verify(data):
    count = 0
    for kind, section in data["tokenizers"].items():
        tok = Tokenizer.from_str(json.dumps(section["tokenizer"]))
        for case in section["cases"]:
            decoded = tok.decode(case["ids"], skip_special_tokens=False)
            encoded = tok.encode(decoded, add_special_tokens=False).ids
            if decoded != case["decoded"] or encoded != case["reencoded"]:
                raise AssertionError(f"{kind}/{case['name']}: HF decode/re-encode changed")
            count += 1
    return count


def record(root):
    data = {"reference": {"repository": "convaiinnovations/laya", "revision": REVISION,
                           "tokenizers": tokenizers.__version__, "files": {}}, "tokenizers": {}}
    for kind, relative in FILES.items():
        contents = (root / relative).read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != SOURCE_HASHES[relative]:
            raise ValueError(f"{relative}: tokenizer does not match the pinned snapshot {REVISION}")
        data["reference"]["files"][relative] = digest
        raw = json.loads(contents)
        cases = vectors(Tokenizer.from_str(contents.decode("utf-8")), kind)
        data["tokenizers"][kind] = {"tokenizer": subset(raw, cases), "cases": cases}
    verify(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
        count = verify(stored)
        if args.checkpoint_root and record(args.checkpoint_root) != stored:
            raise AssertionError("checkpoint tokenizer fixture differs; review before recording")
        print(f"PASS: {count} HF decode/re-encode vectors (tokenizers {tokenizers.__version__})")
    else:
        if not args.checkpoint_root:
            parser.error("recording requires --checkpoint-root at the pinned snapshot")
        data = record(args.checkpoint_root)
        FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {FIXTURE}")


if __name__ == "__main__":
    main()
