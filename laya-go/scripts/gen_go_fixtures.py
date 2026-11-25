#!/usr/bin/env python3
"""Record the goldens laya-go needs that laya-java's fixtures do not hold: go_extra.json.

laya-java/scripts/gen_fixtures.py records what Python answers for the tokenizer corpus, the
decode corpus, language detection and python_json. That is enough to prove the Java port on the
inputs somebody thought of. This port also re-implements what the JVM supplied for free --
Unicode normalization, lowercasing, the pre-tokenizer's scanner -- so it is held to the
reference over far more input than a hand-written corpus, from a fixed seed:

  * normalize       the crate's NFC/NFD/NFKC/NFKD over every code point (a digest), and over
                    random strings of marks, jamo and compatibility characters
  * lowercase       the crate's Lowercase over every code point (a digest)
  * python_lower    CPython's str.lower() on strings built around a capital sigma, the one
                    context-dependent rule
  * single_word     which code points the crate counts as word characters beside a
                    single_word added token (two digests, one per side)
  * pretokenize     ByteLevel (three configurations) and Metaspace (six) on random strings
  * normalizers     Strip, Prepend, Replace (string and regex), Lowercase, Sequence
  * encode_fuzz     ids for random strings on both shipped checkpoints, reaching the heap merge
                    path, added tokens, byte fallback and the unknown token
  * decode_fuzz     decode of random id lists, which produce the invalid UTF-8 a real decode of
                    a truncated slice does
  * round4          Python's round(x, 4) over exact binary ties, random doubles and the edges
  * states          json.loads then serialize_state over 2,000 random JSON texts (int and float
                    spellings, NaN, dups, escapes), the texts Python rejects, and the lone
                    surrogates it accepts and laya-go refuses
  * label_answers   Agent._decode_answers with bool, int, float and str labels, as the JSON
                    bytes Python writes
  * answers_fuzz    5,000 Agent._decode_answers calls on random, tied and near-tied logit rows
  * marker_refusals Agent._encode_state's "only N of its M option markers fit" refusal
  * questions       Agent._check_question over question definitions, every refusal with its text,
                    and for the accepted ones render_options, the labels, the legend and the
                    normalized instructions and option order
  * calibration     apply_calibration_payload on an ONNXAgent-shaped object (every accepted and
                    rejected shape: version, temperature, temperature_by_options, binning_map,
                    identity), apply_binning_map, and calibration_payload
  * predict_plan    ONNXAgent._infer_batch on a stub session: which rows each run holds (shape and
                    per-row checksums), the sort order, the result order, the answers and the
                    usage, over batch sizes, sorting and question sets on both tokenizers

    python laya-go/scripts/gen_go_fixtures.py --out DIR      # needs LAYA_FIXTURE_CHECKPOINTS
    python laya-go/scripts/gen_go_fixtures.py --check        # fail if laya-go/fixtures differs
                                                             # (answers that depend on numpy's
                                                             # float32 exp are not compared)

A checkpoint that cannot be found is an error, never a skip marker.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
GO = os.path.normpath(os.path.join(HERE, ".."))
DEFAULT_OUT = os.path.join(GO, "fixtures")
NAME = "go_extra.json"
MAX = 0x110000
SURROGATES = range(0xD800, 0xE000)
SEED = 20261011


def digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(row.encode("utf-8"))
    return h.hexdigest()


def mapped(cp, text):
    return "%d:%s;" % (cp, ",".join(str(ord(c)) for c in text))


# --- normalization, lowercase, sigma, single_word -------------------------------------------
def normalize_family(rng):
    from tokenizers import normalizers

    forms = {"NFC": normalizers.NFC(), "NFD": normalizers.NFD(),
             "NFKC": normalizers.NFKC(), "NFKD": normalizers.NFKD()}
    sweep = {}
    for name, norm in forms.items():
        sweep[name] = digest(mapped(cp, norm.normalize_str(chr(cp)))
                             for cp in range(MAX) if cp not in SURROGATES)
    marks = [cp for cp in range(0x300, 0x2000) if unicodedata.combining(chr(cp))][:400]
    marks += [0x0345, 0x1DFA, 0x0C3C, 0x07FD, 0x302A, 0x3099, 0x309A, 0x1E4EE, 0x10D69]
    compat = [cp for cp in range(0xA0, 0x1F000)
              if cp not in SURROGATES and unicodedata.normalize("NFKD", chr(cp)) != chr(cp)]
    pools = [
        [ord(c) for c in "aeiouAEIOUnNcCzZ oO"],
        [ord(c) for c in "éèêëÀÁÅçñüÿǆǅ"
         "ﬁﬂ①½ｱｶ㌀㈱℃ÅΩᾳǖṩ"],
        marks, marks,
        list(range(0x1100, 0x1113)), list(range(0x1161, 0x1176)), list(range(0x11A8, 0x11C3)),
        [rng.randrange(0xAC00, 0xD7A4) for _ in range(40)],
        compat,
        [0x200D, 0x200B, 0x20, 0x0A],
    ]
    cases = []
    for _ in range(1500):
        text = "".join(chr(rng.choice(rng.choice(pools))) for _ in range(rng.randrange(1, 9)))
        cases.append({"text": text, **{k.lower(): n.normalize_str(text) for k, n in forms.items()}})
    return {"sweep": sweep, "cases": cases}


def lowercase_family(rng):
    from tokenizers import normalizers

    low = normalizers.Lowercase()
    return {"sweep": digest(mapped(cp, low.normalize_str(chr(cp)))
                            for cp in range(MAX) if cp not in SURROGATES)}


def sigma_family(rng):
    sigma = "Σ"
    pool = ["A", "b", "α", "Α", "Β", sigma, sigma, sigma, " ", "'", ".", "́",
            "‍", "1", "-", "·", "ʰ", "’", "ǅ", "ẞ", "İ", "ͅ",
            "あ", "﻿", ":"]
    cases = []
    for _ in range(1000):
        text = "".join(rng.choice(pool) for _ in range(rng.randrange(1, 8)))
        cases.append({"text": text, "lower": text.lower()})
    cases += [{"text": t, "lower": t.lower()} for t in
              ("ΟΔΣ", "Σ", "AΣ", "ΣA", "AΣB", "ΑΣ Β",
               "A.Σ", "AΣ.", "A'Σ'B", "ΣΣ", "AΣΣ", "ΧΑΟΣ ΣΟΦΟΣ")]
    return {"cases": cases}


def single_word_family(rng):
    from tokenizers import AddedToken, Tokenizer, models

    tok = Tokenizer(models.BPE(vocab={"a": 0}, merges=[], unk_token=None))
    tok.add_tokens([AddedToken("TOK", single_word=True)])
    tid = tok.token_to_id("TOK")
    left, right = [], []
    for cp in range(MAX):
        ch = chr(cp)
        try:
            left.append("1" if tid in tok.encode(ch + "TOK").ids else "0")
            right.append("1" if tid in tok.encode("TOK" + ch).ids else "0")
        except Exception:
            left.append("0")
            right.append("0")
    return {"token": "TOK", "id": tid,
            "left_standalone": digest(left), "right_standalone": digest(right)}


# --- pre-tokenizers ---------------------------------------------------------------------------
TEXT_POOL = (
    ["a", "b", "Hello", "world", "x", "Z", "123", "4", "7.5", "'s", "'t", "'re", "'ve", "'m", "'ll",
     "'d", "'S", "don't", "I'm", "it's", "’s", ",", ".", "!", "?", "-", "_", "/", "(", ")", "$",
     "@", "#", "é", "é", "中", "文", "あ", "한", "م", "א",
     "न", "ก", "Ж", "Ω", "\U0001f600", "‍", "​", "﻿", "½",
     "²", "Ⅷ", "٣", "\U0001d7d8", "\U00010d4a", "\U0002ebf0", "▁", "▁▁"]
    + [" "] * 8 + ["  ", "   ", "\t", "\n", "\r\n", " ", "　", " ", " ", "\u0085"]
)


def fuzz_text(rng, low=1, high=12):
    return "".join(rng.choice(TEXT_POOL) for _ in range(rng.randrange(low, high)))


def pretokenize_family(rng):
    from tokenizers import pre_tokenizers

    texts = ["", " ", "a", "  ", "a  b\t\tc\n\nd\r\ne   f", "abc123", "123abc", "a 1 b", "'s", " 's",
             "x's value", "  x", "x  ", "▁", "▁a▁b", "a▁b", " ▁"]
    texts += [fuzz_text(rng) for _ in range(1200)]
    out = {"texts": texts}
    byte_level = {}
    for prefix in (False, True):
        for use_regex in (True, False):
            pre = pre_tokenizers.ByteLevel(add_prefix_space=prefix, use_regex=use_regex)
            key = "add_prefix_space=%s,use_regex=%s" % (prefix, use_regex)
            byte_level[key] = [[p for p, _ in pre.pre_tokenize_str(t)] for t in texts]
    out["byte_level"] = byte_level
    meta = {}
    for scheme in ("always", "first", "never"):
        for split in (True, False):
            pre = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=scheme, split=split)
            meta["%s,split=%s" % (scheme, split)] = [[p for p, _ in pre.pre_tokenize_str(t)]
                                                     for t in texts]
    out["metaspace"] = meta
    return out


def normalizers_family(rng):
    from tokenizers import Regex, normalizers

    def seq(*items):
        return normalizers.Sequence(list(items))

    specs = [
        ({"type": "Lowercase"}, normalizers.Lowercase()),
        ({"type": "Strip", "strip_left": True, "strip_right": True}, normalizers.Strip(True, True)),
        ({"type": "Strip", "strip_left": True, "strip_right": False}, normalizers.Strip(True, False)),
        ({"type": "Strip", "strip_left": False, "strip_right": True}, normalizers.Strip(False, True)),
        ({"type": "Prepend", "prepend": "▁"}, normalizers.Prepend("▁")),
        ({"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
         normalizers.Replace(" ", "▁")),
        ({"type": "Replace", "pattern": {"String": "ab"}, "content": ""}, normalizers.Replace("ab", "")),
        ({"type": "Replace", "pattern": {"Regex": "[ \\t\\n]+"}, "content": " "},
         normalizers.Replace(Regex("[ \\t\\n]+"), " ")),
        ({"type": "Replace", "pattern": {"Regex": "[0-9]+"}, "content": "#$0"},
         normalizers.Replace(Regex("[0-9]+"), "#$0")),
        ({"type": "Replace", "pattern": {"Regex": "(?i)hello"}, "content": "bye"},
         normalizers.Replace(Regex("(?i)hello"), "bye")),
        ({"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"},
                                              {"type": "Strip", "strip_left": True, "strip_right": True}]},
         seq(normalizers.NFKC(), normalizers.Lowercase(), normalizers.Strip(True, True))),
        ({"type": "Sequence", "normalizers": [{"type": "Replace", "pattern": {"Regex": " {2,}"},
                                               "content": " "}, {"type": "Prepend", "prepend": "_"}]},
         seq(normalizers.Replace(Regex(" {2,}"), " "), normalizers.Prepend("_"))),
    ]
    inputs = ["", " ", "Hello  World", "  padded  ", "ab ab", "ABC def", "ÉCOLE", "a\t\nb",
              "x1y22z333", "ﬁnd ①", "HELLO hello Hello", " nbsp ", "İstanbul",
              "ΣΣAΣ"]
    inputs += [fuzz_text(rng) for _ in range(150)]
    return {"inputs": inputs,
            "specs": [{"spec": spec, "outputs": [norm.normalize_str(t) for t in inputs]}
                      for spec, norm in specs]}


# --- the shipped checkpoints ------------------------------------------------------------------
ENCODE_POOL = (
    ["the", "refund", "payment", "account", "customer", "Please", "cancel", "my", "order", "I'm", "don't",
     "it's", "we've", "THEY'D", "12,345.67", "2026-10-11", "a@b.co", "https://example.com/x?y=1",
     "def f(x):", "{\"k\": [1, 2.5, null]}", "é", "café", "café", "naïve", "über",
     "中文测试", "日本語", "한국어", "مرحبا",
     "שלום", "नमस्ते", "สวัสดี",
     "Привет", "Γεια", "\U0001f600", "\U0001f468‍\U0001f469",
     "❤️", "\U0001f44d\U0001f3fd", "\U00010a00", "ೱ", "\U00016fe0", "\U0001e900",
     "▁", " ", "​", "‮", "﻿", "\u0000", "�", "½", "Ⅷ",
     "[MASK]", "[CLS]", "[SEP]", "[PAD]", "<mask>", "<bos>", "<eos>", "<pad>", "<unk>", "<s>", "</s>",
     "<0x41>", "<0xE2>", "ẞ", "İ", "ᾳ", "Å", "ﬁ"]
    + [" "] * 10 + ["  ", "\t", "\n", "\r\n", "   ", "　"]
)


def encode_text(rng):
    kind = rng.random()
    if kind < 0.08:
        return rng.choice("xqz中あ ") * rng.randrange(33, 220)
    if kind < 0.12:
        return "".join(rng.choice("abcdefghij") for _ in range(rng.randrange(40, 180)))
    return "".join(rng.choice(ENCODE_POOL) for _ in range(rng.randrange(0, 14)))


def checkpoint_tokenizer(name):
    from tokenizers import Tokenizer

    root = os.environ.get("LAYA_FIXTURE_CHECKPOINTS")
    if not root:
        raise SystemExit("gen_go_fixtures: set LAYA_FIXTURE_CHECKPOINTS to a directory holding the "
                         "english and multilingual checkpoints")
    path = os.path.join(root, name, "tokenizer", "tokenizer.json")
    if not os.path.isfile(path):
        raise SystemExit("gen_go_fixtures: no tokenizer at %s" % path)
    return Tokenizer.from_file(path)


def encode_family(rng):
    out = {}
    for name in ("english", "multilingual"):
        tok = checkpoint_tokenizer(name)
        texts = [encode_text(rng) for _ in range(700)]
        out[name] = {"texts": texts,
                     "ids": [tok.encode(t, add_special_tokens=False).ids for t in texts]}
    return out


def decode_family(rng):
    out = {}
    for name in ("english", "multilingual"):
        tok = checkpoint_tokenizer(name)
        size = tok.get_vocab_size(with_added_tokens=True)
        byte_ids = [tok.token_to_id("<0x%02X>" % b) for b in range(256)]
        byte_ids = [i for i in byte_ids if i is not None]
        low = list(range(0, 300))
        cases = []
        for _ in range(900):
            pool = rng.choice([byte_ids, low, None]) or list(range(size))
            ids = [rng.choice(pool) if pool is not None else rng.randrange(size)
                   for _ in range(rng.randrange(0, 10))]
            if rng.random() < 0.1:
                ids.append(size + rng.randrange(5))
            cases.append({"ids": ids,
                          "skipping": tok.decode(ids, skip_special_tokens=True),
                          "keeping": tok.decode(ids, skip_special_tokens=False)})
        out[name] = {"cases": cases}
    return out


GERMAN = "Ich kann mich nicht in mein Konto einloggen und das Passwort funktioniert nicht"


def state_family(rng):
    """How lang.py walks a state: Python's verdict on each container shape Go can spell."""
    from laya import lang

    self_list = [GERMAN]
    self_list.append(self_list)
    self_dict = {"a": GERMAN}
    self_dict["b"] = self_dict
    shapes = {
        "string": GERMAN,
        "map_of_slices": {"a": [GERMAN]},
        "slice_of_maps": [{"a": GERMAN}],
        "map_of_objects": {"a": {"b": GERMAN}},
        "slice_of_slices": [[GERMAN]],
        "tuple": (GERMAN,),
        "seven_deep": [[[[[[[GERMAN]]]]]]],
        "six_deep": [[[[[[GERMAN]]]]]],
        "numbers_and_none": [1, 2.5, None, True],
        "bytes": GERMAN.encode("utf-8"),
        "invalid_bytes": b"\xff\xfe",
        "mixed": {"k": [GERMAN, 3, {"x": None}]},
        "non_string_keys": {1: GERMAN},
        "self_ref_list": self_list,
        "self_ref_dict": self_dict,
    }
    out = {}
    for name, state in shapes.items():
        a = lang.analyse(state)
        out[name] = {"state_text": lang.state_text(state), "is_english": a["is_english"],
                     "language": a["language"], "script": a["script"]}
    return out


def invalid_utf8_family(rng):
    """CPython's bytes.decode('utf-8', 'replace') on random byte strings, and what each shipped
    tokenizer and lang.py make of the result."""
    from laya import lang

    pieces = [b"a", b"b", b" ", b"\xe2\x82\xac", b"\xe2\x82", b"\xe2", b"\xff", b"\xfe", b"\x80",
              b"\xc0\x80", b"\xed\xa0\x80", b"\xf0\x9f\x98\x80", b"\xf0\x9f\x98", b"\xf4\x90\x80\x80",
              b"\xc3\xa9", b"\xc3", b"\xe4\xb8\xad", b"Ich kann nicht"]
    cases = []
    for _ in range(3000):
        raw = b"".join(rng.choice(pieces) if rng.random() < 0.7 else bytes([rng.randrange(256)])
                       for _ in range(rng.randrange(0, 9)))
        cases.append(raw)
    decoded = [r.decode("utf-8", "replace") for r in cases]
    toks = {n: checkpoint_tokenizer(n) for n in ("english", "multilingual")}
    return {"hex": [r.hex() for r in cases], "decoded": decoded,
            "script": [lang.detect_script(d) for d in decoded],
            "state_text": [lang.state_text(d) for d in decoded],
            "ids": {n: [t.encode(d, add_special_tokens=False).ids for d in decoded[:600]]
                    for n, t in toks.items()}}


# --- questions ---------------------------------------------------------------------------------
# (case id, question id, definition). Chosen to reach every branch of Agent._check_question, in the
# order it checks them, and every rendering rule of render_options. A definition holds only what
# JSON can hold, because the Go side reads it from the golden as text.
CH2 = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "x", "b": "y"}}
SC3 = {"type": "score", "instructions": "Rate.", "criteria": ["bad", "ok", "good"]}
NL = {"type": "noul", "instructions": "Holds?"}


def _with(base, **changes):
    out = dict(base)
    for key, value in changes.items():
        if value is _DROP:
            out.pop(key, None)
        else:
            out[key] = value
    return out


_DROP = object()

QUESTION_CASES = [
    # the definition itself
    ("def-list", "q", [1, 2]), ("def-str", "q", "choice"), ("def-int", "q", 5),
    ("def-null", "q", None), ("def-bool", "q", True), ("def-float", "q", 2.5),
    # ids
    ("id-empty", "", CH2), ("id-blank", "  \t", CH2), ("id-nbsp", " ", CH2),
    ("id-apostrophe", "it's", _with(CH2, instructions="")),
    ("id-both-quotes", "a'b\"c", _with(CH2, instructions="")),
    ("id-unicode", "café", _with(CH2, instructions="")),
    ("id-control", "a\tb\x01", _with(CH2, instructions="")),
    ("id-astral", "\U0001f600", _with(CH2, instructions="")),
    ("id-ok", "team", CH2),
    # type
    ("type-missing", "q", _with(CH2, type=_DROP)), ("type-null", "q", _with(CH2, type=None)),
    ("type-number", "q", _with(CH2, type=5)), ("type-typo", "q", _with(CH2, type="chioce")),
    ("type-case", "q", _with(CH2, type="Choice")), ("type-list", "q", _with(CH2, type=["choice"])),
    ("type-dict", "q", _with(CH2, type={"a": [1, 2.5, None, True]})),
    ("type-float", "q", _with(CH2, type=1e16)),
    # instructions
    ("ins-missing", "q", _with(CH2, instructions=_DROP)),
    ("ins-null", "q", _with(CH2, instructions=None)),
    ("ins-empty", "q", _with(CH2, instructions="")),
    ("ins-blank", "q", _with(CH2, instructions=" \t\n")),
    ("ins-nbsp", "q", _with(CH2, instructions=" \u0085")),
    ("ins-ideographic-space", "q", _with(CH2, instructions="　")),
    ("ins-zero-width-is-text", "q", _with(CH2, instructions="​")),
    ("ins-empty-list", "q", _with(CH2, instructions=[])),
    ("ins-empty-dict", "q", _with(CH2, instructions={})),
    ("ins-zero", "q", _with(CH2, instructions=0)),
    ("ins-float", "q", _with(CH2, instructions=2.5)),
    ("ins-true", "q", _with(CH2, instructions=True)),
    ("ins-list", "q", _with(CH2, instructions=[1, "a", None, {"k": 1.0}])),
    ("ins-dict", "q", _with(CH2, instructions={"k": "v", "café": [1, 2], "n": 1e16})),
    ("ins-whitespace-kept", "q", _with(CH2, instructions="  padded  ")),
    # choice criteria
    ("choice-missing", "q", _with(CH2, criteria=_DROP)), ("choice-null", "q", _with(CH2, criteria=None)),
    ("choice-int", "q", _with(CH2, criteria=5)), ("choice-str", "q", _with(CH2, criteria="a")),
    ("choice-empty-dict", "q", _with(CH2, criteria={})), ("choice-empty-list", "q", _with(CH2, criteria=[])),
    ("choice-dict", "q", CH2),
    ("choice-list", "q", _with(CH2, criteria=["x", "y", "z"])),
    ("choice-list-null", "q", _with(CH2, criteria=["a", None])),
    ("choice-list-first-null", "q", _with(CH2, criteria=[None, ["x"]])),
    ("choice-list-list", "q", _with(CH2, criteria=["a", ["b"]])),
    ("choice-list-dict", "q", _with(CH2, criteria=["a", "b", {"k": 1}])),
    ("choice-list-repeat", "q", _with(CH2, criteria=["a", "b", "a"])),
    ("choice-list-int-float-repeat", "q", _with(CH2, criteria=[1, 1.0])),
    ("choice-list-bool-int-repeat", "q", _with(CH2, criteria=["x", True, 1])),
    ("choice-list-false-zero-repeat", "q", _with(CH2, criteria=[False, 0.0])),
    ("choice-list-big-repeat", "q", _with(CH2, criteria=[10 ** 30, float(10 ** 30)])),
    ("choice-list-near-big-ok", "q", _with(CH2, criteria=[2 ** 53 + 1, float(2 ** 53)])),
    ("choice-list-str-and-int-ok", "q", _with(CH2, criteria=[1, "1"])),
    ("choice-list-numbers", "q", _with(CH2, criteria=[2, 2.5, True, "x", "", 1e16, 1e-05, -0.0, 7])),
    ("choice-empty-label", "q", _with(CH2, criteria={"": "nothing", "b": None})),
    ("choice-falsy-criteria", "q", _with(CH2, criteria={"zero": 0, "no": False, "nul": None, "e": "", "l": [], "d": {}})),
    ("choice-structured", "q", _with(CH2, criteria={
        "a": {"desc": "café", "n": 1.0, "big": 123456789012345678901234567890, "neg": -0.0,
              "small": 1e-07, "huge": 1e16, "flag": True, "none": None},
        "b": [1, 2.5, "x", [], {}], "c": 2.5, "d": 7, "e": True})),
    ("choice-unicode", "q", _with(CH2, criteria={"退款": "こんにちは", "\U0001f600": "ل"})),
    ("choice-mask-text", "q", _with(CH2, criteria={"a": "x [MASK] y", "b": "<mask>"})),
    ("choice-control-text", "q", _with(CH2, criteria={"a": "line\nbreak\ttab \"q\" back\\slash", "b": "\x01"})),
    ("choice-dict-order", "q", _with(CH2, criteria={"z": "1", "a": "2", "m": "3"})),
    ("choice-many", "q", _with(CH2, criteria={"o%02d" % i: "c%d" % i for i in range(30)})),
    # score criteria
    ("score-missing", "q", _with(SC3, criteria=_DROP)), ("score-null", "q", _with(SC3, criteria=None)),
    ("score-dict", "q", _with(SC3, criteria={"0": "a"})), ("score-str", "q", _with(SC3, criteria="abc")),
    ("score-empty", "q", _with(SC3, criteria=[])), ("score-only-null", "q", _with(SC3, criteria=[None])),
    ("score-null-second", "q", _with(SC3, criteria=["a", None, "c", None])),
    ("score-ok", "q", SC3),
    ("score-mixed", "q", _with(SC3, criteria=["a", {"d": 1}, 3, 2.5, True, "", [1, [2]], 1e16])),
    ("score-one", "q", _with(SC3, criteria=["only"])),
    ("score-unicode", "q", _with(SC3, criteria=["悪い", "良い"])),
    # noul criteria
    ("noul-plain", "q", NL), ("noul-null-criteria", "q", _with(NL, criteria=None)),
    ("noul-int", "q", _with(NL, criteria=5)), ("noul-list", "q", _with(NL, criteria=[])),
    ("noul-str", "q", _with(NL, criteria="x")),
    ("noul-bad-key", "q", _with(NL, criteria={"maybe": "x"})),
    ("noul-bad-keys-sorted", "q", _with(NL, criteria={"true": "y", "maybe": 1, "Z": 2, "a": 3})),
    ("noul-empty-criteria", "q", _with(NL, criteria={})),
    ("noul-case-keys", "q", _with(NL, criteria={"True": "a", "FALSE": "b"})),
    ("noul-key-collision", "q", _with(NL, criteria={"true": "first", "TRUE": "second"})),
    ("noul-zero-is-description", "q", _with(NL, criteria={"true": 0, "false": False})),
    ("noul-empty-is-default", "q", _with(NL, criteria={"true": "", "false": None})),
    ("noul-structured", "q", _with(NL, criteria={"true": {"k": [1, 2.0]}, "false": "no"})),
    ("noul-only-true", "q", _with(NL, criteria={"true": "yes it does"})),
    ("noul-only-false", "q", _with(NL, criteria={"false": "no it does not"})),
    # option_order
    ("order-ok-choice", "q", _with(CH2, option_order=[1, 0])),
    ("order-ok-identity", "q", _with(CH2, option_order=[0, 1])),
    ("order-ok-score", "q", _with(SC3, option_order=[2, 0, 1])),
    ("order-ok-noul", "q", _with(NL, option_order=[1, 0])),
    ("order-short", "q", _with(CH2, option_order=[0])), ("order-long", "q", _with(CH2, option_order=[0, 1, 2])),
    ("order-repeat", "q", _with(CH2, option_order=[0, 0])), ("order-range", "q", _with(CH2, option_order=[0, 2])),
    ("order-negative", "q", _with(CH2, option_order=[-1, 0])),
    ("order-str", "q", _with(CH2, option_order="01")), ("order-null", "q", _with(CH2, option_order=None)),
    ("order-dict", "q", _with(CH2, option_order={})), ("order-int", "q", _with(CH2, option_order=1)),
    ("order-float", "q", _with(CH2, option_order=[0, 1.0])), ("order-float-whole", "q", _with(CH2, option_order=[1.0])),
    ("order-bool", "q", _with(CH2, option_order=[True, 0])),
    ("order-str-element", "q", _with(CH2, option_order=[0, "1"])),
    ("order-nested", "q", _with(CH2, option_order=[[0], [1]])),
    ("order-empty", "q", _with(CH2, option_order=[])),
    ("order-noul-wrong-size", "q", _with(NL, option_order=[0, 1, 2])),
    ("order-repr-mixed", "q", _with(CH2, option_order=[0, None, 2.5, "x", True, [1], {"a": 1.0}])),
    # labels
    ("labels-on-choice", "q", _with(CH2, labels={"false": "a", "true": "b"})),
    ("labels-on-score", "q", _with(SC3, labels=None)),
    ("labels-ok", "q", _with(NL, labels={"false": "nope", "true": "yep"})),
    ("labels-reversed-keys", "q", _with(NL, labels={"true": "yep", "false": "nope"})),
    ("labels-stripped", "q", _with(NL, labels={"false": " no\t", "true": " yes "})),
    ("labels-null", "q", _with(NL, labels=None)),
    ("labels-default-pair", "q", _with(NL, labels={"false": "false", "true": "true"})),
    ("labels-list", "q", _with(NL, labels=[])), ("labels-empty", "q", _with(NL, labels={})),
    ("labels-str", "q", _with(NL, labels="x")),
    ("labels-missing-true", "q", _with(NL, labels={"false": "a"})),
    ("labels-same", "q", _with(NL, labels={"false": "a", "true": "a"})),
    ("labels-same-after-strip", "q", _with(NL, labels={"false": "a", "true": " a "})),
    ("labels-empty-value", "q", _with(NL, labels={"false": "a", "true": ""})),
    ("labels-blank-value", "q", _with(NL, labels={"false": " ", "true": "x"})),
    ("labels-int-value", "q", _with(NL, labels={"false": 1, "true": "x"})),
    ("labels-capital-keys", "q", _with(NL, labels={"False": "a", "True": "b"})),
    ("labels-extra-key", "q", _with(NL, labels={"false": "a", "true": "b", "x": "c"})),
    ("labels-with-criteria", "q", _with(NL, labels={"false": "nope", "true": "yep"},
                                       criteria={"true": "it holds", "false": "it does not"})),
    ("labels-unicode", "q", _with(NL, labels={"false": "いいえ", "true": "はい"})),
    # precedence: the first fault Python finds is the one it reports
    ("order-blank-instructions-first", "q", _with(CH2, instructions="", criteria=[])),
    ("order-type-before-instructions", "q", _with(CH2, type="x", instructions="")),
    ("order-criteria-before-order", "q", _with(CH2, criteria=[], option_order=[5])),
    ("order-order-before-labels", "q", _with(CH2, option_order=[5], labels={})),
    ("order-label-type-before-repeat", "q", _with(CH2, criteria=["a", "a", None])),
    ("order-first-bad-label-wins", "q", _with(CH2, criteria=[None, ["x"]])),
    ("order-labels-before-noul-labels", "q", _with(SC3, labels={"x": 1})),
]


def questions_family(rng):
    """Python's verdict on each definition: the refusal text, or what the accepted one renders to."""
    import copy

    from laya.agent import Agent
    from laya.common import _resolve_noul_labels, render_criterion, render_options

    cases = []
    for cid, qid, definition in QUESTION_CASES:
        record = {"case": cid, "id": qid, "definition": definition}
        try:
            Agent._check_question(qid, copy.deepcopy(definition))
        except ValueError as refused:
            record["error"] = str(refused)
            cases.append(record)
            continue
        internal = Agent._to_internal(copy.deepcopy(definition))
        record["type"] = internal["t"]
        record["instructions"] = internal["ins"]
        record["options"] = render_options(internal)
        if internal["t"] == "choice":
            record["labels"] = [str(k) for k in internal["crit"]]
        elif internal["t"] == "score":
            record["labels"] = [str(i) for i in range(len(internal["crit"]))]
            record["legend"] = [render_criterion(c) for c in internal["crit"]]
        else:
            record["labels"] = list(_resolve_noul_labels(internal.get("labels")))
        record["option_order"] = internal.get("option_order")
        cases.append(record)
    refused = sum(1 for c in cases if "error" in c)
    if refused < 80 or len(cases) - refused < 30:
        raise AssertionError("the question cases stopped covering both refusals (%d) and "
                             "acceptances (%d)" % (refused, len(cases) - refused))
    return {"cases": cases}


def round4_family(rng):
    """round(x, 4): half to even on the exact binary value, which no scaled-integer trick matches."""
    ties = [k / 2 ** n for n in range(5, 14) for k in range(1, 2 ** n, max(1, 2 ** n // 40))]
    steps = [(2 * j + 1) / 20000.0 for j in range(0, 400)]          # decimal halves, binary inexact
    values = ties + steps + [-v for v in ties[:200]]
    values += [rng.random() for _ in range(2500)]
    values += [rng.uniform(-3, 3) for _ in range(1500)]
    values += [round(rng.random(), 5) + 0.00005 for _ in range(600)]
    values += [0.0, -0.0, 1e-5, -1e-5, 4.9999e-5, 5e-5, 5.0000001e-5, 1.0, 0.99995, 0.99994999,
               1e16, 123456.78905, 1e-300, 5e-324, 1.7976931348623157e308, 0.5, 2.5, 1e22]
    return {"values": values, "rounded": [round(v, 4) for v in values]}


# --- states: json.loads + serialize_state -----------------------------------------------------
_STR_POOL = ["a", "b", "z", " ", "\"", "\\", "/", "\n", "\t", "\x01", "\x7f", "é", "ü", "中", "あ",
             "\u2028", "\u00a0", "😀", "\U0010ffff", "\ufeff", "<", ">", "&", "0", "1e5"]
_NUM_INTS = ["0", "-0", "7", "-7", "42", "123456789012345678901234567890", "-98765432109876543210",
             "9007199254740993", "10000000000000000"]
_NUM_FLOATS = ["0.0", "-0.0", "1.0", "2.5", "-2.5", "0.1", "1e16", "1E16", "1e+16", "1e22", "1e23",
               "1e-5", "1E-05", "1e-4", "0.0001", "0.00001", "123456789.123456789", "5e-324",
               "1.7976931348623157e308", "1e308", "1e309", "-1e400", "1e400", "100E-2", "2.5E+3",
               "0e0", "0.30000000000000004", "1e15", "1e17", "12345678901234567.0"]


def _json_string(rng):
    text = "".join(rng.choice(_STR_POOL) for _ in range(rng.randrange(0, 6)))
    return json.dumps(text, ensure_ascii=rng.random() < 0.5)


def _json_number(rng):
    kind = rng.random()
    if kind < 0.3:
        return rng.choice(_NUM_INTS)
    if kind < 0.6:
        return rng.choice(_NUM_FLOATS)
    if kind < 0.8:
        return str(rng.randint(-10 ** rng.randrange(1, 25), 10 ** rng.randrange(1, 25)))
    if kind < 0.9:
        return repr(rng.uniform(-1, 1) * 10 ** rng.randrange(-20, 25))
    return "%d%s%d" % (rng.randrange(1, 10), rng.choice(["e", "E", "e+", "e-"]), rng.randrange(0, 30))


def _json_text(rng, depth=0):
    ws = lambda: rng.choice(["", "", " ", "\n", "\t ", "\r\n"])
    kind = rng.random()
    if depth >= 4 or kind < 0.35:
        k = rng.random()
        if k < 0.08:
            return rng.choice(["null", "true", "false", "NaN", "Infinity", "-Infinity"])
        if k < 0.4:
            return _json_string(rng)
        return _json_number(rng)
    if kind < 0.65:
        items = [_json_text(rng, depth + 1) for _ in range(rng.randrange(0, 5))]
        return "[" + ws() + ("," + ws()).join(i + ws() for i in items) + "]"
    keys = ["a", "b", "é", "k1", "k2", "中", "a"]       # a repeats, so duplicates occur
    pairs = []
    for _ in range(rng.randrange(0, 5)):
        pairs.append(json.dumps(rng.choice(keys), ensure_ascii=rng.random() < 0.5) + ws() + ":" + ws() + _json_text(rng, depth + 1))
    return "{" + ws() + ("," + ws()).join(p + ws() for p in pairs) + "}"


_BAD_JSON = ["", " ", "01", "-", "+1", ".5", "1.", "1e", "[1,]", "[,1]", "{\"a\":1,}", "{'a':1}", "[", "]", "{",
             "{\"a\"}", "{\"a\":}", "[1 2]", "1 2", "nul", "True", "NaN1", "Infinit", "- Infinity", "\"\t\"", "\"\\q\"",
             "\"\\u12\"", "\"\\u12G4\"", "\"abc", "\"\x00\"", "[1]]", "{}{}", "9" * 4301, "-" + "9" * 4301]
# No deep-nesting case: how deep json.loads recurses before RecursionError depends on the CPython
# build (5,000 levels fail on macOS and parse on the Linux runner). laya-go's own bound,
# pyvalue.MaxDepth, is held by its unit tests.
_LONE_SURROGATES = ["\"\\ud800\"", "\"\\udc00\"", "[\"\\ud800x\"]", "{\"\\udfff\":1}", "\"\\ud800\\u0041\"", "\"\\ud83d\"",
                    "\"a\\ud800\\ud800\""]


def states_family(rng):
    from laya.common import serialize_state

    cases = []
    for _ in range(2000):
        text = _json_text(rng)
        value = json.loads(text)
        cases.append({"text": text, "serialized": serialize_state(value), "list": isinstance(value, list)})
    bad = []
    for text in _BAD_JSON:
        try:
            json.loads(text)
        except (ValueError, RecursionError):
            bad.append(text)
        else:
            raise AssertionError("Python parses %r, so it is not a refusal case" % text[:40])
    for text in _LONE_SURROGATES:
        json.loads(text)              # Python accepts these; laya-go refuses
    return {"cases": cases, "python_rejects": bad, "go_refuses": _LONE_SURROGATES}


# --- label_answers and decode_fuzz: the real _decode_answers ----------------------------------
def _shim(temperature, by_options=None):
    from laya.agent import Agent

    shim = Agent.__new__(Agent)
    shim.temperature = temperature
    shim.temperature_by_options = by_options or {}
    shim.lang_temperatures = {}
    shim.binning_map = None
    return shim


def _answer_text(shim, internal, logits, order, lang=None):
    import numpy as np

    from laya.agent import Agent

    k = len(logits)
    q = dict(internal)
    if order is not None:
        q["option_order"] = order
    row = np.asarray(list(logits) + [99.0] * 3, dtype=np.float32).reshape(1, k + 3)
    act = np.asarray([0.25, -0.5], dtype=np.float32).reshape(1, 2)
    out = Agent._decode_answers(shim, row, act, [{"markers": list(range(k))}], ["x"], {"x": q}, 0, lang)
    return json.dumps(out["x"], ensure_ascii=False, separators=(",", ":"))


def _random_labels(rng, n):
    seen, labels = set(), []
    while len(labels) < n:
        kind = rng.random()
        if kind < 0.3:
            label = rng.choice(["a", "b", "billing", "1", "True", "1.0", "é", "", "x y", "it's", "<b>"]) + str(rng.randrange(0, 3) if rng.random() < 0.3 else "")
        elif kind < 0.45:
            label = rng.choice([True, False])
        elif kind < 0.7:
            label = rng.choice([0, 1, 2, -3, 7, 10 ** 20, 2 ** 53 + 1, 12])
        elif kind < 0.9:
            label = rng.choice([0.5, 1.5, -2.25, 1e16, 1e-05, 1e22, 0.1, 100.0, 2.0, 1e308])
        else:
            label = rng.choice([float("inf"), float("-inf"), 1e400])
        key = ("n", label) if not isinstance(label, str) else ("s", label)
        if key in seen or (not isinstance(label, str) and any(not isinstance(x, str) and x == label for x in labels)):
            continue
        seen.add(key)
        labels.append(label)
    return labels


def label_answers_family(rng):
    from laya.common import render_options

    shim = _shim([1.0, 1.0, 1.0])
    cases = []
    for _ in range(500):
        k = rng.randrange(1, 9)
        labels = _random_labels(rng, k)
        internal = {"t": "choice", "ins": "i", "crit": {c: None for c in labels}}
        if len(internal["crit"]) != k:          # 1 and True and 1.0 are one dict key
            continue
        order = rng.sample(range(k), k) if rng.random() < 0.3 else None
        logits = [round(rng.uniform(-4, 4), rng.choice([1, 3, 6])) for _ in range(k)]
        cases.append({"criteria_text": json.dumps(labels), "logits": logits, "option_order": order,
                      "options": render_options(internal),
                      "answer_text": _answer_text(shim, internal, logits, order)})
    return {"cases": cases}


def _fuzz_logits(rng, k):
    import numpy as np

    kind = rng.random()
    if kind < 0.25:
        return [round(rng.gauss(0, 1), rng.choice([0, 1, 2])) for _ in range(k)]          # many exact ties
    if kind < 0.5:                                                                          # near ties, a few ulps apart
        base = np.float32(rng.uniform(-3, 3))
        out = []
        for _ in range(k):
            v = base
            for _ in range(rng.randrange(0, 4)):
                v = np.nextafter(v, np.float32(np.inf))
            out.append(float(v))
        return out
    if kind < 0.6:
        return [float(np.float32(1.168331) + np.float32(rng.randrange(0, 3)) * np.float32(1.2e-7)) for _ in range(k)]
    scale = rng.choice([0.01, 0.5, 3, 12, 60])
    return [rng.gauss(0, 1) * scale for _ in range(k)]


def decode_fuzz_family(rng):
    from laya.common import clamp_temperature

    cases = []
    for _ in range(5000):
        kind = rng.choice(["choice", "choice", "score", "noul"])
        k = 2 if kind == "noul" else rng.randrange(1, 31) if kind == "choice" else rng.randrange(1, 9)
        if kind == "choice":
            internal = {"t": "choice", "ins": "i", "crit": {"o%d" % i: None for i in range(k)}}
            crit_json = ["o%d" % i for i in range(k)]
        elif kind == "score":
            internal = {"t": "score", "ins": "i", "crit": ["l%d" % i for i in range(k)]}
            crit_json = internal["crit"]
        else:
            internal, crit_json = {"t": "noul", "ins": "i"}, None
        temperature = rng.choice([5.0, 1.0, 0.5, 2.0, rng.uniform(0.3, 6), rng.uniform(0.9, 1.1)])
        order = rng.sample(range(k), k) if rng.random() < 0.3 else None
        logits = _fuzz_logits(rng, k)
        shim = _shim([clamp_temperature(temperature)] * 3)
        import numpy as np

        text = _answer_text(shim, internal, logits, order)
        cases.append({"type": kind, "criteria": crit_json, "temperature": temperature,
                      "logits": [str(np.float32(v)) for v in logits],   # shortest text of the float32 itself
                      "option_order": order, "answer_text": text})
    return {"cases": cases}


def marker_refusals_family(rng):
    """Agent._encode_state's refusal, on real tokenizers: the text, or None when every marker fits."""
    from transformers import AutoTokenizer

    from laya.agent import Agent

    root = os.environ.get("LAYA_FIXTURE_CHECKPOINTS")
    out = {}
    for name in ("english", "multilingual"):
        tok = AutoTokenizer.from_pretrained(os.path.join(root, name, "tokenizer"))
        shim = Agent.__new__(Agent)
        shim.tok, shim.parallel_options = tok, False
        shim.cfg = {"max_len": 512, "head_max_len": 192}
        cases = []
        for options, words in ((2, 1), (10, 6), (10, 30), (40, 6), (120, 4), (3, 40)):
            for max_len, head in ((24, 16), (64, 32), (100, 192), (160, 192), (512, 192), (1024, 256), (12, 192)):
                crit = {"opt%03d" % i: " ".join("w%d_%d" % (i, j) for j in range(words)) for i in range(options)}
                internal = {"t": "choice", "ins": "Pick one.", "crit": crit}
                record = {"options": options, "words": words, "max_len": max_len, "head_max_len": head}
                try:
                    shim._encode_state("state text", ["it's"], {"it's": internal}, max_len, head)
                    record["error"] = None
                except ValueError as refused:
                    record["error"] = str(refused)
                cases.append(record)
        refused = sum(1 for c in cases if c["error"])
        if not 5 <= refused < len(cases) - 5:
            raise AssertionError("the %s marker cases stopped covering both outcomes (%d refused)" % (name, refused))
        out[name] = {"cases": cases}
    return out


# --- predict_plan: ONNXAgent._infer_batch with the session stubbed -------------------------------
PLAN_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
              "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo")

PLAN_QUESTIONS = {
    # three types, three head sizes
    "mix": {
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "asks for money back", "status": "asks where the order is",
                                "other": "anything else"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["not at all", "slightly", "moderately", "very", "critical"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    },
    # a tiny head beside a head that collapses its options (30 options share their first tokens)
    "heads": {
        "pair": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "x", "b": "y"}},
        "wide": {"type": "choice", "instructions": "Which of the many?",
                 "criteria": {"the same long label prefix %02d" % i: "alpha beta gamma delta epsilon %d" % i
                              for i in range(30)}},
        "long": {"type": "noul", "instructions": " ".join(["Does the text mention the thing"] * 12)},
    },
    "one": {"only": {"type": "noul", "instructions": "Holds?"}},
}


def _plan_states(rng, count, low=1, high=40):
    """Strings of random length, then a list state (end kept) and a dict state (order kept)."""
    out = ["%s %s" % (rng.choice(PLAN_WORDS), " ".join(rng.choice(PLAN_WORDS)
                                                       for _ in range(rng.randrange(low, high))))
           for _ in range(count - 2)]
    out.append(["hello there", "I was charged twice", "please refund me"])
    out.append({"subject": "late order", "body": "it has not arrived", "n": 3})
    return out


def _plan_cases(rng):
    mixed = _plan_states(rng, 7)
    # equal lengths, 36 of them, so that an unstable sort has room to reorder them
    ties = ["%s %s" % (w, n) for w in PLAN_WORDS[:12] for n in ("one", "two", "six")]
    long_state = " ".join("word%d" % i for i in range(1500))     # past max_len on both checkpoints
    # partial: enough to cut the wide question's room but not the tiny question's, per checkpoint;
    # longest first, so a sort keyed on the first question's row reorders what the longest-row key ties
    partial = [" ".join("word%d" % i for i in range(n)) for n in (330, 250, 200, 130)]
    odd = ["", "x", long_state, "short one", ["a", "b"], "medium length state of text", ""] + partial
    many = _plan_states(rng, 30, 1, 60)
    cases = []

    def add(name, states, questions, batch_size=None, sort=False, lang=None):
        cases.append({"name": name, "states": states, "questions": questions,
                      "batch_size": batch_size, "sort": sort, "lang": lang})

    add("all in one", mixed, "mix")
    add("all in one, sorted flag ignored", mixed, "mix", sort=True)
    add("chunk 1", mixed, "mix", batch_size=1, sort=True)
    add("chunk equals states", mixed, "mix", batch_size=7, sort=True)
    add("chunk beyond states", mixed, "mix", batch_size=50, sort=True)
    add("chunk between, unsorted", mixed, "mix", batch_size=3)
    add("chunk between, sorted", mixed, "mix", batch_size=3, sort=True)
    add("chunk 2, sorted", mixed, "mix", batch_size=2, sort=True)
    add("negative chunk means all", mixed, "mix", batch_size=-1)
    add("zero chunk means all", mixed, "mix", batch_size=0)
    add("equal lengths, stable", ties, "mix", batch_size=4, sort=True)
    add("equal lengths, unsorted", ties, "mix", batch_size=4)
    add("more than 8 chunks, sorted", many, "mix", batch_size=2, sort=True)
    add("more than 8 chunks, unsorted", many, "mix", batch_size=2)
    add("lookahead boundary", many, "mix", batch_size=4, sort=True)
    add("different head sizes", mixed, "heads", batch_size=3, sort=True)
    add("different head sizes, one run", mixed, "heads")
    add("truncated and empty states", odd, "mix", batch_size=3, sort=True)
    add("truncated and empty, one run", odd, "heads")
    add("truncated, different heads, sorted", odd, "heads", batch_size=3, sort=True)
    add("single state", mixed[:1], "mix")
    add("single state, chunk 5", mixed[:1], "mix", batch_size=5, sort=True)
    add("one one-option question", mixed, "one", batch_size=3, sort=True)
    add("language tag", mixed, "mix", batch_size=3, lang="yo-NG")
    return cases


def _plan_session(log):
    """Stands in for onnxruntime. Its outputs are a function of the integers it is fed, exact in
    float32, so what the Go port feeds it is what decides the answers."""
    import numpy as np

    class Session(object):
        def run(self, names, feed):
            ids, mask = feed["input_ids"], feed["attention_mask"]
            pos, pmask = feed["marker_pos"], feed["marker_mask"]
            rows, length = ids.shape
            sums = [int(ids[r][mask[r] == 1].sum()) for r in range(rows)]
            log.append({"rows": rows, "length": length, "markers": pos.shape[1],
                        "id_sums": sums,
                        "tokens": [int(mask[r].sum()) for r in range(rows)],
                        "marker_sums": [int(pos[r][pmask[r]].sum()) for r in range(rows)],
                        "qtypes": [int(q) for q in feed["qtype"]]})
            logits = np.array([[((s + 31 * j) % 97 - 48) / 8 for j in range(pos.shape[1])]
                               for s in sums], dtype=np.float32)
            acts = np.array([[((s + 17 * c) % 97 - 48) / 8 for c in range(2)] for s in sums],
                            dtype=np.float32)
            return [logits, acts]

    return Session()


def predict_plan_family(rng):
    """The real _infer_batch on a stub agent: chunking, sort order, row routing, answers, usage."""
    from transformers import AutoTokenizer

    from laya.onnx_agent import ONNXAgent

    root = os.environ.get("LAYA_FIXTURE_CHECKPOINTS")
    plan_rng = random.Random("predict_plan")
    cases = _plan_cases(plan_rng)
    out = {}
    for name in ("english", "multilingual"):
        with open(os.path.join(root, name, "rl_agent_config.json"), encoding="utf-8") as handle:
            cfg = json.load(handle)
        log = []
        agent = ONNXAgent.__new__(ONNXAgent)
        agent.tok = AutoTokenizer.from_pretrained(os.path.join(root, name, "tokenizer"))
        agent.cfg = cfg
        agent.session = _plan_session(log)
        # what __init__ derives from the config: the clamp, not a re-implementation of it
        from laya.common import clamp_temperature
        agent.temperature = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
        agent.temperature_by_options = {k: clamp_temperature(v)
                                        for k, v in cfg.get("temperature_by_options", {}).items()}
        agent.lang_temperatures = {}
        agent.binning_map = None
        recorded = []
        for case in cases:
            del log[:]
            results = agent._infer_batch(case["states"], PLAN_QUESTIONS[case["questions"]],
                                         lang=case["lang"], batch_size=case["batch_size"],
                                         sort_by_length=case["sort"])
            recorded.append({
                "name": case["name"], "states": case["states"], "questions": case["questions"],
                "batch_size": case["batch_size"], "sort": case["sort"], "lang": case["lang"],
                "runs": list(log),
                "results": [{"model": r["model"], "answer_ids": list(r["answers"]),
                             "answers": r["answers"], "usage": r["usage"]} for r in results]})
        out[name] = {"cases": recorded}
    out["questions"] = PLAN_QUESTIONS
    return out



# --- calibration payload (runtime half) -------------------------------------------------------
CALIB_CFG = {"max_len": 512, "head_max_len": 192, "option_layout": "sequential",
             "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"choice:2": 2.0},
             "extra": {"a": [1, 2.5, True], "b": None}}
CALIB_IDENTITY = {k: v for k, v in CALIB_CFG.items() if k not in ("temperature", "temperature_by_options")}
OK_T = "[1.5, 2, 0.75]"
OK_BIN = '{"choice:2": {"bins": 3, "values": [0.1, 0.5, 1]}}'


def _calib_cases():
    ident = json.dumps(CALIB_IDENTITY)
    cases = []

    def add(name, text, cfg=None):
        cases.append((name, text, CALIB_CFG if cfg is None else cfg))

    def body(**kw):
        parts = ['"%s": %s' % (k, v) for k, v in kw.items()]
        return "{" + ", ".join(parts) + "}"

    # payload shape
    for name, text in [("array", "[]"), ("number", "1"), ("string", '"x"'), ("null", "null"),
                       ("true", "true")]:
        add("payload_" + name, text)
    # version
    for name, v in [("absent", None), ("null", "null"), ("one", "1"), ("two", "2"), ("three", "3"),
                    ("str_two", '"2"'), ("str_padded", '" 2 "'), ("str_underscore", '"0_2"'),
                    ("str_double_underscore", '"2__0"'), ("str_trailing_underscore", '"2_"'),
                    ("str_plus", '"+2"'), ("str_word", '"abc"'), ("str_empty", '""'),
                    ("str_float", '"2.0"'), ("float_two", "2.0"), ("float_high", "2.9"),
                    ("float_low", "1.9"), ("float_half", "1.5"), ("float_above", "2.5"),
                    ("str_fullwidth", '"\uff12"'), ("str_arabic_indic", '"\u0662"'),
                    ("str_fullwidth_underscore", '"\uff10_\uff12"'), ("str_tab_newline", '"\\t2\\n"'),
                    ("str_nbsp", '"\u00a02\u00a0"'), ("str_superscript", '"\u00b2"'), ("exp", "2e0"), ("nan", "NaN"), ("inf", "Infinity"),
                    ("huge_float", "1e999"), ("bool", "true"), ("false", "false"), ("list", "[2]"),
                    ("object", "{}"), ("negative", "-1"), ("big", "1" + "0" * 30)]:
        kw = {"temperature": OK_T}
        if v is not None:
            kw = {"version": v, **kw}
        add("version_" + name, body(**kw))
    # identity
    v2 = lambda **kw: body(version=2, temperature=OK_T, **kw)
    add("identity_match", v2(model_id_or_path="null", subfolder="null", config=ident))
    add("identity_config_absent", v2())
    add("identity_config_null", v2(config="null"))
    add("identity_model", v2(model_id_or_path='"org/m"', subfolder="null", config=ident))
    add("identity_subfolder", v2(model_id_or_path="null", subfolder='"sub"', config=ident))
    add("identity_both", v2(model_id_or_path='"it\'s"', subfolder='"s"', config='{"max_len": 1}'))
    add("identity_config_differs", v2(config='{"max_len": 256}'))
    add("identity_equal_numbers", v2(config='{"extra": {"b": null, "a": [1.0, 2.5, 1]}, '
        '"option_layout": "sequential", "head_max_len": 192.0, "max_len": 512}'))
    add("identity_bool_as_int", v2(config='{"max_len": 512, "head_max_len": 192, '
        '"option_layout": "sequential", "extra": {"a": [true, 2.5, 1], "b": null}}'))
    add("identity_extra_key", v2(config=json.dumps({**CALIB_IDENTITY, "z": 1})))
    add("identity_with_temps", v2(model_id_or_path="null", subfolder="null", config=json.dumps(CALIB_CFG)))
    add("identity_list", v2(config="[]"))
    add("identity_not_checked_v1", body(version=1, temperature=OK_T, model_id_or_path='"x"'))
    add("identity_empty_cfg", v2(model_id_or_path="null", subfolder="null", config="{}"), {})
    # temperature
    for name, t in [("absent", None), ("null", "null"), ("two", "[1, 2]"), ("four", "[1, 2, 3, 4]"),
                    ("string", '"abc"'), ("object", '{"a": 1, "b": 2, "c": 3}'), ("number", "3"),
                    ("clamped", "[0.1, 9, 2.5]"), ("strings", '["2.5", "x", " 3 "]'),
                    ("string_specials", '["nan", "inf", "0x1p1"]'),
                    ("specials", "[null, NaN, Infinity]"), ("neg_inf", "[-Infinity, 1e999, -1e999]"),
                    ("bools", "[true, false, 1]"), ("zero", "[0, -0.0, 0.5]"),
                    ("nested", '[[1], {}, "1 0"]'), ("edges", "[0.5, 5, 5.0000001]"),
                    ("ints", "[1, 5, 100000000000000000000000]"),
                    ("plain_strings", '["+2", "1e1", "1.5"]'),
                    ("str_underscores", '["1_5", "1_0.5", "2e1_0"]'),
                    ("str_bad_underscores", '["1_.5", "1._5", "_1"]'),
                    ("str_unicode_digits", '["\uff11.\uff15", "\u0663", "\u00b2"]'),
                    ("str_signed_nan", '["+nan", "-NaN", "-Infinity"]'),
                    ("str_inf_words", '["inf", "INFINITY", "infinit"]'),
                    ("str_nbsp", '["\u00a02\u00a0", "\u20032", "2\\u200b"]'),
                    ("str_overflow", '["1e400", "-1e400", "1e-400"]'),
                    ("bool_true_only", "[true, 1.0, 1]")]:
        add("temperature_" + name, body(version=1, **({} if t is None else {"temperature": t})))
    # temperature_by_options
    for name, b in [("absent", None), ("null", "null"), ("empty", "{}"), ("list", "[]"),
                    ("string", '"x"'), ("number", "5"), ("false", "false"),
                    ("mixed", '{"choice:2": 0.01, "choice:3-5": "3", "score:2": true, "noul:2": null, '
                              '"choice:11+": 7, "choice:6-10": 2}'),
                    ("duplicate", '{"choice:2": 9, "choice:2": 0.1}')]:
        kw = {"version": 1, "temperature": OK_T}
        if b is not None:
            kw["temperature_by_options"] = b
        add("by_options_" + name, body(**kw))
    # binning
    def bm(v):
        return body(version=1, temperature=OK_T, binning_map=v)

    for name, v in [("null", "null"), ("empty", "{}"), ("valid", OK_BIN), ("one_bin", '{"c": {"bins": 1, "values": [0.25]}}'),
                    ("ints", '{"c": {"bins": 2, "values": [0, 1]}}'),
                    ("two_buckets", '{"choice:2": {"bins": 2, "values": [0.2, 0.9]}, '
                                    '"score:3-5": {"bins": 1, "values": [0.5]}}'),
                    ("list", "[]"), ("string", '"x"'), ("number", "1"), ("true", "true"),
                    ("entry_list", '{"c": []}'), ("entry_null", '{"c": null}'), ("entry_empty", '{"c": {}}'),
                    ("bins_bool", '{"c": {"bins": true, "values": [0.5]}}'),
                    ("bins_float", '{"c": {"bins": 2.0, "values": [0.5, 0.5]}}'),
                    ("bins_exp", '{"c": {"bins": 1e0, "values": [0.5]}}'),
                    ("bins_str", '{"c": {"bins": "2", "values": [0.5, 0.5]}}'),
                    ("bins_zero", '{"c": {"bins": 0, "values": []}}'),
                    ("bins_negative", '{"c": {"bins": -1, "values": []}}'),
                    ("bins_missing", '{"c": {"values": [0.5]}}'),
                    ("bins_huge", '{"c": {"bins": 100000000000000000000, "values": [0.5]}}'),
                    ("values_short", '{"c": {"bins": 3, "values": [0.5, 0.5]}}'),
                    ("values_long", '{"c": {"bins": 1, "values": [0.5, 0.5]}}'),
                    ("values_missing", '{"c": {"bins": 1}}'),
                    ("values_string", '{"c": {"bins": 1, "values": "a"}}'),
                    ("values_object", '{"c": {"bins": 1, "values": {"a": 1}}}'),
                    ("value_str", '{"c": {"bins": 1, "values": ["0.5"]}}'),
                    ("value_bool", '{"c": {"bins": 1, "values": [true]}}'),
                    ("value_null", '{"c": {"bins": 1, "values": [null]}}'),
                    ("value_list", '{"c": {"bins": 1, "values": [[0.5]]}}'),
                    ("value_nan", '{"c": {"bins": 1, "values": [NaN]}}'),
                    ("value_inf", '{"c": {"bins": 1, "values": [Infinity]}}'),
                    ("value_huge", '{"c": {"bins": 1, "values": [1e999]}}'),
                    ("value_negative", '{"c": {"bins": 1, "values": [-0.1]}}'),
                    ("value_high", '{"c": {"bins": 1, "values": [1.0000001]}}'),
                    ("value_edges", '{"c": {"bins": 2, "values": [0.0, 1.0]}}'),
                    ("second_entry_bad", '{"a": {"bins": 1, "values": [0.5]}, "b": {"bins": 1, "values": [2]}}')]:
        add("binning_" + name, bm(v))
    # the whole thing at once
    add("full", body(version=2, temperature="[0.2, 3, 9]", temperature_by_options='{"choice:2": 0.1, "score:2": 2}',
                     model_id_or_path="null", subfolder="null", config=ident, binning_map=OK_BIN))
    add("duplicate_keys", '{"version": 1, "temperature": [9, 9, 9], "temperature": [2, 2, 2]}')
    return cases


class _CalibObj:
    def __init__(self, cfg):
        self.cfg = cfg   # the parsed rl_agent_config.json; Go has no model id, so it compares against (None, None)


def calibration_family():
    import warnings

    from laya.calibrate import apply_binning_map, apply_calibration_payload, calibration_payload

    cases = []
    for name, text, cfg in _calib_cases():
        obj = _CalibObj(json.loads(json.dumps(cfg)))
        obj.binning_map = "unset"
        record = {"name": name, "payload": text, "config": json.dumps(cfg)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                apply_calibration_payload(obj, json.loads(text))
            except ValueError as exc:
                record["error"] = str(exc)
            else:
                record["temperature"] = obj.temperature
                record["by_options"] = obj.temperature_by_options
                record["binning"] = obj.binning_map
        record["warnings"] = [[w.category.__name__, str(w.message)] for w in caught]
        cases.append(record)

    rng = random.Random(SEED + 8)
    maps = [{"choice:2": {"bins": 3, "values": [0.1, 0.5, 1.0]}},
            {"choice:2": {"bins": 1, "values": [0.25]}, "score:3-5": {"bins": 15, "values": [i / 15 for i in range(15)]}},
            {}]
    buckets = ["choice:2", "choice:3-5", "score:3-5", "noul:2", "choice:11+"]
    confs = [0.0, 1.0, 0.5, 1 / 3, 2 / 3, 0.9999999, 1e-12, 0.3333333333333333, -0.5, 1.5]
    confs += [rng.random() for _ in range(40)]
    binning = [{"map": m, "bucket": b, "confidence": c, "result": apply_binning_map(c, b, m)}
               for m in maps for b in buckets for c in confs]

    writer = []
    for model, sub, cfg, bmap in [(None, None, CALIB_CFG, None), ("org/m", "sub", CALIB_CFG, OK_BIN_MAP),
                                  (None, None, None, {}), (None, None, {}, OK_BIN_MAP)]:
        body = calibration_payload([1.5, 2, 0.75], {"choice:2": 2, "score:3-5": 0.5}, model, sub, cfg, bmap)
        writer.append({"model": model, "subfolder": sub, "config": cfg, "binning_map": bmap,
                       "payload": json.dumps(body)})
    return {"cases": cases, "binning": binning, "writer": writer}


OK_BIN_MAP = json.loads(OK_BIN)

class _OrderAgent:
    """Records the questions predict_shortlist hands to predict, option_order included."""

    def __init__(self):
        self.seen = None

    def predict(self, state, questions, **kwargs):
        self.seen = {qid: dict(qdef) for qid, qdef in questions.items()}
        return {"answers": {}, "model": "stub"}


def _order_embed(texts):
    """Rank option i by a fixed score, so which labels survive the cut is known by reading it."""
    import numpy as np

    return np.asarray([[1.0, 0.0] if i == 0 else [1.0 / (i + 1), float(i % 3)]
                       for i in range(len(texts))], dtype=np.float64)


def shortlist_order_family(rng):
    """predict_shortlist's option_order rule: a narrowed choice loses it, with one warning each."""
    import copy
    import warnings

    from laya import shortlist as mod

    six = {name: "the %s option" % name for name in ("a", "b", "c", "d", "e", "f")}
    three = {"x": "first", "y": "second", "z": "third"}
    cases = []
    for name, questions, k in [
        ("narrowed-loses-order", {"q": {"type": "choice", "instructions": "Which?", "criteria": six,
                                        "option_order": [5, 4, 3, 2, 1, 0]}}, 3),
        ("passthrough-keeps-order", {"q": {"type": "choice", "instructions": "Which?", "criteria": three,
                                           "option_order": [2, 0, 1]}}, 3),
        ("non-choice-keeps-order", {
            "s": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"],
                  "option_order": [2, 1, 0]},
            "n": {"type": "noul", "instructions": "Is it?", "option_order": [1, 0]}}, 2),
        ("two-narrowed-two-warnings", {
            "p": {"type": "choice", "instructions": "P?", "criteria": six, "option_order": [1, 0, 2, 3, 4, 5]},
            "q": {"type": "choice", "instructions": "Q?", "criteria": six, "option_order": [5, 4, 3, 2, 1, 0]},
            "r": {"type": "choice", "instructions": "R?", "criteria": three, "option_order": [1, 2, 0]}}, 4),
        ("narrowed-without-order", {"q": {"type": "choice", "instructions": "Which?", "criteria": six}}, 2),
    ]:
        agent = _OrderAgent()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mod.predict_shortlist(agent, "state", copy.deepcopy(questions), _order_embed, k)
        cases.append({
            "name": name, "k": k, "questions": questions,
            "asked": agent.seen,
            "warnings": [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)],
            "labels": {qid: entry["labels"] for qid, entry in result["shortlist"].items()},
        })
    return {"cases": cases}

def email_fold_family(rng):
    """Which code points `re.I` treats as each literal letter laya/email.py's patterns spell.

    The Go port matches those patterns on text folded to one rune per rune (Go's `(?i)` is a
    different relation: it has no dotless i), so the fold is held to Python over all of Unicode
    for every letter the patterns use. `orbit[L]` lists every other code point that matches L.
    """
    import re
    from laya import email as email_mod

    sources = [p.pattern for p in email_mod._QUOTE_HEADERS]
    sources += [email_mod._ATTRIBUTION_TAIL.pattern, email_mod._ATTRIBUTION_HEAD.pattern,
                email_mod._HEADER_FROM_NAME.pattern, email_mod._HEADER_NEXT.pattern,
                email_mod._SIGNOFF_HEAD.pattern, email_mod._DEVICE_FOOTER.pattern,
                email_mod._DISCLAIMER.pattern, email_mod._DEVICE]
    sources += [m.pattern for m in email_mod._SIGNATURE_MARKERS if hasattr(m, "pattern")]
    accented = sorted({c for src in sources for c in src if ord(c) > 127})
    alphabet = [chr(c) for c in range(ord("a"), ord("z") + 1)] + accented
    pattern = re.compile("|".join("(%s)" % re.escape(c) for c in alphabet), re.I)
    orbit = {c: [] for c in alphabet}
    for cp in range(MAX):
        if cp in SURROGATES or not chr(cp).isalpha():
            continue
        m = pattern.fullmatch(chr(cp))
        if m is not None and chr(cp) != alphabet[m.lastindex - 1]:
            orbit[alphabet[m.lastindex - 1]].append(cp)
    return {"alphabet": "".join(alphabet), "orbit": {k: v for k, v in orbit.items() if v}}


EMAIL_FRAGMENTS = [
    "I cannot log in and need a password reset.", "Please refund the duplicate charge.",
    "On Mon, Jan 5, 2026 at 9:00 AM Ana <ana@x.com> wrote:", "On Monday I wrote:",
    "Em seg., 5 de jan. de 2026 \u00e0s 09:00, Ana <ana@x.com> escreveu:",
    "Em resposta ao que voc\u00ea escreveu:", "El lun, 5 ene 2026, Ana escribi\u00f3:",
    "Le lun. 5 janv. 2026 \u00e0 09:00, Ana a \u00e9crit :", "Le rapport que vous avez \u00e9crit :", "Le rapport que Ana a \u00e9crit :", "El informe que Ana escribi\u00f3:",
    "Le lun. 5 janv. 2026, Ana a \u00e9crit:", "ana@x.com> escreveu:", "support@x.com> a \u00e9crit :",
    "-----Original Message-----", "-----Mensagem original-----", "-----Message d'origine-----",
    "________________", "From: ana@x.com", "From: my side it works", "From: Ana Souza", "Sent: Monday 2026",
    "De: Maria Souza", "Enviado: segunda 2026", "De: 10/09 a 15/09", "De : Marie Dupont",
    "Envoy\u00e9 : lundi 5 janvier 2026", "Date: 5 Jan 2026", "Fecha: 5 enero 2026", "De: ana@x.com",
    "> quoted line", "  > indented quote", "--", "-- ", "Regards,", "Regards, Ana", "Thanks for the reply.",
    "Thanks, \u017canetta", "Thanks, \u0141ukasz", "Best regards", "Warm wishes, Jos\u00e9",
    "Thanks, Jose\u0301", "Thanks, \u0301x", "Kind regards and thanks a lot very much", "Cheers",
    "Atenciosamente,", "Obrigado pelo retorno, mas ainda n\u00e3o consigo entrar.", "Obrigada.",
    "Saludos cordiales", "Gracias de antemano", "Cordialement,", "Merci pour votre aide, mais non.",
    "Merci d'avance", "Bonne journ\u00e9e", "Sent from my iPhone", "Enviado do meu celular",
    "Enviado do meu celular o comprovante ontem.", "Envoy\u00e9 depuis mon iPhone",
    "Sent from my Samsung Galaxy smartphone using Outlook 365", "Get Outlook for iOS",
    "Obter o Outlook para Android", "This email and its contents are confidential and intended solely for you.",
    "This email and its contents are\nconfidential and intended solely for you.", "Is this confidential?",
    "Confidential: I need a refund.", "If you have received this email in error please delete it.",
    "Esta mensagem \u00e9 confidencial e de uso exclusivo do destinat\u00e1rio.",
    "preciso do contrato confidencial", "Antes de imprimir, pense no meio ambiente.",
    "antes de imprimir o boleto, confira o valor", "Ha recibido este mensaje por error.",
    "Ce message est confidentiel et destin\u00e9 \u00e0 son destinataire.", "le contrat confidentiel",
    "Vous avez re\u00e7u ce message par erreur.", "Usage exclusif du destinataire.",
    "The account is locked\nThis email is confidential and intended solely for the addressee.",
    "Please help. This message is confidential and privileged. Thank you.",
    "\u0130f you received this message in error", "TH\u0130S EMAIL IS CONF\u0130DENT\u0130AL AND INTENDED SOLELY",
    "this email is confidential and intended solely", "t\u0131his", "\u017f\u00e9nt from my iPhone",
    "Em\u00a0seg, 1 de jan de 2024 escreveu:", "On\u3000Monday 5 Jan 2026 Ana wrote:",
    "Le\u202flun. 5 janv. 2026, Ana a \u00e9crit :", "-----Original\u00a0Message-----",
    "Thank\u00a0you John", "Thank you\u3000John", "Sent\u00a0from my iPhone", "Sent from\u202fmy iPhone",
    "Ce\u00a0message est confidentiel", "This\u2009email is confidential and intended solely",
    "Em\x0eseg, 1 de jan de 2024 escreveu:", "Thank\x0eyou John", "Sent from my iPhone using x\x0f", "\x0e", "\x0f", "\x7f",
    "This email is confidential\u0663 and intended solely", "This email is confidential\u00a0and intended solely",
    "Sent from my iPhone using x\u0663", "Sent from my iPhone using x\uff11", "Sent from my iPhone \u0663",
    "\u212aind regards", "REGARDS, ANA", "\u00a0Regards,\u00a0", "\u2028", "\x85", "\x1c", "a\u2028b",
    "\u0661\u0662/\u0661\u0663 5 jan Ana escribi\u00f3:", "El \u0661\u0662 Ana escribi\u00f3:",
    "This email\u0301 is confidential and intended solely", "\U0001d400 confidential and intended solely",
    "\u4f60\u597d\uff0c\u8bf7\u9000\u6b3e\u3002", "x" * 70, "", " ", "\t", "a    b\t\tc",
]
EMAIL_JOINERS = ["\n", "\n", "\n", "\n\n", "\r\n", " ", "\n \n", "\\n", "\r", "\n\u00a0\n"]


def email_fuzz_family(rng):
    """clean_email_body and email_state over random runs of the corpus' own fragments.

    The shared corpus is hand-built from the module's reasoning; this joins its rules, its
    near-misses and the Unicode oddities (dotless i, long s, Kelvin, NEL, LS, FS, astral, marks,
    non-ASCII digits) in orders nobody wrote down, at three budgets.
    """
    from laya import email as email_mod

    cases = []
    for _ in range(1500):
        parts = [rng.choice(EMAIL_FRAGMENTS) for _ in range(rng.randrange(1, 16))]
        body = parts[0]
        for part in parts[1:]:
            body += rng.choice(EMAIL_JOINERS) + part
        budget = rng.choice([3000, 3000, 3000, 200, 40, 1, 0, -3])
        case = {"body": body, "max_chars": budget,
                "result": email_mod.clean_email_body(body, max_chars=budget)}
        cases.append(case)
    return {"cases": cases}

# ---------------------------------------------------------------- router
# laya/router.py's routing half, run through the real Router. The probe corpus is the one the .NET
# port records (laya-dotnet/tools/routing_cases.py); only its inputs are used, every answer here is
# recorded from Python now.
ROUTER_TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}

NAME_CASES = [
    "english", "multilingual", "typed-decisions", "en", "laya", "default", "multi", "ml", "laya-multilingual",
    "typed", "typed_decisions", "laya-typed-decisions", "decisions", "EN", "  English ", "\tML\n", "Typed_Decisions",
    "TYPED-DECISIONS", "typed decisions", "", " ", "auto", "papers", "english/", "../english", "x" * 300,
    "\u00a0en\u00a0", "\u2003en\u2003", "\u3000english\u3000", "\u200ben", "en\u200b", "\u212a", "K", "\u0130N",
    "ENGLİSH", "\uff25\uff2e", "\uff45\uff4e", "ΣΑΣ", "ΑΣ", "Σ", "english\u0307", "ﬁ", "ǅ", "ß", "ẞ",
    "it's", 'say "hi"', "it's \"x\"", "x\u200by", "\x07", "\U0001f600", "\ud7ff", "\ue000", "\n", "a\nb",
    "multilingual\x00", "ml ", " ml", "MULTI", "Laya", "LAYA-TYPED-DECISIONS", "typed_decisions ", "decision",
]

CODE_CASES = [
    "en", "EN", "En", "en-US", "en_US", "en_US.UTF-8", "en.UTF-8", "eng", "ENG", "english", "English", " EN ",
    "de", "de-DE", "de_DE.UTF-8", "fr", "zh_CN", "zh-Hans-CN", "pt-BR", "hi", "ar", "qq", "it's", "x\u200by",
    "", " ", "   ", "\t", ".", ".UTF-8", "-", "_", "-en", "_en", "C", "c", "C.UTF-8", "c.utf-8", "POSIX", "posix",
    "POSIX.UTF-8", "und", "UND", "und-Latn", "zxx", "mul", "mul-x", "en-", "en_", "en-.x", "e", "n", "english-us",
    "English_US", "eng-US", "ENGLISH.UTF-8", "en\u00a0", "\u00a0en", "\u200ben", "\uff45\uff4e", "\u212a",
    "\u0130", "i\u0307", "ΣΑΣ", "x" * 500, "en" + "-x" * 100, "e\u0301n", "\U0001f600", "C\u200b", "c-utf8",
    "c_utf8", "posix_x", "Mul", "none", "None", "null", "0", "1", "True",
]

ROUTER_FRAGMENTS = [
    "I was charged twice, please refund the duplicate payment on my invoice.",
    "Please review the error and send me the details of this and what you would like to do.",
    "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist",
    "Eu nao consigo entrar na minha conta e depois eu nao sei mais o que fazer com isso tudo.",
    "Le client a ete facture deux fois et demande un remboursement",
    "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
    "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
    "Gătește-mi o rețetă de sarmale de post pentru mâine.",
    "žluťoučký kůň úpěl ďábelské ódy", "hej min vän", "Hello", "refund me", "12345", "12345 6789", "", "   ", "\n",
    "मुझसे दो बार शुल्क लिया गया", "二重に請求されました", "客户被重复扣款要求退款", "두 번 청구되었습니다",
    "تم خصم المبلغ مرتين", "Հայերեն", "ՀԱՅԵՐԵՆ", "ลูกค้าถูกเรียกเก็บเงินสองครั้ง", "С клиента дважды сняли деньги",
    "Ο πελάτης χρεώθηκε δύο φορές", "MON DES EST LA", "os.path = nao consigo entrar na minha conta",
    "Please send the receipt to Владимир Иванов today", "ok ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎ", "Please " + "ありがとう" * 4,
    "\U0001f600 \U0001f600", "ＡＢＣＤ hello", "x\u200by z", "it's don't \"quoted\"", "a\u0007b nao consigo entrar",
    "Quiero cancelar mi pedido por favor ahora", "Ich kann mich nicht in mein Konto einloggen",
    "Care este ora in Tokyo?", "Cât e ora acum la Tokyo",
]


def _router_decision(decision):
    det = decision["detection"]
    return {
        "model": decision["model"], "repo": decision["repo"], "reason": decision["reason"],
        "workflow": decision["workflow"],
        "detection": None if det is None else {
            "script": det["script"], "language": det["language"], "is_english": det["is_english"],
            "language_undecided": det["language_undecided"], "non_latin_fraction": det["non_latin_fraction"],
            "diacritic_rate": det["diacritic_rate"], "mixed_segment": det["mixed_segment"]},
    }


def _router_build(kwargs):
    """The Router for these kwargs, or the text of what Router() refused."""
    from laya.router import Router
    try:
        return Router(**kwargs), None
    except ValueError as refused:
        return None, str(refused)


def _router_route(router, route_kwargs):
    try:
        return {"decision": _router_decision(router.route(**route_kwargs))}
    except ValueError as refused:
        return {"error": str(refused)}


def _router_questions(rng):
    ids = [i for ids in ROUTER_TD.values() for i in ids] + ["x", "y", "urgency", "Urgency", "", "id with space"]
    ids = sorted(set(ids))
    pools = [list(v) for v in ROUTER_TD.values()]
    for _ in range(40):
        pool = rng.choice(pools)
        pick = rng.sample(pool, rng.randint(max(0, len(pool) - 2), len(pool)))
        if rng.random() < 0.4:
            pick.append(rng.choice(ids))
        pick = list(dict.fromkeys(i for i in pick if i.strip()))
        pools.append(pick)
    return pools


def _noul_questions(ids):
    return {i: {"type": "noul", "instructions": "x"} for i in ids}


def _mutate_text(rng, text):
    ops = [lambda t: t.upper(), lambda t: t.lower(), lambda t: " " + t + "\t", lambda t: t[::-1],
           lambda t: t + "\u200b", lambda t: t + "\n" + t, lambda t: t * 3, lambda t: t.title(),
           lambda t: t.replace(" ", "_"), lambda t: t.replace("-", "_"), lambda t: "\u00a0" + t]
    return rng.choice(ops)(text)


def _router_state(rng):
    frag = lambda: rng.choice(ROUTER_FRAGMENTS)  # noqa: E731
    shape = rng.randrange(7)
    if shape == 0:
        return frag()
    if shape == 1:
        return {"body": frag()}
    if shape == 2:
        return {"subject": frag(), "body": frag()}
    if shape == 3:
        return [frag(), frag()]
    if shape == 4:
        return "\n".join([frag() for _ in range(rng.randint(2, 5))])
    if shape == 5:
        return {"note": frag() * rng.randint(1, 40), "body": frag()}
    return _mutate_text(rng, frag())


def router_family(rng):
    """Router naming, workflow match, question signature, route and route_batch, all from Python."""
    import copy

    sys.path.insert(0, os.path.join(HERE, "..", "..", "laya-dotnet", "tools"))
    try:
        import routing_cases
    finally:
        sys.path.pop(0)
    from laya import router as pyrouter
    from laya.agent import Agent
    from laya.common import _resolve_noul_labels, render_criterion, render_options
    from laya.router import Router

    names = []
    for name in NAME_CASES + [_mutate_text(rng, rng.choice(NAME_CASES[:16])) for _ in range(200)]:
        record = {"name": name, "canonical": pyrouter.canonical_name(name)}
        try:
            record["normalised"] = pyrouter.normalise_name(name)
        except ValueError as refused:
            record["error"] = str(refused)
        spec = pyrouter.resolve_model_spec(name)
        record["spec"] = None if spec is None else list(spec)
        names.append(record)

    codes = [{"code": c, "english": pyrouter._english_from_code(c)}
             for c in CODE_CASES + [_mutate_text(rng, rng.choice(CODE_CASES[:30])) for _ in range(200)]]

    workflows = []
    for ids in list(ROUTER_TD.values()) + _router_questions(rng) + [[], ["urgency"]]:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        workflows.append({"ids": shuffled, "workflow": pyrouter.match_typed_decisions_workflow(_noul_questions(shuffled))})

    signatures = []
    for cid, qid, definition in QUESTION_CASES:
        try:
            Agent._check_question(qid, copy.deepcopy(definition))
        except ValueError:
            continue
        internal = Agent._to_internal(copy.deepcopy(definition))
        render = json.dumps([internal["t"], internal["ins"], render_options(internal),
                             [str(k) for k in internal["crit"]] if internal["t"] == "choice" else
                             [render_criterion(c) for c in internal["crit"]] if internal["t"] == "score" else
                             list(_resolve_noul_labels(internal.get("labels"))),
                             internal.get("option_order")], ensure_ascii=False)
        signatures.append({"case": cid, "questions": {qid: definition}, "render": render,
                           "signature": pyrouter._question_schema({qid: definition})})
    for _ in range(30):
        picks = rng.sample([d for _, _, d in QUESTION_CASES if _accepted(d)], 3)
        questions = {"q%d" % i: copy.deepcopy(d) for i, d in enumerate(picks)}
        signatures.append({"case": "multi", "questions": questions,
                           "signature": pyrouter._question_schema(questions)})
        reordered = dict(reversed(list(questions.items())))
        signatures.append({"case": "multi-reversed", "questions": reordered,
                           "signature": pyrouter._question_schema(reordered)})

    routes = []
    for label, router_kwargs, route_kwargs in routing_cases.ROUTE_CASES:
        router, refused = _router_build(router_kwargs)
        record = {"case": label, "router": router_kwargs, "route": route_kwargs}
        record.update({"router_error": refused} if refused else _router_route(router, route_kwargs))
        routes.append(record)

    # Router() arguments Python refuses, and a state a Router sees in every position.
    for bad in ("bogus", "", "Papers", " english/ "):
        routes.append({"case": "default-%r" % (bad,), "router": {"default": bad}, "route": {"state": "x"},
                       "router_error": _router_build({"default": bad})[1]})
    for key in ("en", " ML ", "Laya", "typed_decisions"):
        kwargs = {"models": {key: "/srv/laya-models/a"}, "standalone_repos": key == "en"}
        router, _ = _router_build(kwargs)
        route = {"state": "x", "model": key}
        routes.append(dict({"case": "models-alias-%r" % key, "router": kwargs, "route": route},
                           **_router_route(router, route)))
    # "~" expands to a home directory that differs per machine; the Go tests pin it against $HOME.
    for source in ("/abs/~/x", "x~", "a~b", ""):
        kwargs = {"models": {"english": source}}
        router, _ = _router_build(kwargs)
        route = {"state": "x", "model": "english"}
        routes.append(dict({"case": "path-%r" % source, "router": kwargs, "route": route},
                           **_router_route(router, route)))
    for sub in ("", None, "sub", "a/b"):
        kwargs = {"models": {"multilingual": ["org/repo", sub]}}
        router, _ = _router_build(kwargs)
        route = {"state": "x", "model": "ml"}
        routes.append(dict({"case": "pair-%r" % (sub,), "router": kwargs, "route": route},
                           **_router_route(router, route)))

    questions_pool = _router_questions(rng)
    for _ in range(900):
        kwargs = {}
        if rng.random() < 0.3:
            kwargs["default"] = rng.choice(["english", "en", "ML", "typed", "multilingual", "Laya"])
        if rng.random() < 0.3:
            kwargs["auto_task_detection"] = True
        if rng.random() < 0.2:
            kwargs["standalone_repos"] = True
        if rng.random() < 0.25:
            kwargs["lang_guess"] = rng.choice(CODE_CASES)
        route = {"state": _router_state(rng)}
        if rng.random() < 0.4:
            route["questions"] = _noul_questions(rng.choice(questions_pool))
        if rng.random() < 0.15:
            route["model"] = rng.choice(NAME_CASES)
        if rng.random() < 0.12:
            route["task"] = rng.choice(NAME_CASES + ["typed_decisions", "Typed-Decisions", "TYPED_DECISIONS"])
        if rng.random() < 0.3:
            route["lang"] = rng.choice(CODE_CASES)
        if rng.random() < 0.25:
            route["lang_guess"] = rng.choice(CODE_CASES)
        router, refused = _router_build(kwargs)
        record = {"case": "fuzz", "router": kwargs, "route": route}
        record.update({"router_error": refused} if refused else _router_route(router, route))
        routes.append(record)

    batches = []
    for _ in range(60):
        router_kwargs = {"auto_task_detection": rng.random() < 0.5}
        router = Router(**router_kwargs)
        requests = []
        for _ in range(rng.randint(0, 6)):
            request = {"state": _router_state(rng), "questions": _noul_questions(rng.choice(questions_pool))}
            if rng.random() < 0.3:
                request["model"] = rng.choice(NAME_CASES[:14])
            if rng.random() < 0.3:
                request["lang"] = rng.choice(CODE_CASES)
            if rng.random() < 0.2:
                request["lang_guess"] = rng.choice(CODE_CASES)
            if rng.random() < 0.1:
                request["task"] = rng.choice(["typed-decisions", "TYPED_decisions", "english", "nope"])
            requests.append(request)
        record = {"router": router_kwargs, "requests": requests}
        try:
            record["decisions"] = [_router_decision(d) for d in router.route_batch(requests)]
        except ValueError as refused:
            record["error"] = str(refused)
        batches.append(record)
    for index, state in ((0, None), (2, None)):
        requests = [{"state": "hello there my friend", "questions": {}} for _ in range(index)] + \
                   [{"state": state, "questions": {}}]
        try:
            Router().route_batch(requests)
        except TypeError as refused:
            batches.append({"router": {}, "requests": requests, "error": str(refused)})
    batches.append({"router": {}, "requests": [], "decisions": []})

    refused = sum(1 for r in routes if "error" in r or "router_error" in r)
    if len(routes) < 1000 or refused < 40 or len(names) < 250:
        raise AssertionError("the router cases stopped covering enough (%d routes, %d refused)"
                             % (len(routes), refused))
    hooks, hook_batches = router_hooks_family(rng)
    return {"names": names, "codes": codes, "workflows": workflows, "signatures": signatures,
            "routes": routes, "batches": batches, "hooks": hooks, "hook_batches": hook_batches,
            "timeout_text": [{"seconds": x, "text": "%g" % x} for x in TIMEOUT_TEXT_VALUES]}


# Python's %g on a timeout: six significant digits, exponent form below 1e-4 and from 1e6.
TIMEOUT_TEXT_VALUES = [0.1234567, 1234.567, 123456.7, 1234567.0, 12345678.9, 0.0001, 0.00012345678, 0.00001234567, 1e-07,
                       0.5, 1.0, 30.0, 100000.0, 999999.5, 999999.4, 1000000.0, 0.1, 0.05, 2.5, 3600.0, 1e9,
                       0.30000000000000004, 1.0000005, 1.0000004, 0.99999949, 7.0000001]

ROUTER_HOOK_MODELS = ["english", "multilingual", "typed-decisions"]


def _hook_class(name, events, behaviour, log, run_ids):
    """A hook object whose class is called `name` and which implements only `events`."""
    from laya.router import RouteDecision, _repo_str

    def on_route(self, ctx):
        run_ids.append(ctx.run_id)
        log.append({"hook": name, "event": "on_route", "model": ctx.decision["model"],
                    "reason": ctx.decision["reason"], "states": [x if isinstance(x, str) else "<%s>" % type(x).__name__ for x in ctx.states],
                    "questions": list(ctx.questions), "run_id_len": len(ctx.run_id),
                    "run_id_shared": ctx.run_id == run_ids[0], "router_is_self": ctx.router is not None})
        kind, _, arg = behaviour.partition(":")
        if kind in ("replace", "replace_raise"):
            ctx.decision = RouteDecision(model=arg, repo=_repo_str(ctx.router.models.get(arg)),
                                         reason="replaced by %s" % name, detection=None, workflow=None)
        if kind == "suffix":
            ctx.decision = RouteDecision(dict(ctx.decision, reason=ctx.decision["reason"] + "|" + name))
        if kind in ("raise", "replace_raise"):
            raise RuntimeError("boom from %s" % name)
        if kind == "sleep":
            time.sleep(float(arg))

    def other(self, ctx):
        log.append({"hook": name, "event": "other"})

    on_route.__qualname__ = name + ".on_route"
    other.__qualname__ = name + ".other"
    methods = {event: (on_route if event == "on_route" else other) for event in events}
    return type(name, (), methods)()


def _router_hooks_case(label, router_kwargs, chain, call_kwargs, route_kwargs, requests=None):
    import warnings

    from laya import hooks as pyhooks
    from laya.router import Router

    log, run_ids, objects = [], [], {"default": [], "installed": [], "call": []}
    for spec in chain:
        objects[spec["where"]].append(_hook_class(spec["name"], spec["events"], spec["behaviour"], log, run_ids))
    record = {"case": label, "router": router_kwargs, "chain": chain, "call": call_kwargs, "route": route_kwargs}
    if requests is not None:
        record["requests"] = requests
    pyhooks.clear_default_hooks()
    try:
        for hook in objects["default"]:
            pyhooks.add_default_hook(hook)
        try:
            router = Router(**dict(router_kwargs, hooks=objects["installed"]))
        except ValueError as refused:
            record["router_error"] = str(refused)
            return record
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if requests is None:
                    record["decision"] = _router_decision(
                        router.route(hooks=objects["call"], **call_kwargs, **route_kwargs))
                else:
                    record["decisions"] = [_router_decision(d) for d in router.route_batch(
                        requests, hooks=objects["call"], **call_kwargs)]
            except Exception as refused:  # noqa: BLE001 -- the text is the golden
                record["error"] = {"type": type(refused).__name__, "message": str(refused)}
        record["warnings"] = [str(w.message) for w in caught]
    finally:
        pyhooks.clear_default_hooks()
    record["calls"] = log
    return record


def router_hooks_family(rng):
    """on_route through the real Router: order, replacement, raise, warn, timeout, per-call overrides."""
    out = []

    def hook(name, where="installed", behaviour="ok", events=("on_route",)):
        return {"name": name, "where": where, "events": list(events), "behaviour": behaviour}

    en = {"state": "I was charged twice, please refund the duplicate payment on my invoice.",
          "questions": _noul_questions(["urgency"])}
    de = {"state": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
          "questions": _noul_questions(["urgency"])}
    td = {"state": "x", "questions": _noul_questions(ROUTER_TD["customer_service"])}
    fixed = [
        ("no-hooks", {}, [], {}, en),
        ("observe", {}, [hook("Seen")], {}, en),
        ("observe-multilingual", {}, [hook("Seen")], {}, de),
        ("replace", {}, [hook("Pin", behaviour="replace:multilingual")], {}, en),
        ("replace-typed", {}, [hook("Pin", behaviour="replace:typed-decisions")], {}, de),
        ("second-hook-sees-the-replacement", {},
         [hook("A", behaviour="replace:multilingual"), hook("B")], {}, en),
        ("last-replacement-wins", {},
         [hook("A", behaviour="replace:multilingual"), hook("B", behaviour="replace:typed-decisions")], {}, en),
        ("suffix-chain-keeps-order", {}, [hook("A", behaviour="suffix"), hook("B", behaviour="suffix"),
                                          hook("C", behaviour="suffix")], {}, en),
        ("defaults-then-installed-then-call", {},
         [hook("Call1", "call", "suffix"), hook("Inst1", "installed", "suffix"), hook("Def1", "default", "suffix"),
          hook("Inst2", "installed", "suffix"), hook("Call2", "call", "suffix"), hook("Def2", "default", "suffix")],
         {}, en),
        ("hook-without-on-route-is-skipped", {}, [hook("Other", events=["on_predict_end"]), hook("Seen")], {}, en),
        ("hook-with-several-events", {}, [hook("Many", events=["on_route", "on_load", "on_evict", "on_error"])], {}, en),
        ("raise", {}, [hook("Bad", behaviour="raise")], {}, en),
        ("raise-stops-the-chain", {}, [hook("A"), hook("Bad", behaviour="raise"), hook("C")], {}, en),
        ("raise-warns-and-continues", {"hooks_raise": False},
         [hook("A"), hook("Bad", behaviour="raise"), hook("C")], {}, en),
        ("replace-then-raise-raises", {}, [hook("Bad", behaviour="replace_raise:multilingual")], {}, en),
        ("replace-then-raise-keeps-the-replacement-when-warned", {"hooks_raise": False},
         [hook("Bad", behaviour="replace_raise:multilingual"), hook("C")], {}, en),
        ("call-overrides-raise-to-warn", {}, [hook("Bad", behaviour="raise"), hook("C")], {"hooks_raise": False}, en),
        ("call-overrides-warn-to-raise", {"hooks_raise": False}, [hook("Bad", behaviour="raise")],
         {"hooks_raise": True}, en),
        ("call-hook-raises", {}, [hook("Bad", "call", "raise")], {}, en),
        ("default-hook-raises", {}, [hook("Bad", "default", "raise"), hook("C")], {}, en),
        ("timeout-raises", {"hooks_timeout": 0.1}, [hook("Slow", behaviour="sleep:0.6")], {}, en),
        ("timeout-stops-the-chain", {"hooks_timeout": 0.1},
         [hook("A"), hook("Slow", behaviour="sleep:0.6"), hook("C")], {}, en),
        ("timeout-text-keeps-six-significant-digits", {"hooks_timeout": 0.1234567}, [hook("Slow", behaviour="sleep:0.6")],
         {}, en),
        ("timeout-text-per-call-shorter", {}, [hook("Slow", behaviour="sleep:0.6")], {"hooks_timeout": 0.1049999}, en),
        ("timeout-warns-and-continues", {"hooks_timeout": 0.1, "hooks_raise": False},
         [hook("A"), hook("Slow", behaviour="sleep:0.6"), hook("C")], {}, en),
        ("timeout-does-not-trip-a-fast-hook", {"hooks_timeout": 5.0}, [hook("A", behaviour="replace:multilingual")], {}, en),
        ("call-timeout-overrides-router", {}, [hook("Slow", behaviour="sleep:0.6")], {"hooks_timeout": 0.1}, en),
        ("call-timeout-loosens-router", {"hooks_timeout": 0.1}, [hook("Slow", behaviour="sleep:0.3")],
         {"hooks_timeout": 5.0}, en),
        ("call-timeout-zero", {}, [hook("A")], {"hooks_timeout": 0}, en),
        ("call-timeout-negative", {}, [hook("A")], {"hooks_timeout": -1}, en),
        ("call-timeout-zero-with-no-hooks", {}, [], {"hooks_timeout": 0}, en),
        ("call-timeout-zero-after-a-route-error", {}, [hook("A")], {"hooks_timeout": 0}, dict(en, model="nope")),
        ("router-timeout-zero", {"hooks_timeout": 0}, [], {}, en),
        ("router-timeout-negative", {"hooks_timeout": -0.5}, [], {}, en),
        ("route-error-runs-no-hook", {}, [hook("A")], {}, dict(en, model="nope")),
        ("explicit-model-then-hook-replaces", {}, [hook("Pin", behaviour="replace:english")], {}, dict(de, model="ml")),
        ("workflow-states-and-questions-seen", {"auto_task_detection": True}, [hook("Seen")], {}, td),
        ("empty-questions-seen", {}, [hook("Seen")], {}, {"state": "hello there my friend"}),
        ("unserialised-hooks", {"hooks_concurrent": False}, [hook("A", behaviour="suffix"), hook("B", behaviour="suffix")],
         {}, en),
    ]
    for label, router_kwargs, chain, call_kwargs, route_kwargs in fixed:
        out.append(_router_hooks_case(label, router_kwargs, chain, call_kwargs, route_kwargs))

    behaviours = ["ok", "ok", "ok", "suffix", "suffix", "replace:english", "replace:multilingual",
                  "replace:typed-decisions", "raise", "replace_raise:multilingual", "sleep:0.6"]
    event_sets = [["on_route"], ["on_route"], ["on_route"], ["on_route", "on_load"], ["on_predict_end"], ["on_error"]]
    questions_pool = _router_questions(rng)
    for i in range(300):
        chain = [hook("H%d" % j, rng.choice(["default", "installed", "installed", "call"]),
                      rng.choice(behaviours), rng.choice(event_sets)) for j in range(rng.randint(1, 4))]
        router_kwargs, call_kwargs = {}, {}
        if rng.random() < 0.3:
            router_kwargs["hooks_raise"] = rng.random() < 0.5
        if rng.random() < 0.3:
            router_kwargs["hooks_timeout"] = rng.choice([0.1, 5.0])
        if rng.random() < 0.2:
            router_kwargs["hooks_concurrent"] = rng.random() < 0.5
        if rng.random() < 0.3:
            call_kwargs["hooks_raise"] = rng.random() < 0.5
        if rng.random() < 0.2:
            call_kwargs["hooks_timeout"] = rng.choice([0.1, 5.0, 0, -1])
        if any(h["behaviour"].startswith("sleep") for h in chain) and "hooks_timeout" not in call_kwargs:
            router_kwargs.setdefault("hooks_timeout", 0.05)
        if rng.random() < 0.25:
            router_kwargs["auto_task_detection"] = True
        route_kwargs = {"state": _router_state(rng)}
        if rng.random() < 0.5:
            route_kwargs["questions"] = _noul_questions(rng.choice(questions_pool))
        if rng.random() < 0.1:
            route_kwargs["model"] = rng.choice(["english", "ml", "typed", "nope"])
        if rng.random() < 0.2:
            route_kwargs["lang"] = rng.choice(CODE_CASES[:30])
        out.append(_router_hooks_case("fuzz-%d" % i, router_kwargs, chain, call_kwargs, route_kwargs))

    batches = []
    for i in range(40):
        chain = [hook("H%d" % j, rng.choice(["default", "installed", "call"]),
                      rng.choice(behaviours[:10]), ["on_route"]) for j in range(rng.randint(1, 3))]
        router_kwargs, call_kwargs = {}, {}
        if rng.random() < 0.3:
            router_kwargs["hooks_raise"] = rng.random() < 0.5
        if rng.random() < 0.3:
            call_kwargs["hooks_raise"] = rng.random() < 0.5
        if rng.random() < 0.2:
            call_kwargs["hooks_timeout"] = rng.choice([5.0, 0])
        requests = []
        for _ in range(rng.randint(0, 4)):
            request = {"state": _router_state(rng), "questions": _noul_questions(rng.choice(questions_pool))}
            if rng.random() < 0.2:
                request["model"] = rng.choice(["english", "typed"])
            requests.append(request)
        batches.append(_router_hooks_case("batch-%d" % i, router_kwargs, chain, call_kwargs, {}, requests=requests))
    batches.append(_router_hooks_case("batch-hook-sees-each-request", {}, [hook("Seen", behaviour="suffix")], {}, {},
                                      requests=[{"state": "one", "questions": {}}, {"state": "two", "questions": {}}]))
    batches.append(_router_hooks_case("batch-timeout-applies-to-every-request", {}, [hook("Slow", behaviour="sleep:0.6")],
                                      {"hooks_timeout": 0.1}, {}, requests=[{"state": "one", "questions": {}}]))
    return out, batches


LIFECYCLE_NAMES = ["english", "multilingual", "typed-decisions", "en", "ML", "typed", " Typed_Decisions "]


def _lifecycle_hook(name, events, behaviour, log):
    """A hook object named `name` that logs `events` and then does `behaviour` ("ok", "raise" or "sleep:S")."""
    def make(event):
        def method(self, ctx):
            agent = getattr(ctx, "agent", None)
            log.append("%s.%s:%s%s" % (name, event, ctx.model, ":" + agent.label if agent is not None else ""))
            kind, _, arg = behaviour.partition(":")
            if kind == "raise":
                raise RuntimeError("boom from %s" % name)
            if kind == "sleep":
                time.sleep(float(arg))
        method.__qualname__ = name + "." + event
        return method
    return type(name, (), {event: make(event) for event in events})()


def _lifecycle_case(label, router_kwargs, hooks, ops):
    """Run `ops` on a real Router whose Agent is a stub; record loaded(), events and errors per op."""
    import types
    import warnings

    import laya.agent as pyagent
    from laya import hooks as pyhooks
    from laya.router import Router, _split

    log, state = [], {"n": 0, "fail": set()}

    class Stub:
        def __init__(self, repo, **kwargs):
            state["n"] += 1
            name = state["by_source"][(repo, kwargs.get("subfolder"))]
            if name in state["fail"]:
                raise ValueError("boom building %s" % name)
            self.label = "built-%d" % state["n"]

    record = {"case": label, "router": router_kwargs, "hooks": hooks, "ops": []}
    pyhooks.clear_default_hooks()
    real = pyagent.Agent
    pyagent.Agent = Stub
    try:
        router = Router(**dict(router_kwargs, hooks=[_lifecycle_hook(h["name"], h["events"], h["behaviour"], log)
                                                      for h in hooks]))
        state["by_source"] = {_split(source): name for name, source in router.models.items()}
        attached = {}
        record["max_loaded"] = router.max_loaded
        for op in ops:
            step = dict(op)
            before = len(log)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    kind = op["op"]
                    if kind == "load":
                        step["agent"] = router.load(op["name"]).label
                    elif kind == "preload":
                        router.preload(op["names"])
                    elif kind == "attach":
                        agent = attached.setdefault(op["agent"], types.SimpleNamespace(label=op["agent"]))
                        router.attach(op["name"], agent)
                    elif kind == "unload":
                        router.unload(op["name"])
                    elif kind == "fail":
                        state["fail"].add(op["name"])
                    elif kind == "heal":
                        state["fail"].discard(op["name"])
                except Exception as refused:  # noqa: BLE001 -- the text is the golden
                    step["error"] = str(refused)
            step["events"] = log[before:]
            step["warnings"] = [str(w.message) for w in caught]
            step["loaded"] = list(router.loaded)
            step["max_loaded"] = router.max_loaded
            record["ops"].append(step)
    finally:
        pyagent.Agent = real
        pyhooks.clear_default_hooks()
    return record


def digests_row(env, argument, pin):
    """One router_digests row: Router(sha256_digests=argument) under LAYA_SHA256_DIGESTS=env (None:
    unset), and the map each checkpoint would be built with given the shared `pin`."""
    from unittest import mock
    from laya.router import Router, _digest_entry, _merge_expected_digests

    row = {"env": env, "argument": argument, "shared": pin}
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LAYA_SHA256_DIGESTS", None)
        if env is not None:
            os.environ["LAYA_SHA256_DIGESTS"] = env
        try:
            router = Router(sha256_digests=argument)
        except ValueError as exc:
            row["error"] = str(exc)
        else:
            row["table"] = {k: router.sha256_digests[k] for k in sorted(router.sha256_digests)}
            row["expected"] = {
                name: _merge_expected_digests(pin, _digest_entry(router.sha256_digests, name))
                for name in ("english", "multilingual", "typed-decisions")}
    return row


def router_digests_family():
    """What the real Router holds and merges for per-checkpoint SHA-256 pins: the
    LAYA_SHA256_DIGESTS layering (flat, nested, mixed, malformed), the `sha256_digests` argument
    (explicit None masks), the `agent_kwargs["expected_sha256"]` channel and the merge. Each row is
    either an error text from Router(...) or the table plus, per checkpoint, the map
    _merge_expected_digests hands Agent (null: none passed). No model runs."""
    a, b, c, d, e = ("%s" % ch * 64 for ch in "abcde")
    english = {"model.safetensors": a}
    multi = {"model.safetensors": b}
    shared_pin = {"tokenizer.json": d}
    envs = [
        None, "", "   ", "{not json", "[]", '"digests"', "{}", "null", "7",
        json.dumps({"model.safetensors": a}),
        json.dumps({"model.safetensors": a, "tokenizer.json": d}),
        json.dumps({"english": english, "multilingual": multi}),
        json.dumps({"en": english}),
        json.dumps({" ML ": multi}),
        json.dumps({"english": {}}),
        json.dumps({"typed": english, "typed-decisions": multi}),
        json.dumps({"english": english, "english ": multi}),
        json.dumps({"engligh": english}),
        json.dumps({"model.safetensors": a, "english": english}),
        json.dumps({"english": english, "model.safetensors": a}),
        json.dumps({"english": english, "x": 1}),
        json.dumps({"english": None}),
        json.dumps({"a": 1}),
        json.dumps({"z": 1, "a": "s", "m": None}),
        json.dumps({"english": english, "multilingual": []}),
        # Python's json.loads reads these; encoding/json does not
        '{"english": NaN}', '{"a": -Infinity}', '[NaN]', "NaN",
        '{"english": {"a": "%s"}, "x": Infinity}' % a,
        # tokens inside strings (escaped quote, escaped backslash) and glued to other text
        '{"english": {"a": "x\\":NaN, y"}}', '{"english": {"a": "\\\\"}, "x": NaN}',
        '{"english": NaNe5}', '{"a": Infinity.5}', '{"english": -Infinity5}', '{"english": {"a": "q"}, "b": NaN1}',
    ]
    arguments = [
        None, {}, {"english": {"model.safetensors": c}}, {"ml": None}, {"engligh": english},
        {"english": {}}, {"english": None, "multilingual": multi}, {"en": english},
        {"typed": {"tokenizer.json": e}},
    ]
    shared = [None, {}, shared_pin, {"model.safetensors": c}, {"model.safetensors": c, "tokenizer.json": d}]
    return [digests_row(env, argument, pin) for env in envs for argument in arguments for pin in shared]


def lifecycle_family():
    """Eviction order, loaded() sequences, attach/preload/unload and hook events from the real Router
    with a stub Agent: no model runs."""
    rng = random.Random(SEED + 3)
    rec = {"name": "Rec", "events": ["on_load", "on_evict"], "behaviour": "ok"}

    def load(n):
        return {"op": "load", "name": n}

    def unload(n):
        return {"op": "unload", "name": n}

    def attach(n, a):
        return {"op": "attach", "name": n, "agent": a}

    def preload(names=None):
        return {"op": "preload", "names": names}

    def fail(n):
        return {"op": "fail", "name": n}

    def heal(n):
        return {"op": "heal", "name": n}

    en, ml, td = "english", "multilingual", "typed-decisions"
    fixed = [
        ("load-cached", {}, [rec], [load(en), load(en), load(ml), load(td), load(td), load(en)]),
        ("lru-touch", {}, [rec], [load(en), load(ml), load(en), load(td), load(ml)]),
        ("max-1", {"max_loaded": 1}, [rec], [load(en), load(ml), load(en)]),
        ("max-3", {"max_loaded": 3}, [rec], [load(en), load(ml), load(td), load(en)]),
        ("aliases", {}, [rec], [load("en"), load("ML"), load(" Typed_Decisions "), load("laya"), load("decisions")]),
        ("unknown-name", {}, [rec], [load("gpt"), load(""), unload("gpt"), preload(["gpt"]), load(en)]),
        ("attach-raises-max", {}, [rec], [attach(en, "A1"), attach(ml, "A2"), attach(td, "A3"), load(en), load(ml)]),
        ("attach-then-load-evicts-oldest", {}, [rec], [attach(en, "A1"), load(ml), load(td), load(en)]),
        ("attach-replaces", {}, [rec], [attach(en, "A1"), attach(en, "A2"), load(en), attach(en, "A2")]),
        ("attach-over-built", {}, [rec], [load(en), attach(en, "A1"), load(en), unload(en), load(en)]),
        ("attach-touches", {}, [rec], [load(en), load(ml), attach(en, "A1"), load(td)]),
        ("attach-alias", {}, [rec], [attach("en", "A1"), attach("multi", "A2"), load("laya")]),
        ("preload-all-raises-max", {"max_loaded": 1}, [rec], [preload(), load(en)]),
        ("preload-staged", {}, [rec], [preload([en]), preload([ml]), preload([td, en]), load(ml)]),
        ("preload-resident-not-touched", {}, [rec], [load(en), load(ml), preload([en]), load(td)]),
        ("preload-with-attached", {"max_loaded": 1}, [rec], [attach(en, "A1"), preload([ml]), preload()]),
        ("preload-duplicates", {}, [rec], [preload([ml, ml, "multi"])]),
        ("unload-one", {}, [rec], [load(en), load(ml), unload(en), unload(en), load(en)]),
        ("unload-all", {}, [rec], [load(en), load(ml), load(en), unload(None), unload(None), load(td)]),
        ("unload-attached", {}, [rec], [attach(en, "A1"), load(ml), unload(en), load(en)]),
        ("build-failure-not-cached", {}, [rec], [fail(ml), load(ml), load(ml), heal(ml), load(ml)]),
        ("build-failure-evicts-nothing", {}, [rec], [load(en), load(ml), fail(td), load(td), heal(td), load(td)]),
        ("preload-failure-stops", {}, [rec], [fail(ml), preload(), heal(ml), preload()]),
        ("hook-raise-on-load", {}, [rec, {"name": "Boom", "events": ["on_load"], "behaviour": "raise"}],
         [load(en), load(en), load(ml)]),
        ("hook-raise-on-load-warn", {"hooks_raise": False},
         [rec, {"name": "Boom", "events": ["on_load"], "behaviour": "raise"}], [load(en), load(ml)]),
        ("hook-raise-on-evict-skips-load", {"max_loaded": 1},
         [{"name": "Boom", "events": ["on_evict"], "behaviour": "raise"}, rec], [load(en), load(ml), load(td)]),
        ("hook-raise-on-evict-warn", {"max_loaded": 1, "hooks_raise": False},
         [{"name": "Boom", "events": ["on_evict"], "behaviour": "raise"}, rec], [load(en), load(ml), load(td)]),
        ("hook-raise-on-unload", {},
         [rec, {"name": "Boom", "events": ["on_evict"], "behaviour": "raise"}], [load(en), load(ml), unload(None), load(en)]),
        ("hook-timeout-on-load", {"hooks_timeout": 0.1},
         [rec, {"name": "Slow", "events": ["on_load"], "behaviour": "sleep:0.6"}], [load(en), load(ml)]),
        ("hook-timeout-warn", {"hooks_timeout": 0.1, "hooks_raise": False},
         [{"name": "Slow", "events": ["on_load"], "behaviour": "sleep:0.6"}, rec], [load(en)]),
        ("hook-order", {}, [{"name": "A", "events": ["on_load", "on_evict"], "behaviour": "ok"},
                            {"name": "B", "events": ["on_evict"], "behaviour": "ok"},
                            {"name": "C", "events": ["on_load"], "behaviour": "ok"}],
         [load(en), load(ml), load(td), unload(None)]),
        ("hook-other-events-skipped", {}, [{"name": "Route", "events": ["on_route"], "behaviour": "ok"},
                                           {"name": "Pred", "events": ["on_predict_start", "on_error"], "behaviour": "ok"}],
         [load(en), unload(en)]),
        ("hook-serial", {"hooks_concurrent": False}, [rec], [load(en), load(ml), load(td), unload(None)]),
        ("hook-attach-no-event", {}, [rec], [attach(en, "A1"), attach(ml, "A2"), attach(en, "A3")]),
        ("standalone-repos", {"standalone_repos": True}, [rec], [load(en), load(ml), load(td)]),
    ]
    out = [_lifecycle_case(*case) for case in fixed]
    raisers = [{"name": "Boom", "events": ["on_load"], "behaviour": "raise"},
               {"name": "Bang", "events": ["on_evict"], "behaviour": "raise"},
               {"name": "Both", "events": ["on_load", "on_evict"], "behaviour": "raise"}]
    for i in range(240):
        ops = []
        for _ in range(rng.randint(1, 14)):
            kind = rng.choices(["load", "preload", "attach", "unload", "fail", "heal"], [8, 2, 2, 3, 1, 1])[0]
            name = rng.choice(LIFECYCLE_NAMES)
            if kind == "load":
                ops.append(load(name))
            elif kind == "preload":
                ops.append(preload(rng.choice([None, [name], rng.sample(LIFECYCLE_NAMES, 2)])))
            elif kind == "attach":
                ops.append(attach(name, rng.choice(["A1", "A2", "A3"])))
            elif kind == "unload":
                ops.append(unload(rng.choice([None, name])))
            elif kind == "fail":
                ops.append(fail(rng.choice(LIFECYCLE_NAMES[:3])))
            else:
                ops.append(heal(rng.choice(LIFECYCLE_NAMES[:3])))
        kwargs = {}
        if rng.random() < 0.7:
            kwargs["max_loaded"] = rng.randint(1, 3)
        hooks = [rec]
        if i % 3 == 0:
            hooks.append(rng.choice(raisers))
            kwargs["hooks_raise"] = rng.random() < 0.5
        out.append(_lifecycle_case("fuzz-%d" % i, kwargs, hooks, ops))
    return out


def _accepted(definition):
    import copy

    from laya.agent import Agent
    try:
        Agent._check_question("q", copy.deepcopy(definition))
    except ValueError:
        return False
    return True


def build():
    import tokenizers

    rng = random.Random(SEED)
    return {
        "meta": {"seed": SEED, "tokenizers": tokenizers.__version__,
                 "cpython_unicode": unicodedata.unidata_version},
        "normalize": normalize_family(rng),
        "lowercase": lowercase_family(rng),
        "python_lower": sigma_family(rng),
        "single_word": single_word_family(rng),
        "pretokenize": pretokenize_family(rng),
        "normalizers": normalizers_family(rng),
        "encode_fuzz": encode_family(rng),
        "decode_fuzz": decode_family(rng),
        "state_shapes": state_family(rng),
        "invalid_utf8": invalid_utf8_family(rng),
        "round4": round4_family(rng),
        "questions": questions_family(rng),
        "states": states_family(rng),
        "label_answers": label_answers_family(rng),
        "answers_fuzz": decode_fuzz_family(rng),
        "marker_refusals": marker_refusals_family(rng),
        "predict_plan": predict_plan_family(rng),
        "calibration": calibration_family(),
        "shortlist_order": shortlist_order_family(rng),
        # `router` stays ahead of `email_fuzz`. Both draw from this one shared `rng`, so whichever
        # runs first fixes the other's inputs: #1311 recorded `router.signatures` with nothing
        # between it and `predict_plan`, and `router/golden_test.go` pins how many of those 107
        # signatures come out byte-identical to Python's. `email_fuzz` has no such constant, only
        # a case count, so it is the one that may be re-drawn. `calibration_family` takes no rng
        # and `shortlist_order_family`/`email_fold_family` consume no draws, so they are free.
        "router": router_family(rng),
        "email_fold": email_fold_family(rng),
        "email_fuzz": email_fuzz_family(rng),
        "lifecycle": lifecycle_family(),
        "router_digests": router_digests_family(),
    }


# The two families whose answers pass through numpy's float32 exp, a SIMD routine whose last bit
# depends on the CPU. --check compares their inputs and leaves the answers to the tests, which
# read a golden recorded on the machine they run on.
EXP_DEPENDENT = ("label_answers", "answers_fuzz")


def _unnumber(value):
    """value with every number replaced by 0, so only its structure is left."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, dict):
        return {k: _unnumber(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unnumber(v) for v in value]
    return value


def _key(value):
    """value spelled as json.dumps writes it as an object key, or None if it cannot be a key."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    return None


def _answer_shape(answer):
    """One answer's type, key sets and order and legend, with no number and no chosen option.

    A choice is blanked only when it is one of the answer's options, so a choice that names no
    option, or is not a scalar, still differs. Option membership is by key spelling, so a label's
    type (false against "false") is not compared here; the Go tests pin it against a golden
    recorded where they run. Anything that is not an answer is kept as it is.
    """
    if not isinstance(answer, dict):
        return answer
    shaped = _unnumber(answer)
    probabilities = answer.get("probabilities")
    if "choice" in answer and isinstance(probabilities, dict) and _key(answer["choice"]) in probabilities:
        shaped["choice"] = "<an option>"  # not None, so a choice changed to null still differs
    return shaped


def _answers_shape(answers):
    if not isinstance(answers, dict):
        return answers
    return {question_id: _answer_shape(answer) for question_id, answer in answers.items()}


def _text_shape(text):
    """answer_text, parsed and shaped; text that is not JSON is kept, so it differs."""
    try:
        return render(_answer_shape(json.loads(text))).rstrip("\n")
    except (TypeError, ValueError, RecursionError):
        return text


def portable(payload):
    out = dict(payload)
    for name in EXP_DEPENDENT:
        out[name] = {"cases": [{**case, "answer_text": _text_shape(case.get("answer_text"))}
                               for case in payload[name]["cases"]]}
    out["predict_plan"] = {
        name: ({"cases": [{**case, "results": [{**r, "answers": _answers_shape(r.get("answers"))}
                                              for r in case["results"]]}
                          for case in value["cases"]]} if name != "questions" else value)
        for name, value in payload["predict_plan"].items()}
    return out


def render(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", metavar="DIR", default=DEFAULT_OUT,
                        help="write %s here (default laya-go/fixtures)" % NAME)
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the file in --out differs")
    args = parser.parse_args(argv)
    path = os.path.join(os.path.abspath(args.out), NAME)
    payload = build()
    fresh = render(payload)
    if args.check:
        try:
            with open(path, encoding="utf-8") as handle:
                current = render(portable(json.load(handle)))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RecursionError):
            current = None
        if current != render(portable(payload)):
            print("gen_go_fixtures: %s is stale or missing; run laya-go/scripts/gen_go_fixtures.py"
                  % path, file=sys.stderr)
            return 1
        print("gen_go_fixtures: %s matches" % path)
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(fresh)
    print("wrote %s (%d bytes)" % (path, len(fresh.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
