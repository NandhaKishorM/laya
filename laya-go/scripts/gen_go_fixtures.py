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
                                         sort_by_length=case["sort"], state_token_counts=True)
            recorded.append({
                "name": case["name"], "states": case["states"], "questions": case["questions"],
                "batch_size": case["batch_size"], "sort": case["sort"], "lang": case["lang"],
                "runs": list(log),
                "results": [{"model": r["model"], "answer_ids": list(r["answers"]),
                             "answers": r["answers"], "usage": r["usage"]} for r in results]})
        out[name] = {"cases": recorded}
    out["questions"] = PLAN_QUESTIONS
    return out



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
