"""Presentation-sensitivity regression checks for a Laya checkpoint.

Why this exists
---------------
``laya-multilingual``'s ``score`` head has a learned prior against the first-listed
level (#131). The fix is a position-balanced retrain, and this is the check that says
whether a retrained checkpoint removed the prior. It needs no labelled data: every
input is fixed in this file, and both checks compare the checkpoint against itself
under different presentations of the same question.

Checks
------
``score_slot0_identical``
    The identical-option control from @AlKor13 in #131. A ``score`` question whose K
    levels all carry the same text, so the rendered options differ only by position
    and the ``level N:`` prefix ``render_options`` always emits. The metric is the raw
    marker logit of slot 0 minus the mean over the K slots, averaged over every state
    and every (text, K) configuration. A checkpoint with no slot-0 deficit sits at or
    above 0; the gate is one-sided (see "Limits" in README.md).

``score_first_slot_permuted``
    Three real levels presented in all 3! = 6 orders for each state. Every level sits
    in every slot exactly twice per state, so a checkpoint whose answer does not
    depend on the order picks the first slot in exactly 1/3 of the decisions. The
    metric is the observed first-slot rate.

``choice_slot0_identical``
    The same control for ``choice`` (#602, part a). Choice keys must be unique, so the
    options are numbered keys with one shared description (``1: a request``,
    ``2: a request``, ...), the counterpart of score's ``level N:`` prefix. Gated at 4
    options, with 3 reported beside it; same gate as the score control. It runs with
    ``--lang`` (or when named in ``--checks``), so the default English report is unchanged.

All metrics read raw marker logits (before temperature) through
``laya_eval.score_cases``, the same forward pass ``research/scripts/bench_local.py``
uses. ``parity`` checks that path against ``Agent.system_one`` before anything is
reported.

Languages
---------
``--lang`` runs the same two checks on fixed states in Japanese, Korean, Hindi or
Turkish (#602): translations of the English states, with the level texts in the same
language. Every one of them routes to ``multilingual`` under ``Router``, so they test
the checkpoint #131 is about on the scripts it serves. The gates are the English ones.
Without ``--lang`` the run and its report are exactly what they were.

Usage
-----
    python research/eval/presentation_checks.py --model convaiinnovations/laya
    python research/eval/presentation_checks.py --model convaiinnovations/laya \\
        --subfolder multilingual --out multilingual.json
    python research/eval/presentation_checks.py --model convaiinnovations/laya \\
        --subfolder multilingual --lang ja,ko,hi,tr --out multilingual-langs.json

Exit status: 0 every check passed, 1 a check failed, 2 the harness could not be
trusted (parity with ``Agent.system_one`` above ``PARITY_TOL``).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Same path handling as laya_eval.py: allow running this file directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.eval.laya_eval import score_cases, softmax_t, temperature_for  # noqa: E402

# Fixed English states written for this check. Urgency varies on purpose: a slot
# prior has to show through states whose honest answer is low, middle and high.
STATES = (
    "Hi, whenever you get a chance, could you update the billing address on our account "
    "to 12 Harbour Street? No rush at all.",
    "Our invoice for March shows two charges for the same seat. Could you look into it "
    "this week?",
    "The CSV export has been failing since this morning and our finance team cannot close "
    "the month until it works.",
    "Just wanted to say thanks for the onboarding call yesterday. Nothing needed from "
    "your side.",
    "We are thinking about adding ten more licences next quarter. Can someone send pricing "
    "when convenient?",
    "Login is down for every user in our company. Nobody can reach the dashboard and our "
    "customers are waiting.",
    "The mobile app sometimes shows the wrong time zone on reports. It is annoying but we "
    "can work around it.",
    "Our API key was revoked by mistake and every production request is returning 401 "
    "errors right now.",
    "Payroll runs tomorrow morning and the integration still rejects our employee file "
    "with a schema error.",
    "Could you add a dark mode to the admin screens at some point? Several of us would "
    "like it.",
)
INSTRUCTIONS = "How urgent is this request?"
IDENTICAL_TEXTS = ("moderate", "a request")
IDENTICAL_KS = (3, 4, 5)
# Option counts for the choice control. The first is gated: 4, as in a 4-option routing
# question; 3 is reported next to it.
CHOICE_KS = (4, 3)
LEVELS = ("Not urgent", "Soon", "Work is blocked")

# Fixed states in other languages, for the multilingual checkpoint (#602). Each tuple follows
# STATES item for item, so the urgency spread is the same in every language; every state routes
# to `multilingual` under `Router` (tested offline). The level texts mirror LEVELS. They were
# written for this check by a non-native speaker, except Japanese; corrections are welcome.
LANGUAGES: Dict[str, Dict[str, Any]] = {
    "en": {"states": STATES, "instructions": INSTRUCTIONS, "identical_texts": IDENTICAL_TEXTS,
           "levels": LEVELS},
    "ja": {
        "states": (
            "お手すきの際に、アカウントの請求先住所をハーバー通り12番地に変更していただけますでしょうか。急ぎではございません。",
            "3月分の請求書に、同じシートの料金が二重に計上されております。今週中にご確認いただけますでしょうか。",
            "今朝からCSVの書き出しが失敗しており、復旧するまで経理部門が月次の締めを行えない状況です。",
            "昨日のオンボーディングのお打ち合わせ、ありがとうございました。こちらから特にお願いすることはございません。",
            "来四半期にライセンスを10件追加することを検討しております。ご都合のよいときにお見積もりをお送りいただけますか。",
            "弊社の全ユーザーがログインできない状態です。誰もダッシュボードにアクセスできず、お客様をお待たせしております。",
            "モバイルアプリのレポートで、ときどきタイムゾーンが誤って表示されます。不便ではありますが、回避はできております。",
            "APIキーが誤って無効化され、現在すべての本番リクエストが401エラーを返しております。",
            "明朝に給与計算の実行を控えておりますが、連携機能が依然としてスキーマエラーで従業員ファイルを受け付けません。",
            "いずれ管理画面にダークモードを追加していただくことは可能でしょうか。社内でも何名か希望しております。",
        ),
        "instructions": "この依頼の緊急度は？",
        "identical_texts": ("中程度", "依頼"),
        "levels": ("急がない", "早めに", "業務が止まっている"),
    },
    "ko": {
        "states": (
            "시간 되실 때 저희 계정의 청구 주소를 하버가 12번지로 변경해 주실 수 있을까요? 전혀 급하지 않습니다.",
            "3월 청구서에 같은 좌석 요금이 두 번 청구되어 있습니다. 이번 주 안에 확인해 주실 수 있을까요?",
            "오늘 아침부터 CSV 내보내기가 계속 실패하고 있어서, 해결될 때까지 재무팀이 월말 마감을 할 수 없습니다.",
            "어제 온보딩 통화 감사했습니다. 저희 쪽에서 따로 필요한 것은 없습니다.",
            "다음 분기에 라이선스를 10개 더 추가하려고 합니다. 편하실 때 가격 정보를 보내 주실 수 있나요?",
            "회사 전체 사용자가 로그인할 수 없습니다. 아무도 대시보드에 접속하지 못하고 고객들이 기다리고 있습니다.",
            "모바일 앱 보고서에서 가끔 시간대가 잘못 표시됩니다. 불편하지만 우회할 수는 있습니다.",
            "API 키가 실수로 취소되어 지금 모든 운영 요청이 401 오류를 반환하고 있습니다.",
            "내일 아침에 급여 처리가 실행되는데, 연동 기능이 아직도 스키마 오류로 직원 파일을 거부하고 있습니다.",
            "나중에 관리 화면에 다크 모드를 추가해 주실 수 있을까요? 저희 중 몇 명이 원하고 있습니다.",
        ),
        "instructions": "이 요청은 얼마나 긴급한가요?",
        "identical_texts": ("보통", "요청"),
        "levels": ("급하지 않음", "빨리", "업무가 멈춤"),
    },
    "hi": {
        "states": (
            "जब भी आपके पास समय हो, क्या आप हमारे खाते का बिलिंग पता 12 हार्बर स्ट्रीट कर सकते हैं? कोई जल्दी नहीं है।",
            "मार्च के इनवॉइस में एक ही सीट के लिए दो बार शुल्क लगा है। क्या आप इस हफ़्ते इसे देख सकते हैं?",
            "आज सुबह से CSV एक्सपोर्ट विफल हो रहा है और जब तक यह ठीक नहीं होता, हमारी वित्त टीम महीने का हिसाब बंद नहीं कर सकती।",
            "कल की ऑनबोर्डिंग कॉल के लिए धन्यवाद। आपकी ओर से कुछ भी करने की ज़रूरत नहीं है।",
            "हम अगली तिमाही में दस और लाइसेंस जोड़ने के बारे में सोच रहे हैं। क्या कोई सुविधानुसार कीमतें भेज सकता है?",
            "हमारी कंपनी के सभी उपयोगकर्ताओं के लिए लॉगिन बंद है। कोई भी डैशबोर्ड तक नहीं पहुँच पा रहा और हमारे ग्राहक इंतज़ार कर रहे हैं।",
            "मोबाइल ऐप की रिपोर्ट में कभी-कभी गलत टाइम ज़ोन दिखता है। यह परेशान करता है, पर हम इससे काम चला सकते हैं।",
            "हमारी API कुंजी गलती से रद्द हो गई है और अभी हर प्रोडक्शन अनुरोध 401 त्रुटि दे रहा है।",
            "कल सुबह पेरोल चलना है और इंटीग्रेशन अब भी स्कीमा त्रुटि के कारण हमारी कर्मचारी फ़ाइल को अस्वीकार कर रहा है।",
            "क्या आप कभी एडमिन स्क्रीन में डार्क मोड जोड़ सकते हैं? हम में से कई लोग इसे चाहते हैं।",
        ),
        "instructions": "यह अनुरोध कितना ज़रूरी है?",
        "identical_texts": ("मध्यम", "एक अनुरोध"),
        "levels": ("ज़रूरी नहीं", "जल्द", "काम रुका हुआ है"),
    },
    "tr": {
        "states": (
            "Müsait olduğunuzda hesabımızdaki fatura adresini Liman Sokak No. 12 olarak güncelleyebilir misiniz? Hiç acelesi yok.",
            "Mart faturamızda aynı kullanıcı lisansı için iki kez ücret görünüyor. Bu hafta içinde bakabilir misiniz?",
            "CSV dışa aktarımı bu sabahtan beri başarısız oluyor ve düzelene kadar finans ekibimiz ay sonu kapanışını yapamıyor.",
            "Dünkü tanıtım görüşmesi için teşekkür etmek istedim. Sizden bir şey gerekmiyor.",
            "Gelecek çeyrekte on lisans daha eklemeyi düşünüyoruz. Uygun olduğunuzda fiyat bilgisi gönderebilir misiniz?",
            "Şirketimizdeki tüm kullanıcılar için giriş çalışmıyor. Kimse panele ulaşamıyor ve müşterilerimiz bekliyor.",
            "Mobil uygulama raporlarda bazen yanlış saat dilimi gösteriyor. Can sıkıcı ama idare edebiliyoruz.",
            "API anahtarımız yanlışlıkla iptal edildi ve şu anda tüm üretim istekleri 401 hatası döndürüyor.",
            "Bordro yarın sabah çalışacak ve entegrasyon çalışan dosyamızı hâlâ şema hatasıyla reddediyor.",
            "Bir ara yönetim ekranlarına karanlık mod ekleyebilir misiniz? Birkaçımız bunu istiyor.",
        ),
        "instructions": "Bu talep ne kadar acil?",
        "identical_texts": ("orta", "bir talep"),
        "levels": ("Acil değil", "Yakında", "İş durdu"),
    },
}

# Gates. Set from the shipped checkpoints' leave-one-out ranges (README.md, "Thresholds").
SLOT0_MIN = -0.20
FIRST_SLOT_MIN = 0.15
PARITY_TOL = 1e-3

ScoreFn = Callable[[str, Sequence[Dict[str, Any]]], List[Any]]


def identical_question(text: str, k: int, instructions: str = INSTRUCTIONS) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": [text] * k}


def choice_identical_question(text: str, k: int, instructions: str = INSTRUCTIONS) -> Dict[str, Any]:
    """Numbered keys with one shared description: `1: <text>`, `2: <text>`, ... (#602).

    Choice keys must be unique, so the options cannot be identical outright; a numbered key is
    the closest counterpart of the `level N:` prefix the score control carries.
    """
    return {"type": "choice", "instructions": instructions,
            "criteria": {str(i + 1): text for i in range(k)}}


def permuted_questions(levels: Sequence[str] = LEVELS,
                       instructions: str = INSTRUCTIONS) -> List[Tuple[Tuple[int, ...], Dict[str, Any]]]:
    """Every order of `levels`, as (order, question); order[slot] indexes `levels`."""
    return [(order, {"type": "score", "instructions": instructions,
                     "criteria": [levels[i] for i in order]})
            for order in itertools.permutations(range(len(levels)))]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def leave_one_out(per_state: Sequence[float]) -> Tuple[float, float]:
    """(min, max) of the metric recomputed with each state left out in turn."""
    n = len(per_state)
    total = sum(per_state)
    loo = [(total - x) / (n - 1) for x in per_state]
    return min(loo), max(loo)


def passes(value: float, threshold: float) -> bool:
    return value >= threshold


def check_score_slot0_identical(score_fn: ScoreFn, states: Optional[Sequence[str]] = None,
                                lang: str = "en") -> Dict[str, Any]:
    spec = LANGUAGES[lang]
    states = spec["states"] if states is None else states
    configs = [(text, k) for text in spec["identical_texts"] for k in IDENTICAL_KS]
    questions = [identical_question(text, k, spec["instructions"]) for text, k in configs]
    per_state: List[float] = []
    per_config: Dict[str, List[float]] = {"%s/K=%d" % c: [] for c in configs}
    per_slot: Dict[str, List[float]] = {"%s/K=%d" % (t, k): [0.0] * k for t, k in configs}
    for state in states:
        slot0 = []
        for (text, k), z in zip(configs, score_fn(state, questions)):
            name = "%s/K=%d" % (text, k)
            z = [float(v) for v in z]  # numpy float32 would survive into the JSON report
            centred = [v - mean(z) for v in z]
            slot0.append(centred[0])
            per_config[name].append(centred[0])
            per_slot[name] = [a + b / len(states) for a, b in zip(per_slot[name], centred)]
        per_state.append(mean(slot0))
    metric = mean(per_state)
    lo, hi = leave_one_out(per_state)
    return {
        "metric": round(metric, 4),
        "threshold": SLOT0_MIN,
        "passed": passes(metric, SLOT0_MIN),
        "leave_one_out": [round(lo, 4), round(hi, 4)],
        "per_state": [round(x, 4) for x in per_state],
        "per_config": {name: round(mean(v), 4) for name, v in per_config.items()},
        "per_slot_centred": {name: [round(x, 4) for x in v] for name, v in per_slot.items()},
    }


def check_score_first_slot_permuted(score_fn: ScoreFn, states: Optional[Sequence[str]] = None,
                                    lang: str = "en") -> Dict[str, Any]:
    spec = LANGUAGES[lang]
    states = spec["states"] if states is None else states
    levels = spec["levels"]
    design = permuted_questions(levels, spec["instructions"])
    questions = [q for _order, q in design]
    by_slot = [0] * len(levels)
    by_label = {label: 0 for label in levels}
    per_state: List[float] = []
    order_invariant = 0
    for state in states:
        picked = []
        for (order, _q), z in zip(design, score_fn(state, questions)):
            slot = max(range(len(z)), key=lambda i: float(z[i]))
            by_slot[slot] += 1
            by_label[levels[order[slot]]] += 1
            picked.append((slot, levels[order[slot]]))
        per_state.append(sum(1 for slot, _ in picked if slot == 0) / len(design))
        order_invariant += len({label for _, label in picked}) == 1
    metric = mean(per_state)
    lo, hi = leave_one_out(per_state)
    return {
        "metric": round(metric, 4),
        "threshold": FIRST_SLOT_MIN,
        "passed": passes(metric, FIRST_SLOT_MIN),
        "leave_one_out": [round(lo, 4), round(hi, 4)],
        "decisions": len(states) * len(design),
        "per_state": [round(x, 4) for x in per_state],
        "argmax_by_slot": by_slot,
        "picks_by_label": by_label,
        "order_invariant_states": order_invariant,
    }


def check_choice_slot0_identical(score_fn: ScoreFn, states: Optional[Sequence[str]] = None,
                                 lang: str = "en") -> Dict[str, Any]:
    """The score control's counterpart for `choice` (#602, part a).

    One choice question whose options are `1: <text>`, `2: <text>`, ...: they differ only by
    position and the numbered key. The metric is slot 0's raw marker logit minus the mean over
    the options, averaged over states and texts, at the first entry of CHOICE_KS; the others are
    reported beside it and not gated. Same gate as the score control.
    """
    spec = LANGUAGES[lang]
    states = spec["states"] if states is None else states
    configs = [(text, k) for text in spec["identical_texts"] for k in CHOICE_KS]
    questions = [choice_identical_question(text, k, spec["instructions"]) for text, k in configs]
    by_k: Dict[int, List[float]] = {k: [] for k in CHOICE_KS}
    per_config: Dict[str, List[float]] = {"%s/K=%d" % c: [] for c in configs}
    for state in states:
        slot0: Dict[int, List[float]] = {k: [] for k in CHOICE_KS}
        for (text, k), z in zip(configs, score_fn(state, questions)):
            z = [float(v) for v in z]
            centred = z[0] - mean(z)
            slot0[k].append(centred)
            per_config["%s/K=%d" % (text, k)].append(centred)
        for k in CHOICE_KS:
            by_k[k].append(mean(slot0[k]))
    gated = by_k[CHOICE_KS[0]]
    metric = mean(gated)
    lo, hi = leave_one_out(gated)
    return {
        "metric": round(metric, 4),
        "threshold": SLOT0_MIN,
        "passed": passes(metric, SLOT0_MIN),
        "leave_one_out": [round(lo, 4), round(hi, 4)],
        "k": CHOICE_KS[0],
        "by_k": {str(k): round(mean(v), 4) for k, v in by_k.items()},
        "per_state": [round(x, 4) for x in gated],
        "per_config": {name: round(mean(v), 4) for name, v in per_config.items()},
    }


CHECKS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "score_slot0_identical": check_score_slot0_identical,
    "score_first_slot_permuted": check_score_first_slot_permuted,
}
# Checks outside the default English run, so its report is unchanged. A `--lang` run adds them;
# `--checks` can name them anywhere.
CHOICE_CHECKS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "choice_slot0_identical": check_choice_slot0_identical,
}
ALL_CHECKS: Dict[str, Callable[..., Dict[str, Any]]] = {**CHECKS, **CHOICE_CHECKS}


def run_checks(score_fn: ScoreFn, names: Optional[Sequence[str]] = None,
               states: Optional[Sequence[str]] = None, lang: str = "en") -> Dict[str, Any]:
    names = list(names or CHECKS)
    results = {name: ALL_CHECKS[name](score_fn, states, lang) for name in names}
    return {"checks": results, "passed": all(r["passed"] for r in results.values())}


def parse_langs(values: Optional[Sequence[str]]) -> List[str]:
    """`--lang ja --lang ko` or `--lang ja,ko`, in order, without repeats. None means English only.

    Raises ValueError naming any code that has no fixed states here.
    """
    if not values:
        return ["en"]
    langs = list(dict.fromkeys(x.strip().lower() for v in values for x in v.split(",") if x.strip()))
    unknown = [x for x in langs if x not in LANGUAGES]
    if unknown or not langs:
        raise ValueError("unknown language %s; fixed states exist for: %s"
                         % (", ".join(unknown) or "(none given)", ", ".join(LANGUAGES)))
    return langs


def routes(states: Sequence[str]) -> Dict[str, int]:
    """Which checkpoint `Router` would pick for each state, counted. Routing only, no weights."""
    from laya.router import Router

    router = Router()
    counts: Dict[str, int] = {}
    for state in states:
        model = router.route(state)["model"]
        counts[model] = counts.get(model, 0) + 1
    return counts


def exit_code(report: Dict[str, Any], parity: float) -> int:
    if not parity <= PARITY_TOL:
        return 2
    return 0 if report["passed"] else 1


def agent_score_fn(agent) -> ScoreFn:
    """Raw marker logits for several questions on one state, one forward pass."""
    def score(state, questions):
        return score_cases(agent, [(state, {str(i): q for i, q in enumerate(questions)})])
    return score


def parity(agent, states: Optional[Sequence[str]] = None, lang: str = "en") -> float:
    """Max |p| difference between this harness and Agent.system_one.

    system_one rounds probabilities to 4 decimals, so agreement shows up as <= 5e-5.
    """
    from laya.common import QTYPES

    spec = LANGUAGES[lang]
    states = spec["states"] if states is None else states
    questions = {"identical": identical_question(spec["identical_texts"][0], 3, spec["instructions"]),
                 "levels": {"type": "score", "instructions": spec["instructions"],
                            "criteria": list(spec["levels"])}}
    score = agent_score_fn(agent)
    worst = 0.0
    for state in states:
        public = agent.system_one(state, questions)["answers"]
        for (qid, _q), z in zip(questions.items(), score(state, list(questions.values()))):
            p = softmax_t(z, temperature_for(agent, QTYPES["score"], len(z)))
            got = [public[qid]["probabilities"][str(i)] for i in range(len(z))]
            worst = max(worst, max(abs(a - float(b)) for a, b in zip(got, p)))
    return worst


def choice_parity(agent, states: Optional[Sequence[str]] = None, lang: str = "en") -> float:
    """`parity` for the choice control's questions, keyed by option name as system_one reports them."""
    from laya.common import QTYPES

    spec = LANGUAGES[lang]
    states = spec["states"] if states is None else states
    questions = {"k%d" % k: choice_identical_question(spec["identical_texts"][0], k, spec["instructions"])
                 for k in CHOICE_KS}
    score = agent_score_fn(agent)
    worst = 0.0
    for state in states:
        public = agent.system_one(state, questions)["answers"]
        for (qid, q), z in zip(questions.items(), score(state, list(questions.values()))):
            p = softmax_t(z, temperature_for(agent, QTYPES["choice"], len(z)))
            got = [public[qid]["probabilities"][key] for key in q["criteria"]]
            worst = max(worst, max(abs(a - float(b)) for a, b in zip(got, p)))
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="presentation-checks",
        description="Label-free presentation-sensitivity checks for a Laya checkpoint.")
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="checkpoint repo id or local path")
    parser.add_argument("--subfolder", default=None,
                        help="checkpoint subfolder, e.g. multilingual")
    parser.add_argument("--device", default="cpu",
                        help="default cpu: fp32 and deterministic, which is what the thresholds were set on")
    parser.add_argument("--checks", default=None,
                        help="comma-separated subset of: %s. Default: %s; with --lang, all of them"
                             % (", ".join(ALL_CHECKS), ", ".join(CHECKS)))
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--lang", action="append", default=None,
                        help="fixed-state language: %s; repeat or comma-separate (--lang ja,ko). "
                             "Without it the run is English and the report is unchanged" % ", ".join(LANGUAGES))
    args = parser.parse_args(argv)

    if args.checks is None:
        names = list(CHECKS) if args.lang is None else list(ALL_CHECKS)
    else:
        names = [x.strip() for x in args.checks.split(",") if x.strip()]
    unknown = [x for x in names if x not in ALL_CHECKS]
    if unknown:
        print("unknown checks: %s" % ", ".join(unknown), file=sys.stderr)
        return 2
    try:
        langs = parse_langs(args.lang)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    import laya

    started = time.time()
    agent = laya.load(args.model, device=args.device, subfolder=args.subfolder)
    agent.model.eval()
    if args.lang is None:
        worst = parity(agent)
        report = run_checks(agent_score_fn(agent), names)
        payload = {
            "config": {
                "model": args.model,
                "subfolder": args.subfolder,
                "device": str(agent.device),
                "states": len(STATES),
                "instructions": INSTRUCTIONS,
                "identical_texts": list(IDENTICAL_TEXTS),
                "identical_ks": list(IDENTICAL_KS),
                "levels": list(LEVELS),
                "laya_version": getattr(laya, "__version__", "unknown"),
            },
            "parity_max_abs_diff": worst,
            **report,
            "seconds": round(time.time() - started, 1),
        }
        print_report(report, worst)
    else:
        by_lang: Dict[str, Any] = {}
        for lang in langs:
            spec = LANGUAGES[lang]
            routed = routes(spec["states"])
            lang_worst = parity(agent, lang=lang)
            extra: Dict[str, Any] = {}
            if any(name in CHOICE_CHECKS for name in names):
                extra = {"parity_score_max_abs_diff": lang_worst,
                         "parity_choice_max_abs_diff": choice_parity(agent, lang=lang),
                         "choice_ks": list(CHOICE_KS)}
                lang_worst = max(lang_worst, extra["parity_choice_max_abs_diff"])
            lang_report = run_checks(agent_score_fn(agent), names, lang=lang)
            by_lang[lang] = {
                "config": {"states": len(spec["states"]), "instructions": spec["instructions"],
                           "identical_texts": list(spec["identical_texts"]),
                           "identical_ks": list(IDENTICAL_KS), "levels": list(spec["levels"]),
                           "router_picks": routed},
                "parity_max_abs_diff": lang_worst,
                **extra,
                **lang_report,
            }
            print("== %s: Router picks %s" % (lang, ", ".join(
                "%s for %d/%d states" % (m, n, len(spec["states"])) for m, n in sorted(routed.items()))))
            print_report(lang_report, lang_worst)
        worst = max(r["parity_max_abs_diff"] for r in by_lang.values())
        report = {"passed": all(r["passed"] for r in by_lang.values())}
        payload = {
            "config": {"model": args.model, "subfolder": args.subfolder, "device": str(agent.device),
                       "langs": langs, "laya_version": getattr(laya, "__version__", "unknown")},
            "languages": by_lang,
            "parity_max_abs_diff": worst,
            **report,
            "seconds": round(time.time() - started, 1),
        }

    code = exit_code(report, worst)
    print("  verdict: %s" % {0: "PASS", 1: "FAIL", 2: "HARNESS MISMATCH"}[code])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
    return code


def print_report(report: Dict[str, Any], worst: float) -> None:
    for name, r in report["checks"].items():
        print("  %-26s %8.4f  >= %5.2f  %s   (leave-one-out %.4f .. %.4f)"
              % (name, r["metric"], r["threshold"], "PASS" if r["passed"] else "FAIL",
                 r["leave_one_out"][0], r["leave_one_out"][1]))
        if "by_k" in r:
            print("  %-26s gated at K=%d; by K: %s" % ("", r["k"], ", ".join(
                "%s %.4f" % (k, v) for k, v in r["by_k"].items())))
    print("  parity vs Agent.system_one: max |dp| %.2e (tolerance %.0e)" % (worst, PARITY_TOL))


if __name__ == "__main__":                      # pragma: no cover
    sys.exit(main())
