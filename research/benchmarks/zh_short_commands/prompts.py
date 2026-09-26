"""Frozen prompts and label policy for the Chinese short-command benchmark.

Nothing in this file may change once a results archive is produced: run.py records
this file's hash and audit.py re-checks it, so a published number can always be
re-derived from the archive alone.

Motivation: #218 reported ~50% accuracy for `laya-multilingual` on 20 short Chinese
voice commands, and that following the documented guidance (adding criteria, a
scenario description, a structured JSON state) made discrimination *worse* — outputs
flattened into the 0.6–1.0 range. That report is not reproducible from the repository
(the prompt text was never published, and no per-case record exists), so this
benchmark freezes the inputs and records every decision. It reproduces the *shape* of
that comparison, not its exact prompts.

The ablation is a ladder over both task shapes, so the guidance claim is tested where
it was made (a single `noul` dimension) and where routing actually happens (six-way
`choice`):

    task B (noul)    noul_plain -> noul_criteria -> noul_scenario -> noul_json_state
    task A (choice)                choice_criteria -> choice_scenario -> choice_json_state

A `choice` question cannot exist without criteria — the criteria keys *are* its
options — so task A starts one rung higher.
"""

LABELS = ("faster", "slower", "stop", "left", "right", "none")

# --------------------------------------------------------------- choice (task A)
CHOICE_INSTRUCTIONS = "这条指令要求机器人做什么？"

CHOICE_CRITERIA = {
    "faster": "要求加快速度",
    "slower": "要求放慢速度",
    "stop": "要求停止运动",
    "left": "要求向左转向",
    "right": "要求向右转向",
    "none": "不是对机器人的运动指令",
}

SCENARIO_PREFIX = "你是一台清洁机器人的控制模块，用户通过语音发出简短指令。"

# ---------------------------------------------------------------- noul (task B)
# Four independent yes/no dimensions. `noul_plain` omits criteria on purpose, so the
# checkpoint reads its default `false:` / `true:` slots — the "plain string, no
# criteria, single question" shape #218 measured. The known label sensitivity of
# `noul` (README, "Honest limits") applies to that rung and is why task A exists.
NOUL_DIMENSIONS = ("wants_faster", "wants_slower", "wants_stop", "is_command")

NOUL_INSTRUCTIONS = {
    "wants_faster": "这条指令是否要求加快速度？",
    "wants_slower": "这条指令是否要求放慢速度？",
    "wants_stop": "这条指令是否要求完全停止？",
    "is_command": "这条指令是否是对机器人的运动指令？",
}

# The criteria a `noul` carries once the ladder adds them. These two keys ARE the
# option text the model reads (docs, "Decision Primitives"), so they spell out the
# meaning rather than leaning on the default `false:` / `true:` labels.
NOUL_CRITERIA = {
    "wants_faster": {"true": "要求加快速度", "false": "不要求加快速度"},
    "wants_slower": {"true": "要求放慢速度", "false": "不要求放慢速度"},
    "wants_stop": {"true": "要求完全停止", "false": "不要求完全停止"},
    "is_command": {"true": "是对机器人的运动指令", "false": "不是对机器人的运动指令"},
}

# The gold boolean each dimension should return, derived from the case's gold label.
NOUL_GOLD = {
    "wants_faster": lambda gold: gold == "faster",
    "wants_slower": lambda gold: gold == "slower",
    "wants_stop": lambda gold: gold == "stop",
    "is_command": lambda gold: gold != "none",
}

# ------------------------------------------------------------------- ladder axes
CHOICE_CONFIGS = ("choice_criteria", "choice_scenario", "choice_json_state")
NOUL_CONFIGS = ("noul_plain", "noul_criteria", "noul_scenario", "noul_json_state")
CONFIGS = CHOICE_CONFIGS + NOUL_CONFIGS

_SCENARIO_CONFIGS = ("choice_scenario", "noul_scenario")
_JSON_STATE_CONFIGS = ("choice_json_state", "noul_json_state")
_NO_CRITERIA_CONFIGS = ("noul_plain",)


def _instructions(base: str, config: str) -> str:
    if config not in CONFIGS:
        raise ValueError("unknown config: %r" % config)
    return (SCENARIO_PREFIX + base) if config in _SCENARIO_CONFIGS else base


def choice_question(config: str) -> dict:
    """The single six-option choice question a config asks."""
    if config not in CHOICE_CONFIGS:
        raise ValueError("not a choice config: %r" % config)
    return {"intent": {"type": "choice",
                       "instructions": _instructions(CHOICE_INSTRUCTIONS, config),
                       "criteria": dict(CHOICE_CRITERIA)}}


def noul_questions(config: str) -> dict:
    """The four dimension questions, with criteria wherever the ladder has them."""
    if config not in NOUL_CONFIGS:
        raise ValueError("not a noul config: %r" % config)
    questions = {}
    for dim in NOUL_DIMENSIONS:
        qdef = {"type": "noul", "instructions": _instructions(NOUL_INSTRUCTIONS[dim], config)}
        if config not in _NO_CRITERIA_CONFIGS:
            qdef["criteria"] = dict(NOUL_CRITERIA[dim])
        questions[dim] = qdef
    return questions


def questions_for(config: str) -> dict:
    return choice_question(config) if config in CHOICE_CONFIGS else noul_questions(config)


def state_for(config: str, text: str):
    """The state the config feeds the model; the json rungs wrap it in a dict."""
    if config not in CONFIGS:
        raise ValueError("unknown config: %r" % config)
    if config in _JSON_STATE_CONFIGS:
        return {"text": text, "source": "voice", "device": "vacuum"}
    return text


def cases_for(config: str, texts):
    """`(state, questions)` pairs in the research/eval harness's own shape."""
    questions = questions_for(config)
    return [(state_for(config, text), questions) for text in texts]
