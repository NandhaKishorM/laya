"""The frozen question every case is asked, as a small ladder of renderings.

Written before any model was run. A rung changes HOW the same six-way question is asked,
never which answer is right.
"""

CRITERIA_ES = {
    "ventas": "quiere comprar, contratar, mejorar su plan o pedir una cotización",
    "soporte": "algo no funciona o está fallando y hay que arreglarlo",
    "facturacion": "un cobro, un pago, una factura, un saldo o un reembolso",
    "operadora": "pide hablar con una persona, sin decir con qué área",
    "colgar": "se despide o dice que ya no necesita nada más",
    "ninguno": "saluda, pregunta algo, o pide que NO lo transfieran ni le cuelguen",
}

CRITERIA_EN = {
    "ventas": "wants to buy, contract, upgrade a plan or get a quote",
    "soporte": "something is broken or failing and needs to be fixed",
    "facturacion": "a charge, a payment, an invoice, a balance or a refund",
    "operadora": "asks to talk to a person, without naming an area",
    "colgar": "says goodbye or that nothing else is needed",
    "ninguno": "greets, asks a question, or asks NOT to be transferred or hung up on",
}

CLINIC_CRITERIA_ES = {
    "citas": "quiere sacar, cambiar o cancelar una consulta médica",
    "laboratorio": "exámenes, análisis y sus resultados",
    "farmacia": "medicamentos y recetas",
    "emergencias": "alguien está en peligro y necesita atención inmediata",
    "recepcion": "pide hablar con una persona, sin decir con qué área",
    "colgar": "se despide o dice que ya no necesita nada más",
    "ninguno": "saluda, pregunta algo, o pide que NO lo transfieran ni le cuelguen",
}

CLINIC_CRITERIA_EN = {
    "citas": "wants to book, change or cancel a medical appointment",
    "laboratorio": "lab tests and their results",
    "farmacia": "medicines and prescriptions",
    "emergencias": "someone is in danger and needs immediate care",
    "recepcion": "asks to talk to a person, without naming an area",
    "colgar": "says goodbye or that nothing else is needed",
    "ninguno": "greets, asks a question, or asks NOT to be transferred or hung up on",
}

MENUS = {
    "cases": {"es": CRITERIA_ES, "en": CRITERIA_EN},
    "clinic": {"es": CLINIC_CRITERIA_ES, "en": CLINIC_CRITERIA_EN},
    # The same two menus over sentences the author of the first two sets did not write.
    "independent_cases": {"es": CRITERIA_ES, "en": CRITERIA_EN},
    "independent_clinic": {"es": CLINIC_CRITERIA_ES, "en": CLINIC_CRITERIA_EN},
    # An angry caller, written by a native speaker. The telephone-company menu.
    "frustration": {"es": CRITERIA_ES, "en": CRITERIA_EN},
}

INSTRUCTIONS_ES = "¿Qué debe hacer el agente con esta llamada?"
INSTRUCTIONS_EN = "What should the agent do with this call?"

SCENARIO_ES = (
    "Una persona llama por teléfono a una empresa y la atiende un agente de voz. "
    "El agente puede seguir conversando, pasar la llamada a un área o terminarla."
)

RUNGS = {
    # Spanish question, Spanish option descriptions, bare utterance.
    "es_criteria": {"instructions": INSTRUCTIONS_ES, "language": "es", "scenario": None},
    # The same, with one sentence that says who is talking to whom.
    "es_scenario": {"instructions": INSTRUCTIONS_ES, "language": "es", "scenario": SCENARIO_ES},
    # English question and descriptions over the Spanish utterance: what an integrator
    # who does not write Spanish would ship.
    "en_criteria": {"instructions": INSTRUCTIONS_EN, "language": "en", "scenario": None},
}


def labels(dataset: str):
    return list(MENUS[dataset]["es"])


def build(rung: str, text: str, dataset: str = "cases"):
    """Returns (state, questions) for one case at one rung."""
    spec = RUNGS[rung]
    state = text if spec["scenario"] is None else f"{spec['scenario']}\n\nLa persona dice: {text}"
    questions = {
        "accion": {
            "type": "choice",
            "instructions": spec["instructions"],
            "criteria": dict(MENUS[dataset][spec["language"]]),
        }
    }
    return state, questions
