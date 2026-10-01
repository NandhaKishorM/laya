"""Builds the frozen Spanish phone-turn cases.

The label policy below was written BEFORE any model was run on these cases. The cases are
hand-written fixtures in Latin American Spanish, in the register a speech recogniser hands
to a voice agent. They are not a held-out corpus and not independently annotated.

Label policy
------------
The caller is on the phone with a voice agent that can do exactly three things: keep
talking, transfer the call to one of four destinations, or end the call.

* ``ventas``       the caller wants to BUY, contract, upgrade or get a quote.
* ``soporte``      something is broken or not working and needs fixing.
* ``facturacion``  a charge, payment, invoice, balance or refund.
* ``operadora``    the caller asks for a person, without naming an area.
* ``colgar``       the caller says goodbye or states nothing else is needed.
* ``ninguno``      anything else: greetings, questions the agent can answer by itself,
                   fragments, and every request that NEGATES a transfer or a hang-up.

A question about information (opening hours, address, how something works) is ``ninguno``:
the agent answers it. A negated request ("no me pase con ventas") is ``ninguno``: acting on
the named destination would be the exact mistake a phone agent must not make.
"""
import json
from pathlib import Path

LABELS = ["ventas", "soporte", "facturacion", "operadora", "colgar", "ninguno"]

# (text, gold, tags)
CASES = [
    # --- ventas: direct -------------------------------------------------------------------
    ("Quiero hablar con ventas, por favor.", "ventas", ["direct"]),
    ("Páseme con el departamento de ventas.", "ventas", ["direct"]),
    ("¿Me comunica con un asesor de ventas?", "ventas", ["direct"]),
    ("Con ventas, por favor.", "ventas", ["direct", "short"]),
    ("Necesito que me transfieran al área comercial.", "ventas", ["direct"]),
    # --- ventas: indirect -----------------------------------------------------------------
    ("Quiero contratar el servicio de internet para mi casa.", "ventas", ["indirect"]),
    ("Me interesa adquirir un plan nuevo para la oficina.", "ventas", ["indirect"]),
    ("Necesito una cotización para veinte líneas telefónicas.", "ventas", ["indirect"]),
    ("Fíjese que quiero cambiarme a un plan más grande, ¿con quién hablo?", "ventas", ["indirect", "hn"]),
    ("Vi la promoción en Facebook y quiero comprarla.", "ventas", ["indirect"]),
    ("Queremos poner el servicio en una sucursal nueva que vamos a abrir.", "ventas", ["indirect"]),
    ("Quisiera agregar otra línea a mi contrato.", "ventas", ["indirect"]),
    ("Ando buscando un plan empresarial, quiero que me lo vendan.", "ventas", ["indirect", "hn"]),
    ("Quiero subir la velocidad de mi plan, ¿quién me lo puede vender?", "ventas", ["indirect"]),
    # --- soporte: direct ------------------------------------------------------------------
    ("Comuníqueme con soporte técnico.", "soporte", ["direct"]),
    ("Necesito hablar con un técnico.", "soporte", ["direct"]),
    ("Páseme con soporte, por favor.", "soporte", ["direct"]),
    ("Con el área técnica, si me hace el favor.", "soporte", ["direct"]),
    ("Quiero reportar una falla con soporte.", "soporte", ["direct"]),
    # --- soporte: indirect ----------------------------------------------------------------
    ("No tengo internet desde ayer en la noche.", "soporte", ["indirect"]),
    ("El teléfono no da tono y no puedo hacer llamadas.", "soporte", ["indirect"]),
    ("Se me cayó el servicio y necesito que lo arreglen.", "soporte", ["indirect"]),
    ("El módem tiene una luz roja parpadeando y no navega.", "soporte", ["indirect"]),
    ("Fíjese que el internet está lentísimo, casi no carga nada.", "soporte", ["indirect", "hn"]),
    ("Las llamadas se cortan a cada rato, así no se puede trabajar.", "soporte", ["indirect"]),
    ("No me funciona el correo de la empresa desde la mañana.", "soporte", ["indirect"]),
    ("Se fue la señal en toda la oficina, necesitamos que vengan a revisar.", "soporte", ["indirect"]),
    ("La extensión de recepción no suena cuando le marcan.", "soporte", ["indirect"]),
    # --- facturacion: direct --------------------------------------------------------------
    ("Páseme con facturación.", "facturacion", ["direct"]),
    ("Necesito hablar con el departamento de cobros.", "facturacion", ["direct"]),
    ("Quiero hablar con alguien de facturación, por favor.", "facturacion", ["direct"]),
    ("Con cuentas por cobrar, por favor.", "facturacion", ["direct"]),
    ("Comuníqueme con el área de pagos.", "facturacion", ["direct"]),
    # --- facturacion: indirect ------------------------------------------------------------
    ("Me cobraron dos veces este mes y quiero que me devuelvan el dinero.", "facturacion", ["indirect"]),
    ("La factura me llegó con un monto que no es el que acordamos.", "facturacion", ["indirect"]),
    ("Ya pagué y me siguen cobrando el mismo recibo.", "facturacion", ["indirect"]),
    ("Necesito que me reenvíen la factura de agosto con el RTN de la empresa.", "facturacion", ["indirect", "hn"]),
    ("Quiero hacer un arreglo de pago porque estoy atrasado.", "facturacion", ["indirect"]),
    ("Me aparece un cargo que no reconozco en el estado de cuenta.", "facturacion", ["indirect"]),
    ("Hice la transferencia hace tres días y no me han aplicado el pago.", "facturacion", ["indirect"]),
    ("Quiero saber por qué me subieron el cobro sin avisarme, necesito que lo corrijan.", "facturacion", ["indirect"]),
    ("Pagué de más y quiero un reembolso.", "facturacion", ["indirect"]),
    # --- operadora: a person, no area -----------------------------------------------------
    ("Quiero hablar con una persona.", "operadora", ["human"]),
    ("Páseme con un humano, por favor.", "operadora", ["human"]),
    ("Necesito hablar con alguien de verdad, no con una máquina.", "operadora", ["human"]),
    ("¿Me puede pasar con una operadora?", "operadora", ["human", "direct"]),
    ("Comuníqueme con recepción.", "operadora", ["human", "direct"]),
    ("Prefiero que me atienda un agente.", "operadora", ["human"]),
    ("Quiero hablar con un representante.", "operadora", ["human"]),
    ("Con alguien, por favor, con quien sea.", "operadora", ["human", "short"]),
    ("Mire, mejor páseme con una persona que me pueda ayudar.", "operadora", ["human", "hn"]),
    ("No quiero hablar con un robot, páseme con alguien.", "operadora", ["human", "negation-in-text"]),
    ("¿Hay alguna persona que me pueda atender?", "operadora", ["human"]),
    ("Déjeme hablar con la secretaria.", "operadora", ["human"]),
    # --- colgar: farewell -----------------------------------------------------------------
    ("Eso era todo, muchas gracias, hasta luego.", "colgar", ["farewell"]),
    ("No, gracias, ya no necesito nada más. Adiós.", "colgar", ["farewell"]),
    ("Listo, muy amable, que tenga buen día.", "colgar", ["farewell"]),
    ("Vaya pues, gracias, hasta luego.", "colgar", ["farewell", "hn"]),
    ("Ya puede colgar, gracias.", "colgar", ["farewell", "direct"]),
    ("Perfecto, era solo eso. Buenas tardes.", "colgar", ["farewell"]),
    ("Bueno, le agradezco, adiós.", "colgar", ["farewell"]),
    ("Nada más, gracias. Chao.", "colgar", ["farewell", "short"]),
    ("Me equivoqué de número, disculpe, hasta luego.", "colgar", ["farewell"]),
    ("Ya me resolvieron, gracias, feliz tarde.", "colgar", ["farewell"]),
    ("Está bien, después llamo. Adiós.", "colgar", ["farewell"]),
    ("Cheque, gracias, eso sería todo.", "colgar", ["farewell", "hn"]),
    # --- ninguno: questions the agent answers ---------------------------------------------
    ("¿A qué hora abren mañana?", "ninguno", ["open"]),
    ("¿Dónde queda la oficina principal?", "ninguno", ["open"]),
    ("¿Hasta qué hora atienden los sábados?", "ninguno", ["open"]),
    ("¿Cuál es el correo para mandar documentos?", "ninguno", ["open"]),
    ("¿Ustedes tienen cobertura en San Pedro Sula?", "ninguno", ["open", "hn"]),
    ("¿Me puede repetir la dirección, por favor?", "ninguno", ["open"]),
    ("¿Qué documentos necesito llevar?", "ninguno", ["open"]),
    ("¿Trabajan los días feriados?", "ninguno", ["open"]),
    ("¿Cómo se llama usted?", "ninguno", ["open"]),
    ("¿Tienen parqueo para clientes?", "ninguno", ["open"]),
    # --- ninguno: greetings and fragments -------------------------------------------------
    ("Aló, buenos días.", "ninguno", ["noise"]),
    ("¿Aló? ¿Me escucha?", "ninguno", ["noise"]),
    ("Sí.", "ninguno", ["noise", "short"]),
    ("Este... espere un momento.", "ninguno", ["noise"]),
    ("Buenas tardes, mire, tengo una consulta.", "ninguno", ["noise", "hn"]),
    ("Un momentito, que estoy buscando el número de contrato.", "ninguno", ["noise"]),
    ("¿Cómo dijo? No le entendí.", "ninguno", ["noise"]),
    ("Ajá, sí, lo escucho.", "ninguno", ["noise"]),
    # --- ninguno: negated transfers -------------------------------------------------------
    ("No me pase con ventas, solo quiero saber el horario.", "ninguno", ["negation"]),
    ("No quiero hablar con soporte, ya hablé con ellos ayer.", "ninguno", ["negation"]),
    ("No necesito que me transfieran a facturación, solo una pregunta.", "ninguno", ["negation"]),
    ("Todavía no me pase con nadie, primero explíqueme usted.", "ninguno", ["negation"]),
    ("No, con la operadora no, prefiero que me lo diga usted.", "ninguno", ["negation"]),
    ("No es para ventas, es solo para saber dónde están ubicados.", "ninguno", ["negation"]),
    ("No tengo ninguna falla, el internet funciona bien, solo llamo por una duda.", "ninguno", ["negation"]),
    ("No tengo problema con el cobro, la factura está bien, era otra cosa.", "ninguno", ["negation"]),
    ("No quiero comprar nada, solo estoy preguntando por el horario.", "ninguno", ["negation"]),
    ("No me transfiera, por favor.", "ninguno", ["negation", "short"]),
    # --- ninguno: negated hang-ups --------------------------------------------------------
    ("No cuelgue, por favor, todavía tengo otra pregunta.", "ninguno", ["negation"]),
    ("Espere, no he terminado, no me corte.", "ninguno", ["negation"]),
    ("No, no es todo, me falta preguntarle otra cosa.", "ninguno", ["negation"]),
    ("Gracias, pero todavía no me despido, tengo una duda más.", "ninguno", ["negation"]),
    # --- hard: the named area is not the one that must act --------------------------------
    ("Ventas me prometió una velocidad y el internet no funciona.", "soporte", ["hard"]),
    ("Soporte ya vino, pero ahora me están cobrando la visita y no estoy de acuerdo.", "facturacion", ["hard"]),
    ("Hablé con facturación y me dijeron que para contratar otra línea hable con ustedes.", "ventas", ["hard"]),
    ("Ya hablé con ventas y con soporte y nadie me resuelve, quiero hablar con una persona.", "operadora", ["hard"]),
    ("Gracias por pasarme antes con soporte; ahora quiero pagar mi factura.", "facturacion", ["hard"]),
    ("El técnico arregló todo, solo quería agradecer. Hasta luego.", "colgar", ["hard"]),
    # --- recogniser-style text: no punctuation, fillers, repetitions ----------------------
    ("eh si buenas mire es que no tengo internet desde la mañana", "soporte", ["asr", "indirect"]),
    ("quiero este quiero contratar un plan para mi negocio", "ventas", ["asr", "indirect"]),
    ("me me cobraron doble la factura de este mes", "facturacion", ["asr", "indirect"]),
    ("paseme con una persona por favor", "operadora", ["asr", "human"]),
    ("bueno gracias eso era todo hasta luego", "colgar", ["asr", "farewell"]),
    ("a que hora cierran hoy", "ninguno", ["asr", "open"]),
    ("no no me pase con ventas solo una pregunta", "ninguno", ["asr", "negation"]),
    ("alo alo si me escucha", "ninguno", ["asr", "noise"]),
]

BORDERLINE = [
    # Reported apart and excluded from the headline: two readers could defend two labels.
    ("¿Cuánto cuesta el plan de cincuenta megas?", "ninguno", ["borderline"]),
    ("Quiero cancelar mi servicio.", "operadora", ["borderline"]),
    ("¿Qué promociones tienen este mes?", "ninguno", ["borderline"]),
    ("Necesito cambiar el titular de la cuenta.", "operadora", ["borderline"]),
]


# A second menu, from a different trade, with different destinations. It exists to tell
# "the model learned to read Spanish" apart from "the model learned this one menu": nothing
# in it shares a destination with the telephone-company menu above.
CLINIC_LABELS = ["citas", "laboratorio", "farmacia", "emergencias", "recepcion", "colgar", "ninguno"]

CLINIC_CASES = [
    # --- citas ----------------------------------------------------------------------------
    ("Quiero sacar una cita con el cardiólogo.", "citas", ["indirect"]),
    ("Necesito cambiar la fecha de mi consulta del jueves.", "citas", ["indirect"]),
    ("Páseme con citas, por favor.", "citas", ["direct"]),
    ("Fíjese que quiero apartar una consulta para mi mamá.", "citas", ["indirect", "hn"]),
    ("Tengo que cancelar la cita que tenía mañana con la doctora.", "citas", ["indirect"]),
    ("quiero agendar con el pediatra para la otra semana", "citas", ["asr", "indirect"]),
    # --- laboratorio ----------------------------------------------------------------------
    ("Quiero saber si ya están listos los resultados de mis exámenes de sangre.", "laboratorio", ["indirect"]),
    ("Comuníqueme con el laboratorio.", "laboratorio", ["direct"]),
    ("Me mandaron a hacerme unos análisis y quiero saber si tengo que ir en ayunas, páseme con ellos.", "laboratorio", ["indirect"]),
    ("Necesito que me reenvíen el resultado de la prueba que me hice el lunes.", "laboratorio", ["indirect"]),
    ("me hicieron un examen de orina y no me han entregado nada", "laboratorio", ["asr", "indirect"]),
    # --- farmacia -------------------------------------------------------------------------
    ("Quiero saber si tienen el medicamento que me recetaron, páseme con farmacia.", "farmacia", ["direct"]),
    ("Necesito surtir una receta.", "farmacia", ["indirect"]),
    ("Con la farmacia, por favor.", "farmacia", ["direct", "short"]),
    ("Me dieron unas pastillas equivocadas y quiero cambiarlas.", "farmacia", ["indirect"]),
    ("Quiero comprar la medicina que me indicó el doctor.", "farmacia", ["indirect"]),
    # --- emergencias ----------------------------------------------------------------------
    ("Mi papá se desmayó y no reacciona, necesito ayuda ya.", "emergencias", ["indirect"]),
    ("Páseme con emergencias, es urgente.", "emergencias", ["direct"]),
    ("Mi hijo se cortó y está sangrando mucho.", "emergencias", ["indirect"]),
    ("Tengo un dolor muy fuerte en el pecho y me cuesta respirar.", "emergencias", ["indirect"]),
    ("necesito una ambulancia mi esposa se cayó y no se puede mover", "emergencias", ["asr", "indirect"]),
    # --- recepcion: a person, no area -----------------------------------------------------
    ("Quiero hablar con una persona.", "recepcion", ["human"]),
    ("Páseme con alguien que me atienda, por favor.", "recepcion", ["human"]),
    ("No quiero hablar con una máquina, comuníqueme con una persona.", "recepcion", ["human", "negation-in-text"]),
    ("Con la recepcionista, por favor.", "recepcion", ["human", "direct"]),
    # --- colgar ---------------------------------------------------------------------------
    ("Muchas gracias, eso era todo. Hasta luego.", "colgar", ["farewell"]),
    ("Vaya pues, muy amable, adiós.", "colgar", ["farewell", "hn"]),
    ("Ya no necesito nada más, gracias.", "colgar", ["farewell"]),
    ("bueno gracias que pase buen dia", "colgar", ["asr", "farewell"]),
    # --- ninguno: questions, greetings ----------------------------------------------------
    ("¿A qué hora abre la clínica?", "ninguno", ["open"]),
    ("¿Dónde están ubicados?", "ninguno", ["open"]),
    ("¿Atienden los domingos?", "ninguno", ["open"]),
    ("Aló, buenas tardes.", "ninguno", ["noise"]),
    ("¿Me escucha?", "ninguno", ["noise", "short"]),
    ("Un momento, que estoy buscando mi carné.", "ninguno", ["noise"]),
    # --- ninguno: negations ---------------------------------------------------------------
    ("No quiero una cita, solo quiero saber el horario.", "ninguno", ["negation"]),
    ("No me pase con el laboratorio, ya tengo mis resultados.", "ninguno", ["negation"]),
    ("No es una emergencia, solo es una pregunta.", "ninguno", ["negation"]),
    ("No necesito la farmacia, solo quería preguntar dónde quedan.", "ninguno", ["negation"]),
    ("No cuelgue, todavía tengo otra duda.", "ninguno", ["negation"]),
    ("No me transfiera con nadie todavía, primero dígame usted.", "ninguno", ["negation"]),
    ("no no es urgente no me pase con emergencias", "ninguno", ["asr", "negation"]),
    # --- hard -----------------------------------------------------------------------------
    ("En el laboratorio me dijeron que para el examen necesito una cita, quiero sacarla.", "citas", ["hard"]),
    ("El doctor de emergencias me dejó una receta y quiero comprarla.", "farmacia", ["hard"]),
    ("Ya saqué la cita, ahora quiero saber si están mis resultados.", "laboratorio", ["hard"]),
]


# A third and fourth set, over the same two menus, whose SENTENCES were not written by the
# author of the first two. A different language model wrote them (glm-4.7-flash, not the
# one that wrote the training data) from briefs about negation; they were then read one by
# one, labelled by hand under the policy at the top of this file, and the ones two readers
# could label differently were dropped. 96 were written, the ones below were kept. The
# labels were fixed before any model was run on them.
#
# Tags: `refused` names an area and turns it down; `redirected` turns one area down and
# asks for another; `need-with-no` states a real need using a negation.
INDEPENDENT_CASES = [
    ("No, ni me meta al área de ventas que yo estoy llamando por una factura.", "facturacion", ["redirected"]),
    ("Soy cliente, no me conecte con los de contratos, necesito hablar con el soporte técnico.", "soporte", ["redirected"]),
    ("Quiero hablar con alguien del área de cobros, no me saque al departamento comercial.", "facturacion", ["redirected"]),
    ("Escúchame bien, no me mande con los de ventas, estoy llamando sobre mi saldo.", "facturacion", ["redirected"]),
    ("Oiga, mira que no me funciona la internet hace tres días", "soporte", ["need-with-no"]),
    ("Holaaa, no me llegó la factura y necesito hablar con alguien", "facturacion", ["need-with-no"]),
    ("Oiga, yo no quiero hablar con nadie, no me pasen al departamento de soporte técnico, llámame directamente.", "ninguno", ["refused"]),
    ("No me pongan con soporte, ya sé lo que hace, ¿podés darme mi número?", "ninguno", ["refused"]),
    ("No me manden al área de soporte, quiero que me respondan acá, al cliente.", "ninguno", ["refused"]),
    ("Mire, no me pongan con el soporte, eso ya lo saben, yo les hablo a ustedes.", "ninguno", ["refused"]),
    ("No me pasen con soporte técnico, llámeme directo al celular, no quiero hablar con nadie más.", "ninguno", ["refused"]),
    ("Oye, ¿saben por qué nunca me funciona el internet?", "soporte", ["need-with-no"]),
    ("Bueno, tengo un problema que no me deja usar la conexión.", "soporte", ["need-with-no"]),
    ("¿Alguien me ayuda con el servicio porque no funciona?", "soporte", ["need-with-no"]),
    ("No me funcionó el equipo que me enviaron, por favor arreglenlo.", "soporte", ["need-with-no"]),
    ("No quiero pasar al de facturación, haganme la atención aquí.", "ninguno", ["refused"]),
    ("Cuidado, no me pasen a facturación, es muy largo.", "ninguno", ["refused"]),
    ("No me manden a facturación, por favor.", "ninguno", ["refused"]),
    ("Ni se les ocurra pasarme a facturación, no estoy interesada.", "ninguno", ["refused"]),
    ("Oiga, estoy llamando por un asunto serio, me está cobrando una mensualidad y no me deja ver la factura ni el detalle del pago.", "facturacion", ["need-with-no"]),
    ("Hola, buenos días, tengo el saldo pendiente y no me está llegando el cobro, necesitan revisar el caso urgente.", "facturacion", ["need-with-no"]),
    ("Buenas tardes, disculpen que llame, me hicieron un cobro adicional y no me aparece la factura para validar el pago.", "facturacion", ["need-with-no"]),
    ("Saludos, llamo para reclamar el saldo que me deben y no puedo generar una orden de pago ni ver el estado del reembolso.", "facturacion", ["need-with-no"]),
    ("¿Sigo en línea?", "ninguno", ["noise"]),
    ("No, espera, no me cuelgues, me falta algo.", "ninguno", ["refused"]),
    ("No me corten la línea, no se vayan.", "ninguno", ["refused"]),
    ("Ya, no me cuelgues, sigo aquí esperando.", "ninguno", ["refused"]),
    ("Me quedé en pausa, no me vayan a cortar.", "ninguno", ["refused"]),
    ("Quiero hablar con el humano, por favor, no quiero hablar con la grabación.", "operadora", ["need-with-no"]),
    ("¿No hay nadie ahí? No quiero hablar con la grabación.", "operadora", ["need-with-no"]),
    ("Oiga, quiero hablar con alguien en vivo, no quiero hablar con la grabación.", "operadora", ["need-with-no"]),
]

INDEPENDENT_CLINIC = [
    ("No, no, no me pasen a esa caja ni al departamento de turnos.", "ninguno", ["refused"]),
    ("Hola, quería ver si se puede arreglar mi cita porque no puedo ir a la clínica", "citas", ["need-with-no"]),
    ("Buenas tardes, necesito saber si hago trámite para cambiar o anular la cita ya que nunca me llegó la confirmación", "citas", ["need-with-no"]),
    ("Quisiera pedir ayuda porque no puedo ir a mi consulta médica y quiero cancelarla o cambiarla", "citas", ["need-with-no"]),
    ("Oiga, mira, por favor, no la pasen para ahí, a lo del laboratorio, no quiero que me hagan análisis de nada, ni nada de eso.", "ninguno", ["refused"]),
    ("Soy yo, no me pasen al área de laboratorio, por favor, no quiero exámenes ni nada.", "ninguno", ["refused"]),
    ("Necesito que revisen los resultados, por favor, ya no me llegó la información que me mandaron.", "laboratorio", ["need-with-no"]),
    ("Oigan, urgente: me dijeron que me envíen los exámenes pero no me funciona el sistema.", "laboratorio", ["need-with-no"]),
    ("¿Alguien ahí me puede llamar para explicarme los resultados? No me puedo mover para ir.", "laboratorio", ["need-with-no"]),
    ("Quiero que me atiendan con mis exámenes, no me funciona lo que me enviaron ayer.", "laboratorio", ["need-with-no"]),
    ("Por favor, necesitan revisar mis análisis, nunca me han dado el resultado que me corresponde.", "laboratorio", ["need-with-no"]),
    ("Quiero hablar con atención humana, no con la farmacia, por favor.", "recepcion", ["redirected"]),
    ("Bueno, hola, soy la señora Rodríguez, tengo una inquietud con el medicamento que me dejaron recetado, no me llegó.", "farmacia", ["need-with-no"]),
    ("Bueno, hola, soy el señor de la mesa cuatro, me dejaron recetar unas cosas pero nunca me las dieron.", "farmacia", ["need-with-no"]),
    ("Oiga, ¿me pueden pasar con el médico de urgencias?", "emergencias", ["direct"]),
    ("Escúcheme, yo no quiero ser atendido en la emergencia.", "ninguno", ["refused"]),
    ("Quiero consulta pero no quiero que me pongan en la sala de emergencias.", "citas", ["redirected"]),
    ("Ayúdeme rápido, por favor, no puedo moverme de la cama", "emergencias", ["need-with-no"]),
    ("Hable conmigo, no quiero hablar con la grabación.", "recepcion", ["need-with-no"]),
    ("No quiero hablar con el sistema, quiero hablar con el humano.", "recepcion", ["need-with-no"]),
]


# Written by a Honduran Spanish speaker who is neither the author of the sets above nor a
# language model, asked for "sentences with a negation". What came back is a kind none of
# the other sets has: the caller is angry AT THE AGENT and insists. Kept exactly as typed,
# which is also how a recogniser would hand them over.
#
# Under the policy at the top none of them asks for a transfer or to end the call, so the
# answer is `ninguno`: the agent must not act. Whether an angry caller should be offered a
# person is a product decision the policy does not make, and a different question.
FRUSTRATION_CASES = [
    ("te estoy pidiendo que no", "ninguno", ["frustration", "short"]),
    ("te dije que no", "ninguno", ["frustration", "short"]),
    ("no estas entendiendo?", "ninguno", ["frustration", "short"]),
    ("maldita sea te dije que no!", "ninguno", ["frustration", "swearing"]),
    ("carajo no entiendo porque repetis lo mismo", "ninguno", ["frustration", "swearing"]),
    ("que tonteras decis", "ninguno", ["frustration"]),
    ("joder esto no sirve", "ninguno", ["frustration", "swearing"]),
    ("no me estas respondiendo lo que te pregunte", "ninguno", ["frustration"]),
    ("eres demasiado tonto", "ninguno", ["frustration", "insult"]),
]


def write(here: Path, name: str, prefix: str, cases, labels) -> None:
    rows = []
    for index, (text, gold, tags) in enumerate(cases, start=1):
        assert gold in labels, (text, gold)
        rows.append({"id": f"{prefix}-{index:03d}", "text": text, "gold": gold, "tags": tags})

    texts = [r["text"] for r in rows]
    assert len(set(texts)) == len(texts), "duplicated case"

    out = here / "data" / f"{name}.jsonl"
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    headline = [r for r in rows if "borderline" not in r["tags"]]
    print(f"{name}: {len(rows)} cases, {len(headline)} in the headline")
    for label in labels:
        print(f"  {label:12s} {sum(1 for r in headline if r['gold'] == label)}")
    tags = sorted({t for r in rows for t in r["tags"]})
    print("  tags:", ", ".join(f"{t}={sum(1 for r in rows if t in r['tags'])}" for t in tags))


def main() -> None:
    here = Path(__file__).parent
    write(here, "cases", "es", CASES + BORDERLINE, LABELS)
    write(here, "clinic", "cl", CLINIC_CASES, CLINIC_LABELS)
    write(here, "independent_cases", "ic", INDEPENDENT_CASES, LABELS)
    write(here, "independent_clinic", "il", INDEPENDENT_CLINIC, CLINIC_LABELS)
    write(here, "frustration", "fr", FRUSTRATION_CASES, LABELS)


if __name__ == "__main__":
    main()
