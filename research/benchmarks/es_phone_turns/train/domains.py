"""The trades and destinations the training data is drawn from.

Inside one trade no two destinations can claim the same sentence: a menu where "a charge I
do not recognise on my card" fits both `tarjetas` and `reclamos` teaches noise, because
the label would depend on which of the two the generator happened to be asked for.

None of them is the telephone company or the clinic of the frozen tests, and none of their
destinations shares a label with a test destination: what the fine-tuned model is asked to
do on the tests is read a menu it has never seen.
"""

DOMAINS = {
    "banco": {
        "business": "un banco",
        "departments": {
            "tarjetas": "tarjetas de crédito y débito: bloqueos, robos, límites, reposiciones",
            "prestamos": "préstamos y créditos: solicitarlos, requisitos, cuotas",
            "cuentas": "cuentas de ahorro y de cheques: abrirlas, saldos, transferencias",
            "remesas": "enviar o cobrar dinero que viene del extranjero",
        },
    },
    "colegio": {
        "business": "un colegio",
        "departments": {
            "admisiones": "matrícula e inscripción de alumnos nuevos",
            "tesoreria": "pago de mensualidades y colegiaturas",
            "registro": "constancias, certificados y notas",
            "direccion": "asuntos de disciplina y hablar con la directora",
        },
    },
    "hotel": {
        "business": "un hotel",
        "departments": {
            "reservaciones": "reservar, cambiar o cancelar una habitación",
            "restaurante": "comida, servicio a la habitación y horarios del comedor",
            "mantenimiento": "algo dañado en la habitación: aire, ducha, televisor",
            "eventos": "salones para bodas, fiestas y reuniones de empresa",
        },
    },
    "concesionaria": {
        "business": "una concesionaria de carros",
        "departments": {
            "repuestos": "piezas y repuestos para el carro",
            "taller": "reparación y mantenimiento del carro",
            "vehiculos": "comprar un carro nuevo o usado",
            "alquiler": "alquilar un carro por días o semanas",
        },
    },
    "aseguradora": {
        "business": "una aseguradora",
        "departments": {
            "siniestros": "reportar un accidente, un robo o un daño cubierto por el seguro",
            "polizas": "contratar, renovar o modificar un seguro",
            "cobranza": "pago de primas y cuotas atrasadas del seguro",
        },
    },
    "paqueteria": {
        "business": "una empresa de envíos",
        "departments": {
            "rastreo": "saber dónde está un paquete",
            "recoleccion": "programar que pasen a recoger un envío",
            "aduanas": "trámites e impuestos de paquetes que vienen del extranjero",
            "casilleros": "abrir un casillero para recibir compras hechas por internet",
        },
    },
    "municipalidad": {
        "business": "una municipalidad",
        "departments": {
            "impuestos": "bienes inmuebles, solvencia municipal e impuestos de negocio",
            "permisos": "permisos de construcción y de operación",
            "catastro": "medidas, escrituras y registro de terrenos",
            "aseo": "tren de aseo, basura y alumbrado público",
        },
    },
    "inmobiliaria": {
        "business": "una inmobiliaria",
        "departments": {
            "alquileres": "alquilar una casa, un apartamento o un local",
            "compraventa": "comprar o vender una propiedad",
            "avaluos": "saber cuánto vale una propiedad",
            "condominios": "cuotas y reparaciones de áreas comunes de un condominio",
        },
    },
    "gimnasio": {
        "business": "un gimnasio",
        "departments": {
            "membresias": "inscribirse, renovar o congelar la membresía",
            "clases": "inscribirse a clases de grupo: zumba, yoga, spinning",
            "entrenadores": "contratar un entrenador personal",
        },
    },
    "universidad": {
        "business": "una universidad",
        "departments": {
            "becas": "solicitar o renovar una beca",
            "matricula": "inscribir clases y pagar el período",
            "biblioteca": "préstamo de libros y multas de la biblioteca",
            "posgrados": "maestrías y doctorados",
        },
    },
    "restaurante": {
        "business": "un restaurante",
        "departments": {
            "reservas": "apartar una mesa",
            "domicilio": "pedir comida para llevar o a domicilio",
            "banquetes": "comida para eventos y fiestas",
        },
    },
    "agencia": {
        "business": "una agencia de viajes",
        "departments": {
            "boletos": "comprar boletos de avión",
            "excursiones": "paquetes turísticos y tours",
            "visas": "trámite de visas y pasaportes",
            "cambios": "cambiar o cancelar un viaje ya comprado",
        },
    },
    "tienda": {
        "business": "una tienda de electrodomésticos",
        "departments": {
            "garantias": "un producto comprado que salió dañado y está en garantía",
            "entregas": "cuándo llega a la casa lo que se compró",
            "credito": "el crédito de la tienda: solicitarlo, cuotas, saldo",
            "instalaciones": "que lleguen a instalar lo comprado: aire acondicionado, lavadora",
        },
    },
}

# The three answers every menu has, each under several names: the model has to read what
# the option SAYS, not recognise a word it saw in training.
HUMAN = [
    ("agente", "pide hablar con una persona, sin decir con qué área"),
    ("asesor", "quiere que lo atienda una persona"),
    ("persona", "no quiere hablar con la máquina, pide un humano"),
    ("atencion", "pide que lo comuniquen con alguien, con quien sea"),
    ("secretaria", "pide hablar con alguien de la oficina, sin área específica"),
]
HANG_UP = [
    ("terminar", "se despide o dice que ya no necesita nada más"),
    ("despedida", "quien llama se está despidiendo"),
    ("finalizar", "ya no queda nada pendiente y quiere terminar la llamada"),
    ("cortar", "dice adiós o que eso era todo"),
]
KEEP_TALKING = [
    ("seguir", "saluda, pregunta algo, o pide que NO lo transfieran ni le cuelguen"),
    ("conversar", "cualquier otra cosa: el agente sigue hablando con quien llama"),
    ("continuar", "no pide pasar la llamada ni terminarla"),
    ("otra_cosa", "una pregunta, un saludo o una frase a medias"),
    ("nada", "no hay que transferir ni colgar"),
]

INSTRUCTIONS = [
    "¿Qué debe hacer el agente con esta llamada?",
    "¿A dónde hay que pasar esta llamada?",
    "¿Qué pide la persona que llama?",
    "Decida qué hacer con la llamada.",
    "¿Cuál es la acción correcta para lo que dijo quien llama?",
    "¿Qué necesita quien llama?",
]

TEST_LABELS = {
    "ventas", "soporte", "facturacion", "operadora", "colgar", "ninguno",
    "citas", "laboratorio", "farmacia", "emergencias", "recepcion",
}

for _domain in DOMAINS.values():
    assert not TEST_LABELS & set(_domain["departments"]), _domain
assert not TEST_LABELS & {label for label, _ in HUMAN + HANG_UP + KEEP_TALKING}
