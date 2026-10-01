"""Asks a local language model for the raw utterances and caches every answer.

The model only WRITES sentences. It never labels one: each request is for sentences of a
kind that is known beforehand, so the label comes from the request and not from the model.

Standard library only. Needs an Ollama server.
"""
import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path

import domains

RAW = Path(__file__).parent / "raw"

STYLE = (
    "Sos hondureño y escribís frases tal como las dice la gente por teléfono en "
    "Centroamérica: español natural, casi siempre de usted, sin formalismos de carta. "
    "Variá mucho: unas frases de cuatro palabras, otras de veinte; unas empiezan con un "
    "saludo, la mayoría no; hombres y mujeres, jóvenes y mayores, apurados y tranquilos. "
    "Una de cada cuatro puede arrancar con una muletilla del habla local. Cada frase tiene "
    "que sonar a algo que alguien diría de verdad y empezar distinto que las demás. "
    "Devolvé SOLO un objeto JSON con la forma {\"frases\": [\"...\", \"...\"]}."
)

KINDS = {
    "indirect": (
        "Escribí {n} frases distintas que diría una persona que llama a {business} y "
        "necesita que se ocupe de su caso el área que atiende esto: «{description}». La "
        "persona QUIERE HACER un trámite o TIENE un problema concreto y lo cuenta. No son "
        "preguntas de información general como horarios, precios o dirección. No nombra el "
        "departamento ni pide que la transfieran."),
    "direct": (
        "Escribí {n} frases distintas de una persona que llama a {business} y pide que la "
        "comuniquen con el área de «{label}» ({description}). TODAS tienen que pedir que "
        "las pasen, comuniquen o transfieran, y nombrar el área. La mitad son muy cortas."),
    "negated": (
        "Escribí {n} frases distintas de una persona que llama a {business} y aclara que "
        "NO necesita el área de «{label}» ({description}): dice que no la pasen ahí, o que "
        "no tiene ese problema, y que solo quiere hacer una pregunta o que la atienda quien "
        "contestó. Cada frase tiene que llevar una negación clara."),
}

# Sentences where a negation describes the PROBLEM, and the caller still needs the area.
# Without them every "no" in the training data points at "keep talking", and the model
# learns the word instead of the meaning: "no tengo internet" is a request, not a refusal.
KINDS["needs_with_no"] = (
    "Escribí {n} frases distintas de una persona que llama a {business} y necesita que se "
    "ocupe de su caso el área que atiende esto: «{description}». En TODAS la persona "
    "cuenta su problema usando una negación: «no me funciona», «no me llegó», «no puedo», "
    "«no me han», «nunca me», «no tengo». La persona SÍ necesita ayuda con eso. No nombra "
    "el departamento.")

POOLS = {
    "farewell": (
        "Escribí {n} frases distintas con las que una persona termina una llamada "
        "telefónica a una empresa: se despide, agradece, dice que eso era todo o que ya no "
        "necesita nada más."),
    "greeting": (
        "Escribí {n} frases distintas de lo que dice una persona al empezar una llamada o a "
        "mitad de ella sin pedir nada todavía: saludos, «¿me escucha?», «un momento», "
        "«¿cómo dijo?», frases a medias, muletillas."),
    "human": (
        "Escribí {n} frases distintas de una persona que llama a una empresa, la atiende un "
        "agente automático y pide hablar con una persona de verdad, sin decir con qué "
        "departamento."),
    "human_with_no": (
        "Escribí {n} frases distintas de una persona que llama a una empresa, la atiende un "
        "agente automático y pide hablar con una persona de verdad. En TODAS usa una "
        "negación para decirlo: «no quiero hablar con una máquina», «no me sirve el "
        "robot», «con la grabación no». No dice con qué departamento."),
    "negated_hangup": (
        "Escribí {n} frases distintas de una persona que está en una llamada y pide que NO "
        "le cuelguen o aclara que todavía NO terminó: «no cuelgue», «espere, me falta otra "
        "cosa», «todavía no me despido». Cada frase tiene que llevar una negación clara."),
    # An angry caller. None of the phrases a native speaker wrote for the frustration test
    # is here: they are held out, and the builder drops anything close to them.
    "frustration": (
        "Escribí {n} frases distintas de una persona que está en una llamada con el agente "
        "automático de una empresa y está MOLESTA con el agente: le reclama que no "
        "entiende, que repite lo mismo, que no contesta lo que le preguntó, que ya le dijo "
        "que no. Algunas llevan malas palabras suaves de Centroamérica. NO pide hablar con "
        "otra persona, NO pide que la transfieran y NO se despide: solo se queja."),
    "frustration_human": (
        "Escribí {n} frases distintas de una persona que está en una llamada con el agente "
        "automático de una empresa, está MOLESTA porque no la entiende, y por eso exige "
        "hablar con una persona de verdad. Tiene que pedir claramente una persona."),
    "negated_human": (
        "Escribí {n} frases distintas de una persona que llama a una empresa y dice que NO "
        "quiere que la pasen con nadie todavía, que prefiere que le responda quien contestó."),
}

OPEN = (
    "Escribí {n} preguntas distintas de información general que haría una persona que llama "
    "a {business}: horarios, dirección, si abren feriados, formas de pago, parqueo, "
    "sucursales, requisitos. Son preguntas que contesta cualquiera en el mostrador; no "
    "piden que las pasen con nadie."
)


def ask(url, model, prompt, temperature):
    body = json.dumps({
        "model": model, "stream": False, "think": False, "format": "json",
        "keep_alive": "30m",
        "messages": [{"role": "system", "content": STYLE},
                     {"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": 2500},
    }).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        content = json.loads(response.read())["message"]["content"]
    try:
        phrases = json.loads(content).get("frases", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    clean = []
    for phrase in phrases:
        if isinstance(phrase, str):
            text = re.sub(r"\s+", " ", phrase).strip().strip('"«»')
            if 3 <= len(text) <= 240:
                clean.append(text)
    return clean


def cached(name, url, model, prompt, rounds, temperature):
    path = RAW / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["phrases"]
    seen, phrases = set(), []
    for _ in range(rounds):
        for phrase in ask(url, model, prompt, temperature):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)
    path.write_text(json.dumps(
        {"name": name, "model": model, "prompt": prompt, "phrases": phrases},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return phrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:11434/api/chat")
    parser.add_argument("--model", default="qwen3.6:35b-a3b-q8_0")
    parser.add_argument("--only", default=None, help="one domain, to try the prompts")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    RAW.mkdir(exist_ok=True)
    started = time.time()

    jobs = []
    for key, domain in domains.DOMAINS.items():
        if args.only and key != args.only:
            continue
        for label, description in domain["departments"].items():
            for kind, template in KINDS.items():
                n = {"indirect": 20, "direct": 8, "negated": 8, "needs_with_no": 8}[kind]
                jobs.append((f"{key}.{label}.{kind}", template.format(
                    n=n, business=domain["business"], label=label,
                    description=description), args.rounds))
        jobs.append((f"{key}.open", OPEN.format(n=15, business=domain["business"]), args.rounds))

    if not args.only:
        for name, template in POOLS.items():
            jobs.append((f"pool.{name}", template.format(n=25), args.rounds * 3))

    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cached, name, args.url, args.model, prompt, rounds, args.temperature): name
            for name, prompt, rounds in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            total += len(future.result())
            if done % 10 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} requests, {total} phrases, "
                      f"{time.time() - started:.0f} s", flush=True)

    print(f"{total} phrases in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
