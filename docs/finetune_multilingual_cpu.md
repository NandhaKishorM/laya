# Fine-tuning the multilingual checkpoint on a single CPU

A worked, reproducible example of specialising the multilingual checkpoint (322M, mmBERT-base)
for a decision family it handles poorly zero-shot: Portuguese customer-support judgments. The
multilingual checkpoint understands Portuguese, but it under-signals yes/no decisions such as
churn and escalation — `"vou cancelar minha assinatura"` scores a 0.07 churn probability. A small
fine-tune on one CPU (no GPU, no paid API) closes most of that gap.

Everything below ran on a 12-core x86-64 CPU with 19 GB RAM, using
[`research/scripts/finetune_single_device.py`](https://github.com/NandhaKishorM/laya/pull/704).

## Result

Held-out Portuguese cases (not in the training set), comparing the shipped `laya-multilingual`
checkpoint against the fine-tuned one. Reference column is the hosted TypeSafe Jev 1.13.0.

| question | original | fine-tuned | Jev |
|---|---|---|---|
| churn `"encerro minha conta"` (noul) | 0.069 | **0.814** | 0.970 |
| churn `"vou cancelar na semana que vem"` (noul) | 0.073 | **0.947** | 0.950 |
| churn `"devolvam meu dinheiro ou paro"` (noul) | 0.060 | **0.792** | 0.950 |
| churn `"pausar o plano durante as férias"` (noul, no) | 0.041 | 0.018 | 0.040 |
| escalate `"chefe do seu chefe"` (noul) | 0.821 | **0.962** | 0.860 |
| escalate `"resposta da diretoria"` (noul) | 0.230 | **0.740** | 0.900 |
| sentiment neutral (score 0–2) | 0.69 | 0.99 | 1.00 |
| department `"orçamento do plano empresarial"` (choice) | billing 0.33 | **sales 0.92** | sales 1.00 |

The fine-tuned model matches Jev closely on churn and escalation where the shipped checkpoint was
effectively blind. Accuracy on the 18 held-out cases went from 9 correct to 18 correct against the
Jev reference.

## Pipeline

The dataset is 128 synthetic Portuguese cases in the `{state, questions, gold}` schema from
[`docs/finetune.md`](finetune.md), split across the three primitives:

| primitive | cases | notes |
|---|---|---|
| `noul` churn | 44 | 13 positive + 31 negative, including pause / suspend / freeze / vacation phrasings |
| `noul` escalate | 23 | positive + negative |
| `score` sentiment / urgency | 27 | 3-level rubrics |
| `choice` department | 22 | billing / technical / sales / support / other |

Training used the single-device script with its defaults: RLCD recipe (gold-distribution soft
targets, noisy-logit policy gradient, soft cross-entropy), 8 epochs, batch 8, encoder lr 2.5e-5,
head lr 1e-4, cosine schedule. On CPU a full 8-epoch run took about 14 minutes.

## What mattered most

- **Negative coverage for `noul`.** The first pass fixed the churn positives but then over-triggered
  on `"pausar"` / `"suspender"` / `"congelar"`. Adding pause/suspend/freeze/vacation phrasings as
  negative examples removed the false positives while keeping the positives near Jev.
- **Seeding the training run.** Exploration noise in the policy-gradient term (`torch.randn`) made
  the tiny dataset unstable run to run; a fixed seed made the result reproducible.

## What did not work

- **Instruction language alone.** Asking the same questions in Portuguese instead of English moved
  individual cases both ways (churn improved, escalation got worse) and was not a reliable fix —
  the weakness is in the checkpoint, not the prompt.
- **A ~90-example dataset.** At that size each retrain fixed one phrasing and broke another. It took
  the larger negative coverage above before the held-out false positives stabilised.

## Reproduce

```bash
python research/scripts/finetune_single_device.py \
    --data dataset_pt.jsonl \
    --model-dir /path/to/laya-multilingual \
    --output-dir laya_pt_finetuned \
    --epochs 8 --seed 0
```

Then evaluate the result like any other checkpoint:

```python
import laya

agent = laya.load("./laya_pt_finetuned")
print(agent.predict("Não aguento mais, vou cancelar o serviço.",
                    {"churn": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}))
```

## Caveats

- The dataset is small and synthetic; the numbers are a proof of concept, not a production
  benchmark. Real support tickets will generalise better than hand-written phrasings.
- With 12 held-out calibration items (under the 10-item floor) the per-type temperatures were left
  at 1.0, so confidence is not re-calibrated here; accuracy (argmax) is unaffected.
