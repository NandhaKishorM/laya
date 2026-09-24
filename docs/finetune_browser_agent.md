# Fine-tuning Laya as a browser-agent decision head

A worked, fully reproducible example of specialising Laya for a decision family it cannot do
zero-shot: picking the next browser action (operation + target element) for
[browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast), whose `/v1/systemone`
request format is the same as `Agent.predict(state, questions)`. Everything below ran on one
RTX 4070 Ti SUPER (16 GB) with no paid API; weights, code and per-run results are at
[huggingface.co/cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser).

## Result

| | typed-decisions, zero-shot | fine-tuned |
|---|---|---|
| element top-1 on held-out pages (2,734 decisions, ~45 candidates each) | 0.10 (chance) | **0.66** (421M) / 0.63 (322M) |
| operation accuracy (CLICK / TYPE_TEXT / SELECT / DONE) | 0.54 | 0.88–0.89 |
| 16 real browser tasks, 3 runs each | 0 % | **62 %** (322M), 50 % (421M) |
| latency per step (3 questions, 30–65 candidates) | 50–200 ms | 41–50 ms (421M), **17–23 ms** (322M) |

The live suite is bimodal: 10 tasks pass 3/3 (category / tab / page navigation, checkbox,
`<select>`, search + submit on some sites) and 6 fail 3/3 (type-then-pick-a-suggestion flows,
pagination that needs a scroll first, Google Flights). Run-to-run variance on live sites is larger
than the gap between the two backbones, so treat them as equivalent and pick by latency.

Checkpoints are ordinary Laya checkpoint directories:

```python
agent = laya.load("laya-browser/v10s")                       # after huggingface-cli download cklxx/laya-browser
agent.cfg["head_max_len"] = agent.cfg["head_max_len_train"]  # 768; the config records the input format too
```

## Pipeline

Every step is a script in `code/finetune/` of the Hub repo; `run_v10.sh` / `run_v10s.sh` run it end to end.

1. **Crawl** 421 real pages (Wikipedia, GitHub, HN, arXiv, HF, demo shops, form-heavy test sites) with
   jev's DOM reader, keeping the element table and page text.
2. **Reverse-generate goals** (5,244): pick an element as the answer, ask a local Qwen3-8B to write the
   goal a user would state to need it. No teacher has to *solve* anything, so labels are clean.
3. **Real DONE states** (700): execute the click in the browser and record the landing page with the
   history as a DONE case.
4. **Step-2 negatives** (659): new goals on those landing pages with the history kept, so "having a
   history" stops predicting DONE.
5. **Mind2Web** ([osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web), 7,296 steps):
   candidates re-rendered as an element table, action history from `action_reprs`, typed values shown
   as the field's current value.
6. **On-policy corrections** (DAgger, 177): run real tasks with the current model, ask a local LLM at
   each step, keep its verdict with the model's own state.
7. **Build → train → calibrate → eval**: Laya's RLCD recipe (gold-distribution soft targets + noisy-logit policy gradient + soft CE),
   single GPU, no gradient checkpointing, 4 epochs (~2 h for 421M, ~1 h for 322M), post-hoc
   temperature, held-out pages / websites for eval.

## What mattered most: the input format

With jev's state passed verbatim (page text + the whole element table as JSON inside `state`) the
1,024-token window truncates most of the table, so the model often never sees the candidate it should
pick. Moving elements *out* of the state and into the option list (full label + role + current value,
`head_max_len` 512 → 768; state keeps title / URL / history / 1.2–1.5k chars of text) was worth more
than any data change: Mind2Web click top-1 0.44 → 0.51 and the live suite 6/16 → 10/16 on the same data.

## What did not work (so you don't repeat it)

- Templated DONE goals ("Open the page titled X, stop once it is open") leak phrasing; the model
  learns *stop when ⇒ DONE*. DONE samples must be real landing pages after an executed action.
- If every DONE sample has exactly one prior action and every click sample none, the model learns
  *any history ⇒ DONE*. Add mid-task negatives.
- Mind2Web alone kills DONE / TYPE_TEXT (no DONE there, CLICK dominates). Re-weight rare operations
  (DONE ×4, TYPE_TEXT / SELECT ×3).
- Truncating page text to 3,000 chars saved nothing (the head dominates the sequence) and cost 0.04 top-1.
- `torch.compile` on variable-length batches recompiles per shape: 6× slower. Turning off gradient
  checkpointing was the real free win (1.25×).
- Confidence-gated escalation to a local 8B or 27B LLM made results *worse*; on these pages the
  fine-tuned 322M model is the better decider (27B with a 300-token thinking budget: 0.861 op acc /
  0.603 top-1 at 4.7 s per step, vs 0.890 / 0.623 at 21 ms). A stronger teacher is needed for further
  DAgger gains.
- jev's DOM reader hides password fields by design and never sees collapsed menus; some "failures" are
  the framework, not the model.

## Reproduce

```bash
huggingface-cli download cklxx/laya-browser --local-dir laya-browser
cd laya-browser/code && uv sync --extra fast
uv run python verify.py v10s            # downloads the checkpoint, answers one recorded browser step
```

`code/finetune/README.md` in that repo has every intermediate number from the first attempt to the
final one, and `results/` holds the per-run suite JSONs behind the table above.
