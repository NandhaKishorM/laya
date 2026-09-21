"""Fine-tuning path: device selection, calibration fitting, and a real (tiny) training run.

CI-safe and weight-free: the test builds a miniature checkpoint in a temp directory, so it
exercises the actual training loop -- forward pass, RLCD reward, policy-gradient step, optimizer
step, temperature fitting, save, and a strict reload of what was saved -- without downloading
anything or needing a GPU.

The point is that the loop is device-agnostic: this runs on CPU here, and the same code path is
what runs on CUDA/ROCm and on an AMD GPU through Metal (MPS).
"""
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import AutoModel, ModernBertConfig  # noqa: E402

from laya.common import DecisionModel, build_model  # noqa: E402
from laya.finetune import (  # noqa: E402
    available_devices,
    collate_train_batch,
    device_report,
    fit_temperature,
    pick_device,
    train_rlcd,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# =============================================================== device selection
info = available_devices()
check_true("devices/report names every backend",
           all(word in device_report() for word in ("CUDA/ROCm", "MPS", "CPU")), device_report())

check("devices/explicit cpu is honoured", pick_device("cpu").type, "cpu")
# an explicit request is not silently overridden, so the error names the device the caller asked for
check("devices/explicit mps is honoured", pick_device("mps").type, "mps")

auto = pick_device()
if info["cuda"]:
    check("devices/auto prefers cuda (or rocm, which reports as cuda)", auto.type, "cuda")
elif info["mps"]:
    check("devices/auto falls back to mps", auto.type, "mps")
else:
    check("devices/auto falls back to cpu", auto.type, "cpu")
check_true("devices/auto picked something real",
           auto.type in ("cuda", "mps", "cpu"), auto.type)


# =============================================================== calibration fitting
# targets generated from a known temperature: the fit should recover it
torch.manual_seed(0)
Z = torch.randn(40, 3) * 2.0
T_true = 2.0
T = torch.softmax(Z / T_true, -1)
fitted = fit_temperature([(Z[i].tolist(), T[i].tolist()) for i in range(len(Z))])
check_true("temperature/fit recovers a known temperature", 1.5 < fitted < 2.6, "fitted %.3f" % fitted)
check("temperature/too few pairs returns the default", fit_temperature([(Z[0].tolist(), T[0].tolist())]), 1.0)


# =============================================================== batching
tiny_items = [{"ids": [1, 2, 3], "markers": [1, 2], "qtype": 0, "target": [0.7, 0.3], "label": 0},
              {"ids": [4, 5], "markers": [1], "qtype": 2, "target": [0.4, 0.6], "label": 1}]
b = collate_train_batch(tiny_items, pad_id=0)
check("collate/shape", tuple(b["input_ids"].shape), (2, 3))
check("collate/pad", b["input_ids"][1].tolist(), [4, 5, 0])
check("collate/marker mask", b["marker_mask"][1].tolist(), [True, False])


# =============================================================== a real training run
def make_tiny_checkpoint(root):
    """A miniature but structurally real checkpoint, so build_model and the loop both run."""
    ecfg = ModernBertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, local_attention=8,
        global_attn_every_n_layers=2, pad_token_id=0, mask_token_id=1,
        cls_token_id=2, sep_token_id=3,
    )
    os.makedirs(os.path.join(root, "encoder"), exist_ok=True)
    ecfg.save_pretrained(os.path.join(root, "encoder"))
    model = DecisionModel(AutoModel.from_config(ecfg), head_layers=1, n_act=2)
    save_file({k: v.half().contiguous() for k, v in model.state_dict().items()},
              os.path.join(root, "model.safetensors"))
    with open(os.path.join(root, "rl_agent_config.json"), "w") as f:
        json.dump({"encoder": "unused-in-this-test", "head_layers": 1, "max_len": 64,
                   "head_max_len": 32, "act_costs": {"escalate": 0.5},
                   "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}, f, indent=2)
    return model


def make_items(n=8):
    torch.manual_seed(1)
    items = []
    for i in range(n):
        k = 2 + (i % 3)
        ids = [2] + [int(x) for x in torch.randint(4, 60, (10,))] + [3]
        markers = list(range(2, 2 + k))
        qtype = i % 3
        if qtype == 0:
            target = [1.0 if j == i % k else 0.0 for j in range(k)]
        else:
            target = [1.0 / k] * k
        items.append({"ids": ids, "markers": markers, "qtype": qtype, "target": target,
                      "label": target.index(max(target))})
    return items


work = tempfile.mkdtemp(prefix="laya_finetune_test_")
try:
    ckpt = os.path.join(work, "base")
    out = os.path.join(work, "finetuned")
    make_tiny_checkpoint(ckpt)
    items = make_items()

    metrics = train_rlcd(
        items, ckpt, out, device="cpu", epochs=1, micro_batch=2, grad_accum=1, group_size=2,
        grad_checkpointing=False, calib_max=4, pad_id=0, log_every=0, log=lambda s: None,
    )
    check("train/device reported", metrics["device"], "cpu")
    check("train/ran a single process", metrics["world_size"], 1)
    check("train/items seen", metrics["items"], len(items))
    check_true("train/updates happened", metrics["updates"] > 0, "updates=%s" % metrics["updates"])
    check_true("train/loss is finite", math.isfinite(metrics["final_loss"]), str(metrics["final_loss"]))
    check_true("train/reward is finite", math.isfinite(metrics["mean_reward"]), str(metrics["mean_reward"]))
    check_true("train/temperatures fitted for all three types",
               len(metrics["temperatures"]) == 3, str(metrics["temperatures"]))

    # what it saved must load back into a freshly built model, strictly
    saved = os.path.join(out, "model.safetensors")
    check_true("save/wrote weights", os.path.exists(saved))
    check_true("save/wrote config", os.path.exists(os.path.join(out, "rl_agent_config.json")))
    check_true("save/wrote encoder config", os.path.exists(os.path.join(out, "encoder", "config.json")))
    with open(os.path.join(out, "rl_agent_config.json")) as f:
        saved_cfg = json.load(f)
    check("save/marked fine-tuned", saved_cfg.get("fine_tuned"), True)
    check_true("save/recorded temperatures", len(saved_cfg["temperature"]) == 3)

    reloaded = build_model(saved_cfg, encoder_dir=os.path.join(out, "encoder"))
    reloaded.load_state_dict(load_file(saved), strict=True)
    check_true("save/strict reload into the same architecture", True)
    # weights are stored half precision, as the published checkpoints are
    check("save/stored as fp16", str(load_file(saved)["scorer.1.weight"].dtype), "torch.float16")

    # a second run with grad accumulation across two steps also completes
    m2 = train_rlcd(items, ckpt, os.path.join(work, "out2"), device="cpu", epochs=2, micro_batch=2,
                    grad_accum=2, group_size=2, grad_checkpointing=False, limit=4, pad_id=0,
                    log_every=0, log=lambda s: None)
    check("train/limit caps the items", m2["items"], 4)
    check_true("train/two epochs ran", m2["updates"] >= 2, "updates=%s" % m2["updates"])
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    FAIL.append("train/raised %s: %s" % (type(e).__name__, e))
finally:
    shutil.rmtree(work, ignore_errors=True)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
