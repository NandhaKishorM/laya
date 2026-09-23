"""TileLang fast path: kernels vs torch reference, and full forward vs the stock model.

Skips unless CUDA + tilelang are available.  Run: python -m pytest tests/test_fast.py -q
"""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("tilelang")
from laya import tl_kernels as K  # noqa: E402

dev = "cuda"
def err(a, b): return (a.float() - b.float()).abs().max().item()


def test_gemm_epilogues():
    M, Kd = 200, 768   # M not a multiple of the tile on purpose
    A = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
    for N, bias, act in [(2304, False, "none"), (768, True, "gelu"), (3072, True, "relu")]:
        W = torch.randn(N, Kd, device=dev, dtype=torch.bfloat16) * 0.02; b = torch.randn(N, device=dev)
        C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        K.gemm_kernel(N, Kd, bias=bias, act=act)(A, W, b, C)
        ref = A.float() @ W.float().T + (b if bias else 0)
        ref = {"none": ref, "gelu": torch.nn.functional.gelu(ref), "relu": torch.relu(ref)}[act]
        assert err(C, ref) < 0.05


def test_geglu():
    M, Kd, F = 256, 768, 1152
    A = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16); Wi = torch.randn(2 * F, Kd, device=dev, dtype=torch.bfloat16) * 0.02
    C = torch.empty(M, F, device=dev, dtype=torch.bfloat16); K.gemm_geglu_kernel(F, Kd)(A, Wi, C)
    x = A.float() @ Wi.float().T
    assert err(C, torch.nn.functional.gelu(x[:, :F]) * x[:, F:]) < 0.05


def test_add_layernorm():
    M, D = 100, 768
    X = torch.randn(M, D, device=dev) * 3000            # fp32 residual stream at ModernBERT-large's real magnitude
    R = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 50
    w = torch.rand(D, device=dev) + 0.5; b = torch.randn(D, device=dev)
    X2 = X.clone(); Y = torch.empty(M, D, device=dev, dtype=torch.bfloat16)
    K.add_ln_kernel(D, residual=True, bias=True)(X2, R, w, b, Y)
    xr = X + R.float()
    assert torch.equal(X2, xr)                              # the stream is updated exactly, in fp32
    assert err(Y, torch.nn.functional.layer_norm(xr, (D,), w, b, 1e-5)) < 0.05


def test_attention_mask_and_window():
    H, Dh, B = 12, 64, 3
    for L, window, dyn in [(80, 0, True), (200, 65, True), (1024, 65, False), (1024, 0, False)]:
        qkv = torch.randn(B, L, 3, H, Dh, device=dev, dtype=torch.bfloat16)
        lens = torch.tensor([L, L - 7, max(1, L // 3)], device=dev, dtype=torch.int32)
        O = torch.empty(B, L, H * Dh, device=dev, dtype=torch.bfloat16)
        K.attn_kernel(None if dyn else B, None if dyn else L, H, Dh, window=window)(qkv, lens, O)
        q, k, v = [qkv[:, :, i].transpose(1, 2).float() for i in range(3)]
        idx = torch.arange(L, device=dev)
        mask = (idx[None, :] < lens[:, None])[:, None, None, :].expand(B, 1, L, L)
        if window:
            mask = mask & ((idx[:, None] - idx[None, :]).abs() <= window)[None, None]
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, L, -1)
        valid = idx[None, :] < lens[:, None]
        assert torch.isfinite(O).all()
        assert (O.float() - ref)[valid].abs().max().item() < 0.02


@pytest.mark.skipif(os.environ.get("LAYA_TEST_MODEL") is None, reason="set LAYA_TEST_MODEL=<repo or path> to run")
def test_full_forward_matches_stock():
    import laya
    from laya.common import QTYPES, build_sequence, collate_items
    agent = laya.load(os.environ["LAYA_TEST_MODEL"], subfolder=os.environ.get("LAYA_TEST_SUBFOLDER"))
    q = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "tech": "bugs", "sales": "pricing"}},
         "urg": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
         "churn": {"type": "noul", "instructions": "Threatens to cancel?"}}
    st = {"body": "We were billed twice, refund now or we cancel. " * 30}
    items = []
    for qid in q:
        qq = agent._to_internal(q[qid]); seq, m = build_sequence(agent.tok, st, qq, agent.cfg["max_len"], agent.cfg["head_max_len"])
        items.append({"ids": seq, "markers": m, "qtype": QTYPES[qq["t"]]})
    b = {k: v.cuda() for k, v in collate_items([items], agent.tok.pad_token_id).items() if torch.is_tensor(v)}
    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=agent.dtype):
            return agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])[0]
    lo = run(); assert agent.accelerate(strict=True); lf = run()
    assert (torch.softmax(lo.float(), -1) - torch.softmax(lf.float(), -1)).abs().max().item() < 0.02
