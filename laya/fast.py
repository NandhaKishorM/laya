"""GPU fast path for laya: TileLang fused kernels + bf16 resident weights + CUDA graphs.

    agent = laya.load("convaiinnovations/laya", fast=True)      # or agent.accelerate()

Requires CUDA and `pip install laya[fast]` (tilelang).  Falls back to the stock forward otherwise.
"""
import sys
import threading
import torch
from . import tl_kernels as K

BF = torch.bfloat16


def _top_two(probs):
    """Return two columns for confidence features, including k == 1."""
    if probs.shape[-1] == 1:
        top1 = probs[:, 0]
        return torch.stack([top1, torch.zeros_like(top1)], dim=-1)
    return probs.topk(2, -1).values


def _bucket_n(n):
    return 1 << max(0, (n - 1).bit_length())


class FastLaya:
    def __init__(self, model, max_len=1024, use_graphs=True, verbose=False):
        self.m = model
        enc = model.encoder
        cfg = enc.config
        dev = next(model.parameters()).device
        self.dev = dev
        self.use_graphs = use_graphs
        # CUDA graphs reuse static input/output buffers.  Keep the complete
        # forward under one lock so concurrent callers cannot overwrite those
        # buffers between replay and head decoding.
        self._forward_lock = threading.RLock()
        self.verbose = verbose
        self.H, self.Dh, self.D = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.hidden_size
        self.F = cfg.intermediate_size
        self.eps = cfg.norm_eps
        self.max_len = max_len
        f32 = lambda t: t.detach().float().contiguous()
        b16 = lambda t: t.detach().to(BF).contiguous()
        zeros = torch.zeros(self.D, device=dev)
        self.zeros = {self.D: zeros, 3 * self.D: torch.zeros(3 * self.D, device=dev), 4 * self.D: torch.zeros(4 * self.D, device=dev),
                      2 * self.F: torch.zeros(2 * self.F, device=dev)}
        # --- encoder weights
        # embeddings are gathered from an exact fp16 copy of the checkpoint values and upcast to fp32, like the stock path
        self.emb_w = enc.embeddings.tok_embeddings.weight.detach().to(torch.float16).contiguous()
        self.emb_ln = f32(enc.embeddings.norm.weight)
        # HF's bidirectional sliding mask keeps keys with |i - j| <= config.sliding_window (= local_attention // 2);
        # ModernBertAttention.sliding_window is that value + 1 (flash-attn's inclusive convention) and must NOT be used here.
        win = getattr(cfg, "sliding_window", None) or cfg.local_attention // 2
        self.layers = []
        for i, lyr in enumerate(enc.layers):
            self.layers.append(dict(
                attn_ln=None if i == 0 else f32(lyr.attn_norm.weight),
                wqkv=b16(lyr.attn.Wqkv.weight), wo=b16(lyr.attn.Wo.weight),
                mlp_ln=f32(lyr.mlp_norm.weight), wi=b16(lyr.mlp.Wi.weight), wo2=b16(lyr.mlp.Wo.weight),
                window=(win if lyr.attention_type == "sliding_attention" else 0), ltype=lyr.attention_type))
        self.final_ln = f32(enc.final_norm.weight)
        # --- rotary tables (rounded through bf16 exactly like HF does before applying)
        rot = enc.rotary_emb
        pos = torch.arange(max_len, device=dev).float()
        self.rope = {}
        for lt in set(cfg.layer_types):
            inv = getattr(rot, f"{lt}_inv_freq").float()
            scl = getattr(rot, f"{lt}_attention_scaling")
            fr = torch.outer(pos, inv)
            self.rope[lt] = ((fr.cos() * scl).to(BF).float().contiguous(), (fr.sin() * scl).to(BF).float().contiguous())
        # --- decision head (nn.TransformerEncoderLayer, norm_first, relu)
        self.type_emb = b16(model.type_emb.weight)
        self.head = []
        for lyr in model.head.layers:
            sa = lyr.self_attn
            self.head.append(dict(
                n1w=f32(lyr.norm1.weight), n1b=f32(lyr.norm1.bias), n2w=f32(lyr.norm2.weight), n2b=f32(lyr.norm2.bias),
                in_w=b16(sa.in_proj_weight), in_b=f32(sa.in_proj_bias), out_w=b16(sa.out_proj.weight), out_b=f32(sa.out_proj.bias),
                l1w=b16(lyr.linear1.weight), l1b=f32(lyr.linear1.bias), l2w=b16(lyr.linear2.weight), l2b=f32(lyr.linear2.bias)))
        # --- kernels (M is dynamic, so these compile once)
        D, F = self.D, self.F
        self.k_qkv = K.gemm_kernel(3 * D, D)
        self.k_o = K.gemm_kernel(D, D)
        self.k_geglu = K.gemm_geglu_kernel(F, D)
        self.k_o2 = K.gemm_kernel(D, F)
        self.k_addln = K.add_ln_kernel(D, residual=True, bias=False, eps=self.eps)
        self.k_addln_b = K.add_ln_kernel(D, residual=True, bias=True, eps=1e-5)
        self.k_ln_b = K.add_ln_kernel(D, residual=False, bias=True, eps=1e-5)
        self.k_in = K.gemm_kernel(3 * D, D, bias=True)
        self.k_out = K.gemm_kernel(D, D, bias=True)
        self.k_ffn1 = K.gemm_kernel(4 * D, D, bias=True, act="relu")
        self.k_ffn2 = K.gemm_kernel(D, 4 * D, bias=True)
        self._rope_k, self._rope_tab, self._attn_k = None, {}, {}
        self.graphs = {}

    # ------------------------------------------------------------------ kernels per shape
    DYNAMIC_MAX_L = 256   # up to here one dynamic-shape attention kernel is as fast as a static one
    LONG_BUCKET = 64      # beyond it, static kernels per (B, L) bucket of this size

    def rope_k(self):
        if self._rope_k is None:
            self._rope_k = K.rope_kernel(self.H, self.Dh)
        return self._rope_k

    def rope_tab(self, ltype, L):
        key = (ltype, L)
        if key not in self._rope_tab:
            cos, sin = self.rope[ltype]
            self._rope_tab[key] = (cos[:L].contiguous(), sin[:L].contiguous())
        return self._rope_tab[key]

    def attn_k(self, B, L, window):
        key = (None, None, window) if L <= self.DYNAMIC_MAX_L else (B, L, window)
        if key not in self._attn_k:
            self._attn_k[key] = K.attn_kernel(key[0], key[1], self.H, self.Dh, window=window)
        return self._attn_k[key]

    # ------------------------------------------------------------------ encoder + head on padded [B, L]
    def _encode(self, ids, lens, qtype):
        """ids [B,L] long (padded), lens [B] int32, qtype [B] long -> hidden [B, L, D] bf16"""
        B, L = ids.shape
        M, D = B * L, self.D
        dev = self.dev
        emb = torch.nn.functional.embedding(ids, self.emb_w).view(M, D).float()
        # residual stream = embeddings.norm(emb), kept in fp32 exactly like the stock autocast path
        X = torch.nn.functional.layer_norm(emb, (D,), self.emb_ln, None, self.eps)
        Y = X.to(BF)                                                            # layer 0 attends to it directly (attn_norm = Identity)
        qkv = torch.empty(M, 3 * D, device=dev, dtype=BF)
        O = torch.empty(M, D, device=dev, dtype=BF)
        G = torch.empty(M, self.F, device=dev, dtype=BF)
        z = self.zeros
        nl = len(self.layers)
        for i, ly in enumerate(self.layers):
            self.k_qkv(Y, ly["wqkv"], z[3 * D], qkv)                            # Y = attn_norm(X) (layer 0: X itself)
            cos, sin = self.rope_tab(ly["ltype"], L)
            self.rope_k()(qkv, cos, sin)
            self.attn_k(B, L, ly["window"])(qkv.view(B, L, 3, self.H, self.Dh), lens, O.view(B, L, D))
            self.k_o(O, ly["wo"], z[D], Y)                                      # Y = attn out
            self.k_addln(X, Y, ly["mlp_ln"], z[D], Y)                           # X += Y ; Y = mlp_norm(X)
            self.k_geglu(Y, ly["wi"], G)
            self.k_o2(G, ly["wo2"], z[D], Y)                                    # Y = mlp out
            nxt = self.layers[i + 1]["attn_ln"] if i + 1 < nl else self.final_ln
            self.k_addln(X, Y, nxt, z[D], Y)                                    # X += Y ; Y = next norm(X)
        # decision head: h = final_norm(x) + type_emb ; 2 x pre-norm transformer layers (relu ffn)
        X = (Y.view(B, L, D).float() + self.type_emb[qtype].float()[:, None, :]).view(M, D).contiguous()   # fp32 stream for the head
        F1 = torch.empty(M, 4 * D, device=dev, dtype=BF)
        for j, h in enumerate(self.head):
            self.k_ln_b(X, Y, h["n1w"], h["n1b"], Y)
            self.k_in(Y, h["in_w"], h["in_b"], qkv)
            self.attn_k(B, L, 0)(qkv.view(B, L, 3, self.H, self.Dh), lens, O.view(B, L, D))
            self.k_out(O, h["out_w"], h["out_b"], Y)
            self.k_addln_b(X, Y, h["n2w"], h["n2b"], Y)                         # X += attn ; Y = norm2(X)
            self.k_ffn1(Y, h["l1w"], h["l1b"], F1)
            self.k_ffn2(F1, h["l2w"], h["l2b"], Y)
            X = X + Y.float()                                                   # residual (fp32, torch, last op)
        return X.view(B, L, D)

    def _encode_graphed(self, ids, lens, qtype):
        key = tuple(ids.shape)
        g = self.graphs.get(key)
        if g is None:
            s_ids, s_lens, s_q = ids.clone(), lens.clone(), qtype.clone()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    self._encode(s_ids, s_lens, s_q)       # warm-up (compiles kernels, allocs)
            torch.cuda.current_stream().wait_stream(st)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                s_out = self._encode(s_ids, s_lens, s_q)
            g = self.graphs[key] = (graph, s_ids, s_lens, s_q, s_out)
            if self.verbose:
                print(f"[fast_laya] captured CUDA graph for shape {key}", file=sys.stderr)
        graph, s_ids, s_lens, s_q, s_out = g
        s_ids.copy_(ids); s_lens.copy_(lens); s_q.copy_(qtype)
        graph.replay()
        return s_out

    # ------------------------------------------------------------------ DecisionModel.forward replacement
    @torch.no_grad()
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        with self._forward_lock:
            return self._forward_unlocked(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder)

    @torch.no_grad()
    def _forward_unlocked(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
        m = self.m
        N, L0 = input_ids.shape
        g = 16 if L0 <= self.DYNAMIC_MAX_L else self.LONG_BUCKET
        L = min(self.max_len, ((L0 + g - 1) // g) * g)
        B = _bucket_n(N)
        ids = torch.zeros(B, L, dtype=torch.long, device=self.dev)
        ids[:N, :L0] = input_ids
        lens = torch.zeros(B, dtype=torch.int32, device=self.dev)
        lens[:N] = attention_mask.sum(1).to(torch.int32)
        qt = torch.zeros(B, dtype=torch.long, device=self.dev)
        qt[:N] = qtype
        h = (self._encode_graphed if self.use_graphs else self._encode)(ids, lens, qt)
        h = h[:N, :L0].float()
        # ---- scorer / act head (tiny; identical to laya.common.DecisionModel.forward)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        mk = torch.gather(h, 1, idx)
        logits = m.scorer(mk).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = torch.softmax(logits, -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        # Choice questions are allowed to contain one criterion.  The stock
        # DecisionModel handles that case, but topk(2) raises when the marker
        # dimension has width one.
        top2 = _top_two(p)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = m.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits
