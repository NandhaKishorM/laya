"""TileLang kernels for the Laya (ModernBERT + decision head) encoder.

All kernels take 16-bit activations (bf16 by default, fp16 with dtype="float16"), accumulate in fp32.  Row count M is a runtime
symbol so one compiled kernel serves every batch/sequence bucket; M must be a
multiple of 16 (the caller pads); out-of-bounds rows are predicated by TileLang.
"""
import tilelang
import tilelang.language as T

ACC = "float"
DTYPES = ("bfloat16", "float16")
FAST = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}


def _dt(dtype):
    if dtype not in DTYPES:
        raise ValueError("tl_kernels dtype must be one of %s, got %r" % (DTYPES, dtype))
    return dtype


def _act(x, kind):
    if kind == "gelu":       # exact erf-GELU, what HF "gelu" means
        return 0.5 * x * (1.0 + T.erf(x * 0.7071067811865476))
    if kind == "relu":
        return T.max(x, 0.0)
    return x


# ----------------------------------------------------------------------------- GEMM
@tilelang.jit(pass_configs=FAST)
def gemm_kernel(N, K, bias=False, act="none", bm=64, bn=128, bk=64, stages=3, threads=128, dtype="bfloat16"):
    """C[M,N] = act(A[M,K] @ W[N,K]^T + b)."""
    DT = _dt(dtype)
    M = T.dynamic("M")

    @T.prim_func
    def main(A: T.Tensor((M, K), DT), W: T.Tensor((N, K), DT), Bv: T.Tensor((N,), ACC), C: T.Tensor((M, N), DT)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bm, bk), DT)
            W_s = T.alloc_shared((bn, bk), DT)
            C_l = T.alloc_fragment((bm, bn), ACC)
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                T.copy(A[by * bm, k * bk], A_s)
                T.copy(W[bx * bn, k * bk], W_s)
                T.gemm(A_s, W_s, C_l, transpose_B=True)
            # `bias` is a Python-level flag, so each branch is one expression (no re-bound TileLang value)
            if bias:
                for i, j in T.Parallel(bm, bn):
                    C_l[i, j] = _act(C_l[i, j] + Bv[bx * bn + j], act)
            else:
                for i, j in T.Parallel(bm, bn):
                    C_l[i, j] = _act(C_l[i, j], act)
            T.copy(C_l, C[by * bm, bx * bn])
    return main


@tilelang.jit(pass_configs=FAST)
def gemm_geglu_kernel(F, K, bm=64, bn=64, bk=64, stages=3, threads=128, dtype="bfloat16"):
    """ModernBERT GLU MLP up-projection, fused:  C[M,F] = gelu(A @ Wi[:F]^T) * (A @ Wi[F:]^T)."""
    DT = _dt(dtype)
    M = T.dynamic("M")

    @T.prim_func
    def main(A: T.Tensor((M, K), DT), W: T.Tensor((2 * F, K), DT), C: T.Tensor((M, F), DT)):
        with T.Kernel(T.ceildiv(F, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bm, bk), DT)
            Wi_s = T.alloc_shared((bn, bk), DT)
            Wg_s = T.alloc_shared((bn, bk), DT)
            Ci = T.alloc_fragment((bm, bn), ACC)
            Cg = T.alloc_fragment((bm, bn), ACC)
            T.clear(Ci); T.clear(Cg)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                T.copy(A[by * bm, k * bk], A_s)
                T.copy(W[bx * bn, k * bk], Wi_s)
                T.copy(W[F + bx * bn, k * bk], Wg_s)
                T.gemm(A_s, Wi_s, Ci, transpose_B=True)
                T.gemm(A_s, Wg_s, Cg, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                Ci[i, j] = _act(Ci[i, j], "gelu") * Cg[i, j]
            T.copy(Ci, C[by * bm, bx * bn])
    return main


# ----------------------------------------------------------------------------- LayerNorm (+residual)
@tilelang.jit(pass_configs=FAST)
def add_ln_kernel(D, residual=True, bias=False, eps=1e-5, bm=4, threads=32, dtype="bfloat16"):
    """X (fp32 residual stream) += R (16-bit branch output, if residual);  Y (16-bit) = LN(X) * w (+ b).

    The residual stream stays in fp32 exactly like the stock autocast path: ModernBERT-large's residual
    activations reach ~3e4, where bf16's 8-bit mantissa would lose ~100 units per add and drift layer by layer."""
    DT = _dt(dtype)
    M = T.dynamic("M")

    @T.prim_func
    def main(X: T.Tensor((M, D), ACC), R: T.Tensor((M, D), DT), Wv: T.Tensor((D,), ACC), Bv: T.Tensor((D,), ACC),
             Y: T.Tensor((M, D), DT)):
        with T.Kernel(T.ceildiv(M, bm), threads=threads) as bx:
            x = T.alloc_fragment((bm, D), ACC)
            xs = T.alloc_fragment((bm, D), ACC)
            mean = T.alloc_fragment((bm,), ACC)
            var = T.alloc_fragment((bm,), ACC)
            Xb = T.alloc_shared((bm, D), ACC)
            Rb = T.alloc_shared((bm, D), DT)
            Yb = T.alloc_shared((bm, D), DT)
            T.copy(X[bx * bm, 0], Xb)
            T.copy(Xb, x)
            if residual:
                T.copy(R[bx * bm, 0], Rb)
                T.copy(Rb, xs)
                for i, j in T.Parallel(bm, D):
                    x[i, j] = x[i, j] + xs[i, j]
                T.copy(x, Xb)
                T.copy(Xb, X[bx * bm, 0])
            T.reduce_sum(x, mean, dim=1)
            for i in T.Parallel(bm):
                mean[i] = mean[i] / D
            for i, j in T.Parallel(bm, D):
                xs[i, j] = (x[i, j] - mean[i]) * (x[i, j] - mean[i])
            T.reduce_sum(xs, var, dim=1)
            for i in T.Parallel(bm):
                var[i] = T.rsqrt(var[i] / D + eps)
            if bias:
                for i, j in T.Parallel(bm, D):
                    xs[i, j] = (x[i, j] - mean[i]) * var[i] * Wv[j] + Bv[j]
            else:
                for i, j in T.Parallel(bm, D):
                    xs[i, j] = (x[i, j] - mean[i]) * var[i] * Wv[j]
            T.copy(xs, Yb)
            T.copy(Yb, Y[bx * bm, 0])
    return main


# ----------------------------------------------------------------------------- RoPE (in place on packed qkv)
@tilelang.jit(pass_configs=FAST)
def rope_kernel(H, Dh, bm=32, threads=128, dtype="bfloat16"):
    """QKV[M, 3*H*Dh] packed as (q|k|v)(h)(d).  Rotates q and k in place (rotate-half convention, fp32 math).
    cos/sin: [L, Dh/2].  Row r has position r % L.  M and L are runtime symbols."""
    DT = _dt(dtype)
    M, L = T.dynamic("M"), T.dynamic("L")
    half = Dh // 2
    W = 2 * H * Dh  # q and k columns

    @T.prim_func
    def main(QKV: T.Tensor((M, 3 * H * Dh), DT), Cos: T.Tensor((L, half), ACC), Sin: T.Tensor((L, half), ACC)):
        with T.Kernel(T.ceildiv(M, bm), threads=threads) as bx:
            for i, c in T.Parallel(bm, W // 2):
                r = bx * bm + i
                pos = r % L
                hh = c // half            # which (q|k, head)
                d = c % half
                c0 = hh * Dh + d
                c1 = c0 + half
                x0 = T.cast(QKV[r, c0], ACC)
                x1 = T.cast(QKV[r, c1], ACC)
                cs = Cos[pos, d]
                sn = Sin[pos, d]
                QKV[r, c0] = T.cast(x0 * cs - x1 * sn, DT)
                QKV[r, c1] = T.cast(x1 * cs + x0 * sn, DT)
    return main


# ----------------------------------------------------------------------------- flash attention (padding mask + sliding window)
@tilelang.jit(pass_configs=FAST)
def attn_kernel(B, L, H, Dh, window=0, bm=64, bn=64, stages=1, threads=128, dtype="bfloat16"):
    """QKV: [B, L, 3, H, Dh] 16-bit (a view of the packed [M, 3*H*Dh] buffer).  Lens: [B] int32 valid length.
    O: [B, L, H*Dh].  window>0 => bidirectional sliding window |i-j| <= window.  Masked scores use a large
    finite negative so fully-masked (padding) rows stay finite.

    B and/or L may be None: they then become runtime symbols (one compile serves every shape, at the
    cost of predicated loads -- ~4x slower for full attention at L=1024, free for short inputs)."""
    DT = _dt(dtype)
    scale = (1.0 / Dh) ** 0.5 * 1.44269504  # log2(e)
    if B is None:
        B = T.dynamic("B")
    if L is None:
        L = T.dynamic("L")
    NEG = -1e9

    @T.prim_func
    def main(QKV: T.Tensor((B, L, 3, H, Dh), DT), Lens: T.Tensor((B,), "int32"), O: T.Tensor((B, L, H * Dh), DT)):
        with T.Kernel(T.ceildiv(L, bm), H, B, threads=threads) as (bx, by, bz):
            Q_s = T.alloc_shared((bm, Dh), DT)
            K_s = T.alloc_shared((bn, Dh), DT)
            V_s = T.alloc_shared((bn, Dh), DT)
            O_s = T.alloc_shared((bm, Dh), DT)
            s = T.alloc_fragment((bm, bn), ACC)
            s_c = T.alloc_fragment((bm, bn), DT)
            o = T.alloc_fragment((bm, Dh), ACC)
            m = T.alloc_fragment((bm,), ACC)
            m_prev = T.alloc_fragment((bm,), ACC)
            sc = T.alloc_fragment((bm,), ACC)
            rs = T.alloc_fragment((bm,), ACC)
            l = T.alloc_fragment((bm,), ACC)
            T.annotate_layout({Q_s: tilelang.layout.make_swizzled_layout(Q_s)})
            T.copy(QKV[bz, bx * bm:(bx + 1) * bm, 0, by, :], Q_s)
            T.fill(o, 0); T.fill(l, 0); T.fill(m, NEG)
            n = Lens[bz]
            if window > 0:
                k_lo = T.max(0, (bx * bm - window) // bn)
                k_hi = T.min(T.ceildiv(L, bn), T.ceildiv(T.min(n, (bx + 1) * bm + window), bn))
            else:
                k_lo = 0
                k_hi = T.ceildiv(n, bn)
            for k in T.Pipelined(k_lo, k_hi, num_stages=stages):
                T.copy(QKV[bz, k * bn:(k + 1) * bn, 1, by, :], K_s)
                for i, j in T.Parallel(bm, bn):
                    qi = bx * bm + i
                    kj = k * bn + j
                    if window > 0:
                        ok = (kj < n) & (qi - kj <= window) & (kj - qi <= window)
                    else:
                        ok = kj < n
                    s[i, j] = T.if_then_else(ok, 0.0, NEG)
                T.gemm(Q_s, K_s, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(QKV[bz, k * bn:(k + 1) * bn, 2, by, :], V_s)
                T.copy(m, m_prev)
                T.reduce_max(s, m, dim=1, clear=False)
                for i in T.Parallel(bm):
                    sc[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(bm, bn):
                    s[i, j] = T.exp2(s[i, j] * scale - m[i] * scale)
                T.reduce_sum(s, rs, dim=1)
                for i in T.Parallel(bm):
                    l[i] = l[i] * sc[i] + rs[i]
                T.copy(s, s_c)
                for i, j in T.Parallel(bm, Dh):
                    o[i, j] = o[i, j] * sc[i]
                T.gemm(s_c, V_s, o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, Dh):
                o[i, j] = o[i, j] / T.max(l[i], 1e-30)
            T.copy(o, O_s)
            T.copy(O_s, O[bz, bx * bm:(bx + 1) * bm, by * Dh:(by + 1) * Dh])
    return main
