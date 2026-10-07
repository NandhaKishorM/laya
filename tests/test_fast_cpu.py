"""TileLang C target: original limitations, fp32 CPU execution, and CUDA parity.

Run: python -m pytest tests/test_fast_cpu.py -q -s
CPU checks need torch, tilelang and a C++ compiler; CUDA parity skips without CUDA.
"""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")
tilelang = pytest.importorskip("tilelang")
from laya import tl_kernels as K  # noqa: E402


@pytest.fixture(params=[torch.float32, torch.bfloat16, torch.float16], ids=["cpu", "bf16", "fp16"])
def dtype(request):
    if request.param != torch.float32 and not torch.cuda.is_available():
        pytest.skip("GPU parity needs CUDA")
    torch.manual_seed(1234)
    return request.param


def rand(*shape, dtype=torch.float32, scale=1):
    return (torch.randn(*shape) * scale).to(dtype).float()


def check(name, actual, expected, tol):
    delta = (actual.float() - expected.float()).abs().max().item()
    print("%s max_abs=%.9g tolerance=%g" % (name, delta, tol))
    assert torch.isfinite(actual).all()
    assert delta <= tol


def gpu(dtype, factory, args, tensors, floating, **kwargs):
    dt = {torch.bfloat16: "bfloat16", torch.float16: "float16"}[dtype]
    values = [x.cuda().to(dtype) if i in floating else x.cuda() for i, x in enumerate(tensors)]
    factory(*args, dtype=dt, **kwargs)(*values)
    return [x.cpu() for x in values]


@pytest.mark.parametrize("bias,act", [(False, "none"), (True, "gelu"), (True, "relu")])
def test_gemm(dtype, bias, act):
    M, N, D = 17, 70, 67
    a, w, b = rand(M, D, dtype=dtype), rand(N, D, dtype=dtype, scale=0.02), rand(N)
    out = torch.empty(M, N)
    opts = dict(bias=bias, act=act)
    K.compile_cpu(K.gemm_kernel, N, D, **opts)(a, w, b, out)
    ref = a @ w.T + (b if bias else 0)
    ref = {"none": ref, "gelu": torch.nn.functional.gelu(ref), "relu": torch.relu(ref)}[act]
    check("gemm/cpu-reference", out, ref, 2e-5)
    if dtype != torch.float32:
        other = gpu(dtype, K.gemm_kernel, (N,D), [a,w,b,torch.empty_like(out)], (0,1,3), **opts)[-1]
        check("gemm/cpu-gpu", out, other, 0.05)


def test_geglu(dtype):
    M, F, D = 17, 70, 67
    a, w = rand(M,D,dtype=dtype), rand(2*F,D,dtype=dtype,scale=0.02)
    out = torch.empty(M,F)
    K.compile_cpu(K.gemm_geglu_kernel,F,D)(a,w,out)
    x = a @ w.T
    check("geglu/cpu-reference", out, torch.nn.functional.gelu(x[:,:F]) * x[:,F:], 2e-5)
    if dtype != torch.float32:
        other = gpu(dtype,K.gemm_geglu_kernel,(F,D),[a,w,torch.empty_like(out)],(0,1,2))[-1]
        check("geglu/cpu-gpu", out, other, 0.05)


@pytest.mark.parametrize("residual,bias", [(False,False),(False,True),(True,False),(True,True)])
def test_layernorm(dtype, residual, bias):
    M, D = 17, 128
    x, r = rand(M,D,scale=3000), rand(M,D,dtype=dtype,scale=50)
    w, b = torch.rand(D)+0.5, rand(D)
    updated, out = x.clone(), torch.empty_like(x)
    opts = dict(residual=residual,bias=bias)
    K.compile_cpu(K.add_ln_kernel,D,**opts)(updated,r,w,b,out)
    refx = x + r if residual else x
    assert torch.equal(updated,refx)
    check("ln/cpu-reference",out,torch.nn.functional.layer_norm(refx,(D,),w,b if bias else None,1e-5),2e-5)
    if dtype != torch.float32:
        other = gpu(dtype,K.add_ln_kernel,(D,),[x.clone(),r,w,b,torch.empty_like(out)],(1,4),**opts)
        assert torch.equal(updated,other[0])
        check("ln/cpu-gpu",out,other[-1],0.05)


def test_rope(dtype):
    M, L, H, D = 35, 7, 2, 64
    qkv = rand(M,3*H*D,dtype=dtype)
    angles = rand(L,D//2)
    cos, sin = angles.cos(), angles.sin()
    out = qkv.clone()
    K.compile_cpu(K.rope_kernel,H,D)(out,cos,sin)
    ref = qkv.reshape(M,3,H,D).clone()
    x0, x1 = ref[:,:2,:,:D//2].clone(),ref[:,:2,:,D//2:].clone()
    cs,sn = [v[torch.arange(M)%L,None,None,:] for v in (cos,sin)]
    ref[:,:2,:,:D//2],ref[:,:2,:,D//2:] = x0*cs-x1*sn,x1*cs+x0*sn
    check("rope/cpu-reference",out,ref.reshape_as(out),2e-6)
    assert torch.equal(out[:,2*H*D:],qkv[:,2*H*D:])
    if dtype != torch.float32:
        other = gpu(dtype,K.rope_kernel,(H,D),[qkv.clone(),cos,sin],(0,))[0]
        check("rope/cpu-gpu",out,other,0.05)


@pytest.mark.parametrize("window,dynamic", [(0,False),(0,True),(17,False),(17,True)])
def test_attention(dtype, window, dynamic):
    B,L,H,D = 3, 80, 2, 32
    qkv = rand(B,L,3,H,D,dtype=dtype)
    lens = torch.tensor([L,39,0],dtype=torch.int32)
    out = torch.empty(B,L,H*D)
    args = (None if dynamic else B,None if dynamic else L,H,D)
    K.compile_cpu(K.attn_kernel,*args,window=window)(qkv,lens,out)
    q,k,v = [qkv[:,:,i].transpose(1,2) for i in range(3)]
    idx = torch.arange(L)
    mask = (idx[None,:]<lens[:,None])[:,None,None,:].expand(B,1,L,L)
    if window:
        mask = mask & ((idx[:,None]-idx[None,:]).abs()<=window)[None,None]
    ref = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape_as(out)
    valid = idx[None,:]<lens[:,None]
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out[2]) == 0
    check("attn/cpu-reference",out[valid],ref[valid],2e-5)
    if dtype != torch.float32:
        other = gpu(dtype,K.attn_kernel,args,[qkv,lens,torch.empty_like(out)],(0,2),window=window)[-1]
        assert torch.isfinite(other).all()
        check("attn/cpu-gpu",out[valid],other[valid],0.02)


# These are the first failures of the original GPU specialization, not a claim
# that the math is impossible on CPU. Fail on changed diagnostics after upgrades.
PROBES = [
    (K.gemm_kernel,(128,64),"The layout for fragment C_l can not be inferred correctly."),
    (K.gemm_geglu_kernel,(64,64),"CPU fill only supports local and global buffers, but got dst scope `local.fragment`."),
    (K.add_ln_kernel,(128,),"CPU reduce only supports local src and local/local.var dst buffers, got src scope `local.fragment` and dst scope `local.fragment`."),
    (K.rope_kernel,(2,64),"Cannot convert type bfloat16 to C type"),
    (K.attn_kernel,(1,64,2,64),"The layout for fragment s_c can not be inferred correctly."),
]


@pytest.mark.parametrize("factory,args,message",PROBES,ids=[p[0].func.__name__ for p in PROBES])
@pytest.mark.parametrize("dtype",K.DTYPES)
def test_original_cpu_failure(factory,args,message,dtype,capfd):
    if tilelang.__version__ != "0.1.14":
        pytest.skip("diagnostics pinned to TileLang 0.1.14; re-probe on upgrade")
    if factory is K.rope_kernel and dtype == "float16":
        with pytest.raises(RuntimeError,match="Compilation Failed!"):
            tilelang.compile(factory.get_tir(*args,dtype=dtype),target="c",pass_configs=K.FAST)
        captured = capfd.readouterr()
        # Pin the type mismatch, not the full compiler sentence: the wording differs
        # between runs of the same compiler ("no matching function for call to
        # 'vec_type<float, 4>::vec_type(half4&)'" vs "no known conversion for argument 1
        # from 'vec_type<float, 4>' to 'vec_type<half_float::half, 4>&&'").
        assert "vec_type<float, 4>" in captured.err
        assert ("no matching function for call to" in captured.err
                or "no known conversion for argument 1" in captured.err)
        print("rope_kernel float16: vec_type<float, 4> / half4 conversion is unavailable")
    else:
        with pytest.raises(tilelang.tvm.error.InternalError) as exc:
            tilelang.compile(factory.get_tir(*args,dtype=dtype),target="c",pass_configs=K.FAST)
        assert message in str(exc.value)
        print(factory.func.__name__ + " " + dtype + ": " + message)


@pytest.mark.parametrize("factory,args,message",PROBES,ids=[p[0].func.__name__ for p in PROBES])
def test_bf16_after_removing_fragments(factory,args,message):
    if tilelang.__version__ != "0.1.14":
        pytest.skip("diagnostics pinned to TileLang 0.1.14; re-probe on upgrade")
    with pytest.raises(tilelang.tvm.error.InternalError) as exc:
        tilelang.compile(factory.get_tir(*args,cpu=True),target="c",
                         pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
    assert "Cannot convert type bfloat16 to C type" in str(exc.value)


def test_gpu_still_rejects_fp32():
    with pytest.raises(ValueError):
        K.gemm_kernel.get_tir(128,64,dtype="float32")


@pytest.mark.parametrize("operation",["fill","sum","max"])
def test_fragment_operations_without_bf16(operation):
    # Isolate GEGLU's fill and LN/attention's reductions from GEMM layout and dtype.
    if tilelang.__version__ != "0.1.14":
        pytest.skip("diagnostics pinned to TileLang 0.1.14; re-probe on upgrade")
    import tilelang.language as T

    @T.prim_func
    def probe(X: T.Tensor((4,128),"float32"),Y: T.Tensor((4,),"float32")):
        with T.Kernel(1,threads=32):
            x = T.alloc_fragment((4,128),"float32")
            y = T.alloc_fragment((4,),"float32")
            for i,j in T.Parallel(4,128):
                x[i,j] = X[i,j]
            if operation == "fill":
                T.fill(y,0)
            elif operation == "sum":
                T.reduce_sum(x,y,dim=1)
            else:
                T.reduce_max(x,y,dim=1)
            for i in T.Parallel(4):
                Y[i] = y[i]

    message = ("CPU fill only supports local and global buffers, but got dst scope `local.fragment`."
               if operation == "fill" else
               "CPU reduce only supports local src and local/local.var dst buffers, got src scope `local.fragment` and dst scope `local.fragment`.")
    with pytest.raises(tilelang.tvm.error.InternalError) as exc:
        tilelang.compile(probe,target="c")
    assert message in str(exc.value)
    print(operation + ": " + message)
