# 07: PyTorch engineering constraints and techniques for gradoscopy

**Scope.** This brief covers the engineering substrate: complex autograd, dtype and device support, FFT performance, compilation, memory, special functions, gradients through randomness, API shape and testing. It does not choose physical models. It does quantify what those models cost in PyTorch.

**Provenance legend.**
- **[V]** means I checked it against a URL, which is given.
- **[M]** means I measured it on the target workstation: Windows 11, NVIDIA RTX 3090 (24 GB), and the project's own `.venv` with `torch 2.14.0+cu130`. The measurement scripts are in `scratchpad/exp{1..5}_*.py`.
- **[R]** means it comes from my recollection or expertise and was not re-verified. Treat [R] items with the stated confidence.

---

## 1. Platform baseline (September 2026)

- **Versions [V].** PyTorch 2.14.0 was released on 2 Sep 2026. Before it came 2.13 (8 Jul 2026), 2.12 (13 May), 2.11 (Mar) and 2.10 (Jan). The cadence is now roughly one release every 6–8 weeks, with 2.15 planned for 28 Oct 2026 and 2.16 for Dec 2026. Version 2.14 supports Python 3.10 to 3.15, CUDA 12.6/13.0/13.2 and ROCm 7.14, and uses C++20. Sources: https://pypi.org/project/torch/ and https://github.com/pytorch/pytorch/blob/main/RELEASE.md. The project `pyproject.toml` pins `torch>=2.8` against the cu130 index, and the lock resolves to 2.14.0.
- **Triton is missing on Windows [M].** The installed Windows wheel ships without Triton. As a result, `torch.compile` (Inductor) on CUDA fails with `TritonMissing`, which I reproduced. The official fork `triton-lang/triton-windows` provides `pip install "triton-windows<3.9"` for PyTorch 2.7–2.14 and states that "`triton.jit` and `torch.compile` just work" on Turing and newer GPUs [V: https://github.com/triton-lang/triton-windows]. On Linux the torch wheel pulls Triton automatically. **Consequence:** compilation must be an optional accelerator and never a correctness dependency.
- **Multi-GPU on Windows [R, high confidence].** NCCL is unavailable on Windows, so only the Gloo backend exists there. Multi-GPU is therefore a Linux feature.
- **Apple MPS [V/R].**
  - MPS has no float64 and no complex128. See the issue https://github.com/pytorch/pytorch/issues/148670.
  - complex64 FFT is supported through MPSGraph. complex32 is "not accepted as an input dtype on MPS" [V: https://github.com/pytorch/pytorch/pull/196375]; that PR, which enables half/bf16 FFT, is still open.
  - Coverage of general complex ops has historically been patchy: issues #119088 and #143140 report crashes in complex elementwise ops and einsum.
  - Plan MPS as a best-effort device that runs complex64 only, with smoke tests.

---

## 2. Complex tensors, autograd and FFT

### 2.1 Gradient convention

- **What `.grad` holds [V/M].** PyTorch computes the conjugate Wirtinger derivative. The docs say: "The gradient computed is ∂L/∂z*... the negative of which is precisely the direction of steepest descent" [V: https://docs.pytorch.org/docs/2.14/notes/autograd.html]. In practice `z.grad = ∂L/∂Re z + i·∂L/∂Im z = 2·∂L/∂z*`. For example, the gradient of |z|² at 1+1j is **2+2j** [M]. This equals the real-view gradient, so optimising a complex `nn.Parameter` with Adam behaves exactly like optimising its (re, im) pair.
- **JAX differs.** The same docs note that JAX returns the conjugate. Any port of formulas or code from Chromatix (JAX) must flip a conj.
- **Rules for a custom `backward` [V].** The general rule from the docs is `grad_in = conj(grad_out)·∂s/∂z* + grad_out·conj(∂s/∂z)`. The cases we actually need reduce to four:
  - For a C-linear map y = A x: `grad_x = A^H grad_y`.
  - For v = t ⊙ u: `grad_u = conj(t)·grad_v` and `grad_t = conj(u)·grad_v`.
  - For a real parameter θ entering y(θ): `grad_θ = Re(Σ conj(grad_y) · ∂y/∂θ)`.
  - For unitary FFT propagation P = ifft∘diag(H)∘fft: `P^H = ifft∘diag(conj H)∘fft`, because fft and ifft with `norm="backward"` are adjoint up to N. Using `norm="ortho"` removes that bookkeeping.
- **Validation [M].** I implemented a hand-written adjoint for multislice using these rules. It matched autograd to **7×10⁻⁶ relative error in float32** (§4.3).

### 2.2 Dtype support matrix

| dtype | CUDA | CPU | MPS | Notes |
|---|---|---|---|---|
| complex64 | full, incl. cuFFT and cuBLAS | full (MKL/pocketfft) | FFT OK, other ops patchy | **default working dtype** |
| complex128 | full, but slow on consumer GPUs | full | **none** | FFT **7.5× slower** than c64 on the 3090 [M]; ~1/64 FP64 rate on GeForce [R] |
| complex32 (`chalf`) | "experimental" warning; fft2, mul, exp, abs, sum, add all run [M] | minimal | rejected | FFT only for power-of-2 sizes [V: https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft2.html; M: error on 300×300] |
| bf16 complex | does not exist | | | |

### 2.3 torch.fft performance [M]

All timings are on the RTX 3090, in ms per `fft2`.

| workload | c64 fft2 | c32 fft2 | c128 fft2 | c64 elementwise multiply |
|---|---|---|---|---|
| 1×512² | 0.020 | 0.022 | 0.114 | 0.011 |
| **64×512²** (128 MiB field) | **0.705** | 0.971 | 5.32 | **0.525** |
| 64×1024² | 2.93 | 3.59 | 21.8 | 2.14 |
| 256×512² | 2.86 | 3.61 | 17.5 | 1.97 |

Four things follow from these numbers:
- **Elementwise multiplies are nearly as expensive as FFTs** (0.525 vs 0.705 ms). Both are bandwidth-bound. A split-step "multiply by transmission, FFT, multiply by H, iFFT" loop therefore spends about 45% of its time in unfused elementwise kernels. The real optimisation target is fusion (§3), not faster FFTs.
- **complex32 is slower than complex64** on Ampere, and it produced **NaN** in a 100-slice multislice run, while c64 gave a relative error of 5×10⁻⁶ [M]. It should be dead on arrival.
- **Size matters.** For B=64 at N=512, 520, 500 and 509 the times were 0.70, 0.77, 1.03 and 1.30 ms. Pad grids to 2^a·3^b·5^c·7^d sizes, since cuFFT has fast radices 2, 3, 5 and 7 [R]. Prime sizes cost up to 1.9×.
- **Plan cache [V].** The cuFFT plan cache is a per-device LRU at `torch.backends.cuda.cufft_plan_cache[i]`, with `max_size` defaulting to 4096 [M] [V: https://docs.pytorch.org/docs/2.13/notes/cuda.html]. Plans are keyed on shape, dtype and stride. Ragged or varying grid sizes therefore cause plan churn and allocator fragmentation. Standardise grid sizes per fidelity tier and set `PYTORCH_ALLOC_CONF=expandable_segments:True` [V, same page].
- **Use `rfft2`/`irfft2`** for every real-valued pipeline: fluorescence PSF convolution and intensity-domain operations. It halves both work and memory.

### 2.4 Precision [M]

These are relative L2 errors against a complex128 reference, for angular-spectrum propagation at 512² with λ=0.5 and dx=0.1.

| case | error |
|---|---|
| c64 propagation, any z (phase argument computed in fp64, then cast) | 3.7×10⁻⁷ |
| c32 propagation | 2.2×10⁻³ |
| c64 with the phase argument `2π·kz·z` computed in **fp32**, z=100 (200 λ) | 1.6×10⁻⁵ |
| same, z=10⁴ (2×10⁴ λ) | **1.6×10⁻³** |
| 100-slice multislice, c64 | 5.3×10⁻⁶ |
| 100-slice multislice, c32 | NaN |

**Rule.** Large phase arguments lose precision in fp32. Two fixes are enough, and complex128 buys nothing more for image formation:
1. Subtract the carrier, e.g. `exp(i(kz−k)z)`, since only relative phase matters.
2. Build phase arguments in fp64, reduce them mod 2π, and then cast to fp32.

**Autocast [V].** `torch.autocast` has no CUDA policy for FFT ops (https://docs.pytorch.org/docs/2.14/amp.html), and autocast only touches floating-point tensors. The danger is the opposite direction: a user's outer autocast context would silently run the simulator's *real* `conv2d`/`matmul` in fp16.
- Also [M], cuDNN convolutions default to **TF32** (`torch.backends.cudnn.conv.fp32_precision == "tf32"`), which gives ~2 orders of magnitude larger error [V, CUDA notes].
- The simulator should therefore wrap its internals in `torch.autocast(device_type, enabled=False)`. Precision-sensitive matmuls, such as matrix Fourier transforms for arbitrary pixel sizes, should run under an explicit IEEE fp32 setting via the 2.9+ API `torch.backends.cuda.matmul.fp32_precision` [V].

---

## 3. Compilation, transforms and custom kernels

### 3.1 torch.compile with complex numbers

What I verified:
- **Dynamo captures complex FFT code.** `torch._dynamo.explain` on a 32-slice complex multislice gives **1 graph and 0 graph breaks** [M]. `backend="aot_eager"` also traces the backward correctly [M].
- **Inductor does not generate code for complex ops.** `torch/_inductor/lowering.py` marks complex inputs as unsupported and warns: "Torchinductor does not support code generation for complex operators. Performance may be worse than eager" [V: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/lowering.py]. Complex ops fall back to ATen kernels, so there is **no fusion**. Issue #125718 ("torch.compile and complex numbers") is still open [V].
- **2.14 adds an experimental opt-in.** `torch._functorch.config.patch(enable_complex_wrapper=True)` enables a `ComplexTensor` subclass. It stores separate real and imaginary tensors and decomposes ops into real ops, with a conversion cost at graph boundaries. `view_as_real`/`view_as_complex` and mutation of complex inputs break aliasing semantics [V: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_complex_number_support.html].
- **FFT is not in ComplexTensor's op table.** The subclass registers elementwise ops, mm/bmm, exp/log and reductions, but no `_fft_c2c`/`_fft_r2c` [V: `torch/_subclasses/complex_tensor/_ops/aten.py`]. How it handles FFT is unclear; treat it as unsupported.
- **The real-view pattern works today.** Keep fields as (re, im) float32 pairs, or convert at the boundaries of the elementwise chain, and write phase screens as `cos/sin` arithmetic. Inductor then fuses the elementwise chains, and the FFTs remain extern calls. With `aot_eager` [M], the real-view formulation reduced saved activations from 2.94 to 2.47 fields per slice. With Inductor I expect a further ~1.5–2× wall-clock gain on the elementwise share, but that could not be measured here without Triton.
- **Compile the per-slice step, not the loop.** A Python loop over 256 slices unrolls into a giant graph with long compile times [R]. Regional compilation of the repeated block is the pattern the PyTorch team recommends [V: https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/].
- **Other flags.** Use `fullgraph=True` in CI to catch regressions; 2.9 added `torch._dynamo.error_on_graph_break()` [V: https://pytorch.org/blog/pytorch-2-9/]. Shape-varying batches trigger recompiles: fix the tier grid sizes and use `mark_dynamic` only on the batch dimension.

### 3.2 torch.func

- **Available APIs [V: https://docs.pytorch.org/docs/2.14/func.api.html].** `vmap`, `grad`, `vjp`/`jvp`, `jacrev`/`jacfwd`, `hessian`, `linearize`, `functional_call` and `stack_module_state`.
- **vmap and randomness.** `vmap` defaults to `randomness="error"`: calling a random op raises [M]. `"different"` and `"same"` behave as named [M, V]. `chunk_size=` trades memory for loop overhead [V].
- **autograd.Function compatibility [V: https://docs.pytorch.org/docs/2.14/notes/extending.func.html].** A custom Function works under torch.func only if it uses the `forward` + `setup_context` split, and either `generate_vmap_rule=True` (pure-torch bodies) or an explicit `vmap` staticmethod. Intermediates must be returned as outputs, not stashed on `ctx`.
  - Consequence: adjoint Functions that take Python callables (e.g. a procedural slice generator) will not be vmappable.
  - Design so that batching is explicit (leading `B` dimension) and vmap is a convenience, never a requirement.
- **High-value uses:**
  - `functional_call(module, sampled_params, inputs)` swaps sampled per-batch parameter tensors into a module.
  - `jacfwd` over a handful of physical parameters gives Fisher information and Cramér–Rao bounds, which are useful for PSF engineering and for sanity-checking learnability. With ≤10 parameters, forward mode is cheap.

### 3.3 CUDA graphs [M]

- **Measurement.** A 32-slice forward pass on 256² with B=1 took 2.29 ms eagerly and **0.80 ms as a graph replay** (2.9×), with bit-identical output. cuFFT captures fine after warm-up on a side stream.
- **Per-op overhead.** A tiny GPU op costs about **8 µs** (2.9 µs on CPU) [M]. Anything with many small ops is launch-bound: Mie recurrences, per-particle loops, small fields.
- **Where it helps.** Large batches (64×512²) are bandwidth-bound and gain nothing. Use graphs, via `torch.compile(mode="reduce-overhead")` or `torch.cuda.make_graphed_callables`, for small-field or high-op-count tiers.
- **Constraints [V: CUDA notes].** Static shapes, no host synchronisation, and a graph-safe RNG (`Generator.graphsafe_set_state`).

### 3.4 Custom kernels

- **Triton via `torch.library`.** Hotspots should be Triton kernels registered with `torch.library.triton_op`. Since 2.6, `torch.compile` can trace into them, unlike opaque `custom_op`s. Gradients go through `torch.library.register_autograd` [V: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html].
- **Complex in Triton [R].** Triton has no complex dtype. Pass `torch.view_as_real(z)` and load interleaved pairs.
- **Candidate kernels, in order:**
  1. Fused `u ← u·exp(iφ)` from a real phase or complex-RI slice (one read of u and φ, one write).
  2. Fused `F ← F·H(kx, ky; z, λ)` with H generated on the fly instead of materialised per z and λ.
  3. Fused intensity accumulation `I += |u|²` over incoherent modes.
  4. Procedural slice rasterisation of geometric primitives: spheres, ellipsoids, SDFs.
- **Mandatory fallback.** Every kernel needs a pure-torch reference implementation, both for fallback (macOS, Windows without Triton) and for tests.
- **FFT callbacks.** `nvmath-python` exposes cuFFT with LTO **prolog/epilog callbacks** on `torch.Tensor` [V: https://docs.nvidia.com/cuda/nvmath-python/0.9.0/host-apis/fft/generated/nvmath.fft.fft.html]. It is the only practical route to fusing the H-multiply into the FFT itself. Consider it a late, optional CUDA-only backend.

### 3.5 Is Cython useful?

**No, not for the engine.**
- Cython compiles CPU code, sits outside autograd, cannot run on the GPU, and would add a compiled build matrix across Windows, macOS and Linux.
- The ~8 µs per-op overhead is Python→ATen dispatch, and Cython does not remove it. CUDA graphs, `torch.compile` and bigger batched ops do.
- The only plausible CPU-side niches are mesh voxelisation and geometry preprocessing. Those are better served by numpy/numba or by doing them in torch on the GPU.

**Recommendation:** pure Python + PyTorch, with optional Triton (via `torch.library`) and optional nvmath. No C++/CUDA extension in v1.

---

## 4. Memory budgets and strategies

### 4.1 Arithmetic for the canonical workload

The workload is a 512² field, complex64, batch 64, 256 slices and up to hundreds of incoherent modes.

| quantity | size |
|---|---|
| one field, 512² c64 | 2 MiB; **×64 batch = 128 MiB** |
| same with 2× anti-aliasing padding (1024²) | 8 MiB; ×64 = 512 MiB |
| float32 phase/RI volume, 64×256×512² | **16 GiB** (complex RI: 32 GiB) |
| gradient of that volume, if it is a leaf | another 16 GiB |
| 200 modes × 64 batch at once | 25 GiB per field copy |

The measured activation cost [M] was B=16, N=512, S=32, loss `mean(W·|u|²)`:

| strategy | peak incl. φ-grad | per slice | time fwd+bwd | grad vs naive |
|---|---|---|---|---|
| naive autograd | 3.0 GB | **2.94 fields** (≈2.44 activations + 0.5 φ-grad) | 163 ms | — |
| `checkpoint` every √S slices (non-reentrant) | 1.30 GB | 1.27 | 133 ms | identical |
| custom reversible adjoint (§4.3) | 0.80 GB | 0.78 (≈9 fields total + φ-grad) | **100 ms** | 7×10⁻⁶ |
| aot_eager-compiled, complex | 3.49 GB | 3.41 | 165 ms | — |
| aot_eager-compiled, real-view | 2.53 GB | 2.47 | 208 ms | — |

Extrapolated to 64×512², 256 slices, no padding:

| strategy | memory | fits a 24 GB card? |
|---|---|---|
| naive autograd | ≈ **78 GiB** activations + 16 GiB φ-grad + 16 GiB volume | no; ~375 GiB with 2× padding |
| √S checkpointing | ≈ 55 fields ≈ 7 GiB | only if the volume and its gradient also fit, which they do not |
| reversible adjoint | ≈ 1.1 GiB | yes, but only if the sample is procedural or parametric, so neither the volume nor its gradient is ever materialised |

**The biggest memory item is the sample, not the field.**

The compute budget points the same way:
- One slice step (2 FFT + 2–3 elementwise ops) costs about **2.5–3 ms** for 64×512² on a 3090 [M].
- 256 slices × 200 incoherent modes therefore takes **~140 s per forward pass** of a 64-image batch, and roughly 3× that with backward.
- Full partially-coherent multislice is a fitting and validation tier, not a bulk-training-data tier. The fast tiers must reach 10–100 ms per batch: PSF convolution, projection or thin-object approximation, and few or zero modes.

### 4.2 Strategy toolbox, in order of preference

1. **No graph when there is nothing to learn.** When no resolved parameter requires grad, which is the normal data-generation case, run under `torch.inference_mode()`. In that branch in-place ops (`u.mul_(t)`, `fft(..., out=)`) are safe. Core ops stay out-of-place under grad mode, because in-place writes to saved tensors raise version-counter errors.
2. **Exact chunked sums over incoherent modes, wavelengths or batch.** Image formation is a *sum* of per-mode intensities. So `img = Σ_chunks checkpoint(render_chunk, params, chunk_ids)` is exact and needs only O(chunk) activations.
   - [M], for M=64 modes, B=8, 512²:

     | configuration | peak memory | time |
     |---|---|---|
     | unchunked | 3.74 GB | 65 ms |
     | chunk 8 + checkpoint | 0.51 GB | 98 ms |
     | chunk 1 + checkpoint | 0.11 GB | 129 ms |

   - Gradients were identical to 1.1×10⁻⁶.
   - Choose the chunk size automatically from `torch.cuda.mem_get_info()` and a per-operator memory estimate.
3. **Adjoint `autograd.Function`s for multislice or beam propagation** (derivation below). Two variants:
   - (a) *Reversible*: O(1) activations. Valid when transmissions are unimodular (phase-only or weakly absorbing) and propagation is unitary.
   - (b) *Segmented*: store the field every K slices and recompute forward inside the segment during backward. Memory is O(S/K + K), and one extra forward pass is needed. Use it for absorbing samples, where dividing by |t| is unstable.
4. **Procedural slices with an in-loop VJP.** The multislice Function receives compact sample parameters (particle positions, radii, RI; SDF parameters) and a slice generator. In backward it regenerates `t_k` under `enable_grad` and immediately contracts `grad_t_k` into the parameter gradients via `torch.autograd.grad`. This removes the 16+16 GiB volume and volume-gradient terms. It is the single most important memory decision.
5. **Generic `torch.utils.checkpoint`** with `use_reentrant=False`, which the docs recommend [V: https://docs.pytorch.org/docs/2.14/checkpoint.html]. Also available: selective checkpointing (`create_selective_checkpoint_contexts`, `CheckpointPolicy.MUST_SAVE/PREFER_RECOMPUTE/..._CPU_OFFLOAD`). Keep `preserve_rng_state=True` whenever noise is sampled inside a checkpointed region.
6. **CPU offload** (`torch.autograd.graph.save_on_cpu`, or the `CPU_OFFLOAD` policy) is **dominated by recompute**. A 128 MiB field over PCIe 4 at ~25 GB/s takes ~5 ms each way. Recomputing a slice step costs ~2.5 ms [M/R].

**Adjoint recurrence** for u_{k+1} = P(t_k ⊙ u_k), in the PyTorch gradient convention:

```
λ_S = grad_out
for k = S-1 … 0:
    λ_v  = P^H λ_{k+1}                 # iFFT(FFT(λ)·conj H)
    v_k  = P^H u_{k+1}                 # reversible variant; else recompute/stash
    u_k  = v_k / t_k                   # = conj(t_k)·v_k when |t_k| = 1
    grad_t_k = conj(u_k) · λ_v         # → contract into params via VJP of slice_fn
    λ_k  = conj(t_k) · λ_v
```

For a real phase φ_k, `grad_φ_k = Re(conj(λ_v) · i·v_k)`. This exact form is what I validated [M].

### 4.3 Mixed precision verdict

- **fp16/complex32:** no. It is slower on Ampere [M], restricted to power-of-2 sizes [V], unstable (NaN [M]) and range-limited (65504; an unnormalised 512² FFT grows by 2.6×10⁵).
- **bf16:** no complex type exists, and 8 mantissa bits cannot hold phase.
- **complex64 everywhere by default.** It is accurate to ~10⁻⁶ over 100 slices once phase arguments are carrier-subtracted.
- **float64/complex128 only for:**
  1. Special-function recurrences (Mie/T-matrix coefficients). These are tiny arrays, so run them in fp64 even on GeForce GPUs, or on CPU (the only option on MPS).
  2. Building large phase arguments before reducing them mod 2π.
  3. `gradcheck`.
- **Policy object.** Put the policy in a `PrecisionPolicy(real=float32, complex=complex64, special=float64, special_device=None)`, never in `torch.set_default_dtype`.

---

## 5. Special functions and numerics

### 5.1 What torch.special gives you [M/V]

- **Real-only.** `torch.special.bessel_j0/j1/y0/y1` and `spherical_bessel_j0` accept **real inputs only**: complex input fails with "not implemented for 'ComplexFloat'" [M].
- **No autograd in 2.14.** Their outputs have no `grad_fn` in 2.14 [M]. `main` now carries derivative formulas for `bessel_j0/j1/y0/y1` and `modified_bessel_*`, but `spherical_bessel_j0` is still `non_differentiable` [V: `tools/autograd/derivatives.yaml` on main, Sept 2026]. Expect this to land in 2.15; do not depend on it.
- **Missing entirely:**
  - spherical Bessel/Hankel of order n
  - complex arguments
  - Zernike polynomials
  - associated Legendre functions
- The psf-generator authors hit the same wall and hand-wrote differentiable J₀/J₁/J₂ [V: https://pmc.ncbi.nlm.nih.gov/articles/PMC12946858/].
- **Consequence:** gradoscopy owns a small `special` module: fp64, batched over (particles × wavelengths), with order recurrences written in torch ops.

### 5.2 Mie (and the road to T-matrix)

**Algorithm [R, standard: Bohren & Huffman 1983; Wiscombe 1980]:**
- Truncate at `n_max = x + 4x^{1/3} + 2`.
- Compute the logarithmic derivative D_n(mx) by **downward** recurrence `D_{n-1} = n/z − 1/(D_n + n/z)`, starting from `n_start = max(n_max, |mx|) + 15` with D=0.
- Compute ψ_n(x) and ξ_n(x) upward. For real x, upward recurrence of ψ_n is stable only while n ≲ x. Use downward recurrence plus normalisation to j₀ beyond that.
- Then a_n and b_n follow from the standard D_n formulas.

**PyMieDiff** is the reference implementation [V: https://arxiv.org/html/2512.08614v1, APL Photonics 2026]:
- Pure-torch core-shell Mie, with downward recurrence for j_n and upward for y_n, plus a continued-fraction fallback.
- Autograd goes through the recurrences natively.
- Its authors say "the recurrences currently require to be performed at double precision".
- It is slower on an RTX 4090 than on CPU, because of FP64 throughput.
- It is only competitive (10×+) when batched over particles × wavelengths.
- Stated weaknesses: instability for very large or plasmonic particles, and vector spherical harmonics only for l=1.

**My measurements [M]:**
- complex64 D_n errs by up to **1–5% relative** for real m (1.33, 1.5) over x ∈ [1, 50], but only 4×10⁻⁷ for strongly absorbing m. **Coefficients must be fp64.**
- A batched D_n recurrence over P=4096 particles with n_start=48 takes about 3 ms on GPU in c64 or c128 alike, because it is launch-bound. CPU takes 3–5 ms.
- **Keep coefficients on whichever device the policy picks, and batch everything into one call.** Capture the call in a CUDA graph if it is on the hot path.

**Autograd strategy:**
- Native autograd through the recurrences is fine for the coefficients: arrays of size P × n_max, with ~100 steps. The tangent-linear recurrence of a stable recurrence is itself stable.
- The memory killer is the *field synthesis*. Evaluating Σ_n over n_max orders at every pixel keeps P × H × W × n_max intermediates alive: 10 particles × 512² × 50 orders × 8 B ≈ 1 GiB per image.
- **Avoid it by exploiting symmetry.** For spheres, S₁(θ) and S₂(θ) depend only on θ. Evaluate them on a 1-D θ grid (≈10³ samples) with the π_n/τ_n upward recurrences (stable), then interpolate differentiably onto the 2-D pupil. Place each particle with a Fourier phase ramp, and do one FFT per image.
- The per-particle cost becomes O(n_max·N_θ + H·W) instead of O(n_max·H·W).
- Use a custom backward, built on d/dz f_n = f_{n−1} − (n+1)/z·f_n, only if profiling shows the coefficient graph matters.

**T-matrix:** the same structure applies (coefficients in fp64, small; fields via far-field or pupil synthesis). It is a later tier. Mature CPU codes exist for validation, e.g. `treams` and `smuthi` [R].

### 5.3 Zernike polynomials

- **Indexing.** Use Noll indexing [R: Noll 1976], with an ANSI alternative.
- **Building the basis.** Construct radial polynomials in fp64 with the explicit factorial sum up to n≈15. Beyond that, cancellation sets in, so switch to a recurrence [R: Andersen, Opt. Express 26, 2018].
- **Storage.** Store the basis as a registered buffer `[J, H_pupil, W_pupil]`: 36 modes × 512² × 4 B ≈ 38 MB.
- **Evaluation.** The pupil phase is `einsum("bj,jhw->bhw", coeffs, basis)`, which is trivially differentiable in the coefficients. Gradients with respect to NA or pupil radius require regenerating the basis on the fly from ρ(NA). It is cheap as long as n stays small.

### 5.4 Interpolation and resampling

- **`grid_sample` rejects complex input** ("grid_sampler_2d_cpu_kernel_impl not implemented for 'ComplexFloat'") [M]. The feature request, issue #67634, has been open since 2021 [V].
  - Workaround [M]: stack `[z.real, z.imag]` as channels, sample, and rebuild with `torch.complex`. This is exact for (bi/tri)linear interpolation.
  - Caveat [R]: `grid_sampler` backward uses atomics and is **nondeterministic on CUDA**.
- **Fourier shift theorem.** Sub-pixel placement multiplies spectra by `exp(−2πi k·r)`. It is differentiable in r and passes `gradcheck` [M]. Rendering all particles as `F = Σ_p A_p·T_p(k)·exp(−2πi k·r_p)` followed by one iFFT gives exact analytic position gradients and no sampling bias. It is periodic, so pad the field by the object support. It is the preferred placement primitive.
- **Band-limited up/downsampling** is done by zero-padding or cropping spectra.
- **Arbitrary output pixel sizes, e.g. pupil → camera.** Use the matrix Fourier transform (two complex matmuls) [R: Soummer et al. 2007] or the chirp-z transform. psf-generator uses a CZT at the cost of "a convolution and three FFTs" [V: PMC12946858]. The MFT needs IEEE-fp32 matmuls, not TF32 (§2.4).

---

## 6. Differentiating through randomness

### 6.1 Silent-zero pitfalls [M, V]

- `torch.poisson(λ)` returns **zero gradient** without any error: `derivatives.yaml` has `self: zeros_like(self)`.
- `torch.normal(mean_tensor, std_tensor)` also returns **zero** gradients for both mean and std.
- `bernoulli` does the same.

A naively written noise model therefore makes learned noise parameters silently stay put. **Rule:** all library randomness goes through gradoscopy's own sampling helpers, each with an explicit gradient estimator. A test asserts non-zero gradients for every learnable noise or distribution parameter.

### 6.2 Reparameterisable distributions [V]

- **`has_rsample=True`:** Normal, LogNormal, Uniform, Gamma and Beta (implicit reparameterisation), Dirichlet, Exponential, Laplace, Cauchy, StudentT, VonMises, Kumaraswamy, RelaxedBernoulli, RelaxedOneHotCategorical and TransformedDistribution. See https://docs.pytorch.org/docs/2.14/distributions.html.
- **Poisson has no `rsample`.** The source samples under `no_grad` [V: `torch/distributions/poisson.py`], and I confirmed `Poisson.has_rsample == False` [M].

### 6.3 Poisson and camera noise options

| estimator | forward value | gradient dy/dλ | when to use |
|---|---|---|---|
| Gaussian reparam `λ + √λ·ε` | approximate (continuous) | 1 + ε/(2√λ) | high counts (λ ≳ 20) |
| plain straight-through `λ + (n−λ).detach()` | exact Poisson | 1 (mean only) | quick default; ignores the noise-level dependence |
| **scaled ST** `λ + √λ·((n−λ)/√λ).detach()` | **exact Poisson** | **1 + ε/(2√λ)** [M] | recommended default: exact samples with Gaussian-consistent gradients |
| Anscombe domain `2√(y+3/8)` | variance-stabilised | — | use in *losses*, not in the sampler |
| score function `f(n)·(n/λ − 1)` | exact | unbiased, high variance | low-count regimes, or when learning noise statistics precisely |

Other camera stages:
- **Read noise:** Gaussian, reparameterised.
- **EMCCD gain:** Gamma(n, g), with an implicit rsample conditional on n.
- **Quantisation:** `round` with straight-through.
- **Saturation:** clamp, optionally a soft clamp for learning.

### 6.4 Discrete choices

- **Particle counts.** Use a padded slot tensor `[B, P_max, …]` plus a presence mask. For learning the count distribution, the mask can be a RelaxedBernoulli/Concrete sample with temperature τ, straight-through-hardened in the forward pass. That gives gradients with respect to presence probabilities.
- **Categorical choices** (particle type, material): Gumbel-softmax, or REINFORCE with a baseline. Collect `log_prob` for all non-reparameterised draws during resolution, so a score-function surrogate `Σ log_prob · stop_grad(loss − baseline)` can be added automatically.
- **Fidelity choices are not random variables.** Do not differentiate through them. What is useful is **surrogate gradients across fidelities**: `y = y_hi.detach() + y_lo − y_lo.detach()`. The values come from the high-fidelity model and the gradients from the cheap one, as an option for parameter fitting.

### 6.5 Learnable data-generation distributions

- **How gradients reach the distribution.** When a distribution's parameters are `nn.Parameter`s and sampling uses `rsample`, gradients flow from any image-space loss. Constrain them with `torch.nn.utils.parametrize` or with softplus/sigmoid transforms.
- **What loss to use.** The loss is usually distributional (sim-to-real), for example:
  - MMD or sliced-Wasserstein on image features;
  - a discriminator;
  - moment matching of intensity histograms and noise power spectra.
- **Budget.** Treat this as a first-class use case, but budget-conscious. It requires the *fast* tiers.

---

## 7. API design trade-offs

### 7.1 What prior art does

- **Chromatix (JAX) [V: https://chromatix.readthedocs.io/en/latest/101/, /training/].**
  - Elements are Equinox modules. A `Field` has shape `(... H W [λ] [pol])`, and ScalarField/VectorField are distinct types.
  - Trainability comes from an `eqx.partition` filter; static fields are discouraged.
  - Shot noise needs an explicit PRNG key.
  - It reports 2–6× single-GPU speed-ups over the *original* PyTorch implementations of several published methods [V: https://pmc.ncbi.nlm.nih.gov/articles/PMC13042145/]. Much of that plausibly comes from XLA fusing complex elementwise ops, which Inductor cannot do (§3.1).
- **TorchOptics [V: https://github.com/MatthewFilipovich/torchoptics].** `nn.Module` elements composed in a `System`. Trainable parameters are declared by passing `nn.Parameter` into constructors.
- **psf-generator [V].** Propagator classes (`ScalarCartesian`, `VectorialSpherical`, …) with a `(z, channel, x, y)` layout.
- **waveorder [V: https://arxiv.org/abs/2412.09775].** A functional, transfer-function operator composition.
- **DeepTrack2 [V: https://github.com/DeepTrackAI/DeepTrack2].** Features + Properties (Python-callable sampling), with `deeptrack/backend` and `deeptrack/pytorch` subpackages, and numpy/torch backend switching. Properties are arbitrary callables, which are non-differentiable by default.

### 7.2 Recommendation: functional core, Module shell, resolve-then-render

**Layer 1, functional core.** Pure functions on tensors with explicit metadata, e.g. `angular_spectrum(u, dz, wl, dx, *, pad, policy)`:
- no hidden state;
- trivially testable with `gradcheck` and usable with `torch.func`.

**Layer 2, components as `nn.Module`s.** Modules give `.to()`, `state_dict`, `parameters()` for optimisers, DDP wrapping and familiarity to DeepTrack and PyTorch users. Each forward delegates to Layer 1. Every scalar or tensor attribute is declared through one `Param` spec:
- `Fixed(v)`, stored as a buffer;
- `Learnable(init, constraint=positive|interval|…)`, an `nn.Parameter` plus a parametrization;
- `Sampled(dist)`, where `dist` is a torch `Distribution` whose own parameters may be any of these kinds, recursively;
- `Derived(fn, *deps)`, the DeepTrack-style dependency, differentiable when `fn` is torch code;
- `Sampled(callable)`, a non-differentiable escape hatch for legacy DeepTrack properties.

**Two-phase execution** (the key decision):
1. `scene = pipeline.resolve(batch_size=B, generator=g)` walks the Param graph. It samples every stochastic quantity (with `rsample` where possible) and returns a flat, typed `Resolved` container of tensors shaped `[B, …]`. It also carries `masks`, `log_prob` terms for score-function estimators, and ground-truth labels.
2. `image = pipeline.render(scene, fidelity=…)` is a deterministic function of `scene` and the fixed or learnable parameters, apart from detector noise, which draws from `g`.

The split gives four things at once:
- labels for free;
- exact replay and caching;
- `functional_call` and `vmap` compatibility;
- a clean place to insert REINFORCE terms.

**Batching semantics.**
- Every resolved quantity has a leading B dimension. Size 1 means shared, and broadcasting is allowed.
- Variable object counts are padded to `P_max` with float masks. Do not use `torch.nested`: its jagged layout does not cover FFT or most of the ops needed [R].
- Per-sample optics (different NA or defocus per sample) is just `[B]` parameters. Transfer functions must be generated from broadcasted parameter tensors, not cached per scalar.

**Field container.** A small dataclass (tensor + dx + λ + medium index + polarization basis), registered as a pytree. Not a `torch.Tensor` subclass: subclasses complicate autograd.Function, compile and vmap.

**Device and dtype.**
- A `SimContext` carries the device, the `PrecisionPolicy`, the grid policy (nice FFT sizes, padding factor) and the `torch.Generator`.
- Components register only buffers and parameters, so `.to()` works.
- Complex dtypes are derived from the real dtype, never hard-coded.
- There are no global `set_default_dtype` calls.

### 7.3 Determinism and seeding

- **Explicit generators.** Always pass explicit `torch.Generator(device=…)` objects. Seed per batch as `hash(base_seed, epoch, batch_idx, rank)`, so results do not depend on worker count.
- **Per-image reproducibility** comes from storing the `Resolved` scene. It is small, and it is needed as labels anyway. Re-seeding per image would kill batching.
- **Nondeterministic CUDA ops [R].** These include `index_add_`/`scatter_add_` and `index_put_(accumulate=True)` (used for splatting point emitters), and the `grid_sample` backward.
- **Deterministic mode.** Offer `deterministic=True`, which calls `torch.use_deterministic_algorithms(True)` and swaps splatting for sort-based or FFT-based rendering.
- **Checkpointing** must keep `preserve_rng_state=True` [V].

### 7.4 Data loading and multi-GPU

- **Generate on the GPU in the main process.** Use a plain iterator, or an `IterableDataset` with `num_workers=0`.
- **Avoid CUDA in DataLoader workers.** Each worker needs a spawn-started process and its own CUDA context of several hundred MB [R], and it adds nothing when the data is born on the GPU.
- **CPU workers are only for CPU-side geometry.**
- **Multi-GPU.** Rank-local generation plus DDP (NCCL on Linux) for joint training. Simulation is embarrassingly parallel over the batch, so model-parallel simulation is out of scope.

---

## 8. Testing pyramid

1. **Analytic unit tests (fp64):**
   - Gaussian-beam propagation vs the closed form.
   - Airy disk vs 2J₁(v)/v.
   - Parseval and energy conservation for unitary propagators and phase-only slices. My experiment exposed that an energy-type loss has an exactly zero gradient there, which makes it a good invariant test.
   - Fourier shift vs integer `roll`.
   - Zernike orthonormality on the unit disk.
   - Mie Q_ext/Q_sca and S₁/S₂ vs scipy-based references (`miepython`, PyMieScatt, PyMieDiff) across dielectric, absorbing, large-x and core-shell cases.
2. **Gradient tests:**
   - `torch.autograd.gradcheck` in complex128 on every functional op and custom Function. The full mode took 0.99 s for a 2×16×16 complex op and `fast_mode=True` took 0.01 s [M]; use full mode on tiny shapes and fast mode in bulk.
   - `gradgradcheck` where second order matters.
   - Adjoint-vs-autograd equality (target ≤1×10⁻⁵ in fp32 [M]).
   - Non-zero-gradient assertions for every Learnable and Sampled parameter.
3. **Cross-fidelity consistency:** each approximate tier must converge to the reference within its declared validity regime. Examples: Born/Rytov multislice vs Mie for weak spheres, PSF-convolution vs multislice for thin samples. Tolerance tables live beside the tier registry.
4. **Device and dtype parity:** CPU vs CUDA vs MPS (c64 only); c64 vs c128.
5. **Statistical tests:** KS or moment tests on sampled parameters and on noise (Poisson mean = variance), and seeded reproducibility.
6. **Learnability end-to-end:** recover defocus, NA, Zernike coefficients, radius and RI, and distribution means from synthetic data within tolerance.
7. **Performance and memory benchmarks:** record `max_memory_allocated` and time per tier on fixed shapes; run on a self-hosted GPU runner (this 3090); fail on regressions of more than 20%.

---

## Implications for gradoscopy's architecture

1. **Language.** Pure Python on PyTorch ≥2.12, with no Cython and no C++ extension in v1. Optional accelerators, each with a pure-torch reference fallback:
   - Triton kernels via `torch.library.triton_op` (needs `triton-windows` on Windows);
   - `torch.compile` on real-view step functions;
   - later, nvmath-python FFT callbacks.

   Correctness must never depend on compilation. Triton is missing by default on this very machine.
2. **Dtype policy.** complex64/float32 everywhere. fp64 only for special-function coefficients, phase-argument construction (carrier-subtracted, mod 2π) and gradcheck. Ban complex32, fp16 and bf16 from the physics path. Run internals under `autocast(enabled=False)` with IEEE fp32 matmuls. Pad grids to 2·3·5·7-smooth sizes and use rfft for real pipelines.
3. **Architecture split.**
   - A functional physics core.
   - `nn.Module` components whose every attribute is a `Param` (Fixed, Learnable, Sampled or Derived).
   - A two-phase **resolve → render** execution model with a typed, batched `Resolved` scene (labels, masks, log-probs).
   - Fidelity is a render-time dispatch over operator implementations that declare validity regimes and memory/compute cost estimates. Fidelity is not a property of the scene description.
4. **Memory as an architectural feature.** Encode these in the operator interface, not as user tricks:
   - Samples are **procedural or parametric by default**, and slices are generated on the fly. A materialised 64×256×512² volume (16 GiB plus 16 GiB of gradient) is the real memory wall.
   - Every multi-step propagator ships a hand-written **adjoint `autograd.Function`**, reversible or segmented-recompute, with in-loop VJP into sample parameters.
   - Every incoherent, spectral or polarisation sum uses **exact chunked checkpointed summation**, with chunk sizes from a memory estimator.
   - `inference_mode` plus in-place fast paths whenever nothing requires grad.
5. **Fidelity tiers sized by measured cost.** A 3090 does ~2.5–3 ms per slice step for 64×512². Full partially-coherent multislice (256 slices × 200 modes) is minutes per batch, so it belongs to validation and fitting. Training-data tiers must stay in the 10–100 ms per batch range (PSF convolution, thin-object or projection, few modes), with surrogate-gradient bridging between tiers available.
6. **Randomness.** Library-owned samplers with explicit estimators:
   - `rsample` wherever it exists;
   - scaled-straight-through Poisson by default;
   - Concrete masks for counts;
   - automatic REINFORCE bookkeeping for non-reparameterisable draws.

   Guard with tests against the silent zero gradients of `torch.poisson` and `torch.normal`.
7. **Batching and devices.** A leading batch dimension everywhere; padding plus masks for ragged particle sets; per-sample optics parameters as `[B]` tensors. GPU-side generation in the main process with explicit `torch.Generator`s. Linux-only multi-GPU via DDP. MPS as best-effort complex64, with fp64 kernels routed to CPU.
8. **Own a small `gradoscopy.special`.** It needs Mie coefficients (fp64, batched, D_n downward and ψ/ξ with a stable-direction switch), Zernike (Noll, recurrence) and complex-safe interpolation. Borrow algorithms and tests from PyMieDiff. Synthesise sphere fields via a 1-D angular function interpolated onto the pupil, never with per-pixel order sums.
9. **Testing is part of the architecture.** Every operator is registered with its analytic references, gradcheck shapes, cross-fidelity tolerance and benchmark shape. The registry doubles as the fidelity dispatch table.
