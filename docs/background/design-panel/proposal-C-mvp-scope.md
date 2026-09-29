# Proposal C — "Seams first, solvers later": a minimum viable architecture for gradoscopy

**Angle:** a skeptical tech lead with a team of 1–3 developers. The goal is the smallest architecture that serves most DeepTrack2 users soon, is differentiable and GPU-native from day one, and has clean seams so it can grow without rewrites. The proposal says explicitly what not to build, which abstractions are premature, and which seams must be right in v0 because they are expensive to change later.

**Evidence tags**
- **[B01]…[B09]** cite the nine research briefs.
- **[M-C]** marks numbers I measured for this proposal on the target machine (RTX 3090, Windows 11, torch 2.14.0+cu130). The scripts are in `scratchpad/propC/`: `bench_v1_tiers.py`, `bench_batching.py`, `bench_abbe.py` and `units_check.py`.
- Everything else is engineering judgement.

---

## 0. The proposal on one page

1. **Scope.** v1 does the two DeepTrack2 workhorses properly [B01 §f, B08 §3]:
   - coherent imaging of Mie spheres (holography, brightfield, darkfield, iSCAT-lite);
   - incoherent fluorescence of point emitters and primitives.

   It adds thin and multislice phase objects (cells), Köhler partial coherence, a real camera model with likelihoods, learnable scene distributions, and a DeepTrack2 adapter. Each item beyond that is scheduled for v2 or later and has a named application trigger.
2. **Spine.** `Spec → sample() → Trace → render(plan) → Batch`.
   - The spec is declarative: frozen dataclasses whose leaves are `Param`s.
   - The trace is a small, flat, fidelity-independent record of tensors. It serves as the renderer input, the ground truth and the replay record.
   - Fidelity is an argument to rendering. It is never a property of the scene.
3. **Two currencies instead of two ports.**
   - Coherent contributions meet as a scattered angular spectrum on one pupil k-grid shared by all wavelengths, referenced to the focal plane z = 0. The reference or unscattered field is carried analytically alongside it.
   - Incoherent contributions meet as irradiance on one object-space grid.
   - All paths evaluate the same `Pupil` function.
   - This interface is the one thing I would defend against every later rewrite pressure.
4. **The optical path has three planes:** sample (slices), pupil and camera, all in object-space coordinates. Magnification is a coordinate scale. Condensers, phase rings, stops, aberrations and engineered masks are all functions on a pupil.
5. **Fidelity is a frozen dataclass of about 10 named knobs with three presets** (`draft`, `standard`, `accurate`). A rule table picks the kernels when the `Simulator` is constructed, and `explain()` prints the choice. Validity is checked against parameter *bounds*. There is no cost model, no search and no operator algebra.
6. **Parameters use five leaf kinds:** `Fixed`, `Learnable`, `Random`, `Derived` and `Count`.
   - Learnable values are stored at O(1) scale.
   - Each random site declares its gradient estimator.
   - Randomness comes from explicit generators.
   - This is our own implementation of about 600 lines. It does not depend on pyro or tensordict.
7. **Units and precision.** µm everywhere, complex64 everywhere, fp64 only for Mie coefficients and phase-argument construction. v1 uses scalar polarization but reserves the polarization axis. All wavelengths share one grid. Time is folded into the batch axis.
8. **Measured fast tiers [M-C]** on the 3090:
   - sparse emitters: 52k images/s forward, 10k images/s with gradients;
   - Mie holography: 2.8k images/s;
   - dense fluorescence: 7.3k images/s;
   - phase contrast with 25 source points: about 400 images/s with gradients;
   - multislice cell: 41 images/s with gradients at 0.88 GiB peak.
9. **Roadmap.**
   - M0 spine → M1 fluorescence → M2 DT2 adapter → M3 Mie → M4 volumes and partial coherence → M5 v1.0.
   - About 20 weeks for two developers. Every milestone has numeric exit criteria.

```
 SPEC (what)                  TRACE (sampled tensors)           PLAN (how; static, cached)
 Scene, Microscope    sample  path -> Tensor[B,...]    plan     grids, pads, kernels,
 leaves: Fixed,     ────────▶ presence masks, log_probs ──────▶ chunking, validity report
 Learnable, Random,           (fidelity-independent)            (rule table, explain())
 Derived, Count                        │                                │
                                       └────────────┬───────────────────┘
                                                    ▼ render(trace, plan)
      coherent path                                   incoherent path
      Mie/dipole (sparse) ─┐                          point emitters (sparse, MFT-ROI) ─┐
      projection/multislice├─ Σ A(k) @ z=0 on k-grid  emission volume (dense, per-z OTF)├─ Σ I(r)
      (dense)             ─┘   × P(k;λ) → IFT          (both use the same P(k;λ))       ─┘
      + analytic reference → |E_r|² + 2Re(E_r*E_s) + |E_s|²   Σ over modes M, wavelengths W
                                                    ▼
                           Camera: pixel-integrate → expected photons →
                           sample (scaled-ST Poisson) | log_prob → ADU      + labels from Trace
```

---

## 1. Guiding principles

1. **Serve the two workhorses first.** Coherent imaging of Mie spheres and incoherent fluorescence of points and primitives account for most DeepTrack2 simulation use [B01 §f]. A third large group uses DeepTrack2 only for data plumbing. Every other feature has to name the application that needs it (§2), or it waits.

2. **Seams before solvers.** A small set of decisions is expensive to change later:
   - units and conventions;
   - tensor layout;
   - the spec → trace → render split;
   - the two currencies;
   - `Param` semantics;
   - the detector API.

   These are frozen in M0. Everything behind them may be naive in v1: planner, kernels, caching, acceleration.

3. **One batched call per kernel kind per batch.** There is no per-sample Python on the hot path. Batching speeds up Mie holography by 33× over a per-sample loop (713 ms vs 21.7 ms for 64 images [M-C]), and that is before DeepTrack2's own graph overhead.

4. **Differentiable by construction, or visibly not.** No hard masks: they give zero radius and position gradients [B06 M]. No bare `torch.poisson`, `torch.normal(tensor, tensor)` or `bernoulli`: all three return silent zero gradients [B07 M]. No caches that detach tensors [B02]. Every learnable leaf gets an automated non-zero-gradient test.

5. **Fidelity is a render-time argument.** `upscale`, `padding`, `dz`, source-point counts and wavelength bins belong to the plan, never to scene objects [B09 §3.4]. The same trace renders at every fidelity, which is what makes fidelity comparisons meaningful.

6. **A readable plan beats a clever planner.** v1 uses a rule table plus `explain()`, in about 300 lines [B09 §4.5]. A cost model or search is added only when three or more implementations compete for one slot *and* users report wrong automatic choices.

7. **Tensor shapes come from declared bounds, not from values.** Grid sizes, padding, ROI sizes and slice counts are derived at plan time from distribution supports and declared `bounds=`. A learnable defocus then cannot change the padding mid-training [B05 §8.4], and CUDA graphs and compilation stay possible.

8. **Pure PyTorch, no required compiler.** Triton is not installed by default on this machine, and Inductor does not generate code for complex operations [B07 §1, §3.1]. Correctness must never depend on `torch.compile`, Triton or CUDA graphs. They are optional accelerators, each with a pure-torch reference.

9. **Measure before accelerating.** A custom adjoint, Triton kernel or CUDA-graph path is added only when a benchmark in the regression suite shows it is needed. The suite is committed in M0.

10. **Adapter first.** DeepTrack2 compatibility is a v1 deliverable with parity tests (M2), not a port done afterwards [B01 §implications 9].

---

## 2. Scope

### 2.1 v1: must ship

| Capability | Justification (applications) |
|---|---|
| **Point emitters, sparse path**: pupil PSF evaluated per emitter on a region of interest (ROI) by matrix DFT (MFT), then scatter-add; Gaussian sprites as the draft tier | SMLM (DECODE-like), fluorescent particle tracking, PSF engineering (DeepSTORM3D, DTEx252) [B08 §2.4, B01 §f] |
| **Primitives as soft band-limited occupancy**: Sphere, Ellipsoid, Capsule, Composite (parts plus local transforms), user `Voxels`, user `SDF` | Bacteria and cell fluorescence (DTGS172), phase objects, masks as labels [B06 §3] |
| **Dense emission** by per-z optical transfer function (OTF) convolution | DeepTrack2 `Fluorescence` parity; labelled cells [B05 §1.2a] |
| **Scalar pupil**: soft aperture, exact defocus, Zernike (optical path difference (OPD) in µm), pixel pupil (amplitude and phase), apodization, Gibson–Lanni OPD, phase ring, stops, DIC shear | Aberration calibration, astigmatic SMLM, phase contrast; all pupil-learning applications [B08 §6.3] |
| **Mie, homogeneous sphere**: 1-D S₁/S₂(θ) interpolated onto the pupil; Rayleigh dipole as the draft tier; layered sphere in v1.x | Holographic characterization, 2D/3D tracking (DTGS121, DTEx203, DTEx205), CATCH-like data [B08 §2.1–2.2] |
| **Coherent volumes**: projection (thin sample) and multislice (free-space angular spectrum, procedural slices, checkpointed) | Phase contrast, quantitative phase imaging (QPI) and brightfield of cells; DeepTrack2 `Brightfield` parity [B08 §2.5–2.6] |
| **Illumination**: on-axis and oblique plane waves; Köhler source as K modes (disc, annulus); unpolarized light as 2 modes; a few-wavelength spectrum | Brightfield realism, phase-contrast halo, darkfield, DTGS106 white light [B05 §2] |
| **Reference handling**: in-line (unscattered), blocked (darkfield), reflected with phase (iSCAT-lite, homogeneous medium); explicit interference terms | Holography, darkfield, iSCAT; float32 cancellation hazard [B08 §2.3] |
| **Camera**: pixel integration, QE, gain, offset, read noise (scalar or per-pixel map), EMCCD, quantization, saturation; `expected` / `sample` / `log_prob` | Needed by every application; system identification needs likelihoods [B05 §4] |
| **Parameters**: `Fixed`, `Learnable`, `Random` with learnable distributions, `Derived`, `Count` (relaxed presence); explicit generator; `condition` and `replay` | Sim-to-real, inverse problems, calibration [B08 §6] |
| **Labels**: trace values, positions (µm and pixels), instance masks, noise-free image, complex field, optical path length (OPL) map | DeepTrack2's paired-data value proposition [B09 §2.8] |
| **Brownian trajectories helper**, time folded into the batch axis | Sequences for tracking (LodeSTAR, MAGIK training data) [B01 §a] |
| **Fidelity presets, `explain()`, validity warnings, `compare()`** | The "tunable fidelity" requirement |
| **DeepTrack2 adapter and batched dataset** | Becoming DeepTrack2's engine |

### 2.2 v2: planned, each with a trigger

| Capability | Trigger / justification |
|---|---|
| Vectorial pupils (Jones, P = 2), 6-basis dipole images, Richards–Wolf | High-NA system identification and fixed-dipole SMLM. Gaussian or scalar fits of fixed dipoles bias localization by up to about 40 nm [B03 §2.2]. Polarization microscopy |
| Stratified media: coverslip Fresnel coefficients, full iSCAT iPSF, TIRF evanescent excitation, supercritical-angle fluorescence (SAF) | Mass photometry and TIRF [B08 §2.3] |
| Mixed-sample coupling C1: Mie injected into the multislice march; fluorescence through a refractive-index (RI) volume | Beads inside cells, tissue [B04 §5] |
| Born/Rytov (with reflection port), SSNP mode, reversible multislice adjoint, angle mini-batching | Optical diffraction tomography (ODT) and weak-object reconstruction [B08 §2.6] |
| SOCS/WOTF fast paths | Only if Abbe with fixed optics becomes a bottleneck [B05 §1.2b] |
| Photophysics (blinking Markov model, bleaching), exposure motion blur, SIM, light-sheet and confocal excitation patterns | SMLM realism, ML-SIM, light-sheet data [B08 §2.7, §2.10] |
| Space-variant PSFs: per-emitter Zernike maps (cheap on the sparse path, v1.x); low-rank dense | FD-DeepLoc/uiPSF-style calibration [B03 §2.3] |
| MFT/CZT detector sampling with continuous pixel size | Learnable magnification and pixel size |
| Meshes (via SDF), uniaxial ε tensors, deepinv export, TensorDict interop, entry-point plugins | Imported cell shapes, polarization tomography, reconstruction reuse |

### 2.3 Later (v3 and plugins)
- Modified-Born-series / Lippmann–Schwinger reference solver.
- Multi-sphere Foldy–Lax coupling and T-matrix import.
- biobeam-class light-sheet in tissue.
- Learned surrogate tiers trained from higher tiers.
- Ray-traced eikonal phase screens.
- Whole-sensor Fourier-ptychography (FPM) tiling.

All of these are legitimate [B04 §6], and none is needed by the majority of DeepTrack2 users.

### 2.4 Explicitly out of scope
- **FDTD, FEM and DDA inside the engine.** No data-generation application needs them online [B08 §5]. Precomputed far-field tables can be imported later instead.
- **Coherent nonlinear optics (SHG, CARS, SRS) and Raman.** No surveyed pipeline depends on them.
- **FLIM and photon timing.**
- **Monte-Carlo radiative transfer.** Phenomenological haze covers it.
- **Lens-design ray tracing.** Objectives are ideal aplanatic systems plus pupil aberrations [B08 §5].
- **Electron and X-ray imaging, biomechanics** (external generators), **instrument control**.
- **A NumPy backend.** DeepTrack2 already carries one. gradoscopy is torch-only [B01].

### 2.5 Abstractions deliberately *not* built in v1 (premature)

| Not building | Why it is premature | Seam kept so it can be added later |
|---|---|---|
| Cost-model planner, conversion-graph search [B06 §4.3, B09 §2.2] | v1 has at most 2–3 kernels per slot. A dictionary lookup is the whole planner | Kernels declare `handles`, `validity` and `grad`. The plan is a list of named stages |
| Operator algebra with type-checked composition and rewrite/fusion passes [B09 §2.3] | The pipeline skeleton is fixed. Fusing pupil multipliers is one `prod()` | Pupil modifiers are a list. Stages are plain callables |
| Two-port S-matrix and Redheffer star products [B04 §4] | Only iSCAT needs reflection in v1, and it needs one backward port with an analytic reference | `direction = ±1` on the pupil spectrum |
| Pyro-style effect-handler stack [B09 §1.10] | `condition=` and `replay` as arguments cover every v1 use case | The trace is the single sampled-value record |
| A generic `IncoherentSum` node with stochastic and analytic estimators [B02 R6] | v1 needs only exact chunked sums | The mode axis M and its weights exist in every coherent tensor |
| Triton kernels, `torch.compile` integration | Unavailable on Windows by default, no complex codegen [B07 §3.1]. Unmeasured need | Kernels are pure functions with static configuration, so they are compile-friendly later |
| Entry-point plugin discovery, pydantic schema, serialization format | No third-party plugins exist yet | Decorator registry. Specs have `to_dict()` with registry names |
| Counter-based per-sample RNG | Batch-level seeding plus stored traces cover reproducibility | Seeds are derived by a function in one module |
| Vectorial optics | 1.5–6× cost; required only for high-NA system identification and fixed dipoles | The P axis is present with size 1 |

---

## 3. Layered architecture and responsibilities

### 3.1 Layers

```
 L5  interop/deeptrack  (DT Features, BatchedDataset)      fit/ (losses, regularizers; v1.x)
       │ imports L4 + specs
 L4  render/   Fidelity · plan (rule table, bounds, validity, explain) · Simulator(nn.Module)
               · labels · compare                     ◀── the ONLY layer that binds specs to kernels
       │ imports everything below
 L3  scene/ optics/ presets.py   (WHAT: frozen dataclasses with Param leaves)
     physics/                    (HOW: pure functions on tensors + Grid; never see Params)
       │ specs import L2+L1          physics imports L1 only
 L2  params/   leaves · dist · sampler · Trace
       │ imports L1
 L1  _core/    units · conventions · Grid/KGrid · PupilSpectrum/Irradiance containers ·
               precision · fft utils · special (Mie, Zernike) · estimators · registry
```

| Layer | Owns | Must not |
|---|---|---|
| `_core` | Units and conventions; `Grid` (static ints and floats); containers; FFT sizing; fp64 phase helpers; special functions; gradient estimators at tensor level (scaled-ST Poisson, ST-round); the kernel registry | import anything from gradoscopy |
| `params` | Leaf types, distribution wrappers, `sample(spec, batch, generator) → Trace`, bounds extraction | know about optics |
| `scene`, `optics`, `presets` | Declarative descriptions: geometry, material, emission, medium; illumination, objective, pupil modifiers, camera; modality presets | import `physics`; hold numerical knobs such as grid, padding or `dz` |
| `physics` | Every numerical kernel: pupil evaluation, MFT-ROI emitters, rasterization, Mie, projection, multislice, Abbe, per-z OTF, detector | import `params` or the specs. Receives tensors, `Grid`s and static configuration only |
| `render` | Planning, validity, `Simulator` (owns learnable parameters and buffers), label renderers, `compare` | contain physics formulas |
| `interop` | DeepTrack2 mapping, batched dataset | be imported by any lower layer |

A 30-line test (`tests/test_layering.py`) parses imports and fails the build if a rule is broken. `grep`-style tests also forbid `torch.manual_seed`, `np.random` and bare `torch.poisson` under `src/`.

### 3.2 Package tree (v1)

```
src/gradoscopy/
  __init__.py            public API re-exports (Scene, Population, Sphere, …, Simulator, Fidelity)
  units.py               nm, um, mm, deg; from_si(); to_pixels()
  _core/
    conventions.py       axis order, Fourier sign, z direction, Zernike indexing (one file, tested)
    grid.py              Grid, KGrid, VolumeGrid, next_smooth(), padding rules
    containers.py        PupilSpectrum, Reference, Irradiance (frozen dataclasses)
    precision.py         PrecisionPolicy, phase_arg() (fp64 → mod 2π → fp32), autocast guard
    special.py           mie_ab() fp64 (downward D_n), pi_tau(), zernike basis (ANSI/Noll)
    estimators.py        poisson_scaled_st, round_st, relaxed_presence, score_surrogate
    registry.py          kernel/geometry/modifier registries (decorators)
  params/
    leaves.py            Fixed, Learnable, Random, Derived, Count, Choice
    dist.py              reparameterized wrappers accepting leaves as args (+ bounds())
    sampler.py           sample(spec, batch_shape, generator, condition=, mode=) -> Trace
    trace.py             Trace (flat path->Tensor, batch_shape, masks, log_probs, meta)
  scene/
    scene.py             Scene, Population, Medium
    geometry.py          Point, Sphere, Ellipsoid, Capsule, Composite/Part, Voxels, SDF
    material.py          Material(n, dispersion), Fluorophore(photons, spectrum)
    texture.py           GaussianRandomField
    placement.py         UniformBox, InsidePart, Brownian (trajectory helper)
  optics/
    illumination.py      PlaneWave, Koehler(Disk|Annulus), Spectrum, Excitation(uniform|fn)
    objective.py         Objective(NA, magnification, immersion_n), Pupil(*modifiers)
    modifiers.py         Aperture, Defocus, Zernike, PixelPupil, Apodization, GibsonLanni,
                         PhaseRing, CentralStop, DICShear
    reference.py         InlineReference, BlockedReference, ReflectedReference
    camera.py            Camera + noise models (Ideal, PoissonGaussian, SCMOS, EMCCD)
  presets.py             Widefield, InlineHolography, Brightfield, Darkfield, ISCAT,
                         PhaseContrast, DIC, QPI
  physics/
    pupil.py             evaluate(modifiers, fx, fy, λ) -> complex
    propagate.py         asm_kernel (H_exp form, band-limited, carrier-subtracted)
    emitters.py          psf_mft_roi(), gaussian_sprites(), scatter_add()
    raster.py            blurred_ball(), sdf_ramp(), patch placement
    projection.py        analytic chord lengths; occupancy z-sum
    mie.py               S(θ) on 1-D grid -> pupil spectrum (per mode), dipole limit
    multislice.py        z-march with procedural slices, segment checkpointing
    abbe.py              source-point sampling on k-grid, chunked checkpointed mode sum
    fluorescence.py      per-z OTF convolution (rfft2), empty-slice compaction
    detect.py            interference assembly, pixel integration, noise pipeline
  render/
    fidelity.py          Fidelity + presets
    plan.py              RULES table, bounds, grid derivation, validity, Plan, explain()
    simulator.py         Simulator(nn.Module): sample(), render(), expected(), parameters(path)
    labels.py            Positions, Mask, OPL, Photons, Param(path)
    compare.py           compare(scene, scope, fid_a, fid_b, …)
  interop/deeptrack/     features.py, dataset.py, units.py, parity/ (golden refs)
  fit/                   losses.py (poisson_nll, mmd, radial_psd, hist_w1), reg.py (v1.x)
```

### 3.3 Division of responsibilities

**Across projects.**
- **DeepTrack2** keeps the pipeline DSL and data plumbing: Features, Properties, Sources, augmentations, sequences, `dt.pytorch.Dataset`.
- **deeplay** keeps the models.
- **gradoscopy** owns:
  - differentiable physical scene sampling;
  - rendering;
  - the detector;
  - labels derived from the scene.

  The adapter (§10) is the only place where DeepTrack2's property DAG meets gradoscopy tensors [B01 §implications 1].

**Across the team.** With 2–3 developers:
- **Developer A** owns the spine: `_core`, `params`, `render`, the adapter.
- **Developer B** owns `physics`.
- **Developer C**, if there is one, owns validation, benchmarks, parity and docs.

With one developer the order of milestones stays the same and the timeline roughly doubles.

---

## 4. Core data model

### 4.1 Units and conventions (frozen in M0, in `_core/conventions.py`)

- **Length is in µm**, internally and in the API. Helpers are `nm = 1e-3`, `um = 1`, `mm = 1e3`. Time is in s. Angles are in rad. Counts are in photons. Wavelengths are vacuum wavelengths in µm. Refractive indices are complex and dimensionless.
- **Zernike coefficients are OPD in µm** (phase = 2π·OPD/λ), which makes them correct for polychromatic light. The basis is ANSI/OSA, RMS-normalized. Converters exist for Noll and for DeepTrack2's `(n, m)` in radians.
- **Coordinates.** Right-handed. x runs along image columns and y along rows, so arrays are `[..., Y, X]`. The x/y origin is the camera ROI's top-left corner mapped to object space. **+z points from the sample toward the objective**, which is the propagation direction of transmitted and collected light. **z = 0 is the nominal focal plane.** An object at z > 0 lies between the focal plane and the objective.
- **Fourier convention.**
  - Fields are e^{i(k·r − ωt)}.
  - `torch.fft.fft2` uses `norm="backward"`, f = `fftfreq` in cycles/µm, and k = 2πf.
  - Forward propagation by +Δz multiplies the angular spectrum by e^{+i k_z Δz}, with k_z = 2π·√((n/λ)² − |f|²). This is computed carrier-subtracted, as (k_z − k_m)Δz, with the argument built in fp64 and reduced mod 2π [B07 §2.4].
- **Radiometry.**
  - An emitter's intensity PSF integrates to its collection efficiency in `accurate` mode and to 1 in `draft`/`standard`.
  - Irradiance is expressed in photons per exposure per simulation pixel.
- **Pupil coordinates.** Normalized ρ = |f|·λ/NA is used for modifiers that are defined on the aperture, such as Zernike and the pixel pupil. Absolute f in cycles/µm is used for physical modifiers such as defocus and Gibson–Lanni.

### 4.2 Grids and sampling bookkeeping

- **`Grid`**: `shape: tuple[int, int]` and `spacing: float` (µm, object space, isotropic in v1), plus `origin`. These are Python numbers owned by the plan, so they are static. `KGrid` is derived from it: `fx, fy = fftfreq(N, d)`. `VolumeGrid` adds `z0`, `dz` and `Nz`.
- **Simulation spacing**: d_sim = pixel_pitch / (M·s), where s is the integer oversampling factor. `oversample="auto"` picks the smallest s such that d_sim ≤ λ_min / (4·NA). Squaring a coherent field doubles its bandwidth [B05 §5]. When the illumination NA is non-zero, the object spectrum up to (NA + NA_ill)/λ must also be representable.
- **Padding** is derived from bounds: r_pad = max|Δz|·tan θ_max + 2λ/NA [B05 §5]. The padded size is rounded up to a 2·3·5·7-smooth number, because prime sizes cost up to 1.9× in cuFFT [B07 §2.3]. Users can pin padding through `Fidelity`, never through scene objects.
- **Common grid for all wavelengths.** Every wavelength uses the same object-space grid and the same k-grid. The pupil support |f| ≤ NA/λ_w differs per wavelength on that grid. This avoids Chromatix's problem of a different dx per wavelength [B02 §2.1], and it turns spectral sums into plain weighted reductions.
- **Sparse emitter pupils** are sampled on their own small N_p × N_p grid (64–128 points) over |f| ≤ NA/λ. They evaluate the *same* modifier functions as the dense k-grid. The pixel pupil is a table in ρ-coordinates, interpolated bilinearly on separate amplitude and phase channels, because `grid_sample` rejects complex input [B07 §5.4].
- **Camera grid.** In v1, integer s×s sum pooling of the ROI is exact box integration and differentiable [B05 §5]. A continuous-coordinate MFT detector, which would make pixel size learnable, is v2.

### 4.3 Containers (fixed rank; size-1 axes broadcast)

```python
@dataclass(frozen=True)
class PupilSpectrum:          # coherent currency: SCATTERED/object field only
    data: Tensor              # complex [B, M, W, P, Ky, Kx]   (Y,X last & contiguous for fft2)
    grid: Grid                # the sim grid whose fftfreq defines (Ky, Kx)
    wavelengths: Tensor       # [W] µm
    mode_weights: Tensor      # [B|1, M]  (sum to 1)
    wl_weights: Tensor        # [W] power weights incl. quadrature (sum to 1)
    n_medium: Tensor          # [] or [B]
    z_ref: float = 0.0        # always the focal plane in v1
    direction: int = +1       # +1 transmitted/forward, -1 reflected/backward (iSCAT)

@dataclass(frozen=True)
class Reference:              # analytic unscattered / reference wave, per mode & λ
    amplitude: Tensor         # complex [B|1, M, W, P]   (0 => blocked, darkfield)
    k_in: Tensor              # [M, W, 2] transverse frequency (snapped to k-grid)
    included_in_spectrum: bool  # True if a dense exit field already carries it

@dataclass(frozen=True)
class Irradiance:             # incoherent currency
    data: Tensor              # real [B, W, Y, X] photons / exposure / sim-pixel
    grid: Grid
```

- **Axes.** B = batch (flattened from any `batch_shape` at the API). M = mutually incoherent coherent modes (source points, polarization states). W = wavelength bins. P = polarization components (**always 1 in v1**). Y, X = space or frequency.
- **Why fixed rank.** Adding an axis later touches every kernel, while a size-1 axis costs nothing [B02 R1, B05 §8.1]. The fixed rank prevents shape bugs, and there are no class variants per axis combination as in Chromatix.
- **Not `torch.Tensor` subclasses and not named tensors** [B07 §7.2].

### 4.4 Light

- **Spectrum.** `Spectrum(wavelengths, weights)` holds power weights including the quadrature weights. `Spectrum.gaussian(center, fwhm, bins="fidelity")` defers the bin count to the fidelity policy; a monochromatic spectrum is just a float.
- **Illumination for transmitted and reflected light.**
  - `PlaneWave(wavelength, direction=(θ, φ) | na=(NA_x, NA_y), polarization=angle | "unpolarized", amplitude)`.
  - `Koehler(spectrum, condenser=Disk(NA) | Annulus(NA_in, NA_out) | Custom(fn), points="fidelity")` produces M source points, snapped to the k-grid to avoid wrap-around seams, with their weights [B05 §2.1–2.2].
- **Coherence model** (the whole v1 story):
  - within a mode everything is coherent: all particles, the volume and the reference;
  - different modes (source points, the two orthogonal polarizations of unpolarized light) add in intensity;
  - wavelengths add in intensity;
  - emitters are mutually incoherent and never go through M; they go through the incoherent path.

  This follows Wolf's coherent-mode expansion, exact for every linear stage [B05 §1.1]. There is no 4-D mutual intensity: at 256² it would need about 34 GB [B02 §2.2].
- **Polarization in v1.** The field is scalar (P = 1). The Mie kernel projects the S-matrix onto the input polarization and an optional analyzer, which is DeepTrack2-compatible (`input_polarization`, `output_polarization`) [B01 §c]. Unpolarized input is represented as two modes on M rather than as a special field type [B02 §3.4].
- **Fluorescence excitation.** `Excitation.uniform(intensity)` or `Excitation.pattern(fn)`, where `fn` is a torch function of xyz that returns relative intensity. It multiplies the emitter photon counts or the density. SIM, TIRF and light-sheet profiles become one-line callables. Physically derived patterns are v2.

### 4.5 Sample

- **`Scene(*populations, medium=Medium(n))`**. The medium holds n(λ). Its v2 slot is the layered stack (immersion, coverslip, sample), which is read by both optics and sample [B06 §5].
- **`Population(geometry, position, rotation, material=None, emission=None, count=None, priority=0, name)`.** A population is one kind of object, with every leaf batched as a struct of arrays over `[*batch_shape, N_max, …]` plus `presence ∈ [0, 1]` [B06 §6, B09 §2.6]. Python loops only over populations, never over objects.
- **Geometry, material and labelling are separate** [B06 §10.2]:
  - Geometry kinds in v1: `Point`, `Sphere(radius)`, `Ellipsoid(radii)`, `Capsule(length, radius)`, `Composite(parts)`, `Voxels(values, spacing, quantity)` and `SDF(fn, bbox)`.
  - `Material(n=complex | callable(λ))` supplies RI contrast.
  - `Fluorophore(photons, spectrum)` supplies emission.
  - One geometry can carry both. The *same* trace then renders in phase contrast and in fluorescence, with masks from the same geometry.
- **Representations are fixed methods, not a conversion graph.** Every volumetric geometry implements `sdf(p)` (exact or first order) and `bbox()`. Optional methods:
  - `occupancy(p, σ)`: the closed-form Gaussian-blurred ball for spheres. Its gradients are exact: dV/dR 4.5239 against 4.5239 [B06 M].
  - `thickness(xy)`: analytic chord lengths, for projection.
  - `mie()`: `Sphere` only.

  The default `occupancy` is the SDF ramp `clamp(0.5 − d/h, 0, 1)`, which gives dV/dR 4.525 against a true 4.524 [B06 M]. Kernels call the method they need. If it is missing, the plan raises an error at construction time.
- **Composite objects.** A `Composite(body=Part(...), nucleus=Part(...), droplets=Part(..., count=…, placement=InsidePart("body")), texture=GaussianRandomField(..., within="body"))` is instanced per object slot.
  - Placement inside an ellipsoidal parent is *reparameterized*: sample u uniformly in the unit ball, then map through the parent's affine transform. No rejection sampling is needed.
  - Children have higher priority.
- **Overlap rule for RI.** Priority "over" compositing in ε, linear to first order in Δn [B06 §6]. Siblings are normalized by occupancy. Emission densities add. `compose="add"` reproduces DeepTrack2's legacy behaviour.
- **Voxel samples.** `Voxels(values, spacing, quantity="dn" | "n" | "density")` has a bounding box and a transform. Rotation uses `grid_sample` on stacked real and imaginary channels [B07 §5.4]. Any torch callable `fn(xyz) → values` is accepted as `Voxels.from_fn`, which covers neural fields [B06 §2.7].

---

## 5. Physics and the fidelity mechanism

### 5.1 How finely to simulate the optical path

Simulate everything in object space, on three planes:

1. **The sample region.** A single plane, or slices along z.
2. **One pupil plane** on the common k-grid. The condenser source, aperture, defocus, aberrations, Gibson–Lanni, phase ring, stops, DIC shear and reference filtering all live here as modifiers or source distributions.
3. **The camera plane**, where magnification is a coordinate scale and pixels are integrated.

There is no tube-lens plane, no intermediate image plane and no relay in v1. Conjugate-plane filters are pupil modifiers. This is enough for every v1 modality [B05 §3.2–3.3]. The seam that could break it is field-dependent optics, which later enter as per-emitter pupils on the sparse path.

### 5.2 Kernels (v1)

Each kernel is a pure function in `physics/`, registered with a `KernelSpec` that records:
- which geometry kinds it handles;
- its contrast type (`coherent` or `emission`);
- the fidelity knob value that selects it;
- a `validity(bounds, ctx)` function;
- its gradient strategy.

| Slot | Kernel | Method (short) | Output currency |
|---|---|---|---|
| emitters | `gaussian` (draft) | Pixel-integrated erf Gaussian, σ ≈ 0.21λ/NA with a z-dependent width [B05 §3.6] | `Irradiance` |
| emitters | `pupil` | U_e(x_r) = Σ_p P(f_p) e^{i(k_z,p − k_m) z_e} e^{i2πf_p·(x_r − x_e)} by MFT (matrix pair) onto an R×R ROI whose size comes from bounds. Sub-pixel offset folded into the MFT matrices, then \|U\|², normalization, × photons, scatter-add | `Irradiance` |
| dense emission | `per_z_otf` | Soft occupancy × density → Σ_z irfft2(rfft2(ρ_z)·OTF_z) with OTF_z from \|IFT(P e^{ik_z z})\|²; empty-slice compaction from the bounding boxes [B01 §b] | `Irradiance` |
| spheres (coherent) | `dipole` (draft) | Mie-dressed α = i6πa₁/k_m³ with radiative correction [B04 §2.1]; valid for x ≲ 0.3 | `PupilSpectrum` |
| spheres (coherent) | `mie` | Coefficients a_n, b_n in fp64 (downward D_n) [B07 §5.2] → S₁, S₂ on a 1-D θ grid of about 1024 points → per mode m, θ(k) = ∠(k, k_in,m) → interpolate → polarization projection → A(k) = −S(θ)/(2π k_m k_z)·e^{−i(k − k_in)·r_p} ([R] normalization, pinned by a unit test against direct field summation). Never uses per-pixel order sums: those need about 75 GB for 100 particles × batch 32 [B08 §2.2] | `PupilSpectrum` |
| spheres (coherent) | `voxel` | Rasterize into the RI volume and use the volume path. This is the consistency tier for mixed samples | via volume |
| volumes (coherent) | `projection` | t = exp(i k₀∫Δn dz − k₀∫κ dz), from analytic chord lengths or an occupancy z-sum, placed at the group's centre plane z_c. Exit field → FFT → back-propagate to z = 0 | `PupilSpectrum` (reference included) |
| volumes (coherent) | `multislice` | u_{s+1} = F⁻¹{H_Δz · F{u_s · e^{i k₀ Δn_s Δz}}}, with H the **free-space** angular-spectrum propagator (evanescent components removed, *not* NA-truncated, fixing DeepTrack2 [B01 §b]) and band-limited when needed. Slices are generated procedurally per step from the populations. Segment checkpointing when gradients are on. Optional obliquity correction 1/cos θ_in | `PupilSpectrum` (reference included) |
| illumination | `abbe` | M source points on the condenser. For projection samples, a tilt is a spectral shift (exact for a thin object [B05 §1.2b]): one object FFT and M shifted pupils. For multislice, M marches, with a cost warning. Chunked and checkpointed mode sums [B07 §4.2] | modes on M |
| pupil | `evaluate` | Product of modifiers on (fx, fy, λ); soft aperture sigmoid(s·(NA/λ − \|f\|)) so NA gets gradients [B02 §2.11]; Zernike basis cached as a buffer | complex `[B\|1, W, 1, Ky, Kx]` |
| image | `assemble` | E_s = IFT(P·A). With an analytic reference, E_r = a_ref·P(k_in)·e^{i2πk_in·r} and I = \|E_r\|² + 2Re(E_r* E_s) + \|E_s\|², computed term by term with no subtraction. This avoids iSCAT cancellation at contrasts of 1e-2 to 1e-5 [B08 §2.3]. Then Σ over M and W | `Irradiance` |
| detector | `camera` | s×s sum-pool, crop, × QE × exposure (+ background) → Poisson (scaled-ST) → EM Gamma (rsample) → read noise (map) → gain, offset → ST-round → clip. `log_prob` is Poisson for low counts, the shifted-Poisson/Gaussian approximation for sCMOS [B05 §4], and the high-gain approximation for EMCCD | ADU `[B, C, H, W]` |

**Combining sparse and dense paths** (principle 3 in §0):
- **Coherent.** Pupil spectra from all coherent populations are summed per mode and wavelength. This is C0 superposition [B04 §5]. If any dense population exists, its exit field already carries the reference, so `Reference.included_in_spectrum=True` and the analytic reference is dropped. Otherwise the analytic reference is used. Mixing Mie spheres with a multislice cell therefore works in v1 (C0) and produces a validity warning that coupling is neglected. C1 injection is v2.
- **Incoherent.** Irradiances are summed on the simulation grid.
- **Both.** A microscope may produce both a coherent and an incoherent contribution, for example fluorescence with transmitted-light leakage. The two irradiances add.

### 5.3 Modalities are presets, not classes

A preset is a function that assembles specs. It contains no physics:

| Preset | Illumination | Reference | Pupil modifiers | Notes |
|---|---|---|---|---|
| `Widefield` | `Excitation` | — | aperture, defocus, Zernike, pixel pupil, Gibson–Lanni | incoherent |
| `InlineHolography` / `Brightfield` | plane wave or Köhler disc | in-line | aperture, defocus, aberrations | `Brightfield` ≈ `InlineHolography` + Köhler |
| `Darkfield` | annulus with NA_in > NA, or central stop | blocked | + `CentralStop` | the reference is outside the aperture |
| `ISCAT` | plane wave | reflected (r ≈ 0.067 at glass/water [B05 §3.4]; round-trip phase 2k_m z) | aperture, defocus | spectrum `direction=-1`; homogeneous medium in v1 |
| `PhaseContrast` | Köhler annulus | in-line | + `PhaseRing` | the halo needs M > 1 |
| `DIC` | Köhler disc | in-line | + `DICShear(shear, bias)` | scalar thin-sample form [B05 §3.3] |
| `QPI` | plane wave | in-line | aperture | output tap `"phase"` = arg(E/E_ref) |

### 5.4 How fidelity is expressed

```python
@dataclass(frozen=True, kw_only=True)
class Fidelity:
    preset: Literal["draft", "standard", "accurate"] = "standard"
    emitters: Literal["gaussian", "pupil"] | None = None          # None -> from preset
    spheres: Literal["dipole", "mie", "voxel"] | None = None      # coherent spheres
    volumes: Literal["projection", "multislice"] | None = None    # coherent volumes
    source_points: int | Literal["auto"] | None = None            # Köhler modes; 1 = coherent
    wavelength_bins: int | None = None
    oversample: int | Literal["auto"] | None = None
    apodization: bool | None = None
    raster_sigma: float = 0.3                                     # × voxel size [B06 §3.3]
    dz: float | Literal["auto"] = "auto"
    pad: float | Literal["auto"] = "auto"
    polarization: Literal["scalar"] = "scalar"                    # "vectorial" in v2
    per_population: Mapping[str, Mapping[str, Any]] = {}          # e.g. {"beads": {"spheres": "voxel"}}
    on_invalid: Literal["warn", "raise", "ignore"] = "warn"
```

| Knob | `draft` | `standard` | `accurate` |
|---|---|---|---|
| emitters | gaussian | pupil | pupil + apodization + collection-efficiency radiometry |
| spheres | dipole if x_max < 0.3, else mie | mie | mie |
| volumes | projection | projection (warn if thicker than the depth of field (DOF)) | multislice |
| source_points (if a condenser is given) | 1 (on-axis) | 9 | auto: rings to σ-dependent convergence (25–49) |
| wavelength_bins | 1 | 1 | 3 (fluorescence) / 5 (broadband coherent) |
| oversample | 1 | auto (Nyquist) | auto × 2 |
| detector | as the camera spec | as the camera spec | as the camera spec |

Fidelity is a *policy*: named presets that expand into explicit, overridable knobs [B09 §1.3]. It is discrete and not differentiated [B07 §6.4]. Knob values double as kernel names, so a plugin kernel `spheres="tmatrix"` is selected by the same mechanism.

### 5.5 How implementations are selected, and when planning happens

- **At `Simulator(...)` construction** (eagerly, so errors surface immediately):
  1. Walk the spec and extract **bounds** for every leaf. Uniform, Beta and similar use their support. Normal and LogNormal use the (1e-4, 1 − 1e-4) quantiles. Learnables use `bounds=` if declared, otherwise init ± 50 % with a warning.
  2. For each population, determine the contrast type: emission if a fluorophore is present and the microscope is incoherent, coherent if a material is present and the microscope is coherent.
  3. Look up `RULES[(geometry_kind, contrast, knob_value)]` to get a kernel. A missing entry is an error that names the population and lists the available knobs.
  4. Run each chosen kernel's `validity(bounds, ctx)` and collect `Violation(kernel, quantity, value, limit, severity, advice)`.
- **At the first `sample()` or `render()` for a given `(batch_shape, device)`**, derive grids, padding, slice counts, ROI sizes and chunk sizes. Chunk sizes come from a simple tensor-size formula and `torch.cuda.mem_get_info()`. The resulting `Plan` is cached. It is a frozen, printable, hashable list of named stages with their static configuration.
- **Never mid-training.** A runtime spot check (cheap reductions such as max defocus, max x, max phase) warns once if a value leaves the planned bounds.

### 5.6 Validity checks (machine-readable, computed from bounds)

| Kernel | Check | Source |
|---|---|---|
| projection | object thickness d > n·λ/NA² (DOF) → warn; phase across an object > 1 rad → info | [B04 §2.2] |
| multislice | max \|Δn\| > 0.1 or scattering half-angle > ~0.5 rad → warn; dz > λ_m/2 → raise | [B04 §2.5] |
| dipole | x = k_m a > 0.3 → warn (use mie) | [B04 §2.1] |
| mie | geometry not a sphere → error; x > 200 → info (fp64, slow); pairwise gap < λ → warn (coupling neglected); coexistence with a dense population → warn (C0) | [B04 §2.8, §5] |
| sampling | d_sim > λ/(4NA) → warn or raise; ROI radius < PSF extent at max \|z\| → auto-grow | [B05 §5] |
| scalar optics | NA > 1.2 → info; any `Fluorophore(dipole="fixed")` → error until v2 | [B03 §2.2, B05 §3.1] |
| abbe × multislice | M·S > 500 slice-steps per image → warn with the cost estimate | [B07 §4.1] |
| learnable geometry | any learnable reaching a hard-mask path → error | [B06 §3] |

### 5.7 How lower fidelities relate to higher ones

- Tiers are **discrete approximations of the same trace**, not a continuum.
- Every lower-tier kernel names its reference kernel and carries a **tolerance table** (regime → error metric). The table is measured by the cross-fidelity suite (§11) and printed by `explain()` next to the chosen kernel, for example "multislice vs projection: ≤ 2 % relative L2 for d ≤ DOF/4".
- `gs.compare(scene, scope, "standard", "accurate", batch_size=8, seed=0)` renders *one* trace at both fidelities and reports relative L2, SSIM and phase RMSE. This is possible only because the trace is independent of fidelity.
- A cross-tier gradient surrogate, y = y_hi.detach() + y_lo − y_lo.detach() [B07 §6.4], is v2.

---

## 6. Parameters, randomness and learnability

### 6.1 Leaf kinds

| Leaf | Meaning | Storage / behaviour |
|---|---|---|
| plain number or `Tensor` (= `Fixed`) | Constant | Registered as a buffer by the Simulator, so `.to()` works |
| `nn.Parameter` | Learnable, TorchOptics convention [B02 §2.2] | Registered as is |
| `Learnable(init, constraint=None, bounds=None, scale=None)` | Learnable with a constraint | Owns an `nn.Parameter` u in **normalized unconstrained space**: value = transform(u)·scale, with scale defaulting to \|init\| or the unit of the quantity. Adam's per-step move is about lr in u, so ≈ lr × scale in physical units. **[M-C]** in SI metres, Adam at lr 1e-2 moves a 1 µm position by 1 cm (10⁴ relative). In µm or with scale normalization it moves 1 % |
| `Random(dist, per="object" \| "sample" \| "batch" \| "frame")` | Random variable. The `dist` arguments may themselves be leaves, which gives learnable distribution parameters | Sampled with the Simulator's generator. The estimator is chosen automatically (§6.2) |
| `Derived(fn, **deps)` | A deterministic torch function of other leaves | Dependencies are passed **by object reference**, not by argument-name matching (DeepTrack2's name matching is opaque and cannot be pickled [B09 §3.9]). Differentiable |
| `Count(max, mean)` | Number of objects | N_max slots plus `presence`. Discrete, with the estimator in §6.2 |
| `Choice(options, probs)` | Categorical choice among populations | Implemented as mutually exclusive presence masks across populations |

- **Paths.** Every leaf gets a path from the spec tree, for example `"beads.shape.radius"` or `"scope.pupil.zernike.coeffs"`.
- **Parameters.** `sim.parameters(pattern=None)` yields the learnable parameters, filtered by glob [B02 §2.10, dLux]. `sim.named_parameters()` is keyed by path.
- **Shared leaves.** A leaf object shared between specs shares its parameter. This is how one learnable sample is rendered by two microscopes.
- **`spec.frozen()`** returns a copy with every `Learnable` replaced by `Fixed(value.detach())`.

**Default plates** (DeepTrack2 semantics, made explicit):
- Leaves inside a `Population` are per object: `[B, N_max, …]`.
- Population-level attributes are per sample.
- **Optics and camera leaves are shared across the batch unless `per="sample"` is set.** Pupils are small, so per-sample optics are cheap when requested: the pupil is evaluated as `[B, W, 1, K, K]` instead of `[1, …]`.

### 6.2 Gradient estimators (declared per site, recorded in the trace)

| Site | Default | Alternatives | Notes |
|---|---|---|---|
| Continuous with `rsample` (Normal, LogNormal, Uniform with learnable bounds, Gamma, Beta, Dirichlet) | pathwise | score | [B07 §6.2] |
| Truncated or bounded (e.g. radius > 0) | pathwise through a transform (softplus / sigmoid) | — | via `TransformedDistribution` |
| `Count` with learnable `mean` | Relaxed-Bernoulli presence, straight-through hardened (τ = 0.5); presence multiplies scattered amplitude, photons or occupancy | score with a moving-average baseline | [B06 §7] |
| `Count` with fixed `mean` | Bernoulli presence, no gradient | — | |
| `Choice` | score with a baseline | Gumbel-softmax mixture (renders every option) | only when options are cheap |
| Poisson shot noise | **scaled straight-through**: λ + √λ·stopgrad((n − λ)/√λ). Exact samples with Gaussian-consistent gradient 1 + ε/(2√λ) | Gaussian reparameterization; score | [B07 §6.3 M] |
| EM gain | Gamma `rsample` (implicit) | — | [B05 §4] |
| Quantization | straight-through round | none | |
| Saturation | hard clip | soft clip | |
| Rejection-sampled non-overlap (adapter only) | none (documented bias) | soft repulsion in v2 | [B06 §7] |

- For score-function sites the trace stores `log_prob`. `batch.score_surrogate(loss)` adds Σ log_prob · stopgrad(loss − baseline), in the manner of Pyro's trace, but without a global handler stack [B09 §1.10].
- **Guard.** `physics/` never calls `torch.poisson`, `torch.normal(tensor, tensor)` or `bernoulli` directly; the estimators in `_core/estimators.py` are the only entry points. The test suite asserts a non-zero gradient for every learnable reachable from `image`.

### 6.3 Batching semantics and seeding

- **Batch shape.** `sim.sample(batch_size | batch_shape, seed=…)`. Any `batch_shape` tuple is accepted, for example `(K, Z)` for bead stacks or `(B, T)` for sequences. Kernels see it flattened to B, and outputs are unflattened.
- **Ragged counts** are handled by padding to N_max with presence. There are no nested tensors [B07 §7.2].
- **Seeding.**
  - `seed: int | torch.Generator`. From an int, gradoscopy derives a device generator by hashing `(seed, "sample")`, and a separate stream for noise by hashing `(seed, "noise")`.
  - Global RNGs are never touched [B09 §3.3].
  - Reproduction uses a **stored trace**: `sim.render(batch.trace)` is bit-exact on the same device and version, *including* noise, because the noise seed is in the trace [B09 §2.7].
  - Nondeterministic CUDA scatter-adds are documented. `deterministic=True` switches emitter splatting to a sort-based path.
- **Data-generation mode.** `sample()` runs under `torch.inference_mode()` automatically when no learnable leaf requires gradients and grad mode is off. This enables in-place fast paths [B07 §4.2].

### 6.4 Ground truth and labels

- Outputs are requested up front: `outputs=("image", "expected", "field", "beads.position", gs.labels.Mask("cell"), …)`. Nothing that is not requested is computed [B09 §2.8].
- There are three sources:
  - **trace values by path**, for example radius, n or Zernike coefficients, each with a presence mask;
  - **label renderers** evaluated from the same trace and geometry at camera or simulation resolution: positions in µm and pixels with an in-FOV flag, instance masks (hard or soft occupancy), OPL maps and photon maps;
  - **stage taps**: the noise-free expected image, the complex camera-plane field, and irradiance per population.
- The label types are plain tensors plus masks. `tv_tensors` wrappers can be added in v1.x.

---

## 7. Composition and extension

**No operator algebra.** Composition is by lists in four places only:
1. `Scene(*populations)`
2. `Pupil(*modifiers)`
3. `Composite(**parts)`
4. `gs.Stack(microscopes)`, which renders one trace through several microscopes into a channel axis, for multimodal data, DPC half-pupils or multi-focus stacks.

This is enough for every v1 modality. DeepTrack2's overloading of the `>> + * ^ & |` operators is an anti-pattern we do not copy [B09 §3.7].

The system is more than the sum of its parts for three reasons:
- a new geometry immediately works in every modality, through `sdf → occupancy`, and in every label renderer;
- a new pupil modifier works on every path, sparse, dense, coherent and incoherent, because they all evaluate the same pupil function;
- a new modality preset reuses every geometry, detector and label.

### 7.1 Adding a primitive

```python
@gs.register.geometry("torus")
@dataclass(frozen=True, kw_only=True)
class Torus(gs.Geometry):
    major: gs.ParamLike            # µm
    minor: gs.ParamLike

    def bbox(self, v):             # v: resolved tensors [B, N, ...]
        r = v.major + v.minor
        return torch.stack([-r, -r, -v.minor], -1), torch.stack([r, r, v.minor], -1)

    def sdf(self, p, v):           # p: local coords [..., 3]; exact SDF
        q = torch.stack([p[..., :2].norm(dim=-1) - v.major, p[..., 2]], -1)
        return q.norm(dim=-1) - v.minor
```

With `sdf` alone, the torus renders in fluorescence (dense), projection, multislice and masks. The planner routes it through the default SDF ramp. Adding `thickness()` or `occupancy()` is an optional speed-up or accuracy improvement.

### 7.2 Adding a kernel (solver)

```python
@gs.register.kernel(
    slot="spheres", name="layered_mie", handles=("LayeredSphere",), contrast="coherent",
    grad="autograd", reference="voxel",  # its declared higher/lower-tier comparison partner
)
def layered_mie(pop: gs.ResolvedPopulation, ctx: gs.KernelContext) -> gs.PupilSpectrum:
    ...  # tensors in, PupilSpectrum out; no Params, no RNG, no globals

@layered_mie.validity
def _(b: gs.Bounds, ctx) -> list[gs.Violation]:
    return [gs.Violation("layered_mie", "x", b.max_size_parameter, 500, "info")] \
        if b.max_size_parameter > 500 else []
```

`Fidelity(spheres="layered_mie")` or `per_population={"vesicles": {"spheres": "layered_mie"}}` selects it. The registry is a dictionary populated by decorators. Discovery through `entry_points(group="gradoscopy.plugins")` is v2 and adds no API change.

### 7.3 Adding a pupil modifier or a modality

```python
@gs.register.modifier("vortex")
@dataclass(frozen=True)
class VortexPhase(gs.PupilModifier):
    charge: gs.ParamLike = 1
    def __call__(self, fx, fy, wl, v, pupil_ctx):   # continuous coordinates -> complex multiplier
        return torch.polar(torch.ones_like(fx), v.charge * torch.atan2(fy, fx))

def SpiralPhaseContrast(wavelength, objective, camera):   # a preset is just assembly
    return gs.Microscope(illumination=gs.PlaneWave(wavelength), reference=gs.InlineReference(),
                         objective=objective, pupil=gs.Pupil(gs.Aperture(), gs.Defocus(),
                         VortexPhase(1)), camera=camera)
```

---

## 8. Performance and memory strategy

### 8.1 Measured anchors on the RTX 3090, torch 2.14, complex64 (`propC/bench_*.py`)

| Workload [M-C] | Forward | Forward + backward | Peak (fwd+bwd) | Images/s (fwd / fwd+bwd) |
|---|---|---|---|---|
| W1 emitters, full-frame pupil FFT per emitter, B64 × 32 emitters, 128² | 8.9 ms | 28.2 ms | 2.0 GiB | 7.2k / 2.3k |
| **W2 emitters, MFT onto a 32² ROI (pupil 64²) + scatter-add, same scene** | **1.24 ms** | **6.1 ms** | 0.61 GiB | **52k / 10k** |
| W3 Mie in the pupil (fp64 coefficients, 1-D S(θ), interpolation, phase ramps), holography, B64 × 5 spheres, 256² | 22.7 ms | 98.6 ms | 1.57 GiB | 2.8k / 650 |
| W4 dense fluorescence, per-z OTF, B16 × Z32 × 256² (PSFs included) | 2.2 ms | 4.6 ms | 0.46 GiB | 7.3k / 3.5k |
| W5 multislice, B8 × 64 slices × 256², 6 procedural soft ellipsoids, naive autograd | — | 153 ms | **6.5 GiB** | — / 52 |
| W5 same, checkpoint every 8 slices | — | 196 ms | **0.88 GiB** | — / 41 |
| Abbe on a thin sample, phase contrast, B16 × 256², K = 9 / 25 / 81 source points | — | 15 / 39 / 46 ms | 0.26 / 0.70 / 2.23 GiB | — / 1.06k / 408 / 349 |
| Abbe K = 81, chunk 8 with checkpoint | — | 63 ms | 0.24 GiB | — / 255 |

| Batching and launch effects [M-C] | Result |
|---|---|
| Mie holography, 64 images: per-sample loop vs one batched call | 713 ms vs 21.7 ms (**33×**) |
| Dense fluorescence, 16 images: loop vs batched | 9.3 ms vs 2.2 ms (4.3×) |
| Emitters: loop vs batched (already batched over emitters internally) | 13.0 ms vs 8.5 ms (1.5×) |
| Mie coefficient recurrence, n_max = 20, P = 5 / 320 / 4096 particles | GPU 8.6 / 8.6 / 8.8 ms (launch-bound); **CPU fp64 2.0 / 2.3 / 9.3 ms** |
| The same, captured as a CUDA graph (P = 320, n_max = 25) | 3.3 ms |

Numbers carried over from the briefs:
- c64 FFT of 64 × 512² takes 0.705 ms, and a same-size elementwise multiply takes 0.525 ms. Elementwise work is about 45 % of a split-step loop [B07 §2.3].
- c128 FFTs are 7.5× slower. complex32 produced NaN in multislice [B07].
- Each tiny GPU op costs about 8 µs; CUDA graphs give 2.9× for small workloads [B07 §3.3].
- Soft voxelization with unfused autograd: 100, 1,000 and 10,000 spheres take 4.6 ms / 0.21 GB, 37 ms / 1.8 GB and 0.99 s / 17.9 GB [B06 §3.4].
- For 64 × 512² × 256 slices, the materialized sample and its gradient (16 + 16 GiB) dominate memory [B07 §4.1].

### 8.2 What the numbers imply

- **The majority tiers already meet the training-throughput target.** A U-Net step on 32 × 256² takes 10–50 ms, so the simulator should deliver at least about 1000 images/s at 256² [B08 §4]. Emitters, dense fluorescence and Mie holography exceed that in naive, unfused PyTorch. **No Triton, compile or CUDA-graph work is planned before v1.0.**
- **The sparse emitter path must be the MFT onto an ROI**, not a full-frame FFT per emitter: 7× faster forward, 3.3× less memory. The ROI size comes from the z-range bounds.
- **Mie coefficients are launch-bound.** Rule: compute them in fp64 on the **CPU** when P·n_max < ~2·10⁴, otherwise on the GPU, optionally as a CUDA graph. Always batch *all particles of all images* into one call. fp64 is mandatory: complex64 D_n has 1–5 % error [B07 §5.2]. The backward pass is 4× the forward and is the first profiling target (a custom backward for the S(θ) synthesis) in M3.
- **Multislice with gradients always checkpoints when slices are procedural.** Naive autograd keeps every per-object slice intermediate (6.5 GiB). Checkpointing every √S slices costs 1.3× time for 7× less memory [M-C]. The reversible or segmented custom adjoint (0.78 fields per slice, 100 ms vs 163 ms [B07 §4.1]) is v2, triggered by ODT.
- **Multislice at about 50 images/s with gradients (B8, 256² × 64) is a fitting and small-dataset tier**, not a bulk training tier. `explain()` says so. Multislice with Abbe multiplies the cost by M and gets a warning.
- **Partial coherence on thin samples is cheap enough for v1**, at 250–1000 images/s with gradients. SOCS/WOTF are not needed yet.

### 8.3 Rules baked into the kernels

1. complex64 and float32 everywhere. fp64 for Mie coefficients, S(θ) and phase arguments (carrier-subtracted, mod 2π). complex32, fp16 and bf16 are banned in physics [B07 §4.3]. Rendering runs under `torch.autocast(enabled=False)` with IEEE-fp32 matmuls for the MFT, because TF32 degrades accuracy by about two orders of magnitude [B02 §3.2, B07 §2.4].
2. FFT sizes are 2·3·5·7-smooth and fixed per plan, which keeps the cuFFT plan cache warm [B07 §2.3]. `rfft2` is used for real intensities.
3. Mode, wavelength and particle sums are **exact chunked sums** with `checkpoint(use_reentrant=False, preserve_rng_state=True)` when gradients are on. The chunk size comes from `mem_get_info()` [B07 §4.2].
4. Propagation kernels use the `H_exp` form, H = exp(Δz·(i(k_z − k_m))), so gradients with respect to Δz and defocus flow [B02 §2.5]. Kernels are cached per plan only when they do not depend on any learnable. A cache never detaches [B02 §2.3].
5. Soft voxelization is patch-based per population, with a custom autograd function that recomputes in backward. This is triggered when a population has more than about 1,000 objects [B06 §3.4]. Triton is an optional later step.
6. Sample volumes are never materialized when the kernel can generate slices procedurally. User `Voxels` are the exception.
7. **Throughput guards.** A benchmark suite of the W1–W5 and Abbe workloads runs on a self-hosted 3090. It fails on a regression of more than 20 %.

---

## 9. User-facing API

All snippets use one consistent API. Lengths are in µm.

### (a) Holography of Mie spheres in a GPU training loop, with a learnable size distribution

```python
import torch
import gradoscopy as gs
from gradoscopy.units import nm, um

# ---- WHAT: the sample; the size distribution's mean and std are learnable ----
r_mean = gs.Learnable(0.30, constraint="positive", bounds=(0.05, 1.5))   # µm
r_std  = gs.Learnable(0.05, constraint="positive", bounds=(0.005, 0.5))

beads = gs.Population(
    gs.Sphere(radius=gs.Random(gs.dist.LogNormal.from_mean_std(r_mean, r_std))),
    material=gs.Material(n=gs.Random(gs.dist.Uniform(1.39, 1.61))),
    position=gs.UniformBox(x="fov", y="fov", z=(-4 * um, 4 * um)),
    count=gs.Count(max=6, mean=3),                                       # 6 slots, Bernoulli presence
    name="beads",
)
scene = gs.Scene(beads, medium=gs.Medium(n=1.33))

scope = gs.presets.InlineHolography(
    wavelength=633 * nm,
    objective=gs.Objective(NA=1.3, magnification=60, immersion_n=1.518),
    pupil=gs.Pupil(gs.Zernike({"coma_x": 0.010})),                      # OPD in µm
    camera=gs.Camera(pixel_size=6.5 * um, shape=(128, 128),
                     photons=gs.Random(gs.dist.Uniform(300, 3000), per="sample"),
                     read_noise=1.5, noise="poisson+gaussian"),
)

sim = gs.Simulator(
    scene, scope, fidelity="standard",
    outputs=("image", gs.labels.Positions("beads", unit="px"), "beads.presence"),
).cuda()
print(sim.explain(batch_size=64))                      # see (d) for the format

tracker = MyUNet(in_channels=1).cuda()
feat    = FrozenEncoder().cuda().eval()                # features for sim-to-real matching
opt_net = torch.optim.AdamW(tracker.parameters(), 1e-3)
opt_sim = torch.optim.Adam(sim.parameters(), 5e-3)     # contains exactly r_mean, r_std
real_it = iter(real_loader)                            # unlabeled experimental crops [64,1,128,128]

for step in range(20_000):
    batch = sim.sample(64, seed=step)                  # ONE batched GPU render (~23 ms fwd)

    # (1) task update on simulated data -- no gradient into the simulator
    heat = tracker(batch["image"].detach())
    loss_task = gs.losses.heatmap(heat, batch["beads.position.px"], batch["beads.presence"])
    opt_net.zero_grad(); loss_task.backward(); opt_net.step()

    # (2) sim-to-real update: pathwise gradient LogNormal.rsample -> x=k·a -> Mie a_n,b_n
    #     -> S(θ) -> pupil -> |E_r+E_s|² -> scaled-ST Poisson -> image -> features -> MMD
    if step % 4 == 0:
        real = next(real_it).cuda()
        loss_sim = gs.losses.mmd(feat(batch["image"]), feat(real))
        opt_sim.zero_grad(); loss_sim.backward(); opt_sim.step()

print(r_mean.value.item(), r_std.value.item())
```

The count is not learnable here (`mean=3` is fixed), so presence carries no gradient. Making it learnable (`mean=gs.Learnable(3.0, bounds=(0, 6))`) switches presence to relaxed-ST Bernoulli, and `explain()` reports the estimator.

### (b) SMLM: a learnable Zernike and pixel pupil fitted to a bead stack, then used to generate training data

```python
K, Z = data.shape[:2]                                   # data: [12 beads, 41 planes, 25, 25] ADU
zs = torch.linspace(-1.0, 1.0, Z)                      # piezo steps, µm

xy  = gs.Learnable(torch.zeros(K, 2), bounds=(-0.3, 0.3))      # offset from crop centre, µm
z0  = gs.Learnable(torch.zeros(K), bounds=(-1.0, 1.0))
phot = gs.Learnable(torch.full((K,), 2e4), constraint="positive")
bg  = gs.Learnable(torch.full((K,), 30.0), constraint="positive")

bead = gs.Population(
    gs.Point(),
    position=gs.Derived(                                 # -> [K, Z, 1 object, 3]
        lambda xy, z0: (torch.cat([xy, z0[:, None]], -1)[:, None, None, :]
                        - torch.stack([0 * zs, 0 * zs, zs], -1)[None, :, None, :].to(xy)),
        xy=xy, z0=z0),
    emission=gs.Fluorophore(photons=gs.Derived(lambda p: p[:, None, None].expand(K, Z, 1), p=phot)),
    name="bead",
)

zern = gs.Zernike(coeffs=gs.Learnable(torch.zeros(22), scale=0.05), indexing="ansi")   # OPD µm
pix  = gs.PixelPupil(phase=gs.Learnable(torch.zeros(64, 64), scale=0.1),
                     amplitude=gs.Learnable(torch.ones(64, 64), constraint="positive"))
scope = gs.presets.Widefield(
    wavelength=670 * nm,
    objective=gs.Objective(NA=1.45, magnification=100, immersion_n=1.518),
    pupil=gs.Pupil(zern, pix, gs.GibsonLanni(sample_n=1.33)),
    camera=gs.Camera(pixel_size=11 * um, shape=(25, 25), qe=0.82, gain=0.47, offset=100,
                     read_noise=1.6, noise="scmos",
                     background=gs.Derived(lambda b: b[:, None, None].expand(K, Z, 1), b=bg)),
)
fit = gs.Simulator(gs.Scene(bead, medium=gs.Medium(n=1.33)), scope,
                   fidelity=gs.Fidelity("accurate", emitters="pupil")).cuda()
# No Random leaves: batch_shape (K, Z) is inferred from the leaf shapes.

def nll():
    mu = fit.expected()["image"]                         # [K, Z, 1, 25, 25] mean ADU
    return -scope.camera.log_prob(data[:, :, None].cuda(), mu).mean()

# Stage 1: Zernikes + emitter nuisances. Stage 2: add the pixel pupil with smoothness priors.
for params, reg in [
    (fit.parameters(exclude="scope.pupil.pixel.*"), lambda: 0.0),
    (fit.parameters(), lambda: 1e-2 * gs.reg.laplacian(pix.phase.value)
                              + 1e-3 * gs.reg.tv(pix.amplitude.value)),
]:
    opt = torch.optim.LBFGS(list(params), max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = nll() + reg(); loss.backward(); return loss
    opt.step(closure)

# ---- reuse the calibrated pupil to generate SMLM training data ----
molecules = gs.Population(
    gs.Point(),
    position=gs.UniformBox(x="fov", y="fov", z=(-0.8 * um, 0.8 * um)),
    emission=gs.Fluorophore(photons=gs.Random(gs.dist.LogNormal.from_mean_std(3000.0, 1500.0))),
    count=gs.Count(max=40, mean=gs.Random(gs.dist.Uniform(2, 30), per="sample")),
    name="mol",
)
train_scope = scope.frozen().replace(camera=scope.camera.frozen().replace(
    shape=(64, 64), background=gs.Random(gs.dist.Uniform(20, 150), per="sample")))
gen = gs.Simulator(gs.Scene(molecules, medium=gs.Medium(n=1.33)), train_scope,
                   fidelity="standard",
                   outputs=("image", gs.labels.Positions("mol", unit="px"), "mol.emission.photons")).cuda()
batch = gen.sample(64, seed=0)                          # inference_mode; ~50k img/s-class path
```

### (c) A cell in phase contrast or QPI, using a procedural or voxelized sample through multislice

```python
cell = gs.Composite(
    body=gs.Part(gs.Ellipsoid(radii=gs.Random(gs.dist.Uniform((6.0, 4.0, 2.5), (9.0, 6.0, 4.0)))),
                 material=gs.Material(n=1.365)),
    nucleus=gs.Part(gs.Ellipsoid(radii=(3.0, 2.2, 1.6)),
                    offset=gs.Random(gs.dist.Normal(0.0, 0.5)),
                    material=gs.Material(n=1.375)),
    droplets=gs.Part(gs.Sphere(radius=gs.Random(gs.dist.Uniform(0.2, 0.5))),
                     count=gs.Count(max=24, mean=10),
                     placement=gs.InsidePart("body", margin=0.3),   # reparameterized, no rejection
                     material=gs.Material(n=1.46)),
    texture=gs.GaussianRandomField(std=0.004, corr_length=0.4, within="body"),
)
scene = gs.Scene(
    gs.Population(cell, position=gs.UniformBox(x="fov", y="fov", z=0.0),
                  rotation=gs.Random(gs.dist.UniformRotation(axis="z")), count=1, name="cell"),
    medium=gs.Medium(n=1.337),
)

pc = gs.presets.PhaseContrast(
    wavelength=gs.Spectrum.gaussian(center=550 * nm, fwhm=40 * nm, bins="fidelity"),
    objective=gs.Objective(NA=0.4, magnification=20),
    condenser=gs.Annulus(NA_inner=0.25, NA_outer=0.28),
    ring=gs.PhaseRing(NA_inner=0.24, NA_outer=0.29, phase=torch.pi / 2, transmission=0.25),
    camera=gs.Camera(pixel_size=6.5 * um, shape=(256, 256), photons=5000, read_noise=2.0),
)
sim = gs.Simulator(scene, pc,
                   fidelity=gs.Fidelity("accurate", volumes="multislice", source_points=25),
                   outputs=("image", "expected", gs.labels.Mask("cell"), gs.labels.OPL("cell"))).cuda()
b = sim.sample(8, seed=0)             # ~0.2 s with grads at 1 source point; ×M with Köhler (see explain)

# QPI of the SAME trace: another microscope, no resampling
qpi = gs.Simulator(scene, gs.presets.QPI(wavelength=550 * nm, objective=pc.objective, camera=pc.camera),
                   fidelity=gs.Fidelity("accurate", volumes="multislice"), outputs=("phase",)).cuda()
phase = qpi.render(b.trace)["phase"]

# Voxelized alternative: a measured RI tomogram (or any torch function of xyz)
tomo = torch.load("cell_ri.pt")                                      # [Z, Y, X] n, 0.2/0.1/0.1 µm
scene_vox = gs.Scene(gs.Population(gs.Voxels(tomo - 1.337, spacing=(0.2, 0.1, 0.1), quantity="dn"),
                                   position=(20.0, 20.0, 0.0), name="cell"), medium=gs.Medium(n=1.337))
```

### (d) Switching fidelity for the same scene and inspecting the planner's choices

```python
for fid in ("draft", "standard", "accurate"):
    print(gs.Simulator(scene, pc, fidelity=fid).explain(batch_size=8))

report = gs.compare(scene, pc, "standard", "accurate", batch_size=8, seed=0,
                    metrics=("rel_l2", "ssim"))       # ONE trace rendered twice
print(report)
```

Illustrative output for `accurate`:

```
Plan  fidelity=accurate  batch=8  cuda:0  complex64                          plan 7c1e0a…
  grid     sim 512×512 @ 0.1625 µm  (camera 256² @ 6.5 µm / 20×, oversample 2)
           padded 540×540 (2²·3³·5); pad 2.9 µm from |z|≤4 µm, NA 0.4
  volume   z ∈ [−4.0, 4.0] µm → 40 slices @ 0.20 µm  (≤ λ_m/2 = 0.206 µm)
  modes    M=25 Köhler annulus NA 0.25–0.28 (snapped to k-grid)   W=3 (530, 550, 570 nm)
  stages
    1 cell       Composite[Ellipsoid×2, Sphere×≤24, GRF] → multislice
                 procedural slices, free-space ASM (carrier-subtracted), checkpoint every 7
    2 pupil      aperture(soft, NA 0.4) · phase_ring(π/2, T=0.25) · defocus(exact)
    3 image      Abbe Σ_m over 25 modes, chunk 5 (checkpointed); Σ_W
    4 camera     sum-pool 2×2 · photons 5000 · Poisson(scaled-ST) + N(0, 2.0²) · ADU 16-bit (ST)
  gradients    cell.radii ← Uniform [pathwise]   droplets.presence [none]
  validity
    WARN  abbe×multislice: 1000 slice-steps/image (M·S). For bulk data use 'standard'.
    info  multislice vs projection tolerance: ≤2 % rel-L2 for d ≤ DOF/4 (measured, tests/fidelity)
  memory  est. peak 1.9 GiB (chunk 5 modes)
```

and for `draft`:

```
    1 cell       Composite → projection (analytic chord lengths; GRF z-integrated)
  modes    M=1 on-axis plane wave (draft)   W=1
  validity
    WARN  projection: cell thickness up to 8.0 µm > DOF n·λ/NA² = 3.4 µm
    WARN  phase contrast with M=1: halo/shade-off not reproduced (needs source_points>1)
```

### (e) Reusing the forward model in an inverse problem: batched Lorenz–Mie hologram fitting

```python
holos = torch.load("holograms.pt").cuda()               # [16, 1, 128, 128], normalized I/I_bg
xyz = gs.Learnable(init_xyz[:, None, :], bounds=((0, 0, -10), (13.9, 13.9, 10)))  # e.g. from LodeSTAR
r   = gs.Learnable(torch.full((16, 1), 0.5), constraint="positive", bounds=(0.1, 2.0))
n   = gs.Learnable(torch.full((16, 1), 1.50), bounds=(1.34, 1.80))
alpha = gs.Learnable(torch.ones(16), bounds=(0.5, 1.5))   # illumination scaling (Lee et al. 2007)

bead = gs.Population(gs.Sphere(radius=r), material=gs.Material(n=n), position=xyz, name="bead")
fit_scope = scope.replace(camera=gs.Camera(pixel_size=6.5 * um, shape=(128, 128), noise=None),
                          scattered_amplitude=alpha)
fit = gs.Simulator(gs.Scene(bead, medium=gs.Medium(n=1.33)), fit_scope,
                   fidelity=gs.Fidelity("accurate", spheres="mie")).cuda()

opt = torch.optim.LBFGS(fit.parameters(), max_iter=100, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad()
    pred = fit.expected()["image"]                        # [16, 1, 128, 128]
    loss = ((pred - holos) ** 2).mean(dim=(1, 2, 3)).sum()   # 16 independent problems, one graph
    loss.backward(); return loss
opt.step(closure)

# Uncertainty: Fisher information from forward-mode Jacobians of the expected image (v1.x helper)
crlb = gs.fisher.crlb(fit, params=("bead.shape.radius", "bead.material.n", "bead.position"),
                      noise=scope.camera)
```

The same pattern covers reconstruction problems:
- **RI tomography**: `gs.Voxels(values=gs.Learnable(torch.zeros(64, 256, 256)), …)` through multislice with checkpointing;
- **deconvolution**: a learnable emission `Voxels` through the per-z OTF;
- **neural fields**: `Voxels.from_fn(mlp)`.

A deepinv `LinearPhysics` export of linear sub-plans is v2 [B09 §1.4].

---

## 10. DeepTrack2 integration strategy

**Position:** a standalone engine with a thin adapter (option B in [B01]), shipped in M2, *before* the coherent work. Replacing DeepTrack2's property DAG (option C) is not on the table. The DAG is DeepTrack2's user value, and deeplay consumes it [B01 §implications].

### 10.1 Three integration levels (all in `gradoscopy.interop.deeptrack`)

1. **Drop-in, per sample (L1).** DeepTrack2-named Features:
   - optics: `Fluorescence`, `Brightfield`, `Holography`, `Darkfield`, `ISCAT`;
   - scatterers: `MieSphere`, `Sphere`, `Ellipsoid`, `Ellipse`, `PointParticle`;
   - aberrations and noise: `Zernike` and the named aberrations, `Poisson`, `Gaussian`, `Background`;
   - illumination: `IlluminationGradient`.

   Scatterers return lightweight `EngineObject` descriptors, a `Wrapper` subclass carrying resolved properties, instead of masks. The optics Feature (`__distributed__ = False`) collects them into a B = 1 trace, renders, and returns `(H, W, 1)` channel-last by default, so existing notebooks run unchanged. Labels like `particle.position` still work because DeepTrack2's property nodes resolve as before.
2. **Batched (L2).** `gs.dt.BatchedDataset(pipeline, batch_size=64, length=…)`:
   1. resolve DeepTrack2's DAG B times on the CPU, which is cheap scalar Python;
   2. stack the descriptors per kind into padded struct-of-arrays traces;
   3. render **once** on the GPU;
   4. return `(B, C, H, W)` tensors plus the resolved labels.

   It can be passed directly as deeplay `train_data`. This removes the per-sample GPU cost; batching alone is worth up to 33× [M-C].
3. **Learnable (L3).** A `gs.Random(...)` or `gs.Learnable(...)` placed as a DeepTrack2 property value is recognised by the adapter. Sampling is then deferred to gradoscopy, so distribution parameters become learnable *inside DeepTrack2 pipelines*. `nn.Parameter` values already pass through DeepTrack2 unchanged as constants [B01 §a].

### 10.2 Semantics mapping

| DeepTrack2 | gradoscopy |
|---|---|
| SI metres; `position_unit="pixel"` | µm at the adapter boundary; pixels via `resolution / magnification` |
| `upscale` | `Fidelity(oversample=…)` |
| `upsample` (scatterer) | `Fidelity(raster_sigma=…)`; soft occupancy always |
| `padding` | automatic from bounds; a larger requested value is honoured with an info message |
| `output_region` | camera ROI |
| `pupil=` Feature | pupil modifier (`LearnablePhaseMask` → `PixelPupil`) |
| `illumination=` Feature | illumination envelope modifier |
| `return_field=True` | output `"field"` |
| `refractive_index` / `intensity` / `value` | `Material(n)` / `Fluorophore(photons)` / `Fluorophore` with a warning |
| `Darkfield` (\|E−1\|²) | blocked reference |
| `ISCAT` (angle convention) | iSCAT-lite with a reflected reference |
| `SampleToMasks` | `gs.labels.Mask` |
| `NonOverlapping` | stays in DeepTrack2's DAG (rejection; documented as non-differentiable) |
| Zernike `(n, m)` in rad | converted to OPD µm at λ |

### 10.3 Parity and deviations
- Golden references come from DeepTrack2 2.0.2 for DTGS121, DTGS131, DTEx203, DTEx205, DTEx252, DTGS106 and DTGS172.
- Metrics are normalized-image relative L2 and label-statistics KS tests.
- `compat="deeptrack2"` exists **only** to pass the golden tests. It reproduces:
  - the NA-truncated inter-slice propagator;
  - additive overlaps;
  - hard supersampled masks.
- The defaults follow the physics, and deviations are documented. This includes the contrast signs disputed in DeepTrack2 issue #445 [B01 §b].

### 10.4 Migration path
1. DeepTrack2 2.x gains an optional extra `deeptrack[gradoscopy]`.
2. A later DeepTrack2 major version makes gradoscopy the default optics backend. `dt.optical` becomes the adapter.
3. The long-term question of whether DeepTrack2's `Property` and gradoscopy's `Param` converge is open (§14).

---

## 11. Testing and validation strategy

This is a pyramid of tests. Every kernel registers its analytic references, gradcheck shapes, cross-fidelity tolerance and benchmark shape [B07 §8].

1. **Analytic and reference tests (fp64):**
   - Airy pattern 2J₁(v)/v;
   - angular-spectrum propagation against a Gaussian beam;
   - Parseval and unitarity;
   - Fourier shift against `roll`;
   - Zernike orthonormality;
   - blurred-ball closed form against the FFT reference (≤ 1e-7 [B06]);
   - Mie Q_ext and S₁/S₂ against miepython (MIT) to ≤ 1e-8 for x ≤ 100, including absorbing m;
   - dipole against Mie for x < 0.1;
   - Mie pupil spectrum against direct near-field summation, which pins the normalization.

   GPL oracles (PyMieDiff, holopy, psfmodels) are optional test extras and are never installed by default [B03 §4.9].
2. **Gradient tests:**
   - `gradcheck` in complex128 on every functional kernel at tiny sizes (full mode) and at bulk sizes (fast mode) [B07 §8];
   - finite-difference checks for radius, n, position, NA, Zernike, defocus and condenser NA;
   - a **non-zero-gradient assertion for every learnable leaf** reachable from `image`, which guards against the silent zero gradients of `torch.poisson` and `torch.normal`;
   - checkpointed and unchecked gradients must be equal to 1e-6.
3. **Cross-fidelity convergence.** This is the fidelity contract, and it lives in `tests/fidelity/`. Initial targets are below; they are calibrated in M3 and M4 and then printed by `explain()`.

   | Pair | Regime | Target |
   |---|---|---|
   | gaussian ↔ pupil emitters | NA ≤ 0.8, \|z\| ≤ 0.2 µm | ≤ 10 % rel L2 |
   | MFT-ROI ↔ full-frame emitters | ROI from bounds | ≤ 1e-4 |
   | dipole ↔ mie | x < 0.3 | ≤ 2 % |
   | projection ↔ multislice | d ≤ DOF/4 | ≤ 2 % |
   | multislice(voxel sphere) ↔ mie | Δn ≤ 0.05, x ≤ 10, dz ≤ λ_m/8 | ≤ 5 % hologram rel L2 |
   | Abbe K ↔ K_ref = 225 | σ ≤ 0.7 | K = 25: ≤ 2 % |
   | soft voxel s = 1 ↔ s = 4 | dense fluorescence | ≤ 1 % |
4. **DeepTrack2 parity:** golden images plus label statistics (§10.3).
5. **Statistical tests:**
   - Poisson mean equals variance;
   - EMCCD variance ≈ 2G²μ;
   - read-noise maps;
   - bit-exact `replay` on the same device;
   - KS tests on sampled leaves.
6. **Learnability end-to-end:** recover each of the following from synthetic data:
   - Zernikes (≤ λ/50 RMS);
   - radius (±1 %);
   - n (±0.003);
   - LogNormal mean and std (±5 %) by moment matching;
   - defocus;
   - NA (soft aperture).
7. **Performance and memory:** the W1–W5 and Abbe workloads, recording `max_memory_allocated` and time on the self-hosted 3090 (nightly). The build fails on a regression of more than 20 %.
8. **Architecture:** layering lint, a ban on global RNG, a ban on bare `torch.poisson`, and a check that every registered kernel has a validity function and a tolerance entry.

CI runs CPU tests on Linux, Windows and macOS with small shapes. MPS gets complex64 smoke tests only [B07 §1].

---

## 12. Roadmap

Two developers, about 20 weeks to v1.0. Milestones are ordered by risk first (API and integration), then value.

```
week   1   3   5   7   9   11  13  15  17  19  21
M0     █████████                                      spine (highest API risk)
M1              ████████████                          pupil + fluorescence
M2                       ██████████                   DeepTrack adapter (overlaps)
M3                             ███████████████        Mie / coherent sparse
M4                                        ██████████████████   volumes + partial coherence
M5                                                     █████████  v1.0 hardening
```

**M0 — Spine (weeks 1–3).**
- Deliverables:
  - units and conventions;
  - `Grid` and the containers;
  - leaves, sampler and `Trace`;
  - a `Simulator` skeleton with Gaussian emitters;
  - a camera with `expected`, `sample` and `log_prob`;
  - estimators;
  - test, benchmark and layering harnesses.
- Exit criteria:
  - `sim.sample(64)` runs end to end on the GPU;
  - non-zero gradients reach the LogNormal mean and std through scaled-ST Poisson;
  - `replay` is bit-exact;
  - gradcheck is green;
  - API review with two DeepTrack2 users. **After this the spec/trace/render split, `Param` semantics and conventions are frozen.**

**M1 — Pupil and fluorescence (weeks 4–7).**
- Deliverables:
  - pupil modifiers (soft aperture, exact defocus, Zernike, pixel pupil, apodization, Gibson–Lanni);
  - MFT-ROI emitters;
  - soft rasterization (Sphere, Ellipsoid, Capsule, SDF, Voxels);
  - per-z OTF;
  - labels (positions, masks);
  - the `Widefield` preset.
- Exit criteria:
  - analytic tests pass;
  - W2 ≥ 20k images/s forward and ≥ 5k images/s with gradients;
  - W4 ≥ 5k images/s;
  - example (b) recovers synthetic Zernikes to ≤ λ/50 RMS in under 2 minutes;
  - a DeepSTORM3D-style mask-learning loop runs.

**M2 — DeepTrack2 adapter (weeks 6–9, overlapping).**
- Deliverables: L1 and L2 for Fluorescence, PointParticle, Sphere, Ellipsoid, Zernike and Poisson; `BatchedDataset`.
- Exit criteria:
  - DTGS131-like and DTEx252-like notebooks run after changing imports only;
  - throughput at B = 64 is at least 10× native DeepTrack2;
  - golden parity is within tolerance.
- Why this early: it validates the integration seam before the coherent work multiplies the surface area.

**M3 — Coherent sparse (weeks 8–13).**
- Deliverables:
  - fp64 Mie coefficients with the CPU/GPU placement rule;
  - S(θ) → pupil;
  - the dipole tier;
  - plane-wave and oblique illumination;
  - unpolarized light as two modes;
  - reference handling (in-line, blocked, reflected);
  - presets `InlineHolography`, `Darkfield` and `ISCAT`;
  - L1 and L2 adapter coverage for these.
- Exit criteria:
  - Mie matches miepython to ≤ 1e-8;
  - holograms match direct summation to ≤ 1e-3;
  - W3 ≥ 2.5k images/s forward and ≥ 1k with gradients, after the custom S(θ) backward;
  - DTGS121, DTEx203 and DTEx205 parity;
  - example (a) recovers the LogNormal mean and std within 5 % from 10k synthetic "real" images;
  - example (e) recovers r within ±1 % and n within ±0.003 at SNR 20.

**M4 — Coherent dense and partial coherence (weeks 12–18).**
- Deliverables:
  - projection and multislice (procedural slices, free-space angular spectrum, checkpointing);
  - Köhler Abbe with chunked, checkpointed sums;
  - `PhaseRing`, `CentralStop` and `DICShear`;
  - `Composite` and GRF texture;
  - presets `PhaseContrast`, `DIC`, `QPI` and `Brightfield` (volumes).
- Exit criteria:
  - convergence tables for projection ↔ multislice and multislice ↔ Mie;
  - a phase-contrast halo test: present at M > 1, absent at M = 1;
  - W5 ≥ 40 images/s at ≤ 1 GiB;
  - Abbe K = 25 ≥ 300 images/s with gradients;
  - example (c) runs;
  - DeepTrack2 `Brightfield` parity in compat mode.

**M5 — v1.0 hardening (weeks 18–21).**
- Deliverables:
  - fidelity presets finalized from the measured tolerance tables;
  - `explain()` wording and `compare()`;
  - docs and five tutorials, examples (a)–(e);
  - performance-regression CI.
- Exit criteria:
  - all of the above green on nightly;
  - three external DeepTrack2 users port a pipeline;
  - the API is frozen at 1.0.

**After v1**, in the order of application demand:
- v2.1 vectorial pupils and dipoles, validated against psf-generator (MIT) (about 4 weeks);
- v2.2 interfaces: iSCAT iPSF, TIRF, SAF (about 4 weeks);
- v2.3 mixed-sample C1 injection and fluorescence through a volume (about 4 weeks);
- v2.4 photophysics, motion blur, SIM and light-sheet excitation (about 3 weeks);
- v2.5 ODT track: Born/Rytov, SSNP, reversible adjoint (about 5 weeks);
- v2.6 throughput work (Triton fused slice and raster kernels, CUDA-graph Mie, SOCS), only when profiling calls for it.

**The planner is upgraded to cost-based selection only when** three or more kernels compete for one slot and users report wrong automatic choices.

---

## 13. Key decisions

| # | Decision | Position | Rationale |
|---|---|---|---|
| 1 | **Units and precision** | **Micrometres** internally and in the API (with `nm` and `um` helpers). The DeepTrack2 adapter converts from SI metres at the boundary. complex64/float32 by default. fp64 only for Mie coefficients, S(θ) and phase-argument construction. No complex32, fp16 or bf16 | (i) **Underflow [M-C]:** for a 10 nm particle in SI metres, α² = 0 in float32 because 1e-47 is below the float32 subnormal range. In µm it is 1.4e-11. Darkfield and iSCAT cross-sections of small particles silently vanish in metres. (ii) **Optimizer ergonomics [M-C]:** Adam at lr 1e-2 moves a metre-valued position by 1 cm. (iii) Microscopists think in µm and nm. (iv) Precision: complex64 gives 5e-6 error over 100 slices once carriers are subtracted; complex32 gives NaN [B07 §2.4] |
| 2 | **Central interchange representation** | **Two currencies, one plane.** The coherent currency is the *scattered* angular spectrum on the common pupil k-grid, referenced to z = 0, with `direction ±1` and an analytic `Reference`. The incoherent currency is irradiance on the object-space grid. No exit-plane currency: exit fields are converted by one FFT plus back-propagation. No two-port S-matrix | Mie, dipoles and emitters produce pupil-plane quantities natively [B04 §4, B08 §7.1]. Exit fields convert exactly in a homogeneous medium. Keeping the reference analytic enables the explicit interference term and darkfield blocking. Two ports [B04 §4] cost a general S-matrix machinery that only v2 iSCAT and TIRF need; `direction` is the cheap seam for it |
| 3 | **Planner sophistication; when planning happens** | A rule table `(geometry kind, contrast, knob) → kernel`, validity from bounds, and `explain()`. Kernel choice happens **at Simulator construction**; grids and chunking at the **first call per (batch_shape, device)**, then cached. No cost model or search | v1 has 2–3 kernels per slot [B09 §3.2, "over-abstract IR too early"]. Eager planning surfaces errors early. Shapes from bounds keep compilation and CUDA graphs possible. `explain()` gives the transparency a clever planner would have to earn |
| 4 | **Parameter and trace system** | **Our own**: five leaf kinds, `Trace` as a flat path→tensor dictionary with masks and log-probs, `condition=` and `replay` as arguments. No dependency on pyro or tensordict; `to_tensordict()` interop comes later | The required features are small (about 600 lines). Pyro brings a global handler stack and parameter store [B09 §1.10]. tensordict's memmap and nested operations are unneeded, and each dependency is a liability on Windows and Python 3.10–3.15 given a torch release every 6–8 weeks [B07 §1] |
| 5 | **nn.Module vs functional core** | **Functional physics core** (pure functions of tensors and a static `Grid`). The **`Simulator` is the only required `nn.Module`**; it owns the learnable parameters and buffers. Specs are frozen dataclasses. `Learnable` leaves hold their own `nn.Parameter` so they can be shared | This keeps physics testable with gradcheck, compatible with `torch.func` and compile-friendly [B07 §7.2]. Making every scene object a Module mixes mutable state with sampling [B09 §3.13]. Users still get `.parameters()`, `.to()` and `state_dict()` |
| 6 | **Sparse vs dense render paths** | **Both in v1.** Sparse: emitters by MFT-ROI, spheres by Mie in the pupil. Dense: soft voxels through the per-z OTF, projection or multislice. They are combined by **summation in the currency domain**: coherent C0 superposition of pupil spectra (the reference is supplied by the dense exit field if there is one), incoherent summation of irradiance. Coupling (C1) is v2 | Most DeepTrack2 data generation is sparse; cells are dense [B08 §7.2]. MFT-ROI is 7× faster than full-frame [M-C]. Voxelizing points destroys sub-voxel positions [B06 §2.5] |
| 7 | **Polarization** | **Scalar fast path only in v1**, with the P axis present at size 1. Unpolarized light is two modes on M. Mie is projected onto the input polarization and analyzer. v2 adds Jones P = 2 in the pupil, 3-D fields at focus and 6-basis dipole images | Vectorial optics is "ideal" almost everywhere and "minimum" only for high-NA system identification [B08 §3]. It costs 1.5–6× [B03 §2.2]. The axis is the expensive part to retrofit, so it is reserved now [B08 §2.13] |
| 8 | **Multi-wavelength grids** | **One object-space grid and one k-grid for all λ.** The pupil support per λ lives on that grid. W is a broadcast axis with power weights; chunked accumulation when W·M is large. The bin count is set by fidelity | This avoids per-λ dx leaking out of Fourier lenses [B02 §2.1] and makes spectral sums a weighted reduction. Nyquist is set by λ_min |
| 9 | **Time and dynamics** | **Not an engine feature in v1.** Time folds into `batch_shape` `(B, T)`. A reparameterized `Brownian` trajectory helper is provided. Blinking, bleaching and motion blur are v2 | Sequences are needed for tracking data but require no physics beyond "render T frames". Photophysics state machines [B03 §2.4] are a v2 application trigger (SMLM realism) |
| 10 | **Dependency policy** | Core: **torch + numpy only**. Optional extras: `deeptrack` (adapter), `triton`/`triton-windows` (future accelerators), `pytorch-finufft` (v2). Test-only: scipy, miepython. GPL code (holopy, PyMieDiff, psfmodels, DECODE) is **never vendored**; it is used only as optional oracles. MIT-licensed psf-generator can be borrowed from | GPL contamination [B03 §4.9]; Windows lacks Triton [B07 §1]; pint and pydantic in hot paths break autograd and compilation [B09 §3.8] |
| 11 | Tensor layout (extra) | Fixed rank `[B, M, W, P, Y, X]`; size-1 axes broadcast | Retrofitting an axis touches every kernel |
| 12 | Learnable scale (extra) | Learnables are stored O(1)-normalized with a constraint transform | Optimizer-agnostic behaviour (Adam, LBFGS) |
| 13 | Mie compute placement (extra) | fp64 coefficients on the CPU below ~2·10⁴ P·n_max, otherwise on the GPU; all particles in one call; S(θ) on a 1-D grid | GPU recurrence is launch-bound at 8.6 ms against 2 ms on the CPU [M-C]; per-pixel order sums are infeasible [B08 §2.2] |

---

## 14. Top risks and open questions

### Risks

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| **Two front-ends**: DeepTrack2 `Property` and gradoscopy `Param` diverge and confuse users | High / high | L3 adapter (`gs.Random` usable as a DeepTrack2 property); one docs page "which DSL when"; revisit convergence after v1 |
| **Mie backward is slow and fragile** (launch-bound fp64 recurrences, 4× forward [M-C]; upward-recurrence instability for n > x, noted in [B01 §c]) | Medium / high | Downward D_n; custom backward for the S(θ) synthesis in M3; CPU placement; CUDA graph; miepython oracle across x and absorbing m |
| **Scalar-only v1 misleads high-NA SMLM users** (fixed dipoles, quantitative intensities at NA > 1) | Medium / medium | A validity error for fixed dipoles; info at NA > 1.2; vectorial is the first v2 item |
| **C0 superposition under-delivers realism for mixed samples** (beads in cells) | Medium / medium | Warning in `explain()`; C1 injection in v2.3; the `voxel` tier for spheres keeps a consistent monolithic option |
| **Soft-voxel memory with many objects** (17.9 GB at 10k spheres, unfused [B06]) | Medium / medium | Chunked populations with checkpointing; a recompute-in-backward autograd function when N > 1,000; Triton only if needed |
| **Thin partial-coherence cost explodes with multislice** (M × S slice-steps) | High / low | Cost warning; guidance to use `standard` for bulk data; SOCS or stochastic source sampling only if users need it |
| **Bounds-based planning is too conservative** (huge pads from loose Normal tails) | Medium / low | Quantile bounds; user-pinnable `pad` and ROI; the plan reports the bound that drove each size |
| **Relaxed-presence and score-function gradients are biased or noisy** for learnable counts | Medium / medium | Documented estimators with a baseline; count learning is recommended through the mean density |
| **Parity with DeepTrack2 physics quirks** (NA-truncated propagator, additive overlap, #445 signs) | Medium / medium | A narrow `compat` flag; documented deviations; statistical rather than bitwise parity |
| **Scope creep** from the nine briefs' long wish lists | High / high | This document's §2.5 "not building" table; each new feature needs an application trigger and an exit criterion |

### Open questions for the DeepTrack2 team

1. **DSL convergence.** Should a future DeepTrack2 extend `Property` with differentiable distributions, making gradoscopy's `Param` the implementation, or keep two DSLs with the adapter bridging them?
2. **Output conventions.** Should the DeepTrack2 adapter's default output stay channel-last `(H, W, 1)`, or should DeepTrack2 3.x move to `(B, C, H, W)`?
3. **Learnable magnification and pixel size.** Is this needed in v1 (it pulls the MFT detector forward), or is a learnable isotropic scale on object positions acceptable until v2?
4. **Licence.** Is it MIT, matching DeepTrack2? This decides what can be borrowed from psf-generator, waveorder and microsim.
5. **Minimum torch version.** `torch>=2.8` as in `pyproject.toml`, or ≥ 2.12 for the newer `fp32_precision` APIs [B07 §2.4]?
6. **Label types.** Adopt `torchvision.tv_tensors` for masks and keypoints in v1 so augmentations keep labels aligned [B09 §1.5], or keep plain tensors?
7. **Default fidelity for the adapter.** Should `standard` match DeepTrack2's current physics as closely as possible, trading correctness for familiarity, or follow physics and document the differences? (This proposal: follow physics.)
