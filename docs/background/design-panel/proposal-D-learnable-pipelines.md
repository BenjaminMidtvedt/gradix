# gradoscopy — Architecture Proposal D: "Learnable data generation and sim-to-real first"

**Angle.** This design is centred on the ML practitioner. The primary product is a **physics-structured generative model** p_θ(frames, labels), where θ spans scene distributions, optics, illumination and the detector. Every workflow the user named is an operation on that model:
- data generation is `sample`;
- calibration is `fit`, either by likelihood or by distribution matching;
- inverse problems are `condition` followed by `fit`;
- optical design is joint training;
- labels are `trace`.

The physics layer's interfaces are *derived* from what those operations need (§5.0), not the other way round.

**Evidence tags.**
- **[B01]…[B09]** cite the nine research briefs.
- **[M-D]** marks measurements I took for this proposal on the target machine (RTX 3090, Windows 11, torch 2.14+cu130). The scripts are in `scratchpad/archD/expA…expD*.py`.
- Numbers without a tag are my engineering estimates.

---

## 0. Executive summary

1. **Spine: Spec → (compiled sampler) → Trace → (planned renderer) → Outputs.**
   - The *Spec* is declarative and says **what** to simulate. Its leaves are `Param`s: fixed, learnable, random with learnable hyperparameters, derived, or discrete choices.
   - The *Trace* is the batched, sampled instance. It holds values, presence masks, base noise, log-probs and seed. It is at once the solver input, the ground truth and the reproducibility record.
   - The *Plan* says **how**. It is a static stage list chosen by a rule-based planner from the fidelity policy, from bounds derived from the distributions, and from which parameters require gradients.
2. **The probabilistic layer (`gradoscopy.prob`) is built first and treated as the core product.** It provides:
   - plates for sampling granularity (batch, image, object, frame, mode, group);
   - a gradient estimator declared per site, with measured defaults;
   - handlers: condition, replay, intervene, mean;
   - a parameter store with constrained, well-conditioned learnables;
   - a **compiled sampler** that fuses sites by (family, plate). Fusion is **8.7× faster** than per-site sampling, which matters because at fast tiers sampling otherwise costs as much as rendering [M-D].
3. **Two-port interchange.**
   - Sparse objects (emitters, Mie and dipole particles) produce **per-object pupil-plane spectra**, rendered either in local ROIs or accumulated on the global k-grid.
   - Dense content (volumes, maps) produces a **grid angular spectrum**, of which the exit-plane field is a view.
   - Both ports carry a transmission/reflection tag and a coherent-mode axis.
   - Measured: the sparse ROI path renders **51 k images/s** (128², 50 emitters, forward) and 18.7 k images/s with backward. It is 3.8–5.7× faster forward (2.5–4.6× with backward) than dense depth strata at ≤ 200 objects [M-D].
4. **Fidelity is a policy object**: named presets expand into per-stage choices, with per-population overrides. The planner selects implementations by capability, then validity over *distribution bounds*, then cost under a memory budget. It records rejected alternatives in `plan.explain()`. Lower tiers are documented approximations of the same Spec, and ladder tests quantify the gap.
5. **Learnability is audited, not assumed.**
   - Converters and stages declare gradient quality per input.
   - `sim.gradient_report()` probes every learnable for zero or NaN gradients.
   - This catches the silent zeros that break sim-to-real work: hard masks; `torch.poisson`; and plain straight-through Poisson, which gives **exactly 0** gradient on a noise-level-sensitive loss [M-D].
6. **Sim-to-real is a first-class subsystem (`gradoscopy.fit`).** It offers:
   - likelihood fitting with the camera's `log_prob` (bead stacks, per-particle characterisation);
   - distribution matching (MMD, sliced-Wasserstein, summary statistics, critic);
   - CRLB/Fisher information;
   - cross-fidelity surrogate gradients;
   - learnable residual "nuisance slots".

   Proof of concept: a LogNormal particle-size distribution learned from *unlabeled* images recovered the median radius to **0.7 %** and σ to **9 %**, at 6 ms per step for B = 256 [M-D].
7. **Units are micrometres internally.** This keeps learnable parameters O(1) for optimisers and float32. Precision is complex64 by default, with fp64 only for special functions and phase-argument construction.
8. **DeepTrack2 integration is adapter first, then a compiler bridge.**
   - First: DT-compatible feature classes and a `BatchedDataset` that resolves the DT property graph per sample and renders once per batch.
   - Then: `gs.from_deeptrack` compiles pipelines, whose introspectable `dt.dist` rules become differentiable sites.
   - Measured: Python per-sample resolution costs about 0.3 ms per image, which caps the adapter at a few thousand images/s. The native path reaches 10⁴–10⁵ images/s [M-D].

---

## Evidence added by this proposal [M-D]

These numbers are referenced throughout.

**Exp A: sparse vs dense fluorescence rendering.**
Setup: scalar pupil with a soft NA 1.4 edge, exact k_z defocus, 6 Zernike modes and a pixel phase map; λ = 0.6 µm; 100 nm pixels; defocus |z| ≤ 0.8 µm. "dense33" means trilinear splatting into 33 z-strata followed by a per-z OTF, with the OTF recomputed per call as it would be for learnable optics. "fullframe" means a per-emitter full-frame Fourier phase ramp.

| config | method | fwd ms | fwd GB | fwd+bwd ms | fwd+bwd GB | images/s (fwd) |
|---|---|---|---|---|---|---|
| B=256, N=20, 64² | sparse ROI 32² | 1.86 | 0.17 | 5.22 | 0.24 | 137 k |
| | dense33 | 7.09 | 0.53 | 13.2 | 0.79 | 36 k |
| | fullframe | 4.78 | 0.47 | 15.1 | 0.79 | 54 k |
| B=64, N=50, 128² | sparse | **1.25** | 0.11 | **3.42** | 0.15 | **51 k** |
| | dense33 | 7.14 | 0.52 | 13.7 | 0.79 | 9.0 k |
| | fullframe | 12.4 | 1.17 | 39.9 | 1.96 | 5.2 k |
| B=64, N=200, 256² | sparse | 5.72 | 0.44 | 22.6 | 0.54 | 11.2 k |
| | dense33 | 26.2 | 2.08 | 104 | 3.17 | 2.4 k |

ROI truncation matters. At |z| ≤ 0.5 µm the 32² ROI (3.2 µm) has a relative L2 error of **5.7 %** against fullframe, from periodic wrap-around. The 64² ROI has **1.7×10⁻⁴**. Energy is conserved in both cases, so the error is invisible to a naive sum check. **The ROI size must therefore be derived by the planner from the defocus bounds and NA.**

**Exp B: gradient estimators.** Values are the mean and std of the per-replicate gradient. The analytic true value is in the last column.

| Poisson, μ = 40, 4096 px | L1 = (mean − 50)² | L2 = (var − 50)² | true |
|---|---|---|---|
| Gaussian reparam μ + √μ·ε | −20.00 ± 0.17 | −19.96 ± 1.34 | −20 |
| plain straight-through | −20.00 ± 0.20 | **0.000 ± 0.000** | −20 |
| **scaled straight-through** | −20.01 ± 0.17 | −19.95 ± 1.36 | −20 |
| score function | −11.3 ± 1014 | −26.0 ± 1062 | −20 |
| score function + leave-one-out baseline | −20.9 ± 28.5 | −20.2 ± 167 | −20 |

| Relaxed counts (64 slots, p = 0.25), dE[L]/dp | mean ± std (true −473) |
|---|---|
| Concrete soft τ = 1.0 | **+80.8 ± 240 (wrong sign)** |
| Concrete soft τ = 0.5 | −240 ± 335 |
| Concrete soft τ = 0.1 | −469 ± 467 |
| **Concrete straight-through τ = 0.5** | −465 ± 430 |
| score function + leave-one-out baseline | −442 ± 894 |
| mean-field (presence = p) | −508 ± 56 (biased, low variance) |

**Exp C: learning a size distribution from unlabeled images.**
- Forward model: analytic sphere form factor → projection phase → pupil with defocus → |E|² → scaled straight-through Poisson.
- Loss: MMD on 32 intensity quantiles plus 16 radial power-spectrum bins.
- Truth is LogNormal(median 0.450 µm, σ 0.25); the initial guess is (0.25 µm, 0.08).
- Result after 600 Adam steps (cosine learning rate, average of the last 200 steps): **median 0.453 µm, σ 0.227**. That is 6.1 ms per step at B = 256, 64².

**Exp D: resolve (scene sampling) overhead.** B = 64, N = 50, 40 random sites.

| strategy | ms per batch |
|---|---|
| naive per-site eager | 1.48 |
| same under `no_grad` | 1.36 |
| **plate-fused draws** | **0.17** |
| naive, captured as a CUDA graph | 0.30 |
| DT2-style per-sample Python/NumPy loop (lower-bound proxy) | 19.0 |

---

## 1. Guiding principles

1. **The simulator is a generative model; physics is its likelihood.** Every public capability should be expressible as sample, condition, score or fit on p_θ(frames, labels | …). If a physics feature cannot be sampled in batch, labelled, or given a declared gradient path, it is not done.
2. **Resolve, then render; never interleave.** All randomness happens in one compiled pass that produces a Trace. Rendering is a deterministic, pure function of (Trace, parameters). The only exception is detector noise, which draws from the same explicit generator. This gives labels, exact replay, paired rendering and REINFORCE bookkeeping at no extra cost.
3. **Granularity is data, not code.** Whether a quantity is drawn per batch, per image, per object, per frame or per coherent mode is a *plate* annotation on the site. No code path is specific to "shared optics" or "per-image optics"; the planner reads the plates.
4. **Every gradient path has a declared quality, and silence is a bug.** Pathwise-exact, relaxed (biased, with a known knob), score (unbiased, high variance), mean-field (biased, low variance) or none. The engine tests that each learnable gets a non-zero gradient and reports the estimator it relies on.
5. **The planner owns "how"; the Spec never names a numerical method.** Grid spacing, ROI size, padding, number of slices, source points and spectral bins are plan decisions derived from fidelity and *bounds of the distributions*. Because learnable distributions drift, plans carry **bound guards** and re-plan when the guards trip.
6. **Sparse first, dense when needed, one output.** Most DL data generation (tracking, SMLM, holography, iSCAT, darkfield) is sparse [B08]. Sparse per-object rendering is the fast default. Dense volumes compose into the same image-plane field or irradiance.
7. **Analytic and band-limited before voxelised.** Form factors, pupil phase ramps and closed-form blurred primitives give exact position and size gradients. Hard rasterisation is never a default path [B06].
8. **Static shapes, explicit memory.** Shapes come from bounds, so CUDA graphs and optional `torch.compile` work. Memory strategy is a stage property, not a user trick: chunked mode sums, reversible multislice adjoints, procedural slices.
9. **No hidden state.** No global RNG, backend, unit context or output cache. Explicit `torch.Generator`s, a `ParamStore` module, and caches keyed on parameter versions that are bypassed under grad.
10. **Parity is an acceptance test, not the architecture.** DeepTrack2 behaviour is reproduced by an adapter and a `dt2` legacy fidelity preset. DT2's known physics shortcuts are fixed at the standard tiers.

---

## 2. Scope

Scope is chosen so that v1 covers the **minimum fidelity of about 16 of the 18 application families** [B08 §3], plus full DeepTrack2 parity, and **every v1 physics feature is randomisable, learnable and labelable**.

### v1 (ships as 1.0)

| Item | Justification (applications) |
|---|---|
| `prob` layer: Param kinds, distributions with learnable arguments, plates, estimators, Trace, handlers, ParamStore, compiled sampler, counter-based RNG | Sim-to-real, PSF engineering and DeepTrack randomised pipelines all need it. It is the gap no existing tool fills [B03 §3.6, B08 §2.16] |
| Point emitters: sparse ROI and global-spectrum rendering, scalar and vectorial (6-basis dipole) PSFs, Zernike and pixel pupil, Gibson–Lanni layer stack | SMLM, fluorescence tracking, PSF engineering (DTEx252) and bead-stack system ID (uiPSF-class) [B08 §2.4, §2.15] |
| Dense fluorescence: emission density from primitives and labelling, depth-strata OTF, pixel integration | Cell/organelle fluorescence, SIM, confocal and light-sheet at minimum fidelity [B08 §2.5, §2.10] |
| Mie (homogeneous and layered, fp64 stable recurrences) and radiation-corrected dipole scatterers, mapped into the pupil | DeepTrack's workhorse #1: brightfield, holography and darkfield of particles; holographic characterisation [B01 (f), B08 §2.2] |
| Primitives (Sphere, Ellipsoid, Capsule, Cylinder, Box, Gaussian), Groups with priority compositing, SDF escape hatch, `Volume` (voxel tensor or `nn.Module` neural field), Gaussian random-field textures; band-limited lowering | Cells, QPI, SyMBac-like phase contrast, virtual-staining pretraining [B06 §10] |
| Coherent volume solvers: projection, slice-wise Born/Rytov, tilt-corrected BPM with reversible and segmented adjoints and procedural slices | Brightfield, QPI and phase contrast; DT2 `Brightfield` parity; small ODT [B04 §6.5, B07 §4] |
| Illumination: plane waves, Köhler disc/annulus (exact chunked or stochastic Abbe), LED arrays, reference beams, analytic excitation patterns (SIM stripes, TIRF decay, Gaussian sheet) | The transmitted-light family plus two-stage fluorescence modalities [B05 §1.3] |
| Pupil modifiers: soft aperture, defocus (H_exp), Zernike (OPD), pixel phase/amplitude, phase ring, central/annular stops, knife edge, scalar DIC shear | Phase contrast, darkfield, DIC, DPC, Hilbert and PSF engineering are "source × pupil" presets [B05 §3.3] |
| Detection: radiometry, pixel integration, Ideal / Poisson–Gaussian / sCMOS (per-pixel maps) / EMCCD / quantisation / saturation, each with `expected`, `sample` and `log_prob` | Every application; calibration needs likelihoods [B05 §4] |
| Labels: trace sites, per-object tables, masks, instance maps, heatmaps, distance maps, OPD/RI projections, per-object images, stage taps (noise-free image, complex field, exit field) | DeepTrack workflows (`&` pairing, `SampleToMasks`), LodeSTAR, U-Nets [B01 (a)] |
| Time: frame plate, Brownian / drift / confined / rotational diffusion, two-state blinking (expected or stochastic), exposure sub-steps (motion blur) | Tracking (DT sequences, MAGIK) and SMLM (DECODE-style multi-frame) [B08 §2.1, §2.4] |
| `fit`: likelihood fitting (L-BFGS/Adam), CRLB, MMD / sliced-W / summary-statistic losses, distribution fitting, cross-fidelity surrogate gradients, residual slots | Sim-to-real and system ID as headline features [B03 §4.7, B08 §2.16] |
| DeepTrack2 adapter: parity classes, `BatchedDataset`, `sim.as_feature()`, `dt2` legacy preset | Adoption path for existing DT users [B01, "Integration strategy"] |

### v2

| Item | Justification |
|---|---|
| SSNP and multi-layer Born modes of the z-march engine | High-NA and thick cells, ODT with oblique illumination [B04 §2.5] |
| C1 composition (Mie/dipole injected into a marched volume) | Beads and nanoparticles tracked *inside* cells; cell-induced aberration of particle images [B04 §5] |
| Stratified-media dipoles, supercritical-angle fluorescence, full reflection-port S-matrix | Quantitative iSCAT/mass photometry iPSF, near-interface SMLM [B08 §2.3] |
| Jones/tensor-ε samples, vectorial DIC, PolScope/PTI presets | Polarisation microscopy [B08 §2.13] |
| SOCS/WOTF fast paths (fixed optics only) | Label-free training throughput when optics are not learned [B05 §1.2] |
| Low-rank/field-dependent PSFs for dense volumes; light sheet through tissue (BPM excitation) | Light-sheet in tissue, large-FOV fluorescence [B08 §2.7] |
| Multi-sphere Foldy–Lax coupling; importer for precomputed T-matrices | Dense colloids, dimers, rods [B04 §2.9] |
| Meshes (winding-number SDF, polyhedron form factor) | Imported cell shapes [B06 §2.4] |
| Nonlinear excitation maps (2P, saturation, parametric STED effective PSF) | Two-photon/STED datasets [B08 §2.11] |

### Later (v3, or plugins behind stable interfaces)

- Modified Born series / Lippmann–Schwinger reference solver with implicit adjoint, which gives validation ground truth [B04 §2.6].
- A learned-surrogate tier trained against the reference tiers (the "reference solvers manufacture fast ones" loop) [B04 §3.7].
- Interop: TorchGDM/DDA; FDTD far-field import; ray-traced eikonal screens; MCX backgrounds.
- Multi-GPU simulation sharding beyond data-parallel generation (Linux/NCCL only) [B07 §1].

### Explicitly out of scope

| Item | Why |
|---|---|
| FDTD/FEM/DDA inside the engine | No DL data-generation workflow needs them online. They enter as imported amplitudes or T-matrices [B08 §5] |
| Coherent nonlinear optics (SHG/CARS/SRS), Raman | No surveyed pipeline depends on them |
| FLIM/TCSPC photon timing | Only a lifetime *label* is in scope |
| Monte Carlo diffuse transport | A phenomenological differentiable turbid-slab operator replaces it |
| Lens design / ray tracing of objectives | The pupil-map abstraction covers every surveyed work [B08 §5] |
| Biological mechanics and growth | External scene generators via a plugin that emits Spec or Trace values (SyMBac used Pymunk) |
| A general PPL inference engine (MCMC, VI) | Export traces to Pyro/sbi; gradoscopy provides the simulator and likelihoods |
| Electron microscopy, X-ray, instrument control | Out of domain |

---

## 3. Layered architecture and responsibilities

### 3.1 The spine

```
            ┌──────────────── ParamStore (nn.Module): learnables in unconstrained, scaled space ───────────┐
            │                                                                                              │
  Spec  ────┼──► SamplerPlan ──► Trace [B,...] ──► RenderPlan ──► Outputs {frames, labels, taps, meta}      │
  (what)    │    (compiled:       values, present,   (stage list:     │                                     │
  frozen    │     fused draws,    base noise ε,       lower→illum→     ├──► losses (task / MMD / NLL)        │
  dataclass │     bound guards)   log_probs, seed,    interact→        │                                     │
  + Param   │                     validity flags      collect→detect)  │                                     │
  leaves    │         ▲                                    ▲           ▼                                     │
            └─────────┴──────── gradients: pathwise / relaxed / score / mean-field ◄──────── backward ─────────┘
                                             Planner(Spec statics, bounds, Fidelity, outputs, grad-set, device)
```

### 3.2 Layers

```
L8  interop/     deeptrack (adapter, BatchedDataset, from_deeptrack), deepinv export, torchvision tv_tensors
L7  fit/         sim-to-real: likelihood fitting, distribution matching, CRLB, surrogates, residual slots
L6  sim/         Simulator(nn.Module), MultiView, outputs/taps, stream (GPU generation), data writers
L5  plan/        Fidelity, bounds + guards, capabilities, planner rules, Stage/Plan, explain, plan cache
L4  labels/      label renderers over lowered representations (masks, heatmaps, tables, OPD, ...)
L3  repr/        lowered representations (views) + converter registry (exactness, grad quality, cost)
L2  physics/     pure functional kernels (tensors in, tensors out): special, mie, dipole, pupil,
                 propagate, multislice, born, coherence, emitters, detect
L1  scene/       declarative Spec types: Sample/Population/Group/Part/Volume, materials, labelling,
                 illumination, objective/pupil, camera, acquisition, presets
L1' prob/        Param kinds, distributions, plates, estimators, Trace, handlers, ParamStore, SamplerPlan
L0  core/        units, Grid/KGrid, typed tensors (Field, AngularSpectrum, ObjectSpectra, Irradiance,
                 Frames, EmitterSet...), axes, precision policy, RNG derivation, registries
```

**Responsibilities and dependency rules.** Imports go strictly downward, and CI enforces this with `import-linter`.

| Layer | Owns | May import | Must never |
|---|---|---|---|
| `core` | units, grids, typed containers, registries, RNG derivation, precision | torch, numpy | know about scenes or physics |
| `prob` | sampling semantics, gradient estimators, traces, learnable storage | core | know about optics |
| `scene` | *what* to simulate; Params as leaves; presets as recipes | core, prob | name numerical methods, grids, ROI sizes or padding |
| `physics` | numerical kernels, custom autograd Functions, adjoints | core | see a `Param`, a `Trace`, a Spec, or a global |
| `repr` | lowering resolved scene values (tensors) to solver-consumable views; exactness and gradient-quality metadata | core, scene (types only), physics | sample randomness |
| `labels` | ground-truth renderers | core, repr | depend on solver choice |
| `plan` | choosing implementations, discretisation, fusion, validity, cost, memory | all below | execute physics at plan time (except cheap bound calculations) |
| `sim` | orchestration: resolve → render → outputs; streaming | all below | contain physics |
| `fit` | objectives and optimisation loops over a Simulator | sim, prob | reach into stages |
| `interop` | foreign APIs | all | be imported by anything else |

The critical rule is **`physics` never sees `Param`s or the Trace**. Stages receive plain tensors bound by the plan. This keeps kernels `gradcheck`-able, CUDA-graph-capturable and compile-friendly [B09 §4.1].

### 3.3 Package and module tree

```
gradoscopy/
  __init__.py              curated public API (gs.Simulator, gs.learnable, gs.Population, ...)
  units.py                 um=1.0, nm=1e-3, mm=1e3, m=1e6; to_m()/from_m(); px helpers
  core/
    grid.py                Grid2D/Grid3D/KGrid (static shape, spacing, origin), nice FFT sizes (2·3·5·7)
    types.py               Field, AngularSpectrum, ObjectSpectra, Irradiance, Frames, ModeSet,
                           EmitterSet, ParticleSet (dataclasses registered as pytrees)
    axes.py                canonical axis order & names; plate→dim mapping
    precision.py           PrecisionPolicy; autocast-off guard; fp64 phase builders (mod 2π, carrier-free)
    registry.py            typed registries + entry-point loading ("gradoscopy.plugins")
    rng.py                 seed derivation H(base, epoch, step, rank); counter-based (per-index) streams
  prob/
    params.py              Fixed, Learnable, Random, Derived, Choice, Count, Process, ModuleParam, External
    dist.py                distribution wrappers with Param arguments; constraints; quantiles for bounds
    plates.py              batch | image | <population> | frame | mode | Group(k)
    estimators.py          pathwise, relaxed(τ, hard), score(baseline), mean_field, none;
                           poisson: scaled_st (default) | gaussian | exact_nograd | score
    trace.py               Trace: values, present, base noise, log_prob, plates, seed, guards
    handlers.py            condition, replay(values|noise), intervene, mean, detach, block, resample
    store.py               ParamStore(nn.Module): raw params by path, transforms, scale, param groups
    compile.py             SamplerPlan: topo-sort, fuse by (family, plate), guards, CUDA-graph capture
    processes.py           Brownian, OU/confined, drift, rotational diffusion, telegraph blinking
  scene/
    sample.py              Sample, Population, Group, Part, Volume, Environment/LayerStack, constraints
    geometry.py            Point, Sphere, LayeredSphere, Ellipsoid, Capsule, Cylinder, Box, Gaussian, SDF
    fields.py              RandomField (GRF), VoxelSource, NeuralSource
    material.py            Dielectric, Dispersive(Cauchy/Sellmeier/Tabulated), Absorbing, Contrast
    emission.py            Fluorophore, Emission(photons, spectrum, dipole, blinking), Labeling
    illumination.py        PlaneWave, Koehler(Disk/Annulus/Custom), LEDArray, Reference, Pattern,
                           LightSheet, Evanescent
    optics.py              Objective, pupil modifiers, DetectionPath (filters, splitters, pinhole)
    camera.py              Ideal, PoissonGaussian, CMOS, SCMOS(maps), EMCCD, QCMOS
    acquisition.py         focus stacks, channels, SIM phases, LED index, frames/exposure/sub-steps
    presets.py             Widefield, TIRF, SMLM, Brightfield, Darkfield, PhaseContrast, DIC, DPC,
                           InlineHolography, OffAxisHolography, ISCAT, LEDArrayIDT, SIM, Confocal
  repr/
    views.py               EmitterSet, ParticleSet, PrimitiveSoA, OccupancyChannels, RIVolume(procedural),
                           DensityVolume, ProjectedMaps, KSpectrum
    convert.py             @lowering registry (exactness, grad quality, cost), shortest-path search
    raster.py              band-limited occupancy (blurred ball, SDF ramp), patch splat (custom autograd)
    formfactor.py          analytic Fourier transforms (ball, ellipsoid, cylinder, revolution, gaussian)
  physics/
    special.py             fp64 Riccati–Bessel/log-derivative recurrences, Bessel J0/J1/J2 + grads,
                           Zernike (ANSI, recurrence), π_n/τ_n
    mie.py                 coefficients (homog., layered), S1/S2 on 1-D θ grid, pupil mapping
    dipole.py              polarisabilities (radiative, Mie-dressed), dipole angular spectra, 6-basis
    pupil.py               pupil assembly (fused modifiers), soft edges, apodisation, GL OPD, vector factors
    propagate.py           BL-ASM (H_exp), CZT, MFT (IEEE fp32), Fresnel
    multislice.py          z-march engine (BPM, tilt-BPM; SSNP/MLB v2) with reversible/segmented adjoint
    born.py                slice-wise Born/Rytov, form-factor Born
    coherence.py           mode reductions: exact chunked+checkpoint, stochastic, random-phase
    emitters.py            sparse ROI / global-spectrum renderers, depth-strata OTF, Gaussian sprites
    detect.py              radiometry, pixel integration, camera samplers + log_probs
  plan/
    fidelity.py            Fidelity presets, axes, per-stage/per-population overrides, schedules
    bounds.py              static bounds from supports/quantiles + headroom; runtime guards
    capabilities.py        Capabilities, Validity, Cost, GradQuality
    planner.py             rules: routing, selection, discretisation, rewrites (fusion), memory
    stages.py              Stage (static config + pure callable), Plan, explain(), hashing, run(until=)
  labels/                  masks, instance, heatmap, distance, keypoints, boxes, tables, OPD, per-object
  sim/
    simulator.py           Simulator(nn.Module): sample/render/expected/log_prob/stream/gradient_report
    multiview.py           MultiView: one trace → several microscopes (paired data)
    outputs.py             output requests, taps, Batch container
    stream.py              SimStream (prefetch on side stream, CUDA graphs, ReuseNoise, augment)
    io.py                  zarr/HDF5 writers with trace + manifest; bead-stack readers
  fit/
    losses.py              mmd, sliced_wasserstein, summary stats, NLL wrappers
    features.py            QuantileSpectrum, RadialPS, MeanVariance, pretrained-encoder adapters
    likelihood.py          fit.likelihood (L-BFGS/Adam/LM), multistart, laplace()
    distribution.py        fit.distribution (matching loops, CRN, schedules)
    fisher.py              crlb, fisher (jacfwd over ≤ tens of params)
    surrogate.py           cross-fidelity gradients, score-function surrogates (LOO baselines)
    residual.py            residual/nuisance slots (pupil residual, background generator, image CNN)
  interop/
    deeptrack/             features.py, batched.py (BatchedDataset), bridge.py (from_deeptrack), units
    deepinv.py             export linear sub-plans as LinearPhysics
    torchvision.py         tv_tensors wrapping of labels
  testing/                 reference solutions, gradcheck/adjoint helpers, ladder harness, estimator tests,
                           benchmark harness (shipped for plugin authors)
```

### 3.4 Why the whole is more than the sum of its parts

The orthogonal axes are geometry kind × material/labelling × modality preset × fidelity × label type × estimator. They meet only through shared types: Param leaves, Trace, representations and the two-port spectra. The consequences:
- A new primitive that implements `sdf` (and optionally `form_factor`) automatically works in BPM, Born, projection, fluorescence (through labelling), instance masks and distance maps. It is also automatically randomisable and learnable.
- A new modality preset automatically works on every primitive and every fidelity.
- A new estimator applies to every site.
- **Paired and multi-view rendering** (`MultiView`) turns one Trace into brightfield and fluorescence, or clean and noisy, or many focus planes. That is the substrate for virtual staining, denoising and multi-modal pretraining, and it exists only because rendering is a pure function of the trace.

---

## 4. Core data model

### 4.1 Units, coordinates, conventions (fixed on day one)

- **Length: micrometres (µm).** Wavelengths are vacuum values in µm. Spatial frequency `f` is in cycles/µm (grid convention, matching `fftfreq`), and angular wavenumber `k = 2πf` is in rad/µm. Time is in seconds, diffusion in µm²/s, and angles in radians. Photon counts are dimensionless. `gs.units` provides `nm = 1e-3` and `m = 1e6`, so `150*nm` is 0.15 µm. The DT adapter converts from metres and pixels at the boundary. See §13.1 for the rationale; the core argument is optimiser conditioning.
- **Fourier convention.** `F(f) = ∫u(r)e^{−i2πf·r}dr` (the torch.fft forward convention). Time dependence is e^{−iωt}, so propagation along +ζ is e^{+i2πζ√((n/λ)²−|f|²)}. The carrier is subtracted in all propagators, and phase arguments are built in fp64 and reduced mod 2π [B07 §2.4].
- **Sample frame.**
  - Right-handed. x is the image column index direction and y the row index direction, both in object space.
  - The lateral origin is the top-left *corner* of the camera FOV, so pixel (i, j) has its centre at ((j+½)p, (i+½)p). Here p is the object-space pixel pitch (camera pixel / M).
  - z = 0 is the coverslip–sample interface, and **+z points into the sample, away from the objective**. This convention works for inverted and upright stands alike.
  - `focus` is the depth of the nominal focal plane. Defocus Δ = z − focus, and positive defocus means farther from the objective.
  - Solvers march in their own propagation frame. The planner inserts the frame flip.
- **Zernike.** ANSI/OSA indexing with RMS normalisation and a Noll converter. Stored as **OPD in µm**; the user unit can be `"waves"`, `"rad"` or `"um"`. Phase = 2π·OPD/λ, which makes Zernike terms correct for polychromatic light automatically [B05 §8.5].
- **Radiometry.** The pupil is normalised so that the PSF integral equals the collection efficiency. Apodisation is √cosθ for focusing and 1/√cosθ for collection, plus the 1/k_z Jacobian [B05 §3.1].
- **Precision.** float32/complex64 by default. fp64 for special-function recurrences, phase-argument construction and `gradcheck`. complex128 is available as a validation mode. **complex32, fp16 and bf16 are banned** from the physics path: they are slower on Ampere and produced NaN in a 100-slice multislice run [B07 §2.3]. Internals run under `autocast(enabled=False)`. MFT matmuls force IEEE fp32 (no TF32).

### 4.2 Grids and sampling bookkeeping

Every tensor-bearing type carries its grid, so sampling travels with the data [B02 §3.1]. All grids are **static per plan**: integer shapes and float spacings fixed at plan time from bounds.

```
camera grid      H×W at object-space pitch p = pixel/M          (output; labels live here too)
  └─ sim grid    (s·H + 2·pad) × (s·W + 2·pad) at p/s, FFT-nice    s = oversample (Nyquist λ/(4·NA) or fidelity)
      ├─ k-grid  fftfreq of sim grid; pupil disc radius NA/λ_ℓ per wavelength bin (soft edge), shared grid
      ├─ ROI     R×R local grids at p/s for sparse objects; R from support bound (defocus, NA, λ_max)
      ├─ volume  Z slices × sim grid, Δz from fidelity; slices generated procedurally, never stored whole
      └─ pupil   N_pupil² in normalised NA coordinates for CZT/vectorial PSFs (decoupled from image grid)
```

Rules:
1. **Padding, oversampling and ROI size are derived from bounds.** The inputs are the maximum |defocus| over the z-distribution support, NA, the maximum λ and the coherence width. For example, the ROI must cover |Δ|·tanθ_max + a few λ/NA. Exp A showed a 5.7 % error when this is ignored.
2. **One common object-space grid for all wavelengths.** Per-λ pupils are different radii on the same k-grid, and pupil-to-camera sampling uses CZT/MFT when the ratio is non-integer [B02 R2].
3. **Pixel integration is sum-pooling by the integer factor s** in v1. Continuous pixel pitch or magnification (learnable) uses a band-limited MFT evaluation with the pixel sinc in the OTF (v1.x).
4. **Shape-affecting learnables must be bounded.** Examples are NA, λ, pixel size and the scale parameters of the z-distribution. `gs.learnable(1.3, constraint=gs.interval(1.0, 1.45))` is required when the parameter changes grids. The planner raises an error for an unbounded learnable that affects shapes.

### 4.3 Field and intensity types (axes)

Canonical axis order. Y and X are last and contiguous for `fft2` [B02 R1]:

```
Field / AngularSpectrum (dense):  [B, A?, T?, M, Λ, P, Y, X]   complex64
ObjectSpectra (sparse):           [B, N, T?, M, Λ, P, R, R]     complex64  + anchor[B,N,2] int, present[B,N]
Irradiance:                       [B, A?, T?, Λ?, Y, X]         float32   (photons / µm² / exposure)
Frames (output):                  [B, T?, C, H, W]              float32   (ADU | e⁻ | photons), C = flattened A
```

- **B** is the image batch.
- **A** is acquisition axes that produce *separate frames*: focus planes, channels, SIM phases, LED index.
- **T** is time frames. Exposure sub-steps are *modes*, not T.
- **M** is summed incoherent modes (source points, dipole basis, exposure sub-steps, diffuser realisations), with weights `w[B?, M]`.
- **Λ** is wavelength bins with weights (spectrum × filter × QE), treated as summed modes that are kept separate until detection.
- **P** is polarisation: 1 (scalar), 2 (Jones, transverse) or 3 (Cartesian).
- "Summed" and "acquisition" are labels on the axis metadata, which settles the dual meaning of "mode" [B05 §1.4].

Every axis may have size 1 or be broadcast. There is **no fixed 6-D layout** [B02 §2.4] and **no torch named tensors** [B07 §7.2]. Axis identity lives in the dataclass metadata.

Metadata on `Field`: `grid`, `wavelengths[Λ]` (or `[B,Λ]` when λ is sampled per image), `n_medium`, `basis ∈ {scalar, xy, xyz}`, `domain ∈ {space, freq}`, `port ∈ {T, R}`, `mode_weights`, `plane_z`.

### 4.4 Light

- **`ModeSet` is the only coherence representation.** It is a set of mutually incoherent coherent modes, each with an incidence k-vector (or a complex envelope on the sim grid), a Jones or Cartesian polarisation, a wavelength bin and a weight. Modes are an axis, never a separate type hierarchy. There is **no mutual-intensity type**: a 256² grid would need about 34 GB [B02 §2.2].
  - Köhler illumination produces source points from a condenser shape. Grid, quadrature and stochastic sampling are plan choices.
  - Unpolarised light is two modes.
  - A reference beam (holography, iSCAT) is flagged `coherent_with=<mode set>`.
- **`Spectrum(wavelengths, weights)`.** Separate excitation and emission spectra, with a Stokes shift through fluorophores. Bins are chosen by the planner from `fidelity.spectral`, using microsim-style centroid binning [B03 §2.1].
- **`ExcitationField`** covers two-stage modalities (TIRF, SIM, light sheet, confocal excitation). It is an analytic callable or grid irradiance at λ_ex, followed by a pointwise *source map*: linear in v1, with saturating or I² variants in v2 [B05 §1.4].
- **Illumination parameters are Params.** Tilt, condenser NA, LED positions, sheet waist and SIM phase are all randomisable per image and learnable, which covers FPM calibration and learned illumination [B08 §2.9, §2.14].

### 4.5 Sample

**Geometry, material and labelling are separate** [B06 §5]:

```python
Part(geometry, material=Dielectric|Dispersive|Contrast, labels=[Labeling(fluorophore, where=...)])
Population(name, template=Part|Group, count=..., position=..., orientation=..., constraints=[...])
Group(parts={...})       # hierarchical composite (cell = body + nucleoid + texture), priority compositing
Volume(source=tensor|nn.Module, extent, quantity="dn"|"n"|"density")  # voxel or implicit neural field
Sample(*populations, volumes=..., background=..., medium=..., environment=LayerStack(...))
```

- **Environment/LayerStack** (immersion, coverslip, sample medium, with design vs actual values) is **one shared object** read by both the pupil (Gibson–Lanni, TIRF) and the sample (n_medium). This removes the classic "medium index in optics ≠ medium index in scatterers" bug [B06 §5].
- **Overlap semantics are explicit per quantity.** Material-like quantities use priority "over" compositing, linear in ε. Density-like quantities add. `compose="add"` is available for DT2 legacy behaviour [B06 §6].
- **Representations (lowered views, in `repr`).**
  - `EmitterSet`: xyz, photons[, T], species, dipole second moments (6), present, id.
  - `ParticleSet`: centre, radii per layer, complex n per layer per λ, present, id. This is the Mie/dipole-capable view.
  - `PrimitiveSoA`: kind, parameters, centre, rotation (6-D), material id, priority, parent, present, id.
  - `OccupancyChannels` / `RIVolume` (procedural slice generator), `DensityVolume`, `ProjectedMaps`, `KSpectrum`.
- **Conversions** are registered with an **exactness tag** (exact | band-limited-exact | approximate | lossy) and a **gradient-quality tag per input** (smooth | relaxed | zero). The planner finds cheapest admissible paths by shortest path over this small graph [B06 §4.3]. The default voxeliser is patch-based analytic band-limited occupancy: the closed-form blurred ball for spheres and the first-order SDF ramp otherwise, with σ = 0.3h. It reproduces dV/dR to 4 digits, where the hard mask gives 0 [B06 §3].
- **Batching many particles.**
  - Struct-of-arrays per kind, `[B, N_max, …]` plus `present[B, N_max]` (float: hard 0/1 forward, relaxed backward).
  - `N_max` is static per plan. Populations are **bucketed** (N_max ∈ {8, 16, 32, …}) to avoid recompiles and plan churn.
  - Python loops run over *kinds*, never objects.
  - Culling uses bbox + blur margin + PSF/defocus margin.
  - Thousands of objects: the sparse ROI renderer scales linearly (Exp A: 200 emitters at 256² in 5.7 ms for B = 64). Dense lowering of more than 10³ primitives needs the custom-autograd splat, because unfused autograd hit 17.9 GB at 10⁴ spheres [B06 §3.4]. Identical shapes can use `F_shape(k)·NUFFT₁` (optional `pytorch-finufft`).
  - Mie coefficients for all particles × wavelengths are one fp64 batched call.

---

## 5. Physics and the fidelity mechanism

### 5.0 From ML needs to physics interfaces (the derivation)

| ML need | Interface requirement | Consequence for the physics layer |
|---|---|---|
| Learnable distribution parameters (size, RI, density, defocus range, texture) | Every sampled quantity reaches pixels along a smooth path | Band-limited geometry (form factors, blurred primitives, SDF ramps); emitters placed by pupil phase ramps; soft pupil and condenser edges; presence *multiplies* amplitude or photons; no hard thresholds in default paths |
| Per-image randomised optics (domain randomisation) | Optical parameters carry plate dims `[B]` or `[1]` | Pupils and OTFs built for broadcast batches; caches keyed by plate and version; shared-optics fast path (compute once per batch) chosen from the plate |
| Labels from the same trace | Stages expose taps and keep object identity | Sparse renders keep the object index, so per-object images, instance masks and visibility come for free; the noise-free image and complex field are taps |
| Throughput ≥ 10⁴ images/s for fast tiers | Static shapes, fused sampling, sparse rendering | Planner fixes N_max, ROI, grids and chunks from bounds; CUDA graphs over sampler and render; inference_mode fast path |
| Calibration to real data | Deterministic expected image plus an exact likelihood | Detector exposes `expected`/`sample`/`log_prob`; physics is deterministic given the trace; nuisance parameters are explicit |
| Paired data (clean/noisy, BF/FL, multi-focus) | One trace, many renders; interventions | Rendering is pure; stage graph can stop at "expected"; `MultiView` shares the resolve |
| Joint optics + network training (PSF engineering) | Cheap forward + backward w.r.t. pupil | Sparse per-emitter pupils (Exp A: 3.4 ms fwd+bwd at B=64, 128²) |
| Learnable distributions drift | Plans must survive parameter changes | Bounds with headroom + runtime guards + re-plan; shape-affecting learnables bounded |
| Inverse problems and reconstruction | Efficient VJPs; adjoints of linear parts | Custom adjoints (multislice reversible, Born adjoint); `as_operator()` / deepinv export |
| Sim-to-real residuals | Hook points that do not corrupt labels | Declared nuisance slots: pupil residual map, background generator, image-domain residual CNN |
| Mixed-fidelity learning | The same Spec renders at any tier | Tiers are views of one Spec; `compare`; cross-fidelity surrogate gradients |

### 5.1 Stage graph (optical path granularity)

The optical path is simulated at **pupil-plane granularity**, not surface by surface. Objectives are ideal aplanatic systems plus pupil maps. Relay and image-side effects are pupil modifiers or detector-side warps and envelopes [B05 §3.2, B08 §5].

```
 Trace ─► [1 lower] ──────────────► EmitterSet | ParticleSet | PrimitiveSoA→RIVolume/Density | Maps
          [2 illuminate] ─────────► ModeSet (M source points × Λ bins × pol)  |  ExcitationField
                                                    │
          [3 interact]  sparse port ──► ObjectSpectra (pupil-plane, per object, per mode)   (Mie, dipole)
                        dense port  ──► AngularSpectrum on global k-grid, ports {T,R}       (proj/Born/BPM)
                        emitters    ──► (incoherent) EmitterSet × excitation → source strengths
                                                    │
          [4 collect]   pupil chain (fused: soft aperture·apodisation·defocus·Zernike·pixel·ring/stop·
                        GL OPD·Jones) — per-λ radius, per-object field-dependent coefficients (sparse)
                                                    │
          [5 image]     ROI ifft (sparse) | global ifft / CZT / MFT (dense) → image-plane field per mode
                        coherent sums: sparse ROIs scatter-added into field; reference added explicitly
                        |·|² → Σ_modes w_m (chunked+checkpoint)  → Irradiance
                        incoherent emitters: PSF model sparse ROI | depth-strata OTF (collapse thm.)
                                                    │
          [6 detect]    radiometry (Λ weights, exposure) → pixel integration (sum-pool/MFT) →
                        camera.expected → camera.sample (estimator) | camera.log_prob(observed)
```

The coherent-mode reduction is exact when chunked and checkpointed. Measured, it cuts peak memory from 3.74 GB to 0.51 GB with identical gradients [B07 §4.2]. Interference-sensitive modalities compute the interference term explicitly (`2Re(E_r*E_s)`) to avoid float32 cancellation at iSCAT contrasts of 10⁻²–10⁻⁵ [B08 §2.3].

### 5.2 Interaction solvers (v1)

| Solver | Port | Consumes | Validity check (plan- and runtime) | Gradient path | Notes |
|---|---|---|---|---|---|
| `emitters.sparse_roi` | incoherent | EmitterSet | ROI ≥ support(Δ_max, NA, λ) | pathwise, exact in xyz, photons and pupil | Exp A default; vectorial = 6 basis images |
| `emitters.global_spectrum` | incoherent/coherent | Emitter/ParticleSet | none (exact) | exact | For large defocus supports |
| `emitters.strata` | incoherent | DensityVolume, EmitterSet | Δz ≤ λ/(2(n−√(n²−NA²))) | trilinear (piecewise-linear in z) | Dense labels |
| `emitters.gaussian` | incoherent | EmitterSet | warns with fixed dipoles (≤ 40 nm bias [B03]) | exact | "sketch" tier |
| `mie.pupil` | T (R via interface, v1.x) | ParticleSet | isolated sphere; distance to interface > λ | fp64 coefficients, autograd through recurrences | S1/S2 on 1-D θ grid, interpolated to pupil |
| `dipole.pupil` | T, R | ParticleSet | x ≲ 0.3 (≲ 1 Mie-dressed) | exact | iSCAT/darkfield of nanoparticles |
| `projection` | T | Maps / form factor k_z=0 | d ≲ nλ/NA² | exact (analytic) | Thin cells, fast QPI |
| `born` / `rytov` | T, R (Born) | KSpectrum or RIVolume | Slaney: Δn·d < 0.35λ; Rytov guarded at zeros of u_in | custom adjoint | Form-factor path avoids voxels |
| `bpm` (+tilt) | T | RIVolume (procedural) | max angle, Δn·Δz; no backscatter (info) | reversible or segmented adjoint with in-loop VJP into sample params | Free-space (not NA-truncated) inter-slice propagator |

Composition in v1 is **C0**: coherent superposition at the image plane for coherent content, and an intensity sum for emitters. The plan records a warning with the fraction of object pairs closer than a coupling distance, computed from the trace at runtime. C1 (injection into the march) is v2, and the contract already has the hooks (`field_at`, source injection at slices) [B04 §5].

### 5.3 Illumination and coherence handling

- **Exact Abbe.** Source points are a mode axis, reduced in memory-bounded chunks with checkpointing.
- **Stochastic Abbe.** K random source points per image per step. This is unbiased for the expected image and its gradient. At training time it doubles as domain randomisation, and the residual speckle is masked by shot noise. It is the default for learnable condensers and pupils [B05 §1.2].
- **SOCS/WOTF** (v2) apply only when optics are fixed and the sample is declared thin or weak: SVD gradients are unstable for degenerate singular values [B05 §8.3].
- **Emitters are incoherent by construction.** They take the sparse or strata path; random-phase ensembles are never the default [B02 §2.1].

### 5.4 Imaging system (pupil) and detection

- **Pupil.** An ordered product of modifiers, fused into one complex map per (λ, plate) by a rewrite pass [B09 §1.5].
  - Soft sigmoid edges make NA, annulus radii and pinholes learnable [B02 §2.11].
  - H_exp-style defocus makes z learnable [B02 §2.5].
  - Per-object Zernike coefficients (field-dependent maps) cost nothing in the sparse path [B08 §2.4].
  - Vectorial mode applies the Richards–Wolf 3×2 and 2×3 maps with Fresnel t_s/t_p from the LayerStack.
  - psf-generator (MIT) is vendorable as the spherical/Bessel reference and for differentiable Bessel functions [B03 §2.2].
- **Detection.** Camera objects are probabilistic modules:
  - `expected(Φ)` gives the mean ADU;
  - `sample(Φ, estimator)` gives frames;
  - `log_prob(observed, Φ)` covers Poisson, Poisson ⊛ Gaussian (sCMOS, with a documented approximation), and EMCCD (Gamma–Poisson).
  - Camera parameters (gain, offset, read-noise maps, QE) are Params, so they can be calibrated.

### 5.5 How fidelity is expressed

`Fidelity` is a **policy**: a named preset that expands to per-axis choices, overridable per axis, per stage and per population. It is discrete and ordered, not continuous [B09 §2.4].

| Axis | sketch | fast | balanced | accurate | reference |
|---|---|---|---|---|---|
| psf | Gaussian (σ from NA, λ) | scalar pupil, exact k_z | scalar + apodisation + GL | vectorial RW + dipole basis + GL + Fresnel | vectorial, field-dependent |
| emitter render | sprites | sparse ROI | auto (sparse / strata) | auto + field maps | per-emitter everywhere |
| particles | dipole | Mie-in-pupil scalar | Mie (vector pupil) | Mie vector, R-port | + coupling (v2) |
| volumes | projection | projection / Born | BPM | BPM (SSNP v2) | SSNP / MBS (v2+) |
| coherence (Köhler) | coherent | stochastic Abbe K = 2 | exact Abbe K = 12 | exact Abbe K = 32 | converged |
| spectral bins | 1 | 1 | 3 | 5 | ≥ 9 |
| oversample | 1 | Nyquist | 2 | 2–3 | 4 |
| detector | Gaussian | Poisson–Gauss (scaled straight-through) | full camera | full + maps | full |
| photophysics | expected | expected | stochastic | stochastic | sub-frame |
| composition | C0 | C0 | C0 | C0 (C1 v2) | C1/C2 (v2) |

```python
gs.Fidelity("balanced")
gs.Fidelity("fast", psf="vectorial", interaction={"cells": "bpm", "beads": "mie"})   # per population
gs.Fidelity("accurate", coherence=gs.Abbe(points=24, sampling="stochastic"), oversample=2)
gs.Fidelity("dt2")        # legacy parity: NA-truncated inter-slice propagator, additive overlaps, ...
```

**Angle-D extras.**
1. **Fidelity schedules**: `gs.FidelitySchedule({0: "fast", 20_000: "balanced"})` for curricula. Each switch is a re-plan served from the plan cache.
2. **Randomised fidelity as model-mismatch augmentation**: `fidelity=gs.OneOf(["fast", "balanced"], p=[.7, .3])` at batch plate. Each option has its own static plan. This makes DeepSTORM3D's ad-hoc "random blur absorbs mismatch" into a principled, declared augmentation [B03 §2.4].

### 5.6 Planner: how implementations are selected, and when

The planner is a **rule-based, deliberately boring** component of roughly 300–600 lines [B09 §4.5]. It runs at `Simulator` construction (lazily on the first `sample`) and on **re-plan triggers**:
- a fidelity change or schedule step;
- an output-request change;
- a change in the set of parameters that require grad;
- a device or batch-size bucket change;
- a **bound-guard trip**.

Steps:
1. **Static analysis of the Spec.**
   - Representation kinds and contrast mechanisms.
   - Coherence (from the preset).
   - Plates of every optical parameter (shared vs per-image).
   - Learnables and their gradient flags.
   - **Bounds**: supports, or quantiles (10⁻⁴ … 1−10⁻⁴) of every Random site evaluated at the current θ, then inflated by a headroom factor (default 1.5× of the range) when the site depends on learnables.
2. **Requirements** come from Fidelity plus physics triggers. Examples: fixed dipoles and NA > 1.2 make the vectorial PSF preferred ("accurate") or produce a warning ("fast"). A learnable condenser NA forces a soft edge and stochastic or exact Abbe (no SOCS).
3. **Candidate generation** by registry lookup, filtered by capabilities: accepted representation, produced port, physics flags, and **gradient quality for every learnable upstream**. A hard rasteriser is rejected if the radius is learnable.
4. **Validity** is evaluated over the bounds. Because the scene is random, validity is a *probability*: the planner estimates it by drawing a validation batch from the sampler plan (cheap) and reports, for example, "Born invalid for 12 % of sampled spheres". It escalates or warns according to the policy.
5. **Cost ranking** under a memory budget (`torch.cuda.mem_get_info` × safety factor). Each stage has a `(flops, bytes)` model. Chunk sizes for mode sums are derived here.
6. **Discretisation** (grids, ROI, padding, Δz, N_θ for Mie, n_max bucket).
7. **Rewrite pass**: fuse pupil modifiers, merge homogeneous propagations, fold pixel integration into OTFs, drop unused outputs, split the plan at `expected` if noise re-draws are requested.
8. **Emit the Plan**, which is a flat list of named stages with static configs, plus the sampler plan and label plan. Plans are cached by a hash of (static Spec structure, bounds bucket, fidelity, outputs, grad set, device, B bucket). `plan.explain()` prints choices, rejected alternatives and reasons, approximations and warnings. `plan.run(trace, until="collect")` supports debugging [B09 §4.7.12].

**Bound guards.** After each resolve, each guarded site computes a flag tensor (value outside planned bounds). The flag is copied non-blocking to pinned host memory and checked **one step later**, so there is no sync on the hot path. The policy is `clamp` (default: clamp, count, and schedule a re-plan at the next step), `replan` or `error`. This is what lets a learnable z-range or size distribution grow during calibration without silent ROI truncation (Exp A's 5.7 % failure mode).

### 5.7 Validity and cross-fidelity relations

- **Every lower tier is an approximation of the same Spec.** Parameters a tier cannot represent follow a declared **fold policy**: `ignore+warn`, `approximate` (for example Gibson–Lanni → effective defocus + Gaussian widening in "sketch"; fixed dipole → isotropic in scalar, with a warning), or `error`.
- **Ladder tests** (§11) measure error vs the next tier across regime grids. Their tables *replace* literature rules of thumb as planner thresholds over time [B04 §6.8].
- `gs.compare(sim, "fast", "accurate", n=64)` renders the same traces under both plans. It reports per-output relative L2, bias in *label-relevant statistics* (for example the centroid bias of localisations), time and memory.
- **Cross-fidelity surrogate gradients**: `Fidelity(forward="accurate", backward="fast")` evaluates y = y_hi.detach() + y_lo − y_lo.detach() from a shared trace [B07 §6.4]. It is intended for calibration where the forward must be accurate but the backward is too expensive. It is documented as biased, and `gradient_report` flags it.

---

## 6. Parameters, randomness and learnability

### 6.1 Param kinds

| Kind | Constructor | Semantics | Storage | Example |
|---|---|---|---|---|
| Fixed | plain value / `gs.fixed(v)` | constant | buffer | `NA=1.4` |
| Learnable | `gs.learnable(init, constraint, plate="batch", scale=None)` | deterministic, optimised | `nn.Parameter` in ParamStore (unconstrained raw) | Zernike vector, pupil map, per-bead xyz (`plate="image"`) |
| Random | `D.Normal(loc, scale, plate=...)`; arguments may be Params | drawn per plate; hyperparameters learnable | nothing (Trace) | `D.LogNormal(gs.learnable(-1.2), gs.learnable(.2,"positive"))` |
| Derived | `gs.derived(fn, *paths)` | deterministic function of other sites | nothing | `z = 2*radius` |
| Choice | `gs.OneOf(options, p)` | discrete choice among sub-Specs | Trace (index + log_prob) | particle kind, modality variant |
| Count | `D.Binomial(total, p)`, `D.Poisson(rate, max)` | presence slots `[B, N_max]` | Trace | particles per image |
| Process | `gs.processes.Brownian(D, dt)` | time series on the frame plate | Trace | trajectories, blinking |
| ModuleParam | `gs.module(nn.Module)` | arbitrary learnable function | ParamStore submodule | neural field, background generator, residual CNN |
| External | callable, DT lambda | per-sample Python, **non-differentiable** | Trace | legacy DeepTrack rules |

**Learnable storage and conditioning.**
- Learnables live in one `ParamStore(nn.Module)`, keyed by Spec path (`"beads/radius/loc"`), in **unconstrained space**:
  - positive → softplus⁻¹, or log for scale-free quantities such as photons;
  - `interval(a, b)` → logit;
  - `unit_vector`, `rotation6d`.
- An optional `scale` normalises magnitude.
- `sim.params[path]` returns the constrained view. `sim.param_groups(lr={"optics": 1e-2, "sample": 3e-2})` builds optimiser groups by Spec branch.
- Specs are immutable. Sharing one `Learnable` object between two Specs shares the parameter (ParamStore dedupes by identity), which is how a fitted objective is reused in a training simulator.

### 6.2 Plates: sampling granularity

| Plate | Value shape | Typical use |
|---|---|---|
| `"batch"` | `[1]` (broadcast) | shared optics, global aberration per batch, per-batch λ |
| `"image"` (default for microscope Random sites) | `[B]` | per-image defocus, tilt, SNR, background, per-image λ |
| `"<population>"` (default for object sites) | `[B, N_max]` | radius, n, position, photons |
| `"frame"` | `[B, T]` or `[B, T, N_max]` nested | trajectories, blinking, drift |
| `"mode"` | `[B, M]` | stochastic Abbe source points, diffuser realisations |
| `gs.plates.Group(k)` | `[B/k]` repeated k× | "same sample, k views", multi-focus augmentation, DT2 `Reuse` semantics |

Hierarchical priors fall out of nesting. A per-image mean radius with per-object scatter models sample-preparation variability, which matters for sim-to-real realism:

```python
radius = D.LogNormal(loc=D.Normal(gs.learnable(-1.2), gs.learnable(0.1, "positive"), plate="image"),
                     scale=gs.learnable(0.15, "positive"))          # plate defaults to the population
```

### 6.3 Gradient estimators (declared per site, defaults measured)

| Site type | Default | Alternatives | Evidence |
|---|---|---|---|
| Continuous with rsample (Normal, LogNormal, Uniform bounds, Gamma/Beta implicit, VonMises, MVN, truncated via icdf) | pathwise | none | [B07 §6.2] |
| Shot noise (Poisson) | **scaled straight-through**: exact Poisson forward, √λ-consistent backward | Gaussian reparam; `exact_nograd`; score | Exp B: scaled straight-through is unbiased on a variance-sensitive loss (−19.95 vs −20); plain straight-through gives **0**; score function std is ~6000× larger |
| EM gain | Gamma rsample (implicit) | — | [B05 §4] |
| Quantisation, saturation | straight-through round; hard or soft clamp | soft clamp for learning | [B07 §6.3] |
| Presence / Count (learnable) | **straight-through Concrete τ = 0.5** (hard forward, so images and labels are exact) | score + leave-one-out baseline (unbiased, ~2× std); mean-field (8× lower std, biased) | Exp B: soft τ = 1 gives the **wrong sign**; straight-through τ = 0.5 gives −465 vs true −473 |
| Categorical Choice | straight-through Gumbel-softmax | score + leave-one-out | Meta-Sim needed REINFORCE [B09 §1.10] |
| Photophysics (stochastic) | score (log_prob of state sequence) | `expected` mode = mean-field occupancy (differentiable in rates) | [B06 §5] |
| Rejection constraints (`NonOverlapping(mode="reject")`) | pathwise conditional, **flagged biased** | `mode="repel"`: unrolled differentiable repulsion (default when positions are learnable) | [B06 §7] |
| External (DT lambdas) | none | — | — |

**Count semantics.** A non-learnable count uses the exact distribution: the first n slots are present. A learnable count defaults to **Binomial slots** with per-slot relaxed presence p = E[n]/N_max. That is a documented approximation of Poisson, and the planner warns when E[n]/N_max > 0.25, where the variance error exceeds 25 %. `estimator=gs.est.score()` keeps the exact Poisson.

**Score-function bookkeeping.** The Trace collects per-image `log_prob` (summed over that image's score sites). `batch.surrogate(per_image_loss)` returns Σᵢ(Lᵢ − bᵢ)ᵢ.detach()·log pᵢ with a leave-one-out baseline across the batch. Batch-level losses such as MMD have no per-image decomposition, so `fit.distribution` uses K sub-batches and a between-sub-batch baseline (documented cost ×K).

**Gradient coverage audit.** `sim.gradient_report()` combines:
- a *static* pass: the planner walks each learnable's path through stage and converter gradient-quality tags;
- a *dynamic* probe: backward of a random linear functional of all outputs over 2–3 batches.

```
learnable                       static path                  estimator            probe |g|   status
beads/radius/loc                form factor → Mie pupil      pathwise             3.1e-02     ok
beads/count/p                   presence × amplitude         relaxed ST τ=0.5     1.2e-01     ok (biased≈small)
camera/read_noise               Poisson–Gauss sampler        pathwise             4.0e-04     ok
condenser/NA_outer              hard annulus in 'fast'       —                    0           ZERO → set soft_edge=True
emission/photons (loss on var)  plain_st Poisson             —                    0           ZERO → use scaled_st
```

### 6.4 Batching semantics

- **B is images.** Every resolved value has a leading B or broadcasts from `[1]`.
- Per-image optics are simply `[B]` parameters. The planner detects shared-plate optics and builds pupils and OTFs once per batch; per-image optics give batched pupils `[B, …]`, cheap because pupils are small [B05 §8.4].
- Variable counts use padding plus presence, bucketed. The packed view `[ΣN, …]` is available to per-object kernels (the PyTorch3D pattern [B09 §1.2]).
- `sim.render(trace, params=...)` supports `torch.func.functional_call`-style parameter swapping (ensembles, meta-learning) [B07 §3.2].

### 6.5 Seeding and reproducibility

- **Explicit generators only.** `sim.sample(seed=gs.seed(base, epoch, step, rank))` derives a `torch.Generator(device)`, so there is no global RNG.
- **Counter-based mode** (`rng="counter"`) generates base noise from `(seed, image_index, site_id)` with Philox-style integer hashing in torch. Image *i* is then identical regardless of batch composition or worker count. This gives DT `Dataset[i]` semantics and deterministic distributed generation at ~1.3× the sampler cost (estimate).
- **The Trace stores base noise ε** for pathwise sites (small: `[B, N]` per site). It enables two replay modes:
  - `gs.replay(trace)` (values mode): bit-exact re-render of a stored batch.
  - `gs.replay(trace, mode="noise")`: re-transform ε under the *current* θ. These are **common random numbers**, which reduce gradient variance in calibration and give sample-average-approximation fitting with deterministic L-BFGS.
- Monte Carlo stages (stochastic Abbe, random-phase) take a separate `seed_grad` stream, used when a backward pass recomputes (checkpointing, `preserve_rng_state=True`) [B09 §1.1].
- Outputs carry `meta = {spec_hash, plan_hash, seed, gradoscopy_version, torch_version, device}`.

### 6.6 Ground truth and label extraction

Labels are **requested outputs of the same Trace**, paid for only when requested [B09 §2.8]:
1. **Trace sites**, addressed by path (`"beads.radius"`, `"beads.present"`, `"objective.pupil.zernike"`). These are free regression targets.
2. **Label renderers** over lowered representations at the camera grid, or any grid:
   - `InstanceMask`, `SemanticMask`, `Heatmap` (with optional per-object weights, for example radius);
   - `DistanceMap`, `Keypoints`, `Boxes`, `EmitterTable` (padded plus mask, with a visibility/in-FOV flag);
   - `OPD` / `RIProjection`, `DensityVolume`;
   - `PerObjectImages` (from sparse ROIs, so instance-separated images are free).
3. **Stage taps**: `expected` (noise-free), `image_field` (complex), `exit_field`, `exit_phase`, `psf`, per-stage irradiance.

Labels use `tv_tensors` types when torchvision is installed, so downstream augmentation stays aligned [B09 §1.5].

**Interventions for paired data.**
- `with gs.intervene({"camera.noise": "off"})` gives clean targets.
- `MultiView({"bf": sim_bf, "fl": sim_fl}, share="sample")` gives virtual-staining pairs.
- `gs.plates.Group(k)` gives k views of one sample.
- `sim.stream(reuse=gs.ReuseNoise(k))` re-draws only camera noise k times from one expected image.

---

## 7. Composition and extension

### 7.1 Registries and mechanics

There is one registry per extension point: `geometry`, `material`, `distribution`, `estimator`, `lowering`, `stage` (solvers of all kinds: psf, interaction, propagate, coherence, detect), `camera`, `pupil_modifier`, `label`, `preset`, `loss`/`feature`.
- Population is by decorators and by `importlib.metadata.entry_points(group="gradoscopy.plugins")`, loaded lazily, plus a `GRADOSCOPY_PLUGINS` environment variable for development [B09 §1.3].
- Duplicate names raise an error unless `override=True`.
- Plugins declare the gradoscopy API version they target.
- Saved Specs use **registry names plus a schema version**, never import paths [B09 §1.11].

**The contract is data types plus capability declarations, not subclass internals.** A plugin states what it accepts and produces, its physics flags, ports, validity predicate, cost model, and gradient quality per input.

### 7.2 Adding a primitive

```python
@gs.register.geometry("torus", params=("major", "minor"))
class Torus(gs.Geometry):
    """Spec-level: params may be any Param kind. Kernels operate on resolved tensors [B,N]."""
    @staticmethod
    def bbox(p):                              # p: dict of tensors; returns [B,N,2,3] local box
        r = p["major"] + p["minor"]
        return gs.geom.local_box(r, r, p["minor"])
    @staticmethod
    def sdf(p, x):                            # x: [..., 3] local coordinates (µm)
        q = torch.stack([x[..., :2].norm(dim=-1) - p["major"][..., None], x[..., 2]], -1)
        return q.norm(dim=-1) - p["minor"][..., None]
    # optional: form_factor(p, k) -> complex; project(p, xy) -> thickness
```

Registering `sdf` automatically adds lowerings: occupancy (SDF ramp, `approx(first-order)`, grad smooth), projection (numerical), masks and distance-map labels. The torus is then usable in BPM, Born, fluorescence (via `Labeling`), all labels, and every Param kind.

### 7.3 Adding a solver

```python
@gs.register.stage("interaction", "ssnp")
class SSNP(gs.Stage):
    caps = gs.Capabilities(
        accepts=(gs.repr.RIVolume,), produces=gs.AngularSpectrum, ports={"T"},
        physics={"multiple_forward_scattering", "nonparaxial"},
        linear_in_incident=True, grad=gs.Grad.CUSTOM_VJP,
        grad_quality={"volume": "smooth"},
    )
    def validity(self, b: gs.Bounds) -> list[gs.Violation]:
        return [gs.Violation("reflections not modelled", "info")] if b.max_dn > 0.05 else []
    def cost(self, p: gs.Problem) -> gs.Cost:
        return gs.Cost(flops=4 * p.n_slices * p.fft_flops, bytes=6 * p.field_bytes)
    def configure(self, p: gs.Problem, fid: gs.Fidelity) -> dict:     # plan time, static
        return dict(dz=fid.slice_thickness(p), memory=fid.memory or "reversible")
    @staticmethod
    def run(modes: gs.ModeSet, vol: gs.repr.RIVolume, *, dz: float, memory: str) -> gs.AngularSpectrum:
        ...                                                          # pure torch; custom autograd.Function
```

The testing kit (`gradoscopy.testing`) gives plugin authors `assert_gradcheck(stage)`, `assert_adjoint(stage)`, `ladder(stage, reference="bpm", regimes=...)` and `benchmark(stage)`. Registering a stage with a ladder entry adds it to CI tables.

### 7.4 Adding a modality

Modalities are **recipes** that wire illumination, pupil modifiers, detection path and camera. They are not physics:

```python
@gs.register.preset("hoffman")
def Hoffman(*, wavelength, objective, slit, modulator, camera, **kw):
    return gs.Microscope(
        illumination=gs.Koehler(source=gs.sources.Slit(**slit), wavelength=wavelength),
        objective=objective.with_pupil(gs.pupil.GradedAmplitude(**modulator)),
        camera=camera, **kw)
```

### 7.5 Adding labels, losses, cameras

```python
@gs.register.label("orientation_field")
def orientation_field(views: gs.repr.Views, grid: gs.Grid2D, *, population: str) -> torch.Tensor: ...

@gs.register.camera("spad_array")
class SPAD(gs.Camera):
    def expected(self, phi, p): ...
    def sample(self, phi, p, gen, estimator): ...
    def log_prob(self, observed, phi, p): ...
```

### 7.6 Operator composition

Inside a plan, stages compose as a typed list with typed boundaries. Composition is checked at plan time (for example, a `FieldOp` cannot follow `Detect`). `gs.stack(...)` creates acquisition axes, and `IncoherentSum(over=...)` expresses mode reductions [B09 §2.3]. Linear sub-plans (Born, pupil filtering, PSF convolution) expose `adjoint`, analytic or via autodiff fallback, and export to `deepinv.physics.LinearPhysics`. Users do not compose raw operators for standard work; presets plus Fidelity cover that. Advanced users can call `gs.physics.*` functionally. There is **no operator-overload soup**: `>>`, `^` and `&` exist only in the DT adapter [B09 §3.7].

---

## 8. Performance and memory strategy

### 8.1 Throughput targets per tier (single RTX 3090)

| Workload | Tier | Target | Basis |
|---|---|---|---|
| Sparse fluorescence, 128², 50 emitters, scalar | fast | ≥ 2×10⁴ images/s (fwd), ≥ 10⁴ (fwd+bwd) | Exp A: 51 k / 18.7 k measured |
| Same, vectorial isotropic (6 basis) | accurate | ≥ 5×10³ images/s | ≈ Exp A / 6 |
| Mie holography, 128², ≤ 10 particles, large defocus (global spectrum) | fast | ≥ 3×10³ images/s | Exp A fullframe 5.2 k at N=50; Mie is 1-D θ table + gather |
| Brightfield BPM 256² × 64 slices, 1 mode | balanced | ≈ 10³ images/s | ~0.7 ms per slice step for 64×256² (scaled from 2.5–3 ms at 64×512² [B07 §4.1]) |
| Same with 24 exact Abbe modes | accurate | ≈ 40–60 images/s | ×24; a calibration and validation tier |
| Calibration step (B = 256, 64², MMD) | — | ≤ 10 ms | Exp C: 6.1 ms |

### 8.2 Techniques, in order of leverage

1. **Sparse-first rendering.** 3.8–5.7× faster (forward) than dense strata and 10× faster than full-frame ramps at 128²/N=50 (Exp A). The planner picks per population using the crossover model N·R²·(cost per ROI) vs N_z·HW·log(HW).
2. **Compiled sampler.** Sites are fused by (family, plate) into single draws: 1.48 → 0.17 ms (Exp D). Without fusion the resolve costs as much as the entire fast render.
3. **`inference_mode` when nothing requires grad**, the normal data-generation case. It enables in-place ops and skips autograd bookkeeping [B07 §4.2].
4. **CUDA graphs over sampler and render** for static-shape plans. Resolve drops from 1.48 to 0.30 ms (Exp D); small-field render gains 2.9× [B07 §3.3]. Graph-safe generators are used. This is the **v1 accelerator that works on Windows without Triton** [B07 §1].
5. **Memory as a stage property.**
   - Coherent-mode sums use exact chunking plus checkpointing: 3.74 → 0.51 GB, gradients identical [B07 §4.2].
   - Multislice ships a **reversible adjoint** (0.80 GB vs 3.0 GB naive for B=16, 512², 32 slices; gradients to 7×10⁻⁶) and a segmented variant for absorbing samples [B07 §4.1].
   - **Procedural slices with in-loop VJP** avoid materialising the 16 GiB volume and its 16 GiB gradient at 64×256×512² [B07 §4.2.4]. This is the single most important memory decision for dense coherent paths.
   - The patch rasteriser is a custom `autograd.Function` (later a Triton kernel), because unfused autograd hit 17.9 GB at 10⁴ spheres [B06 §3.4].
   - Chunk sizes come from `mem_get_info` and per-stage byte models.
6. **Caches keyed on (plate, parameter version, grid), bypassed under grad.** Examples: shared fixed-optics OTF stacks, Mie θ-tables for fixed particles, Zernike bases [B02 §3.8.4, B03 §3.5.6].
7. **FFT hygiene.**
   - complex64; `rfft2` for all real intensity convolutions.
   - Sizes padded to 2·3·5·7-smooth: prime sizes cost up to 1.9× [B07 §2.3].
   - Standardised grid sizes per bucket to avoid cuFFT plan churn.
   - `PYTORCH_ALLOC_CONF=expandable_segments:True`.
8. **Fusion targets** (optional Triton via `torch.library.triton_op`, each with a pure-torch reference): phase-screen multiply, on-the-fly H(k) multiply, |u|² accumulation, and the rasteriser. Elementwise multiplies cost 0.525 ms vs 0.705 ms per FFT at 64×512², so fusion is the real lever, not faster FFTs [B07 §2.3]. `torch.compile` is optional on real-view step functions, because Inductor does not fuse complex ops [B07 §3.1].
9. **Mie.** Coefficients are computed in fp64 (complex64 D_n errs by 1–5 % [B07 §5.2]) and batched over particles × λ in one call. The recurrence is launch-bound (~3 ms regardless of dtype for 4096 particles), so the call is captured in the graph. Fields are synthesised from 1-D S1/S2(θ) tables interpolated to the pupil, never with per-pixel order sums: the naive approach would need about 75 GB for 100 particles × batch 32 [B08 §2.2].

### 8.3 Streaming GPU-side generation

- `sim.stream(seed, prefetch=1, reuse=None, graph=True)` is an infinite iterator that generates in the **main process on the GPU**. There is no CUDA in DataLoader workers, which would mean spawn plus a CUDA context each [B07 §7.4].
- `prefetch=1` renders batch k+1 on a side stream while the model consumes batch k. This helps when either side is launch-bound and is a no-op when the GPU is saturated.
- `reuse=gs.ReuseNoise(k)` splits the plan at `expected` and re-samples the camera k times, which costs only the detector stage. `reuse=gs.Augment(k, flips=True, rot90=True)` applies exact geometric augmentations with label co-transforms. This generalises DT2's `Reuse` [B01 (d)].
- When the simulator has learnables that are being trained jointly (PSF engineering, adversarial sim-to-real), the stream runs in grad mode and gradients flow into `sim.parameters()`.
- `gs.io.write(sim, n, "ds.zarr")` materialises fixed datasets with per-sample traces and a manifest (spec hash, plan hash, seeds).
- Multi-GPU: rank-local generation with seeds derived from the rank, plus DDP for the network (Linux).

---

## 9. User-facing API

All examples use:

```python
import math, torch
import gradoscopy as gs
from gradoscopy import dist as D
from gradoscopy.units import um, nm
```

### (a) Holography of Mie spheres in a GPU training loop with a learnable size distribution

```python
# ---------------- WHAT: randomized scene; the size distribution is learnable ----------------
beads = gs.Population(
    "beads",
    count=D.Binomial(total=8, p=0.6),                         # 8 slots, exact (not learned)
    geometry=gs.Sphere(radius=D.LogNormal(
        loc=gs.learnable(math.log(0.25)),                     # log(µm)
        scale=gs.learnable(0.10, constraint="positive"))),
    material=gs.Dielectric(n=D.Uniform(1.40, 1.60)),          # plate defaults to "beads"
    position=gs.UniformIn(fov=True, margin=2 * um, z=D.Uniform(-6 * um, 6 * um)),
)
sample = gs.Sample(beads, medium=gs.media.Water())

scope = gs.presets.InlineHolography(
    wavelength=532 * nm,
    objective=gs.Objective(
        NA=0.8, magnification=40, immersion=gs.media.Air(),
        pupil=[gs.pupil.Zernike({"spherical": D.Normal(0.0, 0.03, plate="image")}, unit="waves")]),
    illumination=gs.PlaneWave(tilt=D.Normal(0.0, 0.005, plate="image")),     # rad, nuisance
    camera=gs.cameras.CMOS(pixel=3.45 * um, shape=(128, 128), read_noise=2.0,
                           photons_per_pixel=D.Uniform(300.0, 3000.0, plate="image")),
)

sim = gs.Simulator(
    sample, scope, fidelity="fast", batch_size=64, device="cuda",
    outputs={
        "x": "frames",                                        # [64, 1, 128, 128]
        "heat": gs.labels.Heatmap("beads", sigma_px=1.0),     # [64, 1, 128, 128]
        "r": "beads.radius", "n": "beads.n",                  # [64, 8]
        "xyz": "beads.position", "present": "beads.present",  # [64, 8, 3], [64, 8]
    },
)
print(sim.plan.explain())
print(sim.gradient_report())          # both learnables: pathwise via form factor → Mie pupil

# ---------------- Stage 1: calibrate the size distribution on UNLABELED experimental holograms ---
real = gs.io.ImageStream("data/holograms/*.tif", crop=128, batch_size=64, device="cuda")
feats = gs.fit.features.QuantileSpectrum(quantiles=32, radial_bins=16)   # translation invariant
opt = torch.optim.Adam(sim.parameters(), lr=3e-2)                         # only the 2 learnables
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)
for step, x_real in zip(range(600), real):
    fake = sim.sample(seed=gs.seed(0, step))
    loss = gs.fit.mmd(feats(fake["x"]), feats(x_real), bandwidths=(1, 2, 4, 8))
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
print("median r [µm]:", sim.params["beads.radius.loc"].exp().item(),
      "σ:", sim.params["beads.radius.scale"].item())

# ---------------- Stage 2: train a network on the calibrated generator, streamed on the GPU ------
sim.freeze()                                   # learnables → Fixed: inference_mode + CUDA graph plan
net = MyUNet(in_ch=1, out_ch=3).cuda()          # heatmap, radius map, index map
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
for step, b in zip(range(50_000), sim.stream(seed=gs.seed(1), prefetch=1, reuse=gs.ReuseNoise(2))):
    out = net(b["x"])
    loss = gs.losses.heatmap_focal(out[:, :1], b["heat"]) + \
           gs.losses.masked_regression(out[:, 1:], b, keys=("r", "n"), at="xyz", mask="present")
    opt.zero_grad(); loss.backward(); opt.step()
```

Variant: **joint** calibration and training with alternating steps, or `gs.fit.distribution(sim, real, task_loss=...)` for task-driven bilevel objectives. Here the simulator stays in grad mode and the stream yields differentiable batches.

### (b) SMLM: fit a Zernike + pixel pupil to a bead stack, then generate training data with it

```python
stack, meta = gs.io.read_bead_stack("beads/*.tif", roi=31)   # [24, 41, 31, 31] ADU, meta.z_steps [41]

objective = gs.Objective(
    NA=1.45, magnification=100, immersion=gs.media.Oil(),
    stack=gs.LayerStack(coverslip=gs.media.Glass(thickness=170 * um), sample=gs.media.Water()),
    pupil=[
        gs.pupil.Zernike(gs.learnable(torch.zeros(18)), index="ansi", start=4, unit="waves"),
        gs.pupil.PixelPupil(phase=gs.learnable(torch.zeros(64, 64)),
                            amplitude=gs.learnable(torch.ones(64, 64), constraint="positive"),
                            regularizer=gs.reg.Laplacian(1e-3)),
    ],
)
bead = gs.Population(
    "bead", count=1, geometry=gs.Sphere(radius=50 * nm),            # finite bead as emission volume
    emission=gs.Emission(photons=gs.learnable(meta.photons0, "positive", plate="image"),
                         dipole="isotropic"),
    position=gs.learnable(meta.xyz0, plate="image"),                 # [24, 3] per-image latent
)
fit_scope = gs.presets.Widefield(
    wavelength=680 * nm, objective=objective,
    camera=gs.cameras.SCMOS.from_maps("cam/maps.npz", rois=meta.rois),     # per-pixel offset/var/gain
    background=gs.learnable(meta.bg0, "positive", plate="image"),
    acquisition=gs.Acquisition(focus=meta.z_steps),                  # 41 planes → C axis
)
fit_sim = gs.Simulator(gs.Sample(bead), fit_scope, batch_size=24, device="cuda",
                       fidelity=gs.Fidelity("accurate", psf="vectorial"),
                       outputs={"mu": "expected"})
res = gs.fit.likelihood(fit_sim, observed=stack, optimizer="lbfgs", steps=150)   # camera.log_prob
print(res.summary())                     # NLL, per-bead χ², Zernike table [waves], pupil maps
crlb = gs.fit.crlb(fit_sim, sites=["bead.position"], at={"bead.emission.photons": 2000.0})

# ---- training generator that REUSES the fitted objective (shared spec, frozen values) ----------
objective_fit = res.frozen(objective)
emitters = gs.Population(
    "emitters",
    count=D.Poisson(gs.learnable(12.0, "positive"), max=64),           # tunable to real frames later
    geometry=gs.Point(),
    position=gs.UniformIn(fov=True, z=D.Uniform(-0.7 * um, 0.7 * um)),
    emission=gs.Emission(
        photons=D.LogNormal(gs.learnable(math.log(3000.0)), gs.learnable(0.5, "positive")),
        blinking=gs.photophysics.TwoState(k_on=0.05, k_off=0.5, mode="stochastic"),
        dipole="isotropic"),
)
smlm_scope = gs.presets.Widefield(
    wavelength=680 * nm, objective=objective_fit,
    camera=gs.cameras.SCMOS.from_maps("cam/maps.npz", roi=(0, 0, 64, 64)),
    background=D.Uniform(20.0, 120.0, plate="image"),
    acquisition=gs.Acquisition(frames=3, exposure=0.01),               # DECODE-style 3-frame context
)
train_sim = gs.Simulator(gs.Sample(emitters), smlm_scope, batch_size=128, device="cuda",
                         fidelity=gs.Fidelity("balanced", psf="vectorial"),
                         outputs={"x": "frames",                               # [128, 3, 1, 64, 64]
                                  "em": gs.labels.EmitterTable("emitters", frame=1)})
```

The same `objective` object can instead stay learnable inside `train_sim` for **end-to-end PSF engineering** (DTEx252/DeepSTORM3D): train the pupil phase jointly with the localisation network through `sim.stream(...)` in grad mode. The sparse path makes this cheap (Exp A: 3.4 ms fwd+bwd per 64 images, scalar).

### (c) A cell in phase contrast (and QPI) through multislice

```python
cell = gs.Group(parts={
    "body": gs.Part(gs.Capsule(length=D.Uniform(3 * um, 8 * um), radius=D.Uniform(0.45 * um, 0.7 * um)),
                    material=gs.Dielectric(n=D.Normal(1.378, 0.004, plate="cells"))),
    "nucleoid": gs.Part(gs.Ellipsoid(semi_axes=gs.derived(
                            lambda L, r: torch.stack([0.3 * L, 0.6 * r, 0.6 * r], -1),
                            "cells.body.length", "cells.body.radius")),
                        material=gs.Dielectric(n=1.386), parent="body"),
    "texture": gs.Part(gs.RandomField(corr_length=gs.learnable(0.3 * um, "positive"),
                                      std=gs.learnable(0.003, "positive"), spectrum="matern32"),
                       parent="body", compose="add"),                  # Δn texture masked by the body
})
cells = gs.Population("cells", template=cell, count=D.Binomial(total=16, p=0.6),
                      position=gs.UniformIn(fov=True, z=0.8 * um),
                      orientation=gs.rotations.UniformInPlane(),
                      constraints=[gs.NonOverlapping(gap=0.2 * um, mode="repel", steps=8)])
sample = gs.Sample(cells, medium=gs.media.Medium(n=1.335),
                   background=gs.Part(gs.RandomField(corr_length=5 * um, std=5e-4), compose="add"))

scope = gs.presets.PhaseContrast(
    wavelength=550 * nm,
    objective=gs.Objective(NA=1.25, magnification=100, immersion=gs.media.Oil()),
    annulus=gs.Annulus(NA_inner=0.30, NA_outer=0.38),           # condenser; conjugate ring automatic
    ring=gs.pupil.PhaseRing(phase=-math.pi / 2, transmission=gs.learnable(0.25, gs.interval(0.05, 1.0))),
    camera=gs.cameras.SCMOS(pixel=6.5 * um, shape=(256, 256), photons_per_pixel=2000.0),
)
sim = gs.Simulator(
    sample, scope, batch_size=8, device="cuda",
    fidelity=gs.Fidelity("accurate", interaction="bpm", coherence=gs.Abbe(points=24), oversample=2),
    outputs={"x": "frames", "inst": gs.labels.InstanceMask("cells"),
             "opd": gs.labels.OPD(), "phase": "taps.exit_phase"},
)
# SyMBac-style automatic realism matching: learn texture statistics and ring transmission
gs.fit.distribution(sim, real=gs.io.ImageStream("pc/*.tif", crop=256, batch_size=8),
                    features=gs.fit.features.QuantileSpectrum(), steps=300, lr=1e-2)

# QPI of an implicit (neural-field) sample through the same multislice engine
field = gs.Volume(source=SirenMLP(in_dim=3, out_dim=1), quantity="dn",
                  extent=((0, 0, 0), (25.6 * um, 25.6 * um, 6 * um)))
qpi = gs.Simulator(gs.Sample(volumes=[field]),
                   gs.presets.OffAxisHolography(wavelength=633 * nm,
                                                objective=gs.Objective(NA=0.9, magnification=60),
                                                camera=gs.cameras.CMOS(pixel=3.45 * um, shape=(512, 512))),
                   fidelity=gs.Fidelity("accurate", interaction="bpm", memory="reversible"),
                   batch_size=1, outputs={"holo": "frames", "phase": "taps.exit_phase"})
```

### (d) Switching fidelity for the same scene and inspecting the plan

```python
for fid in ["sketch", "fast", "balanced", "accurate"]:
    sim.set_fidelity(fid)                     # re-plan (cached); ParamStore and sampler untouched
    print(sim.plan.explain(short=True))
```

Example output for the holography simulator of (a):

```
Plan 7f3a1c  fidelity=fast  B=64  cuda:0   est. 9.8 ms fwd / 31 ms fwd+bwd, 1.1 GB peak
sampler  11 sites → 4 fused draws (normal×2, uniform×1, binomial×1); graph-capturable
bounds   beads.position.z ∈ [-6.0, 6.0] µm (support); beads.radius ≤ 0.61 µm (q=1e-4 ×1.5 headroom)
[1] lower     beads      Sphere → ParticleSet                 exact        grad(radius,n,xyz): smooth
[2] illum     PlaneWave  1 mode, λ=0.532, scalar, tilt[B]     coherent
[3] interact  beads      mie.pupil (N_θ=512, n_max≤12 fp64)   valid: isolated, 0.3% pairs < 2 µm (C0 warn)
[4] collect   pupil      fused: aperture(soft)·zernike[B]·defocus(H_exp)   per-image pupils [64,256,256]
[5] image     sparse     global-spectrum on 256² (oversample 2)   ROI rejected: support 9.1 µm > 32 px
[6] detect    |E|²       ref+scatter explicit, sum-pool 2×2 → CMOS, Poisson: scaled-ST
rejected: dipole.pupil (x_max=3.4 > 1.0); projection (φ_max=2.1 rad; defocused Mie rings wrong)
```

```
Plan 91be02  fidelity=sketch ...
[3] interact  beads      dipole.pupil (Mie-dressed α)        VIOLATION (x_max=3.4 > 1.0) → policy=warn
[6] detect    Gaussian noise (no Poisson)
```

```python
rep = gs.compare(sim, "fast", "accurate", n=64, seed=0)      # same traces through both plans
print(rep)    # per-output rel-L2, SSIM, centroid bias [nm], radius-feature bias, time, memory
sim.set_fidelity(gs.Fidelity("fast", interaction={"beads": "mie"}, psf="vectorial"))
```

### (e) Reusing the forward model in inverse problems

```python
# (e1) Holographic characterisation of detected particles: same Spec, same fidelity.
crops = gs.io.read_stack("exp/crops.tif", device="cuda")          # [K, 1, 96, 96]
inv = sim.as_inverse(batch_size=crops.shape[0], shape=(96, 96), count={"beads": 1},
                     latents={"beads.radius": 0.4 * um, "beads.n": 1.50,
                              "beads.position": init_xyz},        # become per-image learnables
                     fixed={"illumination.tilt": 0.0})
res = gs.fit.likelihood(inv, observed=crops, optimizer="lbfgs", steps=80,
                        multistart={"beads.position.z": torch.linspace(-6, 6, 7) * um})
r, n = res.values["beads.radius"], res.values["beads.n"]
sd = gs.fit.laplace(inv, res).std()                                # Fisher/Laplace uncertainties

# (e2) Intensity diffraction tomography: learnable voxel Δn through the same BPM stage.
dn = gs.Volume(source=gs.learnable(torch.zeros(96, 256, 256)), quantity="dn",
               extent=((0, 0, 0), (25.6 * um, 25.6 * um, 9.6 * um)))
idt = gs.Simulator(gs.Sample(volumes=[dn]),
                   gs.presets.LEDArrayIDT(wavelength=520 * nm,
                                          objective=gs.Objective(NA=0.65, magnification=40),
                                          leds=gs.LEDArray(grid=(9, 9), pitch=4 * gs.units.mm,
                                                           distance=60 * gs.units.mm)),
                   fidelity=gs.Fidelity("accurate", interaction="bpm", memory="reversible"),
                   batch_size=1, outputs={"mu": "expected"})
opt = torch.optim.Adam(idt.parameters(), lr=5e-3)
for it in range(200):
    for leds in gs.minibatch(range(81), 9):                           # angle mini-batching
        with gs.condition({"acquisition.led": leds}):
            loss = idt.nll(observed[:, leds]) + 1e-4 * gs.reg.tv3d(dn)
        opt.zero_grad(); loss.backward(); opt.step()

# (e3) Linear sub-plan as an operator for external solvers
A = gs.interop.deepinv.as_physics(idt.with_fidelity(interaction="born"))   # LinearPhysics
```

---

## 10. DeepTrack2 integration strategy

**Premise.** DT2's lasting asset is its stochastic property DAG and UX: sampling rules, dependency injection by name, `^`, `&`, Sources and deeplay consumption. Its liabilities are per-sample Python rendering, global state and caches that fight autograd [B01 (g)]. gradoscopy takes over rendering and randomness semantics at scale without breaking DT users.

**Level 1: adapter (v1, milestone M1–M2).** `gradoscopy.interop.deeptrack` provides the following.
- **DT-compatible Features with the same names and semantics**: `Fluorescence`, `Brightfield`, `Holography`, `Darkfield`, `ISCAT`, `IlluminationGradient`, `MieSphere`, `MieStratifiedSphere`, `Sphere`, `Ellipsoid`, `Ellipse`, `PointParticle`, `Zernike` (+ named terms), `Poisson`, `Gaussian`, `Background`.
  - Scatterer `get()` returns lightweight **descriptors** (kind + resolved properties) instead of rasterised volumes.
  - `Microscope.get` packs descriptors into a B = 1 Trace and renders through a gradoscopy plan. Units are converted from metres and pixels to µm, and `(1,C,H,W)` to `(H,W,C)`.
  - `nn.Parameter` properties keep identity, so DT's torch-fitting tutorials keep working, now with non-zero geometry gradients.
- **`BatchedDataset(pipeline, batch_size)`**, drop-in for `dt.pytorch.Dataset`. It resolves the DT property graph B times (Python, per sample), stacks descriptors into one padded Trace, and renders once on the GPU.
  - Measured proxy: per-sample Python resolution of 40 scalar sites costs about **0.3 ms per image**, so this path tops out at about 3×10³ images/s.
  - That is still orders of magnitude above per-sample rendering, but below the native path. The adapter says so in its docs.
- **`sim.as_feature(outputs=...)`** makes any gradoscopy Simulator a DT Feature (`__distributed__=False`), serving per-sample slices from an internally batched render. It is usable with `>>`, `&` and deeplay `fit`.
- **`Fidelity("dt2")`** is a legacy preset used only for **golden parity tests**: NA-truncated inter-slice propagation, additive overlaps, bilinear placement, and the DT2 pupil normalisation. The tests reproduce DTGS121, DTEx203, DTEx205, DTEx252 and DTGS106 to tolerance. Default tiers fix DT2's shortcuts (§5.2).

**Level 2: compiler bridge (v1.x, milestone M5).**
- `gs.from_deeptrack(pipeline)` walks the Feature graph and maps known Features to Spec nodes.
- Properties are mapped as follows:
  - constants and tensors → Fixed;
  - `nn.Parameter` → Learnable (shared);
  - **introspectable sampling rules** → Random sites;
  - opaque lambdas → `External` sites, evaluated per sample, non-differentiable, and reported by `sim.gradient_report()`.
- The introspectable rules come from a small proposed addition to DT2: `dt.dist.Uniform/Normal/LogNormal/...` objects usable wherever a property rule is. They behave like today's lambdas inside DT, and are recognised by the bridge.
- Name-based dependencies with `dt.dist` arguments become `Derived`/Random with Param arguments.
- `^ N` becomes a Population with N slots. `&` label branches reading upstream properties become trace-site outputs.
- Result: existing DT pipelines become native plans (10⁴–10⁵ images/s), with learnable distributions wherever `dt.dist` is used.

**Level 3: DT3 front-end (post-v1, in collaboration with DT maintainers).**
- DT's Feature/Property API becomes a front-end that emits gradoscopy Specs directly.
- DT keeps Sources, augmentations, `Dataset`/deeplay integration and tutorials.
- DT's `optical/` module is deprecated in favour of the engine.
- Caching semantics disappear: the engine is pure, and "update" means drawing a new trace.

**Conventions kept vs broken** [B01, "API conventions"]:
- Kept: class names, `position_unit`, and contrast semantics (`intensity`, `refractive_index`, `value` with a warning).
- Changed:
  - `upscale` maps to a fidelity override (`oversample`).
  - `upsample` becomes a filter setting.
  - Output becomes `(B,C,H,W)` natively, with `(H,W,C)` only in the adapter.
  - Global seeding becomes explicit generators.

---

## 11. Testing and validation strategy

The testing registry doubles as the fidelity dispatch table [B07 §8]. Every stage registers its references, gradcheck shapes, ladder entries and benchmark shapes.

1. **Analytic references (fp64):**
   - Airy pattern and in-focus OTF; Gaussian-beam propagation; Parseval for unitary propagators.
   - Fourier shift vs integer roll; Zernike orthonormality.
   - Sphere form factor vs supersampled voxelisation (0.12 % [B06]).
   - Blurred-ball closed form vs iFFT (3.6×10⁻⁸ [B06]).
   - TIRF depth; iSCAT contrast period λ/(2n).
   - Camera mean and variance (EMCCD ≈ 2G²Φ).
2. **External oracles (test-only extras, never vendored if GPL):**
   - Mie S1/S2 and Q values vs miepython/PyMieDiff (< 10⁻⁶ relative, fp64).
   - Holograms vs holopy `MieLens` (< 10⁻³).
   - Vectorial PSF vs psf-generator (MIT) and psfmodels (GPL, test-only).
   - Propagators vs hcipy/prysm; BPM vs chromatix multislice (with the conj convention flip [B07 §2.1]).
   - Camera statistics vs microsim.
3. **Gradient tests:**
   - `gradcheck` in complex128 for every functional op and custom Function: full mode on tiny shapes, `fast_mode` in bulk [B07 §8].
   - Adjoint-vs-autograd equality ≤ 10⁻⁵ in fp32 (reversible multislice measured 7×10⁻⁶).
   - **Auto-generated non-zero-gradient assertions for every Learnable in every preset × fidelity** (the `gradient_report` probe in CI).
4. **Estimator tests (statistical), modelled on Exp B:**
   - For each registered estimator, bias and variance of dE[L]/dθ vs analytic values on mean-sensitive and variance-sensitive losses, with tolerance of 3σ.
   - Plain straight-through must be *flagged* on variance-sensitive losses.
   - Soft Concrete at large τ must be flagged as biased.
   - Sampler KS tests; Poisson mean = variance; seeded reproducibility; counter-based RNG independent of batch composition.
5. **Cross-fidelity ladder tests** within declared validity regimes:
   - scalar → vectorial at low NA;
   - thin screen → BPM for thin objects;
   - Born → Mie for small Δn·x;
   - dipole → Mie at x < 0.3;
   - BPM → (v2) MBS;
   - sparse ROI → global spectrum (checks ROI sizing; Exp A found 5.7 % with too small an ROI);
   - stochastic Abbe → exact Abbe (convergence ∝ 1/√K);
   - strata → sparse.

   Outputs are error-vs-regime tables committed to the repository that feed planner thresholds.
6. **Learnability end-to-end tests** (nightly):
   - Recover a LogNormal size distribution from unlabeled images (Exp C pass criteria: median ±3 %, σ ±15 %).
   - Recover a Zernike pupil from simulated bead stacks (RMS < λ/50).
   - Recover emitter density and the photon distribution.
   - Recover defocus, NA and condenser NA from phase-contrast images.
7. **DT2 parity** golden images (the `dt2` preset) plus behavioural parity of adapter outputs (shapes, labels, units).
8. **Reproducibility tests:** `replay(trace)` is bit-exact; `replay(mode="noise")` under unchanged θ is bit-exact; plan hashes are stable across runs.
9. **Performance and memory benchmarks** on a self-hosted 3090 runner:
   - Exp A/D-style suites per tier: images/s, fwd+bwd ms, peak GB.
   - Regressions > 20 % fail.
   - Also run on Linux+Triton to track compile gains.

---

## 12. Roadmap

Milestones are ordered by **risk × value**. The highest-novelty risk (the probabilistic core and its ergonomics and speed) goes first, together with the cheapest high-value physics (sparse emitters). Durations assume 2–3 FTE.

| Milestone | Deliverables | Exit criteria |
|---|---|---|
| **M0: Probabilistic core + sparse fluorescence** (wk 0–7) | `core` (units, grids, types, registries, RNG); `prob` (Param kinds, distributions, plates, estimators, Trace, handlers, ParamStore, SamplerPlan with fusion, bound guards); scalar pupil (soft aperture, H_exp defocus, Zernike, pixel pupil); sparse ROI + global-spectrum emitter renderers; planner v0 (ROI from bounds); Poisson–Gaussian camera with expected/sample/log_prob; labels (sites, heatmap, emitter table); `sim.stream` with CUDA graphs | ≥ 2×10⁴ images/s fwd at 128²/50 emitters; resolve ≤ 0.3 ms at 40 sites; estimator test suite passes (Exp B criteria); gradient_report finds zero silent-zero learnables in presets; replay bit-exact; photon/density distribution learning test passes |
| **M1: Calibration loop + DT adapter v1** (wk 7–13) | `fit.likelihood`, `fit.crlb`, `fit.laplace`; sCMOS maps and EMCCD with log_prob; dense strata fluorescence; Gibson–Lanni LayerStack; DT adapter (Fluorescence, PointParticle, Sphere[fluor], Zernike, Poisson; BatchedDataset; as_feature) | Simulated bead-stack pupil recovery RMS < λ/50 in < 60 s (24 beads × 41 z); DTEx252 reproduced natively ≥ 5× faster than DT2 with identical API in the adapter; DTGS126 golden parity |
| **M2: Coherent sparse path** (wk 13–21) | fp64 Mie (homogeneous + layered, stable recurrences), dipole scatterers, Mie-in-pupil via θ-tables; plane waves, reference beams; in-line/off-axis holography, darkfield (stop / annulus), iSCAT-lite (explicit interference term); stochastic and exact Abbe; C0 composition; `gs.compare`; `fit.distribution` | Mie vs miepython < 10⁻⁶; holopy MieLens < 10⁻³; DTGS121/DTEx203/DTEx205 parity; size-distribution recovery from holograms (median ±3 %, σ ±15 %); ≥ 3×10³ images/s for holography fast tier |
| **M3: Dense coherent path** (wk 21–31) | Primitives, Groups, materials, GRF textures, SDF, Volume (voxel/neural); band-limited lowering with custom-autograd splat; projection, form-factor and slice-wise Born/Rytov; tilt-BPM with reversible/segmented adjoint + procedural slices; Köhler presets: brightfield, phase contrast, DIC (scalar), DPC, QPI taps; instance/OPD labels | BPM vs Mie for a 1 µm bead within ladder tolerance; phase-contrast halo present under partial coherence and absent when coherent; 64×256²×128-slice fwd+bwd < 8 GB; SyMBac-like automatic matching demo converges |
| **M4: Accuracy tier + planner maturity** (wk 31–39) | Vectorial RW + 6-basis dipoles + Fresnel; spectral bins; per-object field-dependent Zernike maps; validity-as-probability; ladder-calibrated thresholds; fidelity schedules and randomised fidelity; cross-fidelity surrogate gradients | Vectorial vs psf-generator < 10⁻³; ladder tables committed; `explain()` covers every preset; surrogate-gradient calibration demo |
| **M5: Dynamics + DT bridge + 1.0** (wk 39–47) | Frame plate, Brownian/OU/drift/rotational diffusion, blinking (expected/stochastic), exposure sub-steps; SIM/TIRF/confocal/light-sheet (analytic) presets; `from_deeptrack` + `dt.dist` proposal; docs and tutorials mirroring DT's | DT tutorial suite runs through the bridge; tracking sequence generation ≥ 10³ sequences/s (10 frames, 128²); API freeze; 1.0 release |
| **v2 track** | SSNP/MLB, C1 injection, stratified dipoles and R-port S-matrix, Jones/ε-tensor, SOCS/WOTF, low-rank space-variant PSFs, Foldy–Lax, T-matrix import, meshes, nonlinear excitation | per-item ladder entries against reference tiers |

---

## 13. Key decisions

### 13.1 Units and precision
- **Decision: micrometres internally, as plain floats. Wavelengths in µm, spatial frequency in cycles/µm, time in seconds. float32/complex64 default. fp64 for special functions, phase-argument construction and gradcheck. complex128 as a validation mode. No complex32, fp16 or bf16.**
- **Rationale (angle-specific):** learnable parameters must be O(1) for optimisers. In metres, a radius of 3×10⁻⁷ trained with Adam at lr = 10⁻³ moves by ~3000× its value per step, and Adam's ε (10⁻⁸) is larger than typical gradients' natural scale. Users would need per-parameter scaling everywhere. µm puts positions, radii, wavelengths, z-ranges and OPD in the 10⁻²–10² range, and float32 relative precision is unaffected.
- The unit is still SI-derived and documented. Metres and pixels are converted only at boundaries (the DT adapter, I/O). Positive quantities with wide dynamic range (photons, rates) are log-parameterised in the ParamStore.
- Precision choices follow [B07 §2.4, §4.3]: complex64 is accurate to ~10⁻⁶ over 100 slices when carrier-subtracted, while complex32 gave NaN.

### 13.2 Central interchange representation
- **Decision: two ports of one concept, the angular spectrum at the objective's object-side focal region:**
  1. **sparse** `ObjectSpectra`: per-object, pupil-plane, with local ROI or global k-grid support;
  2. **dense** `AngularSpectrum` on the global k-grid, with `port ∈ {T, R}`. The exit-plane field is its inverse-FFT view.
- **Plus** an incoherent `EmitterSet` channel consumed directly by PSF models, using the shift-invariance collapse.
- **Rationale:**
  - The pupil is the lingua franca [B03 §3.2, B08 §7.1].
  - The sparse port is the performance and learnability sweet spot: exact position gradients from phase ramps; Mie via 1-D tables; per-object aberrations free; 3.8–5.7× faster forward in Exp A.
  - The dense port serves volumes and multislice.
  - Tagging ports now (T/R) avoids an expensive retrofit for iSCAT and TIRF [B04 §6.10].
  - A single exit-plane-only currency would force defocused Mie particles through full-frame fields (10× slower in Exp A). A pupil-only currency would force volumes to be Fourier-transformed at every junction.

### 13.3 Planner sophistication and when planning happens
- **Decision: a rule-based planner with capability filtering, validity over *distribution bounds* (reported as probabilities from a validation draw), cost ranking under a memory budget, and a small rewrite pass. It is not a search-based optimiser (shortest path only over the tiny conversion graph).**
- It runs at Simulator construction (lazily) and on explicit triggers, including **bound-guard trips**, which is the angle-specific need: learnable distributions drift.
- Plans are cached by static signature, and everything is inspectable through `explain()`.
- **Rationale:** debuggability and predictable static shapes (CUDA graphs) matter more than optimality [B09 §3.2]. Planning over bounds, not values, is what keeps shapes static while θ changes [B05 §8.4].

### 13.4 Parameter/trace system; tensordict or pyro?
- **Decision: roll our own `gradoscopy.prob`, estimated at ~1.5–2.5 kLOC. Wrap `torch.distributions` for densities and rsample. Do not depend on Pyro or TensorDict in core. Offer `trace.to_tensordict()` and a Pyro-trace export as optional interop.**
- **Rationale:**
  - Pyro's global handler stack and param store are anti-patterns for an embedded library [B09 §1.10]. We also need semantics Pyro lacks: plates with object-slot presence, bound guards, per-site estimator policy, CUDA-graph-compilable fused draws, stored base noise for common-random-number replay.
  - TensorDict requires a common batch prefix, while our sites mix `[1]`, `[B]`, `[B,N]` and `[B,T,N]`. It is a heavy dependency for a container we can write in a few hundred lines.
  - The prob layer *is* the differentiator [B09 §4.7.2], so owning it is justified.

### 13.5 nn.Module vs functional core
- **Decision: a functional physics core (pure functions and custom autograd Functions on tensors); frozen-dataclass Specs; and exactly one `nn.Module` shell per simulator (`Simulator`, owning `ParamStore` and plan buffers).**
- User `nn.Module`s enter as `ModuleParam` leaves: neural fields, residual CNNs, background generators.
- **Rationale:**
  - Keeps `.to()`, `state_dict`, `parameters()`, DDP and optimisers idiomatic.
  - Avoids the "every scene object is a mutable Module" aliasing trap [B09 §3.13].
  - Kernels stay gradcheck-able and graph-capturable.
  - Sharing a `Learnable` object across Specs shares the parameter, which covers the "fit on beads, then generate" workflow of (b).

### 13.6 Sparse vs dense render paths
- **Decision: both, chosen per population by the planner, combined at the image-plane field (coherent) or irradiance (incoherent).**
  - Sparse has two sub-modes: **ROI** (support ≤ ROI from bounds) and **global spectrum** (large defocus supports).
  - Dense covers strata-OTF fluorescence and grid-spectrum coherent solvers.
- **Rationale:** Exp A shows sparse ROI at 51 k images/s vs 9 k for dense strata at 128²/N=50. It also shows that ROI sizing is a correctness issue (5.7 % error), which is exactly what a bounds-aware planner is for. Real scenes mix both: labelled cells plus single molecules or beads [B03 §3.6.4].

### 13.7 Polarisation representation
- **Decision: a polarisation axis P ∈ {1, 2, 3} is *always present* in field types. The scalar fast path is P = 1. The vectorial path uses the same kernels with P = 2/3 and Richards–Wolf maps. Unpolarised light is two modes. Dipoles are the 6-basis second-moment form.**
- The planner chooses scalar for "fast", unless polarisation optics or fixed dipoles are present or NA > 1.2 in "accurate".
- **Rationale:** retrofitting a polarisation axis later is expensive [B08 §2.13]. The vectorial overhead is small on GPU [B05 §3.1]. Vectorial PSFs are needed for SMLM system ID at high NA (40 nm bias from fixed dipoles with Gaussian or scalar models [B03 §2.2]).

### 13.8 Multi-wavelength grid handling
- **Decision: one common object-space grid for all wavelengths. The pupil lives on the shared absolute-frequency k-grid with a per-λ radius (soft edge). Camera resampling uses MFT/CZT for non-integer ratios. Λ is a summed mode axis with chunked accumulation.**
- Per-image λ (`plate="image"`) gives batched pupils. Spectral bins are a fidelity axis.
- **Rationale:** avoids Chromatix's per-λ grid leak [B02 §2.1], makes the incoherent λ-sum a weighted reduction, and supports learnable or randomised wavelengths without shape changes.

### 13.9 Time/dynamics in v1
- **Decision: yes, minimally.**
  - A `frame` plate (T axis) with Brownian/OU/drift/rotational diffusion processes (reparameterised noise, learnable D and v).
  - Two-state blinking in `expected` (differentiable) or `stochastic` (score) mode.
  - Exposure sub-steps as a summed mode axis (motion blur).
  - No biological mechanics.
- **Rationale:** tracking (DT's core domain, MAGIK, sequences) and SMLM multi-frame context need it [B08 §2.1, §2.4]. It is cheap once T is just another plate. Doing it later would change every label renderer's shape contract.

### 13.10 Dependency policy
- **Decision: core = `torch` (≥ 2.8, tested on latest) + `numpy`. MIT licence, matching DT2.**
- Optional extras:
  - `[fast]`: `triton` / `triton-windows`;
  - `[nufft]`: `pytorch-finufft`;
  - `[io]`: zarr, h5py, tifffile;
  - `[deeptrack]`, `[deepinv]`, `[vision]` (torchvision tv_tensors);
  - `[test]`: scipy, miepython, hcipy, psf-generator; GPL oracles such as PyMieDiff, holopy and psfmodels are installed only in the test environment.
- No pint, pydantic, pyro or tensordict in core. No C++/CUDA extension or Cython in v1 [B07 §3.5]. Every optional accelerator has a pure-torch reference.
- **Rationale:** DT2 users are on all three OSes. Triton is missing on Windows by default [B07 §1]. GPL code cannot be vendored [B03 §0].

### 13.11 Additional angle-D decisions
- **Default estimators come from measurement:**
  - scaled straight-through Poisson;
  - straight-through Concrete τ = 0.5 for learnable presence;
  - score function + leave-one-out baseline as the unbiased fallback;
  - mean-field `expected` modes for low-variance early calibration.
- **The Trace stores base noise** (common-random-number replay, SAA fitting).
- **The compiled sampler** (fused draws) is part of the plan and graph-captured with the render.
- **Paired and multi-view rendering and interventions** are core API, not recipes.

---

## 14. Top risks and open questions

### Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| 1 | The prob layer grows into an ad-hoc PPL: scope creep, confusing semantics | M / H | Freeze a minimal site vocabulary in M0; no inference algorithms in core; design reviews with DT maintainers; export traces to Pyro/sbi for anything else |
| 2 | **Silent estimator bias** misleads calibration conclusions (relaxed counts, straight-through, surrogate gradients) | M / H | Measured defaults; estimator test suite; `gradient_report` shows the estimator per learnable; `fit` results carry an "estimator bias" note; unbiased score fallback for final estimates |
| 3 | **Identifiability**: distribution widths are weakly identified under shot noise (Exp C σ −9 %); features drive bias | H / M | Offer several feature families and learned encoders; report bootstrap intervals across seeds; SBI hooks for posterior widths; document that σ-type parameters need more data |
| 4 | Plan churn as learnable distributions drift (recompiles, CUDA-graph re-capture) | M / M | Headroom in bounds; bucketed shapes; one-step-lagged guards; plan cache; re-plan counted and surfaced |
| 5 | Throughput on Windows without Triton or Inductor complex fusion | M / M | Sparse-first design and CUDA graphs deliver targets without compile (Exp A/D measured without Triton); Triton kernels opt-in |
| 6 | Mie numerics (fp64 on GeForce, large x, absorbing m) | M / M | Tiny arrays; batch and graph-capture; CPU fallback on MPS; downward D_n and stable-direction switching; oracle tests across dielectric, absorbing and large-x cases |
| 7 | Memory in dense learnable-volume problems (ODT, neural fields) | M / H | Reversible or segmented adjoints; procedural slices; angle mini-batching; chunked modes; memory estimator refuses infeasible plans with a clear message |
| 8 | DT2 adoption friction (API differences, parity gaps, the adapter's ~3 k images/s ceiling) | M / H | Adapter first (M1), golden parity tests, `dt.dist` proposal co-designed with DT (the requesting user is a DT core developer), native bridge in M5 |
| 9 | Convention bugs (defocus sign, conj conventions vs JAX ports, apodisation direction) | M / H | Conventions fixed in §4.1 and tested analytically on day one; the JAX-port conj flip is tested explicitly |
| 10 | The planner becomes a black box users fight | M / M | Every choice is pinnable; `explain()` lists rejected alternatives; `plan.run(until=)`; no silent fallbacks |

### Open questions

1. **Poisson–Gaussian likelihood for sCMOS.** Exact truncated convolution, shifted Poisson, or Gaussian with variance μ + σ²? Accuracy at low counts vs speed needs benchmarking against real calibration data.
2. **Default features for distribution matching.** Handcrafted (quantiles plus radial power spectrum, as in Exp C) vs self-supervised encoders trained on real data. Should gradoscopy ship a small pretrained microscopy encoder?
3. **Score-function estimators with batch-level losses** (MMD). Is the K-sub-batch baseline good enough, or should relaxations be mandatory for counts in distribution matching?
4. **Validity as probability.** Should the planner *route* objects per instance (for example, dipole for small spheres and Mie for large ones), paying for both kernels under masks, or route whole populations by bounds only (the v1 proposal)?
5. **Counter-based RNG cost** at very large site counts, and whether to make it the default for dataset materialisation.
6. **Continuous learnable magnification or pixel size.** MFT pixel evaluation vs sub-pixel resampling for the camera stage: which is cheap enough to be the default when these parameters are learnable?
7. **Coupling defaults.** At what inter-particle distance or density should the planner escalate from C0 to C1/C2 (v2) at "accurate"? This needs ladder data.
8. **Spec serialisation of `Derived` lambdas and `External` rules** from DT. Is "importable function or non-serialisable warning" acceptable to DT users who rely on inline lambdas?
9. **Trace container interop.** Should `Batch`/`Trace` implement the TensorDict protocol for Lightning/TorchRL users without depending on it?
10. **Scope of `fit`.** How much bilevel and task-driven optimisation (AutoSimulate-like) belongs in core versus examples?
