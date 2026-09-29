# Proposal A: a physics-first optical-field operator core for gradoscopy

**Angle.** The design is centred on typed optical quantities that carry their own sampling metadata. A small, closed set of physical operator categories acts on them. A modality is a composition of operators. Fidelity is the choice of implementation behind each operator. Everything is chosen for physical correctness, for generality across coherent, partially coherent and fluorescence imaging, and for inspectability.

**Thesis in one paragraph.** A light microscope can be described with five continuous physical descriptions (light, environment, sample, objective, detector) and three sampled carriers (a coherent `Field` made of mutually incoherent modes, an `Intensity`, and an `EmitterSet` of incoherent point sources). Six operator categories (Source, Interact, Propagate, Filter, Transduce, Detect) and four combinators connect them. **Every linear stage in the instrument is diagonal in either the spatial basis or the angular (pupil) basis. The only exceptions are the sample, where light meets matter, and the square-law detector.** gradoscopy should therefore be built as an algebra of basis-typed operators on self-describing fields:
- the planner inserts basis changes (FFT, MFT, CZT, analytic evaluation) and fuses adjacent diagonal operators;
- the classic fast paths (OTF convolution, Hopkins/SOCS, WOTF, per-emitter ROIs) are *rewrites justified by theorems*, not separate models.

The sample is a continuous, resolution-free description that each Interact implementation lowers to the view it needs. The unscattered (zero-order) light is carried **analytically**, never sampled.

**New measurements taken for this proposal.** RTX 3090, torch 2.14+cu130; scripts in `scratchpad/archA/`.

| Measurement | Result |
|---|---|
| Sparse emitters, per-emitter pupil → 32² ROI by batched matrix DFT, B=64 frames × 50 emitters, 64² pupil, scalar | **2.37 ms fwd (27k frames/s at 256²)**, 7.2 ms fwd+bwd w.r.t. all xyz, 0.62 GiB |
| Same, 6-component vectorial dipole basis | 5.9 ms fwd, 18 ms fwd+bwd, 1.4 GiB |
| Same, B=256 × 20 emitters | 3.8 ms fwd (**67k frames/s**) |
| Dense alternative, per-emitter full-frame 256² FFT (lower bound) | ≈2.5k frames/s, ~10× slower |
| complex64 matrix DFT under TF32 vs IEEE fp32 | **4.4e-4 vs 2.9e-7** relative error. TF32 also degrades *complex* matmul, so IEEE must be pinned |
| 1-D Mie-like S(θ) (L=60 orders, 1024 θ) synthesised onto a 64² pupil | 6.9 ms for 640 particles, 23.9 ms for 6400. Launch-bound, so batch every particle into one call |
| Off-grid tilted plane wave, *sampled* on a periodic grid and imaged through a soft pupil | image error **8–10 % rms (up to 90 % max)** from seam ringing; snapping to the grid instead costs 6–8 mrad of illumination angle (25.6 µm FOV) |
| Weak-scatterer interference (iSCAT-like, reference r=0.067), complex64 | sampled total field: **8.7 % / 1.4 % / 0.19 %** error at peak contrast 5e-6 / 5e-5 / 5e-4; analytic background with explicit cross term: **≈2e-7** at all three |

The last two rows drive one of this proposal's central decisions (§4.3).

---

## 1. Guiding principles

1. **The microscope is diagonal almost everywhere, so operators are typed by the basis they are diagonal in.**
   - *Diagonal in angle:* free-space and layered-medium propagation, defocus, the pupil, apodization, aberrations, Fresnel interfaces, engineered masks and Jones pupil optics.
   - *Diagonal in space:* thin screens, image-plane masks, square-law detection and excitation→emission maps.
   - *Neither:* only the thick sample and the incoherent reduction.

   The planner inserts basis changes and fuses consecutive diagonals, much like MONAI's lazy resampling. This is the organising idea of the core.
2. **Physics is continuous; discretisation is a plan decision.**
   - Pupils are functions P(u, λ, h) of direction, wavelength and field position. Sources are angular distributions S(u, λ, pol). Samples are fields of material over ℝ³.
   - Only `Field`/`Intensity` tensors are sampled. **Simulation grids are static plan objects.**
   - Physical sampling *locations* are continuous, differentiable quantities evaluated with continuous-coordinate transforms (MFT/CZT, phase ramps, analytic evaluation). These include camera pixel centres, magnification, emitter positions and source directions.
3. **Light is a set of mutually incoherent coherent modes, and coherence is bookkeeping.** Wolf's coherent-mode representation is exact for every linear stage. Source points, wavelengths, polarisation states, emitters, dipole orientations, exposure sub-samples and speckle realisations are all *the same* weighted reduction Σ_m w_m‖·‖², taken at one place (Detect). There is no separate "partially coherent" or "incoherent" code path.
4. **Keep the zero order analytic.** The unscattered wave (illumination, reference beam, coverslip reflection) is carried as an analytic set of plane waves beside the sampled scattered field. This gives:
   - exact pupil filtering of the zero order (darkfield, phase rings, iSCAT attenuation);
   - seam-free off-grid illumination tilts;
   - float32-safe interference at 10⁻⁶ contrast (measured above).
5. **A closed operator vocabulary; modalities are recipes; fidelity is the choice of implementation.**
   - Adding a modality should mean writing a ≤15-line recipe. If it needs code outside an operator implementation, the vocabulary is wrong.
   - Adding an implementation should upgrade every modality that uses its category.
6. **Fast paths are theorems, not models.** Shift-invariance collapse, thin-sample collapse and weak-object linearisation are planner rewrites guarded by declared properties. Every rewrite has a CI test against its uncollapsed graph.
7. **Every approximation is declared, checked and reported.**
   - Implementations declare validity predicates. These are evaluated at plan time against *bounds* derived from parameter distributions, and optionally at run time on actual values.
   - `plan.explain()` lists the choices made, the alternatives rejected and why.
   - Nothing falls back silently.
8. **Batch-first, static shapes, pure functions, explicit randomness.**
   - One render call per batch.
   - Shapes derive from declared bounds, never from parameter values.
   - No caches that detach gradients.
   - A `torch.Generator` is passed in explicitly; global RNGs are never touched.
9. **Operators are deterministic; parameters carry randomness and learnability.** An operator never samples; a parameter never computes physics. Detector noise is the only stochastic operator, and it draws from the call's generator.
10. **One forward model serves data generation, fitting and reconstruction.** Each model offers `sample()` for training data and `expected()` + `log_prob()` for maximum-likelihood fitting. Linear sub-plans expose an adjoint for reconstruction and deepinv export.

---

## 2. Scope

The scope is chosen so that v1 covers the *minimum* fidelity of about 16 of the 18 application families in brief 08, plus full DeepTrack2 parity, while needing only four solver engines: pupil optics, sparse analytic scatterers and emitters, the z-march, and the detector.

### v1: ships first (DT2 parity plus the application core)

| Capability | Why (applications) |
|---|---|
| Fluorescence from **sparse emitters** (per-emitter pupil → ROI) and **dense densities** (per-z OTF strata); scalar and vectorial dipole PSFs; Zernike (OPD) and pixel pupils; Gibson–Lanni layer stack | SMLM, PSF engineering (DeepSTORM3D), bead calibration (uiPSF/FD-DeepLoc), tracking, cell fluorescence |
| **AngularSource × pupil** transmitted-light family: brightfield (Köhler, partially coherent), darkfield, Zernike phase contrast, DIC (scalar shear form), DPC and oblique, FPM (LED arrays) | cell segmentation (SyMBac-style), QPI, FPM design, virtual-staining pretraining |
| **Interferometric**: in-line and off-axis holography; iSCAT with a *physical* reflected reference (coverslip Fresnel) and dipole/Mie backscatter | holographic characterisation (CATCH, DTEx203/205), mass photometry |
| **Interaction tiers**: thin/projection (analytic projections), Born/Rytov (analytic form factors), multislice BPM (free-space inter-slice propagator, procedural slices, adjoint), Rayleigh dipole, Mie and layered Mie; objects superposed coherently (C0) | brief 04 ladder, lower two thirds; DT parity plus fixes |
| **Two-stage fluorescence** through Transduce: TIRF, linear SIM (analytic plane-wave excitation), analytic light sheet, confocal/ISM/2P via effective-PSF rewrites | ML-SIM, light-sheet deconvolution training, confocal data |
| **Detector**: CCD/EMCCD/sCMOS (per-pixel maps)/qCMOS, each with `expected`, `sample` (documented estimator) and `log_prob`; pixel integration by sum-pool or MTF+MFT | system identification, sim-to-real, all DL training |
| **Params**: Fixed/Learnable/Random/Derived/Choice; pathwise, relaxed, straight-through and score estimators; relaxed counts; Trace, labels and replay | learnable data-generation distributions (the key differentiator) |
| **Time as frames**: Brownian/drift/OU kinematics, 2-state blinking (expected or stochastic), exposure motion blur as an incoherent sum over time sub-samples | tracking sequences, SMLM frames |
| **DT2 adapter** plus a `legacy_dt` fidelity preset (golden parity) | migration path for the user base |

### v2: planned (the contracts are already in place in v1)

| Capability | Why |
|---|---|
| SSNP and multi-layer Born as march modes; C1 injection of sparse objects into the march | beads inside cells, thick label-free cells (ODT) |
| SOCS/TCC and WOTF fast paths (the planner rules exist in v1; kernels in v1.x) | fast partially coherent training data |
| Dipoles near interfaces (SAF, full vectorial iPSF), evanescent-in-water/propagating-in-glass collection | quantitative iSCAT, TIRF SMLM |
| Modified Born series / Lippmann–Schwinger with implicit adjoint (reference tier) | validation, strongly scattering samples |
| Polarised multislice, tensor ε, Jones DIC | PolScope/PTI, birefringent samples |
| Field-dependent aberrations on the *dense* path (product convolution); multisphere Foldy–Lax | large-FOV fluorescence, colloid clusters |
| Mesh import (SDF/winding number); photophysics nonlinearities (saturation, 2P rate, STED effective PSF) | imported cell shapes, super-resolution |
| deepinv export; BPM light-sheet-in-tissue (biobeam-class) | reconstruction, tissue imaging |

### Later / plugins

T-matrix import (SMUTHI/treams tables) for non-spherical particles; TorchGDM effective-dipole interop; learned-surrogate tier trained from reference solvers; ray-traced eikonal screens; MCX backgrounds; nvmath cuFFT callbacks. Each fills a niche (rods, plasmonics, organoids, deep tissue, raw speed) through an *existing* category contract.

### Explicitly out of scope

| Item | Justification |
|---|---|
| FDTD/FEM/DDA inside the core | No surveyed data-generation workflow needs them online. They enter as imported far-field or T-matrix tables through the Interact contract |
| Coherent nonlinear optics (SHG/THG/CARS/SRS), Raman | Emitters are phase-locked to the excitation, which breaks the mode-sum model. No surveyed DL pipeline needs them |
| FLIM/TCSPC, antibunching | Detector time-domain statistics; only a lifetime *label* is in scope |
| Objective lens design / ray tracing | Every surveyed work models the objective as an aplanatic system plus a pupil. External tools can supply OPD or Jones pupil maps |
| Monte-Carlo deep-tissue transport | Replaced by a phenomenological turbid-slab operator |
| Cell mechanics and growth, instrument control, EM/X-ray | Scene-generator plugins (for example Pymunk à la SyMBac) or out of domain |

---

## 3. Layered architecture and responsibilities

### 3.1 Layers

```
 L6  interop / data / learn   deeptrack adapter · deepinv export · SimDataset · MMD/spectra losses · CRLB
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L5  system                   OpticalGraph (recipes of operator nodes) · modalities · Simulator(nn.Module)
                              labels · taps · compare()
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L4  plan                     Fidelity policy · Bounds analysis · Planner (rules, rewrites, sampling,
                              chunking) · Plan (stages, explain, hash) · validity reports
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L3  operators                Operator protocol + Capabilities; implementations per category:
                              source/ interact/ propagate/ filter/ transduce/ detect/ fused/
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L2  physics (descriptions)   Spectrum · AngularSource · Polarization · LayeredMedium · Materials
                              Pupil + modifiers · Objective · Scene (geometry × material × labeling)
                              sample views + lowering registry · DetectionPath · Camera
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L1  params                   Param leaves · learnable distributions · Trace/Site/plates · handlers
                              estimators · ParamStore(nn.Module) · seed derivation
 ─────────────────────────────────────────────────────────────────────────────────────────────
 L0  core + ops               Grid/FreqGrid/PupilGrid/PatchGrid · Field(PlaneWaves ⊕ sampled) · Intensity
                              EmitterSet · conventions · precision policy · registries · special functions
                              ops/: pure torch kernels (fft/mft/czt, propagators, pupil math, multislice
                              steps + adjoints, Mie, dipoles, rasterisers, detector math)
```

**Dependency rules** (enforced by an import-linter test in CI):
1. Imports only go downward.
2. `ops/` (L0) never imports `params` or `physics`. It takes tensors and core types only, so it is gradcheckable and reusable on its own.
3. `physics` descriptions (L2) are frozen dataclasses whose leaves may be `Param`s. They *describe* and never execute kernels; they can say which lowered views they support.
4. Operator implementations (L3) receive **resolved tensors** (a `Trace` slice) plus static configuration. They never see `Param`s, RNGs or globals, except `detect.sensor`, which receives an explicit generator.
5. Only the planner (L4) selects implementations. Descriptions never name a numerical method. `upscale`, `padding` and slice count do not exist on scene objects, although fidelity can pin them.
6. `interop/` is the only place that may import optional third-party packages (deeptrack, deepinv, tensordict).

### 3.2 Responsibilities at a glance

```
     WHAT (continuous)                    HOW (planned)                         RUN (tensors)
 ┌─────────────────────────┐     ┌──────────────────────────────┐     ┌───────────────────────────┐
 │ Scene, Light, Objective, │     │ Planner:                     │     │ Simulator.__call__:        │
 │ Camera, Modality recipe  │ ──▶ │  graph → implementations     │ ──▶ │  resolve Params → Trace    │
 │ Params (Fixed/Learnable/ │     │  rewrites (collapse, fusion) │     │  run Plan stages (pure)    │
 │ Random/Derived/Choice)   │     │  grids, modes, λ-bins,       │     │  detector noise (gen)      │
 │                          │     │  chunks, grad strategy       │     │  labels, taps, meta        │
 └─────────────────────────┘     │  validity + explain()        │     └───────────────────────────┘
                                  └──────────────────────────────┘
          ▲ user edits                  ▲ once per static config             ▲ once per batch
```

### 3.3 Package tree

```
src/gradoscopy/
  __init__.py            # curated public API: Simulator, Scene, Learnable, Random, Fidelity, units, ...
  units.py               # SI constants (m=1, um=1e-6, nm=1e-9), pixel helpers (boundary only)
  conventions.md         # frames, Fourier signs, Zernike indexing, radiometry: the single source of truth
  core/
    grid.py              # Grid2D, FreqGrid, PupilGrid (NA units), PatchGrid (per-mode origins), Grid3D,
                         # nice_fft_size (2·3·5·7-smooth)
    field.py             # Field, PlaneWaves, Ports, Basis(SCALAR|JONES|CARTESIAN), Domain(SPACE|ANGULAR)
    intensity.py         # Intensity, Counts
    emitters.py          # EmitterSet, SourceDensity
    axes.py              # canonical layout [B, A, M, L, P, Y, X] + AcqIndex metadata
    precision.py         # PrecisionPolicy, no_autocast(), ieee_matmul(), fp64 phase builders
    registry.py          # typed registries + entry-point loading
    special/             # bessel_j0/j1/j2 (autograd), spherical/riccati bessel (fp64, stable directions),
                         # mie coefficients (+layered), legendre pi/tau, zernike (ANSI/Noll)
  ops/                   # pure functional kernels (no Params, no RNG)
    fourier.py           # fft helpers, MFT, CZT, Fourier shift, band-limited resample
    propagation.py       # asm (H_exp form), blasm, fresnel, safe kz, plane-wave phase
    pupil.py             # soft aperture, apodization, Gibson–Lanni OPD, Fresnel ts/tp, 2<->3 Jones maps
    multislice.py        # bpm/ssnp steps; reversible + segmented adjoint autograd.Functions
    born.py              # slice-wise born/rytov, form-factor Ewald-cap evaluation
    mie.py               # S1/S2(θ) on 1-D grids, pupil synthesis, backscatter port
    dipole.py            # dipole angular spectra (6-basis), radiative-corrected polarizabilities
    raster.py            # band-limited occupancy (blurred primitives, SDF ramp), projections, patch splat
    detector.py          # pixel integration, noise samplers (scaled-ST Poisson, Gamma EM), likelihoods
  params/
    param.py             # Fixed, Learnable, Random, Derived, Choice
    dist.py              # learnable distributions (nn.Module wrappers of torch.distributions)
    trace.py             # Site, Trace (pytree), plates
    handlers.py          # condition, replay, mean, block, detach (contextvars, call-scoped)
    store.py             # ParamStore(nn.Module): unconstrained parameters, path names, sharing by identity
    rng.py               # seed derivation H(seed, step, rank, site), generator management
  physics/
    light.py             # Spectrum, Polarization, AngularSource (+ presets: PlaneWave, Koehler, LEDArray,
                         # SIMBeams, Evanescent, FocusedBeam, LightSheet, Speckle, ReferenceBeam)
    medium.py            # Medium, LayeredMedium (immersion/coverslip/sample, z-interfaces)
    materials.py         # Constant, Cauchy, Sellmeier, Tabulated, DrudeLorentz, DryMass(dn/dc)
    optics.py            # Pupil + modifiers, Objective, DetectionPath (filters, splitters, pinhole)
    cameras.py           # Camera descriptions (pixel geometry, QE(λ), noise model params)
    sample/
      scene.py           # Scene, Group, Population (SoA, N_max, presence), EmitterPopulation
      geometry.py        # Sphere, LayeredSphere, Ellipsoid, Capsule(solid of revolution), Box, Gaussian,
                         # SDF, FieldComponent (voxels / callables / neural), RandomField
      labeling.py        # Fluorophore, Labeling(volume|surface|points), Photophysics
      views.py           # lowered views: RIVolume, OccupancyChannels, ProjectedMaps, SphereList,
                         # DipoleList, EmitterDensity, KSpectrum
      lowering.py        # conversion registry with exactness tags (exact|bandlimited|approx|lossy)
  operators/
    base.py              # Operator protocol, Capabilities, Violation, Cost, Tier, GradStrategy
    source/  interact/  propagate/  filter/  transduce/  detect/  fused/
  plan/
    fidelity.py  bounds.py  planner.py  rewrites.py  sampling.py  plan.py
  system/
    graph.py             # OpticalGraph + combinators: >>, coherent_sum, acquire, split
    modalities.py        # recipes (Brightfield, Darkfield, PhaseContrast, DIC, DPC, FPM, InlineHolography,
                         # OffAxisHolography, ISCAT, Widefield, TIRF, SIM, LightSheet, Confocal, ...)
    simulator.py         # Simulator(nn.Module): __call__/sample, expected, log_prob, render(trace)
    labels.py            # Positions, Masks, InstanceMasks, DistanceMap, EmitterTable, Values, taps
  learn/                 # losses (MMD, radial spectrum, histogram-W1, mean-variance), CRLB/Fisher, fitting
  data/                  # SimDataset (IterableDataset), writers (zarr/HDF5 + JSON manifest)
  interop/
    deeptrack/           # DT-named Features, BatchedDataset, conversion of conventions, legacy preset
    deepinv.py
  testing/               # reference solutions, gradcheck helpers, convergence harness, plugin conformance
```

---

## 4. Core data model

### 4.1 Units, frames and sign conventions

- **Units.** SI metres, seconds and watts (or photons s⁻¹) throughout the core, stored as plain float tensors. There are no unit objects inside the core; `gs.units` gives constants (`um = 1e-6`) for the boundary. Pixel units exist only in the DT adapter and in label outputs.
- **Frame.** A right-handed world frame. **+z is the direction of light travelling toward the detection objective.** The `LayeredMedium` places its interfaces (immersion | coverslip | sample medium) at explicit z values. An inverted and an upright microscope then differ only in where the coverslip sits in z. Epi-illumination travels −z; transmitted illumination travels +z.
- **Fourier conventions.**
  - Time dependence is e^{−iωt}.
  - A plane wave is e^{+i2π(k⊥·r⊥ + k_z z)} with k in cycles m⁻¹ and k_z = √((n/λ)² − |k⊥|²).
  - The branch is chosen so that Im k_z ≥ 0, i.e. evanescent waves decay.
  - Directions are stored as u = λk⊥ = n sinθ (cos φ, sin φ), which is dimensionless and **λ-independent for a fixed angle**.
- **Pupil conventions.**
  - Aberrations are **optical path difference in metres**, so phase = 2π·OPD/λ and chromatic scaling is automatic.
  - Zernike polynomials use OSA/ANSI indices with RMS normalisation. Noll converters are provided, as is DT's `(n, m)`-in-radians form, which the adapter converts at DT's wavelength.
  - Apodization is √cosθ for focusing and 1/√cosθ for collection, plus the 1/k_z Jacobian of the Debye integral. Direction is a property of the operator instance, never a user switch.
- **Radiometry.** Fields are in √(W m⁻²) per mode. Mode weights and spectral weights are power weights that include quadrature weights. Emission pupils are normalised so that ∫PSF = collection efficiency; for example (1−cosθ_max)/2 ≈ 0.31 for NA 1.4 in oil. The detector converts irradiance × exposure × QE(λ) × pixel area into photoelectrons.

### 4.2 Grids and sampling bookkeeping

```python
@dataclass(frozen=True)
class Grid2D:            # a sampled plane (object-space coordinates)
    shape: tuple[int, int]                  # STATIC: part of the plan, never data-dependent
    spacing: tuple[float, float]            # STATIC for simulation grids (m)
    origin: Tensor                          # [.., 2] may be learnable/batched (sub-pixel offsets)

@dataclass(frozen=True)
class FreqGrid:          # FFT dual of a Grid2D; cycles/m; λ-independent
    of: Grid2D

@dataclass(frozen=True)
class PupilGrid:         # direction space u = λ k⊥ (NA units); λ-independent disc |u| ≤ u_max
    shape: tuple[int, int]; u_max: float

@dataclass(frozen=True)
class PatchGrid:         # many small grids sharing shape/spacing, per-mode origins (sparse ROIs)
    shape: tuple[int, int]; spacing: tuple[float, float]; origin: Tensor   # [B, A, M, 2]
```

Rules:
1. **Simulation grids are static.** Their shape and spacing come from the plan, derived from *bounds*, for example `spacing ≤ λ_min / (4·NA_max)` and padding ≥ PSF support at maximum defocus. They are rounded to 2·3·5·7-smooth sizes, because prime sizes cost up to 1.9× (brief 07).
2. **Physical sample points are continuous.** Camera pixel centres = `origin + (i·pitch/M)` feed a MFT/CZT under IEEE fp32. So pitch, magnification and stage offset are differentiable, and a non-integer supersampling ratio needs no interpolation.
3. **All wavelengths share one spatial grid at the sample and detector planes.** This is Chromatix's per-λ dx problem, fixed.
   - On the dense path, the FFT frequency grid is λ-common, and each λ's pupil is a disc of radius NA/λ on it.
   - On the sparse path, pupils live on a λ-common `PupilGrid` in u, and a per-λ MFT matrix maps them onto the common spatial ROI grid.
4. **Grid compatibility is checked at plan time.** Every operator declares its input and output grid relation (same, dual, resampled, patch), and the planner type-checks the chain.

### 4.3 The `Field` carrier (coherent light as a mode set)

```python
@dataclass(frozen=True, kw_only=True)        # registered as a torch pytree node
class Field:
    sampled: Tensor | None        # complex [B, A, M, L, P, Y, X]: the SCATTERED / non-analytic part (None ≡ 0)
    background: PlaneWaves | None # analytic part: unscattered illumination, references, reflections
    grid: Grid2D | FreqGrid | PupilGrid | PatchGrid
    domain: Domain                # SPACE | ANGULAR
    z: Tensor                     # plane (world frame, m); broadcastable to [B]
    direction: Literal[+1, -1]    # travelling toward (+1) or away from (−1) the detection objective
    medium: Tensor                # n(λ) of the homogeneous medium at this plane, [L] (complex allowed)
    spectrum: Spectrum            # wavelengths [L], power weights [L]
    mode_weights: Tensor | None   # [B?, A?, M] power weights (None = uniform 1)
    basis: Basis                  # SCALAR (P=1) | JONES_XY (P=2) | CARTESIAN (P=3)

@dataclass(frozen=True)
class PlaneWaves:                 # analytic, exact under every diagonal-in-angle operator
    amplitude: Tensor             # complex [B, A, M, J, L, P]  (J mutually COHERENT components per mode)
    direction: Tensor             # [B?, A?, M, J, 2] u = n sinθ (cosφ, sinφ); |u| > n ⇒ evanescent
```

**Canonical layout `[B, A, M, L, P, Y, X]`, fixed rank 7, with size-1 axes broadcasting.** This is a deliberate choice against the "arbitrary leading axes" style.
- *B*: batch (images).
- *A*: acquisition index. It separates frames: focus steps, SIM angle × phase, LED index, time frames, or bead index when fitting. It is a flattened multi-index with `AcqIndex` metadata.
- *M*: incoherent modes summed within one exposure. These are source points, emitters, dipole basis states, unpolarised states, exposure sub-samples or speckle realisations.
- *L*: wavelength bins.
- *P*: polarisation components. **Y, X are last and contiguous** for `fft2`.

Fixed rank keeps every kernel free of ellipsis handling, makes shapes statically checkable, and costs nothing: size-1 or expanded axes are stride-0. HoloTorch's rigidity problem came from a *missing* axis. Here the acquisition multi-index is flattened into A, and the detector channels C appear only after reduction.

**Why the analytic background matters** (principle 4, measured):
- *Off-grid tilts.* An off-grid tilted plane wave sampled on a periodic grid gives 8–10 % rms image error (seam ringing through the pupil). Snapping to the grid costs 6–8 mrad. As `PlaneWaves`, the zero order is exact at any angle. Only the *scattered* part is FFT'd, and it is compactly supported, so it has no seam.
- *Weak interference.* Forming |E_b + E_s|² in complex64 gives 8.7 % error at 5·10⁻⁶ contrast. The explicit cross term 2Re(E_b*E_s) with an analytic E_b gives 2·10⁻⁷.
- *Pupil filtering.* The pupil acts on a plane wave by *evaluation*, E_b ← P(u_b)·E_b. Darkfield (P(u_b) = 0), phase rings (P(u_b) = 0.2·e^{iπ/2}), iSCAT reference attenuation and DIC bias are therefore exact and alias-free.
- *Optical theorem.* The Mie/Born forward scatter interferes with the analytic background, so extinction "shadows" come out right. This doubles as a CI test (§11).

Each operator handles both parts:
- Propagation multiplies plane-wave amplitudes by e^{i2π k_z Δz}.
- A thin screen maps (E_b, E_s) ↦ (E_b, (t−1)·E_b|_grid + t·E_s).
- BPM marches E_s with the background evaluated at each slice.
- Detect computes |E_b|² + 2Re(E_b*E_s) + |E_s|².

`Field.to_sampled()` is a planner-inserted fallback, with a precision warning, for any third-party operator that does not support analytic backgrounds.

**Ports.** `Interact` returns `Ports(plus_z: Field | None, minus_z: Field | None)`. These are angular spectra about a reference plane z_ref: the +z face of the sample bounding box for the +z port, the −z face for the −z port. The objective consumes `plus_z`. Transmission brightfield uses `plus_z` for the transmitted light. Epi-illuminated iSCAT uses `plus_z` for the *backscattered and reflected* light. The v1 −z port is populated by Born, dipole and Mie backscatter and by the analytic Fresnel reflection of the `LayeredMedium`. Multislice declares `ports={+z}` only.

### 4.4 The other carriers

```python
@dataclass(frozen=True)
class Intensity:                  # after the incoherent reduction
    data: Tensor                  # real [B, A, C, Y, X]  (C = detector channels: colour, pol-camera, multi-cam)
    grid: Grid2D
    units: Literal["W/m^2", "photons/s/pixel", "e-", "ADU"]

@dataclass(frozen=True)
class EmitterSet:                 # incoherent point sources (never voxelised by default)
    position: Tensor              # [B, A?, N, 3] world frame (A = frames, for dynamics)
    rate: Tensor                  # [B, A?, N] photons/s emitted (before collection)
    presence: Tensor              # [B, A?, N] ∈ [0, 1]; relaxed or hard
    species: Tensor               # [B, N] int → emission Spectrum table
    dipole: Tensor | None         # [B, N, 6] second moments ⟨μ_iμ_j⟩ (None = isotropic)
    id: Tensor                    # [B, N] persistent identity for tracking labels
```

`SourceDensity` is the voxel counterpart for dense labels (per species, on a `Grid3D`). The step from an `EmitterSet` to a `Field` is a **Source** operator: each emitter is an analytic angular spectrum. The step from `SourceDensity` to `Intensity` is the collapse-(a) rewrite (§5.7).

### 4.5 The light

- **`Spectrum(wavelengths[L], weights[L])`.**
  - Constructors: `Spectrum.line(λ)`, `.band(center, fwhm, bins=None)` and `.table(λ, S)`. Excitation and emission spectra are distinct objects, so the Stokes shift is explicit.
  - When `bins=None` the planner chooses the bin count by fidelity: Gauss–Hermite or midpoint quadrature, with weights that include Δλ.
  - Chunked spectral accumulation is available when L is large.
- **`Polarization`**: `linear(angle)`, `circular(±)`, `jones(e)`, `stokes(s)` or `unpolarized()`. Unpolarised light is **two incoherent modes** on M, which costs the same as hcipy's Jones-on-Stokes contraction. Partial polarisation uses weighted modes from the coherency-matrix eigendecomposition.
- **`AngularSource`: the one illumination abstraction.** Physically it is the cross-spectral density at the condenser (or objective) pupil in Wolf's coherent-mode form W = Σ_g w_g A_g ⊗ A_g*. Each *coherence group* g is a coherent angular spectrum A_g(u). Two constructions:
  1. **Discrete coherent groups**, each a list of plane waves `(amplitude, u)` that interfere.
     - Plane wave: 1 group × 1 wave.
     - SIM frame: 1 group × 2–3 waves, with frames on A.
     - TIRF: 1 wave with |u| > n_sample.
     - LED: 1 wave, with finite LED size as a small incoherent cluster.
     - Off-axis reference: a separate coherent wave.
  2. **Spatially incoherent density S(u, λ)**: Köhler disc or annulus, Rheinberg, arbitrary masks. The *planner* samples it into groups with a strategy chosen by fidelity: exact grid (Abbe reference), quadrature(K) (rings, hexagons or a Fermat spiral with weights), stochastic(K) (fresh directions per batch, unbiased), or SOCS(K) via a rewrite for thin samples with fixed optics.
  3. **Coherent continuous apertures** (focused beams, light-sheet cylindrical pupils, speckle as a random-phase fill) produce a *sampled* angular `Field` rather than `PlaneWaves`.

  Presets are thin constructors: `PlaneWave`, `Koehler(na, annulus=None)`, `LEDArray(positions, height, size)`, `SIMBeams(period, angles, phases, na)`, `Evanescent(angle, n_glass)`, `FocusedBeam(na)`, `LightSheet(na, waist, axis)`, `Speckle(na, realisations)` and `ReferenceBeam(tilt, amplitude)`.

  Illumination-side aberrations and condenser Jones optics are pupil modifiers applied *on the source*. Brightfield, darkfield, phase contrast, oblique, DPC, Hoffman, Rheinberg and FPM are therefore **(AngularSource, Pupil) pairs**, not classes with separate code.

### 4.6 The sample

The sample is continuous and physical: geometry × material × labeling, organised as a hierarchy that flattens into a batched struct of arrays. This follows brief 06, and I adopt its measured conclusions.

- **Scene**: `Scene(environment: LayeredMedium, objects: [...], emitters: [...])`.
  - The environment is the single source of truth for n_medium(λ), the coverslip and the immersion medium. Optics (Gibson–Lanni, TIRF, iSCAT reflection) and sample (medium index) read the same object.
- **Geometry primitives (v1).**
  - `Sphere` and `LayeredSphere` are Mie-capable.
  - `Ellipsoid`, `Capsule` (a solid of revolution with a 1-D-quadrature form factor) and `Box`.
  - `Gaussian` (anisotropic, additive semantics).
  - `SDF` (callable) as the escape hatch that also covers CSG.
  - `FieldComponent`: voxels, torch callables or neural fields, each with a bbox and transform.
  - `RandomField`: Gaussian random-field texture with a learnable spectrum.
  - `Group` with priority compositing.
- **Capabilities per primitive** are introspected by the planner: `bbox`, `occupancy(x, σ)` (band-limited and differentiable), `sdf`, `form_factor(k)`, `project(xy)`, `mie_spec()` and `dipole_limit()`.
- **Materials.** `n(λ)` models: `Constant`, `Cauchy`, `Sellmeier`, `Tabulated`, `DrudeLorentz` and `DryMass` (n = n_m + (dn/dc)·C). Uniaxial ε is reserved for v2.
- **Labeling.** `Labeling(fluorophore, target, mode=volume|surface|points, density)`, where `Fluorophore` holds the excitation and emission spectra, ε, QY, dipole model and photophysics. **One geometry drives both RI contrast and label density**, so multimodal ground truth is consistent by construction.
- **Batching.** One struct of arrays per primitive kind, `[B, N_max, …]`, plus `presence[B, N_max]`, `priority`, `material_id`, `parent` and `id`. Python loops run over *kinds*, never over objects.
- **Overlap semantics** are explicit:
  - material-like quantities use priority "over" compositing, linear in ε;
  - densities add;
  - `compose="add"` is the DT-legacy mode.
- **Lowered views** are what Interact implementations consume. Each comes from the lowering registry with an exactness tag:

| View | From | Exactness | Consumers |
|---|---|---|---|
| `SphereList` / `DipoleList` | Sphere, LayeredSphere / small primitives | exact / approx(x<0.3) | Mie, dipole |
| `KSpectrum` (form factor on requested k-points) | any primitive with `form_factor` | exact | Born/Rytov, OTF fluorescence, projection via the k_z=0 slice |
| `ProjectedMaps` (OPL, absorbance, column density) | analytic projections, or voxel sums | exact / bandlimited | thin screen, 2-D fluorescence |
| `RIVolume` (procedural slice generator or materialised) | band-limited occupancy × materials | bandlimited (σ≈0.3h) | BPM, SSNP, MBS |
| `EmitterDensity` | labeling of volumes/surfaces | bandlimited | OTF-strata fluorescence |
| `EmitterSet` | point labels, SMLM populations | exact | sparse emission |

  **Hard thresholding is never shipped without a surrogate** (DT's masks give dV/dR = 0, as measured in brief 06).

### 4.7 The minimal basis, and why it is minimal

| Kind | Members | Removing it breaks… |
|---|---|---|
| Descriptions (continuous) | Light (Spectrum, AngularSource, Polarization) · Environment (LayeredMedium) · Sample (Scene) · Objective (Pupil) · Detector (Camera, DetectionPath) | — (these are the physical inputs) |
| Carriers (sampled) | `Field` (PlaneWaves ⊕ sampled, mode axis) · `Intensity` · `EmitterSet` | without EmitterSet, fluorescence must voxelise, which costs ~10× and gives zero xyz gradients; without analytic PlaneWaves, darkfield, iSCAT and off-grid Köhler lose accuracy (measured) |
| Operator categories | **Source** · **Interact** · **Propagate** · **Filter** · **Transduce** · **Detect** | without Transduce: TIRF, SIM, light sheet, confocal, 2P, bleaching; without two-port Interact: iSCAT, RICM; without Filter: all pupil engineering |
| Combinators | `>>` (sequence) · `coherent_sum` (interference between arms) · `acquire` (stack on A) · `split` (channels C) | without coherent_sum: off-axis holography with a separate reference arm; without split: multi-camera, biplane, polarisation cameras |

What is **deliberately not** a core abstraction:
- **"Lens"**: the conjugate-plane model makes lenses into Filters plus coordinate scales.
- **"Mutual intensity"**: replaced by mode sets.
- **"PSF"/"OTF"/"TCC"**: derived *views*, i.e. cached intermediates of Source(point) → Filter → Image → Detect.
- **"Noise"**: part of Detect.
- **"Aberration"**: a pupil modifier.
- **"Modality"**: a recipe.
- **`IncoherentSum` as a combinator**: it is *the* Detect reduction, done in exactly one place, so it cannot be applied twice or in the wrong order.

**Coverage check (sufficiency)**: every family in brief 08 expressed in this basis.

| Application | Source | Interact | Filter | Transduce | Detect / combinator |
|---|---|---|---|---|---|
| Particle tracking (BF) | Köhler/PlaneWave | Mie (C0) | pupil + Zernike | — | camera |
| Particle tracking (fluo) | Widefield excitation → emitters | — | pupil | linear | camera, frames on A |
| Holographic characterisation | PlaneWave | Mie / layered Mie | pupil (MieLens-exact) | — | camera or `output="field"` |
| iSCAT / mass photometry | epi PlaneWave (−z) | LayeredMedium reflection + dipole/Mie backscatter (+z port) | pupil (+ reference attenuation) | — | camera; explicit cross term |
| SMLM / PSF engineering | emitters | — | pupil (Zernike, pixel, mask), vectorial | photophysics | sCMOS/EMCCD, frames on A |
| Cell segmentation (PC/DIC/BF) | Köhler annulus/disc | projection → Rytov → BPM | phase ring / shear | — | camera |
| QPI / ODT | PlaneWave / LEDArray | Born/Rytov/BPM (SSNP v2) | pupil | — | field output or intensity, LEDs on A |
| Light sheet | LightSheet excitation | (turbid slab) | pupil | linear | camera, z-planes on A |
| Virtual staining pretraining | Köhler + widefield | BPM/Rytov | pupil | linear | `split` into channels |
| FPM | LEDArray | projection/BPM | pupil (+ calibration) | — | LEDs on A |
| SIM | SIMBeams (coherent group) | — | pupil | linear | angle × phase on A |
| Confocal / ISM / 2P | FocusedBeam | — | pupil (+ pinhole in image plane) | linear / I² | effective-PSF rewrite, or scan on A |
| Darkfield NPs | Köhler annulus (NA_c > NA) | dipole (Drude) / Mie | pupil | — | camera, RGB via `split` |
| Polarisation | PlaneWave (+ Jones pupil) | thin Jones (v1), tensor multislice (v2) | Jones pupil | — | analyser states on A or C |
| End-to-end design, system ID, sim-to-real, recon | any | any | learnable pupil/masks | any | `log_prob`, `expected`, adjoint |

---

## 5. Physics and the fidelity mechanism

### 5.1 The operator contract

```python
class Operator(Protocol):
    category: ClassVar[Category]            # SOURCE | INTERACT | PROPAGATE | FILTER | TRANSDUCE | DETECT
    caps: ClassVar[Capabilities]

    # ---- plan time (static; may inspect descriptions and bounds, never tensors' values) ----
    def validity(self, desc, bounds: Bounds) -> list[Violation]: ...
    def cost(self, desc, problem: Problem) -> Cost: ...                 # flops, bytes (fwd, bwd)
    def configure(self, desc, problem: Problem, policy: Policy) -> Static: ...

    # ---- run time (pure torch; no Params, no RNG except Detect.sensor via explicit generator) ----
    def __call__(self, x, values: TraceView, static: Static): ...

@dataclass(frozen=True)
class Capabilities:
    accepts: frozenset[type]                # e.g. {Field[ANGULAR], SphereList}
    produces: type                          # e.g. Ports, Field[SPACE], Intensity
    diagonal: Literal["space", "angle", "intensity_freq", None]   # drives fusion
    linear_in_field: bool                   # enables mode-sum rewrites / compression
    linear_in_sample: bool                  # enables WOTF/Born-type rewrites
    ports: frozenset[str] = {"+z"}
    polarization: frozenset[int] = {1}      # supported P
    analytic_background: bool = True        # else planner inserts to_sampled() (+warning)
    grad: frozenset[GradStrategy]           # autograd | checkpoint | reversible | segmented | implicit
    grad_quality: Mapping[str, str]         # per input: "exact" | "biased" | "zero"
    tier: int                               # ordinal within category
    approximates: str | None                # the higher-tier implementation it converges to
```

### 5.2 Implementation catalogue (v1 unless marked)

**Source**

| Implementation | Produces | Tier | Notes |
|---|---|---|---|
| `plane_waves` | `PlaneWaves` (analytic) | exact | plane wave, LED, SIM groups, TIRF (complex k_z), references |
| `koehler.single` | 1 mode at the source centroid | 0 | draft; coherent approximation |
| `koehler.quadrature(K)` | K weighted modes | 1–3 | ring/hex/Fermat; error estimate from source smoothness |
| `koehler.stochastic(K)` | K modes, resampled per batch | 2 | unbiased image and gradient; noise hides under shot noise |
| `koehler.grid` | all on-grid directions | reference | exact Abbe |
| `coherent_aperture` | sampled angular `Field` | exact | focused beam, light sheet (own frame), speckle |
| `emit.gaussian` | Intensity patches (bypasses Filter) | 0 | σ≈0.21λ/NA; valid only near focus |
| `emit.scalar` | angular `Field` on `PupilGrid`, M = emitters | 1 | exact k_z defocus inside the emitter's layer; position as a phase ramp |
| `emit.vectorial` | P=2 pupil field, M = emitters × 3 dipole states | 3 | Richards–Wolf collection, 1/√cosθ, t_s/t_p, 6-basis for fixed/wobbling dipoles |
| `emit.tabulated` | Intensity via torch cubic-spline PSF | — | measured PSFs (SMAP/DECODE/uiPSF import); coefficients may be Learnable |

**Interact** (all return `Ports`; the planner assigns an implementation per object group)

| Implementation | Consumes | Ports | Validity predicate (on bounds) | Grad |
|---|---|---|---|---|
| `thin` (projection, scalar or Jones) | ProjectedMaps | +z | thickness ≤ n·λ/NA² (DOF) and max phase gradient; oblique 1/cosθ correction | autograd |
| `born` / `rytov` (slice-wise or form-factor Ewald caps) | KSpectrum or RIVolume | +z, −z (Born) | Born: Δn·d ≤ 0.35λ (Slaney); Rytov: Δn/n_m ≲ 0.05, no zeros in u_in | custom adjoint (linear) |
| `multislice.bpm` (free-space ASM inter-slice propagator, tilt correction) | RIVolume (procedural) | +z | max angle (NA_ill+NA_obj)/n ≲ 0.7; Δn ≲ 0.1 warns | reversible / segmented adjoint; in-loop VJP to primitive params |
| `multislice.legacy_dt` | RIVolume | +z | NA-truncated inter-slice propagator, for **parity tests only** | autograd |
| `mie` (homogeneous, layered) | SphereList | +z, −z | isolated: gap to others/interfaces ≳ λ, else warn C0 | autograd through fp64 recurrences |
| `dipole` (Mie-dressed α, radiative correction) | DipoleList | +z, −z | x ≲ 0.3 (≲1 warns) | autograd |
| `layered.fresnel` | LayeredMedium | +z, −z | planar interfaces | analytic |
| `superpose` (C0 composite) | groups → sum of scattered spectra on a common grid | union | warns "inter-object coupling neglected" when min gap < λ | — |
| `multislice.ssnp` (v2) | RIVolume | +z | higher angles/contrast | adjoint |
| `mbs` / `ls_krylov` (v2) | RIVolume | +z, −z | converges for non-gain media | implicit differentiation |
| `inject` (C1, v2) | RIVolume + SphereList/DipoleList | +z | local-plane-wave incident-field estimate | adjoint |

**Propagate** (diagonal in angle; basis changes are separate implementations)

| Implementation | Notes |
|---|---|
| `asm` | exact k_z with evanescent decay. Uses the `H_exp` form (store i·2πk_z once; H = exp(H_exp·Δz)) so defocus and z gradients flow. Phase arguments are built in fp64 with the carrier subtracted, then cast (brief 07: 1.6e-3 error otherwise at 2·10⁴ λ) |
| `blasm` | Matsushima band limit; default when Δz is large relative to the grid (lensless holography) |
| `fresnel` | paraxial; draft or legacy only |
| `layered` | Fresnel s/p transmission plus layer phase through the stack; Gibson–Lanni is its scalar limit |
| basis changes: `fft`/`ifft` | grid-preserving |
| `mft` | small ROIs or arbitrary pitch/offset/magnification; **IEEE matmul enforced** (TF32 measured at 4.4e-4) |
| `czt` | large outputs with arbitrary sampling |
| `evaluate` | analytic spectra (Mie, dipoles, emitters, form factors) evaluated on the requested points |

**Filter** (diagonal in angle unless stated)

| Implementation | Notes |
|---|---|
| `pupil.scalar` | product of modifiers: soft aperture (sigmoid edge so NA gets gradients), apodization, defocus, Zernike OPD, `LayerStack` OPD, masks, phase rings, stops, knife edge, spiral phase, DIC shear/bias multiplier, pixel phase/amplitude map (bilinear from a `PupilGrid` parameter) |
| `pupil.vectorial` | 3→2 collection or 2→3 focusing Jones maps plus Fresnel t_s/t_p; Jones elements (polariser, waveplate, Wollaston) |
| `pupil.per_object` | sparse path: pupil evaluated at each object's field position h, so field-dependent aberrations cost nothing extra |
| `pupil.low_rank` (v2) | dense field dependence: Σ_k m_k(r)·(P_k-filtered image), product convolution |
| `mask.spatial` | diagonal in space: field stops, image-plane SLM, pinhole (on Intensity for confocal) |

**Transduce** (diagonal in space; Intensity → EmitterSet or SourceDensity)

`linear` (rate = σ_abs(λ_ex)·QY·I), `saturating`, `two_photon` (∝ I²), `photophysics.expected` (master-equation mean occupancy, differentiable in rates), `photophysics.stochastic` (2-state Markov; score-function gradients for the rates), `bleaching` (exp(−k∫I dt)). The excitation is *evaluated at emitter positions* with `Field.evaluate_at(points)`: analytic for PlaneWaves (widefield, SIM, TIRF), interpolated for sampled fields (light sheet, focused beams).

**Detect**

| Implementation | Notes |
|---|---|
| `square_reduce` | Σ_M w_m Σ_L s_λ·QE(λ)·T(λ) Σ_P ‖·‖² into channels C through a weight tensor W[C, L, P]; **explicit background cross term**; scatter-add for `PatchGrid` inputs; intensity computed as re²+im² (no `abs()` singularity) |
| `pixel.sumpool(s)` | integer supersampling |
| `pixel.mtf` | sinc pixel MTF (+ charge diffusion) applied by rfft, then MFT to continuous pixel centres |
| `sensor.{ideal, poisson_gaussian, ccd, emccd, scmos(maps), qcmos}` | each has `expected()`, `sample(gen, estimator)` and `log_prob(measured)`; order follows microsim (Poisson → dark/CIC → full-well → EM Gamma → read → gain/offset → ADC → saturation) |

**Fused implementations** (the planner rewrites of §5.7)

`otf_strata` (collapse a), `sparse_roi` (per-emitter pupil → MFT → ROI → scatter-add), `socs(K)` (collapse b), `wotf` (weak-object linearisation), `effective_psf` (confocal, ISM, 2P, light sheet), `gaussian_splat` (L0: Gaussian primitives convolved with a Gaussian PSF, exactly in real space).

### 5.3 Illumination and coherence

Every source produces modes on M, and the M, L and P reductions happen only in `Detect.square_reduce`. The planner chooses:
- **mode sampling**: the §4.5 strategies;
- **mode chunking**: an exact checkpointed chunked sum, img = Σ_chunks checkpoint(render_chunk). Brief 07 measured 3.74 GB → 0.51 GB at chunk 8 with gradients identical to 1e-6. The chunk size comes from a per-stage memory estimate and `torch.cuda.mem_get_info()`.

Thick samples under partial coherence need one Interact per mode (Abbe). Two cheaper options:
- `stochastic(K)`: fresh directions per batch, unbiased in both value and gradient, with residual noise ~1/√K;
- a random-phase superposition mode, valid because Interact is `linear_in_field`. It costs one march per realisation, but speckle contrast is ~1, so it is only offered when explicitly requested.

Thin samples with fixed optics may use SOCS: the TCC is never formed, and a thin SVD of the stacked shifted pupils gives K kernels (brief 05). **SOCS is refused when any source or pupil parameter requires grad**, because SVD gradients are unstable for the degenerate singular values that symmetric sources produce. The planner falls back to quadrature Abbe and says why.

### 5.4 Imaging system: how granularly to simulate the optical path

**The optical path is modelled at the granularity of conjugate planes, not optical surfaces.** In a well-corrected, telecentric, infinity-corrected microscope, everything between conjugate planes is either:
- a **coordinate map** (magnification, image inversion, distortion as a warp), or
- a **diagonal operator foldable into a pupil** (aberrations, relay filters, phase rings, SLMs in a Fourier plane, polarisation optics, apodization, Fresnel losses).

gradoscopy therefore propagates light explicitly only in these places:
1. inside the sample (Interact);
2. from the sample reference plane to the focal plane through the layered medium (defocus and Gibson–Lanni, both diagonal in angle);
3. in lensless or in-line configurations without an objective (free-space `blasm`);
4. inside the illumination's own frame for sheets and beams.

Everything is simulated in **object space**. The camera grid is `pitch/M` with continuous, learnable M and offset, so the image side needs no Debye/Fresnel modelling (image-side NA/M is tiny). Image-plane elements (pinholes, field stops, a camera tilt modelled as field-dependent defocus) are spatial diagonals or per-object pupil modifiers. External ray tracers can supply a pupil OPD or Jones map; the **pupil is the closure of the objective**.

The `Objective` node therefore expands into Propagate(z_ref → z_focus) ⊗ Filter(pupil) followed by a basis change to the camera grid. Because the first two are diagonal in angle, the planner fuses them into one multiplier per (sample, λ, P), computed once per call and shared by all modes.

### 5.5 Detection

The Detect path is reduce → pixel integrate → sensor, with three execution modes:
- `expected` (noise-free mean, differentiable);
- `sample` (generator and per-stage estimator, §6);
- `log_prob(measured)` (exact Poisson, Poisson⊛Gaussian with per-pixel sCMOS maps, and EMCCD Gamma–Poisson).

Fitting uses `expected` + `log_prob`, and data generation uses `sample`. Stage **taps** can return intermediates: `pupil_field`, `image_field` (complex, for holographic targets), `noiseless`, `per_channel` and `psf`.

### 5.6 How fidelity is expressed

Fidelity is **a named preset that expands into a per-category policy, with per-category and per-object overrides**. It is discrete and ordered; there is no continuity requirement.

```python
gs.Fidelity("balanced")                                          # preset
gs.Fidelity("fast", filter={"polarization": "vectorial"})        # per-category override
gs.Fidelity("balanced", overrides={"objects.beads": {"interact": "mie"},
                                   "objects.cell":  {"interact": "multislice.bpm"}})
gs.Fidelity("accurate", sampling={"oversample": 2.0}, illumination={"modes": 64}, pin={"grid": (512, 512)})
```

| Axis | `draft` | `fast` | `balanced` | `accurate` | `reference` |
|---|---|---|---|---|---|
| Emission | Gaussian sprites | scalar pupil, exact k_z | scalar + apodization + Gibson–Lanni | vectorial 6-basis + Fresnel | vectorial, per-emitter field-dependent pupils |
| Illumination modes | 1 (centroid) | quadrature K≤9 (or SOCS if thin & fixed) | quadrature ≈25 or SOCS(16) | quadrature ≈50 / stochastic | exact grid Abbe |
| Interact, dense | projection | projection / Rytov (by validity) | Rytov or BPM (by validity) | BPM (tilt-corrected); SSNP in v2 | MBS (v2) |
| Interact, spheres | dipole if x<0.3, else projection | Mie | Mie | layered Mie | Mie (+ coupling, v2) |
| Spectral bins | 1 | 1 | 3 | 5 | ≥9 + convergence check |
| Polarisation | P=1 | P=1 | P=1 (P=2 if NA>1.0 and fixed dipoles) | P=2/3 | P=3 |
| Sampling | camera grid | Nyquist (λ/4NA) | 1.25× Nyquist | 2× Nyquist | 3× + slice/mode convergence |
| Detector | expected + Gaussian | Poisson (scaled ST) + read | full chain | full + maps | full |
| `legacy_dt` | reproduces DT2 2.0.2 numerics (hard aperture, NA-truncated slices, bilinear placement, additive overlap) for parity tests | | | | |

### 5.7 The planner: what it does and when

**When.** Planning happens when a `Simulator` is built, and again only when one of these changes:
- (i) the static structure of the spec (object kinds, graph);
- (ii) the fidelity policy;
- (iii) the *bounds*;
- (iv) the device or batch-size bucket.

Plans are cached by a hash of these. Updating a Learnable value **never** triggers re-planning, because shapes derive from bounds. A runtime **bounds guard** checks the resolved Trace against the planned bounds. For example, a learnable mean radius may drift above the radius used to size the padding. The guard warns and schedules a re-plan before the next call, so the current batch is never silently wrong. The target planning cost is under 50 ms. Planning happens neither per batch nor at trace time.

**How (rule-based, deliberately boring).**
1. **Graph expansion.** The modality recipe expands into category nodes, and macro nodes (`Objective`, `Camera`) become their constituents.
2. **Bounds analysis.** Distribution supports and quantiles (99.9 % for unbounded) give max object extent, max Δn, max phase delay, size parameter x, z-range, NA and σ ranges and λ range. Leaves are classified as learnable, discrete or smooth.
   - Distributions with *learnable* parameters are evaluated at the current parameter values, widened by a margin (default 1.25×).
   - The runtime bounds guard enforces the result and triggers a re-plan when training moves a distribution past it.
   - Explicit `bounds=` on a `Learnable` replace the margin.
3. **Candidates.** For each node (per object group for Interact), the registry is filtered by the fidelity policy, `accepts` (plus a Dijkstra search over the small lowering graph weighted by cost and gradient-quality penalty), `validity` (errors reject, warnings are recorded), and `grad_quality` for every learnable input. A learnable radius on a `zero`-gradient path is rejected.
4. **Rewrites** (theorem-guarded; each is logged):
   - **(a) Shift-invariant emission** → `otf_strata`: Σ_e w_e|h(r−r_e;z_e)|² = Σ_z c_z ⊛ |h_z|². Requires lateral invariance of Filter and a dense emitter source.
   - **Sparse vs dense emitters**: choose `sparse_roi` when N_e·R² ≲ N_z·N²·log N. Measured: 50 emitters per 256² frame is 27k frames/s sparse vs ~2.5k dense per-emitter.
   - **(b) Thin sample + fixed optics** → `socs(K)`. **Weak** (max |φ| ≲ 0.3 rad) → `wotf`.
   - **Two-stage point-scanned** → `effective_psf` (h_exc·(h_det ⊛ D)); otherwise scanning becomes an acquisition axis (reference).
   - **Fusion**: adjacent same-basis diagonals multiply into one kernel; consecutive homogeneous propagations add their Δz; pixel MTF folds into the intensity OTF.
5. **Sampling.**
   - Spacing is min(camera pitch/M/s, λ_min/(4·NA_max)).
   - Padding is ≥ z_max·tanθ_max + 2λ/NA, plus coherence width 0.61λ/NA_c and object margins, rounded to smooth sizes.
   - Pupil grid resolution comes from ROI extent and defocus; for the MFT period, λ/du must exceed ROI + support.
   - BPM Δz ≈ λ_m/2…λ_m, with empty slabs merged using bboxes.
   - Spectral bins and mode counts are also set here.
6. **Memory and grad strategy.** Per-stage memory estimates choose chunk sizes and a grad strategy (autograd, checkpoint, reversible, segmented or implicit) under a budget, which defaults to 80 % of free memory.
7. **Emit the `Plan`.** It is an ordered list of `Stage(name, impl, static, in/out types and axis sizes, validity, cost, grad)` plus grids, and it is printable, hashable and serialisable.

**Inspection.**

```text
>>> print(sim.plan.explain())
Plan 3f9c1a  fidelity=fast  device=cuda:0  batch≤64  planned in 21 ms
grids   sample 224×224 @108.3 nm (object space; 224=2^5·7), pad 5.2 µm [z∈±2 µm, NA 1.2, n 1.33]
        camera 128×128 @108.3 nm = 6.5 µm/60  (s=1: 108.3 ≤ λ/4NA = 131 nm), pixel MTF on intensity
modes   illumination Koehler σ=0.17 → 7 modes (hex quadrature, est. rel.err 3e-3); spectrum 1 bin (630 nm)
 1 source.plane_waves        Koehler → PlaneWaves[M=7,J=1,L=1,P=1]                analytic background
 2 interact.mie              objects.beads: Sphere ×12 slots, fp64 coeffs, n_max ≤ 15
                             (x ≤ 5.3: r ≤ 0.40 µm at q99.9 of current LogNormal; bounds guard armed)
                             S(θ) 1024 samples → evaluated on FreqGrid 224², ports {+z}
 3 interact.superpose[C0]    ⚠ inter-particle coupling neglected (bounds: min gap unconstrained)
 4 filter.pupil.scalar       fused: defocus(z_ref→z_f)·aperture(soft)·zernike(ANSI 12)·apod(collect) [angle]
                             per-sample (Zernike is Random, plate=sample) → [64,1,1,224,224] multiplier
 5 propagate.ifft            ANGULAR→SPACE 224² → crop 128²
 6 detect.square_reduce      Σ_M |b+E_s|² with explicit cross term (b analytic)
 7 detect.sensor.scmos       Poisson(scaled-ST) + read 1.6 e⁻ + ADC 16 bit (ST round)
rejected interact.dipole (beads): size parameter x ≤ 5.3 exceeds dipole limit 0.3
         illumination.socs: sample not thin (Mie objects) and pupil varies per sample
cost     fwd ≈ 2.9 GFLOP, peak 0.7 GiB; fwd+bwd ≈ 2.1 GiB at B=64 (grad: autograd)
```

### 5.8 Validity checks

- **Plan time.** Every implementation evaluates its predicate against bounds and returns `Violation(severity ∈ {info, warn, error}, message, quantity, bound, threshold)`. Starting thresholds come from the literature (the §5.2 table; scalar PSFs with fixed dipoles at NA > 1.2 warn about ~40 nm localisation bias per Stallinga & Rieger). They are then **recalibrated from our own cross-fidelity convergence tables** (§11). Errors can be downgraded with `strict=False`.
- **Run time** ("sampling health", optional `diagnostics=True`, a few cheap reductions per batch):
  - fraction of scattered spectral energy in the outer 5 % of the frequency band (aliasing);
  - energy reaching the padding border (wrap-around);
  - maximum per-slice phase encountered;
  - Mie truncation residual |a_{n_max}|;
  - out-of-FOV emitter fraction;
  - bounds-guard status.

  These go into `batch.meta["diagnostics"]`.

### 5.9 How lower fidelities relate to higher ones

1. **One description.** Every tier renders the *same* spec. Parameters a tier cannot represent follow a declared per-parameter policy (`ignore+warn`, `approximate`, `error`). For example, the Gaussian tier folds Gibson–Lanni into an effective defocus plus width.
2. **Declared limits.** Each implementation names the implementation it `approximates` and the regime in which it converges (projection → Rytov → BPM → SSNP → MBS; dipole → Mie; Gaussian → scalar → vectorial; quadrature(K) → grid Abbe as K grows; SOCS(K) → Abbe as K → rank).
3. **Convergence tests** in CI encode these limits and produce error-versus-regime tables that set the validity thresholds.
4. **Cross-fidelity surrogate gradients** are opt-in: `y = y_hi.detach() + y_lo − y_lo.detach()`. The value comes from the accurate tier and the gradient from the fast one, which speeds up fitting.
5. **`gs.compare(scene, scope, ["fast", "accurate"])`** reports rel-L2, SSIM, localisation bias and the like on shared traces (same replayed samples), so users can measure what an approximation costs *for their scene*.

---

## 6. Parameters, randomness and learnability

### 6.1 Parameter kinds

| Kind | Spelling | Storage | Gradient |
|---|---|---|---|
| Fixed | plain number/tensor, or `gs.Fixed(v)` | buffer | none |
| Learnable | `gs.Learnable(init, constraint=None, scale=None, bounds=None, name=None)` | `nn.Parameter` in unconstrained space (`transform_to(constraint).inv`) divided by `scale` | exact |
| Random, with fixed or learnable distribution parameters | `gs.dist.Normal(μ, σ, plate="sample")`; any argument may itself be a Param | distribution parameters in the ParamStore | per-site estimator (§6.3) |
| Derived | `gs.Derived(fn, radius="objects.beads.radius")` | none (computed during resolve) | through `fn` (torch code) |
| Choice | `gs.Choice([...], probs, estimator="relaxed" \| "score" \| "enumerate")` | logits in the ParamStore | relaxed, score or exact enumeration |

**`scale` fixes optimiser conditioning in SI units.** `Learnable(150*nm, scale=nm)` stores 150.0, so L-BFGS/SGD step sizes are sensible while physics stays in metres. `bounds=` feeds the planner; without it, the planner uses a default envelope (×/÷ 2 of init) and the runtime bounds guard.

**Sharing is by identity.** The same `Learnable` object used in two specs (e.g., a pupil calibrated on beads, then reused to generate SMLM data) is one parameter. The `ParamStore` (an `nn.Module` owned by the `Simulator`) exposes path names through `named_parameters()`, `state_dict()` and `.to()`.

### 6.2 Resolve → render, Trace and plates

`sim(batch_size, seed)` does two things:
1. **resolve**: walks the Param graph, samples every site with `rsample` where possible, and returns a `Trace`;
2. **render**: runs the Plan on the Trace.

A `Trace` is a frozen, pytree-registered mapping `path → Site(value[B, …], plate, estimator, log_prob?, mask?)`, with an optional `to_tensordict()` export. **Plates** state granularity: `batch` (one draw shared by the batch), `sample` (per image), `object` (per slot), `frame` (per acquisition/time index). This generalises Kornia's `same_on_batch`.

Handlers are call-scoped context managers implemented with `contextvars`; there is no global stack:
- `gs.condition({path: value})` pins sites, e.g. annotated positions from real data;
- `gs.replay(trace)` reproduces exactly;
- `gs.mean()` gives a deterministic expected scene for debugging;
- `gs.block(paths)` and `gs.detach(paths)` control what is sampled or differentiated.

### 6.3 Gradient estimators for non-reparameterisable parts

All library randomness goes through gradoscopy's own samplers. `torch.poisson`, `torch.normal(tensor, tensor)` and `bernoulli` silently return zero gradients (brief 07). A CI test asserts a **non-zero gradient for every Learnable and every learnable distribution parameter**.

| Quantity | Default estimator | Alternatives |
|---|---|---|
| Continuous sites (size, RI, position, D, …) | pathwise `rsample` (Normal, LogNormal, Uniform bounds, Gamma/Beta implicit) | score |
| Shot noise | **scaled straight-through**: exact Poisson sample n forward, y = λ + √λ·((n−λ)/√λ).detach(), so ∂y/∂λ = 1 + ε/(2√λ) | Gaussian reparameterisation (λ ≳ 20), score function (low counts), `expected` + `log_prob` for fitting |
| EM gain | Gamma `rsample` (implicit reparameterisation) conditional on electrons | — |
| Read noise | Gaussian reparameterised | — |
| Quantisation / saturation | straight-through round / soft clamp (optional) | hard clamp (zero gradient, declared) |
| Object count | N_max slots with presence ~ RelaxedBernoulli(p, τ) hardened by straight-through; p from the count law (Poisson(λ) → Binomial(N_max, λ/N_max)). Presence multiplies scattered amplitude or photons | score function on the count's `log_prob` (unbiased) |
| Categorical (shape kind, material) | Gumbel-softmax mixture of occupancies or amplitudes | score; `enumerate` (render all K, exact expectation, ×K cost) |
| Photophysics | `expected` occupancy (differentiable in rates) | `stochastic` 2-state Markov + score on the rates |
| Monte-Carlo modes (stochastic Abbe, speckle) | pathwise through the realised modes (unbiased) | increase K |
| Constraints (non-overlap) | allow overlap with principled compositing, or soft repulsion (a few unrolled steps) | rejection sampling (declared: drops the acceptance-probability gradient) |
| Fidelity choice | not a random variable | cross-fidelity surrogate (§5.9) |

`batch.score_surrogate(loss, baseline="ema")` adds Σ log_prob·stop_grad(loss − b) for score sites only. It returns 0 when there are none, so it is safe to call always.

### 6.4 Batching semantics

- **B leads everywhere.** Any physical scalar may be `[B]`: per-sample defocus, NA, λ or Zernike vector. Diagonal kernels are generated from broadcast parameter tensors, which is cheap because pupils are small. They are **memoised within one call** (compute once per unique parameter tensor) and never cached across calls when they require grad.
- **Variable counts** use padded `[B, N_max]` slots and presence. `N_max` is the count distribution's `max` (a static plan property), bucketed to avoid recompiles.
- **Acquisition axes** (A) and time frames are plates. Per-frame sites (blinking, diffusion steps) use `plate="frame"`.

### 6.5 Seeding and reproducibility

- `sim(batch_size, seed=s)` builds one `torch.Generator(device)` seeded with H(s, rank). Per-site substreams H(s, site) are optional (a counter-based mode), so adding a site does not shift the others.
- The detector has its own substream, so scene and noise can be varied independently.
- Global RNGs are never touched.
- Exact reproduction comes from **replaying stored Traces**, not from RNG alignment across versions.
- Outputs carry `meta = {spec_hash, plan_hash, version, seed, torch, device}`.

### 6.6 Ground truth and labels

Labels are opt-in outputs of the *same* resolved scene, and are paid for only when requested:
- **Trace values**: any path, e.g. radius, n, xyz, photons, Zernike coefficients.
- **Label renderers**: `Positions` (camera-pixel coordinates after magnification, offsets and in-FOV flags, as tv_tensors `KeyPoints`), `Masks`/`InstanceMasks`/`DistanceMap`/`Density` (from band-limited occupancy at the camera grid with the same transforms), `EmitterTable` (DECODE-like per-frame table with `id`).
- **Taps**: noise-free image, complex image field, pupil field, PSF.

Because labels and images share the Trace, augmentations stay consistent by construction; DeepTrack's `_update_properties` machinery becomes unnecessary.

---

## 7. Composition and extension

### 7.1 Composition

```python
# A modality is a recipe: category nodes + combinators. This is the full brightfield family.
def Brightfield(illumination, objective, camera, *, output="intensity"):
    return gs.graph(
        gs.node.Source(illumination),        # AngularSource → modes (planner picks sampling)
        gs.node.Interact(),                  # the Scene; implementation per object group by planner
        gs.node.Objective(objective),        # expands to Propagate(z_ref→z_f) ⊗ Filter(pupil) → basis change
        gs.node.Camera(camera, output=output),  # Detect: reduce → pixels → sensor (or a field tap)
    )

Darkfield     = lambda na_c, objective, camera: Brightfield(gs.light.Koehler(annulus=na_c), objective, camera)
PhaseContrast = lambda annulus, ring, objective, camera: Brightfield(
                    gs.light.Koehler(annulus=annulus), objective.with_modifier(ring), camera)
FPM           = lambda leds, objective, camera: gs.acquire(Brightfield(leds, objective, camera), over="led")

def OffAxisHolography(illumination, reference, objective, camera):
    return gs.graph(gs.node.Source(illumination), gs.node.Interact(), gs.node.Objective(objective),
                    gs.coherent_sum(gs.node.Source(reference)),   # reference arm joins the image-plane field
                    gs.node.Camera(camera))

def Widefield(emission, objective, camera, excitation=gs.light.Uniform()):
    return gs.graph(gs.node.Source(excitation), gs.node.Transduce(), gs.node.Source.emission(emission),
                    gs.node.Objective(objective), gs.node.Camera(camera))
TIRF = lambda angle, **kw: Widefield(excitation=gs.light.Evanescent(angle), **kw)
SIM  = lambda beams, **kw: gs.acquire(Widefield(excitation=beams, **kw), over=("angle", "phase"))
```

Combinators are few and typed:
- `>>` / `gs.graph` (sequence);
- `coherent_sum` (add fields from a parallel arm; both must share modes and λ);
- `acquire(over=…)` (stack along A);
- `split(channels=…)` (dichroics, polarising splitters, biplane: channels C with per-channel affine registration).

Build-time type checks enforce the carrier types: Filter needs a `Field`, Transduce needs an `Intensity` evaluated at a sample, and Detect ends a chain. Domain changes (space ↔ angle) are inserted automatically by the planner, so users never write FFTs.

### 7.2 Adding a solver (third party)

```python
@gs.register.operator("interact.wpm")            # wave-propagation method (Brenner–Singer), a BPM mode
class WPM(gs.Operator):
    category = gs.Category.INTERACT
    caps = gs.Capabilities(
        accepts={gs.views.OccupancyChannels}, produces=gs.Ports, diagonal=None,
        linear_in_field=True, linear_in_sample=False, ports={"+z"}, polarization={1},
        grad={"segmented", "autograd"}, grad_quality={"geometry": "exact", "material": "exact"},
        tier=4, approximates="interact.mbs",
    )
    def validity(self, desc, b):
        return [gs.Violation("warn", "many index levels; cost ∝ levels", "n_levels", b.n_materials, 6)] \
               if b.n_materials > 6 else []
    def cost(self, desc, p):  return gs.Cost.fft2(p.grid, count=p.n_slices * (p.n_materials + 1), batch=p.batch)
    def configure(self, desc, p, policy): return dict(dz=policy.slice_thickness(p), pad=p.pad)
    def __call__(self, field, values, static):   # pure torch; receives resolved tensors only
        ...
```

Registration can also come from `[project.entry-points."gradoscopy.plugins"]`. The **conformance suite** `gs.testing.conformance(WPM)` runs gradcheck (complex128, tiny shapes), linearity-in-field, energy conservation for lossless samples, analytic-background handling, convergence to the implementation named in `approximates` within the declared regime, and `validity`/`cost` sanity checks. A plugin that passes becomes selectable by the planner and inherits every modality that uses Interact.

### 7.3 Adding a primitive

```python
@gs.register.primitive("torus")
class Torus(gs.sample.Primitive):
    params = ("major", "minor")                       # SoA fields [B, N]
    def bbox(self, v): ...
    def sdf(self, v, x): ...                          # exact torus SDF → occupancy via the SDF ramp for free
    def form_factor(self, v, k): ...                  # optional: enables Born/OTF paths without voxelisation
    def project(self, v, xy): ...                     # optional: enables the thin tier analytically
```

The lowering registry derives the rest: `sdf` → `OccupancyChannels` (approx(first-order)) → `RIVolume`/`EmitterDensity`; `form_factor` → `KSpectrum` (exact); surface sampling → `EmitterSet` for membrane labeling. Implementing `sdf` alone makes a new shape available to BPM, fluorescence, masks and labels.

### 7.4 Adding a modality

This is a recipe (§7.1). New physics goes into a pupil modifier, an AngularSource preset, or at worst a new operator implementation. A Hoffman modulation-contrast recipe, for example, is a slit `AngularSource` plus a graded-amplitude pupil modifier: about eight lines and no new kernels.

### 7.5 Why the whole exceeds the parts

The categories are orthogonal, so contributions multiply:
- a new **primitive** (with `sdf`) works in every Interact tier, every modality and every label renderer;
- a new **Interact** implementation upgrades every transmitted, reflected and interferometric modality;
- a new **pupil modifier** is available to fluorescence, brightfield, iSCAT and SIM at once;
- a new **camera** model serves all modalities.

Reference solvers manufacture fast ones: convergence tables calibrate validity, and in v3 reference outputs train learned surrogates that plug in as tiers. The same objects serve training data, calibration (`log_prob`), design (learnable masks) and reconstruction (adjoints).

---

## 8. Performance and memory strategy

### 8.1 Numbers that shape the design

| Fact (measured on this 3090) | Design consequence |
|---|---|
| fft2 c64 64×512²: 0.705 ms; an elementwise multiply costs 0.525 ms (brief 07) | the loop is bandwidth-bound, so fuse diagonals (planner fusion now, Triton kernels later); avoid materialising H per (z, λ) |
| c128 FFT 7.5× slower; complex32 NaN in multislice (07) | complex64 everywhere; fp64 only for special functions and phase construction |
| Prime FFT sizes up to 1.9× slower (07) | 2·3·5·7-smooth static grids per plan |
| TF32 → 4.4e-4 error in complex matrix DFT (new) | `ieee_matmul()` wraps every MFT/CZT; `no_autocast()` wraps the simulator |
| Sparse per-emitter MFT: 27k frames/s (scalar), 10.8k (vectorial 6-basis), 67k at B=256×20 (new) | `sparse_roi` is the default fluorescence path for point sources; vectorial is affordable in v1 |
| Dense per-emitter FFT ≈ 2.5k frames/s at 256², 50 emitters (new) | the planner crossover rule of §5.7 |
| Multislice B=16, 512², 32 slices: naive 3.0 GB/163 ms; checkpoint 1.30 GB/133 ms; reversible adjoint 0.80 GB/100 ms, grad err 7e-6 (07) | multislice ships with a reversible (lossless) and a segmented (absorbing) adjoint from day one |
| A 64×256×512² RI volume is 16 GiB, plus 16 GiB of gradient (07) | **procedural slices** generated inside the adjoint with an in-loop VJP to primitive parameters; volumes are materialised only when the user supplies voxels |
| Mode chunking M=64: 3.74 GB → 0.51 GB (chunk 8), 0.11 GB (chunk 1), same gradients (07) | exact checkpointed chunked reductions sized from `mem_get_info` |
| CUDA graph replay 2.9× for small workloads; ~8 µs per op (07) | capture the Plan per static config for small tiers (`reduce-overhead`) |
| Mie D_n in c64: 1–5 % error; 4096 particles ≈3 ms launch-bound (07); S(θ) synthesis 6.9 ms for 640 particles (new) | fp64 coefficients on a 1-D θ grid, batched over *all* particles × λ in one call, then evaluated on the pupil; never per-pixel order sums (would be ~1 GiB per image) |
| Patch voxelisation, 10k spheres: 0.99 s, 17.9 GB with autograd (06) | custom autograd function (recompute in backward); optional Triton kernel; bucket by size |
| Inductor does not codegen complex; Triton is absent on Windows (07) | correctness never depends on compile; real-view `torch.compile` of per-slice steps is an optional accelerator |

### 8.2 Execution strategies

1. **No graph when nothing is learned.** If no resolved site requires grad, the run uses `inference_mode` and in-place fast paths. This is the normal data-generation case.
2. **Diagonal fusion.** The pupil-side multiplier (defocus · aperture · aberration · layer stack · masks · Jones) is computed **once per call per unique parameter tensor**, not per mode or per sample, and applied with one fused multiply. A Triton kernel generating H on the fly avoids materialising `[L, Z, Y, X]` kernels (v1.x).
3. **Exact chunked reductions** over M (modes/emitters) and L, with checkpointing. Chunk sizes come from stage memory estimates.
4. **Adjoints as first-class operator capabilities.** Multislice uses reversible or segmented adjoints; Born/Rytov and all linear filters use analytic adjoints; MBS (v2) uses implicit differentiation. The forward solver and the gradient strategy are separate plan choices (the SciML `sensealg` idea).
5. **Static shapes per plan** give stable cuFFT plans (the plan cache is keyed on shapes) and make CUDA graphs viable. Set `PYTORCH_ALLOC_CONF=expandable_segments:True`.
6. **rfft** for every real-valued intensity convolution (OTF strata, pixel MTF).
7. **Culling by bbox** (object support plus PSF and defocus margins). Empty slabs are merged into one propagation step, and empty z-strata are skipped.
8. **Generation happens on the GPU in the main process** (`SimDataset` is an `IterableDataset` with `num_workers=0`). Multi-GPU is rank-local generation plus DDP, Linux only.

### 8.3 Throughput and memory targets (exit criteria in §12)

| Workload | Tier | Target |
|---|---|---|
| SMLM frames 64², 20 emitters, scalar, sCMOS | fast | ≥ 50k frames/s fwd (measured 67k with the kernel alone) |
| Tracking, BF, 256², 5 Mie particles, 7 Köhler modes | fast | ≥ 2k frames/s fwd |
| Phase-contrast cells, 256², projection + SOCS(8) | fast | ≥ 1k frames/s |
| Same scene with BPM (640², 8 slices × 48 modes × 3 λ), B=64 | accurate | ≤ 6 s per batch fwd (≈4 ms per 64-field slice step at 640²); stochastic(K=8) for training ≈ 1 s |
| Bead-stack pupil fit, 20 beads × 41 z × 31², vectorial | accurate | < 60 s to convergence (uiPSF voxel: 40 s on a 3080) |
| ODT recon, 256²×128 RI, per-LED mini-batch of 8, BPM + reversible adjoint | accurate | < 4 GiB peak |

---

## 9. User-facing API

All examples use the same public surface: `gs.Simulator`, `gs.Scene`, `gs.sample`, `gs.materials`, `gs.light`, `gs.optics`, `gs.cameras`, `gs.modalities`, `gs.dist`, `gs.Learnable`, `gs.Fidelity`, `gs.labels`, `gs.units`.

### (a) Brightfield of Mie spheres in a GPU training loop with a learnable size distribution

```python
import torch, gradoscopy as gs
from gradoscopy.units import nm, um
D = gs.dist

r_median = gs.Learnable(250 * nm, constraint="positive", scale=nm, bounds=(50 * nm, 1 * um))
r_spread = gs.Learnable(0.15, constraint="positive", bounds=(0.01, 0.6))

beads = gs.sample.Population(
    "beads",
    geometry = gs.sample.Sphere(radius=D.LogNormal.from_median(r_median, r_spread)),
    material = gs.materials.Constant(n=D.Uniform(1.40, 1.60)),
    position = gs.sample.UniformIn(fov=gs.FOV.padded(1 * um), z=D.Uniform(-2 * um, 2 * um)),
    count    = D.Poisson(4.0, max=12),                     # 12 slots + relaxed presence (planner-visible)
)
scene = gs.Scene(environment=gs.materials.Water(), objects=[beads])

scope = gs.modalities.Brightfield(                          # InlineHolography is the same with a PlaneWave
    illumination = gs.light.Koehler(wavelength=630 * nm, na=0.2),
    objective = gs.optics.Objective(
        na=1.2, magnification=60, immersion=gs.materials.Water(),
        aberrations=gs.optics.Zernike(ansi={12: D.Normal(0.0, 15 * nm)})),   # OPD, per-image random
    camera = gs.cameras.SCMOS(pitch=6.5 * um, shape=(128, 128), qe=0.8, read_noise=1.6,
                              exposure_photons=2e4),
)

sim = gs.Simulator(scene, scope, fidelity="fast", device="cuda",
                   labels=[gs.labels.Positions("beads"), gs.labels.Values("beads.radius")])
print(sim.plan.explain())                                   # see §5.7 for the output

net = MyTracker().cuda()
opt_net = torch.optim.AdamW(net.parameters(), 3e-4)
opt_sim = torch.optim.Adam(sim.parameters(), 1e-2)          # exactly r_median, r_spread
critic = gs.learn.MMD(features=[gs.learn.RadialPowerSpectrum(), gs.learn.IntensityHistogram(64)])

for step, real in enumerate(real_loader):                   # unlabeled experimental crops [64,1,128,128]
    batch = sim(batch_size=64, seed=step)                    # ONE batched GPU render; TensorDict-like
    # 1) train the network on simulated data (simulator graph cut)
    loss_net = net.loss(batch.image.detach(), batch.labels["beads.positions"], batch.labels["beads.radius"])
    opt_net.zero_grad(); loss_net.backward(); opt_net.step()
    # 2) move the size distribution toward the real data
    loss_sim = critic(batch.image, real.cuda())
    loss_sim = loss_sim + batch.score_surrogate(loss_sim)    # no-op here: all sites are pathwise/relaxed
    opt_sim.zero_grad(); loss_sim.backward(); opt_sim.step()
```

Gradients reach `r_median` and `r_spread` through `rsample` → Mie coefficients (fp64 recurrences) → S(θ) → analytic pupil spectra → the explicit interference term → the scaled-ST Poisson.

### (b) SMLM: learnable pupil (Zernike + pixel pupil) fitted to a bead stack, then reused

```python
stack = torch.load("beads.pt").cuda()                       # [20 beads, 41 z, 31, 31] ADU
z_steps = torch.linspace(-1, 1, 41) * um

pupil = gs.optics.Pupil(
    aperture = gs.optics.Aperture(na=1.45, edge=0.005),      # soft edge → NA is learnable if wanted
    stack    = gs.optics.LayerStack(immersion=1.518, coverslip=(1.523, 170 * um), sample=1.33),
    zernike  = gs.optics.Zernike(ansi=range(3, 28), coeffs=gs.Learnable(torch.zeros(25), scale=nm)),
    residual = gs.optics.PixelPupil(shape=(64, 64),
                                    phase=gs.Learnable(torch.zeros(64, 64)),
                                    amplitude=gs.Learnable(torch.ones(64, 64), constraint="positive"),
                                    regularizer=gs.reg.TotalVariation(1e-3)),
)
objective = gs.optics.Objective(pupil=pupil, magnification=100)
camera = gs.cameras.SCMOS(pitch=11 * um, shape=(31, 31), offset=offset_map, variance=var_map, gain=gain_map)

beads = gs.sample.Emitters("beads",
    position=gs.Learnable(xyz0, scale=nm),                   # [20, 3] from peak finding
    photons=gs.Learnable(torch.full((20,), 5e3), constraint="positive"),
    dipole="isotropic", bead_diameter=100 * nm)              # bead-size kernel, uiPSF-style
bead_scope = gs.modalities.Widefield(
    emission=gs.light.Spectrum.band(680 * nm, 30 * nm, bins=3), objective=objective, camera=camera,
    background=gs.Learnable(torch.full((20,), 10.0)),
    acquisition=gs.acq.FocusSteps(z_steps))                  # A = 41
fit = gs.Simulator(gs.Scene(emitters=[beads]), bead_scope,
                   fidelity="accurate", batch_axis="beads")   # vectorial sparse_roi; beads on B

opt = torch.optim.LBFGS(fit.parameters(), lr=1, max_iter=300, line_search_fn="strong_wolfe")
def closure():
    opt.zero_grad()
    mu = fit.expected().counts                               # noise-free ADU [20, 41, 1, 31, 31]
    nll = -camera.log_prob(stack, mu).sum() + fit.regularization()
    nll.backward(); return nll
opt.step(closure)

# ---- generate SMLM training data with the calibrated optics (same Pupil object → shared) ----
pupil.freeze()                                               # Learnables → buffers, in place
blinkers = gs.sample.EmitterPopulation("mol",
    position=gs.sample.UniformIn(fov=gs.FOV, z=D.Uniform(-0.8 * um, 0.8 * um)),
    photons=D.LogNormal.from_median(2000.0, 0.5), count=D.Poisson(15.0, max=64),
    photophysics=gs.photophysics.TwoState(k_on=0.05, k_off=0.5, mode="stochastic"))
smlm = bead_scope.replace(camera=camera.cropped(64, 64), acquisition=gs.acq.Frames(3),
                          background=D.Uniform(20.0, 120.0, plate="sample"))
train_sim = gs.Simulator(gs.Scene(emitters=[blinkers]), smlm, fidelity="accurate",
                         labels=[gs.labels.EmitterTable("mol")])
ds = gs.data.SimDataset(train_sim, batch_size=64, seed=0)    # ~10k+ frames/s vectorial (§8.1)
```

### (c) A cell in phase contrast (and QPI) through multislice, with a primitive-built or implicit sample

```python
cell = gs.sample.Group("cell", parts=dict(
    body     = gs.sample.Capsule(length=D.Uniform(3 * um, 6 * um), radius=D.Uniform(0.45 * um, 0.6 * um),
                                 orientation=D.UniformRotation(), material=gs.materials.Constant(1.365)),
    nucleoid = gs.sample.Ellipsoid(semi_axes=(0.8 * um, 0.3 * um, 0.3 * um), parent="body",
                                   material=gs.materials.Constant(1.380)),
    texture  = gs.sample.RandomField(gs.sample.VonKarman(corr_len=D.Uniform(0.2 * um, 0.6 * um),
                                                         rms_dn=gs.Learnable(0.004)), inside="body"),
))
# Alternative: an implicit / neural RI field
# cell = gs.sample.FieldComponent("cell", fn=ri_mlp, bbox=((-4*um,)*3, (4*um,)*3), quantity="dn")

scene = gs.Scene(environment=gs.materials.Constant(1.335),
                 objects=[gs.sample.Population("cells", template=cell, count=D.Poisson(6.0, max=12),
                                               position=gs.sample.UniformIn(fov=gs.FOV, z=0.0))])

pc = gs.modalities.PhaseContrast(
    illumination = gs.light.Koehler(wavelength=gs.light.Spectrum.band(550 * nm, 40 * nm), annulus=(0.30, 0.34)),
    ring = gs.optics.PhaseRing(na=(0.29, 0.35), phase=torch.pi / 2, transmission=0.2),
    objective = gs.optics.Objective(na=0.95, magnification=40),
    camera = gs.cameras.SCMOS(pitch=6.5 * um, shape=(256, 256)),
)
sim = gs.Simulator(scene, pc, fidelity=gs.Fidelity("accurate", interact="multislice.bpm"),
                   labels=[gs.labels.InstanceMasks("cells"), gs.labels.Values("cells.body.length")])
batch = sim(batch_size=16, seed=0)      # halo/shade-off emerge from annulus × ring overlap, not a tuned offset

# QPI: same scene, coherent plane wave, return the complex image field
qpi = gs.modalities.InlineHolography(gs.light.PlaneWave(550 * nm), pc.objective, pc.camera, output="field")
phase = gs.Simulator(scene, qpi, fidelity="accurate")(batch_size=16, seed=0).field.angle()
```

The BPM consumes a *procedural* `RIVolume`. Slices are generated from the capsule, ellipsoid and texture parameters inside the reversible adjoint, so the volume and its gradient are never materialised and gradients reach `rms_dn` and the capsule sizes.

### (d) Switching fidelity for the same scene and inspecting the planner's choices

```python
for level in ["draft", "fast", "balanced", "accurate"]:
    print(gs.Simulator(scene, pc, fidelity=level).plan.summary())
```
```text
level     interact            illum modes      λ bins  pol  sample grid                 est. fwd (B=64)
draft     thin (projection)   1 (centroid)     1       1    256² @162 nm (camera, s=1)   4 ms   ⚠ undersampled (λ/4NA=145 nm)
fast      thin + SOCS(8)      SOCS 8 kernels   1       1    320² @145 nm → MFT to camera 15 ms  optics fixed ✓; ⚠ thin: 1.2 µm > DOF 0.81 µm
balanced  rytov (slice-wise)  24 (ring quad.)  3       1    400² @116 nm, 8 slices      ≈1 s   Born & Rytov valid (φ_max 0.6 rad); Rytov preferred
accurate  multislice.bpm      48 (ring quad.)  3       1    640² @ 73 nm, 8 slices      ≈5 s   reversible adjoint; WOTF ✗ (φ_max > 0.3 rad)
```
```python
report = gs.compare(scene, pc, fidelities=["fast", "balanced", "accurate"], batch_size=8, seed=0,
                    metrics=["rel_l2", "ssim", "halo_width"])
print(report)       # shared replayed traces → differences are purely numerical/physical

sim = gs.Simulator(scene, pc, fidelity=gs.Fidelity(
    "balanced",
    illumination={"sampling": "stochastic", "modes": 8},          # unbiased, cheaper for training
    overrides={"cells.texture": {"interact": "thin"}},            # per-object override
    strict=True))                                                  # validity warnings → errors
sim.plan.stage("interact").validity                               # structured Violations
sim.plan.run(trace, until="filter.pupil")                         # step through the stages for debugging
```

### (e) Reusing the forward model in an inverse problem

```python
# (e1) Intensity diffraction tomography: recover an RI volume from LED-array images.
vol = gs.sample.VoxelField("ri", values=gs.Learnable(torch.zeros(96, 256, 256)),
                           spacing=(0.2 * um, 0.1 * um, 0.1 * um), quantity="dn")
leds = gs.light.LEDArray(positions=led_xyz, height=60e-3, wavelength=520 * nm)
idt = gs.modalities.Brightfield(leds, gs.optics.Objective(na=0.4, magnification=20),
                                gs.cameras.Ideal(pitch=6.5 * um, shape=(256, 256)))
fwd = gs.Simulator(gs.Scene(environment=gs.materials.Water(), objects=[vol]),
                   gs.acquire(idt, over="led"),
                   fidelity=gs.Fidelity("accurate", interact="multislice.bpm", grad="reversible"))
opt = torch.optim.Adam(fwd.parameters(), 1e-3)
for it in range(500):
    sel = torch.randperm(len(led_xyz))[:8]
    pred = fwd.expected(acquisition=sel).counts               # mini-batch over the acquisition axis
    loss = gs.losses.gaussian_nll(pred, measured[sel], var=pred + 4.0) + 1e-4 * gs.reg.tv3d(vol.values)
    opt.zero_grad(); loss.backward(); opt.step()

# (e2) Linear sub-plans export as operators with adjoints (e.g. widefield deconvolution via deepinv).
wf = gs.Simulator(gs.Scene(objects=[gs.sample.DensityField("dye", values=x0)]), widefield, fidelity="balanced")
A = wf.linear_operator(input="dye.values")                   # otf_strata stage: A, A.adjoint, A.norm()
physics = gs.interop.deepinv.as_physics(wf, input="dye.values")   # deepinv LinearPhysics + noise model

# (e3) Cramér–Rao bounds for PSF design (forward-mode over a handful of parameters)
crlb = gs.learn.crlb(train_sim, sites=["mol.position"], at=trace, camera_model="scmos")
```

---

## 10. DeepTrack2 integration strategy

**Position.** gradoscopy is a standalone engine. It has no dependency on DT; DT depends on it optionally. A thin adapter keeps DT's stochastic-program UX and moves the physics into batched, differentiable operators. DT's property DAG remains DT's asset, and gradoscopy never re-implements it.

**Adapter (`gradoscopy.interop.deeptrack`, shipped in M2):**
1. **DT-named Features.** `Fluorescence`, `Brightfield`, `Holography`, `Darkfield`, `ISCAT`, `Sphere`, `Ellipsoid`, `Ellipse`, `PointParticle`, `MieSphere`, `MieStratifiedSphere`, `Zernike`, `Poisson` and `Gaussian`, with the *same keyword arguments*.
   - Scatterer `get()` returns lightweight `EngineObject(Wrapper)` descriptors holding resolved properties instead of voxel masks. `particle.position`-style label branches keep working because they read property nodes.
   - Optics `get()` translates `NA`, `wavelength`, `magnification`, `resolution`, `refractive_index_medium`, `output_region` and `pupil`/`illumination` into gradoscopy descriptions.
   - `upscale` becomes a pinned `Fidelity(sampling={"oversample": upscale})`. `padding` is ignored with a one-time notice, because the planner derives it (DT's 10 px default is far too small for defocused coherent cases).
2. **Batched resolve.** `BatchedDataset(pipeline, batch_size)` resolves the DT graph B times on the CPU. That is scalar Python, microseconds per sample. It then pads the descriptors to N_max, stacks them into one `Trace`, and renders **once** on the GPU. `__distributed__=False` scatter lists become Populations. The result is `(B, C, H, W)` plus the DT label branches evaluated for each replicate.
3. **Learnability bridge.** `dt.gs.Learnable(...)` and `dt.gs.Uniform/Normal/LogNormal(...)` sampling-rule objects are recognised by the translator and become gradoscopy Params, so distribution parameters become learnable. Plain lambdas still work as non-differentiable `Random(callable)` sites, with a one-time notice. `nn.Parameter` properties pass through as Learnables, as they do in DT today.
4. **Conventions.**
   - pixel ↔ metre via `resolution / magnification`;
   - DT `(x, y)` and z-in-pixels ↔ world frame;
   - Zernike radians at DT's λ ↔ OPD in metres;
   - channel-last ↔ channel-first;
   - DT's `value` semantics become a Material or a Labeling, with the same warning.
5. **Caching.** The adapter calls `update()` internally, so it removes the "call update() after each optimiser step" pitfall of DTDV431 and DTEx252.

**Parity.** The `legacy_dt` preset reproduces DT 2.0.2 numerics: hard aperture, NA-truncated inter-slice propagator, bilinear sub-pixel placement, floored z, additive overlap, per-z OTF. Golden tests against DTGS121, DTEx203, DTEx205, DTEx252 and DTGS106 run with it. The default presets then **fix** those shortcuts, with a documented diff table: free-space inter-slice propagation, physical darkfield and iSCAT geometry, contrast signs (issue #445), and stable Mie recurrences.

**Migration phases.**
- (1) The adapter, opt-in (`dt.optical.use_engine("gradoscopy")`).
- (2) DT's `optical/` becomes thin wrappers when gradoscopy is installed, and `dt.pytorch.Dataset` gains batched resolve.
- (3) DT exposes `Learnable`/`Random` property kinds natively and deeplay consumes `SimDataset` directly.

DT keeps Sources, sequences, augmentations and deeplay glue.

---

## 11. Testing and validation strategy

The testing registry doubles as the fidelity dispatch table: each implementation registers its references, gradcheck shapes, convergence tolerances and benchmark shapes.

1. **Conventions and invariants (property tests, fp64, CPU).**
   - Fourier shift vs `roll`.
   - Propagate(a)∘Propagate(b) = Propagate(a+b).
   - Parseval and unitarity of ASM for propagating components.
   - Energy conservation through phase-only screens.
   - Adjointness ⟨Ax, y⟩ = ⟨x, Aᴴy⟩ for every linear operator.
   - Reciprocity for dipole and Mie.
   - Plane-wave `PlaneWaves` vs the sampled equivalent on-grid, where both must agree to 1e-12.
   - Zernike orthonormality.
2. **Analytic references.**
   - Airy pattern and in-focus OTF.
   - Gaussian-beam propagation.
   - Richards–Wolf focal fields and dipole basis images (Backer–Moerner).
   - TIRF depth d.
   - Collection efficiency (1−cosθ)/2.
   - iSCAT contrast period λ/(2n) versus z.
   - Phase-contrast halo present under Köhler and absent when coherent.
   - **Optical theorem.** Mie extinction from the analytic background × forward-scatter cross term vs Q_ext. This tests the PlaneWaves ⊕ sampled design end to end.
3. **External oracles** (test-only extras, never imported by the package):
   - miepython, scipy and PyMieDiff (GPL, isolated in `tests/oracles/`) for Mie coefficients, S1/S2 and layered spheres across dielectric, absorbing, large-x and core–shell cases;
   - psf-generator (MIT) and psfmodels for scalar/vectorial/Gibson–Lanni PSFs;
   - waveorder for WOTFs;
   - microsim for camera statistics;
   - wavesim for MBS (v2);
   - DT 2.0.2 for `legacy_dt` golden images.
4. **Cross-fidelity convergence harness.** For each `approximates` edge, it runs a regime sweep and records error tables:
   - thin vs BPM vs thickness;
   - Born and Rytov vs Mie vs (x, Δn);
   - BPM vs MBS (v2);
   - dipole vs Mie vs x;
   - scalar vs vectorial vs NA;
   - quadrature(K) vs grid Abbe;
   - SOCS(K) vs Abbe;
   - `otf_strata` vs `sparse_roi` (collapse theorem (a); with emitters on strata planes and Fourier-shift placement they must agree to 1e-6, otherwise the table records the strata-discretisation error vs Δz);
   - sum-pool vs MTF pixel integration.

   The tables are versioned artefacts and **set the validity thresholds** used by the planner.
5. **Gradients.**
   - `gradcheck` in complex128 on every `ops/` kernel and custom Function: full mode on tiny shapes, `fast_mode` in bulk.
   - Adjoint vs autograd (≤1e-5 fp32).
   - **Non-zero-gradient assertions for every Learnable and every learnable distribution parameter** of every example pipeline.
   - Estimator bias and variance checks: scaled-ST Poisson and relaxed presence vs finite differences of expectations.
6. **Statistics and determinism.**
   - KS and moment tests of samplers and camera noise (Poisson mean = variance; EMCCD variance ≈ 2G²Φ).
   - Seeded reproducibility.
   - `replay(trace)` bit-exactness.
   - Device parity (CPU, CUDA, and MPS best-effort in complex64).
7. **Plugin conformance suite** (§7.2), shipped in `gs.testing`.
8. **Learnability end-to-end.** Recover defocus, NA, Zernikes, radius, RI, size-distribution parameters and count rate from synthetic data within tolerance.
9. **Benchmarks.** A self-hosted runner (this 3090) records time and `max_memory_allocated` per tier on fixed shapes and fails on >20 % regressions. The §8.3 targets are asserted.

CI tiers: CPU unit tests on every PR (under 5 min), GPU nightly (convergence, benchmarks), weekly parity and oracle runs.

---

## 12. Roadmap

Milestones are ordered by **risk first** (contracts that are expensive to retrofit), then **value** (DT parity and the SMLM and tracking workhorses). Durations assume 1–2 full-time developers.

| Milestone | Deliverables | Exit criteria |
|---|---|---|
| **M0 Conventions and carriers** (3 wk) | `conventions.md`; Grid/FreqGrid/PupilGrid/PatchGrid; `Field` (PlaneWaves ⊕ sampled, rank-7 layout), `Intensity`, `EmitterSet`; precision policy; registry; `ops/fourier` (FFT/MFT/CZT, IEEE), `ops/propagation` (ASM with H_exp); test harness | property tests of §11.1 green; PlaneWaves vs sampled on-grid ≤1e-12; complex128 gradcheck infrastructure; import-linter rules enforced |
| **M1 Params and emitter path** (5 wk) | Param kinds, Trace, plates, handlers, estimators, ParamStore; planner skeleton (rule tables, bounds, explain); `emit.scalar`, pupil modifiers (soft aperture, Zernike OPD, pixel pupil, LayerStack), `sparse_roi`, `otf_strata`; full camera with `expected`/`sample`/`log_prob`; labels; `Widefield` recipe | ≥50k frames/s SMLM (64², 20 emitters); bead-stack fit recovers synthetic Zernikes to < λ/100 RMS; `otf_strata` ≡ `sparse_roi` to 1e-6 for on-strata emitters; non-zero-gradient suite green |
| **M2 Coherent sparse path + DT adapter** (6 wk) | `plane_waves`, Köhler quadrature/stochastic; `mie` (fp64, layered, ±z ports), `dipole`, `layered.fresnel`; `thin` with analytic projections; explicit-interference Detect; Brightfield/Darkfield/Holography/iSCAT recipes; DT adapter + `legacy_dt` | Mie vs miepython ≤1e-6; optical-theorem test; DTGS121/DTEx203/DTEx205/DTGS106 parity within tolerance; ≥2k frames/s BF tracking tier; iSCAT period λ/(2n) |
| **M3 Dense path** (7 wk) | band-limited voxeliser (custom autograd; optional Triton); `born`/`rytov` with form factors; `multislice.bpm` with procedural slices and reversible/segmented adjoints; PhaseContrast/DIC/DPC/FPM recipes; chunked mode reductions | BPM vs Mie and thin vs BPM convergence tables published; phase-contrast halo test; ODT recon (§8.3) < 4 GiB; 10k-sphere voxelisation < 2 GB |
| **M4 Vectorial and two-stage** (6 wk) | `emit.vectorial` (6-basis), `pupil.vectorial` (Jones, Fresnel), per-object field-dependent pupils; Transduce (linear, photophysics, bleaching); TIRF/SIM/LightSheet/Confocal recipes (effective PSF); spectral bins | vs psf-generator ≤1e-4; Stallinga fixed-dipole bias reproduced; SIM stack generator; DTEx252 end-to-end mask learning reproduced |
| **M5 Fast paths, time, polish** (5 wk) | SOCS/WOTF rewrites; stochastic Abbe; frames, kinematics, motion blur; `gs.compare`; learn toolbox (MMD, spectra, CRLB); CUDA-graph capture; `SimDataset`; docs and tutorials | SOCS→Abbe convergence; PC cells ≥1k frames/s; SyMBac-style automatic distribution matching demo; bounds guard and re-plan tested |
| **M6 = v1.0** (3 wk) | API freeze, deprecations, plugin entry points, conformance suite, DT migration phase 2 PR | all §8.3 targets; three external pilot users (tracking, SMLM, label-free) |
| **v2 track** | SSNP/MLB, C1 injection, MBS/LS (implicit adjoint), interface dipoles/SAF/iPSF, tensor ε, low-rank field dependence, meshes, deepinv export, Foldy–Lax | each lands with convergence tables and conformance tests |

---

## 13. Key decisions

1. **Units and precision.** *SI metres inside, complex64/float32 fields.*
   - fp64 is used for special-function recurrences (Mie, Bessel, Zernike radial polynomials) and for building phase arguments, which are carrier-subtracted and reduced mod 2π before casting.
   - The simulator runs under `autocast(enabled=False)`, and every MFT/CZT under IEEE fp32 matmul (TF32 measured at 4.4e-4).
   - complex32, fp16 and bf16 are banned from physics.
   - Unit handling is limited to `gs.units` constants at the boundary. Optimiser conditioning is handled by `Learnable(scale=…)`, not by changing internal units.
   - *Rationale:* float precision is relative, so metres cost nothing numerically; DT already uses metres; and one convention avoids the mixed-unit bugs seen in prysm.
2. **Central interchange representation.** *Two-port angular spectrum about a reference plane, carried by a `Field` that holds an analytic plane-wave background ⊕ a sampled scattered part, in either the spatial or angular domain.*
   - The objective acts in the angular (pupil) domain, where propagation, pupils and interfaces are all diagonal.
   - Dense solvers hand over exit-plane samples (one FFT away); analytic solvers (Mie, dipoles, emitters) *evaluate* their spectra on whatever grid the plan requests.
   - So "both" domains exist, but the *contract* is the angular spectrum on ports.
   - *Rationale:* it unifies every surveyed modality (brief 04 verdict), and it makes darkfield, phase contrast and iSCAT exact.
3. **Planner sophistication and timing.** *Rule-based, at plan time, cached.*
   - The planner does validity filtering, a small Dijkstra over sample lowerings, theorem-guarded rewrites, fusion, sampling derivation and memory-driven chunking.
   - It runs at `Simulator` construction and re-runs only on static-structure, fidelity, bounds or device/batch-bucket changes. A runtime bounds guard triggers re-planning when learnables drift.
   - There is no cost-model search beyond enumeration, and no per-batch planning.
   - *Rationale:* debuggability (`explain`, `run(until=)`) and static shapes for cuFFT plans, CUDA graphs and compile.
4. **Parameter/trace system.** *Roll our own (≈600 LOC) on top of `torch.distributions`.*
   - Param leaves, Trace (pytree), plates and `contextvars` handlers.
   - No Pyro: its global param store and handler stack are wrong for an embedded library.
   - No hard tensordict dependency: `Trace.to_tensordict()` is an optional export, to be revisited if compile integration makes it compelling.
5. **nn.Module vs functional core.** *Functional core (`ops/`, operator `__call__`) + a thin Module shell.*
   - `Simulator` and `ParamStore` are `nn.Module`s for `.to()`, `parameters()`, `state_dict()` and DDP.
   - Physical descriptions are frozen dataclasses, not Modules, to avoid mutable-state aliasing.
   - Operator implementations are stateless callables with static config.
6. **Sparse vs dense render paths.** *Both from day one, chosen per object group by the planner.*
   - Sparse = per-object analytic angular spectra (coherent, summed on the shared grid) or per-emitter ROIs (incoherent, scatter-added).
   - Dense = grid solvers.
   - Coherent contributions combine in the angular domain on the dense grid (C0 in v1, C1 injection in v2). Incoherent contributions combine in Intensity.
   - The crossover rule is N_e·R² vs N_z·N² log N. Measured: sparse is ~10× faster for 50 emitters at 256².
7. **Polarisation.** *The axis is always present; the plan chooses P ∈ {1, 2, 3}.*
   - Scalar is P=1 through the *same* kernels, so it is fast without being a separate code path.
   - The vectorial 3↔2 mapping lives in `pupil.vectorial`, and unpolarised light is 2 modes.
   - *Rationale:* retrofitting a polarisation axis is the most expensive change there is (brief 08 §2.13).
8. **Multi-wavelength grids.** *One spatial grid for all λ at sample and detector planes.*
   - On the dense path, per-λ pupils are discs of radius NA/λ on a λ-common frequency grid.
   - On the sparse path, a λ-common pupil grid in direction u feeds per-λ MFTs onto the common ROI.
   - Grids are sized for λ_min, bins are chosen by the planner, excitation and emission spectra are separate, and n(λ) comes from Materials.
9. **Time and dynamics.** *In v1, as frames only.*
   - Frames go on A, with kinematics (Brownian, drift, OU), 2-state photophysics (expected or stochastic) and bleaching.
   - Exposure motion blur is an incoherent sum over time sub-samples on M, so no new machinery is needed.
   - Deformations and biology are plugins.
10. **Dependency policy.** *Core = `torch` (≥ 2.9, for the `fp32_precision` matmul API) + `numpy`.*
    - Optional extras: `[triton]` (`triton-windows` on Windows), `[nufft]` (pytorch-finufft), `[deeptrack]`, `[deepinv]`, `[tensordict]`, `[io]` (zarr/h5py).
    - Test-only: scipy, miepython, psf-generator (MIT), waveorder (BSD) and microsim (BSD). PyMieDiff and psfmodels (GPL) are also used as oracles, but only in an isolated `tests/oracles` environment, never imported by package code.
    - No Cython or C++ in v1. License MIT.
11. *Additional:* a **fixed rank-7 axis layout** `[B, A, M, L, P, Y, X]`; **OPD-in-metres Zernikes (ANSI)**; **+z toward the detection objective**. All three are made on day one because they are the sources of silent sign and 2× bugs.

---

## 14. Top risks and open questions

**Risks (with mitigations)**

1. **The dual-part Field adds per-operator complexity.** Every operator must handle the analytic background.
   - *Mitigation:* base-class helpers (`map_background`, `evaluate_background_on(grid)`), a planner-inserted `to_sampled()` fallback with a precision warning, and conformance tests. The measured accuracy gains (§0) justify the cost.
2. **Planner opacity and creep.**
   - *Mitigation:* rule tables in one file, `explain()` and `run(until=)`, per-object pinning, and no search. A "planner changelog" test snapshots plans of the reference pipelines, so rule changes are reviewed as diffs.
3. **Dense learnable volumes exceed memory.** A user-supplied 64×256×512² learnable volume is 32 GiB with its gradient.
   - *Mitigation:* procedural samples by default, reversible adjoints, and batch-1 reconstructions. Where these are unavoidable, the planner errors with an actionable message rather than running out of memory.
4. **Mie on GPU is launch-bound and fp64-bound.** Measured: 6.9 ms for 640 particles (L=60).
   - *Mitigation:* batch all particles × λ into one call, CUDA-graph capture, 1-D θ synthesis, and CPU fp64 routing on MPS. The fallback for very large x is a float64 CPU path.
5. **Validity thresholds are literature rules of thumb** until convergence tables exist. *Mitigation:* M3 and M4 exit criteria require the tables, and the planner reads them.
6. **Partially coherent thick samples explode in cost** (modes × slices, e.g. 140 s per batch naively, brief 07).
   - *Mitigation:* stochastic Abbe (unbiased), SOCS and WOTF where the theorems apply, and explicit tier costs in `explain()`.
7. **Performance ceiling in eager PyTorch.** Complex ops are not fused by Inductor, and Triton is missing on Windows.
   - *Mitigation:* diagonal fusion at plan level, optional real-view Triton kernels behind pure-torch references, and CUDA graphs.
8. **DT semantic mismatches** (pixel conventions, contrast signs #445, `_ID` caching, `^`/list semantics).
   - *Mitigation:* the `legacy_dt` preset, golden tests, and an explicit diff table between the legacy preset and the defaults.
9. **Relaxed-count bias.** Gumbel presence gradients are biased. *Mitigation:* a documented temperature schedule, score-function alternatives, and bias tests.
10. **Scope pressure from two-port and interface physics.** *Mitigation:* v1 fills only the analytic reflection cases (Fresnel coverslip, dipole and Mie backscatter), while the contract is complete.

**Open questions**

1. Should the dense path also use direction-space (u) grids, i.e. per-λ-scaled frequency sampling, to make pupil pixel maps λ-invariant? The cost is resampling at the sample plane. The current answer is no (a λ-common FFT grid, with pixel pupils interpolated per λ), but measure the interpolation error for broadband pixel-pupil learning.
2. Should the analytic background generalise beyond plane waves, to Gaussian beams, spherical references (lensless holography with a diverging reference) and focused spots? Probably a small `AnalyticField` protocol with `evaluate_at`, `propagate` and `filter_by` methods, of which PlaneWaves is one instance.
3. How should the dense path represent field-dependent aberrations: low-rank product convolution, tiles, or ring-symmetric models? This decides the v2 `pupil.low_rank` API.
4. Is a fixed rank-7 layout too rigid for future axes, such as a separate time sub-sample axis or a detector-array axis for ISM? The current answer is to flatten into A or M with metadata, and to revisit if more than two such cases appear.
5. What should the bounds-guard policy be for learnable parameters with no declared bounds? Options are to warn, auto-widen and re-plan, or to require bounds at `Learnable` creation for any parameter that affects grid sizing. The leaning is to require them only for sizing-relevant sites.
6. Is `Trace` better as a TensorDict (compile support, memmap datasets) despite the dependency? Revisit at M5 with measurements.
7. How far should the DT adapter try to translate arbitrary lambdas? Static analysis of callables is fragile. The current answer is recognising only gradoscopy sampling-rule objects and treating everything else as opaque.
8. Can the explicit-interference detector extend cleanly to *partially* coherent references? For example, iSCAT with a scanned illumination has several modes, each with its own reference plane wave. The design supports it (a background per mode), but it needs a test case.
