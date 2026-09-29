# gradoscopy — Proposal B: a declarative physical scene compiled by a planner

*Architect B, 2026-09-24. Angle: users describe **what** (sample, light, objective, detector, fidelity policy). A planner decides **how** (solvers, representation conversions, coherence decomposition, grids, padding, chunking, fusion) from capabilities, validity checks and cost models, in the spirit of Mitsuba scenes plus integrators and of a database query optimizer.*

Evidence tags: **[0x]** cites research brief 0x. **[0x-M]** is a number measured on this RTX 3090 in brief 0x. **[B-M]** is measured for this proposal (`scratchpad/archB/crossover.py`, torch 2.14+cu130, RTX 3090). The `plan.explain()` listings in this document are illustrative mock-ups: their structure is the design, and their numbers are plausible estimates, not measurements.

---

## 0. Summary

gradoscopy is built as a small **optimizing compiler for optical experiments**. There are three objects:

1. **`Scene`** is a declarative, fidelity-free description of the physical experiment: sample, environment, illumination, objective and detection path, detector and acquisition. Its leaves are `Param`s, which can be constant, learnable, random with learnable distributions, derived, or a discrete choice.
2. **`Fidelity`** is a separate policy object. It names a preset and carries per-axis overrides, per-object pins, an accepted approximation severity, a required gradient quality and a budget.
3. **`Plan`** is what the **planner** produces from the Scene *structure*, the parameter *bounds*, the Fidelity, the requested outputs, the device and the set of parameters that need gradients. It is an ordered list of pure-torch stages, together with guards, a cost and memory estimate, a gradient-path audit and a full decision record (`plan.explain()`, `plan.why(...)`, `plan.diff(...)`).

```
  WHAT (user)                    RANDOMNESS                 HOW (planner, once)                 RUN (every batch)
 ┌────────────────────┐  draw  ┌──────────────┐  probe   ┌──────────────────────────┐  Plan  ┌───────────────────────┐
 │ Scene              │ ─────▶ │ Trace [B,…]  │ ───────▶ │ P0 normalize  P5 search  │ ─────▶ │ stage_1 … stage_k     │──▶ image
 │  Sample (+Env)     │        │ values, masks│  bounds  │ P1 bounds     P6 grids   │        │ (pure torch, static   │    labels
 │  Illumination      │        │ log_probs    │          │ P2 partition  P7 memory  │        │  shapes, guards, taps)│    taps
 │  Objective / Path  │        └──────────────┘          │ P3 graph      P8 rewrite │        └───────────────────────┘    meta
 │  Detector, Acq.    │                                  │ P4 require    P9 emit    │
 └────────────────────┘     Fidelity policy ────────────▶└──────────────────────────┘ ──▶ plan.explain()/why()/diff()
```

The distinctive bets are:

- **Logical vs physical operators.** About a dozen *logical physics operators* (Illuminate, Lower, Interact, Emit, SourceMap, Collect, Form, Combine, Reduce, Integrate, Detect, Label) each have many *implementations*. Choosing among them works like a Cascades/Volcano query optimizer. Implementations *require* and *deliver* properties (representation, domain, grid, polarization width, coherence structure, linearity/thinness). Converters act as *enforcers* that are inserted when properties do not match. Selection is cost-based and memoized over a graph that is small enough to search exhaustively (< 50 ms).
- **Shapes come from bounds, never from values.** At plan time the planner draws a *probe* from the Param graph and combines it with declared supports and learnable bounds. From these bounds it derives optical invariants: maximum phase delay, size parameter, defocus range, NA/n and so on. All discretization follows from those invariants. At run time, GPU-side **guards** enforce the assumptions. So a learnable parameter can move freely inside its bounds without changing a shape.
- **The interchange currency is a two-port angular spectrum** (`PortSpectrum`: T and R ports, medium basis, taken before the pupil). It comes in two storage forms: dense `GridSpectrum` and sparse, analytic `ObjectSpectra`. Sparse and dense producers therefore meet in the pupil. Coherent contributions add in k-space. Incoherent contributions add as intensities on the detector grid.
- **Mixed samples are a planner partition problem.** One example is Mie beads inside a voxelized cell. The planner assigns components to *interaction groups*, detects containment and proximity from bounds, and picks a *coupling policy* from a fidelity axis: C0 superpose, C1 inject into the march, C2 iterate, or C3 monolithic. It also handles the bead's host medium and cuts the bead out of the host volume, so the bead's material is not counted twice.
- **The planner audits gradients.** Every learnable parameter's path through the chosen stages is classified as smooth, biased or zero. If a path is zero the planner refuses to compile, for example a hard voxelizer on a learnable radius, and tells the user which knob fixes it.
- **Same description, many fidelities.** `sim.with_fidelity("accurate")` shares the learnable `ParamStore` and swaps only the plan. Fidelities follow a *consistency contract*: same radiometry, same frames, same conventions. There are also *split-fidelity* plans, where the forward pass uses a high tier and the gradients come from a cheap one.

---

## 1. Guiding principles

1. **Physics is described, never scheduled.** A `Scene` contains no numerical knob: no `upscale`, `padding`, `dz`, `n_max`, source-point count, oversampling or solver name. Every such choice is a plan decision. Users can pin any of them through `Fidelity`, never on scene objects. This fixes DeepTrack2's main ergonomic flaw [01, 09].
2. **One description, many plans.** Fidelity is applied at compile time. Every implementation of a logical operator approximates *the same physical operator* and obeys the same consistency contract, so lowering the fidelity changes only approximation error, not calibration.
3. **The planner is boring, deterministic, and explains itself.** It has a fixed, small vocabulary of logical operators; rule-based requirements; exhaustive cost-based enumeration with memoization; and no general-purpose IR and no second autograd graph. Every decision carries a machine-readable reason, and plans for canonical scenes are snapshot-tested like code.
4. **Plan once, run many.** Planning happens at construction, not per batch. Shapes derive from *bounds* (supports, truncations, learnable bounds), never from current values. Replanning is an explicit, logged event.
5. **The angular spectrum is the lingua franca; producers are polymorphic.** Mie, dipoles, emitters, thin screens, Born and multislice all deliver a `PortSpectrum`. Pupil modifiers, stops, rings, DIC shear and learned masks all act on it [03, 04, 05, 08].
6. **Objects are continuous and multi-view; voxels are a lowering.** Components expose capabilities (`occupancy`, `sdf`, `form_factor`, `project`, `mie_spec`, `dipole`, `sample_points`). The conversion graph tags every edge with *exactness* and *gradient quality* [06].
7. **Coherence is a reduction, not a type.** Source points, wavelengths, polarization states, dipole orientations, emitters and sub-exposure time are *mode axes*. `Reduce` has interchangeable estimators: exact, chunked, stochastic, SOCS/WOTF/OTF collapse, and Jones-on-Stokes [05].
8. **Randomness and learnability are data, not control flow.** `Param` leaves feed a pure `Trace`. Solvers never see a `Param`, only tensors. Every random site declares its gradient estimator, and the planner audits all gradient paths [07, 09].
9. **Memory is an architectural feature.** The planner chooses the inference-mode fast path, chunk sizes, checkpointing, reversible or segmented adjoints, and procedural (never materialized) volumes from a memory model and a budget. The largest memory item is the sample, not the field [07-M].
10. **Extension by declaration.** A third party registers data types plus capabilities, validity, cost, gradient quality and a reference test. The planner then composes the new piece with everything else. This keeps integration cost at *N + M + K* (primitives + solvers + modalities) instead of *N·M·K*.

---

## 2. Scope

Scope is chosen so that v1 covers the *minimum* fidelity of about 16 of the 18 application families surveyed in [08], DeepTrack2 parity [01], and the *ideal* fidelity of the two DeepTrack workhorses: coherent Mie imaging and incoherent fluorescence.

### v1 (must ship)

| Item | Justification (applications) |
|---|---|
| Scene/Param/Trace/planner core, `explain()`, presets `sketch…reference` + `deeptrack-2.0.2` parity preset | Everything else depends on it; parity is the DT2 acceptance test |
| Primitives: Sphere, LayeredSphere, Ellipsoid, Cylinder, Capsule (solid of revolution), Box, Gaussian, PointEmitters, SDF escape hatch, `FieldComponent` (voxel tensor / torch callable / neural field), GRF texture, Group hierarchy with priority compositing | Tracking, holography, bacteria (DTGS172), cells, QPI/ODT, neural fields [06, 08] |
| Soft band-limited voxelizer (closed-form blurred ball, SDF ramp; custom autograd) | A hard mask gives **dV/dR = 0** [06-M]; sim-to-real needs geometry gradients |
| Sources: plane wave, Köhler (disk, annulus, custom), LED array, reference beam, analytic excitation (TIRF evanescent, 2D/3D SIM beams, Gaussian/Bessel light sheet) | Brightfield, phase contrast, DIC, DPC, FPM, holography, iSCAT, TIRF, SIM, light sheet [05, 08] |
| Interaction: thin screen (scalar and Jones), Born/Rytov (form-factor and slice-wise), z-march engine with BPM and SSNP, Mie and layered Mie (fp64), Rayleigh/Mie-dressed dipole | Minimum fidelity of cell imaging, QPI, holography, iSCAT, darkfield; SSNP costs about 2× BPM for much better accuracy at high angles [04] |
| Coupling C0 (superpose) and C1 (inject analytic objects into the march) | Mixed samples (beads in cells) are DT's main unmodelled gap [04 §5] |
| Emission: sparse per-emitter ROI (MFT), dense strata OTF, Gaussian sprites, measured/spline PSF import, vectorial dipole basis (6 images) | SMLM, tracking, PSF engineering, system ID; vectorial costs about 6 pupil transforms [05, 08] |
| Pupil: soft aperture, exact defocus, Zernike (OPD in µm, ANSI plus Noll/DT converters), pixel pupil, Gibson–Lanni layer stack, phase ring, stops, DIC shear, knife edge, Jones elements; scalar, Jones and Richards–Wolf vectorial | Every modality lives in the pupil [03, 05] |
| Reduce: exact, chunked+checkpointed, stochastic Abbe, WOTF, shift-invariant OTF collapse | Partial coherence and fluorescence throughput [05] |
| Detectors: Ideal, Poisson–Gaussian, EMCCD, sCMOS maps, each with `expected`, `sample` (estimators) and `log_prob`; pixel MTF, integer and CZT resampling | System ID and MLE require likelihoods [05, 08] |
| Time: frame axis, reparameterized trajectories (Brownian, OU, drift), 2/3-state photophysics (expected and stochastic), motion-blur sub-steps | Tracking and SMLM are DT's core; DT2 has sequences [01, 08] |
| Spectra: few-bin spectral quadrature, separate excitation and emission spectra, dispersive materials | Brightfield with LEDs, fluorescence channels, colour DPC [05] |
| Learnable distributions (rsample, truncation), relaxed counts, score-function bookkeeping | Sim-to-real is the headline differentiator [08 §2.16] |
| DeepTrack2 adapter with batched resolve | Adoption path [01] |

### v2 (planned)

| Item | Justification |
|---|---|
| Modified Born series / Lippmann–Schwinger reference solver with implicit differentiation, validated against wavesim | Reference tier for ladder tests; thick cells and organoids [04] |
| MLB and WPM march modes; C2 coupled iteration; C3′ dipoles embedded in the volume; Foldy–Lax multi-sphere | Dense colloids, bead–cell back-coupling [04 §5] |
| Near-interface dipoles and SAF, full stratified vectorial iSCAT iPSF, R-port Redheffer composition | Quantitative iSCAT and mass photometry, TIRF SMLM [03, 08] |
| Emission through a scattering volume (E3a/E3b/E3c) | Light sheet in tissue, beads imaged through cells [04] |
| Space-variant PSFs (field-dependent Zernike maps, low-rank product convolution) | FD-DeepLoc/uiPSF-class calibration, large FOV [03] |
| SOCS kernels, T-matrix import, tensor-ε samples, meshes (winding number / polyhedron FT) | Speed for fixed optics; rods, birefringence, imported shapes |
| Accuracy-target planning (`Fidelity(target_error=…)`) driven by calibrated error models; auto cost calibration | Needs data from the v1 ladder tests first |
| Nonlinear source maps (2P, saturation, STED effective PSF) | Cheap once SourceMap exists [05, 08] |

### Later (research plugins)

Learned neural-operator rungs trained from reference outputs; DDA/TorchGDM interop; import of FDTD far fields and T-matrices; ray-traced eikonal screens; MCX backgrounds; FPM/whole-slide multi-GPU tiling; biological mechanics generators as plugins. Each depends on v2 data, or adds value only to a niche.

### Explicitly out of scope

- Native FDTD, FEM or DDA solvers. No data-generation application needs them online; they enter only as imported amplitudes or T-matrices [08].
- Coherent nonlinear optics (SHG, CARS, SRS) and Raman. Their emitters are phase-locked, which breaks the incoherent-mode model [05].
- FLIM and photon-timing statistics.
- Monte-Carlo diffuse transport beyond a phenomenological turbid-slab operator.
- Compound-lens ray tracing. Objectives are aplanatic plus pupil functions, and external OPD or Jones maps are accepted instead.
- Electron and X-ray microscopy.
- OCT.
- Instrument control.

---

## 3. Layered architecture and responsibilities

```
 L8 interop/     deeptrack adapter · deepinv export · torchvision tv_tensors · (napari viewer, later)
 L7 learn/       fit (MLE, L-BFGS, LM) · losses (MMD, sliced-W, spectra, histograms) · Fisher/CRLB · score surrogates
 L6 sim/         Simulator(nn.Module) = ParamStore + Sampler + Plan cache · datasets · writers
 L5 plan/        planner passes · Plan/Stage · Report (explain/why/diff) · guards · cost & memory models · calibration · plan cache
 L4 physics/     logical ops + implementations: sources/ interact/ emit/ collect/ form/ reduce/ detect/ labels/ (all pure functions)
 L3 repr/        lowered views (SphereList, EmitterList, OccupancyChannels, RIVolume, EmissionDensity, KSpectrum, ProjectedMaps) + converters
 L2 scene/       spec: components, materials, labeling, hierarchy, environment, illumination, objective & pupil modifiers,
                 detection path, detector, acquisition, presets (modality recipes)
 L1 params/      Param leaves, dist wrappers (truncation, relaxed), plates, SampleContext, Trace, handlers, estimators, probe bounds
 L0 core/        units, Grid/KGrid, Field/PortSpectrum/Irradiance types, axis specs, fft utils, special (fp64 Mie, Bessel,
                 Zernike), precision policy, RNG derivation, registry base, errors
 kernels/        optional Triton kernels, each with a pure-torch reference     testing/  conformance suite, ladders, audits
```

**Responsibilities.**

- **L0 core** owns conventions and the currency types. It knows nothing about scenes.
- **L1 params** owns *all* randomness and learnability: sites, plates, estimators, `Trace`, the unconstrained `ParamStore`, and probe-based bounds.
- **L2 scene** is the declarative spec: frozen `kw_only` dataclasses whose leaves are `Param`s. It contains no tensors of intermediate results and no numerical knobs. Presets such as `PhaseContrast` are *recipes* that expand into explicit components (`scene.expand()` shows them).
- **L3 repr** owns the *views* a solver can consume and the **converter registry**, with an exactness tag, a gradient-quality tag and a cost for each edge.
- **L4 physics** holds implementations of the logical operators as **pure functions** `(inputs: typed tensors, *, static config) → typed tensors`. They see no `Param`, no RNG except an explicit generator for stochastic reductions and noise, and no globals. Each registers a `Capabilities` record, `validity(bounds)`, `cost(problem)`, `configure(bounds, fidelity)` and gradient declarations.
- **L5 plan** is the *only* layer that knows both the scene (L2) and the implementations (L4), and it knows them only through registries. It produces an immutable `Plan`.
- **L6 sim** wires the pieces: `Simulator` is an `nn.Module` that owns learnables (`ParamStore`), the sampler, and a cache of plans keyed by gradient set and fidelity.
- **L7/L8** are consumers.

**Dependency rules** (enforced by an import-linter test in CI):

1. Imports go downward only. L2 never imports L4. L4 never imports L1, L2 or L5.
2. `kernels/` depends only on torch and is never required for correctness. Every kernel has a pure-torch twin.
3. Registries live in L0 and are populated by decorators in L3, L4 and L2 (presets) and by entry points.
4. Only L6 holds `nn.Parameter`s. Scene components are *not* modules (anti-pattern 13 in [09]).

**Package tree (v1):**

```
src/gradoscopy/
  __init__.py                  # gs.Scene, gs.Simulator, gs.Fidelity, gs.Learnable, … curated re-exports
  units.py                     # um=1.0, nm=1e-3, mm=1e3, m=1e6, s=1.0, ms=1e-3; px helpers resolved at plan time
  core/      grid.py  fields.py  axes.py  fft.py  precision.py  rng.py  registry.py  errors.py
  special/   mie.py  bessel.py  zernike.py  quadrature.py            # fp64, batched, autograd-safe
  params/    leaves.py  dist.py  plates.py  trace.py  sampler.py  handlers.py  estimators.py  bounds.py  store.py
  scene/     components.py  geometry/  materials.py  labeling.py  environment.py  illumination.py
             objective.py  pupil.py  detection.py  detector.py  acquisition.py  presets/  scene.py
  repr/      views.py  converters/{voxelize,project,formfactor,mie,dipole,emitters,labels}.py  graph.py
  physics/   ops.py  sources/  interact/{thin,born,march,mie,dipole,couple}.py  emit/  collect/  form/
             reduce/  combine.py  integrate.py  detect/  labels/
  plan/      planner.py  passes/{normalize,bounds,partition,graph,requirements,search,discretize,memory,
             rewrite,emit}.py  plan.py  report.py  guards.py  cost.py  calibrate.py  cache.py  fidelity.py
  sim/       simulator.py  data.py  writers.py
  learn/     fit.py  losses.py  fisher.py
  interop/   deeptrack/  deepinv.py  tv.py
  kernels/   phase_screen.py  voxelize.py  intensity_accum.py  (Triton + torch reference)
  testing/   conformance.py  ladders.py  references/  gradaudit.py
```

---

## 4. Core data model

### 4.1 Units, frames and conventions

- **Lengths are in micrometres internally; everything else is SI-consistent.** Time is in s. Wavenumbers are in rad/µm. Spatial frequencies are in cycles/µm. Irradiance is in photons·µm⁻²·s⁻¹. OPD is in µm, and phase = 2π·OPD/λ. `gs.units` supplies multipliers (`532*nm == 0.532`). The DT2 adapter converts from DT's SI metres and pixels at the boundary. The rationale is in §13.1.
- **Axis order.** Spatial axes are last and contiguous, `(…, Y, X)`: rows y, columns x, as `torch.fft.fft2` wants [02]. Fourier convention: the forward transform uses `e^{−i2πk·r}`; the inverse uses `e^{+i2πk·r}`.
- **World frame.** Right-handed. **+z is the propagation direction of light travelling toward the detection objective.** Transmitted illumination travels +z. Epi and reflected illumination travel −z. The T port collects +z-going light that has passed through the sample. The R port collects +z-going light scattered back by a sample illuminated from the objective side. The `Environment` places interfaces at explicit z (by default, the sample-side coverslip surface is at z = 0). The `nominal_focus` of the objective is a scene parameter. Helpers `gs.depth(d)` and `gs.rel_focus(dz)` save users from sign bookkeeping.
- **Zernike.** ANSI/OSA indexing with RMS normalization, OPD in µm. Converters from Noll and from DT's `(n, m)`-in-radians are provided [01, 05].
- **Radiometry.** Pupils are normalized so that the PSF integral equals the collection efficiency. Apodization is √cosθ for focusing and 1/√cosθ for collection, with a 1/k_z Jacobian [05]. Sources carry absolute photon flux. Cameras convert photons to electrons to ADU.

### 4.2 Grids and sampling bookkeeping

- `Grid(shape: tuple[int,…], spacing: tuple[float,…], origin: tuple[float,…], domain: "space"|"k")` is frozen and static (plain Python numbers) inside a plan. Static spacing is what keeps shapes and cuFFT plans fixed [07].
- `KGrid` is derived from a `Grid` (FFT dual). Pupils are evaluated on the **shared k-grid, in cycles/µm, for every λ**. Only the support radius NA/λ changes with λ, so there is never a per-λ grid (§13.8).
- `CameraSampling` holds pixel pitch, fill factor, ROI, binning, magnification and an offset. Magnification, pixel size and sub-pixel offset may be **tensors** (learnable), because the image→camera map uses a continuous-coordinate MFT/CZT, or integer sum-pooling when the planner can prove the ratio is fixed [05].
- `SamplingPlan` lives inside the `Plan` and records every discretization *with provenance*. Examples: "Δx = 162.5 nm = camera pitch / M (s = 1); Nyquist for intensity λ_min/(4·NA) = 166 nm satisfied"; "pad 32 px = defocus 6.6 µm × tan θ_max 0.75 + 2λ/NA"; "slices Δz = 0.20 µm ≤ λ_m/2". Sizes are rounded up to 2·3·5·7-smooth values, because a prime size costs up to 1.9× [07-M].

### 4.3 Field-like types (the currencies)

All are frozen dataclasses registered as pytrees. They are **not** tensor subclasses [07].

| Type | Tensor layout | Metadata | Role |
|---|---|---|---|
| `Field` | complex `[B, A…, M, Λ, P, Y, X]` | `Grid`, plane z, domain, medium n, basis (`scalar`/`xy`/`xyz`/`sp`), `AxisSpec`, mode weights `w[B?,M]`, spectral weights `wλ[Λ]` | Coherent fields on a plane |
| `PortSpectrum` | `{T: Spec, R: Spec?}` | reference plane z per port, medium basis | **Interchange between Interact/Emit and Collect** |
| ↳ `GridSpectrum` | complex `[B, A…, M, Λ, P, Ky, Kx]` or restricted to pupil-support samples `[…, K]` | `KGrid`, support mask | Dense producers (thin screen, march, Born) |
| ↳ `ObjectSpectra` | lazy: `amp(k⊥, λ) → [B, N, M?, Λ, P, …]` plus positions `[B,N,3]`, presence `[B,N]` | object kind, exact k-range | Sparse producers (Mie, dipole, emitter); evaluated by the consumer on its own k-samples or ROI |
| `SourceSet` | positions `[B,N,3]`, rates `[B,N,(T)]`, species `[B,N]`, dipole second moments `[B,N,6]`, presence, id | spectra per species | Incoherent emitters (logically: modes) |
| `EmissionDensity` | real `[B, S, Z, Y, X]` | Grid, species | Dense fluorophore density |
| `Irradiance` | real `[B, A…, Λ?, (Z), Y, X]` | Grid, units | Intensity on a grid |
| `Frames` | `[B, A…, C, H, W]` | camera metadata | Final output; plus taps |

`AxisSpec` names every non-spatial axis and says whether it is **summed** within an exposure (M sub-modes: `src`, `pol`, `dip`, `sub`, `emit`; the Λ bins) or an **acquisition** axis that produces separate frames (`frame`, `focus`, `phase`, `angle`, `led`, `channel`) [05 §1.4]. M is a flattened product of named sub-modes with a product weight. The planner reshapes it for chunking. P always exists, with width 1, 2 or 3, so going from scalar to vectorial never changes a signature.

### 4.4 Light

- **A source is a physical description of S(k_s, λ, pol, t) plus an optional spatial envelope.** It is *not* a sampled mode set. The planner chooses the sampling: on-grid Abbe, polar quadrature, Fermat spiral, stochastic, SOCS or WOTF.
  - Transmission: `PlaneWave(direction, pol)`, `Kohler(condenser: Disk(NA_c) | Annulus(NA_in, NA_out) | Mask(fn), spectrum)`, `LEDArray(positions, size, spectrum)`.
  - Coherent-light variants: `Laser`, `Speckle(diffuser)`.
  - Excitation: `Evanescent(θ, pol)`, `SIM(beams=pupil points, phases)`, `LightSheet(kind, NA_ls, scan)`, `FocusedBeam`.
  - `ReferenceBeam(amplitude, phase, tilt, path="transmitted"|"reflected")`.
- **Spectra.** `Mono(λ)`, `Band(center, fwhm, shape)`, `Tabulated(λ, S)`. Weights are *power per quadrature sample, including Δλ*. Excitation and emission spectra are separate objects, because fluorescence shifts wavelength [02].
- **Polarization.** A Jones vector, or a Stokes vector for partial or unpolarized light. The planner realizes unpolarized light either as two incoherent modes or as a Jones-matrix-on-Stokes contraction in `Reduce` [02 hcipy].
- **Coherence** is never a flag. Spatial coherence follows from the source extent. Temporal coherence follows from the spectrum. Emitter incoherence follows from the `SourceSet`.
- **Nuisances.** `IlluminationNuisance` (weak tilted or spherical waves: etalons, dust, ghosts) and `IlluminationEnvelope` (gradient, vignetting) cover DT's `IlluminationGradient` physically [01, 05].

### 4.5 Sample

- **Components separate three things:**
  - **Geometry**: a primitive kind plus a transform (translation, rotation as a quaternion or 6D representation, anisotropic scale).
  - **Material**: `n(λ)` complex; `Constant`, `Cauchy`, `Sellmeier`, `Tabulated`, `Drude`; later `Uniaxial`.
  - **Labeling**: `Label(species=Fluorophore(ex, em, brightness, photophysics, dipole), mode="volume"|"surface"|"points", density)`.

  One geometry therefore drives RI contrast *and* label density [06]. This enables multimodal scenes (virtual staining, [08]) and consistent ground truth.
- **`Population(shape, material=…, labels=…, position=…, count=…)`** is a struct-of-arrays over `[B, N_max]`, plus `presence`, `id`, `priority` and `parent`. `N_max` is a *plan* property derived from count bounds. Users never set it except as `max_count=` on a count distribution.
- **`Group(children, transform)`** builds hierarchies such as cell ⊃ {nucleus, droplets, texture, membrane label}. Children have higher priority. Material-like quantities use priority "over" compositing (linear in ε). Density-like quantities add. `compose="add"` gives the legacy DT behaviour [06].
- **`Environment`** holds the layer stack (immersion, coverslip, medium). It is read by both optics (Gibson–Lanni, Fresnel, TIRF) and sample (the default host n). This keeps "medium RI" in one place [06].
- **Views** (capabilities) are declared per kind and introspected by the planner: `bbox`, `occupancy(x, σ)`, `sdf`, `form_factor(k)`, `project(xy)`, `mie_spec`, `dipole(λ)`, `sample_points`. No component ever returns a voxel grid unless the user supplied one.

### 4.6 Optics, detection path, detector, acquisition

- **`Objective(NA, magnification, immersion, design_stack, pupil=[modifiers…], apodization="aplanatic")`.** Modifiers are ordered and fusable: `Aperture(soft_edge)`, `Defocus`, `Zernike`, `PixelPupil`, `LayerStack` (Gibson–Lanni), `PhaseRing`, `CentralStop`, `KnifeEdge`, `SpiralPhase`, `Wollaston`, `Jones`, `PhaseMask`, `AmplitudeMask`, `ModulePupil(nn.Module)`. The same objective serves excitation and emission in epi geometries.
- **`DetectionPath`**: tube lens / magnification, emission filters, splitters (dichroic, polarizing, biplane, multi-camera with per-channel affine), pinhole or detector array (confocal, ISM).
- **`Camera(pixel, qe(λ), noise=Ideal|PoissonGaussian|EMCCD|SCMOS(maps)|QCMOS, bits, offset, gain, full_well)`** with `expected`, `sample` and `log_prob` [05].
- **`Acquisition`** declares acquisition axes: `FocusStack(z)`, `Channels`, `SIMPhases`, `LEDSequence`, `Frames(n, dt, exposure, substeps="auto")`.

### 4.7 The Trace (resolved scene) and batching

`Trace` is a flat, immutable mapping from **site path** to tensor, for example `"sample.beads.radius" → [B, N_max]`. It also carries `batch_size`, per-site plate, masks (`presence`), optional `log_prob` for score sites, the seeds used, and a `spec_hash`. It is registered as a pytree, and `.to_tensordict()` is an optional interop (§13.4). Leading dimensions follow the plate: `batch` → `[1]` (broadcast), `sample` → `[B]`, `object` → `[B, N_max]`, `frame` → `[B, T]`. Optics parameters may be per-sample `[B]`. The planner then generates batched pupils, which is cheap because pupils are small, and disables cross-sample pupil hoisting.

---

## 5. Physics and the fidelity mechanism

### 5.1 The logical operator vocabulary

The planner reasons over a **logical physics graph (LPG)** built from a fixed vocabulary. The node types and their currencies:

| Logical op | Consumes → produces | Meaning |
|---|---|---|
| `Lower(group, view)` | components → view | Representation conversion (an enforcer) |
| `Illuminate(source)` | — → `Excitation` | A mode set: plane-wave batch, field on a plane, field evaluable at points, or irradiance |
| `Interact(group, excitation)` | views + Excitation → `PortSpectrum` | Elastic light–matter interaction |
| `SourceMap(link)` | Irradiance at emitters → rates | Excitation → emission (linear; v2 nonlinear) |
| `Emit(emitters)` | SourceSet/EmissionDensity → `PortSpectrum` (per-emitter modes) | Incoherent sources |
| `Collect(objective)` | PortSpectrum → pupil field | Apply the fused pupil (P = 1/2/3) |
| `Form(path)` | pupil field → image-plane field | FFT, MFT or CZT to a full frame or to ROIs |
| `Combine(reference)` | fields → field(s) | Coherent reference (holography, iSCAT) with an explicit interference term |
| `Reduce(axes, kind)` | fields → Irradiance | Σ_m w_m‖·‖² over summed axes (or a coherent sum) |
| `Integrate(camera)` | Irradiance → expected photons | Pixel MTF, supersample pooling, magnification map |
| `Detect(camera)` | expected → counts/ADU | Sampler with estimator, or `log_prob` |
| `Label(request)` | Trace + views → labels | Label renderers (masks, maps, tables) |

*Logically*, fluorescence is "every emitter is a coherent mode whose PortSpectrum is a dipole angular spectrum times a position phase ramp, reduced incoherently". Brightfield is "every source point is a mode, reduced incoherently". This is one model, and the fast paths (strata OTFs, per-emitter ROIs, SOCS, WOTF) are **implementations that require properties**, such as `shift_invariant`, `thin` or `weak`. They are not separate physics [05 §1.2].

### 5.2 Implementation catalogue (v1 unless marked)

| Logical op | Implementation | Class | Requires (properties) | Validity (initial rule) | Grad |
|---|---|---|---|---|---|
| Interact | `thin_screen` (scalar/Jones) | I1 | projectable view | d ≲ nλ/NA² (DOF) [04] | exact |
| | `born_formfactor` (no voxels) | I2 | `form_factor` | Δn·d < 0.35λ (Slaney) [04] | exact |
| | `born_slices`, `rytov_slices` | I2 | RIVolume | as above; Rytov: Δn/n_m ≲ few %, no zeros in u_in | custom adjoint |
| | `march.bpm` (tilt-corrected) | I4a | RIVolume or procedural slices | scattering angle, Δn·Δz; no backscatter | adjoint (reversible/segmented) |
| | `march.ssnp` | I4b | same | non-paraxial; R only approximate | adjoint |
| | `mie_pupil` (fp64 coefficients, 1D S1/S2 → pupil interpolation) | I3 | `mie_spec`, homogeneous host | isolation ≥ λ + 2a from neighbours and interfaces | autograd |
| | `dipole` (Mie-dressed α) | I3− | `dipole` | x ≲ 0.3 (pure), ≲ 1 (dressed) | exact |
| | `mie_inject` (C1 coupling into a march) | I3+ | march + `mie_spec` | local-plane-wave approximation at the bead | autograd |
| | `mbs`, `ls_krylov` (v2) | I5 | RIVolume in padded box | convergent; iteration count from ε | implicit |
| Emit | `sparse_roi_mft` (per-emitter pupil → ROI) | exact | SourceSet | none (exact for its pupil model) | autograd |
| | `strata_otf` (splat → per-z OTF) | ≈ | shift_invariant, EmissionDensity or splat | Δz_strata vs axial Nyquist | autograd |
| | `gaussian_sprite` | L0 | SourceSet | NA ≲ 0.7, freely rotating dipole [03] | exact |
| | `spline_psf` (measured PSF) | measured | SourceSet | within calibrated z-range | autograd |
| Collect | `scalar`, `jones` (P=2), `richards_wolf` (P=3, dipole basis) | P1 / P2 / P3 | — | scalar warns at NA/n > 0.7 (fixed dipoles, polarization optics) | exact |
| Form | `fft_full`, `pupil_samples_scatter_ifft`, `mft_roi`, `czt` | — | grid or ROI | Nyquist before squaring | exact |
| Reduce | `exact`, `chunked_ckpt`, `stochastic(k)`, `wotf`, `otf_collapse`, `jones_stokes`; `socs(K)` (v2) | — | WOTF: weak and thin; SOCS: thin, and no grad on source/pupil (SVD grads) [05] | exact / unbiased MC / linearized |
| Detect | `ideal`, `poisson_gaussian`, `emccd`, `scmos` × estimator ∈ {gaussian, scaled_ST, score, exact-no-grad} | D0–D2 | — | Gaussian estimator: λ ≳ 20 [05, 07] | per estimator |

### 5.3 Illumination and coherence

`Illuminate` produces an `Excitation` whose *form* is chosen to match the consumer:

- plane-wave batches in k for Abbe with thin, Born, Mie or march;
- a field on a plane for marches;
- `field_at(points)` for dipoles, Mie with non-plane-wave excitation, and fluorescence excitation;
- irradiance for SourceMap.

The planner sizes the mode set:

| Fidelity | Köhler source sampling |
|---|---|
| sketch | 1 mode (coherent approximation) |
| fast | 8-point polar quadrature, or `stochastic(4)` when learnable optics make SOCS unstable |
| balanced | 16–24 quadrature points |
| accurate | 48 points plus spectral bins |
| reference | on-grid Abbe |

SOCS (v2) and WOTF are Reduce implementations that require `thin` (and `weak`) sample properties. Those properties come from P2 partition and P1 bounds: a group is `thin` iff every component satisfies the thin-screen validity, and `weak` iff φ_max < 0.3–0.5 rad. The **gradient set changes the choice.** If the condenser shape or pupil is learnable, SOCS is rejected because SVD gradients at degenerate singular values are unstable, and stochastic or quadrature Abbe is chosen instead [05 §8.3].

### 5.4 Imaging system: how granular is the optical path?

The optical path is simulated at the granularity of **physically meaningful conjugate planes**, not lens surfaces:

1. the condenser pupil (source distribution);
2. the sample volume (interaction), with its layered environment;
3. the objective pupil, where *all* modifiers of all conjugate pupil planes are fused into one complex, possibly Jones-valued, pupil;
4. the image plane (camera sampling), after the detection-path branches.

Relays, 4f filters and SLMs are pupil modifiers. Field-dependent effects go through space-variant rendering (v2). Real lens prescriptions enter as imported OPD or Jones maps. This makes Collect a single fused elementwise multiply per λ and polarization (MONAI-style accumulate-then-apply [09]). It is also what lets learned optics apply to every modality for free.

### 5.5 Detection

`Integrate` does spectral weighting (source × filter × QE), applies the pixel MTF (box × diffusion), and maps to the camera by integer sum-pooling or continuous CZT. `Detect` exposes `expected()`, `sample(estimator)` and `log_prob(measured)`. The sample order follows the converged chain: QE → Poisson → dark/CIC → full well → EM Gamma → read → gain/offset → ADC [03, 05]. The planner picks the estimator from the gradient set:

- `scaled_ST` by default when anything upstream is learnable. It draws an exact Poisson sample and gives the gradient 1 + ε/(2√λ) [07-M].
- `exact-no-grad` in inference mode.
- `score` when the user requests unbiased gradients for noise-statistics learning.

Using `torch.poisson` directly is forbidden, because it silently returns zero gradients [07-M].

### 5.6 Mixed samples and coupling policies

The planner treats mixed samples as a **partitioning problem** (pass P2):

1. **Family candidates per component.** Each component gets candidate interaction families from its views and the registry. A `Sphere` can be `{mie, dipole (if x_max < 0.3), volumetric}`. An `Ellipsoid` can be `{volumetric, born_formfactor, thin}`. A `FieldComponent` can only be `{volumetric}`. `PointEmitters` go to `{emit}`.
2. **Relation graph from bounds.** Containment comes from the `Group` hierarchy (bead ⊂ cell). Proximity comes from probe statistics of pairwise gaps relative to λ + 2a. Interface distance comes from the `Environment`.
3. **Groups.** All volumetric components with overlapping z-ranges form one *march group*, meaning one march. Analytic components form *particle groups*, labelled `free` or `embedded(host)`.
4. **Coupling policy** from the fidelity axis `coupling` and validity:
   - **C0 superpose.** Particles radiate in their host medium, and fields add at the pupil. `free` particles use n_medium. For `embedded` particles, C0 still uses the *host material* index for m = n_p/n_host, and reports an info note that the host's aberration of the bead image is neglected.
   - **C1 inject** (v1). March the volume. At each particle's slice, read the local incident field (amplitude and phase at the centre, direction from the phase gradient), evaluate the Mie response in the host index, convert it to the angular spectrum *at that plane*, add it to the marched field, and continue. The cost is about one extra small FFT per injection plane [04 §5]. This captures shadowing and cell-induced aberration of the bead image.
   - **C2 iterate / C3′ dipoles-in-volume** (v2).
   - **C3 monolithic.** Voxelize the beads too (band-limited) and run the top volumetric solver. This is only valid when a ≫ Δx.
5. **Cut-out-and-fill.** When a component is assigned to an analytic solver but lies inside a volumetric host, the volumetric lowering renders the **host material in its place** (priority compositing with the child's material replaced by the parent's). The bead is therefore not counted twice as both a voxel and a Mie scatterer. This rule lives in the planner, so users never write it.

### 5.7 How fidelity is expressed

**A `Fidelity` is a policy over a lattice of axes.** It is not a scalar, and it is not a different set of classes.

| Axis | Levels (ordered) | sketch | fast | balanced | accurate | reference |
|---|---|---|---|---|---|---|
| geometry filter | hard (parity only) < soft σ=0.5h < soft σ=0.3h, 1.5× oversample < analytic | soft .5h | soft .5h | soft .3h | soft .3h ×1.5 | analytic/k |
| interaction (volumes) | I0 < I1 thin < I2 Born/Rytov < I4a BPM < I4b SSNP < I5 MBS | I1 | ≥I1 | ≥I4a | ≥I4b | I5 (v2) |
| interaction (particles) | projection < dipole < Mie < coupled | projection | Mie | Mie | Mie | coupled |
| coupling | C0 < C1 < C2 < C3 | C0 | C0 | C1 | C1 | C3 |
| coherence sampling | single < stochastic(k) < quadrature(n) < on-grid | single | quad 8 | quad 16–24 | quad 48 | on-grid |
| polarization | scalar-paraxial < scalar < Jones < vectorial | scalar | scalar | auto¹ | auto² | vectorial |
| spectral bins | 1 < n | 1 | 1 | 1–3 | 3–5 | converged |
| spatial variance | invariant < strata < low-rank < per-emitter | invariant | strata | strata | per-emitter | per-emitter |
| detector | D0 < D1 < D2 | D1 | D1 | D2 | D2 | D2 |
| time sub-steps | 1 < n | 1 | 1 | auto | auto | converged |
| max severity accepted | fail < warn < info < ok | warn (silent) | warn | warn-if-no-alt³ | info | ok |

¹ Jones if polarization optics are present, or at NA/n > 0.9. ² Vectorial if NA/n > 0.7, fixed dipoles are present, or polarization optics are present. ³ The **escalation rule**: at `balanced`, an implementation flagged *warn* is accepted only if no *info/ok* alternative costs less than 3× as much.

The overrides:

```python
gs.Fidelity("balanced",
    polarization="vectorial",                          # per-axis minimum (lattice join with the preset)
    coherence={"source_points": 24},                   # pin a discretization
    objects={"sample.cell": {"interaction": "ssnp"}},  # per-object pin (by site path)
    pin={"reduce[modes]": "chunked_ckpt"},             # pin a stage implementation
    max_severity="warn", grad="smooth",                # accepted approximation / required gradient quality
    budget=gs.Budget(memory="18GiB", batch_time="200ms"),
    backward=None,                                     # or a Fidelity: split-fidelity surrogate gradients (§5.11)
    strict=False)                                      # strict=True: any auto-fallback is an error
```

Two Fidelities combine with `|` (lattice join) and `&` (meet). `Fidelity("deeptrack-2.0.2")` pins the legacy implementations: hard masks, NA-truncated inter-slice propagator, bilinear placement, additive merge, and per-slice defocus PSFs. It exists for golden-image regression tests. It warns that geometry gradients are zero and refuses to compile if a geometry learnable is requested.

### 5.8 The planner algorithm

The planner is a pipeline of ten deterministic passes. Each is a plain function `(PlanState) → PlanState` that appends to `state.report`. Each can be run on its own in a debugger (`gs.plan.passes.search(state)`).

```
P0 normalize   expand presets into explicit components; resolve px units → µm using camera/magnification;
               canonical ordering of components; compute StructureSignature (kinds, N_max bounds, views, outputs)
P1 bounds      Bounds = ∩(declared supports/truncations, learnable bounds, probe quantiles)
                 probe: sampler.draw(n=1024, seed=PROBE_SEED) on CPU (scalar work), q=0.999, ×1.1 margin for
                 unbounded sites; derived invariants: φ_max, x_max, Δn_max, extent_z, z_rel_focus range,
                 NA_ill+NA_det, min pairwise gap, host Δn, photon range, λ range
P2 partition   families per component; relation graph; interaction groups; coupling policy candidates (§5.6)
P3 graph       build the LPG: Lower/Illuminate/Interact/Emit/SourceMap/Collect/Form/Combine/Reduce/Integrate/
               Detect/Label nodes; tag axes summed vs acquisition; attach required outputs (dead nodes pruned later)
P4 require     per-node Requirements = Fidelity lattice ⊔ physics triggers ⊔ output needs ⊔ gradient set:
                 e.g. NA/n > 0.7 & fixed dipoles & ≥accurate ⇒ polarization ≥ vectorial
                      learnable radius ⇒ geometry converter grad-quality ≥ smooth
                      output 'field' requested ⇒ Reduce must expose pre-reduction tap
P5 search      memoized, cost-based selection with enforcers (below)
P6 discretize  reconcile per-implementation `configure()` proposals into shared grids: sim Grid (max resolution
               demand, union of paddings), pupil sampling, slice set, source quadrature, n_max, N_θ, ROI size,
               smooth FFT sizes; re-cost with final sizes; if choice changes, iterate P5 once (bounded)
P7 memory      per stage: needs_grad (from gradient set), strategy ∈ {inference_mode, autograd, checkpoint,
               adjoint.reversible, adjoint.segmented, implicit}; chunk sizes for mode/λ/angle reductions from
               the memory model and budget (mem_get_info at plan time); if infeasible → re-enter P5 with
               the offending implementation's cost penalized, or fail with an explanation
P8 rewrite     semantics-preserving fusions: fuse pupil modifiers; merge homogeneous propagations
               (Propagate(a)∘Propagate(b) → Propagate(a+b)); fold pixel MTF into OTFs; hoist grad-free invariants
               (pupils/OTFs/Mie tables) to per-plan or per-batch caches; CSE (shared lowerings across branches);
               dead-output elimination
P9 emit        Plan(stages, guards, report, sampling_plan, key); frozen decisions serializable to JSON
```

**The search (P5)** follows the Cascades/Volcano model. The LPG is a DAG of fewer than about 25 nodes. Each logical op has fewer than about 8 implementations. Properties are tuples such as `(representation, domain, grid_class, P, coherence_form, flags)`.

```python
def best(node, required: Props) -> Choice:
    if (node.id, required) in memo: return memo[(node.id, required)]
    winner = None
    for impl in registry.impls(node.op):                          # sorted by name → deterministic
        caps = impl.capabilities
        if not caps.satisfies(node.requirements):                 # physics features, tier minimums, pins
            report.reject(node, impl, "capability", caps.missing(node.requirements)); continue
        v = impl.validity(node.bounds)                            # list[Violation(severity, msg, est_error)]
        if worst(v) < policy.max_severity(node) and not pinned(node, impl):
            report.reject(node, impl, "validity", v); continue
        g = impl.grad_quality(node.learnables_reaching)           # per input: smooth|biased|zero|none
        if any(q < policy.grad for q in g.values()):
            report.reject(node, impl, "gradient", g); continue
        cfg = impl.configure(node.bounds, policy)                 # static discretization proposal
        cost = impl.cost(node.problem(cfg), device_model)         # Cost(time, bytes_fwd, bytes_bwd, launches)
        for child, need in impl.requires(node, cfg):              # required input properties
            sub = best(child, need)                               # recurse
            cost += sub.cost + enforcer_cost(sub.delivers, need)  # insert converters (∞ if impossible)
        cost += enforcer_cost(impl.delivers(cfg), required)
        cand = Choice(impl, cfg, cost, v, g)
        winner = min(winner, cand, key=lambda c: (c.cost.time_est, -c.impl.tier, c.impl.name))
    memo[(node.id, required)] = winner or report.fail(node)       # fail lists every reject reason
    return winner
```

- **Enforcers** are registered converters with costs. Examples: `components→OccupancyChannels`, `SphereList→ObjectSpectra`, `Field(space)→GridSpectrum` (FFT), `ObjectSpectra→GridSpectrum` (evaluate on pupil samples), `scalar→Jones` lift, `GridSpectrum→ROI` (MFT).
- **Joint constraints** between nodes are properties. For example, `wotf` requires its child Interact to deliver `flags ⊇ {thin, weak, linear_in_sample}`. `sparse_roi_mft` requires the Emit input as a `SourceSet`, not a density.
- **Time estimate.** `time_est = max(flops/F, bytes/BW) + launches·t_launch` per stage, using device constants (§5.10).
- **Ties** go to the higher tier, then to the alphabetical name. Planning is therefore deterministic and snapshot-testable.
- **Runtime.** The search is memoized, so in practice it evaluates about 10³ (node, impl) pairs, far fewer than the raw product of choices. That is well under 50 ms of Python, run once.

**When planning happens.** Eagerly, at `Simulator(...)` construction, so errors surface before training starts. Plans are cached under `key = hash(StructureSignature, quantized Bounds, Fidelity, outputs, device class, grad set, precision policy, gradoscopy version)`.

**Replanning triggers** (always logged, and reported in `sim.plan_history`):

- the scene structure is edited;
- the fidelity or the requested outputs change;
- the device changes;
- the gradient set changes (`sim.requires_grad_(paths)`, or the `train()`/`eval()` plan pair);
- a guard fires under policy `replan`;
- an explicit `sim.replan()`.

Learnable values that stay inside their bounds **never** trigger replanning.

**Guards.** Every `configure()` declares which bounds it consumed, for example "n_max = 37 assumes radius ≤ 1.5 µm" or "pad 32 px assumes |z_rel| ≤ 6.6 µm". P9 turns these into guard kernels:

- a GPU-side max/min reduction over the relevant Trace sites, producing a flag tensor;
- policies `clamp` (default for learnables), `warn`, `replan`, `error`;
- reporting deferred through a non-blocking copy that is checked every K batches, so there is **no host sync on the hot path**. `strict=True` syncs every batch.

Truncated distributions never trigger guards. Guards mostly catch learnables and `Derived` sites without declared bounds.

### 5.9 Validity checks and error models

Each implementation registers `validity(bounds) → [Violation(severity, message, est_error, provenance)]`. Initial thresholds come from the literature and are later **recalibrated by the ladder tests** (§11). The v1 rules include:

| Check | Rule (initial) | Source |
|---|---|---|
| thin screen | extent_z ≤ n·λ/NA² | [04 §2.2] |
| Born | 2π·Δn·d/λ ≲ 0.35·2π (Slaney); WOTF φ ≲ 0.3–0.5 rad | [04], [05] |
| Rytov | Δn/n_m ≲ 0.03 and no excitation nulls | [04] |
| BPM | max(NA_ill, NA_scatter)·Δn; backscatter flagged as info | [04] |
| Mie | isolation gap ≥ λ + 2a; host homogeneous; interface distance ≥ λ | [04 §2.8] |
| dipole | x ≤ 0.3 (bare α), ≤ 1 (Mie-dressed) | [04] |
| scalar pupil | NA/n > 0.7 warn; fixed dipole with Gaussian PSF: bias up to 40 nm (error) | [03] Stallinga |
| sampling | Δx ≤ λ/(4NA) for intensity; field squared only after 2× band padding | [05 §5] |
| padding | pad ≥ z_max·tanθ + 2λ/NA | [05] |
| spectral | N_λ ≥ 4·ΔOPD_max/l_c | [05] |
| iSCAT numerics | contrast < 1e-3 ⇒ explicit interference term required (never subtract \|E_r\|²) | [08] |
| Gaussian Poisson estimator | min expected photons ≥ 20 | [05, 07] |

An `error_model(bounds) → ErrorEstimate(metric, value, provenance)` is optional in v1. It is filled from the ladder tables, so reports say, for example, "BPM vs SSNP: est. 3.1 % rel-L2 at φ_max = 3 rad (calibrated, ladder 2026-11)". v2 accuracy-target planning chooses the cheapest plan whose summed error estimate is at most `target_error`.

### 5.10 Cost and memory model

Each implementation provides `cost(problem) → Cost(flops, bytes_fwd, bytes_saved_for_bwd, launches, peak_transient)`. Device constants come from `gs.calibrate()`, a 20-second microbenchmark in the style of `FFTW_MEASURE`. It measures FFT passes by size class, elementwise bandwidth, IEEE-fp32 matmul throughput and launch overhead, and caches the result per device. The shipped defaults are derived from the measured 3090 numbers:

- elementwise complex multiply 0.525 ms for 64×512² → ~770 GB/s effective [07-M];
- c64 fft2 0.705 ms for 64×512² → ~2.5 memory passes [07-M];
- launch ≈ 8 µs per tiny op [07-M].

The memory model counts saved activations per strategy using the measured multipliers:

- naive autograd 2.94 fields/slice;
- √S checkpoint 1.27;
- reversible adjoint 0.78 [07-M];
- chunked mode sums at O(chunk) [07-M].

`plan.profile(trace)` measures actual per-stage time and memory, prints them next to the predictions, and can feed `gs.calibrate(update_from=profile)`. A wrong cost model only makes a plan slower. It can never make it wrong, because correctness is protected by validity and not by cost.

### 5.11 How lower fidelities relate to higher ones

1. **Consistency contract** (tested): at every tier, the same scene yields the same mean photon count to within the model's declared error, the same coordinate frame and sub-pixel origin, the same sign conventions (contrast signs; compare DT issue #445 [01]) and the same label geometry. Only approximation error differs.
2. **Ladder relation.** Each implementation names its `reference` (the next-higher implementation of the same logical op) and the regime where the two must agree. This generates the cross-fidelity tests (§11) and the error models.
3. **Split-fidelity plans.** `Fidelity("accurate", backward="fast")` compiles two sub-plans that share the Trace and computes `y = y_hi.detach() + y_lo − y_lo.detach()` [07 §6.4]. The forward values are accurate and the gradients are cheap. The planner checks that both sub-plans expose the same learnables with smooth gradients.
4. **`gs.compare_fidelity(sim, "fast", "accurate", n=32, metrics=[...])`** renders the same traces (`replay`) under both plans and reports image metrics, label agreement and cost ratios.

### 5.12 Debuggability

Plans are meant to be read.

- `plan.explain(level="summary"|"decisions"|"full")` shows stages, chosen implementations, grids, estimated cost and memory, warnings, and the gradient-path table.
- `plan.why("interact[cell]")` gives the full decision trace for one node: candidates, reject reasons, costs.
- `plan.diff(other)` compares two plans stage by stage.
- `plan.graph("mermaid")` renders the graph.
- `plan.run(trace, until="collect")` and `plan.run(trace, taps=["interact[cell].T"])` run part of a plan or tap an intermediate.
- `plan.profile(trace)` reports predicted versus measured cost.
- `plan.to_json()` / `gs.Plan.from_json()` give **frozen plans**. A frozen plan pins every decision, so a dataset can store the exact recipe and be regenerated on another machine or version (with a warning if an implementation changed).
- All decisions are logged under `logging.getLogger("gradoscopy.plan")`.

---

## 6. Parameters, randomness and learnability

**Leaves** (L1). Every scene attribute is a `Param`. Python numbers and tensors are auto-wrapped as `Const`.

| Leaf | Meaning | Storage | Gradient |
|---|---|---|---|
| `Const(v)` | fixed | buffer | none |
| `Learnable(init, constraint=, bounds=, scale=)` | trainable | `nn.Parameter` in unconstrained space inside `ParamStore` (`transform_to(constraint)`), normalized by `scale` so optimizers see O(1) values | pathwise |
| `Random(dist, plate=, estimator="auto")` | random; the dist's arguments may be any `Param` (learnable distributions) | — | per estimator |
| `Derived(fn, **deps)` or Param arithmetic (`radius * 2`) | deterministic function of other sites | — | autograd through `fn` |
| `Choice(options, probs=, estimator="relaxed"\|"score"\|"enumerate")` | discrete choice (kind, material, preset) | — | relaxed/score |
| `ModuleParam(module)` | output of an `nn.Module` (neural field, pupil net, texture generator) | the module is registered in `ParamStore` | autograd |
| `Data(source, key)` | values from a dataset (DT `Sources`, real annotations) | — | none |

**Distributions.** `gs.dist` wraps `torch.distributions` so that arguments may be `Param`s. It adds `.truncate(lo, hi)` (inverse-CDF, reparameterized for Normal, LogNormal, Uniform, Exponential), `from_median`/`from_mean_std` constructors, and relaxed variants. **Truncation is how random sites feed shape-affecting quantities.** For example, the planner reports "radius has unbounded support; n_max would be driven by q.999 = 9.6 µm; add `.truncate()`" (§9a).

**Plates.** `batch` draws once per batch, `sample` once per image, `object` once per slot, and `frame` once per time step. Defaults: scene-level fields use `sample`; `Population` fields use `object`. Kornia's `same_on_batch` is the special case `plate="batch"` [09].

**Gradient estimators, chosen per site** (`estimator="auto"` resolves as follows and the choice is shown in `explain`):

- **pathwise** (`rsample`, including implicit reparameterization for Gamma/Beta) for continuous sites;
- **relaxed(τ, hard=True)**: Concrete / Gumbel-softmax with straight-through for `Choice` and presence;
- **score**: REINFORCE with a moving-average baseline. The Trace stores `log_prob`, and `batch.score_surrogate(loss)` adds `Σ log_prob·stopgrad(loss − b)`;
- **none**.

**Discrete counts.** A count becomes `N_max` slots with `presence ~ RelaxedBernoulli(p = E[N]/N_max)`. For Poisson(λ) with λ ≪ N_max, Binomial(N_max, λ/N_max) is a close approximation, and the presence relaxation gives a gradient to λ. The alternative is an exact truncated Poisson with the score estimator. Presence multiplies occupancy, Mie amplitude and photon rate, so every downstream stage respects it.

**Non-overlap.** The default is *no rejection*: overlaps are resolved by compositing rules. The alternatives are `NonOverlapping(method="reject")`, which is flagged in the gradient audit as biased (the acceptance gradient is dropped), and `method="repel", steps=k`, a differentiable unrolled repulsion relaxation [06].

**Detector noise** uses the estimators in §5.5. EMCCD gain uses Gamma `rsample`. Quantization uses straight-through. Saturation uses a soft clamp during learning.

**The gradient audit** (P4/P5, and `testing.gradaudit` empirically). For each `Learnable` and each learnable distribution parameter, the planner walks the chosen stages and composes gradient qualities:

- smooth ⊗ smooth = smooth;
- anything ⊗ zero = zero;
- biased dominates smooth.

A **zero path is a compile error** that names the offending stage and the fix. Example: "`sample.beads.radius` → `occupancy.hard` (preset deeptrack-2.0.2) gives zero gradient; choose geometry ≥ soft or pin `interact[beads]=mie_pupil`". A biased path is a warning. The empirical audit in CI asserts non-zero gradients for every leaf in canonical scenes, which guards against silent zeros such as `torch.poisson` and `torch.normal` [07-M].

**Batching semantics.** Everything is batch-first. Ragged object counts use padding plus presence. Per-sample optics are `[B]` tensors. Nothing loops in Python over samples or objects. The sampler loops only over *site kinds* in topological order, vectorized over `[B, N_max]`.

**Seeding.** `sim.sample(B, seed)` uses one `torch.Generator` per `(seed, site_path)` derived by hashing, so adding a site does not shift the others' streams. An opt-in **counter-based mode** makes image *i* independent of batch composition. Monte-Carlo reductions (stochastic Abbe, speckle) get a separate `seed_grad` stream (Mitsuba [09]). Global RNGs are never touched. Exact reproduction comes from `replay(trace)`, not from RNG luck.

**Handlers** are local to a simulator (an instance-level stack, never a global one): `sim.condition({path: value})`, `sim.replay(trace)`, `sim.mean()` (distribution means, for a deterministic debug image), `sim.detach(paths)` and `sim.block(paths)`.

**Ground truth.**

1. *Trace values* are free regression targets: positions, radii, RI, counts, defocus, Zernike coefficients.
2. *Label renderers* are lowered by the planner with the same converters as the physics, on the camera grid or any requested grid: instance and semantic masks, soft masks, distance maps, centroid heat maps, 3D RI/density volumes, per-object tables with `in_fov` and `visible` flags.
3. *Taps* expose stage intermediates: noise-free expected image, complex image field, per-species images, per-object images, PSF/OTF, excitation irradiance.

Labels are typed as `tv_tensors` (`Mask`, `KeyPoints`, `BoundingBoxes`) so torchvision/Kornia augmentations keep them aligned [09].

---

## 7. Composition and extension

**Registries** (one per extension point, populated by decorators and by `entry_points(group="gradoscopy.plugins")`, loaded lazily; duplicate names are an error unless `override=True`; each plugin declares the API version it targets):

`component`, `view` (capability protocols), `converter`, `impl` (per logical op), `pupil_modifier`, `source`, `detector`, `estimator`, `label`, `preset` (modality recipes and fidelity presets), `rewrite`, `validity/error_model`.

**Adding a primitive.** Declare its views. Every solver and label renderer that consumes those views then works with it automatically.

```python
@gs.register.component("bacillus")
@dataclass(frozen=True, kw_only=True)
class Bacillus(gs.Primitive):                 # capsule with bend, as in DTGS172 [01]
    length: gs.Param; radius: gs.Param; bend: gs.Param = 0.0
    def bbox(self, t: gs.Trace) -> Tensor: ...                         # [B,N,2,3]
    def sdf(self, x: Tensor, t: gs.Trace) -> Tensor: ...               # enables soft occupancy (SDF ramp)
    def form_factor(self, k: Tensor, t: gs.Trace) -> Tensor: ...       # 1D quadrature over the profile [06]
    # no mie_spec → planner never offers Mie for it; project() is derived from form_factor(k_z=0)
```

**Adding a solver.** Register an implementation with its capabilities, validity, cost and gradient quality. It is then available to every modality whose graph contains `Interact`.

```python
@gs.register.impl("interact", "wpm")
class WavePropagationMethod(gs.Impl):
    capabilities = gs.Capabilities(
        requires={"input": gs.props(view=gs.repr.OccupancyChannels, domain="space")},
        delivers=gs.props(currency=gs.GridSpectrum, ports={"T"}, flags={"linear_in_incident"}),
        cls="I4", physics={"multiple_forward_scattering", "wide_angle_steps"},
        polarization={1}, reference="interact.mbs")
    def validity(self, b: gs.Bounds):
        return [gs.Violation("info", "backscatter ignored")] + \
               ([gs.Violation("warn", f"{b.n_materials} index levels → {b.n_materials}× FFT")] if b.n_materials > 6 else [])
    def configure(self, b, fid):  return dict(dz=fid.slice_step(b), levels=b.n_materials)
    def cost(self, p, dev):       return dev.fft(p.nxy, batch=p.batch * p.levels) * p.n_slices
    def grad_quality(self, reaching): return {k: "smooth" for k in reaching}
    grad_strategy = ("adjoint.segmented", "checkpoint")                 # which memory strategies it supports
    def __call__(self, exc: gs.Field, occ, *, dz, levels) -> gs.GridSpectrum: ...  # pure torch
```

**Adding a modality.** In most cases this is a *preset*: a recipe over existing components. It needs no new physics.

```python
@gs.register.preset("hoffman")
def HoffmanModulationContrast(objective, slit_offset_na, modulator_transmission=(0.01, 0.15, 1.0), **kw):
    return gs.Optics(illumination=gs.Kohler(condenser=gs.Slit(offset=slit_offset_na, width=0.05)),
                     objective=objective.with_modifiers(gs.pupil.GradedAmplitude(modulator_transmission)), **kw)
```

**Adding a converter.** `@gs.register.converter(src=Gaussian, dst=OccupancyChannels, exactness="analytic", grad="smooth", cost=...)`.

**Conformance suite.** `gs.testing.conformance(impl)` runs automatically for every registered implementation:

- `gradcheck` in complex128 on tiny shapes;
- a consistency check against its declared `reference` in its validity regime;
- cost-model sanity (predicted vs measured within 3×);
- CPU/CUDA parity;
- the non-zero-gradient audit;
- "pure function" checks (no global RNG use, deterministic under a fixed generator).

Third-party plugins get the same bar as core code.

**Operator composition for power users.** The functional core (`gs.ops`: `propagate`, `pupil`, `mft`, `march`, `mie_spectrum`, `reduce`, …) and a typed operator algebra (`FieldOp >> FieldOp`, `stack`, `IncoherentSum(over=…)`) are public. Plans can also be *edited*:

- `plan.insert_after("form", LearnedResidual())` adds a physics-plus-learned-residual correction, the neural-holography pattern [02];
- `Fidelity(pin={"collect": my_callable})` replaces a stage.

Edited plans are marked `custom` in `explain()`.

**Why this is more than the sum of its parts.** A new primitive with `sdf` and `form_factor` works immediately with thin screen, Born, BPM, SSNP, fluorescence (as a label target), and masks and distance maps (as labels). A new march solver works immediately for phase contrast, DIC, holography, QPI and C1 bead injection. A new pupil modifier works for every modality. A new detector works with every image. The planner is the glue that makes these combinations automatic *and* explicit.

---

## 8. Performance and memory strategy

**Measured anchors** (RTX 3090, torch 2.14+cu130):

| Quantity | Value | Consequence in the planner |
|---|---|---|
| c64 fft2, 64×512² | 0.705 ms; c128 7.5× slower; c32 slower and NaN in multislice [07-M] | complex64 fixed; fp64 only for special functions and phase construction |
| elementwise c64 multiply, 64×512² | 0.525 ms [07-M] | Elementwise costs about the same as an FFT → fusion is the optimization target; cost model counts bytes |
| per tiny op | ~8 µs; a CUDA graph gave 2.9× on a 32-slice 256² forward [07-M] | Static-shape plans are captured as CUDA graphs in inference mode for small fields |
| prime FFT sizes | up to 1.9× slower [07-M] | P6 rounds up to 2·3·5·7-smooth sizes |
| multislice activations | naive 2.94, checkpoint 1.27, reversible adjoint 0.78 fields/slice; reversible 100 ms vs 163 ms naive [07-M] | P7 prefers the reversible adjoint for lossless samples, segmented for absorbing ones |
| chunked mode sums | M=64: 3.74 GB → 0.51 GB (chunk 8 + checkpoint), 65 → 98 ms [07-M] | P7 picks chunk sizes from budget and `mem_get_info` |
| voxelization, 256²×64 grid, P=24 patch | 1000 spheres 37 ms / 1.8 GB; 10k spheres 0.99 s / 17.9 GB (unfused autograd) [06-M] | Custom autograd voxelizer early (kernel twin in Triton); planner charges its memory |
| Mie D_n in complex64 | 1–5 % error for real m [07-M] | Coefficients fp64 always; field synthesis via 1D S1/S2 + pupil interpolation |
| **sparse vs dense emitters** (B=32, 256², pupil 64², ROI 32², dense = 32 z-strata) | sparse: 0.7 µs/emitter fwd, 2.2 µs f+b, ~190 KB/emitter under autograd; dense: flat 10.5 ms fwd / 17 ms f+b, 1.57 GB. Crossover ≈ 300–500 emitters per image [B-M] | Emit implementation chosen by the cost model, not a user flag. Sparse is also exact in z; strata interpolate between planes |

Raw rows for the last line [B-M] (fwd ms / f+b ms / peak MB under f+b):

| N / image | sparse | dense |
|---|---|---|
| 16 | 0.60 / 1.81 / 129 | 10.5 / 16.9 / 1571 |
| 64 | 1.72 / 5.10 / 417 | 10.6 / 17.1 / 1571 |
| 256 | 5.99 / 18.2 / 1569 | 10.5 / 16.9 / 1572 |
| 1024 | 23.2 / 71.0 / 6178 | 10.6 / 17.1 / 1578 |

**Planner-owned strategies:**

1. **Inference-mode fast path.** When the gradient set is empty (the normal data-generation case), the plan runs under `torch.inference_mode()` with in-place variants (`mul_`, `out=`), hoisted invariants (pupils, OTFs, Mie tables for fixed optics), and optional CUDA-graph capture.
2. **Procedural volumes.** For parametric samples in a march with gradients, slices are generated on the fly and the adjoint regenerates them in backward with an in-loop VJP into the geometry parameters [07 §4.2]. This removes the 16 GiB volume plus 16 GiB gradient wall of a 64×256×512² float32 volume. The planner selects this whenever the volume would exceed about 25 % of the budget.
3. **Exact chunked reductions** over modes, λ, angles and emitters, with `checkpoint(use_reentrant=False, preserve_rng_state=True)`.
4. **Pupil-support evaluation for sparse coherent objects.** Mie and dipole spectra are evaluated only at k-samples inside the NA disk, for example 6.9 k-samples for 192² at NA 0.8 and 532 nm. They are scatter-added into the k-grid and inverse-FFTed once per image. Evaluating Mie per pixel would need ~75 GB for 100 particles × batch 32 [08].
5. **Mode layout.** Mode axes are batch-like and the FFTs are batched over `[B·M·Λ·P]`. The planner caps the product by the memory model.
6. **Precision policy.** `PrecisionPolicy(real=f32, complex=c64, special=f64, phase_build=f64)`. Phase arguments are carrier-subtracted and reduced mod 2π in fp64, then cast: 1.6e-3 → 3.7e-7 relative error [07-M]. Internals run under `autocast(enabled=False)` and IEEE fp32 matmuls, because the MFT must not run in TF32.
7. **Compilation is optional, never required.** Per-stage regional `torch.compile` on real-view elementwise chains is used where Triton exists (Inductor does not fuse complex ops, and Triton is missing on Windows by default [07-M]). Triton kernels (fused phase screen, fused H-multiply, intensity accumulation, voxelizer) register via `torch.library.triton_op` and have pure-torch twins.
8. **Throughput targets** (v1 exit criteria): sketch/fast fluorescence and Mie holography at 256² ≥ 1000 images/s on the 3090. Balanced phase contrast of cells at 256², B = 16, 16 source points and a 36-slice BPM should take ≤ 0.5 s per training step.
9. **Multi-GPU.** Rank-local generation plus DDP (Linux/NCCL). Simulation is data-parallel only.

---

## 9. User-facing API

### (a) Holography of Mie spheres in a GPU training loop with a learnable size distribution

```python
import torch, gradoscopy as gs
from gradoscopy import dist as D
from gradoscopy.units import nm, um

r_med = gs.Learnable(0.45 * um, bounds=(0.10 * um, 1.2 * um), constraint="positive")
r_sig = gs.Learnable(0.15, bounds=(0.02, 0.5), constraint="positive")

beads = gs.Population(
    gs.Sphere(radius=D.LogNormal.from_median(r_med, r_sig).truncate(0.05 * um, 1.5 * um)),
    material=gs.Material(n=D.Uniform(1.45, 1.62)),
    position=gs.place.UniformInFOV(z=D.Normal(0.0, 2 * um).truncate(-6.6 * um, 6.6 * um)),
    count=D.Poisson(6.0), max_count=24,
)
scene = gs.Scene(
    sample=gs.Sample(beads, environment=gs.env.WaterOnCoverslip()),
    optics=gs.presets.InlineHolography(
        wavelength=532 * nm, polarization="x",
        objective=gs.Objective(NA=0.8, magnification=40, immersion="water")),
    detector=gs.Camera(pixel=6.5 * um, shape=(128, 128),
                       noise=gs.noise.PoissonGaussian(read=2.0), exposure_photons=D.Uniform(300, 3000)),
)
sim = gs.Simulator(scene, fidelity="fast",
                   outputs=["image", gs.labels.Table("sample.beads", fields=["position", "radius"])],
                   device="cuda")
print(sim.plan.explain())
```

```
Plan a41c07 · fast · cuda:0 RTX 3090 (calibrated 2026-09-24) · grad: {sample.beads.radius: median, sigma}
Outputs  image [64,1,128,128] · table sample.beads [64,24,{position,radius}] + presence
Bounds   radius ≤ 1.50 µm (truncation) → x_max 23.6 · z_rel ∈ [-6.6, 6.6] µm (truncation) · N ≤ 24
Grid     sim 192² @ 162.5 nm (s=1; intensity Nyquist 166 nm ok) · pad 32 px (6.6 µm × tan θ 0.75 + 2λ/NA)
 #  stage              implementation                                   notes                          est fwd
 1  lower[beads]       sphere → SphereList                              exact                          —
 2  illuminate         plane_wave(normal, x-pol) M=1 Λ=1                                               —
 3  interact[beads]    mie_pupil fp64 n_max=37, N_θ=512 → 6 941 k-samp.  C0 superpose                   1.1 ms
 4  collect            aperture(soft) ∘ defocus · P=1 (x-projection)    fused                          0.2 ms
 5  form               k-samples → scatter → ifft 192² → crop 128²                                     0.4 ms
 6  combine            reference(P(k_in)) + scattered, explicit interference term                     0.1 ms
 7  detect             poisson_gaussian · scaled_ST                                                    0.1 ms
 total est 1.9 ms fwd / 5.8 ms fwd+bwd · peak est 0.6 GB (train) · guards: none (all bounds from truncations)
Notes    [info] interact[beads]: cross-polarized far field dropped (P=1); 'balanced' keeps P=2
         [info] bead–bead coupling neglected (C0); gap < λ+2a in 4 % of probes
Gradient sample.beads.radius.median  pathwise(LogNormal.trunc) → mie → collect → form → detect(ST)  smooth
```

```python
net = SizingNet().cuda()
opt_net = torch.optim.AdamW(net.parameters(), 3e-4)
opt_sim = torch.optim.Adam(sim.parameters(), 1e-2)       # only r_med, r_sig (unconstrained, scale-normalized)
feat = gs.learn.features.RadialSpectrum()                # or a frozen CNN embedding

for step, real in enumerate(real_unlabeled_loader):      # real holograms, no labels
    batch = sim.sample(64, seed=step)                     # one GPU call; TensorDict-like Trace + outputs
    img, tab = batch["image"], batch["table", "sample.beads"]
    # 1) train the sizing network on synthetic labels (simulator detached)
    loss_net = net.loss(img.detach(), tab.detach())
    opt_net.zero_grad(); loss_net.backward(); opt_net.step()
    # 2) move the size distribution toward the real data (sim-to-real, pathwise through Mie)
    loss_sim = gs.learn.mmd(feat(img), feat(real.cuda()))
    opt_sim.zero_grad(); loss_sim.backward(); opt_sim.step()   # guards clamp r_med/r_sig into their bounds
```

### (b) SMLM: fit a Zernike + pixel pupil to a bead stack, then generate training data

```python
obj = gs.Objective(NA=1.45, magnification=100, immersion=gs.materials.oil, pupil=[
    gs.pupil.Zernike(ansi=range(3, 22), coeffs=gs.Learnable(torch.zeros(19), bounds=(-0.3 * um, 0.3 * um))),
    gs.pupil.PixelPupil(n=64, phase=gs.Learnable(torch.zeros(64, 64)), amplitude=gs.Learnable(torch.ones(64, 64)),
                        regularizer=gs.reg.Smoothness(1e-3)),
])
beads = gs.Population(                                    # 20 immobilized 100 nm beads, one per ROI
    gs.Sphere(radius=50 * nm), labels=gs.Label(gs.fluor.Custom(em=gs.spectra.Band(670 * nm, 30 * nm))),
    position=gs.Learnable(init_xyz, bounds=gs.bounds.box(xy=0.5 * um, z=(-0.3 * um, 0.3 * um)), plate="sample"),
    photons=gs.Learnable(init_photons, constraint="positive", plate="sample"), count=1,
)
scene = gs.Scene(sample=gs.Sample(beads, environment=gs.env.OilCoverslipWater()),
                 illumination=gs.Widefield(excitation=gs.spectra.Mono(640 * nm)),
                 optics=gs.presets.Widefield(objective=obj),
                 detector=gs.Camera(pixel=6.5 * um, shape=(31, 31), noise=gs.noise.SCMOS(maps=cam_maps),
                                    background=gs.Learnable(torch.full((20,), 100.0), plate="sample")),
                 acquisition=gs.FocusStack(z=torch.linspace(-2, 2, 41) * um))
fit = gs.Simulator(scene, fidelity="accurate", outputs=["expected"], device="cuda")   # no noise sampling
print(fit.plan.explain(level="summary"))
#  collect: richards_wolf P=3, dipole basis (isotropic) — required: NA/n_i=0.955 > 0.7 at 'accurate'
#  emit[beads]: sparse_roi_mft with bead form factor |F_ball(k)| folded into the OTF (exact for 100 nm bead)
#  acquisition 'focus' → A axis (41) · detect: scmos.log_prob (no sampling)

stack = torch.as_tensor(tiff.imread("beads.tif")).cuda()          # [20, 41, 31, 31] ADU
gs.learn.fit(fit, data=stack, loss="nll", optimizer="lbfgs", steps=200,
             params=["optics.objective.pupil.*", "sample.beads.*", "detector.background"])
calibrated = fit.export("optics.objective", freeze=True)          # Const-valued spec, reusable anywhere

# --- SMLM training-data generator with the calibrated pupil ---
emitters = gs.Population(gs.Point(), labels=gs.Label(gs.fluor.AF647(photophysics=gs.photo.TwoState(
               k_on=0.02, k_off=0.5, mode="stochastic"))),
           position=gs.place.UniformInFOV(z=D.Uniform(-1 * um, 1 * um)),
           photons=D.LogNormal.from_median(2000, 0.5).truncate(100, 20000), count=D.Poisson(12), max_count=48)
smlm = gs.Simulator(scene.replace(sample=gs.Sample(emitters), optics=gs.presets.Widefield(objective=calibrated),
                                  detector=gs.Camera(pixel=6.5 * um, shape=(64, 64), noise=gs.noise.SCMOS(cam_maps)),
                                  acquisition=gs.Frames(3)),
                    fidelity=gs.Fidelity("fast", polarization="vectorial"),   # keep the model the pupil was fitted in
                    outputs=["image", gs.labels.Table("sample.emitters", fields=["position", "photons", "on"])])
# plan: emit = sparse_roi_mft (N ≤ 48 ≪ crossover ≈ 300–500), 6 pupil transforms/emitter (dipole basis)
```

### (c) A cell in phase contrast (or QPI) through a multislice solver

```python
cell = gs.Group(
    body=gs.Capsule(length=D.Uniform(8, 14) * um, radius=D.Uniform(2.5, 3.5) * um,
                    material=gs.materials.cytoplasm(n=D.Normal(1.370, 0.005).truncate(1.35, 1.39))),
    nucleus=gs.Ellipsoid(radii=(3 * um, 2.5 * um, 2 * um), material=gs.Material(n=1.39)),
    droplets=gs.Population(gs.Sphere(radius=D.Uniform(0.2, 0.5) * um), material=gs.materials.lipid,
                           count=D.Poisson(8), max_count=20, position=gs.place.InsideParent()),
    texture=gs.RandomField(spectrum=gs.spectra.Matern(ell=gs.Learnable(0.8 * um, bounds=(0.2, 3.0)), nu=1.5),
                           amplitude=gs.Learnable(0.004, bounds=(0.0, 0.02)), mask="body"),
    # implicit alternative: body=gs.FieldComponent(fn=MyNeuralField(), bbox=((-8,-8,0),(8,8,7)))
)
scene = gs.Scene(sample=gs.Sample(gs.Population(cell, count=D.Poisson(3), max_count=6),
                                  environment=gs.env.WaterOnCoverslip()),
                 optics=gs.presets.PhaseContrast(objective=gs.Objective(NA=0.75, magnification=40, immersion="air"),
                                                 annulus=(0.20, 0.26), ring=gs.pupil.PhaseRing(shift=0.5 * torch.pi,
                                                 transmission=0.2)),
                 illumination_spectrum=gs.spectra.Band(550 * nm, 50 * nm),
                 detector=gs.Camera(pixel=6.5 * um, shape=(256, 256), noise=gs.noise.PoissonGaussian(read=1.6)))
sim = gs.Simulator(scene, fidelity="balanced",
                   outputs=["image", gs.labels.InstanceMask("sample.cells"), gs.labels.Map("opl")])
print(sim.plan.explain())
```

```
Plan 9be210 · balanced · grad: {texture.ell, texture.amplitude}
Bounds   extent_z ≤ 7.0 µm · Δn_max 0.11 (lipid) · φ_max 3.4 rad · NA_ill+NA_det = 1.01
Grid     sim 320² @ 162.5 nm (s=1) · 36 slices Δz=0.20 µm (≤ λ_m/2) · smooth sizes
 #  stage             implementation                                     est fwd   est mem(train)
 1  lower[cells]      blurred-SDF occupancy σ=0.3h → OccupancyChannels(4)  procedural (in-loop VJP, 0 B stored)
 2  illuminate        kohler annulus → 16-pt quadrature · Λ=1 (band→1 bin at balanced)
 3  interact[cells]   march.bpm (tilt-corrected) · adjoint.reversible     ~150 ms   ~2.3 GB
 4  collect           soft aperture ∘ phase ring · P=1 (NA/n 0.75, no pol. optics)
 5  form+reduce[src]  fft_full · chunked_ckpt chunk=4
 6  detect            poisson_gaussian · scaled_ST
 7  label.instance    occupancy.project → camera grid (σ=0.3px)     label.opl  project(view) exact
Decisions (why interact[cells]=bpm)
   thin_screen   rejected validity: extent 7.0 µm > DOF 1.30 µm (warn); bpm available at 1.4× cost → escalation
   born_slices   rejected validity: φ_max 3.4 rad ≫ 0.35·2π·(Δn·d/λ) limit (fail)
   ssnp          not required at 'balanced' (min I4a); cost 2.0×
   wotf/socs     rejected: group not thin
Gradient  texture.ell → RandomField(pathwise ε fixed) → occupancy → bpm(adjoint) → … smooth
```

For QPI, swap the preset: `gs.presets.OffAxisHolography(carrier=...)` or `gs.presets.QPI(output="phase")`. The sample and the plan's interaction stage stay the same. Only Collect/Combine and the outputs change.

### (d) Same scene, several fidelities, and what the planner chose

Mixed sample: 200 nm beads inside the cells above, imaged in brightfield-holography.

```python
scene = scene.replace(sample=scene.sample.add_to("cells", beads=gs.Population(gs.Sphere(radius=0.1 * um),
                      material=gs.materials.polystyrene, count=4, position=gs.place.InsideParent())),
                      optics=gs.presets.InlineHolography(wavelength=532 * nm, objective=obj60x))
fast = gs.Simulator(scene, fidelity="fast")
acc  = fast.with_fidelity("accurate")              # same spec, same ParamStore (learnables), different plan
print(gs.plan.diff(fast.plan, acc.plan))
```

```
stage             fast                                         accurate
lower[cells]      project(σ=0.5h) → OPL/absorbance maps        blurred-SDF σ=0.3h ×1.5 → RIVolume 480²×54 (procedural)
lower[beads]      SphereList; host cut-out: n/a                SphereList; host cut-out-and-fill in cell volume
illuminate        plane wave, Λ=1                              plane wave, Λ=1 (laser)
interact[cells]   thin_screen   [warn extent>DOF]              march.ssnp · adjoint.reversible
interact[beads]   mie_pupil C0, host=cytoplasm n=1.37 [info]   mie_inject C1 at 4 planes, local field, host n
collect           scalar                                        jones P=2 ∘ layer_stack(coverslip)
detect            poisson_gaussian                              poisson_gaussian
est fwd / train   4 ms / 11 ms                                  0.9 s / 2.7 s
```

```python
print(acc.plan.why("interact[beads]"))            # candidates, validity, costs, the coupling decision
report = gs.compare_fidelity(fast, acc, n=32)     # replays identical traces through both plans
print(report.table())                             # rel-L2 image error, bead-centroid shift, cost ratio
bal = fast.with_fidelity(gs.Fidelity("fast", objects={"sample.cells.beads": {"coupling": "inject"}}))
```

### (e) Reusing the forward model for an inverse problem

**(e1) Mie characterization of one hologram by maximum likelihood.** This is the xSight/holopy workflow, reusing the scene from (a).

```python
fit_scene = scene.bind({                                   # replace random sites by per-image learnables
    "sample.beads.count": 1,
    "sample.beads.radius":   gs.Learnable(0.5 * um, bounds=(0.1 * um, 1.5 * um)),
    "sample.beads.material.n": gs.Learnable(1.5, bounds=(1.34, 1.70)),
    "sample.beads.position": gs.Learnable(xyz0, bounds=gs.bounds.box(xy=2 * um, z=(-6.6 * um, 6.6 * um))),
})
fit = gs.Simulator(fit_scene, fidelity=gs.Fidelity("accurate", backward="fast"))   # split-fidelity gradients
res = gs.learn.fit(fit, data=hologram_adu, loss="nll", optimizer="lbfgs", steps=100)
print(res.params, res.crlb())                              # Fisher information via jacfwd on ≤10 parameters
```

**(e2) QPI tomography with LED-array illumination.** The sample is a learnable voxel field and is reconstructed through the same march.

```python
vol = gs.FieldComponent(values=gs.Learnable(torch.zeros(48, 256, 256), bounds=(-0.02, 0.12)),
                        bbox=((-20, -20, 0), (20, 20, 9.6)), quantity="delta_n")
tomo = gs.Scene(sample=gs.Sample(vol, environment=gs.env.WaterOnCoverslip()),
                optics=gs.presets.Brightfield(objective=gs.Objective(NA=0.4, magnification=20)),
                illumination=gs.LEDArray(grid=(9, 9), pitch=4 * mm, distance=60 * mm, spectrum=gs.spectra.Band(530*nm, 20*nm)),
                acquisition=gs.LEDSequence("each"),                     # 81 frames = acquisition axis
                detector=gs.Camera(pixel=6.5 * um, shape=(256, 256), noise=gs.noise.PoissonGaussian(read=2)))
rec = gs.Simulator(tomo, fidelity=gs.Fidelity("balanced", interaction="ssnp", budget=gs.Budget(memory="16GiB")),
                   outputs=["expected"])
# plan: angles chunked (P7: chunk 9 LEDs + adjoint.reversible), volume stored once (leaf) + its gradient
opt = torch.optim.Adam(rec.parameters(), 5e-3)
for it in range(300):
    for leds in gs.learn.minibatch(rec.acquisition("led"), size=9):   # stochastic angle mini-batches
        pred = rec.expected(select={"led": leds})
        loss = gs.learn.nll(rec, data[leds], pred) + 1e-4 * gs.reg.tv3d(vol.values)
        opt.zero_grad(); loss.backward(); opt.step()
phys = gs.interop.deepinv.as_physics(rec, wrt="sample.values", linearize_at=vol)   # A, A_adjoint for deepinv solvers
```

---

## 10. DeepTrack2 integration strategy

The goal is to keep DT2's stochastic-program user experience (properties, `>>`, `&`, `^`, Sources, sequences) and move all physics, batching and gradients into gradoscopy [01 option B].

1. **`gradoscopy.interop.deeptrack`** provides DT-named Features (`Brightfield`, `Holography`, `Fluorescence`, `Darkfield`, `ISCAT`, `MieSphere`, `MieStratifiedSphere`, `Sphere`, `Ellipsoid`, `Ellipse`, `PointParticle`, `Zernike`, `Poisson`, `Gaussian`, `IlluminationGradient`). They return **descriptor objects**, not arrays. A descriptor is a `Wrapper` subclass carrying kind plus resolved property values, so `particle.position` labels and `SampleToMasks` keep working.
2. **Template + batched resolve.** At first use, `EngineMicroscope` (`__distributed__=False`) inspects the pipeline once and builds a **gradoscopy scene template**: component kinds, `N_max` from `^ N` or probe maximums, and optics structure. It **probes** the DT DAG (256 resolutions, cheap scalar Python) to derive bounds. It then plans once. For each batch, `BatchedDataset(pipeline, batch_size=B)` calls DT's `update()`+resolve B times (Python, scalar), stacks the descriptors into a `Trace` `[B, N_max]` with presence masks, and runs the plan **once on the GPU**. This removes DT's per-sample GPU loop while keeping its semantics.
3. **Learnables pass through.** DT properties that are tensors with `requires_grad` (DT passes them unchanged [01]) are mapped to `Learnable` sites in the ParamStore by identity. The adapter bypasses DT output caching for rendering, which fixes the stale-cache-versus-autograd problem (#352).
4. **Units** are converted at the adapter boundary: DT SI metres and pixel units become µm. `position_unit="pixel"` resolves through the camera sampling at plan time.
5. **Compatibility switches.** `legacy_layout=True` returns `(H, W, 1)`. `Fidelity("deeptrack-2.0.2")` reproduces DT numerics for golden tests of DTGS121, DTEx203, DTEx205, DTEx252 and DTGS106. The default fidelity for adapter pipelines is `fast`, which *improves* physics: soft geometry, free-space inter-slice propagation, defined overlap rules. A one-time notice explains how to pin parity.
6. **Migration path.**
   - (i) Adapter release, with DT unchanged.
   - (ii) DT's `dt.optical.*` delegates to gradoscopy by default.
   - (iii) Native gradoscopy `Param`s become usable as DT properties (`dt.Value(gs.dist.Normal(...))`), which brings learnable distributions into DT pipelines.
   - (iv) `plan.explain()` is exposed as `pipeline.explain()` in DT.
   - deeplay consumes `BatchedDataset` unchanged.

---

## 11. Testing and validation strategy

1. **Analytic unit tests (fp64).**
   - Airy pattern and in-focus OTF.
   - Gaussian-beam propagation.
   - Parseval and unitarity of propagators; an energy-type loss must give an exactly zero gradient [07].
   - Fourier shift vs `roll`.
   - Zernike orthonormality.
   - Blurred-ball occupancy vs FFT of F_ball·Gaussian (≤ 1e-7 [06-M]).
   - Mie Q_ext, Q_sca, S1 and S2 vs miepython/PyMieScatt (BSD/MIT, test extra). PyMieDiff and holopy are *GPL oracles*, run only in an optional, non-distributed test environment [03, 04].
2. **Gradient tests.**
   - `gradcheck` in complex128 for every registered implementation (full mode on tiny shapes, `fast_mode` in bulk [07-M]).
   - Adjoint vs autograd ≤ 1e-5 in fp32 [07-M].
   - The empirical gradient audit: non-zero gradients for every learnable in the canonical scenes.
   - Learnability end-to-end: recover defocus, NA, Zernikes, radius, RI and distribution means from synthetic data.
3. **Cross-fidelity ladders.** Every implementation is tested against its `reference` inside its validity regime, and the resulting **error tables** feed the planner's error models:
   - thin vs BPM for thin objects;
   - Born vs Mie at small Δn·d;
   - BPM/SSNP vs Mie for voxelized beads (and vs MBS in v2);
   - dipole vs Mie at x < 0.3;
   - scalar vs vectorial at low NA;
   - strata vs sparse emitters;
   - stochastic/quadrature Abbe vs on-grid Abbe;
   - C0 vs C1 vs C3 for beads in a phantom cell.

   CI regenerates a *fidelity atlas* (error versus regime per implementation pair).
4. **Planner tests.**
   - **Golden plan snapshots**: `explain(level="decisions")` text for ~30 canonical scenes is checked in, so planner behaviour changes show up as reviewable diffs.
   - Property-based tests (hypothesis): for random scene structures and fidelities, a compiled plan must have every stage input satisfied, static shapes, guards covering every consumed bound, no zero-gradient path for requested learnables, and deterministic output.
   - Tests for replan triggers and guard policies.
   - Frozen-plan round-trip.
5. **Consistency contract tests.** Photon count, centroid and sign conventions agree across all presets for canonical scenes. Label and image registration agree under augmentation.
6. **DeepTrack parity.** Golden images under `deeptrack-2.0.2` for the five reference notebooks, to tolerance.
7. **Statistics.** Poisson mean = variance; EMCCD variance ≈ 2G²Φ [05]; KS tests on sampled sites; seed and replay bit-exactness; counter-based per-image determinism.
8. **Performance.** A self-hosted 3090 runner records time and `max_memory_allocated` per canonical plan and compares against `plan.profile` predictions. It fails on regressions > 20 % or on cost-model error > 3×. Separate CPU and MPS (c64-only) smoke jobs run too.

---

## 12. Roadmap

Milestones are ordered so that the riskiest architectural bets are proven first on the smallest physics.

| Milestone | Deliverables | Exit criteria |
|---|---|---|
| **M0 Spine** (≈ 4 wk) | L0 units/grids/currencies; L1 Param/Trace/sampler/estimators/truncation; registries; planner passes P0–P9 with *two* implementations per op for one path (Gaussian sprite vs sparse ROI MFT fluorescence); `explain/why/diff`; Poisson–Gaussian camera with `log_prob` | Plan for a fluorescence scene prints and snapshot-tests; planner switches sprite↔ROI by fidelity; gradient audit catches an injected zero path; replay bit-exact; planning < 50 ms |
| **M1 Pupil & incoherent path** (≈ 6 wk) | Scalar pupil, Zernike, pixel pupil, Gibson–Lanni; strata OTF; sparse/dense cost rules calibrated ([B-M] crossover); EMCCD/sCMOS; `learn.fit`; frames and 2-state photophysics | DeepSTORM3D-style mask learning runs; synthetic bead-stack fit recovers 15 Zernikes to < λ/100 RMS; ≥ 1000 img/s at 256² (fast); DTEx252 parity |
| **M2 Coherent particles** (≈ 6 wk) | Sources (plane wave, Köhler quadrature/stochastic, reference); fp64 Mie and layered Mie with pupil synthesis; dipole; thin screen; holography/darkfield/iSCAT presets; C0; DT adapter v0 + parity preset | Mie vs miepython ≤ 1e-6; DTGS121/DTEx203/DTEx205 parity; learnable size distribution converges on synthetic "real" data (example a) |
| **M3 Volumes & mixed samples** (≈ 8 wk) | Soft voxelizer (custom autograd + Triton twin); z-march engine (BPM, SSNP) with reversible/segmented adjoints and procedural slices; Born/Rytov (form factor + slices); WOTF; phase contrast/DIC/DPC presets; P2 partition; C1 injection; cut-out-and-fill | Ladders thin→BPM→SSNP and Born→Mie pass; phase-contrast halo present under partial coherence and absent under coherent illumination [05]; mixed-sample plan (example d) meets its memory estimate within 20 % |
| **M4 v1 release** (≈ 6 wk) | Jones and Richards–Wolf vectorial with dipole basis; spectral bins; TIRF/SIM/light-sheet/confocal two-stage presets; label renderers; CUDA-graph inference path; conformance suite public; docs | Vectorial PSF vs psf-generator; all 18 application reference pipelines of [08 §7.10] run; plugin written by a non-core developer passes conformance |
| **M5 v2** | MBS/LS with implicit differentiation; MLB/WPM; C2/C3′; Foldy–Lax; near-interface dipoles; space-variant PSFs; SOCS; accuracy-target planning; auto calibration | MBS vs wavesim agreement; error models calibrated from the atlas; `Fidelity(target_error=0.02)` selects plans that meet the target on held-out scenes |

---

## 13. Key decisions

**13.1 Units and precision.**

- *Decision:* **micrometres internally** (seconds for time; everything else SI-derived), with multipliers in `gs.units` and SI-metre conversion at the DT boundary. Precision: **complex64/float32** everywhere; **fp64** for special functions, phase-argument construction and gradcheck; complex32, bf16 and fp16 banned from physics.
- *Rationale for µm over metres:*
  - float32 range: a⁶ for a = 1e-7 m is 1e-42, which is subnormal and flushed. In µm it is 1e-6.
  - Optimizer step sizes: Adam's lr is in parameter units, so 1e-2 µm is sensible and 1e-2 m is absurd. `Learnable(scale=)` normalizes further.
  - Microscopy convention and readability; NA/λ ≈ 2 µm⁻¹.
  - Relative precision of phases is unit-independent, so nothing is lost.
- *Rationale for precision:* the measurements in [07-M] (c32 NaN; c128 7.5× slower; carrier subtraction plus fp64 phase build gives 3.7e-7).

**13.2 Central interchange representation.**

- *Decision:* a **two-port angular spectrum (`PortSpectrum`) in the medium basis, before the pupil**, with dense (`GridSpectrum`) and sparse analytic (`ObjectSpectra`) storage forms. Exit-plane real-space fields are a producer convenience converted by an exact FFT enforcer. The post-pupil field is internal to Collect/Form.
- *Rationale:*
  - The objective accepts only NA-limited propagating waves, so consumers evaluate only the pupil support.
  - Mie, dipole and emitter producers need no grid.
  - An R port is required for iSCAT and epi geometries [04].
  - Two ports compose with Redheffer products later.
  - The sparse form keeps per-object gradients exact, as Fourier phase ramps.

**13.3 Planner sophistication and when planning happens.**

- *Decision:* a **Cascades-style, cost-based, exhaustively enumerated selection over a fixed logical vocabulary**, wrapped in deterministic rule passes (bounds, partition, requirements, discretize, memory, rewrite). There is no general IR, no JIT of physics, and no second graph engine. Planning happens **once at construction**, from bounds. Replanning happens only on explicit events. Guards run on the GPU and report asynchronously.
- *Rationale:*
  - The search space is tiny, so exhaustive search is cheap, predictable and snapshot-testable.
  - Query optimizers solved exactly this "logical vs physical operator, properties and enforcers" problem.
  - Static plans enable CUDA graphs and optional `torch.compile`, and keep shapes independent of learnable values [05, 07].
  - Rejecting a general IR follows anti-pattern 2 in [09].

**13.4 Parameter/trace system; tensordict and Pyro.**

- *Decision:* **our own minimal `Param`/`Trace`/`ParamStore`** (≈ 1.5 k LOC): pytree-registered, with explicit generators and handlers local to each simulator. `Trace.to_tensordict()` and a Pyro-model export are optional interop. There is no hard dependency on either library.
- *Rationale:*
  - We need a small, very stable subset: path-keyed tensors, plates, masks, log_probs and replay.
  - tensordict releases track torch closely and add surface area. Pyro's global param store and handler stack are anti-patterns for an embeddable library [09].
  - Owning the leaves lets the planner read learnable bounds, constraints and estimator metadata directly, which the gradient audit depends on.

**13.5 nn.Module vs functional core.**

- *Decision:* **functional core** (L4 implementations are pure functions of tensors plus static config) with **one Module shell** (`Simulator` owns `ParamStore`, sampler and plans). Scene components are frozen dataclasses, not modules. `ModuleParam` lets any `nn.Module` (neural field, pupil network) act as a value.
- *Rationale:*
  - Pure functions make `gradcheck`, `torch.func`, the conformance suite and plan rewriting possible.
  - A single Module gives `.to()`, `state_dict()`, `parameters()` and DDP.
  - Mutable per-component modules cause aliasing between sampling and learning [09].

**13.6 Sparse vs dense render paths.**

- *Decision:* **both are first-class, and the planner chooses per group**. Coherent contributions combine in k-space: sparse objects are evaluated on the pupil-support samples and summed with the dense GridSpectrum before a single inverse FFT, so cross-terms are exact. Incoherent contributions combine as intensities on the detector grid (sparse ROI scatter-add plus dense strata/OTF convolution). Point emitters are *never* voxelized by default.
- *Rationale:* the measured crossover of ≈ 300–500 emitters per 256² image [B-M]. Sparse is exact in z and gives field-dependent pupils for free. Dense is flat-cost for crowded or continuous labels.

**13.7 Polarization representation.**

- *Decision:* the **P axis is always present**, with width 1 (scalar), 2 (Jones, transverse) or 3 (Cartesian). The **planner picks the width per stage**. The scalar fast path is the default below NA/n ≈ 0.7 without polarization optics, fixed dipoles or birefringent samples. Otherwise, or on request, it is lifted to Jones or vectorial. Unpolarized light is a Reduce implementation (two modes, or Jones-on-Stokes contraction).
- *Rationale:*
  - Polarization is expensive to retrofit [08].
  - An always-present axis means no API break.
  - Most training data at NA ≤ 0.8 does not need vectors, and GPU vectorial overhead is minimal when it is needed [05].

**13.8 Multi-wavelength grid handling.**

- *Decision:* **one spatial grid and one k-grid (cycles/µm) shared by every λ**. Pupils are evaluated per λ on that grid, with support NA/λ. The Λ axis is broadcast inside fields and reduced by a chunked `Reduce`. Where pupil-plane sampling in NA units is used (sparse ROI MFT), each λ maps through its own MFT onto the **common** image grid. Materials are lowered once to occupancy channels, and n(λ) is applied per bin as M multiply-adds per voxel.
- *Rationale:* Chromatix's per-λ `dx` leak forces every incoherent sum to reconcile grids [02]. The MFT/CZT approach used by POPPY, hcipy and dLux avoids that leak.

**13.9 Time and dynamics in v1.**

- *Decision:* **yes, but bounded.** The scope is:
  - frame acquisition axes;
  - reparameterized trajectories (Brownian, OU, drift, rotational);
  - 2/3-state photophysics in `Expected` (differentiable) and `Stochastic` (score/relaxed) modes;
  - motion blur as sub-exposure summed modes, with the count chosen by the planner from velocity bounds.

  Biology and mechanics stay out, behind a plugin interface.
- *Rationale:* tracking and SMLM are DT's core applications, and DT2 already has sequences [01, 08]. Time costs only a plate and an axis in this design.

**13.10 Dependency policy.**

- *Decision:* core depends on **torch and numpy only** (numpy for interop and I/O, never in the physics path).
- Optional extras:
  - `[fast]`: triton / triton-windows;
  - `[nufft]`: pytorch-finufft;
  - `[deeptrack]`, `[deepinv]`;
  - `[io]`: zarr, h5py, tifffile;
  - `[cli]`: tyro;
  - `[test]`: scipy, miepython, hypothesis.
- GPL references (PyMieDiff, holopy, psfmodels) are test oracles only. psf-generator (MIT) may be vendored with attribution.
- Excluded: pint, pydantic, xarray and tensordict in core. nvdiffrast is never used (license).
- *Rationale:* permissive licensing for DT2; hot paths free of wrappers [03, 09]; Windows machines without Triton must work [07-M].

**13.11 Fidelity semantics** (a decision the brief does not list, but central to this angle).

- *Decision:* **the minimum-requirement lattice plus the escalation rule is the v1 contract**. Error-target planning is v2, once calibrated error models exist.
- *Rationale:* literature thresholds are good enough to *reject* invalid methods. They are not good enough to *promise* error bounds before the ladder tests produce data.

---

## 14. Top risks and open questions

1. **Planner opacity ("magic").** Users may not trust or understand automatic choices.
   - *Mitigation:* `explain/why/diff` everywhere; golden plan snapshots; `strict=True`; pins; frozen plans stored with datasets; every fallback logged. Open question: is the default verbosity right? Proposal: one-line summary at construction; details on demand.
2. **Bounds for learnables and unbounded distributions.** If a learnable drifts to its bound or a distribution needs a large tail, shapes (n_max, padding) grow and replans cause recompiles or discontinuities mid-training.
   - *Mitigation:* truncation required for shape-affecting random sites; declared bounds required for shape-affecting learnables (the planner lists which); clamp as the default guard policy; `replan` only on request.
   - Open question: should the planner automatically widen bounds by a margin when a learnable approaches them?
3. **Validity thresholds and the error models are initially literature rules of thumb.** Wrong thresholds mean wrong choices.
   - *Mitigation:* ladder tests from M3 onward recalibrate them. Reports say whether an estimate is `literature` or `calibrated`.
4. **Accuracy of C1 injection for beads in cells.** The local-plane-wave approximation of the incident field at the bead is unvalidated.
   - *Mitigation:* mark it `info`/approximate and validate against C3 and MBS in M5. Open question: is a windowed plane-wave decomposition needed near strongly focusing organelles?
5. **Cost-model portability.** 3090 constants may mislead on other GPUs or MPS.
   - *Mitigation:* `gs.calibrate()`; cost only affects speed; profile-based feedback.
6. **Complex numbers and compilation.** Inductor does not fuse complex ops, and Triton is absent on Windows by default. Elementwise work is about 45 % of march time [07-M].
   - *Mitigation:* real-view formulations and Triton kernels as optional accelerators. Open question: does nvmath FFT-callback fusion justify a CUDA-only backend?
7. **DT2 batched-resolve throughput.** Python resolution of B DT DAGs may become the bottleneck at large B, and DT lambdas are unpicklable under Windows `spawn`.
   - Open question: measure DT resolve cost per property. If it is significant, offer a converter from DT property graphs to native `Param` graphs for common patterns.
8. **Scope pressure from vectorial, two-port and two-stage physics in v1.**
   - *Mitigation:* the *contract* ships complete (ports, excitation forms, P axis, source maps); implementations ship partially (R port only for Born, dipole and Mie; E3 in v2).
9. **Gradient-dependent planning surprises.** `train()` and `eval()` may select different implementations (for example SOCS vs Abbe), so the images differ slightly between training and evaluation.
   - *Mitigation:* `explain` shows both plans. `Fidelity(same_plan_for_eval=True)` forces the training plan.
10. **Memory estimates for exotic combinations** (FPM with 300 LEDs × spectral bins × Jones) may be wrong and cause OOM.
    - *Mitigation:* P7 safety margin; a `plan.profile()` dry run on first batch; automatic chunk halving on OOM (logged, not silent).
11. **Open question: how much of DT's name-matched dependency injection to replicate natively?** Proposal: `Derived` plus Param arithmetic covers most uses. Lambdas remain an escape hatch that is flagged as non-serializable.
12. **Open question: should `Fidelity` be allowed to vary per batch element** (mixed-fidelity batches for curriculum learning)? Tentative answer: no. Instead, use two simulators that share the ParamStore and interleave their batches.
