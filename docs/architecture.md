# gradix — Architecture Reference (v1 plan)

*Status: **revision 10 — a standalone engine with a layered public API.** Date: 2026-09-29. Revision 1 (2026-09-24) synthesised design panel A–D, three judge verdicts and research briefs 01–09, and was revised after a five-critic review. Revision 2 moved parameter sampling to the caller. Revision 3 makes gradix useful on its own: public building blocks, composition and convenience levels, with DeepTrack2 integration left for DeepTrack2's optics-pipeline revision (ADR-36, ADR-37). Revision 4 adds feeding data to a built Pipeline (datasets larger than a batch, object counts that vary per image, settings recorded per image, per-image values to refine; ADR-38) and corrects the polarisation detection path. Revision 5 moves polarisation optics into 1.0: vector scattering, Jones pupil optics and thin birefringent samples (ADR-20). Revision 6 adds optical stages before and after the sample for programmable and diffractive optics, including optical neural networks (ADR-39). Revision 7 sets the code-quality and documentation standard: ruff, ty and numpydoc gates, and narrow, milestone-scoped tutorial and example notebooks (ADR-40). Revision 8 names the project gradix; earlier revisions called it gradoscopy. Revision 9 records the lead developer's answers to six open questions: the MIT licence, the torch floor, learnable magnification and Gaussian beams in 1.0, a gradient-aware benchmark for the sCMOS likelihood, and image-keyed EMCCD in v1.x. Revision 10 records the lead developer's decisions after M0 and most of M1 were built: the Gaussian sCMOS likelihood, fast approximate batch-keyed noise until an exact sampler lands before 1.0, CI on GitHub-hosted runners only, and attribute access to outputs. Revision 11 pulls holography forward into a milestone M3a: Mie optics; in-line, off-axis, iSCAT and QWLSI interferometers with the camera's view; differentiable field reconstructions; and a Cramér–Rao bound on what a reconstruction keeps, for designing the illumination and detection of nanobead measurements (ADR-43–ADR-45). Appendix E lists every applied change.
Audience: the lead developer (DeepTrack2 core developer; microscopy + deep learning) and future contributors. **§0 is the plan; everything after it is reference.***

*Evidence tags (all measurements on the target RTX 3090, Windows 11, torch 2.14+cu130):*
- ***[Bxx]*** *research brief xx:* [01](background/research/01-deeptrack.md) *DeepTrack2,* [02](background/research/02-diff-optics-libs.md) *differentiable optics libraries,* [03](background/research/03-microscopy-simulators.md) *microscopy simulators,* [04](background/research/04-light-matter-ladder.md) *light–matter ladder,* [05](background/research/05-imaging-system.md) *imaging system,* [06](background/research/06-sample-representation.md) *sample representation,* [07](background/research/07-pytorch-engineering.md) *PyTorch engineering,* [08](background/research/08-applications-scope.md) *applications and scope,* [09](background/research/09-framework-patterns.md) *framework patterns.*
- ***[A §x]…[D §x]*** *sections of the panel proposals:* [A — field operators](background/design-panel/proposal-A-field-operators.md), [B — scene planner](background/design-panel/proposal-B-scene-planner.md), [C — MVP scope](background/design-panel/proposal-C-mvp-scope.md), [D — learnable pipelines](background/design-panel/proposal-D-learnable-pipelines.md); *judge verdicts in* [judge-verdicts.json](background/design-panel/judge-verdicts.json).
- ***[M-A]…[M-D]*** *measurement scripts written for proposals A–D:* [panel-A](background/evidence/panel-A/), [panel-B](background/evidence/panel-B/), [panel-C](background/evidence/panel-C/), [panel-D](background/evidence/panel-D/). ***[M-J eng | physics | ML]*** *measurements by the judges:* [judges-engineering](background/evidence/judges-engineering/), [judges-physics](background/evidence/judges-physics/), [judges-ml](background/evidence/judges-ml/). ***[B06 M]*** [research-06](background/evidence/research-06/); ***[B07]*** *measurements* [research-07](background/evidence/research-07/).
- ***[Rev]*** *numbers measured during the adversarial review and the later revision checks (same machine); scripts in* [evidence/review](background/evidence/review/) *(one folder per critic, verifier or revision check).*
- *The background documents are inputs. Some contain claims this document corrects (Appendix C); where they disagree, **this document takes precedence**. Untagged numbers are engineering estimates and are marked "estimate" where it matters.*

---

## Contents

0. [Read this first](#0-read-this-first)
1. [Goals, non-goals, design principles](#1-goals-non-goals-design-principles)
2. [Scope](#2-scope) — [capability matrix](#21-capability-matrix-release-plan), [application coverage](#22-application-coverage)
3. [Architecture overview](#3-architecture-overview) — [API levels](#31-api-levels)
4. [Core data model](#4-core-data-model) — [conventions](#41-units-frame-and-conventions-frozen-in-m0-conventionspy-one-tested-file), [carriers](#44-carriers-how-light-is-represented), [sample](#46-sample-how-matter-is-represented)
5. [Physics and the fidelity mechanism](#5-physics-and-the-fidelity-mechanism) — [fidelity](#56-how-fidelity-is-expressed), [planner](#57-the-planner-and-the-pipeline-analysis), [validity](#58-validity-checks), [optical stages](#511-programmable-and-diffractive-optics-optical-neural-networks)
6. [Inputs, gradients, randomness and labels](#6-inputs-gradients-randomness-and-labels) — [input contract](#61-the-input-contract), [keys](#63-randomness-explicit-keys-only), [envelopes](#64-envelopes), [feeding data](#69-feeding-data-to-a-built-pipeline)
7. [Extension and composition model](#7-extension-and-composition-model)
8. [Performance and memory engineering](#8-performance-and-memory-engineering)
9. [API walkthrough](#9-api-walkthrough)
10. [Front ends: bring your own sampling; DeepTrack2 later](#10-front-ends-bring-your-own-sampling-deeptrack2-later) — [contract](#102-the-contract-with-any-front-end), [cookbook](#103-bring-your-own-sampling-a-cookbook)
11. [Verification and validation](#11-verification-and-validation)
12. [Roadmap and engineering practice](#12-roadmap-and-engineering-practice) — [first two weeks](#121-weeks-12-the-first-two-weeks), [budget](#122-budget-and-scope-delta), [milestones](#123-milestones)
13. [Decision records](#13-decision-records)
14. [Risks, open questions and deferred items](#14-risks-open-questions-and-deferred-items)
- [Appendix A — Glossary and notation](#appendix-a--glossary-and-notation) · [B — Prior art](#appendix-b--prior-art-from-the-research-briefs) · [C — Known pitfalls](#appendix-c--known-pitfalls) · [D — Reference listings](#appendix-d--reference-listings) · [E — Changes applied](#appendix-e--changes-applied) · [F — Notes for sampling layers](#appendix-f--notes-for-sampling-layers)

---

## 0. Read this first

### 0.1 What it is

gradix is a **standalone, differentiable light-microscopy renderer** built on PyTorch. It renders a batch of images in one GPU call, and every tensor it receives can get gradients: sample geometry, materials, emitters, optics and camera. The same inputs render at several accuracy levels; the engine decides *how* to compute them and prints why (`explain()`).

It is useful on its own. **It never decides what to render.** Choosing parameter values is the caller's job:
- in standalone use, a few lines of torch (the sampling cookbook, §10.3);
- later, DeepTrack2, whose integration will likely coincide with a revision of how DeepTrack2 builds optics pipelines (§10.4).

The repository lives on the lead developer's GitHub for now, and neither project depends on the other.

### 0.2 Three levels, each usable on its own

The public API has three levels above the kernels. Higher levels only *construct* lower-level objects, so nothing forces you into a container.

1. **Building blocks (L2): elements and data objects.** Data objects describe matter and light as batched tensors in µm: object sets such as `gx.Spheres` and `gx.Emitters`, media, spectra. Elements are small physical operators with one job each: sources, interactions (Mie, Born, multislice, …), imaging (coherent pupil imaging, point-emitter PSFs, per-plane OTFs), excitation and cameras. They exchange typed **carriers** (`PlaneWaves`, `Field`, `ObjectSpectra`, `EmitterSet`, `EmitterDensity`, `Irradiance`), so any element that produces a carrier composes with any element that consumes it. You can call them directly, like functions.
2. **Composition (L3): `gx.Chain` and `gx.Pipeline`.** A Chain wires elements along the physical slots:
   - transmitted, reflected and interferometric light: **Source → Interact → Objective (pupil) → Detect**;
   - fluorescence: **Source → Transduce → Emit → Detect**.

   A Pipeline analyses a Chain once, then renders new data as a pure function: structure-compatible Chains, or values bound by field path (`pipe({"beads.position": …}, key=…)`, §6.9). The analysis sizes static grids from a value **envelope**, checks validity and gradients, and plans memory (chunking, checkpointing). A Pipeline adds keys, outputs, labels, CUDA-graph capture and `explain()`.
3. **Convenience (L4): `gx.plan`, containers and presets.** `gx.Sample` (named object sets and the environment) and `gx.Microscope` (light, objective, camera, acquisition) are modular containers of L2 objects, and presets are functions that build Microscopes. `gx.plan(sample, microscope, fidelity="standard")` **chooses elements** for each population by a readable rule table, builds the Chain, and wraps it in a Pipeline. `plan.chain(...)` shows the Chain it built, which you can edit and run yourself.

Choosing implementations is choosing fidelity. You choose them explicitly at L2 and L3, or through a `Fidelity` policy at L4. Each element declares what it accepts, where it is valid, how good its gradients are and which other element it approximates.

All coherent light meets in one representation: the plane-wave spectrum of the scattered light, carried beside *exact* plane waves for the unscattered, reference and reflected light. Mutually incoherent contributions are summed exactly once, at the detector.

### 0.3 A first example

The same hologram batch at the three levels. The sampling (three lines of plain torch) is the caller's.

```python
import torch, gradix as gx
from gradix.units import nm, um

B, N = 64, 8
r_med = torch.nn.Parameter(torch.tensor(0.30))  # µm; ANY tensor input may be learnable
beads = gx.Spheres(
    position=torch.cat([1.6 + 17.6 * torch.rand(B, N, 2), 10 * torch.rand(B, N, 1) - 5], -1),
    # reparameterised: gradients reach r_med
    radius=r_med * torch.exp(0.15 * torch.randn(B, N)),
    material=1.40 + 0.20 * torch.rand(B, N),
    presence=(torch.rand(B, N) < 0.6).float(),
)
env = gx.env.Homogeneous(n=1.33)
light = gx.light.PlaneWave(wavelength=532 * nm, irradiance=1e5)  # photons/µm²
objective = gx.Objective(NA=0.8, magnification=40)
camera = gx.Camera(pixel_size=6.5 * um, shape=(128, 128), noise=gx.noise.PoissonGaussian(read=2.0))

# L2: call elements directly (grids are sized from these inputs)
modes = light()
field = gx.imaging.Coherent(objective, camera)(gx.interact.Mie(beads)(modes, env), modes, env)
image = camera.sample(camera.expected(field), key=torch.arange(B))

# L3: the same elements in a Chain; the Pipeline fixes grids from an envelope and adds chunking, capture, explain()
chain = gx.Chain(
    light=light,
    scatterers={"beads": gx.interact.Mie(beads)},
    imaging=gx.imaging.Coherent(objective, camera),
    camera=camera,
    environment=env,
)
# ranges in µm
envelope = {"beads.radius": (0.05, 1.5), "beads.position.z": (-6, 6), "objective.focus": (-2, 2)}
# inputs= names the paths later calls may bind
pipe = gx.Pipeline(chain, envelope=envelope, inputs=("beads", "objective.focus"))
image = pipe(chain, key=torch.arange(B))["image"]

# The built Pipeline is then fed data by field path (§6.9): here 1000 recorded images, their rough detections and
# the focus recorded with each. B and N are capacities, so the last batch (40 images) fits, as do fewer beads.
pos, presence = gx.pad(detections)  # 1000 tensors [N_i ≤ 8, 3] µm → [1000, 8, 3] and [1000, 8]
for idx in torch.arange(1000).split(B):
    beads_i = gx.Spheres(position=pos[idx], radius=0.5, material=1.59, presence=presence[idx])
    image = pipe({"beads": beads_i, "objective.focus": focus_set[idx]}, key=idx)["image"]

# L4: containers + a fidelity policy; the planner chooses the elements (here: Mie for spheres at "standard")
sample = gx.Sample({"beads": beads}, environment=env)
# a Microscope of L2 objects
scope = gx.presets.InlineHolography(light=light, objective=objective, camera=camera)
plan = gx.plan(sample, scope, fidelity="standard", envelope=envelope)
print(plan.explain())  # which element renders what, on which grids, and why
image = plan(sample, scope, key=torch.arange(B))["image"]
my_loss(image).backward()  # gradients reach r_med through Mie, the pupil and the camera
```

Given the same inputs, keys and grids, all three produce the same images (a CI test, §11.10); an L2 call that sizes its own grids agrees within the padding tolerance. Values bound by field path give the same images as the equivalent whole Chain. The full workflow, calibrating a size distribution on unlabeled holograms and then training a network, is §9(a).

### 0.4 Answers to the questions

| Question | Answer (details in §) |
|---|---|
| Which scope is feasible yet covers most applications? | v1.0 is **the physics engine**; sampling is the caller's. It ships:<ul><li>pupil PSFs for point and dense emitters, with vectorial sparse PSFs and per-object pupils (M4b);</li><li>analytic particle scatterers (Mie, Mie-dressed dipoles, Born form factors);</li><li>a projection screen for thin samples and a multislice march for thick ones;</li><li>analytic excitation and source quadrature (partial coherence, TIRF, SIM, light sheets);</li><li>polarisation: vector scattering, Jones pupil optics and thin birefringent samples;</li><li>optical stages before and after the sample: programmable devices, diffractive layers and optical neural networks;</li><li>a camera model;</li><li>the three API levels.</li></ul>With sampling supplied by the caller (plain torch today, DeepTrack2 later), that reaches the *minimum* fidelity of 17 of the 18 application families surveyed in [B08], and of optical neural networks (§2.2), in **≈38–52 weeks for two developers** (≈35–49 under the M6 option; §12.2). |
| How are responsibilities split? | **The caller decides what gets rendered**: sampling, dependencies, counts, sequences, value labels, augmentations, datasets, training. **gradix does the physics**:<ul><li>data objects, elements and carriers;</li><li>composition and planning;</li><li>rendering;</li><li>cameras (noise, likelihood, Fisher information);</li><li>geometry-derived labels.</li></ul>DeepTrack2 becomes one caller once integrated. Inside gradix, elements compute, Chains compose, Pipelines analyse and run them, and the planner only chooses elements (§3). |
| What is the most generalizable approach that still allows many methods? | Typed carriers as the contract between small, self-describing elements, arranged along a fixed chain of physical slots. All coherent light meets in one representation (the scattered plane-wave spectrum plus exact plane waves). Methods differ behind a slot, never in the interfaces, and the same elements serve direct calls, hand-built Chains and planned ones (§3, §4.4, §5.1). |
| How are light and the sample represented (voxels? polygons? objects?) | *Light*: a sum of mutually incoherent coherent parts ("modes"). Unscattered and reference light are exact plane waves; only the scattered light is sampled. Fluorescence is carried as point sets, voxel densities and intensity.<br>*Sample*: continuous geometric objects (spheres, ellipsoids, capsules, signed-distance functions) with materials and labels, stored as batched struct-of-arrays tensors with presence masks. Voxels and neural fields are also accepted as inputs. Points stay points. Voxel grids are views produced on demand for the elements that need them. Polygon meshes come in v2 (§4.4–§4.6). |
| How granular is the optical path? | Conjugate planes, not lens surfaces: condenser pupil → sample in its layered environment → one combined objective pupil → image plane and camera sampling. Everything is computed in object space, and magnification is a coordinate scale. Lens prescriptions enter as measured or imported pupil maps. Ordered optical stages before and after the sample (masks, propagation, lenses, apertures) cover relay optics, programmable devices and diffractive optical networks (§5.11). |
| What makes it more than the sum of its parts? | Elements along independent axes (geometry, material, labeling, interaction, imaging, fidelity, camera, label) meet only through shared carriers and data objects. Each addition therefore works with all the others: integration cost grows as a *sum*, not a *product*. Every addition is usable at every level at once: directly, in Chains, and by the planner. The same inputs render through several microscopes and several fidelities, and reference elements calibrate the fast ones (§7.6). |
| How is fidelity tuned? | In discrete steps, never continuously or by gradient.<ul><li>At L2 and L3 you pick elements and their knobs yourself.</li><li>At L4 a frozen `Fidelity` policy with presets `draft`, `standard` and `accurate`, plus `reference` for validation, chooses elements for every population you did not pin, and fills discretisation knobs.</li></ul>Tiers are linked by tested approximation edges with published error tables. Every tier keeps the same radiometry, frames and labels, so `gx.compare()` can measure the difference (§5.6–§5.9). |
| How are simulation parameters learned in data-generation pipelines? | gradix differentiates with respect to every tensor it receives, whatever produced it:<ul><li>an `nn.Parameter`, such as a pupil, an NA or a bead's radius;</li><li>a value the caller sampled with reparameterised torch operations from a distribution with learnable parameters.</li></ul>Each Pipeline reports a gradient quality per input: exact, exact almost everywhere, biased (surrogate or estimator paths), or zero. It refuses an input that requires gradients when every path from it is zero. Camera noise uses a scaled straight-through Poisson estimator, which is exact for losses up to quadratic in counts.<br>Estimators for discrete draws (counts, choices) and truncated distributions belong to the caller's sampling code. The cookbook (§10.3) and Appendix F give tested recipes and the measurements behind them (§6). |

### 0.5 What ships when

Two developers, **≈38–52 weeks** to v1.0 (plan on ≈46, the calendar of §12.3; budget in §12.2).

- **M0 (wk 1–5)** walking skeleton, all three levels thin:
  - units, frame, schemas, carriers, keys, envelopes;
  - Gaussian-sprite and camera elements;
  - Chain/Pipeline v0 (fed whole Chains or values bound by field path) and `gx.plan` v0;
  - a test that the three levels give identical images.
- **M1 (wk 5–12)** pupil PSFs; sparse and dense fluorescence; SMLM, TIRF, SIM and light sheet; per-frame inputs; camera likelihoods; bead-stack pupil fitting; a coherent smoke test.
- **M2 (wk 10–15)** the composable API goes public for incoherent physics:
  - learnable magnification and pixel size;
  - element protocols reviewed and frozen;
  - Chain/Pipeline/plan documented;
  - the sampling cookbook;
  - two external users build fluorescence pipelines at the level of their choice.
- **M3a (≈7–9 weeks, after M1; revision 11)** holography and Mie optics, pulled forward from M3 (ADR-43): Mie and Mie-dressed dipoles; in-line, off-axis, iSCAT and QWLSI microscopes with the camera's view; differentiable field reconstructions; a Cramér–Rao bound on what a reconstruction keeps (ADR-44); shaped illumination for particles; vector producers (P = 2) last. M2 and the later milestones shift by its length, and M3 keeps the rest of its scope.
- **M3 (wk 15–29)** Mie, dipole and Born particles, scalar or vector (P = 2); holography, darkfield (including cross-polarised), iSCAT; Gaussian beams; partial coherence; CUDA graphs; optical stages, shaped illumination and diffractive networks.
- **M4 (wk 28–39)** cells in phase contrast, DIC (scalar and Jones), QPI, FPM and lensless holography; polarisation optics and birefringent samples (crossed polarisers, LC-PolScope); shaped illumination of thick samples; diverging references as a stretch goal.
  - **M4b (wk 29–36, in parallel)** vectorial PSFs and per-object pupils.
  - Alternatively, under the **M6 option**, M4b becomes a v1.1 ≈4–6 weeks after 1.0, and 1.0 lands ≈3–3.5 weeks earlier (≈35–49 weeks, §12.2).
- **M5 (wk 39–46)** hardening: consistency-contract CI, conformance suite, public `gx.compare`, documentation completion, a release candidate used by external users, API freeze (v1.0).

Every milestone ships its tutorial and example notebooks (§12.7). They are the first users of each API, so they surface awkward syntax before a freeze.

### 0.6 Decisions needed from the lead developer

The full table with owners and defaults is §14.2. These are the ones that most affect the plan, in deadline order:

1. **API review of the three levels** with two prospective users (Q16; default: the §3 and §9 sketches) — week 5, when units, schemas and incoherent carriers freeze (element protocols freeze at the week-15 M2 review).
2. **Vectorial optics in 1.0 (M4b) or 1.1 (M6 option)** (Q15) — by the start of M3.
3. **Optical neural networks in 1.0** (Q22; default: the ONN core in 1.0, the extended toolkit in v1.x) — by the start of M3.

On 2026-09-28 the lead developer settled the licence (MIT), the torch floor, learnable magnification, the analytic backgrounds and image-keyed EMCCD; on 2026-09-29 the sCMOS likelihood (the Gaussian, from the §11.6 benchmark), the keyed-noise trade-off and the CI scope (GitHub-hosted runners only) (§14.2).

### 0.7 Key decisions

1. **The caller owns parameter sampling; gradix is a standalone, functional physics engine** ([ADR-32, ADR-37](#13-decision-records)).
2. **Layered public API: building blocks work on their own; Chains and Pipelines compose them; the planner, containers and presets only construct them** ([ADR-36](#13-decision-records)).
3. **Schema-described data objects and elements, and randomness only through explicit keys** ([ADR-33, ADR-34](#13-decision-records)).
4. **µm internally**; SI metres and pixels only at the caller's boundary (`gx.from_si`, `gx.coords`) ([ADR-01](#13-decision-records)).
5. **+z points toward the detection objective** in every element; z = 0 is the sample-side coverslip surface when a layered medium is declared ([ADR-03](#13-decision-records)).
6. **Unscattered and reference light are analytic; scattered light is sampled**, with one radiometry rule at every tier ([ADR-04, ADR-05](#13-decision-records)).
7. **Fidelity is a discrete choice of elements over fixed slots**: explicit at L2 and L3, a readable deterministic rule table at L4. Tiers are linked by tested approximation edges, not a total order ([ADR-09 – ADR-12](#13-decision-records)).
8. **Pure PyTorch**: CUDA graphs first, optional Triton later, no C++ ([ADR-27](#13-decision-records)).

### 0.8 How to read the rest

- **By role:**
  - Lead developer: §0, §3, §9, §12, §14.
  - Physics contributors: §4–§5, §7, §11.
  - Users writing their own sampling: §6, §9 and §10.
  - Future DeepTrack2 integration: §10.4 and Appendix F.
- **Notation.** Symbols and capability IDs are defined in [Appendix A.1](#a1-notation).
  - Tensor axes use single capitals: B A M J L P N K Y X C, plus T for frames, S for species, Z for depth planes, H and W for output pixels, and R for the ROI side.
  - Outside tensor shapes, a few capitals keep their standard physics meaning: the propagator H; the cross-spectral density W and the channel weights W[C, L]; the source density S(u, λ) and the Mie amplitudes S₁/S₂; the EM gain G.
  - Counts are written n_src, n_mat, n_part and n_slices; magnification is Mag; capability IDs are cap-00…cap-39; L0–L4 are the API levels of §3.1.

---

## 1. Goals, non-goals, design principles

### 1.1 Goals

1. **G1 — Physically faithful, tunable simulation** of light microscopy (fluorescence, transmitted light, interferometric, partially coherent) at several discrete fidelity levels that are true approximations of each other.
2. **G2 — Differentiable end to end** with respect to every tensor input: sample geometry, materials, emitters, optics and camera. Parameters learned upstream therefore receive exact gradients, whether they enter directly (`nn.Parameter`) or through reparameterised sampling in the caller's code. This supports sim-to-real, calibration and design, including optical neural networks.
3. **G3 — GPU- and batch-native**: one batched render call per batch; fast tiers reach ≥ 10³ images/s at 256² on one RTX 3090 [B08 §4].
4. **G4 — Paired ground truth by construction.** Images and geometry-derived labels (masks, OPL, pixel positions) come from the same inputs in one call, in one coordinate frame. The caller pairs them with its value labels.
5. **G5 — Standalone and embeddable.** Useful on its own through a layered public API (building blocks, composition, convenience; §3.1), and embeddable in front ends. DeepTrack2 is the first expected front end; its integration is timed to its optics-pipeline revision (ADR-37).
6. **G6 — Extensible by third parties** under a conformance bar equal to the core's. Every extension is immediately usable at every API level.

### 1.2 Non-goals

- **A sampling or probabilistic-programming layer.** The following belong to the caller (ADR-32): the user's own sampling code, for which a tested cookbook ships in the docs (§10.3); DeepTrack2 once integrated; and deeplay:
  - parameter selection, dependencies, replicates, random counts, sequences and photophysics states;
  - value labels, augmentations, datasets, data loaders, training loops and distribution-matching objectives.
- **Data I/O beyond gradix's own artefacts.** gradix writes Pipeline and plan JSON, serialises its inputs (`gx.save_inputs`), and reads camera calibration maps; image readers and dataset writers are out.
- **Full-wave solvers inside the engine** (FDTD, FEM, DDA). They enter as imported far-field tables or T-matrices through the Interact slot [B04 §2.7, B08 §5].
- **Coherent nonlinear optics** (SHG, CARS, SRS) and Raman. Their emitters are phase-locked, which breaks the incoherent-mode model [B05 §1.3].
- **FLIM and photon timing.** Only a lifetime *label* is in scope.
- **Monte-Carlo radiative transfer.** Instead, v1 (M1) ships a phenomenological, differentiable turbid-slab operator on the strata path: Beer–Lambert attenuation × depth-dependent blur, plus a diffuse background labelled incoherent. It is the minimum fidelity that light sheet in tissue needs [B04 §2.12, B08 §2.7].
- **Lens-design ray tracing.** Objectives are aplanatic systems plus pupil maps; external tools may supply OPD or Jones maps.
- **Biology and mechanics** (growth, division). These live upstream, in the caller's code that produces data-object values.
- **A NumPy backend, global state of any kind, or an RNG of its own.** The sanctioned global state is `ieee_matmul()` (scoped, ADR-02) and `gx.render`'s plan cache, which never changes results; element calls, Pipelines and plans touch neither. Randomness enters only as explicit keys (ADR-34).

### 1.3 Design principles

1. **The front end decides; the engine renders.** gradix never chooses a parameter value. It only reports envelopes, validity and gradient quality, so that the front end can choose well (ADR-32).
2. **What vs how.** Data objects and containers never name a numerical method or a discretisation (`upscale`, `padding`, `dz`, source-point count, solver). Those belong to elements, chosen explicitly at L2 and L3 or by the `Fidelity` policy at L4 [B09 §2.1].
3. **Seams before solvers, frozen when first exercised.** The decisions that are expensive to change are frozen as soon as a working path exercises them, never before. At M0, after an API review with two prospective users, the following freeze:
   - units, frame and axis layout;
   - the **data objects and their schemas**;
   - key semantics;
   - the output and label shape contracts;
   - the incoherent carriers (`EmitterSet`, `EmitterDensity`, `Irradiance`).

   At M3 exit, after the optical-theorem, direct-summation and Fresnel-reference tests, the coherent carriers freeze (`PlaneWaves`, `Field`, `ObjectSpectra`, travel tags). Everything behind the seams may start naive [C §1.2].
4. **Physics is continuous; discretisation is static.** In a Pipeline, grids are static objects derived from the **envelope** of the inputs, never from the current values; a direct element call derives them from its own inputs unless `grid=` is given (§3.1). Physical sampling locations (emitter positions, source directions, pupil coordinates) are continuous and differentiable.
5. **Keep the zero order analytic.** The unscattered, reference and reflected light is carried as plane waves and filtered by *evaluation*. Only the compactly supported scattered part is sampled. Inside dense marches, the unscattered light is an exact, periodic on-grid carrier with an analytic residual tilt (§4.4) [A §4.3, M-A].
6. **Coherence is bookkeeping.** Source points, wavelengths, polarisation states, emitters, dipole orientations and exposure sub-steps are all mode axes, reduced once in Detect [B05 §1.1].
7. **Differentiable by construction, or visibly not.** There are no hard masks on paths that can require gradients, and no detaching caches. Each input field has a declared gradient quality, and an input that requires gradients along only zero paths is refused (§6.2). Randomness inside gradix (camera noise, textures, stochastic nodes) is a deterministic function of explicit keys, with library-owned estimators. Bare torch samplers are banned in `src/` (§3.2).
8. **Every approximation is declared, checked and reported.** Each implementation names what it approximates and its validity regime. Nothing falls back silently.
9. **A readable plan beats a clever planner.** The planner is a rule table with no cost-based search in v1 [B09 anti-pattern 2], and Pipelines and Plans explain themselves (`explain()`, `why()`, `diff()`).
10. **Pure function, batch-first, static shapes.** A Pipeline call, `pipe(chain, key=…)` or `pipe(values, key=…)`, has no hidden state:
    - caches never change results, and nothing global is touched;
    - data objects and elements hold exactly the tensors they are given, and outputs are caller-owned;
    - there is no per-sample Python on the hot path.
11. **Measure before accelerating.** Custom adjoints and Triton kernels are added when the committed benchmark suite says so.
12. **Every feature names its application trigger and a numeric exit criterion** [C §2.5].
13. **Every level is public.** Elements work on their own; Chains and Pipelines compose them; the planner, containers and presets only construct them. Nothing below L4 knows about containers (ADR-36).

---

## 2. Scope

### 2.1 Capability matrix (release plan)

Column meanings:
- **First in** is the milestone where a capability first appears, and the pre-release cut at it (v0.1–v0.5 at M0–M4; unstable API).
- **Release**:
  - **1.0** ships by v1.0 (M5, API frozen);
  - **1.1** is the M6 option (§12.2);
  - **1.x** means minor releases after 1.0 (≈3 months);
  - **2** is the next major track;
  - **later** means a plugin or research item;
  - **out** means never in core;

Every capability names its application trigger (principle 12).

| ID | Capability | First in | Release | Application trigger / justification |
|---|---|---|---|---|
| cap-00 | Conventions, units, carriers with background slot and travel tags, fixed-rank axes incl. P and A (frames) | M0 (incoherent frozen); M3 exit (coherent frozen) | 1.0 | Retrofit cost [B08 §2.13, C §1.2] |
| cap-01 | **Data objects and elements with schemas**. Each field declares its quantity, role (shared, per image, per object, per frame), event shape, whether it affects array shapes, and constraints. Includes: the canonical per-call layout; `gx.stack` (padding + presence) over any pytree; `gx.from_si`/`gx.to_si`; structure signatures; envelope extraction; schema export; `gx.tree` utilities; `gx.pad` for ragged lists and value binding by field path (§6.9) | M0 | 1.0 | The contract with any caller; every component introspectable (ADR-33) |
| cap-02 | Camera models: Ideal and Poisson–Gaussian at M0; sCMOS maps, EMCCD, quantisation and saturation at M1. Each has `expected`, `sample(key)` (batch- or image-keyed; image-keyed EMCCD in 1.x) and `log_prob`, plus Fisher information and QE/gain/offset/bit depth/output unit. Pixel integration uses the pixel MTF (§4.3). QCMOS: 1.x (no trigger yet) | M0 | 1.0 | Every application; system ID [B05 §4] |
| cap-03 | Gaussian sprites (draft emitters) | M0 | 1.0 | Huge tracking/detection datasets |
| cap-04 | Scalar pupil. M1: soft aperture, exact defocus (H_exp), Zernike OPD, pixel pupil (OPD), apodisation and layered collection (Fresnel t, Gibson–Lanni). M4: rings, stops, knife edge, DIC shear | M1 | 1.0 | All pupil engineering |
| cap-05 | Sparse emitters: pupil → MFT → ROI (ROI from the envelope) → scatter-add; global-spectrum fallback | M1 | 1.0 | SMLM, tracking, PSF engineering |
| cap-06 | Soft band-limited rasteriser (blurred ball, SDF ramp) with recompute-in-backward and explicit parameter inputs | M1 | 1.0 | Geometry gradients [B06 §3] |
| cap-07 | Dense emission: per-z strata OTF (rFFT), empty-slice compaction; phenomenological turbid slab (Beer–Lambert × depth blur + diffuse background) | M1 | 1.0 | Labelled cells, dense fluorescence, light sheet in tissue [B08 §2.7] |
| cap-08 | Analytic excitation → Transduce (uniform, TIRF evanescent, 2–3-beam SIM, Gaussian sheet), formed on a product grid that resolves the excitation modulation (§4.3) | M1 | 1.0 | TIRF, SIM, light sheet (minimum fidelity) |
| cap-09 | Coherent sparse producers on pupil support: Mie & layered Mie (fp64), Mie-dressed electric + magnetic dipole, Born form factor; vector (P = 2) or co-polarised scalar (P = 1) | M3a (Mie, dipole; Born and layered Mie: M3) | 1.0 | Holography, tracking, darkfield, iSCAT; polarisation-resolved scattering |
| cap-10 | Analytic background + explicit interference; references (in-line, blocked, reflected; off-axis through an image-side reference) | M3a | 1.0 | Holography, iSCAT, darkfield, phase contrast |
| cap-11 | Travel-tagged contributions; layered-medium Fresnel reference; dipole/Mie/Born backscatter | M3a (Born backscatter: M3) | 1.0 | Physical iSCAT/COBRI [B04 §4] |
| cap-12 | Köhler / LED sources with deterministic continuous-node quadrature; on-support single mode. Stochastic nodes: 1.x | M3 | 1.0 | Brightfield realism, PC halo, darkfield, FPM |
| cap-13 | Projection (centre-plane screen with 1/cosθ_out) and obliquity-corrected multislice (procedural slices, checkpointed, residual-demodulated) | M4 | 1.0 | Cells in BF/PC/DIC/QPI; thick-sample brightfield |
| cap-14 | Mixed samples C0: host-index rule (M3), cut-out-and-fill and proximity warnings (M4) | M3 | 1.0 | Beads in cells [B04 §5] |
| cap-15 | Geometry-derived labels: pixel positions (pinned pixel convention), instance/semantic masks, heatmaps, distance maps, OPL, phase, field outputs, emitter tables, per-object images; taps; coordinate helpers (`gx.coords`). Value labels (radii, identities, states) come from the front end | M0 (positions) / M1 | 1.0 | Paired data |
| cap-16 | Per-frame inputs: a frame role (A = T) on any data-object or element field (positions, photons, presence, focus, drift, …), with frame interval and exposure metadata. Trajectories, blinking states and appearance are produced by the front end | M1 | 1.0 | Tracking sequences, SMLM context frames |
| cap-17 | Vectorial 6-basis dipole emission (sparse path), Fresnel t_s/t_p through the stack; photoselection; analyser channels | M4b | 1.0 (1.1 under the M6 option) | High-NA system ID, fixed dipoles (≤ 40 nm bias) [B03 §2.2]; quantitative outputs at NA/n > 0.45 (§5.3); polarisation-resolved fluorescence (orientation, polarisation cameras) |
| cap-18 | Per-object field-dependent pupils (sparse path) | M4b | 1.0 (1.1 under the M6 option) | FD-DeepLoc/uiPSF-style calibration |
| cap-19 | Camera likelihoods and Fisher information (`gx.detect.log_prob`, `gx.detect.fisher`) and `gx.crlb` through forward-mode Jacobians, all at M1 | M1 | 1.0 | System ID, PSF design, CRLB |
| cap-20 | **Composition and convenience**: `gx.Chain`, `gx.Pipeline` (static grids from envelopes, chunking, checkpointing, keys, outputs, capture; `explain`/`validate`/`gradients`/`run`/`diff`), `gx.plan` (rule-chosen elements; `methods=` pinning; `plan.chain()`), `gx.Sample`/`gx.Microscope`, presets | M0 (v0) / M2 (public) | 1.0 | Every workflow; the three API levels (ADR-36) |
| cap-21 | CUDA-graph capture of static Pipelines (`Pipeline.capture`) with fresh output allocations | M2 (incoherent) / M3 | 1.0 | Windows-friendly speed [B07 §3.3] |
| cap-22 | C1 injection of analytic particles into the march (at particle exit plane) | — | 1.x | Beads/NPs inside cells |
| cap-23 | Jones pupils (P = 2) and polarisation optics (polarisers, retarders, compensators, Jones DIC); thin Jones samples (uniaxial materials in projection, `gx.JonesScreen`); per-frame polarisation states | M4 | 1.0 | Polarised-light microscopy and LC-PolScope, Jones DIC, birefringent fibres and crystals |
| cap-24 | Exposure sub-step motion blur (sub-exposure positions on M); bleaching-aware photon budgets are the front end's | — | 1.x | Fast diffusers |
| cap-25 | Confocal/ISM/2P effective-PSF rewrite; 3D-SIM patterns; Richards–Wolf focusing of excitation beams | — | 1.x | Confocal/ISM training data |
| cap-26 | Continuous camera sampling (MFT to continuous pixel centres): learnable magnification, pixel size, rotation and sub-pixel offset; affine registration between channels. The pixel MTF itself is in cap-02 | M2 (coherent paths: M3) | 1.0 | Calibration of magnification and pixel size; multi-camera registration |
| cap-27 | SOCS/WOTF opt-in fast paths (fixed optics); snap-and-shift thin-sample rewrite | — | 1.x | Label-free throughput |
| cap-28 | Front-end hooks: the sampling cookbook as tested documentation, and schema reference pages generated with `gx.schema_of` (schema export and `gx.tree` themselves are cap-01, M0) | M2 | 1.0 | Standalone users bring their own sampling; any front end, DeepTrack2 later (§10) |
| cap-29 | Validity-driven per-instance routing (e.g. dipole vs Mie under masks) | — | 1.x | Straddling populations |
| cap-30 | Lensless / in-line holography without objective (BL-ASM to the sensor) | M4 | 1.0 | Lensless holography ([B08 §2.18]: covered once the propagators exist) |
| cap-31 | Slice-wise Born/Rytov (dense), SSNP march mode, reversible/segmented adjoint, angle mini-batching helpers | — | 2 | ODT/IDT |
| cap-32 | Dipoles near interfaces (LDOS, orientation), full vectorial iPSF; static roughness screens | — | 2 | Quantitative mass photometry, TIRF SMLM |
| cap-33 | Low-rank dense space-variant PSFs; light-sheet BPM through tissue | — | 2 | Large-FOV fluorescence, tissue |
| cap-34 | Tensor-ε samples, polarised multislice | — | 2 | PTI/birefringence |
| cap-35 | Foldy–Lax multi-sphere; meshes via SDF/winding number; deepinv `LinearPhysics` export | — | 2 | Colloid clusters, imported shapes, reconstruction |
| cap-36 | Modified Born series / Lippmann–Schwinger reference solver with implicit adjoint | — | 2 / later | Reference tier, validation |
| cap-37 | T-matrix import; TorchGDM interop; learned surrogate tiers; ray-traced eikonal screens; MCX backgrounds | — | later | Niches through existing slots |
| cap-38 | Optical stages before and after the sample: `Mask` (scalar or Jones, any plane), `Propagate`, `Lens`/`FourierTransform`, `Aperture`, `DiffractiveStack`; shaped coherent illumination; device models `PhaseSLM` and `HeightMap`; region readout | M3 (shaped illumination of sparse producers and `PhaseSLM`: M3a; of dense producers: M4) | 1.0 | Optical neural networks, learned illumination, relay and 4f optics, optical preprocessing |
| cap-39 | Extended ONN toolkit: `DMD`, `DeformableMirror`, `SegmentedMirror` and `Metasurface` devices with binarisation estimators and lookup-table or neural surrogates; nonlinear optical layers; incoherent multi-plane stacks; a physics-aware-training helper | — | 1.x (Q22) | Hardware ONNs, metasurface optics, adaptive optics, optoelectronic networks |
| cap-40 | Shearing interferometry (QWLSI, 1-D shearing gratings, Talbot-type wavefront sensors): a periodic detection-side grating expanded in diffraction orders, each a sheared and tilted copy of the image field (ADR-45) | M3a | 1.0 | Quantitative phase of particles and cells with wavefront sensors |
| cap-41 | Differentiable field reconstructions (`gx.recon`): in-line back-propagation, off-axis sideband demodulation, iSCAT contrast, QWLSI demodulation and gradient integration; linear up to an invertible pointwise step; for measured and rendered frames alike | M3a | 1.0 | Holographic characterisation, training on reconstructed fields, measurement design |
| cap-42 | Reconstruction-aware Cramér–Rao bounds: the Fisher information of a reconstruction's output under the camera noise it propagates, differentiable for design (`gx.crlb(..., reconstruction=)`, ADR-44) | M3a | 1.0 | Designing illumination, pupils and references for particle characterisation through the instrument's own processing |
| — | Sampling layer, value labels, augmentation, datasets, training and distribution matching | — | the caller (own code; DeepTrack2 later) / deeplay | ADR-32 |
| — | FDTD/FEM/DDA solvers, coherent nonlinear optics, FLIM, MC transport, lens design, EM/X-ray | — | out | §1.2 |

Conveniences moved to 1.x by the review (`on_invalid="escalate"`, `Pipeline.graph`, `Pipeline.profile`) are listed in §12.4.

### 2.2 Application coverage

"Min" is the release at which the application's *minimum* fidelity [B08 §3] is met; "Ideal" is the release at which its ideal fidelity is met. Coverage assumes the caller supplies the sampling: plain torch per the cookbook (§10.3), or DeepTrack2 later. The last column records every place where a minimum requirement ("M" in [B08]'s matrix) is not met literally, and how v1 meets it instead.

| # | Application family | Capabilities used | Min | Ideal | Remaining gap after v1 | [B08] minimum not met literally → how v1 meets it |
|---|---|---|---|---|---|---|
| 1 | Particle tracking (BF, fluorescence, darkfield) | cap-02,04,05,09,10,12,15,16 | 1.0 | 1.x | motion blur (cap-24) | — (motion blur is Ideal, not Minimum, in [B08 §2.1]) |
| 2 | Holographic characterisation (Lorenz–Mie) | cap-04,09,10,19 | 1.0 | later | spheroids/clusters (cap-35/37) | — |
| 3 | iSCAT / mass photometry | cap-04,09,10,11 | 1.0 | 2 | near-interface dipole coupling (cap-32) | — |
| 4 | SMLM (astigmatic, DH, tetrapod, learned) | cap-02,04,05,16,17,18 | 1.0 | 2 | near-interface emission (cap-32) | T2 blinking → per-frame on/off states sampled by the caller (§10.3), rendered through per-frame photons (cap-16) |
| 5 | Cell segmentation/tracking (PC, DIC, BF, fluo) | cap-06,07,12,13,14,15 | 1.0 | 1.x | C1 (cap-22) | Born/Rytov → projection/multislice (cap-13). Obliquity-corrected multislice has first Born as its weak limit in the forward port, with no backscatter (§5.2); adequate for forward cell imaging |
| 6 | QPI / optical diffraction tomography | cap-13,19 (+ learnable voxels) | 1.0 | 2 | SSNP, slice Born, adjoints (cap-31) | Born/Rytov → projection/multislice (cap-13). Linearised Born/Rytov reconstruction models (the standard ODT inverse) arrive with cap-31 (v2); v1 reconstructs by autograd through multislice |
| 7 | Light sheet in tissue | cap-07,08 | 1.0 | 2 | BPM excitation, space variance (cap-33) | Phenomenological attenuation and depth blur → turbid slab on the strata path (cap-07) |
| 8 | Virtual-staining pretraining (paired BF/fluo) | cap-07,12,13; one input, two microscopes | 1.0 | 2 | emission through RI volume (v2) | Born/Rytov → multislice, as #5 |
| 9 | Fourier ptychography | cap-12,13, acquisition axis | 1.0 | 2 | tiling (v2); continuous LED calibration is already in v1 | — |
| 10 | SIM (linear 2D) | cap-07,08 | 1.0 | 1.x | 3D-SIM (cap-25) | — |
| 11 | Confocal / ISM | cap-07,08 → cap-25 | 1.x | 2 | — | pinhole/ISM rewrite deferred (cap-25) |
| 12 | Darkfield nanoparticles | cap-09,12 (annulus), dispersive ε | 1.0 | later | rods (cap-37), substrate (cap-32) | — |
| 13 | Polarisation microscopy | cap-23 → cap-34 | 1.0 | 2 | tensor-ε volumes (cap-34) | — |
| 14 | End-to-end optical design | cap-04,05,12,19,38 | 1.0 | 1.x | extended device library (cap-39) | — |
| 15 | System identification (pupil, camera) | cap-02,04,05,17,18,19,26 | 1.0 (1.1 under the M6 option) | 1.0 (1.1 under the M6 option) | — | — (vectorial is "M" in [B08]'s matrix, hence the M6 caveat) |
| 16 | Sim-to-real distribution matching | front-end reparameterised sampling (Appendix F) + cap-01 + any render path | 1.0 | 1.x | per-instance routing (cap-29) | — (requires reparameterised sampling in the caller's code, §10.3) |
| 17 | Physics-informed reconstruction | `expected` + autograd, cap-13 | 1.0 | 2 | adjoints, deepinv export (cap-31, cap-35) | Born/Rytov → multislice, as #6 |
| 18 | Lensless holography | cap-30 | 1.0 | 1.x | diverging references (an M4 stretch goal, else 1.1) | — |
| 19 | Optical neural networks and optical computing (added in revision 6; not in [B08]) | cap-23,38 → cap-39 | 1.0 | 1.x | extended device library, nonlinear layers, incoherent multi-plane stacks (cap-39) | — |

**Seventeen of [B08]'s eighteen families reach minimum fidelity at v1.0** by this table, and so does family #19 (optical neural networks), added in revision 6. Confocal/ISM (#11) reaches it in v1.x. Under the M6 option, system identification (#15) moves to v1.1, leaving 16 of 18 at v1.0.

Three cheap items make the count true, and they are budgeted in §12.2:
- the turbid slab (cap-07);
- per-frame inputs (cap-16), with blinking sampled by the front end;
- the lensless preset (cap-30).

Families #4 and #16 also depend on the caller's sampling: per-frame on/off states, and reparameterised draws for learnable distributions. Both are a few lines of torch in the cookbook (§10.3); DeepTrack2 will provide them once integrated. [B08]'s own "~16 of 18" assumed four things in v1. This plan defers three of them (first-order Born/Rytov volumes, a confocal pinhole/ISM and motion blur) and ships the fourth, a thin-Jones axis. The last column shows how each affected minimum is met instead.

### 2.3 Abstractions deliberately *not* built in v1 (seam kept)

| Not building | Why premature | Seam kept |
|---|---|---|
| A sampling DSL, probabilistic layer, trace/replay machinery or dataset pipeline inside gradix | The caller owns sampling (ADR-32); a second sampling DSL was revision 1's top-rated risk | Schemas and pytree utilities, so any front end can construct and wrap every component (ADR-33) |
| Cost-based plan search (Cascades/Volcano), device-calibrated costs | ≤ 3 implementations compete for any one (population kind, knob); cost-dependent choices change numerics across machines [M-J eng] | Implementations may declare `cost()`; a Pipeline can rank *equivalent* strategies |
| A general user-facing operator algebra with graph rewrites | The pipeline skeleton is fixed; modalities are recipes | Slots are typed; acquisition axes, detection splitters and ordered optical stages cover v1 (§5.11) |
| Redheffer star products / full S-matrix stacks | v1 needs only a single-interface analytic reflection | A `travel` tag (+1/−1) on every contribution |
| Mutual-intensity (4-D) coherence type | 256² would need ~34 GB [B02 §2.2] | Mode axes with weights |
| Custom reversible multislice adjoint | Procedural slices + checkpointing already suffice (§8.1, BPM-with-gradients row) | Stage `grad_strategy` field |
| Cross-call value caches (pupils, OTFs, Mie tables kept between calls) | Caches keyed on tensor versions are the stale-cache class of DT issue #352; per-call cost is small (§8.1) | Per-call hoisting; `Pipeline.capture` for repeated fixed-optics calls |
| Entry-point plugin discovery, schema validation (pydantic) | No third-party plugins yet | Decorator registries; data objects, elements and Pipelines serialise by registry name + schema version |
| Fidelity schedules, randomised fidelity, error-target planning | Needs ladder data first | Plans for several fidelities are cheap to build for the same inputs |
| Per-implementation byte models, a static gradient-quality composition algebra | No v1 consumer; every implementation and plugin would carry the declarations | Probe-chunk memory calibration; capability grad tags; the dynamic gradient probe (§5.7, §6.2) |

---

## 3. Architecture overview

### 3.1 API levels

The public API has four levels above a foundation layer. **Each level is usable on its own, and a higher level only constructs objects of the levels below** (ADR-36). Nothing below L4 knows about `Sample`, `Microscope`, presets or `Fidelity`.

```
 ┌─ callers (outside this repository) ────────────────────────────────────────────────────────────────┐
 │ your own sampling code (plain torch; §10.3) · DeepTrack2 (future integration; §10.4) · fitting     │
 │ tools · GUIs                                                                                       │
 └──────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                │ data objects + elements (tensors, µm) · key · optional envelope / Fidelity
 ┌─ gradix ─────────────────▼─────────────────────────────────────────────────────────────────────┐
 │ L4  convenience  gx.plan(sample, microscope, fidelity) → Plan: chooses elements, builds a Chain,   │
 │                  wraps it in a Pipeline · gx.Sample · gx.Microscope · gx.presets.* · gx.render     │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ L3  composition  gx.Chain: elements wired along the slot skeleton                                  │
 │                  gx.Pipeline(chain): envelope → static grids (SamplingPlan), validity, gradient    │
 │                  table, memory strategy (chunking, checkpointing), keys, outputs and labels,       │
 │                  capture, explain / why / diff / run / JSON                                        │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ L2  building     elements: gx.light.* (sources) · gx.interact.* · gx.excite.* · gx.imaging.* ·     │
 │     blocks       gx.Camera · gx.Objective + gx.pupil.* · gx.optics.* (stages) · gx.devices.*       │
 │                  data objects: object sets (gx.Spheres, gx.Emitters, gx.SDF, gx.Voxels, …),        │
 │                  materials, labelings, media (gx.env.*), spectra · lowering (gx.lower.*)           │
 │                  carriers, the element contract: PlaneWaves · Field · ObjectSpectra · EmitterSet · │
 │                  EmitterDensity · Irradiance · labels and coordinates (gx.labels.*, gx.coords.*)   │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ L1  kernels      gx.ops.*: pure tensor functions (FFT/MFT/CZT, propagation, pupils, Mie, dipole,   │
 │                  Born, multislice step, rasteriser, emitter PSFs, keyed noise, likelihoods)        │
 ├────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ L0  foundations  units · conventions · grids · schemas (gx.field) · keys · precision · registry ·  │
 │                  special functions (fp64 Mie, Bessel, Zernike) · gx.tree · envelopes · grid rules  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
   accel/ (optional): CUDA-graph capture helpers; triton/ kernels each with a torch twin
   testing/ (shipped): conformance suite, ladders, gradient probes, level-equivalence tests, references
```

**What each level promises:**
- **L2: elements run eagerly.** Called directly, an element sizes its own grids from the inputs it receives, unless explicit grids are given (`grid=`, `gx.sampling.*` helpers). This is convenient and exact, but shapes may change when values change.
- **L3: a Pipeline is static.** It fixes every shape from an envelope (`pipe.envelope`), so a training loop, CUDA-graph capture and memory planning see the same structure on every call. It is built once and fed data: whole Chains, or values bound by field path, up to its batch and slot capacities (§6.9). Validity, the gradient table and `explain()` are available for *any* Chain, hand-built or planned.
- **L4: sugar that constructs L3.** `gx.plan` builds nothing that you could not build by hand. `plan.chain(sample, microscope)` returns the Chain it built, and `plan.pipeline` the Pipeline around it.

### 3.2 Packages and dependency rules

| Package | Level | Owns | May import | Must never |
|---|---|---|---|---|
| `units.py`, `conventions.py`, `_core` (grids and grid rules, envelopes, carriers, axes, precision, keys, registry), `special`, `schema`, `tree` | L0 | units, conventions, grids and the rules that size them, envelopes, carriers, schemas, the keyed counter hash, registries, special functions, pytree utilities | torch, numpy | know about elements or physics recipes |
| `ops` | L1 | every numerical kernel, custom autograd Functions, keyed samplers | L0 | see a data object or element; draw from a global or stateful RNG (keys and explicit generators only); read globals |
| `objects`, `devices`, `lower`, `labels`, `coords` | L2 | data objects (object sets, materials, labelings, media, spectra) with capabilities; optical device models that produce transmission maps; lowering to views; geometry-derived labels; pixel ↔ µm conversion | L0, L1; `lower` also `objects`; `labels` and `coords` also `objects`, `lower`, `detect` and `optics` (the Camera and Objective types) | draw unkeyed randomness (textures draw only through `_core/keys.py`) |
| `light`, `interact`, `excite`, `imaging`, `detect`, `optics` | L2 | the elements, one package per slot, each with `validity`, `configure`, eager `__call__`, capability tags and approximation edges | L0, L1, `objects`, `lower` | import `compose`, `plan` or `api` |
| `compose` | L3 | `Chain`, `Pipeline`, the SamplingPlan (the L0 grid rules applied to a Chain's envelope), validity aggregation, gradient table, memory strategy, rewrites, execution (chunking, checkpointing, keys, outputs, capture) | L0–L2 | choose elements (that is L4's job) |
| `recon` | L3 | differentiable field reconstructions from camera frames (§5.12): in-line, off-axis, iSCAT contrast, QWLSI; the optics they read from a Chain | L0–L2, `compose` | render images or own physics beyond the reconstruction's model |
| `plan`, `containers`, `presets`, `api` | L4 | `Fidelity`, the rule table, `Sample`, `Microscope`, presets, `gx.plan`, `gx.render`, `gx.compare`, `gx.crlb`, `save_inputs`/`load_inputs` | everything below | contain physics formulas |

**Enforcement:**
- an `import-linter` contract in CI;
- a check that every registered element has `validity`, a tolerance-table entry, gradcheck shapes (including removable singular points, §7.4) and a benchmark shape;
- a check that every registered data object and element has a complete schema;
- source-level bans under `src/`, with two exceptions: `_core/keys.py`, and the batch-keyed camera samplers in `ops/detect.py`, which draw only from explicit generators passed in. The banned constructs are:
  - global RNG state: `torch.manual_seed`, `np.random`, `random`;
  - `torch.rand/randn/randint/randperm`, every `torch.normal` overload, `torch.poisson`, `torch.bernoulli`, `torch.binomial`, `torch.multinomial`;
  - in-place samplers `Tensor.normal_/bernoulli_/exponential_/geometric_/cauchy_/log_normal_/uniform_/random_`;
  - `.sample(`/`.rsample(` on `Distribution` objects.

### 3.3 Responsibility split across projects and people

| Concern | Owner |
|---|---|
| Parameter values: distributions, dependencies, replicates, random counts, sequences and trajectories, photophysics states, non-overlap | **the caller**: the user's own sampling code in standalone use (cookbook, §10.3); DeepTrack2 once integrated (§10.4) |
| Learnable distributions (reparameterised sampling), estimators for discrete draws, truncation, reproducible per-index sampling | **the caller** (measured recommendations in Appendix F) |
| Value labels (sampled positions, radii, identities), augmentation and label co-transforms, normalisation, datasets, loaders, training, distribution-matching objectives | **the caller** / deeplay |
| Data objects, elements, carriers and schemas; composition (Chain, Pipeline) and planning; rendering; cameras (noise, likelihood, Fisher information); geometry-derived labels and coordinate conversion | **gradix** |

**Repository.** gradix is its own repository, on the lead developer's GitHub for now. It has no dependency on DeepTrack2, and DeepTrack2 has none on it until the integration of §10.4.

**Team of two** (timeline in §12). Every package has one owner.
- *Developer A* owns foundations for data and composition, and the levels users compose with: `schema`, `tree`, `objects`, `labels`, `coords`, `compose`, `plan`, `containers`, `presets`, `api`, `accel` and `testing`.
- *Developer B* owns physics: `units.py`, `conventions.py`, `_core` (grids, carriers, precision, keys), `special`, `ops`, `lower` and the element packages (`light`, `interact`, `excite`, `imaging`, `detect`, `optics`).

In M0, Developer B builds the conventions, grids, carriers, keys and the first elements (Gaussian sprites, camera), while Developer A builds the schema layer, the first data objects, `Chain`/`Pipeline` v0 and `gx.plan` v0.

The **lead developer** owns the tutorials and the level-by-level documentation, and chairs the API reviews with prospective users. They will drive the DeepTrack2 integration when it is scheduled. A third developer, if available, owns validation, benchmarks and the documentation site. With one developer the milestone order is unchanged and the calendar stretches ≈1.8×.

### 3.4 Public namespace

The full package tree is in [Appendix D](#appendix-d--reference-listings). Every public v1 name used in this document resolves as follows; `gx.interop.deepinv` arrives in v2.

| Public name | Module | Level | Contents |
|---|---|---|---|
| `gx.units` (`nm`, `um`, `mm`, `m`, `ms`, …), `gx.from_si`, `gx.to_si` | `units.py`, `schema/` | L0 | units and schema-driven SI conversion (§4.1, §6.1) |
| `gx.field`, `gx.DataObject` (base), `gx.schema_of`, `gx.stack`, `gx.pad`, `gx.tree.*` (`parameters`, `map`, `to`, `replace`) | `schema/`, `tree.py` | L0 | schemas, the pytree base, stacking and padding, pytree utilities (§6.1, §6.9) |
| `gx.PlaneWaves`, `gx.Field`, `gx.ObjectSpectra`, `gx.EmitterSet`, `gx.EmitterDensity`, `gx.Irradiance` | `_core/carriers.py` | L0/L2 | carriers: the element contract (§4.4) |
| `gx.Emitters`, `gx.Spheres`, `gx.LayeredSpheres`, `gx.Ellipsoids`, `gx.Capsules`, `gx.Cylinders`, `gx.Boxes`, `gx.Gaussians`, `gx.SDF`, `gx.Occupancy`, `gx.Voxels`, `gx.JonesScreen`, `gx.NeuralField`, `gx.Texture`, `gx.ObjectSet` (base), `gx.Parent`, `gx.geometry` (decorator), `gx.geom.*` | `objects/` | L2 | object sets (§4.6, §7.3) |
| `gx.Material`, `gx.materials.*`, `gx.Labeling`, `gx.dipole.*`, `gx.env.*`, `gx.Spectrum` | `objects/` | L2 | materials, labels, media, spectra |
| `gx.lower.*` (`ri_volume`, `emitter_density`, `emitter_set`, `projected`, `form_factor`) | `lower/` | L2 | lowering to the views elements consume (§4.6) |
| `gx.light.*` (`PlaneWave`, `Koehler`, `LEDArray`, `ReferenceBeam`, `SIMBeams`, `Evanescent`, `GaussianSheet`, `Shaped`, `Uniform`, `IlluminationNuisance`, `IlluminationEnvelope`), `gx.Annulus`, `gx.Disk`, `gx.Quadrature`, `gx.OnSupport`, `gx.GridAbbe` | `light/` | L2 | source elements (§4.5) |
| `gx.interact.*` (`Mie`, `Dipole`, `Born`, `Projection`, `Multislice`, `Fresnel`, `Superpose`) | `interact/` | L2 | interaction elements (§5.1, §5.2) |
| `gx.excite.*` (`Linear`) | `excite/` | L2 | transduction elements |
| `gx.imaging.*` (`Coherent`, `Sprites`, `PointPSF`, `Strata`) | `imaging/` | L2 | coherent and incoherent imaging elements (§5.3) |
| `gx.Objective`, `gx.pupil.*`, `gx.PupilModifier` (base), `gx.optics.*` (`Mask`, `Propagate`, `Lens`, `FourierTransform`, `Aperture`, `DiffractiveStack`, `Filter`, `Reference`) | `optics/` | L2 | objective, pupil modifiers, optical-stage elements (§5.11) |
| `gx.devices.*` (`PhaseSLM`, `HeightMap`; `DMD`, `DeformableMirror`, `SegmentedMirror`, `Metasurface` in v1.x) | `devices/` | L2 | optical device models (§5.11) |
| `gx.Camera`, `gx.noise.*`, `gx.detect.*` (`sample`, `log_prob`, `fisher`, `to_unit`) | `detect/` | L2 | camera element and functions usable outside a Pipeline (§5.5) |
| `gx.acq.*` | `optics/acquisition.py` | L2 | `FocusStack`, `Frames`, `LEDSequence`, `SIMPhases`, `Channels`, `PolarizationStates` |
| `gx.labels.*`, `gx.coords.*` (`to_pixels`, `from_pixels`, `depth`, `from_focus`, `shift_focus`), `gx.out.*` (`Image`, `Expected`, `Field`, `Tap`, `Regions`) | `labels/`, `coords.py`, `compose/outputs.py` | L2/L3 | labels, coordinates, output specs (§6.6) |
| `gx.Envelope`, `gx.envelope_of`, `gx.sampling.*` (`detection_grid`, `interaction_grid`, `padding`, `roi`, `slices`) | `_core/envelope.py`, `_core/rules.py` | L0 | envelopes and the grid rules, used by elements, by Pipelines and by hand (§4.3, §6.4) |
| `gx.Chain`, `gx.Pipeline`, `gx.SamplingPlan` | `compose/` | L3 | composition and execution (§5.7, §6) |
| `gx.Fidelity`, `gx.plan`, `gx.Plan`, `gx.render`, `gx.compare`, `gx.crlb`, `gx.save_inputs`, `gx.load_inputs` | `plan/`, `api/` | L4 | planning and convenience |
| `gx.Sample`, `gx.Microscope`, `gx.presets.*` | `containers/`, `presets/` | L4 | modular containers and modality recipes |
| `gx.register.*` (`element`, `object_set`, `pupil_modifier`, `preset`, …), `gx.Element`, `gx.Capabilities`, `gx.Edge`, `gx.Violation`, `gx.Cost` | `_core/registry.py` | L0 | plugin API: registries and the element contract (§5.1, §7) |
| `gx.views.*`, `gx.planning.*` (e.g. `slice_dz`), `gx.testing.*`, `gx.ops.*`, `gx.experimental.*` | `lower/`, `_core/rules.py`, `testing/`, `ops/`, `experimental/` | — | views, planning helpers for plugins, conformance and equivalence tests, kernels, unstable features (no SemVer guarantee) |

No `gx.Simulator`, `gx.Learnable`, `gx.dist`, `gx.fit` or `gx.io` exists; those were revision 1's parameter and fitting layers (ADR-32).

### 3.5 Data flow

```
 CALLER: resolve values per image (sampling, dependencies, counts, sequences) → data objects and elements
      │                                                                      (tensors; gx.stack → batch B; gx.from_si)
      ├──────────────── L2 (direct) ────────────────────────────────────────────────────────────────────────────────┐
      │ modes = light();  contrib = gx.interact.Mie(beads)(modes, env);  field = gx.imaging.Coherent(obj, cam)(…)   │
      │ mu = cam.expected(field);  image = cam.sample(mu, key)        grids sized from these inputs (eager)         │
      ├──────────────── L3 ─────────────────────────────────────────────────────────────────────────────────────────┤
      │ chain = gx.Chain(light=…, scatterers=…, emitters=…, excite=…, imaging=…, camera=…, environment=…)           │
      │ pipe = gx.Pipeline(chain, envelope=…, outputs=…)   → SamplingPlan, validity, gradient table, memory         │
      │ out = pipe(chain_like or {path: value}, key=…)     pure; capacities (§6.9); chunked, checkpointed           │
      ├──────────────── L4 ─────────────────────────────────────────────────────────────────────────────────────────┤
      │ plan = gx.plan(sample, microscope, fidelity=…, methods=…)  → rule table chooses elements → Chain → Pipeline │
      │ out = plan(sample, microscope, key=…);   plan.chain(sample, microscope) shows the Chain it built            │
      └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 Inside every render:
   Lower     ─► views: EmitterSet | SphereList | KSpectrum | ProjectedMaps | RIVolume(procedural) | EmitterDensity
   Source    ─► PlaneWaves[B,A,M,J,L,P]  (analytic illumination modes; references)
   Interact  ─► tuple[Contribution] each with travel ±1: ObjectSpectra (sparse, lazy) ⊕ Field (dense) ─┐ coherent
   Objective ─► fused multiplier on pupil support, consuming travel = +1:                        │
               Propagate(z_ref → interface, in n_s)·t(u)·GL(focus)·P(k;λ,h)·√(k_z,i/k_i)          │
               (index-matched: Propagate(z_ref → focus)·P·√cosθ); background by evaluation P(u_b) │
   Transduce ─► at emitters ─► EmitterSet rates; ρ·I_exc on the product grid ─► EmitterDensity    │ incoherent
   Emit      ─► sparse MFT-ROI | strata OTF ─► Irradiance                                        │
   Detect    ─► Σ_{M,L,P} w [|E_b|² + 2Re(E_b*E_s) + |E_s|²] + image-side references + Irradiance ┘
               ─► pixel MTF, sample at pixel centres ─► camera: expected | sample(key) | log_prob(observed)
                                                               ▼
 OUTPUT: image [B,(T,)C,H,W] · expected · fields · geometry labels · taps
         · meta{pipeline hash, chunk sizes, key kind, in_envelope[B], diagnostics}
         (caller-owned tensors, never inference tensors or graph buffers, §6.5)
```

---

## 4. Core data model

### 4.1 Units, frame and conventions (frozen in M0; `conventions.py`, one tested file)

- **Units.** Lengths in **µm**. Wavelengths are vacuum wavelengths in µm; time in s; angles in rad; counts in photons. Spatial frequency f is in cycles/µm (the `fftfreq` convention) and k = 2πf in rad/µm. Directions are stored as **u = λ₀f⊥ = n·sinθ(cosφ, sinφ)**, which is dimensionless and independent of wavelength for a fixed angle. Irradiance is in photons·µm⁻² per exposure. `gx.units` supplies `nm = 1e-3`, `mm = 1e3`, `m = 1e6`. SI metres and pixels exist only at the caller's boundary (`gx.from_si`, `gx.coords`; §6.1). The reason is float32 **exponent range**, not relative precision: α² for a 10 nm particle is exactly 0 in float32 metres but 1.37·10⁻¹¹ in µm [M-C, M-J eng]; a⁶ = 10⁻⁴² m⁶ for a 100 nm particle is subnormal (it evaluates to 1.0005·10⁻⁴², not flushed, but with only ~3 significant digits) [M-J eng]. A secondary reason: Adam moves each parameter by ~lr in parameter units, so lr = 10⁻² is 1 cm per step on a metre-valued position [M-C]; this applies wherever learnable parameters live, i.e. in the front end, which should parameterise them in µm or pixels (Appendix F).
- **Frame.** Right-handed. x runs along image columns and y along rows, so arrays are `[..., Y, X]` with Y, X last and contiguous. The lateral origin is the top-left *corner* of the camera field of view mapped to object space; pixel (i, j) has its centre at ((j+½)p, (i+½)p), with p the object-space pitch. **+z is the direction of travel of light toward the detection objective**, everywhere, including inside every solver (no frame flips). Transmitted illumination travels +z; epi illumination travels −z. **z = 0 is the sample-side surface of the coverslip** when the environment declares a `LayeredMedium`; otherwise it is an arbitrary sample reference. The objective focal plane is `Objective.focus` (a tensor field, default 0). Defocus is Δ = z − focus, so Δ > 0 means the object lies between the focal plane and the objective. Helpers (pure functions in `gx.coords`): `depth(d)` gives the z of a point at depth d into the sample, measured from the objective-side interface (z = −d on an inverted stand); `from_focus(dz, focus)` gives z = focus + dz (+dz toward the objective), which is how many users think of z; `shift_focus(dz)` shifts the focal plane. A population whose z-envelope, including its bounding box, crosses a declared interface or leaves the sample layer is a plan-time error that names the population and the envelope entry (§5.8).
- **Fourier.** Time dependence e^{−iωt}; plane waves e^{+ik·r}. `torch.fft` with `norm="backward"`. Forward propagation by +Δz multiplies an angular spectrum by e^{+ik_zΔz}, with k_z = 2π√((n/λ₀)² − |f⊥|²) on the branch Im k_z ≥ 0 (evanescent waves decay). Kernels use the H_exp form, H = exp(Δz·H_exp), so gradients reach Δz and defocus [B02 §2.5]. Phase arguments are built in fp64 with a carrier subtracted, reduced mod 2π, then cast: without this the error at 2·10⁴ λ is 1.6·10⁻³, with it 3.7·10⁻⁷ [B07 §2.4]. PyTorch returns conjugate-Wirtinger gradients; ports of JAX code (Chromatix) must flip a conj [B07 §2.1].
- **Continuous angular-spectrum convention.** A field in a homogeneous medium of index n_m is E(r⊥, z) = ∫ A(k⊥; z_ref) e^{i[k⊥·r⊥ + k_z(z − z_ref)]} d²k⊥. `conventions.py` fixes the discrete factors (Δk², FFT normalisation). They are pinned by a direct-summation test.
- **Aberrations** are optical path differences in µm (phase = 2π·OPD/λ, which is automatically correct for polychromatic light). This includes pixel pupils (`PixelPupil(opd=…)`); a radian-valued pupil must declare its reference λ and raises if used at another λ. Zernike polynomials use OSA/ANSI indices with RMS normalisation. Converters are provided for Noll indices and for radian-valued coefficients at a declared reference λ; the DeepTrack2 mapping is in Appendix F.8.
- **Apodisation and radiometry (ADR-05), homogeneous index-matched object space.** The interchange quantity is an angular-spectrum *density* sampled uniformly in k⊥. On it:
  - *collection* (object → camera) multiplies by **√cosθ = √(k_z/k_m)**, with k_z taken in the medium where A is defined, and **no extra Jacobian**;
  - *focusing* (pupil → object, e.g. excitation beams) multiplies by **1/√cosθ**;
  - far-field (reference-sphere) amplitudes such as Mie S₁/S₂ (Bohren–Huffman convention, E_s = S·e^{ik_m r}/(−ik_m r)·E₀) or dipole patterns are converted to angular spectra with the stationary-phase factor, **A(k⊥) = −S(θ,φ)/(2π k_m k_z) · e^{−i(k − k_in)·r_p}** (for z_ref = 0; otherwise z_p → z_p − z_ref). This is the only place a 1/k_z appears. The familiar "1/√cosθ collection factor" is the combination of this 1/k_z and the √cosθ above; applying both the 1/√cosθ factor and a 1/k_z Jacobian to an angular spectrum is off by cos²θ, ≈6.6× in amplitude at the pupil edge for NA 1.4 in oil [M-J physics].
- **Layered collection (one rule for the scalar and the vectorial pupil).** When a `LayeredMedium` is declared, which is the normal case for oil objectives imaging water or cells (SMLM, TIRF, iSCAT, bead stacks, example (b)), NA may exceed the sample index n_s. The homogeneous rule then fails: its pupil has a 1/√k_z,s singularity at |u| = n_s whose total depends on the grid, and propagating in the sample toward a deeper focus amplifies evanescent components by 14 / 208 / 4.3·10⁴ at Δ = −0.5 / −1 / −2 µm [Rev]. The rule is:
  1. Object spectra (sparse and dense) are defined in the sample medium and propagated *in that medium only toward the objective-side interface*, e^{ik_z,s(0 − z_p)} with 0 − z_p ≥ 0, so evanescent components always decay. A plan-time assertion forbids any operator that applies e^{ik_zΔz} with Δz < 0 to components with |u| > n of that medium.
  2. They are multiplied by the stack transmission t(u) for all |u| ≤ NA: t_s or the s/p average on the scalar path, t_s/t_p on the vectorial path. Components with n_s < |u| ≤ NA are kept, which models supercritical-angle collection (scalar SAF) at no extra cost; because t ∝ cosθ_s near grazing incidence, the integrable singularity at the critical radius cancels.
  3. Focus and defocus are **one Gibson–Lanni term**, not two: an immersion-medium phase k_z,i·Δf plus the design-versus-actual terms of the stack. The fused Objective multiplier is therefore Propagate(z_ref → interface, in n_s)·t(u)·GL(focus) instead of Propagate(z_ref → focus) (§3.5, §5.1, §5.3). Nothing propagates in the sample medium past the interface.
  4. Apodisation is **√(k_z,i/k_i)**: power flux in the immersion medium.
  5. Every k_z-dependent factor uses a safe square root (double-where), so forward values are unbiased and gradients with respect to n_s and λ stay finite across NA = n_s. If NA exceeds the index of a *homogeneous* medium (no interface declared), the validity check (Pipeline pass P3, or an eager element call) warns that the geometry implies an interface and uses cell-integrated weights (the analytic integral of 1/k_z over the cell) for pupil cells that straddle the critical radius. The support is never clipped at n_s: clipping loses 9–27 % of the collected power on 512²–64² pupil grids [Rev].
  6. Near-interface emission-rate (LDOS) and dipole-orientation effects are vectorial and v2 (cap-32); the validity check reports an *info* note for emitters within ~λ of an interface. Supercritical collection itself is modelled, so no SAF warning is needed.
- **Radiometry.** Fields are in √(photons·µm⁻²) per mode; mode and spectral weights are power weights that include quadrature weights. The PSF of an emitter integrates to its **collection efficiency** at every tier. Tests (§11.2): the index-matched power test ∫PSF = (1 − cosθ_max)/2 for an isotropic scalar emitter (≈0.31 for NA 1.4 in oil; 0.3067 measured [B05 §3.5, Rev]); the analytic *layered* scalar collection efficiency (s-polarised transmission, flux in glass) of 0.709 at NA 1.45, 680 nm, water/glass, emitter at the interface, and 0.575 at 0.1 µm depth [Rev]; convergence of the collected power to 10⁻³ from 64² to 512² pupil grids; a vectorial comparison with psf-generator; finite gradients with respect to n_s across a sweep through NA = n_s.
- **Vector producers (P = 2) and the scalar reduction (P = 1).** At P = 2, the sparse coherent producers (Mie, dipole, Born) give the vector scattered field. For each pupil sample and illumination mode it is expressed in the meridional s/p basis and mapped through the aplanatic objective: s stays azimuthal and p becomes radial (the lens rotation), with √cosθ apodisation, or with the stack's t_s and t_p in a layered medium. The result is a pair of image-side Jones components. Backgrounds and references carry Jones vectors in the same way (the layered reference uses r_s, r_p, t_s and t_p), and Detect squares after the analysers (§5.5a). Born uses the transverse projector (I − k̂k̂ᵀ) on the incident polarisation. At P = 1, the scalar fast path, producers use the lens-rotated, analyser-projected pupil field **S∥ = S₂cos²φ + S₁sin²φ** (x-polarised incidence and an x analyser, or the co-polarised component when no analyser is declared), as holopy's `MieLens` does. For on-axis incidence in a homogeneous medium this is exactly the x component of the P = 2 field. The cross-polarised part (S₂ − S₁)sinφcosφ is dropped at P = 1, and its energy fraction is reported. The object-space Cartesian x component (S₂cosθcos²φ + S₁sin²φ) is *not* used: it differs by 2.8–6.2 % pupil rel-L2 at NA 0.8 and 4.7–12.2 % at NA 1.2 for polystyrene of radius 0.05–0.5 µm [Rev], comparable to the fit accuracy M3 targets. The P = 1 ↔ P = 2 edge (tilted modes, layered stacks, the dropped cross-polarised part) is tabulated in CI, and a CI test compares P = 1 against holopy `MieLens`. `gx.imaging.Coherent(polarization="auto")` runs at P = 2 whenever the Chain holds a polarisation-dependent element (a Jones pupil modifier, an analyser, a birefringent sample or a polarisation-resolved output) and at P = 1 otherwise; a `Fidelity` sets `"vector"` at `accurate` and `reference`.

### 4.2 Axes and tensor layout

Sampled coherent tensors have **fixed rank 7: `[B, A, M, L, P, Y, X]`**. In the angular domain the last two axes are `Ky, Kx` on a `FreqGrid`, or a single flattened `K` over pupil-support samples. Size-1 axes broadcast at zero cost (stride 0). Kernels never need ellipsis indexing, and adding physics never adds an axis.

| Axis | Meaning | Reduced where |
|---|---|---|
| B | images in the batch (the leading dimension of every per-image or per-object field) | never (output) |
| A | acquisition index: *separate frames*: time frames T × focus steps × SIM angle·phase × LED index × pre-split channels; flattened, with `AcqIndex` metadata | never (output, unflattened into T and C) |
| M | mutually incoherent modes summed within one exposure: source nodes, unpolarised states, dipole basis states, exposure sub-steps, speckle realisations | Detect, once |
| J | mutually coherent plane waves inside one mode (SIM beams; sample-side references with `path="transmitted"`/`"reflected"`); only in `PlaneWaves` | inside the mode |
| L | wavelength bins with power weights (source × filter × QE, including quadrature) | Detect, once |
| P | polarisation components: 1 (scalar), 2 (Jones, transverse), 3 (Cartesian) | Detect (‖·‖²) |
| N | object/emitter slots, padded to a bucketed N with `presence` (§6.1) | inside Interact/Emit (coherent sum) or Detect (incoherent) |
| K | pupil-support samples (a static count, fixed with the grids) | scatter into the k-grid |
| Y, X | space or frequency; last and contiguous for `fft2` | — |
| C | detector channels (after Detect; colour, polarisation camera, multi-camera) | output |

**Which carriers are rank 7.** Only `Field.scattered` is rank 7 (`[B, A, M, L, P, Y, X]`, or `[B, A, M, L, P, K]` on pupil support). The others have fixed, documented shapes that broadcast over B and A: `PlaneWaves.amplitude [B|1, A|1, M, J, L, P]`; `EmitterSet` fields `[B, A|1, N, …]`; `EmitterDensity.data [B, A|1, S, Z, Y, X]`; `Irradiance.data [B, A, L|1, Y, X]`; spectra and media `[B|1, L]`. `ObjectSpectra.evaluate` returns the one coherent intermediate with an N axis, `[B, N, A, M, L, P, K]`, whose N (dim 1) is summed before the scatter into the k-grid, so no consumer permutes. An M0 carrier-shape conformance test asserts that every carrier broadcasts over B and A (and L where it has one).

Output frames are **`[B, (T,) C, H, W]`**, channel-first, with non-time acquisition axes flattened into C by default; callers choose their own layouts. Carriers, data objects and elements are frozen dataclasses registered as pytrees. They are not `torch.Tensor` subclasses, and named tensors are not used [B07 §7.2].

### 4.3 Grids and sampling bookkeeping

In a Pipeline, all grids are **static objects** made of Python numbers and derived from the envelope (§6.4); an eager element call derives them the same way from the envelope of its own inputs. Every size is rounded up to a 2·3·5·7-smooth number, because prime FFT sizes cost up to 1.9× [B07 §2.3]. All wavelengths share one spatial grid and one k-grid; only the per-λ pupil radius NA/λ changes. This avoids Chromatix's per-λ dx leak [B02 §2.1].

The grid types are `Grid2D` (a sampled plane: static smooth shape, isotropic spacing in µm, origin at the top-left FOV corner minus padding), `FreqGrid` (its λ-independent FFT dual, optionally carrying a static `PupilSupport` index list whose length is K), `PupilGrid` (direction space u = λ₀f⊥, a disc |u| ≤ u_max), `PatchGrid` (R×R ROIs whose per-object origins are tensors) and `VolumeGrid` (the dense interaction grid). Their definitions are in [Appendix D](#appendix-d--reference-listings).

The rules below are pure functions of the envelope (`gx.sampling.*`, L0). A Pipeline records their results in a **`SamplingPlan`** (pass P5), where every decision keeps its criterion and the envelope entry that drove it; an eager element call applies the same rules to the envelope of its own inputs:

| Decision | Rule | Source |
|---|---|---|
| Detection spacing d_det | pitch/(Mag·s) with the smallest integer s such that d_det ≤ λ_min/(2·max\|u_a − u_b\|) over all coherently interfering pairs: 2NA for scattered–scattered light, \|u_ref\| + NA for an image-side reference with the scattered light. That is λ_min/(4·NA) without a reference (squaring a field doubles its band) and ≈λ_min/(8·NA) for off-axis holography at \|u_ref\| = 3NA | [B05 §5] |
| Interaction spacing d_int (dense solvers) | d_int ≤ λ_min/(2·ν) with ν = n_m when `interaction_band="full"` (the full propagating band, as `accurate` sets) or ν = NA_obj + NA_ill (+10 % margin at `standard`) when `"optical"`, where band-limited voxelisation is used and the dropped band's energy is estimated from form factors at the envelope and reported. The report also gives the in-band attenuation exp(−2π²σ_r²q²) of the raster prefilter at the imaged band edge (§4.6). **Separate from detection Nyquist**; the field is cropped to the pupil before detection | [M-J physics] |
| Transduce product grid (dense emitter densities) | The lateral grid on which Transduce forms rate = ρ·I_exc has Nyquist ≥ f_exc,max + 2NA/λ_em,min, i.e. spacing ≤ λ/(8NA) for SIM at the maximum pattern frequency. The density is rasterised at that spacing (σ_r scaled to it), the product is formed there, OTF_z and the pixel MTF are applied, and the result is decimated to the detection grid. A grid pinned coarser is a validity error. The rule covers every structured excitation (SIM, speckle, `IlluminationNuisance` fringes, TIRF-SIM). Formed on the detection grid instead, the modulated SIM component (all the super-resolution information) errs by 89–95 % [Rev] | standard SIM theory |
| Padding | pad ≥ max over the pupil support of the stationary-phase ray offset \|∇_{k⊥} arg P(k)\| of the fused pupil phase (this covers defocus in the immersion medium and Gibson–Lanni depth aberration) + 2λ/NA, plus the coherence width 0.61λ/NA_c and object margins. For a homogeneous, index-matched medium it reduces to \|Δz\|_max·tanθ_max + 2λ/NA. (With a layered medium, tanθ at sinθ = min(NA, n_s)/n_s is infinite, so the tanθ form is never used there.) | [B05 §5] |
| ROI (sparse emitters) | R·d ≥ 2(stationary-phase offset, as for padding, + 2λ_max/NA) + margin. A 32² ROI where 64² was needed gave **5.7 %** rel-L2 error with energy conserved [M-D] | correctness rule |
| Pupil grid for MFT | N_p = 64–128; period condition λ/du > ROI extent + support | [M-A] |
| Pupil support K | samples with \|f\| ≤ NA/λ_min on the detection `FreqGrid`; never clipped at the sample index (§4.1) | [M-J eng] |
| Slices (multislice) | **One default criterion:** Δz ≤ λ_m/2 and per-slice phase k₀·Δn_max·Δz ≤ 0.5 rad. The §5.8 predicate Δz ≤ λ_m is only the warn limit for a *pinned* dz. Tolerances come from the ladders' dz sweep (λ_m/2, /4, /8); with the obliquity-corrected march λ_m/2 gives 0.4–2.1 % against 0.2–0.6 % at λ_m/8 [Rev]. Empty slabs are merged using bounding boxes | [B04 §2.5] |
| Strata (dense fluorescence) | Δz ≤ λ/(2(n_s − √(n_s² − NA_s²))) with NA_s = min(NA, n_s) on the sample side | [B05 §5] |
| Source nodes | per preset (§5.4); convergence table per σ_coh | [M-C] |
| Wavelength bins | N_λ ≥ 4·ΔOPD_max/l_c, l_c = λ²/Δλ | [B05 §5] |
| Mie n_max, N_θ | n_max = x_max + 4x_max^{1/3} + 2 from the radius envelope (static shape); N_θ ≈ 1024 | [B04 §2.8] |
| Stage planes (optical stages, §5.11) | spacing ≤ half the finest device pitch and ≤ λ/(2 sin θ_max) for the angles present at the plane; pads ≥ the spread d·tanθ_max over each gap plus the element's support; each sampled transfer function band-limited (BL-ASM) | sampling theorem; BL-ASM |
| Chunk sizes (reductions over M, L, emitters) | chosen from free memory when the Pipeline is built, per (batch bucket, device), quantised to a power of two, **recorded in the Pipeline JSON and `out.meta` and included in the Pipeline hash**; a frozen Pipeline (`from_json`) reuses the recorded values, and later changes in free memory only warn (§5.4, §6.3) | determinism |

**Pixel integration.** The expected intensity is multiplied in the frequency domain by the pixel transfer function p²·sinc(f_x p)·sinc(f_y p) and then sampled at pixel centres. This is exact for an intensity band-limited below the grid Nyquist, up to the periodic wrap that the padding and ROI rules already bound. It is folded into OTF_z on the strata path, and costs one rFFT pair per image on the dense coherent path and per R×R ROI on the sparse path (applied on the fine grid before decimation when s > 1). Plain s×s sum-pooling is a midpoint rule, not box integration: with s = 1, which the rule above picks whenever the camera pitch already satisfies λ/(4NA) (as in example (b)), it is point sampling with no pixel MTF, 8.5–9.1 % rel-L2 from exact box averages in focus with an 8–10 % peak excess; s = 3 gives 0.8 % [Rev]. The Gaussian tier is erf-integrated, so sum-pooling would also break the consistency contract (§5.9). Continuous pixel centres arrive in M2 (cap-26). The intensity spectrum, multiplied by the pixel MTF, is sampled by an MFT at the camera's pixel centres, which may follow a learnable magnification, pixel pitch, rotation and sub-pixel offset while the internal grid stays static (it comes from the envelope of those values).

### 4.4 Carriers: how light is represented

The incoherent carriers are frozen at M0; the coherent ones are drafted at M0 and frozen at M3 exit, after the optical-theorem, direct-summation and Fresnel-reference tests (principle 3).

```python
@dataclass(frozen=True)
class PlaneWaves:  # analytic light: illumination modes, sample-side references, reflections
    amplitude: Tensor  # complex [B|1, A|1, M, J, L, P]   (J mutually coherent waves per mode)
    u: Tensor  # real    [B|1, A|1, M, J, 2]      n·sinθ(cosφ, sinφ); |u| > n ⇒ evanescent
    travel: int  # +1: toward the detection objective; −1: away
    z0: float | Tensor  # plane at which `amplitude` is referenced (µm)

    # exact evaluation anywhere; wl [B|1, L]
    def at(self, points: Tensor, n: Tensor, wl: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class Field:  # coherent carrier = analytic background ⊕ sampled scattered part
    # complex [B, A, M, L, P, Ky, Kx] (angular) or [..., Y, X] (space); None ≡ 0
    scattered: Tensor | None
    # Protocol (at, propagate, filter_by): PlaneWaves, GaussianBeams (SphericalWaves: M4 stretch)
    background: Background | None
    grid: Grid2D | FreqGrid
    domain: Literal["space", "angular"]
    z_ref: float | Tensor  # reference plane of `scattered` (µm)
    travel: int  # +1 | −1
    # [B|1, A|1, M, 2] δk of a dense march's demodulated frame (see obligations)
    residual_tilt: Tensor | None
    n_medium: Tensor  # [B|1, L] complex, homogeneous medium at z_ref
    spectrum: Spectrum  # wavelengths [B|1, L] (µm), power weights [B|1, L]
    mode_weights: Tensor  # [B|1, A|1, M]
    basis: Literal["scalar", "jones", "cartesian"]  # P = 1 | 2 | 3


@dataclass(frozen=True)
class ObjectSpectra:  # sparse coherent producers, evaluated lazily by the consumer
    evaluator: str  # "mie" | "dipole" | "born_ff" | plugin name
    # SoA [B, 1, N, ...]: radius, m(λ) = n_p/n_host, α_e, α_m, form-factor params
    params: Mapping[str, Tensor]
    position: Tensor  # [B, A|1, N, 3] µm (A carries frames)
    presence: Tensor  # [B, A|1, N] ∈ [0, 1]
    travel: int  # +1 | −1

    def evaluate(self, f: Tensor, incident: PlaneWaves) -> Tensor:
        ...
        # f: [K, 2] cycles/µm → complex [B, N, A, M, L, P, K], position phase included.
        # The one coherent intermediate with an N axis: N (dim 1) is summed before the scatter into the k-grid.


Contribution = Field | ObjectSpectra | PlaneWaves  # Interact returns tuple[Contribution, ...]


@dataclass(frozen=True)
class EmitterSet:  # incoherent point sources; never voxelised by default
    position: Tensor  # [B, A|1, N, 3] µm (A carries frames)
    photons: Tensor  # [B, A|1, N] expected emitted photons per exposure
    presence: Tensor  # [B, A|1, N]
    species: Tensor  # [B, 1, N] int → emission Spectrum table
    dipole: Tensor | None  # [B, A|1, N, 6] second moments ⟨μiμj⟩; None = isotropic
    id: Tensor  # [B, 1, N] persistent identity (tracking labels)


@dataclass(frozen=True)
class EmitterDensity:  # dense labels: photons per voxel per species on a VolumeGrid
    data: Tensor  # real [B, A|1, S, Z, Y, X]   (A: SIM phases, time frames)
    grid: VolumeGrid


@dataclass(frozen=True)
class Irradiance:  # incoherent currency on the detection grid
    data: Tensor  # real [B, A, L|1, Y, X]  photons per exposure per detection pixel
    grid: Grid2D
```

**Why the zero order is analytic (measured).**
- *Off-grid tilts.* A tilted plane wave sampled on a periodic grid and imaged through a soft pupil gives **8–10 % rms image error** (up to 90 % max) from seam ringing. Snapping it to the grid instead costs 6–8 mrad of angle at a 25.6 µm FOV and makes tilt, condenser NA and LED positions non-differentiable [M-A].
- *Weak interference.* Forming |E_b + E_s|² in complex64 errs by **8.7 % / 1.4 % / 0.19 %** at peak contrast 5·10⁻⁶ / 5·10⁻⁵ / 5·10⁻⁴; the explicit cross term with an analytic E_b errs by ≈2·10⁻⁷ at all three [M-A]. This matters for sparse producers, which never enter a march.
- *Pupil filtering by evaluation.* A plane wave passes the pupil as E_b ← P(u_b)·√cosθ_b·E_b. Darkfield stops (P = 0), phase rings (P = 0.25e^{iπ/2}), iSCAT reference attenuation and DIC bias are exact and alias-free.
- *Optical theorem.* Forward scatter interferes with the analytic background, so extinction shadows come out right. This doubles as a CI test.

**Operator obligations (a closed set).** Only these v1 operators touch a background:
- Source creates it.
- `layered.fresnel` (Interact) creates the analytic zero-order reflection and transmission of a `LayeredMedium` as `PlaneWaves`; Source never does.
- Propagation multiplies its amplitudes by e^{ik_zΔz}, subject to the evanescent rule of §4.1.
- The projection screen maps (E_b, E_s) ↦ (E_b, (t − 1)·E_b|grid + t·E_s) (the scattered form; its cost is negligible). Because (t − 1) is compactly supported, the sampled term has no seam; the screen's scattered spectrum carries the 1/cosθ_out factor (§5.1).
- **Dense marches** (multislice and third-party dense solvers) march the per-mode *total* field in a **residual-demodulated frame**, through `ops.propagate(..., residual_tilt=δk)`. Each mode's tilt is split as k_b = k_grid + δk; the on-grid carrier is exact and periodic, so there is no seam; the pupil is evaluated at q + δk; and E_s = v − carrier is used at detection. With the obliquity correction of §5.1 the step keeps v̂ in k-space: v̂ ← H(q+δk)·(v̂ + O(q+δk)·FFT[(t − 1)·v]), still two FFTs per slice. Tilt, condenser NA and LED positions stay continuous and differentiable. Against a 4×-grid truth, on a tight `accurate` grid with an off-grid Köhler tilt at NA_ill 0.38, the error is 5.69·10⁻⁴ against 5.80·10⁻⁴ for a march carrying the background analytically; full demodulation gives 8.3·10⁻³ (it widens the band) and a naively sampled tilt 1.39·10⁻² [Rev].
- The **scattered-field march**, Ê_s ← H·(Ê_s + O·FFT[(t_s − 1)(E_b(z_s) + E_s)]) with E_b(z_s) evaluated analytically at each slice, stays as an opt-in multislice mode. A deterministic rule selects it when the bound on dense phase contrast is small (φ_max ≲ 0.05 rad) or the zero order is blocked (darkfield). The reason is precision: recovering E_s = v − carrier from a complex64 total-field march has an absolute floor of ≈10⁻⁵·|E_b| (scattered-field error 8.7·10⁻⁵ / 4.2·10⁻³ / 4.1·10⁻² / 0.43 at a total phase of 0.5 / 10⁻² / 10⁻³ / 10⁻⁴ rad, against ≲2·10⁻⁵ for the scattered form) [Rev]. The scattered form costs 1.39–1.46× per march (§8.1).
- The Objective evaluates the pupil at u_b (and at q + δk for demodulated fields).
- Detect forms |E_b|² + 2Re(E_b*E_s) + |E_s|² term by term, and adds image-side references (§4.5).
- An `IlluminationEnvelope` scales that total coherent intensity by |g(r)|², equivalently E_b and E_s before squaring (§4.5).

Sparse producers (Mie, dipole, Born form factor) emit scattered spectra only and never see the background. Third-party dense plugins receive the demodulated frame from `ops.propagate` and have **no background obligation**; there is no `to_sampled()` snapping fallback. Their conformance item is tilt equivariance (§7.4).

**Travel tags.** Every contribution carries one `travel: int` (+1 toward the detection objective, −1 away), spelled identically in every carrier, and Interact returns `tuple[Contribution, ...]`. The Objective consumes `travel == +1`. In transmission, forward-scattered light travels +1. In epi geometries (iSCAT, reflection), the illumination travels −1, and backscatter plus the analytic Fresnel reflection of the coverslip travel +1. v2 layer stacks will consume `travel == -1` from the same tuple, with no type change. Forward-only solvers (projection, multislice) declare `Capabilities.ports = {"+z"}` (forward relative to the illumination), and the Pipeline's validity check refuses them where backscatter is needed (the planner never chooses them there). A single `direction` flag cannot hold transmitted and reflected contributions at once [M-J eng], and a carrier with separate plus/minus slots would make contradictory states (a −1 contribution in the +1 slot) representable.

### 4.5 Light: sources, spectra, coherence

- **`Spectrum(wavelengths[B|1, L], weights[B|1, L])`** (a per-image λ is simply a `[B, L]` tensor). Constructors `Spectrum.line(λ)`, `.band(center, fwhm, bins="fidelity")` and `.table(λ, S)`. Weights are power per bin, including Δλ. Excitation and emission spectra are separate objects (Stokes shift).
- **`Polarization`**: `linear(angle)`, `circular(±1)`, `jones(e)`, `stokes(s)`, `unpolarized()`. Unpolarised light is **two incoherent modes on M**. Partial polarisation uses weighted modes from the eigendecomposition of the coherency matrix. A source's state is the one after the condenser optics (polariser, compensator): it is defined in the condenser pupil and mapped onto each plane-wave mode's s/p basis by the aplanatic condenser. Per-frame states (`gx.acq.PolarizationStates`) express compensator sequences such as LC-PolScope's. A Jones-on-Stokes contraction is a v1.x reduction option.
- **`AngularSource`: the one illumination abstraction.** It is the cross-spectral density at the condenser (or objective) pupil in Wolf's coherent-mode form, W = Σ_g w_g A_g ⊗ A_g* [B05 §1.1]. There are four constructions; the fourth is not part of that pupil cross-spectral density but a separate analytic term added in Detect:
  1. *Discrete coherent groups*, each a list of plane waves `(amplitude, u)` that interfere: a plane wave (1 group × 1 wave), a SIM frame (1 group × 2–3 waves, frames on A), TIRF (1 wave with |u| > n_sample, so k_z is complex), an LED (1 wave; finite size as a small incoherent cluster), an in-line or reflected reference that traverses the sample (`ReferenceBeam(path="transmitted"|"reflected")`, an extra coherent wave in every mode, used by in-line holography and iSCAT).
  2. *Spatially incoherent density S(u, λ)*: a Köhler disc or annulus, Rheinberg, custom masks. The source element samples it into modes with its `nodes=` strategy. Node positions are continuous functions of the source geometry (so the condenser NA is differentiable) and the weights come from a soft source edge.
  3. *Coherent continuous apertures* (shaped illumination, focused beams, light sheets, speckle): a pupil function A(u) sampled on the illumination k-grid. Version 1.0 offers the analytic Gaussian sheet, and shaped illumination at scalar, moderate NA with the 1/√cosθ focusing rule, from illumination stages or `gx.light.Shaped` (§5.11). Vectorial focusing of high-NA beams is v1.x (cap-25). Gaussian beams are the analytic case: their pupil function is a closed-form Gaussian, so they implement the `Background` protocol (`GaussianBeams`: exact evaluation anywhere, the paraxial closed form for waists ≳ 2λ and angular-spectrum quadrature below, analytic propagation and pupil filtering) and serve as illumination and as sample- or image-side references (M3). Diverging references from point sources (`SphericalWaves`, via the Weyl expansion) are the other analytic case, an M4 stretch goal for lensless holography.
  4. *Image-side references* (`ReferenceBeam(path="image")`): the reference of off-axis DHM/QPI, which bypasses the sample and the objective (Mach–Zehnder) and is tilted on the image side. It is a separate analytic plane-wave term that bypasses Interact, P(u) and √cosθ and is added to the image field in Detect's explicit-interference sum. Its direction is given in image-space units and converted to object-space frequency with the magnification, u_ref = Mag·n_img·sinθ_img (≈2.1 for 40× at 3°), with **no |u| ≤ n bound and no evanescent semantics**. Sideband separation needs |u_ref| ≥ 3NA, which a reference passing the pupil (|u| ≤ NA) can never reach, and the detection spacing rule of §4.3 then gives ≈λ/(8NA). The `OffAxisHolography` preset uses `path="image"`; a sideband-separation test pins it.
- **Presets** are thin constructors: `PlaneWave(wavelength, direction|na=(NAx, NAy), polarization, tilt, irradiance)`, `Koehler(spectrum, condenser=Disk(NA)|Annulus(NA_inner, NA_outer)|Custom(fn), irradiance)`, `LEDArray(positions, distance, size, spectrum, irradiance)` with `LEDArray.grid(shape, pitch=, distance=, spectrum=)` and `.select(idx)`, `ReferenceBeam(amplitude, phase, tilt, path="transmitted"|"reflected"|"image", profile="plane"|"gaussian"|"spherical")`, `GaussianBeam(wavelength, waist, focus, direction, polarization, power)`, `Evanescent(angle, n_glass)`, `SIMBeams(period, angles, phases, modulation)`, `GaussianSheet(waist, axis, focus)`, `Shaped(pupil_field=… | field=…, wavelength, irradiance)`, `Uniform(irradiance)`, and `IlluminationNuisance` (weak extra plane waves whose amplitude and phase the front end may sample or learn: etalons, ghosts, dust fringes; these stay analytic). Every source takes an `irradiance` in photons·µm⁻² per exposure at the sample (per image when `[B]`).
- **Envelopes.** `IlluminationEnvelope` g(r) (gradient, non-uniform illumination) multiplies the *total* coherent intensity, |E_b|² + 2Re(E_b*E_s) + |E_s|², by |g(r)|²; equivalently, it scales E_b and the sampled E_s before squaring, because the scattered field is linear in the local incident field. A detection-side vignetting V(r) multiplies the total intensity, including incoherent `Irradiance`. Both are valid when the envelope varies slowly on the PSF scale; a future DeepTrack2 integration maps its `IlluminationGradient` this way (Appendix F.8). Scaling only the background and the cross term would leave darkfield (I = |E_s|²) insensitive to the envelope; a darkfield-under-vignetting test pins the rule.
- **Two-stage physics (Transduce).** An excitation source (any of the above at λ_ex) is evaluated at emitter positions with `PlaneWaves.at(points)`, which is exact, including evanescent decay and SIM interference. For dense densities it is evaluated on the *product grid* of §4.3, whose Nyquist resolves f_exc,max + 2NA/λ_em; on the detection grid the frequency-shifted SIM content would be lost or aliased. It then goes through a pointwise source map (linear in v1: rate ∝ σ_abs(λ_ex)·QY·I_exc; saturating and I² in v2) to per-emitter photons, and the emission stage follows with its own spectrum. This one mechanism gives TIRF, linear SIM, analytic light sheets and structured widefield at minimum fidelity.
- **What is not a type.** Mutual intensity (4-D) is not a type [B02 §2.2]. "PSF", "OTF" and "TCC" are derived views (cached intermediates) of source → pupil → detect, not core abstractions.

### 4.6 Sample: how matter is represented

**Canonical form: continuous, physical, resolution-free, already resolved.** The sample is described by **object sets** (L2 data objects), one population per set; the optional L4 container `gx.Sample` maps population names to object sets plus the environment. Each object set holds one primitive kind as a struct of arrays over `[B, T|1, N]`: a leading batch, an optional frame axis, and object slots with a presence mask. Each object is geometry × material × labeling in a shared world frame.

The values are whatever the front end resolved; gradix never samples them. Voxel grids, projected maps, sphere lists and k-space spectra are *lowered views* that a solver requests with a grid, a wavelength set and a band-limit policy [B06 §0]. A voxel tensor becomes "the sample" only when the front end supplies one.

```python
class ObjectSet(gx.DataObject):  # fields common to every object set (schema roles in §6.1)
    position: Tensor  # [B, T|1, N, 3] µm, world frame (§4.1)
    # [B, T|1, N, 4] unit quaternions (w, x, y, z), local to world; None = identity
    rotation: Tensor | None = None
    presence: Tensor | None = None  # [B, T|1, N] ∈ [0, 1]; None = all present
    # complex n per object [B, 1, N] or a dispersion model (below)
    material: Material | Tensor | None = None
    labeling: Labeling | None = None  # fluorescent label density on the object (below)
    # gx.Parent("cells", index [B, 1, N] int, priority=1, compose="over")
    parent: Parent | None = None
    id: Tensor | None = None  # [B, 1, N] int64 identity for tracking labels


# L4 convenience: gx.Sample(populations: Mapping[str, ObjectSet], environment=gx.env.Homogeneous(n=1.33))
```

- **Object sets (v1).**
  - `Emitters(position, photons, emission, dipole=gx.dipole.Isotropic())`: point emitters; `photons [B, T|1, N]` expected photons per exposure. Never voxelised.
  - Coherent/volumetric primitives, each with its own shape fields:
    - `Spheres(radius)`;
    - `LayeredSpheres(radii [B,1,N,n_layers], materials)` (Mie-capable);
    - `Ellipsoids(semi_axes)`;
    - `Capsules(length, radius)` and `Cylinders(length, radius)`: solids of revolution, form factor by 1-D quadrature;
    - `Boxes(size)`;
    - `Gaussians(sigma)` (anisotropic, additive semantics).
  - `gx.SDF(fn, params, bbox)`: the escape hatch that also covers CSG. It has the signature `fn(x_local, **params) -> sdf`, and `params` are tensors `[B, T|1, N, …]` that receive gradients.
  - `gx.Occupancy(fn, params, bbox)`: a boolean or soft `inside(x)`, the form in which many users (DeepTrack2's among them) write custom scatterers; `hard=True` gives a hard mask. Its gradient quality is reported as `zero` (hard) or `exact` (soft; the softened shape itself is tagged `approx` in the view's exactness); masks come from occupancy, distance maps from an EDT of the rasterised occupancy.
  - `gx.Voxels(values, spacing, quantity="n"|"dn"|"density")`. The default `"n"` is the natural form of a tomogram. `"dn"` is relative to the host medium's n(λ). Arrays from other simulators (e.g. DeepTrack2 volumes) pass through here.
  - `gx.JonesScreen(retardance, orientation | retardance_vector, phase=None, amplitude=None, spacing, position=None)`: a thin anisotropic sample given directly as maps, measured or learnable, by default centred on the field of view at the focal plane. The retardance vector (δcos2θ, δsin2θ) is the well-conditioned form for fitting: it has no wrap at θ = ±π/2 and no sign ambiguity at δ = 0.
  - `gx.NeuralField(module, quantity, bbox)`: a torch `nn.Module` of local coordinates. Its parameters receive gradients through the lowering (below).
  - `gx.Texture` (below).
  - Meshes (winding number → SDF, polyhedron form factor) are v2.
  - `@gx.geometry(kind="sdf"|"occupancy", bbox=...)` turns `def bacillus(x, *, length, radius, bend): ...` into a registered object set whose keyword arguments become schema fields. A porting tutorial converts DTGS172's Bacillus to both forms.
- **Hierarchies.** A cell with a nucleoid, droplets and a texture is several populations linked by `gx.Parent(population, index)`:
  - a child's effective presence is its own presence × its parent's;
  - materials compose by priority (below);
  - label renderers take `parts=` (e.g. `InstanceMask("cells", parts="all")` masks the cell and its children as one instance).

  Where children go inside a parent is the front end's decision; gradix only renders the result.
- **`gx.Texture` is resolution-free and keyed.** A texture is a sum of `features` (default 512) random Fourier features per object instance:
  - wavevectors are k = k̃/corr_length, with k̃ drawn from the normalised spectral density of the chosen family (e.g. `"matern32"`);
  - phases are uniform;
  - amplitudes are ∝ std·√(weights).

  A texture adds its Δn to its parent's material (it is a density-like quantity), so it needs no `compose=`. The random draws are a **pure function of the per-object integer `key [B, 1, N]`** through the counter hash (§6.3), so the texture is identical on any grid, at any fidelity and through any microscope. It is differentiable in `corr_length` and `std`. It is Gaussian only as `features` grows, and this approximation level is declared. White noise sampled on the render grid would change with the grid, and so with fidelity and microscope. A test renders the same inputs at two fidelities and through two microscopes and requires texture correlation > 0.99 on a common grid. The front end chooses the key; a different key is a different texture realisation.
- **Capabilities** are methods that lowerings, elements and the planner introspect on each object-set kind. `bbox` and `sdf` are mandatory for volumetric kinds; the rest are optional.

| Capability | Meaning | Enables |
|---|---|---|
| `bbox()` | [B, T\|1, N, 2, 3] world box, excluding blur margin | culling, slice compaction, ROI and padding |
| `sdf(x)` | exact or first-order signed distance | default occupancy (SDF ramp), masks, distance maps |
| `occupancy(x, σ)` | band-limited, differentiable occupancy (closed-form blurred ball for spheres) | voxel views with exact gradients |
| `form_factor(k)` | analytic Fourier transform of the indicator | Born form factor, OTF fluorescence, projection via the k_z = 0 slice |
| `project(xy)` | analytic chord length / column density | projection screen |
| `mie_spec()` | radii and indices per layer | Mie, dipole |
| `sample_points(n, where)` | deterministic volume or surface quadrature points | surface labeling, bead-volume quadrature |

- **Materials.** `n(λ)` is complex. `gx.Material` offers `Constant`, `Cauchy`, `Sellmeier`, `Tabulated`, `DrudeLorentz` (metals) and `DryMass` (n = n_m + (dn/dc)·C, dn/dc ≈ 0.18–0.19 mL/g). A plain tensor is a constant index. Model parameters may be per object (`[B, 1, N]`) and require gradients. With few distinct materials (n_mat ≲ 8), the lowering emits `OccupancyChannels[n_mat]` once, and dispersion costs n_mat multiply-adds per voxel per λ. When every instance has its own index, a `BakedContrast` channel is used instead. Uniaxial materials, `gx.Material.uniaxial(n_o, n_e, axis)` with the optic axis in the object's local frame, are thin Jones samples in 1.0 (next item); tensor-ε volumes (polarised multislice) are v2 (cap-34).
- **Thin Jones samples.** In the projection element, uniaxial materials add a retardance δ = k₀∫(n_e,eff − n_o)dz along the optic axis projected onto the transverse plane (the slow axis for δ > 0) to the isotropic phase and absorption, where n_e,eff follows the angle between the optic axis and z. The retardance vector (δcos2θ, δsin2θ) is integrated along each ray: exact when the slow axis does not change along the ray, and first order in δ where differently oriented layers overlap (§5.8). The screen is a 2×2 Jones transmission at P = 2, and `gx.JonesScreen` supplies one directly.
- **Labeling.** `gx.Labeling(emission, density | surface_density | photons, where="volume"|"surface", dipole=...)` puts a fluorophore on an object: a photon density per µm³ of volume (`density`) or per µm² of surface (`surface_density`, its own quantity so `from_si` converts it correctly), or a total photon count per object (`photons`, spread over the volume or the surface as `where` says), per exposure. **One geometry drives both RI contrast and label density**, so the same inputs yield consistent brightfield, QPI and fluorescence images and labels [B06 §5]. Photophysics (blinking, bleaching) is the front end's: it supplies the per-frame photons or states.
- **Environment.** `LayeredMedium(immersion=material | (material, thickness, design), coverslip=(material, thickness[, design]), sample=material)`, where each material is a `gx.Material`, a library entry such as `gx.materials.Oil()`, or an index, is the single source of truth for n_medium(λ). It is read by optics (Gibson–Lanni OPD, Fresnel coefficients, TIRF, the iSCAT reference) and by the sample (host index).
- **Overlaps.** Material quantities use priority "over" compositing, linear in ε: a_m = occ_m·rem; rem ← rem·(1 − occ_m); ε = Σ a_m ε_m + rem·ε_medium.
  - Children override parents.
  - Equal-priority siblings are normalised when Σocc > 1.
  - Density quantities add.
  - `compose="add"` gives additive Δn (DeepTrack2 2.0.2's rule) as an explicit opt-in [B06 §6].
- **Lowered views and exactness.**

| View | Produced from | Exactness | Consumers |
|---|---|---|---|
| `SphereList` / `DipoleList` | Spheres, LayeredSpheres / small primitives | exact / approx (≤ 5 % all-angle: dielectric x ≤ 0.75 with a₁ + b₁, metals x ≤ 0.6; §5.8) | mie, dipole |
| `KSpectrum` (form factor at requested k) | any `form_factor` | exact | born_ff, strata OTF of analytic labels |
| `ProjectedMaps` (OPL, absorbance, column density) | `project`, or a z-sum of occupancy | exact / band-limited | projection, 2-D fluorescence |
| `OccupancyChannels` → `RIVolume` (procedural or materialised) | `occupancy` or `sdf` ramp | **approx (Gaussian prefilter σ_r = c·h)**: ≈3–4 % coherent image error at c = 0.3, ≈7–8 % at 0.5 [Rev] / approx (first-order SDF) | multislice |
| `EmitterDensity` | labelings of volumes and surfaces | **approx (Gaussian prefilter)**: ≈1.3 % fluorescence error at c = 0.3 [Rev] | strata OTF |
| `EmitterSet` | `Emitters`, bead quadrature points | exact | sparse emission |

- **The raster prefilter is an in-band blur, not only anti-aliasing.** At σ_r = 0.3h it attenuates the object spectrum by exp(−2π²σ_r²q²) = 0.64 at the band edge q = 1/(2h) of a `standard` interaction grid. The interaction-band report includes this attenuation, and the tolerance table carries the numbers above.
  - With `raster_oversample=2` (set by `accurate` and `reference`), the attenuation at (NA_obj + NA_ill)/λ is kept ≥ 0.9: h is chosen ≤ λ/(2·2ν), about 2× finer, or the volume is rasterised at s = 2 with σ_r = 0.3h_fine and decimated.
  - In-band Gaussian deconvolution is allowed only for linear consumers (strata OTF, born), where its negative lobes are harmless.
- **Lowering selection.** Each view has an ordered preference list of producers: exact analytic, then band-limited, then approximate. `gx.lower` takes the first one that exists for the object set, and that order already puts non-zero-gradient producers first.
  - A hard threshold (DT's masks give dV/dR = 0 [B06 M]) exists only as the explicit opt-in `gx.Occupancy(..., hard=True)`, which reports geometry gradients as `zero`.
  - The default voxeliser is patch-based and band-limited: the closed-form blurred ball for spheres (dV/dR 4.5239 against 4.5239) and the first-order SDF ramp otherwise (4.525 against 4.524) [B06 §3].
  - As built (M1): a solid's capability methods take its shape fields as one mapping (`sdf(x, shape)`, `occupancy(x, blur, shape)`, `surface(x, blur, shape)`, `reach(shape)`, ADR-42). Surface labels are the blurred surface, whose blur is at least 0.6 sample spacings: sampled at spacing h, a Gaussian of width σ sums to its integral within 2·exp(−2π²σ²/h²) (0.2 % at 0.6h, 34 % at 0.3h). Across planes, each object's density is spread by a Lanczos-3 kernel (normalised, so photons are conserved): plane spacings at the axial Nyquist limit sample the PSF's axial band exactly, so a point between planes is imaged correctly only with sinc-like weights; box or linear weights shift its defocus (up to 14 % rel-L2 for a bead between Nyquist planes). Against a point cloud filling a labelled sphere, the strata path is within 0.8–2.4 % at the defaults and 0.3–0.9 % refined (3× oversampling, 0.1 µm planes). Capsule `length` is tip to tip.
  - It is a custom autograd Function that recomputes in backward, because unfused autograd needs 17.9 GB at 10⁴ spheres [B06 §3.4].
  - **Every data-object tensor the lowering reads is an explicit input of that Function.** Callable lowerings (`gx.SDF`, `gx.Occupancy`, `gx.NeuralField`) receive their parameters explicitly too: `RasterFn.apply(values…, *params)`, with `torch.func.functional_call` in backward, returning a gradient for every parameter. Otherwise they are lowered through a chunked non-reentrant checkpoint over patch chunks. Parameters captured by closure would silently receive no gradient: a neural-field SDF inside a recompute Function got `None` for all its weights while d/dradius was correct [Rev].
  - k-space form factors are used only by linear solvers, because their Gibbs ringing produces negative densities (min −0.10) [B06 §3.2].
- **Representation policy (the answer to "voxels? polygons? geometric objects? any?").**
  - Analytic geometric objects are canonical: they are cheap, exactly learnable and serve every solver through capabilities.
  - Voxels are both a lowering and an accepted input (tomograms, segmentations, learned volumes, arrays from other simulators).
  - Neural fields are voxels with a function instead of a tensor.
  - Polygons (meshes) enter in v2 through SDF and form-factor lowerings.
  - Points stay points: voxelising them loses sub-voxel positions and gradients.
  - "Any representation" is supported through *capabilities and a fixed, small set of views*, not through open-ended conversion search.

---

## 5. Physics and the fidelity mechanism

### 5.1 Operator slots and the element contract

The render skeleton is fixed. A modality is a *recipe*: a choice of source, pupil modifiers, references, excitation and acquisition. It is never a new code path.

```
Source ─► [illumination stages] ─► Interact ─► Objective ─► [detection stages] ─► Detect   (transmitted / reflected / interferometric)
Source(ex) ─► [illumination stages] ─► Transduce ─► Emit (fused Objective pupil) ──────► Detect   (fluorescence; two-stage)
```

On incoherent paths there is no separate Objective stage: Emit evaluates the Objective's fused pupil multiplier (P7, §5.7) per emitter (`pupil_mft`, `global_spectrum`) or per stratum (`strata_otf`) and outputs `Irradiance`; `pupil.per_object` and `pupil.vectorial` are pupil evaluators that the sparse Emit implementations consume.

Each slot has **elements**: public classes that can be called directly (L2), wired into a Chain (L3) or chosen by the planner (L4). They obey one contract:

```python
class Element(Protocol):
    slot: ClassVar[Slot]  # SOURCE | STAGE | INTERACT | OBJECTIVE | TRANSDUCE | EMIT | DETECT
    caps: ClassVar[Capabilities]

    # ---- static: sees data-object and element structure and the envelope, never tensor values ----
    def validity(self, desc, envelope: Envelope) -> list[Violation]: ...
    # knobs are constructor arguments; resolves "auto"
    def configure(self, desc, envelope: Envelope) -> Static: ...
    # optional; only ranks numerically equivalent strategies
    def cost(self, p: Problem) -> Cost: ...
    # ---- run time: pure torch; data-object and element tensors and explicit keys only; no globals ----
    # the slot's carriers, then the environment; what a Pipeline calls
    def forward(self, *inputs, static: Static): ...
    # eager L2 call: configure on the envelope of these inputs
    def __call__(self, *inputs, grid=None):
        ...
        # (or on grid=), then forward


@dataclass(frozen=True)
class Capabilities:
    accepts: frozenset[type]  # views / carriers consumed
    produces: type  # tuple[Contribution] | Field | Irradiance | EmitterSet
    # travel directions filled; forward-only solvers: {"+z"}
    ports: frozenset[str] = frozenset({"+z"})
    polarization: frozenset[int] = frozenset({1})
    linear_in_field: bool = True  # enables mode chunking / compression
    linear_in_sample: bool = False  # enables WOTF/Born-type rewrites (v1.x)
    space_variant: bool = False  # Objective: may depend on field position h
    grad: frozenset[str] = frozenset({"autograd"})  # autograd | checkpoint | adjoint | implicit
    grad_quality: Mapping[str, Literal["exact", "exact-a.e.", "biased", "zero"]] = field(
        default_factory=dict
    )
    # optional metadata; default "exact" (§6.2)
    approximates: tuple[Edge, ...] = ()  # (target, regime predicate, tolerance-table id)
```

**Knobs and calls.** An element's knobs (`dz=`, `nodes=`, `psf=`, `method=`, `obliquity=`, …) are constructor arguments, stored like any other field and part of the structure signature. `configure` resolves the knobs left `"auto"` by the element's default rule on the envelope it is given: a Pipeline's envelope, or in an eager call the envelope of that call's own inputs (or `grid=`). A `Fidelity` never reaches an element; at L4 the planner writes its knob values into the fields each element declares for them (`fidelity_knobs`, §5.6).

**Element catalogue.** Each registry name maps to a public element: `interact.*` → `gx.interact.*` (`interact.mie` is `gx.interact.Mie`); `emit.gaussian` → `gx.imaging.Sprites`; `emit.pupil_mft` and `emit.global_spectrum` → `gx.imaging.PointPSF(method="roi"|"global")`; `emit.strata_otf` → `gx.imaging.Strata`; `pupil.*` → options of `gx.imaging.Coherent` and `PointPSF`; `source.*`/`koehler.*` → `gx.light.*` with their `nodes=` strategy; `transduce.linear` → `gx.excite.Linear`; `sensor.*` → `gx.Camera` noise models. "First in" is the milestone that delivers the element; every row with a milestone ships by 1.0 (M4b rows move to 1.1 under the M6 option).

| Slot | Implementation | Consumes → produces | Notes | First in |
|---|---|---|---|---|
| Source | `plane_waves` | the light element's fields → `PlaneWaves` | plane waves, LED, SIM groups, TIRF (complex k_z), sample-side references (Fresnel zero orders come from `layered.fresnel`) | M0 |
| Source | `gaussian_beam` | the light element's fields → `GaussianBeams` background | analytic Gaussian illumination and references: closed-form angular spectrum, exact evaluation anywhere, analytic propagation; consumed like shaped illumination (§5.11) | M3 |
| Source | `spherical_wave` | a point-source position → `SphericalWaves` background | diverging references for lensless point-source holography (Weyl expansion) | M4 (stretch; else 1.1) |
| Source | `reference.image` | the light element's fields → image-side `PlaneWaves` | off-axis reference that bypasses Interact and the pupil; direction in image-space units (§4.5) | M3 |
| Source | `shaped` | pupil function A(u) or sample-plane field → a coherent group: J plane waves for sparse producers, the sampled incident field for dense ones | shaped illumination from illumination stages or given directly (§5.11) | M3 (dense: M4) |
| Stage | `optics.mask`, `optics.propagate`, `optics.lens`, `optics.aperture` | field or pupil function at a plane → the same at the next plane | thin scalar or Jones transmission (from a device or a map, with pixel aperture and placement), BL-ASM propagation, ideal lens or exact 2f transform, soft stops; in illumination and detection stages (§5.11) | M3 |
| Stage | `optics.nonlinear` | field (all modes) → field | saturable absorber, Kerr medium; `linear_in_field=False` | 1.x (Q22) |
| Source | `koehler.on_support(1\|2)` | S(u, λ) → modes | draft: disc → centroid; annulus → two opposite points *on* the ring | M3 |
| Source | `koehler.quadrature(n)` | S(u, λ) → modes | rings / Fermat spiral; continuous nodes, soft-edge weights | M3 |
| Source | `koehler.grid` | S(u, λ) → modes | exact on-grid Abbe (reference) | M3 |
| Source | `koehler.stochastic(n)` | S(u, λ) → modes | **fitting and gradient estimation only**, never data generation; nodes from the numerics key; biased for nonlinear losses (§6.2) | 1.x |
| Interact | `dipole` | SphereList/DipoleList → ObjectSpectra (±z) | Mie-dressed electric and magnetic dipoles, α_e = i6πa₁/k³ and α_m = i6πb₁/k³ (same cost); this equals Mie truncated at n = 1. a₁ and b₁ already contain radiation reaction, so no further radiative correction is applied; P = 2 or P = 1 | M1 (smoke) / M3 |
| Interact | `born_ff` | KSpectrum → ObjectSpectra (±z) | A = i·k₀²(n² − n_m²)·F(k − k_in)/(8π²k_z) on both Ewald caps; no voxels; valid for φ_max = k₀Δn_max·d_max ≤ 0.25 rad (§5.8); at P = 2 the transverse projector acts on the incident polarisation | M3 |
| Interact | `mie` | SphereList → ObjectSpectra (±z) | fp64 coefficients on GPU; S₁/S₂ on a 1-D θ table; evaluated only on pupil support; P = 2 vector, or P = 1 with S∥ (§4.1) | M3 |
| Interact | `projection` | ProjectedMaps → Field (illumination direction) | screen at the group centre plane z_c, (t − 1)·E_b form, back-propagated; the scattered spectrum is multiplied by 1/cosθ_out by default; optional 1/cosθ_in; at P = 2 birefringent samples (uniaxial materials, `gx.JonesScreen`) give a 2×2 thin Jones screen, and isotropic ones carry the illumination's Jones vector | M4 |
| Interact | `multislice` | RIVolume (procedural) → Field (forward) | **obliquity-corrected march**: v̂ ← H·(v̂ + O·FFT[(t − 1)·v]) in the residual-demodulated frame (§4.4), O = k_m/max(k_z, c_min·k_m) on the propagating band, c_min ≈ 0.2 or a smooth taper beyond the collection band (reported by `explain()`); P·O fuse into one multiplier; free-space BL-ASM between slices (evanescent removed, **not** NA-truncated); scattered-field form opt-in; plain BPM opt-in only; at P = 2 the scalar march runs once and the illumination's Jones vector rides along (isotropic samples do not couple polarisations; tensor ε is v2) | M4 |
| Interact | `layered.fresnel` | LayeredMedium → PlaneWaves (±z) | analytic reflection/transmission of the zero order (r_s, r_p, t_s, t_p) | M3 |
| Interact | `superpose` (C0) | groups → union of contributions | host-index rule (M3); cut-out-and-fill, coupling warnings (M4) | M3 |
| Interact | `multislice.plain` | RIVolume → Field | plain BPM with the NA-truncated inter-slice propagator (`gx.interact.Multislice(obliquity=False)`); explicit opt-in for comparisons | M4 |
| Interact | `inject` (C1) | RIVolume + SphereList → Field | Mie/dipole response to the local field, injected at the exit plane z_p + a | 1.x |
| Interact | `born_slices`, `rytov` (flag), `ssnp`, `mbs` | RIVolume → Field | custom/implicit adjoints | 2 |
| Objective | `pupil.scalar` | contributions (travel +1) → image field | fused: Propagate(z_ref → interface, in n_s)·t(u)·GL(focus)·Aperture(soft)·√(k_z,i/k_i)·Zernike·pixel·rings·stops·DIC; index-matched: Propagate(z_ref → focus)·…·√cosθ (§4.1); at P = 2 with the lens rotation and t_s/t_p on the s and p components | M1 (P = 2: M3) |
| Objective | `pupil.per_object` | ObjectSpectra/EmitterSet → image field | pupil evaluated at each object's field position h (space-variant, sparse path); for emitters, a pupil evaluator consumed by the sparse Emit implementations | M4b |
| Objective | `pupil.vectorial` | 6-basis dipole emission → P = 2 | Richards–Wolf collection with Fresnel t_s, t_p through the stack; a pupil evaluator consumed by the sparse Emit implementations | M4b |
| Objective | `pupil.jones` | contributions (travel +1) at P = 2 → image field | 2×2 modifiers fused with the scalar pupil: `Polarizer`, `Retarder` (waveplates, compensators), `JonesMatrix` (constant or maps, learnable), Jones `DICShear` | M4 |
| Objective | focusing maps, `mask.spatial` | — | vectorial focusing of excitation beams; pinholes | 1.x |
| Objective | `pupil.low_rank` | dense field → image field | space-variant product convolution | 2 |
| Transduce | `linear` | excitation at points → EmitterSet photons (× the per-frame photons or states the front end supplies; an emitter's `photons` is its emission under the `reference` irradiance, 1 photon/µm² by default, scaled by I(r)/reference with optional saturation); on the product grid → EmitterDensity | analytic excitation evaluation; photophysics states come from the front end (§6.7). From M4b the excitation is evaluated as a Cartesian vector (P = 3): interfering beams (SIM, TIRF) keep their polarisation-dependent contrast, oriented dipoles absorb E_excᴴ⟨μ̂μ̂ᵀ⟩E_exc (photoselection), and isotropic ones absorb in proportion to \|E_exc\|² | M1 (vector excitation M4b) |
| Emit | `gaussian` | EmitterSet → Irradiance | σ ≈ 0.22λ/NA (L2-optimal) with z-dependent width; pixel-integrated erf | M0 |
| Emit | `pupil_mft` (sparse ROI) | EmitterSet → Irradiance | per-emitter pupil → MFT (IEEE fp32) → R×R ROI → pixel MTF → \|·\|² → scatter-add | M1 |
| Emit | `global_spectrum` | EmitterSet → Irradiance | per-emitter full-frame evaluation; used when the envelope makes the ROI too large | M1 |
| Emit | `strata_otf` | EmitterDensity → Irradiance | Σ_z irfft2(rfft2(ρ_z)·OTF_z·MTF_pix), exact (linear) or circular over the frame and its margin (periodic: ≈2.5× faster, light near one edge wraps to the other; the `dense_boundary` knob); empty-slice compaction (collapse theorem a); optional turbid slab (Beer–Lambert × depth blur + diffuse background) | M1 |
| Emit | `tabulated` | EmitterSet → Irradiance | torch cubic-spline PSF (SMAP/DECODE/uiPSF import); coefficients may be learnable | 1.x |
| Detect | `square_reduce` | Field(s) + image-side references + Irradiance → Irradiance | Σ_{M,L,P} w·[\|E_b\|² + 2Re(E_b*E_s) + \|E_s\|²]; channel weights W[C, L]; P = 2 channels are projected on an analyser a_c before the square (§5.5a); re² + im² (no abs) | M0 |
| Detect | `pixel.mtf` / `pixel.mft_centres` | Irradiance → expected photons | pixel MTF p²·sinc(f_x p)·sinc(f_y p), then sampling at pixel centres (§4.3) / continuous pixel centres | M1 / M2 |
| Detect | `sensor.{ideal, poisson_gaussian, scmos, emccd}`; `qcmos` | expected → output unit | each with `expected`, `sample(key)` (§6.3), `log_prob`, Fisher information | M0/M1; 1.x |

### 5.2 The interaction ladder

The ladder has three families that share one output currency: point/particle producers, volumetric producers and emitters [B04 §1]. **It is not a total order.** Plain BPM's weak limit is Born × cosθ (k_m in place of k_z: the missing obliquity) with no backscatter [M-J physics]. **Obliquity-corrected BPM, the v1 default, has first Born as its weak limit in the forward port**; against the exact scalar sphere it errs by 0.1–0.8 % for x ≥ 3 and 1.6–5 % for x = 1, where plain BPM errs by 11–15 % at NA 0.8 and 27–39 % at NA 1.2 for x = 1–3 [Rev]. Among the rungs for extended, non-spherical objects, Born remains the only cheap one with a backward port. Dipole and projection are limits in opposite regimes (x → 0, and x ≫ 1 with m → 1). The true nestings are encoded as **approximation edges** (§5.9), which hold the tolerances; the table below states only the assumptions.

| Rung | What it assumes | Natural output | Cost | Differentiability |
|---|---|---|---|---|
| dipole | isolated; Mie-dressed a₁ + b₁ within 5 % all-angle for x ≤ 0.75 (dielectrics) or x ≤ 0.6–0.7 (metals); bare quasi-static α only for dielectrics at x ≤ 0.3 | analytic angular spectrum, ±z | O(N·K) | trivial |
| born_ff | weak object: φ_max = k₀·Δn_max·d_max ≤ 0.25 rad for ≈10 % pupil rel-L2 (the error grows as ≈0.41·φ); single scattering | analytic spectra on Ewald caps, ±z | O(N·K) | analytic in position, size, shape |
| mie | isolated sphere in a homogeneous host | exact S₁/S₂ → spectra, ±z | O(n_max·N_θ) per particle + O(N·K) | autograd through fp64 recurrences |
| projection | thickness ≤ DOF = n·λ₀/NA² (NA → min(NA, n_s)); features x ≫ 1; no diffraction inside the object; with 1/cosθ_out on the scattered spectrum | exit screen, forward | one elementwise product + FFTs | trivial |
| multislice (obliquity-corrected BPM) | forward scattering, moderate angles and Δn; no reflection | exit field, forward | 2 FFTs per slice, sequential | autograd + checkpointing |
| C1 inject (v1.x) | local plane-wave incidence at each particle | exit field | + one small FFT per injection plane | autograd |
| born_slices / rytov / SSNP (v2) | weak / smooth / non-paraxial | exit field (± approx R) | 2–4 FFTs per slice | custom adjoints |
| MBS / Lippmann–Schwinger (v2+) | none beyond discretisation; converges for non-gain media | full field in a padded box | N_it × 3-D FFT | implicit differentiation |

### 5.3 The imaging system: how granular the optical path is

**The optical path is modelled at the granularity of conjugate planes, not optical surfaces** [B05 §3.2, A §5.4]:
1. the **condenser pupil**, as a source distribution S(u, λ, pol), shaped by optional illumination stages (§5.11);
2. the **sample volume** inside its layered environment (Interact);
3. **one objective pupil** in which every modifier of every conjugate pupil plane is fused: soft aperture, collection apodisation (√cosθ, or √(k_z,i/k_i) with the stack transmission t(u) in a layered medium), defocus from each contribution's reference plane to the focal plane (in a layered medium: propagation to the interface plus one Gibson–Lanni term, §4.1), Zernike OPD, pixel pupil, phase rings, stops, knife edges, DIC shear/bias, engineered masks, and Jones elements (2×2 multipliers at P = 2: polarisers, retarders, compensators, Jones DIC);
4. the **image plane**, where magnification is a coordinate scale, followed by optional detection stages (§5.11), detection-path branches (channel splitters, filters) and camera sampling.

Everything is simulated **in object space**. Image-side NA/Mag is tiny, so no image-side Debye modelling is needed. The Objective node therefore expands into Propagate ⊗ Filter(pupil) followed by a basis change onto the detection grid, where Propagate is Propagate(z_ref → focus) in an index-matched medium and Propagate(z_ref → interface, in n_s)·t(u)·GL(focus) in a layered one. Because both are diagonal in angle, the Objective's evaluation fuses them (Pipeline pass P7) into **one complex multiplier per (contribution reference plane, λ, P)**, computed once per call per unique parameter tensor and shared by all modes. Pupils are small, so per-image optics (`[B]` fields) cost little: `[B, L, 1, K]` instead of `[1, L, 1, K]`.

Explicit propagation happens only in four places: inside the sample (Interact); from reference planes to the interface or focal plane (a diagonal phase); lensless configurations (BL-ASM to the sensor, M4); and inside an excitation beam's own frame (light sheets). Relay optics, programmable devices and diffractive networks add explicit propagation in the optical stages before and after the sample (§5.11). Image-side references are added after the pupil (§4.5). Objectives are aplanatic systems plus pupil maps; external ray tracers may supply OPD or Jones maps, so **the pupil is the closure of the objective** [B02 §2.8].

**Beyond diagonal-everywhere.** Field-dependent aberrations, non-telecentric vignetting and light-sheet frames are neither space- nor angle-diagonal [M-J physics]. The Objective contract therefore includes a space-variant member from day one: `space_variant=True` implementations receive the field position h. v1 implements it only on the sparse path (`pupil.per_object`), where it is free: Zernike coefficients become smooth functions of each object's (x, y). The dense path refuses h-dependence with an error pointing to the sparse path or to v2 `pupil.low_rank`.

**Scalar vs vectorial.** Even for isotropic (freely rotating) emitters, the z-dipole donut and the depolarisation of the x and y dipoles change the PSF well below NA/n = 0.7. Measured in-focus rel-L2 of the scalar pupil (1/√cosθ form, unit sum) against an aplanatic Richards–Wolf isotropic emitter [Rev]:

| NA/n | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.92 |
|---|---|---|---|---|---|---|---|
| rel-L2 | 1.9 % | 3.5 % | 5.7 % | 8.5 % | 12.3 % | 17.3 % | 26.4 % |

The scalar peak is too high by 2.3–35 %, and the low-NA error follows (NA/n)²/4, consistent with the z-dipole carrying ⟨sin²θ⟩/2 of the collected power (≈13 % at NA/n = 0.7, 24 % at NA 1.4 in oil). Hence `psf="auto"` selects vectorial for sparse emitters above the imaging element's `vectorial_above` NA/n threshold: 0.7 by default (the `standard` value) and 0.45 at `accurate`. Quantitative outputs (`gx.crlb`, system identification) need vectorial whenever NA/n > 0.45, and `gx.crlb` warns otherwise (§5.6). Here n is the index of a homogeneous medium, or the immersion index n_i when a `LayeredMedium` is declared, so NA/n ≤ 1 (NA 1.45 in oil gives 0.955). When no vectorial implementation is registered (before M4b, or in v1.0 under the M6 option), `psf="auto"` falls back to scalar with an info finding that quotes the table above (not a warning, since every high-NA render would raise it); this is never a PlanError. Paths without a v1 vectorial implementation (dense strata, coherent contributions) stay scalar and report the tabulated error. Vectorial imaging is also required for fixed or wobbling dipoles (a scalar or Gaussian fit biases localisation by up to ~40 nm [B03 §2.2]), for polarisation optics, for near-interface emission, and for quantitative intensities at NA > 1 [B05 §3.1]. The vectorial sparse path costs 5.9 ms against 2.37 ms scalar (B = 64 × 50 emitters), i.e. 10.8k frames/s [M-A].

### 5.4 Illumination and coherence handling

Every source produces modes on M. M, L and P are reduced only in `Detect.square_reduce`, through **exact chunked, checkpointed sums**. Chunk sizes come from free memory (analytic estimates of the dominant tensors, or one small probe chunk scaled linearly), are quantised to a power of two, and are recorded in the Pipeline so that the summation order is reproducible (§4.3). Measured: 3.74 GB → 0.51 GB at chunk 8, gradients identical to 1.1·10⁻⁶ [B07 §4.2]. Randomness inside a checkpointed chunk is keyed or drawn before it (§6.3).

| Source-sampling strategy | Use | Evidence |
|---|---|---|
| `on_support(1\|2)` | draft only; the mode(s) must lie **on the source support**: a disc gives its centroid; an annulus gives one point, or by default a symmetric pair, on the ring | Phase contrast: an on-axis mode gives object contrast **−0.31** (wrong sign), the full annulus +5.9, one on-annulus point +9.2. An on-axis mode under a darkfield annulus (NA_c > NA) puts the zero order inside the pupil and turns darkfield into brightfield [M-J physics] |
| `quadrature(n)` rings / Fermat spiral, n_src ≈ 9–25 | **all training-data generation** | Deterministic n_src = 9 leaves 0.8 % rel. RMS against exact Abbe [M-J ML]; n_src = 25 ≤ 2 % target for σ_coh ≤ 0.7 [C §11]; 25 nodes on a 256² thin sample run at ~400 img/s with gradients [M-C] |
| `grid` (exact on-grid Abbe) | `reference` preset, convergence tables | ground truth for the ladder |
| `stochastic(n)` (1.x) | exploratory or large-scale gradient estimation only; nodes drawn from the numerics key (§6.3). Unbiased for the image and for losses linear in the image; for nonlinear fitting losses the objective is biased by O(Var(I_n)) ∝ 1/n_src (e.g. +Var(I_n) for MSE). Final fits use deterministic quadrature, or increase n_src near convergence | Deviation from exact Abbe per image: 4.0 % RMS at n_src = 2, 2.8 % at 4, 1.8 % at 8, 0.9 % at 32, spatially structured fringes, against 1.0 % shot noise at 10⁴ photons/px [M-J physics]; 2.2 % at n_src = 2 in a σ_coh = 0.5 case [M-J ML]. Not masked by shot noise |
| `socs(n)`, `wotf` | v1.x, **opt-in** fast paths; SOCS requires a thin sample and fixed optics (its validity check errors, rather than silently switching, when optics are learnable, because SVD gradients are unstable at degenerate singular values); the 3-D WOTF is valid for thick *weak* objects | [B05 §1.2, M-J physics] |

- **Thin samples.** In v1 each mode's scattered field is computed exactly as FFT[(t − 1)·E_b,m] with continuous node directions. The "one object FFT plus M shifted pupils" trick, which requires on-grid nodes, is a v1.x opt-in rewrite (`snap_and_shift`). The Pipeline reports the angular error λ/(nL) and enables it only when no source field requires gradients. The cost difference is at most 2×.
- **Thick samples.** Each mode needs its own march (Abbe). The cost M × n_slices slice-steps is printed by `explain()` and warned above 500 slice-steps per image. Random-phase superposition (one march per realisation, speckle contrast ≈ 1) is offered only on explicit request [B05 §1.2].
- **Numerical randomness** (stochastic nodes, speckle realisations) is a pure function of the explicit numerics key (§6.3), never of the input values, so the same inputs render at any fidelity [M-J ML]; keyed draws are exact under checkpoint recomputation.

### 5.5 Detection

Detect runs reduce → pixel integration (pixel MTF, §4.3) → sensor, in three execution modes: `expected` (noise-free mean, fully differentiable), `sample` (explicit key; per-stage estimator, §6.2–§6.3) and `log_prob(observed)` (exact Poisson; Poisson ⊛ Gaussian for sCMOS with per-pixel maps, approximated by default as a Gaussian with variance μ + σ²: the benchmark of §11.6, which weighs gradients as well as accuracy, chose it at M1 over the shifted Poisson, and the exact truncated convolution (≈50× slower) serves calibration (§14.2 Q6); high-gain Gamma–Poisson for EMCCD). The sampling order follows the converged chain: QE → Poisson → dark/CIC → full-well clamp → EM Gamma → read noise (per-pixel map) → gain/offset → ADC (straight-through round) → saturation [B03 §3.1, B05 §4]. Background models (constant, learnable low-order field, or a background map supplied by the front end, e.g. a GRF autofluorescence realisation) add expected photons before Poisson.

**Camera element and output unit.** `gx.Camera(pixel_size, shape, qe=1.0, gain=1.0 [ADU/e⁻], offset=0 [ADU], bit_depth=None, noise=..., unit="adu"|"e"|"photons")`, with every field a tensor (so gain and offset can be learnable, or sampled per image by the front end). The `"image"` output is in `unit` (default `"adu"`); `gx.detect.to_unit(frames, camera)` converts real frames into the camera's declared unit. Calibrating against real images without these fields would let gain, offset or illumination mismatches be absorbed into the scene parameters being learned.

**EM gain.** With electron count n from the Poisson stage (an integer-valued float in sample mode, often 0 in the low-light regime EMCCDs exist for), the stage is

  n_safe = where(n > 0, n, 1);  y = where(n > 0, G·Γ(n_safe; gen), 0) + where(n > 0, 0, G·(n − sg(n))).

The forward is the exact Gamma–Poisson chain with y = 0 at n = 0; the backward has no infinities (Gamma sampling at concentration 0 returns an infinite concentration gradient, and a single `where` turns 0·∞ into NaN) and uses the implicit-reparameterisation gradient for n > 0 and dy/dn = G at n = 0. The straight-through term matters: the double-where alone gives dE[y]/dλ biased to 0.59 / 0.82 / 0.975 of the truth at λ = 0.2 / 1 / 3, and with the term it is 1.000 at all three [Rev]. `torch._standard_gamma(…, generator=…)` is a guarded private API (risk 14).

**Taps**: `expected`, `image_field` (complex, per mode, rank 7, in √photons, background phase referenced to z0), `exit_field`, `pupil_field`, per-population irradiance, `psf`. The user-facing field output is the renderer `gx.out.Field(normalize="background"|"incident"|"none", grid="camera"|"detection", layout="complex"|"re_im", reduce=None)`, which returns `[B, C, H, W]`: it sums coherently over L only when L = 1, raises when M > 1 (a partially coherent image has no single field) unless `reduce="stack"`, and on the camera grid uses complex average pooling of the oversampled field. `gx.labels.Phase()` = arg(E/E_b) on the camera grid.

### 5.5a Channels and spectra

Use **A** (unflattened into C on output) when channels differ *before* detection: biplane or multifocus (per-channel `focus`), per-channel pupil modifiers, and chromatic focal shift via `ChromaticDefocus(dz_per_um)`, a pupil modifier that uses its λ argument. Use **W[C, L]** after Detect when optics are shared: emission filters × QE, dichroic bleed-through, and RGB Bayer mosaics (a post-Detect sampling mask; demosaicing is the user's job). **Polarisation analysers** are Jones vectors a_c applied *before* the square, I_c = Σ w·|a_cᴴE|², because weights on squared components cannot express a 45° or circular analyser: those need the cross term Re(E_xE_y*). Analysers act wherever the field is squared (Detect for coherent fields, Emit for vectorial emitters) and need P = 2 fields: vectorial sparse emitters from M4b, and coherent producers from M3. On P = 1 paths only the co-polarised analyser exists (the S∥ reduction, §4.1). A polarisation camera is four analyser channels (0°, 45°, 90°, 135°) plus a 2×2 mosaic mask, as for Bayer. Cameras may differ per channel (maps, gain, offset). Registration between channels is affine (cap-26, M2). Hyperspectral and dispersive detection are out of v1 scope.

### 5.6 How fidelity is expressed

Fidelity is a **frozen policy object**: named presets expand into explicit, overridable knobs. It is discrete and not differentiated. The detector model is *not* a fidelity knob; it belongs to the camera element (`gx.Camera`).

```python
@dataclass(frozen=True)
class Fidelity:
    # the only positional field
    preset: Literal["draft", "standard", "accurate", "reference"] = "standard"
    _: dataclasses.KW_ONLY
    # model choices (None → preset); values double as registry names, so plugins are selectable
    emitters: Literal["gaussian", "pupil"] | str | None = None
    emitter_path: Literal["auto", "sparse", "global", "strata"] | None = None
    psf: Literal["scalar", "vectorial", "auto"] | None = None
    # coherent paths: P = 1 (S∥) or P = 2 (§4.1)
    polarization: Literal["scalar", "vector", "auto"] | None = None
    # sparse producers under shaped light (§5.11)
    illumination_coupling: Literal["exact", "local", "auto"] | None = None
    spheres: Literal["dipole", "born", "mie", "voxel"] | str | None = None
    # non-spherical primitives, coherent
    compact: Literal["born", "projection", "voxel"] | str | None = None
    volumes: Literal["projection", "multislice"] | str | None = None
    coupling: Literal["C0", "C1"] | None = None  # C1 from v1.x
    # OnSupport() | Quadrature(n=…) | GridAbbe() | Stochastic(n=…) (1.x)
    source: Nodes | None = None
    wavelength_bins: int | Literal["auto"] | None = None
    # dense emission at the frame's edges: exact linear or circular convolution (§5.1)
    dense_boundary: Literal["linear", "periodic"] | None = None
    # discretisation ("auto" → derived from the envelope; any value is a pin)
    oversample: int | Literal["auto"] | None = None
    interaction_band: Literal["optical", "full"] | float | None = None
    raster_sigma: float | None = None  # σ_r in units of the voxel size h
    # s: rasterise s× finer, then decimate (§4.6)
    raster_oversample: int | Literal["auto"] | None = None
    dz: float | Literal["auto"] = "auto"
    pad: float | Literal["auto"] = "auto"
    roi: int | Literal["auto"] = "auto"
    # policy
    per_population: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # passed to the Pipeline; "escalate": v1.x, L4 only
    on_invalid: Literal["warn", "raise", "ignore", "escalate"] = "warn"
    # passed to the Pipeline: fraction of free device memory when it is built (§4.3)
    memory_budget: float = 0.8
    backward: str | None = None  # v1.x cross-fidelity surrogate backward; NotImplementedError in v1
```

| Knob | `draft` | `standard` | `accurate` | `reference` (validation only) |
|---|---|---|---|---|
| emitters | Gaussian sprites | pupil | pupil | pupil |
| psf | scalar | scalar; vectorial for sparse emitters when NA/n > 0.7 (otherwise an info line quotes the tabulated error, §5.3) | auto: vectorial for sparse emitters when NA/n > 0.45 or for fixed/wobbling dipoles; other paths scalar with the tabulated error reported; polarisation optics on emitters → the vectorial path | vectorial on sparse emitters; scalar elsewhere (reported) |
| emitter_path | auto (sparse below the calibrated crossover, ≈300–500 emitters per 256² [M-B]) | auto | auto | sparse / global only |
| polarisation (coherent; auto = P = 2 when the Chain is polarisation-dependent) | auto | auto | vector | vector |
| illumination coupling (sparse producers under shaped light; auto = local when the particle is small against the illumination's finest feature) | local | auto | exact | exact |
| spheres (coherent) | dipole if x_max ≤ 0.75 (dielectric) or ≤ 0.6 (plasmonic m), else mie | mie | mie | mie |
| compact (non-spheres, coherent) | projection | born if φ_max = k₀Δn_max·d_max ≤ 0.25 rad; otherwise projection (thin, large features) or the volume group (rendered by the `volumes` knob: projection at `standard` unless pinned) | voxel → volume group | voxel |
| volumes | projection | projection | multislice | multislice, converged dz, full band |
| coupling | C0 | C0 | C0 (C1 from v1.x) | C0 (C1 from v1.x; MBS v2+) |
| source (Köhler/LED density) | on_support (1 disc / 2 annulus) | quadrature(9) | quadrature(25) | grid Abbe |
| wavelength bins | 1 | 1 | auto: N_λ ≥ 4ΔOPD_max/l_c, ≥ 3 for fluorescence | converged |
| dense boundary (strata) | periodic | periodic | linear | linear |
| detection oversample | camera grid (warn if undersampled) | auto: ≤ λ/(4NA), or the reference rule of §4.3 | auto × 1.5 | × 2 |
| interaction band | optical (NA_obj + NA_ill)/λ | optical + 10 % | full n_m/λ | full |
| raster σ_r | 0.5h | 0.3h | 0.3h_fine at s = 2 (in-band attenuation ≥ 0.9 at (NA_obj + NA_ill)/λ, §4.6) | 0.3h at 2× grid |
| slice dz | — | — | min(λ_m/2, 0.5 rad/(k₀Δn_max)) | halved until converged |

All presets obey the **consistency contract** (§5.9): same radiometry, frames, sub-pixel origin, sign conventions and label geometry. Overrides compose: `gx.Fidelity("standard", psf="vectorial", per_population={"beads": {"spheres": "mie"}, "cells": {"volumes": "multislice"}})`. `per_population` accepts the model-choice keys `emitters`, `emitter_path`, `psf`, `spheres`, `compact`, `volumes` and `raster_sigma`; every other knob (source, wavelength bins, grids, policy) is global. `gx.crlb` warns when the Pipeline or Plan it receives uses a scalar PSF for sparse emitters at NA/n > 0.45, because quantitative precision bounds need the vectorial model (§5.3).

**Fidelity at each level.** A `Fidelity` acts only at L4: `gx.plan` uses it to choose an element for every population without an entry in `methods=`, and writes its knob values into the elements it builds, each knob only into the field the element declares for it (`Element.fidelity_knobs`, ADR-42; a field that merely shares a knob's name is left alone). At L3, `gx.Pipeline` resolves the knobs left "auto" in the Chain's elements (oversampling, raster σ, dz, source nodes, λ bins) from the envelope by each element's default rule, and never overrides an explicit value; `on_invalid`, `memory_budget` and `deterministic` are Pipeline arguments, which `gx.plan` passes through. At L2, an element resolves its "auto" knobs the same way, on the envelope of the inputs it is given.

### 5.7 The planner and the pipeline analysis

**When.** At L4, `gx.plan(sample, microscope, fidelity, outputs, envelope, methods, inputs)` normalises its inputs and derives their envelope (P0, P1), expands the fidelity policy and chooses an element for every population (P2), builds a Chain, and hands it to `gx.Pipeline` together with the envelope and the policy's `on_invalid` and `memory_budget`. At L3, `gx.Pipeline(chain, envelope, outputs, inputs=None, batch=None, slots=None, on_invalid="warn", memory_budget=0.8, deterministic=False)` runs P0, P1 and P3–P8 for any Chain, hand-built or planned, so errors surface before anything renders. Chunk sizes are chosen by P6 for the device and the Pipeline's capacities (the batch-size bucket of the example inputs or `batch=`, and each population's slot count or `slots=`), and recorded (§4.3). Calls may bind smaller batches and fewer slots; a larger batch or slot count raises, and `gx.render` re-plans (§6.9). Pipelines and Plans are immutable. `gx.render` keeps a cache of plans, and `Pipeline.capture` of captured CUDA graphs, under `hash(structure signature, capacities, declared inputs, envelope bucket, outputs, deterministic flag, chunk sizes, device class, precision policy, version)`, with the fidelity added for plans. Re-planning happens only on explicit calls, or in `gx.render` when the structure, fidelity, outputs or device change or values leave the envelope (§6.4). **Updating an input tensor never re-plans** while it stays within the envelope, and the set of inputs requiring gradients may change only the memory strategy (ADR-12).

**How (rule-based, deterministic).** The planner (P2, reusing the P0 and P1 functions) and the Pipeline analysis (P0, P1, P3–P8) are ≈2–3 kLOC together, including inspection, JSON and partitioning; M0 builds them for emitters, with `explain()` and a minimal Pipeline JSON (recorded chunk sizes). Passes are plain functions `PlanState → PlanState` that append to a report:

```
P0 normalise   canonical ordering; field paths (§6.4); structure signature (§6.1)
P1 envelope    entries derived from the example inputs (×1.25 headroom, bucketed) for every shape-affecting
               field (schema-declared), overridden entry by entry by front-end probes, then by explicit
               entries; each entry records its source.
               Derived invariants: x_max, φ_max, Δn_max, extent_z, defocus range, NA_ill+NA_det, min gaps, λ range
P2 route       (gx.plan only) expand the fidelity preset into knobs; RULES[(population kind, contrast, knob)] → element
               unless pinned by methods=; partition (containment, proximity) into groups that gx.interact.Superpose
               renders (host-index rule, cut-out-and-fill, C0 warnings); emitter-path rule from N (static)
P3 validity    each implementation's predicates over the envelope → Violations; on_invalid ∈ {warn, raise, ignore};
               escalate (v1.x, L4 only) = the planner re-runs P2 along the unique approximation edge whose higher
               implementation accepts the population kind; ambiguity → PlanError
P4 grad check  capability grad tags per implementation → a gradient-quality table for every input field
               (exact / exact-a.e. / biased / zero; "partial" when some paths are zero); at call time, a field that
               requires gradients along only zero paths ⇒ GradientPathError naming the stage and the fix (§6.2)
P5 sampling    SamplingPlan (§4.3) with provenance; smooth FFT sizes
P6 memory      needs_grad per stage; no_grad vs autograd vs checkpoint; chunk sizes from analytic estimates of the
               dominant tensors (mode/λ/emitter chunks, march activations) or one small probe chunk scaled linearly,
               quantised to a power of two and recorded in the Pipeline JSON; pre-flight refusal of infeasible Pipelines
P7 rewrites    fuse pupil modifiers; merge homogeneous propagations; strata collapse; hoist grad-free invariants
               within a call (pupils, OTFs, Mie tables shared by all modes and wavelengths); no cross-call value
               caches (Pipeline.capture covers repeated fixed-optics calls); drop unused outputs
P8 emit        the Pipeline's frozen state (stages, SamplingPlan, chunk sizes, envelope, gradient table, validity
               report, estimates) → JSON
```

Cost is used only to order *numerically equivalent* strategies (chunk sizes, checkpoint placement, FFT vs MFT for the same grid), and `cost()` is optional in the plugin contract. **Any automatic choice between implementations with different approximation error is a deterministic rule on static quantities** (fidelity knob, envelope, N), never on device-calibrated cost. Otherwise numerics would differ across machines and between training and evaluation [M-J eng].

```python
RULES = {  # (population kind, contrast, knob, value) -> element (registry name)
    ("point", "emission", "emitters", "gaussian"): "emit.gaussian",
    # or global/strata by path rule
    ("point", "emission", "emitters", "pupil"): "emit.pupil_mft",
    ("sphere", "coherent", "spheres", "dipole"): "interact.dipole",
    ("sphere", "coherent", "spheres", "mie"): "interact.mie",
    ("sphere", "coherent", "spheres", "born"): "interact.born_ff",
    # standard: only if φ_max ≤ 0.25 rad
    ("compact", "coherent", "compact", "born"): "interact.born_ff",
    ("compact", "coherent", "compact", "projection"): "interact.projection",
    ("volume", "coherent", "volumes", "projection"): "interact.projection",
    ("volume", "coherent", "volumes", "multislice"): "interact.multislice",
    ("labeled", "emission", "emitter_path", "strata"): "emit.strata_otf",
    # "voxel" values route the population into its volume group (with cut-out-and-fill for hosts)
    # gx.JonesScreen populations, and objects with uniaxial materials, always route to interact.projection in 1.0
}
```

**Inspection.** `pipe.explain()` (and `plan.explain()`, which adds the element choices) prints the chosen elements, grids with provenance, rejected alternatives with reasons, the gradient table, validity, tolerance entries and estimated cost and memory. `plan.why("interact.beads")` gives one node's decision trace. `gx.Pipeline.diff(a, b)` compares two Pipelines stage by stage. `pipe.run(chain, until="objective", taps=[...])` steps through stages. `to_json()` freezes a Pipeline (including chunk sizes) so a dataset can be regenerated exactly. `Pipeline.graph` and `Pipeline.profile` (measured against estimated cost) are v1.x. Illustrative output of `plan.explain()` for the L4 form of example (a) of §9; `pipe.explain()` prints the same without the fidelity field of the header and the element-choice lines (structure is the design; timings are anchored to the measurements cited):

```text
Plan 7f3a1c · fidelity=standard · B≤64 · cuda:0 · complex64/fp64-special · planned in 18 ms
grad set   beads.radius (requires_grad; produced upstream)
envelope   beads.radius ≤ 1.50 µm (explicit) → x_max 23.6 → n_max 37
           beads.position.z ∈ focus ± 6.6 µm (explicit) · N = 8 slots (structure)
           light.tilt, objective.pupil[0].coeffs: per image, not shape-affecting
sampling   detection 210² @ 162.5 nm = 6.5 µm/40× (s=1; λ/(4NA) = 166 nm ✓; pixel MTF)
           pad 39 px ← 6.6 µm·tanθ_max(0.753) + 2λ/NA(1.33 µm) = 6.30 µm · 210 = 2·3·5·7
           pupil support K = 8 270 samples (|f| ≤ NA/λ) · Mie θ-table 1024 · chunks M:1 L:1 (recorded)
 # slot       implementation          notes                                                        est. fwd
 1 source     plane_waves             M=1 · L=1 (532 nm) · P=1 (co-pol S∥; cross-pol incoherent) · tilt per image  —
 2 lower      beads → SphereList      exact · grad(radius, n, xyz) = exact                            —
 3 interact   mie (fp64 on GPU)       travel +1 · C0 · host = medium n 1.33 · on K samples only       ≈13 ms
 4 objective  pupil.scalar (fused)    aperture(soft)·√cosθ·zernike[B]·defocus(z_ref→focus)            <1 ms
 5 form       scatter K → k-grid, ifft   background analytic: P(u_b)·√cosθ_b evaluated               ≈1 ms
 6 detect     square_reduce           |E_b|²+2Re(E_b*E_s)+|E_s|² · pixel MTF · PoissonGaussian(scaled-ST) → ADU
gradients  beads.radius  mie → objective → detect(scaled-ST)
                                        exact(≤quadratic in counts)
validity   info  C0: bead–bead coupling neglected; gap < λ+2a for 3.1 % of beads (validate() on this batch)
           info  dipole rejected: x_max 23.6 > 0.75 (dipole approximates mie as x→0)
tolerance  mie ↔ direct field summation ≤ 1e-3 (ladders/mie_pupil@v1.0)
memory     est. peak 0.6 GiB fwd+bwd at B=64 (autograd; checkpointing not needed)
```

### 5.8 Validity checks

Predicates are evaluated on the envelope when a Pipeline is built (an eager element call evaluates them on the envelope of its inputs), and on actual values by `validate()` on a Pipeline or a Plan. They start from literature thresholds and are **recalibrated from our own ladder tables** (§11). Reports say whether a threshold is `literature` or `calibrated`.

| Implementation | Predicate (initial) | Severity |
|---|---|---|
| projection | object thickness ≤ DOF = n_m·λ₀/NA_s² with NA_s = min(NA, n_s) (a thickness criterion, not a phase criterion); an NA term, 1/cosθ_max − 1 above the tolerance when the 1/cosθ_out factor is disabled (e.g. NA/n_m > 0.3 for 5 %); a feature-size term, x ≫ 1 (anomalous diffraction is invalid for small features regardless of NA) | warn |
| born_ff | φ_max = k₀·Δn_max·d_max (d = maximum chord from the envelope) > 0.25 rad (≈10 % pupil rel-L2; 0.12 rad for 5 %). **Calibrated** from the born_ff ↔ mie ladder (error ≈ 0.41·φ, independent of radius). At `standard` a failure reroutes `compact` (§5.6). Slaney's Δn·d < 0.35λ₀ is a *reconstruction* criterion that allows ≈95 % forward error and is kept only as a literature citation | warn |
| multislice | propagation angle (NA_ill + NA_scatter)/n_m ≲ 0.7 for plain BPM (opt-in; `literature`); for the obliquity-corrected march the angle limit is recalibrated from the M4 ladder, whose exit requires ≤ 5 % at NA 1.2; per-slice phase k₀·Δn_max·Δz ≤ 0.5 rad; Δz ≤ λ_m is the warn limit for a *pinned* dz (the default is λ_m/2, §4.3); backscatter not modelled | warn / info |
| dipole | ≤ 5 % all-angle error: dielectrics x ≤ 0.75 with a₁ + b₁ (≤ 0.5 with a₁ only); metals x ≤ 0.6–0.7 (b₁ does not help; a₂ dominates). Bare quasi-static α only for dielectrics at x ≤ 0.3, never for complex m with a plasmon near λ (gold at 532 nm and silver at 450 nm err by 17–21 % already at x = 0.3) [Rev] | warn |
| mie | homogeneous host; gap ≥ λ + 2a to neighbours and interfaces | warn |
| C0 | inter-object coupling neglected (fraction of close pairs from the envelope and `validate()`) | info |
| psf scalar | sparse emitters rendered with a scalar PSF above the imaging element's `vectorial_above` NA/n (0.7 by default and at `standard`; 0.45 at `accurate` and for quantitative fits); dense and coherent paths quote the tabulated error of §5.3 | info |
| gaussian / scalar with fixed or wobbling dipoles | localisation bias up to ~40 nm | **error** (overridable) |
| gaussian sprite | \|Δz\| beyond ±DOF/2 or NA > 0.7 | warn |
| detection sampling | spacing > λ_min/(2·max\|u_a − u_b\|) (§4.3) | warn with `oversample=1` (the camera grid, as `draft` sets) / error for a larger pinned `oversample` |
| Transduce product grid | pinned coarser than the §4.3 rule under structured excitation | error |
| interaction band | estimated dropped scattered energy > 1 %; in-band raster attenuation < 0.9 with `raster_oversample=2` (as at `accurate`) | warn |
| ROI / padding | smaller than the envelope-derived minimum when pinned | error |
| spectral bins | N_λ < 4·ΔOPD_max/l_c | warn |
| strata | Δz_strata > λ/(2(n_s − √(n_s² − NA_s²))), NA_s = min(NA, n_s) | warn |
| population z-envelope | a population's z-envelope, including its bounding box, crosses a declared interface or leaves the sample layer | **error** (overridable for deliberate cases) |
| near-interface emitters | within ~λ of an interface: emission-rate (LDOS) and orientation effects not modelled (cap-32); supercritical collection *is* modelled (§4.1) | info |
| source | n_src below the convergence table for this σ_coh | info (quotes the tabulated error) |
| source stochastic in data generation (1.x) | per-image error table of §5.4 | warn |
| iSCAT reference `homogeneous` (iSCAT-lite) | constant r; omits Fresnel transmission, r_s/r_p(θ), index change | info: "approximation tier" |
| P = 1 on a polarisation-dependent Chain | a Jones pupil modifier, an analyser other than co-polarised, or a birefringent sample with `polarization="scalar"` pinned | **error** (use P = 2) |
| P = 1 cross-polarised energy | the dropped cross-polarised part exceeds 1 % of the scattered energy at the envelope | info |
| birefringent material outside projection | a uniaxial material routed to Born or multislice, which do not model birefringence in 1.0 (tensor ε is v2) | **error** (route it to projection) |
| stage sampling | grid spacing above half the finest device pitch, or above λ/(2 sin θ_max) for the angles present at a stage plane | **error** when pinned |
| local plane wave | particle diameter > λ/(4·NA_ill), the scale of the illumination's finest feature | warn |
| incoherent light through a multi-plane stage | fluorescence through detection stages beyond the pupil plane (space-variant) | **error** until v1.x (Q22) |
| passive device | \|t\| > 1 anywhere on a passive device | **error** |
| thin Jones sample | as projection (thickness ≤ DOF); differently oriented birefringent layers overlap on a ray with Σδ > 0.3 rad (first-order accumulation) | warn |
| WOTF / SOCS (v1.x) | weak (φ ≲ 0.3–0.5 rad; 3-D WOTF allowed for thick weak objects) / thin and fixed optics | error if violated |
| Rytov (v2) | Δn/n_m ≲ 0.03; no zeros of u_in | error |

**Validity as probability.** `validate()` evaluates every predicate on the actual values and returns per-image and per-object violation masks. A front end aggregates them over sampled batches into fractions ("born_ff invalid for 12 % of sampled vesicles"), which is the right semantics for random scenes [D §5.6]. It is informational and never changes the Pipeline (determinism). Per-instance routing (dipole for small spheres, Mie for large ones, under masks) is v1.x (cap-29).

**Runtime sampling health** (opt-in `diagnostics=True`). Cheap GPU reductions written to `out.meta` without host sync: scattered spectral energy near the band edge (aliasing), energy reaching the padding (wrap), maximum per-slice phase, Mie truncation residual |a_{n_max}|, out-of-FOV fraction, in-envelope flags [A §5.8].

### 5.9 How lower fidelities relate to higher ones

1. **One input, many plans.** Every tier renders the same data objects. Parameters a tier cannot represent follow a declared per-parameter fold policy: `ignore+warn`, `approximate` (e.g. the Gaussian tier folds Gibson–Lanni into an effective defocus and width), or `error`.
2. **Pairwise approximation edges** with regime-bounded tolerance targets. They are tested in CI, tabulated as versioned artefacts, and printed by `explain()`:

| Lower | Higher | Limit | Regime | Initial target |
|---|---|---|---|---|
| gaussian sprite (σ = 0.22λ/NA) | pupil (scalar) | near focus, isotropic | NA ≤ 0.7, \|Δz\| ≤ 0.2 µm | ≤ 20 % rel-L2 (a unit-sum Gaussian cannot represent the Airy rings: 13–15 % at the L2-optimal σ [Rev]); label metrics: centroid bias ≤ 1 nm, FWHM within 5 %, integral exact |
| pupil scalar | pupil vectorial | low NA, isotropic emitter | NA/n ≤ 0.3 | ≤ 2 %; tabulated to NA/n = 0.92 (≈12 % at 0.7, §5.3) |
| P = 1 (co-polarised S∥) | P = 2 (vector Mie/dipole/Born) | scalar reduction | on-axis linear illumination, homogeneous medium, co-polarised analyser | exact there; table vs NA, tilt, stack and dropped cross-polarised energy elsewhere; P = 1 also checked against holopy `MieLens` |
| Jones `DICShear` | scalar `DICShear` | isotropic sample | — | ≤ 10⁻⁵ (identity) |
| local plane wave | J-wave superposition | small particle, slowly varying illumination | d ≤ λ/(4·NA_ill) | table |
| `Shaped` with J = 3 | `SIMBeams` | identical construction | — | ≤ 10⁻⁶ (identity) |
| BL-ASM `Propagate` | Rayleigh–Sommerfeld direct integral (test reference) | inside the band limit | Fresnel number ≥ 0.1 | ≤ 10⁻³ |
| thin Jones screen (first-order accumulation) | Jones product along the ray (test reference) | weak, differently oriented layers | Σδ ≤ 0.3 rad | table |
| pupil tier pixel integration (MTF) | fine box integration | band-limited intensity | pitch ≤ λ/(4NA) | ≤ 1 % |
| sparse ROI | global spectrum | ROI → ∞ | ROI from the envelope | ≤ 10⁻⁴ (a too-small ROI gave 5.7 % [M-D]) |
| strata OTF | sparse | Δz_strata → 0 | emitters on strata planes | ≤ 10⁻⁶; table vs Δz |
| dense-density SIM (product grid) | EmitterSet SIM on the same labels | grid → continuous | maximum pattern frequency | ≤ 2 % on the modulated component |
| dipole (a₁ + b₁) | mie | x → 0 | dielectric x ≤ 0.3 | ≤ 2 %; table per material class (TiO₂ with a₁ only is already 2.1–2.4 % at x = 0.3) |
| born_ff | mie | weak (Rayleigh–Gans) | φ ≤ 0.25 rad | ≤ 10 %; table vs (x, Δn) seeded from the review sweep (4.1 / 12 / 21 / 41 / 95 % at φ = 0.1 / 0.3 / 0.5 / 1.0 / 2.2 rad) [Rev] |
| projection | born_ff / mie | anomalous diffraction | x ≫ 1, m → 1; NA/n_m 0.3–0.9 | table vs (x, NA) |
| projection | multislice | thin sample | d ≤ DOF/4 | ≤ 2 % |
| multislice (voxel sphere) | mie | discretisation → 0 | Δn ≤ 0.05, x = 1–10, NA 0.5 / 0.8 / 1.2, at the preset's dz (λ_m/2 at `accurate`, converged at `reference`) | ≤ 5 % on the scattered field; dz sweep (λ_m/2, /4, /8) tabulated |
| multislice (obliquity-corrected) | plain BPM and the reference | stability | thick strong phantom (Δn 0.1 over 20 µm) | stable; table |
| quadrature(n) | grid Abbe | n → ∞ | σ_coh ≤ 0.7 | n_src = 25: ≤ 2 % |
| on_support | quadrature | coherent limit σ_coh → 0 | disc σ_coh ≤ 0.2; annulus: sign must match | table |
| iSCAT homogeneous | layered.fresnel | index-matched stack | normal incidence | table |
| soft voxel (σ_r = 0.3h) | exact continuous object | prefilter → 0 | coherent (standard grid) and fluorescence (λ/(4NA)) | table (≈3–4 % coherent, ≈1.3 % fluorescence [Rev]) |
| soft voxel s = 1 | s = 4 | — | dense fluorescence | ≤ 1 % |
| C0 | C1 (v1.x) | weak coupling | beads in a phantom cell | table |
| multislice | SSNP (v2) | paraxial | angles ≤ 0.5 rad | table |
| born_slices (v2) | SSNP ⊂ MBS (v2) | weak | — | table |

Edge metrics are computed on the background-removed contrast or on the scattered field, never on the full hologram, where the unit background hides errors (a weak sphere passes any full-hologram test).

3. **Consistency contract** (tested on ~30 canonical scenes). For the same inputs, every tier gives identical absolute radiometry (PSF integral = collection efficiency; mean photons within the declared tolerance; pupil tiers integrate pixels through the MTF and the Gaussian tier through erf, so pixel integration agrees too), the same coordinate frame and sub-pixel origin, the same contrast-sign conventions (checked against physical references, cf. DT issue #445 [B01 §b]) and the same label geometry. Only approximation error differs. Without the contract, learned photon distributions, `compare()` and cross-fidelity surrogates would be tier-dependent [M-J ML].
4. **`gx.compare(sample, microscope, "standard", "accurate", key=…)`** renders the same inputs through both plans and reports rel-L2, SSIM, phase RMSE, bias in label-relevant statistics (e.g. centroid bias in nm), time and memory.
5. **Cross-fidelity surrogate gradients** (v1.x, opt-in; the `backward` field exists in v1 and raises `NotImplementedError`): `Fidelity("accurate", backward="standard")` computes y = y_hi.detach() + y_lo − y_lo.detach() on the same inputs [B07 §6.4]. `gradients()` flags it as biased.

### 5.10 Mixed samples and coupling

Mixed samples (beads in cells, nanoparticles on labelled cells) are a **partition problem** [B §5.6]. Planner pass P2 (or the user, in a hand-built Chain) groups the populations, and `gx.interact.Superpose` renders each group, applying steps 4–5:

1. **Families per object set**: `Spheres` → {dipole, born, mie, voxel}; other primitives → {born, projection, voxel}; `Voxels`, `NeuralField` → {voxel}; `Emitters` → {emit}.
2. **Relations from the envelope** (and, per image, from the values): containment (`gx.Parent` hierarchies, or a bbox inside a host bbox), proximity (gap < λ + 2a, from the envelope and `validate()`) and interface distance (from the environment).
3. **Groups**: volumetric components with overlapping z-ranges form one *march group*; analytic particles form particle groups tagged `free` or `embedded(host)`.
4. **Host-index rule** (C0): an embedded particle's response uses m = n_p/n_host(λ) and x = k₀·n_host·a. It radiates as if embedded in a homogeneous host, is imaged through n_medium, and the neglected host aberration is reported as info.
5. **Cut-out-and-fill**: when an analytic solver owns a particle inside a volumetric host, the host lowering renders *host material* in the particle's footprint (priority compositing with the child's material replaced by the parent's). The particle is never counted twice, as voxels and as a Mie field.
6. **C1 injection (v1.x)**: march the host; at each particle, read the local incident field from the march (amplitude, phase, and a direction from the phase gradient: the local-plane-wave approximation); evaluate the Mie/dipole response in the host index; add its outgoing angular spectrum **at the particle exit plane z_p + a**, where the representation is exact [M-J physics]; continue marching. Cost is roughly one small FFT per injection plane [B04 §5]. This captures cell-induced aberration and shadowing of bead images.
7. C2 (iteration), C3′ (dipoles inside an MBS volume) and interface coupling are v2+.

| Setting (beads in a cell, holography) | Policy | Release | Captures |
|---|---|---|---|
| draft / standard | C0: projection(cell) + Σ Mie(bead, host index) at the pupil, cut-out-and-fill | 1.0 | both images; no mutual effects |
| accurate | C0 with multislice(cell) | 1.0 | thick-cell forward scattering |
| accurate + C1 | multislice with Mie injection at exit planes | 1.x | cell aberrates and shadows beads |
| reference | C0 at converged discretisation (C1 from v1.x) | 1.0 | as accurate, converged |
| reference (v2+) | MBS with beads voxelised or as coupled dipoles | 2+ | everything, including reflections |

### 5.11 Programmable and diffractive optics (optical neural networks)

Light can be shaped before and after the sample by programmable devices (spatial light modulators, digital micromirror devices, micromirror arrays, deformable mirrors) and by fabricated ones (diffractive layers, metasurfaces, diffractive optical elements). Trained and cascaded, such layers form **optical neural networks** (ONNs): from a single learned pupil (PSF engineering) to multi-layer diffractive networks, either fixed after training or trained jointly with the rest of the optics and a digital network. gradix treats them as **optical stages**: thin elements and free-space propagation at planes of the optical path, with device models for the physical implementations. They are ordinary elements, so every ONN inherits batching, envelopes, gradients, keys, capture and the three API levels.

**Where light is shaped.** Each place is a plane of the conjugate-plane model (§5.3):

| Plane | What an element does there | Examples | Where it lives |
|---|---|---|---|
| Illumination pupil (condenser back focal plane, or the objective pupil in epi-illumination) | multiplies the illumination's pupil function | SLM or DMD pupil patterns, learned coherent illumination, metasurface illuminators | `illumination_optics` |
| Illumination field planes (conjugate to the sample) | shapes the field that reaches the sample | DMD-patterned illumination, holographic stimulation, illumination-side diffractive networks | `illumination_optics` |
| Detection pupil (objective back focal plane, or a relayed Fourier plane) | multiplies the pupil (fused, diagonal in angle) | PSF engineering with SLMs, deformable mirrors, DOEs and metasurfaces | `Objective(pupil=[…])` (§5.3) |
| Detection field planes (intermediate image and beyond; right after the sample in lensless set-ups) | acts on the sampled field on its way to the camera | diffractive networks, image-plane masks, 4f optical processors | `detection_optics` |

**Stages in a Chain.** A Chain has two ordered stages, `illumination_optics` and `detection_optics`, each an ordered mapping from names to stage elements, so their fields have paths such as `slm.device.phase` (§6.4). This settles Q20: the slot skeleton plus ordered stages, and no general DAG in v1. The stage elements (`gx.optics.*`):
- `Mask(transmission | device, pitch=None, plane="field"|"pupil", magnification=1, fill_factor=1, offset=(0, 0), rotation=0)`: a thin element, scalar or Jones (P = 2), on its own pixel grid, resampled onto the simulation grid through its pixel aperture. It is differentiable in its values and in its placement, so misalignments can be fitted or sampled;
- `Propagate(distance, medium)`: band-limited angular-spectrum propagation between planes (BL-ASM), differentiable in the distance;
- `Lens(focal_length)` and `FourierTransform(focal_length)`: an ideal thin lens, or the exact 2f transform between focal planes (a scaled MFT); `Filter(mask)` is the 4f shorthand, two transforms around a Fourier-plane mask;
- `Aperture(shape, size)`: field stops and irises, with soft edges;
- `DiffractiveStack(devices, pitch, gaps, offsets=None)`: alternating `Propagate` and `Mask` elements, a diffractive network in one element.

Detection stages act on the sampled total field in image-space coordinates (µm on the camera side, λ in the image medium). They take the Objective's image field inside a field stop, so the analytic background is sampled only where a physical stop keeps the field compact and periodic wrap harmless. In a lensless Chain, the stages start from the sample's exit field in object space.

**Shaped illumination.** This is construction 3 of §4.5 (a coherent pupil function A(u) sampled on the illumination k-grid) at scalar, moderate NA; vectorial focusing of high-NA shaped beams stays v1.x (cap-25). Illumination stages, or `gx.light.Shaped(pupil_field= | field=)` directly, produce it. Consumers read it in three ways:
- Dense producers (projection, multislice) take the incident field on the interaction grid. Its spectrum lies on the grid by construction, and the stop and the padding keep wrap out of the field of view.
- Sparse producers see J plane waves, one per pupil sample (the J axis of `PlaneWaves`). Dipoles need only the field at their own positions, which is exact and cheap. Born sums over J as a k-space convolution of A(u) with the form factor. Mie sums over J in chunks, which is exact, or, with `illumination_coupling="local"`, uses one local plane wave per particle, taken from the field and its phase gradient at the particle; that is valid when the particle is small against the illumination's finest feature (§5.8).
- Transduce evaluates the field at emitter positions and on the product grid; for incoherent patterned excitation, it uses the intensity image of the pattern.

*As built (M3a phase 5).* `gx.light.Shaped(wavelength, na=, pupil_field= | device=, samples=16, irradiance=, center=, travel=)` turns a Q×Q illumination pupil into J = Q² mutually coherent plane waves. Wave j has amplitude √(I/J_eff)·w_j·A(u_j)/√cosθ_j, with a soft circular aperture w_j (J_eff = Σw_j²) and the aplanatic factor in the medium the light is defined in. The executor passes the Chain's medium to light elements that declare `reads_medium`. A phase-only pupil therefore keeps the irradiance I through the focal plane (within 10⁻⁹).

`gx.devices.PhaseSLM(phase | shape, pitch=, wavelength=, zero_order=)` gives the pupil a trainable phase, t = (1 − η)·e^{iφ} + η, scaled by λ_design/λ when a design wavelength is given. The coherent imaging element sums every wave's background and Mie scattering. The Mie sum runs in chunks of waves that keep each Mie tensor below 2²⁴ elements, and the memory estimate counts the waves. Verified:

- a one-cell pupil reproduces `PlaneWave` exactly;
- a linear phase across the pupil shifts the focal field exactly;
- the field is linear in the pupil, complex pupils included;
- chunked and unchunked sums agree to 10⁻¹²;
- the pupil must stay inside the medium's index.

A design loop on the GPU (64² frames, a 12×12 SLM of 144 waves, float64) lowered the bound on a 150 nm bead's radius through the off-axis reconstruction from 3.1 µm to 0.27 µm in 15 Adam steps of 1–2 s. The illumination stages (`Mask(device=slm, plane="pupil")`) arrive in M3 and lower to the same waves.

Partially coherent shaping, such as an LED behind a DMD in the condenser pupil, is a learnable `Custom` source density S(u) (construction 2).

**Device models** (`gx.devices.*`). A device is a data object whose parameters produce a transmission map, `device.transmission(grid, wavelengths, basis)`, including the device's non-idealities and a declared gradient quality; a `Mask` places it. Version 1.0 ships two reference devices:
- `PhaseSLM(pitch, shape, phase | levels, bits=8, fill_factor, crosstalk, lut, zero_order)`: continuous or quantised phase (straight-through and declared `biased`, or a relaxed softmax over levels), a λ-dependent phase response, fringing-field crosstalk as a blur of the phase, and an unmodulated zero-order fraction;
- `HeightMap(heights, material, levels=None, max_height)`: fabricated diffractive layers and DOEs in the thin-element approximation, with phase 2π(n(λ) − n_medium)h/λ, absorption from Im n, and optional discrete height levels.

The extended library is v1.x unless Q22 pulls it forward:
- `DMD`: binary states with binarisation estimators, time-multiplexed grey levels and blaze efficiency;
- `DeformableMirror`: actuator influence functions, coupling and stroke;
- `SegmentedMirror`: piston–tip–tilt micromirror arrays;
- `Metasurface`: meta-atom parameters mapped to Jones transmission through a differentiable lookup table from RCWA or FDTD sweeps, or through a neural surrogate, in the local-periodic approximation.

Every placement has offset, rotation, z-shift and scale fields, which may be per image, so callers can sample fabrication and alignment errors for robust training.

**Training regimes.** All four are ordinary torch:
- *Fixed:* device parameters without gradients, such as a fabricated design loaded with `gx.load_inputs`.
- *Joint:* device parameters as `nn.Parameter`s, trained together with a digital network, other optics (illumination, pupil) and the sample distribution.
- *Physics-aware training* (hardware in the loop): y = y_hw.detach() + y_sim − y_sim.detach(), so the forward pass is the measured hardware output and the backward pass is gradix's. The digital twin is calibrated first (device response, alignment, aberrations) with the system-identification tools of §9(b).
- *Optoelectronic cascades:* a camera, a digital nonlinearity and the SLM of the next stage are two Pipelines chained in torch, differentiable end to end.

Region readout, `gx.out.Regions(masks | boxes)`, returns the power in detector regions: the class scores of a diffractive classifier.

**Incoherent light and nonlinear layers.** Pupil-plane devices work for fluorescence on every path, because the system stays shift-invariant. A multi-plane stack after the image plane is space-variant for incoherent light. Sparse emitters then have to pass it one by one (cost ∝ N × layers), and dense densities need low-rank space-variant models (v2). Both belong to Q22's toolkit, and a multi-plane detection stack on incoherent light raises in 1.0. Optical nonlinearities (saturable absorbers, Kerr media) are elements with `linear_in_field=False`: the Pipeline forms all modes before them and chunks only within them (Q22).

**Sampling and cost.** Each stage plane gets grid rules (§4.3): the spacing is at most half the finest device pitch and at most λ/(2 sin θ_max) for the angles present, pads come from the spread d·tanθ_max over each gap, and each sampled transfer function is band-limited. Every decision is recorded in the SamplingPlan. A 5-layer phase-only diffractive stack on 512² fields runs in 14.4 ms forward and 37 ms forward + backward at B = 64 on the 3090 (≈4.5k and ≈1.7k images/s; 1.2 GiB peak) [Rev].

**Release.** In 1.0:
- M3: stages and stage elements, stage sampling, shaped illumination for sparse producers, `PhaseSLM`, `HeightMap`, region readout and example §9(i);
- M4: shaped illumination for dense producers.

In v1.x unless Q22 pulls them forward: the extended device library, nonlinear layers, incoherent multi-plane stacks and a physics-aware-training helper.

### 5.12 Interferometric microscopes and field reconstruction (M3a)

Holography and interferometric scattering record the scattered field through its interference with a reference. Each modality is a recipe in the slot skeleton of §5.1, a source, a reference path, pupil modifiers or detection stages and a camera, and never a new code path. What the camera records (`"expected"`, `"image"`) is the interferogram itself, with the camera's noise; the field it encodes comes back through a differentiable reconstruction (`gx.recon`), and the field's ground truth is `gx.out.Field` (§5.5; as built, `layout="phase"` is the phase label).

| Modality | Illumination | Reference | Detection | Reconstruction |
|---|---|---|---|---|
| In-line (Gabor) | plane wave, transmitted (travel +1) | the unscattered zero order: the analytic background (§4.4) | objective, camera | `gx.recon.Inline`: background-normalised hologram, back-propagated to a plane (twin image included) |
| Off-axis (DHM) | plane wave, transmitted | tilted image-side plane wave, `ReferenceBeam(path="image")` (§4.5) | objective, camera at the spacing λ/(2(\|u_ref\| + NA)) (§4.3) | `gx.recon.OffAxis`: FFT, a soft window on the +1 sideband, demodulation: the complex image field |
| iSCAT | epi plane wave (travel −1) through a `LayeredMedium` | the coverslip reflection r(u) from `layered.fresnel`, optionally attenuated in the pupil | objective on the immersion side; backscatter collected by the layered rule (§4.1) | `gx.recon.ISCATContrast`: I/I_ref − 1 |
| QWLSI | plane wave, transmitted | none: four sheared copies of the field interfere behind a 2-D grating | `gx.optics.Grating(period, distance)` as a detection stage (ADR-45) | `gx.recon.QWLSI`: four harmonics demodulated, two phase gradients, least-squares integration (an FFT Poisson solver): the complex field |

**Mie spectra on the pupil.** `gx.interact.Mie` computes a_n and b_n in fp64: a downward recurrence of the logarithmic derivative D_n(mx) and upward recurrences for ψ_n and χ_n (as BHMIE and Wiscombe's MIEV0; ψ₁ from its series below x = 0.1), with Wiscombe's n_max = x + 4.05x^{1/3} + 2 from the radius envelope (a static count). As built, they match an independent reference (Bohren–Huffman eq. 4.53 with scipy's spherical Bessel functions) within 10⁻⁹ for x = 0.05–12, including absorbing spheres and m < 1; the upward ψ_n loses relative accuracy only in orders beyond x, whose coefficients are negligible. Its spectra are evaluated lazily on the consumer's pupil samples: π_n and τ_n at the scattering angle between each incident wave and each pupil direction give S₁ and S₂, converted to an angular spectrum by §4.1's A(k⊥) = −S/(2πk_m k_z)·e^{−i(k − k_in)·r_p} and reduced to S∥ at P = 1. For backscatter the incident direction has travel −1. π_n and τ_n are polynomials in cosΘ, so supercritical pupil samples of a layered medium (|u| > n_s, complex cosΘ) continue analytically and give the evanescent part of the scattered field that the interface transmits into the immersion side. Every step is plain torch, so forward mode and double backward work (the bound of ADR-44 differentiates a forward-mode Jacobian). The angular functions depend only on the geometry, so the imaging element computes them once on the pupil grid and contracts them with every sphere's coefficients by batched matrix products; measured on the 3090, 128² holograms of 10 Mie spheres render at 4.3k/s forward and 1.4k/s forward + backward (B = 64, 0.45 GiB peak). The Mie-dressed dipole is the same element truncated at n = 1.

**Off-axis, as built (phase 2).** `gx.light.ReferenceBeam(irradiance, angle=, phase=)` takes the illumination's wavelengths. Its irradiance is referred to the object plane, as the illumination's is, so equal irradiances give equal intensities; its `angle` is the image-side tilt; its `phase` has the setting role, so a `[B, T]` tensor steps it frame by frame (phase-shifting holography). `Microscope.references` holds references by name (schema version 2). `gx.recon.OffAxis` demodulates in real space with the exact carrier (fp64, reduced mod 2π), removes the frame mean first so the strong constant cannot leak into the window when the carrier is off the FFT grid, keeps a soft window two frequency cells beyond NA, and divides by the pixel MTF at the sideband's original frequencies, sinc((f − f_c)·p). What limits it is the true field itself: on a finite frame its spectrum leaks past any window (0.4 % rel-L2 against the unlimited field for a bead centred in a 10 µm frame, independent of the frame size), which is why the exit criterion compares against the window-limited field.

**Interference.** References are analytic (§4.4): |E_r + E_s|² is formed term by term with the explicit cross term, so weak scatterers keep their contrast in complex64 (8.7 % error at a contrast of 5·10⁻⁶ when squaring the sum; ≈2·10⁻⁷ term by term).

**iSCAT.** Illumination through the objective travels −1. At the coverslip, `layered.fresnel` produces the reflected zero order (travel +1, amplitude r(u_in)) and the incident field in the sample (t(u_in)). Spheres in the sample backscatter; their spectra propagate in the sample medium to the interface only (§4.1, rule 1), transmit with t(u), including the supercritical samples, and are collected with √(k_z,i/k_i). Multiple reflections between a particle and the interface are neglected (cap-32, v2); the validity check reports an info note for particles within λ of it. A central pupil filter (`gx.pupil.Filter`: radius, transmission, phase) attenuates or phase-shifts the reference: exactly on the analytic zero order by evaluation, and on the scattered samples inside its support. iSCAT-lite is a homogeneous Sample with an on-axis `ReferenceBeam` of irradiance r²·I and a given phase: the medium belongs to the Sample, so the preset takes no `interface` argument (as built).

**iSCAT, as built (phase 3).** `gx.light.PlaneWave(travel=-1)` is defined in the coverslip at the interface (schema version 2), and `gx.optics.fresnel` holds r and t in wave-vector form on the principal branch, so evanescent waves need no special case. Field amplitudes are normalised to the irradiance in each medium. The reference is the zero order r(g→s)(u_in)·t(g→i)·√(Re k_z,i/k_g), at normal incidence r²·I·p² per pixel (within 10⁻⁶ as built), and the field incident in the sample is t(g→s)·√(n_s/n_g). The backscatter is collected by the layered rule on pupil samples up to the NA, including those beyond n_s, which are evanescent in the sample and transmitted by the interface. The collection weight C/k_z,s folds the f-measure into the transmission, t_s/k_z,s = 2/(k_z,s + k_z,g) and t_p/k_z,s = 2n_s n_g/(n_g²k_z,s + n_s²k_z,g), which stays finite at the critical angle. It is multiplied by t(g→i) and the flux factor √(Re k_z,i/k_s), and at P = 1 the s and p weights mix by the azimuth of the scattering plane, S∥ = c_p·S₂cos²φ + c_s·S₁sin²φ. The pupil phase is k_z,s·depth + k_z,i·focus + the coverslip's Gibson–Lanni mismatch. With all three indices equal, the layered path reproduces the homogeneous one within 10⁻¹⁵ (fp64) for transmitted and epi light alike. The branch point of k_z,s at the critical angle slows the pupil quadrature, so `Coherent(pupil_samples="auto")` takes 128² when the NA exceeds n_s. At NA 1.4 the collected backscatter of a bead of 100 nm diameter changes by 0.68 % from 64² to 128² and by 0.26 % from 128² to 256²; the 10⁻³ goal from 64² was restated accordingly. The contrast oscillates with height at λ/(n_s(1 + ⟨cos θ⟩)), where ⟨cos θ⟩ is averaged over the collected backscatter. With the focus on the particle this is λ/(2n_s) + 1.0 % at NA 0.9 and + 4 % at NA 1.4; a fixed focus adds the defocus phase of the iPSF (0.30 µm at NA 1.4 against λ/(2n_s) = 0.20 µm). `gx.pupil.Filter(radius, transmission, phase)` is a smoothstep disc. The preset's `attenuation` is its amplitude transmission t, on a disc of 5 % of the NA by default, and raises the contrast of particles that fill the pupil by 1/t: 9.96× at t = 0.1 for a bead of 20 nm diameter, falling to 7.2× at 80 nm, where |E_s|² is no longer small. The validity check reports:

- an error for NA ≥ n_i, and an error for scatterers that reach z > 0;
- an info note for scatterers within λ of the coverslip, whose multiple reflections with it are not modelled (cap-32);
- for epi light in a homogeneous medium, a warning when nothing supplies a reference and an info note (the iSCAT-lite approximation tier) when a `ReferenceBeam` does.

`gx.recon.ISCATContrast(reference)` divides by a reference frame and subtracts one. `from_chain` renders the Chain without its scatterers, and without a reference it divides each frame by its median, which is not linear: the bound of ADR-44 takes a fixed reference.

**Shearing interferometry (ADR-45).** A 2-D grating of period Λ sits a distance d before the camera, in image space. Its transmission is a Fourier series t = Σ c_o e^{i2πG_o·r}, so the camera field is Σ_o c_o e^{−iπλdG_o²} e^{i2πG_o·r} E_d(r − λdG_o), where E_d is the image field propagated over d. The propagation over d is a pupil phase (object-space defocus d·n/Mag², paraxial on the image side, where NA/Mag ≲ 0.03), the shift λdG_o is a linear phase on the pupil samples and a translation of each background plane wave, and the tilt G_o is a carrier on the camera grid: each order is exact in the pupil representation and costs one MFT. `kind="checkerboard"` (a 0–π phase checkerboard) and `kind="hartmann"` (Primot's modified Hartmann mask, four equal orders) set the orders; any other periodic mask gives its own. The interferogram carries the harmonics G_o − G_o′, and the harmonic of two orders is E_d(r − λdG_o)·E_d*(r − λdG_o′): its phase is the wavefront difference over the shear λd(G_o − G_o′).

**QWLSI, as built (phase 4).** `gx.optics.Grating(period, distance, rotation=, kind=, max_order=, orders=, coefficients=, focused=)` produces a new carrier, `gx.DiffractionOrders`. It holds the orders' frequencies on the camera side, their coefficients per wavelength bin and the distance, all in fp64 whatever the scene's precision, because the carriers reach hundreds of radians across a frame. `Microscope.detection_optics` holds the grating (schema version 3). The executor passes its orders to the coherent imaging element, which refuses a second grating and refuses image-side references beside one. Λ is the fringe period, and the orders sit at R(rotation)·(m, n)/(2Λ). The kinds:

- `"hartmann"`: four equal orders (±1, ±1) of coefficient 1/2, a lossless idealisation of Primot's mask;
- `"checkerboard"`: the 0–π checkerboard's odd orders up to `max_order`; the light beyond order 3 (19 %) is dropped;
- `"custom"`: any indices and coefficients.

Each order is formed in two parts:

- a phase on the pupil samples and on every background wave: the exact image-side angular spectrum 2πd·[√(1/λ² − |f + G|²) − √(1/λ² − |f|²)], formed without cancellation;
- a carrier e^{i2π·Mag·G·r} on the camera grid (fp64, reduced mod 2π).

That is one MFT per order. Measured on the 3090 (128², 10 beads, B = 64), the Hartmann mask renders 1.5k frames/s forward and 850 forward + backward, against 4.0k and 2.0k for in-line holography: four orders, on a grid sampled twice as finely for the fringes. By default (`focused="camera"`) the camera sits in the image plane with the grating d before it, so each order is an in-focus sheared copy. `focused="grating"` is the form above, with the copies defocused by d. The detection spacing counts the orders' spread, `detection_spacing(separation=λ·Mag·max|G_o − G_p|)`. The validity check warns when the fringes' sideband 1/Λ + NA/(λ·Mag) passes the pixels' band edge.

As built:

- The orders equal a sampled mask propagated by angular spectrum within 10⁻⁴, for both kinds and both focal planes; what remains is the brute force's periodic wrap.
- The Hartmann fringes do not change with d and keep the light: the empty frame's mean is I·p² within 10⁻⁹, and its spectrum holds only the constant and the (±1, 0), (0, ±1) and (±1, ±1)/Λ harmonics.
- A single order shears the image by d·G/√(1/λ² − G²), which is λd/Λ + 2·10⁻⁴. The shear is exact up to the propagation's non-paraxial curvature, 4·10⁻⁵ of the intensity.

`gx.recon.QWLSI` reconstructs in three steps:

1. *Linear part:* it demodulates the constant term and the (2, 0) and (0, 2) harmonics (optionally the diagonals) in soft windows and divides out the pixel MTF. The harmonics' window radius is min(NA/λ + 2 cells, half the fringe frequency); the constant term's window reaches 2NA/λ or the harmonics' windows, whichever is nearer.
2. *Pointwise:* it takes each harmonic's phase relative to the empty field's, and half the logarithm of the constant term's ratio.
3. *Integration:* it solves by least squares with the finite-shear filters T_j(k) = Σ_pairs Re(w_p/W)·(e^{−ik·a_o} − e^{−ik·a_p}), built from the orders' weights and shears (so the checkerboard's unequal pairs and Talbot phases are included), and with T_0 for the amplitude.

Noiseless, against the true field limited to the same window, the phase is within 8·10⁻⁴ rel-L2 for a bead of 1 µm radius and Δn = 0.01 (0.23 rad), and within 7·10⁻⁴ for a 100 nm polystyrene bead (0.011 rad); the checkerboard at Λ = 5 pixels gives 1.3·10⁻². The amplitude channel is limited for phase objects: their intensity signal is second order, and the constant term's window cuts part of its band (15 % complex error for the Δn = 0.01 bead, 0.3 % for the 100 nm bead in a 512² frame). Two findings shaped the build:

- *Calibration precision.* A calibration rendered at another precision than the frames biases every reconstruction that divides by it. A Chain emptied of its populations can compute in float32 where the frames are float64. The 10⁻⁵ rounding of the carriers then became a smooth phase error of about 10⁻³ rad, through the 1/k gain of gradient integration at low frequencies. `gx.recon.without_scatterers(chain)` switches the scatterers off by presence 0 instead, and every `from_chain` uses it. `image_field(part="background")` gives `gx.out.Field(normalize="background")` the frames' precision too.
- *Aliased harmonics.* The camera folds a multi-order mask's higher harmonics onto the fringes. At Λ = 4 pixels the checkerboard's (6, 0) harmonic, at 0.75 cycles/pixel, lands on −1/Λ (31 % phase error on synthetic frames). `QWLSI.aliased()` lists such harmonics and the reconstruction warns. The Hartmann mask has none.

**Reconstructions (cap-41).** `gx.recon` operators are pure torch maps from frames `[B, C, H, W]` to reconstructed data. Each is linear (affine) in the frames up to optional invertible steps (phase extraction and amplitude, pointwise; for QWLSI also the integration of the phase gradients), is built from physical parameters or read from a Chain (`OffAxis.from_chain(chain)` takes the carrier from the reference and the window from NA and the carrier), and applies to measured and rendered frames alike. Being differentiable, they serve fitting, training on reconstructed fields, and design.

**What a reconstruction keeps (ADR-44).** Camera noise is independent across pixels, with variances Σ = diag(v(μ)) (§5.5). A linear reconstruction R maps frames y to z = Ry, whose mean is Rμ(θ) and whose covariance is C = RΣRᵀ; the real and imaginary parts of complex outputs are stacked, because fields reconstructed from real frames are not circular Gaussian. Its Fisher information is F_R = (RJ)ᵀC⁺(RJ), with J = ∂μ/∂θ, which equals ‖ΠWJ‖²: W = Σ^{−1/2} whitens the pixels and Π projects onto the row space of RΣ^{1/2}. The reconstruction keeps exactly the part of the whitened Jacobian inside the subspace it reads, so F_R ≤ F, with equality when R is invertible on the Jacobian's span. Invertible pointwise steps change nothing, so R is the information-losing linear part: windows, crops and projections. `gx.crlb(pipe, chain, wrt=..., reconstruction=R)` computes F_R. RJ comes from forward mode through the render and R, and C⁺RJ from conjugate gradients whose products C·x = R(Σ·Rᵀx) cost one forward- and one reverse-mode pass of R. The result is differentiable without differentiating the solver: with X = C⁺RJ held fixed, F_R = XᵀRJ + (RJ)ᵀX − XᵀCX is stationary in X, so its gradient with respect to any optical parameter is exact. The Gaussian approximation of the propagated noise is reported by `CRLB.explain()`; like the raw bound (§5.5), it counts only the mean's dependence on θ.

**What a reconstruction keeps, as built (phase 5).** `gx.crlb(pipe, chain, wrt=..., reconstruction=R, tolerance=1e-8, iterations=1000)` takes a `gx.recon.Reconstruction` (its `linear` part) or any callable affine in the frames. `is_linear` refuses reconstructions that calibrate on each frame's median, and `Inline.from_chain(normalize=True)` renders a fixed background as the other reconstructions do.

With A = RΣ^{1/2} and c = Σ^{−1/2}J, F_R = ‖Pc‖², where P projects onto A's row space. CGLS on min_y ‖Aᵀy − c‖ finds it: conjugate gradients on RΣRᵀy = RJ, carried on the pixel-space residual s. Each step costs one application of R and one of its adjoint (reverse mode through R). The information ‖c‖² − ‖s‖² rises at every step and never exceeds the raw information. An image stops at the first of:

- ten steps raise its information by less than `tolerance`;
- a step would not shrink its residual;
- `iterations` is reached (with a warning; the result is then a lower bound, so the CRLB errs on the conservative side).

The result is the stationary form in pixel space, F_ij = ⟨v_i, J_j⟩ + ⟨J_i, v_j⟩ − ⟨v_i, Σv_j⟩ with v = Rᵀy held fixed; its gradient through J and Σ is exact at convergence. The solve runs in the basis that whitens the raw information. The radius and index of a 100 nm sphere are nearly degenerate (1 − ρ² ≈ 10⁻¹² in the dipole limit), and a relative tolerance resolves the degenerate direction only when that direction is a column. Measured:

- The bound matches a dense pseudo-inverse within 3·10⁻¹⁵ (32² off-axis).
- An invertible R (`ISCATContrast` with a fixed reference, or a scaling) keeps F within 10⁻⁸.
- The gradient with respect to the reference irradiance agrees with central differences to 3·10⁻⁷, falling as h².
- What a reconstruction keeps depends on the set-up. Through the off-axis reconstruction, a 300 nm bead 0.5 µm out of focus keeps 42–54 % of the raw information (the window discards the in-line interference with the unscattered light), and a 100 nm bead in focus under a strong reference keeps 94 %. A periodic in-line reconstruction keeps 97–99 %, and QWLSI 63 % for the 100 nm bead (per parameter, the diagonal of F). The joint bound on radius and index can lose far more than the diagonal suggests: in-line, a 100 nm bead keeps 99.3 % of each diagonal entry but half the information that separates the two.

Three findings shaped the build:

- *Plain conjugate gradients* on RΣRᵀ were abandoned. Under shaped light the pixel variances span 10¹⁵ (the dark pixels around a focused spot), and the matrix form lost its precision to cancellation. A variance-weighted preconditioner (RΣ⁻¹Rᵀ) slowed convergence instead. The padded in-line reconstruction (zero padding, then cropping) has a continuum of small singular values (time–band limiting), which CGLS resolves slowly: 5·10⁻⁴ short after 50 iterations, always from below.
- *Precision.* `ReferenceBeam` now computes at the precision of its own values. A float64 irradiance under float32 light had left the reference in float32, which limited finite-difference checks to 3·10⁻⁵.
- *Dark pixels.* With an ideal shot-noise camera, pixels that structured illumination leaves nearly dark carry an unphysical excess of raw information (J²/μ as μ → 0). A focused spot gave σ_r = 0.19 nm on the raw frames, against 3.1 µm through the off-axis reconstruction. Designs should model the camera's read noise or a background.

**Designing for the reconstruction.** Because the retained information is differentiable in every optical parameter, the illumination (an illumination-pupil `PhaseSLM` read as J plane waves, §5.11, or a sequence of illumination angles on A), the detection pupil (`PixelPupil`, `Zernike`, `Filter`) and the reference (amplitude, tilt, attenuation) can be optimised for the bound of the parameters a measurement targets, through the reconstruction the instrument will use. This is the DeepSTORM3D idea applied to interferometric characterisation: for nanobeads of 50–250 nm, whose radius and index are nearly degenerate in the dipole limit (α ∝ r³(m² − 1)/(m² + 2)), the design separates them through the higher multipoles and the phase of the scattered field (Ex11).

**Designing for the reconstruction, as built (Ex10, Ex11).** Independent readouts add their information: `bound_a + bound_b` (or `sum(bounds)`) is the bound of frames taken under two states of a modulator, or of two modalities imaging the same sample; it stays differentiable, so a design optimises both states together. At a common dose of 10⁶ photons/µm², with 2 e⁻ of read noise and scalar scattering (P = 1), Ex11 finds:

- *Radius and index are nearly one parameter below ~150 nm.* 1 − ρ² runs from 10⁻⁶–10⁻⁴ at 60 nm to 10⁻²–1 at 250 nm, depending on the microscope. The joint bound on the radius exceeds the radius itself below about 115 nm in iSCAT and 160 nm in the transmission microscopes, while the bound with the index known stays at a few nanometres. For a 100 nm bead through each reconstruction, σ_r is 92 nm jointly in iSCAT and 470–850 nm in the transmission microscopes, against 2–14 nm with the index known.
- *Contrast is not information.* Attenuating iSCAT's reference raises the contrast as 1/t but leaves the bound unchanged while the reference outshines the read noise. It helps only when the camera's full well limits the light.
- *Focusing is not characterisation.* A pupil modulator optimised for one bead focuses the light onto it (a grid of foci, 23× the plane wave's intensity). The gain fades off the design position, and between foci the bound is worse than the plane wave's. Averaging the design objective over bead positions removes most of the gain: coherent waves interfere, so a pupil modulator's illumination always has fringes.
- *Uniform illumination at a designed angle* is translation-invariant by construction, and the design recovers two known techniques. iSCAT chooses total internal reflection just past the critical angle (|u| = 1.356), and off-axis holography, judged through its reconstruction, chooses darkfield (|u| = 1.13 > NA). Each gains 4.6× on the joint radius bound of an 80 nm bead.
- *Two alternating states* at equal photons gain a further 13 % (iSCAT, two total internal reflection azimuths) and 31 % (off-axis: a brightfield state and a darkfield one, whose information is degenerate along different combinations, so the pair's 1 − ρ² exceeds either state's).

These are comparisons, not predictions: total internal reflection and darkfield are where polarisation matters most, so the numbers await the vector model of phase 6.

---

## 6. Inputs, gradients, randomness and labels

gradix has no parameter layer. Its contract with any caller (the user's own sampling code, or DeepTrack2 later) is:
- **tensors in** (schema-described data objects and elements), **tensors out** (images, fields, labels);
- gradients with respect to every tensor input;
- randomness only through explicit keys.

This section specifies that contract. §10 describes front ends (the sampling cookbook, and DeepTrack2 later), and Appendix F holds the measured advice for sampling layers.

### 6.1 The input contract

- **Data objects, elements and Chains** are frozen dataclasses registered as pytrees. Their leaves are tensors, or Python numbers for shared constants.
  - Their static parts are part of the structure signature: population names and kinds, element knobs and enum options, pupil-modifier kinds and order, callables of `gx.SDF`, modules of `gx.NeuralField`, camera shape, acquisition sizes and spectral bin counts.
  - **They store exactly the tensors they are given.** Canonicalising views are created inside each call, never at construction. An element built once around an `nn.Parameter` therefore sees every optimiser update, and it never carries a stale autograd view between iterations.
  - They are edited by copy: `.replace(field=value)` on any of them, or `gx.tree.replace(tree, {"scatterers.cells.obliquity": False})` with pytree attribute paths (Chain field, dictionary key, field).
- **Field schema.** Every field is declared with `gx.field(quantity=…, role=…, event=…, shape_affecting=…, constraint=…, default=…)`.

| Schema entry | Values | Used for |
|---|---|---|
| `quantity` | `length` (µm), `wavelength` (µm, vacuum), `angle` (rad), `index` (complex), `photons`, `irradiance`, `rate` (s⁻¹), `time` (s), `opd` (µm), `dimensionless`, `adu`, `key` (int64) | SI conversion; documentation; unit handling in front ends |
| `role` | `shared` · `image` `[B]` · `frame` `[B, T]` · `object` `[B, T\|1, N]` · `layer` · `wavelength` `[B\|1, L]` | canonical layout, stacking, per-image broadcasting |
| `event` | trailing shape, e.g. `(3,)` for positions, `(4,)` for quaternions | shape checks |
| `shape_affecting` | True for sizes, z, focus, NA, λ, pixel size, magnification, Δn, voxel extents | envelope extraction (§6.4) |
| `constraint` | `positive`, `unit_interval`, `(lo, hi)`, … | validation messages |

- **Canonical layout (per call).** Inside a call every field is broadcast to its full role rank, with singletons where it does not vary. Object fields become `[B, T|1, N, *event]`, and image fields `[B|1, T|1, *event]`. The layout lesson from revision 1 carries over: torch right-aligns when broadcasting, so a per-image `[B]` field combined with a per-object `[B, N]` field silently indexes by object when B == N. With B = N = 8, image 0's objects received the means of images 0–7 [Rev]. The layout is asserted on every call.
- **Presence.** `presence [B, T|1, N] ∈ [0, 1]` weights each object's contribution linearly:
  - coherent scattered field amplitudes are scaled by presence;
  - emitter photons are scaled by presence;
  - volumetric occupancy is scaled by presence in the compositing.

  When presence does not require gradients, slots with presence 0 are culled; the output is unchanged. When it does (relaxed counts in the front end), all N slots are rendered, because the gradient with respect to presence needs their would-be contribution; the cost model charges N [M-J ML].
- **Per-image optics.** Any optics field (light, objective, camera, acquisition; inside a `Microscope` or not) may be per image (`[B]` or `[B, …]`): NA, λ, focus, tilt, Zernike coefficients, gain, read noise, background. The imaging element builds batched pupils `[B, L, P, K]`, which are small, and the Pipeline disables cross-image hoisting for them.
- **Units.**
  - Everything is in µm (§4.1).
  - `gx.from_si(tree)` and `gx.to_si(tree)` (any pytree of data objects, elements, Chains or containers) convert every field by its quantity's power of length, recursively: lengths, wavelengths and OPDs by 10⁶, frequencies and attenuations by 10⁻⁶, irradiances by 10⁻¹², densities by 10⁻¹⁸; dimensionless quantities, angles, counts, times and keys stay unchanged.
  - Pixel coordinates convert through `gx.coords.from_pixels(p, camera, objective)` and `to_pixels(x, camera, objective)`.
- **Stacking.** `gx.stack(items)` stacks B per-image pytrees of identical structure (data objects, elements, Chains, Samples, Microscopes) into one batch.
  - It pads each population to a bucketed N (8, 16, 32, …) with presence 0.
  - It checks that structures agree, naming the first difference.
  - It works on CPU tensors, e.g. in a front end's DataLoader workers; the batch is then moved with `.to(device)`.
  - Populations whose kinds differ between images are an error. The front end groups them, or declares the union with presence masks.
- **Structure signature.** The signature covers:
  - data-object and element types, population names and kinds;
  - slot capacities (N buckets) and T;
  - which optional fields are present, and the role pattern (shared or per image) of each field;
  - dtypes, modifier lists, camera shape, acquisition sizes, spectrum bins and polarisation basis.

  Together with the capacities, the declared inputs, the outputs, the device and the envelope bucket, it keys a Pipeline's captured graphs, and with the fidelity added, `gx.render`'s plan cache.
- **Validation.** Constructors check shapes, dtypes and structure. Value constraints (radius > 0, presence ∈ [0, 1], NA ≤ n_immersion) are checked by `gx.stack` on CPU inputs, where they are free, and by `validate()` on demand. They are never checked implicitly on GPU inputs, where each check would be a host synchronisation.

### 6.2 Gradients

- **Every tensor leaf may require gradients.** Gradients flow through the elements of the Chain, whether the leaf is a front-end `nn.Parameter`, a value the front end sampled with reparameterised torch operations, or a neural field's weights.
- **Gradient table.** `pipe.gradients(chain=None)` (and `plan.gradients(sample, microscope)`) lists every field of the given inputs that requires gradients, with its quality; without arguments, it lists the quality of every field path of the structure:
  - `exact`;
  - `exact-a.e.`: region-dependent zeros such as clips, saturation, or hard edges pinned outside the learnable range;
  - `biased`: surrogate or estimator paths, e.g. scaled-ST Poisson on losses beyond quadratic in counts, or cross-fidelity surrogate backwards (v1.x);
  - `zero`: e.g. `gx.Occupancy(..., hard=True)` masks, or a hard aperture edge pinned by the user.

  The qualities come from the implementations' capability tags (P4 in §5.7). **A field that requires gradients when every path from it is zero raises `GradientPathError` at call time**, naming the stage and the fix; this costs one cheap Python check of `requires_grad` flags. A field with only some paths zero gets a warning naming the missing path.
- **Callables and modules** receive gradients because lowerings take their parameters as explicit inputs (§4.6). `gx.pupil.ModulePupil(module)` and `gx.NeuralField(module)` are called through `torch.func.functional_call`.
- **The gradient set never changes the forward values.** Which inputs require gradients may change only the memory strategy (checkpointing, recompute-in-backward), never the images, the source sampling or the grids (ADR-12). A CI test renders the same inputs and key with and without `requires_grad` and requires identical outputs.
- **Dynamic probe.** `gx.testing.gradient_probe(pipe, chain)` runs one forward and backward at B = 2–3, with taps that name the first stage where the vector-Jacobian product vanishes. It uses two functionals:
  - a random linear functional of all outputs;
  - the noise-energy functional Σ(y − sg(ȳ))².

  Plain straight-through Poisson passes the first functional (|g| = 91.8) but gives exactly 0 on the second [Rev].
- **Estimators inside rendering** are limited to the camera and to numerical randomness. Estimators for *scene* draws (counts, choices, truncations) belong to the front end (Appendix F).

| Stage | Default | Alternatives | Evidence |
|---|---|---|---|
| Poisson shot noise | **scaled straight-through**, implemented as y = n + (λ − sg(λ))·sg(g) with g = where(λ > 0, (n + λ)/(2·max(λ, tiny)), 1). The forward gives exact integer samples. The gradient is (n + λ)/(2λ) = 1 + ε/(2√λ) for λ > 0 and finite (= 1) at λ = 0, where the textbook form λ + √λ·sg((n − λ)/√λ) gives NaN. **Unbiased for losses up to quadratic in counts; biased otherwise** (+2 to +10 % at λ ≲ 3 for Anscombe and log1p losses; per-pixel gradient variance 1/(4λ)) [Rev] | Gaussian reparameterisation (λ ≳ 20); `expected` + `gx.detect.log_prob` (recommended for fits dominated by pixels with λ ≲ 3) | On a variance-sensitive loss, plain straight-through gives **exactly 0**; scaled ST gives −19.95 against the true −20; the score function's std is ~6000× larger [M-D Exp B] |
| EM gain | exact Gamma–Poisson forward with the double-where + straight-through form of §5.5 | — | unbiased at λ = 0.2 / 1 / 3 [Rev] |
| Read noise | Gaussian reparameterised | — | |
| Quantisation / saturation | ST round / hard clip (zero gradient beyond the clip; exact-a.e., declared) | soft clip when learning | |
| Device quantisation (SLM phase levels; DMD states in v1.x) | straight-through, declared `biased` | a relaxed softmax over levels with annealing; continuous design, quantised evaluation | |
| Stochastic source nodes (1.x) | pathwise through realised modes: unbiased for the image and for losses linear in it; biased by O(Var(I_n)) ∝ 1/n_src for nonlinear fitting losses | increase n_src; deterministic quadrature for final fits | §5.4 |

### 6.3 Randomness: explicit keys only

gradix owns no random state. Randomness appears inside rendering in exactly three places, and each is a deterministic function of an explicit key:

| Randomness | Key | Default when absent |
|---|---|---|
| Detector noise (`"image"` output of a camera with a noise model) | `key=` of the render call (`pipe(…)` or `plan(…)`) | `"image"` is not produced without a key; `"expected"` is always available |
| Texture realisations (`gx.Texture`) | per-object `key` field `[B, 1, N]` int64 in the data object | required field |
| Numerical randomness: speckle realisations (v1); stochastic source nodes (1.x) | `numerics_key=` of the render call | required when such an implementation is selected |

- **Key forms.**
  - An `int` is **batch-keyed**: a `torch.Generator` seeded from it, reproducible for the same batch.
  - A `Tensor[B]` of int64 is **image-keyed**: every draw is a counter hash of (key_i, stage, element), so image i's noise is independent of batch composition and order. This is the `Dataset[i]` semantics that reproducible datasets need.

  Image-keyed sampling covers the Ideal, Poisson–Gaussian and sCMOS cameras in v1. EMCCD's Gamma stage is batch-keyed until 1.x, and an image-keyed EMCCD call raises with that explanation. Batch-keyed Poisson draws use `torch.poisson` for speed, and its CUDA sampler is only approximate above λ ≈ 1000: at λ = 4000 the variance is 0.9 % low and the third moment 1.8× high, above λ ≈ 5000 it is a normal approximation, and means stay within 0.01 %. Image-keyed draws are exact but ≈10× slower. A keyed Poisson sampler that is both exact and as fast as `torch.poisson` (a fused kernel) is a 1.0 requirement (M5).
- **Counter hash.** The default is a splitmix64-class finaliser emulated in int64: 30 kernels, ≈0.3 ms eager and ≈0.05 ms as a CUDA graph for 64 × 1020 draws plus a Normal transform [Rev]. Philox-4x32-10 is the high-quality option. Draws map to open-interval uniforms u = (k + 0.5)·2⁻ᵇ.
  - Gaussian draws use the inverse CDF.
  - Poisson draws use the inverse CDF for λ < 10, and transformed rejection (PTRS) with a fixed number of attempts above. The rare pixels that exhaust their attempts take an exact sequential fallback with a static iteration cap.

  The specification is fixed in M0 and tested for exact Poisson statistics (§11.7), with an RNG quality suite (a PractRand or TestU01 SmallCrush subset; cross-image and cross-element correlation checks).
- **Checkpointing is safe by construction.** Keyed draws are pure functions, so recomputing a checkpointed region redraws identical values. Batch-keyed generator draws are made *before* any checkpointed or recompute-in-backward region and passed in as tensors. `preserve_rng_state` protects only the default generators: a backward recomputation that redrew from an advanced explicit generator silently changed a gradient from 1.194 to 0.998 in a measured case [Rev].
- **CUDA graphs.** Keys enter captured graphs as device tensors that are copied into before each replay; a Python-int key would be baked into the captured kernels, and every replay would return the same noise.
- **Purity guarantee.** A Pipeline call `pipe(chain, key=k)` is bit-exact across calls on the same device class and version for a Pipeline built with `deterministic=True` (sort-based splatting instead of atomic scatter-adds; part of the Pipeline hash) and the Pipeline's recorded chunk sizes. With atomics, re-renders differ at the ulp level: 20 re-renders differed in 4.8M expected pixels and hence in ~700 Poisson counts [Rev]. `out.meta` records which mode was used.

### 6.4 Envelopes

An **envelope** is the set of static intervals for every shape-affecting field: sizes, z positions, focus, defocus range, NA, condenser NA, λ range, pixel size, magnification, slot counts N, Δn range and voxel extents. From these the envelope analysis (P1) derives the invariants that set grids, pads, ROIs, Mie n_max, slices and validity: x_max, φ_max, Δn_max, extent_z, the defocus range, NA_ill + NA_det and the λ range.

- **Sources**, later ones overriding earlier ones entry by entry:
  1. **derived** from the example Chain given to `gx.Pipeline` (or the inputs given to `gx.plan`), with headroom: spans are widened by 25 % and rounded outward to buckets (0.25 µm for z and focus, a 10 % log step for sizes). `gx.envelope_of(values)` derives the same entries from a mapping of whole per-image tables, respecting presence, so a fit over a dataset plans its grids once for every image (§6.9);
  2. **probed** by the front end, e.g. by drawing a few hundred samples from its sampler at construction;
  3. **explicit**, `gx.Envelope({"beads.radius": (0.05, 1.5), "beads.position.z": (-6.6, 6.6)})`.

  Sources combine with `|`, later entries overriding earlier ones entry by entry (`gx.envelope_of(data) | {"beads.position.z": (-8, 8)}`), and `explain()` lists every entry with its source.
- **Paths.** Envelope keys, `per_population` overrides, label arguments and `explain()` share one small path grammar:
  - `<population>.<field>[.<component>]`, e.g. `beads.radius`, `beads.position.z`;
  - `<part>.<field>`, where the part is a Chain field (`light`, `camera`, `acquisition`, `environment`, …) or `objective` (the imaging element's objective), e.g. `objective.focus`, `light.wavelength`, `camera.pixel_size`; Microscopes use the same names;
  - `<stage>.<field>` for named elements of `illumination_optics` and `detection_optics` (e.g. `d2nn.offsets`), and `<stage>.device.<field>` for a device's parameters (e.g. `slm.device.phase`).
- **Checks.**
  - A Pipeline call computes per-image in-envelope flags on the device without a host synchronisation and returns them in `out.meta["in_envelope"]` for the caller to read; the Pipeline keeps no state between calls. With `check="raise"` the call synchronises and raises instead.
  - When the inputs are on the CPU, as when a front end has just sampled them in workers, `gx.envelope_of(chain)` is free. The front end then grows the envelope and re-plans before rendering: the **pre-render check** (Appendix F.8).
  - `gx.render`, the convenience entry point, performs that check itself. It re-plans synchronously with a grown envelope when needed and logs each re-plan.
- **Out-of-envelope values are never clamped.** They are rendered with the current Pipeline: the shapes are static, and accuracy may degrade (pads too small, Mie truncated). They are flagged in `out.meta`, and `validate()` or `gx.envelope_of` names the field and the suggested envelope. Clamping would create point masses at the edges and bias whatever the front end is learning [M-J ML].
- **Learnable shape-affecting fields.** If an `nn.Parameter` drives a size, a z position, NA or λ, give it an explicit envelope entry that covers the range it may explore. Otherwise the derived envelope (×1.25 around the initial value) applies, and drifting out of it triggers the flag or a re-plan. Full physical domains (a whole sample layer for z, NA up to n_immersion) are never the default, because they would inflate pads, Mie n_max and K (risk 12).

### 6.5 Batching, outputs and ownership

- **B leads everywhere.** Per-image fields are `[B]`. Ragged counts are padded slots with presence; nested tensors are not used, and a packed view `[ΣN, …]` is available to per-object kernels. Acquisition sizes (A: frames, focus steps, LEDs, SIM phases) are static per Pipeline, while B and each population's N are capacities (§6.9). The Pipeline chunks B·A exactly, and its memory estimates include A.
- **Subsets of an acquisition axis** are just different inputs. To render 9 of 81 LEDs, pass a Chain (or Microscope) whose `LEDArray` holds those 9 (`leds.select(idx)`). A new A size needs a new Pipeline (cached by `gx.render`); the same size with different LED positions uses the same one.
- **Outputs.** `outputs=` of `gx.Pipeline` (or `gx.plan`) names what is computed: a dict from names to output specs, or a tuple of reserved names. Only requested outputs are computed. The call returns an `Output`: a mapping whose outputs are also attributes (`out.image`; `out.pos` for an output named `"pos"`; a label's extras keep dotted keys such as `out["pos.in_fov"]`), and whose `meta` holds the Pipeline hash, chunk sizes, key kind, in-envelope flags and optional diagnostics. The specs are:
  - `"image"`: the camera unit, requires a key if the camera has noise;
  - `"expected"`: the noise-free mean in the camera unit;
  - `gx.out.Field(...)`, `gx.out.Tap(name)`;
  - label renderers (§6.6).
- **Generation mode.** Rendering runs under `torch.no_grad()` automatically when no input requires gradients, never under `inference_mode`. Tensors created under `inference_mode` cannot be saved for backward: a network trained on them raises at its first `backward()`, and a cache created under it breaks later grad-mode calls [Rev].
- **Output ownership.** Every returned tensor is an ordinary tensor that the caller owns. Captured CUDA-graph outputs are static buffers that the next replay overwrites (a held batch changed from 0.001 to 0.842 after one replay [Rev]). `Pipeline.capture` therefore copies final outputs into a *fresh* allocation from the caching allocator, with stream-ordered synchronisation. Buffer reuse is an explicit opt-in (`reuse_output_buffers=True`) with a documented lifetime.
- **Re-drawing only the noise** of a fixed batch: `mu = pipe(…)["expected"]`, then `gx.detect.sample(mu, camera, key=k)` for as many keys as wanted.

### 6.6 Labels and coordinates

- **Value labels are the front end's.** Sampled positions, radii, identities and on/off states are values the caller already holds. gradix does not echo its inputs back as labels.
- **Geometry-derived labels are gradix's**, because they need the lowering, the coordinate frame and the camera grid. They are requested as outputs of the same call, so image and labels share one frame and one sub-pixel convention:
  - `Positions(pop, unit="px"|"um", dims="xy"|"xyz")`, with in-FOV flags;
  - `InstanceMask`, `SemanticMask` (hard or soft occupancy; `parts=` for hierarchies);
  - `Heatmap`, `DistanceMap`;
  - `OPL` / `OPD` maps; `Phase()` (arg(E/E_b) on the camera grid);
  - `EmitterTable` (per frame, padded, with presence, the photons rendered in that frame, in-FOV flags and `id`);
  - `PerObjectImages` (free from sparse ROIs).

  The same renderers are available standalone, `gx.labels.render(label, objects, camera, objective)`, e.g. for a front end's mask features.
- **Coordinates.** `gx.coords.to_pixels` and `from_pixels` apply one pinned pixel convention (integer pixel indices at pixel centres, row/column order as in §4.1); DeepTrack2's convention is recorded in Appendix F.8. `field_of_view(camera, objective)` gives the frame's extent in µm, so `torch.rand(B, N, 2) * gx.coords.field_of_view(camera, objective)` is anywhere in the image, and `pitch(camera, objective)` is a pixel as a length unit. `depth(d)` (z of a point at depth d into the sample, §4.1), `from_focus(dz, focus)` and `shift_focus` are pure helpers for the z semantics of §4.1.
- **Augmentation is the front end's**, including per-replicate parameters and label co-transforms. Physics-side augmentation (random defocus, rotation, illumination tilt) is simply different input values and keeps every label exact.

### 6.7 Sequences

1. **Frames live on A.** Any field with a frame role carries T: positions `[B, T, N, 3]`, photons, presence, focus drift, illumination flicker. `gx.acq.Frames(n, interval=Δt, exposure=t_exp ≤ Δt)` records the timing. v1 renders each frame at mid-exposure; motion blur arrives with cap-24 (v1.x) as sub-exposure positions `[B, T, n_steps, N, 3]` on M.
2. **The front end produces the dynamics**: trajectories (Brownian, drift, confinement), appearance and disappearance (per-frame presence), and blinking states (per-frame photons). The cookbook shows them in torch (§10.3); DeepTrack2's `Sequence` expresses them too. Reparameterised increments written in torch keep the diffusion coefficients learnable.
3. **A is never reduced**, so the Pipeline chunks B·A. For example, 64 sequences × 100 frames × 288² in complex64 is 4.2 GB per coherent tensor, so chunking is required, not optional.
4. **Positions are not confined to the FOV.** Labels carry in-FOV flags per frame.

### 6.8 What the front end owns, and what gradix gives it

| The front end (the caller's sampler; DeepTrack2 later) owns | gradix provides |
|---|---|
| Sampling rules, dependencies, replicates, random counts, sequences, photophysics | Schema-described data objects and elements that any front end can construct and wrap (§6.1); `gx.stack` and `gx.pad` for batching; value binding by field path (§6.9) |
| Learnable distributions; estimators for discrete draws; truncation (Appendix F) | Exact gradients with respect to every input; `gradients()`; presence rendered linearly |
| Envelope probing, and adapting sampling to validity | `gx.envelope_of`, envelopes in `explain()`, `validate()` with per-image and per-object violation masks |
| Value labels, augmentation, datasets, loaders | Geometry-derived labels and coordinate conversion |
| Calibration and fitting loops, distribution-matching objectives | `gx.detect.log_prob` and `gx.detect.fisher` (camera likelihoods), `gx.crlb`, `"expected"` outputs |

### 6.9 Feeding data to a built Pipeline

A Pipeline is built once and then fed data. Its two call forms are one function:
- `pipe(chain, key=…)` takes any structure-compatible Chain (§6.1). It suits callers that build new data objects and elements per batch, as a sampler does (§9a).
- `pipe(values, key=…)` takes a mapping from field paths (§6.4) to new values and binds them onto the Chain the Pipeline was built from, `pipe.template`. A path names a field (`"beads.position"`, `"objective.focus"`) or a whole data object or element (`"beads"`). Fields that are not named keep the template's tensors, read at call time, so a shared `nn.Parameter` in the template is used with its current value. `pipe.bind(values)` returns the bound Chain, and `pipe(values) ≡ pipe(pipe.bind(values))` is a CI test (§11.10). At L4, `gx.plan` takes the same `inputs=`, and `plan(values, key=…)` binds onto its `Sample` and `Microscope` in the same way.

The rules:
- **Declared inputs.** `gx.Pipeline(chain, inputs=("beads", "objective.focus"), …)` names the paths that calls may bind. Declared inputs are planned at the widest role their schema allows (per object, per image), so narrower values, such as a shared scalar, broadcast into them. A path that is not declared may be rebound only with the role its template value has; a wider value raises and names `inputs=`. Structural fields (knobs, camera shape, acquisition sizes) cannot be bound and need a new Pipeline. A captured graph (`pipe.capture()`) copies what a call supplies into its static buffers: every tensor of a whole Chain, or the declared inputs of a values call, and binding an undeclared path raises. The other template tensors are read in place, so in-place optimiser updates are seen.
- **Capacities, not sizes.** The template's batch size (bucketed, or `batch=`) and each population's slot count (or `slots={"beads": 32}`) are capacities. Calls may bind a smaller batch or fewer slots: eager calls run at the given sizes, and a captured graph pads to its captured sizes (presence 0, empty images) and returns only the real images. Per-image results do not depend on this padding (§11.10). A larger batch or slot count raises; `gx.render` re-plans.
- **Ragged object counts.** `gx.pad(*ragged, n=None)` turns lists of per-image `[N_i, …]` tensors into padded `[B, n, …]` tensors and a presence mask, with present slots first; n defaults to the smallest bucket (8, 16, 32, …) that holds the largest N_i. A dataset becomes one table per field. Sorting images by count and slicing each batch to its own largest count (`pos[idx, :n]`) keeps the per-object work close to the number of objects present.
- **Envelopes from the whole dataset.** `gx.envelope_of(values)` accepts whole per-image tables, respecting presence (§6.4), so no batch of a fit leaves the envelope. Values the fit may move, such as refined z positions, still need an explicit entry for the range they may explore: `gx.envelope_of(data) | {"beads.position.z": (-8, 8)}`.
- **Settings known per image.** Settings the user changed between acquisitions (focus, exposure, illumination power, LED, wavelength, camera gain) are per-image values bound at their paths. They may be computed in torch first, for example a recorded focus plus a learned shared offset: `"objective.focus": focus_set[idx] + dz`.
- **Per-image values to refine.** A per-image table, such as rough positions, is gathered by index for each batch, and gradients reach only the rows gathered. The optimiser must also leave the other rows alone, and dense Adam does not: its momentum moved rows that were absent from three consecutive batches by 1.6 learning-rate steps, where `torch.optim.SparseAdam` on a sparse embedding moved them by 0 [Rev]. When no parameter is shared between images, the images are independent problems: fit each batch to convergence and move on (§9g, Appendix F.5).

---

## 7. Extension and composition model

### 7.1 Registries and capability negotiation

There is one registry per extension point: `object_set`, `view`, `lowering`, `element` (one per slot), `pupil_modifier`, `label`, `preset`. Registries are populated by decorators. Entry-point discovery (`[project.entry-points."gradix.plugins"]` plus a `GRADIX_PLUGINS` environment variable for development) arrives in v1.x together with plugin API versioning. Duplicate names are errors unless `override=True`. Saved data objects, elements and Pipelines use **registry names plus a schema version**, never import paths [B09 §1.11]. Every registered data object and element also publishes its schema (`gx.schema_of`), which any front end can use to construct or wrap it (§10.2).

The extension contract is **data types plus capability declarations**, not subclass internals. A Chain checks that consecutive elements agree on carriers (`accepts`/`produces`), ports and polarisation; a Pipeline also checks validity and gradient-quality tags, and the planner matches knob names. Fidelity knob values are registry names, so a plugin is selected exactly like a built-in: `Fidelity(spheres="tmatrix")`.

### 7.2 Composition

There is no operator-overload soup [B09 anti-pattern 7]. Composition happens in a few typed places:
- `gx.Chain(...)`: elements along the slot skeleton, each consuming the carrier the previous slot produces (§3.1);
- `Sample({name: object set})` with `gx.Parent` hierarchies (priority compositing);
- `Objective(pupil=[modifiers…])` (fused into one multiplier, §5.3);
- presets (recipes assembling `Microscope(light, objective, camera, acquisition)`);
- acquisition axes (`FocusStack`, `Frames`, `LEDSequence`, `SIMPhases`) → A;
- detection-path splitters → C;
- the same inputs rendered through several microscopes or fidelities: plain repeated calls (§9c, §9d).

Sample-side references (in-line, reflected) are extra `PlaneWaves` in the same modes; the off-axis reference is an image-side term that Detect adds after the pupil (§4.5), so off-axis holography still needs no separate combinator. Power users may call `gx.ops.*` functionally.

### 7.3 Adding a primitive

```python
@gx.register.object_set("torus")
@dataclass(frozen=True, kw_only=True)
class Tori(gx.ObjectSet):  # position, rotation, presence, material, labeling, parent, id inherited
    # µm, [B, T|1, N]
    major: Tensor = gx.field(quantity="length", role="object", shape_affecting=True)
    minor: Tensor = gx.field(quantity="length", role="object", shape_affecting=True)

    def bbox(self):  # [B, T|1, N, 2, 3] local box; the base maps it to the world box of §4.6
        r = self.major + self.minor
        return gx.geom.local_box(r, r, self.minor)

    def sdf(self, x):  # x: [B, T|1, N, n_pts, 3] local coordinates (µm); exact SDF
        q = torch.stack([x[..., :2].norm(dim=-1) - self.major[..., None], x[..., 2]], -1)
        return q.norm(dim=-1) - self.minor[..., None]

    # optional: form_factor(k) → born_ff, OTF labels; project(xy) → analytic projection
```

With `sdf` alone, the torus renders in projection, multislice, dense fluorescence (via `Labeling`), masks and distance maps. Its fields receive gradients through the SDF ramp: the rasteriser passes the data-object tensors as explicit inputs of its recompute-in-backward Function (§4.6). Its schema documents its fields for any front end, and every interaction element that consumes `sdf` accepts it at once. Adding `form_factor` unlocks `born_ff` and analytic density spectra. The same shape can be written without a class, as `@gx.geometry(kind="sdf", bbox=lambda major, minor: …)` on `def torus(x, *, major, minor)`, or ad hoc as `gx.SDF(fn, params={"major": …, "minor": …}, bbox=…)`. Occupancy-only shapes are accepted with a reported gradient quality (§4.6).

### 7.4 Adding a solver

```python
@gx.register.element(slot="interact", name="wpm")  # wave-propagation method as a march mode
@dataclass(frozen=True)
class WPM(gx.Element):
    volume: gx.ObjectSet  # the objects it renders, held as gx.interact.Multislice holds them
    dz: float | str = "auto"  # a knob: "auto" uses the default rule; a Fidelity may pin it
    caps = gx.Capabilities(
        accepts={gx.views.OccupancyChannels},
        produces=gx.Field,
        ports={"+z"},
        polarization={1},
        linear_in_field=True,
        grad={"checkpoint"},  # grad_quality omitted: defaults to exact
        approximates=(gx.Edge("interact.mbs", regime="piecewise-constant n, forward"),),
    )

    def validity(self, desc, envelope):
        return (
            [gx.Violation("warn", "cost ∝ index levels", "n_levels", envelope.n_materials, 6)]
            if envelope.n_materials > 6
            else []
        )

    def configure(self, desc, envelope):
        dz = gx.planning.slice_dz(envelope) if self.dz == "auto" else self.dz
        return dict(dz=dz, levels=envelope.n_materials)

    # optional
    def cost(self, p):
        return gx.Cost.fft2(p.grid, count=p.n_slices * (p.levels + 1), batch=p.batch)

    # pure torch: lowers self.volume onto static's grid and marches
    def forward(self, modes, env, static):
        # in the residual-demodulated frame (gx.ops.propagate(..., residual_tilt=))
        ...
```

`gx.testing.conformance(WPM)` runs the same bar as core code [A §7.2, B §7]:
- `gradcheck` in complex128 with `check_forward_ad=True` and `check_batched_grad=True`, plus `gradgradcheck`, on tiny shapes that **include removable singular points** (q = 0 and k = k_in for form factors; x → 0 for Mie/dipole spectra and Bessel ratios; θ = 0);
- linearity in the field;
- energy conservation for lossless samples;
- **tilt equivariance**: a one-bin change of the residual tilt δk equals a spectrum roll, up to band-edge content;
- convergence to the declared `approximates` target within its regime;
- `validity` (and `cost`, if declared) sanity;
- CPU/CUDA parity;
- purity: no global RNG, no stateful-generator draw inside checkpointed regions, deterministic under fixed keys;
- carrier shapes that broadcast over B and A;
- the non-zero-gradient check: every input field that requires gradients and reaches the implementation, including through a callable or an `nn.Module` field, receives a finite, non-None, non-zero gradient.

**Custom autograd Functions** (in core and in plugins) use the `forward` + `setup_context` form with a `jvp` staticmethod (or `generate_vmap_rule=True` for pure-torch bodies), and write `backward` in differentiable torch operations (no `once_differentiable`) unless impossibility is documented. `gx.crlb` needs J = ∂μ/∂θ for a few parameters over ~10⁴ pixels, which forward mode (`jacfwd`) or a double backward computes cheaply; a Function without vmap/jvp support or with a once-differentiable backward leaves only one VJP per pixel [Rev]. `gx.crlb` uses forward mode: one `torch.func.jvp` per parameter of an image, for every image at once, since images are independent. Values shared by all images are single parameters, combined with the per-image ones through a Schur complement of the Fisher information. When forward mode is unavailable it falls back to central differences with a warning, and `CRLB.method` and `CRLB.explain()` report which was used (as built, 2026-09-29). **Removable singularities** (q = 0, x = 0, θ = 0, k = k_in) use the double-where pattern: a safe argument substituted in the unused branch plus a Taylor branch. A single `torch.where` returns NaN gradients, because the unused branch still backpropagates 0·∞; on-axis illumination always samples q = 0 exactly [Rev].

A plugin that passes is usable at every level at once, and immediately serves every modality that uses the Interact slot: directly (`WPM(volume)(light(), env)`), in hand-built Chains, and through the planner (`Fidelity(volumes="wpm")`).

### 7.5 Adding a pupil modifier or a modality

```python
@gx.register.pupil_modifier("vortex")
@dataclass(frozen=True)
class VortexPhase(gx.PupilModifier):
    charge: Tensor | float = gx.field(quantity="dimensionless", role="image", default=1.0)

    def __call__(self, fx, fy, wl, ctx):  # continuous pupil coordinates → complex multiplier
        return torch.polar(torch.ones_like(fx), self.charge * torch.atan2(fy, fx))


@gx.register.preset("spiral_phase_contrast")
def SpiralPhaseContrast(*, wavelength, objective, camera):
    # its unscattered light is the in-line reference
    return gx.Microscope(
        light=gx.light.PlaneWave(wavelength),
        objective=objective.with_pupil(VortexPhase(1)),
        camera=camera,
    )
```

A Hoffman modulation-contrast preset is a slit source plus a graded-amplitude modifier: about eight lines, no new kernels [B05 §3.3]. A modifier may also return a 2×2 Jones multiplier, as `gx.pupil.Polarizer`, `Retarder` and `JonesMatrix` do; any Jones modifier makes the imaging element run at P = 2 (§4.1).

### 7.6 Why the whole exceeds the sum of its parts

- **Orthogonal axes meet only through shared types** (the carriers, the data objects, the views). Contributions therefore multiply, while integration cost grows as a *sum* over the axes instead of their *product* [B §1]:
  - a new *primitive* with `sdf` works in every coherent tier, in fluorescence and in every label renderer, and is immediately learnable;
  - a new *interaction element* upgrades every transmitted, reflected and interferometric modality;
  - a new *pupil modifier* is available at once to fluorescence, brightfield, holography, iSCAT and SIM, on the sparse, dense, coherent and incoherent paths alike;
  - a new *camera* serves every modality.
- **Levels compose.** The same elements serve direct calls, hand-built Chains and planned ones. A new element is usable at all three levels the moment it is registered, and the planner can choose it by name. Starting from a planned Chain and swapping one element by hand is a single `replace`.
- **Programmable optics meet simulated optics.** A device model (an SLM, a fabricated diffractive layer) works in any stage of any modality, and one differentiable twin serves design, joint training and physics-aware training on hardware (§5.11).
- **The same inputs, many renderings.** One resolved sample renders through several microscopes (virtual staining, multimodal pretraining) and at several fidelities (`gx.compare`, surrogate gradients), with keyed textures and noise so that only the physics differs.
- **One forward model, four uses:**
  - data generation, with the caller's sampling and a Pipeline;
  - calibration, through `expected` and `gx.detect.log_prob`;
  - design, with learnable optics trained jointly with a network;
  - reconstruction, with autograd now and adjoints and deepinv export in v2.
- **Front ends stay thin.** Data objects and elements are schema-described pytrees, so any front end can construct, stack and introspect them without adapters maintained in gradix: a user's sampler, a GUI, or DeepTrack2 later.
- **Reference elements manufacture fast ones.** Ladder tables calibrate validity thresholds now. In v2+, reference outputs train learned-surrogate elements that plug into the same slots [B04 §3.7].

---

## 8. Performance and memory engineering

### 8.1 Measured anchors (RTX 3090, torch 2.14+cu130) and what they imply

This table holds the **performance** anchors. Accuracy evidence is stated once, in the section that uses it (§4.1, §4.3, §4.4, §5.2–§5.4, §6.2–§6.3), and the decision records cite sections instead of restating numbers.

| Measurement | Result | Design consequence |
|---|---|---|
| c64 `fft2`, 64×512² vs same-size elementwise multiply | 0.705 ms vs 0.525 ms [B07 §2.3] | split-step loops are bandwidth-bound → fuse diagonals; FFT speed is not the lever |
| c128 FFT; complex32 | 7.5× slower; NaN in a 100-slice march [B07] | complex64 everywhere |
| prime FFT sizes | up to 1.9× slower [B07] | 2·3·5·7-smooth static sizes |
| complex MFT under TF32 vs IEEE fp32 | 4.4·10⁻⁴ vs 2.9·10⁻⁷ rel. error [M-A] | `ieee_matmul()` around every MFT/CZT; `no_autocast()` around every render |
| tiny-op launch; CUDA-graph replay | ≈8 µs/op; 2.29 → 0.80 ms (2.9×) on a 32-slice 256² forward [B07 §3.3] | CUDA graphs are the first accelerator |
| sparse emitters, MFT → ROI | 52k img/s fwd, 10k fwd+bwd (B64×32, 128²) [M-C]; 51k / 18.7k (B64×50, 128²) [M-D]; 67k frames/s (B256×20, 64²) [M-A]; vectorial 6-basis 10.8k frames/s [M-A] | sparse path is the default for points |
| full-frame per-emitter FFT vs MFT-ROI | 7.2k vs 52k img/s [M-C] | never full-frame per emitter by default |
| sparse vs dense strata | 3.8–5.7× faster fwd at ≤ 200 emitters [M-D]; crossover ≈300–500 emitters per 256² [M-B] | deterministic path rule on N_max |
| ROI 32² where 64² was needed | 5.7 % rel-L2 error, energy conserved [M-D] | ROI derived from the envelope (correctness) |
| Mie-in-pupil, B64 × 5 spheres, 256² | 1 mode: 3202 → **5020** img/s evaluating on pupil support only; 7 Köhler modes: 842 → **2296** img/s [M-J eng]; full-grid W3: 22.7 ms fwd, 98.6 ms fwd+bwd [M-C] | evaluate sparse coherent objects on pupil-support samples |
| Mie coefficient recurrence | launch-bound ≈8.6 ms for n_part = 5…4096 particles; 3.3 ms as a CUDA graph [M-C]; complex64 D_n errs 1–5 % [B07 §5.2] | fp64 on GPU, all particles × λ in one captured call, n_max from the envelope |
| batched vs per-sample Mie (64 images) | 21.7 vs 713 ms (33×) [M-C] | one batched call per kernel kind |
| dense fluorescence strata, B16 × Z32 × 256² | 2.2 ms fwd, 4.6 ms fwd+bwd (7.3k img/s) [M-C] | |
| BPM 256² × 64 slices, B64, inference | precomputed slices 47.5 ms (1347 img/s); procedural slices of 6 soft ellipsoids 289.5 ms (221 img/s) [M-J eng] | rasterisation dominates → pre-rasterise once per batch in inference; first Triton target |
| multislice march forms, B64, 256² × 64 slices, inference | total-field march 1337–1351 img/s; scattered-field march with analytic E_b 920–926 img/s (1.46×); with precomputed complex transmittance and in-place updates 1562 vs 1123 img/s (1.39×) [Rev] | residual-demodulated total-field march by default; scattered form opt-in (§4.4) |
| BPM with gradients, B8 | naive 6.55 GiB / 134 ms; procedural + non-reentrant checkpoint every 8 slices 0.90 GiB / 182 ms, bit-identical gradients [M-J eng] (C: 6.5 → 0.88 GiB [M-C]); re-runs of the judge script: 41–44 img/s, 44 img/s in the scattered form [Rev] | v1 memory strategy; custom adjoint deferred |
| reversible adjoint (B16, 512², 32 slices) | 0.80 GB / 100 ms vs checkpoint 1.30 GB / 133 ms vs naive 3.0 GB / 163 ms; grad error 7·10⁻⁶ [B07 §4.1] | v2 optimisation (ODT trigger) |
| materialised volume, 64 × 256 × 512² | 16 GiB + 16 GiB gradient [B07 §4.1] | procedural slices; learnable voxel volumes at batch 1 |
| mode chunking, M = 64 | 3.74 → 0.51 GB at chunk 8, gradients identical [B07 §4.2] | exact chunked checkpointed sums |
| thin-sample Abbe, B16 × 256², n_src = 9 / 25 / 81 | 15 / 39 / 46 ms fwd+bwd; 0.26 / 0.70 / 2.23 GiB [M-C] | quadrature(25) is affordable in v1 |
| unfused voxelisation, 10⁴ spheres | 0.99 s, 17.9 GB [B06 §3.4] | recompute-in-backward rasteriser |
| front-end resolve (proxy), B64 × N50, 40 properties | naive per-property draws 1.48 ms; fused 0.17; CUDA graph 0.30; DT-style Python loop 19.0 [M-D] | the front end's cost; recommendations in Appendix F.7 |
| counter-keyed uniform pool + Normal transform, B64 × 1020 uniforms | splitmix64-class hash: 30 kernels, ≈0.30 ms eager, 0.054 ms graphed; pure-torch Philox-4x32-10: 183 kernels, 1.9 ms eager, 0.34 ms graphed [Rev] | cheap counter hash for keyed noise and textures; Philox as a high-quality option (§6.3) |
| Mie-in-pupil calibration step, B256 × 8 slots, 64², grad mode | 21–22 ms fwd, 58–61 ms fwd+bwd, no graphs [Rev] | calibration targets are stated per route (§8.3) |
| CPU (16 threads) vs RTX 3090 | 32-slice 64 × 256² FFT march 0.41 s (idle) to ≈5 s (loaded) vs 16.5 ms; sparse MFT, 64 images × 50 emitters, 87–379 ms vs 0.5–0.6 ms [Rev] | CPU is a correctness tier (§12.5) |
| per-sample Python resolve in a DeepTrack2-style graph (proxy) | ≈0.3 ms/image → single-process ceiling ≈3k img/s [M-D] | front ends should sample in batches; CPU workers with late binding (§10.3) |

### 8.2 Execution strategies (in order of leverage)

1. **One batched call per kernel kind per batch.** No per-sample or per-object Python. Mie coefficients for all particles × wavelengths run in one call.
2. **Sparse first.** Point emitters render through MFT onto bound-sized ROIs. Coherent sparse objects are evaluated only at pupil-support samples, scatter-added into the k-grid, then inverse-FFTed once per (image, mode).
3. **No graph when nothing requires gradients.** `torch.no_grad()` and in-place fast paths are the normal data-generation case; outputs are caller-owned copies (§6.5).
4. **Pre-rasterise in inference, regenerate under gradients.** Without gradients, slices are rasterised once per batch from bbox-culled patches and reused by every mode and wavelength (material occupancy channels make dispersion cost n_mat FMAs). With gradients, slices are regenerated procedurally inside checkpointed segments.
5. **Exact chunked, checkpointed reductions** over modes, wavelengths and emitters (`use_reentrant=False`; keyed or pre-drawn randomness only, §6.3), with chunk sizes chosen from free memory when the Pipeline is built and recorded (§4.3).
6. **Diagonal fusion.** The pupil-side multiplier (defocus · aperture · √cosθ · aberrations · layer stack · masks) is computed once per call per unique parameter tensor. An on-the-fly H kernel (Triton) avoids materialising `[L, Z, Y, X]` kernels later.
7. **Static shapes.** Smooth FFT sizes; N_max and batch-size buckets; a warm cuFFT plan cache; `PYTORCH_ALLOC_CONF=expandable_segments:True`.
8. **`rfft`** for every real intensity convolution (strata OTF, pixel MTF).
9. **Culling.** Bounding boxes plus blur, PSF and defocus margins cull objects; empty slabs merge into one propagation; empty strata are skipped.
10. **CUDA-graph capture** of static Pipelines (`Pipeline.capture`): inputs are copied into static buffers, keys enter as device tensors, and outputs are copied into fresh allocations; re-drawing only camera noise is `gx.detect.sample` on a stored `expected` (§6.5).
11. **Memory is a stage property.** The Pipeline (P6) picks inference mode, autograd or checkpointing per stage under its `memory_budget`, and **refuses infeasible configurations with an actionable message** instead of running out of memory (e.g. "learnable 96×256² voxel volume × 81 LEDs: render acquisition subsets of ≤ 12 LEDs (`leds.select(idx)`, §6.5)").
12. **Rendering on the GPU in the main process.** No CUDA in DataLoader workers [B07 §7.4]; front-end resolution may run in CPU workers that return stacked data objects (§10.3). Multi-GPU is rank-local rendering plus DDP (Linux; NCCL is unavailable on Windows).

### 8.3 Throughput and memory targets (single RTX 3090; milestone exit criteria)

Every target names its route (the implementations it runs) and is set at **≤ 0.8× the median of ≥ 5 runs of the committed script** (for times: ≥ 1.25× the median), so noise and GPU contention do not fail the gate. Targets without a measured anchor are re-baselined when their script lands.

| Workload | Route | Target | Anchor |
|---|---|---|---|
| SMLM frames 64², 20 emitters, sCMOS noise | `pupil_mft`, `psf="scalar"` pinned, pixel MTF, standard | ≥ 30k frames/s fwd | 67k (kernel) [M-A] |
| Sparse fluorescence 128², 50 emitters | `pupil_mft` scalar, standard | ≥ 20k img/s fwd, ≥ 5k fwd+bwd | 51–52k / 10–18.7k [M-C, M-D] |
| Same, vectorial 6-basis | `pupil.vectorial`, accurate | ≥ 5k frames/s | 10.8k [M-A] |
| Dense fluorescence B16 × Z32 × 256² | `strata_otf`, standard (periodic boundary) | ≥ 5k img/s | 7.3k [M-C]; M1: 3.4–3.7k periodic, 1.4k linear |
| Mie holography 256², 5 spheres, 1 mode | `mie` on pupil support, standard | ≥ 3k img/s fwd; ≥ 1k fwd+bwd after the custom S(θ) backward | 5020 fwd [M-J eng]; 650 fwd+bwd full-grid [M-C] |
| Same, vector (P = 2) | `mie` at P = 2, accurate | ≥ 1.5k img/s fwd (estimate: half the P = 1 rate) | measured in M3 |
| Brightfield tracking 256², 5 Mie | `mie` + `koehler.quadrature(9)`, standard | ≥ 1.5k img/s fwd (0.8 × ≈1.94k extrapolated linearly from the 1-mode and 7-mode anchors; re-baselined when measured) | 5020 (1 mode) and 2296 (7 modes) [M-J eng] |
| Thin phase contrast 256² | `projection` + `koehler.quadrature(25)`, `Fidelity("standard", source=gx.Quadrature(n=25))` | ≥ 300 img/s fwd+bwd | ≈400 [M-C] |
| LC-PolScope 256², 5 compensator states, thin Jones screen | `projection` (Jones) + `koehler.quadrature(9)`, P = 2, standard | ≥ 300 frames/s fwd (estimate from the thin phase-contrast anchor) | measured in M4 |
| Multislice 256² × 64 slices, inference, B64 | residual-demodulated total-field march, pre-rasterised slices, accurate | ≥ 1k img/s | 1337–1351 img/s for the plain total-field march [M-J eng, Rev]; the obliquity term adds one fused multiply-add per slice and is re-measured in M4. The scattered-field form reaches ≈1120 img/s only with precomputed complex transmittances (§8.1) |
| Multislice training, B8, 256² × 64 slices | same march, procedural slices + checkpointing, accurate | ≤ 1 GiB peak, ≥ 34 img/s fwd+bwd | 0.88–0.90 GiB; 41–44 img/s [M-J eng, Rev] |
| Sequence, B16 × T100 × 256², 5 Mie, 1 mode | `mie`, chunked over B·A, standard | ≥ 2k frames/s, peak ≤ 8 GiB | estimate from the 1-mode holography anchor; measured in M3 |
| Standalone SMLM generation loop, B64 × 64², 20 emitters, batched torch sampling + keyed sCMOS noise | cookbook sampling + `PointPSF` scalar, standard | ≥ 20k frames/s end to end (estimate; measured in M2) | kernel 67k frames/s [M-A]; sampling 0.17–0.3 ms per batch [M-D] |
| Keyed camera noise, B64 × 256², Poisson–Gaussian | counter hash + keyed Poisson/Gaussian, graphed | ≤ 2 ms per batch (estimate; measured in M0) | 0.054 ms for 64 × 1020 uniforms + a Normal transform (graphed) [Rev] |
| Bead-stack pupil fit, 20 beads × 41 z × 31², vectorial | `pupil.vectorial` + an L-BFGS loop over `gx.detect.log_prob`, accurate | < 60 s to convergence | uiPSF voxel model: 40 s on an RTX 3080 [B03 §2.3] |
| Calibration render step (fwd+bwd; front-end sampling excluded), B256, 64² (born route) | `Fidelity(spheres="born")` (`born_ff`, closest to [M-D Exp C]'s form-factor projection model) | ≤ 10 ms | [M-D Exp C] measured 6.1 ms on a projection model; the born route is measured in M3 |
| Calibration render step (fwd+bwd; front-end sampling excluded), B256 × 8, 64² (Mie route) | `mie`, standard, grad mode, no graphs | ≤ 75 ms until the custom S(θ) backward lands, then re-baselined | 58–61 ms fwd+bwd [Rev] |
| Diffractive stack, 5 phase layers on 512² fields, B64 | `optics.mask` + `optics.propagate` (BL-ASM) in a detection stage | ≥ 3.5k img/s fwd; ≥ 1.3k img/s fwd+bwd | 4.5k fwd, 1.7k fwd+bwd, 1.2 GiB peak [Rev] |

These are regression-checked before each release: the scripts in `benchmarks/` record time and `max_memory_allocated` on a local 3090, and a > 20 % regression blocks the release. CI runs on GitHub-hosted runners, which have no GPU (§11.15).

### 8.4 Accelerators and languages

- **Correctness never depends on `torch.compile`, Triton or CUDA graphs.** Triton is absent from the Windows wheel on this machine, and Inductor does not generate code for complex operations [B07 §1, §3.1].
- **Order of adoption:**
  1. CUDA graphs (M2 for incoherent Pipelines, M3 for all);
  2. regional `torch.compile` of real-view elementwise step functions where Triton exists (optional extra `triton-windows` on Windows);
  3. Triton kernels registered through `torch.library.triton_op`, each with a pure-torch twin: first the procedural rasteriser (procedural slice generation makes the BPM forward ≈6× slower than precomputed slices [M-J eng]), then the fused phase-screen multiply, on-the-fly H(k) multiply and |u|² mode accumulation;
  4. nvmath-python FFT callbacks, late and CUDA-only.
- **No Cython, no C++/CUDA extension.** Cython sits outside autograd, runs only on the CPU and adds a build matrix. The ~8 µs overhead is Python → ATen dispatch, which only batching, graphs and fusion remove [B07 §3.5].

---

## 9. API walkthrough

The examples use plain torch for randomness, because sampling is the caller's (§10.3). Each example uses the level that suits it: L2 direct calls for small fits, L3 Chains and Pipelines for training loops, and L4 plans where choosing elements by fidelity helps. Lengths are in µm. The examples run as executable doctests against the M0 stub classes, in CI and in the M0 API review with prospective users (§12.1). Each becomes an example notebook when its milestone lands (§12.7).

```python
import math, torch
import torch.nn.functional as F
import gradix as gx
from gradix.units import nm, um, mm, ms

# PR CI also runs these examples on the CPU (§11.15)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(dev)  # tensors created below live on dev
```

### (0) Fit one scalar: the first tutorial (L2, direct calls)

```python
focus = torch.nn.Parameter(torch.tensor(0.0))  # µm; natural units (Appendix F)
bead = gx.Emitters(
    position=torch.tensor([[[6.4, 6.4, 0.0]]]),  # [B=1, N=1, 3]
    photons=torch.tensor([[2000.0]]),
    emission=gx.Spectrum.line(600 * nm),
)
env = gx.env.Homogeneous(n=1.518)
camera = gx.Camera(pixel_size=6.5 * um, shape=(128, 128))
opt = torch.optim.Adam([focus], lr=1e-2)  # ≈ 10 nm per step
for _ in range(200):
    psf = gx.imaging.PointPSF(gx.Objective(NA=1.4, magnification=100, focus=focus), camera)
    # eager: grids sized from these inputs
    mu = camera.expected(psf(gx.lower.emitter_set(bead), env))
    loss = F.mse_loss(mu, observed)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Rebuilding the element each step is cheap: elements are small dataclasses that hold the tensors they are given (§6.1).

### (a) Holography of Mie spheres with a learnable size distribution (L3)

```python
r_loc = torch.nn.Parameter(torch.tensor(math.log(0.30)))  # log µm
r_scale = torch.nn.Parameter(torch.tensor(math.log(0.15)))
env = gx.env.Homogeneous(n=1.33)
cam = gx.Camera(
    pixel_size=6.5 * um,
    shape=(128, 128),
    qe=0.8,
    gain=0.47,
    offset=100.0,
    bit_depth=16,
    # "image" is in ADU
    noise=gx.noise.PoissonGaussian(read=2.0),
)


def chain_for(B, g):  # the caller's sampling, in plain torch, feeding L2 objects
    beads = gx.Spheres(
        position=torch.cat(
            [
                0.1625 * (8.0 + 112.0 * torch.rand(B, 8, 2, generator=g)),  # 8–120 px
                -6.0 + 12.0 * torch.rand(B, 8, 1, generator=g),
            ],
            -1,
        ),
        # reparameterised
        radius=torch.exp(r_loc + torch.exp(r_scale) * torch.randn(B, 8, generator=g)),
        material=1.40 + 0.20 * torch.rand(B, 8, generator=g),
        # Binomial(8, 0.6) law
        presence=(torch.rand(B, 8, generator=g) < 0.6).float(),
    )
    light = gx.light.PlaneWave(
        wavelength=532 * nm,
        polarization="x",
        tilt=0.005 * torch.randn(B, 2, generator=g),  # rad; analytic
        # photons/µm²
        irradiance=(300.0 + 2700.0 * torch.rand(B, generator=g)) / 0.1625**2,
    )
    objective = gx.Objective(
        NA=0.8,
        magnification=40,
        pupil=(gx.pupil.Zernike(coeffs=0.02 * torch.randn(B, 1, generator=g), indices=[12]),),
    )
    return gx.Chain(
        light=light,
        scatterers={"beads": gx.interact.Mie(beads)},
        imaging=gx.imaging.Coherent(objective, cam),
        camera=cam,
        environment=env,
    )


# a generator on the device the draws land on (§10.3)
g = torch.Generator(device=dev).manual_seed(0)
pipe = gx.Pipeline(
    chain_for(64, g),
    outputs={
        "x": "image",
        "heat": gx.labels.Heatmap("beads", sigma_px=1.0),
        "xy": gx.labels.Positions("beads", unit="px", dims="xy"),
    },
    envelope={"beads.radius": (0.05, 1.5), "beads.position.z": (-6.6, 6.6)},
)
# §5.7; the gradient table shows beads.radius → mie → objective → scaled-ST camera
print(pipe.explain())

# ---- calibrate the size distribution on UNLABELED holograms (features and MMD: user code or deeplay) ----
opt = torch.optim.Adam([r_loc, r_scale], lr=3e-2)
for step, x_real in zip(range(600), real_loader):  # real frames converted with gx.detect.to_unit
    fake = pipe(chain_for(64, g), key=64 * step + torch.arange(64))["x"]
    loss = mmd(features(fake), features(x_real))
    opt.zero_grad()
    loss.backward()
    opt.step()

# ---- generate training data with the calibrated generator ----
with torch.no_grad():
    fast = pipe.capture(chain_for(64, g))  # CUDA graph; fresh outputs each call (§6.5)
    for step in range(50_000):
        chain = chain_for(64, g)
        b = fast(chain, key=10**9 + 64 * step + torch.arange(64))
        r = chain.scatterers["beads"].objects.radius  # value label: the caller already has it
        ...  # network step: user code or deeplay
```

The calibration stage was demonstrated in [M-D Exp C] with a simpler forward model (form factor → projection phase → pupil → |E|² → scaled-ST Poisson). A LogNormal learned from unlabeled images recovered the median to 0.7 % and σ to 9 %, at 6.1 ms per step (B = 256, 64²). That demonstration used synthetic "real" images from the same generator, so it could not expose a photometric mismatch. That is why the camera carries QE, gain and offset, and why the nightly learnability test generates its "real" set with a different gain, offset and illumination (§11.11).

The count here is a fixed Binomial law. Learning it needs relaxed or score-function presence in the sampling code (the cookbook, §10.3). gradix renders every slot whenever presence requires gradients (§6.1).

The same batch at L4 is `gx.plan(gx.Sample({"beads": beads}, environment=env), gx.presets.InlineHolography(light=light, objective=objective, camera=cam), fidelity="standard", envelope=pipe.envelope)`. Its `plan.chain(...)` equals the Chain above, because the rule table picks `gx.interact.Mie` for spheres at `standard`.

### (b) SMLM: fit a Zernike + pixel pupil to a bead stack, then generate training data with it (L3)

System identification is a plain torch fit over element fields:

```python
stack = ...  # [n_beads, 41, 31, 31] ADU, read by the user
env = gx.env.LayeredMedium(
    immersion=gx.materials.Oil(),
    coverslip=(gx.materials.Glass(), 170 * um),
    # NA > n_s: layered collection rule
    sample=gx.materials.Water(),
)
zern = torch.nn.Parameter(torch.zeros(12))  # OPD µm; ANSI 3, 5–14, 24 (no defocus)
opd = torch.nn.Parameter(torch.zeros(64, 64))
log_amp = torch.nn.Parameter(torch.zeros(64, 64))
xyz = torch.nn.Parameter(xyz0)  # [n_beads, 1, 3] µm
log_photons = torch.nn.Parameter(photons0.log())
log_bg = torch.nn.Parameter(bg0.log())
# offset/gain/variance maps
camera = gx.Camera.scmos(maps="cam/maps.npz", roi=rois, pixel_size=11 * um)


def pupil():
    return (
        gx.pupil.Zernike(coeffs=zern, indices=[3, *range(5, 15), 24]),
        gx.pupil.PixelPupil(opd=opd, amplitude=log_amp.exp()),
    )


def fit_chain():
    # 100 nm bead: labeled volume
    bead = gx.Spheres(
        position=xyz,
        radius=torch.full((n_beads, 1), 0.05),
        labeling=gx.Labeling(
            emission=gx.Spectrum.band(680 * nm, 30 * nm), photons=log_photons.exp()
        ),
    )
    objective = gx.Objective(NA=1.45, magnification=100, pupil=pupil())
    return gx.Chain(
        emitters={"bead": bead},
        imaging=gx.imaging.PointPSF(objective, camera, psf="auto"),
        camera=camera,
        background=log_bg.exp(),
        environment=env,
        # A = 41 focal planes
        acquisition=gx.acq.FocusStack(focus=focus_steps),
    )


pipe = gx.Pipeline(
    fit_chain(), outputs={"mu": "expected"}, envelope={"bead.position.z": (-1.5, -0.05)}
)
# then add opd, log_amp
opt = torch.optim.LBFGS([zern, xyz, log_photons, log_bg], line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    nll = -gx.detect.log_prob(stack, pipe(fit_chain())["mu"], camera).sum() + 1e-3 * laplacian(opd)
    nll.backward()
    return nll


for _ in range(150):
    opt.step(closure)
crlb = gx.crlb(pipe, fit_chain(), wrt=("bead.position",))  # camera Fisher information, jacfwd
```

`psf="auto"` renders vectorial at this NA/n (0.96) once M4b lands, and scalar with a warning before (§5.3). The bead is a labeled 100 nm sphere. The Chain lowers its labeling to volume-quadrature points on the sparse path, which equals the intensity PSF convolved with the (signed) ball form factor within quadrature error, instead of folding |F_ball| into an OTF [M-J physics].

**Training data from the fitted pupil.** Build a Chain with `pupil()` evaluated under `torch.no_grad()` (a frozen copy) and `gx.Emitters` for molecules. The caller samples positions, per-frame on/off states and photons, typically as 3-frame DECODE-style sequences (§10.3). The Pipeline renders `[B, 3, 1, 64, 64]`, with an `EmitterTable` label for the middle frame.

For end-to-end PSF engineering (DeepSTORM3D-style), keep `zern` or `opd` as Parameters in that Chain and train them jointly with the localisation network. The sparse path costs ≈3.4 ms fwd+bwd per 64 images at 128² [M-D].

### (c) A cell in phase contrast (and QPI) through multislice, with a procedural, voxel or neural sample (L4)

This is where the planner helps: at `accurate` it routes the cell body, nucleoid and texture into one multislice march and sizes the slices.

```python
def cells(B, g, corr, std):  # the caller's sampling; corr and std may be Parameters
    n = 16
    L = 3.0 + 5.0 * torch.rand(B, n, generator=g)
    r = 0.45 + 0.25 * torch.rand(B, n, generator=g)
    pos = torch.cat(
        [25.6 * torch.rand(B, n, 2, generator=g), torch.full((B, n, 1), gx.coords.depth(1.0))], -1
    )
    rot = gx.geom.quat_z(2 * math.pi * torch.rand(B, n, generator=g))  # in-plane rotations
    body = gx.Capsules(
        position=pos,
        rotation=rot,
        length=L,
        radius=r,
        material=1.378 + 0.004 * torch.randn(B, n, generator=g),
        presence=(torch.rand(B, n, generator=g) < 0.6).float(),
    )
    parent = gx.Parent("cells", torch.arange(n).expand(B, n))  # child i belongs to cell i
    nucleoid = gx.Ellipsoids(
        position=pos,
        rotation=rot,
        semi_axes=torch.stack([0.3 * L, 0.6 * r, 0.6 * r], -1),
        material=torch.full((B, n), 1.386),
        parent=parent,
    )
    texture = gx.Texture(
        family="matern32",
        corr_length=corr.expand(B, n),
        std=std.expand(B, n),
        key=torch.randint(0, 2**62, (B, n), generator=g),
        parent=parent,
    )
    return gx.Sample(
        {"cells": body, "nucleoids": nucleoid, "texture": texture},
        environment=gx.env.WaterOnCoverslip(),
    )


ring_t = torch.nn.Parameter(torch.tensor(0.25))
corr = torch.nn.Parameter(torch.tensor(0.3))
std = torch.nn.Parameter(torch.tensor(0.003))
cam = gx.Camera(
    pixel_size=6.5 * um,
    shape=(256, 256),
    gain=0.5,
    offset=100.0,
    noise=gx.noise.PoissonGaussian(read=1.6),
)
pc = gx.presets.PhaseContrast(
    light=gx.light.Koehler(
        spectrum=gx.Spectrum.band(550 * nm, 40 * nm),
        condenser=gx.Annulus(NA_inner=0.30, NA_outer=0.38),
        irradiance=2000.0 / 0.1625**2,  # photons/µm²: 2000 per pixel
    ),
    ring=gx.pupil.PhaseRing(phase=-math.pi / 2, transmission=ring_t),  # conjugate to the annulus
    objective=gx.Objective(NA=0.95, magnification=40),
    camera=cam,
)

sample = cells(8, g, corr, std)
plan = gx.plan(
    sample,
    pc,
    fidelity="accurate",  # multislice · quadrature(25) · 3 λ bins
    outputs={
        "x": "image",
        "inst": gx.labels.InstanceMask("cells", parts="all"),
        "opl": gx.labels.OPL("cells"),
    },
    envelope={"texture.corr_length": (0.05, 2.0), "texture.std": (0.0, 0.02)},
)
# halo and shade-off emerge from annulus × ring overlap; no tuned offset
b = plan(sample, pc, key=torch.arange(8))
# the Chain the planner built: Multislice(cells + nucleoids + texture), …
print(plan.chain(sample, pc))

# QPI of the SAME inputs through another microscope; the texture is identical because it is keyed
qpi = gx.presets.QPI(
    light=gx.light.PlaneWave(550 * nm),
    objective=gx.Objective(NA=0.95, magnification=40),
    # no phase ring in the QPI objective
    camera=pc.camera,
)
phase = gx.render(sample, qpi, fidelity="accurate", outputs={"phase": gx.labels.Phase()})["phase"]

# Voxel alternative: a measured RI tomogram; or any torch module of local coordinates
tomo = gx.Voxels(
    values=torch.load("cell_ri.pt", map_location=dev)[None, None],
    spacing=(0.2, 0.1, 0.1),
    quantity="n",
    position=torch.tensor([[[20.0, 20.0, gx.coords.depth(3.0)]]]),
)
nerf = gx.NeuralField(
    module=SirenMLP(3, 1),
    quantity="dn",
    bbox=((-8, -8, -6), (8, 8, 0)),
    position=torch.tensor([[[12.8, 12.8, gx.coords.depth(3.0)]]]),
)
```

The multislice march consumes a *procedural* `RIVolume`.
- In inference, slices are rasterised once per batch (at s = 2 at `accurate`, §4.6) and reused by all 25 modes × 3 wavelengths.
- Under gradients, they are regenerated inside checkpointed segments. The SIREN's weights are explicit inputs of the lowering (§4.6), so gradients reach `corr`, `std`, the capsule sizes, the ring transmission and the network without materialising the volume gradient.

Matching texture statistics and ring transmission to real phase-contrast images, as SyMBac does by hand, is then an optimisation over `corr`, `std` and `ring_t`, with features and losses from the user or deeplay.

### (d) Switching fidelity and inspecting the choices (L4 → L3)

```python
for fid in ("draft", "standard", "accurate"):
    # plans are cheap; inputs are shared
    print(gx.plan(sample, pc, fidelity=fid).explain(level="summary"))
```

Illustrative summary (the structure is the design; timings are estimates scaled from §8.1):

```text
level     interact.cells    source            λ  P  grids                                        est. fwd (B=8)  notes
draft     projection        on_support(2)     1  1  det 288² @162.5 nm (camera; λ/4NA = 145 nm)  ≈3 ms   ⚠ undersampled; ⚠ thickness 1.4 µm > DOF 0.81 µm
standard  projection        quadrature(9)     1  1  det 576² @81 nm (s=2); band optical           ≈15 ms  ⚠ thickness > DOF → consider volumes="multislice"
accurate  multislice(ckpt)  quadrature(25)    3  1  det 576² @81 nm; interaction 288² @162.5 nm  ≈0.3 s  ⚠ 600 slice-steps/image (8 slices × 25 × 3)
                                                     (band n_m/λ ✓; raster s=2); 8 slices @0.20 µm (λ_m/2)
```

```python
std_plan = gx.plan(sample, pc, fidelity="standard")
print(gx.Pipeline.diff(std_plan.pipeline, plan.pipeline))  # stage-by-stage differences
print(plan.why("interact.cells"))  # candidates, validity, knob, reasons
# e.g. "projection invalid (thickness > DOF) for 71 % of cells"
print(plan.validate(sample, pc).summary())
print(
    gx.compare(
        sample,
        pc,
        "standard",
        "accurate",
        key=torch.arange(8),
        # the SAME inputs rendered twice
        metrics=("rel_l2", "ssim", "phase_rmse", "halo_width"),
    )
)

# Take the planner's Chain, swap one element by hand, and run it at L3
chain = plan.chain(sample, pc)
# plain BPM, opt-in; paths as in §6.1
chain = gx.tree.replace(chain, {"scatterers.cells.obliquity": False})
pipe = gx.Pipeline(chain, envelope=plan.pipeline.envelope)
taps = pipe.run(chain, until="objective", taps=["interact.cells"])  # step through the stages
```

A caller that samples scenes can aggregate `validate()` over batches to report violation *fractions* ("born_ff invalid for 12 % of sampled vesicles"), which is the right semantics for random scenes [D §5.6].

### (e) Reusing the forward model in inverse problems

```python
# (e1) Lorenz–Mie characterisation: one problem per crop, all in one batched graph (L3)
n_crops = crops.shape[0]  # [n_crops, 1, 96, 96] in the camera unit
r = torch.nn.Parameter(torch.full((n_crops, 1), 0.4))
n = torch.nn.Parameter(torch.full((n_crops, 1), 1.50))
xyz = torch.nn.Parameter(init_xyz)  # [n_crops, 1, 3]
# photons/µm²: 1500 per pixel
log_irr = torch.nn.Parameter(torch.full((n_crops,), math.log(1500.0 / 0.1625**2)))
offset = torch.nn.Parameter(torch.full((n_crops,), 100.0))


def crop_chain():  # photometry is fitted, not assumed
    cam = cam96.replace(offset=offset)  # gain from calibration (degenerate with irradiance)
    return gx.Chain(
        light=gx.light.PlaneWave(532 * nm, irradiance=log_irr.exp()),
        scatterers={"beads": gx.interact.Mie(gx.Spheres(position=xyz, radius=r, material=n))},
        imaging=gx.imaging.Coherent(gx.Objective(NA=0.8, magnification=40), cam),
        camera=cam,
        environment=gx.env.Homogeneous(1.33),
    )


pipe = gx.Pipeline(
    crop_chain(),
    outputs={"mu": "expected"},
    envelope={"beads.radius": (0.1, 1.5), "beads.position.z": (-8.0, 8.0)},
)
# multistart over z and L-BFGS on -gx.detect.log_prob(crops, pipe(c)["mu"], c.camera) with c = crop_chain(): the user's loop
crlb = gx.crlb(pipe, crop_chain(), wrt=("beads.radius", "beads.material", "beads.position"))

# (e2) Intensity diffraction tomography: learnable voxel Δn through a multislice element (L3)
dn = torch.nn.Parameter(torch.zeros(1, 1, 48, 256, 256))
leds = gx.light.LEDArray.grid(
    (9, 9), pitch=4 * mm, distance=60 * mm, spectrum=gx.Spectrum.line(520 * nm)
)
cam = gx.Camera(pixel_size=6.5 * um, shape=(256, 256), noise=None, unit="photons")


def idt_chain(idx):  # A = len(idx) frames
    volume = gx.Voxels(values=dn, spacing=(0.2, 0.1, 0.1), quantity="dn")
    return gx.Chain(
        light=leds.select(idx),
        scatterers={"ri": gx.interact.Multislice(volume)},
        imaging=gx.imaging.Coherent(gx.Objective(NA=0.4, magnification=20), cam),
        camera=cam,
        environment=gx.env.Homogeneous(1.333),
        acquisition=gx.acq.LEDSequence("each"),
    )


pipe = gx.Pipeline(
    idt_chain(range(9)), outputs={"mu": "expected"}, envelope={"ri.values": (-0.02, 0.12)}
)
opt = torch.optim.Adam([dn], lr=5e-3)
for it in range(200):
    for idx in torch.randperm(81).split(9):  # angle mini-batching bounds memory
        mu = pipe(idt_chain(idx))["mu"]  # same structure (9 LEDs) → same Pipeline
        loss = gaussian_nll(mu, measured[:, idx], var=mu.detach() + 4.0) + 1e-4 * tv3d(dn)
        opt.zero_grad()
        loss.backward()
        opt.step()

# (e3) v2: linear stages (strata OTF, born) export as deepinv LinearPhysics with adjoints
```

### (f) Sequences: 100-frame brightfield tracking with Brownian motion and focus drift (L4)

```python
B, T, N = 16, 100, 8
D = 0.05 + 0.95 * torch.rand(B, 1, N)  # µm²/s per object (learnable if wanted)
# dt = 20 ms; reparameterised increments
steps = torch.randn(B, T, N, 3) * torch.sqrt(2 * D * 0.020)[..., None]
pos = start[:, None] + steps.cumsum(1)  # [B, T, N, 3] µm
beads = gx.Spheres(
    position=pos,
    radius=0.25 + 0.25 * torch.rand(B, 1, N),
    material=torch.full((B, 1, N), 1.59),
    # [B, T, N]: appearance and disappearance
    presence=alive,
)
scope = gx.presets.Brightfield(
    light=gx.light.Koehler(spectrum=gx.Spectrum.band(550 * nm, 40 * nm), condenser=gx.Disk(NA=0.3)),
    objective=gx.Objective(NA=0.75, magnification=40, focus=drift),  # [B, T] µm
    camera=gx.Camera(
        pixel_size=6.5 * um, shape=(256, 256), noise=gx.noise.PoissonGaussian(read=2.0)
    ),
    acquisition=gx.acq.Frames(T, interval=20 * ms, exposure=5 * ms),
)
out = gx.render(
    gx.Sample({"beads": beads}, environment=gx.env.WaterOnCoverslip()),
    scope,
    # chunked over B·A (§6.7)
    fidelity="standard",
    outputs={"x": "image"},
    key=torch.arange(B),
)
frames = out["x"]  # [16, 100, 1, 256, 256]
# labels: the caller's values, in pixels
traj_px = gx.coords.to_pixels(pos, scope.camera, scope.objective)
```

v1 renders each frame at mid-exposure; motion blur within the 5 ms exposure arrives with cap-24 (v1.x).

### (g) Refining rough detections over a dataset (L3)

A detector network found beads in 1000 recorded holograms, with a count that varies per image. The focus was changed between acquisitions and recorded, up to an unknown offset. The fit refines every position, together with the offset and a residual aberration shared by all images. Example (e1) fits one batch of crops; this is the same idea at dataset scale (§6.9).

```python
frames = ...  # [1000, 1, 128, 128] recorded holograms in the camera unit
detections = ...  # 1000 tensors [N_i, 3] µm: rough positions; N_i from 0 to 30
focus_set = ...  # [1000] µm: the focus recorded with each image
pos0, presence = gx.pad(detections)  # [1000, 32, 3] and [1000, 32], present slots first
counts = presence.sum(1).long().cpu()

table = torch.nn.Embedding(1000, 32 * 3, sparse=True)  # per-image positions (µm), refined in place
with torch.no_grad():
    table.weight.copy_(pos0.reshape(1000, -1))
dz = torch.nn.Parameter(torch.tensor(0.0))  # shared: actual minus recorded focus (µm)
zern = torch.nn.Parameter(torch.zeros(3))  # shared: residual astigmatism and spherical (OPD µm)

cam = gx.Camera(
    pixel_size=6.5 * um,
    shape=(128, 128),
    gain=0.47,
    offset=100.0,
    noise=gx.noise.PoissonGaussian(read=2.0),
)
objective = gx.Objective(
    NA=0.8, magnification=40, pupil=(gx.pupil.Zernike(coeffs=zern, indices=[3, 5, 12]),)
)
# certified beads
beads = gx.Spheres(position=pos0[:64], presence=presence[:64], radius=0.50, material=1.59)
template = gx.Chain(
    light=gx.light.PlaneWave(532 * nm, irradiance=2e4),
    scatterers={"beads": gx.interact.Mie(beads)},
    imaging=gx.imaging.Coherent(objective, cam),
    camera=cam,
    environment=gx.env.Homogeneous(1.33),
)
data = {"beads.position": pos0, "beads.presence": presence, "objective.focus": focus_set}
pipe = gx.Pipeline(
    template,
    outputs={"mu": "expected"},
    inputs=("beads.position", "beads.presence", "objective.focus"),
    envelope=gx.envelope_of(data)
    | {"beads.position.z": (-8.0, 8.0), "objective.focus": (-3.0, 3.0)},
)

# updates only the rows a batch used (Appendix F.5)
opt_rows = torch.optim.SparseAdam(table.parameters(), lr=5e-3)
opt_shared = torch.optim.Adam([dz, zern], lr=1e-3)
batches = torch.argsort(counts).split(64)  # images of similar counts batch together
for epoch in range(30):
    for b in torch.randperm(len(batches)).tolist():
        n = max(int(counts[batches[b]].max()), 1)  # this batch's largest count
        idx = batches[b].to(dev)
        pos = table(idx).view(-1, 32, 3)[:, :n]
        mu = pipe(
            {
                "beads.position": pos,
                "beads.presence": presence[idx, :n],
                "objective.focus": focus_set[idx] + dz,
            }
        )["mu"]
        nll = -gx.detect.log_prob(frames[idx], mu, cam).sum()
        opt_rows.zero_grad()
        opt_shared.zero_grad()
        nll.backward()
        opt_rows.step()
        opt_shared.step()
refined = table.weight.detach().view(1000, 32, 3)  # read together with `presence`
```

Three details carry the fit. The envelope comes from all 1000 images, with explicit entries for the values the fit moves (z and the corrected focus), so no batch re-plans. Each batch is sliced to its own largest count, so the per-object work follows the beads present. SparseAdam moves only the rows a batch used; dense Adam would keep moving the other 936 rows through its momentum (§6.9).

If nothing were shared (no `dz`, no `zern`), the 1000 images would be independent problems. Each batch is then fitted to convergence and never revisited, for example with per-image Levenberg–Marquardt on forward-mode Jacobians (`torch.func.jacfwd`, which every element supports because `gx.crlb` needs it). A single L-BFGS over many independent problems would couple them through one line search and one curvature history. Once refined, `gx.crlb(pipe, pipe.bind(values), wrt=("beads.position",))` gives each position's precision.

### (h) Birefringent fibres: crossed polarisers, LC-PolScope and a fitted retardance map (L4)

```python
cam = gx.Camera(pixel_size=6.5 * um, shape=(256, 256), noise=gx.noise.PoissonGaussian(read=1.6))
# the caller's sampling
fibres = gx.Cylinders(
    position=fibre_pos,
    rotation=fibre_rot,
    length=fibre_len,
    radius=0.25,
    # along each fibre
    material=gx.Material.uniaxial(n_o=1.530, n_e=1.536, axis=(0.0, 0.0, 1.0)),
)
env = gx.env.WaterOnCoverslip()
light = gx.light.Koehler(
    spectrum=gx.Spectrum.line(546 * nm), condenser=gx.Disk(NA=0.3), irradiance=5e3
)
objective = gx.Objective(NA=0.75, magnification=40)

crossed = gx.presets.PolarizedLight(
    light=light, objective=objective, camera=cam, polarizer=0.0, analyzer=math.pi / 2
)
# 5 compensator states on A
polscope = gx.presets.PolScope(light=light, objective=objective, camera=cam, swing=0.03)
sample = gx.Sample({"fibres": fibres}, environment=env)
# [B, 1, H, W]
x = gx.render(sample, crossed, fidelity="standard", outputs={"x": "image"}, key=torch.arange(B))[
    "x"
]
# [B, 5, 1, H, W]
frames = gx.render(
    sample, polscope, fidelity="standard", outputs={"x": "image"}, key=torch.arange(B)
)["x"]

# reconstruction as a fit: a learnable thin Jones screen reproduces measured PolScope frames
screen = gx.JonesScreen(
    retardance_vector=torch.nn.Parameter(torch.zeros(1, 2, 512, 512)), spacing=0.08125
)
fit_sample = gx.Sample({"screen": screen}, environment=env)
plan = gx.plan(fit_sample, polscope, fidelity="standard", outputs={"mu": "expected"})
opt = torch.optim.Adam([screen.retardance_vector], lr=1e-2)
for _ in range(300):
    # measured: [1, 5, 1, 256, 256]
    nll = -gx.detect.log_prob(measured, plan(fit_sample, polscope)["mu"], cam).sum()
    opt.zero_grad()
    nll.backward()
    opt.step()
```

Each fibre's optic axis follows its rotation, so the crossed-polariser image brightens as sin²2θ with the fibre's in-plane angle and dims where the fibre tilts toward z. The fit recovers retardance and slow-axis maps (δ and θ from the retardance vector) with the camera likelihood, as an alternative to the closed-form five-frame algorithm, and it extends to samples the closed form does not model (defocus, partial coherence, aberrations).

### (i) An optical neural network: patterned illumination and a diffractive front end, trained with a digital classifier (L3)

A label-free microscope built for one task: a phase SLM patterns the illumination at a field plane, and a three-layer diffractive network between the image plane and the camera processes the light before detection. Both are trained jointly with a small digital head, and misalignments of the fabricated layers are sampled per image for robustness.

```python
slm = gx.devices.PhaseSLM(pitch=12.5, shape=(64, 64), phase=torch.nn.Parameter(torch.zeros(64, 64)))
layers = [
    gx.devices.HeightMap(
        heights=torch.nn.Parameter(torch.zeros(400, 400)),
        material=gx.materials.IP_Dip(),
        # 3D-printed layers
        max_height=1.2,
    )
    for _ in range(3)
]
cam = gx.Camera(pixel_size=4.0, shape=(200, 200), noise=gx.noise.PoissonGaussian(read=2.0))


def onn_chain(cells_batch, B):  # cells_batch: the caller's sampled cells (projection-ready)
    return gx.Chain(
        light=gx.light.PlaneWave(532 * nm, irradiance=1e4),
        illumination_optics={
            "slm": gx.optics.Mask(device=slm, plane="field", magnification=1 / 20),
            # NA_c 0.3
            "condenser": gx.optics.Aperture("disk", size=0.3, plane="pupil"),
        },
        scatterers={"cells": gx.interact.Projection(cells_batch)},
        imaging=gx.imaging.Coherent(gx.Objective(NA=0.4, magnification=20), cam),
        detection_optics={
            "stop": gx.optics.Aperture("square", size=800.0),  # µm, camera side
            "d2nn": gx.optics.DiffractiveStack(
                layers,
                pitch=2.0,
                gaps=2000.0,
                # µm jitter
                offsets=2.0 * torch.randn(B, 3, 2),
            ),
        },
        camera=cam,
        environment=gx.env.Homogeneous(1.33),
    )


head = torch.nn.Linear(10, n_classes)  # the digital part
pipe = gx.Pipeline(onn_chain(*example_batch), outputs={"scores": gx.out.Regions(boxes=class_boxes)})
opt = torch.optim.Adam([slm.phase, *(l.heights for l in layers), *head.parameters()], lr=1e-2)
for step in range(5000):
    cells_batch, labels = sample_cells(64, g)  # the caller's sampling
    # [64, 10] photons
    scores = pipe(onn_chain(cells_batch, 64), key=64 * step + torch.arange(64))["scores"]
    loss = F.cross_entropy(head(scores / scores.sum(1, keepdim=True)), labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

# physics-aware fine-tuning of the built device: measured forward, simulated backward (§5.11)
y_sim = pipe(onn_chain(cells_batch, 64), key=torch.arange(64))["scores"]
scores = y_hw.detach() + y_sim - y_sim.detach()  # y_hw: measured region powers
```

The same Chain with the diffractive stack frozen, and only the head trained, is the fixed-ONN regime. Moving the SLM into the detection pupil (`Objective(pupil=[gx.optics.Mask(device=slm, plane="pupil")])`) turns the example into PSF engineering, and the fluorescence paths accept it unchanged.

---

## 10. Front ends: bring your own sampling; DeepTrack2 later

### 10.1 Position

gradix is a **standalone engine**. It is useful to anyone who can produce tensors, and it never decides parameter values (ADR-32, ADR-37).
- **Standalone users** write their own sampling. In practice this is a few lines of batched torch per population; §10.3 shows the recipes, and Appendix F holds the measurements behind them.
- **DeepTrack2** is expected to become the richest front end. Its integration will likely coincide with a revision of how DeepTrack2 constructs optics pipelines, so it is not designed here in detail and is not a v1 deliverable (§10.4).
- The repository lives on the lead developer's GitHub for now. Neither project depends on the other.

### 10.2 The contract with any front end

| gradix guarantees | The front end provides |
|---|---|
| Schema-described data objects and elements (§6.1), exported as data (`gx.schema_of(cls)`: fields, quantities, roles, event shapes, shape-affecting flags, defaults, docs), with a schema version per class | Values for the fields it wants to set, in µm (or SI through `gx.from_si`, or pixels through `gx.coords`) |
| `gx.stack`, which turns B per-image pytrees (data objects, elements, Chains, Samples, Microscopes) into one padded batch with presence; `gx.pad` for ragged per-image lists | Per-image objects, or batched tensors directly |
| Pure element calls (L2); `gx.Pipeline` and `gx.plan` objects that are deterministic, inspectable and JSON-serialisable, with calls that are pure functions of structure-compatible inputs, or of values bound by field path, and keys (L3/L4) | Keys: an int per batch, or an int64 tensor per image (§6.3) |
| Exact gradients with respect to every tensor input, and a gradient-quality table | Reparameterised torch sampling wherever distribution parameters should be learnable (§10.3) |
| Envelopes, `gx.envelope_of`, and `validate()` with per-image and per-object violation masks | Envelope entries for learnable shape-affecting fields; any adaptation of sampling to validity |
| Physical cameras with `sample`, `log_prob` and Fisher information; geometry-derived labels and pixel coordinates | Value labels, augmentation and label co-transforms |
| SemVer on data objects, elements, schemas and the API from 1.0, with schema changes announced with migration notes | — |

### 10.3 Bring your own sampling: a cookbook

These recipes ship as tested documentation (§11.10), not as an API. Every recipe is plain torch.

```python
import math, torch

g = torch.Generator(device="cuda").manual_seed(seed)  # one explicit generator per stream

# per-image and per-object continuous draws: batched, on the GPU, reparameterised
# gradients reach mu, sigma
radius = torch.exp(mu + sigma * torch.randn(B, N, generator=g, device="cuda"))
z = -6.0 + 12.0 * torch.rand(B, N, 1, generator=g, device="cuda")

# truncated draws on a finite window [a, b]: fp64 inverse CDF, never clamping (Appendix F.2); m and s are tensors
# accurate in the lower tail; ndtr is not
Phi = lambda t: 0.5 * torch.special.erfc(-t / math.sqrt(2))
za, zb = ((a - m) / s).double(), ((b - m) / s).double()
right = za > 0  # mirror windows right of the mean
lo, hi = Phi(torch.where(right, -zb, za)), Phi(torch.where(right, -za, zb))
u = lo + (hi - lo) * torch.rand(B, N, generator=g, device="cuda", dtype=torch.float64)
x = m + s * (torch.where(right, -1.0, 1.0) * torch.special.ndtri(u)).float()

# random counts with a fixed law: N slots with presence (the render culls empty slots)
presence = (torch.rand(B, N, generator=g, device="cuda") < p).float()  # Binomial(N, p)
# learnable counts: score function with a leave-one-out baseline, or ST-Concrete in its calibrated band (Appendix F.1)

# sequences: reparameterised increments (learnable diffusion coefficients)
pos = start[:, None] + (
    torch.randn(B, T, N, 3, generator=g, device="cuda") * (2 * D * dt).sqrt()[..., None]
).cumsum(1)

# reproducible datasets: per-image keys for camera noise, derived from the item index
key = base + torch.arange(B) + B * step  # image i's noise never depends on the batch
```

**Throughput.**
- Batched torch sampling of 40 properties for 64 images × 50 objects takes 0.17–0.3 ms [M-D]. A per-sample Python loop over the same properties takes 19 ms, and per-sample resolution in a DeepTrack2-style graph costs ≈0.3 ms per image (a proxy). So sample in batches.
- For generation without gradients to the sampled values, sampling and `gx.stack` can run in CPU DataLoader workers, which never touch CUDA, while the main process renders.
- Learnable optics tensors (a pupil, an NA) are then **bound in the main process**. Workers return data objects, and the main process builds the elements around its `nn.Parameter`s, so their gradients survive.

### 10.4 DeepTrack2 integration (future)

**Expected shape**, to be designed together with DeepTrack2's revised optics pipelines:
- DeepTrack2 Features wrap gradix elements, or DeepTrack2 builds `gx.Chain`s directly. DeepTrack2's `>>` composition maps naturally onto a Chain.
- DT properties supply the values of element and data-object fields. The exported schemas (quantities, roles) let DeepTrack2 generate those wrappers instead of hand-writing them.
- Rendering uses two-phase batched evaluation: resolve B replicates, `gx.stack`, render once, then apply post-optics Features as batched tensor operations.

**Deliberately not promised:**
- No parity preset reproducing DeepTrack2 2.0.2 numerics. Revision 2's `legacy_dt` preset is dropped. A documentation page, "Differences from DeepTrack2 2.0.2 optics", lists every deliberate deviation with its physical reason: free-space inter-slice propagation, soft geometry, stable Mie recurrences, physical darkfield and iSCAT, partial coherence, contrast signs.
- Behaviour close to the old one remains reachable by opting in: plain BPM (`gx.interact.Multislice(obliquity=False)`), hard masks (`gx.Occupancy(hard=True)`), and additive Δn compositing.

The lessons already learned for an integration are in Appendix F.8:
- the coordinate and unit mapping;
- never invalidate DeepTrack2's cache inside an evaluation;
- per-replicate augmentation parameters;
- worker resolution with late binding;
- the pre-render envelope check.

### 10.5 Optional sampling syntax

**Decision (ADR-35): gradix ships no sampling layer and no sampling syntax.**
- Standalone users randomise with plain torch, using the cookbook (§10.3).
- DeepTrack2 users will get DeepTrack2's syntax through its integration.
- If a DeepTrack2-free syntax is ever wanted, the right move is to extract DeepTrack2's property core into a small shared package that both can use. Re-implementing it inside gradix is the wrong move: two sampling DSLs with drifting semantics was revision 1's top-rated risk.

---

## 11. Verification and validation

Testing is part of the architecture. The registry doubles as the test table: every implementation registers its analytic references, gradcheck shapes, approximation edges with tolerance tables, and a benchmark shape [B07 §8]. §11.1–§11.13 are *verification* (we solve our equations correctly); §11.14 is *validation* (the equations describe real microscopes well enough for training and calibration).

### 11.1 Conventions and invariants (fp64, CPU, property-based)

- Fourier shift against `roll`; Propagate(a)∘Propagate(b) = Propagate(a+b);
- Parseval and unitarity; energy conservation through phase-only screens (an energy-type loss has an exactly zero gradient there);
- adjointness ⟨Ax, y⟩ = ⟨x, Aᴴy⟩ for linear stages; reciprocity for dipole and Mie;
- analytic `PlaneWaves` equal to their sampled equivalent on-grid to 10⁻¹²; Zernike orthonormality;
- explicit sign tests: defocus direction, +z, travel assignment, JAX conj flip, `gx.coords.from_focus`;
- the plan-time assertion that no operator applies e^{ik_zΔz} with Δz < 0 to evanescent components (§4.1);
- the M0 carrier-shape conformance test: every carrier broadcasts over B and A (and L where it has one).

### 11.2 Analytic references

- Airy 2J₁(v)/v and the in-focus OTF; Gaussian-beam propagation;
- **power test ∫PSF = (1 − cosθ_max)/2** in an index-matched medium (pins the apodisation rule); the **analytic layered scalar collection efficiency** (0.709 at NA 1.45, 680 nm, water/glass, h = 0; 0.575 at h = 0.1 µm); pupil-grid convergence of the collected power to 10⁻³ from 64² to 512²; a vectorial comparison with psf-generator; finite gradients with respect to n_s across NA = n_s (§4.1);
- pixel integration by the MTF against fine box integration (≤ 1 %);
- blurred-ball closed form against FFT (≤ 10⁻⁷);
- Mie Q_ext and S₁/S₂ against miepython (≤ 10⁻⁸, x ≤ 100, absorbing m included); **Mie pupil spectrum against direct near-field summation** (pins A = −S/(2πk_m k_z)); **optical theorem** through the analytic background (pins the dual-part Field end to end); the P = 1 reduction against holopy `MieLens`;
- dipole (a₁ + b₁) against Mie for x < 0.1;
- TIRF depth d; iSCAT contrast period λ/(2n) and reference amplitude = Fresnel r;
- phase-contrast halo present under Köhler and absent when coherent, **with the correct sign for on-support single modes**;
- darkfield annulus excludes the zero order; darkfield under vignetting responds to the envelope (§4.5);
- off-axis holography: sidebands separate at |u_ref| ≥ 3NA with the image-side reference (§4.5);
- off-grid tilt seam test (analytic background against a sampled tilt: 8–10 % rms error expected for the latter [M-A]); tilt equivariance of the residual-demodulated march;
- keyed `Texture` correlation > 0.99 across two fidelities and two microscopes (same key);
- free-space stages: Fresnel diffraction of a square aperture against Fresnel integrals; Gaussian-beam propagation through `Propagate` and `Lens`; the 4f identity (two transforms return the inverted input); BL-ASM energy conservation and composition;
- Jones calculus: Malus's law; a uniform retarder between crossed polarisers, I = I₀sin²2θ·sin²(δ/2); circular-analyser responses; the P = 2 x component equals the P = 1 S∥ field for on-axis incidence; the crossed-analyser image of a sphere follows (S₂ − S₁)sinφcosφ.

### 11.3 External oracles (test-only extras; never imported by package code)

- MIT/BSD: miepython, scipy, psf-generator (vectorial, Gibson–Lanni), waveorder (WOTF), microsim (camera statistics), wavesim (MBS, v2);
- in-house fp64 references in `testing/references/`: vector Mie imaging (miepython amplitudes with explicit s/p rotations) and Jones products along rays for thin Jones samples;
- GPL, isolated in a `tests/oracles` environment: PyMieDiff, holopy `MieLens`, psfmodels;
- DT 2.0.2 outputs for the "Differences from DeepTrack2 2.0.2" page (generated once and stored; not a test gate).

### 11.4 Ladders

Each approximation edge of §5.9 gets a regime sweep that produces a versioned error table (the "fidelity atlas"). The tables set validity thresholds and the tolerances printed by `explain()`. Edge metrics are on the background-removed contrast or the scattered field. The multislice ↔ Mie edge runs at NA 0.5 / 0.8 / 1.2 for x = 1–10 at each preset's dz, with a dz sweep (λ_m/2, /4, /8); the obliquity-corrected march has a stability test on a thick, strong phantom (Δn 0.1 over 20 µm) against plain BPM and the reference; dense-density SIM is compared with EmitterSet SIM on the modulated component.

### 11.5 Gradients

- `gradcheck` in complex128 with `check_forward_ad=True` and `check_batched_grad=True`, plus `gradgradcheck`, on every `ops` kernel and custom Function: full mode on tiny shapes, `fast_mode` in bulk. The shapes include q = 0, k = k_in, x → 0 and θ = 0.
- Finite-difference checks for radius, n, position, NA, Zernike, defocus, condenser NA, tilt and LED position, each as a data-object or element field.
- Checkpointed and unchecked gradients agree to 10⁻⁶ **with noise enabled** (keyed speckle and scaled-ST detector) for every checkpointed stage. Batch-keyed generator state is unchanged by backward.
- The **non-zero-gradient check** for every shape, material, optics and camera field in every preset × fidelity (`gx.testing.gradient_probe`, both functionals). It includes fields reached through callables (`gx.SDF`) and modules (`gx.NeuralField`, `ModulePupil`).
- Refusal tests: a field requiring gradients along only zero paths raises `GradientPathError` at call time; partial paths warn.
- A test that the gradient set never changes the forward values: the same inputs and key give identical outputs with and without `requires_grad`.
- An element built once around an `nn.Parameter` gives correct gradients over 10 optimiser steps (no stale views, §6.1).
- EM-gain gradient finiteness on low-λ images with zero-electron pixels.

### 11.6 Camera estimator suite (modelled on [M-D Exp B])

Bias and variance of dE[L]/dθ against analytic values, on mean- and variance-sensitive losses:
- plain straight-through Poisson must be *flagged* (exactly 0 on the variance loss);
- scaled ST must be exact on quadratic losses and, at λ ∈ {0, 0.2, 1, 3} with Anscombe and log1p losses, within a declared bias tolerance;
- EM gain: dE[y]/dλ must be unbiased at λ ∈ {0.2, 1, 3};
- **sCMOS likelihood** (Q6, M1): the shifted-Poisson, Gaussian (μ + σ²) and truncated-convolution approximations against the exact Poisson ⊛ Gaussian likelihood, at 0.1–100 photons and on the E2 calibration maps. The benchmark measures the error of the log-likelihood, the bias and variance of its gradients with respect to μ, gain, offset and read noise, the resulting CRLB, and speed. The winner becomes the `log_prob` default. *Result (M1, 2026-09-29):* the Gaussian is unbiased for μ, gain, offset and read noise from 0.1 to 100 photons, with an efficiency of ≥ 0.93 for μ that falls to 0.48 for the read noise at σ = 0.8; the shifted Poisson is biased at low light; the truncated convolution is exact at ≈50× the cost. The Gaussian is the default, and the convolution stays available (`likelihood="convolution"`).

The estimators for scene draws (ST-Concrete, score + LOO, truncations) belong to sampling code, not to gradix. The cookbook recipes that use them are tested (§11.10) with the reference torch sampler in `tests/`, which also exercises the render with relaxed presence.

### 11.7 Statistics and determinism

- KS tests of the camera samplers; Poisson mean = variance; EMCCD variance ≈ 2G²μ; read-noise maps.
- An RNG quality suite on the counter hash: a PractRand or TestU01 SmallCrush subset, plus cross-image and cross-element correlation checks.
- Keyed Poisson exactness: the inverse CDF below λ = 10, PTRS above, and the fallback path, against the Poisson pmf with χ² tests over λ ∈ [10⁻³, 10⁵].
- **Purity:** the same inputs and key give bit-identical outputs under `deterministic=True` with the recorded chunk sizes.
- **Image-keyed noise is independent of batch composition:** image 17 is identical alone and inside batches of 32 and 64.
- A CUDA-graph capture of a Pipeline call makes no host syncs and gives fresh outputs; the output-ownership test of §6.5.
- Stable Pipeline hashes; the frozen-Pipeline JSON round trip, including chunk sizes.

### 11.8 Consistency contract

Across presets on ~30 canonical scenes: photon counts, centroids, sign conventions and label geometry (§5.9).

### 11.9 Planner, Pipeline and data-object tests

- Golden `explain(level="decisions")` snapshots, so behaviour changes are reviewed as diffs.
- Hypothesis property tests over randomly generated data objects and Chains:
  - static shapes;
  - envelope entries for every shape-affecting field;
  - no all-zero gradient paths;
  - determinism;
  - canonical-layout assertions with B == N == T, where right-aligned broadcasting would otherwise mix images and objects (§6.1).
- `gx.stack`: padding to N buckets with presence 0; errors that name the first structural difference; value checks on CPU inputs.
- `gx.from_si` ∘ `gx.to_si` is the identity on every registered data object and element; the schema has no length field without a quantity.
- Plan-cache keys: structure signature, batch-size bucket, envelope bucket, fidelity, outputs, deterministic flag, device. Re-planning happens only in `gx.render` and on explicit calls.

### 11.10 Level equivalence and the front-end contract

- **Level equivalence** runs on every PR at small shapes and at full size on a local GPU before each release. For every §9 example and every preset × fidelity, three renders must give bit-identical outputs under `deterministic=True`:
  - L2 direct calls given the Pipeline's grids (`grid=`; grid types are L0);
  - the L3 Pipeline;
  - the L4 plan.

  L2 calls that size their own grids must agree within the padding tolerance of the ladder tables. A Chain rebuilt by hand from `plan.chain()`'s printed form must reproduce the plan's outputs.
- **Feeding data** (§6.9):
  - `pipe(values)` equals `pipe(pipe.bind(values))` and the same values passed as a whole Chain;
  - a batch of 40 images in a 64-image Pipeline, eager or captured, equals the same 40 images in a Pipeline built for 40, and neither slot padding (presence 0) nor slicing by count changes any per-image output;
  - `gx.pad` round-trips ragged lists with present slots first, and `gx.envelope_of(values)` ignores absent slots;
  - binding a wider role than planned, a structural field, or an undeclared path into a captured graph raises and names the fix.
- **Schemas:**
  - snapshot tests of every registered data object and element, with a version bump on every change;
  - `gx.from_si` ∘ `gx.to_si` is the identity;
  - `gx.stack` pads to N buckets with presence 0 and names the first structural difference;
  - value checks run on CPU inputs.
- **Coordinates:**
  - `gx.coords` round-trips exactly;
  - an emitter's rendered centroid lies within 0.02 px of its `to_pixels` position at 1000 random sub-pixel positions, with no transposition;
  - the z sign is checked with an astigmatic Zernike;
  - a non-zero ROI offset is handled.
- **The cookbook.** Every recipe of §10.3 runs as a test, with the reference sampler in `tests/`:
  - reparameterised draws reach their parameters;
  - fp64 truncation handles windows with both bounds in one tail;
  - image keys reproduce item i across batch compositions;
  - sampling in CPU workers with late binding in the main process keeps gradients to optics Parameters.

### 11.11 Learnability end to end (nightly)

- Functional fits: Zernikes ≤ λ/50 RMS; radius ±1 %; n ±0.003; defocus, NA and condenser NA from phase-contrast images.
- Distribution learning through the render, using the reference torch sampler: LogNormal median ±3 % and σ ±15 % from unlabeled images. A variant generates its "real" set with gain, offset and illumination different from the generator's initial values.
- Optical neural networks: a 3-layer diffractive stack routes 10 input patterns to 10 detector regions with ≥ 95 % of the power in the right region; the joint classifier of §9(i) reaches ≥ 90 % accuracy on a synthetic two-class phase-object task; physics-aware training against a perturbed twin (misaligned layers, a biased SLM response) recovers ≥ 90 % of the in-silico accuracy.

### 11.12 Performance and memory

The §8.3 workloads run on a local 3090 before each release (`benchmarks/`); a > 20 % regression blocks the release.

### 11.13 Architecture

- import-linter layering;
- source bans (global RNG; stateful and in-place samplers outside `_core/keys.py` and generator-based camera paths; `Distribution.sample`/`rsample`);
- every registered implementation has validity, a tolerance entry, gradcheck shapes and a benchmark shape;
- every registered data object and element has a complete schema;
- the plugin conformance suite (§7.4) ships in `gx.testing`;
- §9 runs as an executable doctest until each sketch's example notebook takes over (§12.7);
- the quality gates, clean on `src/` and `tests/` (`docs/background/` is provenance and stays excluded): ruff lint with the annotation (`ANN`) and numpydoc docstring (`D`) rules; ruff format, which also formats the Python examples of this document; ty for every platform, with warnings as errors; and `numpydoc lint`, run by `pytest`. The package ships a `py.typed` marker.

### 11.14 Validation against experimental data

Each item names a dataset (hosted with a DOI, e.g. on Zenodo, fetched on demand by `gx.testing.experimental`, never shipped in the wheel), the parameters fitted and an acceptance level. Items run weekly on the 3090 runner and are recorded in the fidelity atlas with the gradix version. They are report-only from their milestone; **E1–E3 gate v1.0**. Under the M6 option, E1, which needs the vectorial model at NA 1.4–1.45 as system identification does (§2.2), is reported with the scalar model at v1.0 and gates v1.1. E6 depends on network training and external baselines, so it is reported at v1.0 but does not gate it. Dataset choices and thresholds are initial proposals for the lead developer, preferring in-house DT-lab data where public sets are unavailable.

| # | Milestone | Data | Fitted | Acceptance |
|---|---|---|---|---|
| E1 | M1 | bead z-stacks, 100 and 175 nm beads, NA 1.4–1.45 oil | Zernike + pixel pupil | held-out beads reach reduced χ² ≤ 1.2 under the camera likelihood over ±1.5 µm |
| E2 | M1 | photon-transfer series | gain, offset and read-noise maps | mean–variance slopes reproduced within 2 % |
| E3 | M3 | inline holograms of size-certified polystyrene and silica beads | the (e1) fit | diameter within the certificate uncertainty + 2 %; n within 0.01; residual rings ≤ 1.5× shot noise |
| E4 | M3 | iSCAT of size-calibrated gold nanoparticles or mass-photometry calibrants | reference amplitude and focus only | contrast–size slope within 10 % |
| E5 | M4 | phase contrast or brightfield of beads and bacteria with QPI ground truth | ring transmission, condenser | halo sign and width match; bead phase within 5 % |
| E6 | M5 | annotated experimental sets: holographic sizing, tracking with manual annotations, Cell Tracking Challenge sequences with gold-truth segmentation | networks trained only on data rendered by gradix from the user's own sampling, calibrated per §9(a–b) | no worse than DT2-trained baselines |
| E7 | M4 | LC-PolScope and crossed-polariser images of a calibrated retardance standard, and of birefringent fibres or crystals | compensator swing, retardance and slow-axis maps (the §9(h) fit) | standard's retardance within 2 nm; slow axis within 2° |
| E8 | M4 | SLM-shaped illumination patterns and a fabricated diffractive element, imaged in the lab | SLM phase response, alignment and aberrations | calibrated predictions match the measured intensity patterns within 10 % rel-L2 |

Failures are triaged as model, calibration or data error; a model error opens a ladder item.

### 11.15 CI tiers and platforms

- **Every PR:** the quality gates of §11.13; CPU unit tests on Linux, Windows and macOS (< 5 min, small shapes); all §9 examples at small shapes on CPU; and the documentation build with warnings as errors, with the tutorials executed on CPU and the example notebooks in smoke mode. Hosted macOS runners run CPU only; MPS is best effort and untested in v1 (§12.5).
- **Nightly (GitHub-hosted, CPU):** ladders, learnability, all §9 examples and the example notebooks at reduced sizes; a floor job pinned to torch 2.9.
- **Before each release (local GPU):** the §8.3 benchmarks, full-size level equivalence, CUDA parity and the example notebooks at full size. CI uses GitHub-hosted runners only (2026-09-29), and they have no GPU.
- **Weekly:** oracles and the experimental validation of §11.14.
- **Torch versions:** the latest two minor releases on every PR; the floor (≥ 2.9, raised whenever a needed feature requires a newer torch) nightly.

---

## 12. Roadmap and engineering practice

- **Two developers: ≈38–52 weeks to v1.0** at the scope below; plan on ≈46 (the calendar of §12.3).
- **One developer: ≈1.8× (≈69–93 weeks)**, with the milestone order unchanged.
- **No DeepTrack2-side work is scheduled.** The integration follows DeepTrack2's own optics-pipeline revision (§10.4).

Milestones are ordered by **risk first** (the seams that are expensive to retrofit, then the public API levels), then by value. Revision 1 budgeted ≈38–44 weeks with a parameter layer, an adapter, a simulator, fitting tools and I/O inside gradix. Revision 2 budgeted ≈30–38 with a DeepTrack2 backend contract, revision 5 moved polarisation optics into 1.0, revision 6 added optical stages for optical neural networks, revision 7 set the documentation standard, and revision 9 added learnable magnification and Gaussian beams. §12.2 audits the differences.

### 12.1 Weeks 1–2: the first two weeks

**Before day 1 (lead developer).**
- Name Developers A and B (§3.3).
- Recruit two prospective users (lab members) for the API reviews at weeks 5 and 15.

**Week 1, both developers.**
1. Repository skeleton with the Appendix D packages as empty modules. The uv project already exists, with the quality gates of §11.13 configured and passing: ruff (lint and format), ty, and `numpydoc lint` (run by `pytest`).
2. The import-linter contract of §3.2 and the source-ban test.
3. A GitHub Actions CPU matrix: Linux, Windows and macOS, on the latest two torch minors.
4. A nightly GitHub Actions job stub and the torch-2.9 floor job (GitHub-hosted runners only; GPU checks run locally).
5. `docs/adr/` seeded from §13. The review measurement scripts (tagged [Rev], already under `docs/background/evidence/review/`) re-run on the CI runner to confirm their numbers.
6. The documentation skeleton (§12.7): Sphinx with numpydoc, autosummary, the pydata theme and myst-nb, built in CI with warnings as errors; `docs/tutorials/` and `docs/examples/` holding the scope tables; a pre-commit hook that strips notebook outputs.

**Week 1, Developer B.** `units.py` and `conventions.py`, with tests for:
- the Fourier-shift sign against `roll`, and Propagate(a)∘Propagate(b);
- the direct-summation angular-spectrum normalisation;
- the +z/defocus sign;
- Zernike ANSI/RMS normalisation;
- fp64 carrier-subtracted phase construction [B07 §2.4];
- IEEE-vs-TF32 MFT precision under `ieee_matmul()`.

**Week 1, Developer A.** `schema/` and `tree.py`:
- `gx.field` metadata, quantities and roles;
- the pytree base for data objects and elements (frozen dataclass, schema validation, `.replace`, `.to`);
- `gx.tree.parameters`/`map`/`to`/`replace` (path form);
- `gx.from_si`/`to_si`;
- `gx.stack` with N-bucket padding and presence;
- structure signatures;
- `gx.schema_of` export.

Tests: SI round trips, ragged stacking, per-image vs shared broadcasting, the B == N == T canonical-layout trap, and an element that holds an `nn.Parameter` across optimiser steps.

**Weeks 1–2, lead developer.** A two-page sketch of the three API levels (§3, §9) for the prospective users, with the §0.3 example as the anchor.

**Week 2, Developer B.**
- `grid.py` (smooth sizes); the carriers as frozen pytree dataclasses, with the carrier-shape conformance test.
- The `gx.imaging.Sprites` element (σ = 0.22λ/NA, erf-integrated).
- `_core/keys.py`: the counter hash with open-interval uniforms, and keyed Gaussian and Poisson samplers.
- The `gx.Camera` element (Ideal, Poisson–Gaussian) with `expected`/`sample(key)`/`log_prob`, the NaN-free scaled-ST form, and QE/gain/offset/unit.

**Week 2, Developer A.**
- `gx.Emitters`; `gx.Chain` and `gx.Pipeline` v0, hard-wired for emitters: envelope extraction with headroom and buckets, in-envelope flags, the output-ownership copy, value binding by field path with declared inputs and capacities, `gx.pad`;
- `gx.plan` v0 for emitters with `gx.Sample`/`gx.Microscope`;
- the §9 doctest runner against stub classes;
- the **level-equivalence test** (L2 calls with the Pipeline's grids, the L3 Pipeline and the L4 plan give bit-identical images).

**Week-2 demo:**
- 64 × 128² images render in one GPU call at all three levels, with identical outputs;
- a LogNormal photon distribution sampled with torch `rsample` in a test gets non-zero, finite gradients through scaled-ST Poisson, including at λ = 0;
- image 17's noise is identical in batches of 32 and 64 under image keys;
- the first benchmark number is recorded.

**Weeks 3–5.**
- planner v0 rules (P2) and the Pipeline passes (P0, P1, P3–P8) for emitters; `explain()`; a minimal Pipeline JSON with the recorded chunk sizes;
- position labels;
- the remaining harnesses, including `testing/ladders.py`;
- the week-5 API review with the prospective users, then the M0 freeze.

### 12.2 Budget and scope delta

Estimates are **calendar weeks for the two-developer team after the 1.3–1.5× realism factor**, relative to revision 2's ≈30–38 weeks (itself audited against revision 1's ≈38–44). They are engineering estimates.

| # | Change | Estimate (calendar weeks) |
|---|---|---|
| 0 | Revision 2 total | 30–38 |
| 1 | Remove DeepTrack2-specific deliverables: the `legacy_dt` preset (fluorescence, coherent and brightfield variants), DT coordinate and intensity helpers, DT contract tests and spike support | −(1.5–2) |
| 2 | Make elements a public, documented, stable level: element protocols, eager sizing for direct calls, element-level conformance | +1–1.5 |
| 3 | Split composition from planning: `gx.Chain`/`gx.Pipeline` at L3, the planner emitting Chains, level-equivalence tests | +0.5 |
| 4 | The sampling cookbook (tested recipes) and the "Differences from DeepTrack2 2.0.2" page | +0.25 |
| 5 | Revision 4: feeding data to built Pipelines (value binding, declared inputs, capacities, `gx.pad`, dataset envelopes, example (g)); photoselection and analyser channels in M4b | +0.25–0.5 |
| 6 | Revision 5, polarisation in 1.0 (Q21): coherent vector producers with s/p stack transmission, Jones backgrounds and analysers in Detect (+1–1.5); Jones pupil optics, Jones DIC, thin Jones samples, the PolarizedLight and PolScope presets, E7 (+2–3) | +3–4.5 |
| 7 | Revision 6, optical stages and the ONN core: stage elements, Chain stages and stage-plane sampling (+1–1.5, partly the relay optics that Q20 always implied); shaped coherent illumination for sparse and dense producers (+0.75–1); `PhaseSLM`, `HeightMap`, region readout, example (i), E8 (+0.75–1) | +2.5–3.5 |
| 8 | Revision 7, documentation standard: numpydoc enforced by ruff, `numpydoc lint` and a schema cross-check; the tutorial and example notebooks executed in CI; the friction triage; the documentation share raised from ≥ 10 % to ≈ 15 % of every milestone | +1–2 |
| 9 | Revision 9, the lead developer's answers of 2026-09-28: learnable magnification and pixel size in 1.0 (cap-26 in M2: continuous pixel centres and affine registration; +1–1.5); Gaussian beams as backgrounds, sources and references (M3; +0.75–1); diverging references as an unbudgeted M4 stretch goal | +1.75–2.5 |
| | **Total** (30–38 plus +8.25 to +13.75) | **≈38–52** |

**The M6 option** (the engineering judge's recommendation). Vectorial optics and per-object pupils (≈3.5 weeks) leave the path to 1.0 and become a **v1.1 milestone M6 (≈4–6 weeks, including stabilisation) directly after 1.0**. Milestone M4b disappears, and 1.0 lands ≈3–3.5 weeks earlier (≈35–49 weeks).
- Consequence for coverage (§2.2): system identification (#15) reaches minimum fidelity at v1.1, because vectorial is "M" in [B08]'s matrix. SMLM keeps its v1.0 minimum.
- E1 then gates v1.1 instead of v1.0 (§11.14).
- Decide by the start of M3 (§14.2).

### 12.3 Milestones

```
week     1   5   9   13  17  21  25  29  33  37  41  45
M0 v0.1  █████                                          skeleton at all three levels; schemas, carriers, keys frozen
M1 v0.2      ████████                                   pupil + fluorescence + per-frame inputs; camera likelihoods
M3a v0.3h           ████████                            holography + Mie pulled forward (ADR-43); M2 and later shift by its length
M2 v0.3           ██████                                composable API public (incoherent): protocols, Chain/Pipeline/plan, cookbook, learnable magnification
M3 v0.4                ███████████████                  coherent sparse (P = 1, 2) + references + Gaussian beams + graphs + optical stages; coherent seams frozen
M4 v0.5                             ████████████        coherent dense + partial coherence + lensless + polarisation optics
M4b                                  ████████           vectorial + per-object pupils (developer B; or the M6 option)
M5 v1.0                                        ████████ hardening: contract CI, conformance, docs and notebooks, rc feedback, API freeze
```

**Milestone rule:** every API named in an exit criterion is a deliverable of that milestone or an earlier one. Documentation is built incrementally (≈ 15 % of every milestone, §12.7), and a milestone's notebooks must run before it exits.

| Milestone | Deliverables | Exit criteria (numeric where possible) |
|---|---|---|
| **M0 — v0.1 walking skeleton** (wk 1–5) | <ul><li>`conventions.py`, units, grids; `schema` and `tree`</li><li>data objects `Emitters`, `Spheres`; elements `gx.light.PlaneWave`, `gx.imaging.Sprites`, `gx.Camera` (Ideal, Poisson–Gaussian); `gx.Objective`</li><li>incoherent carriers with fixed shapes; coherent carriers **drafted, not frozen**; fixed-rank axes incl. A and P</li><li>precision policy (`ieee_matmul`); registries; the counter hash and keyed Gaussian/Poisson</li><li>`gx.Chain` and `gx.Pipeline` v0 (P0, P1 and P3–P8 for emitters: envelope → SamplingPlan, `explain()`, minimal JSON, output-ownership contract)</li><li>`gx.plan` v0 (P2 for emitters) with `gx.Sample` and `gx.Microscope`; `gx.render` (plan cache, pre-render envelope check)</li><li>value binding by field path (`pipe(values)`, `bind`), declared inputs, batch and slot capacities, `gx.pad`, `gx.envelope_of(values)`</li><li>`gx.stack`, `from_si`/`to_si`, structure signatures, schema export; position labels</li><li>harnesses: layering lint, bans, gradcheck helpers with forward AD, benchmark runner, conformance skeleton, camera estimator suite, `testing/ladders.py`, the §9 doctest runner, the level-equivalence test</li></ul> | <ul><li>the same batch renders on the GPU through direct element calls, a hand-built Pipeline and `gx.plan`, with bit-identical outputs under `deterministic=True` given the same grids; values bound by field path and a ragged last batch give the same images as whole Chains</li><li>non-zero, finite gradients reach a torch-sampled LogNormal photon median and σ through scaled-ST Poisson</li><li>purity: the same inputs and key give bit-identical outputs; image-keyed noise is independent of batch composition</li><li>the camera estimator suite flags plain ST</li><li>carrier-shape and canonical-layout conformance</li><li>§9 doctests pass against stubs</li><li>**API review with two prospective users → units, frame, axes, schemas, incoherent carriers, key semantics and label shapes frozen**</li></ul> |
| **M1 — v0.2 pupil & fluorescence** (wk 5–12) | <ul><li>pupil modifiers: soft aperture, H_exp defocus, `Zernike`, pixel pupil as OPD, √cosθ apodisation, Gibson–Lanni and the **layered collection rule** from `LayeredMedium`</li><li>pixel MTF</li><li>`gx.imaging.PointPSF` (sparse MFT-ROI, ROI from the stationary-phase rule; global-spectrum fallback)</li><li>`gx.imaging.Strata` + turbid slab</li><li>soft rasteriser (explicit parameter inputs, recompute-in-backward); `Ellipsoids`, `Capsules`, `Cylinders`, `Boxes`, `Gaussians`, `SDF`, `Occupancy`, `Voxels`; `Labeling` (volume/surface, bead quadrature)</li><li>analytic excitation sources + `gx.excite.Linear` on the product grid (uniform, TIRF, SIM beams, Gaussian sheet)</li><li>sCMOS maps and EMCCD (batch-keyed)</li><li>masks, heatmaps, distance maps, emitter tables</li><li>per-frame (A = T) fields and `Frames(interval, exposure)`</li><li>presets Widefield, TIRF, SIM2D</li><li>`gx.detect.log_prob`/`fisher`</li><li>coherent smoke thread: a minimal `gx.interact.Dipole` and `gx.imaging.Coherent` over the drafted carriers</li><li>tutorial T1 (notebook)</li></ul> | <ul><li>power test within 10⁻³; layered collection efficiency within 10⁻³ of 0.709 / 0.575 and grid-converged; Airy</li><li>pixel MTF ↔ fine box ≤ 1 %; sparse ↔ global ≤ 10⁻⁴; strata ↔ sparse ≤ 10⁻⁶ on-plane; dense-density SIM ↔ EmitterSet SIM ≤ 2 % on the modulated component</li><li>≥ 20k img/s fwd and ≥ 5k fwd+bwd (128², 50 emitters); ≥ 30k SMLM frames/s; dense ≥ 5k img/s</li><li>synthetic bead-stack Zernike recovery ≤ λ/50 RMS in < 2 min (scalar), in the loop of §9(b)</li><li>a DeepSTORM3D-style mask-learning loop runs; TIRF depth test</li><li>**coherent smoke thread**: plane-wave source, dipole `ObjectSpectra` on pupil support, scalar pupil imaging and explicit-interference detection pass the optical-theorem and direct-summation tests</li><li>E1/E2 reported</li><li>the sCMOS likelihood benchmark (accuracy and gradients against the exact Poisson ⊛ Gaussian, §11.6) sets the `log_prob` default (Q6)</li><li>tutorial T1 runs in CI, and its `api-friction` issues are triaged</li></ul> |
| **M3a — v0.3h holography and Mie** (≈7–9 weeks after M1; ADR-43) | <ul><li>*Phase 1, Mie and coherent Chains:* `gx.interact.Mie` (fp64 coefficients, S₁/S₂ on the pupil support, n_max from the envelope, forward mode and double backward); `gx.interact.Dipole` as Mie-dressed a₁ + b₁; coherent Chain and Pipeline execution (light → scatterers → imaging → references → detection stages → camera) with envelopes, statics, gradient routes and labels for scatterers; the pupil modifiers on the coherent path; `gx.out.Field` and `labels.Phase`; preset InlineHolography; `gx.recon.Inline`</li><li>*Phase 2, off-axis:* `gx.light.ReferenceBeam(path="image")` with the reference spacing rule; preset OffAxisHolography; `gx.recon.OffAxis`</li><li>*Phase 3, iSCAT:* epi illumination (`PlaneWave(travel=-1)`); `gx.optics.fresnel` (reflected zero order, transmitted incidence); backscatter under the layered collection rule, supercritical samples included; `gx.pupil.Filter`; preset ISCAT (iSCAT-lite as a homogeneous Sample with an on-axis `ReferenceBeam`); `gx.recon.ISCATContrast`</li><li>*Phase 4, QWLSI:* `gx.optics.Grating` and the `gx.DiffractionOrders` carrier (ADR-45); `Microscope.detection_optics`; preset QWLSI; `gx.recon.QWLSI`</li><li>*Phase 5, design:* `gx.crlb(..., reconstruction=)` (ADR-44); shaped illumination for sparse producers from an illumination-pupil `PhaseSLM` (J plane waves, Mie summed over J in chunks): `gx.light.Shaped` and `gx.devices.PhaseSLM`</li><li>*Phase 6, vector:* Mie and dipole at P = 2 with the lens rotation, Jones backgrounds and references, the P = 1 ↔ P = 2 table (Q23)</li><li>examples Ex10 and Ex11 (notebooks)</li><li>taken from M3, whose row keeps them for reference: Mie, dipole, references, Fresnel and backscatter, the three presets, `gx.out.Field`, `labels.Phase`, shaped illumination for sparse producers and `PhaseSLM`</li></ul> | <ul><li>Mie a_n and b_n within 10⁻¹⁰ and S₁, S₂ within 10⁻⁸ of an independent fp64 reference (scipy's spherical Bessel functions; miepython when installed); Mie ↔ dipole ≤ 2 % for dielectrics at x ≤ 0.3; direct-summation normalisation ≤ 10⁻³; the optical theorem from the rendered image within 10⁻³; forward-mode Jacobians within 10⁻⁶ of fp64 central differences</li><li>off-axis sideband separation at \|u_ref\| ≥ 3NA; the off-axis reconstruction equals the true field limited to its window within 10⁻³ rel-L2 in the frame's interior (noiseless; as built 8.8·10⁻⁴, and 4·10⁻⁵ for a point-sampled hologram); in-line refocusing of the rendered field matches the field rendered in focus there within 2 %, and on holograms the twin image biases magnitude focus metrics by ≈0.2 µm at 3 µm (as built)</li><li>iSCAT: the reference amplitude equals r(0) within 10⁻⁶; with the focus on the particle, the contrast oscillates with its height at λ/(2n_s) within 2 % at NA 0.9 (as built 1.0 %; the collected ⟨cos θ⟩ lengthens it, 4 % at NA 1.4); collected backscatter converges across NA = n_s, changing by < 0.5 % from 128² to 256² pupil grids (as built 0.26 %, the critical angle's branch point limiting the order); equal indices reproduce the homogeneous medium within 10⁻¹⁰</li><li>QWLSI: harmonics at the grating's difference frequencies; shear λd/Λ; the recovered phase of a weak phase object within 2 % rel-L2 (noiseless; as built: only those harmonics in the empty frame, a single order's shear exact to 4·10⁻⁵ of the intensity, and a phase error of 8·10⁻⁴ for a bead of 1 µm radius and Δn = 0.01)</li><li>the bound: F_R ≤ F for every parameter; F_R = F within 10⁻⁸ for an invertible R; F_R within 10⁻⁶ of a dense covariance inversion on a small case; its gradient within 10⁻⁴ of finite differences (as built: 3·10⁻¹⁵ against the dense inversion, 3·10⁻⁷ for the gradient)</li><li>≥ 1k holograms/s forward (128², 10 beads, Mie at P = 1, one mode) and ≥ 300 forward + backward, re-baselined when first measured</li><li>Ex10 and Ex11 run in smoke mode in CI, and their friction issues are filed (as built: Ex10 in 8 s and Ex11 in about 2 min on the GPU, a few seconds each in smoke mode; issues #9–#13)</li></ul> |
| **M2 — v0.3 composable API, incoherent** (wk 10–15) | <ul><li>element protocols documented, with reference pages generated from the schemas</li><li>`gx.Chain`/`gx.Pipeline` public: `validate`, `gradients`, `run`/taps, `diff`, `replace`, `capture` for incoherent chains</li><li>`gx.plan` rule table for emitters and labelings, with `methods=` pinning and `plan.chain()`</li><li>`gx.stack` over Chains, Samples and Microscopes</li><li>continuous pixel centres (cap-26): learnable magnification, pixel pitch, rotation and sub-pixel offset through an MFT, and affine registration between channels</li><li>the sampling cookbook (§10.3) as tested documentation</li><li>tutorials T2–T4 and example Ex1 (notebooks, §12.7)</li></ul> | <ul><li>examples (0) and (b) (scalar PSF, without its CRLB line) and an SMLM data-generation loop run at L2, L3 and L4 with identical outputs</li><li>every cookbook recipe passes its test; tutorials T1–T4 and example Ex1 run in CI, and their `api-friction` issues are triaged before the freeze</li><li>magnification and pixel pitch recovered within 0.1 % from synthetic bead images; continuous sampling equals integer-grid sampling within 10⁻⁶ where both apply</li><li>two external users build fluorescence pipelines at the level of their choice and report</li><li>**API review → incoherent element protocols and Chain/Pipeline semantics frozen** (later changes need an ADR)</li></ul> |
| **M3 — v0.4 coherent sparse + references + optical stages** (wk 15–29; the items M3a delivers are listed there) | <ul><li>`gx.interact.Mie` (fp64, homogeneous and layered; GPU; n_max from the envelope; custom S(θ) backward with `jvp`); `gx.interact.Dipole` (Mie-dressed a₁ + b₁); `gx.interact.Born` with the φ predicate</li><li>pupil-support evaluation at P = 2 (vector, with the lens rotation and s/p stack transmission) and at P = 1 (the S∥ convention); analytic background; Jones-vector backgrounds and references; analyser channels in Detect</li><li>Köhler/LED `Quadrature(n)` with continuous nodes + on-support draft; explicit-interference detection</li><li>references: in-line, blocked, reflected; image-side off-axis</li><li>Gaussian beams (`GaussianBeams`, the second `Background` implementation): illumination, and sample- or image-side references</li><li>`gx.interact.Fresnel` reference + backscatter</li><li>presets InlineHolography, OffAxisHolography, Brightfield (particles), Darkfield (including cross-polarised), ISCAT (+ iSCAT-lite)</li><li>host-index rule</li><li>**CUDA-graph capture** for all chains</li><li>`gx.out.Field` and `labels.Phase`</li><li>optical stages (`Mask`, `Propagate`, `Lens`, `FourierTransform`, `Aperture`, `DiffractiveStack`) in illumination and detection, with stage-plane sampling; shaped coherent illumination for sparse producers (J plane waves, local plane waves); `PhaseSLM` and `HeightMap`; `gx.out.Regions`</li><li>tutorial T5 and examples Ex2–Ex4 (notebooks)</li></ul> | <ul><li>Mie coefficients within 10⁻⁸ of miepython; direct-summation normalisation ≤ 10⁻³; optical theorem</li><li>P = 1 against holopy `MieLens`; P = 2 within 10⁻⁴ of the fp64 vector reference, with its x component equal to the P = 1 field for on-axis incidence; the crossed-analyser image of a sphere follows (S₂ − S₁)sinφcosφ; dipole ↔ Mie ≤ 2 % for dielectrics at x ≤ 0.3; born_ff ↔ mie table seeded</li><li>iSCAT period and Fresnel amplitude; darkfield zero-order exclusion; off-axis sideband separation</li><li>Gaussian beams: evaluation against their angular-spectrum integral ≤ 10⁻⁴, the Gouy phase and waist evolution, and a Gaussian-illuminated sphere against the J-wave reference</li><li>≥ 3k img/s holography (1 mode); ≥ 1.5k img/s BF tracking at quadrature(9); ≥ 1k img/s fwd+bwd holography</li><li>examples (a) and (e1, without its CRLB line) run at L2, L3 and L4 with identical outputs</li><li>example (a) recovers a LogNormal (median ±3 %, σ ±15 %) from 10k synthetic "real" images, also with a photometric mismatch</li><li>example (e1), without its CRLB line (M5), recovers r ±1 % and n ±0.003 at SNR 20</li><li>example (g) on 1000 synthetic holograms refines rough starts (100 nm lateral, 0.5 µm axial RMS) to ≤ 10 nm lateral and ≤ 40 nm axial RMS, and the focus offset to ≤ 20 nm (initial targets)</li><li>free-space stages pass the analytic tests (Fresnel square aperture ≤ 10⁻³; 4f identity and BL-ASM energy ≤ 10⁻⁶); `Shaped` with J = 3 equals `SIMBeams`; a 3-layer diffractive stack routes 10 inputs to 10 regions with ≥ 95 % of the power in the right region; example (i) runs</li><li>T5 and Ex2–Ex4 run in CI, and their friction issues are triaged before the coherent freeze</li><li>**coherent carriers and coherent element protocols frozen**; E3/E4 reported</li></ul> |
| **M4 — v0.5 coherent dense + polarisation optics** (wk 28–39) | <ul><li>`gx.interact.Projection`: centre plane, (t−1)E_b, 1/cosθ_out</li><li>`gx.interact.Multislice`: obliquity-corrected, in the residual-demodulated frame; scattered form opt-in; plain BPM opt-in (`obliquity=False`); free-space BL-ASM; procedural slices + checkpointing; pre-rasterised inference; interaction grid from band; raster s = 2 at `accurate`</li><li>per-mode Abbe for thick samples, with cost warnings; shaped illumination for dense producers (sampled incident field)</li><li>PhaseRing, CentralStop, DICShear (scalar and Jones), DPC</li><li>Jones pupil optics (`Polarizer`, `Retarder`, `JonesMatrix`); thin Jones samples (uniaxial materials in the projection element, `gx.JonesScreen`); per-frame polarisation states</li><li>`gx.Parent` hierarchies; keyed `Texture`; cut-out-and-fill</li><li>learnable `Voxels`, `NeuralField`, `ModulePupil`</li><li>stretch goal: diverging references (`SphericalWaves`) for point-source lensless holography; they move to 1.1 if M4 runs late</li><li>lensless preset (cap-30); presets PhaseContrast, DIC (scalar and Jones), DPC, QPI, Brightfield (volumes), FPM, Lensless, PolarizedLight, PolScope</li><li>examples Ex5–Ex6 (notebooks)</li></ul>*Developer A*: hierarchies, textures, cut-out-and-fill, learnable voxels, Jones pupil optics and thin Jones samples, presets. *Developer B*: the march and propagators, interleaved with M4b (wk 29–36), then Jones DIC and the PolScope preset | <ul><li>projection ↔ multislice ≤ 2 % (d ≤ DOF/4); voxel sphere ↔ Mie ≤ 5 % on the scattered field (NA 0.5/0.8/1.2, x = 1–10, dz = λ_m/2)</li><li>march stability test; quadrature(25) ↔ grid Abbe ≤ 2 %; halo and sign tests; seam and tilt-equivariance tests</li><li>training ≤ 1 GiB at ≥ 34 img/s; inference ≥ 1k img/s</li><li>a uniform retarder between crossed polarisers follows I₀sin²2θ·sin²(δ/2) within 10⁻⁵; Jones DIC equals scalar DIC on isotropic samples within 10⁻⁵; the five-frame LC-PolScope algorithm recovers retardance within 1 nm and slow axis within 1° from rendered frames of a test pattern</li><li>dense and sparse routes agree within 1 % on a weak phase object under shaped illumination</li><li>if the stretch goal lands: a point-source in-line hologram of a sphere matches the Weyl-expansion reference</li><li>examples (c), (e2) and (h) run</li><li>Ex5–Ex6 run in CI, and their friction issues are triaged</li><li>E5, E7 and E8 reported</li></ul> |
| **M4b — vectorial and per-object pupils** (wk 29–36, Developer B; or M6) | <ul><li>vectorial 6-basis sparse emitters with Fresnel t_s/t_p under the layered rule</li><li>photoselection (Cartesian excitation) and analyser channels, including polarisation cameras</li><li>per-object field-dependent pupils</li><li>`psf="auto"` thresholds (§5.6)</li></ul> | <ul><li>vectorial PSF within 10⁻³ of psf-generator</li><li>the Stallinga fixed-dipole bias is reproduced by the scalar tier and removed by the vectorial one</li><li>fixed-dipole analyser channels (0°, 45°, 90°, 135°) within 10⁻³ of the vectorial reference; photoselection follows cos² of the angle between dipole and excitation polarisation</li><li>the scalar ↔ vectorial table (≤ 2 % at NA/n ≤ 0.3)</li><li>vectorial bead-stack fit < 60 s; ≥ 5k vectorial frames/s</li></ul> |
| **M5 — v1.0 hardening** (wk 39–46) | <ul><li>public `gx.compare`; `gx.testing.gradient_probe` with the noise-energy functional</li><li>consistency-contract CI; golden Pipeline snapshots + hypothesis planner and data-object tests</li><li>public conformance suite; documentation complete (built since M0), including "Differences from DeepTrack2 2.0.2"; performance gates (the benchmark scripts on a local GPU before each release)</li><li>an exact keyed Poisson sampler as fast as `torch.poisson`: a fused kernel for batch and image keys (§6.3)</li><li>example (f) and example notebook Ex7; every notebook revised against the release candidate; v1.0rc</li></ul> | <ul><li>every §8.3 target met **or re-baselined from a recorded median-of-5 measurement of the committed script**</li><li>contract green across presets; E1–E3 pass (E1 at v1.1 under the M6 option) and E6 is reported</li><li>**v1.0rc published, and feedback from ≥ 3 external users building their own pipelines collected for ≥ 2 weeks, with blocking issues triaged**</li><li>every `api-friction` issue fixed or recorded as a deliberate choice before the freeze</li><li>public API per §12.8 reviewed and tagged 1.0.0 → API frozen</li></ul> |

**Why this order de-risks:**
- **M0** freezes only what its skeleton exercises: units, frame, axes, schemas, incoherent carriers, key semantics and label shapes. It proves that the three levels agree on day-one code, so the layering is tested before there is much to layer.
- **M1** delivers the cheapest high-value physics (sparse fluorescence) and exercises the drafted coherent carriers with a smoke thread.
- **M2** puts the composable API in front of external users while the element set is still small, and freezes the incoherent protocols only then.
- **M3** proves the central physics bet (analytic zero order, travel tags, pupil-support evaluation) on analytic producers, at P = 1 and P = 2, before any dense solver exists. Only then do the coherent carriers and element protocols freeze.
- **M4** adds the expensive dense path and the polarisation optics onto proven contracts.
- **M4b** runs vectorial optics in parallel. Its risk is conventions (apodisation direction, Fresnel), which are pinned by then, and it has a stabilisation window before the freeze. The M6 option removes it from the 1.0 path altogether.
- **M5**'s external gate is feedback on a published release candidate, not outcomes outside the team's control (E6 is reported, not gated, §11.14).

### 12.4 After v1.0

**M6 — v1.1** (only if the M6 option is taken; ≈4–6 weeks directly after 1.0): vectorial 6-basis sparse emitters and per-object field-dependent pupils (cap-17, cap-18), with the M4b exit criteria and the E1 gate (§11.14).

**DeepTrack2 integration**, timed with DeepTrack2's optics-pipeline revision (§10.4, Appendix F.8).

**v1.x** (≈12–14 weeks after 1.0; ordered by demand):
- C1 injection;
- Richards–Wolf focusing of excitation beams (polarised confocal/ISM and focused-beam excitation, with cap-25);
- motion-blur sub-exposure positions (cap-24);
- confocal/ISM/2P effective PSFs; 3D-SIM;
- per-instance validity routing; `on_invalid="escalate"`;
- SOCS/WOTF opt-in and the snap-and-shift rewrite; stochastic source nodes (`Koehler(nodes=gx.Stochastic(n))`, keyed) for fitting;
- tabulated spline PSF import; QCMOS; image-keyed EMCCD;
- an automatic dense boundary: periodic whenever the wrap bleed estimated from the z positions and the pupil stays below a tolerance, linear otherwise;
- entry-point plugins with plugin-API versioning; cross-fidelity surrogate gradients; `Pipeline.graph` and `Pipeline.profile`; general element DAGs if a trigger appears;
- the extended ONN toolkit (cap-39), unless Q22 pulls it into 1.0;
- MPS support if a user trigger appears (§12.5).

**v2 track** (each item lands with ladder tables and conformance tests):
- SSNP; slice-wise Born/Rytov; reversible/segmented adjoints with in-loop VJP and angle mini-batching helpers (ODT);
- near-interface dipoles (LDOS, orientation), vectorial iPSF, roughness screens;
- low-rank dense space-variant PSFs; light-sheet BPM through tissue; emission through RI volumes;
- tensor ε and polarised multislice;
- Foldy–Lax; meshes; deepinv export;
- an MBS reference solver validated against wavesim;
- nonlinear source maps (2P, saturation, STED effective PSF).

### 12.5 Platforms

| Tier | Platform | Scope |
|---|---|---|
| 1 (features and performance targets) | CUDA on Linux and Windows, sm ≥ 7.0 | everything; §8.3 targets on the reference RTX 3090 |
| 1 for correctness, 2 for speed | CPU on x86-64 and arm64 (Linux, Windows, macOS) | All features, with no CUDA graphs (`Pipeline.capture` returns the Pipeline uncaptured, with an info note) and `deterministic=True` by default. One to two orders of magnitude slower than a 3090: ≈25× for a multislice march to ≈140× for sparse MFT on an idle 16-thread desktop CPU, more under load [Rev]; to be re-measured on a reference laptop and published. For tests, small datasets, notebooks and small fits |
| 3 (best effort, untested in v1) | Apple MPS | Complex64 only. The intended design routes **all fp64 work (special functions and phase-argument construction)** to the CPU, with a documented sync cost and cross-device autograd, or uses an on-device double-float (two-fp32) reduction mod 2π validated against CPU fp64. No graphs; correctness-only tolerances. Built when a user trigger appears, then tested on a self-hosted Apple-silicon runner (hosted macOS runners run CPU only) |
| untested | ROCm | — |

`device=` defaults to the device of the input tensors. Memory estimates use host RAM on CPU, and `explain()` prints a per-device-class throughput estimate. v1 has no MPS tests or MPS-specific routes (ADR-25); both are built only when a user trigger appears.

### 12.6 I/O and persistence

gradix reads and writes only its own artefacts:
- **Pipelines and plans:** `to_json()` / `from_json()`, including chunk sizes, so a dataset can be regenerated exactly.
- **Inputs:** `gx.save_inputs(path, obj)` / `gx.load_inputs(path)` for any pytree of data objects and elements. They write tensors plus the registry names and schema versions (§7.1); schema versions ship migration functions.
- **Camera calibration maps:** `gx.Camera.scmos(maps="….npz")` for offset, gain, variance and QE maps.

Image readers (TIFF, OME-Zarr), dataset writers, loaders and experiment metadata belong to the caller: user code, deeplay, or DeepTrack2 later. Validation sets that must match across machines are stored, not regenerated, because bit-exact rendering holds only on the same device class and version (§6.3).

### 12.7 Documentation

Documentation is a deliverable of every milestone, not a phase: **≈ 15 % of every milestone**, and a milestone does not exit until its notebooks run.

**Docstrings.** Every public module, class, method and function has a numpydoc docstring; private helpers have one when their behaviour is not obvious.
- **Content:** a one-line summary in the imperative; Parameters, whose type line states the tensor shape in the axis letters of §4.2 and the unit (`position : Tensor, shape (B, T|1, N, 3), µm`); Returns; Raises; Notes with the physics (equations, validity regime, references with DOIs); See Also; and, for the user-facing API (levels L2–L4), Examples that run as doctests on the CPU at small shapes.
- **Schemas:** the Parameters sections of data objects and elements are checked against their schemas (names, quantities, roles, event shapes), so documentation and schemas cannot drift. Reference pages show both.
- **Gates** (§11.13): ruff's docstring rules with the numpy convention check presence and form, and `numpydoc lint`, run by `pytest`, checks content (every parameter documented, Returns and Raises present, well-formed sections). A second `numpydoc lint` pass requires Examples in the user-facing packages.

**Site.** Sphinx with numpydoc and autosummary, the pydata-sphinx-theme, myst-nb for notebooks, and intersphinx links to torch and numpy. It builds on every PR with warnings as errors, broken cross-references included, and is published from the main branch.

**Tutorials and examples.** Both are Jupyter notebooks. Tutorials teach working with gradix, in order; each example does one concrete thing. The tables are the 1.0 scope: a new notebook needs a group of users that the existing ones do not serve. There are no separate how-to guides: their topics are sections of the notebooks (bring your own sampling in T3, editing a planned Chain in T4, pupil calibration in Ex1, porting a mask-based scatterer in T5). The §9 sketches are the design of the examples, and each becomes an example notebook when its milestone lands. Each notebook states its level, prerequisites and runtime.

| Tutorial | Topics | Level | First in |
|---|---|---|---|
| T1 First images | units and conventions; data objects; elements called directly; the camera (expected and sampled images) | L2 | M1 |
| T2 Chains and Pipelines | envelopes and `explain()`; batches and keys; outputs and labels; feeding data by field path | L3 | M2 |
| T3 Gradients and your own sampling | learnable parameters; reparameterised draws and the cookbook; the gradient table; camera likelihoods | L2–L3 | M2 |
| T4 Fidelity and the planner | presets; `why`, `diff` and `compare`; pinning elements; editing a planned Chain | L4 → L3 | M2 (coherent parts: M3) |
| T5 Extending | a new shape, a pupil modifier and an element, with schemas and the conformance suite | L2 | M3 |

| Example | A concrete task | Sketched in | First in |
|---|---|---|---|
| Ex1 SMLM | fit a pupil to a bead stack, then generate training data with it | §9(b) | M2 |
| Ex2 Holography | learn a size distribution from unlabeled holograms, then train a network | §9(a) | M3 |
| Ex3 Particle characterisation | Lorenz–Mie fits, and refinement over a dataset | §9(e1), §9(g) | M3 |
| Ex4 Optical neural network | patterned illumination and a diffractive front end, trained with a classifier | §9(i) | M3 |
| Ex5 Label-free cells | phase contrast, QPI and intensity diffraction tomography | §9(c), §9(e2) | M4 |
| Ex6 Polarised light | crossed polarisers and LC-PolScope of birefringent fibres | §9(h) | M4 |
| Ex7 Particle tracking | sequences with Brownian motion and focus drift | §9(f) | M5 |
| Ex8 Labelled cells | dense fluorescence of labelled solids under widefield, light-sheet and structured illumination, and in turbid tissue | — | M1 (cap-06) |
| Ex9 PSF engineering | design a phase mask by Fisher information, then end-to-end with a localisation network | M1 exit (mask learning) | M1 |
| Ex10 Interferometric microscopes | in-line, off-axis, iSCAT and QWLSI views of nanobeads, and their reconstructions against the true field | §5.12 | M3a |
| Ex11 Light shaping for nanobead characterisation | minimise the bound on radius and index through each reconstruction, over illumination, pupil and reference | §5.12, ADR-44 | M3a |

Ex8 serves dense fluorescence imaging (cell biology, light-sheet and SIM users) and Ex9 PSF design, groups that Ex1–Ex7 do not reach. Ex10 serves label-free particle characterisation with interferometers (holography, iSCAT, wavefront sensors) and Ex11 measurement design. Ex1 and Ex7 were drafted at M1 against the API as it landed; their milestones complete them.

- **Kept current.** Notebooks live in `docs/tutorials/` and `docs/examples/`, are committed without outputs (a pre-commit hook strips them) and are executed by the documentation build. Tutorials run on the CPU on every PR. Examples run on every PR in smoke mode (small shapes and few iterations, set by one flag at the top of the notebook) and at full size on a local GPU before each release. ruff lints and formats them. A notebook is drafted when its milestone starts, against the API as it lands, and is updated in the same PR as any API change it touches.
- **Finding pain points.** The notebooks are the first users of each API. Whoever writes or updates one files each awkward line as an `api-friction` issue: boilerplate, a surprising name, a concept needed too early, an unhelpful error message. Each milestone's API review triages them, and every issue is either fixed before the next freeze or recorded as a deliberate choice. The number of lines each notebook needs for its task is tracked across milestones, as a coarse measure of ergonomics.
- **Explanations and reference.** Explanations are prose pages: conventions and units; the API levels; carriers as the element contract; fidelity; gradients, keys and purity; one physics page per interaction element, with its validity regime; optical stages; "Differences from DeepTrack2 2.0.2". The reference is the numpydoc API, plus schema tables of the registered data objects, elements and devices, the validity predicates, the tolerance tables and the fidelity atlas. The §10.3 cookbook is a tested page.
- **Owners.** Each developer documents their own packages. The lead developer owns the notebooks and chairs the friction triage. A third developer, if one joins, takes over the site.

### 12.8 Versioning, stability and governance

- **SemVer from 1.0.** The public API is:
  - the names exported by `gradix/__init__`;
  - the documented packages at L2–L4 (the table of §3.4);
  - `testing.conformance`;
  - **the schemas of data objects and elements**, which are the contract with every front end.

  `_core`, `ops` internals, Pipeline internals and `gx.experimental` carry no guarantee. Deprecations warn for ≥ 2 minor releases and ≥ 6 months. After M0, a change to a seam or a schema needs an ADR approved by the lead developer and both package owners. Schema changes are listed in the changelog with migration notes.
- **Numerics policy:**
  - (R1) Bit-exact outputs for the same version, device class, torch version and inputs and keys, under `deterministic=True` with a frozen Pipeline.
  - (R2) Within a minor series, outputs for the same inputs and keys change only within the published tolerance of the affected approximation edge, listed under a changelog heading "Numerics changes". Larger changes, or any convention change, require a major release.
  - (R3) Stored inputs and Pipelines carry schema versions and gradix versions, so they can be re-rendered with a pinned version.
- **Supported versions.** The latest two torch minor releases (floor ≥ 2.9, raised whenever a needed feature requires a newer torch; tested nightly) and Python ≥ 3.10.
- **Plugins.** Distributed as `<name>-<plugin>` packages. Registry names are `<package>.<name>`; core names are unprefixed. Each plugin declares a `requires` range and a plugin-API version. A "verified" listing requires CI passing `gx.testing.conformance` and published tolerance tables. Core never imports plugins.
- **"API frozen"** in the M5 exit means: the public API above reviewed and tagged 1.0.0.

---

## 13. Decision records

Each record has a **Decision** and **Consequences**, plus **Context** and **Rejected** (alternatives named by content) where they add information. Numbers live in the sections cited; the panel history is in [judge-verdicts.json](background/design-panel/judge-verdicts.json) and the review history in Appendix E.

**ADR-00 — Basis of this design.**
*Context:* the judges recommended different bases (physics → field operators, engineering → the MVP scope, ML → learnable pipelines). On 2026-09-25 the lead developer made two decisions: sampling belongs to the caller (ADR-32), and gradix is a standalone engine with a layered public API (ADR-36, ADR-37). On 2026-09-27 they moved polarisation optics into 1.0 (ADR-20) and asked for optical neural networks to be designed in (ADR-39). *Decision:* scope discipline, milestone order and "seams before solvers" from the MVP proposal; physics contracts (analytic zero order, travel tags, slots, approximation edges, conformance, sampling rigour) and composable operators on typed carriers from the field-operator proposal; measured estimator evidence, sizes from static ranges (now envelopes) and output ownership from the learnable-pipelines proposal, whose probabilistic core is handed to sampling layers (Appendix F); planner hygiene (partition, consistency contract, SamplingPlan provenance, golden snapshots) from the scene-planner proposal. *Rejected:* cost-based plan search; randomised fidelity and fidelity schedules in v1; a v1 reversible adjoint; scalar-only v1; grid snapping; tier-dependent radiometry; frame flips; a gradix-owned sampling layer (revision 1); a DeepTrack2 backend contract as a v1 deliverable (revision 2). *Consequences:* ≈38–52 weeks for two developers (§12.2), with the M6 option to move vectorial optics to v1.1.

**ADR-01 — Internal length unit.**
*Context:* small-particle quantities underflow in float32 metres; DT uses SI metres and pixels. *Decision:* µm internally and at gradix's API; SI metres and pixels only at the caller's boundary, through the schema-driven `gx.from_si`/`gx.to_si` and `gx.coords`. *Rejected:* SI metres throughout. *Consequences:* no exponent-range underflow (§4.1); one mechanical conversion at the boundary; learnable parameters, which live in the front end, should be parameterised in natural units (Appendix F).

**ADR-02 — Precision policy.**
*Context:* c128 FFTs are slow, c32 is unstable, TF32 degrades complex matmuls, large phase arguments lose precision, complex64 Mie recurrences err (§8.1). *Decision:* complex64/float32 fields; fp64 for special-function recurrences, phase-argument construction (carrier-subtracted, mod 2π), keyed inverse CDFs and gradcheck; complex32/fp16/bf16 banned from physics; `autocast(enabled=False)`; a `PrecisionPolicy` object, never `set_default_dtype`. IEEE fp32 matmul around MFT/CZT through `torch.backends.cuda.matmul.fp32_precision` (torch ≥ 2.9, tested by a nightly floor job) inside `ieee_matmul()`, **the single sanctioned, scoped mutation of process-global torch state**: exception-safe, restoring the exact previous value, re-entrant, used only on the thread that runs the render call and never held across user callbacks. Reading the legacy `allow_tf32` inside the context raises (documented); CUDA graphs are captured inside it, so captured kernels keep IEEE. A flag-free split-precision (3×TF32) MFT is an option. *Rejected:* global flags; c128 everywhere. *Consequences:* fp64 on GeForce is harmless (tiny arrays). MPS (complex64 only) is best effort: all fp64 work, special functions *and* phase construction, would run on the CPU or use a double-float reduction (§12.5).

**ADR-03 — Frame and z convention.**
*Decision:* +z is the propagation direction toward the detection objective, in every data object, element and solver; z = 0 is the sample-side coverslip surface when a layered medium exists; the focal plane is `Objective.focus`; helpers `gx.coords.depth()`, `from_focus()`, `shift_focus()`; populations whose z-envelope crosses an interface are a plan-time error (§4.1, §5.8). *Rejected:* +z into the sample with planner-inserted frame flips. *Consequences:* no frame flips (a standing source of sign bugs [B05 §8.5]); epi light travels −z, so travel tags are unambiguous; DT's focus-relative z maps through `from_focus` (Appendix F.8).

**ADR-04 — Central interchange representation.**
*Context:* the zero order can be sampled, snapped to the grid, or kept analytic; dense marches could carry it analytically at a per-slice cost. *Decision:* analytic `PlaneWaves` ⊕ sampled scattered spectra for Source, the sparse producers (lazy `ObjectSpectra` on pupil support), the Objective (evaluation at u_b) and Detect. Dense marches march the total field in a **residual-demodulated frame** (exact on-grid carrier, analytic residual tilt), with the scattered-field form as a rule-selected opt-in for weak or zero-order-blocked samples. One `travel` tag per contribution; Interact returns `tuple[Contribution, ...]`. `Field.background` is a `Background` Protocol; `PlaneWaves` and `GaussianBeams` implement it in 1.0, and spherical waves for diverging references are an M4 stretch goal. The incoherent currency is `EmitterSet`/`EmitterDensity`/`Irradiance`. *Rejected:* a sampled total field everywhere (seam errors, weak-interference precision); snapped tilts (non-differentiable, angle error); the analytic background through every slice (1.39–1.46× per march, §8.1); a `Ports(plus_z, minus_z)` carrier and a single `direction` flag (contradictory or insufficient states); a `to_sampled()` snapping fallback for plugins. *Consequences:* seam-free, tilt-differentiable dense marches at no per-slice cost; plugins carry no background obligation and are tested for tilt equivariance (§4.4, §7.4).

**ADR-05 — Apodisation and radiometry.**
*Context:* applying 1/√cosθ collection *and* a 1/k_z Jacobian to an angular spectrum is off by cos²θ; tier-dependent radiometry breaks transfer; the index-matched rule fails when NA > n_s. *Decision:* on the angular-spectrum density, collection × √cosθ and focusing × 1/√cosθ, with 1/k_z only in the far-field conversion; for layered media, the layered collection rule (propagate only toward the interface, stack transmission t(u), one Gibson–Lanni focus term, √(k_z,i/k_i), safe square roots, no clipping at n_s); absolute radiometry identical at every tier (§4.1). *Rejected:* the Jacobian form; per-tier normalisation; clipping the support at n_s. *Consequences:* the index-matched power test, the layered collection-efficiency tests and the direct-summation test pin the rule; supercritical collection is modelled.

**ADR-06 — Coherence representation and source sampling.**
*Decision:* coherent-mode sets with coherence groups (`AngularSource`); deterministic continuous-node quadrature for all data generation; grid Abbe as the reference; draft single modes on the source support; stochastic nodes (v1.x) only for gradient estimation, from the numerics key, and labelled biased for nonlinear losses; SOCS/WOTF opt-in (v1.x) and erroring when invalid (§5.4). *Rejected:* 4-D mutual intensity; stochastic Abbe for training data; centroid single modes. *Consequences:* cost linear in n_src and printed; no Monte-Carlo artefacts in training images.

**ADR-07 — Sample representation.**
*Decision:* continuous analytic primitives are canonical, with introspectable capabilities; a small fixed set of views tagged with exactness (the raster prefilter is tagged approximate, §4.6); voxels, neural fields and occupancy-only shapes accepted; emitters never voxelised; `Texture` resolution-free and keyed; callable lowerings take their parameters as explicit inputs; meshes in v2; priority "over" in ε for materials, densities add. *Rejected:* hard masks (zero gradient); grid-indexed white-noise textures; conversion-graph search. *Consequences:* a new primitive needs only `bbox` + `sdf`; the same inputs render at any fidelity.

**ADR-08 — Optical-path granularity.**
*Decision:* conjugate planes (condenser pupil, sample in its layered environment, one fused objective pupil, image plane plus detection branches); object-space simulation; magnification is a coordinate scale; the Objective slot has a space-variant member from day one (sparse per-object pupils in v1, dense low-rank in v2) (§5.3); ordered optical stages before and after the sample carry relay optics, programmable devices and diffractive networks (§5.11, ADR-39). *Rejected:* surface-by-surface lens models. *Consequences:* one pupil multiplier; learned optics apply to every modality.

**ADR-09 — Fidelity-ladder semantics.**
*Decision:* per-implementation approximation edges with regime-bounded, calibrated tolerances (§5.9); `on_invalid="escalate"` (v1.x, L4 only) makes the planner re-route along the unique edge whose higher implementation accepts the population kind, and ambiguity is a PlanError. *Rejected:* a total-order lattice with join/meet (Born is not a limit of plain BPM; dipole and projection are limits in opposite regimes). *Consequences:* `explain()` prints the tolerance of each chosen edge.

**ADR-10 — Fidelity expression.**
*Decision:* at L4, a frozen knob dataclass (`preset` positional, the rest keyword-only) with presets draft/standard/accurate plus reference; per-population overrides; `methods=` pins elements explicitly. At L3, knobs left "auto" in a Chain's elements are resolved from the envelope by each element's default rule; a `Fidelity` acts only at L4. `compare()` and tolerance tables link the tiers (§5.6). *Rejected (deferred):* lattice policies, schedules, randomised fidelity, error-target planning; a DeepTrack2-parity preset (ADR-37). Surrogate gradients are v1.x. *Consequences:* discrete, readable, sweepable fidelity; plugins selected by knob value or by passing the element.

**ADR-11 — Planner sophistication and timing.**
*Decision:* the L4 planner is a deterministic rule table that chooses an element for every population without an explicit one (P2, reusing the P0 and P1 functions) and emits a Chain. The L3 `gx.Pipeline` runs, for *any* Chain, normalisation, the envelope, validity, the gradient table from capability tags, the SamplingPlan, the memory strategy and a few rewrites (P0, P1, P3–P8; ≈2–3 kLOC in total). Chunk sizes are fixed at plan time (P6) for the Pipeline's capacities and recorded. Cost only orders numerically equivalent strategies (§5.7). *Rejected:* cost-based search; device-calibrated costs; per-element byte models; cross-call value caches. *Consequences:* hand-built and planned Chains get the same analysis; Pipelines and Plans are deterministic across machines, snapshot-testable and cheap to rebuild.

**ADR-12 — Gradient-set-dependent planning.**
*Decision:* which inputs require gradients may change only the memory strategy, never the forward values, the source sampling or the grids; opt-in exceptions (surrogate backward) are explicit and reported. *Rejected:* re-planning when the gradient set changes. *Consequences:* one Pipeline serves training and evaluation; SOCS with learnable optics is an error, not a switch.

**ADR-13 — Parameter and trace system. Superseded by ADR-32 and ADR-34 (revision 2).**
Revision 1 specified gradix's own probabilistic layer: Param kinds, plates, per-site estimators, counter-keyed scene RNG, a Trace, a path grammar and call options. It is removed, except the small field-path grammar of §6.4. Its measured findings are handed to sampling layers in Appendix F, and the counter hash survives only for explicit keys (ADR-34).

**ADR-14 — Estimator defaults (camera).**
*Decision:* scaled straight-through Poisson in its NaN-free form, labelled exact for losses up to quadratic in counts; exact Gamma EM gain with the double-where + straight-through backward; ST round; bare samplers banned; the camera estimator suite in CI (§6.2, §11.6). Estimators for scene draws (presence, counts, choices, truncations) are the front end's (Appendix F). *Rejected:* plain ST (zero variance-loss gradients). *Consequences:* noise-level gradients are non-zero, finite and correctly signed, and their bias is declared.

**ADR-15 — Default plates. Superseded by ADR-32 (revision 2).**
Per-image sampling semantics belong to the caller (DeepTrack2's per-replicate resolution, once integrated). gradix accepts per-image or shared fields exactly as given (§6.1).

**ADR-16 — Envelopes.**
*Decision:* grids, pads, ROIs and Mie n_max come from an envelope over shape-affecting fields. The envelope is derived from example inputs (×1.25 headroom, bucketed), probed by the front end, or explicit. Pipeline calls compute in-envelope flags without a host sync; CPU-resolved inputs get a synchronous pre-render check; `gx.render` re-plans; values are never clamped (§6.4). *Rejected:* truncation and retained-mass machinery inside gradix (revision 1; now the front end's); clamping; automatic re-planning inside `plan(...)` calls (history-dependent grids, invalidated graphs). *Consequences:* shapes stay static while values move within the envelope; every size in `explain()` names its envelope entry; the front end keeps full control of its distributions.

**ADR-17 — Functional core.**
*Decision:* functional kernels (`ops`); data objects, elements and Chains are frozen dataclass pytrees that store exactly the tensors they are given; element calls are pure; `gx.Pipeline(chain)` precomputes static structure and is a pure function of structure-compatible Chains (or values bound onto its template by field path) and keys; `gx.plan` returns a Plan that wraps a Chain builder and a Pipeline; `capture` provides CUDA graphs. There is no Simulator module and no parameter store: learnable parameters are ordinary tensors owned by the caller, and `gx.tree.parameters` collects them from a pytree. *Rejected:* a stateful Simulator module holding parameters (revision 1); `nn.Module` elements (mutable, and awkward for per-batch values). *Consequences:* kernels, elements and pipelines are gradcheck-able (including forward AD) and graph-capturable; per-batch values are just new pytrees; no mutable-module aliasing [B09 anti-pattern 13].

**ADR-18 — Sparse vs dense render paths.**
*Decision:* both in v1, chosen per population by a deterministic N_max rule (pinnable); ROI from the envelope with a global-spectrum fallback; coherent sparse objects evaluated on pupil support and summed in k-space with dense spectra before one inverse FFT; emitters never voxelised; presence that requires gradients renders every slot (§8.1 for the evidence). *Consequences:* the sparse path is the default for points and particles.

**ADR-19 — Dense interaction-grid sampling.**
*Decision:* the interaction grid comes from the declared scattering bandwidth (n_m/λ with `interaction_band="full"`, as `accurate` sets; otherwise an optical cutoff with a dropped-band and in-band-attenuation report), separate from detection Nyquist; the field is cropped to the pupil and squared on a 2×-band grid; dense Transduce products use their own grid (§4.3). *Rejected:* march and product grids from detection Nyquist. *Consequences:* the accurate tier is correct at low NA, and SIM keeps its super-resolution content.

**ADR-20 — Polarisation.**
*Decision:* the P axis is always present, and every source carries a `Polarization` state (linear, circular, Jones, Stokes; unpolarised light is two incoherent modes) from M0. Coherent paths run at P = 2 (vector Mie, dipole and Born, with the lens rotation and s/p stack transmission) or at P = 1 (the S∥ reduction) from M3, and `polarization="auto"` chooses P = 2 whenever the Chain holds a polarisation-dependent element. Jones pupil optics (polarisers, retarders, compensators, Jones DIC) and thin Jones samples (uniaxial materials in projection, `gx.JonesScreen`) arrive in M4. Vectorial 6-basis sparse emitters with Fresnel t_s/t_p, vector excitation (photoselection) and analyser channels arrive in M4b (or v1.1 under the M6 option). Analysers project the field before it is squared (§5.5a). `psf="auto"` switches at the imaging element's `vectorial_above` NA/n (0.7 by default and at `standard`; 0.45 at `accurate` and for quantitative fits). Vectorial focusing of excitation beams is v1.x, and tensor-ε volumes are v2. Fixed dipoles with a scalar or Gaussian PSF are a validity error (§5.8). *Rejected:* scalar adequacy up to NA/n 0.7 (measured errors of §5.3); analysers as weights on squared components (they cannot express 45° or circular analysers); polarisation optics deferred to v1.x (revision 4's default, overturned by the lead developer on 2026-09-27). *Consequences:* quantitative high-NA outputs are vectorial by default; polarisation-resolved scattering (cross-polarised darkfield, polarisation-resolved iSCAT) works from M3, polarisation microscopy (crossed polarisers, LC-PolScope, Jones DIC) from M4, and polarisation-resolved fluorescence from M4b; the budget grows by 3–4.5 weeks (§12.2).

**ADR-21 — Multi-wavelength grids.**
*Decision:* one spatial grid and one k-grid for all λ with per-λ pupil radii; a λ-common pupil grid in u with per-λ MFTs for sparse emitters; L is a summed axis with power weights; spectra `[B|1, L]` so λ can be per image; separate excitation and emission spectra. *Consequences:* no per-λ dx leak [B02 §2.1].

**ADR-22 — Reflection and iSCAT.**
*Decision:* travel tags in v1; physical iSCAT through the analytic `LayeredMedium` Fresnel reference plus dipole/Mie/Born backscatter multiplied by the interface transmission; the constant-r "iSCAT-lite" as a labelled approximation tier; supercritical collection modelled by the layered rule; near-interface emission rates, orientation effects and the vectorial iPSF in v2, with an info note (§4.1). *Rejected:* a single direction flag; a full S-matrix stack in v1. *Consequences:* correct reference and scatter amplitudes and phases, including the λ/(2n) contrast period.

**ADR-23 — Mixed-sample coupling.**
*Decision:* C0 in v1 with the host-index rule, cut-out-and-fill and proximity/containment warnings; C1 in v1.x, injecting at the particle exit plane z_p + a; C2 and C3′ in v2+ (§5.10). *Rejected:* plain superposition without host index; injection at the particle centre plane. *Consequences:* beads in cells are neither double-counted nor given the wrong relative index.

**ADR-24 — Multislice march and memory.**
*Decision:* the obliquity-corrected march (first Born as its weak limit) is the default; plain BPM is available only as an explicit opt-in (`gx.interact.Multislice(obliquity=False)`). v1 uses procedural slices with non-reentrant checkpointing and pre-rasterised inference slices; the reversible/segmented adjoint with in-loop VJP is v2, triggered by ODT or benchmarks (§5.1, §8.1). *Rejected:* plain BPM as the default (missing obliquity); a v1 custom adjoint. *Consequences:* v1 memory within the §8.3 targets; custom Functions that take callables stay rare.

**ADR-25 — Mie placement and synthesis.**
*Decision:* fp64 coefficients on the GPU, all particles × λ in one captured call, n_max from the radius envelope; S₁/S₂ on a 1-D θ table interpolated onto pupil-support samples; a custom S(θ) backward with `jvp` in M3. *Rejected:* CPU placement (host syncs, no graph capture, value-dependent n_max); per-pixel order sums; a CPU-Mie route for MPS in v1. *Consequences:* §8.1 anchors hold; MPS waits for a trigger.

**ADR-26 — Time and dynamics.**
*Decision:* the frame role and A axis exist from M0, and any data-object or element field may be per frame; frames carry interval and exposure metadata; each frame renders at mid-exposure in v1, with sub-exposure positions for motion blur in v1.x; trajectories, appearance, blinking and bleaching are produced by the front end (§6.7). *Rejected:* processes and photophysics inside gradix (revision 1). *Consequences:* SMLM and tracking sequences reach minimum fidelity in v1 with per-frame values produced by the caller (the cookbook, §10.3; DeepTrack2's `Sequence` once integrated); biology stays upstream.

**ADR-27 — Accelerators and languages.**
*Decision:* pure Python on PyTorch; CUDA graphs first; optional Triton via `torch.library.triton_op` with torch twins (rasteriser first); optional regional compile of real-view steps; nvmath late (§8.4). *Rejected:* Cython, C++/CUDA extensions. *Consequences:* correctness never depends on compile or Triton, which Windows wheels may lack.

**ADR-28 — Output layout and labels.**
*Decision:* native outputs are `[B, (T,) C, H, W]`; callers choose their own layouts; geometry-derived labels come from renderers in the same call or from standalone `gx.labels` functions, and value labels from the caller; tables are padded, with presence and in-FOV masks (§6.6). *Rejected:* echoing input values back as labels; layout flags. *Consequences:* one rule everywhere; image and geometry labels share one frame.

**ADR-29 — DeepTrack2 integration.**
*Decision:* not a v1 deliverable. The integration is designed together with DeepTrack2's revised optics pipelines, which are expected to wrap gradix elements or build Chains, with DT properties supplying field values through the schemas (§10.4). gradix never imports DeepTrack2. The lessons from revisions 1 and 2 (cache invalidation, two-phase evaluation, coordinate mapping, workers with late binding) are kept in Appendix F.8. *Rejected:* an adapter and bridge inside gradix (revision 1); a DeepTrack2 backend contract, spike and joint milestone in v1 (revision 2). *Consequences:* the engine's API is designed for standalone use first; the integration reuses the same elements.

**ADR-30 — Dependencies and licence.**
*Decision:* core torch ≥ 2.9 (tested nightly; the floor rises whenever a needed feature requires a newer torch) and numpy; extras `[triton]`, `[nufft]`, `[deepinv]` (v2) and `[test]`; GPL codes (PyMieDiff, holopy, psfmodels, DECODE) as isolated test oracles only; MIT licence (decided on 2026-09-28); psf-generator (MIT) may be vendored with attribution; MPS best effort and untested in v1; no dependency on DeepTrack2. *Rejected:* pint, pydantic, pyro, tensordict or nvdiffrast in core; image-I/O libraries in core. *Consequences:* a small, stable dependency surface on Windows.

**ADR-31 — Generation mode and output ownership.**
*Context:* tensors created under `inference_mode` cannot be saved for backward, and CUDA-graph outputs are overwritten by the next replay. *Decision:* rendering without gradients runs under `no_grad`; every returned tensor, including those of captured Pipelines, is a caller-owned copy in a fresh allocation; buffer reuse is only an explicit opt-in (§6.5). *Rejected:* automatic `inference_mode`; handing out graph buffers or double-buffered pools. *Consequences:* one device copy per batch; rendered batches can be trained on directly and kept safely.

**ADR-32 — The caller owns parameter sampling; gradix is a functional engine.**
*Context:* revision 1 built a sampling and learnability layer inside gradix, beside DeepTrack2's property system. Two DSLs with drifting semantics was revision 1's top-rated risk. The lead developer decided that DeepTrack2's sampling is the right front end, and that standalone users bring their own sampling. *Decision:* the caller decides what is rendered (sampling, dependencies, replicates, counts, sequences, photophysics, value labels, augmentation, datasets) and owns learnable distributions. gradix takes resolved, batched data objects and elements and returns images, fields and geometry labels as a pure, differentiable function. *Rejected:* a gradix-owned probabilistic layer (revision 1); a sampling syntax mirroring DeepTrack2's (ADR-35). *Consequences:* a smaller engine; standalone users write a few lines of torch (the cookbook, §10.3); learnable distributions need reparameterised sampling in the caller's code (risk 6, Appendix F).

**ADR-33 — Schema-described data objects and elements are the contract.**
*Decision:* every field of every data object and element declares its quantity, role, event shape, whether it affects shapes, and constraints. The schemas are exported as data (`gx.schema_of`), versioned, snapshot-tested and under SemVer. They drive SI conversion, stacking, the canonical layout, envelopes, validation and documentation, and they let any front end (DeepTrack2 later, GUIs, fitting tools) wrap every component, plugins included, without hand-written adapters (§6.1). *Rejected:* untyped dict inputs; per-front-end adapters maintained in gradix. *Consequences:* a new component is documented and introspectable at once; schema changes are contract changes, with review and migration notes.

**ADR-34 — Randomness only through explicit keys.**
*Decision:* gradix owns no RNG. Detector noise, textures and numerical randomness are deterministic functions of explicit keys: an int for batch-keyed draws, or an int64 tensor per image for draws independent of batch composition, through the counter hash. Keyed draws make checkpoint recomputation exact, and keys enter CUDA graphs as device tensors (§6.3). *Rejected:* global or hidden generators; a plan-owned RNG state. *Consequences:* `pipe(chain, key=…)` and `plan(sample, microscope, key=…)` are pure functions; a front end's `Dataset[i]` semantics extend to camera noise; image-keyed EMCCD waits for a keyed Gamma sampler (v1.x).

**ADR-35 — No sampling layer or syntax inside gradix.**
*Decision:* standalone users randomise with plain torch, and the documentation ships a tested sampling cookbook (§10.3). DeepTrack2 users will get DeepTrack2's syntax through its integration. If a DeepTrack2-free syntax is ever needed, DeepTrack2's property core is extracted into a shared package rather than re-implemented (§10.5). *Rejected:* a lightweight lambda resolver in gradix that mimics DeepTrack2. *Consequences:* one sampling DSL in the ecosystem; a small, stable engine API.

**ADR-36 — Layered public API: building blocks first.**
*Context:* the lead developer asked for an engine that does not force callers into `Microscope` and `Sample` containers. Containers are acceptable only if they are modular and constructed from lower-level objects. *Decision:*
- **L2 building blocks** (data objects, carriers, elements) are public, self-sufficient and callable directly.
- **L3** `gx.Chain` composes elements along the slot skeleton, and `gx.Pipeline` adds static grids, validity, gradients, memory planning, keys, outputs and capture for any Chain.
- **L4** `gx.plan`, `gx.Sample`, `gx.Microscope` and presets only construct L2 and L3 objects, and `plan.chain()` shows the Chain built.

Carriers are the contract between elements. Every level is tested for equivalence with the others (§11.10). *Rejected:* a closed internal pipeline reachable only through a planner (revisions 1 and 2); a general element DAG in v1 (ordered optical stages suffice, ADR-39). *Consequences:* power users and future front ends compose elements freely; the planner is convenience, not a gate; more public surface to keep stable (risk 3).

**ADR-37 — Standalone project; DeepTrack2 later; no DeepTrack2-parity preset.**
*Decision:* gradix is its own repository (on the lead developer's GitHub for now), useful without DeepTrack2 and with no dependency either way. DeepTrack2 integration follows DeepTrack2's optics-pipeline revision (ADR-29). The `legacy_dt` preset of revisions 1 and 2 is dropped. A documentation page lists the deliberate differences from DeepTrack2 2.0.2 and their physical reasons. Close-to-legacy behaviour remains available by explicit opt-in: plain BPM, hard masks and additive Δn. *Rejected:* golden-parity gating against DeepTrack2 2.0.2 numerics. *Consequences:* ≈1.5–2 weeks saved; no hidden legacy numerics in the engine; migration questions are answered by documentation, not by a preset.

**ADR-38 — Feeding data to built Pipelines.**
*Context:* fits and datasets need per-call data in a Pipeline built once: more images than one batch, object counts that vary per image, settings recorded per image, and per-image values to refine. *Decision:* a Pipeline call takes a structure-compatible Chain or a mapping from field paths to values, bound onto the Pipeline's template (`pipe.bind`). Declared inputs are planned at their widest role. Batch size and slot counts are capacities: eager calls run at the given sizes, and captured graphs pad. `gx.pad` turns ragged per-image lists into padded tables with presence, and `gx.envelope_of` derives envelopes from whole datasets. Per-image parameter tables and their optimisers stay the caller's (§6.9, Appendix F.5). *Rejected:* a gradix dataset or parameter-table class (ADR-32); re-planning per batch; exact-size Pipelines that reject a ragged last batch; implicitly slicing template tensors to a batch. *Consequences:* one Pipeline serves a whole dataset, including its ragged last batch; fits over thousands of images need no special API; a binding costs one dataclass copy per call.

**ADR-39 — Optical stages and programmable optics.**
*Context:* optical neural networks, programmable illumination and diffractive front ends shape light at planes other than the pupil, on both sides of the sample, and are either fixed after training or trained with the rest of the system (lead developer, 2026-09-27). *Decision:* a Chain has ordered `illumination_optics` and `detection_optics` stages of stage elements (`Mask`, `Propagate`, `Lens`, `FourierTransform`, `Aperture`, `DiffractiveStack`). Detection stages act on the sampled field inside a field stop, in image-space coordinates. Shaped illumination is a coherent group with a sampled pupil function, read as a sampled field by dense producers and as J plane waves, or local plane waves, by sparse ones. Physical devices are data objects that produce transmission maps with their non-idealities (`PhaseSLM` and `HeightMap` in 1.0). Stage planes have their own sampling rules. The extended device library, nonlinear layers and incoherent multi-plane stacks are v1.x unless Q22 pulls them forward. *Rejected:* a general element DAG (ordered stages cover the use cases; Q20); ONN layers as pupil modifiers only (they cannot express field planes or cascades); a separate ONN framework beside the engine. *Consequences:* ONNs inherit batching, envelopes, gradients, keys, capture and the three levels; one differentiable twin serves design, joint training and physics-aware training; the budget grows by 2.5–3.5 weeks (§12.2).

**ADR-40 — Code quality and documentation standard.**
*Context:* the lead developer asked for ruff and ty compliance, for numpydoc documentation at module, class, method and function level, and for a narrow set of tutorial and example notebooks that are kept current and help find awkward syntax (2026-09-28). *Decision:* ruff lints every maintained file (including the annotation and numpydoc docstring rules) and formats it, including the Python examples of this document. ty checks for every platform, with warnings as errors, and `numpydoc lint` checks docstring content from `pytest`. The package ships `py.typed`, and `docs/background/` stays excluded as provenance. Docstrings are numpydoc, with units and shapes for every tensor, and the docstrings of data objects and elements are checked against their schemas. The documentation is five tutorial and seven example notebooks with a fixed scope, executed in CI, plus explanations and reference. Each notebook's awkward lines become `api-friction` issues, which the milestone's API review triages. *Rejected:* sphinx-gallery scripts (notebooks were asked for); separate how-to guides (scope; their topics live in the notebooks); mypy or pyright (ty matches the uv and ruff toolchain); Google-style docstrings. *Consequences:* documentation, types and schemas cannot drift silently; every API change pays its documentation cost at once; ty is pre-1.0, so the lockfile pins it and upgrades are deliberate (risk 17); the budget grows by 1–2 weeks (§12.2).

**ADR-41 — Consumers choose the density grid.**
*Context:* dense emission is imaged on a grid set by the camera, the magnification and the oversampling (the strata fine grid, pitch/s), and later dense elements have grids of their own (multislice slices, M4). Shapes, labelings and placed or rotated volumes cannot know that grid, and a voxel volume given on another grid would need resampling (2026-09-29). *Decision:* the consuming imaging element declares the density grid it wants when it is configured, and lowerings rasterise shapes and labelings, or resample volumes, onto that grid. DeepTrack2 solved the same problem this way, and it works well in practice. *Rejected:* grids owned by the objects (every consumer would resample, twice when grids differ); one global grid (wrong for elements with different pitches). *Consequences:* the soft rasteriser, `Labeling` and `Voxels(position=)` (cap-06) share one lowering protocol; densities already on the grid are used as they are.

**ADR-42 — User names and framework names never share a namespace.**
*Context:* names users choose travel through the engine next to names gradix reserves: population and stage names start logical paths (`beads.position`), output names become keys and attributes (`out.pos`), shape fields reached capability methods as keyword arguments (a Gaussian's `sigma` collided with the blur argument), and fidelity knobs were written into any element field that shared their name. Each collision would force users into naming conventions that grow as objects are added (lead developer, 2026-09-29). *Decision:* framework code never merges user names into a namespace of its own: shape fields travel as one mapping, and fidelity knobs reach only the fields an element declares (`fidelity_knobs`). Where a user name must enter a namespace with reserved words, it is checked once, where it is introduced, by one rule (`check_name`): a Python identifier that is not a keyword, and not a reserved word of that namespace. The reserved lists are short and fixed: the Chain's parts and groups for population and stage names; `Output`'s own attributes (`keys`, `values`, `items`, `get`, `meta`) for output names. Mapping keys in schemas follow the same rule, and user geometries (`@gx.geometry`) are checked against the object-set base fields when they are defined. *Rejected:* documenting the collisions (silent wrong results); prefixing every user name (noise in every path). *Consequences:* a name either works everywhere or fails at once with the list of reserved words; new object kinds and plugins add no naming rules.

**ADR-43 — Holography and Mie optics pulled forward (M3a).**
*Context:* the lead developer needs interferometric characterisation of nanobeads now: Mie optics; in-line, off-axis, iSCAT and QWLSI microscopes with the camera's view; differentiable field reconstructions; and illumination designed for the bound on radius and refractive index through the reconstruction (2026-09-29). *Decision:* a milestone M3a follows M1 and precedes M2. It takes from M3 the Mie and Mie-dressed dipole producers, coherent Chains and Pipelines, the pupil modifiers on the coherent path, the references (in-line, reflected, image-side), the Fresnel reference and backscatter in a layered medium, shaped illumination for sparse producers with `PhaseSLM`, `gx.out.Field` and `labels.Phase`, and the InlineHolography, OffAxisHolography and ISCAT presets. It adds QWLSI (ADR-45), reconstructions and the reconstruction-aware bound (ADR-44). Vector producers (P = 2) are its last phase. Born, Köhler quadrature, Gaussian beams, the sampled stage elements, `HeightMap`, region readout and CUDA graphs stay in M3. *Rejected:* waiting for M3 (the application is now); a holography side project outside the engine (it would bypass the carriers, the Pipeline and the bound). *Consequences:* the coherent carriers and the coherent execution of Chains meet real use before the M2 freeze, which then covers them; M3 shrinks by ≈5–6 weeks, and the plan grows by ≈2–3 weeks for QWLSI, the reconstructions and the bound (§12.2).

**ADR-44 — Reconstructions are differentiable operators, and a bound measures what they keep.**
*Context:* an interferometer records intensities, and a reconstruction recovers the field while discarding information (sideband windows, harmonic windows, crops). The lead developer wants to shape light so that the reconstruction keeps the information about a sphere's radius and index (2026-09-29). *Decision:* `gx.recon` operators are pure torch maps from frames to reconstructed data, linear up to an optional invertible pointwise step, for measured and rendered frames alike. `gx.crlb(..., reconstruction=R)` returns the Fisher information of R's output under the camera noise propagated through R: F_R = ‖ΠWJ‖² (§5.12), in the Gaussian approximation with real and imaginary parts stacked, computed by conjugate gradients on matrix-free products of R and its adjoint, and differentiated through a form that is stationary in the solver's output. *Rejected:* bounds on raw frames only (they ignore what the instrument's processing loses); treating a reconstructed field as white circular Gaussian (wrong for real-valued frames and for spatially varying variance); differentiating through the unrolled solver (memory and accuracy). *Consequences:* one function bounds raw data (no R) and any linear processing; design objectives target the data the instrument actually uses; the approximations are reported. *As built (§5.12):* CGLS in whitened pixel space replaces conjugate gradients on RΣRᵀ, which lost precision when shaped light left pixels nearly dark; the stationary form is evaluated in pixel space, and the solve runs in the basis that whitens the raw information.

**ADR-45 — Shearing interferometry as an analytic grating stage.**
*Context:* QWLSI places a 2-D grating a short distance before the camera. Each diffraction order carries a copy of the image field, sheared by λdG and tilted by G, and the camera records their interference. Sampling the field at the grating and propagating it would wrap the analytic background at the frame's edges, and would need a grid fine enough for the grating's angles. *Decision:* `gx.optics.Grating(period, distance, kind | orders)` expands a periodic detection-side mask into its diffraction orders and forms each order analytically: the shift as a linear phase on the pupil samples and a translation of every background plane wave, the tilt as a carrier on the camera grid, and the propagation over d as a pupil phase. It is exact in the pupil representation and paraxial only on the image side (NA/Mag ≲ 0.03). *Rejected:* a sampled mask followed by BL-ASM (edge seams, a finer grid, and no accuracy gained for periodic masks); a phenomenological gradient model (it loses the camera's view and the noise statistics). *Consequences:* QWLSI costs one MFT per order; the same stage covers 1-D shearing gratings and Talbot-type sensors; general masks at field planes remain the sampled `Mask` of M3. *As built (§5.12):* the orders travel as a `DiffractionOrders` carrier; each order's phase is the exact image-side angular spectrum rather than its paraxial form; by default the camera is the plane in focus.

---

## 14. Risks, open questions and deferred items

### 14.1 Risks

| # | Risk | L / I | Mitigation |
|---|---|---|---|
| 1 | Scope creep from the briefs' long wish lists | H / H | the §2.3 "not building" table; every feature names a trigger and an exit criterion; milestone gates; the audited delta tables of §12.2 |
| 2 | Dense-march precision and tilt handling | M / M | residual demodulation inside `ops.propagate`, so plugins carry no background obligation; the scattered-field form selected by rule for weak or darkfield samples; tilt-equivariance conformance (§4.4, §7.4) |
| 3 | **API churn before external users.** Three public levels mean more surface to get right | M / H | level-equivalence tests from M0; API reviews with prospective users at M0 (week 5) and M2; element protocols frozen only after they are exercised (principle 3); SemVer from 1.0; the conformance suite; notebooks executed in CI, which show the cost of every API change and collect friction issues (§12.7) |
| 4 | Validity thresholds are literature rules until ladder tables exist | H / M | M3/M4 exits require tables; reports mark `literature` vs `calibrated`; the review already calibrated born_ff, dipole, scalar-vs-vectorial and Gaussian-sprite limits |
| 5 | Mie numerics and backward speed (launch-bound fp64 recurrences; backward 4× forward) | M / H | downward D_n; miepython oracle across x and absorbing m; custom S(θ) backward; CUDA graphs; all particles in one call |
| 6 | **Standalone users must write their own sampling**, including reparameterised draws when distributions are learnable | M / M | the tested cookbook (§10.3); the measured recommendations of Appendix F; examples that sample in plain torch (§9); the render side proven by a reference sampler in gradix's tests |
| 7 | Identifiability in sim-to-real: distribution widths are weakly identified (σ −9 % in [M-D Exp C]), and photometric nuisances get absorbed into scene parameters | H / M | gain, offset and illumination on the camera and light elements; `gx.detect.to_unit` for real frames; background-normalised features (caller); experimental validation (§11.14) |
| 8 | Memory of learnable volumes (ODT, neural fields) | M / H | batch-1 volumes, angle mini-batching through acquisition subsets, checkpointing; Pipelines refuse infeasible memory configurations with advice; adjoint in v2 |
| 9 | **Sampling cost in Python front ends.** Per-sample loops (19 ms per batch, ≈0.3 ms per image in a DeepTrack2-style graph) can dominate fast renders | M / M | batched torch sampling in the cookbook (0.17–0.3 ms per batch [M-D]); CPU workers with late binding (§10.3); `Pipeline.capture` |
| 10 | Convention bugs (sign, apodisation, conj, layered media, pixel conventions) | M / H | `conventions.py` frozen in M0 with sign tests; power, layered collection-efficiency, direct-summation and optical-theorem tests; the evanescent plan-time assertion; coordinate-convention tests (§11.10) |
| 11 | Windows throughput without Triton | M / M | sparse-first design and graphs meet the targets without compile (measured); Triton opt-in |
| 12 | Overly conservative envelopes (large pads from generous headroom or loose probes) | M / L | the Pipeline names the envelope entry behind each size and its source; explicit entries override; derived headroom is ×1.25, not physical domains |
| 13 | Thick-sample partial coherence cost (M × n_slices slice-steps) | H / L | cost warnings; projection at `standard`; C1/SOCS later |
| 14 | torch release cadence (6–8 weeks) | M / M | minimal dependencies; CI on the latest two torch minors plus a nightly floor job; private APIs limited to pytree registration and `torch._standard_gamma(generator=…)`, each guarded by a version test |
| 15 | Schedule overrun | M / H | the audited 38–52-week budget (§12.2); the M6 option; a hardening-focused M5; exit criteria that name only delivered APIs |
| 16 | ONN transfer to hardware limited by device models (response curves, crosstalk, fabrication, alignment) | M / H | per-image sampling of fabrication and alignment errors; device calibration by system identification; physics-aware training (§5.11); E8 |
| 17 | ty is pre-1.0: its rules and inference change between releases | M / L | `uv.lock` pins the version; upgrades are deliberate PRs that fix or explicitly silence new diagnostics; ruff and the tests do not depend on ty |

### 14.2 Open questions (for the lead developer)

| # | Question | Decide by | Blocks | Owner | Default if undecided |
|---|---|---|---|---|---|
| Q16 | **API review of the three levels** with two prospective users: do the §3 and §9 sketches (element signatures, Chain fields, Pipeline calling convention) fit their workflows? | week 5 (M0 freeze) | schema and carrier freeze | lead dev | the §3 and §9 sketches |
| Q11 | **Rank-7 rigidity.** ISM detector arrays and sub-exposure axes | adopted in M0 | — | Developer B | flatten into A or M with metadata; revisit if more than two such cases appear |
| Q15 | **Vectorial optics in 1.0 (M4b) or 1.1 (M6 option)?** | start of M3 | M4b staffing, the 1.0 date | lead dev | M4b (in 1.0) |
| Q22 | **Optical neural networks in 1.0.** The ONN core (optical stages, shaped illumination, `PhaseSLM` and `HeightMap`, region readout; ≈2.5–3.5 weeks) is in 1.0. Keep it there, or ship it as a v1.1 milestone right after 1.0, with the stage contract kept in 1.0? And should the extended toolkit (DMD, deformable and segmented mirrors, metasurfaces with surrogates, nonlinear layers, incoherent multi-plane stacks, a physics-aware-training helper; ≈3–4 weeks, cap-39) join 1.0? | start of M3 | M3/M4 scope, the 1.0 date | lead dev | the core in 1.0; the extended toolkit in v1.x |
| Q23 | **Vector producers (P = 2) in M3a.** High-NA iSCAT and quantitative bounds need them; the scalar S∥ pupil field is 3–12 % off at NA 0.8–1.2 (§4.1) | M3a phase 6 | Ex11's final numbers | lead dev | **settled 2026-09-29:** P = 2 within M3a, after the four modalities run at P = 1 and before Ex11's final numbers |
| Q24 | **iSCAT geometry.** Spheres in water above the coverslip under the layered rule (supercritical collection, Fresnel reference), or a homogeneous iSCAT-lite first? | M3a phase 3 | the iSCAT exit criteria | lead dev | **settled 2026-09-29:** the full layered geometry; iSCAT-lite stays the preset's `interface=None` (as built: a homogeneous Sample with an on-axis `ReferenceBeam`, since the medium belongs to the Sample) |
| Q25 | **Design variables of the light-shaping study.** Illumination-pupil phase (a SLM, J plane waves), illumination-angle sequences, detection pupil masks, the reference's amplitude, tilt and attenuation | M3a phase 5 | Ex11 | lead dev | **settled 2026-09-29:** detection pupil and reference first (already differentiable and cheap), then the illumination pupil |
| Q26 | **Bead sizes.** Are the 50–250 nm diameters or radii? | M3a phase 5 | Ex11's envelopes | lead dev | **settled 2026-09-29:** a rough operating range, not a specification; each modality's example uses the sizes it resolves above noise (iSCAT reaches far smaller beads than off-axis holography) |
| Q6 | **sCMOS likelihood at low counts:** shifted Poisson, Gaussian with variance μ + σ², or a truncated convolution? Decided by the benchmark of §11.6: accuracy of the log-likelihood and of its gradients against the exact Poisson ⊛ Gaussian, CRLB fidelity and speed, on synthetic counts and the E2 calibration data | M1 exit | `log_prob` default | Developer B | **settled 2026-09-29:** the Gaussian (§11.6); the convolution for calibration |
| Q9 | **Per-instance routing (v1.x):** evaluate both elements under masks, or split populations by predicate? | v1.x planning | cap-29 | — | — |
| Q13 | **C0 → C1 escalation defaults** at `accurate` (needs C0/C1 ladder data) | v1.x planning | cap-22 | — | — |

**Settled by the lead developer on 2026-09-25:**
- *where the engine lives*: its own repository, on the lead developer's GitHub for now;
- *whether gradix ships a sampling layer or syntax*: no (ADR-35);
- *whether DeepTrack2 parity is a goal*: no (ADR-37).

Revision 2's Q1, Q14 and Q18 are closed by these decisions. Q2, Q7, Q8 and Q12 moved to the caller's side with the sampling layer (Appendix F).

**Settled on 2026-09-27:**
- *polarisation in 1.0* (Q21): yes. Coherent vector producers arrive in M3, and Jones optics with thin Jones samples in M4 (ADR-20);
- *Chain topology* (Q20): ordered optical stages before and after the sample, and no general DAG in v1 (ADR-39).

**Settled on 2026-09-28:**
- *licence* (Q4): MIT;
- *torch floor* (Q5): ≥ 2.9, raised whenever a needed feature requires a newer torch;
- *learnable magnification and pixel size in 1.0* (Q3): yes; cap-26 moves to M2;
- *analytic backgrounds* (Q10): Gaussian beams in 1.0 (M3); diverging references are an M4 stretch goal, else 1.1;
- *image-keyed EMCCD* (Q17): v1.x, batch-keyed until then;
- *the sCMOS likelihood* (Q6) stays open, but its method is settled: a benchmark that weighs gradients as well as accuracy decides it at M1 exit;
- *the name* (Q19): **gradix** (grad + optix), imported as `gx`. `abbe` was the runner-up: it honours Ernst Abbe, but it says nothing about gradients and suggests that the engine is limited to Abbe-style simulation. `gradix` was free on PyPI and conda-forge on 2026-09-28; the GitHub account `gradix` is taken, so the repository lives under the lead developer's account.

**Settled on 2026-09-29** (after M0 and most of M1 were built):
- *the sCMOS likelihood* (Q6): the Gaussian with variance μ + σ², chosen by the §11.6 benchmark; the truncated convolution serves calibration, and high photon counts get an ad-hoc treatment if one is needed;
- *keyed noise*: fast approximate batch-keyed Poisson draws now (`torch.poisson`), and a keyed sampler that is exact and as fast before 1.0 (M5, §6.3);
- *CI*: GitHub-hosted runners only, with no self-hosted GPU runner; GPU benchmarks and full-size checks run locally before each release (§11.15).

### 14.3 Deliberately deferred

- **v1.1 (M6 option only):** vectorial sparse emitters and per-object pupils.
- **With DeepTrack2's optics-pipeline revision:** the integration (§10.4, Appendix F.8).
- **v1.x:**
  - C1 injection; vectorial focusing of excitation beams; motion blur; confocal/ISM; 3D-SIM;
  - per-instance routing and `escalate`;
  - SOCS/WOTF, snap-and-shift and keyed stochastic source nodes;
  - spline PSF import; QCMOS; image-keyed EMCCD;
  - the extended ONN toolkit (cap-39), unless Q22 pulls it into 1.0;
  - entry-point plugins; surrogate gradients; `Pipeline.graph`/`profile`; element DAGs if triggered;
  - MPS if triggered.
- **v2:**
  - SSNP, slice Born/Rytov and custom adjoints;
  - interface dipoles (LDOS, orientation) and the vectorial iPSF;
  - low-rank dense space variance and tissue light sheets;
  - tensor ε; Foldy–Lax; meshes; deepinv export;
  - an MBS reference; nonlinear source maps.
- **Later/plugins:** T-matrix import; TorchGDM interop; learned surrogate tiers; eikonal screens; MCX backgrounds; multi-GPU tiling.
- **The caller's side (not gradix's):** sampling layers, reparameterised distributions, estimators for discrete draws, photophysics models, distribution-matching toolkits (Appendix F).

---

## Appendix A — Glossary and notation

### A.1 Notation

| Symbol | Meaning |
|---|---|
| B A M J L P N K Y X C | tensor axes only (§4.2): batch, acquisition, incoherent modes, coherent waves in a mode, wavelength bins, polarisation, object slots, pupil-support samples, space/frequency, detector channels |
| T | frames (the time part of A; `[B, (T,) C, H, W]` outputs) |
| S, Z | species and depth planes of `EmitterDensity.data [B, A\|1, S, Z, Y, X]` |
| H, W | output image height and width (`[B, (T,) C, H, W]`); in formulas H is also the propagation kernel and W the cross-spectral density or the channel-weight matrix W[C, L, P] |
| R | side length of the R×R ROI of a sparse emitter (`PatchGrid`) |
| n_slices | number of slices of a multislice march |
| n_pts | points evaluated per object in `sdf(x)` |
| n_steps | exposure sub-steps (motion blur, v1.x) |
| n_src | number of source quadrature nodes (`Quadrature(n=…)`) |
| n_mat | number of materials |
| n_part | number of particles |
| n_beads, n_z, n_crops | bead count, focal planes, crop count in examples |
| Mag | magnification |
| σ_coh = NA_ill/NA_obj | partial-coherence factor |
| σ_r | raster prefilter width, in units of the voxel size h |
| x = 2π·n_m·a/λ₀, m = n_p/n_host | Mie size parameter and relative index |
| u = n·sinθ(cosφ, sinφ) | direction coordinate (dimensionless, λ-independent) |
| k_z,s, k_z,i | axial wavenumber in the sample and in the immersion medium |
| φ_max = k₀·Δn_max·d_max | maximum phase of a compact object (born_ff validity) |
| t(u), t_s, t_p | stack amplitude transmission (scalar; s- and p-polarised) |
| cap-00 … cap-37 | capability IDs (§2.1) |
| M0 … M6 | milestones (§12.3) |
| L0 … L4 | foundations and the public API levels (§3.1) |
| [Bxx], [A §x]…[D §x], [M-·], [Rev] | evidence tags (see the header) |

### A.2 Glossary

| Term | Meaning here |
|---|---|
| Abbe (method) | Partially coherent imaging as an incoherent sum over source points, each imaged coherently |
| AcqIndex / A axis | Flattened acquisition multi-index (frames, focus steps, SIM angle × phase, LED index); each entry is a separate output frame |
| Analytic background | The unscattered, reference and reflected light carried as `PlaneWaves`, filtered by evaluation, never sampled |
| Angular spectrum | Plane-wave decomposition A(k⊥) of a field on a plane; the coherent interchange currency (object side, before the pupil) |
| Approximation edge | A declared "lower implementation → higher implementation" limit with a regime and a tolerance table |
| Band-limited occupancy | Gaussian-prefiltered (σ_r ≈ 0.3h) voxelisation with exact geometry gradients; the prefilter is an in-band blur, tagged approximate (§4.6) |
| BL-ASM | Band-limited angular-spectrum method: free-space propagation with the evanescent band removed |
| Born / Rytov | First-order weak-scattering approximations of the field (Born) or of its complex phase (Rytov) |
| BPM / multislice | Beam-propagation method: alternate thin phase screens and free-space propagation; v1 uses the obliquity-corrected march |
| C0 / C1 / C2 / C3′ | Coupling policies: superposition / injection into the march / coupled iteration / dipoles inside a full-wave volume |
| Capacity | The largest batch size or slot count a Pipeline accepts; smaller calls run within it (§6.9) |
| Carrier | A typed tensor structure passed between elements (`PlaneWaves`, `Field`, `ObjectSpectra`, `EmitterSet`, `EmitterDensity`, `Irradiance`) |
| Chain | An L3 composition of elements along the slot skeleton (light, scatterers, emitters, excite, imaging, references, camera, background, acquisition, environment) |
| Checkpointing (non-reentrant) | Discarding activations in forward and recomputing them in backward (`use_reentrant=False`); no stateful-generator draws inside |
| COBRI | Coherent brightfield: the transmission analogue of iSCAT |
| Collapse theorems | (a) shift invariance → per-z OTF convolution; (b) thin sample → Hopkins TCC/SOCS; weak object → WOTF |
| Consistency contract | Identical radiometry, frames, signs, pixel integration and label geometry across fidelity tiers |
| Container | An optional L4 grouping of L2 objects: `gx.Sample` (object sets + environment) or `gx.Microscope` (light, objective, camera, …) |
| CRLB | Cramér–Rao lower bound: the inverse Fisher information, the best achievable estimator variance |
| Data object | An L2 description of matter or light as tensors: object sets, materials, labelings, media, spectra |
| Declared input | A field path that a Pipeline's calls may bind, planned at the widest role its schema allows (§6.9) |
| Detect | The slot that reduces modes, integrates pixels and applies the camera model |
| Device | A data object modelling a programmable or fabricated optical element (SLM, height map, …) that produces a transmission map with its non-idealities (§5.11) |
| Diffractive network (D²NN) | Cascaded thin optical layers separated by free-space propagation, trained so that the output intensity performs a task; an optical neural network (§5.11) |
| DOF | Depth of field n·λ₀/NA² (NA → min(NA, n_s)); the projection validity criterion |
| Element | A public L2 operator with one physical job (source, interaction, transduction, imaging, detection), typed by the carriers it consumes and produces |
| Emit | The slot that turns emitter photons into irradiance on the detection grid |
| Envelope | Static intervals for every shape-affecting field, from which grids, pads, ROIs and Mie n_max are derived (§6.4) |
| Ewald cap | The spherical shell \|k\| = k_m on which Born scattering amplitudes live |
| Field path | `<population>.<field>` or `<part>.<field>` (a Chain field such as `light` or `camera`, or `objective`), used by envelopes, overrides, labels and `explain()` (§6.4) |
| Form factor | Analytic Fourier transform of a primitive's indicator function |
| FPM | Fourier ptychographic microscopy (LED-array illumination, synthetic aperture) |
| Front end | The code that decides what gets rendered and owns sampling: the user's own sampler, or DeepTrack2 once integrated (ADR-32) |
| Gibson–Lanni | Model of the depth-dependent aberration of a layered immersion/coverslip/sample stack; one term with defocus (§4.1) |
| H_exp | The exponent form of the propagation kernel, H = exp(Δz·H_exp), so gradients reach Δz |
| Interact | The slot where light meets the sample (Mie, dipole, Born, projection, multislice, …) |
| iPSF | Interferometric PSF of iSCAT, including interface effects |
| iSCAT | Interferometric scattering microscopy: backscatter interfering with a reflected reference |
| Jones pupil | A 2×2 pupil matrix acting on transverse polarisation (P = 2) |
| Key | An explicit integer (batch-keyed), an int64 tensor per image (image-keyed) or a per-object `key` field (textures) from which all randomness inside rendering is derived (§6.3) |
| Köhler illumination | Condenser pupil imaged to infinity; a spatially incoherent source S(u, λ) |
| Lowering / view | Conversion of continuous components into the representation a solver consumes |
| MBS | Modified Born series: a convergent iterative full-wave solver (v2 reference) |
| MFT / CZT | Matrix Fourier transform / chirp-z transform: DFTs onto arbitrary output grids |
| Mie | Exact scattering by a (layered) sphere; S₁/S₂ amplitude functions |
| MMD | Maximum mean discrepancy: a kernel distance between feature distributions |
| Mode (M) | A mutually incoherent coherent field component, summed in intensity within one exposure |
| N (slots) | Static number of object slots of a population, bucketed by `gx.stack` (8, 16, 32, …) |
| Objective (slot) | The fused pupil multiplier plus the basis change onto the detection grid |
| Obliquity correction | Filtering the multislice source term by k_m/k_z so that the march's weak limit is first Born |
| ODT / IDT / QPI | Optical / intensity diffraction tomography; quantitative phase imaging |
| On-support single mode | Draft source sampling whose mode(s) lie on the condenser support |
| OPD / OPL | Optical path difference / length (µm) |
| Optical stage | An ordered list of stage elements (masks, propagation, lenses, apertures) acting before the sample (`illumination_optics`) or after the Objective (`detection_optics`) (§5.11) |
| Physics-aware training | Training hardware with its measured output in the forward pass and the simulated twin's gradient in the backward pass: y = y_hw.detach() + y_sim − y_sim.detach() (§5.11) |
| Pipeline | An immutable L3 object built from a Chain: static grids from an envelope, validity, gradient table and memory strategy; a pure function of structure-compatible Chains, or of values bound onto its template by field path, and keys |
| Pixel MTF | The pixel's transfer function p²·sinc(f_x p)·sinc(f_y p), applied before sampling at pixel centres |
| Plan | An L4 object returned by `gx.plan`: a rule-chosen Chain builder plus the Pipeline around it |
| Plan time | When a Pipeline is built and analyses its Chain, or an element configures itself for an eager call; plan-time errors and assertions fire then |
| PolScope | LC-PolScope: polarised-light microscopy with a liquid-crystal universal compensator and a circular analyser; frames at a few compensator states give retardance and slow-axis maps |
| Population | A named object set: a key of a Chain's `scatterers` or `emitters`, or of a `Sample` (struct of arrays over `[B, T\|1, N]` with presence) |
| Presence | Per-slot weight in [0, 1]; contributions scale linearly with it (§6.1) |
| PSF / OTF | Point-spread function / optical transfer function |
| Pupil support (K) | The frequency samples with \|f\| ≤ NA/λ at which sparse coherent producers are evaluated |
| Residual demodulation | Marching a tilted field as an exact on-grid carrier times a small analytic residual tilt (§4.4) |
| Retardance | Phase difference δ between the slow and fast eigen-polarisations of a birefringent sample; with the slow-axis angle θ it forms the retardance vector (δcos2θ, δsin2θ) |
| Richards–Wolf | Vectorial diffraction theory of high-NA focusing and collection |
| ROI | Region of interest: the R×R patch onto which a sparse emitter is rendered |
| SAF | Supercritical-angle fluorescence: emission collected beyond the critical angle through evanescent coupling |
| SamplingPlan | A Pipeline's record (P5) of every discretisation decision with its criterion and the envelope entry that drove it |
| Scaled straight-through | Poisson estimator: exact integer samples forward, gradient (n + λ)/(2λ) backward; exact for losses up to quadratic in counts |
| Schema | The per-field declaration of quantity, role, event shape, shape-affecting flag and constraints; exported for any front end (§6.1, ADR-33) |
| SDF | Signed distance function of a shape |
| Shaped illumination | A coherent illumination group with a sampled pupil function, from illumination stages or `gx.light.Shaped` (§4.5, §5.11) |
| SIM | Structured-illumination microscopy |
| Slot | One step of the fixed render chain (Source, Interact, Objective, Transduce, Emit, Detect); optical stages sit between steps (§5.11) |
| SMLM | Single-molecule localisation microscopy |
| SOCS / TCC / WOTF | Sum of coherent systems / transmission cross-coefficient / weak-object transfer function |
| SSNP | Split-step non-paraxial method (v2 march) |
| Strata OTF | Per-z-slice optical transfer functions for dense emission (collapse theorem a) |
| Structure signature | The static part of the inputs (kinds, names, N, T, role patterns, modifiers, camera shape, …) that keys a Pipeline's captured graphs and `gx.render`'s plan cache (§6.1) |
| Taps | Named intermediate outputs (`expected`, `image_field`, `exit_field`, …) |
| Template | The Chain a Pipeline was built from; `pipe(values)` binds values onto it by field path (§6.9) |
| TF32 | TensorFloat-32: reduced-precision fp32 matmul on NVIDIA GPUs, disabled around MFTs |
| Thin Jones sample | A thin anisotropic sample represented as a spatially varying 2×2 Jones transmission (isotropic phase and absorption, retardance, slow axis), from uniaxial materials or `gx.JonesScreen` (§4.6) |
| TIRF | Total internal reflection fluorescence: evanescent excitation |
| Transduce | The slot that turns excitation light evaluated at fluorophore positions into emitted photons per emitter (`EmitterSet`), or on the product grid into photon densities (`EmitterDensity`) |
| Travel tag | `travel = +1` (toward the detection objective) or −1 on every coherent contribution |
| Wirtinger gradient | The complex-derivative convention; PyTorch returns conjugate-Wirtinger gradients |
| Zernike (ANSI) | Orthogonal pupil polynomials with OSA/ANSI indexing and RMS normalisation, in µm of OPD |

## Appendix B — Prior art (from the research briefs)

- **DeepTrack2**: https://github.com/DeepTrackAI/DeepTrack2 (optics: `deeptrack/optical/optics.py`, develop branch); deeplay: https://github.com/DeepTrackAI/deeplay; issue #445 (contrast signs): https://github.com/DeepTrackAI/DeepTrack2/issues/445
- **Chromatix** (JAX): https://github.com/chromatix-team/chromatix; Nat. Methods 2026: https://www.nature.com/articles/s41592-026-03121-x
- **TorchOptics**: https://github.com/MatthewFilipovich/torchoptics; arXiv 2411.18591
- **waveorder**: https://github.com/mehta-lab/waveorder; https://arxiv.org/abs/2412.09775
- **microsim**: https://github.com/tlambert03/microsim; stages: https://talleylambert.com/microsim/stages/
- **psf-generator** (MIT, PyTorch): https://github.com/Biomedical-Imaging-Group/psf_generator; https://arxiv.org/abs/2502.03170
- **uiPSF**: https://github.com/ries-lab/uiPSF; https://www.nature.com/articles/s41592-024-02282-x
- **DECODE**: https://github.com/TuragaLab/DECODE; **DeepSTORM3D**: https://github.com/EliasNehme/DeepSTORM3D; **FD-DeepLoc**: https://github.com/Li-Lab-SUSTech/FD-DeepLoc
- **holopy** (theories): https://holopy.readthedocs.io/en/latest/users/theories.html; **miepython**: https://miepython.readthedocs.io/; **PyMieDiff** (GPL): https://arxiv.org/abs/2512.08614
- **SSNP**: https://arxiv.org/abs/2207.06532, https://github.com/bu-cisl/SSNP-IDT; **multi-layer Born**: https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-5-394; **modified Born series**: https://arxiv.org/abs/1601.05997; **wavesim**: https://github.com/IvoVellekoop/wavesim_py
- **biobeam**: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006079
- **TorchGDM**: https://arxiv.org/abs/2505.09545; **SMUTHI**: https://arxiv.org/abs/2105.04259; **treams**: https://github.com/tfp-photonics/treams
- **iSCAT**: Mahmoodabadi et al. iPSF: https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-18-25969&id=434558; sub-10 kDa detection: https://www.nature.com/articles/s41592-023-01778-2
- **Stallinga & Rieger 2010** (fixed-dipole bias): https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-24-24461; **Backer & Moerner 2014** (dipole basis): https://pmc.ncbi.nlm.nih.gov/articles/PMC4317050/
- **SyMBac**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9710168/; **CATCH**: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b10463; **LodeSTAR**: https://www.nature.com/articles/s41467-022-35004-y
- **deepinv**: https://deepinv.org/user_guide/physics/intro.html; **Kornia** augmentation containers: https://kornia.readthedocs.io/en/latest/augmentation.container.html; **torchvision tv_tensors**: https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_tv_tensors.html
- **Mitsuba 3** (variants, plugins, AD integrators): https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html; **Pyro effect handlers**: https://pyro.ai/examples/effect_handlers.html; **Meta-Sim**: https://arxiv.org/abs/1904.11621; **AutoSimulate**: https://arxiv.org/abs/2008.08424
- **PyTorch**: complex autograd: https://docs.pytorch.org/docs/2.14/notes/autograd.html; complex compile support: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_complex_number_support.html; **triton-windows**: https://github.com/triton-lang/triton-windows; **pytorch-finufft**: https://flatironinstitute.github.io/pytorch-finufft/
- **Wuttke polyhedron form factors**: https://arxiv.org/abs/1703.00255; **DeCAF**: https://www.nature.com/articles/s42256-022-00530-3; **Kellman memory-efficient learning**: https://arxiv.org/abs/2003.05551

## Appendix C — Known pitfalls

Errors that earlier drafts, the panel proposals or the research briefs made, and where this design resolves them. The panel history is in [judge-verdicts.json](background/design-panel/judge-verdicts.json); the review history is Appendix E. Pitfalls of sampling layers are in Appendix F.

| Pitfall | Resolution |
|---|---|
| A second sampling DSL beside DeepTrack2's, with an adapter between them (revision 1) | the caller owns sampling; gradix is a standalone functional engine (ADR-32, ADR-37) |
| A planner as the only way to reach the physics (revisions 1 and 2) | public elements and Chains; the planner only constructs them (ADR-36) |
| A DeepTrack2-parity preset in the engine (`legacy_dt`, revisions 1 and 2) | dropped; a differences page and explicit opt-ins (ADR-37) |
| Objects that canonicalise leaf tensors at construction (stale autograd views of `nn.Parameter`s across optimiser steps) | data objects, elements and Chains store exactly the tensors given; canonical views are made per call (§6.1) |
| Right-aligned broadcasting of per-image `[B]` against per-object `[B, N]` fields (mixes images and objects when B == N) | the canonical per-call layout, asserted every call (§6.1) |
| Applying a 1/√cosθ collection factor *and* a 1/k_z Jacobian to an angular spectrum (off by cos²θ) | ADR-05; power and direct-summation tests |
| Using the index-matched apodisation when NA > n_s; propagating in the sample past the interface (evanescent blow-up) | the layered collection rule (§4.1) |
| A single-mode draft at the source centroid (wrong-sign phase contrast; darkfield becomes brightfield) | on-support modes (§5.4); sign test |
| Stochastic Abbe for training data ("hidden under shot noise") | deterministic quadrature (ADR-06) |
| "Metres cost nothing numerically"; "a⁶ subnormal and flushed"; "Adam ε larger than gradients in metres" | µm with the exponent-range rationale (ADR-01) |
| A total-order fidelity lattice | approximation edges (ADR-09) |
| Off-grid tilts sampled or snapped | analytic background; residual demodulation in marches (ADR-04) |
| March grids from detection Nyquist; SIM products formed on the detection grid | interaction and product grids (ADR-19) |
| Tier-dependent radiometry; s×s sum-pooling called "exact" pixel integration | consistency contract; pixel MTF (§4.3) |
| iSCAT-lite as the only iSCAT; the off-axis reference passed through the pupil | Fresnel reference (ADR-22); image-side reference (§4.5) |
| Slaney's reconstruction criterion as a forward validity rule; plain BPM without obliquity | calibrated φ predicate (§5.8); obliquity-corrected march (ADR-24) |
| "Dipole equals Mie at n = 1" with a₁ only; quasi-static α for plasmonic metals | a₁ + b₁ and material-class limits (§5.8) |
| Scalar PSFs "adequate to NA/n 0.7" for isotropic emitters | measured table and `psf="auto"` thresholds (§5.3) |
| \|F_ball\| folded into the OTF for bead extent | volume-quadrature bead = PSF ⊛ signed ball (§9b) |
| C1 injection at the particle centre plane | exit plane z_p + a (ADR-23) |
| A diagonal-everywhere Filter contract | space-variant Objective member (§5.3) |
| "A wrong cost model only makes plans slower" | cost never chooses between numerically different implementations (ADR-11) |
| Illustrative Mie 1.1 ms / 32 px padding; BPM ≈10³ img/s from procedural slices; "sparse ~10× faster" | measured anchors (§8.1) |
| A custom multislice adjoint required in v1 | checkpointing suffices (ADR-24) |
| A single `direction` per spectrum; a `Ports` carrier | travel tags (ADR-04) |
| Mie coefficients on the CPU; n_max via `.item()` | GPU fp64, n_max from the envelope (ADR-25) |
| +z into the sample with frame flips | one convention (ADR-03) |
| Gradient-set-dependent forward models | ADR-12 |
| Unrealistic roadmaps (including ≈30 weeks for revision 1's scope) | audited budgets with scope-delta tables (§12.2) |
| Clamping values to envelope bounds; automatic re-planning mid-stream | envelopes flag, never clamp; re-planning only in `gx.render` or explicitly (ADR-16) |
| Calling DT's `update()` inside the optics Feature (label branch reads a different sample) | recorded for the future integration (Appendix F.8) |
| Absent slots culled when presence requires gradients | every slot rendered; the cost model charges N (§6.1) |
| Stateful-generator draws inside checkpointed regions (backward recomputation redraws) | keyed draws are pure; batch-keyed draws happen before the region (§6.3) |
| Python-int seeds baked into captured CUDA graphs | keys as device tensors (§6.3) |
| Fixed dipoles with scalar/Gaussian PSFs | validity error (§5.8) |
| `inference_mode` outputs and CUDA-graph buffers handed to users | `no_grad` and caller-owned copies (ADR-31) |
| Closure-captured parameters in callable lowerings (silently no gradient) | explicit parameter inputs (§4.6) |
| Grid-indexed white-noise textures (change with fidelity and microscope) | keyed, resolution-free `gx.Texture` (§4.6) |
| Polarisation analysers as weights on squared field components (no 45° or circular analysers) | Jones projection before the square (§5.5a) |
| Dense Adam on a per-image parameter table fitted batch by batch (absent rows keep moving through momentum) | sparse embedding with SparseAdam, or independent per-batch fits (§6.9, Appendix F.5) |

## Appendix D — Reference listings

### D.1 Package tree (v1)

```
src/gradix/
  __init__.py              curated public API (§3.4)
  units.py                 um=1.0, nm=1e-3, mm=1e3, m=1e6, s=1.0, ms=1e-3                               [L0]
  conventions.py           frame, Fourier sign, z direction, Zernike (ANSI) indexing, radiometry, apodisation,
                           layered collection rule, P = 1 reduction (S∥), P = 2 lens rotation and s/p bases [L0]
  tree.py                  pytree utilities: parameters, map, to, replace                                  [L0]
  _core/                                                                                                   [L0]
    grid.py                Grid2D, FreqGrid(+PupilSupport), PupilGrid, PatchGrid, VolumeGrid, nice_size()
    rules.py               grid rules as pure functions of an envelope (detection, interaction, padding, ROI,
                           slices, λ bins, Mie n_max): gx.sampling.*, gx.planning.*
    envelope.py            Envelope (sources combine with |), envelope_of (from Chains or whole per-image tables),
                           derivation with headroom and buckets, field paths
    axes.py                canonical axis order, AcqIndex (flattened acquisition multi-index)
    carriers.py            PlaneWaves, Field, ObjectSpectra, Contribution, EmitterSet, EmitterDensity, Irradiance
    precision.py           PrecisionPolicy, no_autocast(), ieee_matmul() (scoped, re-entrant), fp64 phase builders
    keys.py                counter hash (splitmix64-class; Philox option) → open-interval uniforms; keyed Gaussian,
                           keyed Poisson (inverse CDF / PTRS + exact fallback); key normalisation (int | Tensor[B])
    registry.py            typed registries (object sets, views, lowerings, elements per slot, pupil modifiers,
                           labels, presets); decorators; duplicate-name errors; the plugin contract types
                           Element, Capabilities, Edge, Violation, Cost
    errors.py              PlanError, ValidityError, GradientPathError, StructureError (each names the fix)
  special/                                                                                                 [L0]
    mie.py                 a_n, b_n (homogeneous, layered; downward D_n; fp64), π_n/τ_n, S1/S2 on 1-D θ grid
    bessel.py              J0/J1/J2 with custom backward and jvp (double-where at x = 0); spherical j_n/y_n recurrences
    zernike.py             ANSI/Noll, RMS-normalised radial polynomials (fp64, recurrence beyond n≈15)
  schema/                                                                                                  [L0]
    fields.py              gx.field (quantity, role, event, shape_affecting, constraint, default, doc); quantities
    base.py                pytree base for data objects and elements: frozen dataclass, validation, replace, to, select
    layout.py              canonical per-call layout (role ranks, singleton dims), layout assertions
    stack.py               gx.stack: batching of any pytree, N-bucket padding with presence, structure checks,
                           CPU value checks; gx.pad for ragged per-image lists
    units.py               gx.from_si / gx.to_si (schema-driven, recursive)
    signature.py           structure signatures (Pipeline cache keys)
    export.py              gx.schema_of: schemas as data, schema versions
  ops/                     pure kernels                                                                    [L1]
    fourier.py             fft helpers, MFT (IEEE fp32), CZT, Fourier shift, band-limited resample
    propagate.py           ASM (H_exp form, safe k_z), BL-ASM, residual_tilt demodulation, layered transfer
                           (Fresnel, Gibson–Lanni)
    pupil.py               fused pupil evaluation on FreqGrid / PupilSupport / PupilGrid; soft edges; √cosθ / √(k_z,i/k_i); t(u)
    jones.py               Jones algebra: s/p bases, lens rotation, polarisers and retarders, analyser projections
    emitters.py            gaussian_sprites, psf_mft_roi, global_spectrum, strata_otf, turbid slab, scatter_add (det. option)
    mie.py                 S(θ) → pupil-support spectra (per mode, position phase; P = 2 vector or S∥ for P = 1), backscatter
    dipole.py              Mie-dressed α_e, α_m (a₁, b₁); scalar and 6-basis vectorial angular spectra
    born.py                form-factor Born on Ewald caps (travel ±1)
    thin.py                projection screen at the centre plane, (t−1)·E_b form, 1/cosθ_out; thin Jones screens
    multislice.py          obliquity-corrected march (residual-demodulated; scattered form opt-in; plain BPM opt-in),
                           procedural slices, segment checkpointing
    excitation.py          evaluate PlaneWaves / analytic beams at points and on product grids; transduction maps
    coherence.py           source quadrature (on-support, rings, Fermat, grid), mode chunking helpers
    detect.py              explicit-interference reduction, image-side references, pixel MTF, camera samplers (keyed or
                           explicit generator; EM gain double-where), likelihoods, Fisher information
  objects/                 data objects                                                                    [L2]
    objectset.py           ObjectSet base (position, rotation, presence, material, labeling, parent, id); Parent
    shapes.py              Emitters, Spheres, LayeredSpheres, Ellipsoids, Capsules, Cylinders, Boxes, Gaussians, SDF,
                           Occupancy (hard | soft), Voxels, JonesScreen, NeuralField, Texture (keyed random Fourier features);
                           @geometry decorator; geom helpers (local_box, quat_z, …)
    material.py            Material(n): Constant, Cauchy, Sellmeier, Tabulated, DrudeLorentz, DryMass; uniaxial(n_o, n_e, axis)
    materials.py           library: Oil, Glass, Water, Lipid, …
    labeling.py            Labeling(emission, density | photons, where), dipole.*
    environment.py         Homogeneous, LayeredMedium(immersion, coverslip, sample; design vs actual); presets
    spectrum.py            Spectrum (line, band, table)
  lower/                   views + lowering registry with exactness tags; raster (blurred ball, SDF ramp, patch
                           splat; custom autograd with explicit parameter inputs); form factors; texture synthesis  [L2]
  light/                   source elements: PlaneWave, Koehler(nodes=…), LEDArray (.grid, .select), ReferenceBeam,
                           SIMBeams, Evanescent, GaussianSheet, Shaped, Uniform, IlluminationNuisance, IlluminationEnvelope;
                           Disk, Annulus, Quadrature, OnSupport, GridAbbe                                  [L2]
  interact/                interaction elements: Mie, Dipole, Born, Projection, Multislice, Fresnel, Superpose   [L2]
  excite/                  transduction elements: Linear                                                   [L2]
  imaging/                 imaging elements: Coherent (scalar; vectorial and per-object in M4b), Sprites, PointPSF
                           (roi | global; scalar | vectorial), Strata                                      [L2]
  optics/                  Objective, pupil modifiers (Aperture, Defocus, Zernike, PixelPupil (OPD), Apodization,
                           GibsonLanni, PhaseRing, CentralStop, KnifeEdge, SpiralPhase, DICShear (scalar | Jones),
                           AmplitudeMask, PhaseMask, ChromaticDefocus, ModulePupil, FieldDependent; Jones: Polarizer,
                           Retarder, JonesMatrix); stage elements (Mask, Propagate, Lens, FourierTransform, Aperture,
                           DiffractiveStack, Filter, Reference); acquisition.py (FocusStack, Frames, LEDSequence,
                           Channels, SIMPhases, PolarizationStates)                                         [L2]
  devices/                 PhaseSLM, HeightMap; DMD, DeformableMirror, SegmentedMirror, Metasurface (v1.x):
                           parameters → transmission maps with non-idealities; MetaAtomLibrary (tables, surrogates) [L2]
  detect/                  Camera(qe, gain, offset, bit_depth, unit, noise) + noise models (Ideal, PoissonGaussian,
                           SCMOS(maps), EMCCD; QCMOS in v1.x); Camera.scmos(maps=…); gx.detect.sample / log_prob /
                           fisher / to_unit                                                                [L2]
  labels/                  Positions, InstanceMask, SemanticMask, Heatmap, DistanceMap, OPL, Phase, EmitterTable,
                           PerObjectImages; gx.labels.render; coords.py (to_pixels, from_pixels, depth, from_focus,
                           shift_focus)                                                                    [L2]
  compose/                                                                                                 [L3]
    chain.py               Chain (slot skeleton: light, scatterers, emitters, excite, imaging, references, camera,
                           background, acquisition, environment); replace; eager __call__
    pipeline.py            Pipeline (analysis + runner), bind (values by field path), capacities, capture,
                           run(until=, taps=), diff, to_json/from_json
    sampling.py            SamplingPlan: the _core/rules.py rules applied to a Chain's envelope (P5), with the
                           provenance of every decision
    validity.py            predicates on envelopes and on values, tolerance tables (versioned artefacts from ladders)
    gradients.py           gradient-quality table from capability tags; call-time zero-path refusal
    memory.py              dominant-tensor estimates, probe-chunk calibration, chunk sizes, checkpoint policy
    rewrites.py            pupil fusion, propagation merging, strata collapse, per-call hoisting
    outputs.py             Output (dict-like + meta); out.Image, Expected, Field, Tap, Regions
  plan/                    Fidelity, presets of knobs, per-population overrides; RULES; partition (containment,
                           proximity) into groups for gx.interact.Superpose; the planner (P2) emitting Chains     [L4]
  containers/              Sample, Microscope                                                              [L4]
  presets/                 Widefield, TIRF, SIM2D, Brightfield, Darkfield, PhaseContrast, DIC, DPC, QPI,
                           InlineHolography, OffAxisHolography, ISCAT, FPM, Lensless, PolarizedLight, PolScope
                           (recipes that build Microscopes)                                               [L4]
  api/                     gx.plan / Plan, gx.render (plan cache, pre-render envelope check), gx.compare,
                           gx.crlb (jacfwd + camera Fisher information), gx.save_inputs / gx.load_inputs       [L4]
  experimental/            unstable features, no SemVer guarantee (§12.8)
  accel/                   graphs.py (Pipeline.capture support); triton/ (optional, later)
  testing/                 conformance.py, ladders.py, gradprobe.py, camera_estimators.py, levels.py (equivalence
                           of L2/L3/L4), experimental.py, references/, bench.py
```

The reference torch sampler that gradix's own tests and cookbook tests use lives in `tests/`, outside `src/`, so the source bans of §3.2 hold without exceptions for it. Beside the package: `tests/` (including the docstring gate), `docs/tutorials/` and `docs/examples/` (notebooks, §12.7), the Sphinx sources in `docs/`, and `docs/background/` (provenance: research briefs, panel proposals and measurement scripts, excluded from linting and type checking).

### D.2 Grid types

```python
@dataclass(frozen=True)
class Grid2D:  # a sampled plane in object space
    shape: tuple[int, int]  # static, smooth sizes
    spacing: float  # µm, isotropic in v1
    origin: tuple[float, float]  # µm; top-left corner of the camera FOV minus padding


@dataclass(frozen=True)
class FreqGrid:  # FFT dual of a Grid2D; f in cycles/µm; λ-independent
    of: Grid2D
    support: PupilSupport | None  # static index list of samples with |f| ≤ max_λ NA/λ  → K


@dataclass(frozen=True)
class PupilGrid:  # direction space u = λ0·f⊥ (NA units); λ-independent disc |u| ≤ u_max
    shape: tuple[int, int]
    u_max: float


@dataclass(frozen=True)
class PatchGrid:  # R×R ROIs with shared spacing; per-object origins are tensors [B,N,2]
    shape: tuple[int, int]
    spacing: float


@dataclass(frozen=True)
class VolumeGrid:  # dense interaction or product grid (may be coarser/finer than detection)
    xy: Grid2D
    z0: float
    dz: float
    nz: int
```

## Appendix E — Changes applied

**Revision 11 (2026-09-29): holography pulled forward.** The lead developer asked for holography and Mie optics now (in-line, off-axis, iSCAT and QWLSI, with the camera's view), for differentiable field reconstructions, and for illumination designed for the bound on nanobead radius and index through the reconstruction.
- **New milestone M3a (ADR-43, §12.3):** after M1 and before M2, ≈7–9 weeks; M3 shrinks by ≈5–6 weeks, a net ≈2–3 weeks.
- **New §5.12:** the four interferometers as recipes; QWLSI as an analytic grating stage (ADR-45, cap-40); reconstructions (cap-41) and the bound on what they keep (ADR-44, cap-42).
- **New:** examples Ex10 and Ex11; questions Q23–Q26, settled the same day: P = 2 within M3a before Ex11's final numbers; iSCAT in the full layered geometry; the detection pupil and the reference as the first design variables, then the illumination pupil; bead sizes per modality.
- **As built (phases 1–5 and the examples):** §5.12 records each phase and what Ex10 and Ex11 found, and §5.11 records shaped illumination. `CRLB` bounds add across independent measurements. The Pipeline reports the precision its kernels run in. Vector producers (phase 6) remain. Two iSCAT exit criteria were restated against the measurements: the contrast period is lengthened by the collected ⟨cos θ⟩, and supercritical pupil quadrature converges slowly at the critical angle's branch point. Calibrations of reconstructions render at the frames' precision (`gx.recon.without_scatterers`).

**Revision 10 (2026-09-29): decisions after M0 and M1.** M0 and most of M1 were built and audited. The lead developer settled Q6, the keyed-noise trade-off and the CI scope, and asked for two API changes found while working through tutorial T1.
- **sCMOS likelihood (Q6):** the Gaussian with variance μ + σ² is the `log_prob` default (§11.6: unbiased at 0.1–100 photons; efficiency ≥ 0.93 for μ, down to 0.48 for the read noise at σ = 0.8). The truncated convolution (exact, ≈50× slower) serves calibration, and the shifted Poisson is biased at low light. High photon counts get an ad-hoc treatment if one is needed.
- **Keyed noise:** batch-keyed Poisson draws use `torch.poisson` again, which is approximate on CUDA above λ ≈ 1000 (§6.3); image keys stay exact. A keyed sampler that is exact and as fast is an M5 deliverable (a fused kernel). The §8.3 image-keyed target already required one, so the budget is unchanged.
- **CI:** GitHub-hosted runners only. The self-hosted 3090 runner is dropped (§8.3, §11.12, §11.15, §12.1); GPU benchmarks and full-size checks run locally before each release.
- **Outputs (§6.5):** `Output` exposes every output as an attribute as well as a key (`out.image`, `out.pos`).
- **Coordinates (§6.6):** `gx.coords.field_of_view` gives the frame's extent in µm, and `pitch` works as a pixel unit without a reference tensor.
- **Dense boundary (§5.6):** a `dense_boundary` knob picks the strata convolution: periodic (fast) at `draft` and `standard`, linear (exact) at `accurate` and `reference`. The §8.3 dense target applies to the periodic mode, and choosing the boundary automatically from the z positions and the pupil is a v1.x item.
- **Density grids (ADR-41):** the consuming element declares the grid, and lowerings rasterise or resample onto it, as in DeepTrack2 (cap-06).
- **Smaller decisions:** `psf="auto"` reports its scalar fallback as an info finding, not a warning (§5.3); an emitter's `photons` under excitation is its emission under the `reference` irradiance (§5.1); `gx.from_si` converts every quantity by its power of length, not only lengths (§6.1).
- **Names (ADR-42):** user names and framework names never share a namespace: shape fields travel as one mapping, fidelity knobs reach only declared element fields, and population, stage and output names pass one rule where they are introduced.
- **cap-06 (in progress):** `Solid` shapes (`Spheres`, `Ellipsoids`, `Capsules`, `Cylinders`, `Boxes`, `Gaussians`), `gx.Labeling` with `density`, `surface_density` or `photons`, the raster lowering (§4.6) and Strata's density grid (ADR-41); `@gx.geometry`, placed `Voxels` and bead quadrature follow.
- **CRLB (cap-19):** `gx.crlb` moves from M5 to M1 at the lead developer's request. It takes a Pipeline, a Plan or a Chain with `wrt=` field paths (components such as `beads.position.z` too), gives per-image and shared parameters their joint bound, gives absent objects an infinite bound, stays differentiable for design, and warns about scalar PSFs above NA/n = 0.45 (§5.3). Ex9 uses it.
- **Examples (§12.7):** Ex8 (labelled cells) and Ex9 (PSF engineering) join the scope; Ex1, Ex7, Ex8 and Ex9 run at M1, in smoke mode on every PR.
- **Tracked:** PointPSF core accuracy (the ROI renormalisation and the pupil's soft edge) in https://github.com/BenjaminMidtvedt/gradix/issues/1.

**Revision 9 (2026-09-28): answers to open questions.** The lead developer settled Q3, Q4, Q5, Q10 and Q17, and the method for Q6.
- **MIT** licence (a `LICENSE` file was added); the **torch floor** is ≥ 2.9 and rises whenever a needed feature requires it (`pyproject.toml` now says `torch>=2.9`).
- **Learnable magnification and pixel size in 1.0:** cap-26 (continuous pixel centres by MFT, affine registration between channels) moves from v1.x to M2; application #15 reaches its ideal fidelity at 1.0.
- **Analytic backgrounds:** Gaussian beams become the second `Background` implementation in M3 (illumination and references); diverging references (`SphericalWaves`) are an M4 stretch goal, else 1.1.
- **sCMOS likelihood:** a benchmark in the camera suite (§11.6) compares the approximations on accuracy, gradient bias and variance, CRLB fidelity and speed, and sets the default at M1 exit.
- **Image-keyed EMCCD** stays in v1.x.
- **Budget:** +1.75–2.5 weeks: ≈38–52 for two developers, plan on ≈46; the M2 review moves to week 15.

**Revision 8 (2026-09-28): the name.** The lead developer settled Q19: the project is **gradix**, imported as `gx`. Every `gradoscopy` became `gradix` and every `gs.` became `gx.`, including in the entries below, which were written under the working name gradoscopy; the provenance under `docs/background/` keeps the old name.

**Revision 7 (2026-09-28): code quality and documentation.** At the lead developer's request: ruff and ty compliance, numpydoc documentation, and narrow tutorial and example notebooks.
- **Gates (§11.13):** ruff lint now includes the annotation (`ANN`) and numpydoc docstring (`D`) rules. ruff format covers the Python examples of this document, which were reformatted, with long trailing comments moved onto their own lines. ty 0.0.84 checks every platform, with warnings as errors, and `numpydoc lint` runs from `pytest`. `py.typed` was added, and `docs/background/` is excluded as provenance.
- **Documentation (§12.7, ADR-40):** numpydoc content rules (units and shapes on type lines; Notes with physics and references; Examples for the user-facing API) and a schema cross-check; tutorials T1–T5 and examples Ex1–Ex7 as Jupyter notebooks with a fixed scope, executed in CI; the `api-friction` triage at each API review; how-to guides folded into the notebooks; the documentation share raised to ≈ 15 %.
- **Also:** milestone deliverables and exits list their notebooks; §11.15 runs the gates and the documentation build; risk 17 (ty is pre-1.0).
- **Budget:** +1–2 weeks: ≈37–49 for two developers, plan on ≈44.

**Revision 6 (2026-09-27): optical stages and optical neural networks.** Applied after the lead developer asked to keep optical neural networks in mind, as a way to shape light going into or coming out of the sample, fixed or trained with the rest of the system.
- **New §5.11 and ADR-39:** ordered `illumination_optics` and `detection_optics` stages with `Mask`, `Propagate`, `Lens`, `FourierTransform`, `Aperture` and `DiffractiveStack`; stage-plane sampling rules; shaped illumination (construction 3 of §4.5) read as a sampled field by dense producers and as J or local plane waves by sparse ones; device models (`PhaseSLM` and `HeightMap` in 1.0); training regimes (fixed, joint, physics-aware, optoelectronic); region readout; example §9(i).
- **Settled:** Q20 (ordered stages, no DAG). **New:** Q22 (keep the ONN core in 1.0, and whether the extended toolkit joins it), cap-38 and cap-39, application family #19, risk 16, E8, the `illumination_coupling` knob, and the stage validity rules and approximation edges.
- **Measured:** a 5-layer diffractive stack on 512² fields runs at ≈4.5k images/s forward and ≈1.7k forward + backward at B = 64 (`evidence/review/rev6/d2nn_throughput.py`).
- **Budget:** +2.5–3.5 weeks: ≈36–47 for two developers, plan on ≈42 (§12.2, §12.3).

**Revision 5 (2026-09-27): polarisation in 1.0.** The lead developer settled Q21: both pieces move into 1.0.
- **Coherent vector producers (M3):** Mie, dipole and Born at P = 2, with the lens rotation, s/p stack transmission, Jones backgrounds and references, and analysers in Detect. P = 1 (S∥) stays as the scalar fast path, chosen by `polarization="auto"` (§4.1). The P = 1 path now drops the cross-polarised part and reports its energy, instead of adding it incoherently.
- **Jones optics and thin Jones samples (M4):** `Polarizer`, `Retarder`, `JonesMatrix` and Jones `DICShear` pupil modifiers; uniaxial materials as thin Jones samples in the projection element; `gx.JonesScreen` maps (the retardance vector is the fitting form); per-frame polarisation states; the PolarizedLight and PolScope presets; example §9(h); E7 (report-only).
- **Also:** vector excitation from M4b (SIM and TIRF contrast, photoselection); vectorial focusing of excitation beams joins cap-25 (v1.x); cap-09, cap-23, §2.2 (17 of 18 families at 1.0), §5.1, §5.6 (`polarization` knob), §5.8, §5.9, §8.3, §11 and ADR-20 revised; Q21 closed.
- **Budget:** +3–4.5 weeks: ≈33–44 for two developers, plan on ≈39 (§12.2, §12.3).

**Revision 4 (2026-09-27): feeding data to built Pipelines; polarisation detection.** Applied after the lead developer's review of §0.3 and question about polarisation.
- **Feeding data** (ADR-38, new §6.9, example §9g): Pipeline calls bind values by field path onto the template (`pipe(values)`, `pipe.bind`), with declared inputs planned at their widest role; batch size and slot counts became capacities, so a ragged last batch and varying object counts fit one Pipeline; `gx.pad` for ragged per-image lists; `gx.envelope_of` over whole per-image tables, and envelope sources combine with `|`. §0.3 now feeds 1000 recorded images through the built Pipeline. Appendix F.5 records that dense Adam moves absent rows of a per-image table and SparseAdam does not (`evidence/review/rev4/adam_rows.py`).
- **Polarisation:** analysers are Jones projections before the square, because weights on squared components cannot express 45° or circular analysers (§5.5a); photoselection (the Cartesian excitation field contracted with the dipole orientation tensor) and analyser channels join M4b (cap-17); ADR-20 revised; new open question Q21 on pulling coherent P = 2 producers and Jones optics into 1.0.
- **Name:** Q19 and §0.6 record the main contenders, `abbe` and `gradix`.
- **Budget:** +0.25–0.5 weeks, within ≈30–39 (§12.2).

**Revision 3 (2026-09-25): a standalone engine with a layered public API.** Applied at the lead developer's direction. gradix is useful on its own, and standalone users bring their own sampling. DeepTrack2 integration will coincide with a revision of DeepTrack2's optics pipelines. The engine is not forced into `Microscope`/`Sample` containers; those are optional and constructed from lower-level objects.

- **New ADRs:**
  - ADR-36: layered public API with building blocks first (L2 elements and data objects, L3 `Chain`/`Pipeline`, L4 `gx.plan`/containers/presets).
  - ADR-37: standalone project; DeepTrack2 later; no DeepTrack2-parity preset.
- **Revised ADRs:** ADR-00, 10, 11, 17, 24, 28, 29, 32, 33 and 35.
- **Removed from v1:** the `legacy_dt` preset, DT coordinate and intensity helpers, the DT spike, the joint DeepTrack2 milestone, DT contract and parity tests, and schema-generated DT Features as a deliverable. Plain BPM, hard masks and additive Δn remain as explicit opt-ins, and a "Differences from DeepTrack2 2.0.2" page replaces parity gating.
- **Rewritten sections:**
  - §0, §3 (API levels, packages, ownership, namespace, data flow) and §9 (examples at the level that suits each);
  - §10 ("Front ends: bring your own sampling; DeepTrack2 later", with a tested sampling cookbook);
  - §12 (M2 is now the composable-API milestone; ≈30–39 weeks);
  - §14 (risks 3, 6 and 9 recast; Q16, Q19 and Q20; Q1, Q14 and Q18 closed);
  - §7.6, §11.10 (level equivalence and the front-end contract), Appendix D.1, and the new Appendix F.8 (notes for a future DeepTrack2 integration, including the semantics mapping).
- **Targeted edits:**
  - §1: G2, G4–G6, non-goals and the new principle 13;
  - §2: cap-01, 07, 13, 15, 20 and 28, coverage wording, §2.3;
  - §4.1, §4.2 and §4.6: boundary wording, and Occupancy/Voxels/overlap opt-ins;
  - §5.1: the element contract and catalogue mapping;
  - §5.6: fidelity at each level; §5.7: the planner and the Pipeline analysis;
  - §5.8 and §5.9: no `legacy_dt`;
  - §6: the contract with any caller;
  - §7.1–§7.4: element registries and plugins usable at every level;
  - §8: generic front-end rows and a standalone SMLM target;
  - §11.3, §11.6, §11.11 and §11.14;
  - glossary rows (Chain, Element, Pipeline, Plan, Data object);
  - Appendices C and F.


**Post-revision-3 audit (2026-09-25).** An independent consistency audit of revision 3 raised 25 findings, all applied:
- the element contract: knobs are constructor arguments, `configure(desc, envelope)` resolves "auto", Pipelines call `forward`, and an eager `__call__` sizes grids from its inputs; `Envelope`, the grid rules and `gx.planning` moved to L0, so elements never import `compose`;
- `Fidelity` confined to L4: `on_invalid`, `memory_budget` and `deterministic` are Pipeline arguments, and preset-keyed rules are keyed by knob value (`interaction_band`, the new `raster_oversample`, the imaging element's `vectorial_above`);
- pass ownership: P0 and P1 are shared, P2 is the planner's, P3–P8 are the Pipeline's; partitioned groups render through `gx.interact.Superpose`; `escalate` is L4 only;
- wording: "planner", "container" and "plan" replaced wherever an element, a data object or a Pipeline is meant, and "plan time" added to the glossary;
- DeepTrack2: the coordinate-convention arguments removed from `gx.coords` and `Positions`, and the remaining wording that presented DeepTrack2 as part of v1 rewritten;
- milestones: example (b) uses `psf="auto"` and its CRLB line waits for M5; capture of incoherent Pipelines is M2 (cap-21); `gx.render` is an M0 deliverable; §0.6 is in deadline order, with the week-5 freeze stated precisely;
- examples: `gx.lower.emitter_set`, `@gx.register.object_set`, a device-agnostic §9 preamble with a device-bound generator, `gx.tree.replace`, `plan.pipeline.envelope`, and an envelope for the §0.3 plan;
- the cookbook: every draw names its device, and truncation uses the erfc form with right-tail mirroring, because `torch.special.ndtr` loses the lower tail even in fp64 (F.2; checked by `evidence/review/rev3-audit/ndtr_tail.py`).

**Revision 2 (2026-09-25): DeepTrack2 owns parameter sampling.** Applied at the lead developer's direction: DeepTrack2 stays the front-facing library and selects what gets rendered, and gradix becomes a functional backend for differentiable optics (ADR-32).

- **New ADRs:**
  - ADR-32: DT owns sampling; gradix is a functional backend.
  - ADR-33: schema-described containers are the contract.
  - ADR-34: randomness only through explicit keys.
  - ADR-35: no sampling syntax inside gradix.
- **Superseded ADRs:** ADR-13 (parameter and trace system) and ADR-15 (default plates).
- **Revised ADRs:** ADR-00, 01, 11, 12, 14, 16 (now envelopes), 17 (functional core), 26, 28, 29, 30 and 31, plus wording in ADR-02, 03, 06, 07, 18 and 25.
- **Rewritten sections:**
  - §0 and §1, with the new principle 1 "the front end decides; the engine renders";
  - §2: the capability matrix; cap-01 is now the input containers and schemas, cap-28 is DT's schema-generated Features, and the coverage assumptions are stated;
  - §3: the layer stack with DeepTrack2 above gradix, the ownership table and the namespace;
  - §4.6 and §6 (the input contract, gradients, keys, envelopes, outputs, labels and sequences);
  - §7.3 and §7.6; §9 (all examples in the functional API, with the DT view in (a));
  - §10 ("DeepTrack2 as the front end": the contract, two-phase evaluation, workers with late binding, learnables across the boundary, the optional-syntax decision);
  - §11.5–§11.13; §12 (≈30–38 weeks, with the delta table, the DT spike in M0 and a joint M2); §14 (risks 3, 6 and 9 recast; Q1, Q16, Q17 and Q18 new; Q2, Q7, Q8 and Q12 moved to DT);
  - Appendices C and D.1, the Appendix A glossary, and the new **Appendix F** (the sampling-layer findings handed to DeepTrack2).
- **Targeted edits elsewhere:**
  - "bounds" became the envelope throughout §4.3 and §5;
  - the implementation contract sees the envelope and container tensors;
  - the photophysics Transduce variants were removed (front end);
  - `plan.validate` replaces `sim.validate`;
  - fixes to §8 anchors and targets;
  - §11.3 and §11.14.

**Post-revision-2 audit (2026-09-25).** An independent consistency audit of revision 2 raised 16 findings and a set of nits, all applied: coverage stated as depending on DT roadmap items (with plain torch as the interim route); the Gantt chart redrawn at one column per week; chunk sizes chosen at plan time and the batch-size bucket added to cache keys; no deferred host read of envelope flags (no hidden state); `Texture` without `compose=`; background realisations supplied by the front end; leftover revision-1 wording; `gx.detect` wrappers moved to `api/`; `biased` added to the gradient-quality vocabulary; `gx.render`'s cache named as sanctioned global state; DT-backend ownership wording; example fixes (camera in e1, QPI objective without the ring, irradiance units, capture under `no_grad`, matching DT/torch ranges); the test sampler moved out of `src/`; `gx.save_inputs`/`gx.experimental` in §3.4; envelope precedence in P1; widened source and medium signatures; `deterministic=` on `gx.plan`.

The revision-1 entries below are kept as history. Where they mention the parameter layer, the DT-L1–DT-L6 adapter levels, `Simulator`, the Trace, plates or estimators for scene draws, those parts now live in DeepTrack2 (Appendix F) or were removed.


The adversarial review raised 99 findings (physics, feasibility, DeepTrack2/API, differentiability, completeness). Each was verified; all were applied, those marked *partially confirmed* in the verifier's corrected form. One line each:

**Physics.**
- PHYS-1 Layered collection rule for NA > n_s: propagate only to the interface, stack transmission t(u), one Gibson–Lanni focus term, immersion-side apodisation, stationary-phase pad/ROI rule, layered tests (§4.1, §4.3).
- PHYS-2 born_ff validity recalibrated to φ_max ≤ 0.25 rad; Slaney kept only as a citation; `compact` rerouting at `standard` (§5.2, §5.6, §5.8).
- PHYS-3 Obliquity-corrected march as the default; projection 1/cosθ_out, NA and feature-size terms; edge metrics on the scattered field (§4.4, §5.1, §5.9).
- PHYS-4 Measured scalar-vs-vectorial table; `psf="auto"` thresholds 0.45/0.7; quantitative fits vectorial (§5.3, §5.6).
- PHYS-5 Pixel integration by the pixel MTF, moved into v1 (§4.3, §5.1).
- PHYS-6 Image-side off-axis reference; detection spacing from interfering pairs (§4.3, §4.5).
- PHYS-7 Transduce product grid for structured excitation of dense densities (§4.3).
- PHYS-8 Mie-dressed a₁ + b₁ dipole; material-class validity; no double radiative correction (§5.1, §5.8).
- PHYS-9 Gaussian-sprite edge ≤ 20 % plus label metrics; σ = 0.22λ/NA; NA ≤ 0.7 (§5.1, §5.9).
- PHYS-10 P = 1 reduction pinned to S∥ with incoherent cross-polarisation (§4.1).
- PHYS-11 Illumination envelope scales the total coherent intensity by |g|² (§4.5).
- PHYS-12 Raster prefilter retagged approximate; in-band attenuation report; finer rasters at `accurate` (§4.6).
- PHYS-13 One slice criterion (λ_m/2, 0.5 rad); edges run at the preset's dz (§4.3, §5.8, §5.9).

**Feasibility.**
- FEAS-1 Audited 38–44-week budget with a scope-delta table and the M6 option (§12).
- FEAS-2 Minimal fit core scheduled with the milestones that test it (§12.3).
- FEAS-3 M5 split: M4b in parallel, hardening-only M5, release-candidate feedback gate (§12.3).
- FEAS-4 Counter-keyed RNG kept with a cheap specified hash, measured costs, restricted identity claim, named exceptions (§6.5, ADR-13).
- FEAS-5 Residual-demodulated total-field march by default; scattered form opt-in; no plugin background obligation (§4.4, ADR-04).
- FEAS-6 Coherent carriers drafted at M0, frozen at M3 exit; coherent smoke thread at M1 (§1.3, §12.3).
- FEAS-7 One `travel` tag per contribution; `Ports` deleted (§4.4).
- FEAS-8 Carrier shapes fixed for A and per-image λ; rank-7 carriers named; shape conformance test (§4.2, §4.4).
- FEAS-9 Guards warn and keep the plan; opt-in replan; declared Derived bounds (§6.6).
- FEAS-10 Coverage recounted; three cheap restorations make 16 of 18 true (§2.2).
- FEAS-11 Planner and prob sizes restated; no byte models or version-keyed caches (§5.7).
- FEAS-12 Gradient check reduced to a small static check plus the dynamic probe (§5.7, §6.4).
- FEAS-13 Handlers replaced by explicit call options in the cache keys (§6.7).
- FEAS-14 Every §8.3 target names its route and is set at ≤ 0.8× a median of 5 runs (§8.3).
- FEAS-15 Fidelity table limited to implementations that exist; `escalate` defined for v1.x (§5.6).
- FEAS-16 First-in milestone column; features without triggers moved to v1.x (§2.1).
- FEAS-17 Trace namespaces; `render(trace)` rules; MultiView as a helper (§6.8).
- FEAS-18 Chunk sizes recorded in the plan and `batch.meta` (§4.3).
- FEAS-19 Torch floor tested nightly; MPS best effort and out of PR CI (§11.15, §12.5).
- FEAS-20 Public-namespace table; `io/`, placement and materials in the tree; signatures aligned (§3.4, Appendix D).

**DeepTrack2 / API.**
- API-01 The optics Feature never calls `update()`; pairing tests (§10.2, §11.10).
- API-02 Output-ownership contract; fresh allocations for graph outputs (§6.3).
- API-03 `EngineObject.array` slow path and array passthrough for DT scatterers (§10.2).
- API-04 Pinned DT ↔ gradix coordinate mapping with parity tests (§10.3).
- API-05 `units="dt"` and DT noise semantics in pipeline order (§10.2).
- API-06 Map-style `BatchedDataset` with `__getitems__` (§10.2).
- API-07 Synchronous pre-render bound check for DT paths (§10.2).
- API-08 `as_feature` in no-grad and grad modes (§10.2).
- API-09 Path grammar frozen in M0; examples rewritten (§6.1, §9).
- API-10 Learnable ownership; `sim.freeze()` returns a new Simulator; `regularization()` (§6.1).
- API-11 Jacobian-normalised Learnable conditioning; constraint/bounds precedence (§6.1).
- API-12 DT learnables: user-owned tensors, bounds hints, units, user Aberrations (§10.2).
- API-13 `as_inverse` assigns every Random site (§9e).
- API-14 Camera QE/gain/offset/unit; photometric calibration in the examples (§5.5, §9).
- API-15 Resolution-free `RandomField` (§4.6).
- API-16 Population templates, Part signature, nested plates (§4.6, §6.2).
- API-17 `gx.from_focus`; z-bounds predicate; example (b) fixed (§4.1, §5.8, §9b).
- API-18 Stochastic telegraph blinking in v1; frames with interval and exposure; OPD pixel pupils (§6.4, §6.9).
- API-19 DT sequences batched; `Reuse` and random-count mappings; `Markov` process (§10.2, §10.3).
- API-20 Honest co-transform statement; per-sample augmentation in the DT fast path (§6.8, §10.2).
- API-21 `gx.out.Field` and `gx.labels.Phase` (§5.5).
- API-22 Cheap bounds (`rel=`, `abs=`, `auto_bounds`); `batch_shape` rule; fit-one-scalar tutorial (§6.1, §6.3, §9).
- API-23 `Fidelity.preset` positional; `backward` field (§5.6).
- API-24 Zernike fit set without defocus; one constructor (§9b).
- API-25 One output-layout rule (§4.2, §10.2).
- API-26 Pixel-unit positions for network targets; trace values in µm (§6.8, §9a).
- API-27 `gx.SDF` with parameters, `@gx.geometry`, occupancy-only geometries (§4.6).
- API-28 Namespace and signature consistency; DT-L1…DT-L6 numbering (§3.4, §10).

**Differentiability.**
- DIFF-1 `no_grad` generation, caller-owned outputs, event-fenced guard flags (§6.3, §6.6).
- DIFF-2 No stateful-generator draws inside checkpointed regions (§6.5).
- DIFF-3 NaN-free, integer-exact scaled ST; honest bias labels (§6.4).
- DIFF-4 Presence defaults to score + LOO outside ST-Concrete's band; matched logistic noise (§6.4).
- DIFF-5 Counter hash specified; device-tensor seeds; restricted identity claim; persistent generators (§6.5).
- DIFF-6 fp64 truncated inverse CDFs; envelope auto-truncation; retained-mass reporting (§6.6).
- DIFF-7 Gradient lattice with partial paths; noise-energy probe; CRN finite differences (§6.4, §11.5).
- DIFF-8 Canonical full-rank plate layout (§6.2).
- DIFF-9 Replay bit-exact only with deterministic splatting and a frozen plan; per-site `mode="noise"` (§6.5).
- DIFF-10 Callable lowerings take explicit parameter inputs (§4.6).
- DIFF-11 EM-gain double-where with straight-through term (§5.5).
- DIFF-12 `validate_args=False`; no Distribution objects on the hot path (§6.5).
- DIFF-13 Wider sampler bans (§3.2).
- DIFF-14 Custom Functions support forward AD, vmap and double backward (§7.4).
- DIFF-15 Merged with API-11 (§6.1).
- DIFF-16 Stochastic source nodes labelled biased for nonlinear losses (§5.4, §6.4).
- DIFF-17 Double-where at removable singularities; singular gradcheck points (§7.4).
- DIFF-18 `ieee_matmul()` documented as the sanctioned scoped global mutation (ADR-02).
- DIFF-19 MPS phase construction on the CPU or in double-float; hosted macOS runs CPU only (§12.5).

**Completeness.**
- COMP-01 Fidelity and learnability rows in the §0 answer table.
- COMP-02 §0 rewritten as "Read this first".
- COMP-03 Light restructure: table of contents, reference listings to Appendix D, condensed ADRs, "Known pitfalls", §8.1 as the performance-anchor table. ADRs stayed ≈2.8k words because they gained the review's decisions.
- COMP-04 Notation appendix; capability IDs renamed cap-00…cap-37; overloaded symbols renamed; glossary extended.
- COMP-05 B08 substitution column; coverage count corrected.
- COMP-06 Exit criteria name only delivered APIs (§12.3).
- COMP-07 First two weeks (§12.1).
- COMP-08 Open questions with deadlines, owners and defaults (§14.2).
- COMP-09 Experimental validation plan (§11.14).
- COMP-10 I/O, datasets and persistence (§12.6).
- COMP-11 Sequences (§6.9, example (f), §8.3).
- COMP-12 Channels and spectra (§5.5a).
- COMP-13 Platform support tiers (§12.5).
- COMP-14 Documentation plan and owners (§12.7).
- COMP-15 Versioning, stability and governance (§12.8).
- COMP-16 Supercritical support handled by the layered rule, without clipping (§4.1).
- COMP-17 Network losses left to deeplay and user code (§3.3, §9a).
- COMP-18 Calibration-objective (loss family) table (§6.10).
- COMP-19 §8.3 rows relabelled by exact configuration; "four engines" replaced.

**Final audit.**
- C01 M2 exit keeps DTEx252 and the fluorescence §11.10 tests; DTDV431, DTGS162, the `bf(p) & fl(p)` pairing and the DTDV431-style tests move to the M3 exit (§11.10, §12.3).
- C02 "Hardening only" replaced by hardening plus a listed set of fit/estimator features (§0.5, Gantt, §12.3 prose, risk 15).
- C03 Minimal Mie-dressed dipole `ObjectSpectra` for the coherent smoke thread added to M1; `dipole` first in M1 (smoke) / M3 (§5.1, §12.3).
- C04 "Rank-8 intermediate" restated as the one coherent intermediate with an N axis (§4.2, §4.4).
- C05 `AngularSource` has four constructions (§4.5).
- C06 Public-namespace table completed (plugin API, carriers, views, `gx.Microscope`, `gx.Point`, `gx.ReuseNoise`, `gx.units`, …) and limited to v1 names (§3.4).
- C07 One Microscope field list without `reference`; `microscope.*` explained as the grouping of the Microscope roots; spiral-phase preset rewritten (§3.5, §6.8, §7.2, §7.5).
- C08 Axis capitals completed (T, S, Z, U, H, W, R) in §0.8 and A.1; slice count written n_slices (§5.4, risk 13).
- C09 Undefined `Frames` carrier removed (§5.1, Appendix D).
- C10 Thin phase-contrast target routed through `Fidelity("standard", source=gx.Quadrature(n=25))` (§8.3).
- C11 `compact` at `standard` renders the volume group through the `volumes` knob (§5.6).
- C12 M6 lands ≈4–6 weeks after 1.0 (§0.5).
- C13 §0.6 lists the decisions that most affect the plan, in deadline order, including Q14 and Q6.
- C14 Torch floor decided during week 1 (§12.1).
- C15 v1 has no MPS tests or MPS-specific routes (§12.5).
- C16 ADR field statement matches the records (§13).
- C17 Ideal release set to 2 for SMLM and to later for holography and darkfield (§2.2).
- C18 `units.py` and `conventions.py` shown as top-level L0 files (§3.1, §3.2).
- C19 Appendix D registry list matches §7.1.
- C20 Camera samplers of `ops/detect.py` exempted from the sampler ban with explicit generators (§3.2, §6.4).
- C21 `EmitterSet.species`/`id` and `ObjectSpectra.params` carry the A position `[B, 1, N]` (§4.4).
- C22 Nested-part plate shape includes T (§4.6).
- C23 Example (a) Stage 2 passes `batch_size=64` (§9a).
- C24 All §9 examples run at small shapes on CPU per PR and at full size nightly (§11.15).
- C25 Score + LOO presence ships with Count(Binomial) at M0; M5 keeps learnable-rate Count(Poisson) (§2.1, §12.2, §12.3).
- F01 Fluorescence chain is Emit (through the fused Objective pupil) → Detect; per-object and vectorial pupils are evaluators Emit consumes (§0.2, §5.1).
- F02 Merged with C01.
- F03 Merged with C02.
- F04 `impl/` layer for the registered implementations, with rules, owner and tree entry; contract types in `_core/registry.py` (§3.1–§3.4, Appendix D).
- F05 Scalar fallback with a warning while no vectorial implementation is registered; n in NA/n defined; E1 gates v1.1 under the M6 option (§5.3, §5.6, §11.14, §12.2).
- F06 E6 reported, not gated, at v1.0 (§11.14, §12.3).
- F07 Merged with C13 and C14; Q numbers added (§0.6).
- F08 Q10 and §0.6 ask which further `Background` implementations to add; protocol named `Background` in ADR-04.
- F09 Headline gradient claim lists the v1 exceptions (§0.1).
- F10 Scope answer adds M4b, the M6 count and a correct projection/multislice description (§0.4).
- F11 Emitter densities named among the incoherent carriers (§0.2, §0.4).
- F12 `legacy_dt` excluded from the consistency contract (§0.4, §5.6, §5.9).
- F13 Fixed-rate Poisson counts ship in M1; learnable rate at M5 (§2.1, §6.1, §6.4, §12.3).
- F14 Minimal plan JSON for frozen-plan replay at M0; `fit.reg.Laplacian` at M1; "frozen plans" dropped from M5 (§2.1, §5.7, §12.1, §12.3).
- F15 Merged with C06.
- F16 Applied in C07's form: no `reference` root, since sample-side references are `ReferenceBeam`s in `light` (§4.5).
- F17 Merged with C05; the fourth construction marked as outside the pupil cross-spectral density (§4.5).
- F18 Axis claim qualified and A.1 extended (with C08); physics symbols H, W, S, G kept and named as non-axis uses rather than renamed.
- F19 Fresnel zero orders owned by `layered.fresnel`; it and `IlluminationEnvelope` added to the background obligations (§3.5, §4.4, §5.1).
- F20 Merged with C20.
- F21 Every package has one owner; presets with Developer A, as in M4 (§3.3).
- F22 Merged with C18; the import-linter treats the top-level files as L0 (§3.2).
- F23 Example (b) bead z bound ends at −radius (§9b).
- F24 Fitted pupils are reused by sharing or by a frozen copy (§6.1, §9b).
- F25 Transduce on the product grid produces `EmitterDensity` (§3.5, §5.1, A.2).
- F26 "≤ 3 implementations" counted per (population kind, knob) (§2.3, ADR-11).
- F27 Born is the only cheap backward-port rung for extended non-spherical objects (§5.2).
- F28 Merged with C15.
- F29 `select=` call option defined and keyed; delivered with `fit.minibatch` in M4 (§2.1, §5.7, §6.7, §12.3, Appendix D).
- F30 Multislice angle predicate limited to plain BPM; recalibrated for the obliquity-corrected march (§5.8).
- F31 `standard` emitters cell reads "pupil" (§5.6).
- F32 M6 option moves v1.0 to ≈35–41 weeks; §12.2 estimates in calendar weeks (§0.5, §12.2).
- F33 Lengths default to a 1 µm scale (§6.1).
- F34 Per-population overrides limited to model-choice knobs (§0.4, §5.6).

## Appendix F — Notes for sampling layers

Revision 1 built a probabilistic layer inside gradix, and the panel and reviewers measured a lot about it. Sampling now belongs to the caller (ADR-32): the user's own code in standalone use (the cookbook of §10.3 is built on these notes), and DeepTrack2 once integrated. This appendix hands over what was learned, as recommendations for any sampling layer. Nothing here is a gradix API. Evidence tags point to the scripts in `docs/background/evidence/`.

### F.1 Estimators for sampled values

| Draw | Recommended default | Alternatives | Evidence |
|---|---|---|---|
| Continuous with `rsample` (Normal, LogNormal, Uniform, Exponential; Gamma/Beta/Dirichlet by implicit reparameterisation) | pathwise | score function | [B07 §6.2] |
| Truncated continuous (sizes, z) | fp64 inverse-CDF truncation (F.2) | rejection (biased gradients) | [Rev] |
| Presence under a Binomial count law with learnable p | **score with a leave-one-out baseline** (unbiased; ≈2× the std of ST-Concrete) whenever p's range leaves [0.15, 0.85]; **straight-through Concrete, τ = 0.5** (hard forward, so images and labels are exact) inside that band, as the low-variance option | mean-field presence = p (biased, 8× lower std); τ = 0.25 under evaluation | ST-Concrete at τ = 0.5 is biased, with a p-dependent ratio: 0.86 at p = 0.5, 1.25 at p = 0.05 or 0.95, 1.38 at p = 0.02, on losses linear in presence. It is several-fold off on quadratic count losses at p ≤ 0.05 (N = 8, p = 0.05: −7.89 against a true −1.73) [Rev]. **Soft Concrete at τ = 1 has the wrong sign** (+80.8 against a true −473) [M-D Exp B] |
| Count under a Poisson law with learnable rate | exact truncated-Poisson count (inverse CDF over 0…N), score + LOO | the user states a Binomial law to get relaxed gradients | **the count law must never change because it became learnable** [M-J ML] |
| Discrete choice (`OneOf`) | ST Gumbel-softmax (hard forward) | score + LOO; enumerate all options | Meta-Sim needed REINFORCE [B09 §1.10] |
| Photophysics (blinking) | stochastic telegraph sampling per frame for training data (no gradients to rates); mean-field occupancy when fitting rates | stochastic 2-state Markov + score | expected-mode blinking alone gives wrong training data for context-frame networks (DECODE) |
| Non-overlap | allow overlaps with principled compositing (§4.6), or rejection flagged as biased | a differentiable repulsion step | [B06 §7] |

**Logistic-noise convention.** Draw ST-Concrete's logistic noise as g = log(1 − u) − log u. Its hard sample is then exactly 1[u < p] for the same uniform u that a non-learnable path would threshold. Making p learnable, or freezing it, then never changes the realised images for a seed; the opposite sign convention agrees on only 40 % of slots [Rev].

**Score bookkeeping.** Store per-image log-probabilities summed over score-function draws. A surrogate Σᵢ stopgrad(Lᵢ − bᵢ)·log pᵢ with a leave-one-out baseline adds the score term to any per-image loss. Batch-level losses such as MMD have no per-image decomposition; use sub-batches with a between-sub-batch baseline (× the number of sub-batches in cost).

### F.2 Numerics of truncated sampling

- Run truncated inverse CDFs in **fp64**, with Φ(z) = ½·erfc(−z/√2) evaluated in the tail nearest the window, and mirror windows right of the mean. Clip the result to [a′, b′] only to absorb ≤ 1-ulp rounding. That is not a sample clamp.
- `torch.special.ndtr` evaluates ½(1 + erf(z/√2)) and loses the lower tail even in fp64: it is 1.8 % low at z = −8 and exactly 0 at z = −9 (torch 2.14) [Rev]. Use the erfc form, as the cookbook does (§10.3).
- The textbook fp32 form Φ⁻¹(Φ(a′) + u(Φ(b′) − Φ(a′))) fails in three ways:
  - it returns NaN for every sample when both bounds lie in one tail (`ndtr(−6.28)` is 0 in fp32);
  - it turns the whole batch's gradient into NaN when u = 0;
  - it rounds samples just outside [a, b] [Rev].
- **Never clamp** sampled values to envelope limits: clamping creates point masses at the bounds and biases distribution learning [M-J ML].
- **Report the retained mass** Φ(b′) − Φ(a′). If a learnable distribution drifts into its truncation, learning stalls: dE[x]/dμ fell to 0.02 in a drifted case [Rev].
- Construct `torch.distributions` objects with `validate_args=False` per call, never through the global `set_default_validate_args`. Argument validation is a host synchronisation that breaks CUDA-graph capture. Keep `Distribution` objects off the hot path (use them for `log_prob` and quantiles).

### F.3 Layout of batched draws

Keep every batched draw in a canonical full-rank layout, with singleton dimensions where a value does not vary and event dimensions last:
- per image: `[B, 1, 1]`;
- per object: `[B, 1, N]`;
- per frame and object: `[B, T, N]`.

Right-aligned broadcasting of rank-variable shapes silently mixes images and objects when B == N: with B = N = 8, image 0's objects received the means of images 0–7 [Rev]. Test hierarchical priors and derived properties with B == N == T.

### F.4 Randomness and reproducibility

- **Per-index identity.** To make item i of a dataset independent of batch composition, order and worker count, derive every draw from a counter hash of (seed, item index, property id, element), not from a shared stateful generator. A splitmix64-class hash emulated in int64 costs 30 kernels: ≈0.3 ms eager and ≈0.05 ms as a CUDA graph for 64 × 1020 uniforms plus a Normal transform. Philox-4x32-10 costs 183 kernels, ≈1.9 ms eager and 0.34 ms graphed [Rev]. Map to open intervals, u = (k + 0.5)·2⁻ᵇ. Hash property ids to 64 bits and reject collisions. gradix's `key=` accepts per-image int64 keys, so any front end can pass the same per-index keys for camera noise.
- **CUDA graphs.** Pass seeds as device tensors that are copied into before each replay. A Python-int seed would be baked into the captured kernels, and every replay would return the same batch. Use persistent, re-seeded generator objects; a captured graph binds only the generator objects present at capture.
- **Checkpointing.** Never draw from a stateful generator inside a checkpointed region. `preserve_rng_state` protects only the default generators; in a measured case, recomputation redrew with an advanced explicit generator and changed a gradient from 1.194 to 0.998 [Rev].
- **Common random numbers.** Storing the base noise (the uniforms or standard normals) of pathwise draws allows re-transforming them under updated parameters. This gives low-variance calibration gradients, and deterministic L-BFGS by sample-average approximation, for objectives with no discrete draws and no sampled detector noise [D §6.5].

### F.5 Learnable parameters

- **Natural units.** Adam moves each parameter by ≈ lr in parameter units, so lr = 10⁻² is 1 cm per step on a metre-valued position [M-C]. Parameterise lengths in µm or pixels, and positive quantities in log-space, then convert differentiably.
- **Constrained parameters.** If a wrapper applies a constraint transform T, normalise the raw step by the Jacobian at initialisation:
  - value = T(T⁻¹(init) + u·scale/|T′(T⁻¹(init))|), so that dvalue/du = scale at init for every constraint;
  - the naive value = T(scale·u) moved a bounded Zernike coefficient 20× less than advertised [Rev];
  - default scales by quantity: 1 µm for lengths and OPD, 1 for dimensionless values, |init| only for strictly positive quantities such as photons, counts and rates. Never use |init| for quantities that may be 0 (Zernike coefficients, pixel pupils) [M-J ML].
- **Shape-affecting learnables** (sizes, z, NA, λ) need an envelope entry covering the range they may explore (§6.4). Otherwise gradix's derived envelope (×1.25 around the initial value) is flagged as they drift.
- **Sharing.** The same `nn.Parameter` used in two places (two elements, or two DT properties) is one parameter. A pupil fitted on beads is reused by sharing it (joint training) or by passing a detached copy (frozen generator), as in §9(b).
- **Per-image values over a dataset.** Keep them in a per-image table and gather the rows of each batch by index. Dense Adam still moves rows that are absent from a batch, through its momentum: 1.6 learning-rate steps over three absent batches in a measured case, against 0 for `torch.optim.SparseAdam` on a sparse embedding (`torch.nn.Embedding(..., sparse=True)`) [Rev]. Use a separate dense optimiser for parameters shared between images. When nothing is shared, the images are independent problems: fit each batch to convergence (for example per-image Levenberg–Marquardt on forward-mode Jacobians) instead of cycling through batches. A single L-BFGS over many independent problems couples them through one line search and one curvature history (§6.9, §9g).

### F.6 Calibration objectives

gradix supplies `expected` outputs, camera likelihoods (`gx.detect.log_prob`), Fisher information and `gx.crlb`. The objectives themselves belong to the caller: user code, deeplay, or DeepTrack2 later:

| Family | Where |
|---|---|
| Per-image likelihood (bead stacks, Lorenz–Mie fits) | user loop over `gx.detect.log_prob` (§9b, §9e) |
| Summary statistics, MMD, quantile/radial-spectrum features | deeplay or DT; background-normalised features avoid absorbing photometric nuisances |
| Adversarial critics | any torch critic on rendered outputs; pathwise gradients plus the score surrogate of F.1 |
| Simulation-based inference (NPE) | the caller generates (θ, x) pairs without gradients; interop with the `sbi` package |
| Task-driven bilevel (validation loss on a few labelled real images) | alternating updates or one-step unrolled look-ahead |

### F.7 Throughput of resolution

Measured on the same machine [M-D]: resolving 40 properties for B64 × N50 takes 1.48 ms with naive per-property torch draws, 0.17 ms fused and 0.30 ms as a CUDA graph. A DT-style per-sample Python loop takes 19.0 ms, and per-sample DT resolution costs ≈0.3 ms/image (a proxy). Batched or vectorised resolution is the lever for training-data throughput when gradients are needed. When they are not, CPU workers that resolve and `gx.stack` in parallel feed a main-process renderer (§10.3).

### F.8 Notes for a future DeepTrack2 integration

These notes are what revisions 1 and 2 learned while designing a DeepTrack2 backend. They are recorded here so that the integration can reuse them once DeepTrack2's optics pipelines are revised (§10.4). None of them is a gradix v1 deliverable.

**Lessons:**
- **Never invalidate DeepTrack2's cache inside an evaluation.** An optics Feature that calls `update()` or `invalidate()` resamples every upstream property. A parallel label branch (`optics(p) & p.position`) then reads a different sample from the one rendered [B01 §g]. The training idiom is `pipeline.update()` after each optimiser step (DTDV431, DTEx252).
- **Two-phase evaluation:** per batch, resolve B replicates in Python, `gx.stack` the objects, render once, then apply post-optics Features as batched tensor operations. Reuse each replicate's resolved augmentation parameters; a torchvision v2 transform applied to the whole batch draws one parameter set for all images.
- **Throughput:** per-replicate resolution costs ≈0.3 ms per image (a proxy [M-D]), ≈3k images/s in one process. For generation without gradients to sampled values, resolve and stack in CPU workers, render in the main process, and bind learnable optics tensors there (late binding).
- **Pre-render envelope check:** values are known on the CPU before stacking, so grow the envelope and re-plan before rendering; never drop objects, clamp values or resample replicates. For `particle ^ (lambda: np.random.poisson(20))`, about 8.5 % of 64-image batches exceed the probed N bucket [Rev].
- **Custom scatterers that produce arrays** pass through as `gx.Voxels` (spacing = voxel size/upsample, DT's anchoring convention), with gradient quality `zero` for hard masks. Function-defined shapes can move to `gx.SDF` or `gx.Occupancy`.
- **Learnables:** `nn.Parameter`s in DT properties flow into elements unchanged, and unit conversions must stay differentiable (×10⁶, or × the pixel size). Learnable distributions need reparameterised torch draws in the property rules; NumPy draws reach autograd as constants.

**Semantics mapping, pinned against DeepTrack2 2.0.2.** Develop's PR #464 changed sub-pixel placement, so an integration must test the mapping, not assume it [B01 §c].

| DeepTrack2 2.0.2 | gradix |
|---|---|
| **Lateral position** `position=(p0, p1)` [px; an integer is a pixel *centre*; axis 0 = rows, DT's "x"] | `(x, y) = ((p1 − c0 + ½)·d, (p0 − r0 + ½)·d)` µm, with (r0, c0) the `output_region` origin and d = resolution/magnification. For `position_unit="meter"`: metres → µm with the same axis swap |
| **Axial position** `z` [px of voxel_size_z, relative to the focal plane] | `z = focus + s·z_DT·d_z`, with s = ±1 fixed by an astigmatic-PSF sign test; `Homogeneous(refractive_index_medium)` unless a `LayeredMedium` is supplied |
| `NA`, `wavelength`, `magnification`, `resolution`, `refractive_index_medium` | `Objective`, the light element, `Camera(pixel_size)`, the environment |
| `upscale=(ux, uy, uz)` | `oversample=ux` plus `dz` (element knobs; set by a `Fidelity` at L4) |
| `upsample` (scatterer) | `raster_sigma`; occupancy is soft unless `hard=True` is requested |
| `padding` | derived from the envelope (DT's 10 px default is far too small for defocused coherent cases [B05 §5]) |
| `output_region` | camera ROI, and the lateral offset above |
| `pupil=` (Zernike `(n, m)` in rad; `LearnablePhaseMask`) | `gx.pupil.Zernike` OPD in µm converted at DT's λ; user aberration functions evaluated on a DT-shaped pupil array and resampled differentiably onto gradix's pupil samples; `PixelPupil` / `ModulePupil` |
| `illumination=` (`IlluminationGradient`; noise on the field) | `IlluminationEnvelope` (§4.5); `IlluminationNuisance`, or complex noise on the field output |
| `return_field=True` | `gx.out.Field(normalize="incident", grid="camera")` (\|E_b\| = 1 at the focal plane) |
| `intensity` / `refractive_index` / `value` | `Labeling(photons)` or `Emitters(photons)` / `material` / `material` |
| `Brightfield` / `Holography` | `Brightfield` / `InlineHolography` presets, or a Chain with `gx.imaging.Coherent` |
| `Darkfield` (\|E − 1\|²) | a physical darkfield: annulus with NA_c > NA or a central stop, blocked reference |
| `ISCAT` (a Mie-angle convention) | epi illumination, Fresnel reference from the `LayeredMedium`, backscatter (travel +1); the homogeneous "iSCAT-lite" tier |
| `MieSphere`, `MieStratifiedSphere` (polarisations, `collection_angle`, `coherence_length`) | `gx.interact.Mie` on `Spheres` / `LayeredSpheres`; polarisation on light and analyser; NA from the objective; `coherence_length` → spectral bandwidth |
| `Poisson(snr)`, `Gaussian`, `Background`, `Offset`, `ComplexGaussian` | DT-semantics batched stages in pipeline order (e.g. `Poisson(snr)` rescales each image by its own peak and draws with scaled-ST Poisson), or a physical `gx.Camera` |
| `SampleToMasks` | threshold transform → `gx.labels.InstanceMask` / `SemanticMask` |
| `NonOverlapping`, `^ N`, `^ callable`, `Sequence`, `Reuse`, `seed()` | stay in DT: rejection on descriptors; N slots with presence; bucketed N with the pre-render check; frame-role fields; re-rendering the same inputs; DT seeds its own RNG and derives gradix's `key` from it |
| `(H, W, 1)` output | gradix returns `[B, (T,) C, H, W]`; DT chooses its own per-sample layout |

**Parity references** (for the "differences" page, not as a gate): DTGS121, DTGS131, DTEx203, DTEx205, DTEx252, DTGS106, DTGS162, DTGS172, DTDV431. Compare normalised-image rel-L2 and KS tests on label statistics. Per-sample pairing tests catch a label branch that reads a different sample from the one rendered.
