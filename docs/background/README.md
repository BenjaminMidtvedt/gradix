# Background material for the architecture plan

These files are the inputs to [`../architecture.md`](../architecture.md). The project was called gradoscopy until
2026-09-28, and the documents here use that name. They are kept for provenance and as
a prior-art reference. **They are not authoritative.** Several claims in them were corrected by the judges
and by the adversarial review, and where a background file and `architecture.md` disagree,
`architecture.md` wins.

## How the plan was produced (2026-09-24)

1. **Research.** Nine parallel briefs on prior art, physics and engineering. Where it mattered, the briefs
   include measurements taken on the target machine (RTX 3090, Windows 11, torch 2.14 + CUDA 13.0).
2. **Design panel.** Four independent architecture proposals, each from a different angle (A–D), were
   scored by three judges through different lenses: physics, engineering and ML usability. Several judge
   verdicts were backed by new measurements.
3. **Synthesis.** A single plan was built on the favoured base, with the best ideas from the other
   proposals grafted in. Every error the judges flagged was explicitly avoided.
4. **Adversarial review.** Five critics examined the plan (physics, feasibility, API/DeepTrack2,
   differentiability, completeness). A skeptical verifier then tried to refute each finding, and the
   verified findings were applied to produce `architecture.md`.
5. **Revision 2 (2026-09-25).** At the lead developer's direction, parameter sampling moved out of
   gradix (ADR-32). The design panel proposals and several briefs (notably 01, 06, 07 and 09) assume that
   gradix owns sampling. Appendix F of `architecture.md` hands their sampling-layer findings over to
   whoever writes the sampling.
6. **Revision 3 (2026-09-25).** gradix became a standalone engine with a layered public API: building
   blocks, then Chains and Pipelines, then planner and presets (ADR-36). DeepTrack2 integration is left for
   DeepTrack2's own optics-pipeline revision (ADR-37). Brief 01's integration analysis now lives in
   Appendix F.8 as notes for that future work.
   An independent consistency audit of revision 3 followed, and its 25 findings were applied (Appendix E);
   its check of the cookbook's truncation recipe is in `evidence/review/rev3-audit/`.
7. **Revision 4 (2026-09-27).** After the lead developer's review, Pipelines gained a documented way to be fed
   data (value binding by field path, capacities, ragged padding), and the polarisation detection path was
   corrected. The optimiser check behind the per-image fitting advice is in `evidence/review/rev4/`.
8. **Revision 5 (2026-09-27).** The lead developer moved polarisation optics into 1.0: vector scattering, Jones
   pupil optics and thin birefringent samples (ADR-20).
9. **Revision 6 (2026-09-27).** Optical stages before and after the sample for programmable and diffractive
   optics, including optical neural networks (ADR-39). The diffractive-stack benchmark is in
   `evidence/review/rev6/`.

## Contents

| Path | Cited in the plan as | What it is |
|---|---|---|
| `research/01-deeptrack.md` | [B01] | DeepTrack2 deep-dive: feature graph, optics, scatterers, torch work, pain points, integration options |
| `research/02-diff-optics-libs.md` | [B02] | Differentiable optics libraries (Chromatix, TorchOptics, odak, dO, poppy, hcipy, prysm, …): sampling and API lessons |
| `research/03-microscopy-simulators.md` | [B03] | Microscopy simulators and PSF tools (microsim, psf-generator, uiPSF, DECODE, holopy, waveorder, …) |
| `research/04-light-matter-ladder.md` | [B04] | Light–matter interaction methods as a fidelity ladder; solver contract; mixed samples |
| `research/05-imaging-system.md` | [B05] | Illumination, pupil, detection; coherent-mode decomposition; collapse theorems; camera models |
| `research/06-sample-representation.md` | [B06] | Sample representations, differentiable anti-aliased voxelisation, capabilities, batching |
| `research/07-pytorch-engineering.md` | [B07] | Complex autograd, FFT and memory measurements, torch.compile/Triton status, estimators, testing |
| `research/08-applications-scope.md` | [B08] | 18 application families, their fidelity needs, proposed scope |
| `research/09-framework-patterns.md` | [B09] | Architecture patterns (Mitsuba, PyTorch3D, deepinv, Pyro, Kubric, …) and anti-patterns |
| `design-panel/proposal-A-field-operators.md` | [A §x] | Physics-first optical-field operator core |
| `design-panel/proposal-B-scene-planner.md` | [B §x] | Declarative scene compiled by a planner |
| `design-panel/proposal-C-mvp-scope.md` | [C §x] | Pragmatic, scope-first ("seams first, solvers later") |
| `design-panel/proposal-D-learnable-pipelines.md` | [D §x] | Learnable data generation and sim-to-real first |
| `design-panel/judge-verdicts.json` | — | Judge scores, grafts, key-decision verdicts and the errors found in each proposal |
| `evidence/panel-A` … `panel-D/` | [M-A] … [M-D] | Measurement scripts written for each proposal |
| `evidence/judges-*/` | [M-J eng / physics / ML] | Measurement scripts the judges used to check claims |
| `evidence/research-06/`, `research-07/` | [B06 M], [B07] | Measurement scripts behind the sample-representation and engineering briefs |
| `evidence/review/<critic>/`, `evidence/review/verify-<critic>/` | [Rev] | Checks run by the adversarial-review critics and by the verifiers who tried to refute them |

The evidence scripts are throwaway prototypes and are not part of the package. To re-run one from the repo
root:

```bash
uv run python docs/background/evidence/judges-engineering/bpm_check.py
```

Numbers depend on the GPU, the driver and the torch version.
