# 04 — Light–matter interaction methods as a fidelity ladder

Research brief for **gradoscopy**, a differentiable light-microscopy simulator built on PyTorch and meant to become the DeepTrack2 engine.

Evidence tags: **[V]** means I checked it against the cited URL during this research (Sept 2026). **[R]** means it comes from my own knowledge of the literature and I believe it is correct, but I did not re-check it here. **[?]** marks an estimate or an uncertain claim. Numbers without a tag are my own calculations.

---

## 0. Scope and notation

This brief covers everything between "the illumination reaches the sample" and "light leaves the sample toward the collection optics". That includes emitters located inside the sample. Illumination synthesis, the pupil and aberrations, and the detector are covered in other briefs. They appear here only as the boundary conditions that interaction solvers must meet.

Notation:
- λ0 is the vacuum wavelength and k0 = 2π/λ0.
- n_m is the medium index, k_m = n_m·k0 and λ_m = λ0/n_m.
- The object has index n = n_r + iκ, contrast Δn = n − n_m and relative index m = n/n_m.
- The size parameter is x = k_m·a for radius a.
- The scalar scattering potential is V(r) = k0²(n(r)² − n_m²).
- The accumulated phase delay is φ = k0·Δn·d for thickness d.

Anchor cases, with λ0 = 532 nm in water (λ_m ≈ 0.40 µm):

| Sample | Size | Δn | x | φ |
|---|---|---|---|---|
| Mammalian cell | d ≈ 5–15 µm | 0.01–0.06 (lipid droplets and nucleoli up to about 0.1) | 80–240 | 0.6–10 rad, locally more |
| 1 µm polystyrene bead | — | 0.26 | ≈ 7.9 | ≈ 3.1 rad |
| 5 µm silica bead | — | ≈ 0.12 | ≈ 39 | ≈ 7 rad |
| 80 nm gold nanoparticle | — | ε ≈ −5 + 2i | ≈ 0.6 | dipolar/Mie regime; unusable in any voxel solver at practical grid sizes |

Most DeepTrack-style use sits in the first three rows. That is why the middle of the ladder (multislice, Mie, and their composition) matters most.

**Current DeepTrack2 baseline**, as summarized by my fetch of `deeptrack/optical/optics.py` and `scatterers.py` on the `develop` branch [V, via a tool summary, so the details may be approximate]:
- `Brightfield`, `Holography` and `ISCAT` propagate a voxel refractive-index volume slice by slice in Fourier space, applying `exp(1j * ri_slice * voxel_size * K)` per slice. This is a multislice/BPM scheme. The pupil is applied at the end.
- `MieSphere` and `MieStratifiedSphere` return a `ScatteredField` computed from S1/S2. The "hybrid" mode maps the amplitudes onto pupil spatial frequencies. The field is added coherently at the detector, which is plain superposition with no coupling to the volume.
- `Fluorescence` convolves each z-slice with a defocused PSF and sums the results.

gradoscopy should keep this functionality as its lower rungs and generalize it. Sources: https://github.com/DeepTrackAI/DeepTrack2/tree/develop/deeptrack/optical

---

## 1. The ladder at a glance

The ladder is not one-dimensional. There are three families that share one output currency:
- **volumetric** methods: voxel in, field out;
- **analytic-particle** methods: parametric shape in, multipole/far-field out;
- **statistical/geometric** methods: rays or photons.

Emitters are a fourth concern that runs across all three families.

| Rung | Method | Consumes | Natural output | Validity (rule of thumb) | Cost / memory | Torch differentiability | Release |
|---|---|---|---|---|---|---|---|
| P0 | Point dipole scatterer / point emitter (ASF/PSF splatting) | point list: position, α or brightness, dipole orientation | coherent angular spectrum per point, or incoherent PSF | x ≲ 0.3 for pure dipole; ≲ 1 with Mie-dressed α; sparse | O(N_pts · N_pupil) or an FFT convolution | trivial | v1 |
| V1 | Thin/projection screen (scalar or Jones) | 2D OPD and absorption map, or voxel volume integrated along z | field on the exit plane | d ≲ DOF ≈ λ0·n/NA²; negligible diffraction inside the object | O(N² log N) | trivial | v1 |
| V2a | First Born | voxel V, or analytic V̂(q) | angular spectrum on the forward and backward Ewald caps | Δn·d ≲ 0.35 λ (Slaney) | O(N_z·N² log N), parallel over z | trivial, self-adjoint | v1 |
| V2b | First Rytov | voxel V | exit field | Δn/n_m ≲ a few %, almost any size | same as Born plus log/exp | yes; fails at zeros of u_in | v1 |
| V3a | Multislice BPM (angular-spectrum diffraction plus phase screen; tilt-corrected; WPM variant) | voxel n | exit field | forward-only; moderate Δn; moderate scattering angles; no reflections | 2 FFTs per slice, sequential | autograd plus checkpointing | v1 |
| V3b | Multi-layer Born (MLB) | voxel | exit field plus approximate backscatter | better at oblique illumination | ≈ 2–3× BPM [?] | autograd | v2 |
| V3c | SSNP | voxel | exit field (forward projection) | non-paraxial; higher contrast than BPM; reflections only approximate | ≈ 2× BPM | autograd | v1, as a BPM mode |
| V4a | Convergent/modified Born series, scalar and vector | voxel ε in a padded box | full field in the box, so both transmission and reflection | exact up to discretization; any contrast, but slow at high contrast | N_it × 2 FFTs; N_it ∝ thickness·Δ(n²) | implicit differentiation (one adjoint solve) | v2 (reference rung) |
| V4b | Lippmann–Schwinger with Krylov (SEAGLE, BiCGSTAB, GMRES) | voxel | full field | as V4a; Krylov can stall at high contrast | as V4a plus Krylov vectors | implicit differentiation | v2 (same operator code) |
| V4c | FDFD | Yee voxel grid | full field | exact up to discretization; 3D preconditioning is hard | sparse direct or iterative solve | adjoint | no (adapter at most) |
| V5 | FDTD | Yee grid; dispersive or nonlinear materials | time-domain fields; broadband | anything, including plasmonics; small volumes only | 6 components per cell; 10³–10⁴ steps | adjoint or time-reversal | external validation only |
| A1 | Lorenz–Mie, homogeneous or stratified | analytic sphere: a, n(λ), layers | S1/S2 far-field amplitudes mapped to the pupil angular spectrum; near field | exact for an isolated sphere in a homogeneous medium, any x | O(N_max) coefficients + O(N_max · N_pupil) | custom recurrences; autograd works | v1 |
| A2 | T-matrix: spheroids, cylinders, clusters, substrates | shape parameters, orientation, particle list | T matrix; outgoing VSWF coefficients | EBCM: moderate aspect ratio and x; clusters via Foldy–Lax | O((N_p·L²)³) direct, or iterative | no torch library exists | v2 (sphere clusters), v3 (non-spherical) |
| A3 | DDA / GDM | voxelized particle only | dipole moments, then far and near field | \|m\|·k·d ≲ 0.5–1; accurate for \|m\| ≲ 2 | FFT: O(N log N) per iteration with O(N) memory; dense: O(N²) | adjoint is trivial (symmetric matrix) | v3, or an adapter to TorchGDM |
| G1 | Ray optics / eikonal | mesh or SDF; smooth surfaces | optical-path map (phase screen); ray bundles | x ≳ 10²–10³; no diffraction | rays × intersections | Dr.Jit/Mitsuba, or custom | later |
| G2 | Monte-Carlo radiative transfer / diffusion | μ_s, μ_a, g fields | incoherent intensity; background | thickness ≳ l_s; ensemble average; no phase | photons × steps | hard (path-replay) | later; phenomenological model in v1 |
| E | Incoherent emitters (fluorescence) | point list or density voxels, plus orientation | intensity via PSF, or coherent ASF per emitter | depends on the propagation rung used | see §2.13 | yes | v1 |

---

## 2. Method notes

### 2.1 Point models (P0)

**Small scatterers.** A small particle with x ≲ 0.3 is an induced dipole, p = ε_m·α·E_inc(r_p).
- The quasi-static polarizability is α₀ = 4πa³·(m² − 1)/(m² + 2).
- It should always be radiation-corrected, α = α₀/(1 − i·k_m³·α₀/(6π)). Otherwise extinction is wrong and so is the iSCAT phase. [R]
- A better choice is the Mie-dressed polarizability from the first coefficient, α_e = i·6π·a₁/k_m³, plus a magnetic dipole from b₁. This extends the dipole picture to x ≈ 1. [R] It links P0 to A1 at no extra cost.

The scattered far field is the standard dipole pattern. Mapped onto the pupil, it is a closed-form vector angular spectrum carrying the position phase e^{−i(k−k_in)·r_p}.

Everything here is analytic and differentiable with respect to position, α, n and wavelength. Cost is O(N_particles × N_pupil) with a batched einsum, or one FFT if the particles are first splatted onto a grid and a common pupil is used.

The same machinery is the natural low-fidelity rung for iSCAT, darkfield and holography of nanoparticles. It also shares code with fluorescence emitters (§2.13). In both cases the dipole radiates, near an interface or not; only coherence and the source of p differ.

### 2.2 Thin object / projection approximation (V1)

The exit field is u_exit(x,y) = u_in(x,y)·t(x,y), with t = exp[i·k0·∫(n_r − n_m)dz − k0·∫κ dz]. The anisotropic version is a 2×2 Jones matrix per pixel. chromatix ships both `thin_sample` and `jones_sample` [V, https://github.com/chromatix-team/chromatix/blob/main/src/chromatix/functional/samples.py].

**Validity.** Diffraction inside the object must be negligible at the resolution being imaged. The practical condition is d ≲ n·λ0/NA², the depth of field. Features finer than the NA are filtered out by the pupil anyway. [R]
- At NA 0.3 the approximation holds for objects about 8 µm thick.
- At NA 1.2 it only holds for objects under about 0.5 µm.
- It also ignores the obliquity of illumination. For tilted illumination multiply by 1/cos θ_in.

**Cost** is one elementwise product. It is the ideal rung for fast 2D training data: phase contrast, holography of flat cells, quantitative phase imaging.

**Differentiability** is trivial. The only gradient pitfall is geometric parameters: if the projection is rasterized with hard edges, gradients with respect to position and radius vanish (see §3.3).

### 2.3 First Born (V2a)

u_s(r) = ∫G(r−r′)·V(r′)·u_in(r′)d³r′, with G = e^{ik_m|r|}/(4π|r|).

For plane-wave incidence k_in, the angular spectrum of the scattered field on a plane z is

  Û_s(k⊥; z) = (i/(2k_z))·e^{ik_z z}·V̂(k − k_in),   k = (k⊥, ±k_z),   k_z = √(k_m² − |k⊥|²).

The "+" branch is transmission and the "−" branch is reflection. [R, standard Fourier diffraction theorem]

This is the only cheap rung that produces a **reflected** field (the backward Ewald cap) at the same cost as the transmitted one. That matters for epi-geometries.

**Implementation.** The robust GPU form is slice-wise:

  Û_s(k⊥) = Σ_z (i/(2k_z))·e^{ik_z(z_out−z)}·F2D[V(·,z)·u_in(·,z)]·Δz.

This avoids non-uniform interpolation onto the Ewald sphere, works for any incident field sampled in the volume (focused, structured, evanescent), and is embarrassingly parallel over z. That is one batched FFT, not a sequential march. The alternative is a 3D FFT followed by gather onto the cap. It is faster for many illumination angles, but interpolation error has to be managed. For analytic shapes V̂ is closed-form: the sphere form factor 3(sin qa − qa·cos qa)/(qa)³, and ellipsoids by coordinate scaling. This gives exact, band-limited, differentiable Born scattering with no voxelization at all.

**Validity.** Slaney, Kak and Larsen found Born valid when (relative index change) × diameter < 0.35λ. Rytov had essentially no size constraint, provided the relative index change is below a few percent. [V, https://www.slaney.org/malcolm/purdue/Slaney1984(LimitationsDiffractionTomography).pdf, abstract] In phase terms that is roughly φ ≲ 2 rad. A 1 µm PS bead is borderline and a whole cell is outside the regime. Born also underestimates the index in reconstructions [V, MLB paper, below].

**Cost and memory.** O(N_z·N² log N) FLOPs. The forward pass needs only an accumulator. Because Born is linear in V, the gradient with respect to V is the adjoint Born operator, which is the same slice-wise structure conjugated. A custom `autograd.Function` therefore avoids storing N_z slices.

### 2.4 First Rytov (V2b)

u = u_in·exp(ψ), with ψ = u_s^Born/u_in in the usual sense of the Rytov transform. [R] Cost equals Born plus a pointwise exp.

**Validity.** Rytov is right for large, weakly refracting, smooth objects, which describes cells in low-to-mid NA transmission. It constrains local phase gradients rather than total phase. [R; Chen & Stamnes, Appl. Opt. 37, 2996 (1998) compare the two]

It fails for:
- backscatter;
- strong refraction or total internal reflection;
- structured or focused illumination with zeros in u_in, where the division breaks. This is a hard failure mode that the implementation must guard against.

**Recommendation.** Offer Rytov as a *flag on the Born solver*, not as a separate solver.

### 2.5 Multislice family: BPM, WPM, MLB, SSNP (V3)

**Classic multislice (Feit–Fleck BPM, in the angular-spectrum form).**

  u(z+Δz) = F⁻¹{ e^{ik_zΔz}·F{ u(z)·e^{ik0(n(x,y,z)−n_m)Δz} } }.

The diffraction step is exact for the homogeneous background: it is the angular spectrum, not Fresnel. The approximation lies in the **phase screen**, which assumes each slice imposes phase along z regardless of the local propagation direction. So for FFT-based microscopy BPM, "paraxial vs wide-angle" is really about the interaction step, not the propagator.

Consequences:
- Errors grow with illumination and scattering angle, and with Δn.
- There is no reflection and no backscatter.
- A cheap first-order fix for oblique illumination is to divide the screen phase by cos θ_in.
- Classic Padé "wide-angle BPM" (Hadley) belongs to finite-difference waveguide BPM. It is not the right tool here. [R]

**Wave Propagation Method (WPM; Brenner & Singer 1993).** In each slice, the angular spectrum is propagated with k_z(n_j) for each distinct index n_j, and each pixel picks the result for its own index. This handles strong, piecewise-constant index steps non-paraxially, for example beads, droplets and microoptics. Cost is ∝ (number of index levels) × FFT. Vectorial versions with Fresnel transmission coefficients exist, and so does a recent differentiable WPM for freeform optics. [V, Schmidt et al., "Wave-optical modeling beyond the thin-element-approximation", Opt. Express 2016; differentiable WPM: https://pubmed.ncbi.nlm.nih.gov/40310784/]

WPM is a good BPM *mode* for samples described by a few materials.

**Multi-layer Born (MLB; Chen, Ren, Waller, Optica 7, 394, 2020).** Each slice is treated with first Born instead of a phase screen. This captures oblique scattering and estimates backscatter. The authors argue that multislice "does not optimally accommodate for highly oblique scattering and does not model backward scattering". [V, https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-5-394]

**Split-step non-paraxial (SSNP; Zhu, Wang, Tian, Opt. Express 30, 32808, 2022)** [V, https://arxiv.org/abs/2207.06532]. SSNP rewrites Helmholtz as a first-order system in z for Φ = (φ, ∂φ/∂z)ᵀ, with ∂Φ/∂z = (H₁ + H₂(r))Φ. Each step is Φ(z+Δz) ≈ P·Q(z)·Φ(z).
- In Fourier space, P = [[cos(k_zΔz), sin(k_zΔz)/k_z], [−k_z·sin(k_zΔz), cos(k_zΔz)]], with the evanescent components zeroed.
- In real space, Q = [[1, 0], [k0²(n0² − n²(z))Δz, 1]].
- At the output, the field is back-propagated to the focal plane, pupil-filtered, and the forward-going part is extracted: φ_out = F⁻¹{(½, −j/(2k_z))·F{Φ}}.

It keeps the multislice structure. Cost and memory are about 2× BPM (two field components). The paper states it has "similar computational costs" to BPM and MLB while being more accurate for high-angle illumination and strongly scattering samples [V]. The reference code is PyCUDA with a hand-written autodiff framework [V, https://github.com/bu-cisl/SSNP-IDT], so gradoscopy would reimplement it. That is about 100 lines of torch.

Caveat: SSNP solves an *initial-value* problem for a boundary-value equation. Backward-going light is carried internally but is not correctly bounded, which is why the forward projection F is needed. It is not a reflection-mode solver. [V/R]

**Shared engineering for the whole family:**
- **Cost.** 2 FFTs per slice for BPM, 4 for SSNP, sequential in z. Example: 512² lateral × 128 slices × 16 batch at complex64 is 2048 FFT pairs of 256k points, tens of ms on a modern GPU. [?]
- **Autograd memory.** Stored activations are N_z × N² × 8 B per field per batch item. Example: 16 × 128 × 512² × 8 B ≈ 4.3 GB, plus FFT temporaries. Use `torch.utils.checkpoint` every √N_z slices, which gives about 0.4 GB here.
- **Reversible marching.** For lossless samples, BPM and SSNP steps are invertible. The backward pass can reconstruct activations by inverse marching, the same trick FDTDX uses with Maxwell time-reversibility [V, https://github.com/ymahlau/fdtdx]. It is unstable for absorbing samples, so treat it as an optional memory mode.
- **Sampling.** Lateral Δx ≤ λ0/(2·NA_max), where NA_max covers both illumination and the scattering that matters. Δz ≈ λ_m/2 … λ_m for BPM. [R]

**Recommendation.** Build **one z-marching engine** in which BPM, tilt-corrected BPM, WPM, MLB, SSNP, the Jones/polarized multislice (chromatix has `polarized_multislice_thick_sample` [V]) and the linear slice-wise Born are all instances of a (diffract, interact) step pair. Give it injection hooks at every slice for sources (§5).

### 2.6 Full-wave frequency-domain volume solvers (V4)

**Convergent / modified Born series (Osnabrugge, Leedumrongwatthanakun, Vellekoop, J. Comput. Phys. 322, 113, 2016)** [V abstract, https://arxiv.org/abs/1601.05997]. The method solves ∇²ψ + k(r)²ψ = −S by splitting k(r)² = k0² + iε + V(r), with ε ≥ max|k(r)² − k0²|.
- G = F⁻¹·1/(|p|² − k0² − iε)·F is a damped Green's function applied by FFT.
- The preconditioner is γ = (i/ε)·V.
- The iteration ψ ← (γGV − γ + 1)ψ + γGS converges for any non-gain medium.

Each iteration advances the wavefront a "pseudo-propagation" distance 2k0/ε [V, search snippet of the paper]. So N_it ∝ thickness × Δ(n²)·k0/2, plus extra iterations for resonances and multiple reflections.

My estimates for a 20 µm thick sample:

| Sample | Δ(n²) | Pseudo-propagation per iteration | Iterations for one traversal | Practical total |
|---|---|---|---|---|
| Cell | ≈ 0.13 | ≈ 1.3 µm | ≈ 15 | 50–200 |
| PS beads in water | ≈ 0.76 | ≈ 0.2 µm | ≈ 90 | several hundred |
| Metals | — | — | — | hopeless; use dipoles or Mie instead |

The authors report the method is about 100× faster than pseudospectral time domain, with far higher accuracy [V, search snippet]. Extensions:
- **Vector Maxwell and general media.** Vector Maxwell (Krüger et al. 2017 [R]). Arbitrary linear media, including birefringent, chiral and magnetic, in **MacroMax** (Vettenburg et al., arXiv 1812.10463). MacroMax has an optional PyTorch backend [V, macromax docs and search].
- **Recurrent-network form.** A "physics-defined RNN" formulation ran a 5 mm × 5 mm (8000 λ) problem with 576 M samples. It needed about 35 GB of working memory, 39,002 iterations and 108 minutes, and "no backpropagation is used" [V, https://arxiv.org/abs/2208.01118].
- **Domain decomposition over GPUs** (Mache & Vellekoop, arXiv 2410.02395). It solved 315³ λ³ in 1.4 h on two GPUs [V, https://wavesim.readthedocs.io]. The `wavesim` package is MIT-licensed and supports Helmholtz and Maxwell for "non-magnetic and non-birefringent" media [V, https://github.com/IvoVellekoop/wavesim_py].
  - Its backend changed from PyTorch in older docs to CuPy in PyPI 0.2.0a1 (Nov 2025) [V, https://pypi.org/project/wavesim/].
  - There is **no autograd support** [V: none mentioned].
  - gradoscopy should therefore reimplement MBS in torch. It is short, and wavesim can serve as a regression reference.

**Lippmann–Schwinger with Krylov solvers.** Solve (I − G·diag(V))u = u_in with G applied by FFT. Aperiodic convolution needs zero padding. The truncated-Green's-function (Vico–Greengard) discretization is what Pham et al. use for 3D optical diffraction tomography (IEEE TCI 2020, MATLAB/GlobalBioIm) [V, https://github.com/ThanhAnPham/Lippmann-Schwinger]. [R: padding factor about 2 per dimension, so about 8× the field memory for the kernel.] SEAGLE solves the forward LS with Nesterov-accelerated gradient and differentiates by explicit backpropagation through the iterations [V, https://arxiv.org/abs/1705.04281].
- **Trade-off against MBS.** Krylov methods (BiCGSTAB, GMRES) can converge in fewer matrix-vector products than MBS for moderate contrast. They lack MBS's guaranteed monotone convergence, and GMRES stores m Krylov vectors.
- **Recommendation.** Share one "Helmholtz operator" module (G, V, boundaries) and offer MBS as the default iteration, with Krylov iterations as options.

**Boundaries and memory.** MBS needs absorbing layers or wavesim's wrapping correction (`n_boundary` default 8) [V]. As an example, a cell region of 40 × 40 × 20 µm at 100 nm voxels (about 4 per λ_m, which is enough for a spectral method [R]) is 32 M voxels × 8 B = 256 MB per field. MBS keeps about 3–4 fields, and the vector version triples that. So the total is 1–3 GB with one 3D FFT pair per iteration. My estimate is 1–5 s per illumination on a current GPU [?]. That is fine for validation and small datasets, and too slow as a default training-data generator.

**Differentiability.** Never unroll the iterations: 200 iterations × 256 MB is 50 GB. Wrap the solve in an `autograd.Function` with **implicit differentiation**.
- The Helmholtz operator A is complex-symmetric (reciprocal), so the adjoint solve A^H·λ = g equals conj(A⁻¹·conj(g)). That is the *same* solver on the *same* (absorbing) medium, so MBS convergence is preserved.
- The gradient with respect to V is then a pointwise product of the forward and adjoint fields. One extra solve per backward pass.

**FDFD** (e.g. ceviche) is a sparse Yee-grid system. It is excellent in 2D but needs strong preconditioners in 3D. It adds nothing over MBS/LS for microscopy and is not recommended.

### 2.7 FDTD (V5)

FDTD is the universal method: broadband, dispersive, nonlinear and plasmonic materials. It is also the most expensive.
- **Memory.** Accuracy needs about 15–20 cells per λ in the densest material. A 40 × 40 × 20 µm box at 20 nm is 4×10⁹ cells × 6 components × 4 B ≈ 96 GB before materials and PML. For whole-cell microscopy FDTD is out of scope. It is appropriate for sub-5-µm nanophotonic samples such as plasmonic antennas and metasurface substrates.
- **Differentiable options.**
  - FDTDX (JAX) computes gradients through time-reversibility without storing fields per step and handles billions of cells [V, https://github.com/ymahlau/fdtdx; JOSS https://www.theoj.org/joss-papers/joss.08912/10.21105.joss.08912.pdf].
  - Meep and Tidy3D offer adjoint methods [R].
  - flaport/fdtd has a PyTorch backend but is not built for adjoint efficiency [V, https://github.com/flaport/fdtd].

**Recommendation.** Offer an *offline adapter* that imports near-to-far-field results or T-matrices produced by external FDTD. Do not build a native FDTD.

### 2.8 Lorenz–Mie, homogeneous and stratified (A1)

**Formulation.** a_n and b_n come from Riccati–Bessel functions ψ_n and ξ_n. The far-field amplitudes are:
- S1(θ) = Σ (2n+1)/(n(n+1))·(a_n·π_n + b_n·τ_n)
- S2(θ) = Σ (2n+1)/(n(n+1))·(a_n·τ_n + b_n·π_n)

The angular functions π_n and τ_n follow simple recurrences in cos θ. The truncation is N_max ≈ x + 4.05·x^{1/3} + 2 (Wiscombe) [R]. That gives about 28 terms for a 1 µm-radius bead and about 180 for a 10 µm droplet.

**Stability.** The stable algorithm uses logarithmic derivatives D_n(mx), computed by downward recurrence or a Lentz continued fraction, for complex m. ψ_n(x) is computed upward only while n < x.
- Stratified spheres use the Yang (2003) or Peña & Pal (2009) recurrences. PyMieDiff cites Peña & Pal. [V, https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html]
- Large x wants float64. Consumer GPUs have weak FP64 throughput, but Mie is tiny, so this does not matter.

**Torch constraints.** `torch.special` provides only order-0/1 real-argument Bessel functions: `bessel_j0/j1/y0/y1`, `modified_bessel_i0/i1/k0/k1`, the scaled K variants, `spherical_bessel_j0` and `airy_ai`. There is no arbitrary order and no complex argument [V, https://docs.pytorch.org/docs/2.11/special.html]. `airy_ai` has `not_implemented` backward in `derivatives.yaml` [V]. So Mie must be written from recurrences in elementwise complex torch ops. Autograd through the recurrences is fine.

**Prior art.** **PyMieDiff** (Jackson, De Liberato, Muskens, Wiecha; APL Photonics 11, 046114, 2026) is exactly this: autograd-compatible spherical Bessel and Hankel functions, multilayer coefficients, cross sections, angular scattering and near fields, vectorized over orders, wavelengths and angles [V, https://arxiv.org/abs/2512.08614].
- It is **GPLv3** [V, https://github.com/UoS-Integrated-Nanophotonics-group/MieDiff]. If gradoscopy is MIT/BSD like DeepTrack2, do not vendor it. Implement independently and use it only as a test oracle, in an optional test dependency.
- Its authors note GPU performance is "memory transfer bound" and only pays off at thousands of concurrent evaluations [V]. The lesson is to batch all particles and all wavelengths in the scene into one call.

**Output and embedding.** Mie's natural output is the far-field amplitude matrix. It maps exactly onto the collection pupil: for each pupil sample (k_x, k_y), compute θ and φ relative to k_in, form (E_θ, E_φ), rotate to Cartesian, and multiply by the position phase. That is the exact angular spectrum within the NA, which the imaging model then consumes. HoloPy's `MieLens` theory, "exact scattering from a sphere imaged through a perfect lens", is a validated precedent [V, https://holopy.readthedocs.io/en/master/reference/holopy.scattering.theory.html]. DeepTrack's "hybrid" mode does roughly this today [V].

**Validity.**
- Exact for an isolated sphere in a homogeneous medium.
- Not exact near a coverslip, where the interface reflects the scattered field back onto the particle. Mie–interface coupling is significant within about λ of the interface. [R]
- Illumination that is not a plane wave needs beam-shape coefficients, or superposition over the plane-wave components of the illumination, which costs N_components × Mie evaluations. That is cheap anyway.

**Batching.** Particles with different sizes have different N_max. Pad to the largest N_max with masks. The discrete N_max is non-differentiable but harmless, because the dropped terms are negligible by construction.

### 2.9 T-matrix (A2)

**Formulation.** The T-matrix maps regular-VSWF coefficients of any incident field to outgoing-VSWF coefficients: [p;q] = T·[a;b]. For a sphere T is diagonal and equals the Mie coefficients. Shape-specific methods:
- Spheroids and finite cylinders use EBCM/null-field methods (Mishchenko's codes; NFM-DS). Ill-conditioning grows with aspect ratio and x, and practical codes use extended precision for large x. [R]
- Orientation is applied by Wigner-D rotation. [R]
- Multiple particles are coupled with translation-addition theorems in a Foldy–Lax system of size N_p × 2L(L+2). [R]

**Libraries.**
- **SMUTHI** couples particle T-matrices (spheres, spheroids, finite cylinders) with a scattering-matrix treatment of *planarly layered media* [V, https://arxiv.org/abs/2105.04259]. That is exactly what particles on a coverslip, as in iSCAT, need.
- MSTM (Mackowski) handles large sphere clusters. [R]
- `treams` is **not differentiable**, per the TorchGDM paper's note that its Mie/T-matrix modules rely on treams, "which does not support automatic differentiation" [V, https://arxiv.org/html/2505.09545v1].
- I found **no production-grade differentiable torch T-matrix library** [V: searched; nothing found].

**Architectural value.** The T-matrix is the ideal *exchange currency for compact objects*. Any solver that can compute an object's response to several incident fields can tabulate a T-matrix once, cache it and reuse it for all orientations, positions and illuminations. There is also a community proposal for a standard HDF5 T-matrix data format (Asadova et al., JQSRT 2025) [R/?].

**Recommendation.**
- v2: multi-sphere Foldy–Lax on top of the Mie core, which is differentiable because it is just linear algebra.
- v3: an importer for precomputed T-matrices of non-spherical particles. Gradients with respect to shape are unavailable for those; gradients with respect to position, orientation and illumination remain.

### 2.10 Discrete dipole approximation / Green's dyadic method (A3)

**Formulation.** The particle's volume is discretized into N polarizable dipoles, using lattice-dispersion-relation or filtered polarizabilities, and (I − αG)p = αE_inc is solved.

**Validity.** \|m\|·k·d ≲ 0.5–1, i.e. about 10 dipoles per λ inside the material. Accuracy degrades for \|m\| ≳ 2. [R: Draine & Flatau; ADDA manual]

**Two implementations.**
- **FFT-accelerated iterative (ADDA, DDSCAT).** O(N) memory, O(N log N) per matrix-vector product, handling up to 10⁸–10⁹ dipoles on clusters [R].
- **Dense (TorchGDM).** LU inversion in PyTorch with O(N²) memory, which "limits the possible size to around 10,000–15,000 dipoles" [V, https://arxiv.org/html/2505.09545v1]. It runs on GPU with autograd, costing about 2× the time. It supports 3D and 2D homogeneous environments only, with no layered media yet [V].
  - Its standout idea is **multi-scale hybridization**: fully discretized structures coupled to *effective electric and magnetic dipole models* of other scatterers. The effective models are derived from Mie coefficients or fitted by pseudo-inverse to full simulations [V]. That is precisely the "mixed sample" pattern in §5.
  - License not verified [?].

**Relevance.** DDA beats MBS/LS for compact, high-contrast, arbitrary-shape particles in a large empty space, because only the particle is discretized. Its adjoint is trivial because the interaction matrix is symmetric.

**Recommendation.** Interoperate: accept TorchGDM effective-dipole or discretized results through the point-dipole and T-matrix ports. Do not reimplement in v1.

### 2.11 Geometric ray tracing (G1)

**Validity.** Large, smooth objects with x ≳ 10²–10³ and radius of curvature ≫ λ. Diffraction and resonances are lost. [R]

**Use cases.** Big droplets, embryos, organoids and cleared tissue acting as lenses; capillaries and microfluidic walls; tilted or curved coverslips.

**Coupling to wave methods.** The optical path length along refracted rays gives an eikonal phase screen, a "bent-ray projection approximation". That screen feeds the wave engine.

**Differentiable tools.** Mitsuba 3/Dr.Jit (interop via DLPack) and PyTorch lens-design tracers exist [R].

**Recommendation.** Later-stage and optional. The WPM mode (§2.5) covers much of the same regime in wave form.

### 2.12 Turbid media: Monte-Carlo radiative transfer and diffusion (G2)

**Model.** Photons random-walk with parameters μ_s, μ_a, anisotropy g (Henyey–Greenstein) and n. The output is ensemble-averaged intensity with no phase or speckle.

**Tools.** MCX is GPU-accelerated, "hundreds to a thousand times faster" than single-threaded CPU Monte Carlo, with the Python binding `pmcx` (CUDA-only) [V, https://github.com/fangq/mcx, https://pypi.org/project/pmcx/]. It is not differentiable [V: nothing found]. Differentiable Monte Carlo exists in graphics (path replay, Mitsuba) [R].

**Role in microscopy.**
- The ballistic signal is attenuated as exp(−z/l_s), with l_s ≈ 50–200 µm in tissue at visible to NIR wavelengths [R].
- Scattered light produces diffuse background and haze. Speckle and coherent effects need the V3/V4 wave rungs.

**Recommendation.**
- v1: a phenomenological, differentiable "turbid slab" operator. It attenuates the ballistic field or intensity and adds a background equal to the excitation convolved with a broad kernel, with the kernel width from the diffusion approximation.
- Later: MCX as a non-differentiable offline background generator.
- Always label it incoherent.

### 2.13 Emitters: sources inside the sample (E)

Fluorescence breaks the "incident field in, exit field out" picture twice. The source is *inside* the domain, and emitters are mutually *incoherent*.

**Fidelity sub-ladder:**

- **E0.** Intensity PSF convolution with a scalar or Gaussian PSF, one PSF per depth plane. This is DeepTrack's current approach [V].
- **E1.** Vectorial Richards–Wolf PSF with a fixed or freely rotating dipole, Gibson–Lanni index-mismatch aberration, apodization and Fresnel transmission. The EPFL `psf_generator` is PyTorch and autograd-compatible, with a unified Richards–Wolf framework [V, https://github.com/Biomedical-Imaging-Group/psf_generator; https://arxiv.org/abs/2502.03170].
- **E2.** Dipole near an interface: Hellen–Axelrod or Enderlein angular spectra, including supercritical-angle fluorescence collected at NA > n_m [R].
  - This is the *same* computation as a coherent Rayleigh scatterer near the coverslip in iSCAT, so share the code.
- **E3.** Emitters inside a weakly or moderately scattering volume, through a multislice engine. Three strategies:
  - **(a) Per-emitter propagation.** Inject the emitter's spherical-wave angular spectrum at its slice, march through the slices above it, and sum intensities. Cost is one partial march per emitter. Emitters are batchable on the GPU, but memory scales with the batch.
  - **(b) Random-phase ensemble.** All emitters radiate together with i.i.d. random phases. Run K coherent marches and average the intensities. This is unbiased for the incoherent sum, with variance ∝ 1/K. chromatix's `fluorescent_multislice_thick_sample` does this with a `num_samples` parameter [V]. For training data, the Monte-Carlo noise can be a feature, but it must be controllable.
  - **(c) Reciprocity.** The field in pupil direction k from a dipole p at r_e is ∝ p·E^{(−k)}(r_e). E^{(−k)} is the field inside the sample for a plane wave launched from the detection side. One solve per pupil sample gives the PSFs of *all* emitters at once. This is the right tool for dense emitter fields in full-wave solvers (V4) or with coarse pupil sampling. [R]
- **E4.** The excitation side. Solve the illumination (widefield, TIRF evanescent, light-sheet, confocal spot) with any rung, then map it to emitter brightness: linear, saturating, or ∝ I² for two-photon excitation.
  - The same two-stage **excitation → source → emission** structure covers Raman (incoherent, shifted λ) and SHG/CARS (coherent sources P ∝ χ⁽²⁾E²). The pipeline must therefore support an *intermediate typed quantity* (source density or dipole list) and a *wavelength change* between stages.

---

## 3. Cross-cutting differentiability and GPU concerns

1. **Iterative solvers (V4, A2 Foldy–Lax, A3).** Implicit differentiation through a custom `torch.autograd.Function` is mandatory. The backward pass costs one adjoint solve of the same operator, and memory is O(a few fields). Unrolling is only acceptable at fewer than about 10 iterations. Expose tolerances as fidelity knobs. For consistent gradients, the backward tolerance should be at least as tight as the forward one.
2. **Marching solvers (V3).** Use native autograd with segment checkpointing. Reversible marching is an option for lossless samples. For linear rungs (Born), write custom adjoints to avoid storing the slice stack.
3. **Geometry to grid.** Hard voxelization kills gradients with respect to position, size and shape. Two alternatives:
   - Analytic Fourier-domain rasterization for primitives: sphere, ellipsoid and cylinder form factors, then the inverse FFT. It is band-limited, alias-free and exactly differentiable.
   - Soft occupancy from a signed distance, σ(−sdf/τ) with τ ≈ Δx/2.
   - Both feed every voxel rung. Keep rasterization a separate, cached, differentiable stage.
4. **Discrete fidelity knobs** (method choice, N_max, slice count, iteration count, number of emitter realizations) are not differentiated. That is fine, but the *resampling* between grids must stay differentiable (Fourier resampling or interpolation).
5. **Precision.** complex64 suffices for FFT solvers; wavesim itself uses complex64 [V]. Use float64 inside Mie, T-matrix and translation coefficients for x ≳ 50, then cast down.
6. **Batching ragged scenes.** Per-image particle counts vary. Use padded particle tensors with masks for the analytic rungs and fixed-size volumes for the voxel rungs, so one kernel launch serves a whole training batch.
7. **Learned surrogates as rungs.** Neural operators or "learned Born series" [V exists: https://pubs.aip.org/asa/jel/article/3/5/052401/2887637/A-learned-Born-series-for-highly-scattering-media] can be trained from gradoscopy's own V4 outputs. They then slot in as a fast rung with the *same interface*. This is where the whole becomes more than the sum of its parts: the reference solvers manufacture the fast ones.

---

## 4. Assessing the candidate unifying interface

**Candidate:** *each solver maps (incident field or incident angular spectrum, sample) → outgoing field on an exit plane, or its angular spectrum; a shared imaging model takes over from there.*

**What is right about it.** It covers transmitted-light brightfield, phase contrast, DIC, holography, darkfield and ODT, which is most DeepTrack use. It cleanly separates the specimen from the optics.

The angular spectrum *on the collection pupil grid* is the correct currency. The objective only accepts propagating waves inside its NA, so a solver needs to produce only those. It can do so at whatever internal grid suits it: Mie analytically, BPM on its lateral grid, MBS in its box. The format is grid-agnostic and exactly what a Richards–Wolf imaging model consumes.

**Where it leaks, and the fix for each:**

1. **Multiple scattering between objects handled by different solvers.** Superposing independently computed fields ignores coupling. Examples:
   - a bead inside a cell, whose image is aberrated by the cell;
   - dense colloids;
   - a particle near a membrane.
   → Fix: solvers optionally expose `field_at(points)` or a local expansion, and accept arbitrary incident fields. A composer can then chain or iterate them (§5).
2. **Reflection and epi geometries** (iSCAT, RICM, reflectance confocal). These need a *backward* port. The substrate interface is at once part of the sample, the source of the reference wave, and a mirror that couples back to the particle. Forward-only rungs (V1, V3) simply cannot produce it.
   → Fix: generalize the exit plane to **two ports** (top and bottom planes) and treat every solver as a (possibly partial) **scattering matrix** in the angular-spectrum basis.
   - Planar interfaces are then analytic, diagonal-in-k⊥ Fresnel S-matrices.
   - Stacks compose with the Redheffer star product, which is the SMUTHI and RCWA approach [V: SMUTHI's layered-media S-matrix formalism].
   - Forward-only solvers have zero reflection blocks.
   - The number of inter-block bounces is a fidelity knob (0, 1, or converged).
   - The operator stays matrix-free: apply it to vectors, never form it.
3. **Evanescent illumination (TIRF, TIR scattering).** The incident field is not a propagating wave on an upstream plane: k_z is imaginary and it decays away from the glass.
   → Fix: illumination objects provide both `angular_spectrum()`, which allows complex k_z, and `field_at(points)`.
   - Born and dipole rungs just evaluate the field in the volume.
   - Multislice cannot march an evanescent input meaningfully. Inject it as a volume source, Born-style.
   - Mie needs beam-shape coefficients for inhomogeneous plane waves [R/?; the plane-wave VSWF expansion is analytic in angle, so complex angles should work — verify].
   - In addition, with oil objectives at NA > n_m, the *collected* spectrum includes components that are evanescent in water but propagating in glass. The pupil basis must be defined in the immersion medium, with the glass interface explicitly in the S-matrix stack.
4. **Sources inside the sample** (fluorescence, and any emitter). Covered in §2.13.
   → Fix: an explicit **source port** in the solver input, taking dipole lists or source density. Coherence is declared per source set.
5. **Partial coherence and incoherence** (Köhler/LED illumination, broadband light, emitters). The interface yields one coherent field *per mode*, and intensities add at the detector. Cost multiplies by the number of modes.
   → Fix:
   - A batch dimension for coherent modes in every solver, so the GPU amortizes the modes.
   - A capability flag `linear_in_sample` (Born). It lets the imaging model use transfer-function shortcuts (TCC/WOTF): one 3D convolution instead of N_modes solves.
   - A capability flag `linear_in_incident`, which is true for every rung except nonlinear sources. It enables mode compression, e.g. solving only for the SVD modes of the illumination.
6. **Polarization.** Scalar rungs silently drop vector effects, such as high-NA focusing and depolarization by beads.
   → Fix: the currency always carries two transverse components (or three for internal fields). Scalar solvers declare `scalar=True` and are lifted by an assumed polarization. The composer warns when vector and scalar results mix at high NA.
7. **Near-field detection and very small working distances** (NSOM, SAF). Evanescent parts of the exit field matter. This is covered by the immersion-basis fix in item 3, plus an option to keep |k⊥| up to NA_imm·k0.
8. **Inelastic and nonlinear processes.** There is a wavelength change and a non-field intermediate.
   → Fix: a pipeline of stages with typed intermediates (Field, Intensity, SourceDensity, EmitterList). This is not a single solver call.
9. **Broadband and dispersion.** One solve per wavelength sample, with a batch dimension over λ. Only FDTD gets broadband for free.

**Verdict.** The candidate is the right *default* contract. It becomes safe once it is widened to:
- two ports instead of one exit plane;
- an excitation that may be a plane-wave batch, a field on a plane, a field evaluable in the volume, or internal sources;
- capability flags (linearity, vectorial, ports, accepted excitations, reflection support);
- optional coupling hooks (`field_at`, `respond_to(local_expansion)`).

HoloPy's `ScatteringTheory` offers a close precedent at the small-object level:
- `can_handle(scatterer)`;
- `raw_fields(pos, scatterer, medium_wavevec, medium_index, illum_polarization)`;
- `raw_scat_matrs`;
- a `Lens` wrapper that makes any theory imaging-aware [V, holopy docs].

gradoscopy should generalize that pattern to volumetric solvers and to composition.

---

## 5. Mixed samples and solver composition

A scene is a background plus objects, each with a geometry representation and a material. Common mixtures:
- Mie spheres or gold nanoparticles inside a voxelized cell;
- beads on a coverslip under a cell layer;
- emitters inside tissue.

Five composition policies, in increasing fidelity and cost:

- **C0 — Independent superposition.** Each object sees the bare illumination. Coherent fields add at the pupil; emitters add in intensity. Cost is additive. Valid when objects are sparse and weak relative to their separation. This is DeepTrack today [V].
- **C1 — Ordered forward chaining ("distorted-wave" injection).** Sort objects and slabs by z. March the volumetric engine (V3), and at each analytic object's plane do the following:
  1. Read the local incident field at the particle from the marched field. In the local-plane-wave approximation this is amplitude, phase and direction from the phase gradient. For a better approximation, decompose a small window into plane waves.
  2. Evaluate the Mie, T-matrix or dipole response to that field.
  3. Convert its outgoing field to the angular spectrum *at that slice plane*. This is exact for the half-space beyond the particle.
  4. Add it to the marched field and continue marching.

  This captures cell-induced aberration and shadowing of bead images, which is the main physical gap for tracking inside cells. It costs roughly one extra FFT per injection plane. It neglects back-reaction and backscatter.

  Emitters use the same injection hook with incoherent bookkeeping (§2.13, E3a/E3b). Point dipoles below the voxel scale inject the same way. This is "the particle as a source term", which is what TorchGDM's multi-scale approach does inside a volume-integral solver [V].
- **C2 — Coupled iteration (Foldy–Lax / domain decomposition).** The total field satisfies u = u_in + Σ_j S_j[u|_j]. Solve it with GMRES or fixed-point iteration over solver responses. Fields are translated between objects by angular-spectrum propagation in the homogeneous background, or by VSWF translation between compact objects. Differentiate implicitly. Cost is roughly iterations × Σ solver costs. Use it for dense colloids, particle–interface coupling and particle–cell near contact.
- **C3 — Monolithic rasterization.** Voxelize everything and run the top volumetric rung (V4). This is the simplest correct option, limited by resolution: fine for 1 µm beads, impossible for 20–80 nm nanoparticles.
- **C3′ — Hybrid monolithic.** C3 for resolvable objects, with sub-voxel particles entering as coupled point dipoles in the MBS/LS iteration. Their polarizability comes from Mie a₁ and b₁ with radiative correction. This is DDA inside MBS, and gives the highest practical fidelity for cells with nanoparticle labels.

**Worked example: Mie spheres inside a voxelized cell, brightfield or holography.**

| Fidelity setting | Policy | Cost (256² × 128 grid, 5 beads) | What it captures |
|---|---|---|---|
| draft | C0: BPM(cell) + Σ Mie(bead) at the pupil | ≈ BPM | bead and cell images, no mutual effects |
| standard | C1: SSNP march with Mie injection at the bead planes | ≈ SSNP + 5 small FFTs | cell aberrates the bead images; beads shadow and aberrate the cell below |
| high | C2: SSNP/Mie iteration, 2–3 sweeps | 2–3× standard | bead–cell back-coupling (weak) |
| reference | C3/C3′: MBS vector, beads voxelized or as dipoles | seconds | everything, including reflections |

**The planner.** Composition should be chosen by a small **planner**, not hand-wired by users. Every solver implements `validity(scene_group, excitation) → report` and `cost(...) → estimate`. Validity estimators to implement:
- phase delay φ and the Slaney criterion for Born;
- Δn/n_m for Rytov;
- maximum scattering or illumination angle and Δn·Δz for BPM;
- x and "isolated?" (distance to other objects and interfaces, relative to size and λ) for Mie;
- \|m\|kd for DDA;
- the ε-based iteration estimate for MBS.

User-facing fidelity presets (`draft`, `standard`, `high`, `reference`) map to policies. Per-object overrides remain possible. The planner emits a human-readable fidelity report and warns when a forced choice is out of regime.

---

## 6. Implications for gradoscopy's architecture

1. **Adopt one exchange currency.** Use a **vector angular spectrum on explicit ports** (top and bottom planes, with the immersion-medium basis optional), batched over coherent modes, wavelengths and scene instances. Provide exact converters: field on a plane ↔ spectrum (FFT), spectrum ↔ pupil grid, spectrum ↔ VSWF expansion about a point. These three converters are the glue that lets every rung interoperate.

2. **Build four engines, not twenty solvers.** Each engine hosts many methods:
   - **Z-march engine.** A (diffract, interact) step pair with injection hooks, checkpoint and reversible modes. Hosts thin screen (one step), BPM, tilt-corrected BPM, WPM, MLB, SSNP, Jones/polarized multislice, and slice-wise Born/Rytov.
   - **Helmholtz-operator engine.** FFT Green's function (damped, truncated, vector-projected), potential application, absorbing/wrap boundaries, and implicit-differentiation wrapper. Hosts Born, Born/Neumann series, MBS/CBS (scalar and vector), LS-Krylov, and embedded point-dipole coupling (C3′).
   - **Multipole engine.** Recurrence-based Riccati–Bessel functions (float64), Mie and stratified coefficients, far-field amplitudes to the pupil, near fields, point-dipole polarizabilities; later Foldy–Lax clusters, translation theorems and a T-matrix importer.
   - **Emission engine.** Dipole angular spectra in homogeneous and interface geometries, shared with iSCAT dipoles. Incoherent bookkeeping: per-emitter, random-phase ensemble, and reciprocity modes. Excitation → source mapping.

3. **Specify the solver contract up front, even if v1 fills only part of it:**
   ```python
   class InteractionSolver(Protocol):
       caps: Capabilities            # ports={T,R}, accepts={plane_waves, plane_field, volume_field, sources},
                                     # vectorial, linear_in_incident, linear_in_sample, supports_coupling
       def validity(self, group, exc) -> ValidityReport: ...
       def cost(self, group, exc) -> CostEstimate: ...
       def solve(self, group, exc: Excitation) -> PortFields: ...   # batched over modes, λ, instances
       # optional coupling hooks
       def field_at(self, points) -> Tensor: ...
       def respond(self, local_expansion) -> OutgoingExpansion: ...
   ```
   Iterative solvers must wrap `solve` in implicit differentiation. Linear solvers should expose `apply`/`adjoint` so imaging-level shortcuts (TCC/WOTF, mode compression) become possible.

4. **Separate geometry from physics.** Scene objects hold analytic primitives, SDFs, meshes (later), voxel fields or point sets. Differentiable **representation adapters** convert them per solver:
   - analytic → Fourier-domain rasterization, band-limited and exact;
   - analytic → soft occupancy;
   - sphere → Mie coefficients;
   - small particle → dipole.

   Solvers never see user-level shapes directly.

5. **v1 scope, which covers most DeepTrack use cases:**
   - P0 dipole scatterers (Mie-dressed α);
   - V1 thin screen (scalar and Jones);
   - V2 Born and Rytov (slice-wise, T and R ports);
   - V3 BPM (tilt-corrected) and SSNP;
   - A1 Mie (homogeneous and stratified, own MIT-licensed implementation, validated against PyMieDiff and scattnlay/miepython in tests);
   - E0–E2 emitters (vectorial PSF with dipole orientation and interface);
   - composition C0 and C1 (injection into the march);
   - a phenomenological turbid-slab operator;
   - the planner with validity reports and four presets.

   Ports: the T port is complete. The R port is supported by Born, dipole and Mie rungs, with an analytic coverslip Fresnel S-matrix. That already enables a physically sensible iSCAT at the P0/A1 rungs.

6. **v2:**
   - V4 MBS scalar and vector in torch with implicit adjoint, benchmarked against wavesim (MIT);
   - LS-Krylov as an alternative iteration;
   - MLB and WPM modes;
   - C2 coupled iteration and C3′ hybrid dipole-in-volume;
   - multi-sphere Foldy–Lax;
   - Redheffer star-product port composition for stacks;
   - E3c reciprocity emitters;
   - learned-surrogate rung trained from V4.

7. **v3 and adapters, never core:**
   - T-matrix import for non-spherical particles (SMUTHI/treams-generated, standard format);
   - TorchGDM interop for DDA and effective dipoles;
   - FDTD far-field import;
   - ray-traced eikonal screens;
   - MCX backgrounds;
   - nonlinear and inelastic stages (SHG, Raman) on the excitation → source → emission pipeline.

8. **Build a validation matrix early, and make it a CI asset.** Every rung is checked against the rung above in the regime where both hold:
   - Born vs Mie for small spheres;
   - BPM and SSNP vs MBS for voxelized beads and cell phantoms;
   - Mie vs voxelized MBS;
   - dipole vs Mie at x < 0.3;
   - thin screen vs BPM for thin objects.

   These tests also produce the error-versus-regime tables that calibrate the planner's validity thresholds, instead of relying on literature rules of thumb.

9. **Don't take on GPL code or non-differentiable dependencies in core.** PyMieDiff is GPLv3. treams is non-differentiable. SSNP-IDT is PyCUDA. wavesim has no autograd. Use them as oracles, not as foundations.

10. **Settle the hard cases in the design now**, because they are the leaks that are expensive to retrofit:
    - two ports and the substrate S-matrix (iSCAT, TIRF);
    - an internal-source port with explicit coherence (fluorescence);
    - a coherent-mode batch dimension with linearity flags (partial coherence);
    - `field_at` and `respond` coupling hooks (mixed samples).

    Even if only C0/C1 and T/R-by-Born/Mie are implemented in v1, the contract must already express these so that higher rungs slot in without API breaks.
