# 05 — Modelling the optical path and imaging system (illumination → sample → collection → detector)

Research brief for the gradoscopy architecture plan. Scope: light microscopy only. Sample representation and light–sample interaction are covered in other briefs; here they appear only as an interface ("exit field" or "emitters").

Evidence tags: **[V: url]** means I checked it against the source during this research. **[R]** means it comes from memory or textbook knowledge and I did not re-check it. **[H]** means an engineering heuristic or estimate of mine, to be validated in code.

---

## 0. TL;DR

1. The proposed unifying abstraction ("an incoherent sum over mutually incoherent modes of |coherent image of mode|², then detection") is **exact for every *linear* optical stage**. Mathematically it is Wolf's coherent-mode (Mercer) expansion of the cross-spectral density [R]. It is **not enough on its own**, for three reasons:
   - Fluorescence, confocal, two-photon (2P), light-sheet, SIM and TIRF are **two-stage** processes. A coherent excitation stage produces |E|², a pointwise and possibly nonlinear map turns that into emitter strengths, and an incoherent emission stage follows.
   - The mode axis has two meanings. Some modes are **summed within an exposure** (source points, wavelengths, emitters, time samples). Others are **acquisition axes that produce separate frames** (LED index in Fourier ptychography, SIM phase, scan position, focus step, channel).
   - A few modalities break the assumption that emitters are incoherent (SHG/CARS/SRS). Others are nonlinear (2P, STED, saturated SIM). Both need an explicit nonlinear intensity→source link, or they are out of scope.
2. Two **collapse theorems** give almost all of the speed at the cheap fidelity levels. They should be encoded as planner rules, not left for users to rediscover:
   - (a) **Shift-invariance collapse.** If the coherent map is the same for every emitter up to a translation, the sum over emitter modes becomes a convolution of the emitter density with the intensity PSF, done once per z-slice, with an OTF or low-rank variant.
   - (b) **Thin-sample collapse.** If tilting the illumination only shifts the sample's spectrum (thin or projection sample), the sum over source points becomes the Hopkins TCC. That can be truncated to K SOCS kernels, or linearised into weak-object transfer functions (WOTF) for weak samples.
3. Transmitted-light brightfield, darkfield, phase contrast, oblique, DPC, Hoffman, Rheinberg, Fourier ptychography (FPM) and (with Jones calculus) DIC are all **one physical model: a source distribution S(k_s, λ, pol) plus a pupil filter P(k)**. They should be presets of a single Abbe/Hopkins imaging kernel, not separate classes with separate code paths. iSCAT and holography add one extra coherent term: the reference field.
4. First-class, swappable components: **Illumination (source)**, **Pupil** (built from pupil modifiers, with scalar and vector/Jones variants), **PSF/transfer-function model** (derived views of the pupil: CTF, PSF, OTF, TCC/SOCS, WOTF, analytic Gaussian, measured/learned), **DetectionPath** (magnification, filters, channel splitters), and **Camera**. Solvers, sampling grids, padding, CZT versus FFT, caching and chunking should be internal and chosen by a **planner** driven by a fidelity policy.
5. Differentiability imposes concrete design rules:
   - Use soft-edged apertures, so that NA, condenser NA, pinhole size and ring radii all get gradients.
   - Use continuous-coordinate transforms (CZT or matrix DFT), so that pixel size and magnification get gradients.
   - Derive tensor shapes from *declared parameter bounds*, never from parameter values.
   - Give every optical parameter a batch dimension, so each sample in a batch can have its own optics.
   - The detector should expose `expected`, `sample` (with reparameterised approximations) and `log_prob` (the exact NLL, for fitting to real data).

---

## 1. The unifying abstraction and where it breaks

### 1.1 Formalism

Take a quasi-monochromatic spectral component at frequency ω. A statistically stationary field has a cross-spectral density W(r₁, r₂, ω). Any such W can be expanded as W = Σ_m λ_m(ω) φ_m(r₁) φ_m*(r₂), with λ_m ≥ 0 and orthonormal modes φ_m (Wolf 1982, coherent-mode representation) [R]. A linear optical operator L_ω, which covers free-space propagation, pupils, linear samples and polarisation optics, acts on each mode independently. The intensity after the system is therefore

  I(r) = ∫ dω Σ_m λ_m(ω) ‖(L_ω φ_m)(r)‖²  (the norm sums over polarisation components)

and the detector turns I(r) into counts. The decomposition does not have to be orthogonal. Any representation W = Σ_m w_m ψ_m ψ_m* works, for example the non-orthogonal set of tilted plane waves in Abbe's method. That is what makes the idea practical: **the mode set is simply an extra batch axis M with weights w_m**, the linear operators are vectorised over M, and a reduction Σ_m w_m|·|² is taken at the end.

Common mode sources are:

| Mode family | Why the modes are mutually incoherent | Typical count |
|---|---|---|
| Köhler condenser source points | spatially incoherent extended lamp or LED | 1 (coherent) to 20–500 (Abbe), or K = 5–30 (SOCS) [H] |
| Wavelength bins | different temporal frequencies do not interfere in a time-integrating detector | 3–10 (fluorescence), 5–30 (broadband coherent imaging) [H] |
| Polarisation states | unpolarised light = two orthogonal incoherent states | 2 |
| Emitters or voxels (fluorescence) | spontaneous emission is incoherent between molecules | 1 to 10⁶ (hence collapse theorem (a)) |
| Dipole orientations | rotational averaging within the exposure | 3 (isotropic) or 6 second-moment bases |
| Time samples | camera integrates intensity (motion blur, rotating diffuser) | 1–50 |
| Diffuser/speckle realisations | rotating diffuser, multimode-fibre scrambling | 1–100 |

### 1.2 Collapse theorems (the fast paths)

**(a) Shift-invariant emission.** Suppose emitter e at position r_e produces the field h(r − r_e; z_e), and h depends on z but not on lateral position. Then Σ_e w_e |h(r − r_e; z_e)|² = Σ_z (c_z ⊛ |h_z|²)(r). That is one 2D convolution per z-slice with the intensity PSF, and it is exact. This is what DeepTrack2's `Fluorescence` does today: for each non-empty z-slice it builds `psf = |IFFT(pupil·defocus)|²`, `otf = FFT(psf)` and multiplies [V: https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/optics.py]. microsim (Talley Lambert) does the same with `fftconvolve(f_truth, summed_psf)` after summing wavelength-weighted PSFs per fluorophore [V: https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/modality/_simple_psf.py].

**(b) Thin-sample partially coherent imaging (Abbe → Hopkins → SOCS).** For a thin transmittance t(r) under Köhler illumination with source S(k_s):

- Abbe: I(r) = ∫ S(k_s) |F⁻¹{P(k) t̂(k − k_s)}(r)|² dk_s. One object FFT plus one inverse FFT per source point.
- Hopkins: I(r) = ∬ TCC(k₁,k₂) t̂(k₁) t̂*(k₂) e^{i2π(k₁−k₂)·r} dk₁dk₂, with TCC(k₁,k₂) = ∫ S(k_s) P(k₁+k_s) P*(k₂+k_s) dk_s.
- SOCS: TCC ≈ Σ_{j≤K} α_j Φ_j(k₁)Φ_j*(k₂), so I ≈ Σ_j α_j |F⁻¹{Φ_j t̂}|². That is K coherent systems. Cobb's original SOCS work truncated at about 5–6 kernels in lithography settings [V: http://www-video.eecs.berkeley.edu/papers/ncobb/socs.pdf via search summary].
- **The TCC never needs to be formed.** Stack the shifted, source-weighted pupils as the columns of A, with a_s(k) = √S_s P(k + k_s). Then TCC = A Aᴴ, and a thin SVD A = UΣVᴴ gives the kernels Φ_j = U_j with α_j = σ_j². This is Yamazoe's "stacked pupil shift matrix" [V: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-25-12-3111]. Memory comparison [H]: for a 128×128 frequency grid the full TCC has 16384² ≈ 2.7·10⁸ complex64 entries ≈ 2.1 GB, while A with 300 source points is 16384×300 ≈ 39 MB.
- **Weak-object linearisation (WOTF).** With t = e^{−μ+iφ} ≈ 1 − μ + iφ, the image becomes Î(k) ≈ Bδ(k) + H_μ(k) μ̂(k) + H_φ(k) φ̂(k), where H_μ = −[TCC(k,0) + TCC*(−k,0)] and H_φ = i[TCC(k,0) − TCC*(−k,0)] (my derivation from the Hopkins form; sign conventions vary). This costs a single FFT pair and underlies DPC and QLIPP-style models; waveorder implements this family for 2D/3D phase and polarisation [V (existence, scope): https://github.com/mehta-lab/waveorder; details R]. Valid for |φ|, μ ≪ 1 rad (roughly < 0.3–0.5 rad) [H].
- The PyTorch lithography framework TorchLitho implements both Abbe and Hopkins/SOCS differentiably. It reports cost dropping from O(n⁶) to O(Q·n⁴) and states that Hopkins beats Abbe in time when the optics are fixed [V: https://arxiv.org/html/2409.15306v1].

**The caveat that matters for gradoscopy:** (b) requires that the sample's response to a tilted plane wave is its normal-incidence response shifted in frequency. That holds under the projection approximation. It fails for thick or multiple-scattering samples (multi-slice or Born beyond first order). For those, Abbe with one sample simulation per source mode is the only exact route. There are two cheaper routes for thick samples:
- Sample source modes stochastically: E_{s~S}[|ψ_s|²] = I, so both the image and its gradient are unbiased.
- Use random-phase superpositions: ψ = Σ_s √w_s e^{iθ_s} ψ_s gives E_θ|ψ|² = Σ w_s|ψ_s|² because multi-slice is linear in the incident field. This costs one sample simulation per realisation, but each realisation has speckle contrast of about 1, so the residual contrast falls only as 1/√R [R]. It is acceptable only when heavy shot noise masks the residual.

### 1.3 Which modalities fit

| Modality | Summed modes | Frame axes | Fit | Fast path |
|---|---|---|---|---|
| Widefield fluorescence (incl. PSF engineering, biplane) | emitters × λ × dipole | channels, focus | exact | (a) per-z OTF, per-emitter sprites |
| TIRF / HILO | excitation stage + emitters | — | exact, two-stage | analytic evanescent excitation |
| Confocal / spinning disk / ISM (Airyscan) | emitters | scan position, detector element | exact, two-stage + pinhole integration | effective PSF h_exc·(h_det⊛D) |
| Two-photon | emitters | scan | **only with a nonlinear link** (rate ∝ I_exc²) | effective PSF h_exc(λ₂ₚ)² |
| Light sheet (Gaussian, Bessel, lattice, DSLM) | emitters, sheet-scan/dither times | z-planes | exact, two-stage | h_det(r)·I_sheet(z; x) |
| Linear SIM (2D/3D) | emitters | angle × phase | exact, two-stage | D_j = h_det ⊛ (c·I_j) |
| Nonlinear SIM, STED, RESOLFT | — | — | **needs a nonlinear link**; recommend effective-PSF only | parametric effective PSF |
| Brightfield (Köhler) | source × λ × pol | focus | exact (Abbe) | (b) SOCS or WOTF if thin |
| Darkfield, phase contrast, oblique, DPC, Hoffman, Rheinberg | source × λ | — | exact (source mask + pupil mask) | (b) |
| DIC | source × λ, Jones | bias, shear direction | exact with 2-component fields | (b) + sheared-difference pupil |
| FPM / LED array | per-LED source extent × λ | LED index (or summed if multiplexed) | exact | shifted-spectrum (thin sample) |
| In-line holography | λ × source extent | focus | exact (reference = unscattered wave) | single coherent mode |
| Off-axis holography | λ | — | exact if reference and object share modes | — |
| iSCAT / COBRI | λ (± source extent) | — | exact (reference = reflected or transmitted beam) | — |
| Laser speckle illumination | diffuser realisations | — | exact (stochastic) | random-phase fields |
| Polarisation (LC-PolScope, PTI) | source × pol states | analyser states | exact with vector/Jones fields; needs an anisotropic sample model | WOTF variants (waveorder) |
| SHG / THG / CARS / SRS | — | — | **does not fit**: emitters are phase-locked to the excitation, so emission must be summed coherently | out of scope for v1 |
| FLIM, antibunching, photon timing | — | — | detector time domain | out of scope |

### 1.4 Verdict: the refined abstraction

Replace "every modality = Σ modes |coherent|² then detect" with:

> **An imaging pipeline is a chain of stages. Each stage maps a set of weighted, mutually incoherent coherent modes through a linear operator and reduces them to irradiance (Σ_m w_m‖·‖²). Stages are joined by pointwise *source maps* from irradiance to new mode weights (linear or nonlinear; for example excitation rate ∝ |μ·E|² or ∝ I²). Every mode axis is labelled either *summed* (within one exposure) or *acquisition* (a separate frame). The final irradiance goes to a detector.**

Collapse theorems (a) and (b) are then optimisations the planner applies when a stage declares the needed property (`shift_invariant_laterally`, `thin_sample`, `weak_object`). They are never separate physics.

---

## 2. Illumination

The general description is a source distribution S(k_s, λ, polarisation, t) at the condenser pupil (or back focal plane, BFP), plus an optional spatial envelope in the sample plane. It can be coherent (a few discrete points) or incoherent (a continuous distribution). Concrete cases follow.

**2.1 Coherent tilted plane wave.** U_inc = A e^{i2π k_s·r}, with k_s = n sinθ (cos φ, sin φ)/λ₀. On a periodic FFT grid of side L, k_s must be a multiple of 1/L. Otherwise the wrapped phase ramp creates a seam artefact. The options are:
- snap k_s to the grid, giving an angular error of about λ/(nL cosθ);
- apply the ramp in real space with a tapered window and padding.

DeepTrack2 currently uses a single plane wave per image (`illumination_angle` in `ISCAT`/`Darkfield`) [V: optics.py above].

**2.2 Köhler, partially coherent.** The condenser aperture (disk of radius NA_c/λ, an annulus, or any shape) is the source. The coherence parameter is σ = NA_c/NA_obj. The mutual intensity in the sample plane is J(Δr) = F⁻¹{S}(Δr) (van Cittert–Zernike), so the coherence width is about 0.61λ/NA_c [R]. There are three ways to sample it:
- *Exact on-grid Abbe*: every frequency-grid point inside the condenser disk. That is π(NA_c L/λ)² points, about 7.8·10³ for L = 50 µm, NA_c = 0.5, λ = 0.5 µm.
- *Quadrature*: polar rings or a Fermat spiral with 20–200 points. This is usually converged, because the image varies smoothly with k_s for most samples [H].
- *Stochastic*: K random points per training step. The result is unbiased and the noise disappears under shot noise [H].

Critical illumination (the lamp imaged onto the sample) is Köhler plus a spatial intensity envelope, to good approximation [R].

**2.3 LED arrays and FPM.** LED j at (x_j, y_j) and distance d gives k_j ≈ (x_j, y_j)/(λ√(x_j²+y_j²+d²)). If NA_j < NA_obj the LED gives brightfield; if NA_j > NA_obj it gives darkfield. Each image is |F⁻¹{P(k) t̂(k − k_j)}|² (Zheng, Horstmeyer and Yang 2013 [R]). Several second-order effects are real and worth parameterising, because FPM calibration papers learn them:
- finite LED size (partial coherence per LED);
- spherical wavefront, so the tilt varies across the field of view (FOV);
- LED spectral width of about 20 nm;
- LED position errors.

Multiplexed and DPC patterns are summed over lit LEDs.

**2.4 Lasers and speckle.** With a diffuser or multimode fibre in the condenser pupil: U_inc = F⁻¹{C(k) e^{iφ_rand(k)}}. This is circular-Gaussian speckle with grain size about λ/(2NA_c). N independent realisations integrated in one exposure reduce the contrast to 1/√N [R, Goodman]. Coherent nuisances are the main realism gap in coherent simulators, and should be modelled as a learnable perturbation of U_inc, i.e. a sum of weak tilted or spherical waves. They include:
- dust and etalon fringes;
- Newton rings;
- off-axis ghost reflections.

**2.5 Light sheets.** For a Gaussian sheet from illumination NA_ls:
- waist w₀ ≈ λ₀/(π NA_ls);
- Rayleigh range x_R = π n w₀²/λ₀.
- Example [H arithmetic]: 488 nm, NA 0.05, n = 1.33 gives w₀ ≈ 3.1 µm and x_R ≈ 82 µm.

Other sheet types:
- Bessel: an annular pupil gives a J₀² profile with side lobes.
- Lattice: a coherent sum of discrete k-vectors on a ring, then dithered, i.e. a time-incoherent sum.
- DSLM: the beam is scanned, so the time average is a convolution along the scan axis.

The fast model computes I_sheet(r) analytically or with an angular-spectrum propagation in the sheet's own frame and resamples it into the sample frame. The high-fidelity model propagates through the sample with a beam propagation method (BPM). Biobeam does exactly this with a scalar BPM on the GPU. It also multiplexes 10⁵–10⁶ PSF calculations to capture spatially varying aberrations in tissue [V: https://github.com/maweigert/biobeam; https://arxiv.org/abs/1706.02261]. The sheet frame is rotated 90° from the detection axis, so **illumination and detection need independent coordinate frames**.

**2.6 TIRF (evanescent).** A plane wave with k_∥ = n₁ sinθ/λ₀ > n₂/λ₀ has an imaginary k_z in the sample:
- I(z) = I₀ e^{−z/d}, with d = (λ₀/4π)(n₁² sin²θ − n₂²)^{−1/2} [R].
- Example: λ₀ = 488 nm, n₁ = 1.518, n₂ = 1.33, θ = 66° gives d ≈ 99 nm [H arithmetic].
- Fresnel transmission differs for s and p. p-polarisation carries an E_z component, which matters for dipole excitation.

The angular-spectrum machinery handles this for free if the complex square root is not clipped (evanescent components allowed). HILO is an inclined thin beam, i.e. a tilted sheet.

**2.7 Structured illumination.** SIM excitation is the coherent interference of 2 (2D-SIM) or 3 (3D-SIM) beams from discrete points in the objective pupil: I(r) = |Σ_b a_b e^{i2πk_b·r + iφ_b}|². So SIM illumination is simply "a coherent source made of 2–3 pupil points" pushed through the same machinery. Real systems have limited modulation depth m. Standard acquisition is 3 angles × 3 phases for 2D and 3 × 5 for 3D [R]. With the shift-invariance collapse, frame j is D_j = h_det ⊛ (c · I_j).

**2.8 Focused and scanned beams (confocal, 2P, STED).** The excitation field is the illumination-side PSF, vectorial at high NA and polarisation-dependent, at λ_ex. If it is shift-invariant, scanning is a translation, and the modality collapses to an effective PSF (§3.7). Otherwise it is one excitation computation per scan position, which is only feasible for small scans or as a validation reference.

**2.9 Polychromatic sources.** Represent the spectrum as N_λ bins with weights. Coherence length is l_c ≈ λ²/Δλ. For example, an LED at 530/30 nm gives about 9.4 µm, and a halogen lamp through a 100 nm filter gives about 3 µm [H arithmetic]. Spectral sampling matters most when interference path differences approach l_c: holography far from focus, thick samples, off-axis references. Rheinberg illumination and colour-coded DPC need a **joint** S(k_s, λ). The source abstraction should not force separability.

---

## 3. Collection optics

### 3.1 Pupil-function model

For an infinity-corrected objective, image formation in *object-space coordinates* is a pupil filter. The coherent image field is F⁻¹{P(k; λ, h) · Û_exit(k)}, where h is the field position. It is shift-invariant when P does not depend on h. The pupil is a product of modifiers:

P(k) = Circ(|k| ≤ NA/λ) · A(k) · exp(i·2π/λ · [W_defocus + W_Zernike + W_GL + W_mask](k)) · T_mask(k) · [Jones(k)]

- **Defocus**, exact angular-spectrum form: W_defocus = z·√(n² − λ²|k|²). DeepTrack2 uses this form with its `z_shift` [V: optics.py].
- **Zernike aberrations** (OSA/ANSI or Noll indexing; pick one): W = Σ c_j Z_j(ρ, φ). Store the optical path difference (OPD) in physical length (µm), not radians. Phase then scales automatically as 2π OPD/λ for polychromatic work. Chromatic focal shift is a λ-dependent defocus coefficient.
- **Gibson–Lanni (refractive-index mismatch between immersion, coverslip and sample):** OPD(ρ) = z_p√(n_s² − NA²ρ²) + t_g√(n_g² − NA²ρ²) − t_g*√(n_g*² − NA²ρ²) + t_i√(n_i² − NA²ρ²) − t_i*√(n_i*² − NA²ρ²). Starred quantities are design values [R form; parameterisation V: psf_generator uses `z_p, n_s, n_g, n_g0, t_g, t_g0, n_i, n_i0, t_i0` at https://psf-generator.readthedocs.io/autoapi/psf_generator/index.html]. It produces depth-dependent spherical aberration and focal shift. The paraxial focal-shift ratio is n_s/n_i ≈ 0.88 for water in oil; at high NA the effective value is noticeably smaller [R].
- **Apodization and Jacobians.** For aplanatic *focusing* the amplitude factor is √cosθ. For *collection* from an object-space emitter it is 1/√cosθ (energy conservation). The Debye integral over (k_x, k_y) adds a 1/k_z (i.e. 1/cosθ) Jacobian [R]. psf_generator exposes these as `apod_factor`, `sz_correction` and `cos_factor`, plus a Gaussian `envelope` [V: same docs URL]. **Getting these wrong changes relative intensities between defocus planes and between NAs.** That matters once intensities are learned against real data.
- **Fresnel transmission** at interfaces: separate t_s and t_p per ray. This needs the vectorial model to be meaningful [V: psf_generator paper, https://arxiv.org/html/2502.03170v1].

**Scalar versus vectorial.** Richards–Wolf gives E(ρ) = −(ifk/2π)∬_Ω a(s) e^{iW(s)} e_∞(s) e^{iks·ρ} dΩ. The EPFL psf_generator paper shows that the Cartesian (FFT or chirp-Z) and spherical (Bessel/Simpson) evaluations are equivalent [V: https://arxiv.org/html/2502.03170v1]:
- Spherical form: O(n), converges at 4th order with Simpson's rule, and is fastest on GPU for large PSFs. It needs axisymmetric pupils.
- Cartesian form: required for Zernike terms or phase masks. It uses a chirp-Z transform implemented as three FFTs, costs O(n log n), and converges at 1st–2nd order.
- The vectorial overhead is about 1.5× (Cartesian) to 3× (spherical) on CPU and minimal on GPU.
- They added custom differentiable Bessel functions because PyTorch lacks autograd for Bessel J_n.

Rules of thumb [R/H]:
- Scalar with correct apodization is adequate for NA ≲ 0.7.
- Scalar is also adequate for *intensity shapes* of freely rotating emitters up to fairly high NA.
- Vectorial is required for fixed dipoles, polarisation optics (DIC, PolScope), near-interface emission (supercritical-angle fluorescence, SAF), linearly polarised high-NA focusing (elongated focal spot), and quantitative intensities at NA > 1.

**Jones and vector pupils.** In the collimated space behind the objective the field is transverse (2 components). Focusing maps 2 → 3 components (a 3×2 matrix per k). Collection maps 3 → 2. Standard p/s decomposition: for a dipole μ, the BFP field ∝ (1/√cosθ)[(cosθcosφ, cosθsinφ, −sinθ)·μ · ê_ρ + (−sinφ, cosφ, 0)·μ · ê_φ], multiplied by t_p and t_s at interfaces. The camera field (E_x, E_y) is then 2 FFTs per dipole component [R, consistent with the Green's-tensor model in Backer & Moerner 2014, V: https://pmc.ncbi.nlm.nih.gov/articles/PMC4317050/]. A general polarisation-optics path is a 2×2 Jones pupil J(k) applied to transverse pupil fields.

### 3.2 Magnification and tube lens

Two regimes:
- **Well-corrected, telecentric, infinity-corrected system.** Magnification is purely a coordinate map: camera pixel in object space = pitch/M, with M = f_tube/f_obj, and image-side NA = NA/M. The Debye model is valid at the image side because NA/M is tiny. **Simulate everything in object space.** Treat M as a coordinate scale on the camera grid, which is differentiable if the camera sampling uses a CZT or matrix DFT with continuous coordinates.
- **Physical image-space modelling.** Needed only for intermediate-plane filtering in relays (4f spatial filters in a conjugate pupil), non-telecentric systems, camera tilt, and image-side aberrations. Model these as pupil modifiers expressed in normalised pupil coordinates where possible. Field-dependent effects go through the space-variant machinery (§3.6):
  - distortion is a geometric warp;
  - field curvature is field-dependent defocus;
  - vignetting and flat-field non-uniformity are multiplicative envelopes.

### 3.3 Fourier-plane filtering family

All of these are "source shape × pupil mask" pairs in the Abbe/Hopkins model:
- **Darkfield.** Annular condenser with NA_c > NA_obj, so no zero-order light is collected. Or coherent darkfield with a central stop in the pupil. DeepTrack2's volume `Darkfield` returns |field − 1|² with contrast (Δn)². It is explicitly labelled a non-physical toy [V: optics.py]. That corresponds to an infinitesimal central stop.
- **Zernike phase contrast.** Annular source plus a conjugate phase ring in the objective pupil (≈ π/2 phase shift, transmission ≈ 0.1–0.25) [R]. The characteristic halo and shade-off artefacts come *only* from the finite ring width under partial coherence. A coherent model will not reproduce them, which is a good test case for the Abbe path.
- **DIC.** Polariser, condenser Wollaston/Nomarski, sample, objective prism, analyser. Per source mode, in scalar form: E_out ∝ E(r + s/2)e^{iψ/2} − E(r − s/2)e^{−iψ/2}. For a thin sample this equals the pupil multiplier i·2 sin(π k·s + ψ/2), so DIC is a Fourier filter plus Köhler partial coherence (Preza, Snyder and Conchello 1999 [V: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-16-9-2185]). A full Jones treatment is needed for birefringent samples and high-NA polarisation mixing.
- **Hilbert / Schlieren / Foucault.** Half-plane knife edge (sign(k_x)), or spiral phase e^{iφ_k} for isotropic edge enhancement.
- **Hoffman modulation contrast.** Off-axis slit source plus a graded amplitude modulator in the pupil.
- **Oblique and DPC.** Asymmetric sources.
- **PSF engineering.** Astigmatic (cylindrical lens, i.e. Zernike astigmatism), double-helix, tetrapod, or learned phase masks. DeepSTORM3D learned such a mask end-to-end through a differentiable PSF layer [V: https://github.com/EliasNehme/DeepSTORM3D].

### 3.4 Interferometric modalities

- **In-line holography.** I = |1 + E_s|², where the reference is the unscattered illumination. DeepTrack2's `Holography` is an alias of its coherent multi-slice `Brightfield` [V: optics.py]. A `return_field` output tap is essential for holographic training targets.
- **Off-axis holography.** I = |R|² + |O|² + R*O + RO*, with R = e^{i2πk_r·r}. Separating the terms requires |k_r| ≥ 3NA/λ. Sampling them then requires Δx_eff ≤ λ/(8NA) along the carrier axis, relaxed about √2 for a diagonal carrier [R/H]. The reference arm must be in the same mode set as the object (mutually coherent within each λ and source mode). Otherwise fringe visibility is wrong.
- **iSCAT / COBRI.** E_det = E_r + E_s. With E_r = rE_inc, the glass–water amplitude reflectance is r = (1.52 − 1.33)/(2.85) ≈ 0.067 (0.45 % in intensity) [H arithmetic]. The image is I = |E_r|²(1 + |s|² + 2|s|cos Δφ) with s = E_s/E_r. The contrast is linear in the particle polarisability (∝ a³), versus a⁶ in darkfield. In reflection the phase Δφ includes the round-trip term 2nk_0z plus the Gouy and scattering phases, so contrast oscillates with an axial period of about λ/(2n) [R]. Quantitative iPSFs need a vectorial model with defocus, index mismatch and aberrations. Mahmoodabadi et al. 2020 validated such a model against FDTD and experiment, with 3D localisation over about 10 µm [V: https://arxiv.org/pdf/2006.15332 via search summary]. DeepTrack2's `ISCAT` exposes `illumination_angle`, `amp_factor` (reference attenuation) and input/output polarisation [V: optics.py]. Reference attenuation is a pupil-plane amplitude filter on the zero order, which is another pupil modifier.

### 3.5 Fluorescence specifics: dipoles

For a fixed dipole μ the camera intensity is I = Σ_{ij} μ_iμ_j B_ij(r). The six basis images are B_xx, B_yy, B_zz, B_xy, B_xz, B_yz, computed from the Green's tensor. For rotational mobility, replace μ_iμ_j by the second moments M_ij = ⟨μ_iμ_j⟩. The isotropic case is (B_xx + B_yy + B_zz)/3 [V: Backer & Moerner 2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC4317050/]. The excitation rate is ∝ |μ_abs·E_exc|². Architecturally, a vectorial PSF model should expose the **6 real basis images**, not a single PSF. Isotropic, fixed and wobbling emitters are then all cheap linear combinations, and orientation parameters are differentiable. Near an interface (TIRF or SAF), the same basis needs t_p and t_s including evanescent-to-supercritical coupling. The collection efficiency of an isotropic emitter in a homogeneous medium is (1 − cosθ_max)/2, about 0.31 for NA 1.4 in oil [H arithmetic]. Radiometry should come out of the pupil normalisation, not an arbitrary scalar.

### 3.6 Depth-variant and space-variant PSFs

A ladder of techniques, cheapest first:
1. **Global 3D OTF** (fully shift-invariant, e.g. index-matched). Fastest for z-stacks: one 3D FFT convolution.
2. **Per-z strata.** One 2D OTF per z-slice. This is exact for depth dependence with lateral invariance (Gibson–Lanni). It costs N_z FFT pairs and is the DeepTrack2 approach.
3. **Low-rank expansion** h(r; z_e or field position) ≈ Σ_{k≤K} a_k(z_e) h_k(r), giving image = Σ_k h_k ⊛ (Σ_z a_k(z)c_z). That is K convolutions, and it is differentiable in both a_k and h_k. Examples: PCA depth-variant models (Preza lab) [R]; product-convolution space-variant blur, as in deepinv's `SpaceVaryingBlur`, `ProductConvolutionBlurGenerator` and `TiledSpaceVaryingBlur` [V: https://deepinv.org/api/deepinv.physics.html].
4. **Symmetry-exploiting space variance.** Ring deconvolution uses the rotational symmetry of most microscopes, with a Seidel-coefficient neural surrogate [V: Kohli et al., Nat Methods 22:1311 (2025), https://www.nature.com/articles/s41592-025-02684-5].
5. **Per-emitter rendering**, exact for any variance. Options:
   - a pupil FFT per emitter on a small region of interest (ROI), batched;
   - cubic-spline lookup tables of precomputed or measured PSFs (the SMAP/DECODE style [R]). These are fastest and differentiable in position, but not in optics parameters unless rebuilt;
   - analytic Gaussians. A least-squares Gaussian fit to the widefield PSF gives roughly σ_xy ≈ 0.21λ/NA [R value; paper V: Zhang, Zerubia and Olivo-Marin, Appl. Opt. 46:1819 (2007)].
   Splat PSF patches with a differentiable scatter-add.
6. **Learned or measured PSFs.** uiPSF infers pupil-based or voxel-based PSF models, including field- and depth-dependent aberrations and inter-channel transforms, from bead or single-molecule data [V: https://www.nature.com/articles/s41592-024-02282-x]. gradoscopy should be able to *import* such models as a `PSFModel` and *fit* its own pupils in the same way.
7. **Full wave through the sample** (per emitter or per mode BPM). Reference only.

### 3.7 Two-stage modalities: effective-PSF forms

- **Confocal** (descanned, pinhole D): h_conf(r) = h_exc(r; λ_ex) · [h_det(·; λ_em) ⊛ D](r). Pinhole diameter is given in Airy units, with 1 AU = 1.22λ_em/NA. microsim implements exactly this product: `eff_em_psf = fftconvolve(em_psf, pinhole)`, `out = ex_psf * eff_em_psf`. Its PSF is vectorial Gibson–Lanni, computed via Bessel/Simpson integration [V: https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/psf.py].
- **ISM / Airyscan.** Detector element at offset d gives h_d = h_exc(r) · h_det(r − d), followed by pixel reassignment. Spinning disk adds pinhole crosstalk: a periodic pinhole array convolved in.
- **Two-photon.** h_2p = [h_exc(r; ≈2λ)]². Non-descanned detection means no pinhole.
- **Light sheet.** h(r) = h_det(r) · I_sheet(z; x). This is **shift-variant along the propagation axis x**. For short FOVs (≪ x_R) treat it as invariant. Otherwise use the strata or low-rank approaches along x.
- **SIM.** Emission density × pattern, then h_det. 3D-SIM patterns move with the focal plane, which gives the standard band decomposition. The two-stage direct simulation handles this automatically.
- **TIRF.** Emission density × e^{−z/d}, then (ideally vectorial, near-interface) h_det.

---

## 4. Detection

Recommended pipeline, all on GPU, per pixel p:

1. **Spectral integration and radiometry.** Expected photons: Φ_p = t_exp ∫dλ QE(λ) T_filter(λ) ∫_pixel I(r,λ) dr / (hc/λ). Wavelength bins carry weights w_l = S(λ_l)·T(λ_l)·QE(λ_l). microsim bins spectra with intensity thresholds and a configurable bin count [V: _simple_psf.py above].
2. **Pixel response.** A box of the active width (fill factor, near 1 with microlenses), optionally convolved with a charge-diffusion Gaussian. In Fourier space: MTF_pix(k) = sinc(p_a k_x) sinc(p_a k_y) e^{−2π²σ_d²|k|²} [R].
3. **Background.** Additive photons b_p, which can be a learnable low-order field or out-of-focus/autofluorescence map. Dark current D·t_exp. EMCCD clock-induced charge (CIC).
4. **Shot noise.** e_p ~ Poisson(Φ_p + b_p + D t).
5. **EM gain (EMCCD).** e′ ~ Gamma(shape = e_p, scale = G). The excess noise factor F² → 2, i.e. effectively half the QE [R]. Add serial-register saturation.
6. **Full-well clamp.**
7. **Read noise and ADC.** v = e′ + N(0, σ_{r,p}²); ADU = clip(round(v/g_p + o_p), 0, 2^bits − 1).
8. **sCMOS per-pixel maps.** Offset o_p, variance σ²_p and gain g_p are required for faithful sCMOS statistics [V: Huang et al., Nat Methods 10:653 (2013), https://www.nature.com/articles/nmeth.2488]. Per-pixel QE non-uniformity also matters [V: Sci Rep 2019, https://www.nature.com/articles/s41598-019-53698-x]. Hot pixels and column fixed-pattern noise are optional.

microsim's `_Camera` has almost exactly this order: Poisson → dark (Poisson) → full-well → pre-quantisation binning (CCD/EMCCD) → EM gamma → serial-register cap → Gaussian read × gain → ADC → post-quantisation binning (CMOS) → saturation. Its defaults are read noise 6 e⁻, full well 18 000, 12 bit, offset 100. It lacks pixel size and fill factor ("TODO: add photodiode size") [V: https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/detectors/_camera.py]. Chromatix's `BasicSensor` offers `shot_noise_mode ∈ {None, 'approximate', 'poisson'}` and resampling by sum-pooling or `scale_and_translate` [V: https://raw.githubusercontent.com/chromatix-team/chromatix/main/src/chromatix/elements/sensors.py].

**Differentiability of detection:**
- The *expectation* path, the mean ADU as a function of Φ, is fully differentiable.
- `torch.distributions.Poisson` has **no `rsample`**; its `sample` runs under `no_grad` [V: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/distributions/poisson.py].
- `Gamma` **does** have `has_rsample = True`, so EM gain can be reparameterised [V: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/distributions/gamma.py].
- Options for Poisson:
  - Gaussian reparameterisation Φ + √Φ·ε, which is good above about 10–20 photons [H]. It is used in end-to-end optics design for this reason [V: https://arxiv.org/pdf/2412.09774 via search summary].
  - Straight-through estimator: a Poisson sample forward, identity backward.
  - Score-function estimators for distribution parameters.
- Quantisation: round with a straight-through estimator. Saturation: hard clamp (zero gradient) or a soft clamp.
- **For fitting to real data**, sampling is the wrong tool. The camera should expose `log_prob(measured_ADU | Φ)`, the exact Poisson ⊛ Gaussian (sCMOS) or EMCCD likelihood, so that physical parameters can be fitted by maximum likelihood. uiPSF and Huang 2013 work in this regime.

---

## 5. Sampling, grids, padding and cost

**Lateral Nyquist (object space):**
- Incoherent intensity: the OTF cutoff is 2NA/λ, so Δx ≤ λ/(4NA).
- Coherent imaging: the field needs Δx ≤ λ/(2NA). **|E|² doubles the bandwidth**, so squaring a field that is only sampled at the field Nyquist aliases. Either compute the field on the λ/(4NA) grid or zero-pad in Fourier space before squaring.
- Partially coherent: each mode image is still limited to 2NA_obj/λ in intensity, so λ/(4NA_obj) is sufficient.
- The *sample-plane* grid before the pupil must represent the object spectrum up to (NA_obj + NA_ill)/λ for thin samples. Thick multi-slice samples need up to n/λ (see the sample brief).

**Examples [H arithmetic]:**
- 100×/1.4 oil, 6.5 µm pixels → 65 nm vs a Nyquist limit of 93 nm at 520 nm. The camera grid is already fine enough: supersampling s = 1.
- 60×/1.2 water → 108 nm vs 108 nm. Borderline; use s = 2.
- 20×/0.75 → 325 nm vs 173 nm. The camera undersamples, so simulate at s = 2 and let the pixel integration alias physically.
- Off-axis holography at 20×/0.4 with 3.45 µm pixels → 172 nm vs the λ/(8NA) = 166 nm requirement. Borderline.

**Axial:** Δz ≤ λ/(2(n − √(n² − NA²))). For NA 1.4 in oil at 520 nm that is about 280 nm [H arithmetic].

**Camera grid versus simulation grid:**
- *Integer supersampling s*: compute on a grid with Δx_cam/s spacing, then s×s sum-pool. This is exact box integration and is what DeepTrack2 does with `upscale` + `SumPooling` [V: optics.py].
- *Band-limited evaluation*: multiply the intensity spectrum by MTF_pix, then evaluate at arbitrary pixel centres with a CZT or matrix DFT. This handles non-integer ratios, sub-pixel camera or stage offsets, and makes pixel size and magnification continuous, differentiable parameters. When the camera grid satisfies Nyquist, s = 1 with no upsampling is exact.

**Pupil grid versus image grid.** Decouple them. The pupil needs N_pupil ≥ 2(NA/λ)·L_psf samples across its diameter, where L_psf is the PSF extent to represent without wrap-around. A CZT computes the PSF directly at the camera sampling. psf_generator does this and avoids padding overhead [V: arXiv 2502.03170]. Typical N_pupil is 64–256.

**FOV padding:**
- Linear convolution needs a pad of at least the PSF radius at the maximum defocus: r ≈ |z|tanθ_max + 2λ/NA. For NA 1.4 in oil, tanθ ≈ 2.4, so 5 µm of defocus needs a ~12 µm margin [H arithmetic].
- Coherent plane-wave illumination has no natural edge. Use tapering or a larger pad.
- DeepTrack2's default of `padding=[10,10,10,10]` pixels [V] is far too small for defocused or coherent cases. Padding should be **derived** from the declared defocus range, NA and coherence width, not user-set.

**Spectral sampling [H]:** N_λ ≳ 4·ΔOPD_max/l_c. The PSF scales ∝ λ, so fluorescence with 40–60 nm emission bandwidth converges with 3–5 bins. Broadband holography or iSCAT far from focus needs more.

**Numerics:**
- complex64 is adequate for fields, e.g. a 3 200 rad Gibson–Lanni phase has an error of about 4·10⁻⁴ rad [H].
- Compute pupil OPDs in float64 (pupils are small) and cast afterwards.
- Use `rfft2` for real intensities.

**Cost per image (N×N grid; multipliers relative to one 2D FFT pair):**

| Axis | Choice → multiplier |
|---|---|
| Illumination modes | coherent 1 · WOTF 1 · SOCS K (5–30) · Abbe N_s (20–500) · Abbe on thick sample N_s × (sample sim) |
| Spectrum | ×N_λ (1, 3–10, 10–30) |
| Polarisation | scalar 1 · Jones 2–4 · dipole basis 6 (3 dipoles × 2 camera polarisations) |
| Depth | 1 plane · N_z strata · 3D FFT |
| Shift variance | invariant 1 · low-rank K · tiles T · per-emitter N_e × ROI² |
| Supersampling | ×s² memory and ≈ s² FFT cost |
| Frames | × acquisition count (SIM 9–15, FPM 50–300, focus stack) |

**Memory [H]:** one 512² complex64 field is 2 MiB. A batch of 32 with 64 modes is 4.3 GB for a single stage, and autograd stores 2–4 such tensors per FFT → multiply → IFFT → |·|² chain. **Accumulate modes in chunks with `torch.utils.checkpoint`, or with a custom autograd function that recomputes the chunk in backward.** Peak memory then scales with the chunk size, not with M.

---

## 6. Fidelity ladder (imaging-system side)

Fidelity is multi-axis. Named presets map to per-axis settings, and users can override any axis. The ladder is discrete and needs no continuity.

| Level | Illumination | Pupil / PSF | Shift variance | Spectrum / pol | Detector | Typical use |
|---|---|---|---|---|---|---|
| **L0 "sketch"** | idealised (uniform, or single analytic pattern) | analytic Gaussian/Airy sprites; Gaussian σ derived from NA, λ | invariant | 1 λ, scalar | Poisson–Gaussian or none | detection/tracking nets, huge datasets |
| **L1 "scalar"** (≈ DeepTrack2 parity) | single coherent plane wave; analytic excitation patterns | scalar FFT pupil + defocus + Zernike | per-z strata | 1–3 λ | full CCD/sCMOS/EMCCD statistics | most current DeepTrack uses |
| **L2 "partial coherence"** | Köhler/LED sources via SOCS (fixed optics) or sub-sampled or stochastic Abbe (learnable optics); WOTF for weak samples | scalar + apodization + Gibson–Lanni; phase ring, DIC and darkfield filters | strata or low-rank | 3–10 λ | + per-pixel maps | label-free (phase contrast, DIC, DPC, FPM), quantitative brightfield |
| **L3 "vectorial"** | as L2 + polarised/evanescent sources | Richards–Wolf / Jones, Fresnel t_s, t_p, dipole bases, SAF | low-rank or per-emitter | full spectral, vector | + likelihoods | SMLM, orientation imaging, iSCAT iPSF, high-NA quantitative |
| **L4 "reference"** | full Abbe per mode through the thick sample | vectorial, field-dependent (h-dependent Zernike), per-emitter pupils | per-emitter or per-tile, emission through the sample (BPM) | dense | full | validation, small FOVs, calibration studies |

Two points:
- The same user-facing `Microscope` description must be renderable at every level. This works if components are *physical descriptions* (NA, λ, source shape, aberration OPD, dipole moments), and each level only changes the **numerical view** the planner requests (for example, the Gaussian σ at L0 is derived from the same NA and λ).
- Parameters that lower levels cannot represent are handled by a declared policy per parameter: ignore with a warning, or approximate (for example, fold Gibson–Lanni into an effective defocus plus a Gaussian width).

---

## 7. Prior art and lessons

- **DeepTrack2** (`deeptrack/optical/optics.py`, default branch `develop`) [V]:
  - `Optics(NA, wavelength, magnification, resolution, refractive_index_medium, padding, output_region, pupil, illumination, upscale)`.
  - `Fluorescence`: per-slice OTF, empty slices skipped, sum-pooling downscale.
  - `Brightfield`: coherent multi-slice (`exp(i·Δn·dz·k)` per slice, angular-spectrum `pupil_step`, final `pupil_focus`), `return_field`.
  - `Holography`: alias of `Brightfield`.
  - `ISCAT`: `illumination_angle`, `amp_factor`, polarisation.
  - `Darkfield`: toy |E − 1|².
  - `IlluminationGradient`.
  - NumPy and Torch backends.
  - Lessons: good parameter vocabulary and a working per-slice fast path. Gaps: no partial coherence, no vector optics, hand-set padding, and the physically defined modalities are separate classes rather than source + pupil presets.
- **microsim** (Talley Lambert) [V]:
  - `Widefield`, `Confocal(pinhole_au)` and `Identity` modalities.
  - Vectorial Gibson–Lanni PSFs via Simpson/Bessel.
  - Spectral binning.
  - Solid camera models.
  - NumPy/CuPy/JAX backends. It describes itself as more "application" than "library".
  - Lessons: good schema of Sample/Modality/Detector/Space and truth versus output space. Not differentiable-first and not modular across coherent modalities.
- **Chromatix** (JAX; Nature Methods 2026) [V: https://github.com/chromatix-team/chromatix, bioRxiv 10.1101/2025.04.29.651152]:
  - Fields are `ScalarField`, `ChromaticScalarField`, `VectorField` and `ChromaticVectorField`, shaped `(..., y, x, [λ], [3])` with a spectrum object. Intensity sums over λ (weighted by spectral density) and polarisation.
  - Elements: lenses, masks, propagators, sensors.
  - Reported to "only model completely coherent propagation". Fluorescence is an "incoherent sum of coherently simulated PSFs".
  - Lesson: the `Field` data structure with λ and polarisation axes is right. Partial coherence and two-stage modalities are exactly the gap gradoscopy can fill.
- **psf_generator** (EPFL, PyTorch) [V]:
  - Scalar and vectorial, Cartesian (CZT) and spherical (Simpson/Bessel) propagators.
  - Gibson–Lanni, apodization, envelope and Zernike options.
  - Differentiable Bessel functions.
  - Best reference implementation to validate against, or to wrap for L3 PSFs. **Check its licence before vendoring**; the docs page did not state it.
- **MicroscPSF** (Li, Xue and Blu, JOSA A 2017): Gibson–Lanni as a Bessel-series expansion, reported 498× faster than PSF Generator at 511×511×255 [V: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-34-6-1029].
- **deepinv** [V]: PyTorch physics operators, including `DiffractionBlurGenerator3D` (Zernike pupil), `ConfocalBlurGenerator3D`, space-varying product-convolution and tiled blur, and Poisson/Gaussian/Poisson–Gaussian noise. Useful interop target and a proof that product-convolution is practical in torch.
- **TorchLitho** [V]: differentiable Abbe and Hopkins/SOCS in PyTorch, scalar only.
- **biobeam** [V]: BPM light-sheet simulation, OpenCL, not differentiable. It is the L4 reference design for tissue light-sheet.
- **uiPSF** [V] and **DeepSTORM3D** [V]: evidence that pupil-based differentiable PSF models are the right parameterisation for learning optics from data.
- **OSOG** (arXiv 2606.21381, June 2026) [V abstract]: PyTorch structure-of-arrays, differentiable "wave-optic particles" (40 000 in < 50 ms). It indicates that the demand is for L0/L1-speed particle rendering with differentiability.

---

## 8. Implications for gradoscopy's architecture

### 8.1 Core data types (internal, but stable)

- **`Field`**: complex tensor `[B, A…, M, Λ, P?, Y, X]`.
  - B: batch.
  - A: acquisition axes (frames).
  - M: summed incoherent modes, with weights `w[B?, M]`.
  - Λ: wavelength bins, with weights.
  - P: absent (scalar), 2 (Jones/transverse) or 3 (Cartesian).
  - Metadata: a `Grid` (dx, origin, frame: a rigid transform in sample coordinates), the plane z, the domain (real/Fourier), and the medium index.
  - Use a small dataclass wrapper with named axes. Do **not** use `torch` named tensors, which are poorly supported.
- **`Irradiance`**: real `[B, A…, Λ?, (Z), Y, X]`, in physical units (photons·µm⁻²·s⁻¹).
- **`Emitters` / `SourceDensity`** (from the sample side): point lists (position, brightness, spectrum id, dipole second moments, on/off state) *or* density volumes. Both must be accepted, because collapse (a) and per-emitter rendering are complementary.
- **`Frames`**: ADU or photons `[B, A…, C, H, W]`, plus optional taps (complex camera-plane field, noiseless expectation, per-stage irradiances) for supervision.

### 8.2 First-class swappable components (public API)

1. **`Illumination`**: a physical description S(k_s, λ, pol, t) plus a spatial envelope. It produces either a `Field` mode set (transmitted or reflected light) or an excitation `Irradiance` (fluorescence).
   - Implementations: `PlaneWave`, `Kohler(condenser_na, shape=Disk|Annulus|Custom, sampling=…)`, `LEDArray`, `LaserSpeckle`, `LightSheet(Gaussian|Bessel|Lattice, scanned)`, `Evanescent`, `StructuredPattern`, `FocusedBeam`, `ReferenceBeam`, and a learnable `IlluminationNuisance`.
   - The *sampling strategy* of a Köhler source (grid, quadrature, stochastic, SOCS) is a planner choice, not a user class.
2. **`Pupil`**: an ordered product of **`PupilModifier`s** evaluated on continuous pupil coordinates (ρ, φ), with optional dependence on λ and field position h. The modifiers are `Aperture(NA, soft_edge)`, `Apodization`, `Defocus`, `LayerStack` (Gibson–Lanni and Fresnel), `Zernike(OPD in µm)`, `PhaseMask`, `AmplitudeMask`, `PhaseRing`, `CentralStop`, `KnifeEdge`, `SpiralPhase`, `Wollaston(shear, bias)`, and `Jones` elements. The same `Objective` object serves both excitation and emission paths in epi geometries.
3. **`ImagingModel` / PSF views** derived from a `Pupil`, each a strategy class behind one protocol. Views: `ctf()`, `field_psf(points)`, `intensity_psf(grid)`, `otf()`, `socs(source, K)`, `wotf(source)`, `dipole_basis()`, `gaussian_approx()`. Backends: `ScalarFFT`, `ScalarBessel`, `VectorialCZT`, `VectorialBessel`, `SplineLUT`, `Measured` / `uiPSF`-import, `Neural`. Users choose these only when they want to pin fidelity.
4. **`DetectionPath`**: magnification and tube lens (a coordinate map), emission filter spectra, branch splitters (dichroic, polarising, biplane, multi-camera with per-channel affine registration), confocal pinhole or detector array.
5. **`Camera`**: pixel geometry (pitch, fill factor, diffusion, ROI, binning), spectral QE, noise model (`Ideal`, `PoissonGaussian`, `CCD`, `EMCCD`, `SCMOS(maps)`, `QCMOS`), and exposure and time sampling. Methods: `expected()`, `sample(mode='exact'|'reparam'|'straight_through')`, `log_prob()`.
6. **`Modality` presets**: *thin recipes*, not physics. `Widefield`, `TIRF`, `Confocal`, `TwoPhoton`, `LightSheet`, `SIM`, `Brightfield`, `Darkfield`, `PhaseContrast`, `DIC`, `DPC`, `FPM`, `InlineHolography`, `OffAxisHolography`, `ISCAT`. Each is a constructor that wires Illumination + Pupil modifiers + DetectionPath + Camera, and declares the stage graph (coherent stage → source map → coherent stage). Adding a modality should mostly mean adding a preset.

### 8.3 Internal (planner-owned) details

These should be hidden behind a `Plan`:
- the solver choice (Abbe, SOCS, WOTF, per-slice OTF, low-rank, per-emitter, full-wave);
- grid sizes, supersampling s, padding and tapers;
- CZT versus FFT;
- mode chunk size and checkpointing;
- PSF and SOCS caches.

The plan should be inspectable: `microscope.plan(sample_spec, fidelity="L2").explain()` should print the chosen algorithms, grid sizes, and the approximations each one implies. Planner rules:
- Collapse (a) applies when the stage and pupil declare lateral invariance and the emitters are dense. Use per-emitter rendering when the emitters are sparse, with the crossover from N_e·ROI² versus N_z·N² log N.
- Collapse (b), via SOCS, applies when the sample declares `thin` **and** the optics are fixed (no `requires_grad`). Use Abbe with sub-sampled or stochastic sources when the source or pupil is learnable. SVD gradients are unstable for degenerate singular values, and symmetric sources create exactly those.
- WOTF applies when the sample declares `weak` and the fidelity level is ≤ L2.

### 8.4 Differentiability contract

- Every numeric optical parameter accepts a leading batch dimension (per-sample optics sampled from learnable distributions) and is broadcast through PSF generation. Pupils are small, so batched per-sample PSFs are cheap.
- **Shapes derive from declared bounds**: parameter ranges and distribution supports, never from current values. This keeps `torch.compile` and CUDA graphs viable and prevents a learnable defocus from changing the padding mid-training.
- **Soft geometry everywhere.** Anti-aliased or sigmoid-edged apertures give gradients with respect to NA, condenser NA, ring radii, pinhole size, sheet position, fill factor and wavelength (which moves the pupil radius). Continuous-coordinate CZT and matrix-DFT sampling give gradients with respect to pixel size, magnification and sub-pixel offsets.
- Caches are keyed by parameter version and bypassed when gradients are required.
- Fidelity knobs (N_s, N_λ, K, s, strata count) are discrete hyperparameters, not differentiable, which is acceptable given the user's "not necessarily continuous" requirement. Stochastic mode sampling keeps gradients unbiased at low cost.

### 8.5 Conventions to fix on day one

Unify these early, because they are where silent 2× and sign bugs come from:
- object-space µm;
- Fourier kernel e^{+i2πk·r};
- z axis pointing toward the objective (state defocus sign explicitly);
- OSA/ANSI Zernike indices with RMS normalisation, plus a Noll converter;
- OPD in µm, with phase = 2π·OPD/λ;
- absolute radiometry: pupils normalised so that the PSF integral equals the collection efficiency;
- apodization factors: √cosθ for focusing, 1/√cosθ for collection, and the 1/k_z Jacobian.

### 8.6 Scope recommendation

- **In scope for v1:**
  - widefield fluorescence (incl. TIRF, HILO, PSF engineering, biplane, dipoles);
  - Köhler and LED transmitted-light family (brightfield, darkfield, phase contrast, DIC, DPC, FPM, oblique);
  - in-line and off-axis holography, iSCAT and COBRI;
  - confocal, spinning disk, ISM and 2P via effective PSFs;
  - light sheet (analytic sheet × detection PSF);
  - linear 2D/3D SIM;
  - CCD, EMCCD, sCMOS and qCMOS cameras.
- **Later:** per-scan-position confocal, BPM-through-tissue excitation and emission (L4), field-dependent aberration fields, polarisation tomography.
- **Out of scope:**
  - coherent nonlinear modalities (SHG, CARS, SRS);
  - STED and nonlinear SIM physics (offer parametric effective PSFs only);
  - FLIM and photon-timing statistics;
  - OCT;
  - near-field plasmonics;
  - ray-traced compound-lens design. Accept external OPD or Jones-pupil maps instead, which keeps the pupil abstraction closed.

### 8.7 Build order

1. Scalar pupil + `Defocus`/`Zernike` + per-slice OTF widefield + per-emitter sprites + the full camera with `log_prob`. This gives DeepTrack parity plus radiometry.
2. The coherent `Field` path with mode axis, chunked Abbe, `PlaneWave`/`Kohler`/`LEDArray`/`ReferenceBeam`, and the Fourier-filter modifiers. This makes all transmitted-light presets and iSCAT/holography fall out.
3. Vectorial and Jones pupils, `LayerStack` (Gibson–Lanni and Fresnel), dipole bases, SAF.
4. SOCS and WOTF fast paths with planner rules.
5. Two-stage presets (confocal, 2P, light sheet, SIM, TIRF).
6. Low-rank and space-variant rendering, and uiPSF import.

### 8.8 Validation suite (continuous integration)

- Airy pattern and in-focus analytic OTF.
- Scalar versus vectorial PSFs against psf_generator. Gibson–Lanni against MicroscPSF/psf_generator.
- Abbe versus SOCS convergence in K, and SOCS versus WOTF for weak phase objects.
- Phase-contrast halo present under partial coherence and absent under coherent illumination.
- iSCAT contrast period of λ/(2n) versus z. TIRF depth d.
- Camera mean and variance (EMCCD variance ≈ 2G²Φ).
- Gradient checks against finite differences for NA, λ, Zernike coefficients, pixel size and condenser NA. These tests specifically guard the soft-edge and CZT design.
