# 08: Applications, fidelity requirements and scope for gradoscopy

**Question:** Which applications must a differentiable, PyTorch-based light-microscopy simulator serve? What physics does each one need, at minimum and ideally? Which parameters do people want to learn? And given those answers, what belongs in v1, in v2+, and out of scope?

**How to read the evidence tags:** [V] means I checked the claim in this session, and the URL is given inline or in the source list. [R] means it comes from my own expertise or recollection and was not re-checked. [?] means I am unsure of it. Any number without a tag is an engineering estimate I derived myself.

---

## 0. Summary of conclusions

1. **Most practical applications need only a small set of physics.** That set is: (i) Fourier optics based on the pupil, with scalar and vectorial PSFs, Zernike or pixel pupils, and refractive-index (RI) mismatch; (ii) incoherent fluorescence rendering, both for sparse emitters and for dense volumes; (iii) coherent imaging of thin objects, weak scatterers and multislice volumes, with partial coherence from summing over source points; (iv) exact single-particle scattering (Rayleigh dipole, Mie, layered Mie) superposed without coupling between particles; (v) interferometric and pupil-plane contrast (holography, iSCAT, darkfield, phase contrast, DIC); (vi) realistic camera noise that has both a sampler and a likelihood; (vii) differentiable stochastic scene generation. These seven cover the *minimum* fidelity of roughly 16 of the 18 application families below, and the *ideal* fidelity of about half of them.
2. **The hard cases are few and easy to name.** They are strongly multiple-scattering 3D samples (thick-tissue ODT, light-sheet in tissue), non-spherical exact scatterers (rods, clusters), near-field effects at interfaces (supercritical-angle fluorescence, iSCAT at the coverslip), anisotropic tensor samples, and nonlinear photophysics (STED). These belong in v2 or later, or behind plugin interfaces. Full-wave solvers (FDTD/FEM/DDA), coherent nonlinear optics and electron microscopy are out of scope. Where they matter, they enter as *precomputed scattering amplitudes or T-matrices*, not as solvers inside the engine.
3. **The angular spectrum (pupil) domain works as the lingua franca.** Point emitters, dipoles, Mie spheres, thin objects, Born scatterers and multislice exit fields all map cheaply onto a pupil-plane field. Aberrations, phase masks, phase rings, darkfield stops, DIC shear and learned optics all live in the pupil.
4. **Learnable parameters fall into about ten classes.** Two of them need machinery that existing optics libraries lack: gradients through random scene generation (distribution hyperparameters, relaxed discrete counts), and likelihood-based detector models. Section 6 lists all ten.

---

## 1. A shared vocabulary: fidelity axes

Each application is characterised along six largely independent axes. The levels are ordered by cost and generality. gradoscopy should make them *explicit, discrete choices per stage*.

| Axis | Levels |
|---|---|
| **S: sample representation** | S0 point emitters or point scatterers (dipoles) · S1 analytic primitives (sphere, ellipsoid, spherocylinder, shells) with continuous parameters · S2 2D maps (thickness/OPL, complex transmittance, 2D fluorophore image) · S3 voxel volumes (fluorophore density ρ(r), complex RI n(r); optionally a tensor ε(r)) · S4 implicit/neural fields, meshes |
| **I: interaction** | I0 phenomenological (Gaussian blobs, additive contrast) · I1 incoherent emission ∝ excitation intensity; thin/projection object `t = exp(i k0 ∫Δn dz)` · I2 linearised weak scattering (first Born/Rytov, weak-object transfer functions) · I3 exact single particle (Rayleigh dipole, Mie, layered Mie, T-matrix), superposed without coupling · I4 forward multiple scattering (paraxial/wide-angle BPM, multi-layer Born, SSNP) · I5 rigorous full-wave (Lippmann–Schwinger/Born series, coupled T-matrix, DDA, FDTD) |
| **C: coherence** | C0 fully coherent · C1 fully incoherent · C2 partially coherent via an Abbe sum over source points (Köhler condenser disc/annulus, LED arrays) · C3 Hopkins TCC/SOCS (valid only for thin, shift-invariant objects) · C4 polychromatic (sum over λ with dispersion) |
| **P: aperture/polarisation** | P0 scalar paraxial (quadratic defocus) · P1 scalar non-paraxial (exact `kz`, apodisation, RI-mismatch phase) · P2 vectorial (Richards–Wolf, dipole emission, Fresnel coefficients at interfaces, SAF) · P3 anisotropic samples (Jones/tensor ε) |
| **D: detection** | D0 none/Gaussian · D1 Poisson + Gaussian read noise, gain/offset · D2 camera specific (per-pixel sCMOS maps, EMCCD Gamma–Poisson, quantisation, saturation) · D3 temporal integration (motion blur within the exposure, frame-to-frame correlation) |
| **T: dynamics** | T0 static · T1 kinematics (Brownian/directed motion, drift) · T2 photophysics (blinking, bleaching) · T3 biology (growth, division, mechanics). T3 should be an external plugin. |

The core equations, written in a notation reused below:
- Coherent imaging: `E_img = F⁻¹{ P(k) · Ê_exit(k) }`, `I = |E_img|²`.
- Abbe partial coherence: `I = Σ_s w_s |F⁻¹{ P · Ê_exit^(s) }|²`, with one exit field per source point *s*.
- Fluorescence (shift-invariant): `I = h_det ⊛ (ρ · I_exc)`. For sparse emitters: `I = Σ_j N_j h(r − r_j; z_j, μ_j) + b`.
- Interferometric: `I = |E_r|² + |E_s|² + 2 Re(E_r* E_s)`.
- Multislice: `E_{m+1} = F⁻¹{ e^{i kz Δz} F{ E_m · e^{i k0 Δn_m Δz} } }`.

---

## 2. Application survey

### 2.1 2D/3D particle tracking (brightfield, fluorescence, darkfield)

**Deep-learning task.** Detect and localise particles (x, y, and often z) and sometimes estimate size or RI, then link them into trajectories. This is DeepTrack's core domain: *Helgadottir et al., Optica 2019* [R]; *Midtvedt et al., Appl. Phys. Rev. 2021* [R]; LodeSTAR, *Midtvedt et al., Nat. Commun. 13:7492, 2022* [V]; MAGIK, *Pineda et al., Nat. Mach. Intell. 2023* [R]. Other examples are *Newby et al., PNAS 115:9026, 2018*, a CNN trained on simulated 2D/3D particle videos [V](https://www.pnas.org/doi/10.1073/pnas.1804420115), and the ISBI particle-tracking challenge built on simulated data (Chenouard et al., Nat. Methods 2014) [R].

**What DeepTrack does today.** I checked the source [V](https://github.com/DeepTrackAI/DeepTrack2).
- `deeptrack/optical/optics.py` defines `Microscope`, `Optics`, `Fluorescence`, `Brightfield` (coherent slice-by-slice angular-spectrum propagation, with RI voxels converted to phase via `exp(1j·ri·dz·K)`), `Holography` (alias of Brightfield), `ISCAT`, `Darkfield` (`|field−1|²`), `IlluminationGradient` and `NonOverlapping`.
- Its parameters include `NA`, `wavelength`, `magnification`, `resolution`, `refractive_index_medium`, `padding`, `output_region`, `pupil`, `upscale` and `illumination_angle`.
- `scatterers.py` defines the voxelised `PointParticle`, `Ellipse`, `Sphere` and `Ellipsoid`, the field-based `MieSphere` and `MieStratifiedSphere` (with "geometric" and "hybrid" pupil modes), and `Incoherent`.
- A NumPy/PyTorch backend dispatch exists. Position resampling in `NonOverlapping` is explicitly non-differentiable.

**Minimum fidelity.**
- Fluorescence: S0/S1 with I1, a P1 scalar pupil PSF with defocus and a few Zernikes, pixel integration, D1 noise.
- Brightfield/darkfield of sub-µm to µm particles: an I3 Mie field is needed for correct z-dependent ring patterns. Voxel projection (I1) gives qualitatively wrong defocus behaviour for particles of size ≳λ. Use C0 or a small C2 source sum.

**Ideal fidelity.**
- P2 vectorial PSFs with coverslip RI mismatch and depth-dependent spherical aberration.
- Partial coherence set by the condenser NA. In brightfield it strongly changes contrast and defocus behaviour.
- Plasmonic dispersion for metal nanoparticles in darkfield (see 2.12).
- T1 Brownian motion with exposure-time motion blur, which matters for fast diffusers.
- Realistic background: illumination gradients, dust, out-of-focus debris.

**Sizes.** Images of 64²–512², batches of 8–64, 1 to a few hundred particles per image, sequences of 10–100 frames for linking [R].

**Learned parameters.**
- Inverse problems: per-particle position, radius, RI and intensity.
- Sim-to-real: the distributions of those quantities plus defocus range, aberrations, illumination gradient, noise level and background texture statistics.
- DeepTrack already ships an Optuna-based "automated optimization" example [V], which confirms that users tune simulation parameters today, but by black-box search.

### 2.2 Holographic particle characterisation (Lorenz–Mie; xSight, holopy, pylorenzmie)

**Task.** In-line or off-axis holograms are fitted with `I(r) = |E0 + α E_s(r − r_p; a_p, n_p)|²` to recover radius, RI and 3D position. Lee et al. (Opt. Express 2007) report precision of a few nm in radius and about 1e-3 in index [R]. The commercial xSight (Spheryx) instrument grew out of this line of work [R].

**Published works.**
- **CATCH** (Altman & Grier, J. Phys. Chem. B 2020) is a CNN trained on synthetic Lorenz–Mie holograms, "a representation of Lorenz–Mie theory in 200 kB" [V](https://pubs.acs.org/doi/10.1021/acs.jpcb.9b10463). **pylorenzmie** supplies GPU-accelerated hologram computation and fitting [V].
- **HoloPy** provides the `Mie` (layered via Yang's recursion, x ≤ 1000), `Multisphere` (exact clusters), `DDA` (via external ADDA), `Tmatrix` (axisymmetric: spheroids, cylinders), `MieLens`, `AberratedMieLens` and `Lens` theories [V](https://holopy.readthedocs.io/en/master/reference/holopy.scattering.theory.html). MieLens (Leahy et al., Opt. Express 2020 [R]) models the objective's finite collection angle, which improves accuracy near focus.
- **Deep-learning off-axis holography** (Midtvedt et al., ACS Nano 15:2240, 2021) extracts size *and* RI of subwavelength particles from trajectories two orders of magnitude shorter than standard methods need [V](https://pubs.acs.org/doi/10.1021/acsnano.0c06902).

**Minimum fidelity.** S1 spheres with I3 exact Mie. Born or projection approximations bias size/RI badly for x = k·a ≳ 1 with Δn ≳ 0.05. The coherent C0 regime applies. The scattered field must be at least near-vectorial: with x-polarised illumination the pattern is not rotationally symmetric.

**Ideal fidelity.**
- A MieLens-type pupil model: finite NA plus aberrations, especially spherical aberration from the coverslip.
- Layered spheres (core–shell, vesicles).
- Effective-medium models for porous or fractal aggregates, an approximation the Grier lab uses [R].
- Axisymmetric T-matrix for spheroids.
- Coupled multisphere for dimers.
- Off-axis carrier fringes and camera noise.

**Sizes.** Crops of 64²–256² with one particle each, large batches (32–256).

**Learned parameters.** a_p, n_p, r_p, illumination amplitude α, wavelength, pixel-size/magnification calibration, lens NA, spherical-aberration coefficient, background normalisation.

**Implementation note.** Mie truncation is `n_max ≈ x + 4x^{1/3} + 2` (Wiscombe) [R]. For a 2 µm-radius polystyrene bead in water at 532 nm, x ≈ 2π·1.33·2/0.532 ≈ 31, so about 45 orders are needed.
- Evaluating the near field on a 256² grid naively keeps O(n_max · H·W) complex intermediates alive for autograd. That is about 23 MB per particle in complex64, or roughly 75 GB for 100 particles × batch 32. This is infeasible.
- **Compute Mie in the pupil instead.** For a sphere, the far-field amplitudes S1(θ) and S2(θ) are 1D functions (≈500 samples). Map them onto the 2D pupil with polarisation factors cosφ/sinφ and multiply by the position phase ramp `exp(−i k·r_p)` and the defocus term `exp(i kz z_p)`. This is essentially DeepTrack's "hybrid" mode and HoloPy's MieLens. The cost and memory per particle become negligible, and the method extends naturally to per-particle aberrations.

### 2.3 iSCAT and mass photometry

**Task.** Detect and localise label-free nanoparticles or single proteins. Interferometric contrast is linear in polarisability, and therefore in mass (Young et al., Science 2018 [R]). The iPSF shape encodes z over about 10 µm (Mahmoodabadi et al., Opt. Express 28:25969, 2020, a vectorial Richards–Wolf model with stratified media, validated against FDTD) [V](https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-18-25969&id=434558). ML has pushed the detection limit below 10 kDa using an iForest plus self-supervised FastDVDnet (Dahmardeh et al., Nat. Methods 20, 2023) [V](https://www.nature.com/articles/s41592-023-01778-2). DeepTrack has an `ISCAT` optics class [V].

**Minimum fidelity.**
- S0 Rayleigh dipole with `α = 4πa³(ε_p−ε_m)/(ε_p+2ε_m)` [R].
- A scalar P1 pupil with the correct *relative phase* between reference and scattered light. That phase includes the Gouy phase, the defocus propagation phase and the reflection phase.
- Contrast `c ≈ 2|s/r| cos φ + |s/r|²`.
- Ratiometric (frame-differenced) background, with Gaussian-approximated shot noise at very high photon counts.

**Ideal fidelity.**
- P2 vectorial imaging with stratified media: coverslip interface, Fresnel coefficients, the dipole's near-field coupling to the interface.
- Aberrations.
- Partial spatial coherence from illumination scanning.
- A static background from coverslip roughness (random phase screens). This is the dominant sim-to-real gap for small particles.
- Mie for larger particles (> ~50 nm).

**Numerics.** Contrast ranges from about 1e-2 down to 1e-5 [R]. complex64/float32 is adequate *only if the interference term is computed explicitly* rather than forming `|E_r + E_s|²` and subtracting `|E_r|²`, which causes catastrophic cancellation. A float64 validation mode is essential.

**Learned parameters.** Polarisability or mass calibration, z, the reference–scatter phase offset, aberrations, background-screen statistics, a noise model per ratiometric frame.

### 2.4 Single-molecule localisation microscopy (astigmatic, double-helix, tetrapod, learned PSFs)

**Published works.** All checked this session:
- **DeepSTORM3D** (Nehme et al., Nat. Methods 2020) [V](https://github.com/EliasNehme/DeepSTORM3D).
  - Its forward model is **scalar** Fourier optics with oil/water RI mismatch through Snell's law. There are no Fresnel coefficients.
  - The pupil is a circular aperture combining the NA and medium limits. Defocus is applied as `exp(1j·k_oil·(−NFP)·cosθ_oil)`.
  - The phase mask is a learnable `torch.nn.Parameter`. A per-emitter random Gaussian blur absorbs model mismatch.
  - Noise comprises a non-uniform super-Gaussian background, a differentiable Poisson approximation (`poisson_noise_approx`), and spatially non-uniform read noise with a baseline.
  - Learning the mask took about 30 h on a Titan Xp.
  - The follow-up learns an optimal *pair* of PSFs (arXiv 2009.14303) [V].
- **DECODE** (Speiser et al., Nat. Methods 2021) [V](https://github.com/TuragaLab/DECODE).
  - The PSF is a cubic spline calibrated from bead stacks, not a physical model.
  - Defaults are `img_size: [40, 40]`, `batch_size: 64` and `emitter_av`. Photons are set by `intensity_mu_sig` and `photon_range`, blinking by `lifetime_avg` and background by `bg_uniform`.
  - The camera model is `em_gain`, `e_per_adu`, `baseline`, `read_sigma`, `spur_noise` and `qe`.
  - It uses a 3-frame temporal context [R].
- **uiPSF** (Liu et al., Nat. Methods 21:1082, 2024) [V](https://pmc.ncbi.nlm.nih.gov/articles/PMC12330227/).
  - It is TensorFlow autodiff with three PSF parameterisations: voxel, pixelated pupil and Zernike pupil (~21 modes). The pupil models are vectorial.
  - It learns emitter xyz/photons/background, per-frame drift, bead size, field-dependent aberration maps, inter-channel affine transforms and 4Pi piston phase. It incorporates lattice-light-sheet shear directly into the forward model.
  - The loss is MLE in the Fourier/pupil models and MSE in the voxel model.
  - Runtimes are 40 s to 35 min on an RTX 3080.
- **FD-DeepLoc** (Fu et al., Nat. Methods 2023) fits thousands of bead stacks across the FOV to a GPU *vectorial* PSF to obtain field-dependent Zernike maps. It localises over about 180×180×5 µm³ [V](https://github.com/Li-Lab-SUSTech/FD-DeepLoc).
- **Deep-SMOLM** (Wu et al., Opt. Express 30:36761, 2022) estimates 3D orientation and 2D position of overlapping dipole emitters with engineered dipole-spread functions [V](https://github.com/Lew-Lab/Deep-SMOLM).

**Minimum fidelity.**
- S0 emitters.
- A P1 scalar pupil with RI mismatch plus Zernikes, including astigmatism, double-helix and tetrapod masks, *or* an experimental spline or voxel PSF.
- Pixel integration.
- D1/D2 noise (EMCCD or sCMOS).
- T2 blinking.
- Differentiability with respect to the phase mask, which DeepSTORM3D needs.

**Ideal fidelity.**
- P2 vectorial dipole emission, either freely rotating (averaged over 3 orthogonal dipoles) or fixed/wobbling (SMOLM).
- Supercritical-angle fluorescence near the coverslip.
- Field-dependent aberrations.
- Chromatic effects for multicolour.
- Per-pixel sCMOS offset, variance and gain maps.
- Drift.

**Sizes.** Crops of 40²–128², batches of 16–64, 1–50 emitters per frame, z ranges of a few µm.

**Learned parameters.** Pupil phase (pixel or Zernike), field-dependence coefficients, per-emitter (xyz, N, bg, orientation), camera gain/offset/noise, and for sim-to-real the photon/density/background distributions.

**Architectural consequence.** SMLM is sparse. Per-emitter rendering (pupil → ROI via matrix DFT or chirp-z, then scatter-add) is far cheaper than full-frame 3D convolution. It also makes per-emitter *field-dependent* pupils essentially free: the Zernike coefficients become functions of the emitter's (x, y). A vectorial isotropic emitter costs about 6 pupil transforms instead of 1. A PSF-stack-plus-interpolation path, as in DECODE's spline or uiPSF voxels, must also exist for experimentally calibrated PSFs.

### 2.5 Cell segmentation and tracking (brightfield, phase contrast, DIC, fluorescence)

**Published works.**
- **SyMBac** (Hardo et al., BMC Biol. 2022) [V](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710168/). It renders cells as 3D spherocylinders whose growth and collisions are simulated in Pymunk (a 2D rigid-body engine), producing an OPL image. The OPL image is convolved with a phase-contrast PSF modelled as an "obscured Airy disk" parameterised by NA, λ, pixel size and ring/annulus dimensions. Rendering happens at 3× oversampling followed by downsampling. A small PSF offset controls halo and shade-off. Perlin noise supplies agar texture.
  - Matching to real images is *interactive and manual*: intensity histograms, noise, and the rotational Fourier spectrum. Training uses 2,000–5,000 synthetic images.
  - Error rates are 0.12–0.47% versus 0.93–1.8% for pretrained models.
  - SyMBac is the clearest existing example of the workflow gradoscopy should automate with gradients.
- **Cell Tracking Challenge** synthetic datasets (Fluo-N2DH-SIM+, Fluo-N3DH-SIM+) generated with MitoGen/CytoPacq, which simulate a specific widefield microscope and camera [V](https://celltrackingchallenge.net/2d-datasets/). Other examples are SimuCell (Rajaram et al., Nat. Methods 2012) [R] and the cell-counting tutorials in DeepTrack [V].
- Segmenters such as Cellpose, StarDist and Omnipose are mostly trained on real annotated data [R]. Simulation's value here is domain-specific retraining and exact ground truth.

**Minimum fidelity.**
- S2 OPL/thickness maps (fluorescence: 2D density).
- Linearised transfer-function imaging (I2, weak phase). Phase contrast uses a pupil phase ring. DIC uses `2i·sin(k·s/2 − ψ/2)·T̂(k)`, the shear-and-bias kernel.
- Procedural textures (Perlin/fractal fields).
- D1 noise.

**Ideal fidelity.**
- S3 RI volumes with organelle texture.
- I4 multislice.
- True C2 partial coherence from the condenser annulus. Phase-contrast halo and shade-off emerge naturally from annulus/ring overlap rather than from a tuned offset.
- DIC with Nomarski polarisation.
- Realistic mechanics of shape and division (T3, external).
- Learned texture generators.

**Sizes.** Images of 256²–1024², batches of 4–16, 10–500 cells; time-lapse sequences for tracking.

**Learned parameters.** Texture spectra and amplitude, shape distributions, RI/OPL contrast, condenser NA and ring parameters, defocus, vignetting, noise. **Distribution matching** on histograms and spectra, i.e. SyMBac's manual loop made differentiable.

### 2.6 Quantitative phase imaging and optical diffraction tomography

**Published works.**
- **2D QPI.** Off-axis DHM, TIE from defocus, DPC, and weak-object transfer functions under partial coherence. **waveorder** (PyTorch; phase from brightfield, QLIPP, permittivity tensor imaging; single-scattering models) [V](https://github.com/mehta-lab/waveorder).
- **3D refractive-index tomography.**
  - First Born/Rytov.
  - Multislice/BPM (Kamilov et al., Optica 2016, "learning approach to optical tomography" [R]; Tian & Waller, Optica 2015 [R]; Chowdhury et al., Optica 2019 [R]).
  - **Multi-layer Born** (Chen et al., Optica 7:394, 2020) [V](https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-5-394&id=431219). It handles oblique and backward scattering better than multislice at similar cost. Their 100-iteration reconstructions took 0.67 h (Born), 1.19 h (Rytov), 1.50 h (multislice) and 1.67 h (MLB).
  - SSNP, a non-paraxial multiple-scattering model (Zhu et al., arXiv 2207.06532) [V].
  - Lippmann–Schwinger solvers such as SEAGLE and the convergent Born series [R].
  - **DeCAF**, a neural-field RI representation learned through the IDT forward model with no ground truth (Liu et al., Nat. Mach. Intell. 4:781, 2022) [V](https://www.nature.com/articles/s42256-022-00530-3).
- Chromatix (JAX) demonstrates phase retrieval and holography [V].

**Minimum fidelity.** S3 voxel RI with I2 Born/Rytov (weak, thin cells). Scalar C0 per illumination angle; LED-array IDT sums incoherently over angles.

**Ideal fidelity.** I4 multislice/MLB/SSNP for multiply scattering cells and organisms. Pupil aberration and illumination-angle self-calibration. S4 neural-field samples.

**Sizes and memory.**
- Volumes of 128³ to 512²×200 with 30–200 illuminations; reconstruction is typically batch = 1.
- One 512² complex64 field is 2 MiB. Backprop through 200 slices stores at least 400 MiB of fields per angle, and roughly 1 GB once FFT intermediates are counted. At 100 angles that is about 100 GB.
- **Mini-batching over illumination angles, gradient checkpointing across slices, and (v2) adjoint/reversible propagation** are therefore mandatory. Kellman's memory-efficient learning cut FPM memory from 5.69 GB to 0.062 GB at <2× compute [V](https://arxiv.org/abs/2003.05551).

**Learned parameters.** The RI volume (voxels or a neural field), illumination k-vectors and intensities, pupil, focus offset.

### 2.7 Light-sheet imaging in scattering tissue

**Published works.**
- **biobeam** (Weigert et al., PLoS Comput. Biol. 2018) is GPU (OpenCL) BPM that combines 10⁵–10⁶ multiplexed PSF computations. It reproduces spatially varying aberrations, distortions and adaptive-optics effects [V](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006079).
- **PhaseNet** for sensorless aberration estimation is trained *only on simulated* Zernike PSFs: 5 slices within ±1 µm, first 6 Zernike orders, 4 ms per image. It transfers to real data (Saha et al., Opt. Express 28:29044, 2020) [V](https://arxiv.org/abs/2006.01804).
- Deconvolution trained on synthetic images from the known PSF for Airy/propagation-invariant light sheets (Light Sci. Appl. 2022) [V](https://www.nature.com/articles/s41377-022-00975-6).
- CARE (Weigert et al., Nat. Methods 2018) uses semi-synthetic degradations [R].

**Minimum fidelity.**
- S3 fluorophore density.
- An analytic sheet profile (Gaussian waist vs x, or Bessel/Airy/lattice via a pupil-plane annulus/cubic phase).
- Excitation × density, then a spatially invariant detection PSF.
- Beer–Lambert attenuation and depth-dependent blur/background as phenomenological scattering.

**Ideal fidelity.**
- I4 BPM of the illumination through the tissue RI volume.
- Spatially varying detection PSFs computed by propagating from emitter grid points through the tissue, as biobeam does, then interpolating.
- Tissue RI statistics (fractal/Gaussian random fields).
- Adaptive-optics correction modes.

**Sizes.** Training patches of 64³–128³; whole volumes up to 2048²×1000. Full-fidelity simulation needs chunking. This is **v2**.

**Learned parameters.** Aberration Zernikes and AO corrections, sheet waist and offset, tissue scattering statistics, attenuation.

### 2.8 Label-free virtual staining

**Published works.** Christiansen et al. (Cell 2018) in-silico labelling [R]; Ounkomol et al. (Nat. Methods 2018) [R]; Rivenson et al. (Nat. Biomed. Eng. 2019) [R]. *Cooke et al.*, "Physics-enhanced machine learning for virtual fluorescence microscopy", jointly learns LED-array illumination and the network [V](https://arxiv.org/abs/2004.04306).

**Role of simulation.** Training pairs are almost always real. Simulation helps in two ways:
- Pretraining with synthetic pairs, which requires rendering **one scene in several modalities**: RI for brightfield/QPI plus fluorophore channels.
- Optimising the illumination.

**Fidelity.**
- Minimum: S3 volumes with per-material RI *and* fluorophore labels, C2 brightfield, fluorescence PSF.
- Ideal: realistic organelle geometry and texture.

**Architectural consequence.** A scene must be renderable by several optical systems. The sample model therefore holds *materials with multiple optical properties*, not "an RI volume" or "a fluorescence volume".

### 2.9 Fourier ptychography (FPM)

**Published works.** Zheng, Horstmeyer & Yang (Nat. Photon. 2013) [R]; multiplexed and 3D multislice FPM (Tian et al., 2014/2015) [R]; learned LED-array design (Kellman et al., IEEE TCI 2019, arXiv 1808.03571) [V](https://arxiv.org/abs/1808.03571); *Horstmeyer et al.*, "Convolutional neural networks that teach microscopes how to image", which jointly learns illumination and classifier [V](https://arxiv.org/abs/1709.07223); Chromatix FPM example [V].

**Forward model.** `I_j = |F⁻¹{ P(k) T̂(k − k_j) }|²`. A thin sample (S2), coherent per LED, with multiplexed LEDs summed incoherently. Multislice is used for thick samples.

**Minimum ≈ ideal.** Scalar, thin object, with pupil and LED-position calibration. Adding I4 multislice covers 3D FPM.

**Sizes.** Patches of 128²–512² × 100–300 LEDs [R]. Patches must be tiled across the full sensor.

**Learned parameters.** Complex object, pupil (EPRY-style), LED positions/angles/intensities, illumination patterns (design).

### 2.10 Structured illumination microscopy (SIM)

**Published works.** Gustafsson (J. Microsc. 2000; 3D-SIM 2008) [R]. **ML-SIM** trains a network *purely on simulated* SIM stacks and generalises to real microscopes (Christensen et al., Biomed. Opt. Express 12:2720, 2021) [V](https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-5-2720&id=450173). I believe it used natural-image datasets as sample textures [R/?]. DFCAN/rDL-SIM use real BioSR data [R].

**Forward model.** `I_{θ,φ} = h ⊛ [ρ · (1 + m cos(2π p_θ·r + φ))]`. 3D-SIM uses three-beam interference patterns with axial structure.

**Minimum fidelity.** A 2D widefield pupil PSF with sinusoidal excitation, S2 textures, D1 noise.

**Ideal fidelity.** 3D patterns from coherent pupil-plane beams; vectorial pattern contrast (modulation depth depends on polarisation at high NA); pattern distortions; out-of-focus background; bleaching across the 9–15 raw frames.

**Sizes.** 9–15 frames × 256²–512², batches of 4–16.

**Learned parameters.** Pattern k, phase and modulation depth (classical SIM calibration), OTF/pupil, noise; pattern design.

**Architectural consequence.** SIM, light-sheet, TIRF (evanescent `exp(−z/d)` excitation), confocal and ISM are all **"excitation field × fluorophore density → detection optics (± pinhole/detector array)"**. One generic excitation/detection factorisation covers all of them.

### 2.11 Confocal, ISM (and STED)

**Models.** Confocal: `h_conf = h_exc · (h_det ⊛ pinhole)`. ISM: per-element `h_exc(r) · h_det(r − d)` [R]. Simulation-driven ISM learning exists (Opt. Express 2022, PubMed 35473120) [V](https://pubmed.ncbi.nlm.nih.gov/35473120/). *microsim* covers widefield/confocal with scalar and vectorial PSFs in NumPy/JAX/CuPy (Torch planned). It does not claim differentiability [V](https://github.com/tlambert03/microsim).

**pySTED** has vectorial excitation/depletion PSFs, a two-state photophysics model and a bleaching model. It is validated against real STED and used to train RL agents that transfer sim-to-real. It is *not differentiable* and works on 224×224 training images [V](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491398/).

**Minimum fidelity.** Scalar product PSFs. **Ideal fidelity.** Vectorial depth-dependent PSFs, saturation and bleaching.

STED, saturation and 2-photon excitation are *pointwise nonlinear functions of excitation intensity*, not coherent nonlinear optics. They are cheap to add as "excitation response" functions, so they belong in **v2**.

### 2.12 Darkfield nanoparticle imaging

**Task.** Tracking and spectroscopy of Au/Ag nanoparticles and nanorods, and label-free tracking of vesicles and bacteria.

**Minimum fidelity.** A quasi-static dipole polarisability for spheres, or a Gans ellipsoid for rods, with dispersive ε(λ) from tabulated Johnson–Christy data [R]. An annular condenser (NA_c > NA_obj) drops the zero order. The image is coherent within each illumination angle and incoherent (C2) across the annulus.

**Ideal fidelity.** Mie for larger particles, T-matrix for rods, P2 vectorial imaging (rod orientation → polarisation), spectral integration over source spectrum × camera QE (RGB cameras), and a substrate interface.

**Learned parameters.** Size, aspect ratio, orientation, dielectric function, illumination spectrum.

### 2.13 Polarisation microscopy

**Published works.** LC-PolScope (Oldenbourg) [R]; QLIPP (Guo et al., eLife 2020) and permittivity tensor imaging (Yeh et al., Nat. Methods 2024), both implemented in waveorder with single-scattering transfer functions [V](https://github.com/mehta-lab/waveorder). Fluorescence polarisation and dipole orientation are covered by Deep-SMOLM [V].

**Minimum fidelity.** P3 thin Jones samples (retardance, slow axis, diattenuation) imaged per polarisation component, with a 2×2 Jones pupil.

**Ideal fidelity.** Tensor-ε volumes with vectorial high-NA imaging (3D field components; Richards–Wolf mixing) and weak scattering (PTI) or vectorial BPM.

**Architectural consequence.** Fields must carry an optional polarisation channel axis (1, 2 or 3 components). Elements act as Jones matrices on that axis, and scalar is the 1-component special case. Retrofitting this later is expensive.

### 2.14 End-to-end optical design (learned phase masks, pupils, illumination)

**Published works.**
- DeepSTORM3D's learned mask, and the learned PSF pair [V].
- Horstmeyer 2017 learned LED illumination [V].
- Muthumbi et al., "Learned sensing" (Biomed. Opt. Express 2019), which jointly optimises illumination and pupil transmission for malaria classification [R].
- Kellman 2019 coded illumination for QPI [V].
- Cooke 2020 [V].
- *Pinkard et al.*, "Information-driven design of imaging systems" (arXiv 2405.20559), which uses mutual-information estimates as a design objective [V](https://arxiv.org/abs/2405.20559).
- Differentiable microscopy ∂µ (arXiv 2203.14944) [V].
- Sitzmann et al. (SIGGRAPH 2018), from computational photography [R].
- **Chromatix** (JAX, Nature Methods 2026) is a general differentiable wave-optics library: fields with sampling metadata, lenses, phase/amplitude masks, polarisation, multiple wavelengths, sensors with shot noise; demos include PSF engineering, FPM, CGH and Zernike phase retrieval. Its README mentions no Mie scattering or fluorescence-specific physics [V](https://github.com/chromatix-team/chromatix).

**Requirements.**
- Gradients with respect to pupil phase/amplitude (pixel, Zernike, or SLM with pixelation, phase wrapping and quantisation via straight-through estimators; DM influence functions), LED weights and angles, pattern parameters, polarisation states.
- Discrete choices such as wavelength or channel count go through relaxations or outer-loop search.
- Physical-realisability constraints.
- *Speed*: the simulator sits inside every training step.

### 2.15 System identification (pupil and aberration retrieval, noise calibration)

**Published works.**
- Phase-retrieved pupils from bead stacks (Hanser et al., J. Microsc. 2004) [R].
- uiPSF and FD-DeepLoc [V].
- In-situ PSF retrieval from blinking data (INSPR; Xu et al., Nat. Methods 2020) [R].
- PhaseNet [V].
- sCMOS per-pixel offset/variance/gain calibration (Huang et al., Nat. Methods 2013) [R].
- EMCCD Gamma–Poisson models [R].

**Requirements.**
- **MLE with correct likelihoods**: Poisson; Poisson+Gaussian, approximated as Gaussian with variance μ+σ² or as a shifted Poisson; EMCCD.
- Hundreds of ROIs batched with per-ROI nuisance parameters.
- Levenberg–Marquardt or L-BFGS.
- Fisher information and CRLB via autograd Jacobians of the expected image, which is also the classic PSF-design objective (Shechtman et al., PRL 2014 [R]).

**Sizes.** 10–1000 beads × 21²–41² px × 20–80 z-planes.

### 2.16 Sim-to-real distribution matching

**Published works.**
- *General ML*:
  - "Learning to Simulate" (Ruiz et al., ICLR 2019) uses REINFORCE on simulator parameters [V](https://arxiv.org/abs/1810.02513).
  - **Meta-Sim** (Kar et al., ICCV 2019) learns a distribution transformer over scene grammars using MMD in Inception-feature space plus a task loss [V](https://github.com/nv-tlabs/meta-sim).
  - **AutoSimulate** (Behl et al., ECCV 2020) uses a differentiable local approximation of the bilevel objective and is up to 50× faster than REINFORCE-style methods [V](https://arxiv.org/abs/2008.08424).
- *Microscopy*:
  - SyMBac's manual histogram/spectrum matching [V].
  - DeepTrack's Optuna tuning [V].
  - pySTED's RL sim-to-real [V].
  - **OSOG** (arXiv 2606.21381), a single-author, unreviewed preprint. It claims a PyTorch-native structure-of-arrays differentiable engine using OPD-based wave optics, inverse rendering of continuous optical parameters, and zero-shot sim-to-real detection. Its capability claims (and the <50 ms figure for 40k particles) should be treated with caution [V/?](https://arxiv.org/abs/2606.21381).

**Objective families gradoscopy must enable.**
1. Per-image likelihood or reconstruction, when latents can be jointly inferred (system ID, inverse problems).
2. Summary-statistic matching: histograms, radially averaged power spectra, noise mean–variance curves, PSF-width statistics.
3. MMD with fixed or learned feature kernels.
4. Adversarial training (simulator as GAN generator).
5. Simulation-based inference (neural posterior estimation) over simulator parameters.
6. Task-driven bilevel objectives (validation loss on a few labelled real images).

**Requirements.** Objectives 2–4 need **pathwise gradients through random scene generation**:
- Reparameterised continuous distributions (`rsample`, including implicit reparameterisation for Gamma/Beta [R]).
- Relaxed discrete counts (fixed N_max with soft or Gumbel presence weights).
- Differentiable *soft* rasterisation of primitives, so that positions and sizes have gradients.
- Surrogate gradients for Poisson noise: a Gaussian `λ + √λ·ε` for λ ≳ 10–20, a straight-through estimator, or a score function.
- A per-variable gradient-estimator policy.

### 2.17 Physics-informed reconstruction with the same forward model

**Published works.** DeCAF [V]; unrolled physics-based networks (Kellman) [V]; Richardson–Lucy networks (Li et al., Nat. Methods 2022) [R]; diffusion posterior sampling with a known operator (Chung et al., ICLR 2023) [R]; deep image prior [R].

**Requirements.**
- The *same* forward operator in deterministic mode.
- Efficient vector–Jacobian products (the adjoint).
- A likelihood term.
- Tiling for large FOVs.
- The sample representation replaceable by any `nn.Module` (voxel grid, neural field, generator output).

### 2.18 Brief notes on the remaining modalities
- **TIRF / HiLo / widefield deconvolution.** Covered by the excitation × density factorisation (evanescent excitation) and pupil PSFs.
- **Phase contrast and DIC for non-cell samples** (colloids, crystals). Same pupil machinery as 2.5.
- **Lensless / in-line digital holography.** Angular-spectrum propagation to a sensor without an objective. It is trivially covered once the propagators exist.

---

## 3. Requirements matrix

Legend: **M** means the capability is needed at *minimum* fidelity; **I** means it is needed only at *ideal* fidelity; a blank cell means not needed.

| Application | Scalar pupil PSF (P1) | Vectorial/dipole (P2) | Sparse per-object render | Dense 3D fluo conv | Excitation patterns | Thin/projection coherent | Partial coherence (Abbe) | Mie/dipole (I3) | Multislice (I4) | Born/Rytov (I2) | Interferometric/pupil stops | Jones/polarisation | Camera noise + likelihood | Differentiable random scenes | Dynamics/photophysics | Spatially varying PSF | Memory-efficient grads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Particle tracking | M | I | M | I | | M | I | M | I | | M | | M | M | M | I | |
| Holography (Mie) | M | I | M | | | | | M | | | M | I | M | M | I | | |
| iSCAT / MP | M | I | M | | | | I | M | | | M | I | M | M | I | | |
| SMLM | M | I | M | | | | | | | | | I | M | M | M | I | |
| Cell seg./track | M | | | M | | M | I | | I | M | M | I | M | M | I | | I |
| QPI / ODT | M | | | | | M | M | | I | M | M | | I | | | | M |
| Light-sheet tissue | M | I | | M | M | | | | I | | | | M | M | | I | I |
| Virtual staining | M | | | M | | M | M | | I | M | M | | M | M | | | I |
| FPM | M | | | | | M | M | | I | | M | | M | | | I | M |
| SIM | M | I | | M | M | | | | | | | | M | M | I | | |
| Confocal/ISM | M | I | | M | M | | | | | | M | | M | M | I | | |
| Darkfield NPs | M | I | M | | | | M | M | | | M | I | M | M | M | | |
| Polarisation | M | I | | | | M | M | | I | I | | M | M | | | | |
| End-to-end design | M | I | M | M | M | M | M | I | I | | M | I | M | M | | | M |
| System ID | M | M | M | | | | | | | | | | M | | | M | |
| Sim-to-real | (inherits) | | | | | | | | | | | | M | M | | | |
| Physics-informed recon | M | I | | M | | M | M | | M | M | M | | M | | | | M |

**Reading the matrix.**
- The scalar pupil, sparse rendering, thin/projection coherent imaging, Mie/dipole, camera likelihoods and differentiable random scenes each appear as **M** in most rows. They are the non-negotiable v1 core.
- Vectorial PSFs are **I** almost everywhere and **M** only for system ID at high NA (FD-DeepLoc, uiPSF). Because vectorial PSFs are cheap to add on top of a pupil engine (6 pupil transforms instead of 1), they belong in v1 anyway.
- Multislice is **I** broadly but **M** for physics-informed 3D reconstruction.
- Spatially varying PSFs and memory-efficient gradients are niche but critical where they appear.

---

## 4. Compute and memory budget (engineering estimates)

- **The data-generation throughput target is set by the training step.** A U-Net step on 32×256² images typically takes 10–50 ms on a modern GPU [R]. To avoid becoming the bottleneck, "fast" tiers (L0–L1) should render at least about 1000 images/s at 256². DeepTrack's per-feature Python graph evaluation is, in my assessment, the main throughput limiter today [R/?]. Struct-of-arrays batching avoids it.
- **Sparse PSF rendering.** Consider a pupil of 64², an ROI of 32², and a matrix DFT costing about 2·64²·32 complex MACs per axis pass. With batch 64 × 50 emitters, that is under 1 GFLOP per batch in scalar mode, ×6 for vectorial. This is trivially fast, and memory is dominated by the per-emitter ROIs.
- **Mie in the pupil.** 1D S1/S2 over about 500 θ samples × n_max ≲ 100 orders per particle is negligible. The 2D pupil interpolation per particle costs about the same as one scalar emitter.
- **Partial coherence.** Cost ×N_s source points, typically 20–200. TCC/SOCS reduces this to 5–20 kernels for thin objects only.
- **Multislice ODT.** About 1 GB of activations per illumination angle at 512²×200 (§2.6). This requires angle mini-batching and checkpointing in v1, and adjoint/reversible propagation in v2.
- **Full light-sheet in tissue** (biobeam-style, 10⁵–10⁶ PSFs) is minutes to hours per volume. It is v2 and needs chunked streaming.

---

## 5. Proposed scope

### v1: must ship (covers the minimum fidelity of ~16 of 18 families and DeepTrack parity)

1. **Scene and sample.**
   - S0 emitters with optional dipole orientation.
   - S1 primitives: sphere, ellipsoid, capsule/spherocylinder, layered sphere, with soft anti-aliased rasterisation.
   - S2 maps and S3 voxel volumes of complex RI and fluorophore density.
   - Materials carrying RI (complex, optionally dispersive via a callable), fluorophore brightness/label and polarisability.
   - Any `nn.Module` may act as a sample field, which enables neural fields.
2. **Illumination.**
   - Plane waves (angle, λ, polarisation) and sets of plane waves: Köhler disc or annulus source sampling, LED arrays.
   - Interference patterns for SIM.
   - Analytic Gaussian/Bessel sheets and TIRF evanescent excitation.
   - Few-wavelength polychromatic sums.
3. **Interaction models (tiered).**
   - I1 incoherent emission (excitation × density).
   - I1 projection/thin object.
   - I2 first Born/Rytov.
   - I3 Rayleigh dipole, Mie, layered Mie. Particles are superposed without mutual coupling.
   - I4 multislice: paraxial plus exact-`kz` angular spectrum, with checkpointing.
4. **Imaging.**
   - Pupil-based scalar (P0/P1) and vectorial (P2, Richards–Wolf, dipole emission, single-interface RI mismatch) PSFs.
   - Zernike (with a documented index convention) and pixel pupils.
   - Per-object pupils, which give field-dependent aberrations in sparse mode.
   - ROI rendering via matrix DFT/chirp-z, and full-frame FFT convolution for dense volumes.
   - Interferometric references (in-line and off-axis holography, iSCAT reflection) computed with explicit interference terms.
   - Pupil stops and filters: darkfield, phase ring, DIC shear/bias, DPC half-pupils.
   - Confocal pinhole and ISM.
   - A thin-Jones polarisation channel axis (P3 thin).
5. **Detection.**
   - Magnification and pixel integration (supersample + area average).
   - QE, gain, offset, quantisation, saturation.
   - Poisson, Gaussian read noise, sCMOS maps, EMCCD, each with `sample()`, `log_prob()` and a documented surrogate gradient.
   - Background generators: gradients, and procedural textures with *learnable spectra*.
6. **Stochastic scene generation.**
   - Reparameterised distributions as `nn.Module`s.
   - Relaxed particle counts.
   - Non-overlap via soft penalties or rejection with a stated non-differentiability.
   - T1 Brownian/drift with motion blur.
   - T2 two-state blinking.
7. **Inference utilities.** MLE fitting (LM/L-BFGS), Fisher information/CRLB, and a minimal sim-to-real loss library: histogram/Wasserstein-1D, radial power spectrum, mean–variance curve, MMD.
8. **Validation.**
   - Golden tests against HoloPy/pylorenzmie (Mie holograms) and against published vectorial PSFs.
   - Cross-tier convergence: Born → Mie for small Δn; scalar → vectorial at low NA.
   - A float64 mode.

### v2: planned extensions
- Stratified-media near-field physics: dipoles near interfaces, SAF, the full iSCAT iPSF.
- Wide-angle multiple scattering (MLB, SSNP).
- TCC/SOCS acceleration.
- Axisymmetric T-matrix for spheroids and rods; coupled multisphere clusters.
- Spatially varying PSFs for dense volumes (PSF interpolation/low-rank mixing).
- biobeam-class light-sheet-in-tissue simulation.
- Tensor-ε samples with vectorial propagation (PTI-class).
- Mesh import (rasterisation or analytic polyhedron Fourier transforms).
- Photophysics nonlinearities: saturation, STED depletion, 2-photon.
- Spectral multichannel with dispersion and crosstalk.
- Large-FOV tiling and stitching (FPM, whole-slide).
- Adjoint/reversible propagation.
- Adversarial and SBI toolkits.

### v3 / research plugins
- A Lippmann–Schwinger / convergent Born series solver for strongly scattering volumes.
- A learned surrogate "neural operator" tier trained against higher tiers.

### Explicitly out of scope
- **FDTD, FEM and DDA solvers inside the engine.**
  - Why: the survey shows that no deep-learning data-generation application needs them online. Their role is validation, as in Mahmoodabadi's FDTD check of the iPSF.
  - Instead: support *importing precomputed far-field amplitude tables or T-matrices* through the same scatterer interface as Mie.
- **Coherent nonlinear optics** (SHG, THG, CARS, SRS) and Raman spectral imaging. None of the surveyed data-generation pipelines depend on them.
- **Time-resolved photon statistics** (FLIM TCSPC, antibunching). Only a simple per-fluorophore lifetime *label* belongs in scope.
- **Monte Carlo diffuse photon transport** (deep tissue). Replace it with phenomenological attenuation, blur and background.
- **Macroscopic lens design and ray tracing.** Objectives are modelled as ideal aplanatic systems plus pupil aberrations. Every surveyed work parameterises optics this way.
- **Biological mechanics and growth.** Keep this in external scene generators behind a plugin interface; SyMBac used Pymunk.
- **Electron, X-ray and acoustic imaging.**
- **Instrument control.**

---

## 6. Learnable-parameter use cases that must be first-class

1. **Per-object physical parameters.** Position (xyz), radius, RI (complex), polarisability, photons, dipole orientation, shell thickness. Used by inverse problems, fitting and CRLB.
2. **Distribution hyperparameters of the generator.** Means and variances of size, RI, SNR, density, defocus range, texture spectrum. This is the core of sim-to-real. It requires `rsample`, relaxed counts and soft rasterisation.
3. **Pupil and aberrations.** Zernike coefficients, pixel phase and amplitude, field-dependence coefficients, defocus/focal offset, effective NA, medium, immersion and coverslip RI and thickness.
4. **Engineered optics (design).** Phase masks with SLM/DM constraints, amplitude masks, LED weights and angles, SIM pattern parameters, polarisation states. Discrete choices go through relaxation or outer-loop search.
5. **Illumination calibration.** k-vectors and LED positions, intensities, condenser NA (spatial coherence), illumination non-uniformity.
6. **Sample fields as modules.** Voxel RI and fluorophore volumes, neural fields (DeCAF), generator networks producing textures.
7. **Detector parameters.** Gain, offset, read noise, per-pixel maps, QE, EM gain, pixel crosstalk (a pixel PSF).
8. **Nuisance and background models.** Static iSCAT phase screens, autofluorescence, illumination gradients, debris.
9. **Dynamics and photophysics.** Diffusion coefficients, drift, blinking and bleaching rates.
10. **Fidelity and discrete configuration.** Tier choice, number of source points, oversampling factor. These are not differentiable, but they must be *programmatically sweepable* for accuracy/cost trade-offs and hyperparameter search.

Cross-cutting requirements:
- Parameters can be **shared** across modalities: the same sample rendered by brightfield and fluorescence systems.
- Parameters can be **per-sample or shared across a batch**, which is broadcasting semantics.
- Every random variable declares its **gradient estimator**: pathwise, straight-through, score-function or none.

---

## 7. Implications for gradoscopy's architecture

1. **Make the angular spectrum at the entrance pupil the canonical interface between "interaction" and "imaging".**
   - Every interaction model emits either a coherent exit field on a plane (dense path) or an angular-spectrum/far-field amplitude per object and mode (sparse path).
   - Imaging consumes pupil fields: aberrations, masks, stops, rings, DIC kernels and vectorial Richards–Wolf mapping all act there, then a pupil → image transform follows (FFT or ROI DFT).
   - This single decision unifies fluorescence emitters, Mie/dipole scatterers, thin objects, Born scatterers and multislice exit fields. It also makes learned-optics design trivially available to every modality.

2. **Build two render paths from day one, sparse and dense, and let them compose.**
   - Most deep-learning data generation is sparse: tracking, SMLM, holography, iSCAT, darkfield.
   - Cells, tissue and ODT are dense.
   - A hybrid scene (dense cell background plus sparse particles) sums coherently (fields) or incoherently (intensities), depending on the coherence model.
   - Sparse per-object pupils give field-dependent aberrations for free.

3. **Treat "modes" as a first-class batched axis with a configurable reduction.**
   - Source points (partial coherence), wavelengths, polarisation states, dipole orientations and incoherent emitters are all "compute coherent fields per mode, then sum intensities".
   - Use tensors of shape `(B, M, C, H, W)` (C = polarisation components), with chunked, memory-bounded reduction.
   - The *number of modes is the primary fidelity/cost knob*.

4. **Make fidelity explicit, discrete and per stage, with named presets and validity diagnostics.**
   - Presets: L0 phenomenological; L1 scalar paraxial; L2 scalar non-paraxial with exact single scatterers and partial coherence; L3 vectorial with multiple scattering; L4 external/rigorous.
   - Each preset maps to per-stage model choices, with per-object overrides (for example, Mie for particles and projection for the background).
   - Each model reports its validity regime: size parameter, k0·Δn·L, NA > ~0.7 for scalar, sampling. Warnings are cheap and raising errors is optional.
   - Cross-tier convergence tests are what make the tiers genuinely interchangeable approximations *of the same scene description*.

5. **Keep the scene description physical and fidelity-agnostic; objects expose multiple "views".**
   - A `Sphere` should provide its analytic Fourier transform, a soft voxelisation, a projected thickness, Mie coefficients and a dipole limit.
   - Renderers request the view they need.
   - This is the mechanism for being "more than the sum of its parts": adding one scatterer type makes it available to every modality, and adding one modality makes it available for every scatterer.

6. **Represent scenes as batched struct-of-arrays tensors with presence masks, not as Python object graphs.**
   - Scene tensors are `(B, N_max, …)` per primitive type.
   - Distributions are `nn.Module`s with `rsample`.
   - Counts are relaxed via soft presence.
   - A DeepTrack-style declarative front end can compile to this representation.

7. **Model detectors as probabilistic objects** with `mean()`, `sample()`, `log_prob()` and surrogate gradients. MLE, CRLB, SBI and likelihood-based sim-to-real then need no special-casing, and system identification (uiPSF-, FD-DeepLoc-style) becomes a thin layer on top of the engine.

8. **Engineer memory from the start.**
   - Gradient checkpointing across multislice.
   - Mini-batching over illumination angles and modes.
   - Custom autograd for special-function recurrences: Mie Riccati–Bessel functions, since naive autograd through recurrences is memory-hungry and numerically fragile.
   - complex64 by default, with a float64 validation mode.
   - Explicit interference-term computation for weak-scatterer contrast.

9. **Provide an automatic, unit-aware sampling planner.**
   - It derives internal grid spacing (≤ λ/(4·NA) or finer), padding, pupil sampling and oversampling for pixel integration from NA, λ, pixel size and z range.
   - Silent aliasing is the most common way optics simulators produce plausible but wrong images.

10. **Use DeepTrack parity plus reference pipelines as acceptance tests.**
    - Reproduce DeepTrack's `Fluorescence`, `Brightfield`, `Holography`, `ISCAT`, `Darkfield`, `IlluminationGradient`, `MieSphere`, `MieStratifiedSphere`, the voxel scatterers, noises and aberrations.
    - Add reference pipelines:
      - DECODE-like SMLM data generation (40² frames, EMCCD/sCMOS camera config).
      - A learned-PSF design loop (DeepSTORM3D-like).
      - Mie hologram fitting (matching HoloPy/pylorenzmie to tolerance).
      - SyMBac-like phase-contrast cell generation with *automatic* histogram/spectrum matching.
      - Bead-stack pupil retrieval (uiPSF-like).
      - Small multislice ODT with angle mini-batching.
      - A SIM stack generator.

11. **Keep solvers pluggable behind stable interfaces, and do not build what is out of scope.** Rigorous scatterers enter as precomputed amplitude or T-matrix tables. Biology and tissue enter as scene-generator plugins. Heavy solvers (Lippmann–Schwinger, biobeam-class light-sheet) are v2+ modules that implement the same interaction interface.

12. **Position the project against its neighbours.**
    - Chromatix is general wave optics in JAX with no scatterer or fluorescence physics in its README.
    - waveorder is PyTorch but single-scattering and label-free focused.
    - microsim is realistic fluorescence but not differentiable.
    - HoloPy and pylorenzmie are exact scattering but not differentiable or batched for deep learning.
    - uiPSF is TensorFlow and focused on PSF modelling.
    - DECODE uses a spline PSF.
    - pySTED is not differentiable.
    - **No existing library combines exact particle scattering, fluorescence, partial coherence, vectorial PSFs, probabilistic detectors and differentiable stochastic scene generation in PyTorch.** That combination is gradoscopy's reason to exist, and v1 should be scoped to deliver exactly it.

---

## Sources (checked this session)
- DeepTrack2 repository and source: https://github.com/DeepTrackAI/DeepTrack2 ; `deeptrack/optical/optics.py` and `scatterers.py` (develop branch)
- Chromatix: https://github.com/chromatix-team/chromatix ; Nat. Methods (2026) https://www.nature.com/articles/s41592-026-03121-x ; bioRxiv 10.1101/2025.04.29.651152
- DeepSTORM3D code/physics: https://github.com/EliasNehme/DeepSTORM3D ; PSF pair: https://arxiv.org/abs/2009.14303
- DECODE: https://github.com/TuragaLab/DECODE (reference.yaml)
- uiPSF: https://pmc.ncbi.nlm.nih.gov/articles/PMC12330227/
- FD-DeepLoc: https://github.com/Li-Lab-SUSTech/FD-DeepLoc ; https://www.nature.com/articles/s41592-023-01775-5
- Deep-SMOLM: https://github.com/Lew-Lab/Deep-SMOLM
- SyMBac: https://pmc.ncbi.nlm.nih.gov/articles/PMC9710168/
- Cell Tracking Challenge: https://celltrackingchallenge.net/2d-datasets/
- CATCH: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b10463 ; HoloPy theories: https://holopy.readthedocs.io/en/master/reference/holopy.scattering.theory.html
- Off-axis holography DL: https://pubs.acs.org/doi/10.1021/acsnano.0c06902
- LodeSTAR: https://www.nature.com/articles/s41467-022-35004-y
- iSCAT iPSF: https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-18-25969&id=434558 ; sub-10 kDa: https://www.nature.com/articles/s41592-023-01778-2
- Multi-layer Born: https://opg.optica.org/optica/fulltext.cfm?uri=optica-7-5-394&id=431219 ; SSNP: https://arxiv.org/abs/2207.06532
- DeCAF: https://www.nature.com/articles/s42256-022-00530-3
- waveorder: https://github.com/mehta-lab/waveorder
- biobeam: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006079
- PhaseNet: https://arxiv.org/abs/2006.01804
- Light-sheet DL deconvolution: https://www.nature.com/articles/s41377-022-00975-6
- ML-SIM: https://opg.optica.org/boe/fulltext.cfm?uri=boe-12-5-2720&id=450173
- pySTED: https://pmc.ncbi.nlm.nih.gov/articles/PMC11491398/
- microsim: https://github.com/tlambert03/microsim
- Simulation-driven ISM: https://pubmed.ncbi.nlm.nih.gov/35473120/
- Kellman memory-efficient learning: https://arxiv.org/abs/2003.05551 ; learned coded illumination: https://arxiv.org/abs/1808.03571 ; LearnedDesignFPM: https://github.com/kellman/LearnedDesignFPM
- Horstmeyer 2017: https://arxiv.org/abs/1709.07223 ; Cooke 2020: https://arxiv.org/abs/2004.04306 ; Pinkard 2024: https://arxiv.org/abs/2405.20559 ; ∂µ: https://arxiv.org/abs/2203.14944
- Newby 2018: https://www.pnas.org/doi/10.1073/pnas.1804420115
- Learning to Simulate: https://arxiv.org/abs/1810.02513 ; Meta-Sim: https://github.com/nv-tlabs/meta-sim ; AutoSimulate: https://arxiv.org/abs/2008.08424
- OSOG (unreviewed preprint): https://arxiv.org/abs/2606.21381
