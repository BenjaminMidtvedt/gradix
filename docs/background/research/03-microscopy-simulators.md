# 03 — Microscopy image simulators and PSF modeling tools: a survey of how they split up the problem and how much physics they cover

Research brief for the **gradoscopy** architecture plan (a differentiable, PyTorch-based light-microscopy simulation engine intended to sit under DeepTrack2).

**Legend.** **[V]** means checked against the linked source (repo code, docs, or paper) during this survey (September 2026). **[R]** means recalled from memory and not re-checked here; treat it as likely but unconfirmed. Where a claim is uncertain, the text says so.

---

## 0. TL;DR for the architect

1. Nearly every serious tool uses the same pipeline, whether or not it says so: **ground truth (sample) → emission or scattering (photophysics, spectra, material response) → illumination → optics (the pupil or propagation) → sampling/detector → (image, labels)**. microsim spells this out in its API most clearly ([V] `ground_truth → filtered_emission_rates → optical_image → digital_image`). CytoPacq has the same three-stage split (phantom → OptiGen → AcquiGen), and so do DECODE, DeepSTORM3D and the SMLM challenge code.
2. **The pupil function (the back-focal-plane angular spectrum) is the common interface.** Hanser/pyotf, psf-generator, uiPSF, DeepSTORM3D, waveorder, Fourier ptychography, holopy's `MieLens`, DeepTrack2's `_pupil`, iSCAT PSF models and SIM beam models all meet at a complex, possibly polarized, field on a k-space disk of radius NA/λ. gradoscopy should make this a first-class type.
3. **No existing tool combines all of these:** differentiable, GPU-batched, multi-fidelity, fluorescence *and* coherent label-free, stochastic sample generation with **learnable distribution parameters**, and paired ground-truth outputs.
   - waveorder is closest on the physics side: differentiable, fluorescence plus label-free, PyTorch. It is limited to linear single-scattering transfer functions and is a reconstruction toolkit, not a data generator.
   - microsim is closest on decomposition and spectral realism. It is not differentiable and handles fluorescence only.
   - DeepTrack2 is closest on randomized pipelines across many modalities (fluorescence, brightfield, holography, iSCAT, darkfield), but its physics is simplified.
4. **Fidelity knobs already exist, but each tool has only its own:**
   - holopy: `theory='auto'`, choosing Mie, Multisphere, T-matrix or DDA for the same `Scatterer`.
   - psfmodels: Gaussian, scalar Gibson–Lanni, or vectorial.
   - psf-generator: scalar/vectorial × Cartesian/spherical.
   - microsim: `spectral_bins_per_emission_channel` and `max_psf_radius_aus`.
   - DeepTrack2: `upscale`.
   - waveorder: thin versus thick models.
   - DECODE: Delta, Gaussian or Spline PSF.

   gradoscopy should generalize holopy's split between *description* and *solver* to every stage.
5. **Several practical blockers are already documented:**
   - PyTorch has no autograd for Bessel functions ([V], psf-generator paper, as of torch 2.3).
   - `torch.distributions.Poisson` has no `rsample` [R].
   - microsim's `TorchAPI.fftconvolve` raises `NotImplementedError` [V].
   - xarray/pydantic pipelines get in the way of autograd.
   - Several key reference codes are **GPL-3**: holopy [V], PyMieDiff [V], psfmodels (Aguet C++) [V], and DECODE [R]. They cannot be vendored into an MIT/BSD engine and must be re-implemented from the papers.

---

## 1. Comparison matrix

| Tool | Simulates | Physics fidelity | Differentiable | GPU | Decomposition | Language / license |
|---|---|---|---|---|---|---|
| **microsim** | Widefield, confocal, SIM, and spinning-disk fluorescence (3D, multi-channel, crosstalk) | Vectorial Richards–Wolf PSF, spectral binning, full camera noise | No (not designed for it) | Via CuPy/JAX; torch backend incomplete | Space / Sample(labels) / Modality / ObjectiveLens / OpticalConfig(channels) / Detector / Settings | Python; BSD-3 [V] |
| **DeepTrack2** | Fluorescence, brightfield, holography, iSCAT, darkfield; randomized pipelines | Scalar pupil; slice-by-slice (fluorescence) and multislice phase propagation (coherent); Mie add-on | Partially (torch backend in progress) | Via torch | Scatterer features / Optics(Microscope) / noise features | Python; MIT [R] |
| **psfmodels** | 3D PSFs | Vectorial (Aguet), scalar Gibson–Lanni, Gaussian | No | No | PSF only | C++/pybind; GPL-3 [V] |
| **pyotf** | PSF/OTF; phase retrieval | Hanser pupil; scalar plus vectorial correction; sine/Herschel apodization | No | No (NumPy) | PSF only | Python [R: Apache-2] |
| **MicroscPSF-Py** | Scalar Gibson–Lanni PSF | Bessel-series approximation | No | No | PSF only | Python |
| **EPFL PSF Generator (Java)** | 3D PSFs | Born–Wolf, Gibson–Lanni, Richards–Wolf, variable-RI and others [R] | No | No | PSF only | Java/ImageJ |
| **psf-generator (EPFL, PyTorch)** | 2D/3D PSFs | Scalar or vectorial Richards–Wolf; Cartesian (FFT/CZT) or spherical (Bessel); Gibson–Lanni, Zernike, apodization | **Yes** (custom differentiable Bessel) | **Yes** | PSF only (pupil → focal field) | Python/PyTorch; MIT [V] |
| **uiPSF** | Inverse PSF modeling (beads or in situ) | Voxel, pixel pupil or Zernike pupil; vectorial; field-dependent aberrations; 4Pi; multichannel | **Yes** (TensorFlow) | Yes | PSF + emitter + background + channel transforms | Python/TF |
| **DECODE (simulation)** | SMLM frame sequences | Cubic-spline experimental PSF; blinking; EMCCD/sCMOS | No autograd through the PSF | Yes (CUDA spline) | StructurePrior / EmitterSampler / PSF / Background / Camera | Python/PyTorch; GPL-3 [R] |
| **DeepSTORM3D** | 3D SMLM with learned phase mask | Scalar Fourier optics, oil/water mismatch | **Yes** (mask learned end to end) | Yes | Emitters → pupil phases → PSF → blur → noise | PyTorch |
| **SMLM Challenge 2016** | Benchmark SMLM data | Experimental bead PSFs, 4-state photophysics, EMCCD | No | No | Structure / photophysics / PSF / camera | Java (plus scripts) |
| **TestSTORM / SuReSim / SMIS** | SMLM experiments | Photophysics-heavy (SMIS: multi-state, multi-laser, sub-frame); simple PSFs | No | No | Structure / labeling / photophysics / PSF / camera | MATLAB / Java |
| **biobeam** | Light-sheet in scattering tissue | Scalar BPM (split-step); multiplexed spatially varying PSFs | No | Yes (OpenCL) | dn volume / illumination beam / BPM / detection | Python/OpenCL; BSD [R] |
| **MLB / SSNP** (Waller, Tian) | 3D phase/IDT forward models | Multiple scattering (multi-slice Born; non-paraxial split-step) | Yes (reference codes use autodiff or explicit gradients) | Yes | RI volume / illumination angle / slices / pupil | Python/MATLAB |
| **holopy** | Holograms, scattering | Exact Mie, multisphere (SCSMFO), T-matrix (Mishchenko), DDA (ADDA), lens models (`MieLens`) | No | No | Scatterer / Theory / detector schema (illumination in metadata) | Python + Fortran; GPL-3 [V] |
| **miepython / PyMieScatt** | Single-sphere Mie | Exact | No | No (numba JIT for miepython) | Particle only | Python |
| **PyMieDiff** | Core–shell Mie: cross sections, angular, near field | Exact | **Yes** | **Yes** | Particle only | PyTorch; GPL-3 [V] |
| **treams / smuthi** | T-matrix, lattices, particles near interfaces | Exact (T-matrix; smuthi adds layered media and NFM-DS for non-spheres) | No | smuthi: CUDA for coupling | Particle(s) / layers | Python (+Fortran) |
| **TorchGDM** | Green dyadic method (DDA-like), multi-scale | Full-wave in the dipole approximation | **Yes** | **Yes** | Structure / illumination / observables | PyTorch |
| **waveorder** | Phase, absorption, birefringence, fluorescence density/orientation; widefield, confocal, light-sheet, oblique | First Born / weak object; linear transfer functions; vector Green's tensor | **Yes** | **Yes** | Specimen property vector / illumination pupil / scattering (Green's tensor) / detection pupil | PyTorch; BSD-3 [V] |
| **chromatix** (JAX) | General wave optics (PSF engineering, FP, holography) | Scalar and vector fields, multislice, polychromatic | **Yes** | **Yes** | Field / sources / elements / sensors | JAX |
| **SIM tools** (fairSIM, SIMToolbox, microsim `SIMIllum`, mcSIM, OpenSIM) | SIM raw data (mostly reconstruction tools) | Microsim: beamlet-based 3-beam interference; mcSIM: DMD diffraction | No | mcSIM: CuPy | Illumination pattern × sample → PSF | Java / MATLAB / Python |
| **FP tools** (LearnedDesignFPM, PtyLab) | FP intensity stacks; learned LED patterns | Thin-object coherent pupil model; PtyLab adds multislice and mixed states | LearnedDesignFPM: **Yes** | Yes | Object spectrum / LED k-vectors / pupil / sensor | PyTorch / Py/M/Julia |
| **iSCAT / mass photometry** (Mahmoodabadi iPSF, PiSCAT, Becker 2023, DeepTrack `ISCAT`) | Interferometric PSF, contrast versus mass | Vectorial iPSF with index mismatch; Fourier optics plus atomistic polarizability | Mostly no (DeepTrack partially) | — | Scatterer polarizability / reference field / pupil | Python / MATLAB |
| **Cell phantoms** (CytoPacq, SimuCell, SIMCEP, CellOrganizer) | Synthetic cell populations with ground truth | Geometric/statistical phantoms; simple optics (Gaussian or measured PSF) | No | No | Phantom / optics / acquisition | Web service / MATLAB |

---

## 2. Tool-by-tool notes

### 2.1 End-to-end fluorescence simulators

#### microsim (tlambert03/microsim) — the reference design for decomposition

Sources: [repo](https://github.com/tlambert03/microsim), [docs: stages](https://talleylambert.com/microsim/stages/), [simulation.py](https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/simulation.py). All [V] unless noted.

**Top-level object.** A pydantic `Simulation` with these fields:

```
truth_space: Space
output_space: Space | None
sample: Sample
modality: Modality = Widefield()
objective_lens: ObjectiveLens
channels: list[OpticalConfig] = [FITC]
detector: Detector | None          # discriminated on "camera_type"
exposure_ms: float = 100
settings: Settings
output_path
```

**Spaces.** `ShapeScaleSpace`, `ExtentScaleSpace` and `ShapeExtentSpace` are absolute. `DownscaledSpace` and `UpscaledSpace` are relative to the truth space. The output is derived from the truth space by `img.coarsen(...).sum()`, and only integer downscaling is supported. The truth space is typically about 10 nm voxels; the example in the docs is `scale=(0.02, 0.01, 0.01)` µm with `downscale: 8`.

**Sample.**
- `Sample(labels=[FluorophoreDistribution(distribution, fluorophore, concentration)])`.
- `distribution` is a union `MatsLines | CosemLabel | FixedArrayTruth | RenderableType`.
- `BaseDistribution` is an ABC. Its abstract `render(space, xp)` returns an xarray `DataArray`, and `cache_path()` and `is_random()` help with caching.
- `CosemLabel` pulls real organelle segmentations from Janelia's OpenOrganelle/COSEM zarr pyramids (4 nm base level). It then picks the pyramid level whose scale matches the truth space. This is a strong pattern: use real segmentations as the geometry prior.

**Stages (method names verbatim).**
1. `ground_truth()`, dims `(F, Z, Y, X)`: fluorophore counts per label. Cached on disk, keyed by label, space and seed.
2. `filtered_emission_rates()`, dims `(C, F, W)`: photons/s per channel × fluorophore × wavelength. It is computed as absorption cross-section (from the extinction coefficient) × irradiance × QY × normalized emission spectrum × filter transmission × detector QE. Fluorophores and filters can come from FPbase.
3. `emission_flux()`, dims `(C, F, Z, Y, X)`.
4. `optical_image_per_fluor()` calls `modality.render(truth, em_rates, objective_lens, settings, xp)`. `optical_image()` then sums over F, giving `(C, Z, Y, X)`. Keeping F separate is what makes bleed-through and crosstalk simulation natural.
5. `digital_image()`: rescale to the output space, then apply detector noise and quantization.

**Modalities.** `_PSFModality` is subclassed by `Widefield`, `Confocal(pinhole_au)` and `Identity`. `render` does an `xp.fftconvolve(truth, summed_psf, mode="same")`.
- **Spectral handling.** `bin_spectrum()` groups the emission spectrum into bins (centroid wavelength and summed weight). `_summed_weighted_psf()` then builds `Σ_bins w_b · PSF(λ_b)`. This is a *per-channel effective PSF*, and a very good cheap-to-expensive fidelity knob.
- **Confocal.** `h_conf = h_ex · (h_em ⊗ pinhole_disk)`, per z-plane FFT convolution.
- The `illum/` folder has `SIMIllum2D`/`SIMIllum3D` (angles, `nphases`, `linespacing`, `NA`, `nimm`, `wvl`, `ampcenter`, `nbeamlets=31`, `spotsize`). Each diffraction order is a fan of beamlets, which approximates a partially coherent multimode-fiber spot. There is also a spinning-disc module.

**PSF.** `psf.py` implements a vectorial Richards–Wolf model.
- `vectorial_rz` computes a 2D (r, z) PSF by Simpson integration over θ with J0/J1 Bessel functions. It is then interpolated to 3D (`vectorial_psf`).
- `make_confocal_psf` handles confocal.
- There is `lru_cache` plus disk caching (`MICROSIM_CACHE`).
- There is a JAX Bessel implementation (`_jax_bessel.py`).

**Detector.** `_Camera` has `qe` (float or Spectrum), `full_well`, `read_noise`, `dark_current`, `clock_induced_charge`, `bit_depth`, `offset` and `gain`. Its `simulate()` does:
1. photons × exposure;
2. Poisson;
3. add dark current and CIC;
4. clip to full well;
5. binning (before readout for CCD/EMCCD, after for CMOS);
6. EM gain (`CameraEMCCD`: Gamma with shape equal to the electron count);
7. read noise and quantization;
8. ADC saturation.

**Backend.** `NumpyAPI`, with subclasses `CupyAPI`, `JaxAPI` and `TorchAPI` (`auto` tries CuPy, then JAX, then torch). It abstracts `fftconvolve`, `poisson_rvs`, `norm_rvs`, `map_coordinates`, `j0` and `j1`. **`TorchAPI.fftconvolve` raises `NotImplementedError`**, so torch is effectively unsupported. Arrays are wrapped in xarray (there is an `xarray_jax.py` shim).

**Settings (fidelity knobs).**
- `float_dtype`, `random_seed`.
- `max_psf_radius_aus = 6`: truncates the PSF lateral support.
- `spectral_bins_per_emission_channel = 1` (default): the docs note that more bins give more realism at a memory and time cost.
- `spectral_bin_threshold_percentage = 1`.

**Verdict.** This is the best *decomposition* and the best spectral and detector realism among open simulators. It covers fluorescence only (no coherent or label-free imaging) and is not differentiable. The heavy object model (pydantic plus xarray) is great for reproducibility but hostile to autograd and batching.

What to borrow:
- the named stage API with explicit dimension conventions (`C, F, W, Z, Y, X`);
- truth space versus output space;
- the `(C, F, W)` emission-rate tensor;
- per-bin effective PSFs;
- the camera model.

#### DeepTrack2 (the incumbent to be served)

Source: [optics.py](https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/optics.py) [V]. The package is now split into `deeptrack/optical/{optics, scatterers, holography, aberrations, noises}` and `deeptrack/backend` (including `mie`, `polynomials`, `units`). TensorFlow support was dropped in 2.x.

**Optics classes.**
- `Optics(NA, wavelength, magnification, resolution, refractive_index_medium, padding, output_region, upscale)`.
- `Fluorescence` projects each z-slice through a defocused PSF `|F⁻¹{pupil(z)}|²` and sums over z.
- `Brightfield` (`Holography` is an alias) is coherent. It propagates slice by slice through the contrast volume with a phase `exp(i·Δn·dz·K)`, applies the pupil, and can add `ScatteredField` (Mie) contributions at the detector plane. `return_field` returns the complex field.
- `ISCAT(illumination_angle, amp_factor)`, `Darkfield` and `IlluminationGradient` are also available.

**Pupil and sampling.**
- `_pupil(shape, NA, wavelength, refractive_index_medium, defocus, include_aberration)`, with both `_pupil_numpy` and `_pupil_torch` implementations.
- `upscale` renders at a finer grid, then applies average-pooling (brightfield) or sum-pooling (fluorescence). This is the equivalent of microsim's truth-to-output spaces.

**Strength.** Random *pipelines*: lazily resolved properties sampled per image. This is what users will expect gradoscopy to keep.

**Weaknesses (physics).**
- scalar only;
- no spectral axis;
- no refractive-index mismatch model by default;
- the fluorescence PSF is a stack of 2D defocus PSFs rather than a true 3D vectorial PSF;
- Mie contributions are added rather than consistently coupled with multislice;
- no explicit photophysics.

#### Cell-phantom generators: CytoPacq, SimuCell, SIMCEP, CellOrganizer

- **CytoPacq** ([V] [site](https://cbia.fi.muni.cz/research/simulations/cytopacq.html)) is a web service with three modules: **phantom** (CytoGen, MitoGen with 3D+t mitosis, FiloGen with filopodia) → **OptiGen** (blur, uneven illumination, excitation/emission filters) → **AcquiGen** (dark current, resampling, noise, amplification, ADC). It outputs masks and lineage as ground truth.
- **SimuCell** ([V] Nat Methods 2012) is a declarative MATLAB framework. It covers population → cell → organelle shapes → marker distributions, with dependencies between markers (for example, microenvironment-dependent marker localization). It can import CellOrganizer SLML generative models.
- **SIMCEP** ([V]) uses Gaussian blur plus CCD noise for focus blur.
- **CellOrganizer** ([V]) learns *generative shape and distribution models from images*.

**Lesson.** These tools invest almost everything in the **sample prior** (shape statistics, population structure, time evolution) and almost nothing in optics. In gradoscopy terms they are "sample generators": they should plug into a sample stage, and their parameters are exactly the **distribution parameters** a user will want to learn.

### 2.2 PSF models and libraries

**Physics background** (standard results [R]; the equations below are textbook forms):

- **Hanser pupil formalism** (Hanser et al., *J. Microsc.* 2004).
  - Amplitude PSF: `h(x,y,z) = F⁻¹_2D{ P(kx,ky) · exp(i2πz·√((n/λ)² − k²)) }`, with `P = A·e^{iφ}` supported on `|k| ≤ NA/λ`.
  - Intensity PSF: `|h|²`. The OTF is the 3D Fourier transform of `|h|²`.
  - Phase retrieval from bead z-stacks recovers `P`, including measured aberrations.
  - Cost: one FFT per z-plane, O(N_z·N² log N). It handles arbitrary non-axisymmetric aberrations natively.
- **Gibson–Lanni** (scalar, stratified sample/coverslip/immersion with design versus actual thicknesses and indices).
  - `h(r,z) = |∫₀¹ exp(ik·OPD(ρ)) J₀(k·NA·rρ) ρ dρ|²`.
  - `OPD(ρ) = z_p·n_s·√(1−(NAρ/n_s)²) + t_g n_g√(…) − t_g* n_g*√(…) + t_i n_i√(…) − t_i* n_i*√(…)`.
  - This captures depth-dependent spherical aberration from RI mismatch, the dominant real-world PSF distortion in live-cell imaging.
- **Richards–Wolf vectorial.** For an x-polarized pupil, the focal field is `E ∝ (I₀ + I₂cos2φ, I₂ sin2φ, −2i I₁ cosφ)`, with
  - `I₀ = ∫₀^α √cosθ sinθ (1+cosθ) J₀(kρ sinθ) e^{ikz cosθ} dθ`,
  - `I₁ = ∫ √cosθ sin²θ J₁ e^{…} dθ`,
  - `I₂ = ∫ √cosθ sinθ (1−cosθ) J₂ e^{…} dθ`.

  Pitfall: the apodization is √cosθ for focusing (illumination) but 1/√cosθ for collection (emission, mapped to the camera).
- **Aguet vectorial model** (Aguet thesis, EPFL 2009; used in psfmodels). Richards–Wolf combined with Gibson–Lanni OPD and Fresnel transmission coefficients (`t_s`, `t_p`) through the stratified layers.
- **Stallinga & Rieger** (Opt. Express 2010, [V] abstract). Full vectorial model including dipole orientation, high-NA polarization, interfaces and aberrations.
  - For *freely rotating* dipoles a Gaussian PSF is adequate for 2D localization.
  - For *fixed* dipoles, Gaussian fits give **systematic errors up to about 40 nm**, independent of photon count.

  This defines when a vectorial or dipole model is *required* rather than optional.
- **Dipole-orientation general form** [R] (Backer & Moerner 2014): `I = Σ_{ij} M_ij B_ij(x,y)`, with six basis images `B` and a second-moment tensor `M`. waveorder uses the same second-moment representation.

**Libraries.**

- **psfmodels** ([V] [repo](https://github.com/tlambert03/PSFmodels)): `vectorial_psf`, `scalar_psf`, the `*_centered` variants, `make_psf` and `tot_psf` (light-sheet total PSF). Parameters are `ni, ni0, ns, tg, tg0, NA, wvl, pz`.
  - Built on Aguet's C++ code, **GPL-3**, CPU only.
  - The README itself notes that it is slower than Li et al. 2017.
- **pyotf** ([V] [otf.py](https://raw.githubusercontent.com/david-hoffman/pyotf/master/pyotf/otf.py)).
  - `BasePSF(wl, na, ni, res, size, zres, zsize, vec_corr="none", condition="sine")`, with `vec_corr ∈ {none, x, y, z, total}` and `condition ∈ {none, sine, herschel}`.
  - `HanserPSF(zrange=...)` computes arbitrary z planes. `SheppardPSF(dual=...)` supports 4Pi-like opposing objectives.
  - The vectorial correction multiplies the pupil by polarization factors `Pxx, Pxy, …` and sums the intensities.
  - Also `phase_retrieval.retrieve_phase` and orthonormal Zernikes. NumPy only.
- **MicroscPSF-Py** ([V]; Li, Xue & Blu, JOSA A 2017). Approximates the Gibson–Lanni Kirchhoff integral as a linear combination of rescaled Bessel functions, avoiding numerical integration. It is **reported 498× faster than the Java PSF Generator** at 511×511×255.

  This is a good example of an *approximation that is smooth in its parameters*. A torch port would be cheap and differentiable, provided Bessel gradients are supplied.
- **EPFL PSF Generator (Java/ImageJ; Kirshner et al. 2013)** [R]. Models include Born & Wolf, Gibson & Lanni, Richards & Wolf, a variable-RI Gibson–Lanni, and 2D defocus/astigmatism/Koehler variants. The exact list was not re-checked because of a certificate error on the site.
- **psf-generator (EPFL, PyTorch; Liu et al., *J. Microsc.* 2026, arXiv 2502.03170)** ([V] [paper](https://arxiv.org/html/2502.03170v1), [docs](https://psf-generator.readthedocs.io/autoapi/psf_generator/index.html), MIT license).
  - **Key conceptual result:** the Fourier (Cartesian `s_x, s_y`) and Bessel (spherical `θ, φ`) approaches are two parameterizations of the *same* Richards–Wolf integral `E(ρ) = −(ifk/2π)∬ a(s)e^{iW(s)} e_∞(s) e^{iks·ρ} dΩ`. So all correction factors can be applied to both: Gibson–Lanni `W`, √cosθ apodization, Gaussian envelope, Zernike `W` (Cartesian only), and Fresnel `q_s, q_p` for the vectorial case.
  - **Classes:** `ScalarCartesianPropagator`, `ScalarSphericalPropagator`, `VectorialCartesianPropagator` and `VectorialSphericalPropagator`.
    - Common arguments: `n_pix_pupil=128`, `n_pix_psf=128`, `device`, `zernike_coefficients`, `wavelength`, `na`, `pix_size`, `defocus_step`, `n_defocus`, `apod_factor`, `envelope`, `gibson_lanni`, `n_i`, `n_s`, `t_i`.
    - Vectorial only: `e0x`, `e0y`.
    - Cartesian only: `special_phase_mask`.
    - Spherical only: `cos_factor`, `integrator`.
    - Methods: `compute_focus_field()`.
    - Output layout is `(z, channel, x, y)`, with channel = 3 for vectorial.
  - **Numerics:**
    - Cartesian uses a chirp-Z transform for arbitrary output pixel size, O(n log n) per plane, with 1st–2nd order convergence.
    - Spherical uses Simpson integration (4th order), O(n), vectorized with `torch.vmap`.
    - On CPU, Cartesian is faster below about 512 px and spherical faster above. On GPU (RTX 3090), spherical scaling is almost flat.
    - Scalar runs about 1.5× (Cartesian) to 3× (spherical) faster than vectorial.
  - **Differentiability:** "Automatic differentiation of the Bessel functions is not natively supported by PyTorch as of version 2.3", so they ship differentiable Bessel functions. **This is the most directly reusable PSF component for gradoscopy.**
- **qnano/VectorialPSF** ([V] search result): an interactive vectorial dipole PSF tool with Zernike aberrations and fixed-dipole orientation (TU Delft, Stallinga/Rieger lineage). **Gohlke's `psf`** [R]: scalar confocal and two-photon PSFs (C extension).

### 2.3 Inverse PSF modeling: uiPSF

Sources: [repo](https://github.com/ries-lab/uiPSF), [paper, Nat. Methods 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12330227/). All [V].

**Representations.**
- **Voxel PSF.** Subpixel shifts are done with Fourier phase ramps.
- **Pixelated complex pupil** `h(kx,ky)`.
- **Zernike pupil.**
- **Vectorial** versions of the pupil models.

**Forward model.** `U = F{h·e^{i2πk_z z_i}·e^{i2π(k_x x_i + k_y y_i)}}`, convolved with the bead-shape kernel `g` (to account for bead size), then × photons + background.

**Extensions.**
- **Field-dependent aberrations:** one *Zernike coefficient map* per mode, interpolated at emitter positions, with smoothness regularization.
- **Multichannel:** affine transforms between channels.
- **4Pi:** per-arm Zernikes plus a relative piston.
- **In situ** estimation from blinking SMLM data: iterate localize → re-estimate, usually two iterations.

**Fitting.** MSE plus regularizers (spatial domain) or a Poisson maximum-likelihood loss (Fourier domain), optimized with L-BFGS through TensorFlow autodiff. Runtimes range from 40 s (voxel model, 20 beads) to 35 min (in situ 4Pi, 10⁴ emitters) on an RTX 3080. Outputs are `.h5` files compatible with SMAP and FD-DeepLoc.

**Lesson.** This is the proof that an *autodiff-able optics + emitter + background + detector model* can be fit to real data to calibrate a simulator. gradoscopy's PSF layer should be able to represent every uiPSF parameterization: voxel, pixel pupil, Zernike pupil, field-dependent Zernike maps, and multichannel transforms. Then "calibrate from beads, then generate training data" becomes a single-library workflow.

Related [R]: FD-DeepLoc (Fu et al., Nat. Methods 2023) trains on simulations with field-dependent vectorial PSFs. ZOLA-3D (Aristov 2018) does GPU Zernike-based PSF fitting.

### 2.4 SMLM simulators

- **DECODE** ([V] [API](https://decode.readthedocs.io/en/release-0.10/decode.simulation.html)).
  - **Modules:**
    - `structure_prior.RandomStructure`;
    - `emitter_generator.EmitterSamplerFrameIndependent` / `EmitterSamplerBlinking` (`intensity_mu_sig`, `lifetime`, frame range);
    - `psf_kernel.DeltaPSF` / `GaussianPSF` (optionally astigmatic) / `CubicSplinePSF(xextent, yextent, img_shape, ref0, coeff, vx_size, roi_size, device, max_roi_chunk)`;
    - `background.UniformBackground`;
    - `camera.Photon2Camera(qe, spur_noise, em_gain, e_per_adu, baseline, read_sigma)`, `PerfectCamera`, `SCMOS` (per-pixel read noise);
    - `simulator.Simulation`.
  - **PSF:** the cubic spline PSF comes from SMAP bead calibration (Li et al., Nat. Methods 2018 [R]). It is implemented in a separate C++/CUDA library (TuragaLab/SplinePSF [V]; CUDA limits ROIs to 32×32 = 1024 threads per block).
  - **Gradients:** analytic derivatives exist (used for CRLB) [R], but the PSF is **not a torch autograd op**, so spline coefficients and emitter parameters are not learnable through the simulator.
  - **Lesson:** a sparse, emitter-centric renderer (local ROIs pasted into frames) is **orders of magnitude cheaper** than dense-volume convolution for point-like samples. gradoscopy needs both paths.
- **DeepSTORM3D** ([V] [physics_utils.py](https://raw.githubusercontent.com/EliasNehme/DeepSTORM3D/master/DeepSTORM3D/physics_utils.py); Nehme et al., Nat. Methods 2020).
  - **Pipeline:** `EmittersToPhases` builds per-emitter pupil phases `exp(i(x·Kx + y·Ky + z·Kz))` × defocus aberration × circ. This includes an oil/water (Gibson–Lanni-style) `k_z` treatment in `calc_bfp_grids`. Then `MaskPhasesToPSFs` computes the FFT of mask × phases, normalized to `nphotons`. This is followed by `BlurLayer` (random Gaussian σ ∈ [0.5, 1.5] to absorb model mismatch), crop, and `NoiseLayer` (super-Gaussian non-uniform background, Poisson, read noise).
  - **Scalar only.**
  - **Learning:** the phase mask is a learnable parameter, optimized *jointly* with the localization CNN (end-to-end PSF engineering).
  - **Poisson noise:** differentiable path `input + √input·randn` (Gaussian reparameterization); non-differentiable path via NumPy.
  - **Lesson:** "optics as a trainable layer" works in practice when the layer is a pupil and the noise is reparameterized. It also shows the common hack of *random blur to cover model mismatch*, which gradoscopy should make an explicit "model-mismatch augmentation" component rather than hidden physics.
- **SMLM Challenge 2016** ([V] [Sage et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6684258/)).
  - Structures: microtubules (25 nm, hollow 15 nm core) and pseudo-ER (150 nm tubes) defined as **3D polynomial splines** with fluorophores uniformly sampled on them.
  - Photophysics: 4-state model (ON/OFF/BLEACH/DARK; T_on = 3, T_dark = 2.5, T_bleach = 1.5 frames).
  - PSFs: **experimental** bead PSFs at 10 nm steps (astigmatism, double helix; biplane made by ±250 nm offsets).
  - Camera: EMCCD model (read noise 74.4 e⁻, EM gain 300, spurious charge 0.002, 45 e⁻/ADU), 100 nm pixels.
  - **Lesson:** a community benchmark chose curves as geometry and *measured* PSFs, so gradoscopy must accept tabulated or measured PSFs as a first-class optics backend.
- **TestSTORM** [R] (Sinkó et al. 2014; Novák et al., Sci. Rep. 2017, [V] existence). MATLAB. Models label geometry (antibody linkers), laser-intensity-dependent stochastic photophysics, epi/TIRF illumination profiles, several 3D PSF types, drift and camera noise.
- **SuReSim** ([V] summary via BIII / Nat. Methods 2016). Ground-truth models (epitopes, lines, surfaces, including imported surface meshes [R]) plus labeling (efficiency, antibody length and binding angle, fluorophores per label, unspecific labeling), bleaching, and 2D/3D PSF models. It works largely at the *localization level* [R].
- **SMIS** ([V] [Comm. Biol. 2023](https://github.com/DominiqueBourgeois/SMIS)). MATLAB.
  - Full multi-state photophysics with spectral data per state, photo- and thermally-induced transitions, sub-frame time resolution, unlimited lasers and colors.
  - Diffusion with multiple states (sptPALM), 2D/3D widefield.

  This is the photophysics fidelity ceiling.

**Takeaway.** SMLM tools show that **labeling and photophysics are separate stages** from geometry. The chain is structure → epitopes → labels (linker displacement, efficiency) → fluorophores (state machines) → emitted photons per frame. They also show that **time** (frames and sub-frame state changes) is intrinsic.

### 2.5 Wave-optical propagation and scattering solvers

- **biobeam** ([V] [PLOS CB 2018](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006079), [repo](https://github.com/maweigert/biobeam)).
  - **Core:** scalar BPM (`Bpm3d(dn, size, units, lam, n0, simul_xy)`, `.propagate`, `.psf`). Each step is `u(z+Δz) = F⁻¹{F{u·e^{ik₀Δn Δz}}·e^{ik_zΔz}}` with the exact angular-spectrum kernel. The paper reports high accuracy versus Mie up to about 0.5 rad; this most likely refers to the scattering-angle limit, so treat it as indicative.
  - **Illumination:** cylindrical light sheets, Bessel lattices, custom fields.
  - **Key trick:** **multiplexing**. Hundreds to thousands of *non-overlapping* point sources are propagated in one simulation to get spatially varying detection PSFs, a claimed 20,000× speedup. About 1000 PSFs through 1024³ voxels take under 500 ms on one GPU.
  - Image formation = excitation intensity from the propagated light sheet × fluorophore density → convolution with spatially varying detection PSFs.
  - OpenCL (gputools); not differentiable.
  - **Autograd memory warning** [own estimate]: a 1024³ complex64 BPM stores 1024 slices × 8 MB = **8.6 GB** of activations for reverse mode. gradoscopy needs gradient checkpointing or an adjoint BPM for this tier.
- **Multi-layer Born (MLB)** (Chen, Ren, Liu, Chowdhury, Waller, Optica 2020 [V]) and **SSNP** (Zhu, Wang, Tian, Opt. Express 2022 [V]; repo `bu-cisl/SSNP-IDT`).
  - MLB applies a first-Born scattering step per slice and captures oblique and back-scattering better than BPM.
  - SSNP propagates the field *and its z-derivative* (non-paraxial) and is accurate for high-angle illumination.

  Both are the standard differentiable forward models for 3D RI tomography and are *exactly* the "thick label-free" fidelity tier.
- **holopy** ([V] [theories](https://holopy.readthedocs.io/en/latest/users/theories.html), GPL-3). **Cleanest description/solver split in the survey.**
  - `Scatterer` types: `Sphere`, `Spheres`, `LayeredSphere`, `Spheroid`, `Cylinder`, `Ellipsoid`, `Csg`, …
  - Theories: `Mie` (built-in), `Multisphere` (Mackowski's Fortran SCSMFO, exact multiple scattering), `Tmatrix` (Mishchenko, axisymmetric), `DDA` (external ADDA, any shape, slowest), `MieLens` (analytic Mie through a perfect lens; Leahy et al. 2020, slightly *faster* than `Mie`), `AberratedMieLens` (plus spherical aberration), and `Lens(theory)` (wraps any theory; "orders of magnitude slower"). `theory='auto'` picks one.
  - API: `calc_holo`, `calc_field`, `calc_intensity`, `calc_scat_matrix`, `calc_cross_sections`, with the detector schema from `detector_grid()` / `detector_points()`. Illumination (wavelength, polarization, medium index) is metadata on the schema.
  - Hologram model [R] (Lee et al. 2007): `I/I₀ = |x̂ + α·E_s/E₀|²` with a scaling factor α.
- **miepython** ([V] v3.x): `S1_S2`, `i_par`, `i_per`, efficiencies; numba JIT via `MIEPYTHON_USE_JIT=1`. **PyMieScatt** [R]: Mie plus inversion, core–shell, aerosol-oriented. Neither is GPU or differentiable.
- **PyMieDiff** ([V] [arXiv 2512.08614](https://arxiv.org/abs/2512.08614), APL Photonics 2026). Fully PyTorch, autograd-compatible spherical Bessel/Hankel functions, vectorized Mie coefficients for multi-shell spheres, and far-field, angular and near-field outputs. GPU. **GPL-3.0** ([V] repo), so it cannot be vendored into an MIT/BSD gradoscopy; re-implement or depend on it optionally.
- **treams** ([V]; Beutel et al., CPC 2024). T-matrices in spherical, cylindrical and plane-wave bases, helicity/parity bases, 1D/2D/3D lattices, stratified media. MIT; NumPy/Cython [R]; not differentiable or GPU.
- **smuthi** ([V]). Multiple scattering by particles *in planar layered media*: T-matrix plus S-matrix, NFM-DS (Fortran) for non-spherical particles, CUDA for particle coupling, thousands of particles. The natural reference for "particle on a coverslip" (iSCAT, TIRF scattering).
- **TorchGDM** ([V] arXiv 2505.09545). Green dyadic method (coupled dipoles) in PyTorch with autograd, GPU and multi-scale effective polarizabilities. The only *differentiable arbitrary-shape* full-wave scatterer solver found. Cost scales badly with dipole count, so it suits nanoparticles, not cells.
- **chromatix** ([V] [repo](https://github.com/chromatix-team/chromatix), JAX).
  - `Field` objects carry tensors plus sampling metadata.
  - Elements are composed like NN layers: `ObjectivePointSource`, `PlaneWave`, `PhaseMask`, `AmplitudeMask`, `FFLens`, propagation, multislice thick samples.
  - Sensors: `BasicSensor`, `ShotNoiseIntensitySensor`.
  - Polychromatic and vectorial support.

  This is the best existing model of a *compositional differentiable optics API*, but in JAX. **torchoptics** [V existence] is a smaller PyTorch analogue.

### 2.6 Label-free, phase and polarization: waveorder

Sources: [repo](https://github.com/mehta-lab/waveorder), [arXiv 2412.09775](https://arxiv.org/html/2412.09775v3), BSD-3. [V]

**Models.** `waveorder/models/` contains `isotropic_thin_3d`, `phase_thick_3d`, `isotropic_fluorescent_thin_3d`, `isotropic_fluorescent_thick_3d`, `inplane_oriented_thick_pol3d`, `inplane_oriented_thick_pol3d_vector` and `shift_variant_fluorescent_3d`.

Each model exposes `calculate_transfer_function(...)`, `apply_transfer_function(...)` (the forward simulation) and `apply_inverse_transfer_function(...)` (Tikhonov inversion). An example is `phase_thick_3d.calculate_transfer_function(zyx_shape, yx_pixel_size, z_pixel_size, wavelength_illumination, z_padding, index_of_refraction_media, numerical_aperture_illumination, numerical_aperture_detection, …)`, which returns real and imaginary potential transfer functions. Helpers include `optics.generate_pupil`, `generate_propagation_kernel`, `compute_weak_object_transfer_function_3D` and `generate_tilted_pupil`.

**Physics.**
- The specimen is a vector of physical properties: scattering-potential tensor components (phase, absorption, birefringence, diattenuation) expanded on spherical harmonics, or fluorophore density and dipole second moments.
- Image formation is linear: `d = H_θ(f)`. `H` is assembled from three sub-models: a **scattering model** (Green's tensor spectrum), an **illumination pupil** (Köhler, partially coherent, oblique) and a **detection pupil** (widefield, confocal, light-sheet).
- `θ` (NA, defocus, tilt, and about 8 misalignment modes) is **auto-tuned by backprop**, per tile for shift-variant problems.

**Stated limits.** Single scattering only; no nonlinear contrast; fluorescence is decoupled from the RI; Gaussian (Tikhonov) noise.

**Lesson.** This is the best evidence that **one pupil-based transfer-function core can serve fluorescence and label-free modalities**. It is also the right *cheap tier* for label-free imaging. gradoscopy should include the weak-object/WOTF path as its "fast" coherent and partially coherent tier.

### 2.7 Modality-specific forward models

- **SIM.**
  - Forward model [R, standard]: `D_{θ,φ} = (S · I_{θ,φ}) ⊗ h`, with `I = I₀[1 + m·cos(2πp_θ·r + φ)]` in 2D. 3D SIM uses 5 bands: `D̃(k) = Σ_m OTF_m(k) S̃(k − m·p)`, where the axial modulation enters the band-specific OTFs.
  - fairSIM ([V] packages `sim_algorithm`, `sim_gui`, `linalg`, …) and SIMToolbox (MATLAB, MAP-SIM) are *reconstruction* tools.
  - Forward simulation lives in microsim `SIMIllum2D/3D` (beamlet fans for coherence) [V], mcSIM `simulate_dmd.py` (full DMD blaze/diffraction model; optional CuPy) [V], and OpenSIM (used to generate ML-SIM training data) [V search].
  - **Lesson:** SIM is "illumination pattern × sample, then an incoherent PSF". It needs only a pluggable *illumination intensity field*, plus optionally the coherent pupil-level beam model for fidelity.
- **Fourier ptychography.**
  - Thin-sample forward model: `I_n = |F⁻¹{P(k)·Õ(k − k_n)}|²` per LED. Each LED is a tilted plane wave, which shifts the object spectrum across the pupil.
  - Extensions: multislice for thick samples (the MLB/SSNP lineage), LED position and intensity miscalibration, mixed states for partial coherence (PtyLab [V]: ePIE/mPIE/qNewton engines, Fraunhofer/Fresnel/ASP propagators, CPU/GPU).
  - **LearnedDesignFPM** ([V], Kellman et al., IEEE TCI 2019) learns LED patterns by backprop through the physics in PyTorch. Here the *illumination* is the learnable part, the counterpart of DeepSTORM3D learning the detection pupil.
- **iSCAT / mass photometry.**
  - Signal: `I = |E_r|² + |E_s|² + 2|E_r||E_s|cos φ`. The reference is the coverslip reflection (glass/water amplitude reflectivity ≈ (1.52−1.33)/2.85 ≈ 0.067, about 0.44 % in intensity). Contrast ≈ 2|E_s|/|E_r|·cos φ, and `E_s ∝` polarizability ∝ volume ∝ mass, which gives the linear contrast–mass law (Young et al., Science 2018 [V via Refeyn]).
  - **Mahmoodabadi et al.** (Opt. Express 2020 [V]): vectorial iPSF model including RI mismatch and aberrations, validated against FDTD, with about 10 µm axial range from the iPSF's lateral fringes.
  - **Becker et al.** (ACS Photonics 2023 [V]): Fourier optics plus an **atomistic polarizability model**, simulating mass photometry from first principles, including shape and orientation effects.
  - **PiSCAT** ([V], JOSS 2022): analysis tools plus iPSF stack generation.
  - DeepTrack2 `ISCAT` is a brightfield variant with an illumination angle and a reference amplitude factor.
  - **Lesson:** iSCAT is simply *coherent imaging with a reference field that shares the pupil path*. Where the reference is produced (reflection versus transmission) and its phase (the Gouy phase and the defocus-dependent relative phase) are what matter. The same engine can do brightfield, holography, darkfield (reference blocked) and iSCAT (reflected reference) if the "unscattered field" is an explicit, filterable component.

---

## 3. Cross-cutting analysis

### 3.1 The common decomposition

Mapping each tool onto one set of stages. "—" means the stage is absent or trivial.

| Stage | microsim | DECODE | DeepSTORM3D | SMLM tools | holopy | waveorder | biobeam | CytoPacq | DeepTrack2 |
|---|---|---|---|---|---|---|---|---|---|
| **Geometry / structure prior** | `Sample.labels[*].distribution` (MatsLines, COSEM) | `RandomStructure` | random 3D points | splines, meshes, epitopes | `Scatterer` | specimen property volumes | `dn` volume | CytoGen/MitoGen/FiloGen | scatterer features |
| **Labeling / material** | `Fluorophore` (FPbase), concentration | — | — | linker, efficiency | refractive index `n` | property vector | RI | — | intensity / RI |
| **Photophysics / dynamics** | — (static) | blinking, lifetime | — | 4-state (SMIS: N-state) | — | — | — | mitosis, motion | sequences |
| **Spectral** | `(C,F,W)` rates, binned | — | — | multicolor (SMIS) | single λ | single λ | single λ | filters | single λ |
| **Illumination** | irradiance; SIM / spinning disc | — | — | laser profile | plane wave + polarization | illumination pupil | light-sheet BPM | uneven illumination | illumination gradient |
| **Sample–light interaction** | incoherent emission | emission | emission | emission | Mie / T-matrix / DDA | weak-object Born | BPM | emission | multislice phase + Mie |
| **Optics (pupil/PSF)** | vectorial RW PSF, confocal | spline / Gaussian PSF | learned pupil (scalar) | measured PSFs | lens models | detection pupil | BPM + lens | measured PSF | scalar pupil + Zernike |
| **Sampling** | truth → output (coarsen-sum) | per-pixel spline | FFT grid + crop | pixelization | detector grid | voxel grid | grid | resampling | upscale → pool |
| **Detector** | CCD/EMCCD/CMOS | EMCCD/sCMOS | Poisson + read | EMCCD | — (noise_sd) | Gaussian | — | full acquisition chain | noise features |
| **Ground-truth outputs** | `ground_truth()` volume | `EmitterSet` | positions | localizations | params | properties | — | masks, lineage | arbitrary features |

**Observations.**

- **Split geometry from material.** Tools that separate geometry from material/labeling scale to more applications. Examples are microsim's distribution + fluorophore, SuReSim's structure + labeling, and holopy's shape + index.
- **Keep a species axis until the end.** Tools that keep an explicit per-species axis (microsim's F) until late get crosstalk, bleed-through and multi-channel imaging almost for free.
- **Emitter-centric versus volume-centric samples.** These are two different computational regimes. Points (SMLM, particle tracking) are best rendered sparsely with local PSF patches (DECODE). Continuous structures (cells, tissue) are best rendered as a volume convolution (microsim). DeepTrack2 mostly uses voxelization for both, which is wasteful for points.
- **The detector model has converged.** It is QE → Poisson → dark/CIC → full-well clip → (EM gain: Gamma) → read noise (Gaussian; sCMOS: per-pixel maps) → gain/offset → quantization/saturation. microsim and DECODE are essentially equal here.
- **Ground truth is a product of the pipeline, not an afterthought.** microsim caches `ground_truth()`; DECODE returns `EmitterSet`; CytoPacq returns masks and lineage.

### 3.2 The pupil as the lingua franca

Every coherent-imaging and PSF tool reduces to:

> **source spectrum on the pupil** (emitter dipole radiation, scattered far field S1/S2, object spectrum shifted by the illumination k-vector) × **pupil filter** (NA disk, apodization, Fresnel factors, Gibson–Lanni OPD, Zernike or pixel phase, engineered masks, polarization optics) → **propagation to the image plane** (FFT/CZT, or Bessel integrals for axisymmetric cases) → **|·|² at the detector**, summed over incoherent modes.

Incoherent modes include:
- emission wavelengths (microsim spectral bins);
- dipole orientations (three for isotropic emitters);
- source points of a partially coherent Köhler illumination (waveorder's illumination pupil);
- FP LEDs;
- SIM beamlets (microsim `nbeamlets`).

**Consequence:** a single "sum over incoherent modes of coherent pupil propagations" kernel, plus the incoherent shortcut (precompute `|h|²` and convolve), covers widefield, confocal, SIM, light-sheet detection, brightfield, darkfield, holography, iSCAT, FP and DPC. The alternatives are the scattering solvers (Mie, T-matrix, BPM, MLB), which are *producers* of pupil-plane or image-plane fields, not replacements for the imaging system.

### 3.3 Sample representations in the wild

- **Point sets with attributes** (DECODE, DeepSTORM3D, SMLM tools, holopy `Spheres`): exact positions, differentiable with respect to position, cheap sparse rendering.
- **Analytic primitives** (holopy spheres, spheroids and cylinders; DeepTrack2 `Sphere`/`Ellipsoid`; SMLM-challenge splines): these enable exact solvers (Mie, T-matrix) and analytic Fourier transforms (a sphere's FT is `3(sin kr − kr cos kr)/(kr)³`), so voxelization can be *band-limited and differentiable in radius and position*.
- **Voxel volumes** (microsim truth space, biobeam/waveorder/MLB RI volumes, CytoPacq phantoms, COSEM segmentations): the most general and the only option for tissue and cells, but aliasing makes them non-smooth in geometric parameters unless they are rendered from an implicit representation.
- **Meshes / surfaces:** SuReSim imports surfaces for label placement [R]. **No surveyed tool uses meshes for optics.** Meshes are useful as *geometry authoring* (cell shapes, organelles) and are converted to SDF → soft occupancy → voxels, or to label point clouds by surface sampling.
- **Learned/statistical generators** (CellOrganizer, SimuCell, COSEM-derived priors): these belong in the sample stage as stochastic generators with parameters.

### 3.4 Fidelity knobs that already exist

| Axis | Cheap | Medium | Expensive | Seen in |
|---|---|---|---|---|
| PSF model | Gaussian (DECODE `GaussianPSF`; Zhang 2007 approximation in psfmodels) | scalar Fourier/Bessel, Gibson–Lanni | vectorial RW + Fresnel + dipole orientation | psfmodels, psf-generator, pyotf |
| PSF variance | shift-invariant | depth-variant (G–L per z) | field-dependent Zernike maps; BPM-multiplexed | uiPSF, biobeam, waveorder tiles |
| Spectral | single effective λ | N bins per channel | full spectrum | microsim `spectral_bins_per_emission_channel` |
| PSF support | truncated (`max_psf_radius_aus = 6`) | — | full field | microsim |
| Coherent scattering | projection / thin phase | first Born (WOTF) | multislice BPM / MLB / SSNP → Mie / T-matrix / DDA / GDM | DeepTrack2, waveorder, biobeam, holopy, TorchGDM |
| Lens model | lens-free / ideal | `MieLens` (analytic) | `Lens(theory)` numeric (orders slower) | holopy |
| Sampling | at output pixel | ×2–4 oversampling | ×8+ truth space | microsim downscale, DeepTrack2 `upscale` |
| Photophysics | constant brightness | 2–4-state Markov per frame | N-state, sub-frame, laser-dependent | DECODE, SMLM challenge, SMIS |
| Detector | Gaussian noise | Poisson + read noise | full EMCCD/sCMOS chain with per-pixel maps and saturation | DeepSTORM3D, DECODE, microsim |

### 3.5 Differentiability: patterns and pitfalls observed

1. **Bessel functions.** `torch.special.bessel_j0/j1` exist without autograd (psf-generator paper, torch ≤ 2.3). Higher orders (J₂ for vectorial RW) and spherical Bessel/Hankel functions (for Mie) are missing. Both psf-generator and PyMieDiff wrote their own. Use `J₀' = −J₁`, `J₁' = J₀ − J₁/x`, `J_n' = (J_{n−1} − J_{n+1})/2`, and series expansions near x → 0 to avoid the unstable `2J₁/x − J₀` recurrence.
2. **Poisson noise.** There is no `rsample`, so tools use a Gaussian approximation (`λ + √λ·ε`, as in DeepSTORM3D). EMCCD Gamma has an implicit-reparameterization `rsample` in torch [R]. Quantization and clipping need straight-through or soft surrogates.
3. **Discrete structure.** Particle counts, blinking states and labeling-efficiency Bernoulli draws are non-differentiable. None of the surveyed tools addresses learning their parameters. Candidates are score-function (REINFORCE) estimators, Gumbel-softmax relaxations, or expected-value (mean-field) surrogates, e.g. using `p_label` as a continuous density weight.
4. **Rasterization aliasing.** Voxelizing hard primitives makes gradients with respect to position and size zero almost everywhere. Use analytic Fourier-domain primitives, soft SDF occupancy, or point splatting with a smooth kernel.
5. **Framework hostility.** pydantic + xarray (microsim), NumPy fallbacks (DeepSTORM3D noise), C++/CUDA kernels without backward (DECODE spline) and OpenCL (biobeam) all break the graph. The differentiable path must be pure torch tensors, with metadata carried alongside rather than wrapped around them.
6. **Caching versus gradients.** microsim's `lru_cache` + disk PSF cache is excellent for fixed optics but wrong when optics parameters require grad. The cache must key on parameter *values and versions* and be bypassed or detached when parameters are trainable.
7. **Memory.** Multislice and BPM reverse mode stores every slice. Checkpointing (`torch.utils.checkpoint`) or adjoint propagation is mandatory above about 256³ with batch > 1.
8. **Model mismatch.** DeepSTORM3D's random blur and uiPSF's in situ refinement are two answers to sim-to-real mismatch: augment, or calibrate. A differentiable engine can do both, and the calibration loop is the unique selling point.

### 3.6 Gaps that no existing tool fills

1. **One scene description, many solvers, fluorescence *and* coherent label-free.** waveorder has the physics breadth but only linear single scattering and no stochastic generation. holopy has the solver ladder for scattering but no fluorescence, no GPU and no gradients. microsim has fluorescence depth but no coherent imaging and no gradients.
2. **Learnable distribution parameters.** No tool differentiates image statistics with respect to the parameters of the *random sample generator*: mean and variance of particle radius, photon-count distribution, density, blinking rates, aberration distributions. DeepTrack2's random pipelines resolve values eagerly, and the others use NumPy RNGs.
3. **Calibrated fidelity switching.** No tool reports or estimates the error between tiers (e.g. Gaussian versus vectorial for a given NA, or BPM versus Mie for a given size and contrast), or chooses the tier automatically from validity criteria. holopy's `theory='auto'` is the only precedent, and it is based on scatterer type, not error.
4. **A unified sparse-plus-dense renderer.** Point emitters (sparse ROIs) and volumes (FFT convolution) are handled by different tools. Real samples mix them, e.g. single molecules on a labeled cell with autofluorescence.
5. **Spatially varying optics as a general component.** Field-dependent aberrations (uiPSF), depth-dependent PSFs (Gibson–Lanni) and scattering-induced variation (biobeam) are each solved separately. There is no common "PSF field" abstraction with interpolation and differentiable parameters.
6. **Time.** Photophysics (SMIS, DECODE), motion and diffusion (SMIS, MitoGen, DeepTrack2 sequences) and in-exposure motion blur are fragmented. None is differentiable.
7. **Coupling between channels.** Fluorescence emitted inside a refractive-index volume is imaged through the aberrations of that volume. Only biobeam does this, and it is not differentiable. waveorder explicitly assumes the two are decoupled.
8. **Batch-first GPU data generation with paired labels at the throughput needed for training** (≥ 10³ images/s at 256²). DECODE and DeepSTORM3D achieve this for SMLM only.

---

## 4. Implications for gradoscopy's architecture

These recommendations are opinionated and deliberately concrete.

### 4.1 Adopt a staged pipeline with typed boundaries, modeled on microsim but torch-native

Define the stages as `nn.Module`-like, batch-first components with **explicit typed intermediates** (dataclasses of tensors plus lightweight metadata, never xarray inside the graph):

1. **`SampleGenerator` → `Scene`.** A stochastic program producing geometry and materials, with learnable distribution parameters.
2. **`Labeling / Photophysics` → `EmitterState`.** Per-species, per-time-step emission rates (a `(B, F, T)` or per-emitter tensor).
3. **`Illumination` → `IlluminationModes`.** A list of coherent modes with weights: plane waves, beams, SIM beamlets, LEDs, light-sheet fields; or an intensity field for incoherent excitation.
4. **`Interaction` → `SourceField(s)`.** Emitters become dipole sources; scatterers become scattered fields via the solver tier.
5. **`ImagingSystem` (pupil chain) → `DetectorIrradiance`.** Photons/s/pixel, shape `(B, C, [T], [Z], Y, X)`.
6. **`Detector` → `Counts`.** Plus the **`GroundTruth` bundle** (positions, masks, property volumes, per-species images).

Keep microsim's dimension discipline, `(B, C, F, W, Z, Y, X)` with named axes documented. Keep the species axis F separate until the channel sum (for crosstalk), and keep the wavelength-bin axis W (for spectral fidelity).

### 4.2 Make the `Pupil` / angular spectrum the central abstraction

- **`Pupil` object:**
  - NA; immersion, coverslip and sample stack (n, t, design versus actual → Gibson–Lanni OPD and Fresnel `t_s, t_p`);
  - apodization (sine condition, with **√cosθ versus 1/√cosθ chosen by direction**);
  - aberrations (Zernike coefficients; pixel phase and amplitude; field-dependent Zernike *maps*);
  - engineered masks (phase masks, SLM, DMD);
  - polarization (Jones matrices; scalar mode = 1 component; vectorial = 2 or 3).
- **Two propagation back-ends behind one interface**, following psf-generator's unification: Cartesian FFT/CZT (arbitrary aberrations, arbitrary output pixel) and spherical Bessel-quadrature (axisymmetric, fast on GPU).
  - Port or adapt psf-generator: it is **MIT-licensed**, so it can be vendored.
  - Ship our own autograd Bessel J₀/J₁/J₂ and spherical j_n/y_n with custom backward.
- **Everything coherent goes through this chain:** dipole emission, Mie far fields (S1/S2 mapped onto pupil coordinates, holopy `MieLens`-style), shifted object spectra (FP), and reference fields (iSCAT/holography/darkfield as an explicit filterable "unscattered" component).
- **Incoherent imaging is an *optimization* of the same chain:** precompute `|h|²` per (wavelength bin, dipole mode) and convolve. It should not be a separate model.

### 4.3 Represent the sample as multiple representations with explicit converters

Primary representations, all batchable:

- **`PointSet`** (positions plus attributes: species, brightness, dipole orientation, radius, n).
- **`Primitives`** (spheres, ellipsoids, cylinders, layered spheres, capsules, spline tubes), with analytic FTs and eligibility for exact solvers.
- **`ImplicitField`** (SDF or neural field; meshes are imported to SDF).
- **`VoxelVolume`** (per-species fluorophore density; complex Δn / scattering potential; optional anisotropy tensor for polarization).

**Converters** (all differentiable where possible):

- primitive → band-limited voxels (Fourier-domain rasterization);
- SDF → soft occupancy;
- surface/volume → point labels (labeling model with linker offset and efficiency);
- points → sparse ROI render.

Each solver declares which representations it consumes. The planner converts as needed, like holopy's theory eligibility but generalized.

### 4.4 Make fidelity a policy object resolved per stage, holopy-style, with validity rules

- A user writes one scene plus one microscope, then picks `fidelity="fast" | "balanced" | "exact"` or per-stage overrides: `psf="vectorial"`, `scattering="bpm"`, `spectral_bins=5`, `oversample=4`, `photophysics="markov4"`, `detector="emccd"`.
- **The tiers per stage come straight from the survey:**
  - PSF: Gaussian → scalar → Gibson–Lanni → vectorial → vectorial + dipole orientation.
  - Coherent: projection → first Born/WOTF → BPM/MLB/SSNP → Mie/T-matrix (primitives) → optional GDM plugin.
  - Spatial variance: invariant → depth-variant → field maps → through-sample BPM.
  - Spectral: 1 → N bins.
  - Detector: Gaussian → Poisson + read → full chain.
- **Encode validity checks as code**, warning when a tier is outside its regime:
  - Gaussian PSF with a fixed-dipole emitter (Stallinga: up to 40 nm bias);
  - scalar PSF at NA > 1.2 [R threshold];
  - projection approximation when the phase accumulated across a particle exceeds about 1 rad;
  - Born when `k·a·|Δn|` is not ≪ 1;
  - BPM beyond about 0.5 rad scattering angles (biobeam);
  - sampling finer than Nyquist (pixel > λ/(4NA) in coherent intensity).
- **Provide a `compare_fidelity()` utility** that runs two tiers on the same scene and reports error metrics. This is gap #3, and a key differentiator.

### 4.5 Two renderers: sparse (emitter) and dense (volume), composited

- **Sparse:** per-emitter PSF patches, evaluated analytically (pupil → local CZT), or from a tabulated/measured PSF with differentiable **cubic-spline** interpolation reimplemented in torch (DECODE/SMAP-style coefficients as `nn.Parameter`s, so they are learnable). Scatter-add into frames.
- **Dense:** FFT convolution with per-wavelength-bin OTFs (microsim), or multislice propagation for coherent imaging.
- **The detector sees the sum.** This lets single-molecule, particle-tracking and cell/tissue applications share one engine.

### 4.6 Randomness as a first-class, differentiable citizen

- Every random quantity is drawn from a `gradoscopy.Distribution` whose parameters can be `nn.Parameter`s.
  - Use `rsample` where it exists (Normal, LogNormal, Gamma, Beta, Uniform with learnable bounds via reparameterization).
  - Use relaxed or surrogate estimators for discrete draws: count → expected density or Gumbel-softmax; Bernoulli labeling → continuous weight; blinking → expected on-fraction or REINFORCE option.
- **Noise:** Poisson via selectable estimator (Gaussian reparameterization, as DeepSTORM3D does, versus straight-through exact sampling). EMCCD via Gamma `rsample`. Quantization via straight-through estimator.
- **Seeded, batch-vectorized RNG (per-sample generators)** for reproducibility, as microsim does with `random_seed` and cache keys.

### 4.7 Calibration loop as a headline feature

Package uiPSF-style fitting: bead stacks or in-situ data → fit pupil, field maps, background and camera gain/offset. Then **the same objects** generate training data. Support voxel, pixel-pupil, Zernike-pupil and field-dependent parameterizations. Loss: Poisson/negative-binomial NLL. Optimizers: L-BFGS/Adam.

### 4.8 Performance and memory rules

- **Batch-first** everywhere (target ≥ 10³ 256² images/s for fast-tier fluorescence on one GPU).
- **Ragged emitters** via padding plus masks, or CSR offsets.
- **PSF/OTF cache** keyed on hashed parameter values, auto-bypassed when `requires_grad`.
- **Gradient checkpointing** built into multislice/BPM solvers, which also need a truncation or adjoint option.
- **Multiplexed PSF computation** (biobeam's trick) for spatially variant PSFs: compute non-overlapping point responses in one propagation.
- `complex64` by default, with `complex128` available for Mie/T-matrix at large size parameter.

### 4.9 Licensing strategy

- **Vendorable (permissive):** psf-generator (MIT), microsim (BSD-3), waveorder (BSD-3), treams (MIT). biobeam (BSD [R]) is fine for ideas.
- **GPL-3, do not vendor; reimplement from papers, or keep as optional extras for cross-validation tests only:** holopy, PyMieDiff, psfmodels (Aguet C++), and DECODE [R].

Build a **validation suite** that compares gradoscopy tiers against these references in CI: Mie versus holopy/miepython; vectorial PSF versus psfmodels/psf-generator; BPM versus Mie (biobeam-style); camera statistics versus microsim.

### 4.10 Scope recommendations (feasibility)

**In scope for v1:**
- fluorescence (widefield, confocal, light-sheet detection, SIM via illumination patterns, SMLM/sparse emitters) with scalar, Gibson–Lanni and vectorial PSFs plus spectral bins;
- coherent label-free (brightfield/DPC via partially coherent modes, holography, darkfield, iSCAT) with projection / WOTF / multislice BPM, plus Mie for spheres;
- a full detector chain;
- learnable distributions;
- the calibration loop.

**v2:**
- polarization/birefringence (waveorder-style anisotropic scattering potential);
- MLB/SSNP;
- T-matrix for spheroids and cylinders;
- through-sample aberrated fluorescence (biobeam-style, coupling fluorescence to RI);
- time-resolved photophysics with sub-frame dynamics;
- FP with LED arrays.

**Out of scope, or plugins only:** DDA/GDM and FDTD-class full-wave solvers, nonlinear contrast (two-photon, SHG), and inelastic scattering (Raman). Provide an interface so external solvers can supply pupil-plane fields.

The **"more than the sum of its parts"** comes from four shared layers: one pupil chain, one sample multi-representation with converters, one fidelity policy with validity checks, and one differentiable randomness layer. With these, any modality becomes a short composition, and any parameter anywhere (sample distribution, labeling, illumination, pupil, detector) becomes learnable by gradient descent.

---

### Key sources

- microsim: https://github.com/tlambert03/microsim · https://talleylambert.com/microsim/stages/
- DeepTrack2 optics: https://github.com/DeepTrackAI/DeepTrack2 (develop, `deeptrack/optical/optics.py`)
- psfmodels: https://github.com/tlambert03/PSFmodels · pyotf: https://github.com/david-hoffman/pyotf · MicroscPSF-Py: https://github.com/MicroscPSF/MicroscPSF-Py
- psf-generator: https://github.com/Biomedical-Imaging-Group/psf_generator · https://arxiv.org/abs/2502.03170
- uiPSF: https://github.com/ries-lab/uiPSF · https://www.nature.com/articles/s41592-024-02282-x
- Stallinga & Rieger 2010: https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-24-24461
- DECODE: https://decode.readthedocs.io/en/release-0.10/decode.simulation.html · SplinePSF: https://github.com/TuragaLab/SplinePSF
- DeepSTORM3D: https://github.com/EliasNehme/DeepSTORM3D
- SMLM challenge: https://pmc.ncbi.nlm.nih.gov/articles/PMC6684258/ · SMIS: https://github.com/DominiqueBourgeois/SMIS · SuReSim: https://www.nature.com/articles/nmeth.3775 · TestSTORM: https://www.nature.com/articles/s41598-017-01122-7
- biobeam: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006079
- MLB: https://github.com/Waller-Lab/multi-layer-born · SSNP: https://github.com/bu-cisl/SSNP-IDT
- holopy: https://holopy.readthedocs.io/en/latest/users/theories.html
- miepython: https://miepython.readthedocs.io/ · PyMieDiff: https://arxiv.org/abs/2512.08614 · treams: https://github.com/tfp-photonics/treams · smuthi: https://smuthi.readthedocs.io/ · TorchGDM: https://arxiv.org/abs/2505.09545
- chromatix: https://github.com/chromatix-team/chromatix
- waveorder: https://github.com/mehta-lab/waveorder · https://arxiv.org/abs/2412.09775
- CytoPacq: https://cbia.fi.muni.cz/research/simulations/cytopacq.html · SimuCell: https://www.nature.com/articles/nmeth.2096
- SIM: https://github.com/fairSIM/fairSIM · https://github.com/simtoolbox/SIMToolbox · https://github.com/qi2lab/mcSIM
- FP: https://github.com/kellman/LearnedDesignFPM · https://arxiv.org/abs/2301.06595 (PtyLab)
- iSCAT: https://arxiv.org/abs/2006.15332 · https://joss.theoj.org/papers/10.21105/joss.04024 · https://pubs.acs.org/doi/10.1021/acsphotonics.3c00422
