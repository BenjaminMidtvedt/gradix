# 02 — Survey of differentiable / GPU wave- and ray-optics libraries: architecture patterns and lessons for gradoscopy

**Scope.** This brief compares how existing optics libraries represent fields, sampling, spectra, polarization and coherence, how they propagate, and how they compose "systems". The goal is to decide what gradoscopy (a PyTorch, differentiable, fidelity-tunable light-microscopy engine behind DeepTrack2) should copy and what it should avoid.

**Evidence legend.** **[V]** = verified in September 2026 by reading source or docs at the URL given. **[R]** = recollection, not re-verified; treat as probably right but check before relying on it. **[?]** = uncertain. Source-code summaries were obtained through a fetch tool that paraphrases. Exact signatures quoted below come from source files; small details (default values) should be re-checked before copying code.

---

## 1. At-a-glance comparison

| Library | Framework | Field container carries sampling? | Batch / spectral / polarization layout | Propagators | Coherence | API style | Relevance |
|---|---|---|---|---|---|---|---|
| **Chromatix** | JAX + Equinox | Yes: `u`, `dx`, `origin`, `spectrum` | `(... y x [wv] [p=3])`, arbitrary leading batch | Single-FFT Fresnel, SAS, Fresnel TF, ASM (+band-limit, CZT output), high-NA vectorial ff-lens (CZT), ABCD/Collins, multislice, polarized multislice, MBS (experimental) | Coherent only; incoherent fluorescence by Monte Carlo random phases | Functional core + mirrored Equinox modules; `OpticalSystem` (sequential); `Microscope` (PSF-convolution) | **Highest** (closest analogue) |
| **TorchOptics** | PyTorch | Yes: `PlanarGrid` (shape, spacing, offset, z) | `(..., [3], H, W)`; coherence `(..., H, W, H, W)` | ASM, direct integration (RS), Fresnel variants, AUTO choice | Full 4D mutual intensity | `nn.Module` elements positioned at z; `System` sorted by z | High (PyTorch idioms) |
| **odak** | PyTorch | No: dx, λ, z passed as arguments | 2-D fields; batching via kernel stacks | BL-ASM, ASM, Fresnel TF/IR, Fraunhofer, shifted BL-ASM | Coherent | Functions + non-`nn.Module` `propagator` class | Medium (holography) |
| **HoloTorch** (Meta) | PyTorch | Yes: `ElectricField(data, wavelengths, spacing)` | Fixed 6-D `BTPCHW` | ASM kernel, FFT lens | Coherent (+ time and pupil axes) | `nn.Module` components | Medium (dimension semantics) |
| **waveprop** (LCAV) | NumPy/PyTorch | No | 2-D | Fraunhofer, Fresnel 1/2/multi-step, BL-ASM, shifted, rescaled BLAS, pyFFS interpolation, direct integration | Coherent; polychromatic via loop | Functions | Medium (algorithm catalogue) |
| **neural-holography** | PyTorch | No | 2-D + batch | BL-ASM with `precomped_H`/`H_exp` | Coherent | Functions + learned models | Medium (hybrid physics + learning) |
| **dO / DiffOptics**, **DeepLens**, **Optiland** | PyTorch | Rays: `o`, `d`, λ | `[..., 3]` rays | Differentiable ray tracing; DeepLens adds ray-to-wave hybrid | Incoherent PSF | OO lens groups | Medium (objective aberrations, memory tricks) |
| **deepinv** | PyTorch | No (operators on tensors) | `(B, C, H, W[, D])` | Blur/FFT blur, space-varying blur, Zernike diffraction PSF generators | n/a | `Physics` = A + noise, composition, parameter generators | High (data-generation and inverse-problem API) |
| **LightPipes** | NumPy | Yes: `siz`, `lam`, `N`, `_curvature` | 2-D | Fresnel, Forvard, Steps, LensFresnel/LensForvard (variable grid) | Coherent | Command chain | Low (historical) |
| **diffractsim** | NumPy/CuPy/JAX | Extents + N | 2-D; per-λ loop | ASM, Bluestein `scale_propagate` | Polychromatic via deferred replay | `add(element)`, `propagate(z)` | Low–medium |
| **POPPY** | NumPy (+accel) | Yes: `pixelscale`, `planetype` | 2-D per λ | FFT, MFT, Fresnel with Gaussian-beam switching | Polychromatic via weighted λ loop | `OpticalSystem.add_pupil/add_image/add_detector` | Medium (sampling logic) |
| **hcipy** | NumPy | Yes: `Field.grid` (ndarray subclass) | Flattened points; tensor fields for Jones | FFT/MFT/naive chosen automatically; Fraunhofer, Fresnel, ASM | Jones + input Stokes (partial polarization) | Elements with `forward`/`backward`; agnostic caching | Medium–high (patterns) |
| **prysm** | NumPy/CuPy shim | Yes: `dx`, `space` | 2-D | `focus`/`unfocus` (FFT), fixed-sampling MDFT, ASM `free_space` | Coherent | OO `Wavefront` with hand-written adjoints | Medium |
| **dLux** | JAX + Equinox + Zodiax | Yes: `pixel_scale`, `plane`, `units` | per λ, vmapped | MFT, FFT, Fresnel | Polychromatic via source spectra | Layered systems; path `.set("a.b", v)` | Medium–high (parameter addressing) |
| **waveorder** (Mehta lab) | PyTorch (partly NumPy) | No (explicit args) | `zyx` volumes | Transfer functions (WOTF), Green's tensors, pupils | Partially coherent (illumination pupil), polarization (Stokes) | `calculate_transfer_function` / `apply_…` / `apply_inverse_…` | **High** (linear microscopy fidelity tier) |
| **Multislice/SSNP/MLB codes** | PyCUDA / PyTorch / TF | Mostly ad-hoc | `zyx` RI volume | BPM, SSNP, multi-layer Born | Coherent + illumination-angle sums | Scripts | High (thick-sample physics) |
| **WaveBlocks** | PyTorch | Global `OpticConfig` | depth stacks | RS propagation, lens FT, MLA, camera conv | Incoherent depth sum | `OpticBlock(nn.Module)` + `members_to_learn` | Medium (microscopy-specific) |
| **DeepSTORM3D** (deep-optics exemplar) | PyTorch | No | `(B, emitters, H, W, 2)` real/imag | Pupil phase + FFT per emitter | Incoherent emitter sum | `nn.Module` physical layer | High (emitter rendering pattern) |

---

## 2. Library deep-dives

### 2.1 Chromatix (chromatix-team/chromatix, JAX) — the closest analogue

Published as *Chromatix: a differentiable, GPU-accelerated wave-optics library*, Nature Methods 23, 1388–1398 (2026) [V: https://www.nature.com/articles/s41592-026-03121-x, PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC13042145/]. The authors report 2–6× speedups on one GPU over the original research code and up to 22× on 8 GPUs. The paper states that the library "currently models completely coherent propagation" [V, PMC].

**Package layout** [V: GitHub tree API]: `core/{base,field,spectrum}.py`, `functional/{propagation,lenses,phase_masks,amplitude_masks,polarizers,pupils,rays,samples,sensors,sources,convenience}.py`, `elements/…` (a module wrapper for each functional file), `systems/{optical_system,microscopes}.py`, `ops/{filters,noise,quantization,resample,ops}.py`, `utils/{czt,fft,shapes,initializers}.py`, `data/permittivity_tensors.py` and `experimental/modified_born_series/`. The repository has **migrated from Flax to Equinox**. Trainability is now expressed with `eqx.partition` and a filter spec, and the docs say this avoids "the abuse of static" [V: https://chromatix.readthedocs.io/en/latest/training/].

**Field data model** [V: `core/field.py`, `core/spectrum.py`, `core/base.py`]
- `Field(eqx.Module)` is an immutable pytree with `u` (complex), `dx`, `origin`, `spectrum`, and a class-level `dims: ClassVar[IntEnum]` that maps axis names to negative indices.
- There are four concrete classes built from mixins: `ScalarField` `(... y x)`, `ChromaticScalarField` `(... y x wv)`, `VectorField` `(... y x 3)` and `ChromaticVectorField` `(... y x wv 3)`. Leading batch dimensions are arbitrary.
- `dx` has shape `(2,)` for a monochromatic field and `(wv, 2)` for a chromatic one, so **spacing may vary with wavelength**.
- `Spectrum(wavelength[wv], density[wv])`: the density is normalised to sum to 1. `MonoSpectrum` is the single-wavelength case, and `Spectrum.build()` accepts a float, an array, or a `(wavelengths, densities)` tuple.
- Chromatic intensity is `sum(density * |u|^2, axis=wv)`. Power multiplies by the pixel area.
- Derived properties: `grid`, `f_grid`, `k_grid = 2π f_grid`, `extent = dx * spatial_shape`. Module-level helpers: `pad`, `crop`, `shift_grid`, `shift_field`, `cartesian_to_spherical` (for high-NA work). Operator overloads (`+`, `*`, …) return new Fields.
- `base.py` defines abstract `Sample` mixins (`Absorbing`, `Scattering`, `Fluorescent`, `Volume` with `num_planes`) and `Sensor`/`Resampler` bases. The sample is therefore described by its **material properties**, not by geometry.

**Propagation** [V: `functional/propagation.py`]
- `transform_propagate(field, z, n, pad_width, …)` is the single-FFT Fresnel method, and it **changes dx**.
- `transform_propagate_sas(field, z, n, …)` is the scalable ASM: three FFTs and an automatic 2× pad.
- `transfer_propagate` (Fresnel TF) and `asm_propagate` share the following signature: `(field, z, n, pad_width, cval, absorbing_boundary={'tukey','super_gaussian'}, kykx, shift_yx, output_dx, output_shape, use_czt=True, mode={'full','same'})`. `asm_propagate` adds `remove_evanescent` and `bandlimit`. The band limit follows the shifted-ASM limits of Matsushima (2010).
- All of these call `kernel_propagate(field, propagator, …)`. When `output_dx`/`output_shape` are given, it resamples the output **with a chirp-z transform (CZT)**.
- `z` may be a vector, which gives a batched defocus stack in a single call.
- **Padding is explicit.** `compute_padding_{transform,transfer,exact}(height, λ, dx, z)` estimate the padding from the Fresnel number `N_F = (D/2)^2/(λ z)`, and the caller passes the resulting integer. This matters for JIT, because array shapes must be static.
- `kykx` supplies a tilted-carrier offset, used for oblique illumination.
- Kernels are **not cached**. `compute_asm_propagator` and similar functions are exposed, so a user can precompute a kernel and reuse it.

**Lenses and focusing** [V: `functional/lenses.py`, `functional/convenience.py`]
- `ff_lens(field, f, n, NA=None, inverse=False)` applies a circular pupil `D = 2 f NA / n` and then `optical_fft`. The output spacing is `du = df * |λ z / n|` **for each wavelength**, so a chromatic field leaves the lens with a different grid for every λ.
- `high_na_ff_lens(field, f, n, NA, output_shape, output_dx)` works as follows: it converts to the spherical basis for vector fields, multiplies by `exp(i k s_z f)/s_z` with `s_z = sqrt(max(s_z^2, 0))`, which is a safe square root, and runs a `zoomed_fft` (CZT) to an arbitrary output grid. The default output grid equals the input grid.
- `rays.py` provides ABCD matrices (free space, thin lens, thick plano-convex lens) and `ray_transfer`, which uses Collins' integral. This is a cheap paraxial "system" propagator.

**Sources** [V: `functional/sources.py`]
- `point_source`, `objective_point_source`, `plane_wave`, `gaussian_plane_wave`, `generic_field`. All take `(shape, dx, spectrum, …, scalar=True)` and build the field through `Field.empty(...)`.
- `objective_point_source` uses a **paraxial quadratic** defocus phase `-π (z/f) |r-offset|^2 / L^2` with `L^2 = λ f / n`, then a circular pupil, then power normalisation. It is therefore a low-fidelity defocus model compared with the exact `sqrt(n^2 - ρ^2)` form.

**Samples** [V: `functional/samples.py`]
- `thin_sample` computes `exp(i 2π (dn + i·absorption) · thickness / λ)`. `jones_sample` is its 2×2 Jones version.
- `multislice_thick_sample` runs an ASM kernel between slices and a thin-sample screen at each slice, inside `lax.fori_loop`. It then back-propagates to the centre of the stack and applies an optional NA mask.
- `fluorescent_multislice_thick_sample` handles incoherent fluorescence **by Monte Carlo**: it applies random phases in [0, 2π) to the fluorescence stack, runs forward and backward propagators, and averages intensities over `num_samples`.
- `polarized_multislice_thick_sample` handles 3×3 permittivity tensors (birefringence) with transverse-projection and vectorial scattering operators.
- An experimental modified Born series solver is included.

**Systems** [V: `systems/optical_system.py`, `systems/microscopes.py`]
- `OpticalSystem(elements: Sequence[Callable])` is **strictly sequential**. The first element may take arbitrary arguments (for example a source), and the last may return an array.
- `Microscope` is an Equinox module. Its fields are `system_psf` (a callable that returns a PSF Field or array), `sensor`, `f`, `n`, `NA`, `spectrum`, `padding_ratio`, `taper_width`, `convolution_axes`, and `fast_fft_shape`.
- `__call__` computes the PSF, takes its intensity, crops the padding, applies a sigmoid taper, **resamples to sensor spacing**, runs `fourier_convolution(sample, psf)`, and calls the sensor, which may add noise. The PSF is recomputed on every call.
- `Optical4FSystemPSF` chains `objective_point_source → phase_change → ff_lens`.
- In other words, the "microscope" is a **PSF-convolution system**, not a full field simulation of the sample.

**Sensors, noise and resampling** [V: `functional/sensors.py`, `ops/noise.py`, `ops/resample.py`]
- `basic_sensor(sensor_input, shot_noise_mode={'approximate','poisson'}, resampler, reduce_axis, reduce_parallel_axis_name, input_spacing, noise_key)`. `reduce_parallel_axis_name` is for multi-device sums.
- `shot_noise` samples a **true Poisson** value in the forward pass, while its `custom_jvp` returns the gradient of the Gaussian approximation `x + sqrt(x)·ε`. This is a straight-through surrogate.
- `init_plane_resample` offers either `PoolingPlaneDownsampler` (sum pooling) or `InterpolatingPlaneResampler`, which uses `jax.image.scale_and_translate` (linear, cubic or lanczos) followed by division by `prod(scale)` to conserve energy.

**Lessons from Chromatix**
- *Copy:*
  - Sampling metadata travels with the field.
  - Named axis indices per field class.
  - A functional core with thin module wrappers.
  - Explicit padding with helper estimators.
  - CZT output resampling built into the propagators.
  - Batched `z`.
  - The straight-through Poisson surrogate.
  - A PSF-based `Microscope` as a fast tier.
- *Avoid:*
  - A per-wavelength `dx` that leaks out of Fourier-lens operations. Every later incoherent sum over λ then has to reconcile grids.
  - A sequential-only system.
  - Monte Carlo as the only route to incoherence, which gives noisy gradients and slow convergence.
  - Paraxial defocus inside a "point source" for high-NA use.
  - Placing spatial axes before `wv`/`p`. That order suits JAX, but in PyTorch `fft2` wants Y and X last and contiguous.

### 2.2 TorchOptics (MatthewFilipovich/torchoptics, PyTorch; arXiv 2411.18591)

Source files [V: GitHub tree]: `fields.py`, `planar_grid.py`, `optics_module.py`, `system.py`, `elements/{lens,modulators,polarized_modulators,polarizers,waveplates,beam_splitters,detectors,identity_element}.py`, `propagation/{angular_spectrum_method,direct_integration_method,propagator}.py`, `profiles/…`, `functional/functional.py`.

- **Geometry-first data model** [V]. `Field(PlanarGrid)` stores `data` `(..., H, W)` and a `wavelength`. Every plane (field or element) is a `PlanarGrid` with `shape`, `spacing`, `offset` and `z`.
  - Polarization is a 3-component axis at `POLARIZATION_DIM = -3`, so data has shape `(..., 3, H, W)`.
  - `SpatialCoherence`/`CoherenceField` stores the mutual intensity as `(..., H, W, H, W)`. The paper warns that this scales "quadratically relative to the Field class" [V: https://arxiv.org/html/2411.18591]. Concretely, a 256² grid needs 256⁴ complex64 values, about **34 GB**, and even 128² needs about 2.1 GB.
- **Parameter registration** [V: `optics_module.py`]. `OpticsModule.register_optics_property(name, value, …)`: if `value` is an `nn.Parameter` it is registered as a trainable parameter, otherwise as a buffer. Shapes are validated at creation and on assignment through `__setattr__`, so learnability is chosen **at construction by wrapping in `Parameter`**. The pattern is simple and idiomatic.
- **Propagation** [V: `propagator.py`, `angular_spectrum_method.py`].
  - Methods are `ASM`, `DIM` (direct integration: FFT convolution with the Rayleigh–Sommerfeld impulse response) and `AUTO`, each with a `_FRESNEL` variant.
  - `AUTO` picks ASM when `z < z_c = 2|x_max|Δx/λ` (from Voelz) in at least one axis.
  - Propagation happens in two stages. First the field goes to an intermediate plane that keeps the input spacing and is sized to cover the output plane. Then it is **interpolated** (bilinear, bicubic or nearest) onto the output plane's geometry.
  - The ASM pads 2× by default, uses `kz = sqrt(k² − kx² − ky² + 0j)` (evanescent components decay), and does **not band-limit**.
- **System** [V: `system.py`]. `System(nn.Module)` sorts elements by `z`. `forward` calls `field.propagate_to_plane(element)` and then `element(field)` for each element. `measure(shape, spacing, offset, z)` and `measure_at_plane` produce arbitrary output planes.
- **Lessons.**
  - *Copy:* the Parameter-or-buffer convention; positioning elements by plane geometry; measurement planes as first-class objects.
  - *Avoid:* silent real-space interpolation of complex fields between planes (lossy, aliasing-prone and non-band-limited; use Fourier or CZT resampling); a 4-D mutual intensity as the primary coherence model; an unbanded ASM at large `z`.

### 2.3 odak (kaanaksit/odak, PyTorch; UCL Computational Light Lab)

- Data model [V: `odak/learn/wave/classical.py`]. A field is a **bare complex tensor**, and `k`, `distance`, `dx` and `wavelength` are passed as separate arguments.
  - `propagate_beam(field, k, distance, dx, wavelength, propagation_type="Bandlimited Angular Spectrum", kernel=None, zero_padding=[True, False, True], aperture=1.0, scale=1, samples=[20,20,5,5])`.
  - Supported methods: BL-ASM, ASM, Fresnel TF, Fresnel impulse response (IR), Fraunhofer, and shifted BL-ASM with carrier offsets `offset_fx/fy`.
  - The band limit is `fx_max = 1/sqrt((2 z / X)^2 + 1)/λ`, where X is the padded extent.
  - `get_light_kernels()` builds `[wavelength × distance × pitch × m × n]` kernel stacks.
- `propagators.py`'s `propagator` class **is not an `nn.Module`**. It caches kernels in `self.kernels[depth, channel]` and reuses them through `.detach().clone()` [V]. As a result, **gradients with respect to λ, z and pitch are silently cut** after the first call.
- Lesson: a clean catalogue of classical propagators and good holography tooling. Avoid loose sampling arguments, non-module stateful objects (device, dtype and `state_dict` handling), and caches that detach.

### 2.4 HoloTorch (facebookresearch/holotorch, PyTorch)

- `ElectricField(data, wavelengths: WavelengthContainer, spacing: SpacingContainer, identifier)` asserts `data.ndim == 6` [V: `CGH_Datatypes/ElectricField.py`].
- The dimension classes are `BTPCHW` = Batch, Time, Pupil, Channel, Height, Width, together with `TC`, `C`, `HW` and others [V: `utils/Dimensions.py`]. The containers carry their own `TensorDimension`, so for example wavelengths can vary over T×C and still broadcast correctly.
- The repository appears dormant (a handful of commits) [V].
- Lesson: **declaring the dimension semantics of metadata** (the axes along which wavelength or spacing vary) is the right idea. A fixed 6-D layout is too rigid; use arbitrary leading batch dimensions plus named trailing axes.

### 2.5 waveprop (ebezzam/waveprop, LCAV; NumPy and PyTorch)

- Functions in `rs.py`, `fresnel.py`, `fraunhofer.py`, `spherical.py`, `slm.py`, `color.py` and `simulation.py` [V: tree].
- `angular_spectrum(u_in, wv, d1, dz, bandlimit=True, out_shift=0, d2=None, N_out=None, pyffs=False, aperture=None, in_shift=None, weights=None, return_H=False, return_H_exp=False, H=None, H_exp=None, U1=None, device=None, pad=True)` [V].
  - The band limit is Table 1 of the shifted-ASM paper (Matsushima 2010).
  - Arbitrary output sampling uses either the 2012 "rescaled BLAS" method (chirp multiplication and convolution) or **pyFFS**, a Fourier-series interpolation.
  - **The `H_exp` trick:** store the complex exponent `i·kz` once and evaluate `H = exp(H_exp·dz)`. The kernel is then built in one fused operation and **gradients with respect to `dz` flow**.
- `FarFieldSimulator` resizes the object to set its size, runs an rfft convolution with the PSF, downsamples to the sensor, adds shot noise and quantises [V].
- Lesson: `H_exp` is a small, high-value trick for learnable distances or defocus. It also shows that propagation variants are best organised as a registry.

### 2.6 neural-holography (computational-imaging/neural-holography; Wetzstein lab)

- `propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True, padtype='zero', return_H=False, precomped_H=None, return_H_exp=False, precomped_H_exp=None, dtype=torch.float32)` uses the Matsushima band limit and 2× padding for linear convolution [V].
- `propagation_model.py` adds learnable aberrations, source amplitude and CNN corrections trained **camera-in-the-loop** [V: README].
- Lesson: "physics model + learned residual" is a proven pattern. gradoscopy should let a learned module stand in for any element (for example a pupil-plane or detector-plane CNN), both for calibrating against real data and for closing the sim-to-real gap.

### 2.7 Deep-optics repos: ∂μ (Wadduwage lab), DeepSTORM3D, and others

- **Differentiable Microscopy (∂μ)** [V: arXiv 2203.14944; PMC10942716; code https://github.com/wadduwagelab/OpticalElectronicQPI].
  - Forward model: `I = D(|H_O(x_in)|^2)`. The sample is a thin phase object `A·e^{iφ}`. The optics are either a learnable 4f Fourier filter (256² grid, radius 128) or a 5-layer phase-only D2NN with λ/2-scale "neurons" and Rayleigh–Sommerfeld propagation between layers. The detector is 2×2 average pooling plus Poisson and read noise (σ = 4–6).
  - Training is staged: optics first, then decoder, then end-to-end fine-tuning. The authors report D2NN instability at large fields of view.
  - Lesson: ∂μ is a *methodology* (top-down, learnable elements at key planes), not a library. Its code is task-specific and all sampling is implicit. gradoscopy should make exactly this workflow trivial: "take a standard microscope, mark the Fourier-plane element learnable".
- **DeepSTORM3D** (PyTorch; learned PSF engineering for 3-D single-molecule localisation) [V: `physics_utils.py`].
  - `PhysicalLayer(nn.Module)` renders one PSF per emitter. It multiplies the phase mask by an emitter-specific pupil phase `exp(i(x0·X + y0·Y))·exp(i z0·Z)`, which includes an **oil/water refractive-index-mismatch defocus** via Snell's law in the back focal plane. It then runs a centred FFT, takes |·|², and scales by the photon count.
  - A random Gaussian blur layer (σ between 0.5 and 1.5 pixels) absorbs model mismatch. Poisson noise uses the `x + sqrt(x)·ε` approximation, plus read noise and a non-uniform background.
  - The model is scalar. Complex numbers are stored as real tensors with a trailing dimension of 2 (pre-complex-autograd era), and there is no custom autograd.
  - Lesson: for point emitters, **sub-pixel positions enter as Fourier phase ramps**, which is exactly differentiable and needs no rasterisation. Rendering per emitter costs memory `B × emitters × N²`.
- Other end-to-end deep-optics code (Sitzmann et al. 2018 achromatic EDOF, Metzler et al. 2020 HDR, Chang & Wetzstein 2019 depth, PhaseCam3D 2019) [R]: mostly TensorFlow, one-off scripts that hard-code sampling in the form "pupil height map → PSF → convolve → noise". A curated index is at https://github.com/singer-yang/awesome-deep-optics [V: exists].
- Related PSF tools, useful as ground truth and not as architectures [R]: **uiPSF** (Liu, Ries et al., Nature Methods 2024; TensorFlow; vectorial and 4Pi PSF inverse modelling); psfmodels (C++ Gibson–Lanni and vectorial); pyotf.

### 2.8 Differentiable ray tracing: dO/DiffOptics, DeepLens, Optiland

- **dO** (Wang, Chen and Heidrich, IEEE Transactions on Computational Imaging 2022) [V: https://github.com/vccimaging/DiffOptics].
  - `Ray(o, d, wavelength, mint, maxt)` with batched `[..., 3]` origins and directions and one λ per Ray object. `Transformation(R, t)`, `Material` with `n(λ) = A + B/λ²` from (nD, V), aspheric/XY-polynomial/B-spline surfaces, and lens groups.
  - Two key techniques:
    1. **Implicit differentiation of the ray–surface intersection.** Newton iterations run without autograd, and the gradient is attached at the solution (about 6× memory saving).
    2. **Adjoint back-propagation (ABP).** Compute ∂L/∂image first, then back-propagate through the renderer in ray chunks, so memory is bounded by the chunk size (similar to checkpointing).
- **DeepLens** (Xinge Yang et al.) [V: https://github.com/singer-yang/DeepLens].
  - Interchangeable lens models: `GeoLens` (ray tracing), `DiffractiveLens`, `HybridLens` (ray-trace to the exit pupil, then wave propagation), and `PSFNetLens` (a neural surrogate).
  - All share a `psf()`/`render()` interface. Spatially varying, depth-dependent PSFs are rendered through PSF maps and patch convolution.
  - **This is precisely the "same description, several fidelity backends" pattern that gradoscopy wants.**
- **Optiland** [V: https://github.com/optiland/optiland]: optical design with a switchable NumPy/PyTorch backend, polarization ray tracing, and differentiable non-sequential tracing.
- Lessons for microscopy: ray tracing matters only when the *objective's real prescription* is needed (field-dependent aberrations, stray light). A hybrid "ray → pupil phase/amplitude map → wave PSF" is the right coupling point: the pupil function is the common currency. The implicit-differentiation and ABP tricks transfer directly to any iterative or large-sum computation.

### 2.9 deepinv (PyTorch; inverse-problems library)

- `Physics` models `y = N(A(x))` and is an `nn.Module` [V: https://deepinv.org/user_guide/physics/intro.html].
  - `LinearPhysics` adds `A`, `A_adjoint`, `A_dagger`, `prox_l2`, `compute_norm` and `condition_number`.
  - Operators compose with `*` or `compose()` and stack with `stack()`.
  - Parameters may be given at construction or per call (`physics(x, filter=θ)`), and they stay differentiable (for blind or calibration problems).
- **Generators**: `PhysicsGenerator.step(batch_size)` returns a parameter dict, generators can be added together, and `GeneratorMixture` mixes them.
  - Microscopy-relevant generators: `DiffractionBlurGenerator` (Zernike pupil → |FFT|², Noll indices 4–11 by default, cutoff `fc ∈ [0, 0.25]` for Nyquist, `max_zernike_amplitude=0.15` waves, a super-resolved pupil of 256²), `DiffractionBlurGenerator3D`, `ConfocalBlurGenerator3D`, `ProductConvolutionBlurGenerator` and `TiledBlurGenerator`, plus `SpaceVaryingBlur` [V: https://deepinv.org/api/deepinv.physics.html and the generator page].
  - Noise models: `PoissonGaussianNoise`, `GammaNoise`, and others.
- Lesson: **separate the random parameter sampler (generator) from the deterministic differentiable operator.** This is exactly DeepTrack's sampling semantics, made differentiable. Exposing `A_adjoint` makes gradoscopy's models reusable in reconstruction, and product-convolution is the standard fast tier for space-variant PSFs.

### 2.10 LightPipes, diffractsim, POPPY, hcipy, prysm, dLux (non-differentiable or JAX references)

- **LightPipes** [V: `LightPipes/field.py`]
  - The `Field` holds `_field`, `_siz`, `_lam`, `_curvature` and Gaussian-beam flags. `dx = siz/N`.
  - `LensFresnel`/`LensForvard` propagate in **spherical coordinates with a variable grid** (the grid scales by about (f−z)/f), and `Convert` returns to normal coordinates by applying the stored curvature [R for the scaling law].
  - Lesson: storing an analytic curvature term separately from the sampled field ("reference wave") makes it possible to follow a focusing beam with few samples. POPPY does the same.
- **diffractsim** [V: `polychromatic_simulator.py`]
  - `PolychromaticField(spectrum, extent_x, extent_y, Nx, Ny, spectrum_size, spectrum_divisions=30)`. `add(element)` and `propagate(z)` only *record* steps. `get_colors()` replays the recorded chain for each wavelength and accumulates CIE XYZ → sRGB on the fly, so all spectral fields are never held at once.
  - `scale_propagate` uses Bluestein (chirp-z) propagation. The backend can be NumPy, CuPy or JAX.
  - Lesson: **deferred execution with per-spectral-sample replay and accumulation** keeps spectral memory at O(1) (a memory-versus-parallelism trade-off).
- **POPPY** (NumPy, non-differentiable) [V: `fresnel.py`, `matrixDFT.py`]
  - `Wavefront.planetype` is pupil, image, detector or intermediate, and `pixelscale` is in m/pix or arcsec/pix [R].
  - `matrix_dft(plane, nlamD, npix, offset, inverse, centering={FFTSTYLE, SYMMETRIC, ADJUSTABLE})` is an MFT with cost about O(N·M² + N²·M), and `nlamD` sets the output field of view in λ/D.
  - Fresnel mode tracks Gaussian-beam parameters `w_0`, `z_w0` and `z_r` together with a `spherical` flag, and chooses a propagation path:
    - plane-to-plane (ASM) inside `rayleigh_factor·z_r` (default 2);
    - waist-to-spherical, a single FFT with `pixelscale' = λ|dz|/(N·pixelscale)`;
    - spherical-to-waist.
  - Only `QuadraticLens` updates the beam parameters. Quantities use astropy units.
  - Lesson: automatic regime switching driven by tracked beam parameters works well. astropy units in inner loops are slow; keep units at the API boundary.
- **hcipy** (NumPy) [V: `optics/optical_element.py`, `optics/wavefront.py`, `fourier/fourier_transform.py`]
  - `Field` is an ndarray subclass carrying `.grid`. Values are flattened over the grid points; tensor fields add leading dimensions [R for flattening]. Grids can be regular, separated or unstructured coordinates.
  - **`make_fourier_transform` picks FFT, MFT, naive or ZFT by estimated or measured cost**, given the input and output grids.
  - `OpticalElement.forward/backward` and `get_transformation_matrix_forward` (explicit `E_out = M E_in`).
  - `AgnosticOpticalElement` **caches per-(grid hash, wavelength) instance data** in an LRU `OrderedDict` (`max_in_cache`), and `evaluate_parameter` inspects which of `input_grid`, `output_grid` and `wavelength` a parameter function depends on.
  - `Wavefront(electric_field, wavelength, input_stokes_vector)` stores a Jones field `(2, N)`. With an input Stokes vector, the field becomes a `(2, 2, N)` Jones *matrix* acting on that Stokes vector, and intensity and Stokes values come from the Mueller conversion.
  - Lesson: (a) choose the transform by grid types and cost; (b) cache per-λ kernels; (c) **carry the system's Jones matrix and contract it with the source's coherency at detection**. Point (c) handles unpolarized or partially polarized light at twice the cost of a single coherent propagation.
- **prysm** [V: `propagation/wavefront.py`]
  - `Wavefront(cmplx_field, wavelength, dx, space='pupil'|'psf')`. `focus(efl, Q=2)` and `unfocus` use an FFT with `Q` as the padding factor ("Q at least 2" to avoid aliasing). `free_space(dz, Q, tf)` uses the ASM. There is also `focus_fixed_sampling` (matrix DFT), plus coronagraph helpers.
  - It provides **hand-written adjoints** (`focus_adjoint`, `free_space_adjoint`, `intensity_adjoint`, …) for algorithmic differentiation without an autodiff framework. The backend is a NumPy/CuPy shim.
  - Units are mixed (mm in the pupil, µm in the PSF), which is a known source of confusion.
- **dLux** (JAX + Equinox + Zodiax) [V: https://github.com/LouisDesdoigts/dLux]
  - `Wavefront` carries `pixel_scale`, `wavelength` and `plane`. Optical layers make up `AngularOpticalSystem`/`LayeredOpticalSystem`; sources carry spectra; detectors are layers; propagation uses MFT or FFT, with Fresnel through an ABCD engine.
  - **Parameters are addressed by dotted path strings** (`model.set("aperture.diameter", v)`, `.get`) and updated immutably.
  - Lesson: path addressing maps naturally onto DeepTrack's property system and onto "learn this subset" filters.

### 2.11 waveorder (mehta-lab/waveorder, PyTorch) — the linear-transfer-function fidelity tier

- Paper: *WaveOrder: a differentiable wave-optical framework for scalable biological microscopy with diverse modalities* [V: https://arxiv.org/html/2412.09775].
  - Every transfer function is built from **three submodels: a scattering model, an illumination pupil, and a detection pupil**. The scattering model is a single Green's-tensor spectrum.
  - Specimens are "physically interpretable" potentials: phase, absorption, birefringence and diattenuation, or dipole-orientation moments for fluorescence.
  - It assumes weak, single scattering, which the authors say "break[s] down in optically thick or heterogeneous samples".
  - Shift-variant aberrations are handled by fitting per-tile parameters (defocus, tilt, NA) by backpropagation.
- Code layout [V: `waveorder/models/*.py`, `optics.py`]: one module per modality (`isotropic_thin_3d`, `phase_thick_3d`, `isotropic_fluorescent_thick_3d`, `inplane_oriented_thick_pol3d[_vector]`, `shift_variant_fluorescent_3d`, …), each with the same triad:
  - `calculate_transfer_function(zyx_shape, yx_pixel_size, z_pixel_size, wavelength_illumination, z_padding, index_of_refraction_media, numerical_aperture_illumination, numerical_aperture_detection, invert_phase_contrast, tilt_angle_zenith, tilt_angle_azimuth, pupil_steepness=1e4)`;
  - `apply_transfer_function(...)`;
  - `apply_inverse_transfer_function(..., reconstruction_algorithm={'Tikhonov','TV','RL','RLGC'})`.
- Differentiability details [V: `optics.py`]:
  - **`generate_pupil` uses a sigmoid edge, `sigmoid(steepness·(NA/λ − f_r))`**, so NA receives gradients.
  - `generate_greens_function_z` adds `1e-15` and a `(1 − pupil_support)` term to the denominator to avoid division by zero.
  - The 3-D weak-object transfer function is `compute_weak_object_transfer_function_3D(ill_pupil, ill_pupil, det_pupil, propagation_kernel, greens_function_z, dz)`.
  - Parts of the code are still legacy NumPy/CuPy (for example the SEAGLE vector model and Stokes utilities).
- Lessons.
  - (1) The same operator object serves both simulation and reconstruction.
  - (2) The expensive transfer function is precomputed separately from the cheap apply step, with an obvious cache boundary.
  - (3) Soft pupils make NA learnable.
  - (4) Partially coherent weak-object imaging (brightfield, DPC, phase and polarization microscopy) collapses to *one 3-D convolution per channel*. This is the natural "medium fidelity" tier between PSF convolution and multislice.

### 2.12 Thick-sample multiple-scattering codes (BPM, SSNP, MLB, MBS)

- **BPM / multislice.** `φ(z+Δz) = F⁻¹{e^{i k_z Δz} F{φ}}·e^{i k₀ Δn(r) Δz}`. It is cheap (two FFTs per slice) and forward-scattering only; accuracy falls off at high angles and large Δn. Implementations include Chromatix `multislice_thick_sample` [V] and the PyTorch "multi-slice neural network" tomography code [V: https://github.com/yang980130/Physics-based-3D-tomography-Multi-slice-neural-network].
- **SSNP** (split-step non-paraxial; Lim, Ayoub, Antoine and Psaltis, Light: Science & Applications 2019; later applied to intensity diffraction tomography (IDT) by the Tian lab, 2022).
  - It propagates the state `(φ, ∂φ/∂z)`. The homogeneous step is an exact 2×2 k-space rotation `[[cos k_zΔz, sin(k_zΔz)/k_z], [−k_z sin k_zΔz, cos k_zΔz]]`, and the scattering step is `∂_zφ += k₀²(n₀² − n²)φΔz` [R for exact form].
  - It costs about 2× BPM and is much more accurate at high NA.
  - The reference code **bu-cisl/SSNP-IDT is PyCUDA with its own AD framework** [V: https://github.com/bu-cisl/SSNP-IDT].
- **Multi-layer Born** (Chen, Ren and Waller, Optica 2020) [R] and the **modified Born series** (Osnabrugge 2016; Chromatix `experimental/modified_born_series`) [V: exists] sit higher up in fidelity and cost.
- Memory is the dominant engineering issue. Backpropagating through S slices of an N² field stores O(S) complex fields: for example 200 slices × 2048² × complex64 × roughly 2 saved tensors per slice is about **13 GB**. Options are checkpointing (√S), hand-written adjoints (SSNP-IDT), or **reversible recomputation**: each slice step is invertible when Δn is real and absorption is bounded, so the field can be recomputed backwards from the output (as in reversible networks). The last option is [R], a known technique, and not observed in these codes.

### 2.13 WaveBlocks (pvjosue/WaveBlocks, PyTorch; light-field microscopy)

- `OpticBlock(nn.Module)` holds an `OpticConfig` (a shared global λ, k and n) and `members_to_learn`. The blocks are `WavePropagation` (Rayleigh–Sommerfeld), `Lens` (focal-plane FT), `DiffractiveElement`, `MicroLensArray` and `Camera` (|·|² plus a convolution with the object, summed over depth). The objective PSF stack `[1, nDepths, H, W, 2]` is imported from a file [V: README].
- Lesson: microscopy users want "Lego" blocks plus depth-resolved PSF stacks. A global config object couples everything and blocks polychromatic batches, so put λ and n on the field, not in a singleton.

---

## 3. Cross-cutting analysis

### 3.1 Sampling bookkeeping (the most important design axis)

Observed designs:
1. **Loose arguments** (odak, neural-holography, waveprop, DeepSTORM3D, waveorder functions): `dx`, `λ` and `z` are passed on every call. This is error-prone, and grid changes (after a Fourier transform) have to be tracked by hand.
2. **Sampling on the field** (Chromatix `dx`/`origin`; TorchOptics `spacing`/`offset`/`z`; POPPY `pixelscale`/`planetype`; prysm `dx`/`space`; dLux `pixel_scale`/`plane`; hcipy `grid`; HoloTorch `SpacingContainer`). Grid-changing operations update the metadata automatically. **This is the consensus modern design.**
3. **Grid type drives the algorithm** (hcipy): the input and output grids determine whether FFT, MFT or a naive DFT is used.
4. **Plane geometry drives propagation** (TorchOptics): the target plane's shape, spacing and offset define the output, and the library works out the intermediate padding and interpolation.
5. **Analytic reference-wave separation** (LightPipes curvature, POPPY spherical reference): quadratic phase is kept analytically so few samples are needed.

The key grid-changing operations in a microscope, and how they should be handled:
- **Pupil ↔ focal plane (objective/tube lens).** A plain FFT gives `dx' = λ f/(n N dx)`, which is λ-dependent: Chromatix produces a per-λ dx, while POPPY, hcipy, prysm and dLux avoid this with an MFT or CZT to a *specified* detector sampling. For polychromatic microscopy, the MFT/CZT approach is clearly better. Define the pupil grid in normalised spatial-frequency (NA) units and evaluate each λ with its own MFT onto **one common image-plane grid**. The per-λ cutoff is just a different pupil radius on the same k-grid.
- **Defocus / free space.** ASM keeps dx unchanged; single-FFT Fresnel and SAS change or zoom it. For microscopy (µm-scale z), ASM with a band limit is almost always right; SAS or CZT is needed only when the output field of view must grow (for example long-distance holography).
- **Sample voxel grid → field grid.** The sample's voxel grid must equal, or be resampled to, the field grid. Resampling should be anti-aliased (area-weighted or Fourier-based), and ideally the sample itself is *band-limited analytically* (see §3.8).
- **Field grid → camera pixels.** Integrate intensity over pixel area (sum pooling for integer ratios; otherwise area-weighted or Fourier resampling; Chromatix normalises energy by `prod(scale)`). Magnification is a coordinate scaling at the detector, not a propagation.

### 3.2 Propagator catalogue: validity and cost

| Method | Output dx | Validity / sampling condition | Cost (N² field) | Notes |
|---|---|---|---|---|
| ASM (exact scalar, homogeneous) | = input | H aliases when z > ~N dx²/λ; needs a 2× pad for linear convolution | 2 FFTs of (2N)² | Evanescent components: decay or remove. Differentiable in z and λ; use `H_exp` for z |
| Band-limited ASM (Matsushima & Shimobaba 2009) | = input | Cutoff `f_lim = 1/(λ·sqrt((2Δf·z)² + 1))` with Δf = 1/(padded extent) | same | Default for microscopy defocus |
| Shifted BL-ASM (2010) | = input, shifted window | off-axis windows | same | Tiling and off-axis |
| Fresnel TF | = input | z ≤ N dx²/λ (Voelz) | 2 FFTs | Paraxial |
| Fresnel IR / direct integration (RS) | = input | z ≥ N dx²/λ | ~3 FFTs of (2N)² | TorchOptics `DIM`, WaveBlocks |
| Single-FFT Fresnel ("transform") | λz/(N dx) | Input chirp sampled when z ≥ N dx²/λ | 1 FFT | Zoom is locked to λ and z |
| SAS (Heintzmann, Loetgering & Wechsler, Optica 2023) | Zoomable (target pitch ≥ source) | High-NA valid; distance limit derived in the paper | 3 FFTs at 2× pad | Chromatix `transform_propagate_sas`; PyTorch notebook at https://github.com/bionanoimaging/Scalable-Angular-Spectrum-Method-SAS [V] |
| CZT / Bluestein zoom | Arbitrary window and pitch | Band-limited input | ~3 FFTs of size ≥ N+M−1 per axis | Chromatix `use_czt`, diffractsim |
| MFT (Soummer et al. 2007) | Arbitrary | — | O(M·N² + M²·N) matmuls | Cheaper than a padded FFT for small output regions; uses tensor cores, **but TF32 matmul silently degrades accuracy in PyTorch** (disable for MFT) |
| Debye / Richards–Wolf (vectorial high-NA focus) | Arbitrary via CZT | Debye validity: Fresnel number of the focusing aperture ≫ 1 (true for objectives) | CZT per polarization component | Chromatix `high_na_ff_lens` |
| Collins / ABCD | Scaled | Paraxial | 1–2 FFTs | Chromatix `ray_transfer` |
| Multislice BPM / SSNP / MLB / MBS | = input | Forward-scattering / non-paraxial / Born-series convergent | S × (2–10) FFTs | Thick samples |

### 3.3 Spectra

- **Wavelength axis inside the field** (Chromatix `wv` axis plus `Spectrum(wavelength, density)`; HoloTorch C axis with a `WavelengthContainer`). This is fully parallel but uses memory proportional to the number of wavelengths, and a Fourier lens produces per-λ grids.
- **Loop outside, one λ per wavefront** (hcipy, POPPY `calc_psf(source={'wavelengths', 'weights'})`, dLux vmaps over λ, diffractsim replay-and-accumulate). The core is simpler, and memory stays flat when loops are chunked.
- Recommendation: keep the spectral axis as an ordinary broadcast axis (so the parallel path is the default), and support **chunked accumulation** for large spectra, as diffractsim does. Spectral weights must carry quadrature weights (Δλ), and density semantics must be explicit: either power per sample or normalised to 1 as in Chromatix. Treat excitation and emission spectra as separate objects; fluorescence has a Stokes shift, and Chromatix acknowledges it has no spectrum-shifting model.

### 3.4 Polarization

- 3-component Cartesian `(Ex, Ey, Ez)` fields (Chromatix `VectorField` with p=3; TorchOptics at dimension -3) are required once high-NA focusing produces Ez, near the sample and for dipole emission.
- 2-component Jones fields plus `input_stokes_vector` (hcipy) handle partially polarized or unpolarized input by carrying a Jones *matrix* per pixel, which costs twice a scalar propagation.
- Chromatix supports Jones samples, polarizers and a 3×3 permittivity multislice; waveorder expresses polarization through Stokes and Mueller transfer functions.
- Recommendation: a polarization axis of size 0 (scalar), 2 (transverse Jones in the pupil or paraxial path) or 3 (Cartesian at the focus or sample), with explicit basis conversions at the objective (the Richards–Wolf rotation). Unpolarized light and random dipole orientations should be *coherence reductions* (§3.5), never special field types.

### 3.5 Coherence

Observed approaches:
- Coherent only (Chromatix, odak, most others).
- Full 4-D mutual intensity (TorchOptics; O(N⁴) memory, impractical above about 128²).
- Monte Carlo random phases (Chromatix fluorescence).
- Incoherent sums over explicit emitters (DeepSTORM3D) or depth planes (WaveBlocks).
- Transfer-function models (waveorder WOTF with illumination and detection pupils).

Missing everywhere: a general mechanism that lets the *same* system be evaluated with an exact coherent-mode sum (Abbe source points, emitters, polarization states, wavelengths), a stochastic estimator, or an analytic shortcut (OTF, WOTF, TCC/SOCS). **This is gradoscopy's opportunity.**

### 3.6 System composition

- Almost every library is **sequential**: Chromatix `OpticalSystem`, TorchOptics `System` (z-sorted), hcipy `OpticalSystem`, dLux layers, POPPY `add_*`, WaveBlocks.
- The exceptions are Chromatix `Microscope` (PSF generator → convolution → sensor, i.e. a two-stage *structure* rather than a chain), DeepLens (one interface over ray, wave, hybrid and neural backends), deepinv (algebraic composition and stacking of operators, plus adjoints), and waveorder (a scattering model × illumination pupil × detection pupil).
- A real microscope is a DAG: an illumination branch, sample interaction, a detection branch, incoherent reductions, interferometric sums (holography, iSCAT, where a reference is added to the scattered field), and multi-camera or polarization splits.

### 3.7 Parameters and trainability

- **JAX:** Equinox pytrees with filter-based partitioning (Chromatix, dLux). Every leaf *can* be trained, and the choice is made at optimisation time.
- **PyTorch:** `Parameter`-or-buffer at construction (TorchOptics); a `members_to_learn` list (WaveBlocks); per-call parameter dicts from generators (deepinv).
- **dLux** path strings give uniform addressing for get, set and filtering.

### 3.8 Differentiability caveats observed

1. **Hard pupil edges** give zero gradient with respect to NA or pupil radius. Fix: soft sigmoid pupils (waveorder `pupil_steepness`) or analytic area-coverage anti-aliasing.
2. **Parameter-dependent shapes** (padding from z, grid size from NA or λ) are non-differentiable and trigger recompilation. Chromatix forces explicit `pad_width`. Plan shapes from parameter *bounds*.
3. **Square roots at the evanescent cutoff** (`k_z → 0`) produce infinite or NaN gradients. Fix: clamp plus the double-`where` pattern (Chromatix `sqrt(max(·, 0))` inside `where`; waveorder adds ε to denominators).
4. **Detached kernel caches** (odak) silently kill gradients with respect to λ and z. A cache must be keyed on the parameters and must be bypassed when they require gradients.
5. **Poisson sampling** is not differentiable. Fix: a straight-through estimator with a Gaussian surrogate gradient (Chromatix `custom_jvp`), or the fully Gaussian approximation (DeepSTORM3D, ∂μ).
6. **Iterative solvers** (Newton intersection, Born series, root finding): use implicit differentiation (dO).
7. **Discrete geometry** (voxelised spheres) is non-differentiable with respect to radius or position under hard rasterisation. Fixes: Fourier phase ramps for positions (DeepSTORM3D), analytic Fourier transforms of primitives (a sphere has `3(sin qR − qR cos qR)/(qR)³`), or soft signed-distance rasterisation.
8. **Complex autograd** (PyTorch uses Wirtinger conventions): `angle()` and `abs()` have singular gradients at zero amplitude, so compute intensity as `real² + imag²`.
9. **Monte Carlo incoherence** gives unbiased but high-variance gradients. Offer deterministic alternatives.

### 3.9 Performance and memory patterns observed

- Batch z, λ, emitters and illumination angles as broadcast dimensions (Chromatix batched z; odak kernel stacks).
- Precompute kernels and use `H_exp` for learnable z (waveprop, neural-holography).
- Keep per-grid, per-λ LRU caches (hcipy).
- Choose transforms by cost: MFT for small output regions (POPPY, hcipy, prysm).
- Use rfft and "fast" FFT sizes for intensity convolutions (Chromatix `fast_fft_shape`; waveprop rfft2).
- Use loops (`lax.fori_loop`/`scan`) for multislice, with checkpointing or hand-written adjoints for memory (SSNP-IDT, prysm).
- Chunk over rays or emitters with adjoint back-propagation (dO).
- Scale out with multi-device batch sharding (Chromatix, via JAX).
- Rough footprints:
  - One 2048² complex64 field is 33.5 MB, and 134 MB with 2× padding.
  - A batch of 64 PSF stacks × 64 z-planes × 256² complex64 is about 2.1 GB.

---

## 4. What to copy and what to avoid (summary)

**Copy**
- Sampling on the field, updated by every grid-changing operation (Chromatix, TorchOptics, POPPY, prysm, dLux).
- Named axis semantics per field type (Chromatix `dims` IntEnum; HoloTorch dimension classes for metadata).
- A functional core with thin `nn.Module` wrappers (Chromatix, TorchOptics).
- Parameter-or-buffer registration (TorchOptics) and path-based parameter addressing (dLux).
- A propagator registry with validity and cost-driven automatic selection (TorchOptics AUTO, hcipy `make_fourier_transform`).
- CZT/MFT output resampling to physically specified grids (Chromatix `output_dx`, POPPY `Detector`, hcipy focal grids).
- Band-limited ASM as the default, with `H_exp` for learnable distances.
- The straight-through Poisson surrogate (Chromatix).
- Soft pupils (waveorder).
- Implicit differentiation and adjoint chunking (dO).
- Fourier phase ramps for emitter positions (DeepSTORM3D).
- Jones matrix on an input Stokes vector for partial polarization (hcipy).
- Transfer-function tiers with separate precompute and apply steps, and simulation and inversion sharing operators (waveorder, deepinv).
- A common interface over ray, wave, hybrid and neural backends (DeepLens).
- Parameter generators separate from physics (deepinv).

**Avoid**
- Loose `dx`/`λ` arguments (odak, neural-holography, waveprop).
- Per-λ grids escaping a lens (Chromatix).
- Implicit real-space interpolation between planes (TorchOptics).
- Detached caches (odak).
- Global optical config singletons (WaveBlocks).
- A fixed 6-D layout (HoloTorch).
- Flattened field storage (hcipy).
- Mixed units (prysm) and heavy unit objects in hot loops (POPPY).
- 4-D mutual intensity as the main coherence model (TorchOptics).
- Sequential-only system graphs (nearly all).
- Monte Carlo as the only route to incoherence (Chromatix).
- Paraxial defocus hidden inside "sources" (Chromatix `objective_point_source`).
- Non-module stateful objects (odak `propagator`).

---

## 5. Implications for gradoscopy's architecture

These recommendations are deliberately opinionated.

**R1. One `Field` type with explicit, tensor-friendly layout and metadata.** Make it a frozen dataclass registered as a `torch.utils._pytree` node (so it works with `torch.func.vmap`, `grad` and `torch.compile`) or a `tensordict` tensorclass [?; evaluate compile compatibility]. It holds:
- `u: complex Tensor[*batch, λ, P, Y, X]`, with Y and X **last and contiguous** for `torch.fft.fft2`. `λ` and `P` are always present but may be size 1 or broadcastable; P ∈ {1 scalar, 2 Jones-xy, 3 Cartesian-xyz}.
- `grid: Grid2D(shape: tuple[int, int] (static), spacing: Tensor[2], origin: Tensor[2], domain: "space" | "pupil")`. The pupil domain is expressed in spatial-frequency or NA units, **never λ-dependent pixel sizes**.
- `spectrum: Spectrum(wavelengths[λ], weights[λ])`, with weights being power per sample including the quadrature weight. Excitation and emission are separate Spectrum objects.
- `medium: n` (may be a tensor, learnable).
- `pol_basis`.
- An optional analytic `reference: QuadraticPhase` (the LightPipes/POPPY idea, useful for long-distance holography).

Units: SI metres internally, enforced by a single convention and validated at the API boundary only.

**R2. Invariant: every λ shares one spatial grid at sample and detector planes.** Implement the objective and tube lens as pupil ↔ image operators using per-λ MFT/CZT onto a common, user-specified image grid (in object-space units), with the pupil sampled in NA units. That keeps Chromatix's per-λ dx problem out of the design and makes incoherent λ sums a plain weighted reduction. Magnification is detector-side coordinate scaling (object-space simulation, as in Chromatix `Microscope` and DeepSTORM3D).

**R3. A static "planning" pass separate from the differentiable "execution" pass.** Given a system description, fidelity level, requested outputs (camera ROI, pixel pitch, z-stack) and *bounds* on learnable parameters (max NA, max defocus, λ range), the planner chooses grid sizes, padding (Chromatix's Fresnel-number helpers), propagators (from validity and cost tables, as in TorchOptics AUTO and hcipy), and chunk sizes. It emits an immutable `Plan` of Python ints, strings and callables, cached by configuration. Execution then never changes shapes, which keeps `torch.compile` and CUDA graphs viable and avoids non-differentiable shape logic.

**R4. A propagator registry with declared metadata.** Each propagator is a function `(field, z, *, plan) -> field` registered with `valid_regime(grid, λ, z, n)`, `cost(grid, out_grid)`, `output_grid(...)`, `supports={'z_grad', 'λ_grad', 'zoom', 'offaxis', 'vector'}`. Ship first:
1. BL-ASM with `H_exp`.
2. Fresnel TF/IR.
3. Chirp-z and MFT zoom.
4. SAS.
5. Debye/Richards–Wolf vectorial focusing with CZT.
6. Multislice BPM, then SSNP.

Keep ABCD/Collins and MBS for later.

**R5. The system is a typed DAG of stages, each with multiple fidelity backends behind one interface (the DeepLens pattern, generalised).** The canonical stages are `Illumination → SampleInteraction → Collection (pupil) → ImageFormation → Detector`, plus explicit combinators: `IncoherentSum(over=…)`, `CoherentSum` (for holography, iSCAT and interference with a reference) and `Split` (channels). Examples of backends:
- `SampleInteraction`: thin screen, Born/Rytov, multislice, SSNP, MBS.
- `ImageFormation`: coherent pupil FFT, PSF convolution, depth-stack PSF convolution, product-convolution for space-variant PSFs (deepinv, DeepLens), and the WOTF/transfer function (waveorder).

A **fidelity level is a mapping from stage to backend (plus plan knobs)**, not a flag scattered through the code. The same user-facing `Microscope` description compiles to different graphs.

**R6. Coherence as explicit reductions with interchangeable estimators.** Every "sum over incoherent modes" (emitters, illumination source points, wavelengths, polarization states, dipole orientations, time) is an `IncoherentSum` node that offers:
- (a) exact batched evaluation, with chunked accumulation and adjoint chunking for memory (the dO ABP idea);
- (b) an unbiased stochastic estimator (Chromatix-style random phases or mode subsampling);
- (c) an analytic shortcut when it applies: shift-invariant PSF/OTF convolution, WOTF/TCC for weak objects, or the Jones-matrix-on-Stokes contraction for polarization (hcipy).

Do **not** make a 4-D mutual intensity a first-class field type.

**R7. Sample representation is decoupled from sampling onto the field grid through a "rasterise" contract.** Samples (analytic primitives, meshes, voxel volumes, point clouds of emitters) expose `to_potential(grid3d, λ, band_limit)` and/or `fourier_transform(k)`. Point emitters bypass rasterisation entirely: their positions become pupil phase ramps, which is exact and differentiable (DeepSTORM3D). Analytic primitives should offer analytic Fourier transforms or soft signed-distance rasterisation, so size and shape parameters get gradients.

**R8. Parameters: PyTorch-native, uniformly addressable, distribution-aware.**
- Adopt TorchOptics' rule: `nn.Parameter` means learnable, anything else is a buffer.
- Add dLux-style dotted-path `get`/`set`/`select`, so DeepTrack features can override or sample any property.
- Put random data generation in deepinv-style `Generator`s that return parameter dicts, with *learnable distribution parameters* through reparameterisation (`μ + σ·ε`; relaxed or straight-through estimators for discrete choices).
- Physics operators stay deterministic given the parameters and a `torch.Generator` seed.

**R9. Differentiability rules baked into the core library** (and tested with gradcheck on float64/complex128):
- Soft or anti-aliased pupils and apertures.
- Safe square root and double-`where` at the evanescent cutoff.
- Intensity as `real² + imag²`.
- No parameter-dependent shapes.
- Caches keyed on parameter identity plus version and bypassed when gradients are required.
- Straight-through Poisson with a Gaussian surrogate gradient (a custom `autograd.Function`), with read noise, EM gain and quantisation offering surrogate gradients as well.
- Implicit differentiation for iterative solvers.
- Checkpointed or reversible multislice.
- TF32 disabled around MFT matmuls.

**R10. Make adjoints first-class where cheap.** Linear stages (propagation, pupil filtering, PSF and transfer-function convolution) expose `adjoint()` (hcipy `backward`, deepinv `A_adjoint`, prysm `*_adjoint`). This lets gradoscopy models be dropped into reconstruction and deconvolution (the waveorder-style triad) and enables memory-saving hand-written backward passes for multislice and SSNP.

**R11. Performance defaults.**
- complex64 by default, with complex128 for validation.
- Every independent quantity (z, λ, emitter, illumination angle, polarization state) is a broadcast batch dimension.
- Kernel LRU cache per (grid, λ, z, n, dtype, device), as in hcipy.
- rfft for real-valued intensity convolutions; next-fast FFT sizes.
- MFT when the output region is small.
- Chunked incoherent sums.
- Optional `torch.compile` [?: verify complex-number coverage in current Inductor] and DDP for batch scale-out.

**R12. Cross-validation harness.** Numerical reference tests should compare against:
- hcipy, POPPY and prysm (FFT/MFT Fraunhofer and Fresnel);
- Chromatix (ASM, SAS, CZT, multislice);
- analytic Airy and Richards–Wolf focal fields;
- Mie spheres (DeepTrack's `MieSphere`) for scattering fidelity;
- waveorder transfer functions for brightfield, DPC and polarization.

These tests are what make "fidelity levels" trustworthy: every fast backend should come with a documented error bound against the next-higher level.
