# 01 — DeepTrack2 in depth: what the gradoscopy engine must serve

Research brief for the gradoscopy architecture plan. Target: gradoscopy must be able to act as the simulation backend of DeepTrack2 (DT2).

**Legend.** [V] = verified in source, docs or PR during this research (URL given). [R] = from recollection of the DT2 codebase and papers, not re-verified here. [?] = uncertain or inferred. Line-level claims about `develop` come from `raw.githubusercontent.com` fetches. The fetch tool truncated `optical/optics.py` at about 47k characters, so `develop`'s `_create_volume` / `_merge_placed_volumes` bodies were not visible. For those, the 2.0.1 implementation is quoted instead and marked as such.

**Versions (as of 2026-09-24)** [V] <https://github.com/DeepTrackAI/DeepTrack2/releases>:
- **2.0.0 (2024-08-26).** Moved to PyTorch plus `deeplay` and removed TensorFlow. Pipelines stop returning `Image` by default. Adds `Sources`, the `dt.pytorch` submodule (Dataset, ToTensor), and the iSCAT, darkfield and improved holography modalities.
- **2.0.1 (2025-03-26).** Removed the last TensorFlow leftovers and the `pint<2` pin.
- **2.0.2 (2026-08-05).** Refactored core modules, expanded tests, and improved docs.
- **`develop`** (default branch, last push 2026-09-08). It has a new layout: `deeptrack/{features,properties,sequences,wrappers,elementwise,statistics,types,utils}.py`, `deeptrack/optical/{optics,scatterers,aberrations,noises,augmentations,holography,math}.py`, `deeptrack/backend/{core,_config,mie,polynomials,units,pint_definition}.py` plus `backend/array_api_compat_ext/torch/`, and `deeptrack/{pytorch,sources,deeplay,extras}/`. The legacy `image.py`/`Image` class is gone from `develop` [V] (directory listing <https://github.com/DeepTrackAI/DeepTrack2/tree/develop/deeptrack>).

---

## (a) Core abstractions

### DeepTrackNode: a lazily evaluated, cached DAG [V]
`deeptrack/backend/core.py` (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/backend/core.py>):

- **`DeepTrackDataObject`** holds `_data` and a `_valid` flag.
- **`DeepTrackDataDict`** maps `_ID` tuples of ints to `DeepTrackDataObject`. The key length is fixed by the first entry. Lookups use prefix matching: a longer `_ID` is trimmed, and a shorter one returns a slice of every matching entry. This is how one node caches a different value per replicate (see `Repeat`).
- **`DeepTrackNode(action)`** caches the result of `action(_ID=...)`. It keeps `_dependencies` and `_children` as `WeakSet`s, plus transitive `_all_dependencies` and `_all_children`.
  - `__call__(_ID)` returns the cached value if it is valid. Otherwise it runs `action` and stores the result.
  - `invalidate()` marks this node and every descendant invalid. `update()` resets data in all children. `new()` is `update()` followed by a call. `set_value()` stores a value and invalidates it if it changed.
  - Operators `+ - * / // < > <= >=` build new nodes lazily with `_create_node_with_operator`. `__getitem__` builds an indexing node.
- **Key fact:** the cache is invalidated only by explicit `update()`/`invalidate()` calls. It does not see tensor versions or parameter mutation. `core.py` has no torch or autograd logic at all [V].

### Property, PropertyDict, SequentialProperty [V]
`deeptrack/properties.py` (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/properties.py>). A `Property` is a `DeepTrackNode` whose action comes from a **sampling rule**, dispatched on type:

- **Constants** are returned unchanged. This covers numbers, strings, ndarrays and torch tensors, so an `nn.Parameter` passed as a property keeps its autograd identity.
- **Callables** are inspected with `get_kwarg_names()`. Only arguments whose names match sibling properties (or the special names `_ID`, `replicate_index`, `sequence_index`, `previous_value`, …) are passed in. **Dependencies are wired by keyword name**, for example `Sphere(radius=lambda: ..., z=lambda radius: 2*radius)`.
- **Lists, tuples, dicts and slices** are handled recursively. **Iterators** yield successive values and repeat the last one when exhausted. **Existing nodes** are linked directly.

`PropertyDict` (a subclass of both `DeepTrackNode` and `dict`) resolves properties in repeated passes until every dependency is satisfied. If a pass makes no progress it raises `ValueError`. Its action returns `{name: prop(_ID)}`.

`SequentialProperty` adds `sequence_length`, `sequence_index`, `previous_value` and `previous_values`, and splits the rule into `initial_sampling_rule` (step 0) and `sampling_rule` (later steps). `next_step()` advances the index and invalidates dependents.

### Feature [V]
`deeptrack/features.py` (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/features.py>).

**Construction.** `Feature.__init__(_input=None, **kwargs)`:
1. Captures the backend, dtypes and device from global config.
2. Registers itself as a `DeepTrackNode` whose action is `_action`.
3. Wraps every kwarg as a `Property` in `self.properties`, a `PropertyDict` that is a dependency of the feature.
4. Wraps `_input` in its own `DeepTrackNode`.
5. Creates a random-seed node.

**The subclass contract** is one method: `get(self, data, _ID=(), **kwargs)`. It receives the *sampled* property values as kwargs and returns output data.

**Evaluation:**
- `__call__(data_list=None, _ID=(), **kwargs)` (alias `resolve`) evaluates the feature. Keyword arguments override properties for that call and are pushed to dependencies by `propagate_data_to_dependencies`.
- `update()` invalidates. `new()` is update plus evaluate.
- `batch(batch_size)` just calls `.new()` repeatedly and zips the outputs. `__iter__`/`__next__` also call `.new()`.

**List semantics:**
- Inputs are normalised to lists by `_format_input`.
- `__distributed__ = True` (the default) calls `get` once per list element. `False` calls it once on the whole list.
- `__list_merge_strategy__` is either `MERGE_STRATEGY_OVERRIDE` (0, the default: output replaces input) or `MERGE_STRATEGY_APPEND` (1: output is appended to input). **Scatterers use APPEND**, which is why `particle_a >> particle_b` yields a list of two scatterers [R, consistent with the V docs].

**Backend and dtype methods:** `.torch(device=None, recursive=True)`, `.numpy()`, `.to(device)`, `.get_backend()`, and `.dtype(float=..., int=..., complex=..., bool=...)`. Chains do **not** auto-convert between backends, so mixing them raises errors ([V] <https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/tutorials/4-developers/DTDV421_backends.ipynb>).

**Seeding:** `seed()` seeds Python `random`, NumPy **and** torch, all as *global* RNGs.

**Operators** [V]:
- `>>` builds `Chain`; a bare callable is wrapped in `Lambda`.
- `&` / `__rand__` stacks outputs (`Stack`). This is the idiomatic "image & label" pairing.
- `^` builds `Repeat`.
- `+ - * / // **` chain `Add`, `Subtract`, `Multiply`, `Divide`, `FloorDivide`, `Power`. Comparisons chain `GreaterThan` and the other comparison features.

**Structural features in `features.py`** [V]: `Chain`/`Branch`, `DummyFeature`, `Value`, `Stack`, `Arguments` (shared, overridable pipeline arguments via `bind_arguments`), `Probability`, `Repeat`, `Combine` (evaluate several features on the same input and return a list), `Bind` (fix property values for a sub-feature; `BindResolve` is a deprecated alias, `BindUpdate` is deprecated), and `ConditionalSetProperty`/`ConditionalSetFeature` (both deprecated). Also `Slice`, `Lambda`, `Merge`, `OneOf`, `OneOfDict`, `LoadImage`, `AsType`, `Store`, `Squeeze`, `Unsqueeze`/`ExpandDims`, `MoveAxis`, `Transpose`/`Permute`, `OneHot` and `TakeProperties`.

**Repeat and `_ID`** [R]: `Repeat.get` loops `for n in range(N): image = self.feature(image, _ID=_ID + (n,), replicate_index=_ID + (n,))`. Each replicate gets its own cache slot through the extended `_ID`. `particle ^ N` therefore produces N independently sampled scatterers from one Feature object. The docs confirm that `^` means Repeat [V].

### Image (legacy) → Wrapper, ScatteredVolume, ScatteredField
- **≤ 2.0.1:** the `Image` class wrapped arrays with a list of per-feature property dicts. It implemented NEP-18 (`__array_function__`/`__array_ufunc__`) [R] and offered `append`, `merge_properties_from` and `get_property`. In 2.0 wrapping became opt-in (`store_properties(True)` sets `_wrap_array_with_image`) because it degraded performance. The docs warn that "Setting dt.Value value as an Image object is likely to lead to performance deterioration" [V] (2.0.1 `features.py`).
- **`develop` / 2.0.2** [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/wrappers.py>, PR #448 <https://github.com/DeepTrackAI/DeepTrack2/pull/448>). `Image` is gone. `wrappers.Wrapper` is a `@dataclass(array, properties: dict)` with arithmetic, comparison and logical operators that carry the left operand's properties. It does *not* implement `__torch_function__`/`__array_ufunc__`. Scatterers return `ScatteredVolume` or `ScatteredField` (subclasses of `Wrapper`, with `.position` and `.pos3d`).
- **How labels are made now:** by referencing the *property node* of an upstream feature in a parallel branch, not by reading metadata attached to the image. From DTGS121 [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/tutorials/1-getting-started/DTGS121_tracking_particle_cnn.ipynb>):
  ```python
  pipeline = brightfield_microscope(particle) & (particle.position / IMAGE_SIZE)
  ```
  This works because both branches read the same cached `_ID` slot of `particle.properties["position"]`.

### Sequences, Sources, PyTorch glue [V]
- **Sequences.** `Sequence(feature, sequence_length)` loops over steps and injects `sequence_index`/`sequence_length` into every `SequentialProperty`. `feature.to_sequential(position=lambda previous_value: ...)` replaces the deprecated `Sequential(...)`. Outputs are lists, transposed to a tuple of lists when the feature returns tuples (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/sequences.py>).
- **Sources.** `Source(a=[...], b=[...])` exposes each field as a `SourceDeepTrackNode` that depends on a `_current_index` node. Calling `source[i]()` activates item *i*, so `dt.Value(source.a)` resolves per item. Related pieces: `Product`, `Subset`, `Sources`/`Join`, `random_split`, and `sources/folder.py` (ImageFolder) (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/sources/base.py>).
- **`dt.pytorch.Dataset(pipeline, inputs=None, length=None, replace=False|True|p|callable, float_dtype="default")`.** `__getitem__` calls `pipeline.update()` then `pipeline(inputs[i])` and caches tuples of tensors. Generation is **per item**. `dt.pytorch.ToTensor(dtype, device, add_dim_to_number, permute_mode)` handles the channel-last to channel-first permute (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/pytorch/data.py>).

### Units and backend configuration [V]
- **Units.** `backend/units.py` uses pint with custom units: optical pixels `xpx/ypx/zpx` and simulation pixels `sxpx/sypx/szpx`. `create_context(xpixel, ypixel, zpixel, xscale, yscale, zscale)` redefines them. `get_active_scale()` returns the optical-to-simulation pixel ratio (the upscale), and `get_active_voxel_size()` returns metres. Each feature's `ConversionTable` converts properties (e.g. `position_unit="pixel"|"meter"`) before `get`. **Tensors with `requires_grad` bypass pint and are multiplied by a factor directly**, to keep the graph (added in PR #480).
- **Backends.** `backend/_config.py` has `config.set_backend("numpy"|"torch")`, `set_backend_torch()`, `set_device("cpu"|"cuda"|"mps"|torch.device)` and the context manager `with_backend(...)`. A module-level `xp` proxy routes calls to `array_api_compat.numpy` or `array_api_compat.torch`. `array_api_compat_ext/torch/random` supplies `xp.random` for torch. There is no CuPy or JAX backend, although the developer tutorial says `xp` "can extend to JAX and CuPy". NumPy is CPU-only (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/backend/_config.py>).

**The semantic core to preserve.** A DT pipeline is a *stochastic program*. Each `update()` resamples every property in the DAG once per `_ID` slot. Features consume the sampled scalars and arrays. Labels are just other nodes of the same DAG. The design is elegant for data generation. It is also inherently **per-sample, Python-interpreted, and global-state-driven**: backend, unit context and RNG are all global.

---

## (b) The optics module

### Optics and Microscope [V]
Source: <https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/optics.py>.

**`Optics.__init__` defaults:**
- `NA=0.7`, `wavelength=0.66e-6`, `magnification=10`, `resolution=1e-6` (camera pixel pitch; may be a 3-tuple to give z spacing), `refractive_index_medium=1.33`
- `padding=(10,10,10,10)`, `output_region=(0,0,128,128)`
- `pupil=None` (a Feature or array that modulates the pupil, i.e. aberrations), `illumination=None` (a Feature applied to the incident field), `upscale=1` (int or 3-tuple)

**Sample-plane voxel size** is `voxel_size = resolution / magnification`, broadcast to three components.

**Calling pattern.** `optics(sample)` returns `Microscope(sample, objective=optics)`, which is itself a Feature. `Microscope.get`:
1. Enters `u.context(create_context(*voxel_size, *upscale))`. All scatterers are therefore evaluated on an **upscaled simulation grid**, with `voxel_size`, `padding` and `output_region` scaled to match. These are pushed into the sample with `propagate_data_to_dependencies` [V].
2. Resolves the sample into a list of `ScatteredVolume` / `ScatteredField`. (In 2.0.1 the split used an `is_field` property.)
3. Converts each volume into a *contrast volume* with `objective.extract_contrast_volume(...)`:
   - Fluorescence: `intensity * prod(scale) * array`
   - Brightfield: `(refractive_index - n_medium) * array`
   - otherwise: `value`, with a warning that it is non-physical.
4. Merges the volumes with `_merge_placed_volumes`.
5. Calls `objective.get(volume, limits, fields, **props)`.
6. Downscales to detector pixels. Coherent modalities use `AveragePooling((ux,uy))` on the **intensity**. Fluorescence uses `SumPooling` normalised by `ux*uy*uz` [V].
7. Output is per sample, shape `(H, W, 1)` channel-last. **There is no batch dimension anywhere in optics** [V].

### Pupil and PSF [V]
`Optics._pupil(shape, NA, wavelength, refractive_index_medium, include_aberration, defocus, **kw)` returns a complex stack `(Z, H, W)`:

- The frequency grid is normalised by `R = NA/λ · voxel_size[:2]`, so `ρ² = W² + H²` (with a `+1e-8` offset in 2.0.1).
- The aperture is a hard disc: `P = (ρ² < 1) + 0j`.
- Defocus uses the **exact scalar angular-spectrum** axial wavenumber, restricted to the pupil:
  `z_shift = 2π·n_m/λ · Δz · sqrt(1 − (NA/n_m)²·ρ²)`, then `P_z = P · exp(i·defocus·z_shift)`, with defocus in units of z-voxels.
  The torch path clamps before the square root to stay differentiable (PR #480).
- The optional `pupil` feature (Zernike aberrations, `GaussianApodization`, or a learnable phase mask) multiplies `P` when `include_aberration=True`.
- **Absent:** the aplanatic apodization factor `cos(θ)^{±1/2}`, vectorial (Richards–Wolf) factors, dipole orientation, refractive-index mismatch (Gibson–Lanni / Hanser), and pupil caching. The pupil is **recomputed on every `get()` call** [V].

### Fluorescence (incoherent) [V]
1. The contrast volume is padded (`pad_image_to_fft`, now in `optical.math`).
2. Empty z-planes are skipped: `zero_plane = all(volume == 0, axis=(0,1))`.
3. For each remaining plane at defocus `z` (a linspace over `z_limits`), it computes `PSF_z = |IFFT2(fftshift(P_z))|²` and `OTF_z = FFT2(PSF_z)`, then `image += IFFT2(FFT2(slice_z) · OTF_z) / scale_z`. Padding is cropped afterwards.

This is the standard *3D incoherent imaging = Σ_z 2D convolution with a depth-dependent scalar PSF*. It is exact for a shift-invariant scalar PSF. Cost per sample is about 4 FFTs per non-empty z-plane, at the padded, upscaled size. `ScatteredField` inputs raise `TypeError`.

### Brightfield / Holography (coherent multislice) [V]
`Holography` is an alias of `Brightfield`. The code (quoted from `develop`):
```python
pupils = [_pupil(shape, defocus=[1],            include_aberration=False)[0],   # inter-slice step (1 voxel)
          _pupil(shape, defocus=[-z_limits[1]], include_aberration=True)[0],    # back-propagate to focus
          _pupil(shape, defocus=[0],            include_aberration=True)[0]]
pupil_step = fftshift(pupils[0])
light_in = fft2(self.illumination.resolve(ones(shape)))
K = 2π/λ · n_medium
for i, z in slices:
    light_in = light_in * pupil_step
    if zero_plane[i]: continue
    light = ifft2(light_in)
    light_out = light * exp(1j * ri_slice * voxel_size[-1] * K)   # ri_slice = n - n_medium
    light_in = fft2(light_out)
# add ScatteredField spectra (e.g. Mie) coherently, apply final (aberrated) pupil, hard mask |P|>0
out = ifft2(light_in_focus * fftshift(pupils[-1]))
return out if return_field else |out|²
```

This is a **split-step multislice beam-propagation method** with a thin phase screen per voxel slice. Each slice uses the phase `k₀·n_m·Δz·(n−n_m)`, which is the projection/first-order phase approximation. The model warns: "Brightfield imaging from ScatteredVolume assumes a weak-phase / projection approximation" [V].

Consequences:
- **The inter-slice propagator is NA-limited.** It includes the hard aperture `ρ<1`, so light scattered beyond the objective NA is discarded *between slices*. Physically it should be the free-space propagator, with only evanescent components cut. The effect is small for weak objects but wrong for thick or strongly scattering ones.
- There is no backscatter or reflection, no multiple scattering beyond what BPM captures, and absorption appears only through a complex `n`.
- **Illumination is a unit plane wave at normal incidence.** It can only be modulated by the `illumination` feature, e.g. `IlluminationGradient`, or noise on the field as in `illumination=dt.Gaussian(sigma=0.005) >> dt.Gaussian(sigma=0.005j)` in the LodeSTAR example. There is no condenser aperture and no Köhler partial coherence.
- **Polychromatic light** is not modelled. The DTGS106 tutorial averages images over sampled wavelengths itself [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/tutorials/1-getting-started/DTGS106_particle_image_modalities.ipynb>). The `wavelength` property is a scalar.

### Darkfield, ISCAT, IlluminationGradient [V]
- **`Darkfield(Brightfield)`** defaults to `illumination_angle=π/2`, which is forwarded to Mie scatterers. It returns `|E − 1|²`: the unscattered unit reference is subtracted from the coherent field. For volume scatterers it warns that the contrast is "non-physical and qualitative only".
- **`ISCAT(Brightfield)`** adds `illumination_angle=π` (backscatter), `amp_factor`, `input_polarization` and `output_polarization`. It **does not override `get`**. Per its docstring, these values are only forwarded to the scatterer and "do not relocate the computation". It is a Mie-angle convention, not a reflection-geometry model.
- **Issue #445** (open, 2025-12) reports inverted brightfield and darkfield contrast for a Mie bead compared with experiment (<https://github.com/DeepTrackAI/DeepTrack2/issues/445>).
- **`IlluminationGradient(gradient, constant, vmin, vmax)`** computes the amplitude `X·g₀ + Y·g₁ + constant`, clips it to `[vmin, vmax]`, and applies `E ← A·E/|E|`, which keeps the phase.

### Other optics-module features [V]
- `SampleToMasks(transformation_function, number_of_masks, output_region, merge_method)` rasterises the placed scatterers into label masks. It was made torch-compatible in PR #448.
- `NonOverlapping(feature, min_distance, max_attempts, max_iters)` uses rejection resampling to enforce minimum separation.

### Holography helpers [V]
From <https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/holography.py>:
- `get_propagation_matrix(shape, to_z, pixel_size, wavelength, dx=0, dy=0)` is an angular-spectrum propagator with a lateral shift. It supports NumPy and torch.
- `Rescale`, `FourierTransform`, `InverseFourierTransform` and `FourierTransformTransformation(Tz, Tzinv, i)` are **NumPy-only**. They are used as *z-refocusing augmentations* of complex fields, e.g. in the LodeSTAR holography examples.

---

## (c) Scatterers
Source: <https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/scatterers.py>.

### Base class [V]
`Scatterer(position=(32,32), z=0, value=1, position_unit="pixel", upsample=1, voxel_size, pixel_size, ...)`. Subclasses are split into `VolumeScatterer`, which returns a `ScatteredVolume` occupancy array, and `FieldScatterer`, which returns a `ScatteredField` complex field. `FieldScatterer` ignores `upsample` and warns.

For a `VolumeScatterer` with `upsample>1`, `get()` runs with `voxel_size/upsample`. The result is then **average-pooled back** (`_antialias_volume`), giving soft, partial-volume edges.

### Geometry [V]
- **`PointParticle`** is a `(1,1,1)` array of value `prod(scale)`, so integrated intensity does not change with upscale. `upsample` is stripped.
- **`Ellipse(radius, rotation, transpose)`** is a 2D mask `Xt²/r₀² + Yt²/r₁² < 1` on a grid of `±ceil(max r / min voxel)`, expanded to `(2R, 2R, 1)`. It is scaled by the axial voxel size for fluorescence.
- **`Sphere(radius)`** is `X+Y+Z ≤ 1` on normalised squared coordinates, with shape `(2Rx, 2Ry, 2Rz)`.
- **`Ellipsoid(radius, rotation)`** uses XYZ Euler rotation, then `(XR/r₀)² + (YR/r₁)² + (ZR/r₂)² ≤ 1`.

**All are hard binary masks.** Their gradient with respect to `radius`, shape or rotation is **zero almost everywhere**. The `upsample` antialiasing gives nicer images but gradients stay piecewise constant.

**Custom scatterers** subclass `VolumeScatterer` and return a boolean or float mask. Examples: the DTGS171 sphere-from-meshgrid tutorial, and DTGS172, which builds `Bacillus` (a cylinder with hemispherical caps and optional bending) and `Streptobacillus` (chains) for `dt.Fluorescence()` [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/tutorials/1-getting-started/DTGS172_bacteria.ipynb>).

### Placement and merging
In **2.0.1** [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/2.0.1/deeptrack/optics.py>), `_create_volume`:
1. Chooses the voxel value: `intensity × (sx·sy/sz)`, else `(n − n_medium)`, else `value`.
2. Pads each scatterer by 2 voxels.
3. Applies **sub-pixel x,y placement** by convolving every z-plane with a 3×3 bilinear kernel `[[0,0,0],[0,(1−x)(1−y),(1−x)y],[0,x(1−y),xy]]` using `scipy.ndimage.convolve`, which is not torch.
4. **Floors z**, with no sub-voxel z interpolation.
5. Grows a bounding-box volume on demand (reallocating and copying).
6. **Adds** overlapping scatterers (`+=`). There is no occupancy or union logic, so two overlapping spheres give `2·(n−n_m)` in the overlap.

In **`develop`** [V at PR level]:
- PR #449 ("optics with torch", merged 2026-02-04, <https://github.com/DeepTrackAI/DeepTrack2/pull/449>) "replac[ed] scipy convolution operations with backend-agnostic implementations" and made "subpixel particle placement … preserve torch tensor properties".
- PR #464 (<https://github.com/DeepTrackAI/DeepTrack2/pull/464>) "Enhanced subpixel placement accuracy", fixed fluorescence normalisation, refined upscale semantics, and introduced `_merge_placed_volumes`, which handles multiple scatterers with different intensities or refractive indices.

[?] Whether `develop` now interpolates in z, and whether merging is still additive, could not be confirmed (the file was truncated). Assume additive.

### Mie family [V]
`MieScatterer(FieldScatterer)` parameters:
- `coefficients` (a factory giving `L → (aₙ, bₙ)`)
- `input_polarization=0` (angle, `"circular"`, or `None` for unpolarised), `output_polarization=0` (analyser)
- `offset_z="auto"`, `collection_angle="auto"` (derived from NA), `L="auto"`
- `refractive_index_medium`, `wavelength`, `NA`, `padding`, `output_region`, `polarization_angle`
- `working_distance=1e6`, `position_objective=(0,0)`, `return_fft=False`, `coherence_length=None`, `illumination_angle=0`, `amp_factor=1`, `phase_shift_correction=False`
- `mode="geometric"|"hybrid"`, `pupil`

Subclasses:
- `MieSphere(radius=1e-6, refractive_index=1.45)` uses `mie.coefficients`.
- `MieStratifiedSphere(radius=(…), refractive_index=(…))` uses `mie.stratified_coefficients`, a determinant form over 2·layers matrices, and requires monotonic radii.

**The `"auto"` L** is the Wiscombe criterion `L = ⌊x + 4x^{1/3} + 1⌋`, with `x = 2πa/λ`.

**Geometric mode:**
1. Evaluate the far field on a virtual plane at `offset_z` in front of the particle. For each pixel this gives the distance `R`, `cos θ` relative to the illumination direction, and the azimuth `φ`.
2. Compute `S₁, S₂` from `mie.harmonics` (π/τ recurrences).
3. Weight them by polarisation coefficients (`S₁·sin(φ+pol)`, `S₂·cos(φ+pol)`, 0.5/0.5 for circular or unpolarised, with an optional analyser projection). This yields a **scalar** field `E = −i/(kR)·e^{ikR}·(S₂c₂ + S₁c₁)/amp_factor`.
4. Mask by collection angle, FFT, multiply by the angular-spectrum propagator back to the particle plane (`·e^{−ik·offset_z}`), and optionally apply a Gaussian `coherence_length` damping and the pupil.
5. IFFT, or return the FFT with `return_fft=True`, which is what Brightfield consumes.

**Hybrid mode** maps `S₁/S₂` directly onto spatial frequencies (`sinθ = |q|/k`, cut at `sinθ<1`). It is less field-of-view sensitive.

**Differentiability** (PR #480, merged 2026-06, <https://github.com/DeepTrackAI/DeepTrack2/pull/480>) [V]. Riccati–Bessel functions for torch are computed "using trigonometric functions and recurrence relations", because `torch.special` Bessel functions lack autograd "as of version 2.11.0". The PR reports as learnable: position, radius, refractive index, NA, magnification, wavelength, resolution, n_medium and Zernike coefficients. It also removes unstable `arccos` derivatives and `nan_to_num` in the pupil. NumPy paths still use `scipy.special.jv/yv` [V] (<https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/backend/polynomials.py>).

[?] Numerical caveat (my own expertise, not verified in the code): *upward* recurrence for spherical Bessel `jₙ(x)` is unstable once `n > x`, and Mie codes normally use downward recurrence or the logarithmic-derivative `Dₙ(mx)`. The fetched summary shows the form `(2n+1)/x·f_n − f_{n−1}`. If that is applied upward to `ψₙ` at large `n` or small `x`, precision loss is likely for small particles with large `L`, or for absorbing `m`. This needs testing.

**Mie fields are computed per particle over the whole (padded) output region** and summed coherently. There is no inter-particle multiple scattering and no near-field or evanescent coupling to other objects.

---

## (d) Noise, aberrations, augmentations, math

**Noise** (`optical/noises.py`) [V]:
- `Background`/`Offset(offset)`, `Gaussian(mu=0, sigma=1)`, `ComplexGaussian(mu, sigma)`
- `Poisson(snr=100, background=0, max_val=1e8)`, implemented as `rescale = (snr/peak)²` clipped to `[1e-10, max_val]`, then `poisson(image·rescale)/rescale` with `torch.poisson` on torch. This is **non-differentiable**. PR #525 "poisson noise fix" (2026-09) is recent; its content was not checked.
- Noise features can be placed on the *field* or the *sample*, not only the image. Examples are `optics(particle >> noise)` in DTEx203 and noise inside `illumination=`.
- **There is no camera model**: no QE, gain/EM gain, read-noise map, ADC quantisation, saturation, pixel crosstalk or exposure-time motion blur.

**Aberrations** (`optical/aberrations.py`) [V]:
- `Aberration` subclasses receive the pupil coordinates `rho` (normalised to the aperture) and `theta`.
- `Zernike(n, m, coefficient)` computes the standard radial polynomial with normalisation `√(2n+2)` for `m≠0` and `√(n+1)` for `m=0`, i.e. RMS-normalised (Noll-style) with explicit `(n, m)` and no Noll index. The phase is applied as `exp(i·Z)`, so coefficients are in **radians**.
- Named terms: `Piston`, `VerticalTilt`, `HorizontalTilt`, `ObliqueAstigmatism`, `Defocus`, `Astigmatism`, `ObliqueTrefoil`, `VerticalComa`, `HorizontalComa`, `Trefoil`, `SphericalAberration`, plus `GaussianApodization(sigma, offset)`.
- A user pupil may be any Feature. In DTEx252, `LearnablePhaseMask(nn.Module, dt.Aberration)` holds `nn.Parameter(phase)` and returns `pupil·exp(iφ)` [V].

**Augmentations** (`optical/augmentations.py`) [V]:
- `Augmentation(time_consistent)`, `Reuse(feature, uses, storage)` (reuses expensive simulations N times with different augmentations)
- `FlipLR`, `FlipUD`, `FlipDiagonal` (issue #416: it assumes `(X, Y, …)` layout), `Affine(scale, translate, rotate, shear, order, cval, mode)`, `ElasticTransformation(alpha, sigma, …)`
- `Crop`, `CropToMultiplesOf`, `CropTight`, `Pad`, `PadToMultiplesOf`
- Label consistency comes from `_update_properties`, which mirrors, transforms or shifts positions and output regions.
- Torch paths exist for flips, `Affine` and elastic transforms (via `grid_sample`) and `Pad`. The NumPy paths use `scipy.ndimage`.

**Math** (`optical/math.py`) [V]:
- `Average`, `Clip`, `NormalizeMinMax`, `NormalizeStandard`, `NormalizeQuantile`
- `AverageBlur`, `GaussianBlur`, `MedianBlur`, `AveragePooling`, `SumPooling`, `MaxPooling`, `MinPooling`, `MedianPooling`, `Resize` (OpenCV, NumPy-only), `BlurCV2` and `BilateralBlur` (OpenCV only)
- `pad_image_to_fft`
- `elementwise.py` exposes `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, … as Features, and `statistics.py` exposes the reducers `Sum`, `Mean`, `Median`, `Std`, … [V docs index].

---

## (e) Torch and backend work; deeplay

**Timeline:**
- **2.0.0 (2024-08).** "Migration to PyTorch": TensorFlow removed, `dt.pytorch.Dataset`/`ToTensor`, deeplay for models. **Simulation was still NumPy.**
- **2025.** The `xp` proxy and `array_api_compat` backend (`_config.py`) arrive. Per-feature `.torch()`/`.numpy()`/`.to()`/`.dtype()` appear, together with a "TODOs for Finalization" checklist (issue #348, <https://github.com/DeepTrackAI/DeepTrack2/issues/348>) asking for PyTorch compatibility, docstrings, tests and tutorials across 24 modules (144 items, all open).
- **Optics port to torch:** PR #415 ("fluorescence compatible with torch", 2025-08), #448 (Image removal, merged 2026-01-22), #449 (torch optics, 2026-02-04), #464 (revised optics and scatterers, 2026-04-20), #474 (numpy device fix), #478 (bf-mie fix) and #480 (Mie autograd, 2026-06) [V] (<https://github.com/DeepTrackAI/DeepTrack2/pulls?q=is%3Apr+optics>).
- **Differentiable demos:**
  - `DTGS161_torch_fitting`: fits a *custom* 2D Gaussian feature, not the optics. It notes that `.update()` is needed each step.
  - `DTDV431_mie_position_optimization`: Brightfield + MieSphere, fitting only (x, y), with Adam at lr 0.2 for 50 steps. It says: "DeepTrack caches feature outputs, so update() is needed after each optimizer step."
  - `DTEx252_phase_mask_optimization`: jointly trains a Fluorescence pupil phase mask and a CNN for 3D localisation, DeepSTORM3D-style. It builds each batch with a **Python loop of `pip.update(); pip.resolve()`** and replaces Poisson with a Gaussian approximation for differentiability.
  - PR #512 renames the Mie tutorial to `DTGS162_differentiable_optimization`, adding radius fitting.
  - [V] <https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/tutorials/2-examples/DTEx252_phase_mask_optimization.ipynb>, <https://github.com/DeepTrackAI/DeepTrack2/pull/512>
- **Public positioning.** The SPIE Optics+Photonics 2026 talk 14198-17 (Lech, Granfors, Huang, Midtvedt, Pineda, Bachimanchi, Manzo, Volpe) advertises "a PyTorch backend, enabling GPU acceleration and backpropagation through the optical system" [V] (<https://spie.org/optics-photonics/presentation/Model-free-in-situ-training-of-diffractive-optical-processors-via/14198-17>). Calibration is otherwise done with Optuna in DTGS127: TPE over the aberration type (categorical), coefficient, radius and z, 230+ trials of RMSE against an experimental image. This is exactly the kind of task a differentiable engine should make cheap.

**deeplay** [V] (<https://github.com/DeepTrackAI/deeplay>, releases up to 0.1.4, 2026-01-19):
- PyTorch plus Lightning, with a **configure → build/create** pattern: `DeeplayModule`, `Layer`, `LayerList`, `Sequential`, blocks, components, models.
- Applications: `Regressor`, `Classifier`, `LodeSTAR`, `MAGIK`, GAN/CycleGAN, VAE, …
- DT2 pipelines are passed directly as `train_data` to `model.fit(...)`, e.g. `dl.Regressor(net).fit(train_data=pipeline, batch_size=32, steps_per_epoch=100)` in DTGS121. deeplay therefore consumes DT features as *infinite sample generators*.
- The configure/build pattern is a precedent the gradoscopy team already knows: declarative config that can be mutated before materialisation.

---

## (f) Documented applications and what they use

| Application (source) | Optics | Scatterers / sample | Labels / notes |
|---|---|---|---|
| Single-particle tracking (DTGS121, DTEx212; Optica 2019 DeepTrack 1.0) | `Brightfield(NA=0.9, λ=680 nm, M=10, res=1 µm, padding 32)` | `MieSphere(n=1.58, r=0.5 µm, position=lambda: ...)` | `particle.position` via `&`; CNN via deeplay [V] |
| Multi-particle tracking (DTEx213, DTGS131) | `Fluorescence` | point-like QDs, `particle ^ N` | `SampleToMasks`; U-Net; sim-to-real [V/partial] |
| Particle sizing / holographic characterisation (DTEx203; ACS Nano 15, 2240 (2021), <https://pubs.acs.org/doi/10.1021/acsnano.0c06902>) | `Brightfield(NA=1.3, λ=635 nm)` complex field, Zernike coma | `MieSphere` r 100–400 nm, n 1.37–1.67; `ComplexGaussian` noise on the field; re/im channels | regress r, n [V] |
| 3D tracking in inline holography (DTEx205) | `Brightfield(NA=1.3, λ=633 nm, M=1, res=0.345 µm)` + coma | 2–7 `MieSphere`, z 2–30 µm, r 190–290 nm | Poisson SNR 7–17, `IlluminationGradient`; U-Net [V] |
| LodeSTAR (Nat Commun 13, 7492 (2022)); DTEx231A–J | mostly *experimental single crop + augmentations*; simulated variants use `Brightfield(return_field=True)` + MieSphere, and holographic z via `get_propagation_matrix` / `FourierTransformTransformation` | — | equivariance; mass measurement [V] |
| MAGIK (Nat Mach Intell 5, 71 (2023)); DTEx241 | none (graphs from detections) | — | [V] |
| Aberration characterisation (DTGS126 CNN, DTGS127 Optuna) | `Fluorescence(M=10, λ=660 nm)` + Zernike | `Sphere` | inverse problem [V] |
| PSF engineering (DTEx252) | `Fluorescence(NA=1.45, n=1.33, pupil=LearnablePhaseMask)` | 25–50 points in a 121×121×30 volume | end-to-end optics + CNN [V] |
| Bacteria morphology (DTGS172) | `Fluorescence()` | custom `VolumeScatterer` capsules and chains | [V] |
| Brightfield particle classification (DTGS141) | Brightfield | [?] Mie / volume | [not checked] |
| Cell counting (DTEx215) | none (BBBC039 real data) | — | U-Net density maps [V] |
| Virtual staining (Biophys. Rev. 2, 031401 (2021)) | none (experimental BF → fluorescence cGAN) | — | [V] |
| Neural tissue segmentation (DTEx251) | procedural ssTEM-like images (EM, out of scope) | — | [V] |
| Particle image modalities (DTGS106) | Fluorescence, Brightfield (λ averaged for white light), Darkfield, ISCAT | `Sphere(intensity=1)` / `Sphere(refractive_index=1.42)` | [V] |

**Takeaway.** Two workhorses carry most of DT's simulation value:
1. **Coherent field imaging of Mie spheres**: brightfield, holography, darkfield and iSCAT, with complex-field outputs, aberrations and z-refocusing.
2. **Incoherent fluorescence of points, spheres, ellipses and custom masks**, with Poisson noise and masks as labels.

A third, large group uses DT only for **data plumbing**: Sources, augmentations, `&`-pairing and Dataset. Cell-scale phase objects (thick RI volumes) exist but are thin.

---

## (g) Known limitations and pain points

**Performance and execution model**
- **Per-sample, Python-level graph evaluation.** No batch dimension exists in optics [V]. `Feature.batch` and `Dataset.__getitem__` call `update()`/`new()` per sample [V]. On a GPU this means many small kernel launches, many host–device syncs, and poor occupancy. Scale-out relies on DataLoader worker processes. [?] Lambda-based sampling rules are hard to pickle under Windows or macOS `spawn`.
- **No reuse of expensive invariants.** Pupils, PSFs and OTFs are recomputed on every `get()` [V]. Fluorescence repeats about 4 FFTs per occupied z-plane. The volume bounding box is reallocated as scatterers are added (2.0.1 [V]).
- **Memory.** The contrast volume spans the bounding box of all scatterers plus padding, at *upscaled* resolution and complex dtype. Example: 128² output, upscale 4, 64 z-slices, complex128 gives 512²×256×16 B ≈ 1.07 GB before autograd. Under autograd the multislice loop stores every slice field, so memory is O(N_z·N_xy). There is no checkpointing.
- **Mie cost** scales as O(N_particles · H · W · L) at full padded field size per particle.

**Differentiability gaps (even on `develop`)**
- Hard-edged volume masks give zero gradients for radius, shape and orientation. Floored z in 2.0.1 gives zero z-gradient for volume scatterers [V].
- Sampling rules use `np.random`/`random` in user lambdas. There are no reparameterised distributions, so **distribution parameters are not learnable**, and discrete choices (`^ N`, `OneOf`, `Probability`) are not either.
- `Poisson` is non-differentiable. `MedianBlur`, `NormalizeQuantile` and max/min pooling are too, and holography FT augmentations, `Resize` and `BlurCV2` are NumPy-only [V].
- **Cache semantics fight autograd.** Changing an `nn.Parameter` does not invalidate caches, so users must call `update()` every step [V]. Bug #352 shows stale caches leaking across pipelines (<https://github.com/DeepTrackAI/DeepTrack2/issues/352>). Issue #349 covers `previous()` semantics.
- Units go through pint. Tensors are special-cased only when `requires_grad` [V].

**Global state**
- The backend is global and per-feature. Chains cannot mix backends, and there is no auto-conversion [V].
- The pint unit context (upscale) is global and entered with a `with`. [?] It is not thread-safe.
- `seed()` seeds global `random`, NumPy and torch [V].

**Physics shortcuts** (all [V] unless marked)
- Scalar optics throughout. There is no aplanatic apodization, no vectorial high-NA PSF, no dipole emission pattern, no index-mismatch or coverslip aberration, and no z-dependent spherical aberration beyond what Zernike adds by hand.
- Brightfield is a projection/weak-phase multislice with an **NA-truncated inter-slice propagator**. Illumination is a single coherent plane wave with no condenser NA or partial coherence, so brightfield of thick samples is really "coherent holography". There is no phase-contrast or DIC model, only via custom pupils [?].
- Mie fields are single-particle and scalar-projected. Coherent summation has no inter-particle coupling. ISCAT and Darkfield are parameter conventions on top of Brightfield, not geometry models. Contrast signs are disputed (#445).
- Overlapping volume scatterers **add** their contrast (2.0.1).
- Sub-pixel placement is bilinear, which low-pass filters and shifts power slightly. There is no sub-voxel z (2.0.1).
- Wavelength is scalar, so there are no spectra. There is no camera or detector model beyond Poisson and Gaussian. There is no exposure integration or motion blur for sequences.

**API and ergonomics**
- Outputs are channel-last `(H, W, 1)`, which requires `ToTensor(permute_mode=...)` [V].
- The `_ID` machinery, `__distributed__` and merge strategies are powerful but opaque, and the source of subtle bugs.
- Some notebooks still reference removed TensorFlow-era APIs (`dt.models.lodestar.LodeSTARGenerator`, `dt.models.gnns.MAGIK`) [V], and issue #389 reports the missing `deeptrack.generators`. Documentation has been lagging code.

---

## Requirements implied for the new engine

### Parity: must-have (to replace DT's optics without regressions)
1. **Optical system description.** NA, wavelength, magnification, camera pixel (`resolution`), `n_medium`, `padding`, `output_region` and `upscale`. The same derived `voxel_size = resolution/magnification` convention should be available.
2. **Pupil model.** A hard aperture, an angular-spectrum defocus phase, a pluggable pupil modifier (Zernike with the `(n, m)` API, Noll/ANSI indices and radians, `GaussianApodization`, an arbitrary learnable phase/amplitude map), and an illumination-field modifier (`IlluminationGradient`, noise on the field).
3. **Incoherent fluorescence**: 3D emission volume → Σ_z of 2D convolutions with depth-dependent PSFs, plus exact point-emitter handling that does not voxelise points.
4. **Coherent brightfield/holography**: RI volume → multislice. Also a complex field output (`return_field`), darkfield (`|E−E₀|²`), and an iSCAT-style reference-interference mode.
5. **Scatterers**: `PointParticle`, `Ellipse`, `Sphere`, `Ellipsoid` (rotations), custom mask/volume scatterers, `MieSphere`, `MieStratifiedSphere` (polarisation, collection angle, `coherence_length`), with `intensity` / `refractive_index` / `value` semantics and upsampled antialiasing.
6. **Composition of many scatterers** (the `^ N` equivalent), `NonOverlapping`, and `SampleToMasks`-style label rendering from the *same* scene.
7. **Noise**: Background, Gaussian, ComplexGaussian, Poisson(snr, background).
8. **Detector integration**: average/sum pooling from the upscaled grid to camera pixels.
9. **Propagation utilities**: `get_propagation_matrix` equivalent and complex-field refocusing for augmentation.
10. **Sequences**: time-evolving parameters rendered per frame.

### Should-have (the reason to build a new engine)
- A leading batch dimension everywhere, vectorised over samples and particles.
- Cached pupils and OTFs keyed by parameter values or versions.
- `complex64` by default.
- Differentiable soft or analytic geometry: SDF/coverage rasterisation, sub-voxel xyz via Fourier shift or trilinear splatting.
- Reparameterised samplers with an explicit `torch.Generator`.
- A camera model: QE, gain, EM-gain, read-noise, offset, quantisation with straight-through, saturation.
- Polychromatic illumination and emission.
- A condenser NA and partial coherence.
- Index mismatch and a vectorial PSF as higher-fidelity tiers.
- Free-space inter-slice propagation, not NA-truncated.
- Gradient checkpointing for multislice.
- Stable Mie recurrences (log-derivative or downward).
- Multi-particle Mie coupling via T-matrix as an optional top tier.

### API conventions to keep vs break
**Keep:**
- *Sampling rules as properties.* Constants, callables with dependency injection by name, and tensors, with `nn.Parameter` passing through.
- `optics(sample)` as the imaging call.
- `>>` for composition, `&` for image/label pairing, `^` for replication.
- `position_unit="pixel"|"meter"`, with SI metres as the internal unit.
- Names and semantics of the main classes (`Fluorescence`, `Brightfield`, `MieSphere`, `Zernike`, …) so the adapter is thin.
- `upscale` as the knob for oversampling fidelity.
- The contrast conventions `intensity`, `refractive_index` and `value` (with a warning).

**Break:**
- Channel-last `(H,W,1)` becomes `(B, C, H, W)`.
- The global backend, NumPy support and `xp` go: the engine is torch-only.
- Global pint contexts on the hot path are replaced by explicit `Grid`/`Sampling` objects passed down.
- Implicit output caching inside the rendering kernel goes: the engine should be a *pure function* of tensors.
- The `Image`/`Wrapper` metadata carriage is replaced by a typed `Scene` that labels are derived from.
- Global RNG seeding is replaced by explicit generators.
- Additive merging of overlaps becomes a defined composition rule (sum for emission; max/union or ordered override for RI, selectable).

### Integration strategy options
**A. Engine as DT's optics backend (deep integration).** DT keeps Feature/Property/Sources. `dt.optical.*` classes are reimplemented as thin Features whose `get()` builds engine objects and calls `engine.render`.
- Pros: users see no change and existing notebooks run.
- Cons: DT's per-sample `update()` loop still bottlenecks throughput, and caching vs autograd still clashes. The engine's batch dimension is wasted unless DT gains a batched resolve.

**B. Standalone engine plus a thin DT adapter (recommended).** gradoscopy exposes:
- a typed, batched `Scene` (objects with tensor parameters of shape `(B, N, …)` plus presence masks),
- `Optics`/`Illumination`/`Detector` modules (as `nn.Module`s),
- a fidelity-selectable `render(scene, system, fidelity=...) -> (B, C, H, W)`.

A `gradoscopy.integrations.deeptrack` module then provides:
- `EngineMicroscope(dt.Feature)` with `__distributed__=False`. Its `get(scatterers, **optics_props)` converts the list of `ScatteredVolume`/`ScatteredField`, or better the *unrendered properties* of DT scatterers, into a `Scene` of batch size 1, renders it, and returns `(H,W,1)` to stay compatible.
- `dt`-compatible class names (`Brightfield`, `Fluorescence`, `MieSphere`, `Sphere`, …) that subclass DT Features but hold *parameter descriptors* instead of computing masks. Rasterisation moves into the engine, where it can be differentiable and batched.
- A **batched resolve helper**: resolve the DT property graph B times, which is cheap because it is scalar Python work, stack the results into a `Scene` with batch B, and render once on the GPU. This keeps DT's stochastic-program UX and removes the per-sample GPU cost. It should be exposed as `dt.pytorch.Dataset`-compatible `BatchedDataset(pipeline, batch_size)`.

**C. Replace DT's core with the engine's own graph.** This is not advisable now: DT's property graph *is* its user value, and deeplay consumes it.

**How DT Features wrap engine objects** (concrete):
- A DT scatterer's `get()` returns a lightweight `EngineObject(kind="sphere", params={radius, n, position(3), ...}, properties=...)`, a subclass of `Wrapper`, so `SampleToMasks` and `particle.position` still work.
- `EngineMicroscope` groups the objects by kind and pads them to N_max with a presence mask. It builds the `Scene`, calls `system.render(scene, fidelity)`, and returns the image.
- Learnable tensors flow through unchanged because properties already pass tensors through as constants.
- Caching is removed as an issue by having the adapter call `update()` internally when any leaf tensor's `_version` changed, or by documenting `pipeline.update()()` as the training idiom.

---

## Implications for gradoscopy's architecture

1. **Split "what to simulate" from "how to render", and own only the second.** DT's lasting asset is the stochastic property DAG: dependency injection by name, `_ID` replicates, sequences and Sources. Its liability is doing the physics inside that per-sample DAG. gradoscopy should be a pure-functional, batched torch renderer, `render(scene: Scene[B], system: ImagingSystem, fidelity: Fidelity) -> Tensor[B,C,H,W]`, with no global state, no hidden caches and no NumPy. The DT adapter is the only place where the DAG meets the tensors.

2. **Make the Scene the single source of truth for images *and* labels.** DT today derives labels by re-reading upstream property nodes, and masks via `SampleToMasks`. gradoscopy's `Scene` should hold typed objects (`PointEmitter`, `Sphere`, `Ellipsoid`, `Capsule`/SDF, `Voxels`, `Mesh`, `MieSphere`, `LayeredSphere`) with batched tensor parameters and a `present` mask. Label renderers (positions, masks, distance maps, graphs) should be functions of the same Scene. This keeps augmentation/label consistency by construction and replaces the `_update_properties` machinery.

3. **Lower objects to intermediate representations chosen by fidelity.** A good ladder:
   - **L0**: analytic PSF splatting (Gaussian or tabulated PSF per point; projected analytic phase for spheres).
   - **L1**: 2D thin-sample projection with the DT-parity scalar pupil.
   - **L2**: 3D volume, i.e. incoherent Σ_z convolution for fluorescence and multislice BPM with a *free-space* inter-slice propagator for phase.
   - **L3**: vectorial Richards–Wolf PSFs with index mismatch, Mie/T-matrix fields, and partial-coherence Köhler sums.

   DT parity sits at **L1–L2 plus Mie**. The fidelity policy must be a declarative input (e.g. `Fidelity(optics="scalar", sample="volume", coherence="coherent")`), not a different set of classes, so the same user description renders at any tier.

4. **Differentiable geometry is non-negotiable.**
   - Replace hard masks with SDF-based coverage, e.g. `sigmoid(−sdf/ε)` or analytic partial-volume, with ε tied to the voxel size.
   - Give sub-voxel placement in xyz via trilinear splatting, or exact Fourier shifts for band-limited objects.
   - Render point emitters *off-grid*, as analytic PSF evaluation or Fourier-domain phase ramps, so position gradients are exact and upscale-independent.

   This removes the zero-gradient failure mode of all DT volume scatterers.

5. **Pupil and ImagingSystem as first-class, cached `nn.Module`s.**
   - `Pupil` = aperture ⊗ apodization ⊗ aberration (Zernike, ANSI/Noll, radians, DT `(n, m)` compatible) ⊗ user phase/amplitude map ⊗ defocus kernel.
   - Precompute the frequency grids and angular-spectrum kernels per `(grid, λ, n, NA)`, and memoise per forward pass. Never recompute per sample, and reuse across the batch.
   - Keep the **exact DT-parity pupil** as a "legacy" mode for golden-image regression tests against DT2 2.0.2.

6. **Batch and particle axes everywhere; memory discipline.**
   - `complex64` by default, with an opt-in `float64`/`complex128` mode for Mie validation.
   - Bounding-box volumes should be allocated once per batch at a fixed size, so there are no dynamic reallocations.
   - `torch.utils.checkpoint` over multislice blocks.
   - Mie evaluated on a per-particle cropped support where possible, with FFT-domain summation.

   Target: one `render` call per batch with no Python loop over samples.

7. **Randomness that can be learned.** Provide `gradoscopy.dist` with reparameterised distributions (Normal, LogNormal, Uniform via transforms), an explicit `torch.Generator`, and fixed N_max particles with Bernoulli or Gumbel-softmax presence for learnable counts. Add score-function (REINFORCE) hooks for truly discrete choices such as `OneOf`. DT lambdas can still produce values; the adapter simply stops tracking gradients through them.

8. **Detector as a module with differentiable surrogates.** Use a Poisson→Gaussian (Anscombe-free) reparameterisation or straight-through estimators for sampling, plus gain, offset, read noise, QE, quantisation (STE) and saturation. Integrate pixels from the upscaled grid by area-sum. DTEx252 had to replace Poisson with a Gaussian ad hoc; make that a named, documented mode.

9. **Adapter first, not last.** Ship `gradoscopy.integrations.deeptrack` early, with class names mirroring DT (`Brightfield`, `Fluorescence`, `Darkfield`, `ISCAT`, `MieSphere`, `Sphere`, `Ellipsoid`, `PointParticle`, `Zernike`, `Poisson`), plus a `BatchedDataset` that resolves the DT DAG B times and renders once. Parity tests should reproduce DTGS121, DTEx203, DTEx205, DTEx252 and DTGS106 outputs within tolerance.

10. **Fix known DT physics shortcuts at the higher tiers and document the approximations.**
    - Free-space (not NA-truncated) inter-slice propagation.
    - A defined overlap rule (emission sums; RI via ordered override or max).
    - Real darkfield/iSCAT geometry: annular or oblique illumination, and a reflected reference with its phase.
    - Condenser NA / partial coherence and polychromatic spectra.
    - Stable Mie recurrences.

    Each renderer should declare its validity regime (thin/weak object, NA, coherence) so the fidelity policy can warn, as DT's weak-phase warning does today, but machine-readably.
