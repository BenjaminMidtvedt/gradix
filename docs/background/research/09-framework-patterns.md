# 09 — Framework patterns for a modular, composable, learnable simulation engine

Research brief for **gradoscopy** (PyTorch, differentiable light-microscopy simulation, intended engine for DeepTrack2).

Evidence legend used throughout:
- **[V]** verified in this session against the linked doc or source file (Sept 2026).
- **[R]** from recollection or general expertise, not re-checked here. Treat as likely but confirm before relying on details.
- **[?]** uncertain, flagged explicitly.

---

## 0. Executive framing

Every framework studied here that is "more than the sum of its parts" follows the same four principles:

1. **A declarative description ("what") is kept separate from the machinery that evaluates it ("how").** Mitsuba separates scene dictionaries from integrators, PyTorch3D separates `Meshes` from the rasterizer and shader, deepinv separates the signal `x` from `Physics` and from `PhysicsGenerator`, and microsim separates `Simulation` from `modality.render` and from `Settings`.
2. **A small set of shared data types that every component speaks.** Examples: Mitsuba's `Spectrum`/`Ray`/`SurfaceInteraction`, PyTorch3D's `Fragments`, torchvision's `tv_tensors`, deepinv's tensors plus `TensorList`, chromatix's `Field`, and TensorDict.
3. **Parameters are first-class, named and addressable by path.** Examples: `mi.traverse` and `params.keep(regex)`, zodiax path-based `.set/.get`, Pyro sample sites, and `nn.Module.named_parameters()`.
4. **Randomness is explicit and replayable.** Examples: Kornia `params=` replay, Pyro `trace`/`replay`/`condition`, deepinv `step(batch_size, seed)`, and Kubric `kb.setup(FLAGS)` returning the RNG.

DeepTrack2 already has (3) in a lazy-graph form and parts of (1). Its weak points are exactly the ones gradoscopy must fix: per-sample Python evaluation, global seeding, caching that interacts badly with autograd, and "how" knobs (`upscale`, `upsample`, `padding`) living on "what" objects.

---

## 1. Framework-by-framework findings

### 1.1 Mitsuba 3 / Dr.Jit

**Variants.** Variant names combine five dimensions: backend `scalar|llvm|cuda`, optional `_ad`, color `mono|rgb|spectral`, optional `_polarized`, and optional `_double`. There are "as many as 60 different variants". The pip wheel ships a curated subset, e.g. `scalar_rgb`, `llvm_ad_rgb`, `cuda_ad_spectral_polarized`. Polarization costs "roughly a 1.5-2X increase in rendering time". [V] https://mitsuba.readthedocs.io/en/stable/src/key_topics/variants.html

Variants are chosen with `mi.set_variant(...)`, which is process-global. The type aliases (`mi.Float`, `mi.Spectrum`) resolve against the active variant. [R] Objects created under one variant are not interchangeable with another. [R]
- *Lesson:* physics dimensions such as scalar vs polarized and mono vs spectral are a **fidelity axis**, and Mitsuba makes them compile-time. That buys speed but relies on global state.
- For gradoscopy, these dimensions should be **data-driven**: the shape and axes of a `Field` tensor, chosen per plan. They should never be a global mode.

**Scenes as dictionaries plus a plugin manager.** `mi.load_dict({"type": "sphere", "radius": 10.0, "bsdf": {"type": "dielectric"}})`. The `"type"` key selects a registered plugin, nesting expresses composition, `"type": "ref"` reuses objects, and XML `$param` defaults support templating. [V] https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html
- *Lesson:* a type-tagged nested dict is a universal serialization format for a plugin tree, and it is the same idea as Hydra's `_target_`.
- *Wart:* the integrator (the "how") can live inside the scene dict or be passed to `mi.render(...)`. Mixing the two blurs what vs how.

**Plugins and parameter exposure.**
- Custom plugins subclass e.g. `mi.BSDF` and read constructor arguments from a `Properties` object.
- They register with `mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))`.
- They expose parameters in `traverse(self, cb)` via `cb.put('tint', self.tint, mi.ParamFlags.Differentiable)`, and recompute derived state in `parameters_changed(self, keys)`.
- [V] https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html
- `mi.ParamFlags` also has a **`Discontinuous`** flag. It marks parameters (e.g. vertex positions) whose gradients involve visibility discontinuities, which require special integrators. [R]
- *Lesson:* **parameter metadata that solvers read.** Whether a parameter is differentiable, discontinuous or discrete is something the planner must know.

**Differentiable rendering API.**
- `params = mi.traverse(scene)` returns a flat `SceneParameters` dict with path keys (e.g. `'mesh.vertex_positions'`).
- `params.keep(regex)` filters, and `params.update(opt)` pushes changes and rebuilds acceleration structures.
- `mi.ad.Adam` is dict-like.
- [V] https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/use_optimizers.html

**Custom adjoints instead of taping.**
- `ADIntegrator` defines `render_forward(scene, params, sensor, seed, spp)` and `render_backward(scene, params, grad_in, sensor, seed, spp)`. [V] https://raw.githubusercontent.com/mitsuba-renderer/mitsuba3/master/src/python/python/ad/integrators/common.py
- Path-replay backpropagation (`prb`) exists because taping a full light-transport graph is memory-prohibitive. [R]
- *Lesson:* solvers must be allowed to supply their own VJP/adjoint. For gradoscopy the candidates are multislice/BPM adjoints, partially coherent source sums and long z-stacks.

**PyTorch interop.** `@dr.wrap(source='torch', target='drjit')` around a function that calls `mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)`. [V] https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html
- Separate `seed` and `seed_grad` decorrelate the primal and adjoint Monte Carlo samples.
- The docs warn about synchronization overhead between frameworks.
- *Lessons:* (a) any Monte Carlo solver in gradoscopy (random source sampling, speckle, stochastic emitters) needs a separate gradient seed policy; (b) staying in one framework (PyTorch) end to end matters.

### 1.2 PyTorch3D

**Renderer as rasterizer ∘ shader.** `MeshRenderer.__init__(self, rasterizer, shader)`. `forward(meshes_world, **kwargs)` calls `self.rasterizer(meshes_world, **kwargs)` and then `self.shader(fragments, meshes_world, **kwargs)`. The kwargs let callers override cameras or lights per call. `MeshRendererWithFragments` also returns the intermediate `fragments` (e.g. `zbuf` for depth). [V] https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/pytorch3d/renderer/mesh/renderer.py

The rasterizer output is a fixed intermediate record (`pix_to_face`, `zbuf`, `dist`, `bary_coords`). The CUDA rasterizer is the only non-PyTorch step, and "the rest of the pipeline is implemented purely in PyTorch". [V] https://pytorch3d.org/docs/renderer
- *Lesson:* a **narrow, typed intermediate between stages** is what makes stages swappable. The shader does not care how fragments were produced.
- In gradoscopy the equivalents are the sampled representation (voxel volume, emitter list, slices) and the complex field at the image plane.

**Heterogeneous batching.** `Meshes` offers list, padded (`verts_padded()` gives `N × max(V) × 3`, faces padded with -1) and packed (`verts_packed()` gives `ΣV × 3`) views, with precomputed index maps between them. [V] https://pytorch3d.org/docs/batching
- *Lesson:* variable object counts per image, a certainty in microscopy, need a first-class padded-plus-mask representation with an efficient packed view.

**Fidelity and differentiability as settings.** `RasterizationSettings(image_size, blur_radius, faces_per_pixel, bin_size, max_faces_per_bin, ...)` [R]. `blur_radius > 0` together with `faces_per_pixel > 1` is what gives the soft rasterizer useful silhouette gradients. With `blur_radius = 0` the gradients with respect to edge positions are essentially zero. [R]
- *Lesson:* **"differentiable" is a property of the discretization choice.** A hard voxelization of a sphere (DeepTrack's upsample-then-average-pool approach) gives gradients with respect to radius and position that are zero almost everywhere. gradoscopy's representation converters must declare this, and the planner must choose smooth converters when gradients are requested.

### 1.3 nerfstudio

**Config ↔ implementation pairing.** [V] https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/nerfstudio/configs/base_config.py

```python
@dataclass
class InstantiateConfig(PrintableConfig):
    _target: Type
    def setup(self, **kwargs) -> Any:
        return self._target(self, **kwargs)
```

Every component has a config dataclass whose `_target` points to its implementation. The implementation receives the typed config, so autocomplete works. Whole methods are named presets in `method_configs.py`. `tyro` turns the dataclass tree into a typed CLI. [V] https://docs.nerf.studio/developer_guides/config.html

**Third-party extension via entry points.** Plugins register with `[project.entry-points.'nerfstudio.method_configs'] my-method = 'my_method.my_config:MyMethod'`, where the target is a `MethodSpecification(config=TrainerConfig(...), description=...)`. `NERFSTUDIO_METHOD_CONFIGS="my-method=pkg.mod:obj"` covers uninstalled development plugins. Data parsers have a parallel group. [V] https://docs.nerf.studio/developer_guides/new_methods.html

- *Lesson:* presets of whole configs (`nerfacto`, `nerfacto-big`, `nerfacto-huge` [R]) are effectively a **fidelity knob at method level**. This is a proven UX: named presets that expand to fully explicit, overridable configs.
- *Caveat:* nerfstudio configs describe *training systems*, not physical scenes. They hold no distributions or learnable scene parameters.

### 1.4 deepinv (operator algebra, generators)

This is the closest existing abstraction to "imaging physics as composable, differentiable operators".

**Physics.** [V] https://raw.githubusercontent.com/deepinv/deepinv/main/deepinv/physics/forward.py
- Signature: `Physics.__init__(self, A=lambda x, **kw: x, noise_model=None, sensor_model=lambda x: x, solver="gradient_descent", max_iter=50, tol=1e-4, **kwargs)`.
- `forward` is `self.sensor(self.noise(self.A(x, **kwargs), **kwargs))`. This three-part split (deterministic operator, stochastic noise, nonlinear sensor) is exactly right for microscopy: optics, shot/read noise, then gain/saturation/ADC.

**LinearPhysics.** [V] same source.
- Adds `A_adjoint`, `A_dagger`, `prox_l2`, `compute_norm`, `condition_number` and `adjointness_test()`, which checks `<v, Au>` against `<A^T v, u>`.
- If no adjoint is given, `A_adjoint` falls back to `adjoint_function(self.A, ...)`, which computes the adjoint by autodiff of the linear map. [V]
- *Pattern:* **an explicit fast path with an automatic, correct fallback.**

**Algebra.** [V] https://deepinv.org/user_guide/physics/intro.html
- `physics2 * physics1` composes, giving `ComposedPhysics` or `ComposedLinearPhysics` depending on type.
- `stack()` produces `StackedPhysics` whose measurements are a `TensorList`, for multi-view or multi-channel acquisitions.

**Parameters flow per call.** `physics(x, filter=theta)` or `physics.update(**params)`. Operator structure is static, and θ is data. [V] same page

**PhysicsGenerator.** [V] https://raw.githubusercontent.com/deepinv/deepinv/main/deepinv/physics/generator/base.py
- `step(batch_size=1, seed=None, **kwargs) -> dict` returns a batch of parameters.
- It owns a `torch.Generator` (`self.rng`). String seeds are hashed with SHA-256.
- `gen_a + gen_b` merges the two parameter dicts. `GeneratorMixture` assigns batch elements to generators by probability. `average()` gives the mean operator.
- Physics and generators are **deliberately separate objects**, which is the separation of randomness from operator.
- Microscopy-relevant generators (`DiffractionBlurGenerator` and 3D variants with Zernike pupils, `ConfocalBlurGenerator3D`, `ProductConvolutionBlurGenerator` for space-varying blur) exist. [R]

*Gaps relative to gradoscopy:*
- Generators sample parameters with `torch.Generator`. As far as I know they have no concept of *learnable* distribution parameters or reparameterized sampling [R].
- `x` (the sample) is an image tensor, not a structured scene.
- There is no fidelity selection.

gradoscopy should **interoperate** by exporting any linear sub-plan as a `deepinv.physics.LinearPhysics`, rather than re-inventing reconstruction tooling.

### 1.5 Kornia, torchvision.transforms.v2, MONAI (composable, batched, randomized transforms)

**Kornia.** [V] https://raw.githubusercontent.com/kornia/kornia/main/kornia/augmentation/base.py and https://kornia.readthedocs.io/en/latest/augmentation.container.html
- `_BasicAugmentationBase(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)`.
- `forward_parameters(batch_shape) -> Dict[str, Tensor]` is separated from `forward(input, params=None)`. Parameters are produced by a `_param_generator`, and a per-sample `batch_prob` is stored in the params.
- `AugmentationSequential(..., data_keys=["input","mask","bbox_xyxy","keypoints",...])` applies the *same* sampled geometry to images and labels.
- `aug(x, params=aug._params)` replays exactly. `inverse()` undoes geometric steps.
- *Patterns:*
  - **sample-then-apply as two phases**, with the sampled parameters as a plain dict of tensors;
  - the **`same_on_batch` flag**, which expresses at what granularity randomness is drawn;
  - **label co-transformation**.

**torchvision v2.** [V] https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html and https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_custom_tv_tensors.html
- Transforms implement `make_params(flat_inputs) -> dict` and `transform(inpt, params)`. These have been public at least since 0.21. The earlier private names were `_get_params`/`_transform`. [R]
- Inputs of arbitrary nesting are flattened. Typed tensor subclasses (`tv_tensors.Image`, `Mask`, `BoundingBoxes`, and `KeyPoints` since 0.23, beta: https://github.com/pytorch/vision/releases/tag/v0.23.0) are dispatched by type. Unknown types pass through.
- Third parties add support for new types with `@F.register_kernel(functional="hflip", tv_tensor_cls=MyTVTensor)`.
- *Pattern:* **typed tensors plus a (functional, type) → kernel registry.** This is single-dispatch extension without subclassing the transform.
- *Implication:* gradoscopy's label outputs should use `tv_tensors` types where possible, so downstream augmentation keeps labels consistent for free.

**MONAI.** [V] https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/transforms/transform.py and https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/monai/transforms/lazy/functional.py
- `Randomizable` holds `R: np.random.RandomState`. Its `set_random_state()` and `randomize()` methods use `self.R`, never `np.random`.
- The docstring warns: "deepcopying instance of this class often causes insufficient randomness as the random states will be duplicated". This is a real multi-worker pitfall.
- **Lazy resampling:** spatial transforms push `pending_operations` onto a `MetaTensor`. `apply_pending` multiplies the affines (`combine_transforms`) and resamples **once**. It flushes early only when interpolation arguments are incompatible.
- *Pattern:* **accumulate, fuse, then execute.** This is the operator-fusion idea gradoscopy needs for optics:
  - consecutive pupil-plane multipliers fuse into one pupil;
  - consecutive homogeneous-medium propagations sum their `dz`;
  - shift-invariant incoherent blurs fuse into an OTF product;
  - pixel integration becomes a sinc factor in the OTF.

### 1.6 Kubric and BlenderProc (synthetic data with ground truth)

**Kubric.** [V] https://raw.githubusercontent.com/google-research/kubric/main/kubric/core/scene.py and https://raw.githubusercontent.com/google-research/kubric/main/kubric/renderer/blender.py
- `kb.Scene(resolution=...)` holds assets. `scene += kb.Sphere(...)`.
- Renderers and simulators are **views** linked with `link_view()`. Assets added to the scene are propagated to all linked views, so the scene is the single source of truth. This uses an observer pattern on traitlets.
- The renderer is configured separately: `Blender(scene, samples_per_pixel=128, adaptive_sampling=False, use_denoising=True, motion_blur=None, ...)`.
- `render(frames=None, return_layers=("rgba","backward_flow","forward_flow","depth","normal","object_coordinates","segmentation"))` returns `Dict[str, np.ndarray]`.
- Worker scripts start with `scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)`. Objects are placed with `kb.move_until_no_overlap(obj, simulator, spawn_region=..., rng=rng)`, the physics simulation runs, and outputs are written with `kb.write_image_dict` and `kb.write_json(...)` of `kb.get_scene_metadata(scene)`, `kb.get_instance_info(...)` and `kb.get_camera_info(...)`. [V] https://raw.githubusercontent.com/google-research/kubric/main/challenges/movi/movi_def_worker.py
- *Patterns:*
  - **ground truth = additional render layers of the same scene**, requested by name;
  - **per-instance metadata** exported alongside;
  - **an explicit RNG object threaded through all sampling helpers**;
  - **rejection-sampling helpers** for placement constraints, where DeepTrack has `NonOverlapping`.
- *Caveat:* Kubric is not differentiable and generates one scene at a time in a CPU process. It shows the data contract, not the execution model.

**BlenderProc.** [V] https://raw.githubusercontent.com/DLR-RM/BlenderProc/main/examples/basics/basic/main.py
- An imperative module API: `bproc.init()`, `bproc.loader.load_obj`, `bproc.camera.add_camera_pose`.
- Ground truth is opt-in: `bproc.renderer.enable_normals_output()`, `enable_depth_output(...)`. Then `data = bproc.renderer.render()` and `bproc.writer.write_hdf5(...)`. COCO and BOP writers also exist. [R]
- *Pattern:* **opt-in output channels** (cost is paid only for requested labels) plus **pluggable writers**.
- *Anti-pattern:* a global mutable scene singleton inside Blender.

### 1.7 DeepTrack2 (Feature / Property graph)

All facts below are [V] from `develop`-branch source: https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/properties.py, `backend/core.py`, `features.py`, `optical/optics.py`, `optical/scatterers.py`, `backend/units.py`.

- **Property:** "a rule for sampling values". A rule can be:
  - a constant or tensor;
  - a callable, with dependencies resolved **by matching argument names** to sibling properties (these become child nodes);
  - a list, dict or tuple, sampled elementwise;
  - an iterator;
  - a node.
- `SequentialProperty` adds `sequence_index`, `previous_value` and similar for time series.
- **DeepTrackNode:** a lazily evaluated, cached node. Values are stored per `_ID` tuple (for replicated sub-evaluations), and `_dependencies`/`_children` are tracked as weak sets. Calling a node returns the cached value if valid, otherwise it runs `action` and stores the result. `invalidate()` propagates to children, and `update()` resets the dependency tree. Nodes overload arithmetic, so `a + b` is a lazy node.
- **Feature:** kwargs become a `PropertyDict`, and subclasses implement `get(data, **props)`.
  - Operators: `>>` (chain), `+ - * /` (arithmetic features), `^` (Repeat), `&` (Combine), `|` (probabilistic branch).
  - `update()` invalidates and `new()` updates then evaluates.
  - Backends are switched with `.torch(device)` / `.numpy()`.
- **Batching:** `.batch(batch_size=32)` "generates multiple outputs by repeatedly calling `.new()`, stacking results". **Seeding:** `.seed()` seeds Python `random`, NumPy **and** PyTorch globally.
- **Optics:**
  - `Microscope` combines a sample feature with an objective and validates compatibility. It separates `ScatteredVolume` from `ScatteredField` outputs and merges volumes with `_create_volume` and `_merge_placed_volumes`.
  - `Optics` carries `NA, wavelength, magnification, resolution, refractive_index_medium, padding, output_region, pupil, illumination, upscale`.
  - `upscale` is explicitly the "internal oversampling factor used during image formation". Scatterers additionally have `upsample`, which evaluates on a finer grid and average-pools.
  - `Fluorescence` convolves each z-slice with a z-dependent PSF. `Brightfield` does iterative FFT propagation with a phase-shift `exp(1j * ri_slice * dz * K)` per slice. `Darkfield` and `ISCAT` subclass Brightfield. `MieSphere` produces fields.
- **Units:** built on pint, with custom pixel units (`sxpx`, `xpx`) redefined through a pint `Context`. For gradient-tracked tensors, `ConversionTable.convert()` **extracts conversion factors instead of wrapping in Quantity**, so pint is kept out of the autograd path.

**Assessment.** DeepTrack2 already embodies "declarative properties with sampling rules". Four problems remain:
1. **Evaluation granularity:** one image per Python traversal, so GPU batching is lost.
2. **Global RNG:** reproducibility depends on call order.
3. **A stateful cache** keyed by `_ID` beside autograd: cached tensors can hold stale graphs, `update()` semantics are implicit, and `torch.compile` cannot trace through the graph machinery.
4. **"How" knobs on "what" objects:** `upscale`, `upsample`, `padding`, and the choice of `Brightfield` vs `MieSphere` path are made by the user rather than a planner.

Name-matched lambda dependencies are ergonomic but opaque to static analysis and not picklable. Picklability matters on Windows and macOS, where DataLoader workers use `spawn`.

### 1.8 microsim (a directly comparable microscopy simulator)

[V] https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/simulation.py and `settings.py`
- **Simulation:** a pydantic model with `truth_space`, `output_space`, `sample`, `modality` (default `Widefield`), `objective_lens`, `channels`, `detector`, `exposure_ms`, `settings` and `output_path`.
  - Staged methods: `ground_truth()` gives `(F,Z,Y,X)`, `filtered_emission_rates()` gives `(C,F,W)`, `optical_image()` gives `(C,Z,Y,X)`, then `digital_image()` and `run()`.
  - The modality is dispatched via `self.modality.render(truth, emission_rates, objective_lens=..., settings=..., xp=self._xp)`. The ground truth can be cached to disk.
- **Settings:** holds the *fidelity and execution* knobs separately from the physics:
  - `np_backend`, `device`, `float_dtype`, `random_seed`;
  - `max_psf_radius_aus=6` ("decreasing this can *dramatically* speed up");
  - `spectral_bins_per_emission_channel=1`.
- *Lessons:*
  - (a) a validated, serializable schema of the whole experiment works well for microscopy;
  - (b) **fidelity knobs belong in a separate settings object**;
  - (c) **staged intermediate products** (truth, emission, optical, digital) double as labels and debugging hooks.
- Not differentiable end to end by design [R], and not batched.

### 1.9 Differentiable optics libraries: chromatix, dLux/zodiax

- **chromatix** (JAX): a `Field` object holds the complex tensor plus sampling (spacing) and spectrum. Elements (phase masks, lenses, propagation, polarization, shot noise) are composed "in a style similar to neural network layers", e.g. `cx.objective_point_source(...) → cx.phase_change → cx.ff_lens → field.intensity`. [V] https://raw.githubusercontent.com/chromatix-team/chromatix/main/README.md
- **dLux** (JAX + Equinox + zodiax): optics described "as a series of layers that operate sequentially on a wavefront". Parameters are manipulated by **string paths** (`.set`, `.get`, `.multiply`), and gradients are filtered by path. [V] https://raw.githubusercontent.com/LouisDesdoigts/dLux/main/README.md and https://raw.githubusercontent.com/LouisDesdoigts/zodiax/main/README.md
- *Lessons:*
  - the **Field-carries-its-own-sampling** type prevents a whole class of grid-mismatch bugs;
  - an optical train is naturally a **sequential layer list over a wavefront**;
  - immutable pytree modules make vmap and jit trivial, and in PyTorch the analogue is frozen dataclasses registered with `torch.utils._pytree.register_dataclass` (private module, but used by `torch.export` docs) [V-partial] https://docs.pytorch.org/docs/stable/export/programming_model.html;
  - path-addressable parameters are the ergonomic basis for "learn these, freeze those".
- *Gap:* neither offers representation-agnostic sample descriptions, fidelity planning or data-generation pipelines. They are "how" libraries.

### 1.10 Probabilistic programming: Pyro, torch.distributions, learned simulators

**Pyro effect handlers.** [V] https://pyro.ai/examples/effect_handlers.html
- Every `pyro.sample`/`pyro.param` builds a message dict (`name, fn, args, value, is_observed, infer, scale, cond_indep_stack, done`). The message is passed through `_PYRO_STACK`: each `Messenger`'s `_process_message` runs in reverse order, then default processing, then `_postprocess_message`.
- Handlers `trace`, `condition`, `replay`, `block`, `scale` and `seed` compose without knowing about each other.
- **PyroModule** [V] https://docs.pyro.ai/en/stable/nn.html:
  - `PyroParam(init, constraint=constraints.positive)` gives constrained learnable parameters;
  - `PyroSample(lambda self: Normal(self.x, 1))` gives module attributes that are *sample sites*, with priors that may depend on other attributes;
  - sampling is cached once per `__call__`.
- *Patterns for gradoscopy:*
  - **named sample sites** (path = site name);
  - a **trace** of sampled values, which is exactly the ground-truth record;
  - **condition/replay** to pin some sites (e.g. force positions from a real dataset's annotations) or reproduce a batch;
  - **constraints via bijective transforms** (`torch.distributions.transform_to(constraint)`) for learnable positive or bounded quantities (NA ∈ (0, n), radii > 0).
- *Anti-pattern to avoid:* the *global* handler stack and global param store (`pyro.get_param_store()`). In a library meant to be embedded, pass a context object explicitly. NumPyro's explicit `seed` handler with JAX keys is the cleaner model [R].

**torch.distributions** [R, stable API]:
- `rsample()` (pathwise/reparameterized) is available for Normal, LogNormal, Uniform, Beta, Gamma (implicit reparameterization), Dirichlet, MultivariateNormal, `RelaxedBernoulli` and `RelaxedOneHotCategorical` (Concrete/Gumbel-softmax).
- Poisson, Bernoulli, Categorical and Binomial have `has_rsample=False` and need score-function (REINFORCE) estimators via `log_prob`.
- Also relevant: `batch_shape` vs `event_shape`, `Independent`, `TransformedDistribution`, `MixtureSameFamily`, and `constraints`/`biject_to`/`transform_to`.

**Learning simulator distributions (prior art).** Meta-Sim (Kar et al., ICCV 2019) [V] https://arxiv.org/pdf/1904.11621:
- It learns a network that modifies scene-graph attributes to minimize MMD, computed in Inception feature space, between rendered and real images.
- Because the renderer was not differentiable, the gradient through it was approximated "using the method of finite differences" by perturbing each attribute.
- The downstream-task objective used REINFORCE with a moving-average baseline.
- Continuous attributes used the reparameterization trick. **Categorical attributes were kept immutable.**
- Learning to Simulate (Ruiz et al., ICLR 2019) is purely policy-gradient [R].
- *Lesson:* gradoscopy removes the finite-difference bottleneck for continuous parameters, but still needs:
  - (i) score-function or relaxed estimators for discrete sites (object counts, shape class, modality choices);
  - (ii) a clean way to add `log_prob`-based surrogate losses. The trace must therefore record `log_prob` per site when requested.

### 1.11 Config systems, units, plugins, structured batches

**Hydra `instantiate`** [V] https://raw.githubusercontent.com/facebookresearch/hydra/main/website/docs/advanced/instantiate_objects/overview.md:
- `_target_` holds a fully qualified import path; `_partial_`, `_recursive_` and `_convert_` control instantiation.
- Powerful for experiment configs, but import paths in config files are brittle across refactors and a code-execution surface.
- *Recommendation:* use **registry names** (`"sphere"`, `"widefield"`) rather than import paths in saved scenes. This is Mitsuba's approach.

**Dataclasses vs pydantic vs tyro** [R]:
- Pydantic v2 gives validation, discriminated unions (`Field(discriminator="type")`) and JSON Schema, which is excellent for microsim-style experiment files and GUIs.
- However, torch tensors, `nn.Parameter` and distribution objects need `arbitrary_types_allowed`, validation adds overhead on hot paths, and pydantic models are not pytrees.
- Frozen `@dataclass(kw_only=True)` (3.10+) can be registered as pytrees, is zero-overhead and works with tyro for CLIs.

**Units** [R, plus DeepTrack evidence above]:
- pint wraps NumPy via `__array_function__` but has no first-class torch-tensor support. Wrapping tensors breaks many `torch.*` ops and `torch.compile`.
- DeepTrack itself strips pint for gradient-tracked tensors.
- Industry practice in differentiable physics (chromatix, Mitsuba, dLux) is **SI floats inside, documented units, and helpers at the boundary**.

**Plugin systems** [R]:
- `importlib.metadata.entry_points(group="...")` is the stdlib standard. nerfstudio [V] uses it plus an environment variable for development.
- pluggy (pytest) adds hook specs, ordering and wrappers, which is useful for event hooks and overkill for a component registry.
- torchvision's `register_kernel` [V] is the model for *(operation, type)* dispatch extension.

**Structured batches.** TensorDict is "a batched, nested dict[str, Tensor] that behaves like a tensor" with one shared `batch_size`. It supports indexing and stacking of the whole structure, `.to(device)`, memmap save/load, `torch.compile` coverage and `@tensorclass`. `TensorDict.from_module(module)` plus `to_module` supports functional parameter swapping. [V] https://raw.githubusercontent.com/pytorch/tensordict/main/README.md
- This is a near-perfect container for gradoscopy's *resolved scene*, *trace* and *outputs* (images plus labels plus metadata).

### 1.12 Other precedents for "what vs how" and planning [R]

- **Halide:** an algorithm (what is computed) is separated from a schedule (tiling, vectorization, parallelism, locality). The same algorithm runs under many schedules with identical semantics. This is the purest statement of the fidelity/solver split. The difference here is that gradoscopy's "schedules" change *physics accuracy*, not only performance, so each needs an **accuracy contract**.
- **SciML / DifferentialEquations.jl:** `solve(ODEProblem(...), alg; sensealg=...)`. A problem type is solved by an algorithm, a default polyalgorithm chooses when `alg` is omitted, and the **adjoint method (`sensealg`) is a separate choice from the forward solver**. That is the right factorization for gradoscopy too: forward solver ≠ gradient strategy (autograd, checkpointed, custom adjoint, or score-function).
- **FFTW planner:** `FFTW_ESTIMATE` vs `FFTW_MEASURE`. A cheap heuristic plan or a measured plan is cached per problem signature. gradoscopy plans should be cached per *static structure* (grid, representation types, solver set).
- **MuJoCo MJX:** a static `Model` (structure, compiled once) is kept separate from batched `Data` (state). `jax.vmap(mjx.step, in_axes=(None, 0))`. For gradoscopy: static **Plan** plus batched **resolved-scene tensors**.
- **ONNX Runtime execution providers:** the graph is partitioned by provider capability, with automatic CPU fallback for unsupported ops. This is capability negotiation with fallbacks, and a caution: silent fallbacks confuse users, so they must be logged.

---

## 2. Cross-cutting patterns extracted

### 2.1 What vs how
The robust decomposition in all mature systems is **Spec → (sampling) → Resolved instance → (planning) → Plan → (execution) → Outputs**:

| Stage | Examples | gradoscopy analogue |
|---|---|---|
| Spec (declarative, may contain distributions) | Mitsuba dict, Kubric scene, microsim `Simulation`, DeepTrack Features | `Scene` of frozen dataclasses with `Param` leaves |
| Sampling | deepinv `step()`, Kornia `forward_parameters`, Pyro trace | `scene.draw(ctx) -> Trace` (TensorDict, batch dim B) |
| Plan (how) | Mitsuba integrator plus variant, PyTorch3D rasterizer/shader, Halide schedule | `Plan` chosen by planner from fidelity plus capabilities |
| Execute | `mi.render`, `renderer(meshes)`, `physics(x, **θ)` | `plan(trace) -> Outputs` (pure torch, compile-able) |

Key rule: **the Spec never names a numerical method or a discretization**, not even `upscale` or `padding`. Grid spacing, oversampling, padding, PSF support, spectral bins and source-point counts are all Plan decisions derived from `Fidelity` plus scene bounds. Users can still *pin* any of them through the fidelity object.

### 2.2 Dispatch, capability negotiation, fallbacks
Three levels, from simplest to most general:
1. **Type dispatch** (torchvision `register_kernel`, `functools.singledispatch`, plum multiple dispatch [R]): (operation, representation type) → kernel. Good for converters and label renderers.
2. **Capability matching:** each solver declares what it accepts and guarantees. The planner filters by requirements, then ranks by a cost model. This is ONNX Runtime-like.
3. **Search over conversion graphs:** representations are nodes and converters are edges with (cost, differentiability, fidelity loss). The planner finds the cheapest path from the sample's native representation to a solver's accepted input. A shortest-path search suffices because the graph is small.

Fallback rule: **always explicit in the plan**. `plan.explain()` lists chosen solvers, rejected alternatives and reasons ("MieSolver rejected: object shape Ellipsoid not supported"). Outputs carry `meta.plan_hash`.

### 2.3 Operator algebra for optical systems
From deepinv (compose, stack, adjoint, noise/sensor split), chromatix/dLux (layers on a Field), and MONAI (lazy fusion):
- **Typed operator categories:**
  - `FieldOp`: linear in the complex field, e.g. `Propagate(dz)`, `Pupil(P)`, `PhaseMask`, `Aperture`, `Lens`.
  - `Scatter(sample)`: linear in the incident field, nonlinear in the sample.
  - `Detect`: Field → Intensity, `|·|²` summed over polarization.
  - `IntensityOp`: linear on intensity, e.g. `Convolve(PSF/OTF)`, `PixelIntegrate`, `SpectralFilter`.
  - `Noise` (stochastic) and `Sensor` (nonlinear: gain, saturation, quantization).
- **Composition** `B @ A` (or `>>`) is type-checked at build time; for example, `Pupil` cannot follow `Detect`. **Stacking** `stack(A1, A2)` covers multi-channel, multi-focus, phase-stepping or SIM, producing a leading axis.
- **Coherence combinator:** partially coherent Köhler imaging is `IncoherentSum(over=source_points, op=Detect ∘ Imaging ∘ Illuminate(s))`, which is Abbe's method. The source-point count is a fidelity parameter. TCC/SOCS decompositions are alternative *implementations* of the same node.
- **Algebraic rewrites** (planner, MONAI-style accumulate-and-fuse):
  - `Propagate(a)∘Propagate(b) → Propagate(a+b)` in a homogeneous medium;
  - adjacent pupil multipliers fuse;
  - `Convolve(h1)∘Convolve(h2) → Convolve(OTF1·OTF2)`;
  - `PixelIntegrate∘Convolve` becomes an OTF × pixel-sinc;
  - downsampling after a band-limited convolution can be folded in.
- **Adjoint:** linear ops expose `adjoint`, analytic or by autodiff fallback (deepinv's `adjoint_function` pattern). This enables custom memory-efficient VJPs and export to deepinv.

### 2.4 The fidelity knob
Evidence: Mitsuba (variant, `spp`, `max_depth`, integrator), PyTorch3D (`blur_radius`, `faces_per_pixel`), microsim (`max_psf_radius_aus`, `spectral_bins_per_emission_channel`), DeepTrack (`upscale`, `upsample`) and nerfstudio (presets).

Distilled design:
- **Fidelity is a policy, not a scalar.** A named preset (`"preview" < "fast" < "balanced" < "accurate" < "reference"`) expands into:
  - (a) **physics requirements**: vectorial or scalar PSF, polarization, multiple scattering, spectral bins, partial-coherence sampling;
  - (b) **discretization parameters**: oversampling, padding, PSF support radius, z-step and slice thickness;
  - (c) **solver preferences** and (d) numeric precision.
- It can be **overridden per subsystem and per field**, e.g. `Fidelity("fast", optics={"psf": "vectorial"})`.
- It is **discrete and ordered** (a lattice), not continuous. Continuous knobs such as oversampling are pinned values inside a preset.
- Every solver declares a **validity predicate** evaluated against scene *bounds*, including distribution supports. Examples:
  - Born is valid when max phase delay `2π·Δn·d/λ ≪ 1`;
  - paraxial PSFs are valid only at low NA;
  - Mie applies to spheres only.
  
  The planner warns or escalates when violated.
- **Fidelity ladder tests**, each tier compared against the next tier within its validity regime, turn the knob into a trustworthy contract rather than a vibe.

### 2.5 Randomness and learnable parameters in one abstraction
Requirements: constants, learnable constants, random variables with fixed parameters, random variables with **learnable parameters**, derived values, and discrete choices, all addressable by path.

- A single `Param` leaf type with subclasses `Const`, `Learnable(init, constraint)`, `Random(dist, plate=...)` (dist arguments may themselves be `Param`s), `Derived(fn, deps)` and `Choice(options, probs, estimator=...)`.
- Sampling happens only under an explicit `SampleContext` (batch size, `torch.Generator`, device, handlers). It returns a **Trace** containing values, `log_prob` where needed, and the plate (granularity) of each site.
- **Plates** generalize Kornia's `same_on_batch`: `plate="batch"` draws once per batch, `"sample"` per image, `"object"` per object slot, `"frame"` per time step.
- **Gradient estimators** are chosen per site:
  - `pathwise` (rsample) is the default for continuous sites;
  - `relaxed(τ)` covers presence, class and Concrete sites;
  - `score` adds REINFORCE with a baseline, using the trace's `log_prob`;
  - `none` stops gradients.
  
  This mirrors the forward-solver versus sensealg split.
- **Handlers**, as local context managers rather than a global stack:
  - `condition({path: value})`;
  - `replay(trace)`;
  - `mean()`, which uses distribution means to give a deterministic debugging image;
  - `block(paths)`;
  - `detach(paths)`.
- **Learnable storage:** learnable leaves are materialized as `nn.Parameter`s in unconstrained space (`transform_to(constraint).inv`) inside a module owned by the Simulator. This lets `sim.parameters()`, `state_dict()`, `.to()` and standard optimizers work. Path names become `named_parameters()` keys.

### 2.6 Batching semantics
- Batch-first everywhere, on the GPU. No per-sample Python loop (the DeepTrack `.batch()` anti-pattern).
- Variable object counts use padded `[B, N_max, …]` tensors with `mask[B, N_max]` and optional soft `presence[B, N_max]`. `N_max` is a static plan property, and bucketing avoids recompiles. The PyTorch3D padded/packed/list trio is the model; a packed view is useful for per-object kernels.
- Additional plates as named axes, in a canonical order: `B, S (source points), W (wavelengths), P (polarization), Z, Y, X`. Plans include or omit axes; dims are documented with jaxtyping. PyTorch named tensors remain a prototype and should not be used [R].
- Parameter sharing across the batch is a plate choice, not a special code path.
- Containers are TensorDict or `@tensorclass` for Trace, ResolvedScene and Outputs.

### 2.7 Serialization and reproducibility
- The **Spec** serializes to a type-tagged dict/JSON/YAML, keyed by registry name as in Mitsuba, with a schema version. Learnable values go to a `state_dict`. Callables are allowed in memory, but `Derived` functions must be importable module-level functions (by qualified name) for the spec to be serializable. A lambda marks the spec "non-serializable" with a warning.
- **Reproducibility:**
  - Seeds are *derived*: `seed(sample_i) = H(global_seed, epoch, index)`. Simple batch-level determinism comes from one `torch.Generator` per batch seeded by `H(global_seed, batch_index)`.
  - Per-sample determinism independent of batch composition needs counter-based RNG (Philox-style hashing of `(seed, index, site)` in torch integer ops). Offer it as an option.
  - **Most importantly, store traces.** Exact reproduction comes from `replay(trace)` (Kornia `params=`, Pyro `replay`), not from hoping the RNG stream lines up across library versions.
- Outputs carry `meta = {spec_hash, plan_hash, gradoscopy_version, seed, torch_version, device}`.
- Separate `seed` and `seed_grad` for any Monte Carlo solver (Mitsuba).
- Never touch global RNGs (DeepTrack `.seed()`, MONAI deepcopy pitfall).

### 2.8 Ground-truth / label extraction
- Labels are **additional named outputs of the same resolved scene**, requested up front so their cost is paid only when asked. This is Kubric's `return_layers` and BlenderProc's `enable_*_output`.
- Three sources:
  - (a) **trace values**: positions, radii, RI, counts, defocus and aberration coefficients, i.e. free regression targets;
  - (b) **label renderers**: instance and semantic masks at detector resolution, distance or heat maps, centroid maps, 3D volumes, per-object bounding boxes and keypoints, rendered with the same geometry converters;
  - (c) **stage intermediates**: noise-free image, image-plane complex field, per-object images, PSF and OTF, as in microsim's staged products and PyTorch3D's `fragments`.
- Types: `tv_tensors.Image/Mask/BoundingBoxes/KeyPoints` so torchvision and Kornia augmentations keep labels aligned.
- Per-object tables are padded with a mask; visibility and in-FOV flags are computed as in Kubric's `compute_visibility`.

### 2.9 Extension by third parties
- One registry per extension point (`representation`, `converter`, `solver`, `psf_model`, `noise`, `label`, `preset`), populated by decorators and by `entry_points(group="gradoscopy.plugins")`, loaded lazily on first lookup. Also support a dev-mode environment variable as in nerfstudio.
- The extension contract is **data types plus capability declarations**, not subclass internals. A plugin must declare accepted and produced types, capabilities, validity, cost and gradient support.
- Duplicate names are an error unless `override=True`. Plugins declare the gradoscopy API version they target.

---

## 3. Anti-patterns to avoid

1. **Magic lazy graphs with caching beside autograd.** DeepTrack's `DeepTrackNode` caches tensors per `_ID` with implicit invalidation. With autograd this yields stale graphs ("backward through the graph a second time"), leaked memory from retained graphs, and gradients through values sampled in a previous step. It also blocks `torch.compile`. Use **eager, pure functions over explicit Trace tensors**. Autograd *is* the graph; do not build a second one.
2. **Over-abstract IR too early.** Building a general dataflow compiler before three or more solvers exist costs debuggability and gives no leverage. Start with a rule-based planner that emits a *flat list of named stages* you can step through in a debugger, print and time.
3. **Global mutable state:** Mitsuba's global variant, Pyro's global param store and handler stack, DeepTrack `.seed()` seeding global RNGs, BlenderProc's global scene. Use explicit context objects.
4. **"How" on "what" objects:** `upscale`, `padding`, `upsample`, or an integrator inside the scene. This forces users to be numerical-methods experts and prevents automatic fidelity selection.
5. **Silent fallbacks and silent validity violations:** e.g. quietly using the Born approximation on a strongly scattering cell. Every fallback and violated validity condition is recorded in the plan and surfaced.
6. **"Differentiable" in name only:** hard voxelization, binary masks, `argmax` and `round` in the sample path. Converters declare gradient quality, and the planner refuses or warns when a requested learnable parameter would get zero or biased gradients.
7. **Operator-overload soup.** DeepTrack's `>> + * ^ & |` on Features makes code hard to read, and `+` means both arithmetic and stacking. Keep at most `@` or `>>` (compose) and explicit `stack()`/`sum_over()` functions.
8. **Units objects in hot paths.** Pint Quantities wrapped around tensors break compile and performance. Use SI floats inside and conversion helpers at the boundary.
9. **Lambdas as the only way to express dependencies.** They are unpicklable (Windows DataLoader workers use spawn), not serializable and invisible to the planner. Offer declarative `Derived` and relation helpers, and allow lambdas as an escape hatch only.
10. **Per-sample Python loops** (`.batch()` calling `.new()` N times) and **dynamic shapes per sample**, which cause recompiles. Use padding, masks and buckets.
11. **A god object** (`Microscope` doing sample merging, compatibility checks, upscaling, propagation and downscaling). Split it into Scene, Planner, Plan stages and Detector.
12. **Stringly-typed config with import paths** (raw Hydra `_target_` in saved datasets). Use registry names plus a schema version.
13. **Making every scene object an `nn.Module`.** Mutable module state mixed with sampling leads to aliasing bugs. Specs should be immutable; learnables live in one owned `ParamStore` module; sampled values live in the Trace.

---

## 4. Implications for gradoscopy's architecture

### 4.1 Recommended layered architecture

```
L7  gradoscopy.learn      fitting sim params to real data (MMD / sliced-W / critic), estimators, curricula
L6  gradoscopy.data       Simulator.sample(), IterableDataset, label specs, writers (zarr/HDF5 + JSON manifest)
L5  gradoscopy.plan       Planner: (Scene, Fidelity, outputs, device, budget) -> Plan; explain(); cache
L4  gradoscopy.solvers    image-formation kernels with Capabilities (incoherent conv, Born/Rytov, BPM/multislice,
                          Mie, Abbe source-sum, Richards-Wolf/Gibson-Lanni PSFs, detector models)
L3  gradoscopy.repr       sample representations + converter graph (Primitives, Emitters, SDF, Mesh, Voxel RI /
                          emission, Slices) with declared differentiability
L2  gradoscopy.scene      declarative frozen dataclasses: Sample, Illumination, Optics, Detector, Scene
L1  gradoscopy.params     Param leaves (Const/Learnable/Random/Derived/Choice), plates, SampleContext, Trace,
                          handlers (condition/replay/mean/block), estimators, ParamStore(nn.Module)
L0  gradoscopy.core       Grid, Field, Intensity, typed tensors, units helpers (SI floats), RNG derivation,
                          registries, FFT/padding utilities
interop/  deeptrack (Feature adapter), deepinv (Physics export), torchvision tv_tensors
```

The dependency rule is strictly downward. L2 must not import L4. Solvers never see `Param`s, only tensors from the Trace. This is what makes solvers compile-able and testable in isolation.

### 4.2 Core contracts, in priority order
1. **`Grid` and `Field`** are the lingua franca. `Field(data[B,…,Y,X] complex, grid, wavelength, n_medium, basis)` carries its sampling, as in chromatix. Every FieldOp checks grid compatibility at plan time, not run time.
2. **The Trace** is a TensorDict with `batch_size=[B]`, keys = site paths (`"sample.particles.radius"`), plus `mask`, `presence` and optional `log_prob`. It is at once the solver input, the ground-truth source and the reproducibility record.
3. **Capabilities** form a frozen dataclass per solver or converter:
   - `accepts`, `produces`: representation types;
   - `contrast`: `{"emission","phase","absorption","scattering"}`;
   - `physics`: `{"vectorial","polarization","multiple_scattering","partial_coherence","spectral"}`;
   - `tier`: an ordinal;
   - `grad`: one of `{"autograd","checkpointed","custom_vjp","none"}`, plus per-input gradient quality;
   - `validity(bounds) -> list[Violation]`;
   - `cost(problem) -> (flops, bytes)`.
4. **Plan** is an ordered list of named stages (plain callables with static config) plus metadata. It is printable, serializable and hashable. It lives inside an `nn.Module` for device and dtype placement only.

### 4.3 User-facing API sketch (target ergonomics)

```python
import torch, gradoscopy as gs
from gradoscopy import dist as D                 # torch.distributions wrappers accepting Param args
from gradoscopy.units import nm, um, NA_         # plain SI floats: nm = 1e-9

# ---- WHAT: declarative scene; leaves may be constants, learnables, or distributions ----
particles = gs.Population(
    shape    = gs.Sphere(radius=D.LogNormal(gs.learn(torch.log(torch.tensor(150*nm))), 0.25)),
    material = gs.Dielectric(n=D.Uniform(1.38, 1.60)),
    position = gs.UniformInFOV(z=D.Normal(0.0, 1*um)),
    count    = D.Poisson(gs.learn(8.0), max=64),      # discrete -> relaxed presence by default
    constraints = [gs.NonOverlapping(min_gap=50*nm)],
)
scene = gs.Scene(
    sample       = gs.Sample(particles, medium=gs.Medium(n=1.33)),
    illumination = gs.Koehler(wavelength=D.Uniform(520*nm, 540*nm, plate="batch"),
                              condenser_NA=0.4),
    optics       = gs.Brightfield(objective=gs.Objective(NA=1.3, magnification=60, immersion_n=1.518),
                                  aberrations=gs.Zernike({"spherical": gs.learn(0.0)})),
    detector     = gs.Camera(pixel_size=6.5*um, shape=(256, 256), qe=0.8,
                             read_noise=D.Uniform(1.0, 2.0), baseline=100, bits=16),
)

# ---- HOW: fidelity policy -> plan ----
sim = gs.Simulator(
    scene,
    fidelity=gs.Fidelity("balanced", optics={"psf": "vectorial"}),   # per-subsystem override
    outputs=["image", gs.labels.Positions(), gs.labels.InstanceMasks(), "noise_free", "field"],
    device="cuda",
)
print(sim.plan.explain())
# stage 1 primitives->RI voxels (soft, antialiased)  | grad(radius,position): smooth
# stage 2 Abbe source-sum, 9 points                   | tier balanced
# stage 3 multislice BPM, dz=100nm, 2x oversample     | Born rejected: max phase 1.9 rad > 0.3
# stage 4 vectorial pupil, pixel OTF fused            | fused: pupil*zernike*defocus
# stage 5 Poisson-Gaussian camera (pathwise approx.)  | ...

batch = sim.sample(batch_size=32, seed=0)      # TensorDict [32]: image, labels.*, trace.*, meta
batch["image"].shape                            # [32, 1, 256, 256]
batch["labels", "positions"]                    # tv_tensors.KeyPoints-compatible + mask

# ---- LEARN: fit the simulator's distributions/optics to real data ----
opt = torch.optim.Adam(sim.parameters(), lr=1e-2)   # only gs.learn(...) leaves appear here
for step, real in enumerate(real_loader):
    fake = sim.sample(32, seed=step)
    loss = gs.learn.mmd(feat(fake["image"]), feat(real))
    loss = loss + fake.score_surrogate(loss)          # adds REINFORCE terms for 'score' sites, if any
    opt.zero_grad(); loss.backward(); opt.step()

# ---- handlers: pin, replay, debug ----
with gs.condition({"sample.particles.position": annotated_xyz}):
    img = sim.sample(1, seed=1)["image"]
img_again = sim.render(gs.replay(batch.trace))       # bit-exact re-render of a stored batch
with gs.mean(): sim.sample(1)                          # deterministic "expected" scene

# ---- training data ----
ds = gs.data.SimDataset(sim, batch_size=64, seed=42)   # IterableDataset; GPU gen in main proc
# interop
feature = gs.interop.deeptrack.as_feature(sim)         # usable inside DeepTrack2 pipelines
phys = gs.interop.deepinv.as_physics(sim.plan, linear_part=True)  # when the stage chain is linear
```

### 4.4 Extension API sketch

```python
@gs.register.solver("bpm_multislice")
class MultisliceBPM(gs.Solver):
    capabilities = gs.Capabilities(
        accepts={gs.repr.RIVolume}, produces=gs.Field,
        contrast={"phase", "absorption"}, physics={"multiple_forward_scattering"},
        tier=gs.Tier.ACCURATE, grad="checkpointed",
    )
    def validity(self, b: gs.Bounds):            # scene bounds incl. distribution supports
        return [gs.Violation("backscatter ignored", severity="info")] if b.max_dn > 0.1 else []
    def cost(self, p: gs.Problem):               # used by the planner to rank candidates
        return gs.Cost(flops=p.n_slices * p.nxy * log2(p.nxy) * 10, bytes=p.nxy * 16 * p.batch)
    def configure(self, p: gs.Problem, fid: gs.Fidelity) -> dict:   # static, plan-time
        return dict(dz=fid.slice_thickness(p), pad=fid.padding(p))
    def __call__(self, field: gs.Field, vol: gs.repr.RIVolume, *, dz, pad) -> gs.Field:
        ...                                      # pure torch; no Params, no RNG, no globals

# pyproject.toml of a third-party package
# [project.entry-points."gradoscopy.plugins"]
# tmatrix = "gs_tmatrix:register"
```

### 4.5 Planner design (keep it boring)
1. **Static analysis of the Scene:** representation types present; contrast mechanism; coherence; NA; conservative bounds from distribution supports and quantiles, e.g. max object extent, max Δn, max phase delay, z-range; which leaves are learnable, and their Mitsuba-style flags (`smooth`, `discontinuous`, `discrete`).
2. **Requirements** come from Fidelity plus physics triggers. Example: NA/n > 0.9 and the fidelity is at least "balanced" → require vectorial. Emission-only → the incoherent path.
3. **Candidate generation:** registry lookup filtered by capability, with converter paths found by Dijkstra on (cost, gradient-quality penalty).
4. **Ranking:** by the cost model under the memory budget. Record rejected alternatives and reasons.
5. **Discretization:** grid, padding, oversampling, slice thickness and PSF support are derived from fidelity rules plus scene bounds (Nyquist for NA/λ, PSF radius in Airy units as in microsim). Everything is pinnable.
6. **Rewrite pass:** fuse pupil multipliers, merge propagations, fold pixel OTF, drop unused outputs.
7. **Emit the Plan** and cache it by `hash(static_structure, fidelity, outputs, device)`.

Start as roughly 300 lines of explicit rules. Grow toward search only when needed.

### 4.6 Scope phasing (feasibility)
- **v0.1 — core plus the incoherent path.**
  - L0–L2, and L1 in full (it is the hardest to retrofit).
  - Representations: Primitives, point emitters and emission voxels.
  - Solvers: 2D/3D PSF convolution with Gaussian, scalar Debye and Gibson–Lanni PSF models, pixel integration and a Poisson–Gaussian camera.
  - Labels and the DeepTrack adapter.
  
  This already covers most fluorescence and SMLM training data.
- **v0.2 — the coherent path.** RI voxels; first Born/Rytov transfer functions; multislice BPM; Mie for spheres; brightfield, darkfield, holography and iSCAT detection; Abbe source-sum for partial coherence.
- **v0.3 — accuracy.** Vectorial (Richards–Wolf) PSFs and polarization; spectral sampling; meshes and SDFs; time series (trace plates over frames); deepinv export.
- **Out of core scope (plugins):** full-wave FDTD/FDFD, T-matrix for arbitrary shapes, nonlinear or multiphoton physics beyond intensity-power PSFs, electron microscopy.

### 4.7 Opinionated recommendations (summary)
1. Adopt **Spec → Trace → Plan → Outputs** as the architectural spine. Solvers accept only tensors and typed data (Field, representations), never Params.
2. Make **Param/Trace/plates/handlers (L1)** the first thing built and the most carefully tested. It is the key differentiator relative to deepinv, Kornia and DeepTrack.
3. Make **Fidelity a policy object with named presets, per-subsystem overrides, validity predicates and ladder tests**. Never put numerical knobs on scene objects.
4. Use **typed operator categories** (FieldOp, IntensityOp, Detect, Noise, Sensor) with composition and stacking checked at plan time, a small rewrite/fusion pass, adjoints for linear ops, and export to deepinv.
5. **Batch-first, padded-plus-mask, TensorDict containers, static shapes per plan.** Plans are `torch.compile` units.
6. **Gradient strategy is separate from the forward solver:** pathwise, relaxed, score or none per site; autograd, checkpointed or custom-VJP per stage; `seed_grad` for Monte Carlo stages.
7. **Label outputs come from the same resolved scene**, are opt-in, and are typed with `tv_tensors`.
8. **Serialize specs with registry names and a schema version; persist traces.** Reproduce by replay, not by RNG luck.
9. **SI floats internally, with unit helpers at the boundary.** Pixel-unit inputs are resolved at plan time.
10. **Entry-point plugin registries with capability declarations**, and no silent fallbacks.
11. **Dataclasses (frozen, kw_only) for specs**, with pydantic or tyro as optional front-ends. Leave room for a microsim-style experiment file later.
12. **Stay debuggable:** each plan stage can be run, printed, timed and visualized individually (`sim.plan.run(trace, until="stage3")`). Avoid a second graph engine on top of autograd.
