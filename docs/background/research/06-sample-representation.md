# 06: Representing the sample for multiple differentiable solvers

Status: research brief for the gradoscopy architecture plan. Written 2026-09-24.
Legend: **[V]** means I checked it against the linked source during this research. **[R]** means it comes from memory or domain knowledge and was not re-checked. **[M]** means I measured it myself in this workspace (RTX 3090, torch 2.14+cu130, float64 for the accuracy tests, float32 for timing). The scripts are `scratchpad/verify_voxelization.py` and `scratchpad/verify_bandlimit.py`.

---

## 0. Summary

1. **The sample should be a continuous, resolution-free description, not a voxel grid.** A voxel grid, point list, sphere list, projected map or k-space spectrum is a *lowered view* that a solver requests with a specific `Grid`, wavelength set and band-limit policy. DeepTrack2 already has part of this split (`VolumeScatterer` vs `FieldScatterer`), but its volume scatterers produce hard-thresholded masks at render time.
2. **The main technical risk is the "moving edge" problem.** DeepTrack2 thresholds masks (`X+Y+Z <= 1`) and then average-pools supersampled masks. Both give **exactly zero gradient** with respect to radius and position. In my measurement the hard mask gives dV/dR = 0 and d(centroid)/dx0 = 0 **[M]**. Analytic band-limited occupancy fixes this at no extra cost. Two ways work: the closed-form Gaussian-blurred primitive, or analytic form factors in k-space. Both reproduce dV/dR and d(centroid)/dx0 to 4 or more digits **[M]**.
3. **Objects should expose capabilities**: `bbox`, `occupancy(x, σ)`, `sdf`, `form_factor(k)`, `sample_points`, `project`, `mie_spec`. A **conversion registry** tags each conversion as exact, band-limited-exact, approximate or lossy. Solvers declare what they accept, in order of preference. A planner chooses per object group and falls back along the chain. It raises an error when the requested fidelity forbids a lossy step.
4. **Geometry, material and labeling are separate.** One cell geometry drives refractive-index (RI) contrast for brightfield and label density for fluorescence at the same time. This gives consistent multimodal data and ground truth.
5. **Batch everything as struct-of-arrays per primitive kind.** Use a `[B, N_max, ...]` layout with a presence mask. This covers thousands of instances, variable counts, `torch.compile`, and relaxed learnable counts.
6. **Resolve overlaps by priority "over" compositing of material occupancies.** Children override parents, and siblings mix by occupancy. Emitter densities add. DeepTrack-style additive Δn stays available as a mode.

---

## 1. What each solver needs from the sample

| Solver family (fidelity tier) | Needs | Best source representation | Notes |
|---|---|---|---|
| Thin-sample / projection (OPL maps, SyMBac-style) | 2D maps: ∫Δn dz, ∫κ dz, emitter column density | Analytic projection (sphere chord `2√(R²−ρ²)`), or the k_z=0 slice of the form factor (projection-slice theorem) | Fastest tier. SyMBac renders an OPL image and convolves it with a PSF, with an optional 3D mode **[V]** [SyMBac](https://pmc.ncbi.nlm.nih.gov/articles/PMC9710168/) |
| Linear 3D imaging: first Born/Rytov (weak scatterers), incoherent fluorescence with an OTF | Scattering potential V(k) = k0²(ε−ε_m)^(k) on Ewald caps, or emitter density ρ̂(k) | **Analytic form factors** evaluated on the solver's k-grid. No voxelization is needed | Exact band-limiting comes for free, and gradients with respect to position and size are analytic |
| Multislice BPM / WPM / (modified) Born series | Complex Δn (or Δε) slices on a regular grid with Δx ≲ λ/(2n) | Voxelized occupancy × material, anti-aliased | Chromatix's `multislice_thick_sample(field, absorption_stack, dn_stack, n, thickness_per_slice, ...)` is the reference API shape. Its thin-slice transmission is `exp(i2π(dn + i·absorption)·Δz/λ)` **[V]** [chromatix samples.py](https://raw.githubusercontent.com/chromatix-team/chromatix/main/src/chromatix/functional/samples.py) |
| Polarized multislice (birefringence) | 3×3 dielectric/scattering-potential tensor per voxel, `[d,h,w,3,3]` | Voxelized tensor from uniaxial material + orientation field | Chromatix `polarized_multislice_thick_sample(field, potential_stack[d,h,w,3,3], ...)` **[V]**. PTI (Yeh et al., Nat. Methods 2024) assumes uniaxial symmetry per voxel **[V]** [PTI](https://www.nature.com/articles/s41592-024-02291-w) |
| Lorenz–Mie | Sphere list: center, radius (or shell radii), complex n(λ), medium n(λ) | Analytic `Sphere` / `LayeredSphere` only | DeepTrack2 `MieSphere` and `MieStratifiedSphere` produce a `ScatteredField` that is summed coherently **[V]**. PyMieDiff is a differentiable PyTorch implementation for multilayer spheres (`Particle.get_mie_coefficients()`, `get_angular_scattering()`, near fields) **[V]** [PyMieDiff](https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html) |
| T-matrix / multi-particle coupling | Particle geometry (axisymmetric for EBCM), orientation, material; T-matrices; positions | Analytic primitive plus an externally computed T-matrix. Rotations and translations of a T-matrix are differentiable even if its computation is not | I found no mature differentiable PyTorch T-matrix code. TorchGDM (PyTorch, autograd, GPU) mixes discretized structures with effective-dipole / global-polarizability models **[V]** [TorchGDM](https://arxiv.org/abs/2505.09545). A 2025 JQSRT paper proposes a T-matrix data format **[V, title only]** |
| DDA / coupled dipoles | Dipole lattice with polarizability from ε (Clausius–Mossotti) | Voxel occupancy × material | TorchGDM **[V]** |
| Widefield / confocal / light-sheet / TIRF fluorescence | Emitter density volume *or* point list, species (spectra, brightness), excitation field | Points stay points (k-space phase ramps / NUFFT). Continuous labels become density voxels | Voxelizing points destroys sub-voxel position information. DeepTrack's `PointParticle` is "single voxel scaled by voxel volume" **[V]** |
| SMLM / tracking | Per-frame point list: xyz, photons, on/off state, species, dipole | `EmitterSet`-like struct | DECODE's `EmitterSet(xyz, phot, frame_ix, id, …)` and `LooseEmitterSet(t0, ontime, intensity)` with `_distribute_framewise()` **[V]** [DECODE emitter.py](https://raw.githubusercontent.com/TuragaLab/DECODE/master/decode/generic/emitter.py) |

Takeaway: no single representation serves all of these. The sample model has to be **polymorphic**, and **conversion is a first-class subsystem**.

---

## 2. Survey of candidate representations

### 2.1 Voxel grids (Δn, κ, emitter density, tensor)
- **Strengths.** Every grid solver consumes them directly. They represent anything, including measured tomograms, OpenOrganelle/COSEM segmentations (microsim's `CosemLabel` **[V]**) and learned volumes. Gradients with respect to voxel values are trivial.
- **Weaknesses.**
  - Memory. Complex64 at 512×512×256 is 512 MB per wavelength per batch element **[R, arithmetic]**, and BPM autograd stores about one field per slice on top of that.
  - They are resolution-locked, so resampling needs anti-aliasing.
  - Geometric parameters (position, size, orientation) are not natively learnable. They are only learnable through a differentiable voxelizer.
- **Placement and instancing.** `torch.nn.functional.grid_sample` with affine transforms is differentiable with respect to the transform. It aliases when minifying, so it needs a mip pyramid or a Fourier-domain resample.
- **Verdict.** Essential as (a) the *lowered* form for grid solvers and (b) a user-supplied `VoxelField` component with its own bbox and transform. It should not be the canonical scene representation.

### 2.2 Analytic primitives (sphere, ellipsoid, spheroid, cylinder, capsule/rod, box, superquadric)
They are cheap, exactly parameterized and learnable. Many have closed-form Fourier transforms, and spheres feed Mie directly. Form factors of the indicator function, with the convention F(k)=∫_V e^{−ik·x} dx, are listed below. **[R]**, textbook results, and I checked the sphere numerically **[M]**:

| Shape | F(k) |
|---|---|
| Ball, radius R | `4π(sin kR − kR cos kR)/k³ = V·3 j₁(kR)/(kR)`. Checked against the FFT of an 8×-supersampled voxelization: 0.12% max relative error below half Nyquist **[M]** |
| Ellipsoid x = A u + c (u in unit ball, A = R·diag(a,b,c)) | `|det A| · e^{−ik·c} · F_ball,R=1(‖Aᵀk‖)` |
| Cylinder, radius R, length L, axis ê | `πR²L · [2J₁(k⊥R)/(k⊥R)] · sinc(k∥L/2)`, with sinc x = sin x/x |
| Any solid of revolution with profile r(s) (capsule, spheroid, rod with hemispherical caps, "pear" bacteria) | `∫ ds e^{−ik∥s} πr(s)² · 2J₁(k⊥r(s))/(k⊥r(s))`, a **1D quadrature**. Exact up to quadrature error, and differentiable with respect to the profile parameters |
| Box, sides a_i | `∏ a_i sinc(k_i a_i/2)` |
| Anisotropic Gaussian, weight w, mean μ, covariance Σ | `w · e^{−ik·μ} · e^{−½ kᵀΣk}` |
| Point | `e^{−ik·r}` |
| Polyhedron | Divergence theorem: `F(q) = Σ_faces (−i q·n_f/q²) ∫_f e^{iq·r} dS` (sign depends on convention). Stokes' theorem reduces each face to edge sums. The removable singularities need series expansions. Wuttke gives a numerically stable algorithm, implemented in BornAgain **[V]** [arXiv:1703.00255](https://arxiv.org/abs/1703.00255), [J. Appl. Cryst. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8056765/) |

Composite cells (for example a capsule with an ellipsoidal nucleus and a textured interior) are *groups* of primitives in a hierarchy (§6), not a new primitive type.

**Verdict:** these should be the backbone of v1.

### 2.3 Signed distance functions and CSG
- **Exact SDFs** exist for the sphere, box, capsule, cylinder and torus.
- **Ellipsoids and superquadrics** only have bounds. However, the first-order distance `d ≈ f(x)/‖∇f(x)‖` for an implicit function `f` (f<0 inside) is accurate near the surface. Near the surface is the only place an occupancy ramp needs it **[R]**.
- **CSG** is cheap on SDFs: union = min, intersection = max, difference = max(a, −b), plus smooth-min variants. These are only approximate SDFs away from the surface, which is again fine near it.
- **Occupancy from an SDF:** `occ = clamp(0.5 − d/h, 0, 1)`, a linear ramp that approximates the voxel partial-volume fraction. A sigmoid `σ(−d/τ)` also works. On a sphere (R=0.6 µm, h=50 nm) the ramp gives dV/dR = 4.525 against a true value of 4.524, and d(centroid)/dx0 = 1.004 **[M]**. With τ = h/4 the sigmoid biases volume up by 0.4% on a convex surface **[M]**.
- **Verdict:** SDFs are the universal *escape hatch*. Any user `torch` function `sdf(x)` becomes a differentiable, anti-aliased component. They have no closed-form FT.

### 2.4 Triangle meshes
- **Differentiable renderers:**
  - nvdiffrast: rasterize, interpolate, texture, antialias. Its antialias op supplies the visibility gradients that point-sampled coverage cannot. It is CUDA-only and licensed under the **NVIDIA Source Code License**; commercial use needs a separate license, which is a problem for an MIT/BSD DeepTrack dependency **[V]** [nvdiffrast](https://nvlabs.github.io/nvdiffrast/).
  - PyTorch3D (BSD) has SoftRas-style mesh rasterization and the Pulsar sphere renderer **[R]**.
  - Kaolin (Apache-2.0 **[R]**) has differentiable `FlexiCubes` and `marching_tetrahedra` (SDF→mesh). Its `trianglemeshes_to_voxelgrids` and `gs_to_voxelgrid` are not documented as differentiable, and `gs_to_voxelgrid` is explicitly "not differentiable" **[V]** [Kaolin conversions](https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.conversions.html).
- **Why they are the wrong abstraction here:** these tools render *surfaces with occlusion* onto a camera. Microscopy needs *volumes* (RI, density) or *transmission through* them.
- **Mesh → volume paths that are differentiable:**
  - (a) Generalized winding number, a sum of per-triangle solid angles (Jacobson 2013, fast version Barill 2018) **[R]**. It is smooth in the vertices away from the surface. For anti-aliasing, combine it with the unsigned distance into a signed distance and use the ramp.
  - (b) The polyhedron form factor (Wuttke). This is exact, but costs O(F·N_k), which is only feasible for small meshes or coarse k-grids.
  - (c) Slice-wise 2D polygon coverage.
- **Verdict:** v2. Useful for imported cell shapes and for mesh-deforming simulations (vertex trajectories).

### 2.5 Point emitters
SMLM, tracking and sub-diffraction labels need exact sub-voxel positions.
- **Render without voxelizing.** Use phase ramps `e^{−ik·r_j}` on the pupil/OTF grid, a direct PSF evaluation per emitter, or a spline-interpolated experimental PSF.
- **Many emitters.** NUFFT type-1 turns point sets into uniform frequencies. `pytorch-finufft` provides `finufft_type1` and `finufft_type2` with backward in *both values and positions*, and GPU support via cufinufft **[V]** [pytorch-finufft docs](https://flatironinstitute.github.io/pytorch-finufft/), [install](https://flatironinstitute.github.io/pytorch-finufft/installation.html).
- **Required attributes:** xyz, photons or brightness, species, dipole orientation plus mobility (for vectorial and polarization PSFs), per-frame on/off state and an id.

### 2.6 Gaussian mixtures / 3D Gaussian splatting
- **Why they fit:**
  - The FT is analytic, `w e^{−ik·μ} e^{−½kᵀΣk}`.
  - Projection along z gives an analytic 2D Gaussian.
  - Convolving with a Gaussian PSF approximation gives a Gaussian with covariance Σ+Σ_psf. That is an **exact, analytic image in real space** for a "fast" fluorescence tier.
  - Gaussians are naturally band-limited, so they are self-antialiasing when σ ≳ 0.5h.
- **Do not adopt the 3DGS rendering model.** 3DGS alpha-composites with occlusion ordering. Fluorescence and weak scattering are *additive and non-occluding*. R²-Gaussian identified an "integration bias" when 3DGS projection is reused for tomography **[V]** [R²-Gaussian](https://openreview.net/forum?id=fMWrTAe5Iy).
- **Recent uptake in tomographic and optical inverse problems:** GS-DOT (absorption as a sum of anisotropic Gaussians), Gaussian-splatting holography, CryoSplat, and 3D Gaussian reconstruction for Fourier light-field microscopy **[V, abstracts only]**: [GS-DOT](https://arxiv.org/html/2604.23675), [GSH](https://arxiv.org/html/2509.20774), [CryoSplat](https://arxiv.org/pdf/2508.04929).
- **Verdict:** v1 primitive (`Gaussian`, batched). It is also the natural *learnable free-form* representation for fitting samples to data. Additive semantics only.

### 2.7 Neural implicit fields
- A coordinate MLP maps x to (Δn, κ) or density, as in DeCAF (Nat. Mach. Intell. 2022), which represents a continuous complex RI field from intensity-only diffraction tomography **[V]** [DeCAF](https://www.nature.com/articles/s42256-022-00530-3).
- To lower one, evaluate it at voxel centers. Band-limiting comes from the positional-encoding bandwidth (or Mip-NeRF-style integrated encodings) **[R]**.
- Mostly useful for *reconstruction* and for learned sample *generators*.
- **Verdict:** it costs nothing to support as a generic `FieldComponent(fn: x→values, bbox, transform)`, which is the same interface as `VoxelField`. No special machinery is needed.

### 2.8 Procedural textures (cell interiors)
- **Gaussian random fields.** Take white noise ε in k-space, multiply by √P(k), then inverse FFT. P can be Matérn / von Kármán, `(1+(kℓ)²)^{−(ν+3/2)}`, the fractal-tissue RI spectrum model **[R]**.
- **Why they suit learning:** they are reparameterizable. With ε fixed, gradients flow to the amplitude, ℓ and ν, so texture statistics become learnable distribution parameters.
- **Usage:** mask by the parent occupancy, n = n_cell + δn_GRF·occ_cell.
- Perlin/simplex noise and "MatsLines"-style procedural fibres (microsim **[V]** [microsim](https://github.com/tlambert03/microsim)) are other `FieldComponent`s.

### 2.9 Time-dependent samples
Parameters become functions of time. There are four patterns:
1. **Trajectories** `[B, T, N, 3]` from SDE integrators with reparameterized noise: Brownian `Δr = √(2DΔt)·ε`; OU/confined; drift. These are differentiable with respect to D, v and the confinement.
2. **Rotational diffusion** of anisotropic particles and dipoles.
3. **Deformations:**
   - time-varying affine per primitive (for example cell length(t) for growth);
   - backward-warp displacement fields `φ(x,t)` applied through `grid_sample` for voxel and neural fields;
   - vertex trajectories for meshes.
4. **Photophysics time series:** blinking and bleaching.

Motion blur means integrating over the exposure with S sub-steps. The renderer should treat T (and the sub-steps) as a batch axis. Identity must persist across time (an `id` per instance) for tracking labels. Neural space-time models for dynamic multi-shot imaging (Cao & Waller, Nat. Methods 2024) show the same "parameters as functions of t" pattern in reconstruction **[R]**.

---

## 3. Differentiable, anti-aliased voxelization (the moving-edge problem)

### 3.1 The problem
For a quantity `Q(θ) = ∫_{Ω(θ)} f dx`, the Reynolds transport theorem gives `dQ/dθ = ∫_Ω ∂f/∂θ dx + ∮_{∂Ω} f (v·n) dS`, where v = ∂x_boundary/∂θ. The second term is carried *entirely by the boundary*.
- Point-sampling a hard indicator discards the boundary term, so gradients with respect to position, size, orientation and shape are zero almost everywhere.
- Supersampling followed by average pooling (DeepTrack2's `upsample` → `_antialias_volume` via `AveragePooling`) reduces aliasing, but the result is still piecewise constant in θ. **The gradient is still zero.**
- The same issue shows up in differentiable graphics as the visibility/silhouette problem, handled by edge sampling (redner), reparameterization (Mitsuba), soft rasterization, and nvdiffrast's analytic antialias op **[R; nvdiffrast V]**.

For volumes, the fix is simpler than in graphics: **render the band-limited (pre-filtered) object instead of the object itself.** Pre-filtering turns the boundary term into a smooth volume term.

### 3.2 Methods

| Method | How | Gradients | Aliasing | Cost |
|---|---|---|---|---|
| Hard threshold (DeepTrack2 now) | `mask = f(x) <= 1` | **0** | High (≈2% in-band spectral error, normalized by F(0), for R=0.3 µm, h=50 nm) **[M]** | O(voxels) |
| Supersample s× and average-pool | partial-volume estimate | **0** | Low (0.37–0.40%) **[M]** | s³× more |
| SDF ramp `clamp(0.5−d/h,0,1)` | first-order partial volume | Good (dV/dR 4.525 vs 4.524) **[M]** | 0.62% **[M]** | O(voxels) |
| Closed-form Gaussian-blurred ball | `½[erf((R−r)/√2σ)+erf((R+r)/√2σ)] − σ/(r√(2π))·[e^{−(R−r)²/2σ²} − e^{−(R+r)²/2σ²}]` | Exact for the blurred object (4.5239 = 4.5239) **[M]** | 0.54% at σ=0.3h; 1.10% at σ=0.5h (in-band attenuation dominates) **[M]** | O(voxels), one erf and two exp per voxel |
| k-space form factor | `F(k)·e^{−ik·c}·W(k)` → iFFT | Exact | 0 in-band by construction, but **Gibbs**: min −0.10, max 1.17 (negative densities) **[M]** | O(N_obj·N_k), or NUFFT |
| Gaussian-mixture fit of the shape | analytic | exact | self-band-limited | O(#Gaussians·voxels_local) |

I checked the blurred-ball formula against the inverse FFT of `F_ball(k)·e^{−½k²σ²}`: the maximum absolute difference is 3.6×10⁻⁸ **[M]**. At r→0 it reduces to the χ₃ CDF, as it should. Every non-hard method recovers d(centroid)/dx0 = 1.000 to 1.004 **[M]**.

### 3.3 Choosing the band-limit
The optics sets the relevant sample bandwidth, not the grid.
- **Transmission imaging:** the transverse band is |k⊥| ≤ k0(NA_ill + NA_det) ≤ 2k0·NA. For λ = 0.5 µm and NA 1.4 that edge sits at 0.56·k_Nyq on a 50 nm grid.
- **Pre-filter width:** a Gaussian with σ≈0.3h matches the box filter's variance (h²/12). At that band edge its attenuation is about 13%; at σ=0.5h it is about 32% **[R, arithmetic]**. So use **σ ≈ 0.3h by default**.
- **Linear imaging models:** the known pre-filter can be divided out of the transfer function.
- **Rule of thumb:** oversample the sample grid 1.5–2× relative to the optical band, then pre-filter at σ≈0.3h.
- **Where k-space form factors fit:** they are best for *linear solvers that never voxelize*: Born/Rytov, fluorescence via OTF, and 2D projections via the k_z=0 slice. They are unsuitable as a default density voxelizer because Gibbs ringing makes densities negative. Negative densities are harmful for Poisson noise, but harmless for Δn in linear models.

### 3.4 Cost of many objects
The approach below is patch-based. Each object gets a local P³ patch around its integer anchor. Occupancy is evaluated for all objects in a kind group at once as an `[N, P³]` tensor, then `index_add_` places it into the grid.

Measured on a 256×256×64 grid with P = 24, forward plus backward, unfused PyTorch **[M]**:

| Spheres | Time | Peak memory |
|---|---|---|
| 100 | 4.6 ms | 0.21 GB |
| 1000 | 37 ms | 1.8 GB |
| 10000 | 0.99 s | 17.9 GB |

- **Memory is the bottleneck.** Autograd keeps every elementwise intermediate of `[N, P³]`.
- **Fix:** a custom `autograd.Function` or a Triton kernel that recomputes in backward and accumulates `∂L/∂R_j, ∂L/∂c_j` per object. Memory then becomes O(grid + N). I expect 5–10× speedups, but that is an estimate, not a measurement.
- **Culling:** bounding boxes plus the blur margin plus the PSF/defocus margin cull objects outside the field of view. Empty z-ranges can be merged into a single propagation step in BPM.
- **Alternative for identical shapes (monodisperse beads, fixed-size emitters):** `F_shape(k)·NUFFT₁({c_j}, w_j)`. This costs O(N_k log N_k + N), and gradients with respect to positions come from pytorch-finufft.
- **Polydisperse populations:** use the real-space patch path, bucketed by size so that P stays tight.

---

## 4. Capabilities, lowerings and the conversion graph

### 4.1 Capability protocol (per primitive kind, batched)
```python
class Component(Protocol):                    # struct-of-arrays over [B, N]
    kind: str
    def bbox(self) -> Tensor                   # [B,N,2,3] world-space, excluding blur margin
    # optional capabilities; presence is introspected by the planner
    def occupancy(self, x: Tensor, sigma: float) -> Tensor        # band-limited, differentiable
    def sdf(self, x: Tensor) -> Tensor                             # exact or first-order
    def form_factor(self, k: Tensor) -> Tensor                     # complex, analytic
    def project(self, xy: Tensor, axis=2) -> Tensor                # thickness / column density
    def sample_points(self, n, where: Literal["volume","surface"], gen) -> Tensor
    def mie_spec(self) -> SphereList                               # spheres/layered spheres only
    def t_matrix(self, wavelength, lmax) -> Tensor                 # future: via external solver
```

### 4.2 Lowered representations ("views") that solvers consume
- `OccupancyChannels(grid, [M])`: one channel per material, not per object.
- `RIVolume(grid, λ)`: complex Δn or Δε, possibly `[W, Z, Y, X]`.
- `EpsTensorVolume`: 6 unique components.
- `EmitterDensity(grid, [S species])`.
- `EmitterList`.
- `SphereList`.
- `KSpaceSpectrum(kgrid)`.
- `ProjectedMaps(OPL, absorbance, column density)`.
- `Labels`: see §9.

### 4.3 Conversion registry
```python
@lowering(src=Sphere, dst=OccupancyChannels, exactness="bandlimited", cost=patch_cost)
def sphere_to_occupancy(sph, grid, policy): ...
@lowering(src=Ellipsoid, dst=SphereList, exactness="lossy")   # equal-volume sphere; only if the policy allows
@lowering(src=SDFComponent, dst=OccupancyChannels, exactness="approx(first-order)")
@lowering(src=AnyWithFormFactor, dst=KSpaceSpectrum, exactness="exact")
@lowering(src=EmitterList, dst=EmitterDensity, exactness="lossy(subvoxel)")
```
- **Planner.** Each solver declares its accepted views in order of preference, for example `Mie: [SphereList]` plus `fallback_for_rest: RIVolume → Born/BPM`. The planner does a shortest-path search over the conversion graph for each component group. The graph is small, so enumeration is enough.
- **Fidelity policy.** A `FidelityPolicy` sets the allowed exactness (for example, "no lossy"), the grid oversampling, the filter (`box | gaussian(σ) | ideal+window`), the quadrature orders, the dispersion sample count W, the number of time sub-steps, and the photophysics mode. An impossible plan raises an error that says which component blocked it.
- **Hybrid plans.** Example: spheres to Mie fields, everything else to a Born/BPM volume, fields superposed. These are legitimate single-scattering approximations, but each plan should carry a *warning tag* that records the neglected coupling.

---

## 5. Material models

- **`Material` (an nn.Module; its parameters may be learnable).**
  - `n(λ) → complex`, broadcastable over a wavelength tensor.
  - Subclasses: `Constant`, `Cauchy` (n = A + B/λ² + C/λ⁴), `Sellmeier` (n² = 1 + Σ B_iλ²/(λ²−C_i)), `Tabulated` (interpolated refractiveindex.info-style data), `DrudeLorentz` (metals such as gold nanoparticles for iSCAT/dark-field).
  - The absorption coefficient is α = 4πκ/λ.
  - Chromatix uses (dn, absorption) per slice **[V]**; PyMieDiff uses `MatConstant` / `MatDatabase` **[V]**. A `MatDatabase`-like tabulated class covers both.
  - **Biological convenience:** `n = n_medium + α_c·C`, with dn/dc ≈ 0.18–0.19 mL/g for protein **[R]**. This lets users specify dry-mass concentrations, which is what QPI measures.
  - **Typical values [R, approximate]:** water 1.333, cytoplasm 1.36–1.39, lipid droplets about 1.46–1.48, silica about 1.45, polystyrene about 1.59.
- **Anisotropic materials.** `Uniaxial(n_o, n_e, axis)` gives ε = n_o²I + (n_e²−n_o²)aaᵀ. The axis can be a per-instance vector or a spatial orientation field (fibres, spindles). The lowered view is the `[..., 3, 3]` tensor that chromatix's polarized multislice consumes **[V]**. Defer to v2 together with vectorial solvers, but reserve the slot in the data model now.
- **`Fluorophore` / species.**
  - Spectral and photometric fields: excitation and emission spectra, extinction coefficient, quantum yield, lifetime. microsim's `Fluorophore(excitation_spectrum, emission_spectrum, extinction_coefficient, quantum_yield, lifetime_ns)` is a good template, including FPbase lookup **[V]** [microsim sample.py](https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/sample/sample.py). Brightness is ε·QY.
  - `photophysics`, with three modes:
    - `None`
    - `Expected(k_on, k_off, k_bleach)`: a deterministic mean occupancy from the 2- or 3-state master equation. It is differentiable with respect to the rates and gives the fast tier.
    - `Stochastic(...)`: Gillespie / discrete-time Markov sampling. It is not differentiable with respect to the rates, so use score-function gradients or a Gumbel relaxation.
  - `dipole`: `Isotropic | Fixed(orientation) | Wobble(cone_angle)`. Needed for vectorial and polarization-resolved PSFs. Anisotropy follows from this.
- **Labeling is its own object.** `Labeling(species, target=component_or_group, mode=volume|surface|points(n or density), concentration)`. microsim's `FluorophoreDistribution(distribution, fluorophore, concentration)` is the same idea **[V]**. One geometry then yields both Δn (brightfield/QPI) and label density (fluorescence), which is how the simulator becomes "more than the sum of its parts".
- **Environment / mounting stack.** The scene root holds `LayeredMedium([Immersion(n_i, t_i), Coverslip(n_g, t_g), SampleMedium(n_s)])`, with z = 0 at the coverslip/sample interface. Optics consumes it for Gibson–Lanni / vectorial depth aberrations and TIRF evanescent excitation. The sample consumes it for n_medium(λ). One source of truth avoids the common inconsistency between the "medium RI" in optics and in scatterers.

---

## 6. Scene organization

- **Hierarchy.** `Scene → Groups → Components`. Each node has a local transform: translation, rotation as a quaternion or the continuous 6D representation for learning, and anisotropic scale. A world transform is composed on flattening. A "cell" is a group: capsule body, ellipsoidal nucleus, GRF interior texture masked by the body, membrane labeling on the body surface. Groups are templates, and instancing a group replicates its component batches.
- **Canonical flattened form (what solvers see).** For each primitive kind, one struct-of-arrays batch of shape `[B, N_max, …]`:
  - `center`, `rotation`, `size…`
  - `material_id` (int)
  - `priority` (int, from depth in the hierarchy)
  - `parent` (int)
  - `presence` (float in [0,1]; 0 means padding)
  - `id`, for time/tracking

  Padding plus a mask is preferable to nested/jagged tensors because it keeps `torch.compile`/`vmap` friendly and lets presence be relaxed (§7). Python loops happen over *kinds*, not objects.
- **Instancing prototypes** (a voxel or neural `FieldComponent` reused N times): an affine `grid_sample` per instance in real space, or `F_proto(Aᵀk)·|det A|·e^{−ik·c}` in k-space.
- **Overlap semantics.** Make this explicit per quantity:
  - **Material-like quantities** (n, ε, tensor): **priority "over" compositing** of occupancy channels, with the highest priority first:
    `a_m = occ_m·rem; rem ← rem·(1−occ_m); ε = Σ a_m ε_m + rem·ε_medium`.
    Children (higher priority) replace the parent's material. Siblings with equal priority are normalized when Σocc > 1. Mixing is linear in ε. That is the Wiener upper bound, and it matches the scattering potential used by Born/BPM/DDA. To first order in Δn it is the same as mixing linearly in n **[R]**.
  - **Density-like quantities** (fluorophores, dry-mass add-ons): add them.
  - **Legacy mode:** DeepTrack2 superposes `(n_i − n_medium)·mask` additively, so overlapping particles double their contrast **[V]** [DeepTrack2 optics.py](https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/optics.py). Keep this as `compose="add"`, but it should not be the default.
- **Material channels vs baked values.**
  - With M ≲ 8 materials, lower to `OccupancyChannels[M]` once. Evaluate `ε(λ)` per wavelength by a weighted sum, so dispersion costs M multiply-adds per voxel per λ, not a re-voxelization. Material parameters stay learnable without touching geometry.
  - When every instance has its own RI (a polydisperse bead suspension with random n_i), use a `BakedContrast` channel Σ occ_i·(n_i(λ0) − n_m) plus a per-family dispersion approximation.
- **Bounding boxes.**
  - The per-component bbox is expanded by the blur margin (about 4σ).
  - For culling, expand further by the PSF lateral extent plus defocus spread, so objects just outside the field of view still scatter or blur into it.
  - The z-extent of all bboxes drives slice compaction in BPM and the choice of non-empty planes for fluorescence convolution. DeepTrack2 already crops empty slices (`crop_empty=True`) **[V]**.

---

## 7. Randomization and learnable distribution parameters

DeepTrack2 samples properties through Python callables (`position=lambda: ...`) re-evaluated on `update()`. That is flexible but opaque to autograd. For gradoscopy:

- **`Param` is one of three kinds:**
  - a fixed tensor;
  - a learnable `nn.Parameter`;
  - `Random(dist)`, where `dist` is an `nn.Module` wrapping `torch.distributions` with learnable parameters.
- **`SceneSampler(nn.Module).sample(B, generator) → Scene`.** The sampled Scene is plain tensors. Randomness is explicit through `torch.Generator`, which gives reproducibility and per-worker streams.
- **Gradient estimators, in order of preference:**
  1. Reparameterization (`rsample`). This covers Normal, LogNormal, Uniform with respect to its bounds, MultivariateNormal, and Gamma/Beta/Dirichlet via implicit reparameterization in PyTorch **[R]**.
  2. Relaxations (`RelaxedBernoulli`, `RelaxedOneHotCategorical`, straight-through).
  3. Score function / REINFORCE with baselines, using the exposed `log_prob`.

  Each `Random` records which estimator applies, so the pipeline can refuse an impossible gradient or warn about it.
- **Discrete structure.**
  - *Particle count:* use `N_max` slots with `presence ~ RelaxedBernoulli(p)`, or hard presence plus straight-through. Presence multiplies occupancy and emitter photons, which gives a gradient to the learnable density p. A Poisson count with `N_max` truncation and the score-function estimator is the unbiased alternative.
  - *Choice among kinds* (sphere vs ellipsoid): relaxed categorical mixtures of occupancy.
- **Constraints such as non-overlap.**
  - Rejection sampling keeps the sample reparameterized but drops the gradient of the acceptance probability, which biases it.
  - Preferred: a few unrolled steps of a differentiable repulsion relaxation (gradient descent on an overlap energy), or simply allowing overlaps with principled compositing (§6).
- **What gets learned.**
  - Distribution parameters: size mean/σ, D, RI mean, texture spectrum, label density, blinking rates.
  - Optics parameters. The loss can be downstream task loss, MMD or adversarial matching of real images, or direct fitting.
  - Meta-Sim learned scene-graph attributes with MMD, but had to use **finite-difference** gradients through a non-differentiable renderer **[V]** [Meta-Sim](https://arxiv.org/abs/1904.11621). gradoscopy's differentiable voxelization (§3) makes those gradients exact and cheap, which is a concrete advantage worth stating in the pitch.

---

## 8. Time

- Add a `T` axis (and optional exposure sub-steps S) to the SoA batches. `scene.at(t)` is a view, and renderers vectorize over T like B.
- **Built-in processes:** Brownian, OU/confined, directed, rotational diffusion, growth (length(t)) and a user callable. All use reparameterized noise.
- **Photophysics:** state sequences `[B,T,N]` (stochastic mode) or expected occupancy (fast mode).
- **Bleaching:** `b(t) = b0·exp(−k_b∫I_exc dt)`, which couples to the illumination.
- **Deformation fields:** a `Warp` modifier applied to any `FieldComponent`/SDF, evaluating f(φ⁻¹(x,t)).

---

## 9. Ground truth is a lowering too

DeepTrack's value is paired data. The same capabilities should produce labels on any grid: binary or soft masks (occupancy), instance-id maps, SDF/distance maps, centroids (from parameters directly), RI and density maps, and per-frame emitter lists with `id`. Labels are rendered with the same transforms, time axis and `presence`, so they stay consistent with the images by construction. This should be `scene.render_labels(grid, kinds=[...])`, not per-feature ad-hoc code.

---

## 10. Implications for gradoscopy's architecture

1. **The canonical sample is continuous and physical.** Use µm and world frame, with z along the optical axis and z = 0 at the coverslip interface. Pixels exist only in the DeepTrack adapter. Voxel grids are *views* requested with `(Grid, wavelengths, FidelityPolicy)`. Never store a grid as "the sample" unless the user supplied one.
2. **Separate geometry, material and labeling:** `Component(geometry) + Material + Labeling(species)`. This one decision makes brightfield/QPI and fluorescence of the *same* virtual specimen consistent, and makes materials learnable independently of shapes.
3. **v1 primitive set (batched SoA, all with `bbox` and `occupancy`, most with `form_factor`):**
   - `Sphere`, `LayeredSphere` (Mie-capable)
   - `Ellipsoid` / spheroid
   - `Cylinder`
   - `Capsule`, a solid of revolution with FT by 1D quadrature
   - `Box`
   - `Gaussian` (anisotropic)
   - `PointEmitters`
   - `FieldComponent`, for user voxels, `torch` callables and neural fields, with bbox and transform
   - `SDFComponent`, the escape hatch that also covers CSG
   - `RandomField` (GRF texture)
   - `Group` / hierarchy with priority compositing

   Defer to v2: meshes (winding number / polyhedron FT), uniaxial ε-tensors (with vectorial solvers), deformation warps. Defer to v3: T-matrix interop.
4. **Default voxelizer:** real-space, patch-based, analytic band-limited occupancy.
   - Use the closed-form blurred ball for spheres, and the first-order SDF ramp for other shapes. The Gaussian-blurred closed forms for ellipsoids are left as future work.
   - σ = 0.3h by default. `box` (supersample) and `ideal` (k-space) are optional filters.
   - Implement it as a custom autograd kernel (Triton) early. Unfused autograd hits 18 GB at 10k spheres **[M]**.
   - Never ship a hard-threshold path without a straight-through/soft surrogate. Hard thresholding is DeepTrack2's current behaviour, and it gives zero geometry gradients **[M]**.
5. **Linear-model solvers use form factors directly.** Born/Rytov, OTF-based fluorescence and projection tiers call `form_factor(k)` (and NUFFT for points and monodisperse groups) and skip voxelization entirely. This is the "fast yet exact within the linear model" tier. It also makes subpixel position and size gradients analytic.
6. **Build a conversion registry with exactness tags plus a planner.** Solvers declare accepted views in preference order. The `FidelityPolicy` bounds allowed exactness and resolutions. The planner produces hybrid plans (for example Mie spheres plus a BPM remainder) that carry explicit approximation warnings. This gives "same user-facing description, different back-ends".
7. **Compositing rules:**
   - Material-like quantities: priority "over", linear in ε.
   - Density-like quantities: additive.
   - Legacy DeepTrack mode: `compose="add"`.
   - Lower materials as occupancy channels (M small) so dispersion and learnable RI cost M FMAs per voxel per λ.
8. **Batching contract:** `[B, (T,) N_max, …]` + `presence`. Group by kind. No per-object Python. Bounding boxes drive culling (including the PSF margin), BPM slice compaction and patch sizes (bucket polydisperse populations by size).
9. **Randomness is a module.** `SceneSampler(nn.Module)` built from `Param`s, where a param is fixed, learnable or `Random(dist)`. Each `Random` records its gradient estimator (reparameterized, relaxed or score-function). Discrete counts use relaxed presence. The sampled Scene is pure tensors plus a seed, so it can be cached, replayed, rendered as labels, and checkpointed for BPM memory recompute.
10. **Emitters are never voxelized by default.** `EmitterSet` (xyz, photons, species, dipole, state[T], id) renders through phase ramps / NUFFT / PSF evaluation. The photophysics modes are `Expected` (differentiable) and `Stochastic` (score-function or relaxed).
11. **The Environment stack (immersion/coverslip/medium) is a shared scene-root object** read by both sample and optics. Declare it once.
12. **Compatibility adapter.** Map DeepTrack2's `Sphere`/`Ellipsoid`/`Ellipse`/`PointParticle`/`MieSphere`/`MieStratifiedSphere` and its `value`/`position_unit`/`upsample` semantics onto these components in a thin `deeptrack` shim. `upsample` becomes a filter setting. `value` becomes a Material, or an intensity via Labeling.

Suggested external dependencies:
- pytorch-finufft (optional extra)
- PyMieDiff, as a reference or an optional backend. Check its license before vendoring **[license unverified]**.
- Avoid nvdiffrast (license). Kaolin/PyTorch3D are optional, only for mesh I/O and mesh↔SDF conversion in v2.

---

## Sources
- DeepTrack2 scatterers (develop): https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/scatterers.py ; optics: https://raw.githubusercontent.com/DeepTrackAI/DeepTrack2/develop/deeptrack/optical/optics.py
- Chromatix samples: https://raw.githubusercontent.com/chromatix-team/chromatix/main/src/chromatix/functional/samples.py
- microsim: https://github.com/tlambert03/microsim ; https://raw.githubusercontent.com/tlambert03/microsim/main/src/microsim/schema/sample/sample.py
- Wuttke polyhedron form factors: https://arxiv.org/abs/1703.00255 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8056765/
- Kaolin conversions: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.conversions.html ; nvdiffrast: https://nvlabs.github.io/nvdiffrast/
- pytorch-finufft: https://flatironinstitute.github.io/pytorch-finufft/
- PyMieDiff: https://uos-integrated-nanophotonics-group.github.io/MieDiff/index.html ; https://arxiv.org/html/2512.08614v1 ; TorchGDM: https://arxiv.org/abs/2505.09545
- DECODE EmitterSet: https://raw.githubusercontent.com/TuragaLab/DECODE/master/decode/generic/emitter.py
- PTI: https://www.nature.com/articles/s41592-024-02291-w ; DeCAF: https://www.nature.com/articles/s42256-022-00530-3
- SyMBac: https://pmc.ncbi.nlm.nih.gov/articles/PMC9710168/ ; Meta-Sim: https://arxiv.org/abs/1904.11621
- Gaussian-primitive tomography: https://openreview.net/forum?id=fMWrTAe5Iy ; https://arxiv.org/html/2604.23675 ; https://arxiv.org/html/2509.20774 ; https://arxiv.org/pdf/2508.04929
