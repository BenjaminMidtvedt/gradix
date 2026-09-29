"""Explicit random keys: the counter hash, keyed uniforms, and keyed Gaussian and Poisson draws.

gradix owns no random state (§6.3). A key is either

- a Python ``int``: *batch-keyed*. Draws come from a :class:`torch.Generator` seeded from the key
  and a stream name (:func:`generator`); they reproduce for the same batch;
- an int64 ``Tensor[B]``: *image-keyed*. Every draw is a counter hash of (key of image i,
  stream, element index within the image), so image i's draws do not depend on the batch it is
  rendered in, nor on its position there.

The counter hash is SplitMix64 emulated in int64 arithmetic: a Weyl sequence
``seed + (j + 1)·γ`` passed through the SplitMix64 finaliser, with logical shifts emulated by
masking. The counter j is the element's flat index within the image plus an optional offset, so a
caller rendering an image in pieces (frames in chunks) passes each piece's absolute start and
gets the draws of the whole image; :func:`fold_in` derives independent keys from a key and a
number. Draws map to open-interval uniforms ``u = (k + 0.5)·2⁻ᵇ`` in float64: b = 52 for single
uniforms (so u ≤ 1 − 2⁻⁵³ < 1 exactly), b = 32 for the two halves of one hash where a pair is
needed.

Gaussian draws use the inverse CDF. Poisson draws (the spec is fixed at M0, so every
implementation, eager or fused, reproduces the same counts):

1. λ < 10: the sequential inverse CDF of one 53-bit uniform, for 24 steps;
2. λ ≥ 10: transformed rejection (PTRS, Hörmann 1993) with 4 attempts, each on one hash split
   into two 32-bit uniforms;
3. the rare elements left over (the inverse CDF beyond 24 steps, or all 4 attempts rejected) are
   gathered into a static-size buffer and drawn exactly by the inverse CDF: the regularised
   incomplete gamma function gives P(X < lo), the pmf recursion extends it over 33 counts from
   ``lo``, and the few draws outside that window finish by bisection. An unfinished search
   continues with its *own* uniform; rejected elements use a fresh one.

Non-finite rates give NaN counts and never touch the shared buffers, so one broken image cannot
change another's draws. If more elements need the buffers than they hold (a probability far
below 10⁻⁹ per call for finite rates), the unfinished ones are NaN, never silently wrong.

Every loop has a static trip count, so the samplers make no host synchronisation. The draws are
bit-exact for a given gradix version on a given device type; CPU and CUDA agree bit for bit on
uniforms and Poisson counts, and within a few ulp (≈10⁻¹⁵) on normals, whose inverse CDF is
evaluated by different libraries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import torch
from torch import Tensor

from gradix._core.errors import StructureError

__all__ = [
    "Key",
    "check_key",
    "fold_in",
    "generator",
    "mix64",
    "normal",
    "poisson",
    "ptrs",
    "stream_id",
    "uniform",
]

Key: TypeAlias = "int | Tensor"
"""A random key: an ``int`` (batch-keyed) or an int64 ``Tensor[B]`` (image-keyed)."""

_MASK64 = (1 << 64) - 1


def _i64(c: int) -> int:
    """Map an unsigned 64-bit constant into signed int64 range."""
    c &= _MASK64
    return c - (1 << 64) if c >= 1 << 63 else c


_GAMMA = _i64(0x9E3779B97F4A7C15)
_M1 = _i64(0xBF58476D1CE4E5B9)
_M2 = _i64(0x94D049BB133111EB)
_TWO_M52 = 2.0**-52

POISSON_ICDF_BELOW = 10.0
"""Rates below this use the sequential inverse CDF; rates at or above it use PTRS."""
POISSON_ICDF_STEPS = 24
"""Steps of the inverse-CDF search; P(X > 24) < 3·10⁻⁴ for λ < 10, and those finish exactly in
the fallback buffer."""
POISSON_PTRS_ATTEMPTS = 4
"""PTRS attempts per element. One attempt accepts with probability 0.75 at λ = 10 and ≈0.89 for
λ ≥ 10³, so at most ≈0.4 % of the elements reach the fallback."""
POISSON_WINDOW = 32
"""Width of the fallback's window search: 33 consecutive counts around a start that is exact for
unfinished inverse-CDF searches (just past step 24) and a Cornish–Fisher estimate otherwise."""
POISSON_FALLBACK_STEPS = 24
"""Bisection steps of the last-resort search: exact for rates below 10⁹ (the bracket
``λ ± (40·√λ + 40)`` then spans fewer than 2²⁴ counts), far above any camera's full well."""
POISSON_FALLBACK_RATE = 0.0045
"""Worst-case fraction of elements that reach the fallback; the buffer holds this fraction plus
ten standard deviations plus 64, so it overflows with probability far below 1e-20."""


def _srl(x: Tensor, shift: int) -> Tensor:
    """Logical right shift of int64 values."""
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def mix64(x: Tensor) -> Tensor:
    """Apply the SplitMix64 finaliser to int64 values.

    Parameters
    ----------
    x : Tensor
        Values of dtype int64, any shape.

    Returns
    -------
    Tensor
        Hashes of dtype int64, same shape; the bit pattern equals the unsigned SplitMix64 finaliser.
    """
    x = (x ^ _srl(x, 30)) * _M1
    x = (x ^ _srl(x, 27)) * _M2
    return x ^ _srl(x, 31)


def _mix64_int(x: int) -> int:
    """SplitMix64 finaliser on a Python int (unsigned 64-bit arithmetic)."""
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def stream_id(name: str) -> int:
    """Hash a stream name to a signed 64-bit integer (FNV-1a, stable across runs and versions).

    Parameters
    ----------
    name : str
        Stream name, such as ``"camera/poisson"``.

    Returns
    -------
    int
        The hash, in int64 range.
    """
    h = 0xCBF29CE484222325
    for byte in name.encode():
        h = ((h ^ byte) * 0x100000001B3) & _MASK64
    return _i64(h)


def check_key(key: object, batch: int | None = None) -> int | Tensor | None:
    """Validate a key and return it in canonical form.

    Parameters
    ----------
    key : int, Tensor or None
        The key: ``None`` (no randomness), an ``int`` (batch-keyed), or an integer
        ``Tensor[B]`` (image-keyed).
    batch : int, optional
        The batch size an image key must match.

    Returns
    -------
    int, Tensor or None
        The key; an image key as an int64 tensor.

    Raises
    ------
    StructureError
        If the key has the wrong type, dtype or shape.
    """
    if key is None:
        return None
    if isinstance(key, bool):
        raise StructureError("a key cannot be a bool", fix="pass an int or an int64 Tensor[B]")
    if isinstance(key, int):
        return key
    if isinstance(key, Tensor):
        if key.is_floating_point() or key.is_complex() or key.dtype == torch.bool:
            msg = f"an image key must be an integer tensor, got {key.dtype}"
            raise StructureError(msg, fix="pass torch.arange(B) or another int64 Tensor[B]")
        if key.ndim != 1:
            msg = f"an image key must have shape [B], got {list(key.shape)}"
            raise StructureError(msg, fix="pass an int for batch-keyed draws, or a 1-D Tensor[B]")
        if batch is not None and key.shape[0] != batch:
            msg = f"the image key has {key.shape[0]} entries but the batch has {batch} images"
            raise StructureError(msg, fix="pass one key per image")
        return key.to(torch.int64)
    msg = f"a key must be an int or an int64 Tensor[B], got {type(key).__name__}"
    raise StructureError(msg)


def generator(key: int, stream: str, device: torch.device | str) -> torch.Generator:
    """Return a generator seeded from a batch key and a stream name.

    Parameters
    ----------
    key : int
        The batch key.
    stream : str
        Stream name; different streams give independent generators.
    device : torch.device or str
        Device of the generator.

    Returns
    -------
    torch.Generator
        A fresh generator. Draw from it only outside checkpointed regions (§6.3).
    """
    seed = _mix64_int(_mix64_int(key ^ stream_id(stream)) + 0x9E3779B97F4A7C15)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed & ((1 << 63) - 1))
    return gen


def fold_in(keys: Tensor, data: int | Tensor) -> Tensor:
    """Derive new image keys from keys and a number, such as a frame or an iteration index.

    Parameters
    ----------
    keys : Tensor
        Image keys of dtype int64, shape [B].
    data : int or Tensor
        The number to fold in: a Python int, or an int64 tensor broadcastable to [B].

    Returns
    -------
    Tensor
        New int64 keys [B], independent of ``keys`` and of other ``data`` values.
    """
    if isinstance(data, float) or (isinstance(data, Tensor) and data.is_floating_point()):
        raise StructureError("fold_in takes integer data (a frame or iteration index)")
    salt = torch.as_tensor(data, dtype=torch.int64, device=keys.device)
    return mix64(mix64(salt + _GAMMA) ^ keys)


def _seeds(keys: Tensor, stream: str) -> Tensor:
    """Per-image seeds of a stream: [B] int64."""
    return mix64(mix64(keys ^ stream_id(stream)) + _GAMMA)


def _bits(keys: Tensor, stream: str, shape: Sequence[int], offset: int = 0) -> Tensor:
    """Counter-hash bits [B, *shape] (int64) for element indices offset … offset + n − 1."""
    count = math.prod(shape)
    j = torch.arange(offset + 1, offset + count + 1, device=keys.device, dtype=torch.int64)
    h = mix64(_seeds(keys, stream)[:, None] + j * _GAMMA)
    return h.reshape(keys.shape[0], *shape)


def _to_unit(bits: Tensor) -> Tensor:
    """Map hash bits to a uniform in (0, 1): the top 52 bits, centred, so u < 1 exactly."""
    return (_srl(bits, 12).to(torch.float64) + 0.5) * _TWO_M52


def uniform(keys: Tensor, stream: str, shape: Sequence[int], *, offset: int = 0) -> Tensor:
    """Draw open-interval uniforms, keyed per image.

    Parameters
    ----------
    keys : Tensor
        Image keys of dtype int64, shape [B].
    stream : str
        Stream name.
    shape : sequence of int
        Per-image shape of the draw.
    offset : int, default 0
        Flat index of the first element within the image, for drawing an image in pieces.

    Returns
    -------
    Tensor
        Uniforms in (0, 1), float64, shape [B, *shape]. Element ``[i, …]`` depends only on
        ``keys[i]``, the stream and the element's flat index within the image.
    """
    return _to_unit(_bits(keys, stream, shape, offset))


def normal(keys: Tensor, stream: str, shape: Sequence[int], *, offset: int = 0) -> Tensor:
    """Draw standard normal values, keyed per image (inverse CDF of :func:`uniform`).

    Parameters
    ----------
    keys : Tensor
        Image keys of dtype int64, shape [B].
    stream : str
        Stream name.
    shape : sequence of int
        Per-image shape of the draw.
    offset : int, default 0
        Flat index of the first element within the image.

    Returns
    -------
    Tensor
        Standard normal values, float64, shape [B, *shape]. CPU and CUDA agree within a few
        ulp (different ``ndtri`` implementations).
    """
    return torch.special.ndtri(uniform(keys, stream, shape, offset=offset))


def _poisson_icdf(u: Tensor, lam: Tensor) -> tuple[Tensor, Tensor]:
    """Sequential inverse CDF for small rates; returns (counts, finished)."""
    p = torch.exp(-lam)
    cdf = p.clone()
    k = torch.zeros_like(lam)
    for i in range(1, POISSON_ICDF_STEPS + 1):
        k = k + (u > cdf).to(k.dtype)
        p = p * lam / i
        cdf = cdf + p
    return k, u <= cdf


def _poisson_bisect(u: Tensor, lam: Tensor) -> Tensor:
    """Exact inverse CDF by bisection on P(X ≤ k) = Q(k + 1, λ) (static trip count)."""
    spread = 40.0 * torch.sqrt(lam) + 40.0
    lo = torch.clamp(torch.floor(lam - spread), min=0.0)
    hi = torch.ceil(lam + spread)
    for _ in range(POISSON_FALLBACK_STEPS):
        mid = torch.floor(0.5 * (lo + hi))
        ok = torch.special.gammaincc(mid + 1.0, lam) >= u
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid + 1.0)
    return hi


def _uniform_pair(keys: Tensor, stream: str, shape: Sequence[int]) -> tuple[Tensor, Tensor]:
    """Two open-interval uniforms per element from the halves of one hash (b = 32)."""
    h = _bits(keys, stream, shape)
    high = _srl(h, 32).to(torch.float64)
    low = (h & 0xFFFFFFFF).to(torch.float64)
    scale = 2.0**-32
    return (high + 0.5) * scale, (low + 0.5) * scale


def _poisson_ptrs(keys: Tensor, stream: str, lam: Tensor) -> tuple[Tensor, Tensor]:
    """PTRS with a fixed number of attempts; returns (draws, accepted)."""
    shape = tuple(lam.shape[1:])
    pairs = [
        _uniform_pair(keys, f"{stream}/ptrs{attempt}", shape)
        for attempt in range(POISSON_PTRS_ATTEMPTS)
    ]
    return ptrs(lam, pairs)


def ptrs(lam: Tensor, pairs: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Draw Poisson counts by transformed rejection (PTRS, Hörmann 1993), one attempt per pair.

    Parameters
    ----------
    lam : Tensor
        Rates ≥ 10, float64.
    pairs : sequence of (Tensor, Tensor)
        Open-interval uniforms ``(u, v)`` per attempt, broadcastable to ``lam``.

    Returns
    -------
    tuple of (Tensor, Tensor)
        The draws (exact where accepted) and the accepted mask.
    """
    slam = torch.sqrt(lam)
    loglam = torch.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    log_invalpha = torch.log(1.1239 + 1.1328 / (b - 3.4))
    vr = 0.9277 - 3.6224 / (b - 2.0)
    result = torch.zeros_like(lam)
    done = torch.zeros_like(lam, dtype=torch.bool)
    for u, v in pairs:
        uu = u - 0.5
        us = 0.5 - torch.abs(uu)
        k = torch.floor((2.0 * a / us + b) * uu + lam + 0.43)
        k_safe = torch.clamp(k, min=0.0)
        fast = (us >= 0.07) & (v <= vr)
        reject = (k < 0) | ((us < 0.013) & (v > us))
        lhs = torch.log(v) + log_invalpha - torch.log(a / (us * us) + b)
        rhs = -lam + k_safe * loglam - torch.lgamma(k_safe + 1.0)
        accept = fast | (~reject & (lhs <= rhs))
        result = torch.where(accept & ~done, k_safe, result)
        done = done | accept
    return result, done


def poisson(keys: Tensor, stream: str, lam: Tensor) -> Tensor:
    """Draw Poisson counts, keyed per image.

    The draw is not differentiable; camera estimators attach gradients (§6.2).

    Parameters
    ----------
    keys : Tensor
        Image keys of dtype int64, shape [B] (moved to the rates' device).
    stream : str
        Stream name.
    lam : Tensor
        Non-negative rates, shape [B, ...] or [1, ...] (shared by every image); negative values
        are treated as 0.

    Returns
    -------
    Tensor
        Integer-valued counts, float64, shape [B, ...]; NaN where the rate is not finite.

    Raises
    ------
    StructureError
        If the rates' leading axis is neither 1 nor B.
    """
    keys = keys.to(lam.device)
    b = keys.shape[0]
    if lam.ndim == 0 or lam.shape[0] not in (1, b):
        got = list(lam.shape)
        raise StructureError(f"keyed Poisson rates need a leading axis of 1 or B = {b}, got {got}")
    rate = torch.clamp(lam.detach().to(torch.float64), min=0.0)
    if rate.shape[0] != b:
        rate = rate.expand(b, *rate.shape[1:])
    bad = ~torch.isfinite(rate)
    rate = torch.where(bad, 0.0, rate)
    shape = tuple(rate.shape[1:])
    small = rate < POISSON_ICDF_BELOW
    counts_small, finished = _poisson_icdf(
        uniform(keys, f"{stream}/icdf", shape), torch.where(small, rate, 0.0)
    )
    counts_large, accepted = _poisson_ptrs(
        keys, stream, torch.where(small, POISSON_ICDF_BELOW, rate)
    )
    counts = torch.where(small, counts_small, counts_large)
    done = torch.where(small, finished, accepted) | bad
    counts = _poisson_fallback(keys, stream, rate, counts, done, small)
    return torch.where(bad, math.nan, counts)


def _poisson_window(u: Tensor, lam: Tensor, lo: Tensor) -> tuple[Tensor, Tensor]:
    """Exact inverse CDF on the window ``[lo, lo + W]``; returns (counts, found)."""
    base = torch.where(lo > 0, torch.special.gammaincc(torch.clamp(lo, min=1.0), lam), 0.0)
    offsets = torch.arange(POISSON_WINDOW + 1, dtype=lam.dtype, device=lam.device)
    ks = lo[:, None] + offsets  # [S, W + 1]
    log_first = torch.xlogy(lo, lam) - lam - torch.lgamma(lo + 1.0)
    log_ratio = torch.log(lam)[:, None] - torch.log(ks[:, 1:])
    log_pmf = log_first[:, None] + torch.cat(
        [torch.zeros_like(lo)[:, None], log_ratio.cumsum(1)], 1
    )
    cdf = base[:, None] + torch.exp(log_pmf).cumsum(1)
    hit = cdf >= u[:, None]
    found = hit[:, -1] & (base < u)
    first = torch.argmax(hit.to(torch.int8), dim=1).to(lam.dtype)
    return lo + first, found


def _cornish_fisher(u: Tensor, lam: Tensor) -> Tensor:
    """Approximate u-quantile of Poisson(λ): the start of the window search for λ ≥ 10."""
    z = torch.special.ndtri(u)
    q = (
        lam
        + torch.sqrt(lam) * z
        + (z * z - 1.0) / 6.0
        + (z**3 - 7.0 * z) / (72.0 * torch.sqrt(lam))
    )
    return torch.floor(q)


def _poisson_fallback(
    keys: Tensor, stream: str, rate: Tensor, counts: Tensor, done: Tensor, small: Tensor
) -> Tensor:
    """Finish the leftover elements exactly, in a static-size buffer (``nonzero_static``).

    An unfinished inverse-CDF search continues with its own uniform (so the draw stays an exact
    inversion); an element whose PTRS attempts were all rejected is redrawn with a fresh uniform
    (valid because rejection is independent of the accepted value). Uniforms are keyed on each
    element's own flat index, so the result does not depend on the batch. Elements that do not
    fit the buffers are NaN.
    """
    flat_rate = rate.reshape(-1)
    total = flat_rate.numel()
    if total == 0:
        return counts
    per_image = max(math.prod(rate.shape[1:]), 1)
    expected = POISSON_FALLBACK_RATE * total
    slots = min(total, 64 + math.ceil(expected + 10.0 * math.sqrt(expected)))
    todo = (~done).reshape(-1)
    # Padding entries point at a dummy element past the end, so scatters never collide.
    idx = torch.nonzero_static(todo, size=slots, fill_value=total)[:, 0]
    need = idx < total
    safe = torch.where(need, idx, 0)
    image = torch.div(safe, per_image, rounding_mode="floor")
    j = safe - image * per_image + 1
    fresh = mix64(_seeds(keys, f"{stream}/fallback")[image] + j * _GAMMA)
    own = mix64(_seeds(keys, f"{stream}/icdf")[image] + j * _GAMMA)
    is_small = small.reshape(-1)[safe]
    u = _to_unit(torch.where(is_small, own, fresh))
    lam = torch.where(need, flat_rate[safe], 1.0)
    half = POISSON_WINDOW // 2
    lo = torch.where(
        is_small,
        torch.full_like(lam, POISSON_ICDF_STEPS + 1),
        torch.clamp(_cornish_fisher(u, torch.clamp(lam, min=1.0)) - half, min=0.0),
    )
    k, found = _poisson_window(u, lam, lo)
    # The window misses only in the far tails: finish those few by bisection in a small buffer.
    spill = min(slots, 64)
    missed = ~found & need
    pick = torch.nonzero_static(missed, size=spill, fill_value=slots)[:, 0]
    picked = pick < slots
    safe_pick = torch.where(picked, pick, 0)
    exact = _poisson_bisect(u[safe_pick], lam[safe_pick])
    k_ext = torch.cat([k, k.new_zeros(1)])
    k_ext = k_ext.scatter(0, torch.where(picked, pick, slots), exact)
    unresolved = missed.clone()
    unresolved_ext = torch.cat([unresolved, unresolved.new_zeros(1)])
    unresolved_ext = unresolved_ext.scatter(
        0, torch.where(picked, pick, slots), torch.zeros_like(picked)
    )
    k = torch.where(unresolved_ext[:slots], math.nan, k_ext[:slots])
    out = torch.cat([counts.reshape(-1), counts.new_zeros(1)])
    out = out.scatter(0, idx, k)[:total]
    # Elements that needed the buffer but did not fit in it are NaN, never silently wrong.
    placed = torch.zeros(total + 1, dtype=torch.bool, device=rate.device).scatter(0, idx, need)
    out = torch.where(todo & ~placed[:total], math.nan, out)
    return out.reshape(counts.shape)
