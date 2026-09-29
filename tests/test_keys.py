"""Keys (§6.3, §11.7): the counter hash, keyed uniforms, Gaussian and Poisson draws."""

import math

import pytest
import scipy.stats
import torch
from scenes import DEVICES

from gradix._core import keys as K
from gradix._core.errors import StructureError


def _splitmix_reference(state, count):
    """SplitMix64 (Steele, Lea & Flood) in plain Python."""
    mask = (1 << 64) - 1
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out.append(z ^ (z >> 31))
    return out


@pytest.mark.parametrize("device", DEVICES)
def test_counter_hash_is_splitmix64(device):
    seeds = K._seeds(torch.tensor([0, 5, -3], device=device), "stream")
    bits = K._bits(torch.tensor([0, 5, -3], device=device), "stream", (4,))
    for row, seed in enumerate(seeds.tolist()):
        expected = _splitmix_reference(seed & ((1 << 64) - 1), 4)
        got = [int(v) & ((1 << 64) - 1) for v in bits[row].tolist()]
        assert got == expected
    assert _splitmix_reference(0, 1) == [0xE220A8397B1DCDAF]  # the published first output


@pytest.mark.parametrize("device", DEVICES)
def test_uniforms_open_interval_and_moments(device):
    u = K.uniform(torch.arange(8, device=device), "u", (64, 64))
    assert u.dtype == torch.float64 and u.shape == (8, 64, 64)
    assert (u > 0).all() and (u < 1).all()
    assert abs(float(u.mean()) - 0.5) < 3e-3
    assert abs(float(u.var()) - 1 / 12) < 1e-3


def test_image_keyed_draws_ignore_batch_composition():
    keys = torch.arange(64) * 7 + 3
    full = K.uniform(keys, "s", (5, 7))
    alone = K.uniform(keys[17:18], "s", (5, 7))
    shuffled = K.uniform(keys.flip(0), "s", (5, 7))
    assert torch.equal(alone[0], full[17])
    assert torch.equal(shuffled.flip(0), full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_cuda_agree_bitwise():
    keys = torch.arange(6)
    assert torch.equal(K.uniform(keys, "x", (33,)), K.uniform(keys.cuda(), "x", (33,)).cpu())
    lam = torch.rand(6, 33, dtype=torch.float64) * 40
    assert torch.equal(K.poisson(keys, "p", lam), K.poisson(keys.cuda(), "p", lam.cuda()).cpu())


def test_streams_are_independent():
    keys = torch.arange(4)
    a = K.uniform(keys, "camera/poisson", (4096,)).flatten()
    b = K.uniform(keys, "camera/read", (4096,)).flatten()
    assert abs(float(torch.corrcoef(torch.stack([a, b]))[0, 1])) < 0.02
    assert K.stream_id("a") != K.stream_id("b")


def test_normal_draws():
    z = K.normal(torch.arange(4), "n", (50_000,)).flatten()
    assert abs(float(z.mean())) < 0.01 and abs(float(z.std()) - 1) < 0.01
    assert scipy.stats.kstest(z.numpy(), "norm").pvalue > 1e-4


def _chi2_pvalue(counts, lam):
    counts = counts.flatten().long().numpy()
    n = counts.size
    kmax = int(counts.max()) + 1
    observed = torch.bincount(torch.from_numpy(counts), minlength=kmax + 1).double().numpy()
    ks = range(kmax + 1)
    expected = [n * scipy.stats.poisson.pmf(k, lam) for k in ks]
    expected[-1] = n * scipy.stats.poisson.sf(kmax - 1, lam)  # the last bin takes the upper tail
    # merge bins with expected < 5 (from both ends toward the mode)
    obs_m, exp_m, acc_o, acc_e = [], [], 0.0, 0.0
    for o, e in zip(observed, expected, strict=True):
        acc_o += o
        acc_e += e
        if acc_e >= 5:
            obs_m.append(acc_o)
            exp_m.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        obs_m[-1] += acc_o
        exp_m[-1] += acc_e
    if len(exp_m) < 2:
        return 1.0
    stat = sum((o - e) ** 2 / e for o, e in zip(obs_m, exp_m, strict=True))
    return float(scipy.stats.chi2.sf(stat, len(exp_m) - 1))


@pytest.mark.parametrize("lam", [1e-3, 0.3, 3.0, 9.99, 10.0, 31.0, 1e3, 1e5])
def test_keyed_poisson_is_exact(lam):
    rate = torch.full((4, 50_000), lam, dtype=torch.float64)
    counts = K.poisson(torch.arange(4) + 11, "chi2", rate)
    assert torch.equal(counts, counts.round())
    assert _chi2_pvalue(counts, lam) > 1e-4
    assert abs(float(counts.mean()) - lam) < 6 * math.sqrt(lam / counts.numel()) + 1e-9


def test_poisson_fallback_path_is_exact(monkeypatch):
    # one PTRS attempt leaves ≈25 % of the draws to the exact bisection fallback
    monkeypatch.setattr(K, "POISSON_PTRS_ATTEMPTS", 1)
    rate = torch.full(
        (2, 100), 12.0, dtype=torch.float64
    )  # 200 draws: every rejection fits the buffer
    samples = torch.cat([K.poisson(torch.tensor([s, s + 1]), "fb", rate) for s in range(0, 400, 2)])
    assert _chi2_pvalue(samples, 12.0) > 1e-4


def test_poisson_zero_rate_and_negative_rates():
    rate = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
    assert torch.equal(K.poisson(torch.tensor([1]), "z", rate), torch.zeros_like(rate))


def test_check_key():
    assert K.check_key(None) is None
    assert K.check_key(5) == 5
    key = K.check_key(torch.arange(3, dtype=torch.int32))
    assert isinstance(key, torch.Tensor) and key.dtype == torch.int64
    with pytest.raises(StructureError):
        K.check_key(True)
    with pytest.raises(StructureError):
        K.check_key(torch.tensor(3))
    with pytest.raises(StructureError):
        K.check_key(torch.rand(3))
    with pytest.raises(StructureError):
        K.check_key(torch.arange(3), batch=4)


def test_batch_keys_reproduce():
    g1 = K.generator(42, "camera", "cpu")
    g2 = K.generator(42, "camera", "cpu")
    assert torch.equal(torch.rand(5, generator=g1), torch.rand(5, generator=g2))
    g3 = K.generator(42, "other", "cpu")
    assert not torch.equal(
        torch.rand(5, generator=K.generator(42, "camera", "cpu")), torch.rand(5, generator=g3)
    )


def test_truncated_inverse_cdf_continues_exactly(monkeypatch):
    # three steps leave ≈14 % of λ = 2 draws unfinished (within the buffer); the fallback continues
    # each inversion with its own uniform, which keeps the draw an exact inverse CDF
    monkeypatch.setattr(K, "POISSON_ICDF_STEPS", 3)
    rate = torch.full((2, 100), 2.0, dtype=torch.float64)
    samples = torch.cat(
        [K.poisson(torch.tensor([s, s + 1]), "icdf", rate) for s in range(0, 400, 2)]
    )
    assert _chi2_pvalue(samples, 2.0) > 1e-4
    monkeypatch.undo()
    exact = torch.cat([K.poisson(torch.tensor([s, s + 1]), "icdf", rate) for s in range(0, 400, 2)])
    assert torch.equal(samples, exact)  # the same inversion, whatever the step count


def test_window_misses_finish_by_bisection(monkeypatch):
    # one attempt and a 3-count window push many rejected draws past the window into the spill path
    monkeypatch.setattr(K, "POISSON_PTRS_ATTEMPTS", 1)
    monkeypatch.setattr(K, "POISSON_WINDOW", 2)
    rate = torch.full((2, 100), 12.0, dtype=torch.float64)
    samples = torch.cat(
        [K.poisson(torch.tensor([s, s + 1]), "spill", rate) for s in range(0, 400, 2)]
    )
    assert _chi2_pvalue(samples, 12.0) > 1e-4


def test_non_finite_rates_do_not_leak_into_other_images():
    clean = torch.full((1, 128, 128), 10.0, dtype=torch.float64)
    broken = torch.full((1, 128, 128), float("nan"), dtype=torch.float64)
    keys = torch.tensor([3, 4])
    alone = K.poisson(keys[1:], "p", clean)
    together = K.poisson(keys, "p", torch.cat([broken, clean]))
    assert torch.isnan(together[0]).all()
    assert torch.equal(together[1:], alone)


def test_uniforms_stay_below_one_and_offsets_draw_the_same_elements():
    top = K._to_unit(torch.tensor([-1], dtype=torch.int64))
    assert float(top) < 1.0
    keys = torch.tensor([11, 12])
    whole = K.uniform(keys, "s", (10,))
    assert torch.equal(whole[:, 5:], K.uniform(keys, "s", (5,), offset=5))


def test_fold_in_derives_independent_deterministic_keys():
    keys = torch.arange(4)
    a, b = K.fold_in(keys, 1), K.fold_in(keys, 2)
    assert torch.equal(a, K.fold_in(keys, 1)) and not torch.equal(a, b)
    assert len(set(torch.cat([keys, a, b]).tolist())) == 12


def test_shared_rates_with_image_keys():
    keys = torch.arange(4)
    lam = torch.full((1, 16), 3.0, dtype=torch.float64)
    shared = K.poisson(keys, "p", lam)
    assert shared.shape == (4, 16)
    assert torch.equal(shared, K.poisson(keys, "p", lam.expand(4, 16)))
    with pytest.raises(StructureError, match="leading axis"):
        K.poisson(keys, "p", torch.ones(3, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_keys_follow_cuda_rates():
    lam = torch.full((2, 64), 5.0, device="cuda", dtype=torch.float64)
    keys = torch.tensor([1, 2])
    assert torch.equal(K.poisson(keys, "p", lam).cpu(), K.poisson(keys, "p", lam.cpu()))


def test_keyed_uniforms_are_uncorrelated_across_images_elements_and_streams():
    n = 200_000
    keys = torch.arange(1, 9)
    u = K.uniform(keys, "q", (n,))
    limit = 5.0 / math.sqrt(n)

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).mean() / (a.std() * b.std()))

    for i in range(len(keys) - 1):
        assert abs(corr(u[i], u[i + 1])) < limit  # adjacent keys
    assert abs(corr(u[0, :-1], u[0, 1:])) < limit  # adjacent elements
    assert abs(corr(u[0], K.uniform(keys[:1], "r", (n,))[0])) < limit  # streams
    # uniformity: a coarse chi-square over 64 bins
    counts = torch.histc(u.flatten(), bins=64, min=0.0, max=1.0)
    expected = u.numel() / 64
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < scipy.stats.chi2.ppf(1 - 1e-4, df=63)


def _batch_moments(device, lam):
    from gradix.ops.detect import poisson

    n = 1 << 21
    x = poisson(torch.full((n,), lam, device=device, dtype=torch.float64), 11, "moments")
    x = x.double()
    mean = float(x.mean())
    centred = x - mean
    return n, mean, float((centred * centred).mean()), float((centred**3).mean())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lam", [100.0, 3999.0])
def test_batch_keyed_poisson_means_and_variances_hold_at_high_rates(device, lam):
    # batch keys use torch.poisson (fast): the mean and variance hold to 0.01 % and 1.5 %
    n, mean, var, _third = _batch_moments(device, lam)
    assert abs(mean / lam - 1.0) < 1e-4 + 5.0 * math.sqrt(1.0 / (lam * n))
    assert abs(var / lam - 1.0) < 0.015


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lam", [100.0, 3999.0])
def test_batch_keyed_poisson_is_exact(device, lam, request):
    # exact on CPU; on CUDA curand's shape is approximate above λ ≈ 1000 (variance −0.9 %,
    # third moment 1.8× at 4000), accepted until the fast exact sampler planned before 1.0
    # (strict: this starts passing, and so fails, once that sampler lands)
    if device == "cuda" and lam > 1000.0:
        reason = "curand's Poisson shape is approximate above λ ≈ 1000"
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    n, mean, var, third = _batch_moments(device, lam)
    assert abs(mean - lam) < 5.0 * math.sqrt(lam / n)
    assert abs(var / lam - 1.0) < 5.0 * math.sqrt(2.0 / n) + 1e-3
    assert abs(third / lam - 1.0) < 0.25  # sampling error of μ₃ ≈ √(15λ/n)·… at these sizes
