"""Q6 benchmark: sCMOS likelihood approximations against the exact Poisson ⊛ Gaussian (§11.6).

Run with ``uv run python benchmarks/q6_scmos.py``. Prints, per approximation, light level and
read noise, the log-likelihood error and the bias, efficiency and claimed Cramér–Rao bound of
the estimates of λ, gain, offset and read noise (deterministic quadrature, gain 2 ADU/e⁻,
offset 100 ADU), then the forward + backward time per million pixels; the timings are
appended to ``benchmarks/results.json``.
"""

from pathlib import Path

import torch

from gradix.testing import likelihoods
from gradix.testing.bench import BenchResult, record


def main() -> None:
    """Sweep λ ∈ [0.1, 100] and σ ∈ {0.8, 1.6, 3.0} e⁻ and time each approximation."""
    results = likelihoods.run()
    header = "approximation     σ     λ   ll err | bias/SE: λ  gain offset read | efficiency ..."
    print(header)
    for r in results:
        bias = " ".join(f"{r.bias_se[p]:+6.2f}" for p in likelihoods.PARAMETERS)
        eff = " ".join(f"{r.efficiency[p]:5.3f}" for p in likelihoods.PARAMETERS)
        crlb = " ".join(f"{r.crlb_ratio[p]:5.3f}" for p in likelihoods.PARAMETERS)
        print(
            f"{r.approximation:16s} {r.read:4.1f} {r.lam:6.1f} {r.ll_error:8.1e} | {bias} |"
            f" {eff} | crlb {crlb}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timings = []
    for name in likelihoods.APPROXIMATIONS:
        ms = likelihoods.speed(name, device=device)
        print(f"{name:16s} {ms:8.2f} ms per 1M pixels (forward + backward, float32, {device})")
        timings.append(
            BenchResult(
                name=f"q6/{name}/fwd+bwd",
                median_ms=ms,
                times_ms=(ms,),
                items=1 << 20,
                device=device,
            )
        )
    record(timings, Path(__file__).with_name("results.json"))


if __name__ == "__main__":
    main()
