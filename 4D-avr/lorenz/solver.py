#!/usr/bin/env python

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor

from sda.lorenz.utils import make_chain, posterior, weak_4d_var, log_prior, log_likelihood
from sda.sda.utils import emd


def freq_to_params(freq: str) -> Tuple[float, int]:
    if freq == "lo":
        return 0.05, 8
    if freq == "hi":
        return 0.25, 1
    raise ValueError(f"Unknown frequency mode: {freq}")


def make_observation_operator() -> Callable[[Tensor], Tensor]:
    chain = make_chain()
    return lambda x: chain.preprocess(x)[..., :1]


def compute_stats(
    x: Tensor,
    x_ref: Tensor,
    y: Tensor,
    A: Callable[[Tensor], Tensor],
    sigma: float,
    step: int,
) -> Dict[str, float]:
    return {
        "log_px": log_prior(x).mean().item(),
        "log_py": log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item(),
        "w1": emd(x, x_ref).item(),
    }


def sample_reference_posterior(
    y: Tensor,
    freq: str,
    n_samples: int = 1024,
) -> Tuple[Tensor, Tensor]:
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()

    x_ref = posterior(y, A=A, sigma=sigma, step=step)[:n_samples]
    x_ref_2 = posterior(y, A=A, sigma=sigma, step=step)[:n_samples]

    return x_ref, x_ref_2


def sample_weak_4dvar(
    y: Tensor,
    freq: str,
    n_samples: int = 1024,
    iterations: int = 16,
) -> Tensor:
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()
    chain = make_chain()

    out = []
    for _ in range(n_samples):
        x0 = chain.prior(())
        x0 = chain.trajectory(x0, length=64, last=True)
        x_init = chain.trajectory(x0, length=65)
        x_map = weak_4d_var(
            x=x_init,
            y=y,
            A=A,
            sigma=sigma,
            step=step,
            iterations=iterations,
        )
        out.append(x_map)

    return torch.stack(out, dim=0)


def evaluate_reference(
    y: Tensor,
    freq: str,
    n_samples: int = 1024,
) -> Tuple[Tensor, Dict[str, float]]:
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()

    x_ref, x_ref_2 = sample_reference_posterior(y, freq=freq, n_samples=n_samples)
    stats = compute_stats(x_ref, x_ref_2, y, A=A, sigma=sigma, step=step)

    return x_ref, stats


def evaluate_weak_4dvar(
    y: Tensor,
    freq: str,
    x_ref: Tensor,
    n_samples: int = 1024,
    iterations: int = 16,
) -> Dict[str, float]:
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()

    x_w4d = sample_weak_4dvar(y, freq=freq, n_samples=n_samples, iterations=iterations)
    return compute_stats(x_w4d, x_ref, y, A=A, sigma=sigma, step=step)