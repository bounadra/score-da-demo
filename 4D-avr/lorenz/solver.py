#!/usr/bin/env python

from typing import Callable, Dict, Optional, Tuple, Union
import sys
from pathlib import Path

import torch
from torch import Tensor

# Try sda installed, fallback to source
try:
    from sda.lorenz.utils import make_chain, posterior, weak_4d_var, log_prior, log_likelihood
except ModuleNotFoundError:
    project_sda_root = Path(__file__).resolve().parents[2] / "sda"
    if str(project_sda_root) not in sys.path:
        sys.path.insert(0, str(project_sda_root))
    from lorenz.utils import make_chain, posterior, weak_4d_var, log_prior, log_likelihood

try:
    from sda.sda.utils import emd
except ModuleNotFoundError:
    from sda.utils import emd

from config import SCENARIOS


def freq_to_params(freq: str) -> Tuple[float, int]:
    """Map frequency string to observation params."""
    cfg = SCENARIOS.get(freq)
    if cfg is None:
        raise ValueError(f"Unknown frequency: {freq}")
    return cfg.sigma_y, cfg.obs_step


def make_observation_operator() -> Callable[[Tensor], Tensor]:
    """Create observation operator (first component of state)."""
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
    """Compute log_px, log_py, w1 statistics."""
    return {
        "log_px": log_prior(x).mean().item(),
        "log_py": log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item(),
        "w1": emd(x, x_ref).item(),
    }


def evaluate_reference(
    y: Tensor,
    freq: str,
    n_samples: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Evaluate reference posterior using SDA's posterior."""
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()
    
    # SDA posterior is CPU-only
    y_cpu = y.cpu()
    x_ref = posterior(y_cpu, A=A, sigma=sigma, step=step)[:n_samples]
    x_ref_2 = posterior(y_cpu, A=A, sigma=sigma, step=step)[:n_samples]
    
    stats = compute_stats(x_ref, x_ref_2, y_cpu, A=A, sigma=sigma, step=step)
    return x_ref, stats


def evaluate_weak_4dvar(
    y: Tensor,
    freq: str,
    x_ref: Tensor,
    n_samples: int = 1024,
    iterations: int = 16,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Evaluate weak 4DVar using SDA's weak_4d_var optimizer.
    
    NOTE: weak_4d_var in SDA is untested code (never called in their evals).
    We initialize from posterior samples rather than raw prior.
    """
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()
    
    # SDA operations are CPU-only
    y_cpu = y.cpu()
    x_ref_cpu = x_ref.cpu()
    
    # Initialize from posterior, then refine with weak_4d_var
    x_init_particles = posterior(y_cpu, A=A, sigma=sigma, step=step)[:n_samples]
    
    x_samples = []
    for i in range(n_samples):
        x_init = x_init_particles[i]  # Use posterior sample as initialization
        
        # Refine with weak 4DVar if iterations > 0
        if iterations > 0:
            x_map = weak_4d_var(
                x=x_init,
                y=y_cpu,
                A=A,
                sigma=sigma,
                step=step,
                iterations=iterations,
            )
        else:
            # iterations=0: just use posterior initialization as-is
            x_map = x_init
        
        x_samples.append(x_map)
    
    x_w4d = torch.stack(x_samples, dim=0)
    return compute_stats(x_w4d, x_ref_cpu, y_cpu, A=A, sigma=sigma, step=step)
