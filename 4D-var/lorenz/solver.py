#!/usr/bin/env python

from typing import Callable, Dict, Optional, Tuple, Union
import sys
from pathlib import Path

import torch
from torch import Tensor


import os
import random
import numpy as np


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Try sda installed, fallback to source
try:
    from sda.lorenz.utils import make_chain, posterior, log_prior, log_likelihood
except ModuleNotFoundError:
    project_sda_root = Path(__file__).resolve().parents[2] / "sda"
    if str(project_sda_root) not in sys.path:
        sys.path.insert(0, str(project_sda_root))
    from lorenz.utils import make_chain, posterior, log_prior, log_likelihood

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
    background_std: float = 1.0,
    maxiter: int = 200,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Evaluate weak 4DVar with explicit weak-constraint 4DVar objective.

    Cost minimized for each particle initialization:
    J(x) = 0.5 * ||(x0 - xb) / sigma_b||^2 - log_prior(x) - log_likelihood(y|x)
    """
    sigma, step = freq_to_params(freq)
    A = make_observation_operator()

    if background_std <= 0:
        raise ValueError("background_std must be strictly positive for 4DVar.")
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    
    # SDA operations are CPU-only
    y_cpu = y.cpu()
    x_ref_cpu = x_ref.cpu()
    
    # 4DVar background initialization: draw full trajectories from model prior.
    chain = make_chain()
    x_prev = chain.prior((n_samples,))
    traj = [x_prev]
    for _ in range(64):
        x_prev = chain.transition(x_prev)
        traj.append(x_prev)
    x_init_particles = torch.stack(traj, dim=1)
    
    x_samples = []
    for i in range(n_samples):
        x_init = x_init_particles[i]
        
        if maxiter == 0:
            x_map = x_init
        else:
            x_b = x_init[0].clone()
            x_opt = torch.nn.Parameter(x_init.clone())
            optimizer = torch.optim.LBFGS((x_opt,), max_iter=maxiter, line_search_fn="strong_wolfe")

            def closure() -> Tensor:
                optimizer.zero_grad()
                background_term = 0.5 * ((x_opt[0] - x_b) / background_std).square().sum()
                loss = background_term - log_prior(x_opt) - log_likelihood(y_cpu, x_opt, A, sigma, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite weak-4dvar objective")
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
                x_map = x_opt.detach()
                if not torch.isfinite(x_map).all():
                    x_map = x_init
            except (RuntimeError, FloatingPointError):
                x_map = x_init
        
        x_samples.append(x_map)
    
    x_w4d = torch.stack(x_samples, dim=0)
    return compute_stats(x_w4d, x_ref_cpu, y_cpu, A=A, sigma=sigma, step=step)
