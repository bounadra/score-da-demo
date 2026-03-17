from dataclasses import dataclass


@dataclass(frozen=True)
class LorenzConfig:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.025
    window_length: int = 250


@dataclass(frozen=True)
class AssimilationConfig:
    obs_step: int = 8
    sigma_y: float = 0.05
    background_std: float = 1.0
    maxiter: int = 200
    seed: int = 42


SCENARIOS = {
    "lo": AssimilationConfig(obs_step=8, sigma_y=0.05, background_std=1.0, maxiter=200, seed=42),
    "hi": AssimilationConfig(obs_step=1, sigma_y=0.25, background_std=1.0, maxiter=200, seed=42),
}