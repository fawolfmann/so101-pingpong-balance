"""Load the PPO controller and query its action. Shared by the viewer, the evals and hardware/run_real.py."""
from __future__ import annotations

import numpy as np
import torch

from sim.ppo import ActorCritic


def load_controller(path) -> ActorCritic:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    obs_dim = next(v for k, v in ck["model"].items() if k.startswith("actor") and k.endswith("weight")).shape[1]
    ac = ActorCritic(obs_dim)
    ac.load_state_dict(ck["model"])
    ac.act_hist = (obs_dim - 14) // 5  # the env must be built with this act_hist so controller_obs() has the right size
    ac.train_args = ck.get("args", {})  # the env settings it was trained with (latency, servo randomization, ball noise, grip)
    return ac.eval()


@torch.no_grad()
def controller_action(ac: ActorCritic, tobs: np.ndarray) -> np.ndarray:
    """The controller's mean action (no exploration noise), clipped to the env's range."""
    return np.clip(ac.dist(torch.as_tensor(tobs[None], dtype=torch.float32)).mean[0].numpy(), -1, 1)
