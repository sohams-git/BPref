import numpy as np
import torch
import torch.nn.functional as F
import gym
import os
import random
import math
import dmc2gym
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

from collections import deque
from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv
from collections import deque
from skimage.util.shape import view_as_windows
from torch import nn
from torch import distributions as pyd
    

def make_env(cfg):
    """
    Create environment based on cfg.env.
    - If cfg.env looks like a Gym Mujoco id (e.g., HalfCheetah-v2), use our Gym adapter.
    - Else, fall back to existing DMC/Metaworld logic (imports remain lazy).
    """
    # ---- Gym Mujoco route ----
    env_name = getattr(cfg, "env", None)
    if isinstance(env_name, str) and env_name.startswith("HalfCheetah"):
        # Use the adapter you created
        try:
            from bpref.envs.gym_halfcheetah import GymHalfCheetah
        except Exception as e:
            raise RuntimeError("Gym adapter import failed: %r" % (e,))
        add_time = bool(getattr(cfg, "add_time", False))
        max_ep_len = int(getattr(cfg, "max_ep_len", 1000))
        seed = int(getattr(cfg, "seed", 0))
        env = GymHalfCheetah(env_id=env_name, add_time=add_time, max_ep_len=max_ep_len, seed=seed)
        return env

    # ---- Fall back to DMC / Metaworld (original code path) ----
    # Delay imports so Gym runs don't require these packages.
    try:
        import dmc2gym
    except Exception:
        # If someone tries a DMC env without dmc2gym installed, fail clearly.
        raise RuntimeError("DMC env requested but dmc2gym is not available. Install or stub only for Gym runs.")
    # Original behavior for DMC-style env names (your repo’s default)
    # NOTE: adapt this if your repo expects domain/task split.
    # For example, if cfg.env='dog_stand', the original code probably did something like:
    domain_task = getattr(cfg, "env", "dog_stand")
    # You may have original parameters like frame_skip, visualize_reward, etc.
    env = dmc2gym.make(
        domain_task.split('_')[0],            # domain (best-effort)
        domain_task[len(domain_task.split('_')[0])+1:],  # task
        seed=int(getattr(cfg, "seed", 0)),
        visualize_reward=False
    )
    return env

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def set_seed_everywhere(seed):
    """
    Minimal seeding helper used by train_PEBBLE.py.
    Seeds Python, NumPy, and (if available) PyTorch CUDA.
    """
    import os, random
    import numpy as _np
    random.seed(int(seed))
    _np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        # torch not installed or older version: fine for non-torch code paths
        pass
# ---- Running mean/std helpers (NumPy + Torch) ----
class RunningMeanStd:
    """Welford-style running mean/std in NumPy. Keeps numerical stability for streaming updates."""
    def __init__(self, shape=(), epsilon=1e-4):
        import numpy as np
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        import numpy as np
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        import numpy as np
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2  = m_a + m_b + delta**2 * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self):
        import torch
        # Add a tiny eps for numerical stability
        return torch.sqrt(self.var + torch.finfo(self.mean.dtype).eps)


class TorchRunningMeanStd:
    """Torch version used by agents; mirrors RunningMeanStd but stores tensors."""
    def __init__(self, shape=(), device="cpu", dtype=None, epsilon=1e-4):
        import torch
        dtype = torch.float32 if dtype is None else dtype
        self.device = torch.device(device)
        self.mean  = torch.zeros(shape, device=self.device, dtype=dtype)
        self.var   = torch.ones(shape,  device=self.device, dtype=dtype)
        self.count = torch.tensor(float(epsilon), device=self.device, dtype=dtype)

    def update(self, x):
        import torch
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, device=self.device, dtype=self.mean.dtype)
            if x.ndim == 0:
                x = x.view(1, *self.mean.shape)
            batch_mean = x.mean(dim=0)
            batch_var  = x.var(dim=0, unbiased=False)
            batch_count = torch.tensor(float(x.shape[0]), device=self.device, dtype=self.mean.dtype)
            self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        import torch
        with torch.no_grad():
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * (batch_count / tot_count)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2  = m_a + m_b + delta.pow(2) * (self.count * batch_count / tot_count)
            new_var = M2 / tot_count
            self.mean, self.var, self.count = new_mean, new_var, tot_count

    @property
    def std(self):
        import torch
        # Add a tiny eps for numerical stability
        return torch.sqrt(self.var + torch.finfo(self.mean.dtype).eps)
# ---- Small MLP builder used by actor/critic ----
def mlp(in_dim, hidden_dim, out_dim, hidden_depth, activation=None, output_activation=None):
    import torch.nn as nn
    if activation is None:
        activation = nn.ReLU
    layers = []
    last = int(in_dim)
    for _ in range(int(hidden_depth)):
        layers += [nn.Linear(last, int(hidden_dim)), activation()]
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)
# ---- Initialization helper used by actor/critic ----
def weight_init(m):
    """Orthogonal init (good default for RL). Applies to Linear/Conv; zeros bias."""
    import torch
    import torch.nn as nn
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.orthogonal_(m.weight) if hasattr(m, "weight") and m.weight is not None else None
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ---- Simple eval_mode context manager for agents ----
from contextlib import contextmanager
@contextmanager
def eval_mode(model):
    try:
        import torch
        prev = getattr(model, "training", None)
        if hasattr(model, "train"): model.train(False)
        yield model
        if hasattr(model, "train") and prev is not None: model.train(prev)
    except Exception:
        # If it's not a torch Module, just yield
        yield model
# ---- Target network update helpers ----
def soft_update_params(net, target_net, tau):
    """Polyak averaging: target <- tau*net + (1-tau)*target."""
    import torch
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.lerp_(p.data, float(tau))

def hard_update_params(net, target_net):
    """Hard copy: target <- net."""
    import torch
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(p.data)
