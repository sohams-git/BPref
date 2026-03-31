#!/usr/bin/env python3
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from bpref.envs import make_env as make_bpref_env
from reward_model import RewardModel

# Try importing your actor implementation (BPref fork usually has this)
from agent.actor import DiagGaussianActor


def load_cfg(run_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    return OmegaConf.to_container(cfg, resolve=True)


def env_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, (tuple, list)) else out


def env_step(env, act):
    out = env.step(act)
    if isinstance(out, (tuple, list)) and len(out) == 5:
        nxt, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return nxt, float(r), done, info
    nxt, r, done, info = out
    return nxt, float(r), bool(done), info


def corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def actor_action(actor, obs, device="cpu", deterministic=True):
    """
    Robustly get an action from DiagGaussianActor across common implementations.
    Uses deterministic (mean) action by default.
    """
    o = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)

    # common patterns:
    # 1) actor(o) -> dist ; dist.mean / dist.sample()
    # 2) actor(o, deterministic=True) -> action
    # 3) actor.act(o, deterministic=True) -> action
    if hasattr(actor, "act"):
        a = actor.act(o, deterministic=deterministic)
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        return a.squeeze()

    try:
        out = actor(o, deterministic=deterministic)
        if torch.is_tensor(out):
            out = out.detach().cpu().numpy()
            return out.squeeze()
    except TypeError:
        pass

    dist = actor(o)
    # dist could be (mu, std) tuple or a distribution
    if isinstance(dist, (tuple, list)) and len(dist) >= 1:
        mu = dist[0]
        if torch.is_tensor(mu):
            mu = mu.detach().cpu().numpy()
        return np.asarray(mu).squeeze()

    # torch distribution
    if hasattr(dist, "mean") and hasattr(dist, "sample"):
        a = dist.mean if deterministic else dist.sample()
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        return a.squeeze()

    raise RuntimeError("Could not extract action from actor; inspect agent/actor.py call signature.")


@torch.no_grad()
def rm_step_reward(inner_net, obs, act, device="cpu"):
    s = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)
    a = torch.tensor(np.asarray(act, dtype=np.float32)[None, :], device=device)
    x = torch.cat([s, a], dim=1)  # your RM expects concat, in_features=14
    out = inner_net(x)
    return float(out.detach().cpu().numpy().squeeze())


def build_rm_member(cfg, ds, da):
    rm = RewardModel(
        ds, da,
        ensemble_size=1,
        lr=float(cfg.get("reward_lr", 3e-4)),
        mb_size=int(cfg.get("reward_batch", 128)),
        size_segment=int(cfg.get("segment", 1)),
        activation=str(cfg.get("activation", "tanh")),
        large_batch=int(cfg.get("large_batch", 1)),
        label_margin=float(cfg.get("label_margin", 0.0)),
        teacher_beta=float(cfg.get("teacher_beta", -1)),
        teacher_gamma=float(cfg.get("teacher_gamma", 1)),
        teacher_eps_mistake=float(cfg.get("teacher_eps_mistake", 0)),
        teacher_eps_skip=float(cfg.get("teacher_eps_skip", 0)),
        teacher_eps_equal=float(cfg.get("teacher_eps_equal", 0)),
    )
    rm.construct_ensemble()
    return rm


def main():
    run_dir = "/home/sohams/BPref/runs/hopper/pebble_hopper_b2000_seg50_s3"
    cfg = load_cfg(run_dir)

    env_name = cfg["env"]  # gym-hopper
    env = make_bpref_env(env_name)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    device = "cpu"   # set "cuda" if you want, but keep consistent
    episodes = 30

    # -------- load actor --------
    actor = DiagGaussianActor(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=int(cfg["agent"]["params"]["actor_cfg"]["params"]["hidden_dim"]),
        hidden_depth=int(cfg["agent"]["params"]["actor_cfg"]["params"]["hidden_depth"]),
        log_std_bounds=cfg["agent"]["params"]["actor_cfg"]["params"]["log_std_bounds"],
    ).to(device)
    actor_sd = torch.load(os.path.join(run_dir, "actor_1000000.pt"), map_location=device)
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()

    # -------- load RM ensemble members (use ensemble[0]) --------
    rm_files = [
        "reward_model_1000000_0.pt",
        "reward_model_1000000_1.pt",
        "reward_model_1000000_2.pt",
    ]
    rm_nets = []
    for f in rm_files:
        rm = build_rm_member(cfg, obs_dim, act_dim)
        sd = torch.load(os.path.join(run_dir, f), map_location=device)
        rm.ensemble[0].load_state_dict(sd, strict=True)
        rm.ensemble[0].to(device)
        rm.ensemble[0].eval()
        rm_nets.append(rm.ensemble[0])

    true_returns = []
    rm_returns = []
    rm_returns_per_step = []
    lengths = []

    for ep in range(episodes):
        obs = env_reset(env)
        done = False
        tr = 0.0
        rr = 0.0
        steps = 0

        while not done:
            act = actor_action(actor, obs, device=device, deterministic=True)

            # Ensure within env bounds (some actors output already scaled; this is a safe clamp)
            act = np.clip(act, -action_high, action_high)

            # RM ensemble mean reward
            step_r = float(np.mean([rm_step_reward(net, obs, act, device=device) for net in rm_nets]))

            nxt, r, done, info = env_step(env, act)

            tr += r
            rr += step_r
            steps += 1
            obs = nxt

        true_returns.append(tr)
        rm_returns.append(rr)
        rm_returns_per_step.append(rr / max(steps, 1))
        lengths.append(steps)

        print(f"ep={ep:03d} steps={steps:4d} true={tr:10.2f} rm_sum={rr:10.2f} rm_per_step={rr/max(steps,1):8.4f}")

    print("\n=== Summary (Policy rollouts: deterministic actor) ===")
    print(f"env: {env_name} episodes: {episodes} device: {device}")
    print(f"true: mean={np.mean(true_returns):.2f} std={np.std(true_returns):.2f} min={np.min(true_returns):.2f} max={np.max(true_returns):.2f}")
    print(f"rm_sum: mean={np.mean(rm_returns):.2f} std={np.std(rm_returns):.2f}")
    print(f"rm_per_step: mean={np.mean(rm_returns_per_step):.4f} std={np.std(rm_returns_per_step):.4f}")
    print(f"corr(true, rm_sum) = {corr(true_returns, rm_returns):.3f}")
    print(f"corr(true, rm_per_step) = {corr(true_returns, rm_returns_per_step):.3f}")
    print(f"corr(steps, rm_sum) = {corr(lengths, rm_returns):.3f}")
    print(f"corr(steps, true) = {corr(lengths, true_returns):.3f}")


if __name__ == "__main__":
    main()
