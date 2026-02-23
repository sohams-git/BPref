#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

from bpref.envs import make_env as make_bpref_env
from reward_model import RewardModel
from agent.actor import DiagGaussianActor


# -------------------- Helpers --------------------
def load_cfg(run_dir: str) -> dict:
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    return OmegaConf.to_container(cfg, resolve=True)


def env_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, (tuple, list)) else out


def env_step(env, act):
    out = env.step(act)
    # Gymnasium-style: (obs, reward, terminated, truncated, info)
    if isinstance(out, (tuple, list)) and len(out) == 5:
        nxt, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return nxt, float(r), done, info
    # Gym-style: (obs, reward, done, info)
    nxt, r, done, info = out
    return nxt, float(r), bool(done), info


def corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pick_latest_checkpoint(run_dir: str, prefix: str) -> str:
    """
    Finds latest file like actor_1000000.pt for prefix='actor_'.
    Returns filename (not full path).
    """
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.pt$")
    best = None
    best_step = -1
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best = fn
    if best is None:
        raise FileNotFoundError(f"No checkpoints found for prefix={prefix} in {run_dir}")
    return best


def find_rm_members(run_dir: str, step: int = None):
    """
    Returns sorted list of reward model member files.
    Supports:
      reward_model_<step>_<k>.pt
    If step is None, auto-pick largest step found.
    """
    pat = re.compile(r"^reward_model_(\d+)_(\d+)\.pt$")
    matches = []
    for fn in os.listdir(run_dir):
        m = pat.match(fn)
        if not m:
            continue
        s = int(m.group(1))
        k = int(m.group(2))
        matches.append((s, k, fn))

    if not matches:
        raise FileNotFoundError(f"No reward_model_*.pt files found in {run_dir}")

    if step is None:
        step = max(s for (s, _, _) in matches)

    members = [fn for (s, k, fn) in sorted(matches) if s == step]
    if not members:
        raise FileNotFoundError(f"No reward model members found at step={step} in {run_dir}")
    return step, members


@torch.no_grad()
def actor_action(actor, obs, device="cpu", deterministic=True):
    o = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)

    # common patterns:
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

    if isinstance(dist, (tuple, list)) and len(dist) >= 1:
        mu = dist[0]
        if torch.is_tensor(mu):
            mu = mu.detach().cpu().numpy()
        return np.asarray(mu).squeeze()

    if hasattr(dist, "mean") and hasattr(dist, "sample"):
        a = dist.mean if deterministic else dist.sample()
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        return a.squeeze()

    raise RuntimeError("Could not extract action from actor; inspect agent/actor.py")


@torch.no_grad()
def rm_step_reward(inner_net, obs, act, device="cpu"):
    s = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)
    a = torch.tensor(np.asarray(act, dtype=np.float32)[None, :], device=device)
    x = torch.cat([s, a], dim=1)
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


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Hydra run directory (contains .hydra/config.yaml + actor_*.pt)")
    ap.add_argument("--env", type=str, default=None, help="Override env name (e.g., gym-hopper, gym-ant, gym-walker2d, gym-swimmer, gym-reacher, gym-humanoid)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic actor sampling instead of deterministic mean")
    ap.add_argument("--rm_step", type=int, default=None, help="Load RM at this step (default: auto-pick latest)")
    ap.add_argument("--actor_step", type=int, default=None, help="Load actor at this step (default: auto-pick latest)")
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = load_cfg(run_dir)

    # pick env
    env_name = args.env if args.env is not None else cfg["env"]
    env = make_bpref_env(env_name)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_high = np.asarray(env.action_space.high, dtype=np.float32)

    device = args.device
    episodes = int(args.episodes)
    deterministic = not args.stochastic

    # -------- load actor --------
    if args.actor_step is None:
        actor_file = pick_latest_checkpoint(run_dir, "actor_")
    else:
        actor_file = f"actor_{args.actor_step}.pt"

    actor_cfg = cfg["agent"]["params"]["actor_cfg"]["params"]
    actor = DiagGaussianActor(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=int(actor_cfg["hidden_dim"]),
        hidden_depth=int(actor_cfg["hidden_depth"]),
        log_std_bounds=actor_cfg["log_std_bounds"],
    ).to(device)

    actor_sd = torch.load(os.path.join(run_dir, actor_file), map_location=device)
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()

    # -------- load RM ensemble members --------
    rm_step, rm_files = find_rm_members(run_dir, step=args.rm_step)
    rm_nets = []
    for f in rm_files:
        rm = build_rm_member(cfg, obs_dim, act_dim)
        sd = torch.load(os.path.join(run_dir, f), map_location=device)
        rm.ensemble[0].load_state_dict(sd, strict=True)
        rm.ensemble[0].to(device)
        rm.ensemble[0].eval()
        rm_nets.append(rm.ensemble[0])

    print("\n=== Eval setup ===")
    print("run_dir:", run_dir)
    print("env:", env_name)
    print("device:", device)
    print("episodes:", episodes)
    print("actor:", actor_file)
    print("rm_step:", rm_step, "rm_members:", rm_files)
    print("deterministic:", deterministic)
    print("==================\n")

    true_returns, rm_returns, rm_returns_per_step, lengths = [], [], [], []

    for ep in range(episodes):
        obs = env_reset(env)
        done = False
        tr = 0.0
        rr = 0.0
        steps = 0

        while not done:
            act = actor_action(actor, obs, device=device, deterministic=deterministic)
            act = np.clip(act, -action_high, action_high)

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

    env.close() if hasattr(env, "close") else None

    print("\n=== Summary ===")
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