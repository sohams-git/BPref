#!/usr/bin/env python3
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from bpref.envs import make_env as make_bpref_env
from reward_model import RewardModel


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


@torch.no_grad()
def rm_step_reward(inner_net, obs, act, device="cpu"):
    s = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :], device=device)
    a = torch.tensor(np.asarray(act, dtype=np.float32)[None, :], device=device)
    x = torch.cat([s, a], dim=1)   # matches in_features=14
    out = inner_net(x)
    return float(out.detach().cpu().numpy().squeeze())


def corr(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    run_dir = "/home/sohams/BPref/runs/hopper/pebble_hopper_b2000_seg50_s3"
    cfg = load_cfg(run_dir)

    env_name = cfg["env"]  # "gym-hopper"
    env = make_bpref_env(env_name)

    ds = int(np.prod(env.observation_space.shape))
    da = int(np.prod(env.action_space.shape))

    device = "cpu"   # switch to "cuda" if you want
    episodes = 30

    member_files = [
        "reward_model_1000000_0.pt",
        "reward_model_1000000_1.pt",
        "reward_model_1000000_2.pt",
    ]

    # Build one RewardModel wrapper per member, but we'll only use ensemble[0] as the active net
    inner_nets = []
    for mf in member_files:
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

        sd = torch.load(os.path.join(run_dir, mf), map_location=device)
        rm.ensemble[0].load_state_dict(sd, strict=True)
        rm.ensemble[0].to(device)
        rm.ensemble[0].eval()
        inner_nets.append(rm.ensemble[0])

    true_returns = []
    rm_returns_members = [[] for _ in inner_nets]
    rm_returns_mean = []

    for ep in range(episodes):
        obs = env_reset(env)
        done = False
        true_ret = 0.0
        rm_rets = [0.0 for _ in inner_nets]
        steps = 0

        while not done:
            act = env.action_space.sample()

            step_rms = [rm_step_reward(net, obs, act, device=device) for net in inner_nets]

            nxt, r, done, info = env_step(env, act)
            true_ret += r
            for i, rr in enumerate(step_rms):
                rm_rets[i] += rr

            obs = nxt
            steps += 1

        rmean = float(np.mean(rm_rets))
        true_returns.append(true_ret)
        rm_returns_mean.append(rmean)
        for i in range(len(inner_nets)):
            rm_returns_members[i].append(rm_rets[i])

        print(
            f"ep={ep:03d} steps={steps:4d} true={true_ret:10.2f} "
            + " ".join([f"rm{i}={rm_rets[i]:10.2f}" for i in range(len(inner_nets))])
            + f" mean={rmean:10.2f}"
        )

    print("\n=== Summary ===")
    print(f"env: {env_name} episodes: {episodes}")
    print(f"true: mean={np.mean(true_returns):.2f} std={np.std(true_returns):.2f}")
    print(f"rm_mean: mean={np.mean(rm_returns_mean):.2f} std={np.std(rm_returns_mean):.2f}")
    print(f"corr(true, rm_mean) = {corr(true_returns, rm_returns_mean):.3f}")

    cors = []
    for i in range(len(inner_nets)):
        c = corr(true_returns, rm_returns_members[i])
        cors.append(c)
        print(f"rm{i}: std={np.std(rm_returns_members[i]):.2f} corr(true, rm{i})={c:.3f}")

    best_i = int(np.nanargmax(cors))
    print(f"\nBest single member by corr: rm{best_i}  corr={cors[best_i]:.3f}")
    print("Recommendation: use ensemble mean (average of 0/1/2) when exporting.")


if __name__ == "__main__":
    main()
