#!/usr/bin/env python3
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(path):
    steps, mean, std = [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(float(row["step"])))
            mean.append(float(row["mean"]))
            std.append(float(row["std"]))
    idx = np.argsort(steps)
    steps = np.array(steps, dtype=np.int64)[idx]
    mean  = np.array(mean, dtype=np.float64)[idx]
    std   = np.array(std,  dtype=np.float64)[idx]
    return steps, mean, std

def moving_avg(x, k):
    if k <= 1: return x
    w = np.ones(k, dtype=np.float64) / k
    return np.convolve(x, w, mode="same")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT group CSV (from tb_mean_std_accum.py)")
    ap.add_argument("--pebble", required=True, help="PEBBLE group CSV (from tb_mean_std_accum.py)")
    ap.add_argument("--tag", default="eval/episode_reward")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--title", default="GT vs PEBBLE (mean ± std)")
    ap.add_argument("--out", default="compare_gt_vs_pebble.png")
    args = ap.parse_args()

    s1, m1, sd1 = read_csv(args.gt)
    s2, m2, sd2 = read_csv(args.pebble)
    m1s, m2s = moving_avg(m1, args.smooth), moving_avg(m2, args.smooth)

    plt.figure(figsize=(9,4.8))
    plt.plot(s1, m1s, linewidth=2, label="GT-SAC mean")
    plt.fill_between(s1, m1s - sd1, m1s + sd1, alpha=0.20)
    plt.plot(s2, m2s, linewidth=2, label="PEBBLE mean")
    plt.fill_between(s2, m2s - sd2, m2s + sd2, alpha=0.20)
    plt.title(args.title)
    plt.xlabel("Step"); plt.ylabel(args.tag)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print("[ok] saved", args.out)

if __name__ == "__main__":
    main()
