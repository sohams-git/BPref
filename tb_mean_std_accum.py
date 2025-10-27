#!/usr/bin/env python3
import argparse, os, glob, math, csv
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalar_series(run_dir, tag):
    ea = EventAccumulator(run_dir); ea.Reload()
    try: scalars = ea.Scalars(tag)
    except KeyError: return None, None
    steps = np.array([e.step for e in scalars], dtype=np.int64)
    vals  = np.array([e.value for e in scalars], dtype=np.float64)
    return steps, vals

def align_on_intersection(step_arrays):
    common = None
    for s in step_arrays:
        ss = set(s.tolist())
        common = ss if common is None else (common & ss)
        if not common: break
    return np.array(sorted(common), dtype=np.int64) if common else np.array([], dtype=np.int64)

def extract_values_at_steps(steps_src, vals_src, steps_target):
    idx = {int(s): i for i, s in enumerate(steps_src)}
    out = np.full((steps_target.shape[0],), np.nan, dtype=np.float64)
    for j, st in enumerate(steps_target):
        i = idx.get(int(st))
        if i is not None: out[j] = vals_src[i]
    return out

def main():
    ap = argparse.ArgumentParser(description="Mean/std over runs for TensorBoard scalars (EventAccumulator).")
    ap.add_argument("runs", nargs="+", help="Run dirs or glob patterns (contain event files).")
    ap.add_argument("--tag", action="append", default=["eval/episode_reward"], help="Scalar tag. Repeatable.")
    ap.add_argument("--csv", default="tb_mean_std.csv", help="Output CSV (tag,step,mean,std,n_runs)")
    ap.add_argument("--plot", default=None, help="Optional PNG for first tag")
    ap.add_argument("--union", action="store_true", help="Use union of steps (default: intersection)")
    args = ap.parse_args()

    # expand globs
    run_dirs = []
    for pat in args.runs:
        m = glob.glob(pat); run_dirs.extend(m if m else [pat])
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        print("[error] No valid run directories found."); return

    rows = []
    for tag in args.tag:
        per_steps, per_vals = [], []
        for rd in run_dirs:
            steps, vals = load_scalar_series(rd, tag)
            if steps is None or steps.size == 0:
                print("[warn] tag '{}' missing/empty in: {}".format(tag, rd)); continue
            per_steps.append(steps); per_vals.append(vals)

        if not per_steps:
            print("[warn] No data for tag '{}'".format(tag)); continue

        target = (np.array(sorted(set(np.concatenate(per_steps).tolist())), dtype=np.int64)
                  if args.union else align_on_intersection(per_steps))
        if target.size == 0:
            print("[warn] no common steps for tag '{}'; try --union".format(tag)); continue

        aligned = [extract_values_at_steps(s, v, target) for s, v in zip(per_steps, per_vals)]
        M = np.stack(aligned, axis=0)
        mean = np.nanmean(M, axis=0); std = np.nanstd(M, axis=0, ddof=0); n = np.sum(~np.isnan(M), axis=0)

        if args.plot and tag == args.tag[0]:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8,4.5))
                plt.plot(target, mean, label="{} mean".format(tag), linewidth=2)
                plt.fill_between(target, mean-std, mean+std, alpha=0.22, label="±1 std")
                plt.xlabel("step"); plt.ylabel(tag); plt.grid(True, alpha=0.3)
                plt.legend(); plt.tight_layout()
                plt.savefig(args.plot, dpi=150); print("[ok] saved plot:", args.plot)
            except Exception as e:
                print("[warn] plotting failed:", e)

        for st, mu, sd, nn in zip(target, mean, std, n):
            if not (mu == mu): continue
            rows.append((tag, int(st), float(mu), float(sd), int(nn)))

    if not rows:
        print("[error] nothing to write."); return
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "step", "mean", "std", "n_runs"]); w.writerows(rows)
    print("[ok] wrote:", args.csv)

if __name__ == "__main__":
    main()
