#!/usr/bin/env bash
# Usage: bash scripts/halfcheetah/500/oracle/run_PEBBLE.sh [sampling_scheme] [seed]
# sampling_scheme: 0=uniform, 1=disagreement, 2=entropy
set -e

SAMPLING=${1:-1}
SEED=${2:-0}

# Activate your venv if needed (comment out if you already did it in the shell):
# source ~/.venvs/bpref-py36/bin/activate

# MuJoCo env vars are assumed set in your shell (MUJOCO_PY_MUJOCO_PATH, MUJOCO_GL, LD_LIBRARY_PATH)

# This calls the PEBBLE trainer exactly like the DMC scripts do, but with Gym HalfCheetah-v2.
python train_PEBBLE.py \
  --env_id HalfCheetah-v2 \
  --teacher oracle \
  --max_feedback 500 \
  --segment_length 50 \
  --sampling_scheme ${SAMPLING} \
  --total_timesteps 1000000 \
  --eval_interval 10000 \
  --seed ${SEED} \
  --logdir runs/halfcheetah/pebble_oracle_b500_s${SEED}_sam${SAMPLING}
