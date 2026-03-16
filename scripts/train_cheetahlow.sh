#!/usr/bin/env bash
set -euo pipefail

# LunarLander safety baseline (matches argparse defaults)
python3 train.py \
  --exp_name "exp0101-eigen-cheetah" \
  --env_name "Halfcheetah-run-low-v5" \
  --seed 97 \
  --policy "Gaussian" \
  --automatic_entropy_tuning True \
  --alpha 0.01 \
  --gamma 0.99 \
  --lr 0.0005 \
  --tau 0.005 \
  --target_update_interval 1 \
  --batch_size 512 \
  --replay_size 50000 \
  --hidden_size 512 \
  --lambda_value 400 \
  --epsilon 1e-3 \
  --gamma_target 1.0 \
  --episodes_per_epoch 10 \
  --gradient_steps_per_epoch 64 \
  --start_steps 1000 \
  --eval_epoch_ratio 100 \
  --save_epoch_ratio 500 \
  --num_episodes 80000 "$@"
