#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root:
#   bash my_scripts/run_dugp_official_experiments.sh assist2015 0

DATASET=${1:-assist2015}
FOLD=${2:-0}
SAVE_DIR=${3:-saved_model_dugp_official}
SEEDS=(3407 42 2024)
MODES=(full fixed_fusion group_add alpha_only no_distance no_uncertainty no_behavior)

export DUGP_GROUP_K=${DUGP_GROUP_K:-8}
export DUGP_ALPHA_FEAT_DIM=${DUGP_ALPHA_FEAT_DIM:-5}
export DUGP_REQUIRE=${DUGP_REQUIRE:-1}

cd examples

# AKT backbone baseline.
for SEED in "${SEEDS[@]}"; do
  python wandb_akt_train.py \
    --dataset_name "${DATASET}" \
    --model_name akt \
    --emb_type qid \
    --save_dir "../${SAVE_DIR}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --use_wandb 0 \
    --add_uuid 1

  # DUGP official variants.
  for MODE in "${MODES[@]}"; do
    python wandb_dugp_train.py \
      --dataset_name "${DATASET}" \
      --model_name akt_dugp \
      --emb_type qid \
      --save_dir "../${SAVE_DIR}" \
      --fold "${FOLD}" \
      --seed "${SEED}" \
      --dugp_mode "${MODE}" \
      --use_wandb 0 \
      --add_uuid 1
  done
done
