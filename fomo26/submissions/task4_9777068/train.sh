#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
: "${PRETRAIN_CHECKPOINT:?Set PRETRAIN_CHECKPOINT to BrainAxL run-19726 last.ckpt}"
: "${ASPARAGUS_DATA:?Set ASPARAGUS_DATA to the processed task-data root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/task4_9777068}"
export PYTHONPATH="${REPO_ROOT}/asparagus:${PYTHONPATH:-}"
export ASPARAGUS_MODELS="${ASPARAGUS_MODELS:-${REPO_ROOT}/runs}"
export SKIP_TEST=true

export SMALL_OBJECT_SAMPLER=monai_label_classes
export MONAI_LABEL_RATIOS='1 2 2'
export MONAI_NUM_CLASSES=3
export SMALL_OBJECT_LOSS=monai_gdfl
export MONAI_FOCAL_GAMMA=2.0
export MONAI_LAMBDA_GDL=1.0
export MONAI_LAMBDA_FOCAL=1.0
export MONAI_GDFL_W_TYPE=square
export SMALL_OBJECT_ROT_PROB=0.08
export SMALL_OBJECT_SCALE_PROB=0.10
export SMALL_OBJECT_ROT_DEGREES='8 8 8'
export SMALL_OBJECT_SCALE_MIN=0.90
export SMALL_OBJECT_SCALE_MAX=1.10
export SMALL_OBJECT_VAL_MODE=sliding_image
export SMALL_OBJECT_VAL_OVERSAMPLE_FG=1.0
export SEG_POSTPROCESS=argmax_cc
export SEG_POST_THRESHOLD=0.30
export SEG_POST_MIN_SIZE=4
export SEG_POST_KEEP_COMPONENTS=2
export SEG_POST_FILL_HOLES=true
export CKPT_MONITOR=val/foreground_dice_mean
export CKPT_MONITOR_MODE=max

"${PYTHON_BIN}" \
  "${REPO_ROOT}/fomo26/submissions/shared/run_finetune_seg_small_object.py" \
  --config-name default_finetune_seg \
  "hydra.run.dir=${OUTPUT_DIR}" \
  +model=dolphins_xlstm_unet_b \
  task=SEG010_FOMO26_TrigeminalNeuralgia \
  root=fomo26 \
  stem=task4_9777068 \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  data.fold=0 \
  "checkpoint_path=${PRETRAIN_CHECKPOINT}" \
  checkpoint_run_id=null \
  load_checkpoint_name=last.ckpt \
  training.load_decoder=false \
  training.batch_size=1 \
  training.seed=42 \
  training.epochs=200 \
  training.warmup_epochs=10 \
  training.decoder_warmup_epochs=10 \
  training.train_batches_per_epoch_per_device=300 \
  training.val_batches_per_epoch_per_device=16 \
  training.patch_size='[128,96,96]' \
  training.check_val_every_n_epoch=2 \
  training.accumulate_grad_batches=1 \
  model.deep_supervision=true \
  model.finetune_lr=0.0002 \
  model.ckpt_every_n_epoch=25 \
  model.min_test_patch_size='[128,96,96]' \
  hardware.num_workers=8 \
  hardware.precision=bf16-mixed \
  logger.progress_bar=true \
  logger.log_every_n_steps=50 \
  logger.wandb_logging=false \
  logger.mlflow_logging=false \
  logger.log_images_every_n_epoch=999 \
  +model.ckpt_monitor=val/foreground_dice_mean \
  +model.ckpt_monitor_mode=max
