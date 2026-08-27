#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
: "${PRETRAIN_CHECKPOINT:?Set PRETRAIN_CHECKPOINT to BrainAxL run-19726 last.ckpt}"
: "${ASPARAGUS_DATA:?Set ASPARAGUS_DATA to the processed task-data root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/task2_9777071}"
FOLDS="${FOLDS:-0 1 2 3 4}"

export PYTHONPATH="${REPO_ROOT}/asparagus:${PYTHONPATH:-}"
export ASPARAGUS_MODELS="${ASPARAGUS_MODELS:-${REPO_ROOT}/runs}"
export SKIP_TEST=true

# Exact training-time sampler/loss settings of the five submitted checkpoints.
export SMALL_OBJECT_SAMPLER=monai_posneg
export MONAI_POS=1.0
export MONAI_NEG=1.0
export SMALL_OBJECT_LOSS=monai_gdfl
export MONAI_FOCAL_GAMMA=1.5
export MONAI_LAMBDA_GDL=1.5
export MONAI_LAMBDA_FOCAL=1.0
export MONAI_GDFL_W_TYPE=square
export SMALL_OBJECT_ROT_PROB=0.1
export SMALL_OBJECT_SCALE_PROB=0.1
export SEG_POSTPROCESS=none
export SEG_POST_FILL_HOLES=false
export SEG_POST_KEEP_COMPONENTS=1
export SEG_POST_MIN_SIZE=32
export SEG_POST_THRESHOLD=0.4
export CKPT_MONITOR=val/foreground_dsc_nsd_mean
export CKPT_MONITOR_MODE=max

for fold in ${FOLDS}; do
  run_dir="${OUTPUT_ROOT}/fold_${fold}"
  "${PYTHON_BIN}" \
    "${REPO_ROOT}/fomo26/submissions/shared/run_finetune_seg_small_object.py" \
    --config-name default_finetune_seg \
    "hydra.run.dir=${run_dir}" \
    +model=dolphins_xlstm_unet_b \
    task=SEG009_FOMO26_Meningioma \
    root=fomo26 \
    "stem=task2_9777071/fold_${fold}" \
    data.train_split=split_80_10_10 \
    "data.fold=${fold}" \
    "checkpoint_path=${PRETRAIN_CHECKPOINT}" \
    checkpoint_run_id=null \
    load_checkpoint_name=last.ckpt \
    training.load_decoder=true \
    training.batch_size=1 \
    training.seed=42 \
    +training.encoder_learning_rate=0.0000962411 \
    +training.decoder_learning_rate=0.0004812056 \
    training.epochs=80 \
    training.warmup_epochs=5 \
    training.decoder_warmup_epochs=8 \
    training.train_batches_per_epoch_per_device=160 \
    training.val_batches_per_epoch_per_device=16 \
    training.patch_size='[160,160,32]' \
    training.check_val_every_n_epoch=2 \
    training.accumulate_grad_batches=1 \
    model.deep_supervision=true \
    model.finetune_lr=0.0004812056 \
    model.ckpt_every_n_epoch=25 \
    model.min_test_patch_size='[160,160,32]' \
    hardware.num_workers=8 \
    hardware.precision=bf16-mixed \
    logger.progress_bar=true \
    logger.log_every_n_steps=50 \
    logger.wandb_logging=false \
    logger.mlflow_logging=false \
    logger.log_images_every_n_epoch=999 \
    +model.ckpt_monitor=val/foreground_dsc_nsd_mean \
    +model.ckpt_monitor_mode=max
done
