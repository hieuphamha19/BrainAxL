#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
: "${PRETRAIN_CHECKPOINT:?Set PRETRAIN_CHECKPOINT to BrainAxL run-19726 last.ckpt}"
: "${ASPARAGUS_DATA:?Set ASPARAGUS_DATA to the processed Task 3 root}"

FINETUNE_BIN="${FINETUNE_BIN:-asp_finetune_reg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/task3_9777067}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEED_BASE="${SEED_BASE:-4610}"
export PYTHONPATH="${REPO_ROOT}/asparagus:${PYTHONPATH:-}"
export ASPARAGUS_MODELS="${ASPARAGUS_MODELS:-${REPO_ROOT}/runs}"
export REGRESSION_LOSS=mse
export REGRESSION_TARGET_NORMALIZE=false
export REGRESSION_TARGET_SCALE=1
export SKIP_TEST=true

for fold in ${FOLDS}; do
  "${FINETUNE_BIN}" \
    --config-name projects/fomo26/baseline/finetune_t3_reg \
    "hydra.run.dir=${OUTPUT_ROOT}/fold_${fold}" \
    "checkpoint_path=${PRETRAIN_CHECKPOINT}" \
    checkpoint_run_id=null \
    load_checkpoint_name=last.ckpt \
    model.pretrain_net=dolphins_xlstm_unet_b \
    model.seg_net=dolphins_xlstm_unet_b \
    model.dimensions=3D \
    +model.starting_filters=40 \
    +model.xlstm_stages='[3,4]' \
    model.cls_net=dolphins_xlstm_unet_b_clsreg \
    model._cls_net._target_=asparagus.modules.networks.dolphins_xlstm_unet.dolphins_xlstm_unet_b_clsreg \
    +model._cls_net.starting_filters=40 \
    +model._cls_net.xlstm_stages='[3,4]' \
    +model._cls_net.late_fusion=false \
    model._cls_net.late_fusion=false \
    training.load_decoder=true \
    training.batch_size=2 \
    training.epochs=60 \
    training.target_size='[128,128,128]' \
    training.limit_train_batches=1.0 \
    training.limit_val_batches=1.0 \
    training.warmup_epochs=0 \
    training.check_val_every_n_epoch=1 \
    "training.seed=$((SEED_BASE + fold))" \
    model.finetune_lr=0.0003 \
    model.decoder_dropout_rate=0.0 \
    model._cls_net.dropout_op_kwargs.decoder_dropout_rate=0.0 \
    transforms.cpu_tr_transforms=CPU_clsreg_train_transforms_nonzero_crop \
    transforms.cpu_val_transforms=CPU_clsreg_val_test_transforms_nonzero_crop \
    data.train_split=split_probe24842_5fold_train80val20 \
    data.test_split=TEST_probe24842_5fold \
    "data.fold=${fold}" \
    hardware.num_workers=8 \
    +lightning._data_module.use_random_datasampler=false \
    +lightning._lightning_module.encoder_learning_rate=0.0001 \
    +lightning._lightning_module.decoder_learning_rate=0.001 \
    hardware.precision=bf16-mixed \
    hardware.compile_mode=default \
    logger.wandb_logging=false \
    logger.mlflow_logging=false
done
