# Training and adaptation

This guide separates the common BrainAxL pretraining run from the exact
adaptation paths used in submissions 9777066–9777071. The accompanying
[`training_logs/`](../training_logs/README.md) directory contains sanitized
epoch records and selected-checkpoint summaries.

## Before training

Install the project and create `asparagus/.env` as described in the root
[README](../README.md#installation). Processed tasks must satisfy the
[data contract](../README.md#data-contract). The commands below require data
obtained under the original dataset or challenge terms.

Set checkpoint paths explicitly. This avoids depending on a local experiment
database:

```bash
export PRETRAIN_CHECKPOINT=/absolute/path/to/brainaxl_run19726.ckpt
export ASPARAGUS_DATA=/absolute/path/to/processed/tasks
```

## Self-supervised pretraining

The reported run uses 64-cubed random crops, 60% block masking, AdamW, and
bfloat16 mixed precision. It optimizes reconstruction MSE plus a
variance/covariance regularizer on a 512-dimensional multiscale projection:

```text
loss = reconstruction_mse + 0.01 * (variance + 0.04 * covariance)
```

Launch the checked-in recipe:

```bash
cd asparagus
asp_pretrain --config-name projects/brainaxl/pretrain
```

The exact run used batch size 128, 1,733 optimizer steps per epoch, 20 epochs,
and 34,660 total updates. It had no learning-rate warm-up and used cosine
decay from `1e-4` with weight decay `3e-5`. The validation split was monitored
during training; no TEST split was read. See the
[20-epoch log](../training_logs/pretraining_run19726_epochs.csv) and
[machine-readable summary](../training_logs/runs.json).

The final submission family initializes from run 19726. Checkpoint files are
not committed because of their size and the source data's access conditions.
They are consolidated in the public
[Hugging Face weight repository](model-weights.md), released after the FOMO26
TEST deadline and a final disclosure audit.

## Full fine-tuning

Only Tasks 2, 3, and 4 use end-to-end neural-network fine-tuning.

### Task 2 — meningioma segmentation

Task 2 fine-tunes a five-fold BrainAxL U-Net ensemble on FLAIR and DWI. It uses
balanced positive/negative `160 x 160 x 32` patches, generalized-Dice/focal
loss, deep supervision, five warm-up epochs, and an eight-epoch decoder
warm-up. Encoder and decoder learning rates are `9.62411e-5` and
`4.812056e-4`, respectively.

```bash
cd fomo26/submissions/task2_9777071
bash train_folds.sh
```

Each fold runs 80 epochs with 160 training and 16 validation batches per
epoch. Checkpoints maximize `val/foreground_dsc_nsd_mean`. Folds 0–1 are from
the confirmation run and folds 2–4 from the original run; their training
overrides are identical. The exact selected epochs and fold metrics are in
[`task2_9777071_folds.csv`](../training_logs/task2_9777071_folds.csv).

### Task 3 — brain-age regression

Task 3 fine-tunes five `dolphins_xlstm_unet_b_clsreg` models, the historical
checkpoint-compatible name for the BrainAxL regression model. T1 volumes are
nonzero-cropped with a 16-voxel margin, then center-cropped or padded to
`128^3`. Training uses MSE, batch size 2, 60 epochs, encoder LR `1e-4`,
head LR `1e-3`, and fold seeds 4610–4614.

```bash
cd fomo26/submissions/task3_9777067
bash train_folds.sh
```

The complete 300-row five-fold optimization history is in
[`task3_9777067_epochs.csv`](../training_logs/task3_9777067_epochs.csv).
Evaluation across 494 out-of-fold cases produced MAE 4.8398, RMSE 6.3408, and
Pearson r 0.9320.

### Task 4 — trigeminal neuralgia segmentation

Task 4 fine-tunes one BrainAxL U-Net from the pretrained encoder and a freshly
initialized decoder. It uses T2 input, `128 x 96 x 96` foreground-aware
patches, generalized-Dice/focal loss, deep supervision, LR `2e-4`, and 200
epochs.

```bash
cd fomo26/submissions/task4_9777068
bash train.sh
```

Validation runs every two epochs and checkpointing maximizes mean foreground
Dice. The best logged value is 0.548638 at epoch 165. See the
[200-epoch log](../training_logs/task4_9777068_epochs.csv).

## Frozen-feature adaptation

Tasks 1 and 5 freeze every BrainAxL parameter, extract features once, and fit
balanced logistic regression heads. Therefore they have fit summaries rather
than epoch logs.

Task 1 fits two `C=0.05` heads on 21 subjects and averages their standardized
logits:

```bash
cd fomo26/submissions/task1_9777066
python extract_features.py --manifest train.csv --output features.npz
python fit_head.py --features features.npz \
  --output models/task1_head_params_3ch_ens_mm_s3.npz
```

Task 5 fits one `C=0.05` head on stage-3 features from 48 subjects. Each
subject contributes an original and a pydeface-2.0.2 view, for 96 fit rows:

```bash
cd fomo26/submissions/task5_9777069
python extract_features.py --manifest train_domains.csv --output features.npz
python fit_head.py --features features.npz \
  --output models/task5_head_params_t1_ap140_s3mean_domainaug_C0.05.npz
```

Submission 9777070 performs no task-label adaptation. It freezes the encoder
and semantic projector and emits a 1,024-dimensional vector; the official
Tasks 6/7 evaluator trains its own linear probe.

## Interpreting the public logs

The logs are evidence for the released recipes, not a promise of bitwise
reproduction. CUDA kernels, framework versions, data order, and preprocessing
can change numerical results. The public records contain only training and
local-validation metrics. They do not expose hidden TEST labels, predictions,
subject identifiers, or machine-specific paths.
