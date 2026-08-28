# Task 2 — submission 9777071

Leaderboard filename: `task2 (3).sif`; audited local artifact: `task2.sif`.

This is a five-fold full fine-tune of `dolphins_xlstm_unet_b` initialized from
BrainAxL run 19726. It uses FLAIR and DWI, patch size `160 x 160 x 32`, balanced
positive/negative sampling, generalized-Dice/focal loss, encoder LR
`9.62411e-5`, decoder LR `4.812056e-4`, 80 epochs, and deep supervision.
Checkpoints maximize validation mean DSC/NSD.

```bash
export PRETRAIN_CHECKPOINT=/path/to/brainaxl_run19726.ckpt
export ASPARAGUS_DATA=/path/to/processed/data
bash train_folds.sh
```

For inference, place each stripped checkpoint at
`model/fold{0..4}_best.ckpt`. The submitted `model/config.json` uses a 0.30
threshold, one score-ranked connected component, 0.5 overlap, batch size 1,
and no TTA.

```bash
export FOMO26_TASK2_MODEL_DIR=$PWD/model
python predict.py --flair flair.nii.gz --dwi dwi.nii.gz \
  --output segmentation.nii.gz
```

Folds 0–1 came from the confirmation run and folds 2–4 from the original run;
their stored Hydra overrides are identical. The five submitted checkpoints are
an ensemble, not five independent submission variants.

Selected epochs and per-fold validation metrics are published in
[`training_logs/task2_9777071_folds.csv`](../../../training_logs/task2_9777071_folds.csv).
